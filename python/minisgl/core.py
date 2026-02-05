"""
核心数据结构模块

定义了 LLM 推理系统中的核心数据结构：
- SamplingParams: 采样参数配置
- Req: 单个推理请求的抽象
- Batch: 批量推理请求的抽象
- Context: 全局推理上下文
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Literal

import torch

# 类型检查时的导入 避免循环依赖
if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend, BaseAttnMetadata
    from minisgl.kvcache import BaseCacheHandle, BaseKVCache


@dataclass
class SamplingParams:
    """
    采样参数配置
    
    用于控制 LLM 生成过程中的采样策略和生成限制。
    
    Attributes:
        top_k: Top-K 采样参数。当 temperature=0 时, 选择概率最高的 k 个 token；
               当 top_k=1 时, 等价于贪婪解码(greedy decoding)
        ignore_eos: 是否忽略 EOS(End of Sequence)token。如果为 True, 即使生成 EOS 也会继续生成
        temperature: 采样温度。0.0 表示贪婪解码(确定性的), >0 表示随机采样
                    温度越高, 生成越随机多样
        max_tokens: 最大生成 token 数量限制, 防止无限生成
    """
    top_k: int = 1
    ignore_eos: bool = False
    temperature: float = 0.0
    max_tokens: int = 1024


class Req:
    """
    单个推理请求的抽象
    
    表示一个完整的 LLM 推理请求, 包含输入序列、缓存状态、采样参数等信息。
    请求的生命周期包括：
    1. Prefill 阶段：处理初始输入序列
    2. Decode 阶段：逐个生成新的 token
    
    Attributes:
        host_ids: 存储在 CPU 上的完整 token ID 序列(包括输入和已生成的部分)
        table_idx: 在页表(page table)中的索引位置
        cached_len: 已经缓存到 KV Cache 中的 token 数量(这些 token 的 KV 已被计算和存储)
        device_len: 当前在设备(GPU)上的序列长度(包括输入和已生成的 token)
        max_device_len: 设备上序列的最大允许长度(等于输入长度 + 最大输出长度)
        uid: 请求的唯一标识符
        sampling_params: 采样参数配置
        cache_handle: KV Cache 的句柄 用于管理该请求的缓存
    
    状态关系：
        cached_len <= device_len <= max_device_len
        - cached_len: 已缓存的长度(不需要重新计算 attention)
        - device_len: 当前序列长度(可能包含新生成的 token, 但还未缓存)
        - max_device_len: 最大允许长度(防止超出限制)
    """
    
    def __init__(
        self,
        *,
        input_ids: torch.Tensor,
        table_idx: int,
        cached_len: int,
        output_len: int,
        uid: int,
        sampling_params: SamplingParams,
        cache_handle: BaseCacheHandle,
    ) -> None:
        """
        初始化请求对象
        
        Args:
            input_ids: 输入的 token ID 序列 必须在 CPU 上
            table_idx: 页表中的索引位置
            cached_len: 已缓存的 token 数量(从缓存中匹配到的前缀长度)
            output_len: 期望的最大输出 token 数量
            uid: 请求的唯一标识符
            sampling_params: 采样参数配置
            cache_handle: KV Cache 句柄
        """
        # 输入序列必须在 CPU 上 以便后续操作(如拼接新 token)
        assert input_ids.is_cpu

        self.host_ids = input_ids  # 存储在 CPU 上的完整序列
        self.table_idx = table_idx  # 页表索引
        self.cached_len = cached_len  # 已缓存的长度
        self.device_len = len(input_ids)  # 当前设备上的序列长度(初始等于输入长度)
        self.max_device_len = len(input_ids) + output_len  # 最大允许长度
        self.uid = uid  # 唯一标识符
        self.sampling_params = sampling_params  # 采样参数
        self.cache_handle = cache_handle  # 缓存句柄

        # 验证状态的合理性
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len

    @property
    def remain_len(self) -> int:
        """
        剩余可生成的 token 数量
        
        返回还可以生成多少个 token 才会达到最大长度限制。
        
        Returns:
            剩余可生成的 token 数量 = max_device_len - device_len
        """
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        """
        本次需要扩展的长度(需要计算 KV Cache 的新 token 数量)
        
        返回当前设备序列中还未缓存的 token 数量。
        这些 token 需要在 forward 过程中计算并存储 KV Cache。
        
        Returns:
            需要扩展的长度 = device_len - cached_len
        """
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        """
        完成一个 token 的生成
        
        在 decode 阶段, 每生成一个新的 token, 调用此方法更新状态：
        - cached_len 更新为 device_len(表示当前所有 token 都已缓存)
        - device_len 增加 1(表示新生成了一个 token, 但还未缓存)
        
        工作流程：
        1. 生成新的 token(decode 阶段)
        2. 调用 complete_one() 更新状态
        3. 下一次 forward 时会计算并缓存新 token 的 KV
        """
        self.cached_len = self.device_len  # 当前所有 token 都已缓存
        self.device_len += 1  # 增加一个 token(新生成的)

    def append_host(self, next_token: torch.Tensor) -> None:
        """
        将新生成的 token 追加到 CPU 上的序列中
        
        在 decode 阶段生成新 token 后, 将其追加到 host_ids 中保存。
        这样可以跟踪完整的生成序列。
        
        Args:
            next_token: 新生成的 token ID(标量或单元素张量)
        """
        self.host_ids = torch.cat([self.host_ids, next_token])

    def can_decode(self) -> bool:
        """
        判断是否可以继续 decode
        
        检查是否还有剩余空间可以生成新的 token。
        
        Returns:
            True 如果还有剩余空间可以生成 token
            False 如果已达到最大长度限制
        """
        return self.remain_len > 0

    def __repr__(self) -> str:
        """
        返回请求的字符串表示 用于调试
        """
        return (
            f"{type(self)}(table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )


class Batch:
    """
    批量推理请求的抽象
    
    将多个请求组合成一个批次 以便并行处理。
    批次可以处于两种阶段：
    - prefill: 处理初始输入序列的填充阶段
    - decode: 逐个生成 token 的解码阶段
    
    Attributes:
        reqs: 批次中包含的真实请求列表
        phase: 批次阶段 "prefill" 或 "decode"
        input_ids: 批次的输入 token ID 张量(由 scheduler 设置)
                   形状通常为 (batch_size, seq_len) 或 (total_tokens,)
        out_loc: 输出位置索引(由 scheduler 设置)用于确定每个请求的输出位置
        padded_reqs: 填充后的请求列表(可能包含虚拟请求用于对齐)
                    由 scheduler 设置 用于处理不同长度的序列
        attn_metadata: 注意力机制的元数据(由 attention backend 设置)
                      包含位置信息等 attention 计算所需的数据
    """
    
    def __init__(self, *, reqs: List[Req], phase: Literal["prefill", "decode"]):
        """
        初始化批次对象
        
        Args:
            reqs: 批次中包含的请求列表
            phase: 批次阶段 "prefill" 或 "decode"
        
        Note:
            input_ids、out_loc、padded_reqs 和 attn_metadata 字段
            需要在创建后由相应的组件设置
        """
        self.reqs = reqs  # 真实请求列表
        self.phase: Literal["prefill", "decode"] = phase  # 批次阶段
        # 这些字段由 scheduler 设置
        self.input_ids: torch.Tensor  # 批次的输入 token IDs
        self.out_loc: torch.Tensor  # 输出位置索引
        self.padded_reqs: List[Req]  # 可能包含一些虚拟请求用于填充对齐
        # 这个字段由 attention backend 设置
        self.attn_metadata: BaseAttnMetadata  # 注意力机制的元数据

    @property
    def is_prefill(self) -> bool:
        """
        判断是否是 prefill 阶段
        
        Returns:
            True 如果是 prefill 阶段
        """
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        """
        判断是否是 decode 阶段
        
        Returns:
            True 如果是 decode 阶段
        """
        return self.phase == "decode"

    @property
    def size(self) -> int:
        """
        返回批次中真实请求的数量
        
        Returns:
            真实请求的数量(不包括填充的虚拟请求)
        """
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        """
        返回填充后的批次大小
        
        Returns:
            包括填充虚拟请求后的批次大小
        """
        return len(self.padded_reqs)


class Context:
    """
    全局推理上下文
    
    管理整个推理系统的全局状态 包括 KV Cache、注意力后端、页表等。
    使用单例模式通过全局函数访问。
    
    Attributes:
        _batch: 当前活跃的批次(None 表示没有活跃批次)
        page_table: 页表 用于管理 KV Cache 的物理存储位置
                    形状为 2D 张量 存储在 CUDA 设备上
        kv_cache: KV Cache 对象 存储和管理键值缓存
        attn_backend: 注意力后端 负责执行 attention 计算
        page_size: 页面大小(当前固定为 1)
    """
    
    def __init__(
        self,
        *,
        page_size: int,
        kv_cache: BaseKVCache,
        attn_backend: BaseAttnBackend,
        page_table: torch.Tensor,
    ):
        """
        初始化上下文对象
        
        Args:
            page_size: 页面大小(当前只支持 page_size=1)
            kv_cache: KV Cache 对象
            attn_backend: 注意力后端
            page_table: 页表张量 必须是 2D、CUDA、int32、连续存储
        """
        self._batch: Batch | None = None  # 当前活跃批次
        self.page_table = page_table
        # 验证页表的格式要求
        assert (
            self.page_table.dim() == 2  # 2D 张量
            and self.page_table.is_cuda  # 必须在 CUDA 设备上
            and self.page_table.dtype == torch.int32  # int32 类型
            and self.page_table.is_contiguous()  # 内存连续
        )
        self.kv_cache = kv_cache  # KV Cache
        self.attn_backend = attn_backend  # 注意力后端
        assert page_size == 1  # 当前只支持 page_size=1

    def set_batch(self, batch: Batch):
        """
        设置当前活跃的批次
        
        Args:
            batch: 要设置的批次对象
        
        Raises:
            AssertionError: 如果已经有活跃的批次
        """
        assert self._batch is None  # 确保之前没有活跃批次
        self._batch = batch

    def reset_batch(self):
        """
        重置当前批次
        
        清除当前活跃的批次, 设置为 None。
        
        Raises:
            AssertionError: 如果当前没有活跃的批次
        """
        assert self._batch is not None  # 确保有活跃批次
        self._batch = None

    @contextmanager
    def forward_batch(self, batch: Batch):
        """
        上下文管理器：安全地设置和重置批次
        
        使用 Python 的 contextmanager 装饰器, 确保在 forward 过程中
        正确设置批次, 并在完成后自动重置。
        
        使用示例：
            with ctx.forward_batch(batch):
                # 执行 forward 操作
                output = model.forward(...)
        
        Args:
            batch: 要设置的批次对象
        
        Yields:
            不返回任何值, 仅作为上下文管理器使用
        """
        self.set_batch(batch)
        try:
            yield  # 执行 forward 操作
        finally:
            self.reset_batch()  # 确保总是会重置批次

    @property
    def batch(self) -> Batch:
        """
        获取当前活跃的批次
        
        Returns:
            当前活跃的批次对象
        
        Raises:
            AssertionError: 如果当前没有活跃的批次
        """
        assert self._batch is not None, "Global batch is not set"
        return self._batch


# 全局上下文单例
_GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context):
    """
    设置全局上下文
    
    在系统初始化时调用 设置全局上下文单例。
    
    Args:
        ctx: 要设置的上下文对象
    
    Raises:
        AssertionError: 如果全局上下文已经被设置过
    """
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    """
    获取全局上下文
    
    返回之前设置的全局上下文单例。这是访问全局状态的标准方式。
    
    Returns:
        全局上下文对象
    
    Raises:
        AssertionError: 如果全局上下文尚未被设置
    
    使用示例：
        ctx = get_global_ctx()
        batch = ctx.batch
        kv_cache = ctx.kv_cache
    """
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX
