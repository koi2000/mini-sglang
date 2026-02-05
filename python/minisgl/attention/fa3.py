"""
FlashAttention 3 (FA3) 注意力后端模块

实现基于 FlashAttention 3 的注意力计算后端。
使用 sgl-kernel 库提供的高性能 FlashAttention 实现 支持 PagedAttention 和 CUDA Graph。

主要特性：
- 支持 PagedAttention: 使用页表管理 KV Cache
- 支持 CUDA Graph: 优化 decode 阶段性能
- 支持变长序列: 使用累积序列长度(cu_seqlens)处理不同长度的序列
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple

import torch

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData, make_positions

if TYPE_CHECKING:
    from minisgl.core import Batch
    from minisgl.kvcache import BaseKVCache
    from minisgl.models import ModelConfig


@dataclass
class FA3CaptureData(BaseCaptureData):
    """
    FlashAttention 3 的 CUDA Graph 捕获数据
    
    继承自 BaseCaptureData 用于存储 CUDA Graph 捕获时需要的所有数据。
    这些数据在捕获时固定 在重放时更新。
    """
    pass


@dataclass
class FA3Metadata(BaseAttnMetadata):
    """
    FlashAttention 3 的注意力元数据
    
    存储 FlashAttention 计算所需的所有元数据信息。
    
    Attributes:
        cu_seqlens_k: Key 的累积序列长度 形状为 (batch_size + 1,)
                      用于指示每个请求在 K 中的起始和结束位置
        cu_seqlens_q: Query 的累积序列长度 形状为 (batch_size + 1,)
                      用于指示每个请求在 Q 中的起始和结束位置
        cache_seqlens: 每个请求的缓存序列长度 形状为 (batch_size,)
        max_seqlen_k: 所有请求中 K 的最大序列长度
        max_seqlen_q: 所有请求中 Q 的最大序列长度
        page_table: 页表 形状为 (batch_size, max_seqlen_k)
                    用于将逻辑位置映射到物理 KV Cache 页面
        positions: 位置张量(继承自 BaseAttnMetadata)
    
    累积序列长度说明：
        cu_seqlens = [0, seq_len_0, seq_len_0 + seq_len_1, ...]
        例如：如果有 3 个请求 长度分别为 [2, 3, 1]
        则 cu_seqlens = [0, 2, 5, 6]
        表示：请求 0 在 [0, 2) 请求 1 在 [2, 5) 请求 2 在 [5, 6)
    """
    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cache_seqlens: torch.Tensor
    max_seqlen_k: int
    max_seqlen_q: int

    page_table: torch.Tensor

    def get_positions(self) -> torch.Tensor:
        """
        获取位置张量
        
        Returns:
            位置张量
        """
        return self.positions

    def get_last_indices(self, bs: int) -> torch.Tensor:
        """
        获取批次中每个请求的最后一个 token 的索引
        
        用于在 decode 阶段定位每个请求的最后一个 token 位置。
        
        Args:
            bs: 批次大小(实际请求数量 不包括填充)
        
        Returns:
            每个请求最后一个 token 的索引 形状为 (bs,)
        
        算法：
            cu_seqlens_q[1:bs+1] 是每个请求的结束位置
            减 1 得到最后一个 token 的索引
        """
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class FlashAttentionBackend(BaseAttnBackend):
    """
    FlashAttention 3 注意力后端
    
    实现基于 FlashAttention 3 的注意力计算 使用 sgl-kernel 库。
    支持 PagedAttention 和 CUDA Graph 优化。
    
    Attributes:
        config: 模型配置
        kvcache: KV Cache 对象
        capture: CUDA Graph 捕获数据
        max_graph_bs: 最大支持的 CUDA Graph 批次大小
        capture_bs: 已捕获的批次大小列表
        scale: Softmax 缩放因子 = 1 / sqrt(head_dim)
        page_table: 页表 用于管理 KV Cache 的物理存储位置
    """
    
    def __init__(self, config: ModelConfig, kvcache: BaseKVCache, page_table: torch.Tensor):
        """
        初始化 FlashAttention 后端
        
        Args:
            config: 模型配置
            kvcache: KV Cache 对象
            page_table: 页表张量
        """
        self.config = config
        self.kvcache = kvcache
        self.capture: FA3CaptureData | None = None  # CUDA Graph 捕获数据
        self.max_graph_bs = 0  # 最大支持的 CUDA Graph 批次大小
        self.capture_bs: List[int] = []  # 已捕获的批次大小列表
        # Softmax 缩放因子 = 1 / sqrt(head_dim) 用于缩放注意力分数
        self.scale = config.head_dim**-0.5
        self.page_table = page_table

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        """
        执行注意力计算
        
        这是注意力机制的核心方法 执行 QKV 注意力计算。
        
        Args:
            q: Query 张量
            k: Key 张量
            v: Value 张量
            layer_id: 当前层的 ID
            batch: 批次对象
        
        Returns:
            注意力输出张量
        
        工作流程：
        1. 获取元数据
        2. 将新的 K 和 V 存储到 KV Cache
        3. 调用 FlashAttention 实现执行注意力计算
        """
        metadata = batch.attn_metadata
        assert isinstance(metadata, FA3Metadata)
        
        # 将新的 K 和 V 存储到 KV Cache
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        
        # 调用 FlashAttention 实现
        return _fa3_sgl_impl(
            q=q,
            k_cache=self.kvcache.k_cache(layer_id),  # 从 KV Cache 获取 K
            v_cache=self.kvcache.v_cache(layer_id),  # 从 KV Cache 获取 V
            page_table=metadata.page_table,  # 页表
            cache_seqlens=metadata.cache_seqlens,  # 缓存序列长度
            cu_seqlens_q=metadata.cu_seqlens_q,  # Query 累积序列长度
            cu_seqlens_k_new=metadata.cu_seqlens_k,  # Key 累积序列长度
            max_seqlen_q=metadata.max_seqlen_q,  # Query 最大序列长度
            softmax_scale=self.scale,  # Softmax 缩放因子
        )

    def prepare_metadata(self, batch: Batch) -> None:
        """
        准备注意力元数据
        
        计算并准备 FlashAttention 计算所需的所有元数据。
        包括累积序列长度、页表等。
        
        Args:
            batch: 批次对象 其 attn_metadata 字段会被设置
        
        工作流程：
        1. 计算每个请求的序列长度(Query 和 Key)
        2. 计算累积序列长度(cu_seqlens)
        3. 准备页表
        4. 创建 FA3Metadata 对象
        """
        reqs = batch.padded_reqs  # 包括填充的请求

        padded_size = len(reqs)
        # Query 序列长度：需要扩展的长度(新计算的 token 数量)
        seqlens_q = [req.extend_len for req in reqs]
        # Key 序列长度：设备上的总长度(包括已缓存的)
        seqlens_k = [req.device_len for req in reqs]
        # 已缓存的长度
        cached_lens = [req.cached_len for req in reqs]
        # 计算最大序列长度
        max_seqlen_k = max(seqlens_k)
        max_seqlen_q = max(seqlens_q)
        
        # CPU 张量参数(使用 pin_memory 加速传输)
        cpu_kwargs = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        device = self.kvcache.device
        # 缓存序列长度：每个请求的 Key 序列长度
        cache_seqlens = torch.tensor(seqlens_k, **cpu_kwargs)
        cache_seqlens = cache_seqlens.to(device, non_blocking=True)
        
        # Key 的累积序列长度：[0, seq_len_0, seq_len_0 + seq_len_1, ...]
        cu_seqlens_k = torch.tensor([0] + seqlens_k, **cpu_kwargs).cumsum_(dim=0)
        cu_seqlens_k = cu_seqlens_k.to(device, non_blocking=True)

        # Query 的累积序列长度 根据情况选择不同的计算方式
        if max_seqlen_q == 1:
            # Decode 阶段：每个请求只有一个 Query token
            # cu_seqlens_q = [0, 1, 2, ..., padded_size]
            cu_seqlens_q = torch.arange(0, padded_size + 1, device=device, dtype=torch.int32)
        elif all(l == 0 for l in cached_lens):  # prefill with no cache hit
            # Prefill 阶段且没有缓存命中：Query 和 Key 长度相同
            cu_seqlens_q = cu_seqlens_k
        else:  # normal extend prefill, with partial cache hit
            # 正常的扩展 prefill：有部分缓存命中
            # Query 只包含新计算的 token
            cu_seqlens_q = torch.tensor([0] + seqlens_q, **cpu_kwargs).cumsum_(dim=0)
            cu_seqlens_q = cu_seqlens_q.to(self.kvcache.device, non_blocking=True)

        # 生成位置信息
        positions = make_positions(device, reqs)
        
        # 准备页表：为每个请求提取对应的页表条目
        page_table = self.page_table
        new_page_table = torch.stack([page_table[req.table_idx, :max_seqlen_k] for req in reqs])

        # 创建元数据对象并设置到批次中
        batch.attn_metadata = FA3Metadata(
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_q=cu_seqlens_q,
            positions=positions,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            page_table=new_page_table,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        """
        初始化 CUDA Graph 捕获
        
        为指定的批次大小列表创建捕获数据结构。
        这通常在系统初始化时调用一次。
        
        Args:
            max_seq_len: 最大序列长度
            bs_list: 需要捕获 CUDA Graph 的批次大小列表
        
        Raises:
            AssertionError: 如果捕获已经初始化
        """
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        # 创建捕获数据结构(预分配最大批次大小)
        capture = FA3CaptureData.create(max_bs, max_seq_len, self.kvcache.device)
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)  # 排序以便后续查找

    def prepare_for_capture(self, batch: Batch) -> None:
        """
        准备 CUDA Graph 捕获
        
        在捕获 CUDA Graph 之前调用 准备批次数据。
        使用预分配的捕获数据结构 确保数据格式符合 CUDA Graph 捕获的要求。
        
        Args:
            batch: 用于捕获 CUDA Graph 的批次对象
        
        Raises:
            AssertionError: 如果批次大小不在捕获列表中或捕获未初始化
        """
        assert (bs := batch.size) in self.capture_bs and self.capture
        capture = self.capture
        # 使用捕获数据结构中的预分配张量创建元数据
        metadata = FA3Metadata(
            cu_seqlens_k=capture.cu_seqlens_k[: bs + 1],  # 只使用前 bs+1 个元素
            cu_seqlens_q=capture.cu_seqlens_q[: bs + 1],
            positions=capture.positions[:bs],
            cache_seqlens=capture.seq_lens[:bs],
            max_seqlen_k=capture.page_table.size(1),  # 页表的列数
            max_seqlen_q=1,  # decode only: decode 阶段每个请求只有一个 Query token
            page_table=capture.page_table[:bs, :],
        )
        # 设置批次的数据和元数据
        batch.attn_metadata = metadata
        batch.input_ids = capture.input_ids[:bs]
        batch.out_loc = capture.out_loc[:bs]

    def prepare_for_replay(self, batch: Batch) -> None:
        """
        准备 CUDA Graph 重放
        
        在重放 CUDA Graph 之前调用 更新捕获数据结构。
        将当前批次的数据复制到捕获数据结构中 以便 CUDA Graph 重放时使用。
        
        Args:
            batch: 用于重放 CUDA Graph 的批次对象
        
        Raises:
            AssertionError: 如果元数据类型不正确或批次大小不在捕获列表中
        
        Note:
            cu_seqlens_q 在 decode 阶段总是 [0, 1, 2, ..., bs] 所以不需要更新
        """
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, FA3Metadata)
        assert self.capture is not None and bs in self.capture_bs
        
        # cu_seqlens_q is always [0, 1, 2, ..., bs] for decode (i.e. no-op)
        # 更新捕获数据结构中的动态数据
        self.capture.input_ids[:bs].copy_(batch.input_ids)
        self.capture.out_loc[:bs].copy_(batch.out_loc)
        self.capture.cu_seqlens_k[: bs + 1].copy_(metadata.cu_seqlens_k)
        self.capture.positions[:bs].copy_(metadata.positions)
        self.capture.seq_lens[:bs].copy_(metadata.cache_seqlens)
        self.capture.page_table[:bs, : metadata.max_seqlen_k].copy_(metadata.page_table)


def _fa3_sgl_impl(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k_new: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    sm_margin: int = 0,
    window_size: Tuple[int, int] = (-1, -1),  # -1 means infinite context window
    softcap: float = 0.0,  # 0.0 means deactivated
    num_splits: int = 0,  # Can be tuned for speed
    pack_gqa: bool | None = None,  # Can be tuned for speed
    o: torch.Tensor | None = None,  # Can be used to save memory
) -> torch.Tensor:
    """
    FlashAttention 3 的核心实现
    
    调用 sgl-kernel 库的 FlashAttention 实现执行注意力计算。
    支持 PagedAttention 和变长序列。
    
    Args:
        q: Query 张量
        k_cache: Key Cache 张量(从 KV Cache 中获取)
        v_cache: Value Cache 张量(从 KV Cache 中获取)
        page_table: 页表 用于将逻辑位置映射到物理 KV Cache 页面
        cache_seqlens: 每个请求的缓存序列长度
        cu_seqlens_q: Query 的累积序列长度
        cu_seqlens_k_new: Key 的累积序列长度
        max_seqlen_q: Query 的最大序列长度
        softmax_scale: Softmax 缩放因子 = 1 / sqrt(head_dim)
        sm_margin: Softmax margin(用于数值稳定性)
        window_size: 注意力窗口大小 (-1 表示无限上下文窗口)
        softcap: Softcap 参数(0.0 表示禁用)
        num_splits: 分割数量(可用于调优速度)
        pack_gqa: 是否打包 GQA(可用于调优速度)
        o: 输出张量(可选 可用于节省内存)
    
    Returns:
        注意力输出张量
    
    Raises:
        ImportError: 如果 sgl-kernel 库未安装
        AssertionError: 如果张量的最后一个维度不连续
    
    Note:
        这个函数是 FlashAttention 3 的核心实现 直接调用底层 C++/CUDA kernel。
        所有输入张量的最后一个维度必须是连续的(contiguous)。
    """
    # 检查 sgl-kernel 库是否已安装
    try:
        import sgl_kernel.flash_attn  # noqa: F401
    except ImportError:
        raise ImportError(
            "sgl_kernel.flash_attn is not found. Please install it with `pip install sgl-kernel`.\n"
            "If you're sure it's correctly installed, try `apt update && apt install libnuma1`."
        )

    # 验证所有输入张量的最后一个维度是连续的
    # 这是 CUDA kernel 的要求
    for x in (k_cache, v_cache, q, page_table, cache_seqlens, cu_seqlens_q, cu_seqlens_k_new):
        assert x.stride(-1) == 1, "this tensor must have contiguous last dimension"

    # 调用 sgl-kernel 的 FlashAttention forward 操作
    out, *_ = torch.ops.sgl_kernel.fwd.default(  # type: ignore
        q,  # Query
        k_cache,  # Key Cache
        v_cache,  # Value Cache
        None,  # k (新 Key 不使用 因为使用 k_cache)
        None,  # v (新 Value 不使用 因为使用 v_cache)
        None,  # q_v (Query-Value 不使用)
        o,  # 输出张量(可选)
        cu_seqlens_q,  # Query 累积序列长度
        None,  # cu_seqlens_k (旧 Key 累积序列长度 不使用)
        cu_seqlens_k_new,  # 新 Key 累积序列长度
        None,  # seqused_q (Query 使用的序列 不使用)
        cache_seqlens,  # 缓存序列长度
        max_seqlen_q,  # Query 最大序列长度
        None,  # max_seqlen_k (Key 最大序列长度 不使用)
        page_table,  # 页表
        None,  # kv_batch_idx_ (KV 批次索引 不使用)
        None,  # leftpad_k_ (Key 左填充 不使用)
        None,  # rotary_cos (旋转位置编码 cos 不使用)
        None,  # rotary_sin (旋转位置编码 sin 不使用)
        None,  # rotary_seqlens (旋转序列长度 不使用)
        None,  # q_descale (Query 反缩放 不使用)
        None,  # k_descale (Key 反缩放 不使用)
        None,  # v_descale (Value 反缩放 不使用)
        softmax_scale,  # Softmax 缩放因子
        True,  # causal (因果注意力 mask)
        window_size[0],  # 左窗口大小
        window_size[1],  # 右窗口大小
        0,  # attention_chunk (注意力分块 不使用)
        softcap,  # Softcap 参数
        True,  # rotary_interleaved (旋转交错 不使用)
        None,  # scheduler_metadata (调度器元数据 不使用)
        num_splits,  # 分割数量
        pack_gqa,  # 是否打包 GQA
        sm_margin,  # Softmax margin
        None,  # q_v_descale (Query-Value 反缩放 不使用)
    )

    return out
