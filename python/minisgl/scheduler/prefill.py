"""
Prefill 阶段调度器模块

负责管理 LLM 推理的 Prefill 阶段 即处理初始输入序列的阶段。
主要功能包括：
1. 将待处理请求添加到 prefill 批次
2. 匹配前缀缓存 避免重复计算
3. 支持分块处理超长序列
4. 管理资源分配和预算控制
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Batch, Req
from minisgl.utils import init_logger

from .utils import PendingReq

if TYPE_CHECKING:
    from minisgl.kvcache import BaseCacheHandle
    from minisgl.message import UserMsg

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager

logger = init_logger(__name__)


class ChunkedReq(Req):
    """
    分块请求类
    
    用于处理超长输入序列 当输入序列太长无法一次性处理时 将其分成多个块。
    每个块作为一个独立的请求进行处理 但共享相同的缓存句柄和页表索引。
    
    与普通 Req 的区别：
    - 不能追加新 token(因为还在 prefill 阶段)
    - 不能进入 decode 阶段(因为输入还未完全处理)
    """
    
    def append_host(self, next_token: torch.Tensor) -> None:
        """
        禁止追加 token
        
        分块请求在 prefill 阶段不能追加新 token 因为输入序列还未完全处理。
        """
        raise NotImplementedError("ChunkedReq should be sampled")

    def can_decode(self) -> bool:
        """
        判断是否可以进入 decode 阶段
        
        分块请求不能进入 decode 阶段 因为输入序列还未完全处理。
        
        Returns:
            False 表示不能进入 decode 阶段
        """
        return False


@dataclass
class PrefillAdder:
    """
    Prefill 请求添加器
    
    负责将待处理的请求添加到 prefill 批次中 同时管理资源分配和预算控制。
    
    Attributes:
        token_budget: 本次批次的 token 预算 限制可以处理的 token 数量
        reserved_size: 保留的缓存大小 用于正在运行的 decode 请求
        cache_manager: 缓存管理器 负责缓存分配和前缀匹配
        table_manager: 页表管理器 负责页表索引分配
    """
    
    token_budget: int
    reserved_size: int
    cache_manager: CacheManager
    table_manager: TableManager

    def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
        """
        尝试为请求分配资源
        
        为请求分配缓存句柄和页表索引。如果资源不足 返回 None。
        
        Args:
            req: 待处理的请求
        
        Returns:
            Tuple[BaseCacheHandle, int] | None: 
            - 如果成功 返回 (缓存句柄, 页表索引)
            - 如果失败 返回 None
        
        工作流程：
        1. 检查页表是否有可用槽位
        2. 匹配请求的前缀缓存
        3. 估算所需资源(扩展长度 + 输出长度)
        4. 检查是否有足够的缓存空间
        5. 锁定缓存句柄
        6. 再次检查资源(锁定后可能发生变化)
        7. 分配页表索引
        8. 如果匹配到前缀缓存 设置页表中的缓存部分
        """
        # 检查页表是否有可用槽位
        if self.table_manager.available_size == 0:
            return None

        # 匹配请求的前缀缓存 查找已存在的缓存
        handle, match_indices = self.cache_manager.match_req(req)
        cached_len = handle.cached_len  # 已缓存的长度
        
        # TODO: better estimate policy
        # 计算需要扩展的长度(未缓存的输入部分)
        extend_len = req.input_len - cached_len
        # 估算总需求：扩展长度 + 输出长度
        estimated_len = extend_len + req.output_len

        # 检查是否有足够的缓存空间(包括保留空间)
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None
        
        # 锁定缓存句柄 保护匹配到的前缀缓存
        self.cache_manager.lock(handle)
        
        # 锁定后再次检查资源(因为锁定会改变可用空间)
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            # 资源不足 解锁并返回 None
            self.cache_manager.unlock(handle)
            return None

        # 分配页表索引
        table_idx = self.table_manager.allocate()
        
        # 如果匹配到前缀缓存 需要设置页表中的缓存部分
        if cached_len > 0:  # NOTE: set the cached part
            # 获取页表中对应位置的 token ID 和页表条目
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]
            page_entry = self.table_manager.page_table[table_idx][:cached_len]
            # 将匹配到的 token IDs 复制到设备(使用 pin_memory 加速)
            device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
            # 将匹配到的缓存索引复制到页表
            page_entry.copy_(match_indices)

        return handle, table_idx

    def _add_one_req(
        self,
        pending_req: PendingReq,
        cache_handle: BaseCacheHandle,
        table_idx: int,
        cached_len: int,
    ) -> Req:
        """
        将请求添加到批次中
        
        根据 token 预算决定处理整个请求还是分块处理。
        
        Args:
            pending_req: 待处理的请求
            cache_handle: 已分配的缓存句柄
            table_idx: 已分配的页表索引
            cached_len: 已缓存的长度
        
        Returns:
            Req: 创建的请求对象(可能是 ChunkedReq 或 Req)
        
        工作流程：
        1. 计算剩余需要处理的长度
        2. 根据 token 预算决定分块大小
        3. 如果分块大小 < 剩余长度 创建 ChunkedReq
        4. 更新 token 预算和保留大小
        5. 将 token IDs 复制到设备的页表中
        6. 创建并返回请求对象
        """
        # 计算剩余需要处理的输入长度
        remain_len = pending_req.input_len - cached_len
        # 分块大小 = min(token 预算, 剩余长度)
        chunk_size = min(self.token_budget, remain_len)
        # 判断是否需要分块处理
        is_chunked = chunk_size < remain_len
        # 根据是否需要分块选择请求类型
        CLS = ChunkedReq if is_chunked else Req
        
        # 更新 token 预算
        self.token_budget -= chunk_size
        # 更新保留大小：剩余输入长度 + 输出长度
        # 这些空间需要为后续处理保留
        self.reserved_size += remain_len + pending_req.output_len
        
        # NOTE: update the tokens ids only; new pages will be allocated in the scheduler
        # 将本次处理的 token IDs 复制到设备的页表中
        _slice = slice(cached_len, cached_len + chunk_size)
        device_ids = self.table_manager.token_pool[table_idx][_slice]
        # 使用 pin_memory 和非阻塞复制加速传输
        device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)
        
        # 创建请求对象
        return CLS(
            input_ids=pending_req.input_ids[: cached_len + chunk_size],  # 只包含已处理的部分
            table_idx=table_idx,
            cached_len=cached_len,  # 已缓存的长度
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
        )

    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        """
        尝试将一个请求添加到批次中
        
        这是主要的入口方法 处理两种情况：
        1. 如果请求是分块请求的延续 直接使用已有资源
        2. 如果是新请求 先分配资源再添加
        
        Args:
            pending_req: 待处理的请求
        
        Returns:
            Req | None: 如果成功添加 返回请求对象；否则返回 None
        
        工作流程：
        1. 检查 token 预算是否足够
        2. 如果是分块请求的延续 使用已有资源
        3. 如果是新请求 先尝试分配资源
        4. 如果资源分配成功 添加请求
        """
        # 检查 token 预算
        if self.token_budget <= 0:
            return None

        # 如果是分块请求的延续 使用已有的缓存句柄和页表索引
        if chunked_req := pending_req.chunked_req:
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=chunked_req.cache_handle,
                table_idx=chunked_req.table_idx,
                cached_len=chunked_req.cached_len,
            )

        # 如果是新请求 先尝试分配资源
        if resource := self._try_allocate_one(pending_req):
            cache_handle, table_idx = resource
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cache_handle.cached_len,
            )

        # 资源不足 无法添加
        return None


@dataclass
class PrefillManager:
    """
    Prefill 阶段管理器
    
    管理 prefill 阶段的调度 负责将待处理请求组织成批次。
    
    Attributes:
        cache_manager: 缓存管理器
        table_manager: 页表管理器
        decode_manager: Decode 阶段管理器 用于获取正在运行的请求信息
        pending_list: 待处理的请求列表
    """
    
    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    pending_list: List[PendingReq] = field(default_factory=list)

    def add_one_req(self, req: UserMsg) -> None:
        """
        添加一个新请求到待处理列表
        
        Args:
            req: 用户消息 包含请求的输入和采样参数
        """
        self.pending_list.append(PendingReq(req.uid, req.input_ids, req.sampling_params))

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        """
        调度下一个 prefill 批次
        
        根据 token 预算和可用资源 从待处理列表中组织一个批次。
        
        Args:
            prefill_budget: Prefill 阶段的 token 预算 限制批次大小
        
        Returns:
            Batch | None: 如果成功创建批次 返回 Batch 对象；否则返回 None
        
        工作流程：
        1. 检查是否有待处理请求
        2. 创建 PrefillAdder 考虑正在运行的 decode 请求占用的资源
        3. 遍历待处理列表 尝试添加请求
        4. 如果请求被分块 记录到 chunked_list
        5. 更新待处理列表(分块请求 + 未处理的请求)
        6. 返回创建的批次
        """
        # 检查是否有待处理请求
        if len(self.pending_list) == 0:
            return None

        # estimated offset due to in-flight decode
        # 创建 PrefillAdder 考虑正在运行的 decode 请求占用的资源
        # reserved_size 用于为 decode 请求保留缓存空间
        adder = PrefillAdder(
            token_budget=prefill_budget,
            reserved_size=self.decode_manager.inflight_tokens,  # 正在运行的 decode 请求占用的 token 数
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
        )
        
        reqs: List[Req] = []  # 本次批次中的请求
        chunked_list: List[PendingReq] = []  # 被分块的请求(需要后续继续处理)
        
        # 遍历待处理列表 尝试添加请求
        for pending_req in self.pending_list:
            if req := adder.try_add_one(pending_req):
                # 清除之前的 chunked_req 标记
                pending_req.chunked_req = None
                # 如果请求被分块 记录到 chunked_list
                if isinstance(req, ChunkedReq):
                    pending_req.chunked_req = req
                    chunked_list.append(pending_req)
                reqs.append(req)
            else:
                # 无法添加更多请求(资源不足或预算用完)
                break  # We cannot add more requests
        
        # 如果没有成功添加任何请求 返回 None
        if len(reqs) == 0:
            return None
        
        # 更新待处理列表：
        # 1. 分块请求放在前面(优先处理)
        # 2. 未处理的请求放在后面
        self.pending_list = chunked_list + self.pending_list[len(reqs) :]
        
        # 返回创建的 prefill 批次
        return Batch(reqs=reqs, phase="prefill")

    @property
    def runnable(self) -> bool:
        """
        判断是否有可运行的请求
        
        Returns:
            True 如果有待处理的请求；否则 False
        """
        return len(self.pending_list) > 0
