"""
缓存管理器模块

提供 KV Cache 的分配、释放、匹配和完整性检查功能。
CacheManager 是调度器与底层缓存管理系统的接口层。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.kvcache import BaseCacheHandle, create_cache_manager

if TYPE_CHECKING:
    from .utils import PendingReq


class CacheManager:
    """
    缓存管理器
    
    负责管理 KV Cache 的物理存储分配和逻辑缓存管理。
    主要功能包括：
    1. 缓存页面的分配和释放
    2. 请求前缀的匹配（查找已存在的缓存）
    3. 缓存句柄的锁定/解锁（保护正在使用的缓存）
    4. 缓存完整性检查
    
    设计思路：
    - 使用页表(page table)管理物理存储
    - 使用底层缓存管理器(如 RadixCacheManager)管理逻辑缓存结构
    - 维护空闲槽位列表(_free_slots)用于快速分配
    - 当空闲槽位不足时，通过驱逐(eviction)释放空间
    
    Attributes:
        _free_slots: 空闲的缓存页面索引列表 存储在设备上
        device: PyTorch 设备
        manager: 底层缓存管理器(如 RadixCacheManager) 负责逻辑缓存管理
        num_pages: 总的缓存页面数量
    """
    
    def __init__(self, device: torch.device, num_pages: int, type: str):
        """
        初始化缓存管理器
        
        Args:
            device: PyTorch 设备(通常是 CUDA 设备)
            num_pages: 总的缓存页面数量
            type: 缓存管理器类型 如 "radix" 或 "naive"
        
        Note:
            当前只支持 page_size=1 即每个页面存储一个 token 的 KV
        """
        # TODO: support page_size > 1
        # 初始化空闲槽位列表 包含所有页面的索引 [0, 1, 2, ..., num_pages-1]
        self._free_slots = torch.arange(num_pages, dtype=torch.int32, device=device)
        self.device = device
        # 创建底层缓存管理器(如 RadixCacheManager)
        self.manager = create_cache_manager(device=device, type=type)
        self.num_pages = num_pages

    def _free(self, indices: torch.Tensor) -> None:
        """
        释放缓存页面索引 将其加入空闲列表
        
        内部方法 用于将不再使用的缓存页面索引回收。
        
        Args:
            indices: 要释放的缓存页面索引张量 形状为 (num_indices,)
        """
        if len(indices) > 0:
            # 将释放的索引追加到空闲列表
            self._free_slots = torch.cat([self._free_slots, indices])

    def match_req(self, req: PendingReq):
        """
        匹配请求的前缀 查找已存在的缓存
        
        在 Radix Tree 中查找与请求输入序列匹配的最长前缀。
        用于实现前缀缓存(Prefix Caching)功能 避免重复计算相同前缀的 KV。
        
        Args:
            req: 待处理的请求对象 包含输入序列等信息
        
        Returns:
            Tuple[BaseCacheHandle, torch.Tensor]: 
            - handle: 匹配前缀的缓存句柄
            - indices: 匹配前缀在缓存中的索引位置
        
        Note:
            只匹配 input_len - 1 个 token 因为最后一个 token 需要用于生成下一个 token
        """
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        # 匹配除最后一个 token 外的所有输入 token
        # 最后一个 token 需要用于生成 所以不参与匹配
        return self.manager.match_prefix(req.input_ids[: input_len - 1])

    @property
    def available_size(self) -> int:
        """
        获取可用的缓存大小
        
        返回当前可以分配的缓存页面数量。
        包括：
        - 空闲槽位数量
        - 可驱逐的缓存大小(通过 eviction 可以释放的空间)
        
        Returns:
            可用的缓存页面数量
        """
        return self.manager.size_info.evictable_size + len(self._free_slots)

    def lock(self, handle: BaseCacheHandle) -> None:
        """
        锁定缓存句柄 保护缓存不被驱逐
        
        当请求开始使用某个缓存前缀时 需要锁定对应的句柄。
        锁定的缓存不会被 evict 操作驱逐。
        
        Args:
            handle: 要锁定的缓存句柄
        """
        self.manager.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        """
        解锁缓存句柄 允许缓存被驱逐
        
        当请求不再使用某个缓存前缀时 解锁对应的句柄。
        解锁后的缓存可以被 evict 操作驱逐(如果引用计数为0)。
        
        Args:
            handle: 要解锁的缓存句柄
        """
        self.manager.lock_handle(handle, unlock=True)

    def allocate(self, needed_len: int) -> torch.Tensor:
        """
        分配缓存页面
        
        分配指定数量的缓存页面索引。如果空闲槽位不足，
        会通过驱逐(eviction)释放空间。
        
        Args:
            needed_len: 需要分配的页面数量
        
        Returns:
            torch.Tensor: 分配的页面索引张量 形状为 (needed_len,)
        
        算法流程：
        1. 如果空闲槽位足够 直接从空闲列表分配
        2. 如果空闲槽位不足：
           a. 计算需要驱逐的数量 = needed_len - free_len
           b. 调用底层管理器的 evict 方法驱逐缓存
           c. 将空闲槽位和驱逐释放的槽位合并
           d. 从合并后的列表分配所需数量
        
        Raises:
            AssertionError: 如果驱逐后仍无法满足需求
        """
        # 检查空闲槽位是否足够
        if needed_len <= (free_len := len(self._free_slots)):
            # 空闲槽位足够 直接分配
            allocated = self._free_slots[:needed_len]
            self._free_slots = self._free_slots[needed_len:]
            return allocated

        # 空闲槽位不足 需要驱逐缓存释放空间
        # NOTE: len(evicted) + free_len >= needed_len
        # 计算需要驱逐的数量
        evicted = self.manager.evict(needed_len - free_len)
        # 合并空闲槽位和驱逐释放的槽位
        merged = torch.cat([self._free_slots, evicted])
        # 确保合并后有足够的空间
        assert len(merged) >= needed_len, "Eviction did not free enough space."

        # 从合并后的列表分配所需数量
        allocated = merged[:needed_len]
        # 更新空闲列表
        self._free_slots = merged[needed_len:]
        return allocated

    def free_and_cache_finished_req(
        self,
        old_handle: BaseCacheHandle,
        input_ids: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        """
        释放请求的缓存并插入到缓存树中
        
        当请求完成 prefill 阶段后 将其 KV Cache 插入到 Radix Tree 中。
        这样可以供后续请求复用相同的前缀。
        
        工作流程：
        1. 将请求的完整序列插入到缓存树中
        2. 释放重复的部分(如果前缀已存在于缓存中)
        3. 解锁旧的缓存句柄
        
        Args:
            old_handle: 请求之前使用的缓存句柄
            input_ids: 请求的完整输入序列 token IDs
            indices: 请求使用的缓存页面索引
        
        详细说明：
        - insert_prefix 返回已存在于缓存中的前缀长度(in_cache_len)
        - 如果 in_cache_len > old_handle.cached_len 说明有部分缓存是重复的
        - 重复的部分(indices[old_handle.cached_len : in_cache_len])需要释放
        - 最后解锁旧的句柄 允许其被驱逐(如果引用计数为0)
        """
        # 将请求的序列插入到缓存树中
        # 返回已存在于缓存中的前缀长度
        in_cache_len = self.manager.insert_prefix(input_ids, indices)
        # 释放重复的缓存页面
        # 如果 in_cache_len > old_handle.cached_len 说明有部分缓存是重复的
        # 这些重复的页面需要释放回空闲列表
        self._free(indices[old_handle.cached_len : in_cache_len])
        # 解锁旧的缓存句柄 允许其被驱逐(如果引用计数为0)
        self.unlock(old_handle)

    def check_integrity(self) -> None:
        """
        检查缓存管理器的完整性
        
        验证缓存状态的一致性 确保：
        1. 底层缓存管理器的完整性
        2. 空闲槽位数量 + 已使用缓存大小 = 总页面数量
        
        这是调试和验证的重要方法 用于检测内存泄漏或状态不一致。
        
        Raises:
            RuntimeError: 如果完整性检查失败
        """
        # 检查底层缓存管理器的完整性
        self.manager.check_integrity()
        # 验证：空闲槽位 + 已使用缓存 = 总页面数
        if len(self._free_slots) + self.manager.size_info.total_size != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_slots({len(self._free_slots)}) +"
                f" total_size({self.manager.size_info.total_size}) != num_pages({self.num_pages})"
            )
