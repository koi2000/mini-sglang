from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from .base import BaseCacheHandle, BaseCacheManager, SizeInfo

'''
RadixTreeNode 表示radix tree的结点
'''
class RadixTreeNode:
    counter: int = 0

    def __init__(self, tic: int | None = None) -> None:
        # children 负责记录当前节点的子节点
        self.children: Dict[int, RadixTreeNode] = {}
        # 记录自己的父节点
        self._parent: RadixTreeNode | None = None
        # 引用计数
        self.ref_count: int = 0
        # static变量用于记录当前节点的唯一id
        self.uuid = RadixTreeNode.counter
        RadixTreeNode.counter += 1
        # 记录当前节点被访问的时间
        self.timestamp = tic or time.monotonic_ns()

        # these fields should be updated later
        # 记录当前节点的key
        self._key: torch.Tensor
        # 记录当前节点的value
        self._value: torch.Tensor
        # 记录当前节点的长度
        self._length: int

    def set_key_value(self, key: torch.Tensor, value: torch.Tensor) -> None:
        assert len(key) == len(value)
        self._key = key
        self._value = value
        self._length = len(key)

    def set_parent(self, parent: RadixTreeNode) -> None:
        self._parent = parent
        parent.children[int(self._key[0].item())] = self

    @property
    def length(self) -> int:
        return self._length

    @property
    def parent(self) -> RadixTreeNode:
        assert self._parent is not None
        return self._parent

    @property
    def value(self) -> torch.Tensor:
        return self._value

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_match_len(self, input_ids: torch.Tensor) -> int:
        from minisgl.kernel import fast_compare_key

        # compare key and input_ids, find the first diff
        return fast_compare_key(self._key, input_ids)

    def _split_at(self, pos: int) -> RadixTreeNode:
        ''''
        将当前节点从pos位置分割成两个节点
        当前节点的key变成前pos个
        新节点的key为当前key的pos+1到最后
        新建的节点会变成parent
        '''
        assert 0 < pos < self.length
        parent = self.parent
        # 创建一个新的节点
        new_node = RadixTreeNode(self.timestamp)
        # 设置新的节点的key和value, 从当前节点的后pos个字符开始
        new_node.set_key_value(self._key[:pos], self._value[:pos])
        # 设置新的节点的父节点
        new_node.set_parent(parent)
        # 设置新的节点的引用计数
        new_node.ref_count = self.ref_count

        self.set_key_value(self._key[pos:], self._value[pos:])
        self.set_parent(new_node)

        return new_node

    def __lt__(self, other: RadixTreeNode) -> bool:
        return self.timestamp < other.timestamp


@dataclass(frozen=True)
class RadixCacheHandle(BaseCacheHandle):
    node: RadixTreeNode


class RadixCacheManager(BaseCacheManager):
    """
    Radix Tree 实现的 KV Cache 管理器
    
    使用 Radix Tree(基数树)数据结构来高效存储和管理 KV Cache。
    主要功能包括：
    1. 前缀匹配：快速查找输入序列的最长匹配前缀
    2. 前缀插入：将新的序列插入到树中
    3. 缓存保护：通过引用计数保护正在使用的缓存
    4. 缓存驱逐：使用 LRU 策略驱逐不常用的缓存节点
    """
    
    def __init__(self, device: torch.device):
        """
        初始化 Radix Cache Manager
        
        Args:
            device: PyTorch 设备 用于创建张量
        """
        self.device = device
        # 创建一个空的张量 用于返回空结果
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        super().__init__()
        # 创建根节点 根节点是树的起点
        self.root_node = RadixTreeNode()
        # 根节点的引用计数设为1 确保根节点始终被保护 不会被驱逐
        self.root_node.ref_count = 1  # root is always protected
        # 可驱逐的缓存大小(引用计数为0的节点)
        self.evictable_size = 0
        # 受保护的缓存大小(引用计数>0的节点)
        self.protected_size = 0

    # 并非进行了足够的抽象 仍默认使用radix 
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """
        锁定或解锁缓存句柄
        
        通过修改从节点到根节点路径上所有节点的引用计数来保护/取消保护缓存。
        当引用计数>0时 节点受保护 不会被驱逐; 当引用计数=0时 节点可被驱逐。
        
        Args:
            handle: 要锁定/解锁的缓存句柄
            unlock: 如果为True 则解锁句柄(减少引用计数); 如果为False 则锁定句柄(增加引用计数)
        
        工作流程：
        - 锁定(unlock=False)：从节点向上遍历到根节点 增加路径上所有节点的引用计数
        - 解锁(unlock=True)：从节点向上遍历到根节点 减少路径上所有节点的引用计数
        """
        assert isinstance(handle, RadixCacheHandle)
        node = handle.node
        
        if unlock:
            # 解锁：减少引用计数 使节点可能被驱逐
            while not node.is_root():
                node = node.parent
                node.ref_count -= 1
                assert node.ref_count > 0  # 确保引用计数不会变成负数
                # 如果引用计数变为0 节点从受保护变为可驱逐
                if node.ref_count == 0:
                    self.evictable_size += node.length
                    self.protected_size -= node.length
        else:
            # 锁定：增加引用计数 保护节点不被驱逐
            while not node.is_root():
                node = node.parent
                # 如果节点之前是可驱逐的(ref_count=0) 现在变为受保护的
                if node.ref_count == 0:
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1

    def match_prefix(self, input_ids: torch.Tensor) -> Tuple[RadixCacheHandle, torch.Tensor]:
        """
        匹配输入序列的最长前缀
        
        在 Radix Tree 中查找与 input_ids 匹配的最长前缀 并返回匹配的缓存句柄和对应的索引。
        这个操作是只读的 不会修改缓存结构。
        
        Args:
            input_ids: 输入序列的 token IDs 形状为 (seq_len,)
        
        Returns:
            handle: 匹配前缀的缓存句柄 包含匹配长度和对应的节点
            indices: 匹配前缀在缓存中的索引 形状为 (matched_len,)
                    如果没有任何匹配 返回空张量
        
        算法：
        1. 使用 _walk 方法在树中查找最长匹配前缀
        2. 从匹配节点向上遍历到根节点 收集所有节点的 value(索引)
        3. 将收集的索引反转并拼接 得到完整的前缀索引序列
        """
        # 在树中查找最长匹配前缀
        node, prefix_len = self._walk(input_ids)
        
        # 如果没有匹配到任何前缀(prefix_len=0) 返回空结果
        if prefix_len == 0:
            assert node.is_root() and node is self.root_node and prefix_len == 0
            return RadixCacheHandle(prefix_len, node), self.empty_tensor
        
        # 从匹配节点向上遍历到根节点 收集所有节点的 value(索引)
        value_list: List[torch.Tensor] = []
        while not node.is_root():
            value_list.append(node.value)
            node = node.parent
        
        # 反转列表 因为是从叶子到根收集的 需要从根到叶子的顺序
        value_list.reverse()
        # 拼接所有索引 得到完整的前缀索引序列
        return RadixCacheHandle(prefix_len, node), torch.cat(value_list)

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> int:
        """
        将新的前缀插入到缓存中
        
        在 Radix Tree 中插入新的序列前缀。如果前缀已经存在 则不插入。
        
        Args:
            input_ids: 要插入的输入序列 token IDs 形状为 (seq_len,)
            indices: 要存储的缓存索引 形状为 (seq_len,) 与 input_ids 一一对应
        
        Returns:
            int: 已经存在于缓存中的前缀长度。调用者应该释放这些索引 因为它们没有被使用。
        
        算法：
        1. 使用 _walk 查找最长匹配前缀
        2. 如果还有未匹配的部分 创建新节点并插入到树中
        3. 新插入的节点默认是可驱逐的(ref_count=0)
        """
        # 查找最长匹配前缀
        node, prefix_len = self._walk(input_ids)
        assert prefix_len <= len(input_ids)
        
        # 如果还有未匹配的部分 需要插入新节点
        if prefix_len < len(input_ids):
            # 创建新节点 存储未匹配的部分
            new_node = RadixTreeNode()
            new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:])
            new_node.set_parent(node)
            # 新节点默认是可驱逐的
            self.evictable_size += new_node.length
        
        return prefix_len

    def _walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
        """
        在 Radix Tree 中遍历 查找与 input_ids 匹配的最长前缀
        
        这是核心的树遍历算法 从根节点开始 沿着树向下查找最长匹配的前缀。
        
        Args:
            input_ids: 要匹配的输入序列 token IDs 形状为 (seq_len,)
        
        Returns:
            node: 匹配结束的节点(可能是部分匹配)
            prefix_len: 匹配的前缀长度
        
        算法流程：
        1. 从根节点开始
        2. 对于每个输入 token 查找对应的子节点
        3. 如果找到子节点 计算匹配长度(可能只匹配节点的一部分)
        4. 如果只匹配了节点的一部分 需要分割节点
        5. 更新访问时间戳(用于 LRU 驱逐策略)
        """
        prefix_len = 0  # 已匹配的前缀长度
        indice_len = len(input_ids)  # 输入序列的总长度
        node = self.root_node  # 从根节点开始
        tic = time.monotonic_ns()  # 当前时间戳 用于更新访问时间

        # 继续匹配直到处理完所有输入
        while prefix_len < indice_len:
            # 获取当前要匹配的 token ID
            this_id = int(input_ids[prefix_len].item())
            
            # 如果当前节点没有对应的子节点 匹配结束
            if this_id not in node.children:
                return node, prefix_len

            # 移动到子节点
            node = node.children[this_id]

            # 计算当前节点与剩余输入序列的匹配长度
            # NOTE: at least 1 char is matched, so match_len >= 1
            match_len = node.get_match_len(input_ids[prefix_len:])
            prefix_len += match_len

            # 如果只匹配了节点的一部分 需要分割节点
            # 例如：节点存储的是 [1,2,3,4] 但只匹配了 [1,2]
            # 需要将节点分割成 [1,2] 和 [3,4] 两个节点
            if match_len != node.length:
                node = node._split_at(match_len)
                return node, prefix_len

            # 完全匹配了当前节点 更新访问时间戳(用于 LRU 驱逐策略)
            node.timestamp = tic

        # 完全匹配了整个输入序列
        return node, prefix_len

    def evict(self, size: int) -> torch.Tensor:
        """
        驱逐缓存以释放空间
        
        使用 LRU(Least Recently Used)策略驱逐最久未使用的缓存节点。
        只驱逐引用计数为0的叶子节点(可驱逐节点)。
        
        Args:
            size: 需要驱逐的缓存大小
        
        Returns:
            torch.Tensor: 被驱逐的缓存索引 形状为 (evicted_size,)
                         实际驱逐的大小可能大于请求的大小(因为按节点为单位驱逐)
        
        Raises:
            AssertionError: 如果请求的驱逐大小超过可驱逐的大小 或者无法收集足够的节点
        
        算法：
        1. 收集所有可驱逐的叶子节点(ref_count=0 且是叶子节点)
        2. 使用最小堆(按时间戳排序)选择最久未使用的节点
        3. 逐个驱逐节点 直到达到目标大小
        4. 如果父节点在子节点被驱逐后变成叶子节点且可驱逐 也加入堆中
        """
        # 如果不需要驱逐 返回空张量
        if size == 0:
            return self.empty_tensor
        
        # 确保请求的驱逐大小不超过可驱逐的大小
        assert (
            size <= self.evictable_size
        ), f"Cannot evict {size}, only {self.evictable_size} is evictable"

        # 收集所有可驱逐的叶子节点
        leave_nodes = self._collect_leave_nodes_for_evict()
        # 使用最小堆 按时间戳排序(最久未使用的节点在堆顶)
        heapq.heapify(leave_nodes)
        evicted_indices: List[torch.Tensor] = []  # 存储被驱逐的索引
        evicted_size = 0  # 已驱逐的大小

        # 继续驱逐直到达到目标大小
        while evicted_size < size:
            # 确保还有节点可以驱逐
            assert (
                leave_nodes
            ), f"Cannot evict enough cache, need {size}, only {evicted_size} evicted"
            
            # 从堆中取出最久未使用的节点(时间戳最小的节点)
            node = heapq.heappop(leave_nodes)
            # 确保节点是可驱逐的叶子节点 且不是根节点
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            
            # 记录被驱逐的节点信息
            evicted_size += node.length
            evicted_indices.append(node.value)
            self.evictable_size -= node.length
            
            # 从父节点的子节点字典中删除当前节点
            parent = node.parent
            del parent.children[int(node._key[0].item())]
            
            # 如果父节点在删除子节点后变成叶子节点 且也是可驱逐的 加入堆中
            # NOTE: root is always protected, so won't be evicted
            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leave_nodes, parent)

        # 拼接所有被驱逐的索引并返回
        return torch.cat(evicted_indices)

    def _collect_leave_nodes_for_evict(self) -> List[RadixTreeNode]:
        """
        收集所有可以驱逐的叶子节点
        
        遍历整个 Radix Tree 找出所有满足以下条件的节点：
        1. 是叶子节点(没有子节点)
        2. 引用计数为0(未被锁定保护)
        
        Returns:
            List[RadixTreeNode]: 可驱逐的叶子节点列表
        
        算法：
        使用深度优先搜索(DFS)遍历整个树 收集符合条件的叶子节点
        """
        nodes: List[RadixTreeNode] = [self.root_node]  # 待遍历的节点栈
        leave_nodes: List[RadixTreeNode] = []  # 可驱逐的叶子节点列表

        # 深度优先搜索遍历整个树
        while len(nodes) > 0:
            node = nodes.pop()
            
            # 如果是叶子节点且可驱逐 加入列表
            if node.is_leaf():
                if node.ref_count == 0:
                    leave_nodes.append(node)
            else:
                # 如果不是叶子节点 将其所有子节点加入栈中继续遍历
                for child in node.children.values():
                    nodes.append(child)

        return leave_nodes

    def reset(self) -> None:
        """
        重置缓存管理器
        
        清空所有缓存数据 恢复到初始状态。
        当前未实现。
        """
        raise NotImplementedError("RadixManager.reset is not implemented")

    @property
    def size_info(self) -> SizeInfo:
        """
        获取缓存大小信息
        
        Returns:
            SizeInfo: 包含可驱逐大小和受保护大小的信息对象
        """
        return SizeInfo(
            evictable_size=self.evictable_size,
            protected_size=self.protected_size,
        )

    def check_integrity(self) -> None:
        """
        检查缓存的完整性
        
        验证 Radix Tree 的结构是否正确 例如：
        - 引用计数是否正确
        - 父子关系是否一致
        - 大小统计是否准确
        
        当前未实现。
        """
        pass
