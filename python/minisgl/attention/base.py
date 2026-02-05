"""
注意力机制基础模块

定义注意力机制的标准接口和抽象类。
支持不同的注意力实现(如 FlashAttention、FlashInfer 等) 并提供混合后端用于优化不同阶段。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import torch
    from minisgl.core import Batch


@dataclass
class BaseAttnMetadata(ABC):
    """
    注意力元数据基类
    
    存储注意力计算所需的元数据信息 如位置信息等。
    不同的注意力实现可以定义自己的元数据子类。
    
    Attributes:
        positions: 位置张量 用于记录每个 token 的位置信息
    """
    positions: torch.Tensor

    @abstractmethod
    def get_last_indices(self, bs: int) -> torch.Tensor:
        """
        获取批次中每个请求的最后一个 token 的索引
        
        用于在 decode 阶段定位每个请求的最后一个 token 位置。
        
        Args:
            bs: 批次大小
        
        Returns:
            每个请求最后一个 token 的索引张量 形状为 (bs,)
        """
        ...


class BaseAttnBackend(ABC):
    """
    注意力后端抽象基类
    
    定义注意力机制的标准接口 包括：
    - Forward 计算
    - 元数据准备
    - CUDA Graph 支持
    
    不同的注意力实现(如 FlashAttention、FlashInfer)需要实现这些接口。
    """
    
    @abstractmethod
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        """
        执行注意力计算
        
        这是注意力机制的核心方法 执行 QKV 注意力计算。
        
        Args:
            q: Query 张量 形状通常为 (batch_size, num_heads, seq_len, head_dim)
            k: Key 张量 形状通常为 (batch_size, num_heads, seq_len, head_dim)
            v: Value 张量 形状通常为 (batch_size, num_heads, seq_len, head_dim)
            layer_id: 当前层的 ID 用于访问对应的 KV Cache
            batch: 批次对象 包含请求信息和元数据
        
        Returns:
            注意力输出张量 形状通常为 (batch_size, num_heads, seq_len, head_dim)
        """
        ...

    @abstractmethod
    def prepare_metadata(self, batch: Batch) -> None:
        """
        准备注意力元数据
        
        在 forward 之前调用 准备注意力计算所需的元数据。
        元数据会被存储在 batch.attn_metadata 中。
        
        Args:
            batch: 批次对象 其 attn_metadata 字段会被设置
        """
        ...

    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        """
        初始化 CUDA Graph 捕获
        
        为指定的批次大小列表初始化 CUDA Graph 捕获环境。
        这通常在系统初始化时调用一次。
        
        Args:
            max_seq_len: 最大序列长度
            bs_list: 需要捕获 CUDA Graph 的批次大小列表
        """
        ...

    @abstractmethod
    def prepare_for_capture(self, batch: Batch) -> None:
        """
        准备 CUDA Graph 捕获
        
        在捕获 CUDA Graph 之前调用 准备批次数据。
        确保批次数据格式符合 CUDA Graph 捕获的要求。
        
        Args:
            batch: 用于捕获 CUDA Graph 的批次对象
        """
        ...

    @abstractmethod
    def prepare_for_replay(self, batch: Batch) -> None:
        """
        准备 CUDA Graph 重放
        
        在重放 CUDA Graph 之前调用 更新批次数据。
        确保批次数据格式符合 CUDA Graph 重放的要求。
        
        Args:
            batch: 用于重放 CUDA Graph 的批次对象
        """
        ...


class HybridBackend(BaseAttnBackend):
    """
    混合注意力后端
    
    组合两个不同的注意力后端：
    - prefill_backend: 用于 prefill 阶段(处理初始输入序列)
    - decode_backend: 用于 decode 阶段(逐个生成 token)
    
    这种设计允许为不同阶段使用最优的注意力实现：
    - Prefill 阶段：通常使用 FlashAttention 等优化实现处理长序列
    - Decode 阶段：通常使用 FlashInfer 等优化实现处理单 token 生成
    
    优势：
    - 不同阶段可以使用不同的优化策略
    - 提高整体性能
    - 保持接口统一
    """
    
    def __init__(
        self,
        prefill_backend: BaseAttnBackend,
        decode_backend: BaseAttnBackend,
    ) -> None:
        """
        初始化混合后端
        
        Args:
            prefill_backend: Prefill 阶段的注意力后端
            decode_backend: Decode 阶段的注意力后端
        """
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        """
        执行注意力计算
        
        根据批次阶段选择对应的后端执行计算。
        
        Args:
            q: Query 张量
            k: Key 张量
            v: Value 张量
            layer_id: 层 ID
            batch: 批次对象
        
        Returns:
            注意力输出张量
        """
        # 根据批次阶段选择对应的后端
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.forward(q, k, v, layer_id, batch)

    def prepare_metadata(self, batch: Batch) -> None:
        """
        准备注意力元数据
        
        根据批次阶段选择对应的后端准备元数据。
        
        Args:
            batch: 批次对象
        """
        # 根据批次阶段选择对应的后端
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.prepare_metadata(batch)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        """
        初始化 CUDA Graph 捕获
        
        只对 decode 后端初始化 CUDA Graph 因为通常只有 decode 阶段使用 CUDA Graph。
        
        Args:
            max_seq_len: 最大序列长度
            bs_list: 批次大小列表
        """
        # 只对 decode 后端初始化 CUDA Graph
        # Prefill 阶段通常不使用 CUDA Graph(因为序列长度变化较大)
        self.decode_backend.init_capture_graph(max_seq_len, bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        """
        准备 CUDA Graph 捕获
        
        只对 decode 后端准备捕获。
        
        Args:
            batch: 批次对象
        """
        # 只对 decode 后端准备捕获
        self.decode_backend.prepare_for_capture(batch)

    def prepare_for_replay(self, batch: Batch) -> None:
        """
        准备 CUDA Graph 重放
        
        只对 decode 后端准备重放。
        
        Args:
            batch: 批次对象
        """
        # 只对 decode 后端准备重放
        self.decode_backend.prepare_for_replay(batch)
