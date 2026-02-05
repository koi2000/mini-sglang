"""
分布式通信实现模块

提供分布式通信的抽象接口和多种实现方式。
支持 Tensor Parallelism 中的 all_reduce 和 all_gather 操作。

通信操作说明：
- all_reduce: 所有进程对张量执行归约操作(如求和) 结果广播到所有进程
- all_gather: 收集所有进程的张量 拼接后返回给所有进程

实现方式：
1. TorchDistributedImpl: 使用 PyTorch 原生分布式通信
2. PyNCCLDistributedImpl: 使用 PyNCCL 进行高性能通信(可选)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from minisgl.distributed import DistributedInfo
    from minisgl.kernel import PyNCCLCommunicator


@dataclass
class DistributedImpl(ABC):
    """
    分布式通信实现的抽象基类
    
    定义分布式通信的标准接口 包括 all_reduce 和 all_gather 操作。
    不同的实现类可以提供不同的通信后端。
    """
    
    @abstractmethod
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行 all_reduce 操作
        
        所有进程对输入张量执行归约操作(通常是求和) 结果广播到所有进程。
        操作是 in-place 的 输入张量会被修改。
        
        Args:
            x: 输入张量 形状为 (batch_size, ...)
        
        Returns:
            归约后的张量 形状与输入相同
        
        示例：
            如果有 4 个进程 每个进程有张量 [1, 2, 3]
            all_reduce 后所有进程都得到 [4, 8, 12] (求和)
        """
        ...

    @abstractmethod
    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行 all_gather 操作
        
        收集所有进程的输入张量 在第一个维度上拼接后返回给所有进程。
        
        Args:
            x: 输入张量 形状为 (batch_size, ...)
        
        Returns:
            拼接后的张量 形状为 (batch_size * world_size, ...)
        
        示例：
            如果有 4 个进程 每个进程有张量 [1, 2]
            all_gather 后所有进程都得到 [1, 2, 1, 2, 1, 2, 1, 2]
        """
        ...


@dataclass
class TorchDistributedImpl(DistributedImpl):
    """
    基于 PyTorch 的分布式通信实现
    
    使用 PyTorch 原生的分布式通信 API 实现 all_reduce 和 all_gather。
    这是默认的实现方式 适用于大多数场景。
    """
    
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """
        使用 PyTorch 的 all_reduce 实现
        
        如果只有一个进程 直接返回输入张量。
        否则对所有进程的张量执行求和归约。
        
        Args:
            x: 输入张量
        
        Returns:
            归约后的张量
        """
        tp_size = dist.get_world_size()
        # 单进程情况 无需通信
        if tp_size == 1:
            return x
        # 执行 all_reduce 操作(求和)
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        """
        使用 PyTorch 的 all_gather 实现
        
        如果只有一个进程 直接返回输入张量。
        否则收集所有进程的张量并在第一个维度上拼接。
        
        Args:
            x: 输入张量 形状为 (batch_size, ...)
        
        Returns:
            拼接后的张量 形状为 (batch_size * world_size, ...)
        """
        tp_size = dist.get_world_size()
        # 单进程情况 无需通信
        if tp_size == 1:
            return x
        
        # 计算输出形状：第一个维度乘以进程数
        shape = list(x.shape)
        shape[0] = shape[0] * tp_size
        # 创建输出张量
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        # 执行 all_gather 操作
        dist.all_gather_into_tensor(out, x)
        return out


@dataclass
class PyNCCLDistributedImpl(DistributedImpl):
    """
    基于 PyNCCL 的分布式通信实现
    
    使用 PyNCCL 进行高性能的分布式通信。
    PyNCCL 是 NVIDIA 的 NCCL 库的 Python 绑定 提供更好的性能。
    
    Attributes:
        comm: PyNCCL 通信器对象
    """
    
    comm: PyNCCLCommunicator

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """
        使用 PyNCCL 的 all_reduce 实现
        
        Args:
            x: 输入张量
        
        Returns:
            归约后的张量
        
        Note:
            PyNCCL 的 all_reduce 是 in-place 操作 直接修改输入张量。
        """
        self.comm.all_reduce(x, "sum")
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        """
        使用 PyNCCL 的 all_gather 实现
        
        Args:
            x: 输入张量 形状为 (batch_size, ...)
        
        Returns:
            拼接后的张量 形状为 (batch_size * world_size, ...)
        """
        from .info import get_tp_info

        # 获取进程数
        world_size = get_tp_info().size
        # 计算输出形状
        output_shape = list(x.shape)
        output_shape[0] *= world_size
        # 创建输出张量
        result = x.new_empty(output_shape)
        # 执行 all_gather 操作
        self.comm.all_gather(result, x)
        return result


class DistributedCommunicator:
    """
    分布式通信器
    
    使用插件模式管理不同的分布式通信实现。
    支持多个实现 使用最后一个(通常是最高优先级的)实现。
    
    Attributes:
        plugins: 分布式通信实现列表 默认包含 TorchDistributedImpl
    """
    
    plugins: List[DistributedImpl] = [TorchDistributedImpl()]

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行 all_reduce 操作
        
        使用插件列表中的最后一个实现(通常是最高优先级的)。
        
        Args:
            x: 输入张量
        
        Returns:
            归约后的张量
        """
        return self.plugins[-1].all_reduce(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行 all_gather 操作
        
        使用插件列表中的最后一个实现(通常是最高优先级的)。
        
        Args:
            x: 输入张量
        
        Returns:
            拼接后的张量
        """
        return self.plugins[-1].all_gather(x)


def enable_pynccl_distributed(
    tp_info: DistributedInfo, tp_cpu_group: torch.distributed.ProcessGroup, max_bytes: int
) -> None:
    """
    启用基于 PyNCCL 的分布式通信
    
    初始化 PyNCCL 通信器并将其添加到插件列表。
    如果启用 PyNCCL 它会成为最高优先级的实现(在列表末尾)。
    
    Args:
        tp_info: Tensor Parallelism 信息
        tp_cpu_group: CPU 进程组 用于同步
        max_bytes: 通信的最大字节数 用于预分配缓冲区
    
    Note:
        - 如果只有一个进程 直接返回(无需通信)
        - PyNCCL 实现会被添加到插件列表末尾 成为默认实现
        - 这允许在运行时切换通信后端
    """
    # 单进程情况 无需通信
    if tp_info.size == 1:
        return
    
    from minisgl.kernel import init_pynccl

    # 初始化 PyNCCL 通信器
    comm = init_pynccl(
        tp_rank=tp_info.rank,
        tp_size=tp_info.size,
        tp_cpu_group=tp_cpu_group,
        max_size_bytes=max_bytes,
    )

    # 将 PyNCCL 实现添加到插件列表末尾(成为默认实现)
    DistributedCommunicator.plugins.append(PyNCCLDistributedImpl(comm))


def destroy_distributed() -> None:
    """
    销毁所有分布式通信插件
    
    清理所有已注册的分布式通信实现。
    通常在程序退出时调用 确保资源正确释放。
    """
    DistributedCommunicator.plugins = []
