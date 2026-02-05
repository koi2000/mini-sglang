"""
分布式信息管理模块

管理 Tensor Parallelism (TP) 的全局信息。
使用单例模式存储和访问分布式配置 包括当前进程的 rank 和总进程数。

Tensor Parallelism 说明：
- 将模型的不同层或部分分布到多个 GPU 上
- 每个 GPU 处理模型的一部分 通过通信协调完成计算
- rank: 当前进程在 TP 组中的编号(从 0 开始)
- size: TP 组中的总进程数
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedInfo:  # should not export from here
    """
    分布式信息数据类
    
    存储 Tensor Parallelism 的配置信息。
    使用 frozen=True 确保不可变 防止意外修改。
    
    Attributes:
        rank: 当前进程在 TP 组中的编号 范围 [0, size)
        size: TP 组中的总进程数
    
    Note:
        这个类不应该直接从本模块导出 应该通过 get_tp_info() 等函数访问。
    """
    rank: int
    size: int

    def __post_init__(self):
        """
        后初始化验证
        
        验证 rank 和 size 的有效性。
        确保 rank 在有效范围内。
        
        Raises:
            AssertionError: 如果 rank 不在 [0, size) 范围内
        """
        assert 0 <= self.rank < self.size

    def is_primary(self) -> bool:
        """
        判断是否是主进程(rank 0)
        
        在分布式系统中 rank 0 通常负责一些特殊任务 如：
        - 日志输出
        - 进度条显示
        - 某些初始化操作
        
        Returns:
            True 如果当前进程是 rank 0；否则 False
        """
        return self.rank == 0


# 全局单例：存储 TP 信息
# 使用模块级变量实现单例模式
_TP_INFO: DistributedInfo | None = None


def set_tp_info(rank: int, size: int) -> None:
    """
    设置 Tensor Parallelism 信息
    
    在系统初始化时调用 设置全局的 TP 信息。
    只能设置一次 防止重复设置导致的不一致。
    
    Args:
        rank: 当前进程的 rank
        size: TP 组的总进程数
    
    Raises:
        RuntimeError: 如果 TP 信息已经被设置过
    
    使用场景：
        通常在 Engine 初始化时调用 例如：
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
    """
    global _TP_INFO
    # 防止重复设置
    if _TP_INFO is not None:
        raise RuntimeError("TP info has been set")
    _TP_INFO = DistributedInfo(rank, size)


def get_tp_info() -> DistributedInfo:
    """
    获取 Tensor Parallelism 信息
    
    返回全局的 TP 信息。如果信息尚未设置 抛出异常。
    
    Returns:
        DistributedInfo: TP 信息对象
    
    Raises:
        RuntimeError: 如果 TP 信息尚未被设置
    
    使用场景：
        在需要访问 TP 信息的地方调用 例如：
        tp_info = get_tp_info()
        if tp_info.is_primary():
            logger.info("This is rank 0")
    """
    if _TP_INFO is None:
        raise RuntimeError("TP info has not been set")
    return _TP_INFO


def try_get_tp_info() -> DistributedInfo | None:
    """
    尝试获取 Tensor Parallelism 信息
    
    与 get_tp_info() 不同 如果信息尚未设置 返回 None 而不是抛出异常。
    用于可选场景 例如在初始化阶段可能尚未设置 TP 信息。
    
    Returns:
        DistributedInfo | None: TP 信息对象 如果尚未设置则返回 None
    
    使用场景：
        在不确定 TP 信息是否已设置的场景中使用 例如：
        tp_info = try_get_tp_info()
        if tp_info is not None and tp_info.is_primary():
            logger.info("This is rank 0")
    """
    return _TP_INFO


__all__ = ["DistributedInfo", "set_tp_info", "get_tp_info", "try_get_tp_info"]
