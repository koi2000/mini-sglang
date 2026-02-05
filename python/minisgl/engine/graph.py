"""
CUDA Graph 运行器模块

实现 CUDA Graph 的捕获和重放功能 用于优化 decode 阶段的性能。
CUDA Graph 可以捕获 CUDA kernel 的执行序列 然后通过重放来减少 CPU-GPU 通信开销。
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Batch, Req, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import init_logger
from tqdm import tqdm

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend
    from minisgl.models import BaseLLMModel

logger = init_logger(__name__)


def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
    """
    确定 CUDA Graph 的批次大小列表
    
    根据配置和可用内存 确定需要捕获 CUDA Graph 的批次大小列表。
    批次大小列表用于覆盖常见的批次大小 以便后续重放时能够使用预捕获的 graph。
    
    Args:
        cuda_graph_bs: 用户指定的批次大小列表 如果提供则直接使用
        cuda_graph_max_bs: 最大批次大小限制
        free_memory: GPU 可用内存(字节)
    
    Returns:
        批次大小列表 例如 [1, 2, 4, 8, 16, 24, ...]
    
    策略：
    - 如果用户指定了批次大小列表 直接使用
    - 否则根据 GPU 内存自动确定：
      - H200(>80GB): 最大批次大小 256
      - 其他 GPU: 最大批次大小 160
    - 批次大小列表：包含 1, 2, 4 以及 8 的倍数直到最大批次大小
    """
    # 如果用户指定了批次大小列表 直接使用
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    # 将字节转换为 GB
    free_memory_gb = free_memory / (1 << 30)
    
    # 根据 GPU 内存确定最大批次大小
    if cuda_graph_max_bs is None:
        if free_memory_gb > 80:  # H200 等大内存 GPU
            cuda_graph_max_bs = 256
        else:  # 其他 GPU
            cuda_graph_max_bs = 160

    # 如果最大批次大小小于 1 则禁用 CUDA Graph
    if cuda_graph_max_bs < 1:
        return []

    # 生成批次大小列表：包含 1, 2, 4 以及从 8 开始每 8 个递增直到最大值
    # 例如：[1, 2, 4, 8, 16, 24, 32, ...]
    return [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))


def mem_GB(size: int) -> str:
    """
    将字节大小转换为 GB 字符串
    
    Args:
        size: 字节大小
    
    Returns:
        格式化的字符串 例如 "16.00 GiB"
    """
    return f"{size / (1024**3):.2f} GiB"


def get_free_memory(device: torch.device) -> int:
    """
    获取 GPU 可用内存
    
    Args:
        device: PyTorch 设备
    
    Returns:
        可用内存大小(字节)
    """
    return torch.cuda.mem_get_info(device)[0]


class GraphRunner:
    """
    CUDA Graph 运行器
    
    负责捕获和重放 CUDA Graph 用于优化 decode 阶段的性能。
    
    CUDA Graph 的优势：
    - 减少 CPU-GPU 通信开销
    - 提高执行效率
    - 适用于 decode 阶段(因为计算模式相对固定)
    
    工作原理：
    1. 初始化时捕获不同批次大小的 CUDA Graph
    2. 运行时根据批次大小选择对应的 graph 进行重放
    3. 使用内存池优化内存分配
    
    Attributes:
        max_graph_bs: 最大支持的批次大小
        graph_map: 批次大小到 CUDA Graph 的映射字典
        graph_bs_list: 排序后的批次大小列表(用于填充批次)
        logits: 用于存储模型输出的张量(最大批次大小)
        attn_backend: 注意力后端
        dummy_req: 虚拟请求(用于填充批次)
    """
    
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
    ):
        """
        初始化 CUDA Graph 运行器
        
        Args:
            stream: CUDA 流 用于执行 graph
            device: PyTorch 设备
            model: LLM 模型
            attn_backend: 注意力后端
            cuda_graph_bs: 用户指定的批次大小列表
            cuda_graph_max_bs: 最大批次大小限制
            free_memory: GPU 可用内存
            max_seq_len: 最大序列长度
            vocab_size: 词汇表大小
            dummy_req: 虚拟请求 用于填充批次和捕获 graph
        """
        # 确定需要捕获的批次大小列表
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        
        # 如果没有批次大小列表 禁用 CUDA Graph
        if len(cuda_graph_bs) == 0:
            logger.info_rank0("CUDA graph is disabled.")
            self.max_graph_bs = 0
            self.graph_map = {}
            return

        # 对批次大小列表进行去重和排序(从大到小)
        cuda_graph_bs = sorted(set(cuda_graph_bs), reverse=True)
        self.max_graph_bs = max(cuda_graph_bs)
        
        # 分配用于存储模型输出的张量(最大批次大小)
        self.logits = torch.empty(
            (self.max_graph_bs, vocab_size),
            dtype=torch.float16,
            device=device,
        )
        self.attn_backend = attn_backend
        
        # 初始化注意力后端的 graph 捕获准备
        attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=cuda_graph_bs)

        # 同步 GPU 清空缓存 重置内存统计 为捕获做准备
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {cuda_graph_bs}")
        free_memory = get_free_memory(device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        # warm up by capturing a graph and then destroying it
        # 预热：捕获一个 graph 然后销毁 用于初始化 CUDA 状态
        # 这有助于稳定后续的捕获过程
        g = torch.cuda.CUDAGraph()
        batch = Batch(reqs=[dummy_req] * self.max_graph_bs, phase="decode")
        attn_backend.prepare_for_capture(batch)
        with get_global_ctx().forward_batch(batch):
            # 先执行一次 forward 用于预热
            self.logits[:] = model.forward()
            # 捕获 graph
            with torch.cuda.graph(g, stream=stream):
                self.logits[:] = model.forward()
        del g  # 删除临时 graph

        # 开始为每个批次大小捕获 graph
        graph_list: List[Tuple[int, torch.cuda.CUDAGraph]] = []
        pbar = tqdm(
            cuda_graph_bs,
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # 只在主 rank 显示进度条
        )

        pool = None  # 内存池 用于优化内存分配
        for bs in pbar:
            # 更新进度条显示当前批次大小和可用内存
            free_memory = get_free_memory(device)
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            pbar.refresh()
            
            g = torch.cuda.CUDAGraph()
            
            # 如果不是最大批次大小 需要准备新的批次
            if bs != self.max_graph_bs:
                batch = Batch(reqs=[dummy_req] * bs, phase="decode")
                self.attn_backend.prepare_for_capture(batch)
            
            # 捕获 graph
            with get_global_ctx().forward_batch(batch):
                # 先执行一次 forward
                self.logits[:bs] = model.forward()
                # 捕获 graph(使用内存池优化)
                with torch.cuda.graph(g, pool=pool, stream=stream):
                    self.logits[:bs] = model.forward()
            
            # 从第一个 graph 获取内存池 后续 graph 共享这个内存池
            if pool is None:
                pool = g.pool()
            
            # 保存 graph
            graph_list.append((bs, g))

        free_memory = get_free_memory(device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

        # 构建批次大小到 graph 的映射字典
        self.graph_map = dict(graph_list)
        # 保存排序后的批次大小列表(从小到大) 用于填充批次
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        """
        判断是否可以使用 CUDA Graph
        
        CUDA Graph 只能在以下条件满足时使用：
        1. 批次处于 decode 阶段(计算模式相对固定)
        2. 批次大小不超过最大支持的批次大小
        
        Args:
            batch: 批次对象
        
        Returns:
            True 如果可以使用 CUDA Graph；否则 False
        """
        return batch.is_decode and batch.size <= self.max_graph_bs

    def replay(self, batch: Batch) -> torch.Tensor:
        """
        重放 CUDA Graph
        
        使用预捕获的 CUDA Graph 执行 forward 过程。
        这比直接调用 forward 更高效 因为减少了 CPU-GPU 通信开销。
        
        Args:
            batch: 批次对象
        
        Returns:
            模型输出的 logits 形状为 (batch.size, vocab_size)
        
        Raises:
            AssertionError: 如果批次不能使用 CUDA Graph
        """
        assert self.can_use_cuda_graph(batch)
        # 根据填充后的批次大小选择对应的 graph
        g = self.graph_map[batch.padded_size]
        # 准备重放(更新输入数据等)
        self.attn_backend.prepare_for_replay(batch)
        # 重放 graph
        g.replay()
        # 返回实际批次大小的 logits(不包括填充的虚拟请求)
        return self.logits[: batch.size]

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    def destroy_cuda_graphs(self) -> None:
        """
        销毁 CUDA Graph
        
        释放所有捕获的 CUDA Graph 占用的资源。
        
        重要：必须在释放 NCCL 资源之前调用此方法 否则可能导致程序挂起。
        这是因为 CUDA Graph 可能持有某些需要在 NCCL 资源释放前清理的资源。
        """
        del self.graph_map
        gc.collect()  # 强制垃圾回收

    def pad_batch(self, batch: Batch) -> int:
        """
        填充批次到合适的 CUDA Graph 批次大小
        
        为了使用 CUDA Graph 需要将批次填充到预捕获的批次大小。
        选择第一个大于等于当前批次大小的预捕获批次大小。
        
        Args:
            batch: 批次对象 将被修改(添加 padded_reqs)
        
        Returns:
            填充的虚拟请求数量
        
        示例：
            如果 graph_bs_list = [1, 2, 4, 8, 16]
            当前 batch.size = 5
            则选择 8 填充 3 个虚拟请求
        """
        # 如果可以使用 CUDA Graph 选择第一个大于等于当前批次大小的预捕获批次大小
        # 否则不填充
        padded_size = (  # choose the first available batch size
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if self.can_use_cuda_graph(batch)
            else batch.size
        )
        # 使用虚拟请求填充批次
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)
        return batch.padded_size - batch.size
