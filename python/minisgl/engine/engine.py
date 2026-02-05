"""
推理引擎模块

实现 LLM 推理引擎的核心功能 包括模型加载、KV Cache 管理、分布式通信等。
这是整个推理系统的核心组件 负责协调各个模块的工作。
"""

from __future__ import annotations

from datetime import timedelta
from typing import Dict, NamedTuple, Tuple

import torch
from minisgl.attention import create_attention_backend
from minisgl.core import Batch, Context, Req, set_global_ctx
from minisgl.distributed import destroy_distributed, enable_pynccl_distributed, set_tp_info
from minisgl.kvcache import create_kvcache
from minisgl.layers import set_rope_device
from minisgl.models import create_model, load_hf_weight
from minisgl.utils import divide_even, init_logger, torch_dtype

from .config import EngineConfig
from .graph import GraphRunner, get_free_memory, mem_GB
from .sample import BatchSamplingArgs, Sampler

logger = init_logger(__name__)


class ForwardOutput(NamedTuple):
    """
    Forward 输出结果
    
    包含模型 forward 后生成的 token 和同步事件。
    
    Attributes:
        next_tokens_gpu: 在 GPU 上的下一个 token IDs 形状为 (batch_size,)
        next_tokens_cpu: 在 CPU 上的下一个 token IDs(用于后续处理)
        copy_done_event: CUDA 事件 用于同步 CPU-GPU 数据传输完成
    """
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event


def create_page_table(shape: Tuple[int, int], device: torch.device) -> torch.Tensor:
    """
    创建页表
    
    页表用于管理 KV Cache 的物理存储位置。
    每个请求在页表中有一个条目 记录其 KV Cache 的页面索引。
    
    Args:
        shape: 页表形状 (max_running_req + 1, max_seq_len)
        device: PyTorch 设备
    
    Returns:
        初始化为零的页表张量 形状为 shape
    """
    return torch.zeros(shape, dtype=torch.int32, device=device)


def _align_up_32(num: int) -> int:
    """
    将数字向上对齐到 32 的倍数
    
    用于内存对齐优化 提高访问效率。
    
    Args:
        num: 要对齐的数字
    
    Returns:
        对齐后的数字(向上取整到 32 的倍数)
    
    示例：
        _align_up_32(1) = 32
        _align_up_32(32) = 32
        _align_up_32(33) = 64
    """
    return (num + 31) // 32 * 32


class Engine:
    """
    LLM 推理引擎
    
    负责管理整个推理系统的核心组件 包括：
    - 模型加载和初始化
    - KV Cache 和页表管理
    - 分布式通信
    - CUDA Graph 支持
    - Forward 和采样流程
    
    Attributes:
        model: LLM 模型
        kv_cache: KV Cache 对象
        page_table: 页表 用于管理 KV Cache 的物理存储位置
        attn_backend: 注意力后端
        ctx: 全局上下文
        sampler: 采样器
        graph_runner: CUDA Graph 运行器
        dummy_req: 虚拟请求 用于填充批次和捕获 CUDA Graph
    """
    
    def __init__(self, config: EngineConfig):
        """
        初始化推理引擎
        
        Args:
            config: 引擎配置对象 包含模型路径、分布式信息等
        """
        self.model_config = config.model_config
        # 设置 Tensor Parallelism 信息
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)

        # 确保 CUDA 尚未初始化(避免冲突)
        assert not torch.cuda.is_initialized()
        # 设置设备(根据 TP rank)
        self.device = torch.device(f"cuda:{config.tp_info.rank}")
        torch.cuda.set_device(self.device)
        # 创建专用的 CUDA 流
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype

        # 初始化分布式通信
        self.tp_cpu_group = self._init_communication(config)
        # 获取初始可用内存(在所有 TP rank 上同步)
        init_free_memory = self._sync_get_memory()[1]
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        # load model and determine number of pages
        # 设置 RoPE 设备
        set_rope_device(self.device)
        # 使用 meta 设备创建模型(不分配实际内存 只创建结构)
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config.model_path, config.model_config)
        # 加载模型权重
        self.model.load_state_dict(self._load_weight_state_dict(config))
        # 确定 KV Cache 的页面数量
        self.num_pages = self.dummy_page = self._determine_num_pages(init_free_memory, config)
        
        # 创建 KV Cache(+1 用于虚拟页面)
        self.kv_cache = create_kvcache(
            num_layers=self.model_config.num_layers,
            num_kv_heads=self.model_config.num_kv_heads,
            num_pages=self.num_pages + 1,  # +1 for dummy page
            head_dim=self.model_config.head_dim,
            device=self.device,
            dtype=self.dtype,
        )
        
        # NOTE: make page table 128 aligned (32 * sizeof(int32) == 128 bytes)
        # 对齐最大序列长度到 32 的倍数(内存对齐优化)
        self.max_seq_len = _align_up_32(min(config.max_seq_len, self.num_pages))
        
        # 创建页表(+1 用于虚拟请求)
        self.page_table = create_page_table(  # + 1 for dummy request
            (config.max_running_req + 1, self.max_seq_len),
            device=self.device,
        )
        
        # 创建注意力后端
        self.attn_backend = create_attention_backend(
            config.model_config,
            self.kv_cache,
            config.attention_backend,
            self.page_table,
        )
        
        # 创建全局上下文
        self.ctx = Context(
            page_size=1,
            kv_cache=self.kv_cache,
            attn_backend=self.attn_backend,
            page_table=self.page_table,
        )
        set_global_ctx(self.ctx)
        
        # 创建采样器
        self.sampler = Sampler(self.device)

        # 获取初始化后的可用内存
        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        # cuda graph related
        # 创建虚拟请求 用于填充批次和捕获 CUDA Graph
        self.dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,  # 使用最后一个页表索引
            cached_len=0,
            output_len=1,
            uid=-1,  # 虚拟 UID
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )
        # 将虚拟请求的页表条目填充为虚拟页面索引
        self.page_table[self.dummy_req.table_idx].fill_(self.dummy_page)
        
        # 创建 CUDA Graph 运行器
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=init_free_memory,
            max_seq_len=self.max_seq_len,
            vocab_size=self.model_config.vocab_size,
            dummy_req=self.dummy_req,
        )

    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        """
        初始化分布式通信
        
        根据配置选择通信后端：
        - 单 rank 或使用 pynccl: 使用 gloo 后端
        - 多 rank: 使用 nccl 后端(用于 GPU 通信) 同时创建 gloo 组(用于 CPU 通信)
        
        Args:
            config: 引擎配置
        
        Returns:
            CPU 进程组 用于 CPU 侧的同步操作
        """
        # 单 rank 或使用 pynccl 时使用 gloo 后端
        if config.tp_info.size == 1 or config.use_pynccl:
            torch.distributed.init_process_group(
                backend="gloo",  # Gloo 后端 支持 CPU 和 GPU
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            # 如果使用 pynccl 启用 pynccl 分布式支持
            if config.use_pynccl:
                max_bytes = (
                    config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
                )
                enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        else:
            # 多 rank 时使用 nccl 后端(用于 GPU 通信)
            torch.distributed.init_process_group(
                backend="nccl",  # NCCL 后端 专用于 GPU 通信
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            # 创建 gloo 组用于 CPU 侧的同步操作
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            assert tp_cpu_group is not None
        return tp_cpu_group

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        """
        加载模型权重
        
        根据配置选择加载真实权重或生成虚拟权重(用于测试)。
        
        Args:
            config: 引擎配置
        
        Returns:
            模型权重的 state_dict
        """
        if config.use_dummy_weight:
            # 使用随机权重(用于测试)
            return {
                k: torch.randn_like(v, device=self.device)
                for k, v in self.model.state_dict().items()
            }
        else:
            # 从 HuggingFace 模型加载真实权重
            return {
                k: v.to(self.dtype)  # 转换为指定数据类型
                for k, v in load_hf_weight(config.model_path, self.device).items()
            }

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        """
        确定 KV Cache 的页面数量
        
        根据可用内存和配置计算可以分配的 KV Cache 页面数量。
        
        Args:
            old_free_memory: 加载模型前的可用内存
            config: 引擎配置
        
        Returns:
            KV Cache 的页面数量
        
        计算逻辑：
        1. 计算每个页面占用的内存(包括 K 和 V)
        2. 如果指定了 num_page_override 直接使用
        3. 否则根据可用内存和内存比例计算
        4. 确保至少有 1 个页面可用
        """
        # 获取加载模型后的可用内存
        new_free_memory = self._sync_get_memory()[1]
        
        # 计算每个页面占用的内存
        # 公式: 2(K+V) * head_dim * num_kv_heads_per_rank * page_size * dtype_size * num_layers
        cache_per_page = (
            2  # key + value
            * self.model_config.head_dim
            * divide_even(self.model_config.num_kv_heads, config.tp_info.size)  # 每个 rank 的 KV heads
            * config.page_size
            * self.dtype.itemsize
            * self.model_config.num_layers
        )
        
        # 如果指定了页面数量覆盖 直接使用
        num_pages = config.num_page_override
        if num_pages is None:
            # 计算模型占用的内存
            model_memory = old_free_memory - new_free_memory
            # 计算可用于 KV Cache 的内存
            available_memory = int(config.memory_ratio * old_free_memory) - model_memory
            # 计算可以分配的页面数量
            num_pages = available_memory // cache_per_page

        # 确保至少有 1 个页面可用
        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-tokens"
        real_kv_size = num_pages * cache_per_page
        logger.info(f"Allocating {num_pages} pages for KV cache, K + V = {mem_GB(real_kv_size)}")
        return num_pages

    def _sync_get_memory(self) -> Tuple[int, int]:
        """
        在所有 TP rank 上同步获取内存信息
        
        获取所有 TP rank 上的最小和最大可用内存。
        用于确保所有 rank 的内存使用平衡。
        
        Returns:
            Tuple[int, int]: (最小可用内存, 最大可用内存)
        
        Raises:
            RuntimeError: 如果不同 rank 之间的内存差异超过 2GB
        
        算法：
        1. 同步 GPU 清空缓存
        2. 获取当前 rank 的可用内存
        3. 使用 all_reduce 获取所有 rank 的最小和最大内存
        4. 检查内存是否平衡
        """
        # 同步 GPU 清空缓存 重置内存统计
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        
        # 获取当前 rank 的可用内存
        free_memory = get_free_memory(self.device)
        
        # 创建张量用于 all_reduce 操作
        # [free_memory, -free_memory] 用于同时获取最小和最大值
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        
        # 在所有 rank 上执行 MIN reduce 操作
        # 第一个元素得到最小内存 第二个元素得到最大内存(取负后求最小)
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        
        # 检查内存是否平衡(差异不应超过 2GB)
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
            logger.error(
                f"Memory across TP ranks are imbalanced:"
                f" min {mem_GB(min_free_memory)}, max {mem_GB(max_free_memory)}"
            )
            raise RuntimeError("Memory across TP ranks are imbalanced")

        return min_free_memory, max_free_memory

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        """
        执行批次 forward 和采样
        
        这是推理引擎的核心方法 执行以下步骤：
        1. 设置批次上下文
        2. 执行模型 forward(使用 CUDA Graph 或常规 forward)
        3. 更新请求状态
        4. 采样下一个 token
        5. 将结果复制到 CPU
        
        Args:
            batch: 批次对象
            args: 采样参数
        
        Returns:
            ForwardOutput: 包含下一个 token IDs 和同步事件
        
        Raises:
            AssertionError: 如果当前 CUDA 流不是引擎的专用流
        """
        # 确保使用正确的 CUDA 流
        assert torch.cuda.current_stream() == self.stream
        
        # 设置批次上下文
        with self.ctx.forward_batch(batch):
            # 如果可以使用 CUDA Graph 使用 graph 重放
            if self.graph_runner.can_use_cuda_graph(batch):
                logits = self.graph_runner.replay(batch)
            else:
                # 否则使用常规 forward
                logits = self.model.forward()

        # 更新所有请求的状态(标记完成一个 token 的生成)
        for req in batch.reqs:
            req.complete_one()

        # 采样下一个 token(只使用实际请求的 logits 不包括填充)
        next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
        # 将结果复制到 CPU(非阻塞)
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        # 记录复制完成事件 用于同步
        copy_done_event = torch.cuda.Event()
        copy_done_event.record()
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    def shutdown(self) -> None:
        """
        关闭引擎
        
        清理所有资源 包括：
        1. 销毁 CUDA Graph
        2. 销毁分布式进程组
        3. 清理分布式资源
        
        重要：必须在程序退出前调用此方法 确保资源正确释放。
        """
        # 销毁 CUDA Graph(必须在释放 NCCL 资源之前)
        self.graph_runner.destroy_cuda_graphs()
        # 销毁分布式进程组
        torch.distributed.destroy_process_group()
        # 清理分布式资源
        destroy_distributed()
