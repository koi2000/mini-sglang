"""
调度器模块 - mini-sglang 的核心调度逻辑

该模块实现了推理服务的调度器，负责：
1. 接收和处理用户请求
2. 管理 prefill（预填充）和 decode（解码）两个阶段的批处理
3. 实现重叠调度以提高 GPU 利用率
4. 管理 KV cache、token 池和页表
5. 与 tokenizer 和引擎协调工作
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
import torch.nn.functional as F
from minisgl.core import Batch, Req
from minisgl.env import ENV
from minisgl.message import (
    BaseBackendMsg,
    BatchBackendMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from minisgl.utils import init_logger
from transformers import AutoTokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .table import TableManager

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)


def _make_2d_indices(table_2d: torch.Tensor, ranges: List[Tuple[int, int, int]]) -> torch.Tensor:
    """
    将 2D 表格的坐标范围转换为 1D 索引，用于高效访问二维张量中的不连续区域。
    
    这个函数的核心作用是将多个二维坐标范围转换为一维索引，以便能够通过 view(-1)[indices] 
    的方式高效访问二维张量中的指定元素。
    
    应用场景：
    - 从 token_pool (2D: [num_requests, max_seq_len]) 中加载每个请求的部分 token
    - 避免使用循环逐个请求处理，提高内存访问效率

    示例：对于一个 3x4 的 2D 表格，其底层 1D 索引为：
        [[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]]
    给定范围 [(0, 1, 3), (2, 0, 2)]，表示：
        - 第 0 行的列 1-3：索引 [1, 2]
        - 第 2 行的列 0-2：索引 [8, 9]
    返回的 1D 索引为：[1, 2, 8, 9]

    Args:
        table_2d (torch.Tensor): 2D 表格张量（必须是连续的）
        ranges (List[Tuple[int, int, int]]): 范围列表，每个元组 (entry, begin, end) 表示：
            - entry: 行索引（表格中的第几行）
            - begin: 列起始索引（包含）
            - end: 列结束索引（不包含）
    
    Returns:
        torch.Tensor: 包含所有指定位置的 1D 索引的张量，位于与 table_2d 相同的设备上
    """
    # 确保输入是 2D 且连续的张量
    assert table_2d.dim() == 2 and table_2d.is_contiguous()
    
    # 获取行步长（stride），即每行有多少个元素
    STRIDE = table_2d.stride(0)
    
    # 计算需要的总索引数量
    needed_size = sum(end - begin for _, begin, end in ranges)
    
    # 在 CPU 上创建固定内存（pinned memory）的索引数组，用于高效 CPU->GPU 传输
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    
    offset = 0
    # 遍历每个范围，计算对应的 1D 索引
    for entry, begin, end in ranges:
        length = end - begin
        offset += length
        # 计算 1D 索引：行索引 * 行步长 + 列索引
        # 例如：entry=2, begin=0, end=2, STRIDE=4 → 索引为 [8, 9]
        torch.arange(
            begin + entry * STRIDE,  # 起始 1D 索引
            end + entry * STRIDE,    # 结束 1D 索引
            dtype=torch.int32,
            out=indices_host[offset - length : offset],  # 直接写入输出数组
        )
    
    # 异步传输到 GPU（non_blocking=True）
    return indices_host.to(table_2d.device, non_blocking=True)


# 为了支持重叠调度（overlap scheduling），我们需要缓存前向传播的输入输出数据
# 这样可以在执行当前批次时，同时处理上一批次的结果，避免 CPU-GPU 空闲等待（IMA: Idle Memory Access）
class ForwardInput(NamedTuple):
    """
    前向传播的输入数据封装。
    
    包含执行一次前向传播所需的所有信息：
    - batch: 批次数据（包含请求、token、注意力元数据等）
    - sample_args: 采样参数（温度、top_p、top_k 等）
    - load_indices: 用于从 token_pool 加载 token 的 1D 索引
    - write_indices: 用于将新生成的 token 写回 token_pool 的 1D 索引
    """
    batch: Batch                      # 批次数据
    sample_args: BatchSamplingArgs    # 批次采样参数
    load_indices: torch.Tensor        # token 加载索引
    write_indices: torch.Tensor       # token 写入索引


# 前向传播数据：包含输入和输出的完整数据对
ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class Scheduler(SchedulerIOMixin):
    """
    调度器类 - 负责协调整个推理服务的核心逻辑。
    
    主要职责：
    1. 请求管理：接收、调度和处理用户请求
    2. 批处理：将 prefill 和 decode 请求组织成批次
    3. 资源管理：管理 KV cache、token 池、页表等
    4. 执行协调：与引擎协调执行前向传播
    5. 结果处理：处理生成的 token，判断完成状态
    6. 重叠调度：支持计算和元数据处理的并行执行
    """
    
    def __init__(self, config: SchedulerConfig):
        """
        初始化调度器。
        
        Args:
            config (SchedulerConfig): 调度器配置，包含模型路径、批次大小、缓存策略等
        """
        from minisgl.engine import Engine

        # 初始化推理引擎（包含模型、注意力后端、采样器等）
        self.engine = Engine(config)
        
        # 初始化 I/O mixin，用于与外部通信（接收请求、发送结果）
        super().__init__(config, self.engine.tp_cpu_group)

        # ==================== CUDA 流管理 ====================
        # 使用独立的 CUDA 流来重叠元数据处理与计算
        # 这样可以在 engine.stream 执行模型前向传播的同时，在 self.stream 上处理其他任务
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)  # 调度器的元数据处理流
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)  # 引擎的计算流上下文
        torch.cuda.set_stream(self.stream)  # 设置当前流为调度器流

        # ==================== 管理器初始化 ====================
        # TableManager: 管理请求的页表槽位和 token 池
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        
        # CacheManager: 管理 KV cache 的分配、释放和 prompt cache
        self.cache_manager = CacheManager(self.device, self.engine.num_pages, config.cache_type)
        
        # DecodeManager: 管理处于 decode 阶段的请求
        self.decode_manager = DecodeManager()
        
        # PrefillManager: 管理处于 prefill 阶段的请求，支持 chunked prefill
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        # ==================== 其他配置 ====================
        self.tp_info = config.tp_info  # 张量并行信息
        self.finished_reqs: Set[Req] = set()  # 已完成但还在批次中的请求集合
        
        # Tokenizer 相关
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id  # 结束符 token ID
        
        # 快速访问的引用
        self.page_table = self.engine.page_table  # 页表：[max_running_req, max_seq_len]
        self.token_pool = self.table_manager.token_pool  # token 池：[max_running_req, max_seq_len]
        self.prefill_budget = config.max_extend_tokens  # 每次调度允许的最大 prefill token 数

    def _process_last_data(
        self, last_data: ForwardData | None, ongoing_data: ForwardData | None
    ) -> None:
        """
        处理上一批次的前向传播结果。
        
        这是重叠调度的关键部分：当前批次在 GPU 上执行时，CPU 同时处理上一批次的结果。
        
        主要任务：
        1. 等待 GPU->CPU 的 token 拷贝完成
        2. 判断每个请求是否完成（达到最大长度、遇到 EOS、满足停止条件）
        3. 释放已完成请求的资源（页表、KV cache）
        4. 将结果发送给 tokenizer 进行解码
        
        Args:
            last_data: 上一批次的前向传播数据（输入+输出）
            ongoing_data: 当前正在执行的批次数据（用于判断哪些请求仍在运行）
        """
        if last_data is None:
            return
        
        # 解包上一批次的数据
        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        
        # 等待 GPU->CPU 的异步拷贝完成
        copy_done.synchronize()
        
        # 准备发送给 tokenizer 的解码消息
        reply = BatchTokenizerMsg(data=[])

        max_seq_len = self.engine.max_seq_len
        
        # 遍历批次中的每个请求，处理生成的 token
        for i, req in enumerate(batch.reqs):
            # 跳过已经完成的请求和分块请求（ChunkedReq 不需要处理输出）
            if req in self.finished_reqs or isinstance(req, ChunkedReq):
                continue

            # 获取生成的下一个 token
            next_token_id = next_tokens_cpu[i]
            req.append_host(next_token_id.unsqueeze(0))  # 添加到请求的 host token 序列
            next_token = int(next_token_id.item())
            
            # 判断请求是否完成
            finished = req.remain_len <= 0  # 已生成足够的 token
            if not req.sampling_params.ignore_eos:
                finished |= next_token == self.eos_token_id  # 遇到结束符
            if req.device_len >= max_seq_len - 1:
                # 达到最大序列长度，强制完成
                finished = True
                logger.warning_rank0(f"Request {req.uid} reached {max_seq_len = }, dropped.")
            
            # 添加解码消息
            reply.data.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

            # 如果请求完成，标记并从 decode_manager 中移除
            if finished:
                self.finished_reqs.add(req)
                self.decode_manager.remove_req(req)
                logger.debug_rank0("Request %s is finished", req)

        # ==================== 资源释放 ====================
        # 只释放已完成且不在当前批次中的请求
        # 如果请求在当前批次中，即使完成也要等到下次才能释放（避免数据竞争）
        ongoing_reqs = ongoing_data[0].batch.reqs if ongoing_data else []
        for req in self.finished_reqs.difference(ongoing_reqs):
            # 释放页表槽位
            self.table_manager.free(req.table_idx)
            
            # 释放 KV cache，并可能将其缓存到 prompt cache
            self.cache_manager.free_and_cache_finished_req(
                req.cache_handle,
                req.host_ids[: req.cached_len],  # 请求的 token 序列
                self.page_table[req.table_idx, : req.cached_len],  # 对应的物理页号
            )

        # 只保留仍在运行批次中的已完成请求
        self.finished_reqs.intersection_update(ongoing_reqs)
        
        # 发送结果给 tokenizer 进行文本解码
        self.send_result(reply)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        """
        处理单个接收到的消息。
        
        消息类型：
        1. BatchBackendMsg: 批量消息，递归处理其中的每条消息
        2. ExitMsg: 退出消息，触发调度器关闭
        3. UserMsg: 用户请求消息，添加到 prefill 队列
        
        Args:
            msg: 接收到的消息
        """
        if isinstance(msg, BatchBackendMsg):
            # 批量消息：递归处理每条子消息
            for msg in msg.data:
                self._process_one_msg(msg)
                
        elif isinstance(msg, ExitMsg):
            # 退出消息：抛出异常以停止调度循环
            raise KeyboardInterrupt
            
        elif isinstance(msg, UserMsg):
            # 用户请求消息：验证并添加到 prefill 队列
            logger.debug_rank0("Received user msg: %s", msg)
            
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            
            # 验证输入长度：必须小于最大序列长度（至少能生成 1 个 token）
            if input_len >= max_seq_len:
                return logger.warning_rank0(
                    f"Input sequence len {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            
            # 调整最大输出长度：不能超过剩余空间
            max_output_len = max_seq_len - input_len
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            
            # 将请求添加到 prefill 管理器
            self.prefill_manager.add_one_req(msg)
            
        else:
            # 未知消息类型
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        """
        为批次准备前向传播所需的所有数据和元数据。
        
        准备步骤：
        1. 分配 KV cache：为本次生成的 token 分配物理页
        2. 填充批次：如果使用 CUDA Graph，需要填充到固定大小
        3. 生成索引：计算从 token_pool 加载和写入的 1D 索引
        4. 更新页表：将分配的物理页号写入页表
        5. 准备元数据：为注意力计算准备元数据（位置、长度、指针等）
        6. 准备采样参数：为每个请求准备采样配置
        
        Args:
            batch: 待执行的批次
            
        Returns:
            ForwardInput: 封装好的前向传播输入数据
        """
        # ==================== 1. 分配 KV cache ====================
        # 计算本次需要的总 cache 大小（每个请求需要存储 extend_len 个 token 的 KV）
        needed_size = sum(r.extend_len for r in batch.reqs)
        # 从 cache_manager 分配物理页，返回物理页号数组
        batch.out_loc = self.cache_manager.allocate(needed_size)
        
        # ==================== 2. 批次填充（CUDA Graph 优化）====================
        # CUDA Graph 要求批次大小固定，因此需要填充
        if padding_size := self.engine.graph_runner.pad_batch(batch):
            # 用 dummy_page 填充 out_loc
            batch.out_loc = F.pad(batch.out_loc, (0, padding_size), value=self.engine.dummy_page)
        
        # ==================== 3. 生成 token 索引 ====================
        # load_indices: 从 token_pool 加载哪些 token（本次要处理的 token）
        # 对于每个请求，加载 [cached_len, device_len) 范围的 token
        # - cached_len: 已经在 GPU 上的 token 数量
        # - device_len: 总共需要处理的 token 数量（包括新添加的）
        load_indices = _make_2d_indices(
            self.token_pool, [(r.table_idx, r.cached_len, r.device_len) for r in batch.padded_reqs]
        )
        
        # write_indices: 将生成的新 token 写到 token_pool 的哪个位置
        # 对于每个请求，写入位置是 [device_len, device_len + 1)
        write_indices = _make_2d_indices(
            self.token_pool, [(r.table_idx, r.device_len, r.device_len + 1) for r in batch.reqs]
        )
        
        # ==================== 4. 更新页表 ====================
        # 将分配的物理页号写入页表对应位置
        # 注意力机制会读取页表来找到对应的 KV cache
        self.page_table.view(-1)[load_indices] = batch.out_loc
        
        # ==================== 5. 准备注意力元数据 ====================
        # 计算注意力所需的元数据（seq_lens, start_loc, block_table 等）
        self.engine.attn_backend.prepare_metadata(batch)
        
        # ==================== 6. 封装返回 ====================
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),  # 准备采样参数
            load_indices=load_indices,
            write_indices=write_indices,
        )

    def _schedule_next_batch(self) -> ForwardInput | None:
        """
        调度下一个要执行的批次。
        
        调度策略：
        1. 优先调度 prefill 批次（在预算范围内）
        2. 如果没有 prefill 或超出预算，则调度 decode 批次
        
        TODO: 支持其他调度策略，例如 DECODE first（优先 decode 以降低延迟）
        
        Returns:
            ForwardInput | None: 准备好的前向传播输入，如果没有可调度的请求则返回 None
        """
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        return self._prepare_batch(batch) if batch else None

    def _load_token_ids(self, input: ForwardInput) -> None:
        """
        从 token_pool 加载 token IDs 到批次。
        
        使用预先计算的 load_indices 从 2D token_pool 中高效加载所需的 token。
        这避免了逐请求循环加载，提高了性能。
        
        Args:
            input: 前向传播输入，包含批次和加载索引
        """
        batch, load_indices = input.batch, input.load_indices
        batch.input_ids = self.token_pool.view(-1)[load_indices]

    def _write_token_ids(self, input: ForwardInput, output: ForwardOutput) -> None:
        """
        将生成的 token IDs 写回 token_pool。
        
        使用预先计算的 write_indices 将新生成的 token 写回对应位置。
        
        Args:
            input: 前向传播输入，包含写入索引
            output: 前向传播输出，包含生成的 token
        """
        self.token_pool.view(-1)[input.write_indices] = output.next_tokens_gpu

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        """
        执行完整的前向传播流程。
        
        步骤：
        1. 加载 token IDs
        2. 调用引擎执行模型前向传播和采样
        3. 将生成的 token 写回 token_pool
        4. 将非分块请求添加到 decode_manager（继续解码）
        
        Args:
            forward_input: 前向传播输入数据
            
        Returns:
            ForwardOutput: 前向传播输出（包含生成的 token 和同步事件）
        """
        # 1. 从 token_pool 加载输入 token
        self._load_token_ids(forward_input)
        
        # 2. 执行模型前向传播和采样
        batch, sample_args = forward_input.batch, forward_input.sample_args
        forward_output = self.engine.forward_batch(batch, sample_args)
        
        # 3. 将生成的 token 写回 token_pool
        self._write_token_ids(forward_input, forward_output)
        
        # 4. 将批次中的请求添加到 decode_manager
        # （下一次迭代时这些请求可能会被调度为 decode 批次）
        self.decode_manager.add_reqs(forward_input.batch.reqs)
        
        return forward_output

    def run_when_idle(self) -> None:
        """
        当调度器空闲时执行的后台任务。
        
        主要用于：
        - 日志记录
        - cache_manager 完整性检查
        - 其他维护任务
        """
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        重叠调度的主循环 - 核心优化机制。
        
        重叠调度的原理：
        ┌─────────────────────────────────────────────────────────────┐
        │ 时间轴                                                       │
        ├─────────────────────────────────────────────────────────────┤
        │ GPU (engine.stream):  [Batch N 前向传播] [Batch N+1 前向]   │
        │ CPU (self.stream):    [处理 Batch N-1] [调度 Batch N+1]     │
        └─────────────────────────────────────────────────────────────┘
        
        通过使用两个 CUDA 流，实现：
        1. GPU 执行当前批次的同时，CPU 处理上一批次的结果
        2. 有效隐藏 CPU 延迟（token 拷贝、资源释放、消息发送等）
        3. 提高 GPU 利用率，减少空闲时间
        
        执行流程：
        1. 接收并处理新消息（如果有待处理数据则非阻塞）
        2. 调度下一个批次
        3. 在引擎流中启动前向传播（GPU 异步执行）
        4. 在调度器流中处理上一批次结果（CPU 同时执行）
        5. 返回当前批次数据，作为下次循环的 last_data
        
        Args:
            last_data: 上一批次的前向传播数据（待处理结果）
            
        Returns:
            ForwardData | None: 当前批次的前向传播数据（下次循环处理）
        """
        # ==================== 1. 接收和处理消息 ====================
        # 决定是否阻塞等待消息：
        # - 如果有待处理的数据或可运行的请求，则非阻塞（继续执行）
        # - 否则阻塞等待新消息（避免空转）
        blocking = not (
            last_data  # 有上一批次数据需要处理
            or self.prefill_manager.runnable  # 有可调度的 prefill 请求
            or self.decode_manager.runnable   # 有可调度的 decode 请求
        )
        
        # 接收并处理所有待处理的消息
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # ==================== 2. 调度下一批次 ====================
        forward_input = self._schedule_next_batch()
        ongoing_data = None
        
        # ==================== 3. 启动前向传播（GPU 异步执行）====================
        if forward_input is not None:
            # 切换到引擎流，在 GPU 上执行前向传播
            with self.engine_stream_ctx:
                # 等待调度器流完成（确保元数据准备完成）
                self.engine.stream.wait_stream(self.stream)
                
                # 执行前向传播（GPU 异步执行，CPU 立即返回）
                ongoing_data = (forward_input, self._forward(forward_input))

        # ==================== 4. 处理上一批次结果（CPU 同时执行）====================
        # 此时 GPU 正在执行 ongoing_data 的前向传播
        # CPU 同时处理 last_data 的结果（token 拷贝、资源释放等）
        self._process_last_data(last_data, ongoing_data)
        
        # 返回当前批次数据，作为下次循环的 last_data
        return ongoing_data

    def normal_loop(self) -> None:
        """
        正常调度循环 - 不使用重叠优化的简化版本。
        
        执行流程（顺序执行，无重叠）：
        1. 接收并处理消息
        2. 调度下一批次
        3. 执行前向传播（阻塞等待完成）
        4. 处理结果
        
        相比 overlap_loop：
        - 更简单，易于调试
        - 但 GPU 利用率较低（CPU 处理时 GPU 空闲）
        - 主要用于调试或禁用重叠调度时使用
        """
        # 1. 决定是否阻塞等待消息
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        
        # 2. 接收并处理消息
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # 3. 调度并执行下一批次
        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        # 4. 立即处理当前批次的结果（同步执行）
        self._process_last_data(ongoing_data, None)

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        """
        调度器的主入口 - 永久运行直到收到退出信号。
        
        根据环境变量选择调度模式：
        1. ENV.DISABLE_OVERLAP_SCHEDULING=True: 使用 normal_loop（顺序执行）
        2. ENV.DISABLE_OVERLAP_SCHEDULING=False: 使用 overlap_loop（重叠执行，默认）
        
        使用 @torch.inference_mode() 装饰器：
        - 禁用梯度计算和自动微分
        - 降低内存占用，提高推理性能
        
        Raises:
            KeyboardInterrupt: 当收到 ExitMsg 时由 _process_one_msg 抛出
        """
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            # ==================== 正常模式（顺序执行）====================
            with self.engine_stream_ctx:
                # 同步流，确保所有操作在同一流中
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            # ==================== 重叠模式（并行执行，默认）====================
            # 确保当前在调度器流中
            assert torch.cuda.current_stream() == self.stream
            
            data = None  # 初始没有待处理的数据
            while True:
                # 每次循环：
                # - 处理上次的 data（如果有）
                # - 启动新批次并返回其数据
                # - 下次循环处理这次返回的数据
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        """
        关闭调度器并清理资源。
        
        清理步骤：
        1. 同步 CUDA 设备，等待所有操作完成
        2. 同步所有 rank（多卡场景）
        3. 关闭引擎（释放模型、内存等）
        """
        # 1. 等待所有 CUDA 操作完成
        torch.cuda.synchronize(self.device)
        
        # 2. 同步所有进程（张量并行场景）
        self.sync_all_ranks()
        
        # 3. 关闭引擎并释放资源
        self.engine.shutdown()
