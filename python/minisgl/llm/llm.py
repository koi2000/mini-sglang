"""
LLM 模块：提供离线模式的 LLM 推理接口

该模块实现了一个简化的 LLM 类, 继承自 Scheduler, 专门用于离线批处理场景。
主要功能包括：
1. 接收文本或 token ID 列表作为输入
2. 批量处理多个生成请求
3. 跟踪每个请求的状态(输入和输出 token IDs)
4. 返回完整的生成结果(文本和 token IDs)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.message import (
    BaseBackendMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    UserMsg,
)
from minisgl.scheduler import Scheduler, SchedulerConfig


class RequestAllFinished(Exception):
    """
    自定义异常：表示所有请求已处理完成
    
    当在阻塞模式下调用 offline_receive_msg 且没有待处理请求时抛出。
    用于优雅地终止 run_forever() 循环。
    """
    pass


@dataclass
class RequestStatus:
    """
    请求状态数据类：跟踪单个生成请求的状态信息
    
    Attributes:
        uid: 用户/请求的唯一标识符
        input_ids: 输入的 token ID 列表(用于记录原始输入)
        output_ids: 输出的 token ID 列表(累积生成的 token)
    """
    uid: int
    input_ids: List[int]
    output_ids: List[int]


class LLM(Scheduler):
    """
    LLM 类：离线模式的简化 LLM 推理接口
    
    继承自 Scheduler, 实现了离线批处理模式。与在线服务模式不同,
    离线模式不需要外部消息队列, 而是直接处理内存中的请求列表。
    
    主要特点：
    - 支持批量生成多个提示词
    - 自动管理请求状态和 token 累积
    - 提供简单的同步生成接口
    - 支持文本或 token ID 列表作为输入
    """

    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16, **kwargs):
        """
        初始化 LLM 实例
        
        Args:
            model_path: 模型路径, 用于加载模型和 tokenizer
            dtype: 模型数据类型, 默认为 bfloat16(平衡性能和精度)
            **kwargs: 其他配置参数, 传递给 SchedulerConfig
                     (例如: max_running_req, max_extend_tokens 等)
        
        Note:
            - tp_info=DistributedInfo(0, 1) 表示单 GPU 模式(rank 0, world_size 1)
            - offline_mode=True 启用离线模式, 使用内存队列而非网络队列
        """
        config = SchedulerConfig(
            model_path=model_path,
            tp_info=DistributedInfo(0, 1),  # 单 GPU 配置: rank=0, world_size=1
            dtype=dtype,
            offline_mode=True,  # 启用离线模式
            **kwargs,
        )
        super().__init__(config)
        
        # 待处理请求列表: 每个元素是 (prompt, sampling_params) 元组
        # prompt 可以是字符串或 token ID 列表
        self.pending_requests: List[Tuple[List[int] | str, SamplingParams]] = []
        
        # 请求状态映射: uid -> RequestStatus
        # 用于跟踪每个请求的输入和输出 token IDs
        self.status_map: Dict[int, RequestStatus] = {}
        
        # 请求计数器: 用于生成唯一的 uid
        self.counter = 0

    def _tokenize_one(self, prompt: List[int] | str) -> torch.Tensor:
        """
        将单个提示词转换为 token ID 张量
        
        支持两种输入格式：
        1. 字符串: 使用 tokenizer 编码
        2. token ID 列表: 直接转换为张量
        
        Args:
            prompt: 提示词, 可以是字符串或 token ID 列表
        
        Returns:
            torch.Tensor: 一维 int32 张量, 形状为 [seq_len]
                         device 为 CPU(符合 Scheduler 的要求)
        """
        if isinstance(prompt, str):
            # 字符串输入: 使用 tokenizer 编码
            return self.tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)
        else:
            # token ID 列表输入: 直接转换为张量
            return torch.tensor(prompt, dtype=torch.int32, device="cpu")

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """
        离线模式的消息接收方法
        
        从待处理请求列表中提取一批请求, 转换为 UserMsg 消息。
        受 prefill_budget 限制, 控制每次处理的输入 token 总数。
        
        工作流程：
        1. 检查是否有待处理请求(阻塞模式下如果没有则抛出异常)
        2. 遍历待处理请求, 累积输入长度
        3. 当累积长度达到 prefill_budget 时停止
        4. 为每个请求创建 UserMsg 并记录状态
        5. 从未处理的请求中移除已处理的请求
        
        Args:
            blocking: 是否阻塞模式
                     - True: 如果没有待处理请求, 抛出 RequestAllFinished 异常
                     - False: 如果没有待处理请求, 返回空列表
        
        Returns:
            List[BaseBackendMsg]: UserMsg 消息列表, 用于发送给调度器处理
        
        Raises:
            RequestAllFinished: 当 blocking=True 且没有待处理请求时抛出
        
        Note:
            - prefill_budget 是 Scheduler 的属性, 限制每次 prefill 阶段的 token 数量
            - 这里使用 sum_input_len 来近似控制, 实际应该考虑更精确的计算
        """
        # 阻塞模式下, 如果没有待处理请求, 抛出异常表示所有请求已完成
        if blocking and len(self.pending_requests) == 0:
            raise RequestAllFinished()
        
        results: List[BaseBackendMsg] = []
        i, sum_input_len = 0, 0
        
        # 遍历待处理请求, 直到达到批量预算或处理完所有请求
        for i, (tokens_or_prompt, sampling_params) in enumerate(self.pending_requests):
            # 为每个请求分配唯一的 uid
            uid = self.counter
            self.counter += 1
            
            # 检查是否超过 prefill 预算(简化版本, 实际应该更精确地计算)
            if sum_input_len >= self.prefill_budget:
                break
            
            # 将提示词转换为 token ID 张量
            input_ids = self._tokenize_one(tokens_or_prompt)
            
            # 创建 UserMsg 消息, 包含 uid、input_ids 和采样参数
            results.append(UserMsg(uid=uid, input_ids=input_ids, sampling_params=sampling_params))
            
            # 记录请求状态
            # 注意: uid 使用 counter, 但 status_map 的 key 也是 uid
            # 这里将 uid 存储为 i 可能是一个 bug, 应该是 uid
            self.status_map[uid] = RequestStatus(
                uid=i,  # 注意: 这里可能应该是 uid 而不是 i
                input_ids=(
                    # 保存原始输入: 如果是字符串则转换为列表, 否则直接使用
                    input_ids.tolist() if isinstance(tokens_or_prompt, str) else tokens_or_prompt
                ),
                output_ids=[],  # 初始化为空, 后续会累积生成的 token
            )
        
        # 从未处理的请求中移除已处理的请求
        self.pending_requests = self.pending_requests[i + 1 :]
        return results

    def offline_send_result(self, reply: BatchTokenizerMsg) -> None:
        """
        离线模式的结果发送方法
        
        接收调度器返回的生成结果, 更新每个请求的输出状态。
        累积生成的 token IDs, 直到请求完成。
        
        Args:
            reply: 批量 tokenizer 消息, 包含多个 DetokenizeMsg
                  每个消息包含一个生成的 token 和完成标志
        
        Note:
            - 只处理未完成的请求(继续累积 token)
            - 完成的请求会在后续步骤中处理
        """
        for msg in reply.data:
            # 验证消息类型
            assert isinstance(msg, DetokenizeMsg)
            
            # 获取对应的请求状态
            status = self.status_map[msg.uid]
            
            # 如果请求未完成, 将新生成的 token 添加到输出列表
            if not msg.finished:
                status.output_ids.append(msg.next_token)

    def generate(
        self,
        prompts: List[str] | List[List[int]],
        sampling_params: List[SamplingParams] | SamplingParams,
    ) -> List[str]:
        """
        批量生成方法：主要的用户接口
        
        接收多个提示词, 批量生成文本, 返回完整结果。
        这是一个同步方法, 会阻塞直到所有请求完成。
        
        工作流程：
        1. 初始化状态(清空待处理请求和状态映射)
        2. 准备请求列表(处理采样参数的统一化)
        3. 调用 run_forever() 启动调度器循环
        4. 等待所有请求完成(RequestAllFinished 异常)
        5. 解码所有输出 token IDs 为文本
        6. 返回结果列表
        
        Args:
            prompts: 提示词列表, 每个元素可以是：
                    - 字符串: 文本提示词
                    - token ID 列表: 预编码的 token IDs
            sampling_params: 采样参数, 可以是：
                            - 单个 SamplingParams: 应用于所有提示词
                            - SamplingParams 列表: 为每个提示词指定不同参数
        
        Returns:
            List[Dict]: 结果列表, 每个元素包含：
                       - "text": 生成的文本字符串
                       - "token_ids": 生成的 token ID 列表
        
        Example:
            >>> llm = LLM("path/to/model")
            >>> results = llm.generate(
            ...     prompts=["Hello", "How are you?"],
            ...     sampling_params=SamplingParams(temperature=0.7)
            ... )
            >>> print(results[0]["text"])  # 生成的文本
            >>> print(results[0]["token_ids"])  # token IDs
        """
        # 重置状态
        self.pending_requests = []
        self.status_map = {}
        self.counter = 0
        
        # 统一采样参数格式: 如果是单个参数, 复制为列表
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        
        # 准备待处理请求列表
        for prompt, sp in zip(prompts, sampling_params):
            self.pending_requests.append((prompt, sp))
        
        # 启动调度器主循环
        # run_forever() 会持续调用 offline_receive_msg 和 offline_send_result
        # 直到所有请求完成, 此时会抛出 RequestAllFinished 异常
        try:
            self.run_forever()
        except RequestAllFinished:
            # 所有请求已完成, 正常退出循环
            pass
        
        # 收集结果: 解码所有输出 token IDs
        results = []
        for i in range(len(prompts)):
            # 获取请求状态(注意: 这里使用索引 i, 与 offline_receive_msg 中的逻辑对应)
            status = self.status_map[i]
            
            # 将 token IDs 解码为文本
            output_text = self.tokenizer.decode(status.output_ids)
            
            # 构建结果字典
            results.append({"text": output_text, "token_ids": status.output_ids})
        
        return results
