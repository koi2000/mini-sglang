"""
Tokenizer 服务器模块：提供独立的 tokenization 服务进程

该模块实现了一个基于 ZMQ 消息队列的 tokenizer 服务器, 可以在独立进程中运行。
主要功能包括：
1. 接收 tokenize 请求(将文本转换为 token IDs)
2. 接收 detokenize 请求(将 token IDs 转换为文本)
3. 批量处理消息以提高效率
4. 将处理结果发送到相应的后端或前端队列
"""

from __future__ import annotations

import multiprocessing as mp
from typing import List

import torch
from minisgl.message import (
    BaseBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchBackendMsg,
    BatchFrontendMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    TokenizeMsg,
    UserMsg,
    UserReply,
)
from minisgl.utils import ZmqPullQueue, ZmqPushQueue, init_logger
from transformers import AutoTokenizer, LlamaTokenizer


def _unwrap_msg(msg: BaseTokenizerMsg) -> List[BaseTokenizerMsg]:
    """
    解包消息：将批量消息展开为单个消息列表
    
    该函数用于统一处理单个消息和批量消息：
    - 如果输入是 BatchTokenizerMsg, 返回其内部的 data 列表
    - 如果输入是单个消息, 将其包装为列表返回
    
    Args:
        msg: 可能是单个消息或批量消息的 BaseTokenizerMsg 对象
    
    Returns:
        List[BaseTokenizerMsg]: 单个消息的列表, 确保返回格式统一
    """
    if isinstance(msg, BatchTokenizerMsg):
        return msg.data
    return [msg]


@torch.inference_mode()
def tokenize_worker(
    *,
    tokenizer_path: str,
    addr: str,
    create: bool,
    backend_addr: str,
    frontend_addr: str,
    local_bs: int,
    tokenizer_id: int = -1,
    ack_queue: mp.Queue[str] | None = None,
) -> None:
    """
    Tokenizer 工作进程主函数
    
    这是一个独立运行的 worker 进程, 负责处理 tokenization 和 detokenization 请求。
    使用 ZMQ 消息队列进行进程间通信, 支持批量处理以提高效率。
    
    工作流程：
    1. 初始化 ZMQ 队列(接收请求、发送结果)
    2. 加载 tokenizer 模型
    3. 创建 TokenizeManager 和 DetokenizeManager
    4. 进入主循环：
       a. 从队列接收消息(支持批量接收直到达到 local_bs)
       b. 将消息分类为 tokenize 和 detokenize 请求
       c. 分别处理两类请求
       d. 将结果发送到相应的队列
    
    Args:
        tokenizer_path: tokenizer 模型路径, 用于加载预训练的 tokenizer
        addr: ZMQ 接收队列地址, 用于接收 tokenization 请求
        create: 是否创建接收队列(True 表示创建新队列, False 表示连接到已有队列)
        backend_addr: ZMQ 发送队列地址, 用于发送 tokenize 结果到后端
        frontend_addr: ZMQ 发送队列地址, 用于发送 detokenize 结果到前端
        local_bs: 本地批量大小(local batch size), 控制每次处理的消息数量
                 用于批量处理以提高效率
        tokenizer_id: tokenizer 实例的唯一标识符, 用于日志记录和调试
        ack_queue: 可选的确认队列, 用于通知主进程 worker 已准备就绪
                  如果提供, 会在初始化完成后发送就绪消息
    
    Note:
        - 使用 @torch.inference_mode() 装饰器禁用梯度计算, 提高推理性能
        - 支持 KeyboardInterrupt 优雅退出
        - 消息处理支持批量优化, 会尽量收集消息直到达到 local_bs 或队列为空
    """
    # 初始化 ZMQ 消息队列
    # send_backend: 用于将 tokenize 结果发送到后端(模型推理服务)
    send_backend = ZmqPushQueue(backend_addr, create=False, encoder=BaseBackendMsg.encoder)
    
    # send_frontend: 用于将 detokenize 结果发送到前端(用户接口)
    send_frontend = ZmqPushQueue(frontend_addr, create=False, encoder=BaseFrontendMsg.encoder)
    
    # recv_listener: 用于接收来自其他组件的 tokenization 请求
    recv_listener = ZmqPullQueue(addr, create=create, decoder=BatchTokenizerMsg.decoder)
    
    # 确保批量大小大于 0
    assert local_bs > 0
    
    # 加载 tokenizer 模型
    # use_fast=True 使用快速版本的 tokenizer(通常基于 Rust 实现, 性能更好)
    tokenizer: LlamaTokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    
    # 初始化日志记录器, 使用 tokenizer_id 区分不同的 worker 实例
    logger = init_logger(__name__, f"tokenizer_{tokenizer_id}")

    # 延迟导入避免循环依赖
    from .detokenize import DetokenizeManager
    from .tokenize import TokenizeManager

    # 创建 tokenization 和 detokenization 管理器
    tokenize_manager = TokenizeManager(tokenizer)
    detokenize_manager = DetokenizeManager(tokenizer)

    # 如果提供了确认队列, 通知主进程 worker 已准备就绪
    if ack_queue is not None:
        ack_queue.put(f"Tokenize server {tokenizer_id} is ready")

    try:
        # 主循环：持续处理消息直到收到中断信号
        while True:
            # 步骤 1: 接收并解包第一个消息
            pending_msg = _unwrap_msg(recv_listener.get())
            
            # 步骤 2: 批量收集消息
            # 继续从队列中获取消息, 直到：
            # - 已收集的消息数量达到 local_bs(批量大小)
            # - 或者队列为空(没有更多消息可获取)
            # 这样可以提高批量处理的效率
            while len(pending_msg) < local_bs and not recv_listener.empty():
                pending_msg.extend(_unwrap_msg(recv_listener.get()))

            logger.debug(f"Received {len(pending_msg)} messages")

            # 步骤 3: 将消息分类为 detokenize 和 tokenize 请求
            # detokenize_msg: 需要将 token IDs 转换为文本的消息(用于生成输出)
            detokenize_msg = [m for m in pending_msg if isinstance(m, DetokenizeMsg)]
            
            # tokenize_msg: 需要将文本转换为 token IDs 的消息(用于模型输入)
            tokenize_msg = [m for m in pending_msg if isinstance(m, TokenizeMsg)]
            
            # 验证：所有消息都应该被分类(不应该有其他类型的消息)
            assert len(detokenize_msg) + len(tokenize_msg) == len(pending_msg)
            
            # 步骤 4: 处理 detokenize 请求(将 token IDs 转换为文本)
            if len(detokenize_msg) > 0:
                # 批量执行 detokenization, 返回文本列表
                replies = detokenize_manager.detokenize(detokenize_msg)
                
                # 构建批量前端消息, 包含所有 detokenize 结果
                batch_output = BatchFrontendMsg(
                    data=[
                        UserReply(
                            uid=msg.uid,  # 用户 ID, 用于关联请求和响应
                            incremental_output=reply,  # 增量输出文本(用于流式生成)
                            finished=msg.finished,  # 是否完成标志
                        )
                        for msg, reply in zip(detokenize_msg, replies, strict=True)
                    ]
                )
                
                # 如果只有一个结果, 直接发送单个消息而不是批量消息(优化)
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                
                # 将结果发送到前端队列
                send_frontend.put(batch_output)

            # 步骤 5: 处理 tokenize 请求(将文本转换为 token IDs)
            if len(tokenize_msg) > 0:
                # 批量执行 tokenization, 返回 token ID 张量列表
                tensors = tokenize_manager.tokenize(tokenize_msg)
                
                # 构建批量后端消息, 包含所有 tokenize 结果
                batch_output = BatchBackendMsg(
                    data=[
                        UserMsg(
                            uid=msg.uid,  # 用户 ID
                            input_ids=t,  # token ID 张量(模型输入)
                            sampling_params=msg.sampling_params,  # 采样参数(温度、top_p 等)
                        )
                        for msg, t in zip(tokenize_msg, tensors, strict=True)
                    ]
                )
                
                # 如果只有一个结果, 直接发送单个消息而不是批量消息(优化)
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                
                # 将结果发送到后端队列(模型推理服务)
                send_backend.put(batch_output)
    
    except KeyboardInterrupt:
        # 优雅处理中断信号(Ctrl+C), 允许进程正常退出
        pass
