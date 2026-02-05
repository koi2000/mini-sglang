"""
Tokenizer 模块：提供文本 tokenization 功能

该模块包含 TokenizeManager 类，用于管理文本的 tokenization 过程。
主要功能是将输入文本（包括聊天格式的对话）转换为模型可以处理的 token ID 序列。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch
from minisgl.message import TokenizeMsg

# TYPE_CHECKING 用于类型检查时的条件导入，避免运行时导入不必要的依赖
# 这样可以减少启动时间和避免循环导入问题
if TYPE_CHECKING:
    from transformers import LlamaTokenizer


class TokenizeManager:
    """
    Tokenization 管理器类
    
    负责将文本消息转换为模型可以处理的 token ID 张量。
    支持两种输入格式：
    1. 普通字符串文本
    2. 聊天格式的消息列表（会通过 chat template 转换为格式化字符串）
    """

    def __init__(self, tokenizer: LlamaTokenizer) -> None:
        """
        初始化 TokenizeManager
        
        Args:
            tokenizer: LlamaTokenizer 实例，用于执行实际的 tokenization 操作
                      该 tokenizer 通常来自 transformers 库，支持 LLaMA 系列模型
        """
        self.tokenizer = tokenizer

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        """
        将多个消息 tokenize 为 token ID 张量列表
        
        该方法处理两种类型的输入：
        1. 如果 msg.text 是列表：使用 apply_chat_template 将聊天消息列表
           转换为符合模型格式的字符串（例如：添加特殊标记、格式化角色等）
        2. 如果 msg.text 是字符串：直接使用该字符串
        
        然后将文本编码为 token ID，并转换为 int32 类型的 PyTorch 张量。
        
        Args:
            msgs: TokenizeMsg 对象列表，每个对象包含需要 tokenize 的文本内容
                  msg.text 可以是字符串或聊天消息列表（如 [{"role": "user", "content": "..."}])
        
        Returns:
            List[torch.Tensor]: token ID 张量列表，每个张量是一维的 int32 类型张量
                               形状为 [seq_len]，表示一个输入序列的 token IDs
        
        Note:
            当前实现是逐个处理消息（TODO: batch tokenization），
            未来可以优化为批量处理以提高效率
        """
        results: List[torch.Tensor] = []
        
        # TODO: batch tokenization
        # 当前实现：逐个处理每个消息
        # 优化方向：可以批量处理多个消息以提高性能
        for msg in msgs:
            # 检查消息文本是否为列表格式（聊天格式）
            if isinstance(msg.text, list):
                # 使用 chat template 将聊天消息列表转换为格式化字符串
                # apply_chat_template 会根据模型的配置将消息列表转换为特定格式
                # 例如：添加 <|im_start|>、<|im_end|> 等特殊标记，或格式化角色信息
                prompt = self.tokenizer.apply_chat_template(
                    msg.text,  # 聊天消息列表，格式如 [{"role": "user", "content": "..."}]
                    tokenize=False,  # 只格式化，不进行 tokenization（我们稍后会单独 tokenize）
                    add_generation_prompt=True,  # 添加生成提示，表示这是用于生成任务的输入
                )
                # 类型断言：确保返回的是字符串类型
                # apply_chat_template 在 tokenize=False 时应该返回字符串
                assert isinstance(prompt, str)
            else:
                # 如果 msg.text 已经是字符串，直接使用
                prompt = msg.text
            
            # 将文本编码为 token IDs
            # encode 方法将字符串转换为 token ID 序列
            # return_tensors="pt" 表示返回 PyTorch 张量格式
            # 返回形状通常是 [1, seq_len]，即批次大小为 1 的二维张量
            input_ids: torch.Tensor = (  # type: ignore
                self.tokenizer.encode(prompt, return_tensors="pt")
            )
            
            # 将二维张量展平为一维，并转换为 int32 类型
            # view(-1) 将 [1, seq_len] 展平为 [seq_len]
            # to(torch.int32) 确保数据类型为 int32（某些模型要求特定的数据类型）
            results.append(input_ids.view(-1).to(torch.int32))
        
        return results
