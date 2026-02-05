"""
Qwen3 模型实现模块

该模块实现了 Qwen3 系列大语言模型的完整架构, 包括：
1. Qwen3DecoderLayer: 单个 Transformer 解码器层
2. Qwen3Model: 完整的模型主干(不包括语言模型头)
3. Qwen3ForCausalLM: 完整的因果语言模型(包括 LM head)

主要特性：
- 使用 RMSNorm 进行层归一化
- 使用 RoPE (Rotary Position Embedding) 注意力机制
- 使用 Gated MLP (SwiGLU 激活函数)
- 支持残差连接和预归一化架构
- 支持张量并行(TP)和词汇表并行
- 使用 NVTX 进行性能分析和调试
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.cuda.nvtx as nvtx
from minisgl.core import get_global_ctx
from minisgl.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding

from .base import BaseLLMModel
from .utils import GatedMLP as Qwen3MLP
from .utils import RopeAttn as Qwen3Attn

if TYPE_CHECKING:
    from .config import ModelConfig


class Qwen3DecoderLayer(BaseOP):
    """
    Qwen3 解码器层：单个 Transformer 解码器层
    
    实现标准的 Transformer 解码器层结构, 采用预归一化(Pre-LayerNorm)架构：
    
    架构流程：
    1. 输入层归一化 (input_layernorm)
    2. 多头自注意力 (self_attn) - 使用 RoPE 和 QK 归一化
    3. 残差连接
    4. 后注意力层归一化 (post_attention_layernorm)
    5. MLP 前馈网络 (mlp) - 使用 Gated MLP (SwiGLU)
    6. 残差连接
    
    关键特性：
    - 使用 RMSNorm 进行归一化(比 LayerNorm 更高效)
    - 支持残差连接, 提高训练稳定性
    - 使用融合的归一化操作(RMSNormFused)提高性能
    - 支持 QK 归一化(has_qk_norm=True), 提高注意力稳定性
    """

    def __init__(self, config: ModelConfig, layer_id: int):
        """
        初始化 Qwen3 解码器层
        
        Args:
            config: 模型配置对象, 包含隐藏层大小、注意力头数等参数
            layer_id: 层编号, 用于标识不同的层(用于 NVTX 性能分析)
        """
        # 多头自注意力层
        # 使用 RoPE (Rotary Position Embedding) 注意力机制
        # has_qk_norm=True 表示对 Q 和 K 进行归一化, 提高注意力稳定性
        self.self_attn = Qwen3Attn(config, layer_id, has_qk_norm=True)
        
        # MLP 前馈网络
        # 使用 Gated MLP (SwiGLU 激活函数), 比标准 MLP 更高效
        self.mlp = Qwen3MLP(config)
        
        # 输入层归一化(注意力层之前)
        # 使用融合的 RMSNorm 实现, 提高性能
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,  # 数值稳定性参数
        )
        
        # 后注意力层归一化(MLP 之前)
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # 保存层 ID, 用于 NVTX 性能分析和调试
        self._layer_id = layer_id

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        执行完整的解码器层计算, 包括注意力、MLP 和残差连接。
        
        Args:
            x: 输入张量, 形状为 [batch_size, seq_len, hidden_size]
            residual: 残差连接张量, 形状与 x 相同
                     如果为 None, 则使用 x 作为残差
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - 输出张量 x: 经过本层处理后的特征
                - 残差张量 residual: 用于下一层的残差连接
        
        Note:
            - 使用 NVTX 范围标记进行性能分析
            - RMSNormFused 会自动处理残差连接
            - 第一层时 residual 为 None, 后续层会传递累积的残差
        """
        # 步骤 1: 输入层归一化 + 残差连接
        # RMSNormFused 会同时进行归一化和残差连接
        x, residual = self.input_layernorm.forward(x, residual)
        
        # 步骤 2: 多头自注意力
        # 使用 NVTX 标记性能分析范围, 便于在 Nsight Systems 等工具中查看
        with nvtx.range(f"MHA_{self._layer_id}"):
            x = self.self_attn.forward(x)
        
        # 步骤 3: 后注意力层归一化 + 残差连接
        x, residual = self.post_attention_layernorm.forward(x, residual)
        
        # 步骤 4: MLP 前馈网络
        with nvtx.range(f"MLP_{self._layer_id}"):
            x = self.mlp.forward(x)
        
        return x, residual


class Qwen3Model(BaseOP):
    """
    Qwen3 模型主干：完整的 Transformer 模型(不包括语言模型头)
    
    实现完整的 Qwen3 模型架构, 包括：
    1. 词嵌入层 (embed_tokens)
    2. N 个解码器层 (layers)
    3. 输出层归一化 (norm)
    
    架构流程：
    input_ids → Embedding → Layer 0 → Layer 1 → ... → Layer N → Norm → output
    
    关键特性：
    - 支持词汇表并行(在张量并行模式下)
    - 使用 OPList 管理多个解码器层
    - 残差连接在层之间累积传递
    """

    def __init__(self, config: ModelConfig):
        """
        初始化 Qwen3 模型
        
        Args:
            config: 模型配置对象, 包含所有模型超参数
        """
        # 词嵌入层
        # 使用词汇表并行嵌入, 在张量并行模式下会将词汇表分片到不同 GPU
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,  # 词汇表大小
            embedding_dim=config.hidden_size,  # 隐藏层维度
        )
        
        # 解码器层列表
        # 创建 config.num_layers 个解码器层, 每个层有唯一的 layer_id
        self.layers = OPList(
            [Qwen3DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        
        # 输出层归一化
        # 在所有解码器层之后进行最终的归一化
        self.norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        执行完整的模型前向传播, 从输入 token IDs 到隐藏状态。
        
        Args:
            input_ids: 输入 token ID 张量, 形状为 [batch_size, seq_len]
        
        Returns:
            torch.Tensor: 输出隐藏状态, 形状为 [batch_size, seq_len, hidden_size]
        
        Note:
            - 残差连接在层之间累积传递
            - 第一层时 residual 为 None, 后续层使用累积的残差
            - 最终归一化会处理残差连接并返回归一化后的输出
        """
        # 步骤 1: 词嵌入
        # 将 token IDs 转换为嵌入向量
        with nvtx.range("Embedding"):
            x = self.embed_tokens.forward(input_ids)
        
        # 初始化残差连接(第一层时使用 None)
        residual: torch.Tensor | None = None
        
        # 步骤 2: 通过所有解码器层
        # 逐层处理, 每层都会更新 x 和 residual
        for layer in self.layers.op_list:
            with nvtx.range(f"Layer_{layer._layer_id}"):
                x, residual = layer.forward(x, residual)
        
        # 步骤 3: 最终层归一化
        # 归一化并处理最终的残差连接
        # 返回 [0] 是因为 RMSNormFused.forward 返回 (output, residual) 元组
        return self.norm.forward(x, residual)[0]


class Qwen3ForCausalLM(BaseLLMModel):
    """
    Qwen3 因果语言模型：完整的语言模型(包括模型主干和语言模型头)
    
    这是最终的用户接口类, 包含：
    1. 模型主干 (model): Qwen3Model 实例
    2. 语言模型头 (lm_head): 将隐藏状态映射到词汇表 logits
    
    架构流程：
    input_ids → Model → hidden_states → LMHead → logits
    
    关键特性：
    - 支持权重共享(tie_word_embeddings): 可以将 embedding 和 lm_head 权重共享
    - 支持词汇表并行: lm_head 在张量并行模式下会分片
    - 从全局上下文获取输入, 便于与调度器集成
    """

    def __init__(self, config: ModelConfig):
        """
        初始化 Qwen3 因果语言模型
        
        Args:
            config: 模型配置对象, 包含所有模型超参数
        """
        # 模型主干：包含嵌入层、解码器层和输出归一化
        self.model = Qwen3Model(config)
        
        # 语言模型头：将隐藏状态映射到词汇表 logits
        # ParallelLMHead 支持：
        # - 词汇表并行(在张量并行模式下)
        # - 权重共享(如果 tie_word_embeddings=True, 与 embedding 共享权重)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,  # 词汇表大小
            embedding_dim=config.hidden_size,  # 隐藏层维度
            tie_word_embeddings=config.tie_word_embeddings,  # 是否共享权重
            # 如果启用权重共享, 传递 embedding 层作为共享权重
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        
        # 调用父类初始化
        super().__init__()

    def forward(self) -> torch.Tensor:
        """
        前向传播
        
        执行完整的语言模型前向传播, 从输入 token IDs 到输出 logits。
        从全局上下文获取输入, 便于与调度器集成。
        
        Returns:
            torch.Tensor: 输出 logits, 形状为 [batch_size, seq_len, vocab_size]
                        表示每个位置对每个词汇的未归一化概率
        
        Note:
            - 使用全局上下文获取输入, 而不是直接接收参数
            - 这允许调度器管理批次和上下文
            - 使用 NVTX 标记性能分析范围
        """
        # 从全局上下文获取当前批次
        ctx = get_global_ctx()
        
        # 步骤 1: 通过模型主干获取隐藏状态
        # 从 input_ids 到隐藏状态 [batch_size, seq_len, hidden_size]
        output = self.model.forward(ctx.batch.input_ids)
        
        # 步骤 2: 通过语言模型头获取 logits
        # 从隐藏状态到词汇表 logits [batch_size, seq_len, vocab_size]
        with nvtx.range("LMHead"):
            logits = self.lm_head.forward(output)
        
        return logits


__all__ = ["Qwen3ForCausalLM"]
