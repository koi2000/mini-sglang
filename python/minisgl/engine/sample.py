"""
采样器模块

实现从模型 logits 中采样下一个 token 的功能。
支持两种采样模式：
1. 贪婪解码(Greedy Decoding)：温度 <= 0 时选择概率最高的 token
2. 温度采样(Temperature Sampling)：温度 > 0 时使用温度缩放进行随机采样
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from minisgl.core import Batch


@dataclass
class BatchSamplingArgs:
    """
    批次采样参数
    
    存储批次中每个请求的采样参数。
    
    Attributes:
        temperatures: 温度张量 形状为 (batch_size,)
                     如果为 None 表示所有请求都使用贪婪解码(温度 <= 0)
    """
    temperatures: torch.Tensor | None


class Sampler:
    """
    采样器类
    
    负责从模型的 logits 输出中采样下一个 token。
    支持贪婪解码和温度采样两种模式。
    
    采样流程：
    1. prepare(): 准备采样参数 检查是否需要温度采样
    2. sample(): 根据参数选择采样模式并执行采样
    3. _sample(): 执行带温度的实际采样
    """
    
    def __init__(self, device: torch.device) -> None:
        """
        初始化采样器
        
        Args:
            device: PyTorch 设备 用于存储温度张量
        """
        self.device = device

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        """
        准备批次采样参数
        
        检查批次中所有请求的采样参数 确定是否需要温度采样。
        如果所有请求的温度都 <= 0 则使用贪婪解码(不设置温度)。
        否则为每个请求准备温度参数。
        
        Args:
            batch: 批次对象 包含多个请求
        
        Returns:
            BatchSamplingArgs: 采样参数对象
        
        工作流程：
        1. 检查所有请求的温度是否都 <= 0
        2. 如果是 返回 None(表示贪婪解码)
        3. 否则 为每个请求准备温度参数(确保温度 >= MIN_T)
        4. 将温度张量传输到设备上(使用 pin_memory 和非阻塞传输优化)
        """
        # 检查所有请求是否都使用贪婪解码(温度 <= 0)
        if all(r.sampling_params.temperature <= 0.0 for r in batch.reqs):
            # 所有请求都使用贪婪解码 不需要温度参数
            return BatchSamplingArgs(temperatures=None)
        
        # 最小温度值 防止除零错误和数值不稳定
        MIN_T = 1e-5
        
        # 为每个请求准备温度参数
        # 确保温度 >= MIN_T 避免数值问题
        return BatchSamplingArgs(
            temperatures=torch.tensor(
                [max(r.sampling_params.temperature, MIN_T) for r in batch.reqs],
                dtype=torch.float32,
                pin_memory=True,  # 使用固定内存 加速 CPU-GPU 传输
            ).to(self.device, non_blocking=True)  # 非阻塞传输到设备
        )

    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        """
        执行采样
        
        根据采样参数选择采样模式：
        - 如果 temperatures 为 None：使用贪婪解码(选择概率最高的 token)
        - 否则：使用温度采样(带随机性的采样)
        
        Args:
            logits: 模型输出的 logits 形状为 (batch_size, vocab_size)
            args: 采样参数对象
        
        Returns:
            torch.Tensor: 采样得到的 token IDs 形状为 (batch_size,)
        
        Note:
            使用 NVTX 范围标记用于性能分析
        """
        # 使用 NVTX 范围标记 用于 CUDA 性能分析工具(如 Nsight)
        with torch.cuda.nvtx.range("Sampler"):
            # 如果没有温度参数 使用贪婪解码
            if args.temperatures is None:
                # 选择每个样本中 logits 最大的索引(即概率最高的 token)
                return torch.argmax(logits, dim=-1)
            
            # 否则使用温度采样
            return self._sample(logits, args.temperatures)

    def _sample(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        """
        执行带温度的实际采样
        
        实现温度采样算法：
        1. 将 logits 除以温度(温度缩放)
        2. 应用 softmax 得到概率分布
        3. 使用多项式采样从概率分布中采样
        
        Args:
            logits: 模型输出的 logits 形状为 (batch_size, vocab_size)
            temperatures: 温度张量 形状为 (batch_size,)
        
        Returns:
            torch.Tensor: 采样得到的 token IDs 形状为 (batch_size,)
        
        算法说明：
        - 温度缩放：logits / temperature
          - 温度越高 分布越平滑(更随机)
          - 温度越低 分布越尖锐(更确定)
          - 温度 = 1 时 保持原始分布
        - Softmax：将缩放后的 logits 转换为概率分布
        - Multinomial：从概率分布中采样一个 token
        
        优化：
        - 使用 in-place 操作(div_, out=)减少内存分配
        - temperatures.unsqueeze(-1) 将形状从 (batch_size,) 变为 (batch_size, 1)
          以便与 logits (batch_size, vocab_size) 进行广播
        """
        # 温度缩放：将 logits 除以温度
        # temperatures.unsqueeze(-1) 将形状从 (batch_size,) 变为 (batch_size, 1)
        # 以便与 logits (batch_size, vocab_size) 进行广播
        logits.div_(temperatures.unsqueeze(-1))
        
        # 应用 softmax 将 logits 转换为概率分布
        # 使用 in-place 操作减少内存分配
        torch.softmax(logits, dim=-1, out=logits)
        
        # 使用多项式采样从概率分布中采样一个 token
        # num_samples=1 表示每个样本采样一个 token
        # view(-1) 将形状从 (batch_size, 1) 变为 (batch_size,)
        return torch.multinomial(logits, num_samples=1).view(-1)
