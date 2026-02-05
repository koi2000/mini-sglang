"""
模型权重加载模块：从 HuggingFace 加载和预处理模型权重

该模块提供从 HuggingFace 模型仓库或本地目录加载模型权重的功能。
主要特性包括：
1. 支持本地目录和 HuggingFace 仓库 ID
2. 支持张量并行(TP)的权重分片
3. 优化权重布局(合并 QKV、gate_up 投影)
4. 使用 safetensors 格式安全加载权重
"""

from __future__ import annotations

import glob
import os
from typing import Dict

import safetensors
import torch
from huggingface_hub import snapshot_download
from minisgl.distributed import get_tp_info
from minisgl.utils import divide_up
from tqdm.asyncio import tqdm


class DisabledTqdm(tqdm):
    """
    禁用进度条的 Tqdm 类
    
    用于在下载模型时禁用进度条显示, 避免在非交互式环境或日志中产生噪音。
    继承自 tqdm, 但默认禁用所有输出。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)


def _shard_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    分片状态字典：为张量并行(TP)准备权重
    
    在张量并行模式下, 需要将模型权重分片到不同的 GPU 上。
    不同的层需要不同的分片策略：
    
    1. 沿 dim=0 分片(列分片)：
       - Q/K/V 投影、gate/up 投影
       - 这些层的输出会被分片到不同 GPU
    
    2. 沿 dim=1 分片(行分片)：
       - O 投影、down 投影
       - 这些层的输入来自不同 GPU, 需要分片权重
    
    3. 词汇表分片：
       - embedding 和 lm_head
       - 按词汇表维度分片, 每个 GPU 负责一部分词汇
    
    4. 其他层：
       - 保持不变(如 LayerNorm、RMSNorm 等)
    
    Args:
        state_dict: 完整的模型权重字典
    
    Returns:
        Dict[str, torch.Tensor]: 分片后的权重字典, 只包含当前 rank 需要的权重
    
    Note:
        - 使用 chunk() 方法将张量均匀分片
        - 使用 divide_up() 确保词汇表分片时向上取整, 处理不能整除的情况
    """
    shard_state_dict: Dict[str, torch.Tensor] = {}
    tp_info = get_tp_info()
    r = tp_info.rank  # 当前进程的 rank (0, 1, 2, ...)
    n = tp_info.size  # TP 组的总进程数
    
    # 需要沿 dim=0 分片的层(列分片)
    # 这些层的输出特征会被分片到不同 GPU
    SPLIT_DIM_0_LIST = [
        ".q_proj",  # Query 投影
        ".k_proj",  # Key 投影
        ".v_proj",  # Value 投影
        ".gate_proj",  # Gate 投影(用于 SwiGLU 等激活函数)
        ".up_proj",  # Up 投影(用于 MLP)
    ]
    
    # 需要沿 dim=1 分片的层(行分片)
    # 这些层的输入来自不同 GPU, 需要分片权重以匹配
    SPLIT_DIM_1_LIST = [
        ".o_proj",  # Output 投影(注意力输出)
        ".down_proj",  # Down 投影(MLP 输出)
    ]
    
    for key, value in state_dict.items():
        # 检查是否需要沿 dim=0 分片
        if any(key.count(sub) for sub in SPLIT_DIM_0_LIST):
            # 将张量沿 dim=0 分成 n 份, 取第 r 份
            # 例如: shape [4096, 4096] -> [1024, 4096] (n=4, r=0)
            shard_state_dict[key] = value.chunk(n, dim=0)[r]
        
        # 检查是否需要沿 dim=1 分片
        elif any(key.count(sub) for sub in SPLIT_DIM_1_LIST):
            # 将张量沿 dim=1 分成 n 份, 取第 r 份
            # 例如: shape [4096, 4096] -> [4096, 1024] (n=4, r=0)
            shard_state_dict[key] = value.chunk(n, dim=1)[r]
        
        # 处理词汇表相关的层(embedding 和 lm_head)
        elif key.count("lm_head") or key.count("embed_tokens"):
            num_embeddings = value.shape[0]  # 词汇表大小
            
            # 计算每个分片的词汇表大小(向上取整)
            # 例如: 50000 词汇, 4 GPU -> 每个 GPU 12500
            num_embeddings_per_partition = divide_up(num_embeddings, n)
            
            # 计算当前 rank 负责的词汇表范围
            vocab_start_idx = r * num_embeddings_per_partition
            vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
            
            # 提取对应的词汇表切片
            # 例如: [50000, 4096] -> [12500, 4096] (n=4, r=0)
            shard_state_dict[key] = value[vocab_start_idx:vocab_end_idx, :]
        
        else:
            # 其他层(如 LayerNorm、RMSNorm 等)保持不变
            # 这些层在每个 GPU 上都有完整副本
            shard_state_dict[key] = value
    
    return shard_state_dict


def _merge_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    合并状态字典：优化权重布局以提高计算效率
    
    将某些相关的投影层合并, 以减少矩阵乘法的次数：
    
    1. QKV 合并：
       - 将 q_proj、k_proj、v_proj 合并为 qkv_proj
       - 在注意力计算时, 可以一次性计算 Q、K、V, 减少内存访问
    
    2. Gate-Up 合并：
       - 将 gate_proj、up_proj 合并为 gate_up_proj
       - 在 MLP 计算时, 可以并行计算 gate 和 up, 提高效率
    
    这种合并是常见的优化技术, 可以减少 kernel 启动开销和内存访问。
    
    Args:
        state_dict: 权重字典(可能已经分片)
    
    Returns:
        Dict[str, torch.Tensor]: 合并后的权重字典
    
    Note:
        - 合并后的权重在 dim=0 上拼接
        - 原始的分立权重会被删除
        - 如果遇到 .k_proj、.v_proj 或 .up_proj(但不是 .gate_proj),
          会跳过, 因为它们会在处理 .q_proj 或 .gate_proj 时被处理
    """
    filtered_state_dict: Dict[str, torch.Tensor] = {}
    
    # 使用 list() 创建副本, 因为我们在迭代时可能会删除键
    for key in list(state_dict.keys()):
        # 处理 QKV 投影：合并 q_proj、k_proj、v_proj
        if key.count(".q_proj"):
            q_proj = state_dict[key]
            k_proj = state_dict[key.replace(".q_proj", ".k_proj")]
            v_proj = state_dict[key.replace(".q_proj", ".v_proj")]
            
            # 创建新键名, 将 .q_proj 替换为 .qkv_proj
            new_key = key.replace(".q_proj", ".qkv_proj")
            
            # 沿 dim=0 拼接: [d, h] + [d, h] + [d, h] -> [3d, h]
            # 例如: [4096, 4096] * 3 -> [12288, 4096]
            filtered_state_dict[new_key] = torch.cat([q_proj, k_proj, v_proj], dim=0)
            
            # 删除原始的分立权重
            del state_dict[key]
            del state_dict[key.replace(".q_proj", ".k_proj")]
            del state_dict[key.replace(".q_proj", ".v_proj")]
        
        # 处理 Gate-Up 投影：合并 gate_proj、up_proj
        elif key.count(".gate_proj"):
            gate_proj = state_dict[key]
            up_proj = state_dict[key.replace(".gate_proj", ".up_proj")]
            
            # 创建新键名, 将 .gate_proj 替换为 .gate_up_proj
            new_key = key.replace(".gate_proj", ".gate_up_proj")
            
            # 沿 dim=0 拼接: [d, h] + [d, h] -> [2d, h]
            # 例如: [4096, 11008] * 2 -> [8192, 11008]
            filtered_state_dict[new_key] = torch.cat([gate_proj, up_proj], dim=0)
            
            # 删除原始的分立权重
            del state_dict[key]
            del state_dict[key.replace(".gate_proj", ".up_proj")]
        
        # 跳过已经被合并的权重
        # 这些权重在处理 .q_proj 或 .gate_proj 时已经被处理
        elif key.count(".k_proj") or key.count(".v_proj") or key.count("up_proj"):
            continue
        
        else:
            # 其他权重保持不变
            filtered_state_dict[key] = state_dict[key]
    
    return filtered_state_dict


def load_hf_weight(model_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    从 HuggingFace 加载模型权重
    
    支持两种输入方式：
    1. 本地目录路径：直接加载目录中的 safetensors 文件
    2. HuggingFace 仓库 ID：从 HuggingFace Hub 下载模型
    
    加载流程：
    1. 确定模型文件夹路径(本地或下载)
    2. 加载所有 .safetensors 文件
    3. 如果启用张量并行, 分片权重
    4. 将权重移动到指定设备
    5. 合并相关权重以优化计算
    
    Args:
        model_path: 模型路径, 可以是：
                   - 本地目录路径(如 "/path/to/model")
                   - HuggingFace 仓库 ID(如 "meta-llama/Llama-2-7b-hf")
        device: 目标设备, 权重将被移动到此设备
    
    Returns:
        Dict[str, torch.Tensor]: 处理后的权重字典
                                - 键: 权重名称(可能已合并, 如 "qkv_proj")
                                - 值: 权重张量(已移动到指定设备, 可能已分片)
    
    Raises:
        ValueError: 如果 model_path 既不是本地目录也不是有效的 HuggingFace 仓库 ID
    
    Example:
        >>> weights = load_hf_weight("meta-llama/Llama-2-7b-hf", torch.device("cuda:0"))
        >>> print(weights.keys())  # 查看权重名称
    """
    # 步骤 1: 确定模型文件夹路径
    if os.path.isdir(model_path):
        # 本地目录：直接使用
        hf_folder = model_path
    else:
        # HuggingFace 仓库 ID：尝试下载
        try:
            hf_folder = snapshot_download(
                model_path,
                allow_patterns=["*.safetensors"],  # 只下载 safetensors 文件
                tqdm_class=DisabledTqdm,  # 禁用进度条
            )
        except Exception:
            raise ValueError(
                f"Model path '{model_path}' is neither a local directory nor a valid HuggingFace repository ID"
            )

    # 步骤 2: 查找并加载所有 safetensors 文件
    # 注意: 注释说的是 *.pt, 但实际代码使用的是 *.safetensors
    files = glob.glob(f"{hf_folder}/*.safetensors")
    state_dict: Dict[str, torch.Tensor] = {}
    
    # 按文件名排序, 确保加载顺序一致
    for file in sorted(files):
        # 使用 safetensors 安全加载(避免 pickle 安全风险)
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            # 加载文件中的所有张量
            for name in f.keys():
                state_dict[name] = f.get_tensor(name)

    # 步骤 3: 如果启用张量并行, 分片权重
    if get_tp_info().size > 1:
        state_dict = _shard_state_dict(state_dict)

    # 步骤 4: 将所有权重移动到指定设备
    state_dict = {k: v.to(device) for k, v in state_dict.items()}
    
    # 步骤 5: 合并相关权重以优化计算
    return _merge_state_dict(state_dict)
