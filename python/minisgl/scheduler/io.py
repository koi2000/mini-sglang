"""
调度器 I/O 模块

处理调度器与 tokenizer 之间的通信。
支持单 rank 和多 rank 模式 使用 ZMQ 进行进程间通信。
"""

from torch._C._distributed_c10d import ProcessGroup


from __future__ import annotations

from typing import TYPE_CHECKING, Final, List

import torch
from minisgl.message import BaseBackendMsg, BaseTokenizerMsg, BatchTokenizerMsg
from minisgl.utils import ZmqPubQueue, ZmqPullQueue, ZmqPushQueue, ZmqSubQueue, init_logger

if TYPE_CHECKING:
    from .config import SchedulerConfig

logger = init_logger(__name__)


class SchedulerIOMixin:
    """
    调度器 I/O 操作的 Mixin 类
    
    这是一个 Mixin 类 用于为调度器提供 I/O 功能。
    负责处理调度器与 tokenizer 之间的通信。
    
    通信架构：
    - 使用 ZMQ(ZeroMQ)进行进程间通信
    - 支持单 rank 和多 rank 模式
    - 在多 rank 模式下 rank 0 负责接收消息并广播给其他 rank
    
    通信模式：
    1. 单 rank 模式：直接与 tokenizer 通信
    2. 多 rank 模式：
       - Rank 0: 从 tokenizer 接收消息 广播给其他 rank
       - Rank 1+: 从 rank 0 接收广播的消息
    3. 离线模式：用于测试 不进行实际网络通信
    
    公共接口：
        receive_msg: 从 tokenizer 接收消息
        send_result: 向 tokenizer 发送结果
        sync_all_ranks: 同步所有 rank(CPU 侧)
    """

    def __init__(self, config: SchedulerConfig, tp_cpu_group: torch.distributed.ProcessGroup):
        """
        初始化 I/O 组件
        
        Args:
            config: 调度器配置 包含网络地址等信息
            tp_cpu_group: Tensor Parallelism 的 CPU 进程组 用于多 rank 同步
        """
        tp_info = config.tp_info
        self.tp_cpu_group: Final[ProcessGroup] = tp_cpu_group  # CPU 进程组 用于同步
        
        # 离线模式：用于测试 不进行实际网络通信
        if config.offline_mode:
            self.receive_msg = self.offline_receive_msg
            self.send_result = self.offline_send_result
            return  # early exit

        # 只有主 rank(rank 0)需要与 tokenizer 直接通信
        if tp_info.is_primary():
            # 从 tokenizer 接收消息的队列(Pull 模式)
            self._recv_from_tokenizer: Final = ZmqPullQueue(
                config.zmq_backend_addr,
                create=True,  # 创建队列
                decoder=BaseBackendMsg.decoder,  # 消息解码器
            )
            # 向 tokenizer 发送结果的队列(Push 模式)
            self._send_into_tokenizer: Final = ZmqPushQueue(
                config.zmq_detokenizer_addr,
                create=config.backend_create_detokenizer_link,
                encoder=BaseTokenizerMsg.encoder,  # 消息编码器
            )

        # 默认使用单 rank 模式
        recv = self._recv_msg_single_rank
        send = self._reply_tokenizer_rank0
        
        # 如果是多 rank 模式 需要设置不同的接收和发送方法
        if tp_info.size > 1:
            if tp_info.is_primary():
                # Rank 0：接收消息并广播给其他 rank
                recv = self._recv_msg_multi_rank0
                # 用于向其他 rank 广播消息的队列(Pub 模式)
                self._send_into_ranks: Final = ZmqPubQueue(
                    config.zmq_scheduler_broadcast_addr, 
                    create=True, 
                    encoder=BaseBackendMsg.encoder
                )
            else:
                # Rank 1+：从 rank 0 接收广播的消息
                recv = self._recv_msg_multi_rank1
                send = self._reply_tokenizer_rank1  # 非主 rank 不发送结果
                # 从 rank 0 接收广播消息的队列(Sub 模式)
                self._recv_from_rank0: Final = ZmqSubQueue(
                    config.zmq_scheduler_broadcast_addr,
                    create=False,  # 不创建队列(由 rank 0 创建)
                    decoder=BaseBackendMsg.decoder,
                )

        # 设置公共接口
        self.receive_msg = recv
        self.send_result = send

    def run_when_idle(self):
        """
        空闲时运行的回调函数
        
        在阻塞接收消息时调用 用于在等待消息时执行其他任务。
        需要在子类中实现。
        """
        raise NotImplementedError("should be implemented")

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """
        离线模式：接收消息
        
        用于测试模式 不进行实际网络通信。
        需要在子类中实现。
        
        Args:
            blocking: 是否阻塞等待消息
        
        Returns:
            接收到的消息列表
        """
        raise NotImplementedError("should be implemented")

    def offline_send_result(self, reply: BatchTokenizerMsg) -> None:
        """
        离线模式：发送结果
        
        用于测试模式 不进行实际网络通信。
        需要在子类中实现。
        
        Args:
            reply: 要发送的批次消息
        """
        raise NotImplementedError("should be implemented")

    def sync_all_ranks(self) -> None:
        """
        同步所有 rank
        
        使用进程组的 barrier 操作 确保所有 rank 同步。
        用于确保多 rank 之间的协调。
        """
        self.tp_cpu_group.barrier().wait()

    def _recv_msg_single_rank(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """
        单 rank 模式：接收消息
        
        直接从 tokenizer 接收消息 不涉及多 rank 通信。
        
        Args:
            blocking: 如果为 True 阻塞等待至少一条消息；否则只接收已存在的消息
        
        Returns:
            接收到的消息列表
        """
        pending_msgs: List[BaseBackendMsg] = []
        
        # 如果阻塞模式 等待至少一条消息
        if blocking:
            self.run_when_idle()  # 在等待时执行其他任务
            pending_msgs.append(self._recv_from_tokenizer.get())
        
        # 接收所有已存在的消息
        while not self._recv_from_tokenizer.empty():
            pending_msgs.append(self._recv_from_tokenizer.get())
        
        return pending_msgs

    def _recv_msg_multi_rank0(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """
        多 rank 模式: Rank 0 接收消息
        
        Rank 0 负责从 tokenizer 接收消息 并广播给其他 rank。
        
        Args:
            blocking: 如果为 True 阻塞等待至少一条消息
        
        Returns:
            接收到的消息列表
        
        工作流程：
        1. 如果阻塞模式 接收一条消息并立即广播
        2. 收集所有待处理的消息(原始字节)
        3. 广播消息数量给其他 rank
        4. 逐个广播消息并解码
        """
        pending_msgs: List[BaseBackendMsg] = []
        
        # 如果阻塞模式 接收一条消息并立即广播
        if blocking:
            raw = self._recv_from_tokenizer.get_raw()  # 获取原始字节
            self._send_into_ranks.put_raw(raw)  # 立即广播给其他 rank
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))  # 解码并添加到列表

        # 收集所有待处理的原始消息
        pending_raw_msgs: List[bytes] = []
        while not self._recv_from_tokenizer.empty():
            pending_raw_msgs.append(self._recv_from_tokenizer.get_raw())

        # 广播消息数量给所有 rank 确保其他 rank 知道要接收多少条消息
        src_tensor = torch.tensor(len(pending_raw_msgs))
        self.tp_cpu_group.broadcast(src_tensor, root=0).wait()

        # 逐个广播消息
        for raw in pending_raw_msgs:
            self._send_into_ranks.put_raw(raw)  # 广播原始字节
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))  # 解码并添加到列表
        
        return pending_msgs

    def _recv_msg_multi_rank1(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """
        多 rank 模式: Rank 1+ 接收消息
        Rank 1+ 从 rank 0 接收广播的消息。
        
        Args:
            blocking: 如果为 True 阻塞等待至少一条消息
        
        Returns:
            接收到的消息列表
        
        工作流程：
        1. 如果阻塞模式 接收一条消息
        2. 从 rank 0 接收消息数量
        3. 根据消息数量接收所有消息
        """
        pending_msgs: List[BaseBackendMsg] = []
        
        # 如果阻塞模式 接收一条消息
        if blocking:
            pending_msgs.append(self._recv_from_rank0.get())

        # 确保所有 rank 都有相同数量的消息
        # 从 rank 0 接收消息数量
        dst_tensor = torch.tensor(-1)
        self.tp_cpu_group.broadcast(dst_tensor, root=0).wait()
        dst_length = int(dst_tensor.item())

        # 根据消息数量接收所有消息
        for _ in range(dst_length):
            pending_msgs.append(self._recv_from_rank0.get())
        
        return pending_msgs

    def _reply_tokenizer_rank0(self, reply: BatchTokenizerMsg) -> None:
        """
        多 rank 模式：Rank 0 向 tokenizer 发送结果
        
        只有 rank 0 负责向 tokenizer 发送结果。
        
        Args:
            reply: 要发送的批次消息
        
        优化：
        - 如果只有一条消息 直接发送单个消息对象
        - 如果有多条消息 发送批次消息对象
        """
        num_reply = len(reply.data)
        logger.debug_rank0(f"Replying to tokenizer: {num_reply} messages")
        
        # 如果只有一条消息 直接发送单个消息(避免不必要的批次包装)
        if num_reply == 1:
            self._send_into_tokenizer.put(reply.data[0])
        # 如果有多条消息 发送批次消息
        elif num_reply > 1:
            self._send_into_tokenizer.put(reply)

    def _reply_tokenizer_rank1(self, reply: BatchTokenizerMsg) -> None:
        """
        多 rank 模式：Rank 1+ 向 tokenizer 发送结果
        
        非主 rank 不发送结果 只有 rank 0 负责发送。
        这是为了避免重复发送。
        
        Args:
            reply: 要发送的批次消息(实际上不使用)
        """
        _ = reply  # do nothing for non-primary ranks
