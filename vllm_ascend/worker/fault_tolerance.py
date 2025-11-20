import torch
import functools
import queue
import torch.distributed as dist

from vllm.config import VllmConfig
from datetime import timedelta
from typing import Callable,List
from memory_block_info import MemoryBlockInfo
from vllm.logger import logger
from vllm_ascend.worker.common import FaultAction
from vllm_ascend.worker.recovery_chain import UCEHandler, RecoveryHandler, ForceStopHandler, NetworkHandler
from vllm_ascend.worker.recovery_context import RecoveryContext
from common import FaultToleranceLevel,RecoveryStatus

class FaultTolerance:
    _recovery_group = None
    def __init__(self,vllm_config:VllmConfig,model,level: FaultToleranceLevel = FaultToleranceLevel.OFF):
        self.model = model
        self.vllm_config = vllm_config
        self.level = level
        self.fault_queue = queue.Queue()
        self.memory_info = MemoryBlockInfo(self.model)
        self.recovery_chain = self._build_recovery_chain()

        # 分布式属性初始化
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        self.memory_info.initialize()
        self._init_recovery_group()
    #TODO: 不同并行策略下，通信组的初始化方式待调整（所需参数待确认）
    def _init_recovery_group(self):
        """初始化恢复专用进程组"""
        if not dist.is_initialized() or self.world_size == 1:
            return

        # 创建恢复专用进程组
        FaultTolerance._recovery_group = dist.new_group(
            ranks=None,  # 包含所有rank
            timeout=timedelta(minutes=5),
            backend="gloo",  # 使用gloo后端
        )

        logger.info("Recovery process group initialized successfully")

    def _build_recovery_chain(self) -> RecoveryHandler:
        """initialize recovery chain"""
        force_stop_handler = ForceStopHandler()
        network_handler = NetworkHandler()
        uce_handler = UCEHandler()

        force_stop_handler.set_next(network_handler).set_next(uce_handler)

        return force_stop_handler

    def fault_tolerance_decorator(self, func: Callable) -> Callable:
        """fault tolerance decorator"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # disable fault tolerance
            if self.level == FaultToleranceLevel.OFF.value:
                return func(*args, **kwargs)
            # enable fault tolerance
            while True:
                try:
                    output = func(*args, **kwargs)
                    return output
                except Exception as e:
                    recovery_context = RecoveryContext(
                        self.model,
                        self.level,
                        e,
                        self.rank,
                        self.vllm_config.model_config.model,
                        self.memory_info,
                        fault_queue=self.fault_queue
                    )
                    ft_action = self._handle_exception(recovery_context)
                    if torch.equal(ft_action,FaultAction.RECOMPUTE.value):
                        continue
                    elif torch.equal(ft_action,FaultAction.RAISE_EXCEPTION.value):
                        raise e
                    elif torch.equal(ft_action,FaultAction.RETURN.value):
                        return None
                    else:
                        raise e

        return wrapper

    def _handle_exception(self, ctx: RecoveryContext) -> torch.Tensor:
        try:
            # 1. 责任链处理异常,并返回故障恢复状态
            local_recovery_status = self.recovery_chain.handle(ctx)  # 返回Tensor
            # 2. 故障恢复状态上报，请求决策获取
            ft_action = self._coordinate_recovery(local_recovery_status)
            # 3. 返回请求处理操作
            return ft_action

        except Exception as inner_e:
            #TODO: 处理恢复失败时的异常
            logger.error(f"Error in exception handling: {inner_e}")
            raise inner_e

    def _coordinate_recovery(self, local_recovery_status:torch.Tensor) -> FaultAction:
        """
        Rank 0 Gather Recovery Status and decide global fault action
        """
        if not dist.is_initialized() or self.world_size == 1:
            return self._single_node_decision(local_recovery_status)

        # 收集所有rank的状态
        all_status_tensors = self._gather_recovery_statuses(local_recovery_status)

        if self.rank == 0:
            ft_actions = self._analyze_global_status(all_status_tensors)
            return self._scatter_ft_actions(ft_actions)
        else:
            return self._receive_ft_actions()

    def _single_node_decision(self, local_recovery_status: torch.Tensor) -> torch.Tensor:
        """单机情况下依据本地恢复情况直接做决策"""
        if torch.equal(local_recovery_status, RecoveryStatus.SUCCESS_RECOMPUTE):
            return FaultAction.RECOMPUTE
        else:
            return FaultAction.RAISE_EXCEPTION

    def _gather_recovery_statuses(self, local_recovery_status:torch.Tensor) -> List[torch.Tensor]:
        """使用gather收集恢复状态"""
        if self.rank == 0:
            # Rank 0 准备接收缓冲区
            gather_list = [torch.zeros_like(local_recovery_status) for _ in range(self.world_size)]
            dist.gather(
                local_recovery_status,
                gather_list=gather_list,
                dst=0,
                group=FaultTolerance._recovery_group
            )
            return gather_list
        else:
            # 其他rank只发送，不接收
            dist.gather(local_recovery_status, gather_list=None, dst=0)
            return []  # 非rank0返回空列表

    def _analyze_global_status(self, all_recovery_statuses: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Analyze global status and generate decisions
        """
        success_ranks = []
        failure_ranks = []

        for rank, recovery_status in enumerate(all_recovery_statuses):
            if torch.equal(recovery_status, RecoveryStatus.SUCCESS_RECOMPUTE):
                success_ranks.append(rank)
            elif torch.equal(recovery_status, RecoveryStatus.FAILED_ABORT):
                failure_ranks.append(rank)
            else:
                logger.warning(f"Unknown status tensor from rank {rank}: {recovery_status}")
                failure_ranks.append(rank)

        logger.info(f"Global recovery: {len(success_ranks)} success, {len(failure_ranks)} failure")

        decisions = []
        if not failure_ranks:  # 全部恢复成功，执行重推
            logger.info("All ranks recovered, issuing RECOMPUTE to all")
            decisions = [FaultAction.RECOMPUTE] * self.world_size
        elif not success_ranks:  # 全部恢复失败，均抛出异常
            logger.warning("All ranks failed, issuing RAISE_EXCEPTION to all")
            decisions = [FaultAction.RAISE_EXCEPTION] * self.world_size
        else:  # 一部分恢复成功，一部分恢复失败
            logger.warning(f"Partial recovery - success ranks: {success_ranks}")
            for rank in range(self.world_size):
                if rank in success_ranks:
                    decisions.append(FaultAction.RETURN)
                else:
                    decisions.append(FaultAction.RAISE_EXCEPTION)

        return decisions

    def _scatter_ft_actions(self, ft_actions: List[torch.Tensor]) -> torch.Tensor:
        """分发决策"""
        # Rank 0分发决策
        recv_ft_action = torch.tensor([0])
        dist.scatter(
            recv_ft_action,
            scatter_list=ft_actions,
            src=0,
            group=FaultTolerance._recovery_group
        )
        return recv_ft_action

    def _receive_ft_actions(self) -> torch.Tensor:
        """非Rank 0接收决策"""
        recv_ft_action = torch.tensor([0])
        dist.scatter(
            recv_ft_action,
            scatter_list=None,
            src=0,
            group=FaultTolerance._recovery_group
        )
        return recv_ft_action

    def destroy_recovery_group(self):
        """销毁恢复进程组"""
        if FaultTolerance._recovery_group is None:
            return

        logger.info("Destroying recovery process group")
        try:
            dist.destroy_process_group(FaultTolerance._recovery_group)
            FaultTolerance._recovery_group = None
            logger.info("Successfully destroyed recovery process group")
        except Exception as e:
            logger.error(f"Failed to destroy recovery process group: {e}")
