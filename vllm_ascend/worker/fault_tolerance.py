import torch
import functools
import queue
import threading
import torch_npu
import torch.distributed as dist

from datetime import timedelta
from typing import Callable,List
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.distributed.parallel_state import get_pp_group,get_tp_group
from vllm_ascend.worker.memory_block_info import MemoryBlockInfo
from vllm_ascend.worker.fault_aware import FaultAware
from vllm_ascend.worker.common import FaultAction,FaultToleranceLevel,RecoveryStatus
from vllm_ascend.worker.recovery_chain import UCEHandler, RecoveryHandler, ForceStopHandler, NetworkHandler
from vllm_ascend.worker.recovery_context import RecoveryContext

class FaultTolerance:
    _recovery_group = None
    def __init__(self,vllm_config:VllmConfig,model,level: FaultToleranceLevel = FaultToleranceLevel.OFF):
        self.model = model
        self.vllm_config = vllm_config
        self.level = level
        self.fault_queue = queue.Queue()
        self.memory_info = MemoryBlockInfo(self.model)
        self.recovery_chain = self._build_recovery_chain()

        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        self._init_recovery_group()
        self.memory_info.initialize()

        self.aware_event = threading.Event()
        if self.level != FaultToleranceLevel.OFF:
            FaultAware(
                self.rank,self.world_size,self.fault_queue,aware_event=self.aware_event
            ).start()

    def _init_recovery_group(self):
        """
        Initialize the global communication group for reporting abnormal status to fault_aware.
        """
        if not dist.is_initialized() or self.world_size == 1:
            return

        FaultTolerance._recovery_group = dist.new_group(
            ranks=None,
            timeout=timedelta(minutes=5),
            backend="gloo",
        )

        logger.info(f"Recovery group initialization successful for rank {self.rank}")

    def _build_recovery_chain(self) -> RecoveryHandler:
        """initialize recovery chain"""
        force_stop_handler = ForceStopHandler()
        network_handler = NetworkHandler()
        uce_handler = UCEHandler()

        force_stop_handler.set_next(network_handler).set_next(uce_handler)

        return force_stop_handler

    def fault_tolerance_decorator(self, func: Callable) -> Callable:
        """fault tolerance decorator is used to modify the execute_model for exception handling."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Level 0:disable fault tolerance
            if self.level == FaultToleranceLevel.OFF.value:
                return func(*args, **kwargs)
            # Enable fault tolerance
            while True:
                try:
                    output = func(*args, **kwargs)
                    return output
                except Exception as e:
                    # Encapsulate the context information required for fault recovery.
                    recovery_context = RecoveryContext(
                        model=self.model,
                        level=self.level,
                        exception=e,
                        rank=self.rank,
                        model_or_path=self.vllm_config.model_config.model,
                        memory_block_info=self.memory_info,
                        fault_queue=self.fault_queue
                    )
                    ft_action = self._handle_exception(recovery_context)
                    if torch.equal(ft_action,FaultAction.RECOMPUTE):
                        self.aware_event.set()
                        logger.info(f"Begin token re-inference at rank {self.rank}")
                        continue
                    elif torch.equal(ft_action,FaultAction.RAISE_EXCEPTION):
                        logger.info(f"Raise exception at rank {self.rank}")
                        # TODO: Need to clear cache for current batch and destroy all group
                        raise e
                    elif torch.equal(ft_action,FaultAction.RETURN):
                        logger.info(f"Abort current batch at rank {self.rank}")
                        # TODO: Need to clear cache for current batch and destroy all group
                        return None
                    else:
                        # TODO: Need to clear cache for current batch and destroy all group
                        logger.info(f"Unknown fault action found at rank {self.rank} ")
                        raise e

        return wrapper

    def _handle_exception(self, ctx: RecoveryContext) -> torch.Tensor:
        """
        Handle exception in recovery_chain and get fault action for the current batch
        """
        try:
            # 1. Handle exception in recovery_chain and get recovery status
            local_recovery_status = self.recovery_chain.handle(ctx)
            # 2. Report recovery status and get fault action
            ft_action = self._coordinate_recovery(ctx,local_recovery_status)
            return ft_action
        except Exception as inner_e:
            logger.error(f"Handle exception failed at rank {self.rank},get exception {inner_e}")
            return FaultAction.RAISE_EXCEPTION

    def _coordinate_recovery(self,ctx:RecoveryContext, local_recovery_status:torch.Tensor) -> torch.Tensor:
        """
        Rank 0 gathers recovery status and determines fault actions for each rank
        Recovery status is categorized into restart recovery and fault recovery
        Failure at any recovery stage will cause re-inference to fail
        Therefore, re-inference is executed only when both restart recovery and fault recovery succeed
        """

        # determine fault action for single rank situation
        if not dist.is_initialized() or self.world_size == 1:
            reinit_status = self._restart_and_reinit(ctx)
            if torch.equal(reinit_status,RecoveryStatus.SUCCESS):
                return self._single_node_decision(local_recovery_status)
            else:
                return FaultAction.RAISE_EXCEPTION
        #TODO:Should refactor codes below
        if self.vllm_config.parallel_config.distributed_executor_backend != (
            "external_launcher") and not get_pp_group().is_last_rank:
            get_pp_group().send_tensor_dict(torch.tensor([0]),all_gather_group=get_tp_group())

        all_recovery_status = self._gather_statuses(local_recovery_status)
        reinit_status = self._restart_and_reinit(ctx)
        all_reinit_status = self._gather_statuses(reinit_status)
        if self.rank == 0:
            has_failed = any(torch.equal(status, RecoveryStatus.FAILED) for status in all_reinit_status)
            if has_failed:
                reinit_actions = self._analyze_global_status(all_reinit_status)
                return self._scatter_ft_actions(reinit_actions)
            else:
                ft_actions = self._analyze_global_status(all_recovery_status)
                return self._scatter_ft_actions(ft_actions)
        else:
            return self._receive_ft_actions()

    def _single_node_decision(self, local_status: torch.Tensor) -> torch.Tensor:
        """
        Single rank situation,determine fault action base on local status
        """
        if torch.equal(local_status, RecoveryStatus.SUCCESS):
            return FaultAction.RECOMPUTE
        else:
            return FaultAction.RAISE_EXCEPTION

    def _gather_statuses(self, local_status:torch.Tensor) -> List[torch.Tensor]:
        """Rank 0 gathers status from each rank"""
        try:
            if self.rank == 0:
                gather_list = [torch.zeros_like(local_status) for _ in range(self.world_size)]
                dist.gather(
                    local_status,
                    gather_list=gather_list,
                    dst=0,
                    group=FaultTolerance._recovery_group
                )
                return gather_list
            else:
                # 其他rank只发送，不接收
                dist.gather(local_status, gather_list=None, dst=0,group=FaultTolerance._recovery_group)
                return []  # 非rank0返回空列表
        except Exception as inner_e:
            logger.error(f"Gather status failed,get exception:{inner_e}")
            if self.rank == 0:
                return [RecoveryStatus.FAILED for _ in range(self.world_size)]
            return []

    def _analyze_global_status(self, all_recovery_statuses: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Analyze status and generate decisions
        """
        success_ranks = []
        failure_ranks = []

        for rank, recovery_status in enumerate(all_recovery_statuses):
            if torch.equal(recovery_status, RecoveryStatus.SUCCESS):
                success_ranks.append(rank)
            elif torch.equal(recovery_status, RecoveryStatus.FAILED):
                failure_ranks.append(rank)
            else:
                logger.warning(f"Unknown status tensor from rank {rank}: {recovery_status}")
                failure_ranks.append(rank)

        logger.info(f"Global recovery: {len(success_ranks)} success, {len(failure_ranks)} failure")

        decisions = []
        if not failure_ranks:
            logger.info("All ranks recovered, Determine RECOMPUTE for all rank")
            decisions = [FaultAction.RECOMPUTE] * self.world_size
        elif not success_ranks:
            logger.warning("All ranks failed, Determine RAISE_EXCEPTION for all rank")
            decisions = [FaultAction.RAISE_EXCEPTION] * self.world_size
        else:
            logger.warning(f"Partial recovery - success ranks: {success_ranks}")
            for rank in range(self.world_size):
                if rank in success_ranks:
                    decisions.append(FaultAction.RETURN)
                else:
                    decisions.append(FaultAction.RAISE_EXCEPTION)

        return decisions

    def _scatter_ft_actions(self, ft_actions: List[torch.Tensor]) -> torch.Tensor:
        """Rank 0 distributed fault action to each rank"""
        recv_ft_action = torch.tensor([0])
        dist.scatter(
            recv_ft_action,
            scatter_list=ft_actions,
            src=0,
            group=FaultTolerance._recovery_group
        )
        return recv_ft_action

    def _receive_ft_actions(self) -> torch.Tensor:
        """Rank 1 ...N receive fault action"""
        recv_ft_action = torch.tensor([0])
        dist.scatter(
            recv_ft_action,
            scatter_list=None,
            src=0,
            group=FaultTolerance._recovery_group
        )
        return recv_ft_action

    def _restart_and_reinit(self,ctx:RecoveryContext) -> torch.Tensor:
        """
        Restart device and reinit process group
        """
        try:
            torch_npu.npu.restart_device(torch.npu.current_device())
            torch.distributed.reinit_process_group(group=None,rebuild_link=False)
            reinit_status = RecoveryStatus.SUCCESS
        except Exception as inner_e:
            logger.error(f"Failed to restart and reinit process group for rank {self.rank},get exception :{inner_e}")
            ctx.exception = inner_e
            reinit_status = RecoveryStatus.FAILED
        return reinit_status

    def destroy_recovery_group(self):
        """Destroy recovery process group and fault_aware"""
        if FaultTolerance._recovery_group is None:
            return

        logger.info("Destroying recovery process group")
        try:
            dist.destroy_process_group(FaultTolerance._recovery_group)
            FaultTolerance._recovery_group = None
            logger.info("Successfully destroyed recovery process group")
        except Exception as e:
            logger.error(f"Failed to destroy recovery process group: {e}")

