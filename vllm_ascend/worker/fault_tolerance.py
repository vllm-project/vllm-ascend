import torch
import functools
import queue
import threading
import torch_npu
import torch.distributed as dist

from datetime import timedelta
from typing import Any, Callable,List
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

from vllm_ascend.worker.fault_aware import FaultAware
from vllm_ascend.worker.common import FaultAction,RecoveryStatus
from vllm_ascend.worker.recovery_handler import RecoveryHandlerManager, ForceStopHandler, NetworkHandler
from vllm_ascend.worker.recovery_context import RecoveryContext

class FaultTolerance:
    _recovery_group = None
    _sync_group = None

    def __init__(self,vllm_config:VllmConfig,model_runner,execute_model_func):
        self.model_runner = model_runner
        self.execute_model_func = execute_model_func
        self.vllm_config = vllm_config

        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        self.fault_queue = queue.Queue()
        self.recovery_handler_manager = self._build_recovery_handler_manager()

        self._init_recovery_group()
        self._init_sync_group()

        self.state_backup = {}
        self.aware_event = threading.Event()
        self.stop_event = threading.Event()
        FaultAware(
            self.rank,self.world_size,self.fault_queue,aware_event=self.aware_event,stop_event=self.stop_event
        ).start()

    def _init_recovery_group(self):
        """
        Initialize the global communication group for status collection
        """
        if not dist.is_initialized() or self.world_size == 1:
            return

        FaultTolerance._recovery_group = dist.new_group(
            ranks=None,
            timeout=timedelta(minutes=5),
            backend="gloo",
        )

        logger.info(f"Recovery group initialization successful for rank {self.rank}")
    def _init_sync_group(self):
        """
        Initialize the global communication group for synchronization
        """
        if not dist.is_initialized() or self.world_size == 1:
            return
        FaultTolerance._sync_group = dist.new_group(
            ranks=None,
            timeout=timedelta(minutes=5),
            backend="hccl"
        )

    def _build_recovery_handler_manager(self) -> RecoveryHandlerManager:
        """initialize recovery chain"""
        recovery_handler_manager = RecoveryHandlerManager()

        force_handler = ForceStopHandler()
        network_handler = NetworkHandler()

        recovery_handler_manager.register_handler(force_handler)
        recovery_handler_manager.register_handler(network_handler)

        return recovery_handler_manager

    def execute_model_decorator(self,func:Callable,max_retries: int,dummy_run: bool) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.state_backup = self._create_essential_state_backup(*args, **kwargs)
            for attempt in range(max_retries + 1):
                try:
                    output = func(*args, **kwargs)
                    if dummy_run:
                        self._all_gather_for_sync_group()
                    return output
                except Exception as e:
                    if attempt >= max_retries:
                        logger.warning(f"Max retries {max_retries} exceeded at rank {self.rank}，raising exception: {e}")
                        raise e
                    # Encapsulate the context information required for fault recovery.
                    recovery_context = RecoveryContext(
                        exception=e,
                        fault_queue=self.fault_queue,
                        back_up=self.state_backup
                    )
                    ft_action = self._handle_exception(recovery_context)
                    if torch.equal(ft_action, FaultAction.RECOMPUTE):
                        self.aware_event.set()
                        logger.info(f"Begin token re-inference at rank {self.rank}")
                        continue
                    elif torch.equal(ft_action, FaultAction.RAISE_EXCEPTION):
                        logger.info(f"Raise exception at rank {self.rank}")
                        raise e
                    elif torch.equal(ft_action, FaultAction.RETURN):
                        logger.info(f"Abort current batch at rank {self.rank}")
                        return EMPTY_MODEL_RUNNER_OUTPUT
            return EMPTY_MODEL_RUNNER_OUTPUT
        return wrapper

    def sample_token_decorator(self,func:Callable,max_retries: int) -> Callable:
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    output = func(*args, **kwargs)
                    if output is not None:
                        self._all_gather_for_sync_group()
                    return output
                except Exception as e:
                    if attempt >= max_retries:
                        logger.warning(f"Max retries {max_retries} exceeded at rank {self.rank}，raising exception: {e}")
                        raise e
                    # Encapsulate the context information required for fault recovery.
                    recovery_context = RecoveryContext(
                        exception=e,
                        fault_queue=self.fault_queue,
                        back_up=self.state_backup
                    )
                    ft_action = self._handle_exception(recovery_context)
                    if torch.equal(ft_action, FaultAction.RECOMPUTE):
                        self.aware_event.set()
                        logger.info(f"Begin re-execute model at rank {self.rank}")
                        # re-execute model first
                        self.execute_model_func(*self.state_backup['args'], **self.state_backup['kwargs'])
                        logger.info(f"Begin token re-inference at rank {self.rank}")
                        continue
                    elif torch.equal(ft_action, FaultAction.RAISE_EXCEPTION):
                        logger.info(f"Raise exception at rank {self.rank}")
                        raise e
                    elif torch.equal(ft_action, FaultAction.RETURN):
                        logger.info(f"Abort current batch at rank {self.rank}")
                        return EMPTY_MODEL_RUNNER_OUTPUT
            return EMPTY_MODEL_RUNNER_OUTPUT

        return wrapper

    def _handle_exception(self, ctx: RecoveryContext) -> torch.Tensor:
        """
        Handle exception in recovery_chain and get fault action for the current batch
        """
        handler = self.recovery_handler_manager.find_handler(ctx)
        # No target exception ,return raise Exception
        if handler is None:
            return FaultAction.RAISE_EXCEPTION
        # Wait until stop_device finished
        self.stop_event.wait()
        self.stop_event.clear()
        self._all_gather_for_recovery_group()
        logger.info("Synchronized Successfully,begin to clean fault")
        clean_status = self._clean_fault(ctx)
        recover_action = self._coordinate_recovery(clean_status)
        if not torch.equal(recover_action,FaultAction.RECOMPUTE):
            return recover_action
        #Begin to recover
        logger.info("Begin to recover error")
        recovery_status = handler.recover(ctx)
        recovery_action = self._coordinate_recovery(recovery_status)
        return recovery_action

    def _coordinate_recovery(self,local_status:torch.Tensor) -> torch.Tensor:
        """
        Rank 0 gathers recovery status and determines fault actions for each rank
        Recovery status is categorized into clean status and recovery status
        Failure at any recovery stage will cause re-inference to fail
        Therefore, re-inference is executed only when both restart recovery and fault recovery succeed
        """
        # determine fault action for single rank situation
        if not dist.is_initialized() or self.world_size == 1:
            return self._single_node_decision(local_status)
        # gather recovery status
        all_status = self._gather_statuses(local_status)
        if self.rank == 0:
            ft_actions = self._analyze_global_status(all_status)
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

    def _clean_fault_queue(self):
        while not self.fault_queue.empty():
            try:
                self.fault_queue.get_nowait()
            except queue.Empty:
                break

    def _clean_fault(self, ctx: RecoveryContext) -> torch.Tensor:
        """
        Clean the abnormal status,restart device and reinit process group
        """
        self._clean_fault_queue()
        try:
            torch_npu.npu.restart_device(torch.npu.current_device())
            torch.distributed.reinit_process_group(group=None, rebuild_link=False)
            self._restore_essential_state(ctx.back_up)
            reinit_status = RecoveryStatus.SUCCESS
        except Exception as inner_e:
            logger.error(f"Failed to clean fault for rank {self.rank},get exception :{inner_e}")
            ctx.exception = inner_e
            reinit_status = RecoveryStatus.FAILED
        return reinit_status

    def _all_gather_for_recovery_group(self):
        local_status = torch.tensor([self.rank])
        gather_list = [torch.zeros_like(local_status) for _ in range(self.world_size)]
        logger.debug(f"Rank {self.rank} waiting for all ranks to throw exceptions")
        try:
            dist.all_gather(gather_list, local_status,group=FaultTolerance._recovery_group)
        except Exception as inner_e:
            logger.error(f"All gather failed for _recovery_group,exception for recovery_group:{inner_e}")
            raise inner_e

    def _all_gather_for_sync_group(self):
        local_status = torch.tensor([self.rank],dtype=torch.int32,device="npu")
        gather_list = [torch.zeros_like(local_status) for _ in range(self.world_size)]
        logger.debug(f"Rank {self.rank} waiting for all ranks to finish execute_model")
        try:
            dist.all_gather(gather_list, local_status,group=FaultTolerance._sync_group)
            torch.npu.synchronize()
        except Exception as inner_e:
            logger.error(f"All gather failed for _sync_group,exception :{inner_e}")
            raise inner_e

    def _gather_statuses(self, local_status:torch.Tensor) -> List[torch.Tensor]:
        """
        Rank 0 gathers status from each rank
        """
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
                dist.gather(local_status, gather_list=None, dst=0,group=FaultTolerance._recovery_group)
                return []
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
        """
        Rank 0 distributed fault action to each rank
        """
        recv_ft_action = torch.tensor([0])
        dist.scatter(
            recv_ft_action,
            scatter_list=ft_actions,
            src=0,
            group=FaultTolerance._recovery_group
        )
        return recv_ft_action

    def _receive_ft_actions(self) -> torch.Tensor:
        """
        Rank 1 ...N receive fault action
        """
        recv_ft_action = torch.tensor([0])
        dist.scatter(
            recv_ft_action,
            scatter_list=None,
            src=0,
            group=FaultTolerance._recovery_group
        )
        return recv_ft_action

    def _create_essential_state_backup(self,*args,**kwargs) -> dict:
        backup = {}
        if not hasattr(self.model_runner,'requests') or not hasattr(self.model_runner,'input_batch'):
            return backup
        # Backup input for execute_model
        backup['args'] = args
        backup['kwargs'] = kwargs

        # Backup common state
        backup['generator_state'] = torch_npu.npu.get_rng_state()

        # Backup requests
        requests_backup = {}
        for req_id,state in self.model_runner.requests.items():
            if state is None:
                continue
            req_backup = {
                'output_token_ids' : state.output_token_ids.copy() if state.output_token_ids else [],
                'num_computed_tokens' : state.num_computed_tokens,
                'block_ids':tuple(block_id.copy() for block_id in state.block_ids) if state.block_ids else [],
            }
            requests_backup[req_id] = req_backup

        backup['requests_essential'] = requests_backup
        # Backup input_batch
        ib = self.model_runner.input_batch

        backup['_req_ids'] = ib._req_ids.copy()
        backup['req_output_token_ids'] = ib.req_output_token_ids.copy()
        backup['req_id_to_index'] = dict(ib.req_id_to_index)
        backup['num_blocks_per_row'] = [
            bt.num_blocks_per_row.copy() for bt in ib.block_table.block_tables
        ]
        if ib.batch_update_builder._removed is not None:
            backup['_removed'] = ib.batch_update_builder._removed.copy()
        else:
            backup['_removed'] = None
        if ib.batch_update_builder.added is not None:
            backup['added'] = ib.batch_update_builder.added.copy()
        else:
            backup['added'] = None
        backup['spec_token_ids'] = [spec_list.copy() for spec_list in ib.spec_token_ids]

        essential_arrays = ['token_ids_cpu','num_tokens','num_tokens_no_spec','num_computed_tokens_cpu','num_accepted_tokens_cpu']

        for attr_name in essential_arrays:
            if hasattr(ib,attr_name):
                attr_value = getattr(ib,attr_name)
                if attr_value is not None:
                    backup[attr_name] = attr_value.copy()

        if hasattr(ib,'prev_sampled_token_ids') and ib.prev_sampled_token_ids is not None:
            backup['prev_sampled_token_ids'] = ib.prev_sampled_token_ids.clone()

        # Backup eplb updator
        if hasattr(self.model_runner,'eplb_updator'):
            eplb = self.model_runner.eplb_updator
            backup['update_info_all'] = (
                eplb.update_info_all.copy()
                if hasattr(eplb,'update_info_all')
                else None
            )
            backup['reqs'] = (
                eplb.reqs.copy()
                if hasattr(eplb,'reqs')
                else None
            )
            backup['cur_iterations'] = (
                eplb.cur_iterations
                if hasattr(eplb,'cur_iterations')
                else None
            )
        return backup

    def _restore_essential_state(self,backup):
        if not backup:
            return
        # clear execute_model_state
        self.model_runner.execute_model_state = None
        # Rollback common state
        if 'generator_state' in backup:
            torch_npu.npu.set_rng_state(backup['generator_state'])

        # Rollback request state
        if 'requests_essential' in backup and hasattr(self.model_runner,'requests'):
            for req_id,req_backup in backup['requests_essential'].items():
                if req_id in self.model_runner.requests:
                    state = self.model_runner.requests[req_id]
                    state.output_token_ids = req_backup['output_token_ids']
                    state.num_computed_tokens = req_backup['num_computed_tokens']
                    state.block_ids = req_backup['block_ids']

        # Rollback inputbatch state
        if hasattr(self.model_runner,'input_batch'):
            ib = self.model_runner.input_batch

            if '_req_ids' in backup:
                ib._req_ids[:] = backup['_req_ids']
            if 'req_output_token_ids' in backup:
                ib.req_output_token_ids[:] = backup['req_output_token_ids']
            if 'req_id_to_index' in backup:
                ib.req_id_to_index.clear()
                ib.req_id_to_index.update(backup['req_id_to_index'])
            if 'num_blocks_per_row' in backup:
                for i, bt in enumerate[Any](ib.block_table.block_tables):
                    bt.num_blocks_per_row[:] = backup['num_blocks_per_row'][i]
            if '_removed' in backup:
                ib.batch_update_builder._removed = backup['_removed']
            if 'added' in backup:
                ib.batch_update_builder.added = backup['added']
            if 'spec_token_ids' in backup:
                ib.spec_token_ids = backup['spec_token_ids']

            essential_arrays = ['token_ids_cpu','num_tokens','num_tokens_no_spec','num_computed_tokens_cpu','num_accepted_tokens_cpu']
            for attr_name in essential_arrays:
                if attr_name in backup and hasattr(ib,attr_name):
                    target = getattr(ib,attr_name)
                    if target is not None and backup[attr_name] is not None:
                        target[:] = backup[attr_name]
            if 'prev_sampled_token_ids' in backup and hasattr(ib,'prev_sampled_token_ids'):
                ib.prev_sampled_token_ids = backup['prev_sampled_token_ids']

        # Rollback eplb state
        if hasattr(self.model_runner,'eplb_updator'):
            eplb = self.model_runner.eplb_updator
            if 'update_info_all' in backup:
                eplb.update_info_all = backup['update_info_all']
            if 'reqs' in backup:
                eplb.reqs = backup['reqs']
            if 'cur_iterations' in backup:
                eplb.cur_iterations = backup['cur_iterations']