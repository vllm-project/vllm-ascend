import copy
import functools
import gc
import hashlib
import json
import queue
import threading
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
import torch_npu
from vllm.logger import logger
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.token_reinference.common import FaultAction, FaultToleranceLevel, RecoveryStatus
from vllm_ascend.token_reinference.fault_aware import FaultAware
from vllm_ascend.token_reinference.recovery_context import RecoveryContext
from vllm_ascend.token_reinference.recovery_handler import ForceStopHandler, NetworkHandler, RecoveryHandlerManager


class FaultTolerance:
    _recovery_group = None
    _sync_group = None

    def __init__(
        self, vllm_config, model_runner, execute_model_func, max_retry_times=3, interval_m=5, max_backup_batches=2
    ):
        self.model_runner = model_runner
        self.execute_model_func = execute_model_func
        self.vllm_config = vllm_config
        self.max_retry_times = max_retry_times
        self.interval_m = interval_m
        self.max_backup_batches = max_backup_batches

        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        self.fault_queue: queue.Queue = queue.Queue()
        self.recovery_handler_manager = self._build_recovery_handler_manager()

        self._init_recovery_group()
        self._init_sync_group()

        self.state_backup: dict = {}
        self.async_batch_to_backup: dict = {}
        self.aware_event = threading.Event()
        self.stop_event = threading.Event()
        FaultAware(
            self.rank, self.world_size, self.fault_queue, aware_event=self.aware_event, stop_event=self.stop_event
        ).start()

    @classmethod
    def init_fault_tolerance(cls, worker, vllm_config, model_runner):
        additional_config = get_ascend_config()
        fault_tolerance_config = additional_config.fault_tolerance_config
        level = fault_tolerance_config.level
        max_reinference_times = fault_tolerance_config.max_reinference_times

        if level == FaultToleranceLevel.LEVEL_0.value:
            return None

        ft = cls(
            vllm_config=vllm_config,
            model_runner=model_runner,
            execute_model_func=worker.execute_model,
            max_retry_times=max_reinference_times,
        )
        return ft

    def _init_recovery_group(self):
        """
        Initialize the global communication group for status collection
        """
        if not dist.is_initialized() or self.world_size == 1:
            return

        FaultTolerance._recovery_group = dist.new_group(
            ranks=None,
            timeout=timedelta(minutes=self.interval_m),
            backend="gloo",
        )

        logger.info("Recovery group initialization successful for rank %s", self.rank)

    def _init_sync_group(self):
        """
        Initialize the global communication group for synchronization
        """
        if not dist.is_initialized() or self.world_size == 1:
            return
        FaultTolerance._sync_group = dist.new_group(
            ranks=None, timeout=timedelta(minutes=self.interval_m), backend="hccl"
        )

    def _build_recovery_handler_manager(self) -> RecoveryHandlerManager:
        """initialize recovery chain"""
        recovery_handler_manager = RecoveryHandlerManager()

        force_handler = ForceStopHandler()
        network_handler = NetworkHandler()

        recovery_handler_manager.register_handler(force_handler)
        recovery_handler_manager.register_handler(network_handler)

        return recovery_handler_manager

    @classmethod
    def execute_model_decorator(cls, dummy_run: bool) -> Callable:
        def decorator(func):
            @functools.wraps(func)
            def wrapper(worker_self, *args, **kwargs):
                ft = getattr(worker_self, "fault_tolerance", None)
                if ft is None:
                    return func(worker_self, *args, **kwargs)
                for attempt in range(ft.max_retry_times + 1):
                    if not dummy_run:
                        batch_key = ft._generate_scheduler_output_key(args[0])
                        if batch_key in ft.async_batch_to_backup:
                            logger.debug("Current batch might be token reinference batch")
                            ft.state_backup = ft.async_batch_to_backup[batch_key]
                            ft._restore_essential_state(ft.state_backup)
                            keys_to_remove = [k for k in ft.async_batch_to_backup if k != batch_key]
                            for key in keys_to_remove:
                                del ft.async_batch_to_backup[key]
                        else:
                            map_size = len(ft.async_batch_to_backup)
                            if map_size >= ft.max_backup_batches:
                                oldest_key = next(iter(ft.async_batch_to_backup))
                                del ft.async_batch_to_backup[oldest_key]
                            ft.state_backup = ft._create_essential_state_backup(*args, **kwargs)
                            ft.async_batch_to_backup[batch_key] = ft.state_backup
                    else:
                        ft.state_backup = ft._create_essential_state_backup(*args, **kwargs)
                    try:
                        output = func(worker_self, *args, **kwargs)
                        if dummy_run:
                            ft._all_gather_for_sync_group()
                        return output
                    except Exception as e:
                        if attempt >= ft.max_retry_times:
                            logger.warning(
                                "Max retries %s exceeded at rank %s，raising exception: %s",
                                ft.max_retry_times,
                                ft.rank,
                                e,
                            )
                            raise e
                        # Encapsulate the context information required for fault recovery.
                        recovery_context = RecoveryContext(
                            exception=e, fault_queue=ft.fault_queue, back_up=ft.state_backup, is_dummy_run=dummy_run
                        )
                        ft_action = ft._handle_exception(recovery_context)
                        if torch.equal(ft_action, FaultAction.RECOMPUTE):
                            ft.aware_event.set()
                            logger.info("Begin token re-inference at rank %s", ft.rank)
                            continue
                        elif torch.equal(ft_action, FaultAction.RAISE_EXCEPTION):
                            logger.info("Raise exception at rank %s", ft.rank)
                            raise e
                        elif torch.equal(ft_action, FaultAction.RETURN):
                            logger.info("Abort current batch at rank %s", ft.rank)
                            return EMPTY_MODEL_RUNNER_OUTPUT
                        else:
                            logger.error("Unexpected FaultAction %s, aborting", ft_action)
                            raise RuntimeError("Unknown fault action")
                return EMPTY_MODEL_RUNNER_OUTPUT

            return wrapper

        return decorator

    @classmethod
    def sample_token_decorator(cls) -> Callable:
        def decorator(func):
            @functools.wraps(func)
            def wrapper(worker_self, *args, **kwargs):
                ft = getattr(worker_self, "fault_tolerance", None)
                if ft is None:
                    return func(worker_self, *args, **kwargs)
                for attempt in range(ft.max_retry_times + 1):
                    try:
                        output = func(worker_self, *args, **kwargs)
                        if output is not None:
                            ft._all_gather_for_sync_group()
                        return output
                    except Exception as e:
                        if attempt >= ft.max_retry_times:
                            logger.warning(
                                "Max retries %s exceeded at rank %s，raising exception: %s",
                                ft.max_retry_times,
                                ft.rank,
                                e,
                            )
                            raise e
                        # Encapsulate the context information required for fault recovery.
                        recovery_context = RecoveryContext(
                            exception=e, fault_queue=ft.fault_queue, back_up=ft.state_backup, is_dummy_run=False
                        )
                        ft_action = ft._handle_exception(recovery_context)
                        if torch.equal(ft_action, FaultAction.RECOMPUTE):
                            ft.aware_event.set()
                            logger.info("Begin re-execute model at rank %s", ft.rank)
                            # re-execute model first
                            ft.execute_model_func(*ft.state_backup["args"], **ft.state_backup["kwargs"])
                            logger.info("Begin token re-inference at rank %s", ft.rank)
                            continue
                        elif torch.equal(ft_action, FaultAction.RAISE_EXCEPTION):
                            logger.info("Raise exception at rank %s", ft.rank)
                            raise e
                        elif torch.equal(ft_action, FaultAction.RETURN):
                            logger.info("Abort current batch at rank %s", ft.rank)
                            return EMPTY_MODEL_RUNNER_OUTPUT
                        else:
                            logger.error("Unexpected FaultAction %s, aborting", ft_action)
                            raise RuntimeError("Unknown fault action")

                return EMPTY_MODEL_RUNNER_OUTPUT

            return wrapper

        return decorator

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
        if not torch.equal(recover_action, FaultAction.RECOMPUTE):
            return recover_action
        # Begin to recover
        logger.info("Begin to recover error")
        recovery_status = handler.recover(ctx)
        recovery_action = self._coordinate_recovery(recovery_status)
        return recovery_action

    def _coordinate_recovery(self, local_status: torch.Tensor) -> torch.Tensor:
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
            if ctx.is_dummy_run:
                self._restore_essential_state(ctx.back_up)
            reinit_status = RecoveryStatus.SUCCESS
        except Exception as inner_e:
            logger.error("Failed to clean fault for rank %s,get exception :%s", self.rank, inner_e)
            ctx.exception = inner_e
            reinit_status = RecoveryStatus.FAILED
        return reinit_status

    def _all_gather_for_recovery_group(self):
        local_status = torch.tensor(self.rank)
        gather_list = [torch.zeros_like(local_status) for _ in range(self.world_size)]
        logger.debug("Rank %s waiting for all ranks to throw exceptions", self.rank)
        try:
            dist.all_gather(gather_list, local_status, group=FaultTolerance._recovery_group)
        except Exception as inner_e:
            logger.error("All gather failed for _recovery_group,exception for recovery_group:%s", inner_e)
            raise inner_e

    def _all_gather_for_sync_group(self):
        local_status = torch.tensor(self.rank, dtype=torch.int32, device="npu")
        gather_list = [torch.zeros_like(local_status) for _ in range(self.world_size)]
        logger.debug("Rank %s waiting for all ranks to finish execute_model", self.rank)
        try:
            dist.all_gather(gather_list, local_status, group=FaultTolerance._sync_group)
            torch.npu.current_stream().synchronize()
        except Exception as inner_e:
            logger.error("All gather failed for _sync_group,exception :%s", inner_e)
            raise inner_e

    def _gather_statuses(self, local_status: torch.Tensor) -> list[torch.Tensor]:
        """
        Rank 0 gathers status from each rank
        """
        try:
            if self.rank == 0:
                gather_list = [torch.zeros_like(local_status) for _ in range(self.world_size)]
                dist.gather(local_status, gather_list=gather_list, dst=0, group=FaultTolerance._recovery_group)
                return gather_list
            else:
                dist.gather(local_status, gather_list=None, dst=0, group=FaultTolerance._recovery_group)
                return []
        except Exception as inner_e:
            logger.error("Gather status failed,get exception:%s", inner_e)
            if self.rank == 0:
                return [RecoveryStatus.FAILED for _ in range(self.world_size)]
            return []

    def _analyze_global_status(self, all_recovery_statuses: list[torch.Tensor]) -> list[torch.Tensor]:
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
                logger.warning("Unknown status tensor from rank %s: %s", rank, recovery_status)
                failure_ranks.append(rank)

        logger.info("Global recovery: %s success, %s failure", len(success_ranks), len(failure_ranks))

        decisions = []
        if not failure_ranks:
            logger.info("All ranks recovered, Determine RECOMPUTE for all rank")
            decisions = [FaultAction.RECOMPUTE] * self.world_size
        elif not success_ranks:
            logger.warning("All ranks failed, Determine RAISE_EXCEPTION for all rank")
            decisions = [FaultAction.RAISE_EXCEPTION] * self.world_size
        else:
            logger.warning("Partial recovery - success ranks: %s", success_ranks)
            for rank in range(self.world_size):
                if rank in success_ranks:
                    decisions.append(FaultAction.RETURN)
                else:
                    decisions.append(FaultAction.RAISE_EXCEPTION)

        return decisions

    def _scatter_ft_actions(self, ft_actions: list[torch.Tensor]) -> torch.Tensor:
        """
        Rank 0 distributed fault action to each rank
        """
        recv_ft_action = torch.tensor(0)
        dist.scatter(recv_ft_action, scatter_list=ft_actions, src=0, group=FaultTolerance._recovery_group)
        return recv_ft_action

    def _receive_ft_actions(self) -> torch.Tensor:
        """
        Rank 1 ...N receive fault action
        """
        recv_ft_action = torch.tensor(0)
        dist.scatter(recv_ft_action, scatter_list=None, src=0, group=FaultTolerance._recovery_group)
        return recv_ft_action

    def _generate_scheduler_output_key(self, scheduler_output):
        new_req_ids = [req.req_id for req in scheduler_output.scheduled_new_reqs]
        cached_req_ids = [req_id for req_id in scheduler_output.scheduled_cached_reqs.req_ids]
        cached_num_tokens = [num_tokens for num_tokens in scheduler_output.scheduled_cached_reqs.num_computed_tokens]
        finished_req_ids = sorted(scheduler_output.finished_req_ids)
        key_data = {
            "new_req_ids": new_req_ids,
            "cached_req_ids": cached_req_ids,
            "cached_num_tokens": cached_num_tokens,
            "finished_req_ids": finished_req_ids,
        }

        key_json = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(key_json.encode("utf-8")).hexdigest()

    def _create_essential_state_backup(self, *args, **kwargs) -> dict:
        backup: dict[str, Any] = {}
        if not hasattr(self.model_runner, "requests") or not hasattr(self.model_runner, "input_batch"):
            return backup
        # Backup input for execute_model
        backup["args"] = args
        backup["kwargs"] = kwargs

        # Backup common state
        backup["generator_state"] = torch_npu.npu.get_rng_state()

        # Backup requests
        requests_backup = {}
        for req_id, state in self.model_runner.requests.items():
            if state is None:
                continue
            req_backup = {
                "output_token_ids": state.output_token_ids.copy(),
                "num_computed_tokens": state.num_computed_tokens,
                "block_ids": tuple(block_id.copy() for block_id in state.block_ids),
            }
            requests_backup[req_id] = req_backup

        backup["requests_essential"] = requests_backup
        # Backup input_batch
        ib = self.model_runner.input_batch

        backup["_req_ids"] = ib._req_ids.copy()
        backup["req_output_token_ids"] = [(sub.copy() if sub is not None else None) for sub in ib.req_output_token_ids]
        backup["req_id_to_index"] = dict(ib.req_id_to_index)
        backup["spec_token_ids"] = [spec_list.copy() for spec_list in ib.spec_token_ids]
        backup["num_blocks_per_row"] = [bt.num_blocks_per_row.copy() for bt in ib.block_table.block_tables]
        builder = ib.batch_update_builder
        backup["_removed"] = copy.deepcopy(builder._removed)
        backup["added"] = copy.deepcopy(builder.added)

        essential_arrays = [
            "token_ids_cpu",
            "num_tokens",
            "num_tokens_no_spec",
            "num_computed_tokens_cpu",
            "num_accepted_tokens_cpu",
        ]
        for attr_name in essential_arrays:
            if hasattr(ib, attr_name):
                attr_value = getattr(ib, attr_name)
                if attr_value is not None:
                    backup[attr_name] = attr_value.copy()

        if hasattr(ib, "prev_sampled_token_ids") and ib.prev_sampled_token_ids is not None:
            backup["prev_sampled_token_ids"] = ib.prev_sampled_token_ids.clone()

        # Backup eplb updator
        if hasattr(self.model_runner, "eplb_updator"):
            eplb = self.model_runner.eplb_updator
            backup["update_info_all"] = (
                copy.deepcopy(eplb.update_info_all) if hasattr(eplb, "update_info_all") else None
            )
            backup["reqs"] = copy.deepcopy(eplb.reqs) if hasattr(eplb, "reqs") else None
            backup["cur_iterations"] = copy.deepcopy(eplb.cur_iterations) if hasattr(eplb, "cur_iterations") else None

        return backup

    def _restore_essential_state(self, backup):
        if not backup:
            return
        # clear execute_model_state
        self.model_runner.execute_model_state = None
        # Rollback common state
        if "generator_state" in backup:
            torch_npu.npu.set_rng_state(backup["generator_state"])

        # Rollback request state
        if "requests_essential" in backup and hasattr(self.model_runner, "requests"):
            for req_id, req_backup in backup["requests_essential"].items():
                if req_id in self.model_runner.requests:
                    state = self.model_runner.requests[req_id]
                    state.output_token_ids = req_backup["output_token_ids"]
                    state.num_computed_tokens = req_backup["num_computed_tokens"]
                    state.block_ids = req_backup["block_ids"]

        # Rollback input_batch state
        if hasattr(self.model_runner, "input_batch"):
            ib = self.model_runner.input_batch

            if "_req_ids" in backup:
                ib._req_ids[:] = backup["_req_ids"]
            if "req_output_token_ids" in backup:
                ib.req_output_token_ids[:] = backup["req_output_token_ids"]
            if "req_id_to_index" in backup:
                ib.req_id_to_index.clear()
                ib.req_id_to_index.update(backup["req_id_to_index"])
            if "num_blocks_per_row" in backup:
                for i, bt in enumerate(ib.block_table.block_tables):
                    bt.num_blocks_per_row[:] = backup["num_blocks_per_row"][i]
            if "spec_token_ids" in backup:
                ib.spec_token_ids[:] = backup["spec_token_ids"]

            if "_removed" in backup:
                ib.batch_update_builder._removed[:] = backup["_removed"]
            if "added" in backup:
                ib.batch_update_builder.added[:] = backup["added"]

            essential_arrays = [
                "token_ids_cpu",
                "num_tokens",
                "num_tokens_no_spec",
                "num_computed_tokens_cpu",
                "num_accepted_tokens_cpu",
            ]
            for attr_name in essential_arrays:
                if attr_name in backup and hasattr(ib, attr_name):
                    target = getattr(ib, attr_name)
                    target[:] = backup[attr_name]
            if "prev_sampled_token_ids" in backup and hasattr(ib, "prev_sampled_token_ids"):
                ib.prev_sampled_token_ids = backup["prev_sampled_token_ids"]

        # Rollback eplb state
        if hasattr(self.model_runner, "eplb_updator"):
            eplb = self.model_runner.eplb_updator
            if "update_info_all" in backup:
                eplb.update_info_all = backup["update_info_all"]
            if "reqs" in backup:
                eplb.reqs = backup["reqs"]
            if "cur_iterations" in backup:
                eplb.cur_iterations = backup["cur_iterations"]

        # clean up cache after rollback
        gc.collect()
        torch_npu.npu.empty_cache()
