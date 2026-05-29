# mypy: ignore-errors
import torch
import torch.distributed as dist
import vllm.v1.core.sched.scheduler as scheduler_module
import vllm.v1.engine.core as engine_core_module
from vllm.logger import logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.core import DPEngineCoreProc
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.structured_output import StructuredOutputManager

_ORIGINAL_SCHEDULER = Scheduler


def _balance_scheduling_enabled(vllm_config) -> bool:
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    return bool(additional_config.get("enable_balance_scheduling", False))


class BalanceScheduler(Scheduler):
    def __init__(
        self,
        vllm_config,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        hash_block_size: int | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        super().__init__(
            vllm_config,
            kv_cache_config,
            structured_output_manager,
            block_size,
            hash_block_size,
            mm_registry,
            include_finished_set,
            log_stats,
        )
        self._balance_enabled = _balance_scheduling_enabled(vllm_config)
        if self._balance_enabled:
            self.balance_queue = [
                torch.tensor([0], dtype=torch.int, device="cpu")
                for _ in range(self.vllm_config.parallel_config.data_parallel_size)
            ]

    def balance_gather(self, dp_group):
        if not self._balance_enabled:
            return
        running_tensor = torch.tensor([len(self.running)], dtype=torch.int, device="cpu")
        dist.all_gather(self.balance_queue, running_tensor, group=dp_group)

    def schedule(self) -> SchedulerOutput:
        if not self._balance_enabled:
            return super().schedule()
        balance_full = max(t.item() for t in self.balance_queue) == self.max_num_running_reqs
        if not balance_full:
            return super().schedule()

        # Reuse the paired vLLM scheduler implementation and only suppress
        # WAITING admission for this step when another DP rank is already full.
        # This avoids carrying a stale copy of Scheduler.schedule as vLLM evolves.
        waiting = self.waiting
        skipped_waiting = self.skipped_waiting
        temp_waiting = create_request_queue(self.policy)
        temp_skipped_waiting = create_request_queue(self.policy)
        self.waiting = temp_waiting
        self.skipped_waiting = temp_skipped_waiting
        try:
            scheduler_output = super().schedule()
        finally:
            self.waiting = waiting
            self.skipped_waiting = skipped_waiting
            if temp_skipped_waiting:
                self.skipped_waiting.prepend_requests(temp_skipped_waiting)
            if temp_waiting:
                self.waiting.prepend_requests(temp_waiting)
        return scheduler_output


class BalanceDPEngineCoreProc(DPEngineCoreProc):
    def run_busy_loop(self):
        """Core busy loop of the EngineCore for data parallel case."""

        # Loop until process is sent a SIGINT or SIGTERM
        while self._handle_shutdown():
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # Publish request counts before and after GPU step to ensure freshness.
            self._maybe_publish_request_counts()

            if self.eep_scaling_state is not None:
                _ = self.eep_scaling_state.progress()
                if self.eep_scaling_state.is_complete():
                    if self.eep_scaling_state.worker_type == "removing":
                        raise SystemExit
                    self.process_input_queue_block = True
                    self.eep_scaling_state = None

            # 2) Step the engine core.
            executed = self._process_engine_step()
            self._maybe_publish_request_counts()

            local_unfinished_reqs = self.scheduler.has_unfinished_requests()
            if not executed:
                if not local_unfinished_reqs and not self.engines_running:
                    # All engines are idle.
                    continue

                # We are in a running state and so must execute a dummy pass
                # if the model didn't execute any ready requests.
                self.execute_dummy_batch()

            # 3) All-reduce operation to determine global unfinished reqs.
            self.engines_running = self._has_global_unfinished_reqs(local_unfinished_reqs)
            balance_gather = getattr(self.scheduler, "balance_gather", None)
            if balance_gather is not None:
                balance_gather(self.dp_group)

            if not self.engines_running:
                if self.dp_rank == 0 or not self.has_coordinator:
                    # Notify client that we are pausing the loop.
                    logger.debug("Wave %d finished, pausing engine loop.", self.current_wave)
                    # In the coordinator case, dp rank 0 sends updates to the
                    # coordinator. Otherwise (offline spmd case), each rank
                    # sends the update to its colocated front-end process.
                    client_index = -1 if self.has_coordinator else 0
                    self.output_queue.put_nowait(
                        (
                            client_index,
                            EngineCoreOutputs(wave_complete=self.current_wave),
                        )
                    )
                # Increment wave count and reset step counter.
                self.current_wave += 1
                self.step_counter = 0

        raise SystemExit


engine_core_module.DPEngineCoreProc = BalanceDPEngineCoreProc
scheduler_module.Scheduler = BalanceScheduler
