# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import torch
import torch_npu
from vllm.distributed.eplb.eplb_state import EplbModelState, _commit_eplb_maps
from vllm.distributed.parallel_state import (
    get_dp_group,
    get_ep_group,
    get_tp_group,
)
from vllm.distributed.utils import set_gloo_backend_timeout
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager
from vllm.model_executor.models.interfaces import get_mixture_of_experts_model
from vllm.v1.fault_tolerance.utils import FaultToleranceRequest
from vllm.v1.worker.sentinel.eplb_redistribute import (
    mark_dead_expert_slots_inplace,
    redistribute_expert_placement,
)
from vllm.v1.worker.sentinel.gpu_worker_sentinel import (
    WorkerSentinel as GPUWorkerSentinel,
)

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.eplb.eplb_state import refresh_model_routing_tables
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.worker.sentinel.eplb_redistribute import (
    build_orig_to_dense_rank_table,
    densify_routing_table_physical_ids,
    reload_experts_from_disk,
)
from vllm_ascend.worker.sentinel.graph_recovery import recover_acl_graphs
from vllm_ascend.worker.sentinel.uce_fault import (
    HbmUceNonKvFault,
    HbmUceUnrecoverableFault,
    classify_uce_after_stop,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    from vllm_ascend.worker.sentinel.kv_cache_region_registry import KvCacheRegionRegistry


_sentinel: "WorkerSentinel | None" = None
_RETRY_CLASSIFICATION_POLL_SEC = 0.05
_RETRY_CLASSIFICATION_RECOVERABLE = b"recoverable"
_RETRY_CLASSIFICATION_NON_KV = b"non_kv"


@dataclass(frozen=True)
class _RetryContext:
    round_id: str
    dp_size: int
    dp_rank: int
    local_size: int
    local_rank: int
    timeout: float
    prefix: str
    store: Any


def fault_barrier_wrapper(func: Callable):
    """Barrier between device faults and the async step loop.

    On the first fault, quarantine the worker and propagate the original
    exception to EngineCore. Do not reset the device here: recovery starts
    only when EngineCore dispatches the FT command selected by Motor.
    While quarantined, wrapped methods raise without touching the device.
    """

    def wrapped(self, *args, **kwargs):
        sentinel = getattr(self, "worker_sentinel", _sentinel)
        if sentinel is None:
            return func(self, *args, **kwargs)
        if sentinel.worker_faulted:
            # Pending async outputs can own tensors allocated on other streams.
            # Dropping them on a faulted device may record an allocator event.
            sentinel._fault_keepalive.append((self, args, kwargs))
            raise RuntimeError("worker quarantined after device fault") from None
        # Retire completed outputs on the worker thread only after a later
        # forward succeeds. Never let the output thread destroy pinned buffers
        # while the worker is discovering a device fault. Outputs completed
        # during this forward stay owned until the next successful forward.
        completed = getattr(sentinel, "_completed_async_outputs", None)
        retire_count = (
            len(completed) if completed is not None and getattr(func, "__name__", None) == "execute_model" else 0
        )
        try:
            result = func(self, *args, **kwargs)
            if retire_count and not sentinel.worker_faulted:
                del completed[:retire_count]
            return result
        except SystemExit:
            raise
        except Exception as exc:
            # Keep traceback locals alive until device restart. Converting this
            # exception to a string in WorkerProc otherwise destroys tensors
            # while the allocator cannot safely record stream events.
            sentinel._fault_keepalive.append(exc)
            if sentinel.worker_faulted:
                raise RuntimeError("worker quarantined after device fault") from None
            sentinel.worker_faulted = True
            logger.warning(
                "[FT] Quarantining %s after fault; awaiting FT command: %s",
                getattr(self, "rank", "worker"),
                exc,
                exc_info=True,
            )
            raise

    return wrapped


def retain_completed_async_output(output: object) -> None:
    """Transfer completed output ownership to the worker/recovery thread.

    Native get_output may succeed just as another thread observes UCE. Keep
    its pinned NumPy backing storage alive across that race without polling
    device events or retaining every output for the lifetime of the worker.
    """
    if _sentinel is not None:
        _sentinel._completed_async_outputs.append(output)


class WorkerSentinel(GPUWorkerSentinel):
    """Per-worker sentinel for fault tolerance on Ascend NPU.

    Handles commands dispatched from EngineCoreSentinel via collective_rpc,
    including device restart and DP group re-initialization on retry, and
    MC2 elastic_info masking + expert redistribution on scale_down.
    """

    def __init__(self, worker: "Worker", device: torch.device):
        global _sentinel
        self.device = device
        self.worker = worker
        # Set once a device-touching method faults, to keep this worker off the
        # device until FT recovery rebuilds the groups.
        self.worker_faulted = False
        self._unrecoverable_uce = None
        self._non_kv_repair_attempted = False
        self.retry_cache_plan = None
        self._fault_keepalive: list[object] = []
        self._completed_async_outputs: list[object] = []
        _sentinel = self

    def query_mask(self, ft_request: FaultToleranceRequest) -> dict:
        """Report the dead-rank mask (upstream convention: 0=live, 1=dead)."""
        return {"mask": get_ep_all2all_manager().query_active_mask().tolist()}

    def _stop_device(self) -> None:
        NPUPlatform.set_device(self.device)
        torch_npu.npu.stop_device(self.device.index)

    def _classify_uce(self) -> None:
        if self._unrecoverable_uce is not None:
            raise self._unrecoverable_uce
        registry = getattr(self.worker.model_runner, "kv_cache_region_registry", None)
        if registry is not None:
            registry.uce_ranges.clear()
        try:
            fault = classify_uce_after_stop(registry, self.device.index)
        except HbmUceUnrecoverableFault as fault:
            self._unrecoverable_uce = fault
            raise
        if isinstance(fault, HbmUceUnrecoverableFault):
            self._unrecoverable_uce = fault
            raise fault

    def _restart_device(self) -> None:
        torch_npu.npu.restart_device(self.device.index)
        torch_npu.distributed.reinit_process_group(None, False)
        torch.npu.synchronize()

    def reset_device(self) -> None:
        # Keep the supplied scale-down path unchanged.
        self._stop_device()
        self._restart_device()

    @staticmethod
    def _wait_for_retry_decision(store, decision_key, ready_keys, deadline, timeout_reason) -> bytes:
        """Use one immutable decision even when readiness races a timeout."""
        while True:
            if store.check([decision_key]):
                return store.get(decision_key)
            if store.check(ready_keys):
                return store.compare_set(decision_key, "", "approved")
            if time.monotonic() >= deadline:
                return store.compare_set(decision_key, "", f"rejected: {timeout_reason}")
            time.sleep(_RETRY_CLASSIFICATION_POLL_SEC)

    def _build_retry_context(self, ft_request: FaultToleranceRequest) -> _RetryContext:
        params = ft_request.params
        round_id = params["recovery_round"]
        if not round_id or round_id != ft_request.request_id:
            raise ValueError("HBM retry requires the current recovery request_id")
        parallel = self.worker.parallel_config
        dp_size = int(params["dp_group_size"])
        dp_rank = int(params["dp_group_rank"])
        local_size = parallel.world_size
        local_rank = self.worker.rank % local_size
        if dp_size < 1 or not 0 <= dp_rank < dp_size:
            raise ValueError("Invalid HBM retry DP membership")
        timeout = float(parallel.fault_tolerance_config.engine_recovery_timeout_sec)
        if timeout <= 0:
            raise ValueError("HBM retry classification timeout must be positive")
        store = torch.distributed.TCPStore(
            params["dp_master_ip"],
            params["recovery_store_port"],
            is_master=False,
            wait_for_workers=False,
            timeout=timedelta(seconds=timeout),
        )
        prefix = f"ascend_hbm_retry_classification/{round_id}"
        return _RetryContext(
            round_id=round_id,
            dp_size=dp_size,
            dp_rank=dp_rank,
            local_size=local_size,
            local_rank=local_rank,
            timeout=timeout,
            prefix=prefix,
            store=store,
        )

    def _classify_and_report_local_blocks(self, context: _RetryContext) -> "KvCacheRegionRegistry | None":
        decision_key = f"{context.prefix}/decision"
        local_blocks = []
        classification = _RETRY_CLASSIFICATION_RECOVERABLE
        registry = getattr(self.worker.model_runner, "kv_cache_region_registry", None)
        try:
            if getattr(self, "_non_kv_repair_attempted", False):
                raise RuntimeError("Non-KV UCE repair already attempted; engine restart required")
            self._stop_device()
            self._classify_uce()
            if registry is None or not registry.regions:
                raise RuntimeError("HBM retry requires registered KV cache regions")
            local_blocks = sorted(registry.partial_block_ids(registry.num_blocks))
        except HbmUceNonKvFault as exc:
            classification = _RETRY_CLASSIFICATION_NON_KV
            logger.error("[FT] Non-KV UCE requires device repair and engine restart: %s", exc)
        except Exception as exc:
            reason = f"DP{context.dp_rank}/worker{context.local_rank}: {type(exc).__name__}: {exc}"
            context.store.compare_set(decision_key, "", f"rejected: {reason}")
            logger.error("[FT] HBM retry classification rejected: %s", reason)
        context.store.set(f"{context.prefix}/blocks/{context.dp_rank}/{context.local_rank}", json.dumps(local_blocks))
        context.store.set(f"{context.prefix}/ready/{context.dp_rank}/{context.local_rank}", classification)
        logger.info(
            "[FT] Retry classify ready: request_id=%s DP%d/worker%d",
            context.round_id,
            context.dp_rank,
            context.local_rank,
        )
        return registry

    def _wait_for_classification(self, context: _RetryContext, deadline: float) -> None:
        decision_key = f"{context.prefix}/decision"
        ready_keys = [
            f"{context.prefix}/ready/{dp}/{rank}" for dp in range(context.dp_size) for rank in range(context.local_size)
        ]
        decision = self._wait_for_retry_decision(
            context.store,
            decision_key,
            ready_keys,
            deadline,
            f"HBM retry classification timed out after {context.timeout}s: request_id={context.round_id}",
        )
        if decision != b"approved":
            raise RuntimeError(f"HBM retry classification failed: {decision.decode()}")
        if any(context.store.get(key) == _RETRY_CLASSIFICATION_NON_KV for key in ready_keys):
            # Every worker has stopped and classified before any rank enters
            # the collective device repair. Never repeat it on another retry.
            self._non_kv_repair_attempted = True
            logger.warning(
                "[FT] Non-KV UCE: repairing device before failing retry; request_id=%s rank=%s",
                context.round_id,
                self.worker.rank,
            )
            self._restart_device()
            raise HbmUceUnrecoverableFault(
                region="non_kv_cache",
                address=None,
                reason="device repair completed; engine restart required",
            )

    def _agree_dp_aliases(
        self, context: _RetryContext, registry: "KvCacheRegionRegistry", blocks: set[int], deadline: float
    ) -> set[int]:
        iteration = 0
        while True:
            expanded = registry.expand_aliases(blocks)
            phase = f"{context.prefix}/aliases/{context.dp_rank}/{iteration}"
            context.store.set(f"{phase}/{context.local_rank}", json.dumps(sorted(expanded)))
            keys = [f"{phase}/{rank}" for rank in range(context.local_size)]
            while not context.store.check(keys):
                if time.monotonic() >= deadline:
                    raise RuntimeError("HBM retry KV alias consensus timed out")
                time.sleep(_RETRY_CLASSIFICATION_POLL_SEC)
            agreed = {block for key in keys for block in json.loads(context.store.get(key))}
            if agreed == blocks:
                return blocks
            blocks = agreed
            iteration += 1
            if time.monotonic() >= deadline:
                raise RuntimeError("HBM retry KV alias consensus timed out")

    def _build_and_agree_cache_plan(
        self, context: _RetryContext, registry: "KvCacheRegionRegistry", deadline: float
    ) -> None:
        cache_decision_key = f"{context.prefix}/cache_decision"
        reports = [
            json.loads(context.store.get(f"{context.prefix}/blocks/{context.dp_rank}/{rank}"))
            for rank in range(context.local_size)
        ]
        blocks = {block for report in reports for block in report}
        try:
            blocks = self._agree_dp_aliases(context, registry, blocks, deadline)
            hbm_detected = any(
                json.loads(context.store.get(f"{context.prefix}/blocks/{dp}/{rank}"))
                for dp in range(context.dp_size)
                for rank in range(context.local_size)
            )
            self.retry_cache_plan = {
                "request_id": context.round_id,
                "blocks": sorted(blocks),
                "hbm_detected": hbm_detected,
            }
        except Exception as exc:
            context.store.compare_set(cache_decision_key, "", f"rejected: {exc}")
        context.store.set(f"{context.prefix}/cache_ready/{context.dp_rank}/{context.local_rank}", b"1")
        cache_ready = [
            f"{context.prefix}/cache_ready/{dp}/{rank}"
            for dp in range(context.dp_size)
            for rank in range(context.local_size)
        ]
        cache_decision = self._wait_for_retry_decision(
            context.store,
            cache_decision_key,
            cache_ready,
            deadline,
            "KV plan consensus timed out",
        )
        if cache_decision != b"approved":
            self.retry_cache_plan = None
            raise RuntimeError(f"HBM retry KV plan failed: {cache_decision.decode()}")

    def _prepare_retry(self, ft_request: FaultToleranceRequest) -> None:
        """Agree on UCE classification and DP-local KV blocks before NPU reinit."""
        context = self._build_retry_context(ft_request)
        logger.info(
            "[FT] Retry classify start: request_id=%s DP%d/worker%d",
            context.round_id,
            context.dp_rank,
            context.local_rank,
        )
        self.worker_faulted = True
        self.retry_cache_plan = None
        registry = self._classify_and_report_local_blocks(context)
        deadline = time.monotonic() + context.timeout
        self._wait_for_classification(context, deadline)
        assert registry is not None
        # Block IDs are DP-local. Never union unrelated DP allocation spaces.
        self._build_and_agree_cache_plan(context, registry, deadline)
        logger.info("[FT] HBM retry classification approved for all workers: request_id=%s", context.round_id)

    def retry(self, ft_request: FaultToleranceRequest):
        logger.info(
            "[FT] Worker recovery received: request_id=%s instruction=%s rank=%s",
            ft_request.request_id,
            ft_request.instruction,
            self.worker.rank,
        )
        if ft_request.instruction == "retry":
            self._prepare_retry(ft_request)
            self._restart_device()
            logger.info("[FT] Retry device restarted: request_id=%s rank=%s", ft_request.request_id, self.worker.rank)
        else:
            # scale_down calls this override from the upstream worker flow.
            self.reset_device()
        super().retry(ft_request)
        logger.info("[FT] Recovery groups rebuilt: request_id=%s rank=%s", ft_request.request_id, self.worker.rank)
        if ft_request.instruction == "retry":
            registry = getattr(self.worker.model_runner, "kv_cache_region_registry", None)
            if registry is None or not registry.regions:
                raise RuntimeError("HBM retry requires registered KV cache regions")
            if self.retry_cache_plan is None:
                raise RuntimeError("Missing HBM retry cache plan")
            registry.invalidate_blocks(self.retry_cache_plan["blocks"])
            torch.npu.synchronize()
            # The flag is agreed across DP/TP, unlike the local damaged blocks.
            # Every participant must recapture in the same collective order.
            if self.retry_cache_plan.get("hbm_detected"):
                recover_acl_graphs(self.worker, ft_request.request_id)
        self._release_fault_state()
        self.worker_faulted = False
        logger.info("[FT] Worker recovery complete: request_id=%s rank=%s", ft_request.request_id, self.worker.rank)

    def _release_fault_state(self) -> None:
        """Only called after device recovery and worker state cleanup succeed."""
        for obj in self._fault_keepalive:
            if isinstance(obj, BaseException):
                traceback.clear_frames(obj.__traceback__)
        self._fault_keepalive.clear()
        if completed := getattr(self, "_completed_async_outputs", None):
            completed.clear()

    def _clean_worker_state(self):
        metadata = getattr(self.worker.model_runner.model_state, "device_metadata", None)
        if metadata is not None:
            metadata.reset_after_recovery()
        super()._clean_worker_state()

    def init_num_local_experts(self) -> None:
        """Record the per-rank physical expert slot count after model load."""
        if self.worker.model_runner.eplb_state is None:
            return
        eplb_model_state = self._eplb_model_state()
        num_local_experts = eplb_model_state.physical_to_logical_map.shape[1] // get_ep_group().world_size
        get_ep_all2all_manager().set_num_local_physical_experts(num_local_experts)

    def scale_down(self, ft_request: FaultToleranceRequest):
        """Scale down over the surviving DP ranks, reusing the upstream flow.

        ``super().scale_down`` runs the deterministic dead-rank masking and
        expert redistribution, dispatching ``retry`` / ``_redistribute_experts``
        to the Ascend overrides below. Ascend adds its platform preconditions
        and a dummy-batch runnability check on top.
        """
        self._validate_scale_down_preconditions()
        super().scale_down(ft_request)

        # Verify the redistributed model is runnable before reporting healthy.
        parallel_config = self.worker.parallel_config
        timeout = timedelta(seconds=self.worker.parallel_config.fault_tolerance_config.engine_recovery_timeout_sec)
        if parallel_config.data_parallel_size > 1:
            set_gloo_backend_timeout(get_dp_group().cpu_group, timeout)
        if parallel_config.tensor_parallel_size > 1:
            set_gloo_backend_timeout(get_tp_group().cpu_group, timeout)
        self.worker.execute_dummy_batch()
        self.activate_cpu_group_timeouts(ft_request)
        torch.npu.synchronize()

    def _validate_scale_down_preconditions(self) -> None:
        if not self.worker.use_v2_model_runner:
            raise ValueError("[FT] scale_down on Ascend NPU requires the v2 model runner.")
        model_runner = self.worker.model_runner
        eplb_config = self.worker.parallel_config.eplb_config
        if model_runner.eplb_state is None or eplb_config.num_redundant_experts <= 0:
            raise ValueError(
                "[FT] scale_down requires EPLB with num_redundant_experts > 0 to re-host the dead rank's experts."
            )
        self._resolve_drafter()
        ascend_config = get_ascend_config()
        if ascend_config.enable_fused_mc2:
            raise ValueError(
                "[FT] scale_down is not supported with enable_fused_mc2: the "
                "fused dispatch_ffn_combine operators take no elastic_info."
            )
        if ascend_config.enable_mc2_hierarchy_comm:
            raise ValueError(
                "[FT] scale_down (elastic_info) is mutually exclusive with mc2 hierarchy comm (comm_alg='hierarchy')."
            )
        if not hasattr(torch_npu, "npu_moe_distribute_dispatch_v2"):
            raise ValueError(
                "[FT] scale_down requires npu_moe_distribute_dispatch_v2 "
                "(aclnn V3+); please upgrade the CANN/torch_npu version."
            )

    def _redistribute_experts(self, dead_ep_ranks: set[int]) -> None:
        """Redistribute experts onto the surviving slots after scale-down.

        Overrides the upstream flow so reassigned weights reload through the
        Ascend reloader: the upstream reloader writes via ``model.load_weights``,
        which cannot produce Ascend's runtime expert layout (transpose / NZ /
        per-slot lists / quant scales). The shared redistribution steps (mark
        dead slots, steal spare slots, rebuild the logical maps) are kept; on
        top of that, refreshes the Ascend kernel-facing routing tables into
        the densified id space and shrinks the MC2 physical-expert width.

        Each EPLB-registered model (main model and drafter) is redistributed
        against its own placement and reloaded from its own checkpoint; results
        are never shared across models.
        """
        model_runner = self.worker.model_runner

        models: list[tuple[torch.nn.Module, EplbModelState, str]] = [
            (model_runner.model, self._eplb_model_state(), self.worker.vllm_config.model_config.model)
        ]
        drafter = self._resolve_drafter()
        if drafter is not None:
            draft_layer = drafter[1].model.moe_layers[0]
            draft_ep_size = getattr(draft_layer, "routed_experts", draft_layer).moe_config.ep_size
            target_ep_size = get_ep_group().world_size
            if draft_ep_size != target_ep_size:
                raise ValueError(
                    f"[FT] scale_down requires the drafter's EP size ({draft_ep_size}) to match the "
                    f"target's ({target_ep_size}); otherwise expert weights are reloaded into the "
                    f"wrong physical slots."
                )
            models.append(drafter)

        for model, model_state, model_path in models:
            self._redistribute_model(model, model_state, model_path, dead_ep_ranks)

    def _redistribute_model(
        self,
        model: torch.nn.Module,
        model_state: EplbModelState,
        model_path: str,
        dead_ep_ranks: set[int],
    ) -> None:
        """Redistribute one model and reload through its checkpoint namespace.

        ``model`` owns the checkpoint name mapper and may wrap ``model_state.model``.
        """
        p2l = model_state.physical_to_logical_map
        num_logical = model_state.logical_replica_count.shape[1]
        num_local_experts = p2l.shape[1] // get_ep_group().world_size

        mark_dead_expert_slots_inplace(p2l, dead_ep_ranks, num_local_experts)
        reassignments = redistribute_expert_placement(p2l, num_logical, num_local_experts)
        logger.info(
            "[FT] %s expert redistribution: moe_layers=%d, num_logical=%d, reassignments=%d",
            type(model).__name__,
            p2l.shape[0],
            num_logical,
            len(reassignments),
        )
        _commit_eplb_maps(model_state, p2l.cpu())
        # Keep the model-side expert map: slot ownership is unchanged, and the
        # map is physical-id indexed, so p2l (logical ids) must not be written into it.
        if reassignments:
            reload_experts_from_disk(model, model_path, reassignments)
        refresh_model_routing_tables(model_state)
        self._densify_routing_tables(model_state)

    def _resolve_drafter(self) -> tuple[torch.nn.Module, EplbModelState, str] | None:
        """Resolve a drafter's EPLB state and checkpoint, rejecting unregistered MoE."""
        model_runner = self.worker.model_runner
        drafter = getattr(getattr(model_runner, "speculator", None), "model", None)
        draft_moe = get_mixture_of_experts_model(drafter)
        if draft_moe is None and not getattr(drafter, "moe_layers", None):
            return None

        for model_state in model_runner.eplb_state.model_states.values():
            if model_state.model is draft_moe:
                speculative_config = self.worker.vllm_config.speculative_config
                if speculative_config is None or speculative_config.draft_model_config is None:
                    raise ValueError("[FT] scale_down requires draft_model_config for an EPLB-registered drafter.")
                return drafter, model_state, speculative_config.draft_model_config.model

        raise ValueError(
            "[FT] scale_down requires the MoE drafter to be registered with EPLB; "
            "otherwise its expert placement would keep pointing at dead ranks."
        )

    def _densify_routing_tables(self, eplb_model_state) -> None:
        """Renumber the kernel-facing routing tables into the densified id space.

        The MC2 kernels consume the routing tables in the densified id space,
        so the kernel-facing values are renumbered in place after the refresh.
        The dead set is cumulative (accumulated across recovery rounds), so it
        is derived from the manager's mask rather than this round's ranks.
        """
        p2l = eplb_model_state.physical_to_logical_map
        ep_world_size = get_ep_group().world_size
        num_local_experts = p2l.shape[1] // ep_world_size
        active_mask = get_ep_all2all_manager().query_active_mask()
        dead_ranks = {rank for rank, is_dead in enumerate(active_mask.tolist()) if is_dead}
        orig_to_dense_rank = build_orig_to_dense_rank_table(ep_world_size, dead_ranks)
        for layer in eplb_model_state.model.moe_layers:
            routing_table = getattr(getattr(layer, "eplb_state", None), "expert_replica_routing_table", None)
            if routing_table is not None:
                densify_routing_table_physical_ids(routing_table, orig_to_dense_rank, num_local_experts)
