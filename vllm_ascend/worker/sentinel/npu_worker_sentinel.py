# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from datetime import timedelta
from typing import TYPE_CHECKING

import torch
import torch_npu
from vllm.config import LoadConfig, ModelConfig
from vllm.distributed.eplb.eplb_state import EplbModelState, _commit_eplb_maps
from vllm.distributed.parallel_state import (
    get_dp_group,
    get_ep_group,
    get_tp_group,
)
from vllm.distributed.utils import set_gloo_backend_timeout
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager
from vllm.model_executor.model_loader.utils import get_draft_load_config
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
from vllm_ascend.ascend_forward_context import (
    MoECommType,
    get_mc2_tokens_capacity,
    select_moe_comm_method,
)
from vllm_ascend.distributed.eplb.eplb_state import refresh_model_routing_tables
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.worker.sentinel.eplb_redistribute import (
    densify_routing_table_physical_ids,
    reload_experts_from_disk,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker


_sentinel: "WorkerSentinel | None" = None


def fault_barrier_wrapper(func: Callable):
    """Barrier between device faults and the async step loop.

    On the first device-touching fault (e.g. an EP-group allreduce failing on a
    dead peer that poisons the model stream) it quarantines the worker and
    resets the device immediately, before any tensor teardown on the broken
    stream can std::terminate the process. The fault is then raised through the
    executor's failure channel instead of being reported as an empty success;
    retry lifts the quarantine only after the groups are rebuilt.
    """

    def wrapped(self, *args, **kwargs):
        sentinel = _sentinel
        if sentinel is None:
            return func(self, *args, **kwargs)
        if sentinel.worker_faulted:
            raise RuntimeError(f"[FT] worker is quarantined after fault: {sentinel.fault_info}")
        try:
            return func(self, *args, **kwargs)
        except SystemExit:
            raise
        except Exception as exc:
            sentinel.worker_faulted = True
            sentinel.fault_info = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "[FT] Quarantining %s after fault: %s",
                getattr(self, "rank", "worker"),
                exc,
            )
            try:
                sentinel.reset_device()
            except Exception:
                logger.exception(
                    "[FT] self device reset failed on %s.",
                    getattr(self, "rank", "worker"),
                )
            # Drop the frames that pin device tensors of the broken stream.
            exc.__traceback__ = None
            raise RuntimeError(f"[FT] worker quarantined after fault: {sentinel.fault_info}")

    return wrapped


class WorkerSentinel(GPUWorkerSentinel):
    """Per-worker sentinel for fault tolerance on Ascend NPU.

    Handles commands dispatched from EngineCoreSentinel via collective_rpc,
    including device restart and DP group re-initialization on retry, and
    MC2 elastic_info masking + expert redistribution on scale_down.
    """

    def __init__(self, worker: "Worker", device: torch.device):
        # Upstream's platform-independent guards; MoE drafts are supported here,
        # so its draft check is intentionally omitted.
        if worker.parallel_config.prefill_context_parallel_size > 1:
            raise ValueError("[FT] fault tolerance on Ascend NPU does not support PCP yet.")
        if not worker.vllm_config.use_v2_model_runner:
            raise ValueError("[FT] fault tolerance on Ascend NPU requires the v2 model runner.")
        global _sentinel
        self.device = device
        self.worker = worker
        # Set once a device-touching method faults, to keep this worker off the
        # device until FT recovery rebuilds the groups.
        self.worker_faulted = False
        self.fault_info: str | None = None
        _sentinel = self

    def query_mask(self, ft_request: FaultToleranceRequest) -> dict:
        """Report the dead-rank mask (upstream convention: 0=live, 1=dead)."""
        return {"mask": get_ep_all2all_manager().query_active_mask().tolist()}

    def reset_device(self) -> None:
        NPUPlatform.set_device(self.device)
        torch_npu.npu.stop_device(self.device.index)
        torch_npu.npu.restart_device(self.device.index)
        torch_npu.distributed.reinit_process_group(None, False)
        torch.npu.synchronize()

    def retry(self, ft_request: FaultToleranceRequest):
        # Reset first so hung device collectives are aborted, then run the
        # base flow and lift the quarantine after the groups are rebuilt.
        self.reset_device()
        super().retry(ft_request)
        self.worker_faulted = False
        self.fault_info = None

    def init_num_local_experts(self) -> None:
        """Record the per-rank physical expert slot count after model load."""
        if self.worker.model_runner.eplb_state is None:
            return
        eplb_model_state = self._eplb_model_state()
        num_local_experts = eplb_model_state.physical_to_logical_map.shape[1] // get_ep_group().world_size
        get_ep_all2all_manager().set_num_local_physical_experts(num_local_experts)
        # The MC2 capacity exists only now; surface a config scale_down will reject.
        self._check_mc2_only_execution(raise_on_error=False)

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
        self._check_mc2_only_execution(raise_on_error=True)
        if not hasattr(torch_npu, "npu_moe_distribute_dispatch_v2"):
            raise ValueError(
                "[FT] scale_down requires npu_moe_distribute_dispatch_v2 "
                "(aclnn V3+); please upgrade the CANN/torch_npu version."
            )

    def _max_reachable_num_tokens(self) -> int:
        """Largest batch in tokens: max_num_batched_tokens or the largest graph capture size."""
        capture_sizes = self.worker.vllm_config.compilation_config.cudagraph_capture_sizes or [0]
        return max(self.worker.vllm_config.scheduler_config.max_num_batched_tokens, max(capture_sizes))

    def _check_mc2_only_execution(self, *, raise_on_error: bool) -> None:
        """Require MC2 for every reachable batch; warn or raise when it fails.

        Dead ranks stay in the device EP group; only MC2 excludes them via
        elastic_info, so AllToAll/AllGather would target the stopped rank.
        """
        capacity = get_mc2_tokens_capacity()
        if capacity is None:
            message = "[FT] scale_down requires the MC2 token capacity to be initialized."
        else:
            max_tokens = self._max_reachable_num_tokens()
            comm_type = select_moe_comm_method(max_tokens, self.worker.vllm_config)
            if comm_type is MoECommType.MC2:
                return
            message = (
                f"[FT] scale_down requires MC2 for every reachable batch, but a "
                f"{max_tokens}-token batch selects {comm_type} (MC2 capacity: {capacity}); "
                f"lower --max-num-batched-tokens."
            )
        if raise_on_error:
            raise ValueError(message)
        logger.warning("[FT] scale_down will be rejected at recovery time: %s", message)

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
        vllm_config = self.worker.vllm_config

        models: list[tuple[torch.nn.Module, EplbModelState, ModelConfig, LoadConfig]] = [
            (model_runner.model, self._eplb_model_state(), vllm_config.model_config, vllm_config.load_config)
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

        for model, model_state, model_config, load_config in models:
            self._redistribute_model(model, model_state, model_config, load_config, dead_ep_ranks)

    def _redistribute_model(
        self,
        model: torch.nn.Module,
        model_state: EplbModelState,
        model_config: ModelConfig,
        load_config: LoadConfig,
        dead_ep_ranks: set[int],
    ) -> None:
        """Redistribute one model and reload through its own checkpoint config."""
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
            reload_experts_from_disk(model, model_config, load_config, reassignments)
        refresh_model_routing_tables(model_state)
        self._densify_routing_tables(model_state)

    def _resolve_drafter(self) -> tuple[torch.nn.Module, EplbModelState, ModelConfig, LoadConfig] | None:
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
                return (
                    drafter,
                    model_state,
                    speculative_config.draft_model_config,
                    get_draft_load_config(self.worker.vllm_config),
                )

        raise ValueError(
            "[FT] scale_down requires the MoE drafter to be registered with EPLB; "
            "otherwise its expert placement would keep pointing at dead ranks."
        )

    def _densify_routing_tables(self, eplb_model_state) -> None:
        """Renumber the kernel-facing routing tables into the densified id space.

        The MC2 kernels consume the routing tables in the densified id space,
        so the kernel-facing values are renumbered in place after the refresh.
        The mapping comes from the manager, which owns the cumulative dead set
        used for elastic_info, so both stay on the same numbering.
        """
        p2l = eplb_model_state.physical_to_logical_map
        num_local_experts = p2l.shape[1] // get_ep_group().world_size
        orig_to_dense_rank = get_ep_all2all_manager().orig_to_dense_rank_table()
        for layer in eplb_model_state.model.moe_layers:
            routing_table = getattr(getattr(layer, "eplb_state", None), "expert_replica_routing_table", None)
            if routing_table is not None:
                densify_routing_table_physical_ids(routing_table, orig_to_dense_rank, num_local_experts)
