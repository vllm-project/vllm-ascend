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
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager
from vllm.model_executor.model_loader.utils import get_draft_load_config
from vllm.model_executor.models.interfaces import get_mixture_of_experts_model
from vllm.v1.fault_tolerance.utils import FaultToleranceRequest
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
    densify_routing_tables,
    reload_experts_from_disk,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker


_sentinel: "WorkerSentinel | None" = None


def _num_local_experts(model_state: EplbModelState) -> int:
    """How many per-expert weight tensors each original EP rank hosts (dead ranks keep theirs)."""
    return model_state.physical_to_logical_map.shape[1] // get_ep_group().world_size


def fault_barrier_wrapper(method: Callable):
    """Wrap a worker method so a device fault is handled where it happens.

    If the wrapped method raises (for example, an EP all-reduce times out
    because a peer died), mark the worker as faulted, stop and restart its
    device, then re-raise the error. While the flag is set, wrapped calls fail
    immediately instead of touching the broken device; ``retry`` clears it once
    the process groups have been rebuilt.
    """

    def wrapped(self, *args, **kwargs):
        sentinel = _sentinel
        if sentinel is None:
            return method(self, *args, **kwargs)
        if sentinel.worker_faulted:
            raise RuntimeError(f"[FT] worker is faulted and will not use the device until retry: {sentinel.fault_info}")
        try:
            return method(self, *args, **kwargs)
        except Exception as exc:
            rank = getattr(self, "rank", None)
            rank_desc = f"rank {rank}" if isinstance(rank, int) else "worker"
            sentinel.worker_faulted = True
            sentinel.fault_info = f"{type(exc).__name__}: {exc}"
            logger.warning("[FT] Marking %s faulted after: %s", rank_desc, exc)
            try:
                sentinel.reset_device()
            except Exception:
                logger.exception("[FT] device reset failed on %s.", rank_desc)
            # Drop the frames that pin device tensors of the broken stream.
            exc.__traceback__ = None
            raise RuntimeError(f"[FT] worker faulted: {sentinel.fault_info}")

    return wrapped


class WorkerSentinel(GPUWorkerSentinel):
    """Per-worker sentinel for fault tolerance on Ascend NPU.

    Handles commands dispatched from EngineCoreSentinel via collective_rpc,
    including device restart and DP group re-initialization on retry, and
    MC2 elastic_info masking + expert redistribution on scale_down.
    """

    def __init__(self, worker: "Worker", device: torch.device):
        # Two constraints: PCP is not supported yet, and fault tolerance needs
        # the v2 model runner.
        if worker.parallel_config.prefill_context_parallel_size > 1:
            raise ValueError("[FT] fault tolerance on Ascend NPU does not support PCP yet.")
        if not worker.vllm_config.use_v2_model_runner:
            raise ValueError("[FT] fault tolerance on Ascend NPU requires the v2 model runner.")
        global _sentinel
        self.device = device
        self.worker = worker
        # Set when a device-touching method fails; cleared by retry. While set,
        # wrapped methods fail fast instead of using the device.
        self.worker_faulted = False
        self.fault_info: str | None = None
        _sentinel = self

    def query_mask(self, ft_request: FaultToleranceRequest) -> dict:
        """Report the dead-rank mask (upstream convention: 0=live, 1=dead)."""
        return {"mask": get_ep_all2all_manager().query_active_mask().tolist()}

    def reset_device(self) -> None:
        """Stop and restart the NPU, then rebuild the process groups."""
        NPUPlatform.set_device(self.device)
        torch_npu.npu.stop_device(self.device.index)
        torch_npu.npu.restart_device(self.device.index)
        torch_npu.distributed.reinit_process_group(None, False)
        torch.npu.synchronize()

    def retry(self, ft_request: FaultToleranceRequest):
        # Reset first so hung device collectives are aborted, then run the
        # upstream retry and clear the fault flag once the groups are rebuilt.
        self.reset_device()
        super().retry(ft_request)
        self.worker_faulted = False
        self.fault_info = None

    def init_num_local_experts(self) -> None:
        """Record how many per-expert weight tensors this rank hosts after model load."""
        if self.worker.model_runner.eplb_state is None:
            return
        get_ep_all2all_manager().set_num_local_physical_experts(_num_local_experts(self._eplb_model_state()))
        # The MC2 token capacity is known only after model load, so check it
        # now and warn if scale_down would be rejected at recovery time.
        self._check_comm_method_supports_scale_down(raise_on_error=False)

    def scale_down(self, ft_request: FaultToleranceRequest):
        """Remove dead DP ranks and recover.

        Runs the upstream scale-down flow with Ascend preconditions and a
        dummy-batch runnability check on top.
        """
        self._validate_scale_down_preconditions()
        super().scale_down(ft_request)

        # Imported lazily: vLLM v0.33.0 does not export it, and normal worker
        # startup imports this module without running scale_down.
        from vllm.distributed.utils import set_gloo_backend_timeout

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
        """Check the config and kernel preconditions scale_down requires."""
        model_runner = self.worker.model_runner
        eplb_config = self.worker.parallel_config.eplb_config
        if model_runner.eplb_state is None or eplb_config.num_redundant_experts <= 0:
            raise ValueError(
                "[FT] scale_down requires EPLB with num_redundant_experts > 0 to re-host the dead rank's experts."
            )
        self._find_eplb_drafter()
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
        self._check_comm_method_supports_scale_down(raise_on_error=True)
        if not hasattr(torch_npu, "npu_moe_distribute_dispatch_v2"):
            raise ValueError(
                "[FT] scale_down requires npu_moe_distribute_dispatch_v2 "
                "(aclnn V3+); please upgrade the CANN/torch_npu version."
            )

    def _max_reachable_num_tokens(self) -> int:
        """Largest batch in tokens: max_num_batched_tokens or the largest graph capture size."""
        capture_sizes = self.worker.vllm_config.compilation_config.cudagraph_capture_sizes or [0]
        return max(self.worker.vllm_config.scheduler_config.max_num_batched_tokens, max(capture_sizes))

    def _check_comm_method_supports_scale_down(self, *, raise_on_error: bool) -> None:
        """Check that every reachable batch uses a MoE communication method that scale-down supports.

        Scale-down currently supports MC2 (MegaMoE is planned): the MC2
        dispatch/combine kernels take the ``elastic_info`` tensor (a per-EP-rank
        live/dead mask passed on every call) and skip dead ranks. Dead ranks
        stay in the device EP group, so AllToAll/AllGather would keep targeting
        the stopped rank.
        """
        capacity = get_mc2_tokens_capacity()
        if capacity is None:
            message = "[FT] scale_down requires the MC2 token capacity to be initialized."
        else:
            max_tokens = self._max_reachable_num_tokens()
            comm_type = select_moe_comm_method(max_tokens, self.worker.vllm_config)
            if comm_type is MoECommType.MC2:
                return
            # TODO: accept MegaMoE/fused MC2 once dispatch_ffn_combine takes elastic_info.
            message = (
                f"[FT] scale_down requires MC2 for every reachable batch, but a "
                f"{max_tokens}-token batch selects {comm_type} (MC2 capacity: {capacity}); "
                f"lower --max-num-batched-tokens."
            )
        if raise_on_error:
            raise ValueError(message)
        logger.warning("[FT] scale_down will be rejected at recovery time: %s", message)

    def _redistribute_experts(self, dead_ep_ranks: set[int]) -> None:
        """Redistribute the dead ranks' experts onto the surviving ranks' per-expert tensors.

        Runs the upstream redistribution steps, then reloads the moved experts'
        weights through the Ascend reloader and renumbers the routing tables the
        kernels read so dead ranks are skipped. Each EPLB-registered model (main
        model and drafter) is redistributed against its own placement and
        reloaded from its own checkpoint.
        """
        model_runner = self.worker.model_runner
        vllm_config = self.worker.vllm_config

        models: list[tuple[torch.nn.Module, EplbModelState, ModelConfig, LoadConfig]] = [
            (model_runner.model, self._eplb_model_state(), vllm_config.model_config, vllm_config.load_config)
        ]
        drafter = self._find_eplb_drafter()
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
        # Imported lazily: the helper was added after vLLM v0.33.0, and normal
        # worker startup imports this module without running this path.
        from vllm.v1.worker.sentinel.eplb_redistribute import (
            mark_dead_expert_slots_inplace,
            redistribute_expert_placement,
        )

        physical_to_logical = model_state.physical_to_logical_map
        num_logical = model_state.logical_replica_count.shape[1]
        num_local_experts = _num_local_experts(model_state)

        mark_dead_expert_slots_inplace(physical_to_logical, dead_ep_ranks, num_local_experts)
        moved_experts = redistribute_expert_placement(physical_to_logical, num_logical, num_local_experts)
        logger.info(
            "[FT] %s expert redistribution: moe_layers=%d, num_logical=%d, moved_experts=%d",
            type(model).__name__,
            physical_to_logical.shape[0],
            num_logical,
            len(moved_experts),
        )
        _commit_eplb_maps(model_state, physical_to_logical.cpu())
        # The model's expert map is indexed by physical expert id and no id
        # changes here; the logical-id map must not overwrite it.
        if moved_experts:
            reload_experts_from_disk(model, model_config, load_config, moved_experts)
        refresh_model_routing_tables(model_state)
        densify_routing_tables(
            model_state.model.moe_layers,
            get_ep_all2all_manager().orig_to_dense_rank_table(),
            num_local_experts,
        )

    def _find_eplb_drafter(self) -> tuple[torch.nn.Module, EplbModelState, ModelConfig, LoadConfig] | None:
        """Find the MoE drafter's EPLB state and checkpoint, rejecting an unregistered drafter."""
        model_runner = self.worker.model_runner
        drafter = getattr(getattr(model_runner, "speculator", None), "model", None)
        draft_moe = get_mixture_of_experts_model(drafter)
        if draft_moe is None and not getattr(drafter, "moe_layers", None):
            return None
        state = next((s for s in model_runner.eplb_state.model_states.values() if s.model is draft_moe), None)
        spec = self.worker.vllm_config.speculative_config
        if state is None:
            raise ValueError(
                "[FT] scale_down requires the MoE drafter to be registered with EPLB; "
                "otherwise its expert placement would keep pointing at dead ranks."
            )
        if spec is None or spec.draft_model_config is None:
            raise ValueError("[FT] scale_down requires draft_model_config for an EPLB-registered drafter.")
        return drafter, state, spec.draft_model_config, get_draft_load_config(self.worker.vllm_config)
