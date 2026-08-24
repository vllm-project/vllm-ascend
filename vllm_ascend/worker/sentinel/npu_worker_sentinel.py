# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import torch
import torch_npu
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager
from vllm.v1.fault_tolerance.utils import FaultToleranceRequest
from vllm.v1.worker.sentinel.gpu_worker_sentinel import (
    WorkerSentinel as GPUWorkerSentinel,
)

from vllm_ascend.platform import NPUPlatform
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.eplb.state import refresh_model_routing_tables
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.worker.sentinel.eplb_redistribute import (
    build_local_reload_plan,
    check_redundancy_sufficient,
    compute_dead_ep_ranks,
    densify_routing_table_physical_ids,
    mark_dead_expert_slots_inplace,
    rebuild_logical_expert_maps,
    redistribute_expert_placement,
    reload_experts_from_disk,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker


class WorkerSentinel(GPUWorkerSentinel):
    """Per-worker sentinel for fault tolerance on Ascend NPU.

    Handles commands dispatched from EngineCoreSentinel via collective_rpc,
    including device restart and DP group re-initialization on retry, and
    MC2 elastic_info masking + expert redistribution on scale_down.
    """

    def __init__(self, worker: "Worker", device: torch.device):
        self.worker = worker
        self.device = device
        self.dp_rank = worker.parallel_config.data_parallel_rank
        self.dp_size = worker.parallel_config.data_parallel_size
        self.data_parallel_master_ip = worker.parallel_config.data_parallel_master_ip

    def query_mask(self, ft_request: FaultToleranceRequest) -> dict:
        """Report the dead-rank mask (upstream convention: 0=live, 1=dead).

        Pure CPU read: on NPU the mask is only ever written host-side during
        FT recovery, so on a first fault this is all zeros and the
        orchestrator's retry-vs-scale_down decision must come from its own
        cluster knowledge.
        """
        return {"mask": get_ep_all2all_manager().query_active_mask().tolist()}

    def reset_device(self) -> None:
        NPUPlatform.set_device(self.device)
        torch_npu.npu.stop_device(self.device.index)
        torch_npu.npu.restart_device(self.device.index)
        torch_npu.distributed.reinit_process_group(None, False)
        torch.npu.synchronize()

    def retry(self, ft_request: FaultToleranceRequest):
        # reset the device first so any hung device-side collectives are
        # aborted, then run the base class flow (synchronize, clean worker
        # state, re-initialize the DP group).
        self.reset_device()
        super().retry(ft_request)

    def scale_down(self, ft_request: FaultToleranceRequest):
        """Scale down over the surviving DP ranks, mirroring the upstream
        GPUWorkerSentinel.scale_down structure."""
        self._validate_scale_down_preconditions()
        params = ft_request.params
        tp_size = self.worker.parallel_config.tensor_parallel_size
        dead_ep_ranks = compute_dead_ep_ranks(params["dead_dp_ranks"], tp_size)
        eplb_model_state = self._eplb_model_state()
        ep_world_size = get_mc2_group().world_size

        check_redundancy_sufficient(
            eplb_model_state.logical_replica_count.shape[1],
            eplb_model_state.physical_to_logical_map.shape[1] // ep_world_size,
            ep_world_size,
            dead_ep_ranks,
        )
        # Suppress EPLB before retry so the base flow skips the EPLB group
        # reinit; the expert placement is fixed by the redistribution below.
        self.worker.model_runner.eep_eplb_suppressed = True
        # Device reset + worker-state cleanup + DP gloo group rebuilt with
        # the densified membership; the base retry also replays the
        # cumulative dead set into the elastic_info mask via update_mask.
        self.retry(ft_request)

        self._redistribute_experts(dead_ep_ranks)

        # Verify the redistributed model is runnable before reporting healthy.
        self.worker.execute_dummy_batch()
        torch.npu.synchronize()

        logger.info(
            "[FT] Worker scale_down complete: dp_group_size=%d, "
            "dp_group_rank=%d, dead_ep_ranks=%s, eplb_suppressed=True",
            params["dp_group_size"],
            params["dp_group_rank"],
            sorted(dead_ep_ranks),
        )

    def _validate_scale_down_preconditions(self) -> None:
        if not self.worker.use_v2_model_runner:
            raise ValueError("[FT] scale_down on Ascend NPU requires the v2 model runner.")
        model_runner = self.worker.model_runner
        eplb_config = self.worker.parallel_config.eplb_config
        if model_runner.eplb_state is None or eplb_config.num_redundant_experts <= 0:
            raise ValueError(
                "[FT] scale_down requires EPLB with num_redundant_experts > 0 to re-host the dead rank's experts."
            )
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
        """One-shot expert redistribution after scale-down.

        Cannot be inherited from the upstream sentinel: the upstream version
        applies the new placement via `rebuild_model_expert_maps` (GPU expert
        maps), reloads weights in the GPU runtime layout, and calls
        `sync_num_dispatchers_for_nixl_ep`. On Ascend the placement must
        instead be propagated into the Ascend routing tables
        (`refresh_model_routing_tables`), reassigned weights must be rewritten
        in the NPU runtime layout (transpose / NZ cast / per-slot lists — the
        `reload_experts_from_disk` in eplb_redistribute.py), and the MC2
        expert width must be shrunk via `set_num_physical_experts`. Only the
        pure placement math is shared (imported from upstream); the apply
        steps are platform-specific.

        Deterministic, no cross-rank communication: every surviving rank
        computes bit-identical placements from the shared EPLB model state.
        The EPLB maps keep the upstream slot model (full width, original
        physical ids); the densified rank numbering lives only in the
        elastic_info tensor maintained by the all2all manager.
        """
        eplb_model_state = self._eplb_model_state()
        p2l = eplb_model_state.physical_to_logical_map
        l2p = eplb_model_state.logical_to_physical_map
        lrc = eplb_model_state.logical_replica_count
        num_logical = lrc.shape[1]
        ep_world_size = get_mc2_group().world_size
        num_local_experts = p2l.shape[1] // ep_world_size
        # The original EP rank; never rewritten by scale-down.
        ep_rank = get_mc2_group().rank_in_group

        p2l_before = p2l.cpu().clone()
        mark_dead_expert_slots_inplace(p2l, dead_ep_ranks, num_local_experts)
        redistribute_expert_placement(p2l, num_logical, num_local_experts)
        rebuild_logical_expert_maps(p2l, l2p, lrc)
        # Propagate the new placement into the Ascend routing tables; the
        # table shape is unchanged so this copy_'s into the storage captured
        # by graphs.
        refresh_model_routing_tables(eplb_model_state)

        # The MC2 kernels require the densified physical-id space while scaled
        # down (dispatch routes via table2[expert_id // num_local]; combine
        # silently drops ids >= the shrunk physical expert count). The refresh
        # above rebuilds the tables with original full-width ids, so renumber
        # the kernel-facing values in place. Original ids only route correctly
        # when the dead ranks are a suffix; a dead rank in the middle would
        # misroute tokens or crash the kernel on a -1 rank lookup.
        orig_to_dense_rank = get_ep_all2all_manager().to_densified_rank_table()
        for layer in eplb_model_state.model.moe_layers:
            routing_table = getattr(getattr(layer, "eplb_state", None), "expert_replica_routing_table", None)
            if routing_table is not None:
                densify_routing_table_physical_ids(routing_table, orig_to_dense_rank, num_local_experts)

        reload_plan = build_local_reload_plan(p2l_before, p2l.cpu(), ep_rank, num_local_experts)
        if reload_plan:
            reload_experts_from_disk(
                self.worker.model_runner.model,
                self.worker.vllm_config,
                reload_plan,
            )

        # Shrink the physical-expert width the MC2 kernels see so it matches
        # the surviving slots; this also flips elastic_info into scaling-down
        # mode (the mask replayed during retry kept the flag at 0 until the
        # new width was known).
        get_ep_all2all_manager().set_num_physical_experts((ep_world_size - len(dead_ep_ranks)) * num_local_experts)

        logger.info(
            "[FT] Expert redistribution: num_logical=%d, ep_world_size=%d, num_local_experts=%d, reloaded_slots=%s",
            num_logical,
            ep_world_size,
            num_local_experts,
            sum(len(v) for v in reload_plan.values()),
        )
