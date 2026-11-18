# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rebuild ACL graphs after a coordinated HBM UCE recovery."""

import torch
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode, set_current_vllm_config
from vllm.logger import logger
from vllm.platforms import current_platform

from vllm_ascend.compilation import acl_graph
from vllm_ascend.device_allocator.sleep_mem_optimized import AclGraphSleepWakeupManager


@torch.inference_mode()
def recover_acl_graphs(worker, request_id: str) -> None:
    """Run while all workers remain quarantined, after device recovery and KV cleanup.

    Keep persistent model and KV tensor addresses. Replace the graph pool
    after dropping graph-owned events, task handles and workspaces;
    capture_model then rebuilds the supported V2 decode graph entries.
    Exceptions must reach the existing EngineCore recovery vote.
    """
    runner = worker.model_runner
    if runner.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
        return
    if not worker.use_v2_model_runner:
        raise RuntimeError("HBM UCE graph recapture requires the V2 model runner")
    _recover_v2_graphs(worker, request_id)


def _recover_v2_graphs(worker, request_id: str) -> None:
    runner = worker.model_runner
    graph_manager = runner.cudagraph_manager
    if graph_manager is None:
        raise RuntimeError("HBM UCE graph recovery requires a V2 graph manager")
    has_encoder_graph = runner.model_state.supports_mm_inputs and runner.model_state.encoder_runner.has_cudagraph()
    if runner.speculator is not None or has_encoder_graph:
        raise RuntimeError("HBM UCE V2 recapture does not support draft or encoder graphs yet")
    descriptors = [desc for descs in graph_manager._capture_descs.values() for desc in descs]
    if not descriptors or any(
        desc.cg_mode != CUDAGraphMode.FULL
        or desc.num_ubatches != 1
        or desc.num_active_loras != 0
        or desc.uniform_token_count != runner.decode_query_len
        for desc in descriptors
    ):
        raise RuntimeError("HBM UCE V2 recapture requires uniform FULL decode graphs without LoRA or microbatching")

    logger.info("[FT] Invalidating V2 ACL graphs after HBM UCE: request_id=%s rank=%s", request_id, worker.rank)
    with set_current_vllm_config(runner.vllm_config):
        torch.npu.synchronize()
        manager = AclGraphSleepWakeupManager(runner.vllm_config, lambda: runner)
        manager.sleep()
        _renew_graph_pool(runner)
        # These are graph-pool output aliases; capture installs fresh outputs.
        graph_manager.hidden_states = None
        graph_manager.aux_hidden_states = []
        graph_manager.intermediate_tensors = None
        try:
            runner.capture_model()
            set_cudagraph_capturing_enabled(False)
            graph_manager.recovery_replayed_graphs = set()
            for desc in descriptors:
                runner._dummy_run(desc.num_tokens, uniform_decode=True, skip_eplb=True)
                torch.npu.synchronize()
            if not set(descriptors).issubset(graph_manager.recovery_replayed_graphs):
                raise RuntimeError("HBM UCE recovery did not replay all recaptured V2 graphs")
        finally:
            graph_manager.recovery_replayed_graphs = None
            set_cudagraph_capturing_enabled(False)
    logger.info("[FT] V2 ACL graphs recaptured and replay verified: request_id=%s rank=%s", request_id, worker.rank)


def _renew_graph_pool(runner) -> None:
    # The last graph release retires its allocator pool (use_count == 0).
    # Reusing that handle for capture triggers create_or_incref_pool's assert.
    # Persistent model/KV allocations stay intact; only graph temporaries move.
    pool = current_platform.graph_pool_handle()
    type(current_platform)._global_graph_pool = pool
    for wrapper in list(acl_graph._acl_graph_wrappers):
        wrapper.graph_pool = pool
    graph_manager = getattr(runner, "cudagraph_manager", None)
    if graph_manager is not None:
        graph_manager.pool = pool
