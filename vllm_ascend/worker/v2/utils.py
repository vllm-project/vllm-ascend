from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
from vllm.compilation import breakable_cudagraph
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.compilation.acl_graph import (
    get_draft_graph_params,
    get_graph_params,
    weak_ref_workspaces,
)
from vllm_ascend.compilation.updatable_graph import UpdatableGraph
from vllm_ascend.core.kv_cache_interface import is_circular_kv_cache_spec
from vllm_ascend.utils import super_kernel_scope, weak_ref_tensor, weak_ref_tensors
from vllm_ascend.worker.v2.attn_utils import ring_state_update_skipped

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner


@contextmanager
def torch_cuda_wrapper():
    try:
        torch.cuda.Event = torch.npu.Event
        torch.cuda.Stream = torch.npu.Stream
        torch.cuda.stream = torch.npu.stream
        torch.cuda.default_stream = torch.npu.default_stream
        torch.cuda.current_stream = torch.npu.current_stream
        torch.cuda.graph_pool_handle = torch.npu.graph_pool_handle
        torch.cuda.CUDAGraph = UpdatableGraph
        torch.cuda.graph = torch_npu_graph_wrapper
        torch.cuda.synchronize = torch.npu.synchronize
        torch.cuda.set_stream = torch.npu.set_stream
        torch.cuda.current_device = torch.npu.current_device
        torch.cuda.mem_get_info = torch.npu.mem_get_info
        breakable_cudagraph.weak_ref_tensor = weak_ref_tensor
        breakable_cudagraph.weak_ref_tensors = weak_ref_tensors
        logger.info_once("Wrapping torch.cuda with torch.npu.")
        yield
    finally:
        pass


@contextmanager
def communicator_switch():
    import vllm.distributed.device_communicators.cuda_communicator

    from vllm_ascend.distributed.device_communicators.npu_communicator import NPUCommunicator

    CudaCommunicator = vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator
    vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator = NPUCommunicator
    logger.debug("Switched CudaCommunicator -> NPUCommunicator for graph capture.")

    try:
        yield
    finally:
        vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator = CudaCommunicator
        logger.debug("Restored CudaCommunicator after graph capture.")


@contextmanager
def torch_npu_graph_wrapper(*args, **kwargs):
    # MRV2-specific cleanup hook: intentionally reuse the graph context
    # manager's exit to weak-ref graph workspaces after each capture,
    # without adding another upstream monkey patch.
    enable_super_kernel = get_ascend_config().ascend_compilation_config.enable_super_kernel
    graph = args[0] if args else (kwargs.get("cuda_graph") or kwargs.get("graph"))
    try:
        with torch.npu.graph(*args, **kwargs), super_kernel_scope("full_model", enable_super_kernel):
            yield
        if enable_super_kernel and graph is not None:
            graph.super_kernel_optimize(
                optimize_options={
                    "dcci_after_kernel_end": [".*"],
                },
            )
            logger.info_once("Super kernel optimization is enabled for ACL graph capture.")
    finally:
        weak_ref_workspaces(get_graph_params())
        weak_ref_workspaces(get_draft_graph_params())


def prepare_v41_source_rope(runner: "NPUModelRunner") -> None:
    """Validate and cache V4.1 source RoPE tables on compressor builders.

    Runner V1 wires this through ``enable_device_metadata`` inside
    ``initialize_attn_backend``; runner V2 keeps metadata tasks
    synchronous (``_publish_task`` runs them inline), so only the RoPE
    cache initialization is needed here. ``build`` raises without it.
    """
    # Lazy import avoids the model/cache registration cycle.
    from vllm_ascend.attention.dsa_v41 import AscendDSAV41MetadataBuilder

    for groups in runner.attn_groups:
        for attn_group in groups:
            for builder in attn_group.metadata_builders:
                if isinstance(builder, AscendDSAV41MetadataBuilder):
                    builder.prepare_source_rope()


def prepare_v41_dummy_ring_state(runner: "NPUModelRunner", num_reqs: int) -> None:
    """Assign live ring pages to dummy requests for V4.1 graph runs.

    V4.1's compressor ring state owns one private page per request.
    Upstream zero-fills dummy block tables, which would alias every
    dummy request onto page 0; assign distinct live state IDs
    1..num_reqs and zero those ring pages so graph capture/replay see
    a clean ring instead of stale or aliased state. Fully skipped for
    dummy batches marked skip_gdn_state_update (mirrors MRV1, which
    suppresses both the ring prep and the state writes there).
    """
    if ring_state_update_skipped():
        return
    forward_context = runner.compilation_config.static_forward_context
    for gid, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        if not is_circular_kv_cache_spec(group.kv_cache_spec):
            continue
        if num_reqs >= runner.kv_cache_config.num_blocks:
            raise ValueError("Insufficient ring pages for dummy graph requests")
        block_table = runner.block_tables.input_block_tables[gid]
        block_table[:num_reqs, 0] = torch.arange(
            1,
            num_reqs + 1,
            dtype=block_table.dtype,
            device=block_table.device,
        )
        for name in group.layer_names:
            forward_context[name].kv_cache[0][1 : num_reqs + 1].zero_()
