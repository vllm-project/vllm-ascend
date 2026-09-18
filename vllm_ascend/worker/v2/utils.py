from contextlib import contextmanager

import torch
from vllm.compilation import breakable_cudagraph
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.compilation.acl_graph import (
    get_draft_graph_params,
    get_graph_params,
    weak_ref_workspaces,
)
from vllm_ascend.utils import super_kernel_scope, weak_ref_tensor, weak_ref_tensors


@contextmanager
def torch_cuda_wrapper():
    try:
        torch.cuda.Event = torch.npu.Event
        torch.cuda.Stream = torch.npu.Stream
        torch.cuda.stream = torch.npu.stream
        torch.cuda.default_stream = torch.npu.default_stream
        torch.cuda.current_stream = torch.npu.current_stream
        torch.cuda.graph_pool_handle = torch.npu.graph_pool_handle
        torch.cuda.CUDAGraph = torch.npu.NPUGraph
        torch.cuda.graph = torch_npu_graph_wrapper
        torch.cuda.synchronize = torch.npu.synchronize
        torch.cuda.set_stream = torch.npu.set_stream
        torch.cuda.current_device = torch.npu.current_device
        torch.cuda.mem_get_info = torch.npu.mem_get_info
        # GPU V2 prefetch/offload calls this CUDA API during load_model.
        torch.cuda.is_current_stream_capturing = torch.npu.is_current_stream_capturing
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
def torch_npu_graph_wrapper(graph, *args, **kwargs):
    # MRV2-specific cleanup hook: intentionally reuse the graph context
    # manager's exit to weak-ref graph workspaces after each capture,
    # without adding another upstream monkey patch.
    enable_super_kernel = get_ascend_config().ascend_compilation_config.enable_super_kernel
    try:
        with torch.npu.graph(graph, *args, **kwargs), super_kernel_scope("full_model", enable_super_kernel):
            yield
        if enable_super_kernel:
            graph.super_kernel_optimize(
                optimize_options={
                    "dcci_after_kernel_end": [".*"],
                },
            )
            logger.info_once("Super kernel optimization is enabled for ACL graph capture.")
    finally:
        weak_ref_workspaces(get_graph_params())
        weak_ref_workspaces(get_draft_graph_params())
