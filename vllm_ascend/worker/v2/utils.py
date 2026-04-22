from contextlib import contextmanager

import torch
from vllm.logger import logger

from vllm_ascend.compilation.acl_graph import get_draft_graph_params, get_graph_params, weak_ref_workspaces


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

    try:
        yield
    finally:
        vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator = CudaCommunicator


@contextmanager
def torch_npu_graph_wrapper(*args, **kwargs):
    # MRV2-specific cleanup hook: intentionally reuse the graph context
    # manager's exit to weak-ref graph workspaces after each capture,
    # without adding another upstream monkey patch.
    try:
        with torch.npu.graph(*args, **kwargs):
            yield
    finally:
        weak_ref_workspaces(get_graph_params())
        weak_ref_workspaces(get_draft_graph_params())
