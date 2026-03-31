from contextlib import contextmanager

import torch
from vllm.logger import logger


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
        torch.cuda.graph = torch.npu.graph
        torch.cuda.synchronize = torch.npu.synchronize
        torch.cuda.set_stream = torch.npu.set_stream
        torch.cuda.current_device = torch.npu.current_device
        torch.cuda.mem_get_info = torch.npu.mem_get_info
        logger.info_once("Wrapping torch.cuda with torch.npu.")
        yield
    finally:
        pass
