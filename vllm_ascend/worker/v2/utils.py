from contextlib import contextmanager

import torch


@contextmanager
def torch_cuda_wrapper():
    ori_event = torch.cuda.Event
    ori_stream = torch.cuda.Stream
    ori_default_stream = torch.cuda.default_stream
    ori_current_stream = torch.cuda.current_stream
    ori_graph_pool_handle = torch.cuda.graph_pool_handle
    ori_cuda_graph_cls = torch.cuda.CUDAGraph
    ori_cuda_graph_func = torch.cuda.graph
    try:
        torch.cuda.Event = torch.npu.Event
        torch.cuda.Stream = torch.npu.Stream
        torch.cuda.default_stream = torch.npu.default_stream
        torch.cuda.current_stream = torch.npu.current_stream
        torch.cuda.graph_pool_handle = torch.npu.graph_pool_handle
        torch.cuda.CUDAGraph = torch.npu.NpuGraph
        torch.cuda.graph = torch.npu.graph
        yield
    finally:
        # revert back torch cuda properties, so it will still raise error
        # to call cuda ops in npu environment.
        torch.cuda.Event = ori_event
        torch.cuda.Stream = ori_stream
        torch.cuda.default_stream = ori_default_stream
        torch.cuda.current_stream = ori_current_stream
        torch.cuda.graph_pool_handle = ori_graph_pool_handle
        torch.cuda.CUDAGraph = ori_cuda_graph_cls
        torch.cuda.graph = ori_cuda_graph_func
