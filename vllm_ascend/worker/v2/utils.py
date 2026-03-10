from contextlib import contextmanager

import vllm
import torch

from vllm_ascend.worker.v2.block_table import AscendBlockTables


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
        yield
    finally:
        pass


@contextmanager
def block_table_wrapper():
    try:
        # vllm-ascend need to initialize slot mapping as torch.int32 dtype,
        # but vllm default is torch.int64 dtype.
        vllm.v1.worker.gpu.block_table.BlockTables = AscendBlockTables
        yield
    finally:
        pass
