from contextlib import contextmanager

import torch
import vllm
from vllm.logger import logger

from vllm_ascend.worker.v2.block_table import AscendBlockTables
from vllm_ascend.worker.v2.model_states import init_asecnd_model_state


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


@contextmanager
def block_table_wrapper():
    try:
        # vllm-ascend need to initialize slot mapping as torch.int32 dtype,
        # but vllm default is torch.int64 dtype.
        vllm.v1.worker.gpu.model_runner.BlockTables = AscendBlockTables
        logger.info_once("Wrapping BlockTables with AscendBlockTables.")
        yield
    finally:
        pass


@contextmanager
def model_states_wrapper():
    try:
        # prepare_attn in AscendModelState is different from vllm,
        # we need to override init_model_state.
        vllm.v1.worker.gpu.model_runner.init_model_state = init_asecnd_model_state
        logger.info_once("Wrapping init_model_state with init_asecnd_model_state.")
        yield
    finally:
        pass
