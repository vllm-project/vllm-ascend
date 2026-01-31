# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Optional
import copy

import torch

from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id)
from vllm.forward_context import get_forward_context, override_forward_context, DPMetadata
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.worker.ubatching import make_ubatch_contexts
from vllm.v1.worker.gpu_ubatch_wrapper import (
    UBatchWrapper as GPUUBatchWrapper,
    UbatchMetadata
)

logger = init_logger(__name__)


@dataclass
class ACLGraphMetaData:
    aclgraph: torch.npu.NPUGraph
    ubatch_metadata: UbatchMetadata
    outputs: Optional[Any] = None


@contextmanager
def _torch_cuda_wrapper():
    """NPU特有的上下文管理器，用于替换torch.cuda API为torch.npu API"""

    class _EventPlaceholder:

        def __init__(self, *args, **kwargs) -> None:
            self.record = lambda: None
            self.synchronize = lambda: None

    class _StreamPlaceholder:

        def __init__(self, *args, **kwargs) -> None:
            pass

    # Backup original
    original_cuda = None
    if hasattr(torch, 'cuda'):
        original_cuda = torch.cuda

    try:
        # replace cuda APIs with npu APIs
        torch.cuda.Event = _EventPlaceholder
        torch.cuda.Stream = torch.npu.Stream
        torch.cuda.default_stream = torch.npu.default_stream
        torch.cuda.current_stream = torch.npu.current_stream
        torch.cuda.stream = torch.npu.stream
        torch.cuda.set_stream = torch.npu.set_stream
        yield
    except Exception:
        torch.cuda.Event = _EventPlaceholder
        torch.cuda.Stream = _StreamPlaceholder
        torch.cuda.default_stream = _StreamPlaceholder
        torch.cuda.current_stream = _StreamPlaceholder
        torch.cuda.stream = _StreamPlaceholder
        torch.cuda.set_stream = torch.npu.set_stream
    finally:
        if original_cuda:
            torch.cuda = original_cuda
        else:
            del torch.cuda


class _EmptyContextManager:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class UBatchWrapper(GPUUBatchWrapper):
    def __init__(self, runnable: Callable, vllm_config: VllmConfig,
                 runtime_mode: CUDAGraphMode, device: torch.npu.device):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.comm_stream = torch.npu.Stream(device=device)

        self.ready_barrier = threading.Barrier(
            self.vllm_config.parallel_config.num_ubatches + 1
        )

        self.aclgraphs: dict[int, ACLGraphMetaData] = {}
        self.cudagraphs = self.aclgraphs

        self.aclgraph_wrapper = None
        self.cudagraph_wrapper = None  # 为了兼容父类引用
        self.graph_pool = None
        if runtime_mode is not CUDAGraphMode.NONE:
            self.aclgraph_wrapper = ACLGraphWrapper(
                runnable, vllm_config, runtime_mode=runtime_mode)
            self.cudagraph_wrapper = self.aclgraph_wrapper
            self.graph_pool = current_platform.get_global_graph_pool()

        self.sm_control = _EmptyContextManager()
        
        self.device = device

    @staticmethod
    def _create_sm_control_context(vllm_config: VllmConfig):
        return _EmptyContextManager()

    @_torch_cuda_wrapper()
    def _capture_ubatches(self, ubatch_metadata, model) -> torch.Tensor:
        """
        NPU：Capture a ACLGraph for a microbatched run.
        """

        @torch.inference_mode()
        def _capture_ubatch_thread(results, ubatch_metadata):
            torch.npu.set_device(self.device)
            ubatch_context = ubatch_metadata.context
            # NPU特殊逻辑：不需要初始化blas_handle
            with ubatch_context:
                model_output = model(
                    input_ids=ubatch_metadata.input_ids,
                    positions=ubatch_metadata.positions,
                    intermediate_tensors=ubatch_metadata.intermediate_tensors,
                    inputs_embeds=ubatch_metadata.inputs_embeds,
                )

            results.append((ubatch_metadata.context.id, model_output))
            # TODO(jcz):这里需要同步吗？如果不同步，会导致crash
            import time
            time.sleep(2)

        results: list[tuple[int, torch.Tensor]] = []
        compute_stream = ubatch_metadata[0].context.compute_stream
        num_tokens = 0
        for metadata in ubatch_metadata:
            num_tokens += metadata.num_tokens

        # Ubatches will manually manage the forward context, so we override
        # it to None here so we can have it restored correctly later
        with override_forward_context(None):
            ubatch_threads = []
            for metadata in ubatch_metadata:
                thread = threading.Thread(target=_capture_ubatch_thread,
                                          args=(
                                              results,
                                              metadata,
                                          ))
                ubatch_threads.append(thread)
                thread.start()
            self.ready_barrier.wait()  # Wait for all threads to be ready

            # Capture the ACLGraph
            aclgraph_metadata = \
                ACLGraphMetaData(
                            aclgraph=torch.npu.NPUGraph(),
                            ubatch_metadata=ubatch_metadata,
                        )
            if self.graph_pool is not None:
                set_graph_pool_id(self.graph_pool)
            else:
                set_graph_pool_id(current_platform.graph_pool_handle())
            with torch.npu.graph(aclgraph_metadata.aclgraph,
                                  stream=compute_stream,
                                  pool=self.graph_pool):
                ubatch_metadata[0].context.cpu_wait_event.set()
                for thread in ubatch_threads:
                    thread.join()
                sorted_results = [value for position, value in sorted(results)]
                result = torch.cat(sorted_results, dim=0)
                aclgraph_metadata.outputs = result
            self.aclgraphs[num_tokens] = aclgraph_metadata
        return aclgraph_metadata.outputs

    @_torch_cuda_wrapper()
    def _run_ubatches(self, ubatch_metadata, model) -> torch.Tensor:

        @torch.inference_mode()
        def _ubatch_thread(results, model, ubatch_metadata):
            with ubatch_metadata.context:
                model_output = model(
                    input_ids=ubatch_metadata.input_ids,
                    positions=ubatch_metadata.positions,
                    intermediate_tensors=ubatch_metadata.intermediate_tensors,
                    inputs_embeds=ubatch_metadata.inputs_embeds,
                )
            results.append((ubatch_metadata.context.id, model_output))
            torch.npu.synchronize()

        results: list[tuple[int, torch.Tensor]] = []

        # Ubatch threads will manually manage the forward context, so we
        # override it to None here so we can have it restored correctly
        # after all threads have finished
        with override_forward_context(None):
            ubatch_threads = []
            for metadata in ubatch_metadata:
                thread = threading.Thread(target=_ubatch_thread,
                                            args=(
                                                results,
                                                model,
                                                metadata,
                                            ))
                ubatch_threads.append(thread)
                thread.start()
            self.ready_barrier.wait()  # Wait for all threads to be ready
            ubatch_metadata[0].context.cpu_wait_event.set()
            for thread in ubatch_threads:
                thread.join()
            sorted_results = [value for position, value in sorted(results)]
            result = torch.cat(sorted_results, dim=0)
            return result

    @_torch_cuda_wrapper()
    def _make_ubatch_metadata(self, ubatch_slices, attn_metadata, input_ids,
                              positions, inputs_embeds, intermediate_tensors,
                              compute_stream, dp_metadata, batch_descriptor,
                              aclgraph_runtime_mode, afd_metadata) -> list[UbatchMetadata]:
        # Create one forward context per ubatch
        forward_contexts = []
        for i, ubatch_slice in enumerate(ubatch_slices):
            forward_context = copy.copy(get_forward_context())

            dp_size = self.vllm_config.parallel_config.data_parallel_size
            ubatch_num_tokens_across_dp = torch.tensor(
                [ubatch_slice.num_tokens] * dp_size, device="cpu", dtype=torch.int32
            )
            ubatch_dp_metadata = DPMetadata.make(
                self.vllm_config.parallel_config,
                attn_metadata[i] if attn_metadata is not None else None,
                ubatch_slice.num_tokens,
                ubatch_num_tokens_across_dp,
            )
            forward_context.dp_metadata = ubatch_dp_metadata
            forward_context.ubatch_idx = i
            forward_context.attn_metadata = attn_metadata[i] if attn_metadata is not None else None
            forward_context.no_compile_layers = self.vllm_config.compilation_config.static_forward_context
            forward_context.cudagraph_runtime_mode = aclgraph_runtime_mode
            forward_context.batch_descriptor = batch_descriptor
            forward_context.afd_metadata = afd_metadata
            forward_context.num_ubatches = len(ubatch_slices)
            forward_context.afd_comm_event = torch.npu.Event()
            forward_contexts.append(forward_context)

        ubatch_ctxs = make_ubatch_contexts(
            num_micro_batches=len(ubatch_slices),
            comm_stream=self.comm_stream,
            compute_stream=compute_stream,
            forward_contexts=forward_contexts,
            ready_barrier=self.ready_barrier)

        ubatch_metadata: list[UbatchMetadata] = []
        for i, ubatch_slice in enumerate(ubatch_slices):
            sliced_input_ids, sliced_positions, sliced_inputs_embeds, \
            sliced_intermediate_tensors = \
                self._slice_model_inputs(
                    ubatch_slice.token_slice, input_ids, positions,
                    inputs_embeds, intermediate_tensors)
            ubatch_metadata.append(
                UbatchMetadata(
                    context=ubatch_ctxs[i],
                    input_ids=sliced_input_ids,
                    positions=sliced_positions,
                    inputs_embeds=sliced_inputs_embeds,
                    intermediate_tensors=sliced_intermediate_tensors,
                    num_tokens=ubatch_slice.token_slice.stop -
                    ubatch_slice.token_slice.start))

        return ubatch_metadata

    @_torch_cuda_wrapper()
    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        ubatch_slices = forward_context.ubatch_slices
        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode
        afd_metadata = forward_context.afd_metadata
        
        # If there's no ubatching, just run the runnable object
        if ubatch_slices is None:

            # This is to account for the case where ubatching was aborted.
            # When we capture full graphs we only capture one graph per shape,
            # meaning that if we have a ubatched graph for the current
            # num_tokens, we don't have a non-ubatched one. Without this
            # check, the graph wrapper will try to capture a graph
            # for this shape during a normal run.
            if cudagraph_runtime_mode is CUDAGraphMode.FULL:
                assert batch_descriptor is not None
                if batch_descriptor.num_tokens in self.aclgraphs:
                    cudagraph_runtime_mode = CUDAGraphMode.NONE

            if cudagraph_runtime_mode in (CUDAGraphMode.NONE,
                                          CUDAGraphMode.PIECEWISE):
                return self.runnable(*args, **kwargs)
            else:
                assert self.aclgraph_wrapper is not None
                return self.aclgraph_wrapper(*args, **kwargs)

        attn_metadata = forward_context.attn_metadata
        num_tokens = 0
        for ubatch_slice in ubatch_slices:
            num_tokens += ubatch_slice.num_tokens
        input_ids = kwargs['input_ids']
        positions = kwargs['positions']
        intermediate_tensors = kwargs['intermediate_tensors']
        inputs_embeds = kwargs['inputs_embeds']
        compute_stream = torch.npu.current_stream()

        dp_metadata = forward_context.dp_metadata

        # We shouldn't be here unless we are running with multiple DP ranks
        assert dp_metadata is not None

        if num_tokens not in self.aclgraphs \
            and cudagraph_runtime_mode is CUDAGraphMode.FULL:
            ubatch_metadata = self._make_ubatch_metadata(
                ubatch_slices=ubatch_slices,
                attn_metadata=attn_metadata,
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                compute_stream=compute_stream,
                dp_metadata=dp_metadata,
                batch_descriptor=batch_descriptor,
                aclgraph_runtime_mode=CUDAGraphMode.NONE,
                afd_metadata=afd_metadata)
            return self._capture_ubatches(ubatch_metadata, self.model)
        elif num_tokens in self.aclgraphs \
            and cudagraph_runtime_mode is CUDAGraphMode.FULL:
            aclgraph_metadata = self.aclgraphs[num_tokens]
            aclgraph_metadata.aclgraph.replay()
            print("UBatchWrapper replay")
            return aclgraph_metadata.outputs
        else:
            ubatch_metadata = self._make_ubatch_metadata(
                ubatch_slices=ubatch_slices,
                attn_metadata=attn_metadata,
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                compute_stream=compute_stream,
                dp_metadata=dp_metadata,
                batch_descriptor=batch_descriptor,
                aclgraph_runtime_mode=CUDAGraphMode.NONE,
                afd_metadata=afd_metadata)
            return self._run_ubatches(ubatch_metadata, self.model)
