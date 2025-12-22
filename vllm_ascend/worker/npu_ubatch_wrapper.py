# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

import vllm.envs as envs
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_pp_group, tensor_model_parallel_all_gather
from vllm.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id)
from vllm.forward_context import (get_forward_context,
                                  override_forward_context)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
#from vllm.utils import has_deep_gemm
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.v1.worker.gpu_ubatch_wrapper import UbatchMetadata
from vllm_ascend.utils import enable_sp, dbo_current_stream
from vllm_ascend.worker.ubatching import make_ubatch_contexts, dbo_yield
from vllm_ascend.ascend_forward_context import create_ascend_forward_context

logger = init_logger(__name__)


@dataclass
class AscendUbatchMetadata(UbatchMetadata):
    pass


# TODO: adapt to npu graph mode
@dataclass
class AscendNPUGraphMetaData:
    aclgraph: torch.npu.NPUGraph
    ubatch_metadata: AscendUbatchMetadata
    outputs: Optional[Any] = None


class NPUCoreControlContextManager:

    def __init__(self, comm_aiv_core: int, comm_aic_core: int,
                 curren_stream: Any):
        """
        Context manager for controlling aiv/aic core num. 
        Upon entering the context, it sets the number of cores
        allocated for communication and computation to comm_core and
        total_cores - comm_core respectively. Upon exiting, it restores the
        allocation to use all available npu cores.

        Args:
            comm_aiv_core (int): The number of aiv cores to allocate for communication. 
                (The remainder will be used for computation.)
            comm_aic_core (int): The number of aic cores to allocate for communication. 
                (The remainder will be used for computation.)
            set_comm_core_limit (Callable[[int], None]): 
                A function that sets the number of aiv/aic for communication.
            set_compute_core_limit (Callable[[int], None]): 
                A function that sets the number of aiv/aic for computation.
        """

        props = torch.npu.get_device_limit(torch.npu.current_device())
        cube_core_num = props["cube_core_num"]
        vector_core_num = props["vector_core_num"]

        assert comm_aic_core < cube_core_num
        assert comm_aiv_core < vector_core_num
        self.total_cube_core = cube_core_num
        self.total_vector_core = vector_core_num
        self.comm_aic_core = comm_aic_core
        self.comm_aiv_core = comm_aiv_core
        self.comp_aic_core = self.total_cube_core - comm_aic_core
        self.comp_aiv_core = self.total_vector_core - comm_aiv_core
        self.current_stream = curren_stream

    def __enter__(self):
        torch.npu.set_stream_limit(self.current_stream,
                                   cube_num=self.comm_aic_core,
                                   vector_num=self.comm_aiv_core)

    def __exit__(self, exc_type, exc_value, traceback):
        torch.npu.reset_stream_limit(self.current_stream)


class AscendUBatchWrapper:

    def __init__(self, runnable: Callable, vllm_config: VllmConfig,
                 runtime_mode: CUDAGraphMode, device: torch.npu.device):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.comm_stream = torch.npu.Stream(device=device)
        # Two ubatch threads plus the main thread
        self.ready_barrier = threading.Barrier(3)

        self.cudagraphs: dict[int, AscendNPUGraphMetaData] = {}

        self.cudagraph_wrapper = None
        self.graph_pool = None
        if runtime_mode is not CUDAGraphMode.NONE:
            self.cudagraph_wrapper = CUDAGraphWrapper(
                runnable, vllm_config, runtime_mode=runtime_mode)
            self.graph_pool = current_platform.get_global_graph_pool()

        self.device = device

    def __getattr__(self, key: str):
        # allow accessing the attributes of the runnable.
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        raise AttributeError(f"Attribute {key} not exists in the runnable of "
                             f"cudagraph wrapper: {self.runnable}")

    def unwrap(self) -> Callable:
        # in case we need to access the original runnable.
        return self.runnable

    def _capture_ubatches(self, ubatch_metadata, model) -> torch.Tensor:
        """
        Capture a cudagraph for a microbatched run.

        The logic here is somewhat complicated because we need to make sure that
        each of the ubatch threads initialize the cuda context before we start
        the graph capture.

        The flow is as follows:
        1. The main thread starts up each ubatch thread. Each thread will 
        initialize its cuda context (torch.cuda.current_blas_handle())
        before going to sleep upon entering the ubatch_context.

        2. The main thread starts the graph capture and wakes up the first 
        ubatch thread.

        3. Each ubatch thread runs the model to completion and returns the 
        completed output tensors back to the main thread.

        4. The main thread stores the captured cudagraph along with its metadata
        and returns
        """

        @torch.inference_mode()
        def _capture_ubatch_thread(results, ubatch_metadata):
            torch.npu.set_device(self.device)
            ubatch_context = ubatch_metadata.context
            with torch.npu.stream(ubatch_context.compute_stream):
                _ = torch.npu.current_blas_handle()
            with torch.npu.stream(ubatch_context.comm_stream):
                _ = torch.npu.current_blas_handle()
            with ubatch_context:
                model_output = model(
                    input_ids=ubatch_metadata.input_ids,
                    positions=ubatch_metadata.positions,
                    intermediate_tensors=ubatch_metadata.intermediate_tensors,
                    inputs_embeds=ubatch_metadata.inputs_embeds,
                )

            results.append((ubatch_metadata.context.id, model_output))

        results: list[tuple[int, torch.Tensor]] = []
        compute_stream = ubatch_metadata[0].context.compute_stream
        num_tokens = ubatch_metadata[0].num_tokens + \
            ubatch_metadata[1].num_tokens

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
            self.ready_barrier.wait()  # Wait for both threads to be ready

            # Capture the cudagraph
            cudagraph_metadata = \
                AscendNPUGraphMetaData(
                            aclgraph=torch.npu.NPUGraph(),
                            ubatch_metadata=ubatch_metadata,
                        )
            if self.graph_pool is not None:
                set_graph_pool_id(self.graph_pool)
            else:
                set_graph_pool_id(current_platform.graph_pool_handle())
            with torch.npu.graph(cudagraph_metadata.aclgraph,
                                 stream=compute_stream,
                                 pool=self.graph_pool):
                ubatch_metadata[0].context.cpu_wait_event.set()
                for thread in ubatch_threads:
                    thread.join()
                sorted_results = [value for position, value in sorted(results)]
                result = torch.cat(sorted_results, dim=0)
                cudagraph_metadata.outputs = result
            self.cudagraphs[num_tokens] = cudagraph_metadata
        return cudagraph_metadata.outputs

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
                # 1. queue the comp kernel from other thread
                # 2. record the comp kernel of cur thread
                # 3. switch to comm stream
                # 4. wait for the npu to finish the cur thread's comp kernel

                # for thread 2, when it reach this point, it will also record and yield back to thread 1
                dbo_current_stream().synchronize()
                dbo_yield()

            results.append((ubatch_metadata.context.id, model_output))

        results: list[tuple[int, torch.Tensor]] = []

        # Ubatch threads will manually manage the forward context, so we
        # override it to None here so we can have it restored correctly
        # after both threads have finished
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
            self.ready_barrier.wait()  # Wait for both threads to be ready
            ubatch_metadata[0].context.cpu_wait_event.set()
            for thread in ubatch_threads:
                thread.join()
        sorted_results = [value for position, value in sorted(results)]
        # if enable sp, we should unpad the model output first
        if get_forward_context().sp_enabled and get_pp_group().is_last_rank:
            for i in range(2):
                sorted_results[i] = tensor_model_parallel_all_gather(
                    sorted_results[i], 0)

                pad_size = ubatch_metadata[i].context.forward_context.pad_size
                if pad_size > 0:
                    sorted_results[i] = sorted_results[i][:-pad_size, :]
        if not get_pp_group().is_last_rank:
            # Merge the IntermediateTensors in pp scenarios
            result = self._merge_intermediate_tensors(sorted_results)
        else:
            result = torch.cat(sorted_results, dim=0)
        # update outside forward context
        get_forward_context().dbo_enabled = True
        return result

    def _make_ubatch_metadata(
            self, ubatch_slices, attn_metadata, input_ids, positions,
            inputs_embeds, intermediate_tensors, compute_stream, dp_metadata,
            batch_descriptor,
            cudagraph_runtime_mode) -> list[AscendUbatchMetadata]:

        # Create one forward context per ubatch

        forward_contexts = []
        cur_forward_context = get_forward_context()
        # Construct forward context list based on the current forward context
        for i, ubatch_slice in enumerate(ubatch_slices):
            forward_contexts.append(
                create_ascend_forward_context(
                    cur_forward_context,
                    attn_metadata=attn_metadata[i]
                    if attn_metadata is not None else None,
                    vllm_config=self.vllm_config,
                    dp_metadata=dp_metadata,
                    ubatch_slices=ubatch_slices,
                    batch_descriptor=batch_descriptor,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    ubatch_num=i,
                    positions=positions,
                ))

        ubatch_ctxs = make_ubatch_contexts(
            num_micro_batches=len(ubatch_slices),
            comm_stream=self.comm_stream,
            compute_stream=compute_stream,
            forward_contexts=forward_contexts,
            ready_barrier=self.ready_barrier)

        ubatch_metadata: list[AscendUbatchMetadata] = []
        for i, ubatch_slice in enumerate(ubatch_slices):
            sliced_input_ids, sliced_positions, sliced_inputs_embeds, \
            sliced_intermediate_tensors = \
                self._slice_model_inputs(
                    ubatch_slice.token_slice, input_ids, positions,
                    inputs_embeds, intermediate_tensors)
            ubatch_metadata.append(
                AscendUbatchMetadata(
                    context=ubatch_ctxs[i],
                    input_ids=sliced_input_ids,
                    positions=sliced_positions,
                    inputs_embeds=sliced_inputs_embeds,
                    intermediate_tensors=sliced_intermediate_tensors,
                    num_tokens=ubatch_slice.token_slice.stop -
                    ubatch_slice.token_slice.start))

        return ubatch_metadata

    def _slice_model_inputs(self, tokens_slice: slice, input_ids, positions,
                            inputs_embeds, intermediate_tensors):
        sliced_input_ids = input_ids[tokens_slice]
        # if we are using mrope. Mrope adds an additional dimension to the
        # positions tensor
        if positions.ndim == 2:
            sliced_positions = positions[:, tokens_slice]
        else:
            sliced_positions = positions[tokens_slice]
        sliced_inputs_embeds = inputs_embeds[
            tokens_slice] if inputs_embeds else None
        # consider pp scenario
        if intermediate_tensors is not None:
            # if enable sp, dbo should not split intermediate tensors using token_slice
            # instead, it should calculate the tensor lens after reduce scatter
            if enable_sp():
                tp_size = get_tensor_model_parallel_world_size()
                start = (tokens_slice.start + tp_size - 1) // tp_size
                if start != 0:
                    stop = start + (tokens_slice.stop - tokens_slice.start +
                                    tp_size - 1) // tp_size
                else:
                    stop = (tokens_slice.stop + tp_size - 1) // tp_size
                tokens_slice = slice(start, stop)

            sliced_intermediate_tensors = intermediate_tensors[
                tokens_slice] if intermediate_tensors else None
        else:
            sliced_intermediate_tensors = None

        return (sliced_input_ids, sliced_positions, sliced_inputs_embeds,
                sliced_intermediate_tensors)

    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        ubatch_slices = forward_context.ubatch_slices
        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode

        # If there's no ubatching, just run the runnable object
        if ubatch_slices is None:

            # This is to account for the case where ubatching was aborted.
            # When we capture full graphs we only capture one graph per shape,
            # meaning that if we have a ubatched  cudagraph for the current
            # num_tokens, we don't have a non-ubatched one. Without this
            # check, the cudagraph wrapper will try to capture a cudagraph
            # for this shape during a normal run.
            if cudagraph_runtime_mode is CUDAGraphMode.FULL:
                assert batch_descriptor is not None
                if batch_descriptor.num_tokens in self.cudagraphs:
                    cudagraph_runtime_mode = CUDAGraphMode.NONE

            if cudagraph_runtime_mode in (CUDAGraphMode.NONE,
                                          CUDAGraphMode.PIECEWISE):
                return self.runnable(*args, **kwargs)
            else:
                assert self.cudagraph_wrapper is not None
                return self.cudagraph_wrapper(*args, **kwargs)

        attn_metadata = forward_context.attn_metadata
        num_tokens = (ubatch_slices[0].token_slice.stop -
                      ubatch_slices[0].token_slice.start) * 2
        input_ids = kwargs['input_ids']
        positions = kwargs['positions']
        intermediate_tensors = kwargs['intermediate_tensors']
        inputs_embeds = kwargs['inputs_embeds']
        compute_stream = torch.npu.current_stream()

        dp_metadata = forward_context.dp_metadata

        # We shouldn't be here unless we are running with multiple DP ranks
        # assert dp_metadata is not None

        if num_tokens not in self.cudagraphs \
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
                cudagraph_runtime_mode=CUDAGraphMode.NONE)
            return self._capture_ubatches(ubatch_metadata, self.model)
        elif num_tokens in self.cudagraphs \
            and cudagraph_runtime_mode is CUDAGraphMode.FULL:
            cudagraph_metadata = self.cudagraphs[num_tokens]
            cudagraph_metadata.aclgraph.replay()
            return cudagraph_metadata.outputs
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
                cudagraph_runtime_mode=CUDAGraphMode.NONE)
            return self._run_ubatches(ubatch_metadata, self.model)

    def _merge_intermediate_tensors(self, intermediate_tensor_list):

        assert len(intermediate_tensor_list) == 2
        result = {}
        for key in intermediate_tensor_list[0].tensors:
            result[key] = torch.cat([
                intermediate_tensor_list[0].tensors[key],
                intermediate_tensor_list[1].tensors[key]
            ],
                                    dim=0)

        res = IntermediateTensors(result)
        return res
