#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/vllm/worker/worker.py
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import gc
from typing import Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.distributed
from torch import nn
from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed import (ensure_kv_transfer_initialized,
                              ensure_model_parallel_initialized,
                              init_distributed_environment,
                              set_custom_all_reduce)
from vllm.logger import logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import (ExecuteModelRequest, IntermediateTensors,
                           SequenceGroupMetadata, SequenceGroupMetadataDelta)
from vllm.utils import GiB_bytes, bind_kv_cache
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.enc_dec_model_runner import EncoderDecoderModelRunner
from vllm.worker.model_runner_base import ModelRunnerBase
from vllm.worker.worker_base import (LocalOrDistributedWorkerBase, WorkerBase,
                                     WorkerInput)

from vllm_ascend.device_allocator.camem import CaMemAllocator
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.utils import try_register_lib
from vllm_ascend.worker.model_runner import NPUModelRunner
from vllm_ascend.worker.pooling_model_runner import NPUPoolingModelRunner


class NPUWorker(LocalOrDistributedWorkerBase):
    """A worker class that executes (a partition of) the model on a NPU.
    Each worker is associated with a single NPU. The worker is responsible for
    maintaining the KV cache and executing the model on the NPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
            self,
            vllm_config: VllmConfig,
            local_rank: int,
            rank: int,
            distributed_init_method: str,
            is_driver_worker: bool = False,
            model_runner_cls: Optional[Type[ModelRunnerBase]] = None) -> None:
        # TODO: Remove this line after fixing the hard-coding issue in VLLM later.
        from torch_npu.contrib import transfer_to_npu  # noqa: F401

        # Register ops and patch when worker init.
        from vllm_ascend import ops  # noqa: F401
        from vllm_ascend import patch  # noqa: F401

        WorkerBase.__init__(self, vllm_config=vllm_config)
        # Try to import mindie_turbo to accelerate vLLM inference.
        try_register_lib(
            "mindie_turbo",
            "MindIE Turbo is installed. vLLM inference will be accelerated with MindIE Turbo."
        )
        # distribute related config
        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        # Return hidden states from target model if the draft model is an
        # mlp_speculator
        speculative_config = self.speculative_config
        model_config = self.model_config
        speculative_args = {} if speculative_config is None \
            or (speculative_config.draft_model_config.hf_config.model_type ==
                model_config.hf_config.model_type) \
            or (speculative_config.draft_model_config.hf_config.model_type
                not in ["medusa", "mlp_speculator", "eagle", "deepseek_mtp"]) \
                    else {"return_hidden_states": True}

        ModelRunnerClass: Type[ModelRunnerBase] = NPUModelRunner
        if model_config.runner_type == "pooling":
            ModelRunnerClass = NPUPoolingModelRunner
        elif self.model_config.is_encoder_decoder:
            ModelRunnerClass = EncoderDecoderModelRunner
        self.model_runner: ModelRunnerBase = ModelRunnerClass(
            vllm_config=self.vllm_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=is_driver_worker,
            **speculative_args,
        )
        if model_runner_cls is not None:
            self.model_runner = model_runner_cls(self.model_runner)

        # Uninitialized cache engine. Will be initialized by
        # initialize_cache.
        self.cache_engine: List[CacheEngine]
        # Initialize gpu_cache as embedding models don't initialize kv_caches
        self.gpu_cache: Optional[List[List[torch.Tensor]]] = None
        self._seq_group_metadata_cache: Dict[str, SequenceGroupMetadata] = {}

        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        if envs.VLLM_TORCH_PROFILER_DIR:
            # lazy import so that torch_npu is not required for normal use.
            import torch_npu
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info("Profiling enabled. Traces will be saved to: %s",
                        torch_profiler_trace_dir)

            experimental_config = torch_npu.profiler._ExperimentalConfig(
                export_type=torch_npu.profiler.ExportType.Text,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
                msprof_tx=False,
                aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
                l2_cache=False,
                op_attr=False,
                data_simplification=False,
                record_op_args=False,
                gc_detect_threshold=None,
            )

            self.profiler = torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                with_stack=True,
                profile_memory=True,
                with_modules=True,
                experimental_config=experimental_config,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir))
        else:
            self.profiler = None

    def sleep(self, level: int = 1) -> None:
        NPUPlatform.set_device(self.device)
        free_bytes_before_sleep = NPUPlatform.mem_get_info()[0]
        allocator = CaMemAllocator.get_instance()
        allocator.sleep(offload_tags=("weights", ) if level == 1 else tuple())
        free_bytes_after_sleep, total = NPUPlatform.mem_get_info()
        freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        used_bytes = total - free_bytes_after_sleep
        assert freed_bytes >= 0, "Memory usage increased after sleeping."
        logger.info(
            "Sleep mode freed %.2f GiB memory, "
            "%.2f GiB memory is still in use.", freed_bytes / GiB_bytes,
            used_bytes / GiB_bytes)

    def wake_up(self) -> None:
        allocator = CaMemAllocator.get_instance()
        allocator.wake_up()

    def init_device(self) -> None:
        if self.device_config.device.type == "npu":
            self.device = torch.device(f"npu:{self.local_rank}")
            NPUPlatform.set_device(self.device)
            NPUPlatform.empty_cache()
            self.init_npu_memory = NPUPlatform.mem_get_info()[0]
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        self._init_worker_distributed_environment(self.vllm_config, self.rank,
                                                  self.distributed_init_method,
                                                  self.local_rank)
        # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        if self.vllm_config.model_config.enable_sleep_mode:
            allocator = CaMemAllocator.get_instance()
            assert allocator.get_current_usage() == 0, (
                "Sleep mode can only be "
                "used for one instance per process.")
            context = allocator.use_memory_pool(tag="weights")
        else:
            from contextlib import nullcontext
            context = nullcontext()  # type: ignore
        with context:
            self.model_runner.load_model()

    def start_profile(self):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        self.profiler.start()

    def stop_profile(self):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        self.profiler.stop()

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        self.model_runner.save_sharded_state(
            path,
            pattern=pattern,
            max_size=max_size,
        )

    def save_tensorized_model(
        self,
        tensorizer_config: TensorizerConfig,
    ) -> None:
        self.model_runner.save_tensorized_model(
            tensorizer_config=tensorizer_config, )

    @NPUPlatform.inference_mode()
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Profiles the peak memory usage of the model to determine how many
        KV blocks may be allocated without OOMs.
        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of NPU and CPU blocks
        that can be allocated with the remaining free memory.
        .. tip::
            You may limit the usage of NPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        NPUPlatform.empty_cache()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        self.model_runner.profile_run()

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        free_npu_memory, total_npu_memory = NPUPlatform.mem_get_info()
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        peak_memory = self.init_npu_memory - free_npu_memory
        assert peak_memory > 0, (
            "Error in memory profiling. "
            f"Initial free memory {self.init_npu_memory}, current free memory"
            f" {free_npu_memory}. This happens when the NPU memory was "
            "not properly cleaned up before initializing the vLLM instance.")

        cache_block_size = self.get_cache_block_size_bytes()
        num_npu_blocks = int(
            (total_npu_memory * self.cache_config.gpu_memory_utilization -
             peak_memory) // cache_block_size)
        num_cpu_blocks = int(self.cache_config.swap_space_bytes //
                             cache_block_size)
        num_npu_blocks = max(num_npu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        gc.collect()
        # TODO: don`t need impl this func after empty_cache in
        # Worker.determine_num_available_blocks() unified`
        NPUPlatform.empty_cache()
        return num_npu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Allocate NPU and CPU KV cache with the specified number of blocks.
        """
        raise_if_cache_size_invalid(num_gpu_blocks,
                                    self.cache_config.block_size,
                                    self.cache_config.is_attention_free,
                                    self.model_config.max_model_len)

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks
        if self.vllm_config.model_config.enable_sleep_mode:
            allocator = CaMemAllocator.get_instance()
            context = allocator.use_memory_pool(tag="kv_cache")
        else:
            from contextlib import nullcontext
            context = nullcontext()  # type: ignore
        with context:
            self._init_cache_engine()
        self._warm_up_model()

    def _init_cache_engine(self):
        assert self.cache_config.num_gpu_blocks is not None
        self.cache_engine = [
            CacheEngine(self.cache_config, self.model_config,
                        self.parallel_config, self.device_config)
            for _ in range(self.parallel_config.pipeline_parallel_size)
        ]
        import torch_npu
        for ve in range(self.parallel_config.pipeline_parallel_size):
            num_layers = len(self.cache_engine[ve].gpu_cache)
            for i in range(num_layers):
                torch_npu.npu_format_cast(self.cache_engine[ve].gpu_cache[i],
                                          2)
        self.gpu_cache = [
            self.cache_engine[ve].gpu_cache
            for ve in range(self.parallel_config.pipeline_parallel_size)
        ]
        bind_kv_cache(self.compilation_config.static_forward_context,
                      self.gpu_cache)

    def _warm_up_model(self) -> None:
        # model capture is not supported, thus we just set seed here.
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    @property
    def do_metadata_broadcast(self) -> bool:
        return self.parallel_config.tensor_parallel_size > 1

    @property
    def kv_cache(self) -> Optional[List[List[torch.Tensor]]]:
        return self.gpu_cache

    @torch.inference_mode()
    def prepare_worker_input(
            self, execute_model_req: ExecuteModelRequest) -> WorkerInput:
        virtual_engine = execute_model_req.virtual_engine
        num_steps = execute_model_req.num_steps
        num_seq_groups = len(execute_model_req.seq_group_metadata_list)
        # `blocks_to_swap_in` and `blocks_to_swap_out` are cpu tensors.
        # they contain parameters to launch cudamemcpyasync.
        blocks_to_swap_in = torch.tensor(execute_model_req.blocks_to_swap_in,
                                         device="cpu",
                                         dtype=torch.int64).view(-1, 2)
        blocks_to_swap_out = torch.tensor(execute_model_req.blocks_to_swap_out,
                                          device="cpu",
                                          dtype=torch.int64).view(-1, 2)
        # `blocks_to_copy` is a gpu tensor. The src and tgt of
        # blocks to copy are in the same device, and `blocks_to_copy`
        # can be used directly within cuda kernels.
        blocks_to_copy = torch.tensor(execute_model_req.blocks_to_copy,
                                      device=self.device,
                                      dtype=torch.int64).view(-1, 2)

        return WorkerInput(
            num_seq_groups=num_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            virtual_engine=virtual_engine,
            num_steps=num_steps,
        )

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    @torch.inference_mode()
    def execute_worker(self, worker_input: WorkerInput) -> None:
        virtual_engine = worker_input.virtual_engine
        # Issue cache operations.
        if (worker_input.blocks_to_swap_in is not None
                and worker_input.blocks_to_swap_in.numel() > 0):
            self.cache_engine[virtual_engine].swap_in(
                worker_input.blocks_to_swap_in)
        if (worker_input.blocks_to_swap_out is not None
                and worker_input.blocks_to_swap_out.numel() > 0):
            self.cache_engine[virtual_engine].swap_out(
                worker_input.blocks_to_swap_out)
        if (worker_input.blocks_to_copy is not None
                and worker_input.blocks_to_copy.numel() > 0):
            self.cache_engine[virtual_engine].copy(worker_input.blocks_to_copy)

    def _get_cached_seq_group_metadata(
            self,
            seq_group_metadata_list: List[Union[SequenceGroupMetadata,
                                                SequenceGroupMetadataDelta]],
            finished_request_ids: List[str]) -> List[SequenceGroupMetadata]:
        """Return a list of cached Sequence Group Metadata after updating its
        state.

        It is used because scheduler only sends delta to workers to reduce
        the data payload size. The function also cleans up cache based on
        a given `finished_request_ids`.
        """
        new_seq_group_metadata_list = []
        for metadata_or_delta in seq_group_metadata_list:
            request_id = metadata_or_delta.request_id
            if request_id not in self._seq_group_metadata_cache:
                # The first prefill.
                assert isinstance(metadata_or_delta, SequenceGroupMetadata)
                self._seq_group_metadata_cache[request_id] = metadata_or_delta
            else:
                # The first prefill is already cached.
                if isinstance(metadata_or_delta, SequenceGroupMetadataDelta):
                    self._seq_group_metadata_cache[request_id].apply_delta(
                        metadata_or_delta)
                else:
                    # If metadata snapshot is sent again, it is
                    # preempted. Reset the cache because we need to start
                    # from scratch.
                    assert isinstance(metadata_or_delta, SequenceGroupMetadata)
                    self._seq_group_metadata_cache[
                        request_id] = metadata_or_delta

            new_seq_group_metadata_list.append(
                self._seq_group_metadata_cache[request_id])

        # Clean up finished ids
        for finished_id in finished_request_ids:
            del self._seq_group_metadata_cache[finished_id]

        return new_seq_group_metadata_list

    def _execute_model_spmd(
        self,
        execute_model_req: ExecuteModelRequest,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Optional[List[SamplerOutput]]:
        if execute_model_req is not None:
            new_seq_group_metadata_list = self._get_cached_seq_group_metadata(
                execute_model_req.seq_group_metadata_list,
                execute_model_req.finished_requests_ids)

            execute_model_req.seq_group_metadata_list = (
                new_seq_group_metadata_list)
        output = super()._execute_model_spmd(execute_model_req,
                                             intermediate_tensors)
        return output

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.model_runner.list_loras()

    def add_prompt_adapter(
            self, prompt_adapter_request: PromptAdapterRequest) -> bool:
        raise NotImplementedError(
            "Prompt Adapter is not implemented for NPU backend currently.")

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        raise NotImplementedError(
            "Prompt Adapter is not implemented for NPU backend currently.")

    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        raise NotImplementedError(
            "Prompt Adapter is not implemented for NPU backend currently.")

    def list_prompt_adapters(self) -> Set[int]:
        raise NotImplementedError(
            "Prompt Adapter is not implemented for NPU backend currently.")

    @property
    def max_model_len(self) -> int:
        return self.model_config.max_model_len

    @property
    def vocab_size(self) -> int:
        return self.model_runner.vocab_size

    def get_cache_block_size_bytes(self) -> int:
        """Get the size of the KV cache block size in bytes.
        """
        return CacheEngine.get_cache_block_size(self.cache_config,
                                                self.model_config,
                                                self.parallel_config)

    def _init_worker_distributed_environment(
            self,
            vllm_config: VllmConfig,
            rank: int,
            distributed_init_method: Optional[str] = None,
            local_rank: int = -1,
            backend: str = "hccl") -> None:
        """Initialize the distributed environment."""
        parallel_config = self.parallel_config
        set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)
        init_distributed_environment(parallel_config.world_size, rank,
                                     distributed_init_method, local_rank,
                                     backend)
        ensure_model_parallel_initialized(
            parallel_config.tensor_parallel_size,
            parallel_config.pipeline_parallel_size)
        ensure_kv_transfer_initialized(vllm_config)


def raise_if_cache_size_invalid(num_gpu_blocks, block_size, is_attention_free,
                                max_model_len) -> None:
    if is_attention_free and num_gpu_blocks != 0:
        raise ValueError("No memory should be allocated for the cache blocks "
                         f"for an attention-free model, but {num_gpu_blocks}"
                         "blocks are allocated.")
    if not is_attention_free and num_gpu_blocks <= 0:
        raise ValueError("No available memory for the cache blocks. "
                         "Try increasing `gpu_memory_utilization` when "
                         "initializing the engine.")
    max_seq_len = block_size * num_gpu_blocks
    if not is_attention_free and max_model_len > max_seq_len:
        raise ValueError(
            f"The model's max seq len ({max_model_len}) "
            "is larger than the maximum number of tokens that can be "
            f"stored in KV cache ({max_seq_len}). Try increasing "
            "`gpu_memory_utilization` or decreasing `max_model_len` when "
            "initializing the engine.")
