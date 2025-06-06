#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/vllm/worker/gpu_worker.py
#

import gc
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch_npu
from torch_npu.op_plugin.atb._atb_ops import _register_atb_extensions
from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment,
                              set_custom_all_reduce)
from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized
from vllm.logger import logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.utils import bind_kv_cache
from vllm.v1.worker.worker_base import WorkerBase

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.distributed.parallel_state import init_ascend_model_parallel
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.utils import try_register_lib
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class NPUWorker(WorkerBase):

    def __init__(
            self,
            vllm_config: VllmConfig,
            local_rank: int,
            rank: int,
            distributed_init_method: str,
            is_driver_worker: bool = False,
            # Additional parameters for compatibility with vllm
            **kwargs):
        """Initialize the worker for Ascend."""
        # register patch for vllm
        from vllm_ascend.utils import adapt_patch
        adapt_patch()
        # Register ops when worker init.
        from vllm_ascend import ops
        ops.register_dummy_fusion_op()
        _register_atb_extensions()
        # init ascend config
        init_ascend_config(vllm_config)

        super().__init__(vllm_config=vllm_config,
                         local_rank=local_rank,
                         rank=rank,
                         distributed_init_method=distributed_init_method,
                         is_driver_worker=is_driver_worker)
        # Try to import mindie_turbo to accelerate vLLM inference.
        try_register_lib(
            "mindie_turbo",
            "MindIE Turbo is installed. vLLM inference will be accelerated with MindIE Turbo."
        )
        if self.cache_config.cache_dtype == "auto":
            self.cache_dtype = self.model_config.dtype
        else:
            self.cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                self.cache_config.cache_dtype]

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        self.profiler = self._init_profiler()

    def sleep(self, level: int = 1) -> None:
        logger.error("Sleep mode is only supported on v0")

    def wake_up(self, tags: Optional[list[str]] = None) -> None:
        logger.error("Sleep mode is only supported on v0")

    def init_device(self):
        if self.device_config.device.type == "npu":
            self.device = torch.device(f"npu:{self.local_rank}")
            NPUPlatform.set_device(self.device)
            NPUPlatform.empty_cache()
            self.init_npu_memory = NPUPlatform.mem_get_info()[0]
        else:
            info = f"Not support device type: {self.device_config.device}"
            logger.error(info)
            raise RuntimeError(info)
        # Initialize the distributed environment.
        self._init_worker_distributed_environment()
        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Init ModelRunner here, so that we have access to self.device.
        self.model_runner = NPUModelRunner(self.vllm_config, self.device)

    def determine_available_memory(self) -> int:
        kv_caches: Dict[str, torch.Tensor] = {}
        kv_cache_spec = self.model_runner.get_kv_cache_spec()
        for layer_name, layer_spec in kv_cache_spec.items():
            if isinstance(layer_spec, FullAttentionSpec):
                # Use an empty tensor instead of `None`` to force Dynamo to pass
                # it by reference, rather by specializing on the value ``None``.
                npu_k_cache = torch.tensor([],
                                           dtype=layer_spec.dtype,
                                           device=self.device)
                npu_v_cache = torch.tensor([],
                                           dtype=layer_spec.dtype,
                                           device=self.device)
                kv_caches[layer_name] = (npu_k_cache, npu_v_cache)
            else:
                raise NotImplementedError

        runner_kv_caches: List[torch.Tensor] = []
        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            runner_kv_caches)

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

        gc.collect()
        # TODO: don`t need impl this func after empty_cache in
        # Worker.determine_num_available_blocks() unified`
        NPUPlatform.empty_cache()
        usable_memory_size = total_npu_memory * self.cache_config.gpu_memory_utilization - peak_memory
        npu_kv_cache_bytes = max(usable_memory_size, 0)
        logger.info(
            f"Available memory: {usable_memory_size}, total memory: {total_npu_memory}"
        )
        return int(npu_kv_cache_bytes)

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        output = self.model_runner.execute_model(scheduler_output)
        return output if self.is_driver_worker else None

    def load_model(self) -> None:
        self.model_runner.load_model()

    def compile_or_warm_up_model(self) -> None:
        warmup_sizes = self.vllm_config.compilation_config.compile_sizes.copy()
        if not self.model_config.enforce_eager:
            warmup_sizes = [
                x for x in warmup_sizes if x not in
                self.vllm_config.compilation_config.cudagraph_capture_sizes
            ]
        for size in sorted(warmup_sizes, reverse=True):
            logger.info("Compile and warming up model for size %d", size)
            self.model_runner._dummy_run(size)
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model()
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate NPU KV cache with the specified kv_cache_config."""
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def initialize_cache(self, kv_cache_configs: List[KVCacheConfig]) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""
        kv_cache_config = kv_cache_configs[self.rank]
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def profile(self, is_start: bool = True):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        if is_start:
            self.profiler.start()
        else:
            self.profiler.stop()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_runner.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def execute_dummy_batch(self) -> None:
        runner = self.model_runner
        num_tokens = 1
        if runner.dp_size > 1:
            max_num_tokens, with_prefill = runner._get_forward_metadata_across_dp(
                1, False)
        if envs_ascend.VLLM_ENABLE_MC2 or runner.torchair_graph_enabled:
            if not with_prefill:
                num_tokens = max_num_tokens
            num_tokens = runner.select_torchair_padded_batch_size(num_tokens)
        runner._dummy_run(num_tokens,
                          is_compile=False,
                          with_prefill=with_prefill)

    def _init_worker_distributed_environment(self) -> None:
        """Initialize the distributed environment."""
        parallel_config = self.vllm_config.parallel_config
        set_custom_all_reduce(
            not self.parallel_config.disable_custom_all_reduce)
        init_distributed_environment(self.parallel_config.world_size,
                                     self.rank, self.distributed_init_method,
                                     self.local_rank, "hccl")
        ensure_model_parallel_initialized(
            self.parallel_config.tensor_parallel_size,
            self.parallel_config.pipeline_parallel_size)
        init_ascend_model_parallel(
            parallel_config.expert_parallel_size,
            parallel_config.expert_tensor_parallel_size,
            parallel_config.world_size_across_dp,
        )
        ensure_kv_transfer_initialized(self.vllm_config)

    def _init_profiler(self):
        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        if envs.VLLM_TORCH_PROFILER_DIR:
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info("Profiling enabled. Traces will be saved to: %s",
                        torch_profiler_trace_dir)

            experimental_config = torch_npu.profiler._ExperimentalConfig(
                export_type=torch_npu.profiler.ExportType.Text,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                msprof_tx=False,
                aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
                l2_cache=False,
                op_attr=False,
                data_simplification=False,
                record_op_args=False,
                gc_detect_threshold=None,
            )

            return torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                with_stack=False,
                profile_memory=False,
                with_modules=False,
                experimental_config=experimental_config,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir))
        else:
            return None
