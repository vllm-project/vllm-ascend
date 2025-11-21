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
import os
from typing import Dict, List, Optional

import ray
import torch
import torch.distributed as dist
import torch.nn as nn
import torch_npu
import vllm.envs as envs
from omni.adaptors.vllm.platform import NPUPlatform
from omni.adaptors.vllm.utils import (check_block_num_cache_exist,
                                      check_torchair_cache_exists, clear_var,
                                      delete_torchair_cache_file,
                                      read_block_num_from_file,
                                      write_block_num_to_file)
# from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from omni.adaptors.vllm.worker.npu_model_runner import NPUModelRunner
from omni.models.common.config.model_config import model_extra_config
from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed import (ensure_model_parallel_initialized, get_dp_group,
                              get_world_group, init_distributed_environment,
                              set_custom_all_reduce)
from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized
from vllm.logger import logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.platforms import current_platform
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, GiB_bytes
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerBase

__origin_get_device_properties__ = torch.npu.get_device_properties


class NPUDeviceProperties:

    def __init__(self, device):
        self.properties = __origin_get_device_properties__(device)
        self.multi_processor_count = self.properties.multi_processor_count \
            if hasattr(self.properties, 'multi_processor_count') else 0


def get_device_properties(device):
    return NPUDeviceProperties(device)


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

        if envs.VLLM_ENABLE_V1_MULTIPROCESSING:
            if envs.VLLM_USE_RAY_SPMD_WORKER:
                dp_size = vllm_config.parallel_config.data_parallel_size
                node_count = len(ray.nodes())
                dp_rank_size = int(dp_size / node_count)
                if dp_rank_size >= 1:
                    local_dp_rank = vllm_config.parallel_config.data_parallel_rank_local
                    vllm_config.parallel_config.data_parallel_rank_local = local_dp_rank % dp_rank_size
            super().__init__(
                vllm_config=vllm_config,
                local_rank=local_rank +
                vllm_config.parallel_config.data_parallel_rank_local *
                vllm_config.parallel_config.tensor_parallel_size -
                int(os.environ.get("SERVER_OFFSET", 0)),
                rank=rank,
                distributed_init_method=distributed_init_method,
                is_driver_worker=is_driver_worker)
        else:
            super().__init__(vllm_config=vllm_config,
                             local_rank=local_rank,
                             rank=rank,
                             distributed_init_method=distributed_init_method,
                             is_driver_worker=is_driver_worker)
        # Try to import mindie_turbo to accelerate vLLM inference.
        # try_register_lib(
        #     "mindie_turbo",
        #     "MindIE Turbo is installed. vLLM inference will be accelerated with MindIE Turbo."
        # )
        if self.cache_config.cache_dtype == "auto":
            self.cache_dtype = self.model_config.dtype
        else:
            self.cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                self.cache_config.cache_dtype]

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        torch.cuda.get_device_properties = get_device_properties

        vllm_config.model_config.disable_cascade_attn = True
        self._init_graph_options()

    def sleep(self, level: int = 1) -> None:
        if not NPUPlatform.is_sleep_mode_available():
            logger.error("Sleep mode is only supported on v0")
            return

        from omni.adaptors.vllm.npu_mem_pool import NpuMemAllocator
        free_bytes_before_sleep = NPUPlatform.mem_get_info()[0]
        allocator = NpuMemAllocator.get_instance()
        allocator.sleep(offload_tags=("weights", ) if level == 1 else tuple())
        free_bytes_after_sleep, total = NPUPlatform.mem_get_info()
        freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        used_bytes = total - free_bytes_after_sleep
        assert freed_bytes >= 0, "Memory usage increased after sleeping."
        logger.info(
            "Sleep mode freed %.2f GiB memory, "
            "%.2f GiB memory is still in use.", freed_bytes / GiB_bytes,
            used_bytes / GiB_bytes)

    def wake_up(self, tags: Optional[list[str]] = None) -> None:
        if not NPUPlatform.is_sleep_mode_available():
            logger.error("Sleep mode is only supported on v0")
            return

        from omni.adaptors.vllm.npu_mem_pool import NpuMemAllocator
        allocator = NpuMemAllocator.get_instance()
        allocator.wake_up(tags=tags)

    def init_device(self):
        if self.device_config.device.type == current_platform.device_type:
            if int(os.getenv("NO_NPU_MOCK", "0")):
                self.device = torch.device(f"cpu")
            else:
                self.device = torch.device(
                    f"{current_platform.device_type}:{self.local_rank}")
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
        self.profiler = self._init_profiler()

    def _init_graph_options(self):
        from vllm.config import CompilationLevel
        from vllm.utils import supports_dynamo

        self.enable_torchair_graph_mode = (
            self.vllm_config.npu_compilation_config.level
            > CompilationLevel.NO_COMPILATION and supports_dynamo())
        self.use_cached_npu_graph = self.vllm_config.npu_compilation_config.use_ge_graph_cached

    def page_size_bytes(self) -> int:
        # For MLA we only store a single latent vector
        coef = 1 if self.vllm_config.model_config.use_mla else 2
        config = self.vllm_config.model_config.hf_config
        block_size = self.vllm_config.cache_config.block_size
        kv_lora_rank = config.kv_lora_rank
        qk_rope_head_dim = config.qk_rope_head_dim
        return coef * config.num_hidden_layers * block_size * (
            kv_lora_rank + qk_rope_head_dim) * 2

    def determine_available_memory(self) -> int:
        if int(os.getenv("NO_NPU_MOCK", "0")):
            return int(100000000)

        cur_npu_kv_cache_bytes = self._compute_kv_cache_bytes()
        if not self.enable_torchair_graph_mode:
            clear_var()
            # Only For Prefill Stage
            if model_extra_config.operator_opt_config.use_omni_placement:
                self.model_runner.planner.start_dynamic_optimize_expert_load_balance(
                )
            return cur_npu_kv_cache_bytes

        last_use_kv_cache_bytes = cur_npu_kv_cache_bytes

        if self.use_cached_npu_graph:
            dp_size = get_world_group().world_size
            if check_torchair_cache_exists() and check_block_num_cache_exist():
                logger.info("Currently use graph cache")
                npu_kv_cache_bytes = read_block_num_from_file(
                    torch.distributed.get_rank())
                if npu_kv_cache_bytes == -1:
                    raise RuntimeError(
                        f"Read npu_kv_cache_bytes: {npu_kv_cache_bytes} error, "
                        f"please make sure the block num cache file is correct"
                    )
                old_kv_cache_bytes = torch.tensor([npu_kv_cache_bytes],
                                                  device="cpu")
                all_kv_cache_bytes = [
                    torch.tensor([0],
                                 dtype=old_kv_cache_bytes.dtype,
                                 device="cpu") for _ in range(dp_size)
                ]
                dist.all_gather(all_kv_cache_bytes,
                                old_kv_cache_bytes,
                                group=get_world_group().cpu_group)
                for kv_cache_bytes in all_kv_cache_bytes:
                    if kv_cache_bytes != old_kv_cache_bytes:
                        raise RuntimeError(
                            f"The block num data of some ranks has been modified, origin: {old_kv_cache_bytes}, now: {kv_cache_bytes}"
                        )

                clear_var(old_kv_cache_bytes, all_kv_cache_bytes)
                last_use_kv_cache_bytes = npu_kv_cache_bytes
            else:
                logger.warning("Currently no graph cache available")
                delete_torchair_cache_file()
                cur_npu_kv_cache_bytes = torch.tensor([cur_npu_kv_cache_bytes],
                                                      device="cpu")
                all_kv_cache_bytes = [
                    torch.tensor([0],
                                 dtype=cur_npu_kv_cache_bytes.dtype,
                                 device="cpu") for _ in range(dp_size)
                ]
                dist.all_gather(all_kv_cache_bytes,
                                cur_npu_kv_cache_bytes,
                                group=get_world_group().cpu_group)
                kv_cache_bytes = int(min(all_kv_cache_bytes[:]))
                write_block_num_to_file(torch.distributed.get_rank(),
                                        kv_cache_bytes)
                clear_var(cur_npu_kv_cache_bytes, all_kv_cache_bytes)
                last_use_kv_cache_bytes = kv_cache_bytes

        return last_use_kv_cache_bytes

    def _compute_kv_cache_bytes(self):
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        gc.collect()
        NPUPlatform.empty_cache()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        self.model_runner.profile_run()
        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        is_pd_seperate_d = self.vllm_config.kv_transfer_config is not None and self.vllm_config.kv_transfer_config.kv_role == "kv_consumer"
        if is_pd_seperate_d and os.getenv("ASCEND_PLATFORM", "A3") == "A2":
            NPUPlatform.empty_cache()

        free_npu_memory, total_npu_memory = NPUPlatform.mem_get_info()
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        peak_memory = self.init_npu_memory - free_npu_memory
        if peak_memory <= 0:
            raise RuntimeError(
                "Error in memory profiling. "
                f"Initial free memory {self.init_npu_memory}, current free memory"
                f" {free_npu_memory}. This happens when the NPU memory was "
                "not properly cleaned up before initializing the vLLM instance."
            )
        usable_memory_size = total_npu_memory * self.cache_config.gpu_memory_utilization - peak_memory
        npu_kv_cache_bytes = max(usable_memory_size, 0)
        logger.info(
            f"Available memory: {usable_memory_size}, total memory: {total_npu_memory}"
        )
        return int(npu_kv_cache_bytes)

    def load_kv_cache(self, info_load_reqs) -> List[int]:
        result = self.model_runner.load_kv_cache(info_load_reqs)
        return result

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        if envs.VLLM_TORCH_PROFILER_DIR:
            if not self.profile_already_start and scheduler_output.total_num_scheduled_tokens >= self.profiler_token_threshold:
                self.profiler.start()
                self.profile_already_start = True
                self.profile_step = 0

        output = self.model_runner.execute_model(scheduler_output)
        if envs.VLLM_TORCH_PROFILER_DIR:
            if self.profile_already_start and not self.profile_finished:
                self.profile_step += 1
            if not self.profile_finished and self.profile_step > self.profiler_stop_step:
                self.profiler.stop()
                self.profile_finished = True
        return output if self.is_driver_worker else None

    def load_model(self) -> None:
        if NPUPlatform.is_sleep_mode_available():
            allocator = NpuMemAllocator.get_instance()
            assert allocator.get_current_usage() == 0, (
                "Sleep mode can only be "
                "used for one instance per process.")
            context = allocator.use_memory_pool(tag="weights")
        else:
            from contextlib import nullcontext
            context = nullcontext()
        with context:
            self.model_runner.load_model()

    def compile_or_warm_up_model(self) -> None:
        if self.enable_torchair_graph_mode:
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
        if NPUPlatform.is_sleep_mode_available():
            allocator = NpuMemAllocator.get_instance()
            context = allocator.use_memory_pool(tag="kv_cache")
        else:
            from contextlib import nullcontext
            context = nullcontext()
        with context:
            if model_extra_config.operator_opt_config.use_omni_cache:
                self.model_runner.initialize_omni_kv_cache(kv_cache_config)
            else:
                self.model_runner.initialize_kv_cache(kv_cache_config)

    def initialize_cache(self, kv_cache_configs: List[KVCacheConfig]) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""
        self.initialize_from_config(kv_cache_configs)

    def profile(self, is_start: bool = True):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        if is_start:
            self.profiler.start()
        else:
            self.profiler.stop()

    def execute_dummy_batch(self) -> None:
        self.model_runner._dummy_run(1)
        if model_extra_config.operator_opt_config.use_omni_placement:
            self.model_runner.planner.place_experts()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_runner.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def _init_worker_distributed_environment(self) -> None:
        """Initialize the distributed environment."""
        additional_config = self.vllm_config.additional_config
        parallel_config = self.vllm_config.parallel_config
        set_custom_all_reduce(
            not self.parallel_config.disable_custom_all_reduce)
        init_distributed_environment(self.parallel_config.world_size,
                                     self.rank, self.distributed_init_method,
                                     self.local_rank, "hccl")
        ensure_model_parallel_initialized(
            self.parallel_config.tensor_parallel_size,
            self.parallel_config.pipeline_parallel_size)
        ensure_kv_transfer_initialized(self.vllm_config)

    def _init_profiler(self):
        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        # PROFILER_TOKEN_THRESHOLD=1
        # PROFILER_STOP_STEP=5
        self.profile_already_start = True
        self.profile_step = 0
        self.profile_finished = True
        if envs.VLLM_TORCH_PROFILER_DIR:
            self.profiler_token_threshold = int(
                os.environ.get('PROFILER_TOKEN_THRESHOLD', "1"))
            self.profiler_stop_step = int(
                os.environ.get('PROFILER_STOP_STEP', "5"))
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
