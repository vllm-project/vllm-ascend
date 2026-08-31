#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#
import gc

import psutil
import torch
import torch_npu
from vllm.logger import logger
from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.mem_utils import MemorySnapshot, memory_profiling
from vllm.utils.torch_utils import set_random_seed  # noqa: E402
from vllm.v1.core.kv_cache_utils import get_kv_cache_groups
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec, UniformTypeKVCacheSpecs

from vllm_ascend._310p.model_runner_310p import NPUModelRunner310
from vllm_ascend.utils import is_rc_device, vllm_version_is
from vllm_ascend.worker.worker import NPUWorker, init_workspace_manager


class NPUWorker310(NPUWorker):
    def _create_model_runner(self):
        if self.use_v2_model_runner:
            from vllm_ascend._310p.worker.v2.model_runner import NPUModelRunner310V2

            model_runner = NPUModelRunner310V2(self.vllm_config, self.device)
            logger.info_once("Using NPUWorker310 and NPUModelRunner310V2.")
            return model_runner

        model_runner = NPUModelRunner310(self.vllm_config, self.device)
        logger.info_once("Using NPUWorker310 and NPUModelRunner310.")
        return model_runner

    def init_device(self):
        self.device = self._init_device()
        torch_npu.npu.set_compile_mode(jit_compile=False)

        init_workspace_manager(self.device, num_ubatches=1)
        self.model_runner = self._create_model_runner()

    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        from vllm_ascend._310p.sharded_state_loader_310p import ShardedStateLoader310

        ShardedStateLoader310.save_model(
            self.model_runner.model,
            path,
            pattern=pattern,
            max_size=max_size,
        )

        ShardedStateLoader310.generate_quant_description(
            self.model_runner.model,
            path,
            self.vllm_config.quant_config,
        )

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Profiles the peak memory usage of the model to determine how much
        memory can be used for KV cache without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculates the free memory that can be used for KV cache in
        bytes.
        """
        GiB = lambda b: b / GiB_bytes
        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        with memory_profiling(
            self.init_snapshot,
            weights_memory=int(self.model_runner.model_memory_usage),
        ) as profile_result:
            self.model_runner.profile_run()
            free_memory, total_memory = torch.npu.mem_get_info()
            # The host memory or device memory for RC devices refers to the available portion of memory
            # which cannot be obtained via torch.npu.mem_get_info()
            if is_rc_device():
                free_memory = psutil.virtual_memory().available
            torch_memory = torch.npu.memory_reserved()
            non_torch_memory_before_empty_cache = total_memory - free_memory - torch_memory

        self.non_torch_memory = profile_result.non_torch_increase
        self.peak_activation_memory = profile_result.torch_peak_increase
        non_torch_memory_cleared_by_empty_cache = non_torch_memory_before_empty_cache - self.non_torch_memory

        free_gpu_memory = profile_result.after_profile.free_memory
        assert self.init_snapshot.free_memory > free_gpu_memory, (
            "Error in memory profiling. "
            f"Initial free memory {GiB(self.init_snapshot.free_memory)} GiB, "
            f"current free memory {GiB(free_gpu_memory)} GiB. "
            "This happens when other processes sharing the same container "
            "release GPU memory while vLLM is profiling during initialization. "
            "To fix this, ensure consistent GPU memory allocation or "
            "isolate vLLM in its own container."
        )

        # Divide the available memory by 2, to reserved more memory for other operators workspace and other cache
        # This could avoid OOM with default gpu_memory_utilization
        # The 310P RC device shares the host memory and device memory.
        # Therefore, the space available for allocating KV cache and Mamba cache needs to be calculated
        # based on the already occupied space of the system memory.

        if is_rc_device():
            vm = psutil.virtual_memory()
            self.available_kv_cache_memory_bytes = (self.requested_memory - (vm.total - vm.available)) // 2
        else:
            self.available_kv_cache_memory_bytes = (
                self.requested_memory - profile_result.non_kv_cache_memory - non_torch_memory_cleared_by_empty_cache
            ) // 2

        self.available_kv_cache_memory_bytes = self._scale_kv_cache_memory_for_multi_group(
            self.available_kv_cache_memory_bytes,
        )

        logger.debug(profile_result)
        logger.info_once(
            "Available KV cache memory: %.2f GiB (halved for workspace)",
            GiB(self.available_kv_cache_memory_bytes),
            scope="local",
        )
        return int(self.available_kv_cache_memory_bytes)

    def _scale_kv_cache_memory_for_multi_group(self, available_memory: int) -> int:
        """Scale the KV cache budget for vllm main's multi-group layout.

        vLLM #51718 derives num_blocks from the largest KV cache group's
        bytes-per-block, but the 310P runner keeps per-layer contiguous buffers
        for every group. Per-layer sizing then totals num_blocks * (sum of ALL
        groups' pages), which exceeds available memory whenever more than one
        group is non-trivial. Scale the advertised budget by bytes_per_block /
        sum(pages) so the engine derives a num_blocks (and block pool) small
        enough for the per-layer buffers to fit.
        """
        if vllm_version_is("0.27.1"):
            return available_memory
        kv_cache_spec = self.get_kv_cache_spec()
        if not isinstance(kv_cache_spec, dict):
            return available_memory
        kv_cache_groups = get_kv_cache_groups(self.vllm_config, kv_cache_spec)
        if not kv_cache_groups:
            return available_memory
        # vLLM #51718 removed the DSV4-specific packed planner. Ascend restores
        # that shared-tuple layout in patch_kv_cache_utils, so DSV4 already fits
        # all groups in one physical budget and must not take the generic
        # per-layer multi-group scale below.
        for group in kv_cache_groups:
            group_spec = group.kv_cache_spec
            specs = (
                group_spec.kv_cache_specs.values() if isinstance(group_spec, UniformTypeKVCacheSpecs) else (group_spec,)
            )
            if any(getattr(spec, "model_version", None) == "deepseek_v4" for spec in specs):
                return available_memory

        # vLLM #51718 overlays KV cache groups in one standardized backing
        # allocation. Do not shrink the planner budget if a future 310P runner
        # can consume the compact Attention/Mamba shared layout directly.
        per_layer_specs = []
        for group in kv_cache_groups:
            group_spec = group.kv_cache_spec
            if isinstance(group_spec, UniformTypeKVCacheSpecs):
                per_layer_specs.extend(group_spec.kv_cache_specs.values())
            else:
                per_layer_specs.append(group_spec)
        has_attention = any(isinstance(spec, AttentionSpec) for spec in per_layer_specs)
        has_mamba = any(isinstance(spec, MambaSpec) for spec in per_layer_specs)
        model_runner = getattr(self, "model_runner", None)
        layout = self.vllm_config.cache_config.get_resolved_kv_cache_layout()
        if (
            has_attention
            and has_mamba
            and layout.is_layer_compact
            and layout.is_block_compact
            and self.vllm_config.kv_transfer_config is None
            and getattr(model_runner, "supports_standardized_shared_kv_backing", False)
            and not getattr(model_runner, "use_sparse", False)
            and not getattr(model_runner, "use_compress", False)
        ):
            return available_memory

        bytes_per_block = 0
        sum_pages = 0
        for group in kv_cache_groups:
            group_pages = 0
            for layer_name in group.layer_names:
                group_spec = group.kv_cache_spec
                if isinstance(group_spec, UniformTypeKVCacheSpecs):
                    layer_spec = group_spec.kv_cache_specs[layer_name]
                else:
                    layer_spec = group_spec
                group_pages += layer_spec.page_size_bytes
                sum_pages += layer_spec.page_size_bytes
            bytes_per_block = max(bytes_per_block, group_pages)
        if bytes_per_block > 0 and sum_pages > bytes_per_block:
            scale = bytes_per_block / sum_pages
            logger.info(
                "310P per-layer KV layout scales the multi-group budget by %.4f "
                "(%d bytes/block over %d total page bytes) so per-layer "
                "buffers fit within device memory.",
                scale,
                bytes_per_block,
                sum_pages,
            )
            return int(available_memory * scale)
        return available_memory

    def _warm_up_atb(self):
        # 310p device do not support torch_npu._npu_matmul_add_fp32 atb ops
        logger.info_once("Skip warm-up atb ops for 310P device.")

    def _init_device(self):
        device = torch.device(f"npu:{self.local_rank}")
        torch.npu.set_device(device)

        # This lazy import avoids torch_npu re-initialization in patch
        # Note that this should be imported after torch.npu.set_device
        # to avoid repeated set_device in extra processes

        gc.collect()
        torch.npu.empty_cache()

        # take current memory snapshot
        self.init_snapshot = MemorySnapshot(device=device)
        self.requested_memory = self.init_snapshot.total_memory * self.cache_config.gpu_memory_utilization
        if is_rc_device():
            self.init_snapshot.free_memory = psutil.virtual_memory().available
            logger.info_once("Root Complex (RC) mode: host and device memory are shared.")
        if self.init_snapshot.free_memory < self.requested_memory:
            GiB = lambda b: round(b / GiB_bytes, 2)
            raise ValueError(
                f"Free memory on device "
                f"({GiB(self.init_snapshot.free_memory)}/"
                f"{GiB(self.init_snapshot.total_memory)} GiB) on startup "
                f"is less than desired GPU memory utilization "
                f"({self.cache_config.gpu_memory_utilization}, "
                f"{GiB(self.requested_memory)} GiB). Decrease GPU memory "
                f"utilization or reduce GPU memory used by other processes."
            )

        if (
            self.parallel_config.data_parallel_size > 1
            and self.parallel_config.data_parallel_size_local > 0
            and self.parallel_config.distributed_executor_backend not in ["ray", "external_launcher"]
            and self.vllm_config.parallel_config.data_parallel_backend != "ray"
            and self.vllm_config.parallel_config.nnodes_within_dp == 1
        ):
            visible_device_count = torch.npu.device_count() if torch.npu.is_available() else 0
            assert self.parallel_config.local_world_size <= visible_device_count, (
                f"local_world_size ({self.parallel_config.local_world_size}) must "
                f"be less than or equal to the number of visible devices "
                f"({visible_device_count})."
            )

        # Initialize the distributed environment.
        self._init_worker_distributed_environment()
        # Set random seed.
        set_random_seed(self.model_config.seed)

        return device
