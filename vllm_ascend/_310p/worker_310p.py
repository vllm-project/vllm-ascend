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
import subprocess

import psutil
import torch
import torch_npu
from vllm.config import CUDAGraphMode
from vllm.logger import logger
from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.mem_utils import MemorySnapshot, format_gib, memory_profiling
from vllm.utils.torch_utils import set_random_seed  # noqa: E402
from vllm.v1.worker.worker_base import CompilationTimes

from vllm_ascend._310p.compile_warmup import build_piecewise_compile_warmup_sizes
from vllm_ascend._310p.model_runner_310p import NPUModelRunner310
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.cpu_binding import bind_cpus
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type
from vllm_ascend.worker.worker import NPUWorker, init_workspace_manager

_IS_RC_DEVICE: bool | None = None


def _is_rc_device() -> bool:
    global _IS_RC_DEVICE
    if _IS_RC_DEVICE is None:
        try:
            # Use lspci to detect if the device is in RC mode.
            # In EP mode, "accelerators" typically appears in the output.
            result = subprocess.run(["lspci"], capture_output=True, text=True, check=True)
            _IS_RC_DEVICE = not any("accelerators" in line.strip() for line in result.stdout.splitlines())
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to False if lspci is unavailable or fails.
            _IS_RC_DEVICE = False
    return _IS_RC_DEVICE


class NPUWorker310(NPUWorker):
    def init_device(self):
        self.device = self._init_device()
        torch_npu.npu.set_compile_mode(jit_compile=False)

        init_workspace_manager(self.device, num_ubatches=1)

        self.model_runner = NPUModelRunner310(self.vllm_config, self.device)
        logger.info_once("Using NPUWorker310 and NPUModelRunner310.")

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
            if _is_rc_device():
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

        if _is_rc_device():
            self.available_kv_cache_memory_bytes = (self.requested_memory - psutil.virtual_memory().used) // 2
        else:
            self.available_kv_cache_memory_bytes = (
                self.requested_memory - profile_result.non_kv_cache_memory - non_torch_memory_cleared_by_empty_cache
            ) // 2

        logger.debug(profile_result)
        logger.info_once(
            "Available KV cache memory: %.2f GiB (halved for workspace)",
            GiB(self.available_kv_cache_memory_bytes),
            scope="local",
        )
        return int(self.available_kv_cache_memory_bytes)

    def _warm_up_atb(self):
        # 310p device do not support torch_npu._npu_matmul_add_fp32 atb ops
        logger.info_once("Skip warm-up atb ops for 310P device.")

    def compile_or_warm_up_model(self) -> CompilationTimes:
        warmup_sizes = (self.vllm_config.compilation_config.compile_sizes or []).copy()
        cg_capture_sizes: list[int] = []
        if not self.model_config.enforce_eager:
            if self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
                cg_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
                cg_capture_sizes = [] if cg_sizes is None else cg_sizes

            compile_ranges = self.vllm_config.compilation_config.get_compile_ranges()
            warmup_sizes = build_piecewise_compile_warmup_sizes(
                self.vllm_config,
                warmup_sizes,
                cg_capture_sizes,
                compile_ranges,
            )

        for size in sorted(warmup_sizes, reverse=True):
            logger.info("Compile and warming up model for size %d", size)
            self.model_runner._dummy_run(size)

        npugraph_memory_bytes = 0
        if not self.model_config.enforce_eager:
            npugraph_memory_bytes = self.model_runner.capture_model()

        if hasattr(self, "npugraph_memory_estimate") and self.npugraph_memory_estimate > 0:
            GiB = lambda b: round(b / GiB_bytes, 2)
            diff = abs(npugraph_memory_bytes - self.npugraph_memory_estimate)
            logger.info(
                "ACL graph pool memory: %s GiB (actual), %s GiB (estimated), difference: %s GiB (%.1f%%).",
                GiB(npugraph_memory_bytes),
                GiB(self.npugraph_memory_estimate),
                GiB(diff),
                100 * diff / max(npugraph_memory_bytes, 1),
            )

        if self.cache_config.kv_cache_memory_bytes is None and hasattr(self, "peak_activation_memory"):
            redundancy_buffer = 150 * (1 << 20)
            non_kv_memory = (
                self.model_runner.model_memory_usage
                + self.peak_activation_memory
                + self.non_torch_memory
                + npugraph_memory_bytes
            )
            self.npugraph_memory_bytes = npugraph_memory_bytes
            suggested_to_requested = int(self.requested_memory) - non_kv_memory - redundancy_buffer
            suggested_to_gpu_limit = int(self.init_snapshot.free_memory) - non_kv_memory - redundancy_buffer
            msg = (
                f"Free memory on device "
                f"({format_gib(self.init_snapshot.free_memory)}/"
                f"{format_gib(self.init_snapshot.total_memory)} GiB) on startup. "
                f"Desired GPU memory utilization is "
                f"({self.cache_config.gpu_memory_utilization}, "
                f"{format_gib(self.requested_memory)} GiB). "
                f"Actual usage: {format_gib(self.model_runner.model_memory_usage)} GiB "
                f"for weights, {format_gib(self.peak_activation_memory)} GiB for peak "
                f"activation, {format_gib(self.non_torch_memory)} GiB for non-torch "
                f"memory, {format_gib(npugraph_memory_bytes)} GiB for NPU graph memory. "
                f"Replace gpu_memory_utilization with "
                f"`--kv-cache-memory={suggested_to_requested}` "
                f"({format_gib(suggested_to_requested)} GiB) to fit into requested "
                f"memory, or `--kv-cache-memory={suggested_to_gpu_limit}` "
                f"({format_gib(suggested_to_gpu_limit)} GiB) to fully utilize NPU "
                f"free memory. Current KV cache memory: "
                f"{format_gib(self.available_kv_cache_memory_bytes)} GiB."
            )
            logger.info(msg)

        if get_ascend_device_type() != AscendDeviceType.A5:
            self._warm_up_atb()
        if get_ascend_config().enable_cpu_binding:
            try:
                bind_cpus(self.local_rank)
            except Exception as e:
                logger.warning("Bind cpus failed in rank%s: %s Skip binding cpu.", self.local_rank, e)
        set_random_seed(self.model_config.seed)
        return CompilationTimes(
            language_model=self.vllm_config.compilation_config.compilation_time,
            encoder=getattr(
                self.vllm_config.compilation_config,
                "encoder_compilation_time",
                0.0,
            ),
        )

    def _init_device(self):
        device = torch.device(f"npu:{self.local_rank}")
        torch.npu.set_device(device)

        # This lazy import avoids torch_npu re-initialization in patch
        # Note that this should be imported after torch.npu.set_device
        # to avoid repeated set_device in extra processes

        gc.collect()
        torch.npu.empty_cache()

        # take current memory snapshot
        self.init_snapshot = MemorySnapshot()
        self.requested_memory = self.init_snapshot.total_memory * self.cache_config.gpu_memory_utilization
        if _is_rc_device():
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
