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

import copy
import gc
import inspect
import logging
from types import NoneType
from typing import Any

import torch
import torch.nn as nn
import torch_npu
from torch_npu.op_plugin.atb._atb_ops import _register_atb_extensions
from torch_npu.profiler import dynamic_profile as dp
from vllm.config import CUDAGraphMode, VllmConfig, set_current_vllm_config
from vllm.distributed import ensure_model_parallel_initialized, get_pcp_group, init_distributed_environment
from vllm.distributed.ec_transfer import ensure_ec_transfer_initialized
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_initialized,
    ensure_kv_transfer_shutdown,
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorHandshakeMetadata
from vllm.distributed.parallel_state import Handle, get_pp_group, get_tp_group
from vllm.logger import logger
from vllm.lora.request import LoRARequest
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.tasks import SupportedTask
from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.mem_utils import MemorySnapshot, format_gib, memory_profiling
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, AsyncModelRunnerOutput, DraftTokenIds, ModelRunnerOutput
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.gpu_worker import AsyncIntermediateTensors
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase
from vllm.v1.worker.workspace import init_workspace_manager

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config
from vllm_ascend.batch_invariant import init_batch_invariance
from vllm_ascend.core.profiling_chunk_predictor import (
    _attach_profiling_chunk_execution_time,
)
from vllm_ascend.cpu_binding import bind_cpus
from vllm_ascend.device_allocator.camem import CaMemAllocator
from vllm_ascend.device_allocator.sleep_mem_optimized import SleepWakeupManager
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_cache_layout import (
    build_layerwise_cache_layout,
    build_layerwise_reuse_layout,
    get_gva_layerwise_config,
    get_layerwise_physical_layer_index,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    get_host_device_memory_usage_ratio,
)
from vllm_ascend.distributed.parallel_state import init_ascend_model_parallel
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper
from vllm_ascend.utils import (
    AscendDeviceType,
    check_ascend_device_type,
    enable_sp,
    get_ascend_device_type,
    register_ascend_customop,
    setup_ascend_local_comm_res,
    vllm_version_is,
)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

torch._dynamo.trace_rules.clear_lru_cache()  # noqa: E402
from torch._dynamo.variables import TorchInGraphFunctionVariable  # noqa: E402
from vllm.utils.torch_utils import set_random_seed  # noqa: E402

torch_non_c_binding_in_graph_functions_npu = dict.fromkeys(
    ["torch.npu.current_stream"],
    TorchInGraphFunctionVariable,
)  # noqa: E402
torch_non_c_binding_in_graph_functions_npu["torch.npu.stream"] = TorchInGraphFunctionVariable  # noqa: E402
torch._dynamo.trace_rules.torch_name_rule_map.append(torch_non_c_binding_in_graph_functions_npu)  # noqa: E402


# These control buffers are created outside the weights mem-pool, so Level-1
# sleep must save them explicitly. They are matched against the trailing
# segment of each buffer name (e.g. "..._dsa_cp_hadamard").
_allowed_names = (
    "_dsa_cp_hadamard",
    "_dsa_hadamard",
)


class NPUWorker(WorkerBase):
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
        # Additional parameters for compatibility with vllm
        **kwargs,
    ):
        """Initialize the worker for Ascend."""
        if not envs_ascend.COMPILE_CUSTOM_KERNELS:
            logger.warning(
                "COMPILE_CUSTOM_KERNELS is set to False. "
                "In most scenarios, without custom kernels, vllm-ascend will not function correctly."
            )

        # register patch for vllm
        from vllm_ascend.utils import adapt_patch

        adapt_patch()

        # Register ops when worker init.
        from vllm_ascend import ops

        ops.register_dummy_fusion_op()
        if get_ascend_device_type() != AscendDeviceType.A5:
            _register_atb_extensions()
        register_ascend_customop(vllm_config)
        # init ascend config and soc version
        init_ascend_config(vllm_config)
        from vllm_ascend.logger import configure_ascend_file_logging

        configure_ascend_file_logging()
        check_ascend_device_type()

        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        if self.cache_config.cache_dtype == "auto":
            self.cache_dtype = self.model_config.dtype
        else:
            self.cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[self.cache_config.cache_dtype]

        # Profiler is lazily initialized on first profile(is_start=True) call (RFC #6954)
        self.profiler_config = vllm_config.profiler_config
        self.profiler: TorchNPUProfilerWrapper | None = None
        self.npugraph_memory_bytes = 0
        if vllm_config.model_config and vllm_config.model_config.enable_sleep_mode:
            # Buffers saved before sleep
            self._sleep_saved_buffers: dict[str, torch.Tensor] = {}
        self.sleep_wakeup_manager = SleepWakeupManager(vllm_config, self, lambda: getattr(self, "model_runner", None))

        # Weight transfer engine is created in `load_model` once the model
        # is available, since the engine needs a reference to the model.
        self.weight_transfer_engine = None
        self._weight_update_active = False

        # FixMe: this is a patch to fix the issue cause by https://github.com/vllm-project/vllm/commit/de94289a98d7ec52a5ef02719e01a1db8b505170
        from vllm.model_executor.layers.linear import WEIGHT_LOADER_V2_SUPPORTED

        if "UnquantizedLinearMethod" in WEIGHT_LOADER_V2_SUPPORTED:
            WEIGHT_LOADER_V2_SUPPORTED.remove("UnquantizedLinearMethod")

        self.use_v2_model_runner = self.vllm_config.use_v2_model_runner
        if not vllm_version_is("0.27.1") and getattr(self.vllm_config, "is_mm_encoder_only", False):
            # Upstream moved the V2 encoder-only path into MMEncoderModelRunner,
            # which vllm-ascend does not implement; keep the V1 runner, which
            # still handles encoder-only instances.
            self.use_v2_model_runner = False
        self._pp_send_work: list[Handle] = []

        ascend_compilation_config = get_ascend_config().ascend_compilation_config
        if ascend_compilation_config.enable_npugraph_ex and ascend_compilation_config.enable_static_kernel:
            # Prevent duplicate triggers, execute the exit logic only once
            shutdown_request = False

            def signal_handler(signum, frame):
                nonlocal shutdown_request
                if not shutdown_request:
                    shutdown_request = True
                    self.uninstall_static_kernel()
                    raise SystemExit()

            # Either SIGTERM or SIGINT will terminate the worker
            import signal

            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

    def uninstall_static_kernel(self):
        import fcntl
        import os
        import subprocess

        ascend_home_path = os.environ["ASCEND_HOME_PATH"]
        static_kernel_dir_path = os.path.join(ascend_home_path, "opp/static_kernel")
        uninstall_script_path = os.path.join(static_kernel_dir_path, "ai_core/uninstall.sh")
        lock_file_path = os.path.join(static_kernel_dir_path, "uninstall.lock")

        if not os.path.exists(uninstall_script_path):
            return
        with open(lock_file_path, "w") as lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                subprocess.Popen(
                    ["bash", uninstall_script_path],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except (BlockingIOError, OSError):
                return
            finally:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    if os.path.exists(lock_file_path):
                        os.remove(lock_file_path)
                except Exception:
                    return

    def sleep(self, level: int = 1) -> None:
        free_bytes_before_sleep = torch.npu.mem_get_info()[0]
        model = self.model_runner.model
        if level == 1:
            self._sleep_saved_buffers = {
                name: buffer.cpu().clone()
                for name, buffer in model.named_buffers()
                if name.rsplit(".", maxsplit=1)[-1] in _allowed_names
            }
        else:
            self._sleep_saved_buffers = {name: buffer.cpu().clone() for name, buffer in model.named_buffers()}

        rl_config = get_ascend_config().rl_config
        cleanup_enabled = rl_config.enabled and rl_config.sleep_mode_extra_cleanup
        if cleanup_enabled:
            self.sleep_wakeup_manager.sleep()

        allocator = CaMemAllocator.get_instance()
        allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())
        free_bytes_after_sleep, total = torch.npu.mem_get_info()
        freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        used_bytes = total - free_bytes_after_sleep
        assert freed_bytes >= 0, "Memory usage increased after sleeping."

        logger.info(
            "Sleep mode (level=%s) freed %.2f GiB memory, %.2f GiB memory is still in use.",
            level,
            freed_bytes / GiB_bytes,
            used_bytes / GiB_bytes,
        )

    def wake_up(self, tags: list[str] | None = None) -> None:
        nz_mode = get_ascend_config().weight_nz_mode
        if nz_mode:
            raise ValueError(
                "FRACTAL_NZ mode is enabled. This may cause model parameter precision issues "
                "in the RL scenarios. Please set weight_nz_mode=0 via --additional-config, "
                "or enable additional_config.rl_config (enabled: true) which disables NZ "
                "automatically."
            )
        allocator = CaMemAllocator.get_instance()
        allocator.wake_up(tags=tags)

        # Restore the buffers after level 2 sleep
        if len(self._sleep_saved_buffers):
            model = self.model_runner.model
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}

        if tags is None or "kv_cache" in tags:
            self.model_runner.post_kv_cache_wake_up()

        rl_config = get_ascend_config().rl_config
        cleanup_enabled = rl_config.enabled and rl_config.sleep_mode_extra_cleanup
        if cleanup_enabled:
            self.sleep_wakeup_manager.wakeup(tags)

        if not vllm_version_is("0.27.1"):
            self.synchronize_device()

    def _check_weight_transfer_engine(self) -> None:
        if self.weight_transfer_engine is None:
            raise RuntimeError(
                "Weight transfer not configured. Please set weight_transfer_config to enable weight transfer."
            )

    def init_weight_transfer_engine(self, init_info: dict) -> None:
        """Initialize the HCCL weight transfer process group with the trainer."""
        self._check_weight_transfer_engine()
        assert self.weight_transfer_engine is not None
        typed_init_info = self.weight_transfer_engine.parse_init_info(init_info)
        self.weight_transfer_engine.init_transfer_engine(typed_init_info)

    def _check_nz_disabled(self) -> None:
        if get_ascend_config().weight_nz_mode:
            raise ValueError(
                "FRACTAL_NZ mode is enabled. This may cause model parameter "
                "precision issues in the RL scenarios. Please set weight_nz_mode=0 "
                "via --additional-config, or enable additional_config.rl_config "
                "(enabled: true) which disables NZ automatically."
            )

    def start_weight_update(self) -> None:
        """Begin a new weight update; prepares the model for layerwise reload."""
        self._check_weight_transfer_engine()

        if self._weight_update_active:
            raise RuntimeError(
                "start_weight_update called while a weight update is already active. Call finish_weight_update first."
            )

        self._check_nz_disabled()

        assert self.weight_transfer_engine is not None
        self.weight_transfer_engine.start_weight_update()
        self._weight_update_active = True

    def update_weights(self, update_info: dict | list[dict]) -> None:
        """Receive a chunk of weights from the trainer and load them in place."""
        self._check_weight_transfer_engine()
        assert self.weight_transfer_engine is not None

        # state machine driven by start/finish.
        if not self._weight_update_active:
            raise RuntimeError("start_weight_update must be called before update_weights.")

        if isinstance(update_info, list):
            # A list indexed by global worker rank across data parallel replicas.
            parallel_config = self.vllm_config.parallel_config
            local_update_info = update_info[parallel_config.data_parallel_rank * parallel_config.world_size + self.rank]
        else:
            local_update_info = update_info

        try:
            self.weight_transfer_engine.update_weights(local_update_info)
        except BaseException:
            self._weight_update_active = False
            raise

    def finish_weight_update(self) -> None:
        """Finish the current weight update; runs layerwise postprocessing."""
        self._check_weight_transfer_engine()

        if not self._weight_update_active:
            raise RuntimeError("start_weight_update must be called before finish_weight_update.")

        assert self.weight_transfer_engine is not None
        self.weight_transfer_engine.finish_weight_update()
        self._weight_update_active = False

    def shutdown(self) -> None:
        if ensure_kv_transfer_shutdown is not None:
            ensure_kv_transfer_shutdown()

        if self.profiler is not None:
            self.profiler.shutdown()

        if weight_transfer_engine := getattr(self, "weight_transfer_engine", None):
            weight_transfer_engine.shutdown()

        if model_runner := getattr(self, "model_runner", None):
            shutdown_fn = getattr(model_runner, "shutdown", None)
            if callable(shutdown_fn):
                shutdown_fn()

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def _init_device(self):
        # vLLM v0.24.0 (PR #45026) removed automatic per-process device
        # isolation for DP workers. Mirror gpu_worker.py::init_device:
        # shift self.local_rank by dp_local_rank * tp_pp_world_size so
        # that each DP group binds to a distinct set of NPUs.
        parallel_config = self.parallel_config
        if (
            parallel_config.distributed_executor_backend not in ("ray", "external_launcher")
            and parallel_config.data_parallel_backend != "ray"
            and parallel_config.nnodes_within_dp == 1
            # vllm-ascend: when the user pre-shards devices via
            # --device-ids (which becomes assigned_physical_gpu_ids),
            # each child process already binds to its own NPU(s); the
            # DP local_rank shift below would push local_rank past the
            # length of the per-rank device list and trip the assert
            # in this same method. Skip the shift in that case.
            and parallel_config.assigned_physical_gpu_ids is None
        ):
            dp_local_rank = parallel_config.data_parallel_rank_local
            if dp_local_rank is None:
                dp_local_rank = parallel_config.data_parallel_index
            tp_pp_world_size = parallel_config.pipeline_parallel_size * parallel_config.tensor_parallel_size
            self.local_rank += dp_local_rank * tp_pp_world_size

        # Publish the logical-to-physical mapping for topology queries.
        assigned_physical_gpu_ids = parallel_config.assigned_physical_gpu_ids
        if assigned_physical_gpu_ids is not None:
            from vllm.platforms.interface import set_assigned_physical_gpu_ids

            set_assigned_physical_gpu_ids(assigned_physical_gpu_ids)
            assert self.local_rank < len(assigned_physical_gpu_ids), (
                f"local_rank {self.local_rank} is out of bounds for "
                f"assigned_physical_gpu_ids {assigned_physical_gpu_ids}"
            )
            if parallel_config.distributed_executor_backend not in ("ray", "external_launcher"):
                assert parallel_config.local_world_size <= len(assigned_physical_gpu_ids), (
                    f"local_world_size ({parallel_config.local_world_size}) "
                    f"exceeds assigned_physical_gpu_ids count "
                    f"({len(assigned_physical_gpu_ids)})"
                )
        else:
            visible_device_count = torch.npu.device_count() if torch.npu.is_available() else 0
            assert self.local_rank < visible_device_count, (
                f"DP adjusted local rank {self.local_rank} is out of bounds for {visible_device_count} devices."
            )

        visible_device_index = current_platform.logical_device_id_to_visible_device_id(self.local_rank)
        device = torch.device(f"{current_platform.device_type}:{visible_device_index}")

        torch.npu.set_device(device)

        # Import _inductor for graph mode execution with triton
        # This lazy import avoids torch_npu re-initialization in patch
        # Note that this should be imported after torch.npu.set_device
        # to avoid repeated set_device in extra processes
        from vllm.triton_utils import HAS_TRITON

        if HAS_TRITON:
            import torch_npu._inductor  # noqa: F401

        gc.collect()
        torch.npu.empty_cache()

        if get_ascend_device_type() == AscendDeviceType.A5:
            setup_ascend_local_comm_res(self.local_rank, self.vllm_config.kv_transfer_config)

        # take current memory snapshot
        self.init_snapshot = MemorySnapshot(device=device)
        self.requested_memory = self.init_snapshot.total_memory * self.cache_config.gpu_memory_utilization
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
        # Initialize device properties used by triton kernels.
        init_device_properties_triton()

        return device

    def init_device(self):
        # NOTE: KEEP device the member of `NPUWorker`, as it will be checked
        # in ray scenario. see https://github.com/vllm-project/vllm/pull/26845
        # for more details
        self.device = self._init_device()
        # Initialize workspace manager
        num_ubatches = 1
        init_workspace_manager(self.device, num_ubatches)
        # Init ModelRunner here, so that we have access to self.device.
        if self.use_v2_model_runner:
            logger.warning("npu model runner v2 is in developing, some features doesn't work for now.")
            from vllm_ascend.worker.v2.model_runner import NPUModelRunner as NPUModelRunnerV2

            self.model_runner = NPUModelRunnerV2(self.vllm_config, self.device)
        else:
            self.model_runner = NPUModelRunner(self.vllm_config, self.device)

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Profiles the peak memory usage of the model to determine how much
        memory can be used for KV cache without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculates the free memory that can be used for KV cache in
        bytes.
        """
        GiB = lambda b: b / GiB_bytes

        # Fast path: user has explicitly specified KV cache size via
        # --kv-cache-memory. Still run profile_run() to compile the model,
        # but skip the memory profiling calculation entirely.
        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
            self.model_runner.profile_run()
            logger.info(
                "Initial free memory %.2f GiB, reserved %.2f GiB for KV Cache "
                "as specified by kv_cache_memory_bytes, skipping memory profiling. "
                "This does not respect the gpu_memory_utilization config. "
                "Only use kv_cache_memory_bytes when you want manual control of "
                "KV cache memory size. If OOM'ed, check the difference of initial "
                "free memory between the current run and the previous run where "
                "kv_cache_memory_bytes is suggested and update it correspondingly.",
                GiB(self.init_snapshot.free_memory),
                GiB(kv_cache_memory_bytes),
            )
            kv_cache_memory_bytes = self.update_available_memory_for_sparse_kv_offload(kv_cache_memory_bytes)
            return kv_cache_memory_bytes

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        with memory_profiling(
            self.init_snapshot,
            weights_memory=int(self.model_runner.model_memory_usage),
        ) as profile_result:
            self.model_runner.profile_run()

            # Record torch peak INSIDE the context and BEFORE graph capture,
            # so that graph pool allocations don't inflate the activation peak.
            # The memory_profiling context will also compute torch_peak_increase
            # on exit, but we override it below with this pre-graph value.
            profile_torch_peak = torch.npu.memory_stats(self.device).get("allocated_bytes.all.peak", 0)

        # Override torch_peak_increase with the pre-graph-capture value to
        # avoid double-counting graph pool memory as activation memory.
        profile_result.torch_peak_increase = profile_torch_peak - profile_result.before_profile.torch_peak
        profile_result.non_kv_cache_memory = (
            profile_result.non_torch_increase + profile_result.torch_peak_increase + profile_result.weights_memory
        )

        # Save per-category memory for use in compile_or_warm_up_model() (step 5).
        self.peak_activation_memory = profile_result.torch_peak_increase
        self.non_torch_memory = profile_result.non_torch_increase

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
        self.available_kv_cache_memory_bytes = self.requested_memory - profile_result.non_kv_cache_memory

        extra_config = get_gva_layerwise_config(self.vllm_config.kv_transfer_config)
        if extra_config is not None:
            memory_info = getattr(self, "_gva_layerwise_memory_info", None)
            if memory_info is None:
                num_layers = self.model_config.get_num_layers(self.parallel_config)
                layout = build_layerwise_cache_layout(num_layers, extra_config)
                num_buffer_assignments = len(layout.storage_indices)
                factor = num_layers / num_buffer_assignments if layout.has_layer_reuse else 1.0
            else:
                num_layers, num_buffer_assignments, factor = memory_info
            if factor != 1.0:
                self.available_kv_cache_memory_bytes = int(self.available_kv_cache_memory_bytes * factor)
                logger.info(
                    "Layerwise KV cache reuse maps %d layers onto %d buffer assignments; "
                    "scale logical KV budget by %.3f.",
                    num_layers,
                    num_buffer_assignments,
                    factor,
                )

        logger.debug(profile_result)
        logger.info_once(
            "Available KV cache memory: %.2f GiB", GiB(self.available_kv_cache_memory_bytes), scope="local"
        )
        self.available_kv_cache_memory_bytes = self.update_available_memory_for_sparse_kv_offload(
            self.available_kv_cache_memory_bytes,
        )

        return int(self.available_kv_cache_memory_bytes)

    def update_available_memory_for_sparse_kv_offload(self, available_memory):
        """
        A simple patch for Sparse KV offload: add additional available_memory according to the
        ratio of host memory (kv) and dev memory (indexer), so we can allocate blocks for indexer cache
        using all original available device memory without modify original kv_spec or vllm code.
        For further optimization, consider to merge this logic to vllm kv_cache_utils.py,
        or reuse hisparse's host pool logic after it's merged to vllm.
        """
        GiB = lambda b: b / GiB_bytes
        sparse_kv_offload_config = get_ascend_config().sparse_kv_offload_config
        if not sparse_kv_offload_config.enabled:
            return available_memory
        keep_device_kv_cache = sparse_kv_offload_config.keep_device_kv_cache
        if keep_device_kv_cache:
            needed_dram_size_bytes = available_memory
        else:
            kv_cache_spec = getattr(self, "kv_cache_spec", None) or self.get_kv_cache_spec()
            host_device_memory_usage_ratio = get_host_device_memory_usage_ratio(kv_cache_spec)
            needed_dram_size_bytes = host_device_memory_usage_ratio * available_memory
        if needed_dram_size_bytes > sparse_kv_offload_config.dram_size_per_dp_GB * (1 << 30):
            raise ValueError(
                f"Needed dram size ({GiB(needed_dram_size_bytes)} GB) is larger than "
                f"user specified dram size ({sparse_kv_offload_config.dram_size_per_dp_GB} GB). "
                "Please increase sparse_kv_offload_config.dram_size_per_dp_GB if available on your device."
            )
        if not keep_device_kv_cache:
            available_memory += needed_dram_size_bytes
            logger.info_once(
                "Sparse KV offload is enabled, enlarge total available memory to %.2f GiB",
                GiB(available_memory),
                scope="local",
            )
        return int(available_memory)

    def log_memory_stats(self) -> None:
        """Profiles the torch reserved memory, torch allocated memory in execute_model()."""
        if not logger.isEnabledFor(logging.DEBUG):
            return
        self.torch_reserved = torch.npu.memory_reserved()
        self.torch_allocated = torch.npu.memory_allocated()
        logger.debug(
            "torch reserved memory: %.2f GiB, torch allocated memory: %.2f GiB",
            self.torch_reserved / GiB_bytes,
            self.torch_allocated / GiB_bytes,
        )

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        self.log_memory_stats()
        # enable msMonitor to monitor the performance of vllm-ascend
        if get_ascend_config().msmonitor_use_daemon:
            dp.step()

        if self._pp_send_work:
            for handle in self._pp_send_work:
                handle.wait()
            self._pp_send_work = []

        intermediate_tensors = None
        forward_pass = scheduler_output.total_num_scheduled_tokens > 0
        if forward_pass and not get_pp_group().is_first_rank:
            if enable_sp():
                all_gather_group = None
            else:
                all_gather_group = get_tp_group()
            tensor_dict, comm_handles, comm_postprocess = get_pp_group().irecv_tensor_dict(
                all_gather_group=all_gather_group
            )
            assert tensor_dict is not None
            intermediate_tensors = AsyncIntermediateTensors(
                tensor_dict,
                comm_handles=comm_handles,
                comm_postprocess=comm_postprocess,
            )

        if self.profiler is not None:
            self.profiler.step()

        output = self.model_runner.execute_model(scheduler_output, intermediate_tensors)
        if isinstance(output, (ModelRunnerOutput, AsyncModelRunnerOutput, NoneType)):
            return output

        assert isinstance(output, IntermediateTensors)
        parallel_config = self.vllm_config.parallel_config
        assert parallel_config.distributed_executor_backend != ("external_launcher") and not get_pp_group().is_last_rank
        if enable_sp():
            all_gather_group = None
        else:
            all_gather_group = get_tp_group()
        self._pp_send_work = get_pp_group().isend_tensor_dict(
            output.tensors,
            all_gather_group=all_gather_group,
        )

        # Align with upstream GPUWorker: Model Runner V2 has no
        # kv_connector_output to propagate from non-last PP ranks. Model Runner
        # V1 must continue below to handle the PP + KV connector path.
        if self.use_v2_model_runner:
            return None

        kv_connector_output = output.kv_connector_output
        if not kv_connector_output:
            return None

        # In case of PP with kv transfer, we need to pass through the
        # kv_connector_output
        if not kv_connector_output.finished_sending and not kv_connector_output.finished_recving:
            return EMPTY_MODEL_RUNNER_OUTPUT
        output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.kv_connector_output = kv_connector_output
        return output

    @torch.inference_mode()
    def sample_tokens(self, grammar_output: "GrammarOutput") -> ModelRunnerOutput | AsyncModelRunnerOutput:
        if not self.use_v2_model_runner:
            return self.model_runner.sample_tokens(grammar_output)

        output = self.model_runner.sample_tokens(grammar_output)
        _attach_profiling_chunk_execution_time(
            self.model_runner.ascend_config.scheduler_config.profiling_chunk_config,
            self.model_runner,
            output,
        )
        return output

    def load_model(self) -> None:
        if self.vllm_config.model_config.enable_sleep_mode:
            allocator = CaMemAllocator.get_instance()
            assert allocator.get_current_usage() == 0, "Sleep mode can only be used for one instance per process."
            context = allocator.use_memory_pool(tag="weights")
        else:
            from contextlib import nullcontext

            context = nullcontext()  # type: ignore

        with context, set_current_vllm_config(self.vllm_config):
            self.model_runner.load_model()

        if self.vllm_config.weight_transfer_config is not None:
            from vllm.distributed.weight_transfer.factory import (
                WeightTransferEngineFactory,
            )

            self.weight_transfer_engine = WeightTransferEngineFactory.create_engine(
                self.vllm_config.weight_transfer_config,
                self.vllm_config,
                self.device,
                self.model_runner.get_model(),
            )

    def compile_or_warm_up_model(self) -> CompilationTimes:
        # Note: need to adapt for graph mode.
        warmup_sizes = (self.vllm_config.compilation_config.compile_sizes or []).copy()
        if not self.model_config.enforce_eager:
            cg_capture_sizes: list[int] = []
            if self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
                cg_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
                cg_capture_sizes = [] if cg_sizes is None else cg_sizes
                warmup_sizes = [x for x in warmup_sizes if x not in cg_capture_sizes]

            compile_ranges = self.vllm_config.compilation_config.get_compile_ranges()
            # For each compile_range, if none of the batch sizes
            # in warmup_sizes or cudagraph_capture_sizes are in the range,
            # add the end of the range to ensure compilation/warmup.
            all_sizes = set(cg_capture_sizes)
            all_sizes.update([x for x in warmup_sizes if isinstance(x, int)])
            for compile_range in compile_ranges:
                if not any(x in compile_range for x in all_sizes):
                    warmup_sizes.append(compile_range.end)

        for size in sorted(warmup_sizes, reverse=True):
            logger.info("Compile and warming up model for size %d", size)
            self.model_runner._dummy_run(size)

        from vllm_ascend.model_executor.warmup.kernel_warmup import kernel_warmup

        kernel_warmup(self)

        npugraph_memory_bytes = 0
        if not self.model_config.enforce_eager:
            npugraph_memory_bytes = self.model_runner.capture_model()

        # Suggest an optimal --kv-cache-memory value for future runs.
        # Only emitted when we ran full profiling (kv_cache_memory_bytes was not
        # pre-specified) so that peak_activation_memory etc. are available.
        # non_kv_memory already includes NPU graph memory, so the suggestion
        # accounts for all measured memory categories. A 150 MiB buffer is kept
        # because memory_profiling may slightly underestimate non-torch
        # allocations (ACL context, HCCL buffers, driver layer, etc.).
        if self.cache_config.kv_cache_memory_bytes is None and hasattr(self, "peak_activation_memory"):
            redundancy_buffer = 150 * (1 << 20)  # 150 MiB safety margin
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
                f"{format_gib(self.available_kv_cache_memory_bytes)} GiB. "
                f"After warmup: torch reserved memory {format_gib(torch.npu.memory_reserved())} GiB, "
                f"torch allocated memory {format_gib(torch.npu.memory_allocated())} GiB."
            )
            logger.info(msg)

        # Call ATB matmul to warm up; otherwise, the first operation (ReshapeAndCache)
        # may cause performance degradation at runtime.
        if get_ascend_device_type() != AscendDeviceType.A5:
            self._warm_up_atb()
        # Bind after warmup so hot allocations are already materialized on the
        # worker process before migratepages/taskset run.
        if get_ascend_config().enable_cpu_binding:
            try:
                bind_cpus(self.local_rank)
            except Exception as e:
                logger.warning("Bind cpus failed in rank%s: %s Skip binding cpu.", self.local_rank, e)

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)
        return CompilationTimes(
            language_model=self.vllm_config.compilation_config.compilation_time,
            # `encoder_compilation_time` was added after v0.19.1 (vLLM #39240); fall
            # back to 0.0 so the older release still constructs CompilationTimes.
            encoder=getattr(
                self.vllm_config.compilation_config,
                "encoder_compilation_time",
                0.0,
            ),
        )

    def _warm_up_atb(self):
        x = torch.rand((2, 4), dtype=torch.float16).npu()
        weight = torch.rand((2, 4), dtype=torch.float16).npu()
        c = torch.rand((4, 4), dtype=torch.float32).npu()
        torch_npu._npu_matmul_add_fp32(x, weight, c)

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    @torch.inference_mode()
    def profile_prefill_latency(self, num_tokens: int) -> float:
        """
        Profile prefill latency for a given number of tokens.

        This runs a real model forward pass and measures the execution time.
        Used for profiling-based dynamic chunk sizing.

        In PP (Pipeline Parallelism) mode:
        - All workers execute the forward pass to stay synchronized
        - Only the timing from PP0 (first rank) is meaningful for scheduling
        - PP0 includes all the pipeline stages' latency when using async scheduling

        Args:
            num_tokens: Number of tokens to profile

        Returns:
            Latency in milliseconds
        """
        import time

        # Clamp to valid range
        num_tokens = min(num_tokens, self.scheduler_config.max_num_batched_tokens)
        num_tokens = max(num_tokens, 1)

        # Synchronize all devices before timing
        # This ensures clean measurement in PP/TP scenarios
        torch.npu.synchronize()

        # In PP mode, we still run on all ranks to keep them synchronized
        # but only the first rank's timing is used for scheduling decisions
        is_first_pp_rank = get_pp_group().is_first_rank

        start = time.perf_counter()

        # Run a real model forward with attention enabled in eager mode.
        # _dummy_run handles PP internally (intermediate tensors, etc.).
        old_max_num_reqs = self.model_runner.max_num_reqs
        try:
            self.model_runner.max_num_reqs = 1

            dummy_run_kwargs = {
                "num_tokens": num_tokens,
                "force_attention": True,
                "is_profile": False,
                "cudagraph_runtime_mode": CUDAGraphMode.NONE,
            }

            if "profile_cpp" in inspect.signature(self.model_runner._dummy_run).parameters:
                dummy_run_kwargs["profile_cpp"] = True

            self.model_runner._dummy_run(**dummy_run_kwargs)
        finally:
            self.model_runner.max_num_reqs = old_max_num_reqs

        # Synchronize after forward to ensure NPU operations complete
        torch.npu.synchronize()

        latency_ms = (time.perf_counter() - start) * 1000

        # Log for debugging in PP mode
        if not is_first_pp_rank:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[ProfilingChunk] PP rank %s: profiled %s tokens, latency=%.2f ms (not used)",
                    get_pp_group().rank_in_group,
                    num_tokens,
                    latency_ms,
                )

        return latency_ms

    def get_kv_connector_handshake_metadata(
        self,
    ) -> dict[tuple[int, ...], KVConnectorHandshakeMetadata] | None:
        """Get KV connector metadata from this worker if available."""
        if not has_kv_transfer_group():
            return None

        connector = get_kv_transfer_group()

        # Return None for connectors that don't need to exchange handshake
        # metadata across workers.
        if (metadata := connector.get_handshake_metadata()) is None:
            return None
        tp_rank = get_tp_group().rank_in_group
        pp_rank = get_pp_group().rank_in_group
        pcp_size = get_pcp_group().world_size
        if pcp_size > 1:
            pcp_rank = get_pcp_group().rank_in_group
            return {(pp_rank, pcp_rank, tp_rank): metadata}
        return {(pp_rank, tp_rank): metadata}

    def _get_layerwise_kv_cache_memory_info(
        self,
        kv_cache_spec: dict[str, KVCacheSpec],
        extra_config: dict[str, Any],
    ) -> tuple[int, int, float]:
        if not kv_cache_spec:
            return 0, 0, 1.0
        base_layers = self.model_config.get_num_layers(self.parallel_config)
        physical_layers = {get_layerwise_physical_layer_index(layer_name, base_layers) for layer_name in kv_cache_spec}
        num_layers = len(physical_layers)
        if num_layers < base_layers:
            return num_layers, num_layers, 1.0
        reuse_layout = build_layerwise_reuse_layout(
            kv_cache_spec,
            base_layers,
            extra_config,
        )
        if not reuse_layout.has_layer_reuse:
            return num_layers, num_layers, 1.0
        num_buffer_assignments = len(reuse_layout.buffer_slots)

        logical_page_bytes = sum(spec.page_size_bytes for spec in kv_cache_spec.values())
        physical_page_bytes = 0
        for slot in reuse_layout.buffer_slots:
            physical_page_bytes += reuse_layout.layer_cache_specs[slot[0]].main.spec.page_size_bytes
            for layer in slot:
                indexer = reuse_layout.layer_cache_specs[layer].indexer
                if indexer is not None:
                    physical_page_bytes += indexer.spec.page_size_bytes
                    break
        return num_layers, num_buffer_assignments, logical_page_bytes / physical_page_bytes

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        kv_cache_spec = self.model_runner.get_kv_cache_spec()
        extra_config = get_gva_layerwise_config(self.vllm_config.kv_transfer_config)
        if extra_config is not None:
            self._gva_layerwise_memory_info = self._get_layerwise_kv_cache_memory_info(
                kv_cache_spec,
                extra_config,
            )
        if get_ascend_config().sparse_kv_offload_config.enabled:
            # reserve kv_cache_spec for sparse kv offload memory profile usage.
            self.kv_cache_spec = kv_cache_spec
        return kv_cache_spec

    def update_max_model_len(self, max_model_len: int) -> None:
        """Update max_model_len after auto-fit to NPU memory.

        This is called when max_model_len=-1 is used and the engine
        automatically determines the maximum context length that fits
        in GPU memory. Workers need to update their cached max_model_len
        to match the engine's decision.
        """
        self.model_config.max_model_len = max_model_len
        if self.model_runner is not None:
            self.model_runner.update_max_model_len(max_model_len)
        logger.debug("Updated max_model_len to %s", max_model_len)

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate NPU KV cache with the specified kv_cache_config."""
        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)
        if self.vllm_config.model_config.enable_sleep_mode:
            allocator = CaMemAllocator.get_instance()
            context = allocator.use_memory_pool(tag="kv_cache")
        else:
            from contextlib import nullcontext

            context = nullcontext()  # type: ignore
        with context:
            self.model_runner.initialize_kv_cache(kv_cache_config)

        # MRV2's scheduler emits new_block_ids_to_zero whenever this flag is
        # set, so its worker-side consumer must use the same condition. Keep the
        # narrower Eagle3 condition for MRV1, where zeroing was introduced only
        # for the multi-step speculative-decode reuse issue.
        speculative_config = self.vllm_config.speculative_config
        needs_mrv1_eagle_zeroing = (
            speculative_config is not None
            and speculative_config.method == "eagle3"
            and speculative_config.num_speculative_tokens > 1
        )
        should_init_kv_zeroer = kv_cache_config.needs_kv_cache_zeroing and (
            self.use_v2_model_runner or needs_mrv1_eagle_zeroing
        )
        # Keep bookkeeping buffers outside the sleep-mode KV-cache pool so they
        # survive sleep/wake cycles.
        if should_init_kv_zeroer and hasattr(self.model_runner, "_init_kv_zero_meta"):
            self.model_runner._init_kv_zero_meta()

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        # Check if profiling is enabled (RFC #6954 - align with upstream vLLM)
        if self.profiler_config is None or self.profiler_config.profiler is None:
            raise RuntimeError(
                "Profiling is not enabled. Please set --profiler-config to enable "
                "profiling. Example: "
                "'--profiler-config.profiler=torch --profiler-config.torch_profiler_dir"
                "=YOUR_DIR_PATH_TO_DUMP_TRACE'"
            )

        if is_start:
            from vllm.distributed.utils import get_worker_rank_suffix

            rank_suffix = get_worker_rank_suffix(global_rank=self.rank)
            trace_name = f"{profile_prefix}_{rank_suffix}" if profile_prefix else rank_suffix

            if self.profiler is None:
                self.profiler = TorchNPUProfilerWrapper(self.profiler_config, trace_name)
                logger.debug("Starting torch profiler with trace name: %s", trace_name)
                self.profiler.start()  # type: ignore[attr-defined]
            else:
                # Profiler already initialized. Restart profiling but keep
                # the original trace name from the first initialization.
                self.profiler.start()
        else:
            if self.profiler is None:
                logger.warning("Profiler was not started, nothing to stop.")
                return
            self.profiler.stop()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_runner.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def reset_encoder_cache(self) -> None:
        self.model_runner.reset_encoder_cache()

    def execute_dummy_batch(self) -> None:
        self.log_memory_stats()
        num_tokens = getattr(self.model_runner, "uniform_decode_query_len", 1)
        self.model_runner._dummy_run(num_tokens, uniform_decode=True)

    def _init_worker_distributed_environment(self) -> None:
        """Initialize the distributed environment."""
        init_batch_invariance()
        init_distributed_environment(
            self.parallel_config.world_size, self.rank, self.distributed_init_method, self.local_rank, "hccl"
        )
        ensure_model_parallel_initialized(
            self.parallel_config.tensor_parallel_size,
            self.parallel_config.pipeline_parallel_size,
            self.parallel_config.prefill_context_parallel_size,
            self.parallel_config.decode_context_parallel_size,
        )
        init_ascend_model_parallel(self.parallel_config)
        ensure_ec_transfer_initialized(self.vllm_config)

    def get_supported_pooling_tasks(self):
        return self.model_runner.get_supported_pooling_tasks()

    def get_supported_tasks(self) -> "tuple[SupportedTask, ...]":
        return self.model_runner.get_supported_tasks()

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.model_runner.take_draft_token_ids()

    def update_config(self, overrides: dict[str, Any]) -> None:
        self.model_runner.update_config(overrides)

    def reload_weights(self, *args, **kwargs) -> None:
        self.model_runner.reload_weights(*args, **kwargs)

    def check_health(self) -> None:
        import subprocess

        logger.debug("check_health starting for rank %s...", self.local_rank)
        try:
            result = subprocess.run(
                ["npu-smi", "info", "-i", str(self.local_rank), "-t", "health"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                parse_text_output(result.stdout)
                logger.debug("check_health success for rank %s.", self.local_rank)
            else:
                logger.warning("query NPU card %s fail: %s", self.local_rank, result.stderr)
        except subprocess.TimeoutExpired:
            logger.warning("query NPU card %s timeout.", self.local_rank)
        except FileNotFoundError:
            logger.warning("npu-smi tool not found.")
        except Exception as e:
            logger.error("query NPU card %s fail: %s", self.local_rank, e)
        return


def parse_text_output(output) -> None:
    lines = output.strip().split("\n")
    for i, line in enumerate(lines):
        line = line.strip()
        if "Health" in line:
            if line.split(":")[-1].strip() != "OK":
                raise RuntimeError("NPU card health status is not OK")
    return
