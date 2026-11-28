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
import logging
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from datetime import timedelta
from threading import RLock
from types import NoneType
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import torch
import torch_npu
from torch_npu.op_plugin.atb._atb_ops import _register_atb_extensions
from torch_npu.profiler import dynamic_profile as dp
from vllm import envs
from vllm.config import CUDAGraphMode, VllmConfig, set_current_vllm_config
from vllm.distributed import ensure_model_parallel_initialized, get_pcp_group, init_distributed_environment
from vllm.distributed.ec_transfer import (
    ensure_ec_transfer_initialized,
    ensure_ec_transfer_shutdown,
    get_ec_transfer,
    has_ec_transfer,
)
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_initialized,
    ensure_kv_transfer_shutdown,
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorHandshakeMetadata
from vllm.distributed.parallel_state import get_pp_group, get_tp_group, get_world_group
from vllm.logger import logger
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.mem_utils import MemorySnapshot, format_gib, memory_profiling
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.core.kv_cache_config_builder import get_kv_cache_config_builder
from vllm.v1.core.kv_cache_planning import unify_hybrid_kv_cache_specs
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker import gpu_worker
from vllm.v1.worker.gpu_worker import AsyncIntermediateTensors
from vllm.v1.worker.gpu_worker import Worker as GPUWorker
from vllm.v1.worker.startup_plan import (
    maybe_apply_startup_plan,
    maybe_save_startup_plan,
)
from vllm.v1.worker.worker_base import CompilationTimes
from vllm.v1.worker.workspace import init_workspace_manager

import vllm_ascend.device_allocator.sleep_mode_backend  # noqa: F401  (registers camem)
import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import KVPPConfig, get_ascend_config, init_ascend_config
from vllm_ascend.batch_invariant import init_batch_invariance
from vllm_ascend.compilation.static_kernel import initialize_static_kernel_runtime, release_static_kernels
from vllm_ascend.core import kv_cache_dtype_handlers  # noqa: F401
from vllm_ascend.core.kv_cache_placement import (
    KVPPPhysicalCachePlan,
    create_kvpp_cache_allocation_plan,
    register_kvpp_draft_layers,
)
from vllm_ascend.cpu_binding import bind_cpus
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.device_allocator.camem import CaMemAllocator
from vllm_ascend.device_allocator.sleep_mem_optimized import SleepWakeupManager
from vllm_ascend.distributed.eplb.eplb_state import AscendEplbState
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_context import uses_sfa_dspark_kv_transfer
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.buffer import DEVICE_BUFFER_ALIGNMENT
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cache_planner import (
    get_layerwise_gva_config,
    plan_layerwise_cache_reuse,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    apply_layerwise_kv_cache_plan,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    plan_sparse_kv_offload_memory,
)
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te
from vllm_ascend.distributed.parallel_state import init_ascend_model_parallel
from vllm_ascend.observability.runtime_guard.hooks import runtime_guard_idle_step
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper
from vllm_ascend.utils import (
    check_ascend_device_type,
    enable_custom_op,
    enable_sp,
    register_ascend_customop,
    register_ascend_fla_customop,
    register_device_print,
    setup_ascend_local_comm_res,
    shared_expert_tp_enable,
)

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig


_STATIC_KERNEL_SHUTDOWN_TIMEOUT = timedelta(seconds=3)


# Both supported versions resolve grouping through the platform's builder.
def get_kv_cache_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    return get_kv_cache_config_builder(vllm_config).get_kv_cache_groups(vllm_config, kv_cache_spec)


torch._dynamo.trace_rules.clear_lru_cache()  # noqa: E402
from torch._dynamo.variables import TorchInGraphFunctionVariable  # noqa: E402
from vllm.utils.torch_utils import set_random_seed  # noqa: E402

torch_non_c_binding_in_graph_functions_npu = dict.fromkeys(
    ["torch.npu.current_stream"],
    TorchInGraphFunctionVariable,
)  # noqa: E402
torch_non_c_binding_in_graph_functions_npu["torch.npu.stream"] = TorchInGraphFunctionVariable  # noqa: E402
torch._dynamo.trace_rules.torch_name_rule_map.append(torch_non_c_binding_in_graph_functions_npu)  # noqa: E402

# if true, allow tensor initialization and casting with internal format (e.g., NZ)
torch.npu.config.allow_internal_format = True


_GPU_WORKER_TRITON_LOCK = RLock()


@contextmanager
def allow_v2_model_runner_without_triton() -> Iterator[None]:
    """Relax the GPU worker's Triton check during Ascend construction."""
    # Serialize callers so overlapping constructors cannot restore the flag
    # while another constructor using this context is still running.
    with _GPU_WORKER_TRITON_LOCK:
        has_triton = getattr(gpu_worker, "HAS_TRITON", True)
        if has_triton:
            yield
            return
        gpu_worker.HAS_TRITON = True
        try:
            yield
        finally:
            gpu_worker.HAS_TRITON = has_triton


class NPUWorker(GPUWorker):
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

        from vllm_ascend.utils import adapt_patch

        adapt_patch()

        from vllm_ascend.patch.worker.patch_kda_recoverssm import patch_kda_recoverssm

        patch_kda_recoverssm(vllm_config)

        # Register ops when worker init.
        from vllm_ascend import ops

        ops.register_dummy_fusion_op()
        if get_current_hardware_profile().supports(HardwareCapability.ATB_EXTENSIONS):
            _register_atb_extensions()
        register_ascend_customop(vllm_config)
        # init ascend config and soc version
        ascend_config = init_ascend_config(vllm_config)
        from vllm_ascend.logger import configure_ascend_file_logging

        configure_ascend_file_logging(ascend_config.ascend_log_path)
        check_ascend_device_type()

        if getattr(vllm_config.parallel_config, "enable_fault_tolerance", False):
            raise NotImplementedError("Ascend does not support the GPU fault-tolerance sentinel.")

        # 310P runs Model Runner V2 without Triton. Replace this context manager
        # with a _validate_triton() override once all supported vLLM versions
        # include the upstream hook.
        # https://github.com/vllm-project/vllm/pull/68320
        with allow_v2_model_runner_without_triton():
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

        self.npugraph_memory_bytes = 0
        # Set once the RL extra cleanup has torn down HCCL groups and ACL graphs,
        # cleared when the KV cache wakes and the graphs are recaptured.
        self._extra_cleanup_pending = False
        self.sleep_wakeup_manager = SleepWakeupManager(vllm_config, self, lambda: getattr(self, "model_runner", None))

        # FixMe: this is a patch to fix the issue cause by https://github.com/vllm-project/vllm/commit/de94289a98d7ec52a5ef02719e01a1db8b505170
        from vllm.model_executor.layers.linear import WEIGHT_LOADER_V2_SUPPORTED

        if "UnquantizedLinearMethod" in WEIGHT_LOADER_V2_SUPPORTED:
            WEIGHT_LOADER_V2_SUPPORTED.remove("UnquantizedLinearMethod")

        self._static_kernel_runtime_initialized = False
        self._kvpp_cache_allocation_plan: KVPPPhysicalCachePlan | None = None

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

    def checkpoint_prepare(self) -> None:
        raise NotImplementedError("GPUWorker.checkpoint_prepare is not validated on Ascend.")

    def checkpoint_restore(self) -> None:
        raise NotImplementedError("GPUWorker.checkpoint_restore is not validated on Ascend.")

    def _scoped_allocator_max_split(self, max_split_size_mb: int):
        raise NotImplementedError("GPUWorker._scoped_allocator_max_split is not validated on Ascend.")

    def handle_ft_command(self, ft_request):
        raise NotImplementedError("GPUWorker.handle_ft_command is not validated on Ascend.")

    def _compile_or_warm_up_model(self) -> CompilationTimes:
        raise NotImplementedError("GPUWorker._compile_or_warm_up_model is not validated on Ascend.")

    def _maybe_activate_jit_monitor(self) -> None:
        raise NotImplementedError("GPUWorker._maybe_activate_jit_monitor is not validated on Ascend.")

    def _get_cudagraph_capture_context(self) -> AbstractContextManager[None]:
        raise NotImplementedError("GPUWorker._get_cudagraph_capture_context is not validated on Ascend.")

    def supports_draft_weight_updates(self) -> bool:
        return False

    def _set_draft_weight_update_target(self) -> None:
        raise NotImplementedError("GPUWorker._set_draft_weight_update_target is not validated on Ascend.")

    def get_compilation_match_table(self) -> dict[str, int]:
        raise NotImplementedError("GPUWorker.get_compilation_match_table is not validated on Ascend.")

    def get_encoder_timing_stats(self) -> dict[str, dict[str, float | int]]:
        raise NotImplementedError("GPUWorker.get_encoder_timing_stats is not validated on Ascend.")

    def annotate_profile(self, scheduler_output):
        raise NotImplementedError("GPUWorker.annotate_profile is not validated on Ascend.")

    @property
    def randomize_dummy_inputs(self) -> bool:
        raise NotImplementedError("GPUWorker.randomize_dummy_inputs is not validated on Ascend.")

    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        raise NotImplementedError("GPUWorker.save_sharded_state is not validated on Ascend.")

    def save_tensorized_model(self, tensorizer_config: "TensorizerConfig") -> None:
        raise NotImplementedError("GPUWorker.save_tensorized_model is not validated on Ascend.")

    def start_draft_weight_update(self) -> None:
        raise NotImplementedError("GPUWorker.start_draft_weight_update is not validated on Ascend.")

    def elastic_ep_execute(self, execute_method: str, *args, **kwargs):
        raise NotImplementedError("GPUWorker.elastic_ep_execute is not validated on Ascend.")

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

    def set_metrics_level(self, level: str) -> str:
        """Set this worker process's MS metric level. Called by collective RPC."""
        from vllm_ascend.observability.ms_metrics.control import apply_process_metrics_level

        return apply_process_metrics_level(level)

    def sleep(self, level: int = 1) -> None:
        # Resolve the backend before cleanup or KV unregistration so a bad
        # backend configuration leaves the worker state intact.
        backend = self.sleep_mode_backend
        self.synchronize_device()
        free_bytes_before_sleep = torch.npu.mem_get_info()[0]
        # Level 2 keeps everything the weight update does not rewrite; the
        # allocator backs up mixed/non-parameter weights-pool allocations.
        discardable = self._discardable_parameters(self.model_runner.model) if level == 2 else set()

        rl_config = get_ascend_config().rl_config
        cleanup_enabled = rl_config.enabled and rl_config.sleep_mode_extra_cleanup
        if cleanup_enabled:
            # Set first so the worker-side pending state survives a partial
            # teardown failure and is ready for wake_up() recovery once the
            # caller invokes it.
            self._extra_cleanup_pending = True
            self.sleep_wakeup_manager.sleep()
        self.synchronize_device()

        # Unregister Mooncake before the KV connector hook, matching discard()
        # (which unregisters Mooncake before delegating to super().discard()).
        # sleep() and wake_up() do not call super() yet, so they repeat the KV
        # connector hooks of upstream's GPUWorker by hand; RFC #22572 P6
        # converges both to super(), which drops these copies.
        global_te.unregister_buffer()
        if has_kv_transfer_group():
            get_kv_transfer_group().unregister_kv_caches()
        backend.suspend(discardable)
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

    def disconnect_kv_transfer_peers(self) -> None:
        self._drain_store_send_queues()
        global_te.disconnect_all_peers()

    def _drain_store_send_queues(self) -> None:
        if not has_kv_transfer_group():
            return
        connector = get_kv_transfer_group()
        children = getattr(connector, "_connectors", None)
        targets = children if children is not None else (connector,)
        for target in targets:
            drain = getattr(target, "drain_queued_saves", None)
            if drain is not None:
                drain()

    def wake_up(self, tags: list[str] | None = None) -> None:
        # Weights stay resident across a KV-only release/wake, so FRACTAL_NZ
        # weights are unaffected and only weight-bearing wakes reject NZ.
        wake_weights = tags is None or "weights" in tags
        if wake_weights:
            self._check_nz_disabled()
        self.sleep_mode_backend.resume(tags)
        if tags is None or "kv_cache" in tags:
            # Reverse the sleep/discard unregister order (and match future
            # super().wake_up() in RFC #22572 P6): restore KV connector caches
            # before re-registering Mooncake buffers.
            if has_kv_transfer_group():
                get_kv_transfer_group().restore_kv_caches()
            global_te.reregister_buffer()

        # Restore only what a sleep tore down: a discard releases the KV pool
        # but leaves HCCL groups and ACL graphs valid.
        if self._extra_cleanup_pending:
            self.sleep_wakeup_manager.wakeup(tags)
            # A weights-only wake restores HCCL but leaves graph recapture to
            # the KV-cache wake, so the restore is complete only then.
            if tags is None or "kv_cache" in tags:
                self._extra_cleanup_pending = False
        self.synchronize_device()

    def discard(self, tags: tuple[str, ...]) -> None:
        if "weights" in tags:
            # Fail before unregistering buffers or releasing memory if the
            # matching weights wake would reject FRACTAL_NZ.
            self._check_nz_disabled()
        if "kv_cache" in tags:
            # Mooncake hands KV-cache device addresses to the NIC for one-sided
            # reads. Unregister them before the memory is released, as sleep
            # does; wake_up(["kv_cache"]) registers them again.
            self.synchronize_device()
            global_te.unregister_buffer()
        super().discard(tags)

    def _check_nz_disabled(self) -> None:
        if get_ascend_config().weight_nz_mode:
            raise ValueError(
                "FRACTAL_NZ mode is enabled. This may cause model parameter "
                "precision issues in the RL scenarios. Please set weight_nz_mode=0 "
                "via --additional-config, or enable additional_config.rl_config "
                "(enabled: true) which disables NZ automatically."
            )

    def _start_weight_update(self, is_draft: bool = False) -> None:
        if is_draft:
            raise NotImplementedError("Draft weight updates are not supported on Ascend.")
        # Report a missing engine before checking the Ascend layout.
        self._check_weight_transfer_engine()
        self._check_nz_disabled()
        # The inherited lifecycle tracks session state, dispatches rank payloads
        # and refreshes the model runner once the target weights are in place.
        super()._start_weight_update()

    def shutdown(self) -> None:
        model = getattr(getattr(self, "model_runner", None), "model", None)
        close_engram = getattr(model, "close_shared_engram", None)
        engram_error = None
        if callable(close_engram):
            try:
                close_engram()
            except Exception as exc:
                engram_error = exc
        state = getattr(getattr(self, "model_runner", None), "eplb_state", None)
        if isinstance(state, AscendEplbState):
            state.close()
        if ensure_kv_transfer_shutdown is not None:
            ensure_kv_transfer_shutdown()
        if ensure_ec_transfer_shutdown is not None:
            ensure_ec_transfer_shutdown()

        if (profiler := self.profiler) is not None:  # type: ignore[has-type]
            profiler.shutdown()

        if weight_transfer_engine := getattr(self, "weight_transfer_engine", None):
            weight_transfer_engine.shutdown()

        if model_runner := getattr(self, "model_runner", None):
            from vllm_ascend.worker.v2.kv_connector import C8VScaleKVConnector

            if isinstance(getattr(model_runner, "kv_connector", None), C8VScaleKVConnector):
                model_runner.kv_connector = model_runner.kv_connector.release_cache_views()
            shutdown_fn = getattr(model_runner, "shutdown", None)
            if callable(shutdown_fn):
                shutdown_fn()
        if self._static_kernel_runtime_initialized:
            self._release_static_kernel_runtime()
        if engram_error is not None:
            raise engram_error

    def _release_static_kernel_runtime(self) -> None:
        world_group = get_world_group()
        try:
            # Bound the collective itself so group teardown can finish after the local wait times out.
            work = world_group.cpu_group.barrier(timeout=_STATIC_KERNEL_SHUTDOWN_TIMEOUT)
            work.wait(timeout=_STATIC_KERNEL_SHUTDOWN_TIMEOUT)
        except RuntimeError as exc:
            logger.warning(
                "Rank %d could not confirm graph release. Static kernel packages may remain under "
                "static_kernel_install/ in the worker's working directory. After all workers in this instance "
                "have exited, remove only its <timestamp>_<pid> subdirectory. Gloo reported: %s",
                world_group.rank,
                exc,
            )
            return

        release_static_kernels()
        self._static_kernel_runtime_initialized = False

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def _get_gpu_memory_utilization(self) -> float:
        utilization = self.cache_config.gpu_memory_utilization
        # vLLM #56492 defers the default until the KV allocation mode is known.
        if utilization is None:
            return self.cache_config.resolved_gpu_memory_utilization
        return utilization

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

        if enable_custom_op():
            register_device_print()

        # Import _inductor for graph mode execution with triton
        # This lazy import avoids torch_npu re-initialization in patch
        # Note that this should be imported after torch.npu.set_device
        # to avoid repeated set_device in extra processes
        from vllm.triton_utils import HAS_TRITON

        if HAS_TRITON:
            import torch_npu._inductor  # noqa: F401

        gc.collect()
        torch.npu.empty_cache()

        if get_current_hardware_profile().supports(HardwareCapability.LOCAL_KV_COMM_RESOURCE):
            setup_ascend_local_comm_res(visible_device_index, self.vllm_config.kv_transfer_config)

        # take current memory snapshot
        self.init_snapshot = MemorySnapshot(device=device)
        self.requested_memory = self.init_snapshot.total_memory * self._get_gpu_memory_utilization()
        if self.init_snapshot.free_memory < self.requested_memory:
            GiB = lambda b: round(b / GiB_bytes, 2)
            raise ValueError(
                f"Free memory on device "
                f"({GiB(self.init_snapshot.free_memory)}/"
                f"{GiB(self.init_snapshot.total_memory)} GiB) on startup "
                f"is less than desired GPU memory utilization "
                f"({self._get_gpu_memory_utilization()}, "
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
        compilation = get_ascend_config().ascend_compilation_config
        static_kernel_compilation = "disabled"
        if compilation.enable_static_kernel:
            static_kernel_compilation = (
                "enabled through model compilation" if compilation.enable_npugraph_ex else "enabled during graph warmup"
            )
        logger.info_once(
            "Static kernel compilation is %s. SuperKernel is %s.",
            static_kernel_compilation,
            "enabled" if compilation.enable_super_kernel else "disabled",
            scope="global",
        )
        # FLA import-time capability queries must run on the selected worker
        # device, and registration must finish before any model is constructed.
        # TODO: Drop this separate call when supported vLLM FLA or Triton Ascend
        # versions fix import-time NPU init and general registration is restored.
        register_ascend_fla_customop()

        self._init_workspace_and_model_runner()

    def _init_workspace_and_model_runner(self) -> None:
        # Initialize workspace manager
        num_ubatches = 1
        init_workspace_manager(self.device, num_ubatches)
        self.model_runner = self._make_model_runner()

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)

    def _make_model_runner(self):
        if self.vllm_config.is_mm_encoder_only:
            from vllm.v1.worker.mm_encoder_model_runner import MMEncoderModelRunner

            from vllm_ascend.worker.v2.utils import torch_cuda_wrapper

            with torch_cuda_wrapper():
                return MMEncoderModelRunner(self.vllm_config, self.device)
        from vllm_ascend.worker.v2.model_runner import NPUModelRunner

        model_runner = NPUModelRunner(self.vllm_config, self.device)
        compilation = get_ascend_config().ascend_compilation_config
        if compilation.enable_static_kernel and not compilation.enable_npugraph_ex:
            initialize_static_kernel_runtime(self.vllm_config, capture=True)
            self._static_kernel_runtime_initialized = True
        return model_runner

    def _apply_kv_offload_decode_memory_constraints(
        self,
        available_device_memory_bytes: int,
    ) -> int:
        GiB = lambda b: b / GiB_bytes
        sparse_kv_offload_config = get_ascend_config().sparse_kv_offload_config
        if not sparse_kv_offload_config.enabled:
            return int(available_device_memory_bytes)

        kv_cache_spec = getattr(self, "kv_cache_spec", None) or self.get_kv_cache_spec()
        dram_limit_bytes = int(sparse_kv_offload_config.dram_size_per_dp_GB * GiB_bytes)
        budget = plan_sparse_kv_offload_memory(
            kv_cache_spec=kv_cache_spec,
            vllm_config=self.vllm_config,
            available_device_memory_bytes=available_device_memory_bytes,
            dram_limit_bytes=dram_limit_bytes,
            keep_device_kv_cache=sparse_kv_offload_config.keep_device_kv_cache,
        )
        logger.info_once(
            "Sparse KV offload memory plan: npu_limit=%d blocks, "
            "dram_limit=%d blocks, workload_limit=%d blocks, "
            "final=%d blocks (%s limited), planner=%.2f GiB, "
            "host=%.2f GiB, device=%.2f GiB, alignment_reserve=%.2f GiB.",
            budget.npu_limit_blocks,
            budget.dram_limit_blocks,
            budget.workload_limit_blocks,
            budget.final_num_blocks,
            budget.limiting_factor,
            GiB(budget.final_planner_bytes),
            GiB(budget.planned_host_bytes),
            GiB(budget.planned_device_bytes),
            GiB(budget.host_alignment_reserve_bytes),
            scope="local",
        )
        return int(budget.final_planner_bytes)

    def _apply_kvpp_memory_budget(self, available_bytes: int) -> int:
        self.available_kv_cache_memory_bytes = available_bytes
        plan = self._kvpp_cache_allocation_plan
        if plan is None:
            return available_bytes
        num_blocks = plan.get_num_blocks(available_bytes)
        return num_blocks * sum(spec.page_size_bytes for spec in plan.logical_cache_spec.values())

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Profiles the peak memory usage of the model to determine how much
        memory can be used for KV cache without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculates the free memory that can be used for KV cache in
        bytes.
        """
        GiB = lambda b: b / GiB_bytes

        maybe_apply_startup_plan(self)

        if self.vllm_config.attention_config.hisparse_config is not None:
            from vllm_ascend.hisparse.backend import prepare_hisparse_workspaces

            prepare_hisparse_workspaces(self.vllm_config.compilation_config.static_forward_context)

        randomize_inputs = envs.VLLM_RANDOMIZE_DP_DUMMY_INPUTS and self.parallel_config.data_parallel_size > 1

        # Fast path: user has explicitly specified KV cache size via
        # --kv-cache-memory. Still run profile_run() to compile the model,
        # but skip the memory profiling calculation entirely.
        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
            with set_current_vllm_config(self.vllm_config):
                self.model_runner.profile_run(randomize_inputs=randomize_inputs)
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
            # Explicit and startup-plan budgets are physical bytes too. Apply
            # the same private multi-group allocation correction as profiling
            # before the offload and KVPP budget conversions.
            kv_cache_memory_bytes = self._scale_kv_cache_memory_for_multi_group(kv_cache_memory_bytes)
            return self._apply_kvpp_memory_budget(
                self._apply_kv_offload_decode_memory_constraints(kv_cache_memory_bytes)
            )

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        with (
            memory_profiling(
                self.init_snapshot,
                weights_memory=int(self.model_runner.model_memory_usage),
            ) as profile_result,
            set_current_vllm_config(self.vllm_config),
        ):
            self.model_runner.profile_run(randomize_inputs=randomize_inputs)

            # Record torch peak INSIDE the context and BEFORE graph capture,
            # so that graph pool allocations don't inflate the activation peak.
            # The memory_profiling context will also compute torch_peak_increase
            # on exit, but we override it below with this pre-graph value.
            profile_torch_peak = torch.npu.memory_stats(self.device).get("allocated_bytes.all.peak", 0)

        # Override torch_peak_increase with the pre-graph-capture value to
        # avoid double-counting graph pool memory as activation memory.
        profile_result.torch_peak_increase = profile_torch_peak - profile_result.before_profile.torch_peak
        if self.vllm_config.attention_config.hisparse_config is None:
            profile_result.non_kv_cache_memory = (
                profile_result.non_torch_increase + profile_result.torch_peak_increase + profile_result.weights_memory
            )
        else:
            # Reuse Core's measured total, including owners retained from an
            # earlier profile. Preserve this authority for startup suggestions.
            self._hisparse_profiled_non_kv_memory = profile_result.non_kv_cache_memory

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
        self.available_kv_cache_memory_bytes = self._scale_kv_cache_memory_for_multi_group(
            self.available_kv_cache_memory_bytes,
        )

        logger.debug(profile_result)
        logger.info_once(
            "Available KV cache memory: %.2f GiB", GiB(self.available_kv_cache_memory_bytes), scope="local"
        )
        self.available_kv_cache_memory_bytes = self._apply_kv_offload_decode_memory_constraints(
            self.available_kv_cache_memory_bytes
        )
        return self._apply_kvpp_memory_budget(self.available_kv_cache_memory_bytes)

    def _scale_kv_cache_memory_for_multi_group(self, available_memory: int) -> int:
        """Scale the KV cache budget for vllm main's multi-group layout.

        vLLM #51718 derives num_blocks from the largest KV cache group's
        bytes-per-block, but some Ascend runners keep per-layer contiguous
        buffers for every group. Per-layer sizing then totals
        num_blocks * (sum of ALL groups' pages), which exceeds available
        memory whenever more than one group is non-trivial. Scale the
        advertised budget by bytes_per_block / sum(pages) so the engine
        derives a num_blocks (and block pool) small enough for the per-layer
        buffers to fit.
        """
        if self.vllm_config.attention_config.hisparse_config is not None:
            # The combined worker allocates one shared slab with alignment
            # headroom after Core plans the number of blocks.
            transfer = self.vllm_config.kv_transfer_config
            if transfer is not None and transfer.has_connector("AscendStoreConnector"):
                return max(0, available_memory - (DEVICE_BUFFER_ALIGNMENT - 1))
            return available_memory
        kv_cache_spec = self.get_kv_cache_spec()
        if not isinstance(kv_cache_spec, dict):
            return available_memory
        kv_cache_groups = get_kv_cache_groups(self.vllm_config, kv_cache_spec)
        if not kv_cache_groups:
            return available_memory
        # Ascend's DSV4/V4.1 and GLM-Next planners already account for shared
        # physical storage across all groups. Their budgets must not take the
        # generic per-layer scale, including when the runner uses sparse SFA.
        for group in kv_cache_groups:
            group_spec = group.kv_cache_spec
            specs = (
                group_spec.kv_cache_specs.values() if isinstance(group_spec, UniformTypeKVCacheSpecs) else (group_spec,)
            )
            if any(
                getattr(spec, "model_version", None) in {"deepseek_v4", "deepseek_v41", "glm5_next"} for spec in specs
            ):
                return available_memory

        # vLLM #51718 overlays KV cache groups in one standardized backing
        # allocation. For block-compact layouts, Ascend can
        # preserve that contract for attention/Mamba and eligible full/sliding
        # attention models through supported cache views. Do not
        # shrink the planner budget when the runner can consume that layout.
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
        shares_attention_pool = False
        # V2 is the only model runner; its KV-cache helpers apply unconditionally.
        from vllm_ascend.worker.v2.attn_utils import can_share_full_and_sliding_kv_cache

        shares_attention_pool = can_share_full_and_sliding_kv_cache(self.vllm_config, per_layer_specs)
        # TODO: The Step-3.5/3.7 models will be sunset in the future. Once they
        # are, remove this is_step_model exemption: this line and the
        # `or is_step_model` budget gate below.
        is_step_model = self.model_config.hf_text_config.model_type in (
            "step3p5",
            "step3p7",
            "step5",
            "step5_vl",
            "step5_mtp",
        )
        if (
            has_attention
            and (has_mamba or is_step_model or shares_attention_pool)
            and layout.is_block_compact
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
                "Ascend per-layer KV layout scales the multi-group budget by %.4f "
                "(%d bytes/block over %d total page bytes) so per-layer "
                "buffers fit within device memory.",
                scale,
                bytes_per_block,
                sum_pages,
            )
            return int(available_memory * scale)
        return available_memory

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

        if self._pp_send_work:  # type: ignore[has-type]
            for handle in self._pp_send_work:  # type: ignore[has-type]
                handle.wait()
            self._pp_send_work = []  # type: ignore[var-annotated]

        intermediate_tensors = None
        forward_pass = scheduler_output.total_num_scheduled_tokens > 0
        if forward_pass and not get_pp_group().is_first_rank:
            if enable_sp(self.vllm_config):
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

        if (profiler := self.profiler) is not None:  # type: ignore[has-type]
            profiler.step()

        output = self.model_runner.execute_model(scheduler_output, intermediate_tensors)
        if self.model_runner.is_pooling_model and output is None:
            output = self.model_runner.pool()  # type: ignore
        if isinstance(output, (ModelRunnerOutput, AsyncModelRunnerOutput, NoneType)):
            return output

        assert isinstance(output, IntermediateTensors)
        parallel_config = self.vllm_config.parallel_config
        assert parallel_config.distributed_executor_backend != ("external_launcher") and not get_pp_group().is_last_rank
        if enable_sp(self.vllm_config):
            all_gather_group = None
        else:
            all_gather_group = get_tp_group()
        self._pp_send_work = get_pp_group().isend_tensor_dict(
            output.tensors,
            all_gather_group=all_gather_group,
        )

        # Model Runner V2 has no kv_connector_output to propagate from
        # non-last PP ranks.
        return None

    @torch.inference_mode()
    def sample_tokens(self, grammar_output: "GrammarOutput") -> ModelRunnerOutput | AsyncModelRunnerOutput:
        output = self.model_runner.sample_tokens(grammar_output)
        return output

    def load_model(self) -> None:
        if self.vllm_config.model_config.enable_sleep_mode:
            allocator = CaMemAllocator.get_instance()
            assert allocator.get_current_usage() == 0, "Sleep mode can only be used for one instance per process."
            context = allocator.use_memory_pool(tag="weights")
        else:
            context = nullcontext()  # type: ignore

        # vLLM #57516 reclaims cached memory inside `load_model` via
        # `torch.accelerator.empty_cache()`. Inside the sleep-mode pluggable
        # memory pool `torch.npu.empty_cache()` trips "captures_underway.empty()",
        # so neutralize it while weights load on both supported lanes; the pool
        # is reclaimed by sleep()/wake_up().
        empty_cache_context: AbstractContextManager[Any]
        if self.vllm_config.model_config.enable_sleep_mode:
            empty_cache_context = patch("torch.npu.empty_cache", lambda: None)
        else:
            empty_cache_context = nullcontext()

        with context, empty_cache_context, set_current_vllm_config(self.vllm_config):
            self.model_runner.load_model()

        if shared_expert_tp_enable():
            from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner

            if not any(
                isinstance(module, AscendMoERunner)
                and module.ascend_shared_experts is not None
                and module.ascend_shared_experts.finegrained_tp_group is not None
                for module in self.get_model().modules()
            ):
                raise ValueError("shared_expert_tensor_parallel_size > 1 requires a layer using shared-expert TP.")

        if has_ec_transfer():
            get_ec_transfer().start_worker_services()

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
            if self.vllm_config.attention_config.hisparse_config is not None:
                non_kv_memory = self._hisparse_profiled_non_kv_memory + npugraph_memory_bytes
            self.npugraph_memory_bytes = npugraph_memory_bytes
            suggested_to_requested = int(self.requested_memory) - non_kv_memory - redundancy_buffer
            suggested_to_gpu_limit = int(self.init_snapshot.free_memory) - non_kv_memory - redundancy_buffer
            msg = (
                f"Free memory on device "
                f"({format_gib(self.init_snapshot.free_memory)}/"
                f"{format_gib(self.init_snapshot.total_memory)} GiB) on startup. "
                f"Desired GPU memory utilization is "
                f"({self._get_gpu_memory_utilization()}, "
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

            if suggested_to_requested > 0:
                maybe_save_startup_plan(self, suggested_to_requested)

        # Call ATB matmul to warm up; otherwise, the first operation (ReshapeAndCache)
        # may cause performance degradation at runtime.
        if get_current_hardware_profile().supports(HardwareCapability.ATB_WARMUP):
            self._warm_up_atb()
        # Keep thread affinity after warmup and capture. Engram HOST_UVA tables
        # are already registered here; process-wide migration must not revisit
        # their pinned backing, which may also be shared across NUMA nodes.
        if get_ascend_config().enable_cpu_binding:
            engram_config = getattr(self.vllm_config, "engram_config", None)
            try:
                bind_cpus(
                    self.local_rank,
                    npu_id=current_platform.device_id_to_physical_device_id(self.local_rank),
                    migrate_memory=not (engram_config is not None and engram_config.cpu_offload),
                )
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

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        kv_cache_spec = self.model_runner.get_kv_cache_spec()
        kvpp_config = KVPPConfig.from_vllm_config(self.vllm_config)
        if kvpp_config.size > 1:
            register_kvpp_draft_layers(
                self.vllm_config,
                self.model_runner,
                kv_cache_spec,
                is_last_pp_rank=get_pp_group().is_last_rank,
            )
            speculative_config = self.vllm_config.speculative_config
            if (
                speculative_config is not None
                and speculative_config.use_eagle()
                and any(isinstance(spec, SlidingWindowSpec) for spec in kv_cache_spec.values())
            ):
                # Use the same full-allocation specs for KVPP budgeting and
                # the engine's cache groups. Attention compute stays windowed.
                kv_cache_spec = dict(kv_cache_spec)
                unify_hybrid_kv_cache_specs(kv_cache_spec)
            kvpp_rank = (
                get_pcp_group().rank_in_group * self.vllm_config.parallel_config.tensor_parallel_size
                + get_tp_group().rank_in_group
            )
            self._kvpp_cache_allocation_plan = create_kvpp_cache_allocation_plan(
                self.vllm_config,
                kv_cache_spec,
                kvpp_rank,
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

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        """Return the sleep-mode pool for ``tag``, or a no-op context."""
        if self.vllm_config.model_config.enable_sleep_mode:
            allocator = CaMemAllocator.get_instance()
            return allocator.use_memory_pool(tag=tag)
        return nullcontext()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate NPU KV cache with the specified kv_cache_config."""
        speculator = getattr(self.model_runner, "speculator", None)
        speculative = self.vllm_config.speculative_config
        if (
            speculative is not None
            and speculative.method == "dspark"
            and speculator is not None
            and (
                uses_sfa_dspark_kv_transfer(self.vllm_config)
                or get_layerwise_gva_config(self.vllm_config.kv_transfer_config) is not None
            )
        ):
            # Connector construction must see loader-owned draft caches before
            # planning target layerwise scratch buffers or pool transfers.
            kv_cache_config.dspark_draft_layer_names = tuple(sorted(speculator.draft_attn_layer_names))
        if (
            speculative is not None
            and speculative.method == "dflash"
            and speculator is not None
            and get_layerwise_gva_config(self.vllm_config.kv_transfer_config) is not None
        ):
            kv_cache_config.dflash_draft_layer_names = tuple(sorted(speculator.draft_attn_layer_names))
        # Plan from this rank's native descriptors before connectors bind to the cache.
        reuse_layout = plan_layerwise_cache_reuse(kv_cache_config, self.vllm_config)
        if reuse_layout is not None:
            kv_cache_config = apply_layerwise_kv_cache_plan(
                kv_cache_config, self.vllm_config.kv_transfer_config, reuse_layout
            )
            logger.info(
                "Layerwise reuse uses one %d-byte device backing; independent layers=%s.",
                reuse_layout.kv_cache_tensors[0].size,
                reuse_layout.independent_layers,
            )
        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)
        # Restrict the discardable kv_cache pool to backing cache allocations.
        # Persistent metadata created during initialize_kv_cache must stay
        # outside this pool so sleep/wake does not drop its contents.
        self.model_runner.initialize_kv_cache(
            kv_cache_config,
            kv_cache_allocation_context=self._maybe_get_memory_pool_context(tag="kv_cache"),
        )

        from vllm_ascend.worker.v2.kv_connector import maybe_wrap_c8_v_scale_connector

        maybe_wrap_c8_v_scale_connector(self.model_runner, kv_cache_config, self.vllm_config.kv_transfer_config)

        # MRV2's scheduler emits new_block_ids_to_zero whenever this flag is
        # set, so its worker-side consumer must use the same condition.
        should_init_kv_zeroer = kv_cache_config.needs_kv_cache_zeroing
        # Keep bookkeeping buffers outside the sleep-mode KV-cache pool so they
        # survive sleep/wake cycles.
        if should_init_kv_zeroer and hasattr(self.model_runner, "_init_kv_zero_meta"):
            self.model_runner._init_kv_zero_meta()

    def profile(
        self,
        is_start: bool = True,
        profile_prefix: str | None = None,
        *,
        delay_iterations: int | None = None,
        max_iterations: int | None = None,
        profiler_kwargs: dict | None = None,
    ):
        # Check if profiling is enabled (RFC #6954 - align with upstream vLLM)
        if self.profiler_config is None or self.profiler_config.profiler is None:
            raise RuntimeError(
                "Profiling is not enabled. Please set --profiler-config to enable "
                "profiling. Example: "
                "'--profiler-config.profiler=torch --profiler-config.torch_profiler_dir"
                "=YOUR_DIR_PATH_TO_DUMP_TRACE'"
            )

        profiler = self.profiler  # type: ignore[has-type]
        if is_start:
            from vllm.distributed.utils import get_worker_rank_suffix

            rank_suffix = get_worker_rank_suffix(global_rank=self.rank)
            trace_name = f"{profile_prefix}_{rank_suffix}" if profile_prefix else rank_suffix

            if profiler is None:
                profiler = TorchNPUProfilerWrapper(self.profiler_config, trace_name)
                self.profiler = profiler
                logger.debug("Starting torch profiler with trace name: %s", trace_name)
            # Else: profiler already initialized. Restart profiling but keep the
            # original trace name from the first initialization.
            # vLLM main supports per-session profiler kwargs; the release
            # wrapper has no slot for them.
            profiler.set_profiler_kwargs(profiler_kwargs)
            profiler.start(delay_iterations=delay_iterations, max_iterations=max_iterations)
        else:
            if profiler is None:
                logger.warning("Profiler was not started, nothing to stop.")
                return
            profiler.stop()

    def synchronize_device(self) -> None:
        torch.npu.synchronize()

    def get_encoder_cudagraph_stats(self) -> dict[str, Any] | None:
        """Return serializable encoder graph state for diagnostics.

        Utility RPCs use safe serialization by default, so callers must invoke
        a worker method by name instead of sending a Python callback. Keep the
        result limited to primitive containers supported by that serializer.
        """
        model_state = getattr(self.model_runner, "model_state", None)
        encoder_runner = getattr(model_state, "encoder_runner", None)
        manager = getattr(encoder_runner, "cudagraph_manager", None)
        if manager is None:
            # Model Runner V1 owns the manager directly.
            manager = getattr(self.model_runner, "encoder_cudagraph_manager", None)
        if manager is None:
            return None
        return {
            "manager": type(manager).__name__,
            "captured": manager.is_captured(),
            **manager.get_cumulative_stats(),
        }

    # Lockstep wave sync for idle DP ranks (never burns manual_dump);
    # lives in the runtime_guard hook, see runtime_guard_idle_step.
    @runtime_guard_idle_step
    def execute_dummy_batch(self) -> None:
        self.log_memory_stats()
        num_tokens = self.model_runner.decode_query_len
        # Idle DP worker RPCs also need the current config for replay metadata.
        with set_current_vllm_config(self.vllm_config):
            self.model_runner._dummy_run(num_tokens, uniform_decode=True, skip_gdn_state_update=True)

    def _init_worker_distributed_environment(self) -> None:
        """Initialize the distributed environment."""
        init_batch_invariance()
        init_distributed_environment(
            self.parallel_config.world_size, self.rank, self.distributed_init_method, self.local_rank, "hccl"
        )
        group_config = self.vllm_config
        if group_config.engram_config is not None and self.parallel_config.prefill_context_parallel_size > 1:
            # Skip upstream's DP-only Engram group so Ascend can create
            # _ENGRAM_DP once with DP x PCP ranks. Copy the config to preserve
            # the original Engram settings for Ascend and model initialization.
            group_config = copy.copy(group_config)
            group_config.engram_config = None
        with set_current_vllm_config(group_config):
            ensure_model_parallel_initialized(
                self.parallel_config.tensor_parallel_size,
                self.parallel_config.pipeline_parallel_size,
                self.parallel_config.prefill_context_parallel_size,
                self.parallel_config.decode_context_parallel_size,
            )
        init_ascend_model_parallel(self.parallel_config, self.vllm_config.engram_config)
        ensure_ec_transfer_initialized(self.vllm_config)

    def get_supported_pooling_tasks(self):
        return self.model_runner.get_supported_pooling_tasks()

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
