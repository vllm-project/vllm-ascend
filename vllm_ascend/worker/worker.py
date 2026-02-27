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
from types import NoneType

import torch
import torch.nn as nn
import torch_npu
import vllm.envs as envs_vllm
from torch_npu.op_plugin.atb._atb_ops import _register_atb_extensions
from torch_npu.profiler import dynamic_profile as dp
from vllm.config import CUDAGraphMode, VllmConfig, set_current_vllm_config
from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
from vllm.distributed.ec_transfer import ensure_ec_transfer_initialized
from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized, get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.logger import logger
from vllm.lora.request import LoRARequest
from vllm.sequence import IntermediateTensors
from vllm.tasks import SupportedTask
from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.mem_utils import MemorySnapshot, memory_profiling
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, AsyncModelRunnerOutput, DraftTokenIds, ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerBase
from vllm.v1.worker.workspace import init_workspace_manager

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config
from vllm_ascend.batch_invariant import init_batch_invariance
from vllm_ascend.cpu_binding import bind_cpus
from vllm_ascend.device_allocator.camem import CaMemAllocator
from vllm_ascend.distributed.parallel_state import init_ascend_model_parallel
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.utils import (
    AscendDeviceType,
    check_ascend_device_type,
    enable_flash_comm_v1,
    get_ascend_device_type,
    register_ascend_customop,
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
        # Import _inductor for graph mode execution with triton
        # This lazy import avoids torch_npu re-initialization in patch
        from vllm.triton_utils import HAS_TRITON

        if HAS_TRITON:
            import torch_npu._inductor  # noqa: F401
        # Register ops when worker init.
        from vllm_ascend import ops

        ops.register_dummy_fusion_op()
        if get_ascend_device_type() != AscendDeviceType.A5:
            _register_atb_extensions()
        register_ascend_customop(vllm_config)
        # init ascend config and soc version
        init_ascend_config(vllm_config)
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

        self.profiler = self._init_profiler()
        if vllm_config.model_config and vllm_config.model_config.enable_sleep_mode:
            # Buffers saved before sleep
            self._sleep_saved_buffers: dict[str, torch.Tensor] = {}

        # FixMe: this is a patch to fix the issue cause by https://github.com/vllm-project/vllm/commit/de94289a98d7ec52a5ef02719e01a1db8b505170
        from vllm.model_executor.layers.linear import WEIGHT_LOADER_V2_SUPPORTED

        if "UnquantizedLinearMethod" in WEIGHT_LOADER_V2_SUPPORTED:
            WEIGHT_LOADER_V2_SUPPORTED.remove("UnquantizedLinearMethod")

        self.use_v2_model_runner = envs_vllm.VLLM_USE_V2_MODEL_RUNNER

        npugraph_ex_config = get_ascend_config().npugraph_ex_config
        if npugraph_ex_config.enable and npugraph_ex_config.enable_static_kernel:
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
        # Save the buffers before level 2 sleep
        if level == 2:
            model = self.model_runner.model
            self._sleep_saved_buffers = {name: buffer.cpu().clone() for name, buffer in model.named_buffers()}
        allocator = CaMemAllocator.get_instance()
        allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())
        free_bytes_after_sleep, total = torch.npu.mem_get_info()
        freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        used_bytes = total - free_bytes_after_sleep
        assert freed_bytes >= 0, "Memory usage increased after sleeping."
        logger.info(
            "Sleep mode freed %.2f GiB memory, %.2f GiB memory is still in use.",
            freed_bytes / GiB_bytes,
            used_bytes / GiB_bytes,
        )

    def wake_up(self, tags: list[str] | None = None) -> None:
        if envs_ascend.VLLM_ASCEND_ENABLE_NZ:
            raise ValueError(
                "FRACTAL_NZ mode is enabled. This may cause model parameter precision issues "
                "in the RL scenarios. Please set VLLM_ASCEND_ENABLE_NZ=0."
            )
        allocator = CaMemAllocator.get_instance()
        allocator.wake_up(tags=tags)

        hidden_size = self.vllm_config.model_config.hf_text_config.hidden_size
        model = self.model_runner.model
        if tags is None or "weights" in tags:
            for name, param in model.named_parameters():
                if "w2_weight" in name and param.shape[2] == hidden_size:
                    parts = name.split(".")
                    param_name = parts[-1]
                    parent_module = model.get_submodule(".".join(parts[:-1]))

                    w2_data = param.transpose(1, 2)
                    w2_data = torch.nn.Parameter(w2_data, requires_grad=False)
                    setattr(parent_module, param_name, w2_data)
                elif "w13_weight" in name and param.shape[1] == hidden_size:
                    parts = name.split(".")
                    param_name = parts[-1]
                    parent_module = model.get_submodule(".".join(parts[:-1]))

                    w13_data = param.transpose(1, 2)
                    w13_data = torch.nn.Parameter(w13_data, requires_grad=False)
                    setattr(parent_module, param_name, w13_data)

        # Restore the buffers after level 2 sleep
        if len(self._sleep_saved_buffers):
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def _init_device(self):
        device = torch.device(f"npu:{self.local_rank}")
        torch.npu.set_device(device)

        gc.collect()
        torch.npu.empty_cache()

        # take current memory snapshot
        self.init_snapshot = MemorySnapshot()
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

        # binding cpu
        if get_ascend_config().enable_cpu_binding:
            try:
                bind_cpus(self.local_rank)
            except Exception as e:
                logger.warning(f"Bind cpus failed in rank{self.local_rank}: {e} Skip binding cpu.")
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
        self.available_kv_cache_memory_bytes = (
            self.requested_memory - profile_result.non_kv_cache_memory - non_torch_memory_cleared_by_empty_cache
        )

        logger.debug(profile_result)
        logger.info_once(
            "Available KV cache memory: %.2f GiB",
            GiB(self.available_kv_cache_memory_bytes),
            scope="local",
        )
        return int(self.available_kv_cache_memory_bytes)

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        # enable msMonitor to monitor the performance of vllm-ascend
        if envs_ascend.MSMONITOR_USE_DAEMON:
            dp.step()

        intermediate_tensors = None
        forward_pass = scheduler_output.total_num_scheduled_tokens > 0
        if forward_pass and not get_pp_group().is_first_rank:
            # If flashcomm1 is used, this all_gather_group parameter needs to be removed, otherwise
            # it will conflict with the all-gather operation in flashcomm1.
            if enable_flash_comm_v1():
                all_gather_group = None
            else:
                all_gather_group = get_tp_group()
            intermediate_tensors = IntermediateTensors(
                get_pp_group().recv_tensor_dict(all_gather_group=all_gather_group)
            )

        output = self.model_runner.execute_model(scheduler_output, intermediate_tensors)
        if isinstance(output, (ModelRunnerOutput, AsyncModelRunnerOutput, NoneType)):
            return output

        assert isinstance(output, IntermediateTensors)
        parallel_config = self.vllm_config.parallel_config
        assert parallel_config.distributed_executor_backend != ("external_launcher") and not get_pp_group().is_last_rank
        # If flashcomm1 is used, this all_gather_group parameter needs to be removed, otherwise
        # it will conflict with the all-gather operation in flashcomm1.
        if enable_flash_comm_v1():
            all_gather_group = None
        else:
            all_gather_group = get_tp_group()
        get_pp_group().send_tensor_dict(output.tensors, all_gather_group=all_gather_group)

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
        return self.model_runner.sample_tokens(grammar_output)

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

    def compile_or_warm_up_model(self) -> None:
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
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model()
        # Call ATB matmul to warm up; otherwise, the first operation (ReshapeAndCache)
        # may cause performance degradation at runtime.
        if get_ascend_device_type() != AscendDeviceType.A5:
            self._warm_up_atb()
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    def _warm_up_atb(self):
        x = torch.rand((2, 4), dtype=torch.float16).npu()
        weight = torch.rand((2, 4), dtype=torch.float16).npu()
        c = torch.rand((4, 4), dtype=torch.float32).npu()
        torch_npu._npu_matmul_add_fp32(x, weight, c)

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    def get_kv_connector_handshake_metadata(self) -> dict | None:
        """Get KV connector metadata from this worker if available."""
        if not has_kv_transfer_group():
            return None

        connector = get_kv_transfer_group()

        # Return None for connectors that don't need to exchange handshake
        # metadata across workers.
        if (metadata := connector.get_handshake_metadata()) is None:
            return None
        return {self.rank: metadata}

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

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
        logger.debug("Updated max_model_len to %d", max_model_len)

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate NPU KV cache with the specified kv_cache_config."""
        if self.vllm_config.model_config.enable_sleep_mode:
            allocator = CaMemAllocator.get_instance()
            context = allocator.use_memory_pool(tag="kv_cache")
        else:
            from contextlib import nullcontext

            context = nullcontext()  # type: ignore
        with context:
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
        self.model_runner._dummy_run(num_tokens=self.model_runner.decode_token_per_req, uniform_decode=True)

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
        ensure_kv_transfer_initialized(self.vllm_config)
        ensure_ec_transfer_initialized(self.vllm_config)

    def _init_profiler(self):
        # Torch profiler. Enabled through profiler_config:
        # --profiler-config.profiler=torch --profiler-config.torch_profiler_dir=/path/to/save/trace
        profiler_config = self.vllm_config.profiler_config
        if profiler_config.profiler == "torch" and profiler_config.torch_profiler_dir:
            if envs_ascend.MSMONITOR_USE_DAEMON:
                raise RuntimeError("MSMONITOR_USE_DAEMON and torch profiler cannot be both enabled at the same time.")
            torch_profiler_trace_dir = profiler_config.torch_profiler_dir
            logger.info("Profiling enabled. Traces will be saved to: %s", torch_profiler_trace_dir)

            experimental_config = torch_npu.profiler._ExperimentalConfig(
                export_type=torch_npu.profiler.ExportType.Text,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                msprof_tx=False,
                aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
                l2_cache=False,
                op_attr=False,
                data_simplification=True,
                record_op_args=False,
                gc_detect_threshold=None,
            )

            return torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                with_stack=False,
                profile_memory=profiler_config.torch_profiler_with_memory,
                # NOTE: torch_npu.profiler.with_modules is equivalent to torch.profiler.with_stack.
                # The with_stack option in torch_npu.profiler introduces significant time overhead.
                with_modules=profiler_config.torch_profiler_with_stack,
                experimental_config=experimental_config,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(torch_profiler_trace_dir),
            )
        else:
            return None

    def get_supported_pooling_tasks(self):
        return self.model_runner.get_supported_pooling_tasks()

    def get_supported_tasks(self) -> "tuple[SupportedTask, ...]":
        return self.model_runner.get_supported_tasks()

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.model_runner.take_draft_token_ids()

    def check_health(self) -> None:
        import subprocess

        logger.info("check_health Start!")
        try:
            result = subprocess.run(
                ["npu-smi", "info", "-i", str(self.local_rank), "-t", "health"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                parse_text_output(result.stdout)
                logger.info("check_health success!")
            else:
                logger.info(f"query NPU card {self.local_rank} fail: {result.stderr}")
        except subprocess.TimeoutExpired:
            logger.info(f"query NPU card  {self.local_rank} timeout.")
        except FileNotFoundError:
            logger.info("npu-smi tool not found.")
        except Exception as e:
            logger.info(f"query NPU card {self.local_rank} fail: {e}")
        return


def parse_text_output(output) -> None:
    lines = output.strip().split("\n")
    for i, line in enumerate(lines):
        line = line.strip()
        if "Health" in line:
            if line.split(":")[-1].strip() != "OK":
                raise RuntimeError("NPU card health status is not OK")
    return
