#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#

from contextlib import suppress
from typing import Any

import torch
import torch_npu
from vllm.config import ProfilerConfig
from vllm.logger import logger
from vllm.profiler.wrapper import WorkerProfiler

from vllm_ascend.ascend_config import get_ascend_config


class TorchNPUProfilerWrapper(WorkerProfiler):
    """Subclass of vLLM ``WorkerProfiler`` that wires in ``torch_npu.profiler``."""

    def __init__(self, profiler_config: ProfilerConfig, trace_name: str) -> None:
        super().__init__(profiler_config)
        self._uses_schedule = profiler_config.warmup_iterations > 0 or profiler_config.wait_iterations > 0
        # profiler.start() consumes schedule step 0, so this tracks the
        # remaining non-active steps advanced by profiler.step().
        self._warmup_steps_remaining = max(
            profiler_config.wait_iterations + profiler_config.warmup_iterations - 1,
            0,
        )
        self.profiler: Any = self._create_profiler(profiler_config, trace_name)

    @staticmethod
    def _create_profiler(profiler_config: ProfilerConfig, trace_name: str) -> Any:
        if profiler_config.profiler != "torch":
            raise RuntimeError(f"Unrecognized profiler: {profiler_config.profiler}")
        if not profiler_config.torch_profiler_dir:
            raise RuntimeError("torch_profiler_dir cannot be empty.")
        msmonitor_use_daemon = False
        with suppress(RuntimeError):
            msmonitor_use_daemon = get_ascend_config().msmonitor_use_daemon
        if msmonitor_use_daemon:
            raise RuntimeError(
                "additional_config.msmonitor_use_daemon and torch profiler cannot be enabled at the same time."
            )

        profiler_schedule = None
        if profiler_config.warmup_iterations > 0 or profiler_config.wait_iterations > 0:
            profiler_schedule = torch_npu.profiler.schedule(
                skip_first=0,
                wait=profiler_config.wait_iterations,
                warmup=profiler_config.warmup_iterations,
                active=profiler_config.active_iterations,
                repeat=1,
            )
            logger.info_once(
                "NPU profiler schedule configured: wait=%d, warmup=%d, active=%d",
                profiler_config.wait_iterations,
                profiler_config.warmup_iterations,
                profiler_config.active_iterations,
            )

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=torch_npu.profiler.ExportType.Text,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            msprof_tx=False,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
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
            schedule=profiler_schedule,
            record_shapes=profiler_config.torch_profiler_record_shapes,
            with_stack=False,
            profile_memory=profiler_config.torch_profiler_with_memory,
            # NOTE: torch_npu.profiler.with_modules is equivalent to torch.profiler.with_stack.
            # The with_stack option in torch_npu.profiler introduces significant time overhead.
            with_modules=profiler_config.torch_profiler_with_stack,
            with_flops=profiler_config.torch_profiler_with_flops,
            experimental_config=experimental_config,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                profiler_config.torch_profiler_dir,
                worker_name=trace_name,
            ),
        )

    def _start(self) -> None:
        self.profiler.start()

    def _stop(self) -> None:
        self.profiler.stop()

    def _profiler_step(self) -> bool:
        if self._uses_schedule:
            self.profiler.step()
            if self._warmup_steps_remaining > 0:
                self._warmup_steps_remaining -= 1
                return False
        return True

    def annotate_context_manager(self, name: str):
        return torch.profiler.record_function(name)
