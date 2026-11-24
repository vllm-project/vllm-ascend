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

import os
from contextlib import suppress
from typing import Any

import torch_npu
from vllm.config import ProfilerConfig
from vllm.profiler.wrapper import WorkerProfiler

from vllm_ascend.ascend_config import get_ascend_config


class TorchNPUProfilerWrapper(WorkerProfiler):
    """Subclass of vLLM ``WorkerProfiler`` that wires in ``torch_npu.profiler``."""

    def __init__(self, profiler_config: ProfilerConfig, trace_name: str) -> None:
        super().__init__(profiler_config)
        selected_ranks = os.environ.get("KIMI_NPU_PROFILE_RANKS")
        selected = True
        if selected_ranks is not None:
            from vllm.distributed import get_world_group

            rank = get_world_group().rank
            selected = rank in {int(value) for value in selected_ranks.split(",") if value.strip()}
            print(f"KIMI_NPU_PROFILE_RANK_FILTER rank={rank} enabled={selected}", flush=True)
        self.profiler: Any = self._create_profiler(profiler_config, trace_name) if selected else None

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

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=torch_npu.profiler.ExportType.Text,
            profiler_level=(torch_npu.profiler.ProfilerLevel.Level0
                            if os.environ.get("KIMI_PROFILE_PLAN_ROOT") and os.environ.get("KIMI_NPU_PROFILE_LEVEL") == "level0"
                            else torch_npu.profiler.ProfilerLevel.Level1),
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
            with_stack=(
                profiler_config.torch_profiler_with_stack
                and os.environ.get("VERL_NPU_PROFILER_WITH_STACK", "0") == "1"
            ),
            profile_memory=profiler_config.torch_profiler_with_memory,
            # NOTE: torch_npu.profiler.with_modules is equivalent to torch.profiler.with_stack.
            # The with_stack option in torch_npu.profiler introduces significant time overhead.
            with_modules=profiler_config.torch_profiler_with_stack,
            experimental_config=experimental_config,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                profiler_config.torch_profiler_dir,
                worker_name=trace_name,
                # Collect first; parse in a separate process to keep training
                # away from profiler-analysis memory/SQLite contention.
                analyse_flag=os.environ.get("KIMI_NPU_PROFILE_DEFER_ANALYSIS", "0") != "1",
            ),
        )

    def _start(self) -> None:
        if self.profiler is not None:
            self.profiler.start()

    def _stop(self) -> None:
        if self.profiler is not None:
            self.profiler.stop()

    def _profiler_step(self) -> bool:
        return True
