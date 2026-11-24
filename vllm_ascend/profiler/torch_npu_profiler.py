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

import dataclasses
import os
from contextlib import suppress
from copy import deepcopy
from typing import Any
from uuid import uuid4

import torch_npu
from pydantic import TypeAdapter
from vllm.config import ProfilerConfig
from vllm.logger import init_logger
from vllm.profiler.wrapper import WorkerProfiler

from vllm_ascend.ascend_config import NPUProfilerConfig, NPUProfilerExperimentalConfig, get_ascend_config

logger = init_logger(__name__)


class TorchNPUProfilerWrapper(WorkerProfiler):
    """Subclass of vLLM ``WorkerProfiler`` that wires in ``torch_npu.profiler``."""

    def __init__(self, profiler_config: ProfilerConfig, trace_name: str = "") -> None:
        super().__init__(profiler_config)
        self._validate_config(profiler_config)
        self._profiler_config = deepcopy(profiler_config)
        self._trace_name = trace_name
        self._base_profiler_kwargs = self._get_base_profiler_kwargs(profiler_config)
        self._profiler_kwargs = deepcopy(self._base_profiler_kwargs)
        self.profiler: Any = None

    def set_output_name(self, worker_name: str) -> None:
        if not self._active:
            self._trace_name = worker_name

    def set_profiler_kwargs(self, profiler_kwargs: dict | None) -> None:
        if self._active:
            return
        self._profiler_kwargs = self._resolve_profiler_kwargs(self._base_profiler_kwargs, profiler_kwargs)

    @staticmethod
    def _get_base_profiler_kwargs(profiler_config: ProfilerConfig) -> dict[str, Any]:
        npu_config = NPUProfilerConfig()
        with suppress(RuntimeError):
            npu_config = get_ascend_config().npu_profiler_config
        activities = profiler_config.torch_profiler_activities
        defaults = dict(
            record_shapes=profiler_config.torch_profiler_record_shapes,
            profile_memory=profiler_config.torch_profiler_with_memory,
            with_flops=profiler_config.torch_profiler_with_flops,
            with_modules=profiler_config.torch_profiler_with_stack,
            with_stack=npu_config.with_stack,
            activities=["NPU" if activity == "PrivateUse1" else activity for activity in activities]
            if activities is not None
            else ["CPU", "NPU"],
            output_dir=profiler_config.torch_profiler_dir,
            experimental_config=dataclasses.asdict(npu_config.experimental_config),
        )
        return TorchNPUProfilerWrapper._resolve_profiler_kwargs(defaults, None)

    @staticmethod
    def _resolve_profiler_kwargs(defaults: dict[str, Any], overrides: dict | None) -> dict[str, Any]:
        if overrides is not None and not isinstance(overrides, dict):
            raise ValueError("NPU profiler_kwargs must be an object")
        result = deepcopy(defaults)
        boolean_options = {"record_shapes", "profile_memory", "with_stack", "with_modules", "with_flops"}
        for name, value in (overrides or {}).items():
            if name not in defaults:
                raise ValueError(f"Unsupported NPU profiler keyword argument: {name}")
            if name == "experimental_config":
                if not isinstance(value, dict):
                    raise ValueError("NPU profiler experimental_config must be an object")
                result[name].update(value)
            else:
                result[name] = deepcopy(value)
        for name in boolean_options:
            if not isinstance(result[name], bool):
                raise ValueError(f"NPU profiler keyword argument {name} must be a boolean")
        activities = result["activities"]
        if (
            not isinstance(activities, list)
            or not activities
            or any(not isinstance(activity, str) or activity not in ("CPU", "NPU") for activity in activities)
            or len(activities) != len(set(activities))
        ):
            raise ValueError("NPU profiler activities must be a non-empty list of unique CPU/NPU names")
        output_dir = result["output_dir"]
        if not isinstance(output_dir, str) or "\x00" in output_dir or not os.path.isabs(output_dir):
            raise ValueError("NPU profiler output_dir must be an absolute local directory path")
        experimental = TypeAdapter(NPUProfilerExperimentalConfig).validate_python(result["experimental_config"])
        result["experimental_config"] = dataclasses.asdict(experimental)
        return result

    @staticmethod
    def _validate_config(profiler_config: ProfilerConfig) -> None:
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

    @staticmethod
    def _create_profiler(
        profiler_config: ProfilerConfig, trace_name: str, profiler_kwargs: dict[str, Any] | None = None
    ) -> Any:
        TorchNPUProfilerWrapper._validate_config(profiler_config)
        kwargs = (
            deepcopy(profiler_kwargs)
            if profiler_kwargs is not None
            else TorchNPUProfilerWrapper._get_base_profiler_kwargs(profiler_config)
        )
        experimental = kwargs.pop("experimental_config")
        level = experimental["profiler_level"]
        mstx = experimental["mstx"]
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=torch_npu.profiler.ExportType.Text,
            profiler_level=getattr(torch_npu.profiler.ProfilerLevel, level),
            mstx=mstx,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            l2_cache=False,
            op_attr=False,
            data_simplification=True,
            record_op_args=False,
            gc_detect_threshold=None,
        )

        output_dir = kwargs.pop("output_dir")
        kwargs.update(
            activities=[getattr(torch_npu.profiler.ProfilerActivity, activity) for activity in kwargs["activities"]],
            experimental_config=experimental_config,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                output_dir,
                worker_name=trace_name,
            ),
        )
        return torch_npu.profiler.profile(**kwargs)

    def _start(self) -> None:
        # A unique name also separates sessions with the same prefix or timestamp.
        trace_name = f"{self._trace_name}_{uuid4().hex}"
        try:
            self.profiler = self._create_profiler(self._profiler_config, trace_name, self._profiler_kwargs)
            self.profiler.start()
        except Exception:
            if self.profiler is not None:
                with suppress(Exception):
                    self.profiler.stop()
            self.profiler = None
            raise

    def _call_start(self) -> None:
        # The upstream hook suppresses errors and leaves the session active.
        # Immediate failures must instead reach the start_profile request.
        try:
            self._start()
        except Exception:
            # start() holds the lifecycle lock, so calling stop() would deadlock.
            self._active = False
            self._running = False
            self._active_iteration_count = 0
            self._profiling_for_iters = 0
            if self._delay_iters == 0:
                raise
            # A delayed start runs in the inference path, after the request has
            # returned. Report the failure without interrupting model execution.
            logger.exception("Failed to start delayed NPU profiler")
        else:
            self._running = True

    def _stop(self) -> None:
        try:
            if self.profiler is not None:
                self.profiler.stop()
        finally:
            self.profiler = None

    def _profiler_step(self) -> bool:
        return True
