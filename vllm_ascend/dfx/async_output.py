#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

"""Async model-runner output wrappers that run DFX checks after D2H."""

from __future__ import annotations

from typing import Any

from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import AsyncOutput
from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput


class AscendAsyncGPUModelRunnerOutput(AsyncGPUModelRunnerOutput):
    """Async v1 output that runs token/logprob anomaly checks after D2H.

    With ``--async-scheduling``, ``_bookkeeping_sync`` leaves sampled tokens /
    logprobs on device. Detection must wait until ``get_output()`` materializes
    them on CPU (same place RejectionSampler.parse_output runs).
    """

    def __init__(self, *args: Any, runner: Any | None = None, **kwargs: Any):
        # Backward-compat: older call sites passed dumper=
        dumper = kwargs.pop("dumper", None)
        super().__init__(*args, **kwargs)
        self._runner = runner
        if self._runner is None and dumper is not None:
            self._runner = getattr(dumper, "runner", None)

    def get_output(self) -> ModelRunnerOutput:
        output = super().get_output()
        if self._runner is None:
            return output
        self._runner.dfx.check_after_sample(
            sampled_token_ids=output.sampled_token_ids,
            logprobs_lists=output.logprobs,
            req_ids=output.req_ids,
        )
        return output


class AscendAsyncOutput(AsyncModelRunnerOutput):
    """Async v2 output that runs DFX checks after ``AsyncOutput`` D2H completes.

    Under async scheduling upstream ``sample_tokens`` returns ``AsyncOutput``
    before CPU materialization; detection must wait until ``get_output()``.
    """

    def __init__(self, inner: AsyncOutput, runner: Any):
        self._inner = inner
        self._runner = runner

    def get_output(self) -> ModelRunnerOutput:
        output = self._inner.get_output()
        self._runner.dfx.check_after_sample(
            sampled_token_ids=output.sampled_token_ids,
            logprobs_lists=output.logprobs,
            req_ids=output.req_ids,
        )
        return output
