#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#
import vllm.envs as envs
from vllm.config.vllm import VllmConfig
from vllm.logger import init_logger

from vllm_ascend.utils import is_310p

logger = init_logger(__name__)
_ORIGINAL_VALIDATE_V2_MODEL_RUNNER = VllmConfig._validate_v2_model_runner


def _patched_use_v2_model_runner(self) -> bool:
    """Return VLLM_USE_V2_MODEL_RUNNER env directly.

    The upstream use_v2_model_runner gate-keeps the v2 runner with
    per-model architecture whitelists, Triton availability checks, and
    feature-support inspections. On Ascend the v2 runner is controlled
    purely by the VLLM_USE_V2_MODEL_RUNNER environment variable;
    model-compatibility decisions are deferred to the NPU runner itself.
    """
    use_v2 = envs.VLLM_USE_V2_MODEL_RUNNER
    if use_v2 is not None:
        return use_v2
    return False


def _patched_validate_v2_model_runner(self) -> None:
    """Allow 310P to use its local non-Triton V2 implementations."""
    if not is_310p():
        _ORIGINAL_VALIDATE_V2_MODEL_RUNNER(self)
        return

    # Keep upstream feature validation. Only the global HAS_TRITON check is
    # platform-specific: 310P replaces reachable Triton-backed classes and
    # rejects unsupported first-release features in NPUModelRunner310V2.
    unsupported = self._get_v2_model_runner_unsupported_features()
    if unsupported:
        raise ValueError(f"Model Runner V2 does not yet support: {', '.join(unsupported)}")

    if self.reasoning_config is not None:
        logger.warning_once(
            "Model Runner V2 does not yet support the thinking_token_budget "
            "request parameter. Set VLLM_USE_V2_MODEL_RUNNER=0 if this is required."
        )


VllmConfig.use_v2_model_runner = property(_patched_use_v2_model_runner)
VllmConfig._validate_v2_model_runner = _patched_validate_v2_model_runner
