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

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None


def use_v2_model_runner(vllm_config: VllmConfig) -> bool:
    """Select the Ascend model runner.

    Model Runner V1 has been removed, so Model Runner V2 is the only
    implementation and this always returns ``True``.
    """
    return True


def _v1_runner_requested() -> bool:
    """Whether the environment explicitly selects Model Runner V1."""
    from vllm import envs

    value = getattr(envs, "VLLM_USE_V2_MODEL_RUNNER", None)
    if value is not None:
        return value is False
    # Upstream drops the entry together with Model Runner V1. Read the raw
    # variable so a stale ``=0`` still fails instead of being ignored.
    raw = os.getenv("VLLM_USE_V2_MODEL_RUNNER")
    if raw is None:
        return False
    try:
        return int(raw) == 0
    except ValueError:
        return False


def _validate_v2_model_runner(self) -> None:
    """Reject the configurations Model Runner V2 cannot serve.

    The feature list comes from upstream so it keeps growing with it. Only the
    Triton precondition is dropped: an Ascend deployment always ships Triton,
    while the CPU test lanes and config-only hosts import neither, and there
    is no V1 runner left to fall back to.
    """
    unsupported = self._get_v2_model_runner_unsupported_features()
    if unsupported:
        raise ValueError(f"Model Runner V2 does not yet support: {', '.join(unsupported)}")


def apply_v2_model_runner_config_patch() -> None:
    """Install Ascend runner selection on ``VllmConfig``.

    Re-apply in every process that reads ``use_v2_model_runner`` (frontend,
    workers, engine-core). Pickled configs do not carry this class patch.

    An explicit ``VLLM_USE_V2_MODEL_RUNNER=0`` raises: there is no V1 runner to
    select any more, and ignoring the setting would hide a stale deployment
    script.
    """
    if _v1_runner_requested():
        raise ValueError(
            "Model Runner V1 has been removed from vllm-ascend, so VLLM_USE_V2_MODEL_RUNNER=0 "
            "cannot be honored. Unset it to run Model Runner V2."
        )
    from vllm.config.vllm import VllmConfig

    VllmConfig.use_v2_model_runner = property(use_v2_model_runner)
    # The guard keeps this working once upstream renames the hook.
    if hasattr(VllmConfig, "_validate_v2_model_runner"):
        VllmConfig._validate_v2_model_runner = _validate_v2_model_runner
