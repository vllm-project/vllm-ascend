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


def validate_v2_model_runner_env() -> None:
    """Reject an explicit request for the removed Model Runner V1.

    Model Runner V2 is the only implementation, so ``=0`` cannot be honored.
    Failing closed keeps a stale deployment script from silently running V2.
    """
    if _v1_runner_requested():
        raise ValueError(
            "Model Runner V1 has been removed from vllm-ascend, so VLLM_USE_V2_MODEL_RUNNER=0 "
            "cannot be honored. Unset it to run Model Runner V2."
        )
