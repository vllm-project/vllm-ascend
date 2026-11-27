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

import pytest

import vllm_ascend.mrv2_utils as mrv2_utils


@pytest.mark.parametrize("value", ["0", "00", " 0 "])
def test_explicit_v1_request_is_rejected(monkeypatch, value):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", value)

    with pytest.raises(ValueError, match="VLLM_USE_V2_MODEL_RUNNER=0"):
        mrv2_utils.validate_v2_model_runner_env()


def test_v1_request_is_rejected_after_upstream_drops_the_entry(monkeypatch):
    from vllm import envs

    # Upstream removes the env entry together with Model Runner V1.
    monkeypatch.setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", None, raising=False)
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")

    with pytest.raises(ValueError, match="VLLM_USE_V2_MODEL_RUNNER=0"):
        mrv2_utils.validate_v2_model_runner_env()


@pytest.mark.parametrize("value", [None, "1", "true", "yes"])
def test_v2_or_unset_request_is_allowed(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    else:
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", value)

    mrv2_utils.validate_v2_model_runner_env()
