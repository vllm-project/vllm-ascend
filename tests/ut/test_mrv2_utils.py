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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import vllm_ascend.mrv2_utils as mrv2_utils
from vllm_ascend.mrv2_utils import use_v2_model_runner

_CONFIGS = [SimpleNamespace(), MagicMock()]
_CONFIG_IDS = ["empty-config", "magicmock-config"]


@pytest.mark.parametrize("config", _CONFIGS, ids=_CONFIG_IDS)
def test_v2_is_the_only_runner(config):
    assert use_v2_model_runner(config) is True


def test_validate_v2_model_runner_keeps_the_upstream_feature_list():
    config = SimpleNamespace(_get_v2_model_runner_unsupported_features=lambda: ["suffix"])

    with pytest.raises(ValueError, match="suffix"):
        mrv2_utils._validate_v2_model_runner(config)


def test_validate_v2_model_runner_accepts_a_supported_config():
    config = SimpleNamespace(_get_v2_model_runner_unsupported_features=lambda: [])

    mrv2_utils._validate_v2_model_runner(config)


def test_explicit_v1_request_is_rejected(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")

    with pytest.raises(ValueError, match="VLLM_USE_V2_MODEL_RUNNER=0"):
        mrv2_utils.apply_v2_model_runner_config_patch()


def test_v1_request_is_rejected_after_upstream_drops_the_entry(monkeypatch):
    from vllm import envs

    # Upstream removes the env entry together with Model Runner V1.
    monkeypatch.setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", None)
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")

    with pytest.raises(ValueError, match="VLLM_USE_V2_MODEL_RUNNER=0"):
        mrv2_utils.apply_v2_model_runner_config_patch()


def test_apply_config_patch_is_wired(monkeypatch):
    from vllm.config.vllm import VllmConfig

    original_property = VllmConfig.use_v2_model_runner
    original_validate = getattr(VllmConfig, "_validate_v2_model_runner", None)

    mrv2_utils.apply_v2_model_runner_config_patch()
    assert isinstance(VllmConfig.use_v2_model_runner, property)
    assert VllmConfig.use_v2_model_runner.fget is mrv2_utils.use_v2_model_runner
    if original_validate is not None:
        # Ascend owns this hook so the GPU-only Triton precondition does not
        # reject configs built in environments without Triton.
        assert VllmConfig._validate_v2_model_runner is mrv2_utils._validate_v2_model_runner

    # Re-applying is harmless.
    mrv2_utils.apply_v2_model_runner_config_patch()

    # Restore the upstream class state so later tests are unaffected.
    monkeypatch.setattr(VllmConfig, "use_v2_model_runner", original_property)
    if original_validate is not None:
        monkeypatch.setattr(VllmConfig, "_validate_v2_model_runner", original_validate)
