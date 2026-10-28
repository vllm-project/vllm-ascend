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

_SUPPORTED_CONFIGS = [
    SimpleNamespace(),
    SimpleNamespace(model_config=None),
    SimpleNamespace(model_config=SimpleNamespace(runner_type="pooling")),
    SimpleNamespace(model_config=SimpleNamespace(is_pooling_model=True)),
    SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_mm_encoder=True)),
    SimpleNamespace(speculative_config=SimpleNamespace(method="ngram")),
    SimpleNamespace(speculative_config=SimpleNamespace(method="ngram_gpu")),
    SimpleNamespace(speculative_config=SimpleNamespace(method="draft_model")),
    SimpleNamespace(speculative_config=SimpleNamespace(method="draft_model", parallel_drafting=True)),
    SimpleNamespace(speculative_config=SimpleNamespace(method="eagle3", parallel_drafting=False)),
    SimpleNamespace(additional_config={"enable_dsa_cp": False}),
    SimpleNamespace(additional_config={"enable_dsa_cp": "false"}),
    SimpleNamespace(additional_config={"enable_dsa_cp": 0}),
    SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="dflash",
            enforce_eager=True,
            draft_model_config=SimpleNamespace(architectures=["DFlash2DraftModel"]),
        )
    ),
]

_SUPPORTED_IDS = [
    "empty-config",
    "no-model",
    "pooling-runner",
    "pooling-model",
    "vl-encoder-graph",
    "ngram",
    "ngram-gpu",
    "draft-model",
    "draft-model-parallel-drafting",
    "eagle3-without-parallel-drafting",
    "dsa-cp-disabled",
    "dsa-cp-string-false",
    "dsa-cp-int-zero",
    "dflash2-eager",
]

_UNSUPPORTED_CASES = [
    (SimpleNamespace(additional_config={"draft_window_size": 512}), "draft_window_size"),
    (SimpleNamespace(additional_config={"enable_dsa_cp": True}), "enable_dsa_cp"),
    (SimpleNamespace(additional_config={"enable_dsa_cp": "true"}), "enable_dsa_cp"),
    (SimpleNamespace(additional_config={"enable_dsa_cp": 1}), "enable_dsa_cp"),
    (SimpleNamespace(speculative_config=SimpleNamespace(method="suffix")), "suffix speculative decoding"),
    (
        SimpleNamespace(
            speculative_config=SimpleNamespace(
                method="dflash",
                enforce_eager=False,
                draft_model_config=SimpleNamespace(architectures=["DFlash2DraftModel"]),
            )
        ),
        "dflash2 graph",
    ),
    (
        SimpleNamespace(speculative_config=SimpleNamespace(method="eagle3", parallel_drafting=True)),
        "p-eagle",
    ),
    (
        SimpleNamespace(speculative_config=SimpleNamespace(method="eagle", parallel_drafting=True)),
        "p-eagle",
    ),
]

_UNSUPPORTED_IDS = [
    "draft-window-size",
    "enable-dsa-cp",
    "enable-dsa-cp-string",
    "enable-dsa-cp-int",
    "suffix-speculative-decoding",
    "dflash2-graph",
    "p-eagle",
    "p-eagle-method-eagle",
]


@pytest.mark.parametrize("config", _SUPPORTED_CONFIGS, ids=_SUPPORTED_IDS)
def test_v2_is_the_only_runner(config):
    assert use_v2_model_runner(config) is True


@pytest.mark.parametrize("env_value", [True, None])
def test_environment_cannot_select_a_removed_runner(monkeypatch, env_value):
    monkeypatch.setattr(mrv2_utils.envs_vllm, "VLLM_USE_V2_MODEL_RUNNER", env_value)

    assert use_v2_model_runner(SimpleNamespace()) is True


def test_magicmock_config_reports_no_unsupported_features():
    assert mrv2_utils._unsupported_features(MagicMock()) == []


@pytest.mark.parametrize("config", _SUPPORTED_CONFIGS, ids=_SUPPORTED_IDS)
def test_supported_configs_pass_validation(config):
    assert mrv2_utils._unsupported_features(config) == []
    mrv2_utils._validate_v2_model_runner(config)


@pytest.mark.parametrize("config, expected", _UNSUPPORTED_CASES, ids=_UNSUPPORTED_IDS)
def test_unsupported_features_are_reported(config, expected):
    assert mrv2_utils._unsupported_features(config) == [expected]


@pytest.mark.parametrize("config, expected", _UNSUPPORTED_CASES, ids=_UNSUPPORTED_IDS)
def test_unsupported_features_fail_validation(config, expected):
    with pytest.raises(ValueError, match="no fallback"):
        mrv2_utils._validate_v2_model_runner(config)


def test_unsupported_feature_message_names_the_feature():
    config = SimpleNamespace(speculative_config=SimpleNamespace(method="suffix"))

    with pytest.raises(ValueError, match="suffix speculative decoding"):
        mrv2_utils._validate_v2_model_runner(config)


def test_apply_config_patch_rejects_an_explicit_v1_request(monkeypatch):
    monkeypatch.setattr(mrv2_utils.envs_vllm, "VLLM_USE_V2_MODEL_RUNNER", False)

    with pytest.raises(ValueError, match="VLLM_USE_V2_MODEL_RUNNER=0"):
        mrv2_utils.apply_v2_model_runner_config_patch()


def test_apply_config_patch_is_wired(monkeypatch):
    from vllm.config.vllm import VllmConfig

    original_property = VllmConfig.use_v2_model_runner
    original_validate = VllmConfig._validate_v2_model_runner

    monkeypatch.setattr(mrv2_utils.envs_vllm, "VLLM_USE_V2_MODEL_RUNNER", None)

    mrv2_utils.apply_v2_model_runner_config_patch()
    assert isinstance(VllmConfig.use_v2_model_runner, property)
    assert VllmConfig.use_v2_model_runner.fget is mrv2_utils.use_v2_model_runner

    # Upstream GPU-specific Triton and feature gates must not change Ascend's choice.
    VllmConfig._validate_v2_model_runner(SimpleNamespace())

    # Re-applying is harmless.
    mrv2_utils.apply_v2_model_runner_config_patch()
    VllmConfig._validate_v2_model_runner(SimpleNamespace())

    # Restore the upstream class state so later tests are unaffected.
    monkeypatch.setattr(VllmConfig, "use_v2_model_runner", original_property)
    monkeypatch.setattr(VllmConfig, "_validate_v2_model_runner", original_validate)
