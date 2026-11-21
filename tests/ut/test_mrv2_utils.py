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


def test_decode_query_limits_are_stable_across_config_revalidation():
    upstream = MagicMock(return_value={6: 1})
    config = SimpleNamespace(additional_config={}, uniform_decode_query_len=6)

    assert mrv2_utils.resolve_mrv2_decode_query_lens(config, 6, upstream) == {6: 1}
    upstream.assert_called_once_with(config, 6)

    config.additional_config[mrv2_utils.MRV2_DECODE_QUERY_LIMITS_KEY] = {"6": 4}
    assert mrv2_utils.resolve_mrv2_decode_query_lens(config, 6, upstream) == {6: 4}
    assert mrv2_utils.resolve_mrv2_decode_query_lens(config, 1, upstream) == {6: 1}
    assert upstream.call_count == 2


def test_draft_decode_sizes_are_stable_across_config_revalidation():
    upstream = MagicMock(return_value=[1, 2])
    config = SimpleNamespace(additional_config={})

    assert mrv2_utils.resolve_mrv2_draft_decode_request_sizes(config, upstream) == [1, 2]
    upstream.assert_called_once_with(config)

    config.additional_config[mrv2_utils.MRV2_DRAFT_DECODE_REQUEST_SIZES_KEY] = [4]
    assert mrv2_utils.resolve_mrv2_draft_decode_request_sizes(config, upstream) == [4]
    upstream.assert_called_once_with(config)

    config.additional_config[mrv2_utils.MRV2_DRAFT_DECODE_REQUEST_SIZES_KEY] = []
    assert mrv2_utils.resolve_mrv2_draft_decode_request_sizes(config, upstream) == []
    upstream.assert_called_once_with(config)


def test_varlen_decode_sizes_are_stable_per_query_len():
    upstream = MagicMock(side_effect=lambda _config, query_len: [query_len])
    config = SimpleNamespace(additional_config={})

    assert mrv2_utils.resolve_mrv2_varlen_decode_token_sizes(config, 6, upstream) == [6]
    upstream.assert_called_once_with(config, 6)

    config.additional_config[mrv2_utils.MRV2_VARLEN_DECODE_TOKEN_SIZES_KEY] = {
        "6": [8, 16],
        1: [],
    }
    assert mrv2_utils.resolve_mrv2_varlen_decode_token_sizes(config, 6, upstream) == [8, 16]
    assert mrv2_utils.resolve_mrv2_varlen_decode_token_sizes(config, 1, upstream) == []
    assert mrv2_utils.resolve_mrv2_varlen_decode_token_sizes(config, 7, upstream) == [7]
    assert upstream.call_count == 2


def test_explicit_v1_request_is_rejected(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")

    with pytest.raises(ValueError, match="VLLM_USE_V2_MODEL_RUNNER=0"):
        mrv2_utils.apply_v2_model_runner_config_patch()


def test_v1_request_is_rejected_after_upstream_drops_the_entry(monkeypatch):
    from vllm import envs

    # Upstream removes the env entry together with Model Runner V1.
    monkeypatch.setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", None, raising=False)
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")

    with pytest.raises(ValueError, match="VLLM_USE_V2_MODEL_RUNNER=0"):
        mrv2_utils.apply_v2_model_runner_config_patch()


def test_apply_config_patch_is_wired(monkeypatch):
    from vllm.config.vllm import VllmConfig

    original_property = VllmConfig.use_v2_model_runner
    original_validate = getattr(VllmConfig, "_validate_v2_model_runner", None)
    resolver_markers = {
        "get_decode_cudagraph_query_lens": "_mrv2_limits",
        "get_draft_decode_cudagraph_request_sizes": "_mrv2_draft",
        "get_varlen_decode_cudagraph_token_sizes": "_mrv2_varlen",
    }
    originals = {name: getattr(VllmConfig, name) for name in resolver_markers}

    mrv2_utils.apply_v2_model_runner_config_patch()
    assert isinstance(VllmConfig.use_v2_model_runner, property)
    assert VllmConfig.use_v2_model_runner.fget is mrv2_utils.use_v2_model_runner
    if original_validate is not None:
        assert VllmConfig._validate_v2_model_runner is mrv2_utils._validate_v2_model_runner
    for name, marker in resolver_markers.items():
        resolver = getattr(VllmConfig, name)
        assert getattr(resolver, marker) is True
        expected = getattr(originals[name], "_vllm_ascend_upstream", originals[name])
        assert resolver._vllm_ascend_upstream is expected

    # Re-applying is harmless.
    patched = {name: getattr(VllmConfig, name) for name in resolver_markers}
    mrv2_utils.apply_v2_model_runner_config_patch()
    assert all(getattr(VllmConfig, name) is resolver for name, resolver in patched.items())

    # Restore the upstream class state so later tests are unaffected.
    monkeypatch.setattr(VllmConfig, "use_v2_model_runner", original_property)
    if original_validate is not None:
        monkeypatch.setattr(VllmConfig, "_validate_v2_model_runner", original_validate)
    for name, resolver in originals.items():
        monkeypatch.setattr(VllmConfig, name, resolver)
