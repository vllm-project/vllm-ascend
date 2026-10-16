#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_ascend.attention import utils as attention_utils
from vllm_ascend.attention.context_parallel.mla_cp import AscendMlaDCPMetadataBuilder
from vllm_ascend.attention.mla_v1 import AscendMLAMetadataBuilder
from vllm_ascend.attention.sfa_v1 import AscendSFAMetadataBuilder
from vllm_ascend.attention.utils import (
    DEFAULT_CHUNKED_PREFILL_WORKSPACE_CAPACITY,
    KIMI_K3_CHUNKED_PREFILL_WORKSPACE_CAPACITY,
    ascend_chunked_prefill_workspace_size,
    is_kimi_k3_model,
    resolve_chunked_prefill_workspace_capacity,
    select_chunked_prefill_workspace_capacity,
)
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile

K3_CAPACITY = 512 * 1024
DEFAULT_CAPACITY = 128 * 1024
ALL_HARDWARE_PROFILES = [get_hardware_profile(device_type) for device_type in AscendDeviceType]


def _model_config(architectures=("DeepseekV3ForCausalLM",), model_type="deepseek_v3", max_model_len=4096, **kwargs):
    return SimpleNamespace(
        architectures=list(architectures),
        hf_config=SimpleNamespace(model_type=model_type),
        max_model_len=max_model_len,
        **kwargs,
    )


def _vllm_config(model_config=None, max_num_seqs=16, block_size=128, pcp_size=1, dcp_size=1, interleave=128):
    return SimpleNamespace(
        model_config=model_config or _model_config(),
        cache_config=SimpleNamespace(block_size=block_size),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size,
            decode_context_parallel_size=dcp_size,
            cp_kv_cache_interleave_size=interleave,
        ),
    )


@pytest.fixture
def ascend_config(monkeypatch):
    config = SimpleNamespace(chunked_prefill_workspace_size=None)
    monkeypatch.setattr(attention_utils, "get_ascend_config", lambda: config)
    return config


@pytest.fixture
def mock_logger(monkeypatch):
    logger = MagicMock()
    monkeypatch.setattr(attention_utils, "logger", logger)
    return logger


def test_capacity_constants_match_policy():
    assert DEFAULT_CHUNKED_PREFILL_WORKSPACE_CAPACITY == DEFAULT_CAPACITY
    assert KIMI_K3_CHUNKED_PREFILL_WORKSPACE_CAPACITY == K3_CAPACITY


@pytest.mark.parametrize("hardware_profile", ALL_HARDWARE_PROFILES)
@pytest.mark.parametrize(
    "architectures,model_type",
    [
        (("KimiK3ForCausalLM",), "kimi_linear"),
        (("KimiK3MTPModel",), "unknown"),
        (("K3DSparkModel",), "unknown"),
        ((), "kimi_linear"),
        ((), "k3_dspark"),
        (("KimiLinearForCausalLM",), "kimi_linear"),
        (("KimiK3ForConditionalGeneration",), "kimi_k3"),
        # The HF model type alone identifies K3 when architectures are unusual.
        (("SomeWrapperForCausalLM",), "kimi_k3"),
        ((), "kimi_k3"),
    ],
)
def test_kimi_k3_selects_512k_on_every_hardware(architectures, model_type, hardware_profile):
    model_config = _model_config(architectures=architectures, model_type=model_type)

    assert is_kimi_k3_model(model_config)
    assert select_chunked_prefill_workspace_capacity(model_config, hardware_profile) == K3_CAPACITY


@pytest.mark.parametrize("hardware_profile", ALL_HARDWARE_PROFILES)
@pytest.mark.parametrize(
    "model_config",
    [
        _model_config(),
        _model_config(architectures=("KimiK25ForConditionalGeneration",), model_type="kimi_k25"),
        # A checkpoint path mentioning K3 does not select the K3 policy.
        _model_config(model="/data/weights/Kimi-K3-Instruct", tokenizer="/data/weights/Kimi-K3-Instruct"),
        SimpleNamespace(architectures=None, hf_config=None),
    ],
)
def test_other_models_fall_back_to_128k(model_config, hardware_profile):
    assert not is_kimi_k3_model(model_config)
    assert select_chunked_prefill_workspace_capacity(model_config, hardware_profile) == DEFAULT_CAPACITY


@pytest.mark.parametrize("location", ["hf_config", "hf_text_config", "text_config"])
@pytest.mark.parametrize("identity", [{"architectures": ["KimiLinearForCausalLM"]}, {"model_type": "kimi_linear"}])
def test_nested_kimi_k3_config_selects_policy(location, identity):
    model = _model_config()
    config = SimpleNamespace(**identity)
    if location == "text_config":
        model.hf_config.text_config = config
    else:
        setattr(model, location, config)

    assert is_kimi_k3_model(model)
    assert select_chunked_prefill_workspace_capacity(model, ALL_HARDWARE_PROFILES[0]) == K3_CAPACITY


def test_resolve_uses_heuristic_and_current_hardware_when_unset(ascend_config, monkeypatch):
    hardware_profile = ALL_HARDWARE_PROFILES[0]
    selector_inputs = []

    def selector(model_config, profile):
        selector_inputs.append((model_config, profile))
        return 4096

    monkeypatch.setattr(attention_utils, "get_current_hardware_profile", lambda: hardware_profile)
    monkeypatch.setattr(attention_utils, "select_chunked_prefill_workspace_capacity", selector)
    vllm_config = _vllm_config()

    capacity, _ = resolve_chunked_prefill_workspace_capacity(vllm_config)

    assert capacity == 4096
    assert selector_inputs == [(vllm_config.model_config, hardware_profile)]
    assert ascend_config.chunked_prefill_workspace_size is None


@pytest.mark.parametrize("configured_capacity", [1, 64 * 1024, 1024 * 1024])
def test_explicit_capacity_wins_over_kimi_k3_default(ascend_config, monkeypatch, configured_capacity):
    ascend_config.chunked_prefill_workspace_size = configured_capacity
    selector = MagicMock(return_value=K3_CAPACITY)
    monkeypatch.setattr(attention_utils, "select_chunked_prefill_workspace_capacity", selector)
    vllm_config = _vllm_config(_model_config(architectures=("KimiK3ForConditionalGeneration",)))

    capacity, source = resolve_chunked_prefill_workspace_capacity(vllm_config)

    assert capacity == configured_capacity
    assert source == "additional_config.chunked_prefill_workspace_size"
    selector.assert_not_called()


@pytest.mark.parametrize(
    "architectures,expected",
    [
        (("DeepseekV3ForCausalLM",), DEFAULT_CAPACITY),
        (("KimiK3ForConditionalGeneration",), K3_CAPACITY),
    ],
)
def test_default_capacity_caps_long_context_workspace(ascend_config, mock_logger, architectures, expected):
    # 8 * 128K = 1M rows of demand exceed both defaults.
    vllm_config = _vllm_config(_model_config(architectures=architectures, max_model_len=128 * 1024))

    assert ascend_chunked_prefill_workspace_size(vllm_config) == expected
    assert ascend_config.chunked_prefill_workspace_size is None
    mock_logger.warning_once.assert_called_once()
    args = mock_logger.warning_once.call_args.args
    assert args[1:3] == (1024 * 1024, expected)
    assert args[3] == ("Kimi K3 model policy" if expected == K3_CAPACITY else "default fallback policy")


def test_demand_below_capacity_is_not_capped(ascend_config, mock_logger):
    ascend_config.chunked_prefill_workspace_size = 64 * 1024
    # demand = max(8 * 4096, 4 * 16 * 128) = 32768 rows.
    vllm_config = _vllm_config(_model_config(max_model_len=4096))

    assert ascend_chunked_prefill_workspace_size(vllm_config) == 32768
    mock_logger.warning_once.assert_not_called()


def test_demand_equal_to_capacity_is_not_reported(ascend_config, mock_logger):
    ascend_config.chunked_prefill_workspace_size = 32768
    vllm_config = _vllm_config(_model_config(max_model_len=4096))

    assert ascend_chunked_prefill_workspace_size(vllm_config) == 32768
    mock_logger.warning_once.assert_not_called()


def test_demand_above_capacity_is_clamped_and_reported(ascend_config, mock_logger):
    ascend_config.chunked_prefill_workspace_size = 32767
    vllm_config = _vllm_config(_model_config(max_model_len=4096))

    assert ascend_chunked_prefill_workspace_size(vllm_config) == 32767
    mock_logger.warning_once.assert_called_once()
    message, *args = mock_logger.warning_once.call_args.args
    assert "clamps the workspace" in message
    assert args == [32768, 32767, "additional_config.chunked_prefill_workspace_size", 32767]


def test_minimum_capacity_overrides_smaller_cap_and_is_reported(ascend_config, mock_logger):
    ascend_config.chunked_prefill_workspace_size = 1024
    # floor = 16 * 128 = 2048 rows > C; demand = max(8 * 64, 4 * 2048) = 8192.
    vllm_config = _vllm_config(_model_config(max_model_len=64))

    assert ascend_chunked_prefill_workspace_size(vllm_config) == 2048
    mock_logger.warning_once.assert_called_once()
    message, *args = mock_logger.warning_once.call_args.args
    assert "max_num_seqs * block_size" in message
    assert "above the capacity" in message
    assert args == [8192, 1024, "additional_config.chunked_prefill_workspace_size", 2048, 2048]


@pytest.mark.parametrize("builder_cls", [AscendMLAMetadataBuilder, AscendSFAMetadataBuilder])
@pytest.mark.parametrize(
    "configured,architectures,expected",
    [
        (None, ("DeepseekV3ForCausalLM",), DEFAULT_CAPACITY),
        (None, ("KimiLinearForCausalLM",), K3_CAPACITY),
        (256 * 1024, ("KimiLinearForCausalLM",), 256 * 1024),
    ],
)
def test_mla_and_sfa_builders_use_capacity_policy(
    ascend_config, mock_logger, builder_cls, configured, architectures, expected
):
    ascend_config.chunked_prefill_workspace_size = configured
    vllm_config = _vllm_config(_model_config(architectures=architectures, max_model_len=128 * 1024))

    assert builder_cls.determine_chunked_prefill_workspace_size(vllm_config) == expected
    assert ascend_config.chunked_prefill_workspace_size == configured


def test_pcp_dcp_minimum_raises_workspace_above_configured_capacity(ascend_config, mock_logger):
    ascend_config.chunked_prefill_workspace_size = 4096
    # floor = 32 * 128 = 4096 rows; PCP/DCP needs 2 * 32 * lcm(128, 128 * 16) rows.
    vllm_config = _vllm_config(
        _model_config(max_model_len=8192), max_num_seqs=32, block_size=128, pcp_size=2, dcp_size=16
    )

    assert AscendMlaDCPMetadataBuilder.determine_chunked_prefill_workspace_size(vllm_config) == 131072
    assert ascend_config.chunked_prefill_workspace_size == 4096
