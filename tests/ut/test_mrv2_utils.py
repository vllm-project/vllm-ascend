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


@pytest.mark.parametrize("env_value", [True, False])
def test_environment_override_wins(monkeypatch, env_value):
    monkeypatch.setattr(mrv2_utils.envs_vllm, "VLLM_USE_V2_MODEL_RUNNER", env_value)

    assert use_v2_model_runner(SimpleNamespace()) is env_value


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(),
        SimpleNamespace(model_config=None),
        SimpleNamespace(
            model_config=SimpleNamespace(
                runner_type="generate",
                is_attention_free=True,
                architecture="UnknownModel",
                architectures=["UnknownModel"],
            )
        ),
        SimpleNamespace(
            speculative_config=SimpleNamespace(
                method="unknown_method",
                num_speculative_tokens_per_batch_size=[[1, 256, 4]],
            )
        ),
        SimpleNamespace(
            speculative_config=SimpleNamespace(
                method="dflash",
                enforce_eager=True,
                draft_model_config=SimpleNamespace(architectures=["DFlash2DraftModel"]),
            )
        ),
        SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_connector="MooncakeConnectorV2"),
        ),
    ],
    ids=[
        "empty-config",
        "no-model",
        "unknown-attention-free-generate",
        "dynamic-unsupported-speculation",
        "dflash2-eager",
        "mooncake-pd",
    ],
)
def test_v2_is_default_outside_the_blacklist(monkeypatch, config):
    monkeypatch.setattr(mrv2_utils.envs_vllm, "VLLM_USE_V2_MODEL_RUNNER", None)
    monkeypatch.setattr(mrv2_utils, "is_310p", lambda: False)

    assert use_v2_model_runner(config) is True


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(lora_config=object()),
        SimpleNamespace(model_config=SimpleNamespace(architectures=["HYV3ForCausalLM"])),
        SimpleNamespace(model_config=SimpleNamespace(architecture="HYV3ForCausalLM")),
        SimpleNamespace(model_config=SimpleNamespace(architectures=["Gemma4ForCausalLM"])),
        SimpleNamespace(model_config=SimpleNamespace(architectures=["Gemma4ForConditionalGeneration"])),
        SimpleNamespace(model_config=SimpleNamespace(architectures=["Gemma4UnifiedForConditionalGeneration"])),
        SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=["Gemma4ForCausalLM"])),
        ),
        SimpleNamespace(model_config=SimpleNamespace(runner_type="pooling")),
        SimpleNamespace(model_config=SimpleNamespace(is_pooling_model=True)),
        SimpleNamespace(model_config=SimpleNamespace(is_encoder_decoder=True)),
        SimpleNamespace(additional_config={"enable_kvpp": True}),
        SimpleNamespace(ec_transfer_config=object()),
        SimpleNamespace(
            model_config=SimpleNamespace(multimodal_config=SimpleNamespace(mm_encoder_only=True)),
        ),
        SimpleNamespace(additional_config={"draft_window_size": 512}),
        SimpleNamespace(speculative_config=SimpleNamespace(method="suffix")),
        SimpleNamespace(speculative_config=SimpleNamespace(method="ngram")),
        SimpleNamespace(speculative_config=SimpleNamespace(method="ngram_gpu")),
        SimpleNamespace(speculative_config=SimpleNamespace(parallel_drafting=True)),
        SimpleNamespace(
            speculative_config=SimpleNamespace(
                method="dflash",
                enforce_eager=False,
                draft_model_config=SimpleNamespace(architectures=["DFlash2DraftModel"]),
            )
        ),
        SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_connector="AscendStoreConnector")),
        SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                kv_connector="AscendStoreConnector",
                kv_connector_extra_config={"backend": "memcache"},
            )
        ),
    ],
    ids=[
        "lora",
        "hy3-preview",
        "hy3-preview-architecture",
        "gemma4-causal",
        "gemma4-conditional",
        "gemma4-unified",
        "gemma4-hf-config",
        "pooling-runner",
        "pooling-model",
        "encoder-decoder",
        "kvpp",
        "vl-encoder-disaggregation",
        "vl-encoder-only",
        "draft-window-size",
        "suffix-speculative-decoding",
        "ngram-speculative-decoding",
        "ngram-gpu-speculative-decoding",
        "parallel-drafting",
        "dflash2-graph",
        "kv-pool-connector",
        "kv-pool-memcache",
    ],
)
def test_blacklisted_features_default_to_v1(monkeypatch, config):
    monkeypatch.setattr(mrv2_utils.envs_vllm, "VLLM_USE_V2_MODEL_RUNNER", None)
    monkeypatch.setattr(mrv2_utils, "is_310p", lambda: False)

    assert use_v2_model_runner(config) is False


def test_310p_defaults_to_v1(monkeypatch):
    monkeypatch.setattr(mrv2_utils.envs_vllm, "VLLM_USE_V2_MODEL_RUNNER", None)
    monkeypatch.setattr(mrv2_utils, "is_310p", lambda: True)

    assert use_v2_model_runner(SimpleNamespace()) is False


def test_310p_env_override_still_wins(monkeypatch):
    monkeypatch.setattr(mrv2_utils.envs_vllm, "VLLM_USE_V2_MODEL_RUNNER", True)
    monkeypatch.setattr(mrv2_utils, "is_310p", lambda: True)

    assert use_v2_model_runner(SimpleNamespace()) is True


def test_magicmock_config_does_not_trip_the_blacklist(monkeypatch):
    monkeypatch.setattr(mrv2_utils.envs_vllm, "VLLM_USE_V2_MODEL_RUNNER", None)
    monkeypatch.setattr(mrv2_utils, "is_310p", lambda: False)

    assert use_v2_model_runner(MagicMock()) is True


def test_blacklist_does_not_override_explicit_env(monkeypatch):
    monkeypatch.setattr(mrv2_utils.envs_vllm, "VLLM_USE_V2_MODEL_RUNNER", True)

    assert use_v2_model_runner(SimpleNamespace(lora_config=object())) is True
    assert use_v2_model_runner(SimpleNamespace(model_config=SimpleNamespace(architectures=["HYV3ForCausalLM"]))) is True
    assert (
        use_v2_model_runner(SimpleNamespace(model_config=SimpleNamespace(architectures=["Gemma4ForCausalLM"]))) is True
    )
    assert use_v2_model_runner(SimpleNamespace(model_config=SimpleNamespace(runner_type="pooling"))) is True
    assert use_v2_model_runner(SimpleNamespace(model_config=SimpleNamespace(is_encoder_decoder=True))) is True
    assert use_v2_model_runner(SimpleNamespace(additional_config={"enable_kvpp": True})) is True
    assert use_v2_model_runner(SimpleNamespace(ec_transfer_config=object())) is True
    assert use_v2_model_runner(SimpleNamespace(additional_config={"draft_window_size": 512})) is True
    assert (
        use_v2_model_runner(SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_connector="AscendStoreConnector")))
        is True
    )


def test_default_v2_logs_selection(monkeypatch):
    info_calls = []
    monkeypatch.setattr(mrv2_utils.envs_vllm, "VLLM_USE_V2_MODEL_RUNNER", None)
    monkeypatch.setattr(mrv2_utils, "is_310p", lambda: False)
    monkeypatch.setattr(mrv2_utils.logger, "info_once", lambda *args: info_calls.append(args))

    assert use_v2_model_runner(SimpleNamespace()) is True
    assert len(info_calls) == 1


def test_blacklist_logs_fallback(monkeypatch):
    warning_calls = []
    monkeypatch.setattr(mrv2_utils.envs_vllm, "VLLM_USE_V2_MODEL_RUNNER", None)
    monkeypatch.setattr(mrv2_utils.logger, "warning_once", lambda *args, **kwargs: warning_calls.append(args))

    assert use_v2_model_runner(SimpleNamespace(lora_config=object())) is False
    assert len(warning_calls) == 1
    assert "LoRA" in warning_calls[0][1]


def test_validation_is_decoupled_from_upstream():
    mrv2_utils._validate_v2_model_runner(object())
    mrv2_utils._validate_v2_model_runner(SimpleNamespace())


def test_apply_config_patch_is_wired(monkeypatch):
    from vllm.config.vllm import VllmConfig

    original_property = VllmConfig.use_v2_model_runner
    original_validate = VllmConfig._validate_v2_model_runner

    monkeypatch.setattr("vllm.config.vllm.HAS_TRITON", False)

    mrv2_utils.apply_v2_model_runner_config_patch()
    assert isinstance(VllmConfig.use_v2_model_runner, property)
    assert VllmConfig.use_v2_model_runner.fget is mrv2_utils.use_v2_model_runner

    # Upstream GPU-specific validation must not change Ascend's default.
    VllmConfig._validate_v2_model_runner(object())

    # Re-applying is harmless.
    mrv2_utils.apply_v2_model_runner_config_patch()
    VllmConfig._validate_v2_model_runner(object())

    # Restore the upstream class state so later tests are unaffected.
    monkeypatch.setattr(VllmConfig, "use_v2_model_runner", original_property)
    monkeypatch.setattr(VllmConfig, "_validate_v2_model_runner", original_validate)
