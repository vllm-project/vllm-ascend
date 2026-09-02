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

import pytest

import vllm_ascend.mrv2_utils as mrv2_utils
from vllm_ascend.mrv2_utils import (
    _v2_model_runner_environment_ready,
    is_default_v2_model_runner_model,
    is_supported_v2_model_runner_feature,
    use_v2_model_runner,
)

DEFAULT_V2_ARCH = "Qwen3ForCausalLM"


def _make_model_config(**kwargs) -> SimpleNamespace:
    attrs = {
        "runner_type": "generate",
        "is_hybrid": False,
        "is_attention_free": False,
        "architectures": ["SomeModelForCausalLM"],
    }
    attrs.update(kwargs)
    return SimpleNamespace(**attrs)


def _make_vllm_config(model_config=None, speculative_config=None) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=model_config,
        speculative_config=speculative_config,
    )


class TestIsDefaultV2ModelRunnerModel:
    def test_whitelisted_architecture(self):
        config = _make_vllm_config(model_config=_make_model_config(architectures=[DEFAULT_V2_ARCH]))

        assert is_default_v2_model_runner_model(config) is True

    def test_unknown_architecture(self):
        config = _make_vllm_config(model_config=_make_model_config())

        assert is_default_v2_model_runner_model(config) is False

    def test_none_model_config(self):
        assert is_default_v2_model_runner_model(_make_vllm_config(model_config=None)) is False

    def test_non_generate_runner_type(self):
        config = _make_vllm_config(
            model_config=_make_model_config(runner_type="embedding", architectures=[DEFAULT_V2_ARCH])
        )

        assert is_default_v2_model_runner_model(config) is False

    def test_hybrid_model(self):
        config = _make_vllm_config(model_config=_make_model_config(is_hybrid=True, architectures=[DEFAULT_V2_ARCH]))

        assert is_default_v2_model_runner_model(config) is False

    def test_attention_free_model(self):
        config = _make_vllm_config(
            model_config=_make_model_config(is_attention_free=True, architectures=[DEFAULT_V2_ARCH])
        )

        assert is_default_v2_model_runner_model(config) is False


class TestIsSupportedV2ModelRunnerFeature:
    def test_without_speculative_config(self):
        config = _make_vllm_config(speculative_config=None)

        assert is_supported_v2_model_runner_feature(config) is True

    @pytest.mark.parametrize("method", ["eagle", "mtp", "dflash"])
    def test_whitelisted_methods(self, monkeypatch, method):
        monkeypatch.setattr(mrv2_utils.logger, "info_once", lambda *args: None)
        config = _make_vllm_config(speculative_config=SimpleNamespace(method=method))

        assert is_supported_v2_model_runner_feature(config) is True

    @pytest.mark.parametrize("method", ["ngram", "ngram_gpu", "unknown_method"])
    def test_unsupported_method(self, method):
        config = _make_vllm_config(speculative_config=SimpleNamespace(method=method))

        assert is_supported_v2_model_runner_feature(config) is False

    def test_whitelisted_method_logs_info(self, monkeypatch):
        info_calls = []
        monkeypatch.setattr(mrv2_utils.logger, "info_once", lambda *args: info_calls.append(args))
        config = _make_vllm_config(speculative_config=SimpleNamespace(method="eagle"))

        assert is_supported_v2_model_runner_feature(config) is True
        assert len(info_calls) == 1


class TestV2ModelRunnerEnvironmentReady:
    def test_unsupported_feature(self):
        config = _make_vllm_config(speculative_config=SimpleNamespace(method="ngram"))

        assert _v2_model_runner_environment_ready(config) is False

    def test_without_triton_on_non_310p(self, monkeypatch):
        monkeypatch.setattr(mrv2_utils, "is_310p", lambda: False)
        monkeypatch.setattr("vllm.triton_utils.HAS_TRITON", False)
        warning_calls = []
        monkeypatch.setattr(mrv2_utils.logger, "warning_once", lambda *args: warning_calls.append(args))
        config = _make_vllm_config(speculative_config=None)

        assert _v2_model_runner_environment_ready(config) is False
        assert len(warning_calls) == 1

    def test_with_triton_on_non_310p(self, monkeypatch):
        monkeypatch.setattr(mrv2_utils, "is_310p", lambda: False)
        monkeypatch.setattr("vllm.triton_utils.HAS_TRITON", True)
        config = _make_vllm_config(speculative_config=None)

        assert _v2_model_runner_environment_ready(config) is True

    def test_without_triton_on_310p(self, monkeypatch):
        # 310P does not require Triton for the V2 model runner.
        monkeypatch.setattr(mrv2_utils, "is_310p", lambda: True)
        monkeypatch.setattr("vllm.triton_utils.HAS_TRITON", False)
        config = _make_vllm_config(speculative_config=None)

        assert _v2_model_runner_environment_ready(config) is True


class TestUseV2ModelRunner:
    @pytest.mark.parametrize("env_value", [True, False])
    def test_env_override_wins(self, monkeypatch, env_value):
        monkeypatch.setattr(mrv2_utils.envs_vllm, "VLLM_USE_V2_MODEL_RUNNER", env_value)
        config = _make_vllm_config(model_config=_make_model_config())

        assert use_v2_model_runner(config) is env_value

    def test_default_enabled_for_whitelisted_model(self, monkeypatch):
        monkeypatch.setattr(mrv2_utils.envs_vllm, "VLLM_USE_V2_MODEL_RUNNER", None)
        monkeypatch.setattr(mrv2_utils, "_v2_model_runner_environment_ready", lambda _config: True)
        config = _make_vllm_config(model_config=_make_model_config(architectures=[DEFAULT_V2_ARCH]))

        assert use_v2_model_runner(config) is True

    def test_default_disabled_when_environment_not_ready(self, monkeypatch):
        monkeypatch.setattr(mrv2_utils.envs_vllm, "VLLM_USE_V2_MODEL_RUNNER", None)
        monkeypatch.setattr(mrv2_utils, "_v2_model_runner_environment_ready", lambda _config: False)
        config = _make_vllm_config(model_config=_make_model_config(architectures=[DEFAULT_V2_ARCH]))

        assert use_v2_model_runner(config) is False

    def test_non_whitelisted_model_falls_back(self, monkeypatch):
        monkeypatch.setattr(mrv2_utils.envs_vllm, "VLLM_USE_V2_MODEL_RUNNER", None)
        warning_calls = []
        monkeypatch.setattr(mrv2_utils.logger, "warning_once", lambda *args: warning_calls.append(args))
        config = _make_vllm_config(model_config=_make_model_config())

        assert use_v2_model_runner(config) is False
        assert len(warning_calls) == 1

    def test_none_model_config_falls_back(self, monkeypatch):
        monkeypatch.setattr(mrv2_utils.envs_vllm, "VLLM_USE_V2_MODEL_RUNNER", None)
        warning_calls = []
        monkeypatch.setattr(mrv2_utils.logger, "warning_once", lambda *args: warning_calls.append(args))
        config = _make_vllm_config(model_config=None)

        assert use_v2_model_runner(config) is False
        assert len(warning_calls) == 1


class TestV2ModelRunnerValidationPatch:
    def test_validation_is_decoupled_from_upstream(self):
        # The Ascend V2 runner decision is fully owned by use_v2_model_runner,
        # so the replacement validation must never raise (e.g. the upstream
        # Triton / feature-support checks do not apply).
        mrv2_utils._validate_v2_model_runner(object())
        mrv2_utils._validate_v2_model_runner(SimpleNamespace())

    def test_apply_config_patch_is_wired(self, monkeypatch):
        from vllm.config.vllm import VllmConfig

        original_property = VllmConfig.use_v2_model_runner
        original_validate = VllmConfig._validate_v2_model_runner

        monkeypatch.setattr("vllm.config.vllm.HAS_TRITON", False)

        mrv2_utils.apply_v2_model_runner_config_patch()
        assert isinstance(VllmConfig.use_v2_model_runner, property)
        assert VllmConfig.use_v2_model_runner.fget is mrv2_utils.use_v2_model_runner

        # The upstream Triton check must no longer run.
        VllmConfig._validate_v2_model_runner(object())

        # Re-applying is harmless.
        mrv2_utils.apply_v2_model_runner_config_patch()
        VllmConfig._validate_v2_model_runner(object())

        # Restore the upstream class state so later tests are unaffected.
        monkeypatch.setattr(VllmConfig, "use_v2_model_runner", original_property)
        monkeypatch.setattr(VllmConfig, "_validate_v2_model_runner", original_validate)
