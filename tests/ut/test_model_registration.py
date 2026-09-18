# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

import pytest

import vllm_ascend.models as models


@pytest.mark.parametrize("release", [False, True])
def test_v41_registration_respects_upstream_version(monkeypatch, release):
    registered = {}
    monkeypatch.setattr(models, "vllm_version_is", lambda version: release)
    monkeypatch.setattr(models.ModelRegistry, "register_model", registered.__setitem__)
    models.register_model()
    assert ("DeepseekV41ForCausalLM" in registered) is not release
    assert ("DeepseekV41DSparkModel" in registered) is not release
    assert "DeepseekV4ForCausalLM" in registered
    assert "DSparkDraftModel" in registered
    assert "Qwen3DSparkModel" in registered


@pytest.mark.parametrize(
    "module_name",
    [
        "vllm_ascend.patch.platform.patch_kv_cache_utils",
        "vllm_ascend.models.deepseek_v41.cache_config",
        "vllm_ascend.attention.dsa_v41",
        "vllm_ascend.worker.model_runner_v1",
        "vllm_ascend.spec_decode.llm_base_proposer",
    ],
)
def test_shared_modules_import_on_both_upstream_versions(module_name):
    # Run in both CPU-UT lanes. Main-only V4.1 dependencies must not prevent
    # unrelated models from importing the common patch/model-runner path.
    assert importlib.import_module(module_name) is not None


def test_hidden_state_drafters_respect_upstream_version():
    from vllm_ascend.spec_decode.llm_base_proposer import _HIDDEN_STATE_DRAFTER_TYPES
    from vllm_ascend.utils import vllm_version_is

    names = {cls.__name__ for cls in _HIDDEN_STATE_DRAFTER_TYPES}
    assert "DSparkDeepseekV4ForCausalLM" in names
    assert "Qwen3DSparkForCausalLM" in names
    assert ("DSparkDeepseekV41ForCausalLM" in names) is not vllm_version_is("0.28.0")
