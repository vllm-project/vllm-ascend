# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Regress Flash-Next's architecture config hook before model construction."""

import importlib
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.config import VllmConfig
from vllm.model_executor.models.config import (
    HybridAttentionMambaModelConfig,
    Qwen3_5ForConditionalGenerationConfig,
    Qwen4ExpForConditionalGenerationConfig,
)

import vllm_ascend.patch.platform  # noqa: F401

PATCH_MODULE = "vllm_ascend.patch.platform.patch_qwen4_exp_config"
PATCH_PATH = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/platform/patch_qwen4_exp_config.py"


def _config(*, architecture="Qwen4ExpForConditionalGeneration", method="dflash", use_v2=True):
    text_config = SimpleNamespace(
        hc_count=4,
        ple_layer_ids=[1],
        indexer_n_heads=8,
        mamba_ssm_dtype="bfloat16",
        rope_parameters={"rope_theta": 10000.0, "mrope_section": [1, 1, 1], "mrope_interleaved": True},
    )
    hf_config = SimpleNamespace(rope_parameters=dict(text_config.rope_parameters))
    return SimpleNamespace(
        model_config=SimpleNamespace(
            architecture=architecture,
            is_hybrid=True,
            convert_type="none",
            hf_config=hf_config,
            hf_text_config=text_config,
            multimodal_config=None,
        ),
        parallel_config=SimpleNamespace(enable_dbo=False, ubatch_size=1, pipeline_parallel_size=1),
        cache_config=SimpleNamespace(mamba_ssm_cache_dtype="auto"),
        speculative_config=None if method is None else SimpleNamespace(method=method, num_speculative_tokens=3),
        use_v2_model_runner=use_v2,
    )


def _verify(config):
    # Exercise the same architecture lookup used during VllmConfig creation.
    # Hybrid page planning is covered separately and would require full models.
    with patch.object(HybridAttentionMambaModelConfig, "verify_and_update_config") as hybrid:
        VllmConfig.try_verify_and_update_config(config)
    hybrid.assert_called_once_with(config)


def test_platform_import_installs_config_patch_before_model_loading():
    assert PATCH_MODULE in sys.modules
    assert Qwen4ExpForConditionalGenerationConfig.verify_and_update_config.__module__ == PATCH_MODULE


def test_unpatched_verifier_reproduces_dflash_allowlist_rejection():
    patch_module = importlib.import_module(PATCH_MODULE)
    config = _config()

    with pytest.raises(NotImplementedError, match="native MTP"):
        patch_module._orig_verify_qwen4_exp_config(config)


@pytest.mark.parametrize("architecture", ["Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"])
def test_dflash_config_dispatch_preserves_speculation_and_normalizes_cache(architecture):
    config = _config(architecture=architecture)
    speculative_config = config.speculative_config

    _verify(config)

    assert config.speculative_config is speculative_config
    assert speculative_config.method == "dflash"
    assert speculative_config.num_speculative_tokens == 3
    assert config.cache_config.mamba_ssm_cache_dtype == "bfloat16"
    assert config.model_config.config_updated


def test_dflash_validation_view_does_not_temporarily_mutate_live_config():
    config = _config()
    speculative_config = config.speculative_config
    original_parent = Qwen3_5ForConditionalGenerationConfig.verify_and_update_config

    def observe_parent(view):
        assert view is not config
        assert view.speculative_config is None
        assert view.cache_config is config.cache_config
        assert view.model_config is config.model_config
        assert config.speculative_config is speculative_config
        assert config.speculative_config.method == "dflash"
        original_parent(view)

    with patch.object(
        Qwen3_5ForConditionalGenerationConfig, "verify_and_update_config", side_effect=observe_parent
    ) as parent:
        _verify(config)
    parent.assert_called_once()


@pytest.mark.parametrize(
    ("architecture", "language_model_only", "stripped"),
    [
        ("Qwen4ExpForCausalLM", False, True),
        ("Qwen4ExpForConditionalGeneration", True, True),
        ("Qwen4ExpForConditionalGeneration", False, False),
    ],
)
def test_dflash_keeps_text_and_language_model_only_rope_normalization(architecture, language_model_only, stripped):
    config = _config(architecture=architecture)
    config.model_config.multimodal_config = SimpleNamespace(language_model_only=language_model_only)

    _verify(config)

    for hf_config in (config.model_config.hf_config, config.model_config.hf_text_config):
        assert ("mrope_section" not in hf_config.rope_parameters) is stripped
        assert ("mrope_interleaved" not in hf_config.rope_parameters) is stripped
        assert hf_config.rope_parameters["rope_theta"] == 10000.0


def test_dflash_keeps_explicit_mamba_cache_dtype():
    config = _config()
    config.cache_config.mamba_ssm_cache_dtype = "float32"

    _verify(config)

    assert config.cache_config.mamba_ssm_cache_dtype == "float32"


@pytest.mark.parametrize(
    ("section", "field", "value", "error", "message"),
    [
        ("text", "hc_count", 1, ValueError, "hc_count"),
        ("parallel", "enable_dbo", True, NotImplementedError, "dual-batch overlap"),
        ("parallel", "ubatch_size", 2, NotImplementedError, "microbatching"),
        ("parallel", "pipeline_parallel_size", 2, NotImplementedError, "pipeline_parallel_size=1"),
    ],
)
def test_dflash_retains_upstream_model_constraints(section, field, value, error, message):
    config = _config()
    target = config.model_config.hf_text_config if section == "text" else config.parallel_config
    setattr(target, field, value)

    with pytest.raises(error, match=message):
        _verify(config)

    assert config.speculative_config.method == "dflash"


def test_dflash_requires_v2_model_runner():
    config = _config(use_v2=False)
    with pytest.raises((ValueError, NotImplementedError), match="[Vv]2|[Vv]ersion 2"):
        _verify(config)
    assert config.speculative_config.method == "dflash"


@pytest.mark.parametrize("method", [None, "mtp", "ngram", "ngram_gpu"])
def test_existing_supported_methods_keep_upstream_behavior(method):
    config = _config(method=method, use_v2=False)
    speculative_config = config.speculative_config

    _verify(config)

    assert config.speculative_config is speculative_config
    assert config.cache_config.mamba_ssm_cache_dtype == "bfloat16"
    if speculative_config is not None:
        assert speculative_config.method == method


@pytest.mark.parametrize("method", ["eagle3", "dspark", "draft_model"])
def test_other_unsupported_methods_are_still_rejected(method):
    config = _config(method=method)
    with pytest.raises(NotImplementedError, match="native MTP"):
        _verify(config)
    assert config.speculative_config.method == method


def test_repeated_import_keeps_installed_verifier():
    verifier = Qwen4ExpForConditionalGenerationConfig.verify_and_update_config
    first_import = importlib.import_module(PATCH_MODULE)
    second_import = importlib.import_module(PATCH_MODULE)

    assert first_import is second_import
    assert Qwen4ExpForConditionalGenerationConfig.verify_and_update_config is verifier


def test_patch_is_noop_when_upstream_qwen4_exp_is_unavailable(monkeypatch):
    verifier = Qwen4ExpForConditionalGenerationConfig.verify_and_update_config
    old_config = ModuleType("vllm.model_executor.models.config")
    monkeypatch.setitem(sys.modules, old_config.__name__, old_config)
    monkeypatch.setattr(importlib.import_module("vllm.model_executor.models"), "config", old_config)

    runpy.run_path(str(PATCH_PATH))

    assert not hasattr(old_config, "Qwen4ExpForConditionalGenerationConfig")
    assert Qwen4ExpForConditionalGenerationConfig.verify_and_update_config is verifier
