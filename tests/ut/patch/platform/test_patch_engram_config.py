# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import builtins
import importlib.util
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_ascend.utils import vllm_version_is


def test_engram_patch_is_noop_without_upstream_config(monkeypatch):
    original_import = builtins.__import__

    def without_engram(name, *args, **kwargs):
        if name == "vllm.config.engram":
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_engram)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    root = Path(__file__).resolve().parents[4]
    namespace = runpy.run_path(str(root / "vllm_ascend/patch/platform/patch_engram_config.py"))
    assert "verify_model_config" not in namespace


@pytest.fixture
def engram_config():
    config_module = pytest.importorskip("vllm.config.engram")
    return SimpleNamespace(
        model_config=SimpleNamespace(
            architecture="DeepseekV41ForCausalLM", hf_text_config=SimpleNamespace(engram_layer_ids=[1])
        ),
        speculative_config=None,
        engram_config=config_module.EngramConfig(cpu_offload=True, dp_shared_memory=True),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8,
            data_parallel_size=4,
            data_parallel_size_local=2,
            nnodes=2,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_elastic_ep=False,
            use_ubatching=False,
        ),
        load_config=SimpleNamespace(load_format="safetensors"),
    )


@pytest.mark.parametrize("draft,shared", [(False, None), (True, None), (False, False), (True, True)])
def test_native_engram_resolution_on_npu(engram_config, draft, shared):
    from vllm.config import EngramConfig, VllmConfig
    from vllm.platforms import current_platform

    from vllm_ascend.platform import _validate_engram_config

    config = engram_config
    original = EngramConfig(cpu_offload=True, dp_shared_memory=shared) if shared is not None else None
    config.engram_config = original
    if draft:
        target = config.model_config
        config.model_config = SimpleNamespace(architecture="DeepseekV41DSparkModel")
        config.speculative_config = SimpleNamespace(target_model_config=target, draft_model_config=config.model_config)
    assert not current_platform.is_cuda()
    VllmConfig._resolve_and_verify_engram_config(config)
    _validate_engram_config(config)
    assert type(config.engram_config) is EngramConfig
    if original is not None:
        assert config.engram_config is original
    # vLLM main resolves the default to True when the caller leaves it unset;
    # v0.30.0 keeps the unset value as False.
    if vllm_version_is("0.30.0"):
        expected_dp_shared_memory = bool(shared)
    else:
        expected_dp_shared_memory = True if shared is None else bool(shared)
    assert config.engram_config.dp_shared_memory == expected_dp_shared_memory


@pytest.mark.parametrize(
    "section,field,value,error",
    [
        ("model_config", "architecture", "UnsupportedModel", "non-empty n-gram"),
        ("engram_config", "embedding_across_dp", True, "embedding_across_dp"),
        ("parallel_config", "tensor_parallel_size", 16, "TP=1/2/4/8"),
        ("parallel_config", "pipeline_parallel_size", 2, "PP=PCP=DCP"),
        ("parallel_config", "prefill_context_parallel_size", 2, "PP=PCP=DCP"),
        ("parallel_config", "decode_context_parallel_size", 2, "PP=PCP=DCP"),
        ("load_config", "load_format", "pt", "indexed safetensors"),
    ],
)
def test_ascend_engram_limits(engram_config, section, field, value, error):
    from vllm.config import VllmConfig

    from vllm_ascend.platform import _validate_engram_config

    setattr(getattr(engram_config, section), field, value)
    with pytest.raises(ValueError, match=error):
        VllmConfig._resolve_and_verify_engram_config(engram_config)
        _validate_engram_config(engram_config)


def test_engram_patch_retains_layer_check(engram_config):
    engram_config.model_config.hf_text_config.engram_layer_ids = []
    with pytest.raises(ValueError, match="non-empty n-gram"):
        engram_config.engram_config.verify_model_config(engram_config.model_config)


def test_platform_leaves_non_engram_models_alone(engram_config):
    from vllm_ascend.platform import _validate_engram_config

    engram_config.engram_config = None
    engram_config.model_config.architecture = "OtherModel"
    _validate_engram_config(engram_config)
    assert engram_config.engram_config is None
