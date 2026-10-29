# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
from vllm.config import CacheConfig

from vllm_ascend.models.glm5next.recoverssm_config import use_glm_recoverssm, validate_glm_recoverssm
from vllm_ascend.patch.platform import patch_replayssm_config


def make_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="glm5_next_text")),
        speculative_config=SimpleNamespace(),
        cache_config=SimpleNamespace(
            use_replayssm=True, use_kda_recoverssm=False, mamba_cache_mode="none", enable_prefix_caching=False
        ),
        mamba_config=SimpleNamespace(enable_stochastic_rounding=False),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1
        ),
        kv_transfer_config=None,
    )


def test_recoverssm_uses_default_off_upstream_cache_switch():
    assert CacheConfig().use_replayssm is False
    assert CacheConfig(use_replayssm=True).use_replayssm is True


def test_default_does_not_change_native_cache_contract():
    config = make_config()
    config.cache_config.use_replayssm = False
    assert not use_glm_recoverssm(config)
    assert not config.cache_config.use_kda_recoverssm


@pytest.mark.parametrize("connector", [None, "MooncakeConnectorV2", "MooncakePullConnector"])
def test_configuration_validation_derives_recoverssm_before_worker_initialization(connector):
    config = make_config()
    if connector is not None:
        config.kv_transfer_config = SimpleNamespace(kv_connector=connector)
    assert use_glm_recoverssm(config)
    assert not config.cache_config.use_kda_recoverssm
    assert patch_replayssm_config.validate_mamba_cached_kernel(config) is config
    assert config.cache_config.use_kda_recoverssm


def test_initialization_requires_speculative_decoding():
    config = make_config()
    config.speculative_config = None
    with pytest.raises(ValueError, match="requires speculative decoding"):
        patch_replayssm_config.validate_mamba_cached_kernel(config)
    assert not config.cache_config.use_kda_recoverssm


@pytest.mark.parametrize(
    "section,field,value,message",
    [
        ("cache_config", "mamba_cache_mode", "align", "Mamba cache mode none"),
        ("cache_config", "enable_prefix_caching", True, "prefix caching disabled"),
        ("parallel_config", "pipeline_parallel_size", 2, "PP=1 and PCP=1"),
        ("parallel_config", "prefill_context_parallel_size", 2, "PP=1 and PCP=1"),
        ("parallel_config", "decode_context_parallel_size", 2, "DCP=1"),
        (None, "kv_transfer_config", SimpleNamespace(kv_connector="AscendStoreConnector"), "requires Mooncake"),
    ],
)
def test_initialization_rejects_unsupported_execution(section, field, value, message):
    config = make_config()
    setattr(getattr(config, section) if section else config, field, value)
    with pytest.raises(ValueError, match=message):
        validate_glm_recoverssm(config)
    assert not config.cache_config.use_kda_recoverssm


def test_other_models_and_disabled_switch_keep_upstream_validation(monkeypatch):
    config = make_config()
    called = []

    def original_validate(value):
        called.append(value)
        return value

    monkeypatch.setattr(patch_replayssm_config, "_original_validate", original_validate)
    config.model_config.hf_text_config.model_type = "kimi_k3"
    assert not use_glm_recoverssm(config)
    assert patch_replayssm_config.validate_mamba_cached_kernel(config) is config
    config.model_config.hf_text_config.model_type = "glm5_next_text"
    config.cache_config.use_replayssm = False
    assert patch_replayssm_config.validate_mamba_cached_kernel(config) is config
    assert called == [config, config]
