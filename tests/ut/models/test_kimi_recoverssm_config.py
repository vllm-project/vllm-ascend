# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

from vllm_ascend.models.kimi_k3.recoverssm_config import (
    supports_request_indexed_kda_recoverssm,
    use_request_indexed_kda_recoverssm,
    validate_request_indexed_kda_recoverssm,
)
from vllm_ascend.patch.platform.patch_kimi_k31_config import KimiK31TextConfig


def make_config(text_config=None):
    if text_config is None:
        text_config = KimiLinearConfig(linear_attn_config={"kda_layers": [1], "full_attn_layers": []})
    return SimpleNamespace(
        model_config=SimpleNamespace(
            architecture="UnknownWrapper",
            hf_config=SimpleNamespace(model_type="unrelated"),
            hf_text_config=text_config,
        ),
        speculative_config=SimpleNamespace(method="dspark", num_speculative_tokens=7),
        cache_config=SimpleNamespace(
            use_replayssm=True, use_kda_recoverssm=False, mamba_cache_mode="none", enable_prefix_caching=False
        ),
        mamba_config=SimpleNamespace(enable_stochastic_rounding=False),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1
        ),
        kv_transfer_config=None,
        use_v2_model_runner=True,
    )


@pytest.mark.parametrize(
    "text_config_cls",
    [KimiLinearConfig, KimiK31TextConfig],
)
@pytest.mark.parametrize("prefix_caching", [False, True])
def test_text_config_contract_selects_recoverssm(text_config_cls, prefix_caching):
    text_config = text_config_cls(linear_attn_config={"kda_layers": [1], "full_attn_layers": []})
    config = make_config(text_config)
    config.cache_config.enable_prefix_caching = prefix_caching
    config.cache_config.mamba_cache_mode = "align" if prefix_caching else "none"
    assert supports_request_indexed_kda_recoverssm(config)
    assert use_request_indexed_kda_recoverssm(config)
    validate_request_indexed_kda_recoverssm(config)
    config.cache_config.use_replayssm = False
    assert not use_request_indexed_kda_recoverssm(config)


@pytest.mark.parametrize("linear_config", [None, {"kda_layers": [], "full_attn_layers": [1]}])
def test_text_config_without_kda_layers_is_not_selected(linear_config):
    config = make_config(KimiLinearConfig(linear_attn_config=linear_config))
    assert not supports_request_indexed_kda_recoverssm(config)
    assert not use_request_indexed_kda_recoverssm(config)


def test_missing_model_config_is_not_selected():
    config = make_config()
    config.model_config = None
    assert not supports_request_indexed_kda_recoverssm(config)
    assert not use_request_indexed_kda_recoverssm(config)


def test_unrelated_model_is_not_selected():
    config = make_config(SimpleNamespace(model_type="kimi_linear", linear_attn_config={"kda_layers": [1]}))
    assert not supports_request_indexed_kda_recoverssm(config)
    assert not use_request_indexed_kda_recoverssm(config)


@pytest.mark.parametrize(
    "section,field,value,message",
    [
        (None, "speculative_config", None, "requires speculative decoding"),
        ("speculative_config", "num_speculative_tokens", 0, "requires speculative decoding"),
        ("speculative_config", "num_speculative_tokens", None, "requires speculative decoding"),
        ("speculative_config", "method", "dflash", "supports DSpark"),
        ("speculative_config", "method", "mtp", "supports DSpark"),
        ("speculative_config", "enable_adaptive_verification", True, "adaptive verification"),
        (None, "use_v2_model_runner", False, "Model Runner V2"),
        ("mamba_config", "enable_stochastic_rounding", True, "stochastic rounding"),
        ("cache_config", "enable_prefix_caching", True, "Mamba cache mode align"),
        ("cache_config", "mamba_cache_mode", "align", "Mamba cache mode none"),
        ("parallel_config", "pipeline_parallel_size", 2, "PP=PCP=DCP=1"),
        ("parallel_config", "prefill_context_parallel_size", 2, "PP=PCP=DCP=1"),
        ("parallel_config", "decode_context_parallel_size", 2, "PP=PCP=DCP=1"),
    ],
)
def test_unsupported_configuration_is_rejected_before_enabling(section, field, value, message):
    config = make_config()
    setattr(getattr(config, section) if section else config, field, value)
    with pytest.raises(ValueError, match=message):
        validate_request_indexed_kda_recoverssm(config)
    assert not config.cache_config.use_kda_recoverssm


@pytest.mark.parametrize("connector", ["MooncakeConnectorV2", "MooncakePullConnector"])
@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
@pytest.mark.parametrize("prefix_caching", [False, True])
def test_recoverssm_accepts_mooncake_roles(connector, role, prefix_caching):
    config = make_config()
    config.kv_transfer_config = SimpleNamespace(kv_connector=connector, kv_role=role)
    config.cache_config.enable_prefix_caching = prefix_caching
    config.cache_config.mamba_cache_mode = "align" if prefix_caching else "none"
    validate_request_indexed_kda_recoverssm(config)


@pytest.mark.parametrize("connector", ["MooncakeConnectorV2", "MooncakePullConnector"])
@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
def test_recoverssm_mooncake_requires_sd_conv_layout(connector, role, monkeypatch):
    config = make_config()
    config.kv_transfer_config = SimpleNamespace(kv_connector=connector, kv_role=role)
    monkeypatch.setattr("vllm_ascend.models.kimi_k3.recoverssm_config.is_conv_state_dim_first", lambda: True)
    with pytest.raises(ValueError, match="requires SD convolution cache layout"):
        validate_request_indexed_kda_recoverssm(config)


def test_recoverssm_without_transfer_allows_dim_first_layout(monkeypatch):
    monkeypatch.setattr("vllm_ascend.models.kimi_k3.recoverssm_config.is_conv_state_dim_first", lambda: True)
    validate_request_indexed_kda_recoverssm(make_config())


@pytest.mark.parametrize(
    ("connector", "role", "prefix", "message"),
    [
        ("OtherConnector", "kv_consumer", False, "MooncakeConnectorV2 or MooncakePullConnector"),
    ],
)
def test_recoverssm_rejects_unverified_transfer(connector, role, prefix, message):
    config = make_config()
    config.kv_transfer_config = SimpleNamespace(kv_connector=connector, kv_role=role)
    config.cache_config.enable_prefix_caching = prefix
    config.cache_config.mamba_cache_mode = "align" if prefix else "none"
    with pytest.raises(ValueError, match=message):
        validate_request_indexed_kda_recoverssm(config)


@pytest.mark.parametrize("text_config_cls", [KimiLinearConfig, KimiK31TextConfig])
def test_platform_validator_derives_kda_flag(text_config_cls):
    from vllm_ascend.patch.platform import patch_replayssm_config

    config = make_config(text_config_cls(linear_attn_config={"kda_layers": [1], "full_attn_layers": []}))
    assert patch_replayssm_config.validate_mamba_cached_kernel(config) is config
    assert config.cache_config.use_kda_recoverssm


def test_platform_validator_preserves_upstream_when_disabled(monkeypatch):
    from vllm_ascend.patch.platform import patch_replayssm_config

    config = make_config()
    config.cache_config.use_replayssm = False
    calls = []

    def original_validate(value):
        calls.append(value)
        return value

    monkeypatch.setattr(patch_replayssm_config, "_original_validate", original_validate)
    assert patch_replayssm_config.validate_mamba_cached_kernel(config) is config
    assert calls == [config]
    assert not config.cache_config.use_kda_recoverssm
