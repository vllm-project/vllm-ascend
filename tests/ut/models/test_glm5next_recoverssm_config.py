# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
from vllm.config import CacheConfig

from vllm_ascend.models.glm5next.recoverssm_config import use_glm_recoverssm, validate_glm_recoverssm
from vllm_ascend.patch.platform import patch_replayssm_config


def make_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="glm5_next_text"), enforce_eager=False),
        speculative_config=SimpleNamespace(),
        cache_config=SimpleNamespace(
            use_replayssm=True, use_kda_recoverssm=False, mamba_cache_mode="none", enable_prefix_caching=False
        ),
        mamba_config=SimpleNamespace(enable_stochastic_rounding=False),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            tensor_parallel_size=1,
            data_parallel_size=1,
        ),
        kv_transfer_config=None,
        use_v2_model_runner=True,
        additional_config={},
    )


def test_recoverssm_uses_default_off_upstream_cache_switch():
    assert CacheConfig().use_replayssm is False
    assert CacheConfig(use_replayssm=True).use_replayssm is True


def test_default_does_not_change_native_cache_contract():
    config = make_config()
    config.cache_config.use_replayssm = False
    assert not use_glm_recoverssm(config)
    assert not config.cache_config.use_kda_recoverssm


@pytest.mark.parametrize(
    "connector", [None, "MooncakeConnectorV2", "MooncakePullConnector", "AscendStoreConnector", "MultiConnector"]
)
@pytest.mark.parametrize("prefix_caching", [False, True])
def test_configuration_validation_derives_recoverssm_before_worker_initialization(connector, prefix_caching):
    config = make_config()
    config.cache_config.enable_prefix_caching = prefix_caching
    config.cache_config.mamba_cache_mode = "align" if prefix_caching else "none"
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
        ("cache_config", "enable_prefix_caching", True, "Mamba cache mode align"),
        ("parallel_config", "pipeline_parallel_size", 2, "PP=1"),
        ("parallel_config", "prefill_context_parallel_size", 2, "requires a KV producer"),
        ("parallel_config", "decode_context_parallel_size", 2, "DCP=1"),
    ],
)
def test_initialization_rejects_unsupported_execution(section, field, value, message):
    config = make_config()
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError, match=message):
        validate_glm_recoverssm(config)
    assert not config.cache_config.use_kda_recoverssm


@pytest.mark.parametrize("connector", ["MooncakeConnectorV2", "MooncakePullConnector"])
@pytest.mark.parametrize("prefix_caching", [False, True])
@pytest.mark.parametrize("enforce_eager", [False, True])
def test_pcp_prefill_producer_and_dp_decode_consumer(connector, prefix_caching, enforce_eager):
    for producer in (True, False):
        config = make_config()
        config.cache_config.enable_prefix_caching = prefix_caching
        config.cache_config.mamba_cache_mode = "align" if prefix_caching else "none"
        config.model_config.enforce_eager = enforce_eager
        config.parallel_config.prefill_context_parallel_size = 8 if producer else 1
        config.parallel_config.data_parallel_size = 1 if producer else 8
        config.kv_transfer_config = SimpleNamespace(
            kv_connector=connector, kv_role="kv_producer" if producer else "kv_consumer"
        )
        assert patch_replayssm_config.validate_mamba_cached_kernel(config) is config
        assert config.cache_config.use_kda_recoverssm


@pytest.mark.parametrize(
    "section,field,value,message",
    [
        ("parallel_config", "tensor_parallel_size", 2, "PCP requires TP=1"),
        ("kv_transfer_config", "kv_role", "kv_consumer", "PCP requires a KV producer"),
        ("kv_transfer_config", "kv_role", "kv_both", "PCP requires a KV producer"),
    ],
)
def test_pcp_rejects_unsupported_execution(section, field, value, message):
    config = make_config()
    config.parallel_config.prefill_context_parallel_size = 8
    config.model_config.enforce_eager = True
    config.kv_transfer_config = SimpleNamespace(kv_connector="MooncakeConnectorV2", kv_role="kv_producer")
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError, match=message):
        patch_replayssm_config.validate_mamba_cached_kernel(config)
    assert not config.cache_config.use_kda_recoverssm


def test_pcp_store_enables_recoverssm():
    config = make_config()
    config.parallel_config.prefill_context_parallel_size = 2
    config.cache_config.enable_prefix_caching = True
    config.cache_config.mamba_cache_mode = "align"
    config.kv_transfer_config = SimpleNamespace(kv_connector="AscendStoreConnector", kv_role="kv_producer")
    assert patch_replayssm_config.validate_mamba_cached_kernel(config) is config
    assert config.cache_config.use_kda_recoverssm


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


@pytest.mark.parametrize("connector", [None, "MooncakeConnectorV2", "MooncakePullConnector"])
@pytest.mark.parametrize("method", ["mtp", "dspark"])
def test_qwen_selector_enables_shared_lifecycle_only_after_validation(connector, method):
    config = make_config()
    config.model_config.dtype = torch.bfloat16
    config.model_config.hf_text_config = SimpleNamespace(
        model_type="qwen3_5_moe_text",
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    config.speculative_config = SimpleNamespace(method=method, num_speculative_tokens=4)
    if connector is not None:
        config.kv_transfer_config = SimpleNamespace(kv_connector=connector, kv_role="kv_consumer")
    config.use_v2_model_runner = True
    config.cache_config.mamba_ssm_cache_dtype = "bfloat16"
    config.cache_config.mamba_cache_dtype = "auto"
    config.parallel_config.tensor_parallel_size = 4
    with pytest.raises(ValueError, match="FP32 SSM"):
        patch_replayssm_config.validate_mamba_cached_kernel(config)
    assert not config.cache_config.use_kda_recoverssm
    config.cache_config.mamba_ssm_cache_dtype = "float32"
    assert patch_replayssm_config.validate_mamba_cached_kernel(config) is config
    assert config.cache_config.use_kda_recoverssm
