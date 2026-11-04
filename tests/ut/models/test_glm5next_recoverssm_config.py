# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
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


@pytest.mark.parametrize("connector", [None, "MooncakeConnectorV2", "MooncakePullConnector"])
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
        ("parallel_config", "prefill_context_parallel_size", 2, "requires a Mooncake KV producer"),
        ("parallel_config", "decode_context_parallel_size", 2, "DCP=1"),
        (None, "kv_transfer_config", SimpleNamespace(kv_connector="MultiConnector"), "requires Mooncake"),
    ],
)
def test_initialization_rejects_unsupported_execution(section, field, value, message):
    config = make_config()
    setattr(getattr(config, section) if section else config, field, value)
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
        ("kv_transfer_config", "kv_role", "kv_consumer", "PCP requires a Mooncake KV producer"),
        ("kv_transfer_config", "kv_role", "kv_both", "PCP requires a Mooncake KV producer"),
        ("kv_transfer_config", "kv_connector", "AscendStoreConnector", "requires a Mooncake KV producer"),
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


def make_store_config(**extra):
    config = make_config()
    config.parallel_config.tensor_parallel_size = 2
    config.cache_config.enable_prefix_caching = True
    config.cache_config.mamba_cache_mode = "align"
    config.kv_transfer_config = SimpleNamespace(kv_connector="AscendStoreConnector", kv_connector_extra_config=extra)
    return config


@pytest.mark.parametrize("backend", ["mooncake", "memcache"])
def test_bulk_store_prefix_reuse_enables_recoverssm(backend):
    config = make_store_config(backend=backend)
    assert patch_replayssm_config.validate_mamba_cached_kernel(config) is config
    assert config.cache_config.use_kda_recoverssm


@pytest.mark.parametrize("option", ["use_layerwise", "load_async", "save_decode_cache"])
def test_store_rejects_execution_before_accepted_state_is_ready(option):
    config = make_store_config(**{option: True})
    with pytest.raises(ValueError, match=f"{option}=false"):
        validate_glm_recoverssm(config)


@pytest.mark.parametrize("option", ["prefill_tp_size", "decode_tp_size"])
def test_store_rejects_heterogeneous_tp(option):
    config = make_store_config(**{option: 4})
    with pytest.raises(ValueError, match="matching TP sizes"):
        validate_glm_recoverssm(config)


def test_store_requires_prefix_caching():
    config = make_store_config()
    config.cache_config.enable_prefix_caching = False
    config.cache_config.mamba_cache_mode = "none"
    with pytest.raises(ValueError, match="requires prefix caching"):
        validate_glm_recoverssm(config)


def test_store_requires_v2_commit_lifecycle():
    config = make_store_config()
    config.use_v2_model_runner = False
    with pytest.raises(ValueError, match="requires Model Runner V2"):
        validate_glm_recoverssm(config)


def test_store_rejects_kvpp():
    config = make_store_config()
    config.additional_config["enable_kvpp"] = True
    with pytest.raises(ValueError, match="does not support KVPP"):
        validate_glm_recoverssm(config)


def test_store_rejects_unqualified_backend():
    config = make_store_config(backend="yuanrong")
    with pytest.raises(ValueError, match="mooncake or memcache"):
        validate_glm_recoverssm(config)


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
