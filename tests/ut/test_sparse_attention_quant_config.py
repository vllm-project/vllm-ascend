# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import TypeAdapter, ValidationError

from vllm_ascend.ascend_config import AscendConfig, SparseAttentionQuantConfig, draft_config_context


def make_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=["MiniMaxM3SparseForCausalLM"])),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1, nnodes=1
        ),
        kv_transfer_config=None,
        cache_config=SimpleNamespace(cache_dtype="fp8"),
    )


def test_default_preserves_native_behavior():
    config = SparseAttentionQuantConfig()
    assert not config.enable_a8c4
    config.validate(make_config())


def test_draft_does_not_inherit_target_a8c4():
    config = make_config()
    config.additional_config = {"sparse_attention_quant_config": {"enable_a8c4": True}}
    with draft_config_context():
        assert not SparseAttentionQuantConfig.from_vllm_config(config).enable_a8c4
    assert SparseAttentionQuantConfig.from_vllm_config(config).enable_a8c4
    assert config.additional_config["sparse_attention_quant_config"]["enable_a8c4"]


def test_typed_draft_config_masks_a8c4_before_validation():
    config = AscendConfig(sparse_attention_quant_config={"enable_a8c4": True})

    # Stop at the next validator after observing the effective sparse config,
    # so this regression does not depend on unrelated full VllmConfig fields.
    class ValidationReached(Exception):
        pass

    def observe(sparse_config, vllm_config):
        assert not sparse_config.enable_a8c4
        raise ValidationReached

    with (
        draft_config_context(),
        patch.object(SparseAttentionQuantConfig, "validate", observe),
        pytest.raises(ValidationReached),
    ):
        config.derive_and_validate(make_config())


def test_draft_scope_restores_target_config_after_error():
    config = make_config()
    config.additional_config = {"sparse_attention_quant_config": {"enable_a8c4": True}}
    with pytest.raises(RuntimeError, match="draft loading failed"), draft_config_context():
        with draft_config_context():
            assert not SparseAttentionQuantConfig.from_vllm_config(config).enable_a8c4
        assert not SparseAttentionQuantConfig.from_vllm_config(config).enable_a8c4
        raise RuntimeError("draft loading failed")
    assert SparseAttentionQuantConfig.from_vllm_config(config).enable_a8c4


def test_typed_config_rejects_invalid_boolean():
    with pytest.raises(ValidationError):
        TypeAdapter(SparseAttentionQuantConfig).validate_python({"enable_a8c4": "invalid"})


def test_supported_single_node_config():
    with patch("vllm_ascend.ascend_config.is_950", return_value=True):
        SparseAttentionQuantConfig(enable_a8c4=True).validate(make_config())


@pytest.mark.parametrize(
    "field", ["pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size", "nnodes"]
)
def test_unvalidated_parallelism_is_rejected(field):
    config = make_config()
    setattr(config.parallel_config, field, 2)
    with patch("vllm_ascend.ascend_config.is_950", return_value=True), pytest.raises(ValueError, match="single-node"):
        SparseAttentionQuantConfig(enable_a8c4=True).validate(config)


def test_kv_transfer_is_rejected():
    config = make_config()
    config.kv_transfer_config = SimpleNamespace(kv_connector="MooncakeConnectorV2")
    with patch("vllm_ascend.ascend_config.is_950", return_value=True), pytest.raises(ValueError, match="KV transfer"):
        SparseAttentionQuantConfig(enable_a8c4=True).validate(config)


@pytest.mark.parametrize(
    "hardware,architecture,dtype",
    [
        (False, "MiniMaxM3SparseForCausalLM", "fp8"),
        (True, "OtherModel", "fp8"),
        (True, "MiniMaxM3SparseForCausalLM", "auto"),
    ],
)
def test_unsupported_device_model_or_cache_is_rejected(hardware, architecture, dtype):
    config = make_config()
    config.model_config.hf_config.architectures = [architecture]
    config.cache_config.cache_dtype = dtype
    with patch("vllm_ascend.ascend_config.is_950", return_value=hardware), pytest.raises(ValueError):
        SparseAttentionQuantConfig(enable_a8c4=True).validate(config)
