# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import TypeAdapter, ValidationError

from vllm_ascend.ascend_config import SparseAttentionQuantConfig


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
