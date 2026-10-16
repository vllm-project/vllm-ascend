# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_ascend.attention.offload_capability import (
    DsaOffloadAttentionBackendCapability,
    DsaOffloadCacheMemory,
    is_dsa_offload_backend,
    is_speculative_cache_layer,
)


def test_layer_role_uses_explicit_names_and_layer_depth():
    assert is_speculative_cache_layer("decoder.mtp.block", num_hidden_layers=32)
    assert is_speculative_cache_layer("model.layers.32.attn", num_hidden_layers=32)
    assert not is_speculative_cache_layer("decoder.block.0.attn", num_hidden_layers=32)


def test_dsa_offload_requires_explicit_backend_capability():
    class UnsupportedBackend:
        pass

    class InheritedBackend(DsaOffloadAttentionBackendCapability):
        pass

    class SupportedBackend(DsaOffloadAttentionBackendCapability):
        @classmethod
        def supports_dsa_offload(cls) -> bool:
            return True

        @classmethod
        def get_dsa_offload_cache_memory(cls, *, layer_name, kv_cache_spec):
            del cls, layer_name, kv_cache_spec
            return DsaOffloadCacheMemory(1, 1)

    assert not is_dsa_offload_backend(UnsupportedBackend)
    assert not is_dsa_offload_backend(InheritedBackend)
    assert is_dsa_offload_backend(SupportedBackend)


def test_dsa_offload_capability_defaults_are_disabled():
    capability = DsaOffloadAttentionBackendCapability

    assert not capability.supports_dsa_offload()
    assert not capability.is_dsa_offload_attention_impl(object())
    assert capability.get_dsa_offload_layer_id("model.layers.0.attn") is None
    assert capability.get_dsa_offload_topk() is None
    assert not capability.is_dsa_offload_speculative_layer(
        layer_name="model.layers.0.attn",
        kv_cache_spec=object(),
    )
