# SPDX-License-Identifier: Apache-2.0
"""Unit regressions for Qwen4Exp MTP hybrid KV-cache groups."""

from types import SimpleNamespace

import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.spec_decode.qwen4_exp import Qwen4ExpMTPProposer


def _attention_spec() -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
    )


def test_map_draft_layer_accepts_direct_group_spec() -> None:
    layer_name = "model.mtp.0.self_attn"
    spec = _attention_spec()
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=[layer_name], kv_cache_spec=spec)
        ],
    )
    proposer = SimpleNamespace(_draft_attn_layer_names={layer_name})

    layer_to_gid, layer_to_spec = (
        Qwen4ExpMTPProposer._map_draft_layers_to_groups(proposer, config)
    )

    assert layer_to_gid == {layer_name: 0}
    assert layer_to_spec == {layer_name: spec}


def test_map_draft_layer_accepts_uniform_group_spec() -> None:
    layer_name = "model.mtp.0.self_attn"
    spec = _attention_spec()
    uniform = UniformTypeKVCacheSpecs.from_specs({layer_name: spec})
    assert uniform is not None
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=[layer_name], kv_cache_spec=uniform)
        ],
    )
    proposer = SimpleNamespace(_draft_attn_layer_names={layer_name})

    layer_to_gid, layer_to_spec = (
        Qwen4ExpMTPProposer._map_draft_layers_to_groups(proposer, config)
    )

    assert layer_to_gid == {layer_name: 0}
    assert layer_to_spec == {layer_name: spec}
