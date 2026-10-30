# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference allocation must retain Draft history without promoting Target."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

from vllm_ascend.worker.v2.spec_decode.dflash2.resident_kv import apply_dflash2_reference_kv_specs


def sliding():
    return SlidingWindowSpec(
        block_size=128,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=2048,
        extra_retained_tokens=7,
    )


@pytest.mark.parametrize("speculator", [None, SimpleNamespace(), SimpleNamespace(_dflash2_full_kv_reference=False)])
def test_original_methods_keep_identical_spec_dictionary(speculator):
    original = {"target": sliding(), "draft": sliding()}
    assert apply_dflash2_reference_kv_specs(original, speculator) is original


def test_only_identified_draft_layers_gain_history_retention():
    original = {"target": sliding(), "draft": sliding()}
    owner = SimpleNamespace(_dflash2_full_kv_reference=True, draft_attn_layer_names={"draft"})
    actual = apply_dflash2_reference_kv_specs(original, owner)
    assert actual is not original
    assert actual["target"] is original["target"]
    assert type(original["draft"]) is SlidingWindowSpec
    assert type(actual["draft"]) is FullAttentionSpec
    assert actual["draft"].sliding_window == 2048
    for field in ("block_size", "num_kv_heads", "head_size", "dtype", "dcp_sharded"):
        assert getattr(actual["draft"], field) == getattr(original["draft"], field)
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=132096),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        max_in_flight_tokens=512,
    )
    assert actual["draft"].max_memory_usage_bytes(config) > original["draft"].max_memory_usage_bytes(config)


def test_unknown_identified_cache_cannot_silently_become_full():
    owner = SimpleNamespace(_dflash2_full_kv_reference=True, draft_attn_layer_names={"draft"})
    with pytest.raises(NotImplementedError):
        apply_dflash2_reference_kv_specs({"draft": object()}, owner)


def test_missing_consumer_cannot_be_omitted():
    owner = SimpleNamespace(_dflash2_full_kv_reference=True, draft_attn_layer_names={"draft"})
    with pytest.raises(KeyError):
        apply_dflash2_reference_kv_specs({}, owner)
