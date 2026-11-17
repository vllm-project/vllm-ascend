# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Page planning preserves cache geometry without changing owner specs."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, SlidingWindowSpec

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.core.kv_cache_planning import align_hybrid_kv_cache_specs


def _attention():
    return FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=4, dtype=torch.float16)


def _state(elements):
    return MambaSpec(block_size=4, shapes=((elements,),), dtypes=(torch.float32,))


@pytest.mark.parametrize("include_attention", [False, True])
def test_shared_pages_preserve_owner_specs(include_attention):
    state = _state(32)
    specs = {"large_state": state, "small_state": _state(4)}
    if include_attention:
        specs["attention"] = _attention()
    original = specs.copy()
    planned = align_hybrid_kv_cache_specs(specs)

    assert all(spec.page_size_bytes == 128 for spec in planned.values())
    assert specs == original
    assert specs["small_state"].page_size_padded is None
    assert planned["large_state"] is state
    assert align_hybrid_kv_cache_specs(planned) == planned


@pytest.mark.parametrize("specs", [{}, {"attention": _attention()}])
def test_nonhybrid_specs_are_unchanged(specs):
    assert align_hybrid_kv_cache_specs(specs) is specs


def test_contiguous_kv_segments_grow_before_page_padding():
    full = _attention()  # 64 bytes: 32 K + 32 V
    sliding = SlidingWindowSpec(
        block_size=4, num_kv_heads=1, head_size=2, dtype=torch.float16, sliding_window=16
    )  # 32 bytes: 16 K + 16 V
    state = replace(_state(16), page_size_padded=80)
    planned = align_hybrid_kv_cache_specs({"full": full, "sliding": sliding, "state": state})

    assert planned["sliding"].block_size == 8
    assert planned["sliding"].real_page_size_bytes == full.real_page_size_bytes
    assert planned["sliding"].sliding_window == 16
    assert all(spec.page_size_bytes == 80 for spec in planned.values())
    assert sliding.block_size == 4 and sliding.page_size_padded is None


def test_descriptor_layout_does_not_resize_scheduler_blocks_or_state_pages():
    full = _attention()
    state = _state(4)
    planned = align_hybrid_kv_cache_specs({"full": full, "state": state}, descriptor_native=True)
    assert planned == {"full": full, "state": state}

    larger_state = _state(32)
    planned = align_hybrid_kv_cache_specs({"full": full, "state": larger_state}, descriptor_native=True)
    assert planned["full"].block_size == full.block_size
    assert planned["full"].page_size_bytes == 128
    assert planned["state"] is larger_state


@pytest.mark.parametrize("head_size,head_size_v", [(3, 3), (2, 6)])
def test_planning_rejects_incompatible_contiguous_kv_pages(head_size, head_size_v):
    # 48 versus 64 bytes is not divisible; 16K+48V versus 32K+32V has
    # equal page sizes but incompatible K/V segments.
    sliding = SlidingWindowSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=head_size,
        head_size_v=head_size_v,
        dtype=torch.float16,
        sliding_window=16,
    )
    specs = {"full": _attention(), "sliding": sliding, "state": _state(32)}
    with pytest.raises(ValueError, match="Cannot align hybrid attention K/V pages"):
        align_hybrid_kv_cache_specs(specs)


def test_planning_pads_mla_without_resizing_token_blocks():
    # Its 24-byte latent page is not a divisible dense K/V segment.
    mla = AscendMLAAttentionSpec(block_size=4, num_kv_heads=1, head_size=3, dtype=torch.float16)
    planned = align_hybrid_kv_cache_specs({"full": _attention(), "mla": mla, "state": _state(32)})
    assert planned["mla"].block_size == mla.block_size
    assert planned["mla"].head_size == mla.head_size
    assert planned["mla"].page_size_bytes == 128
    assert mla.page_size_padded is None


def test_indexer_keeps_its_own_page_class():
    indexer = AscendSFAIndexerCacheSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.int8,
        scale_dim=1,
        scale_dtype=torch.float16,
        cache_sparse_li_c8=True,
    )
    specs = {"full": _attention(), "state": _state(32), "indexer": indexer}
    planned = align_hybrid_kv_cache_specs(specs)
    assert planned["indexer"] is indexer
    assert indexer.page_size_bytes == 40
    assert planned == specs
    assert planned["full"].page_size_bytes == 64
    assert planned["state"].page_size_bytes == 128


def test_builder_aligns_before_default_grouping(monkeypatch):
    planning = pytest.importorskip("vllm.v1.core.kv_cache_planning")
    from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder

    specs = {"full": _attention(), "state": _state(32)}
    upstream = Mock(return_value=[])
    monkeypatch.setattr(planning.DefaultKVCacheConfigBuilder, "get_kv_cache_groups", upstream)
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        model_config=SimpleNamespace(supports_kv_cache_grouping=False, hf_config=SimpleNamespace(model_type="qwen3")),
    )
    assert AscendKVCacheConfigBuilder().get_kv_cache_groups(config, specs) == []

    planned = upstream.call_args.args[1]
    assert planned["full"].page_size_bytes == 128
    assert specs["full"].page_size_padded is None
