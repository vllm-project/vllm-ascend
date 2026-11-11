# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Page planning preserves cache geometry without changing owner specs."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, SlidingWindowSpec

from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec
from vllm_ascend.core.kv_cache_planning import align_hybrid_kv_cache_specs


def _attention():
    return FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=4, dtype=torch.float16)


def _state(elements):
    return MambaSpec(block_size=4, shapes=((elements,),), dtypes=(torch.float32,))


@pytest.mark.parametrize("include_attention", [False, True])
def test_shared_pages_keep_existing_padding_and_owner_specs(include_attention):
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
    assert planned["full"].page_size_bytes == planned["state"].page_size_bytes == 128


def test_builder_aligns_before_default_grouping(monkeypatch):
    planning = pytest.importorskip("vllm.v1.core.kv_cache_planning")
    from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder

    specs = {"full": _attention(), "state": _state(32)}
    upstream = Mock(return_value=[])
    monkeypatch.setattr(planning.DefaultKVCacheConfigBuilder, "get_kv_cache_groups", upstream)
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
    )
    assert AscendKVCacheConfigBuilder().get_kv_cache_groups(config, specs) == []

    planned = upstream.call_args.args[1]
    assert planned["full"].page_size_bytes == 128
    assert specs["full"].page_size_padded is None


def test_builder_fallback_aligns_once_and_uses_unpatched_grouping(monkeypatch):
    import vllm.v1.core.kv_cache_planning as upstream_planning

    from vllm_ascend.core import kv_cache_config_builder as builder_module
    from vllm_ascend.patch.platform import patch_kv_cache_utils  # noqa: F401

    # Importing Ascend patches must leave the public upstream grouping entry intact.
    assert upstream_planning.get_kv_cache_groups.__module__ == "vllm.v1.core.kv_cache_planning"
    specs = {"full": _attention(), "state": _state(32)}
    upstream = Mock(return_value=[])
    align = Mock(wraps=builder_module.align_hybrid_kv_cache_specs)
    monkeypatch.setattr(upstream_planning, "get_kv_cache_groups", upstream)
    monkeypatch.setattr(builder_module, "align_hybrid_kv_cache_specs", align)
    monkeypatch.setattr(builder_module, "get_qwen4_exp_kv_cache_groups", lambda *_: None)
    monkeypatch.setattr(builder_module, "is_deepseek_v41_cache", lambda *_: False)
    monkeypatch.setattr(builder_module, "group_and_unify_kv_cache_specs", lambda *_: None)
    monkeypatch.setattr(builder_module, "_get_kimi_k3_dspark_mixed_kv_cache_groups", lambda *_: None)
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        model_config=SimpleNamespace(supports_kv_cache_grouping=False),
    )
    assert builder_module.AscendKVCacheConfigBuilder().get_kv_cache_groups(config, specs) == []
    align.assert_called_once_with(specs)
    upstream.assert_called_once()
    assert upstream.call_args.args[1]["full"].page_size_bytes == 128
    assert specs["full"].page_size_padded is None
