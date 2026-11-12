# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Regression tests for the centralized DeepSeek V4 shared-tuple geometry.

RFC #21053 (standardize KV cache allocation process): both
``vllm_ascend.core.kv_cache_config_builder`` (the pluggable builder, used on
vLLM main) and ``vllm_ascend.patch.platform.patch_kv_cache_utils`` (the
monkey-patch lane) now read this one geometry
(``vllm_ascend.models.deepseek_v4.cache_config``). These tests exercise that
shared module directly, with hard-coded expected values computed by hand
from vLLM's own ``page_size_bytes`` formula

    page_size_bytes = num_kv_heads * (block_size // tokens_per_state)
                       * (head_size + head_size_v) * dtype_itemsize

(``head_size_v`` defaults to 0 for MLA/SWA-MLA specs - MLA stores a single
latent vector, no separate V), so a regression in the geometry - not merely
a disagreement between the two callers of the same helper - would fail
them.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.models.deepseek_v4.cache_config import (
    DeepseekV4CacheLayout,
    _get_kv_cache_groups_uniform_groups,
    deepseek_v4_pool_bytes_per_block,
    get_deepseek_v4_cache_layout,
    group_and_unify_kv_cache_specs,
    is_deepseek_v4_groups,
)
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    _get_kv_cache_config_deepseek_v4_main,
)

# dtype itemsize used throughout: torch.float16 -> 2 bytes.
_DTYPE = torch.float16
_ITEMSIZE = 2


def _mla_spec(*, block_size: int, head_size: int, tokens_per_state: int = 1) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=_DTYPE,
        tokens_per_state=tokens_per_state,
        model_version="deepseek_v4",
    )


def _swa_spec(*, block_size: int, head_size: int, sliding_window: int) -> SlidingWindowMLASpec:
    return SlidingWindowMLASpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=_DTYPE,
        sliding_window=sliding_window,
    )


def _page_size_bytes(*, block_size: int, head_size: int, tokens_per_state: int = 1) -> int:
    """Hand computation of vLLM's AttentionSpec.page_size_bytes formula.

    head_size_v defaults to 0 for MLA/SWA-MLA specs (a single latent vector,
    no separate V), so the per-state content is just ``head_size`` wide.
    """
    num_states = block_size // tokens_per_state
    return 1 * num_states * head_size * _ITEMSIZE


def _vllm_config_no_override() -> SimpleNamespace:
    return SimpleNamespace(cache_config=SimpleNamespace(num_gpu_blocks_override=None))


def _uniform_group(layer_specs: dict, layer_names: list[str] | None = None) -> KVCacheGroupSpec:
    spec = UniformTypeKVCacheSpecs.from_specs(layer_specs)
    assert spec is not None
    return KVCacheGroupSpec(layer_names=layer_names or list(layer_specs.keys()), kv_cache_spec=spec)


# --- single page-size bucket -------------------------------------------------


def test_single_page_size_bucket_no_mtp():
    # block_size=16, tokens_per_state=1 -> num_states=16; head_size=8 ->
    # page_size_bytes = 16 * 8 * 2 = 256 (head_size_v defaults to 0).
    spec = _mla_spec(block_size=16, head_size=8)
    assert _page_size_bytes(block_size=16, head_size=8) == 256
    group = _uniform_group({"attn.0": spec, "attn.1": spec})

    layout = get_deepseek_v4_cache_layout([group])

    assert layout.page_sizes == (256,)
    assert layout.bucketed == ({256: ("attn.0", "attn.1")},)
    assert layout.mtp_layer_names == ()
    assert layout.mtp_page_size == 0
    assert layout.num_layer_tuples == 2
    assert layout.bytes_per_tuple == 256
    assert layout.pool_bytes_per_block == 512
    assert deepseek_v4_pool_bytes_per_block([group]) == 512


# --- multiple page-size buckets ----------------------------------------------


def test_multiple_page_size_buckets_within_one_group():
    # small: num_states=16, head_size=4 -> page = 16*4*2 = 128.
    # large: num_states=16, head_size=8 -> page = 16*8*2 = 256.
    small = _mla_spec(block_size=16, head_size=4)
    large = _mla_spec(block_size=16, head_size=8)
    assert _page_size_bytes(block_size=16, head_size=4) == 128
    assert _page_size_bytes(block_size=16, head_size=8) == 256
    group = _uniform_group({"small": small, "large": large})

    layout = get_deepseek_v4_cache_layout([group])

    assert layout.page_sizes == (128, 256)
    assert layout.bucketed == ({128: ("small",), 256: ("large",)},)
    assert layout.num_layer_tuples == 1
    assert layout.bytes_per_tuple == 128 + 256 == 384
    assert layout.pool_bytes_per_block == 384


def test_multiple_page_size_buckets_across_two_groups():
    # Group 0 (the "full-MLA" group) defines the canonical bucket set and
    # already contains both page-size buckets (small indexer-like layers and
    # large attention-like layers at the same logical block_size). Group 1
    # (the "C128" group) has one layer per logical position whose page size
    # must already match one of group 0's buckets - this is the real DSV4
    # shape the original duplicated geometry comment describes ("11 C4
    # layers and 10 C128 layers").
    small = _mla_spec(block_size=16, head_size=4)  # page 128
    large = _mla_spec(block_size=16, head_size=8)  # page 256
    group0 = _uniform_group({"c4.indexer": small, "c4.attn": large})
    group1 = _uniform_group({"c128.attn": large})

    layout = get_deepseek_v4_cache_layout([group0, group1])

    assert layout.page_sizes == (128, 256)
    assert layout.bucketed == (
        {128: ("c4.indexer",), 256: ("c4.attn",)},
        {256: ("c128.attn",)},
    )
    # Longest bucket across both groups is 1 layer; no MTP.
    assert layout.num_layer_tuples == 1
    assert layout.pool_bytes_per_block == 128 + 256


# --- MTP / draft trailing slots -----------------------------------------------


def test_mtp_trailing_slot_is_placed_after_every_ordinary_bucket():
    small = _mla_spec(block_size=16, head_size=4)  # page 128
    large = _mla_spec(block_size=16, head_size=8)  # page 256
    group = _uniform_group(
        {"small": small, "large": large, "model.mtp.layers.0.self_attn": large},
        layer_names=["small", "large", "model.mtp.layers.0.self_attn"],
    )

    layout = get_deepseek_v4_cache_layout([group])

    assert layout.page_sizes == (128, 256)
    assert layout.bucketed == ({128: ("small",), 256: ("large",)},)
    assert layout.mtp_layer_names == ("model.mtp.layers.0.self_attn",)
    assert layout.mtp_page_size == 256
    # One ordinary tuple slot (max bucket length 1) plus one trailing MTP slot.
    assert layout.num_layer_tuples == 2
    assert layout.pool_bytes_per_block == (128 + 256) * 2


def test_mtp_trailing_slot_materializes_after_ordinary_descriptors():
    """Cross-check: the layout's bucketing matches what the tensor
    materializer (patch_kv_cache_utils._get_kv_cache_config_deepseek_v4_main)
    actually builds, without re-deriving the geometry a second time."""
    small = _mla_spec(block_size=16, head_size=4)  # page 128
    large = _mla_spec(block_size=16, head_size=8)  # page 256
    group = _uniform_group(
        {"small": small, "large": large, "model.mtp.layers.0.self_attn": large},
        layer_names=["small", "large", "model.mtp.layers.0.self_attn"],
    )
    layout = get_deepseek_v4_cache_layout([group])
    expected_num_blocks = 5
    available_memory = layout.pool_bytes_per_block * expected_num_blocks

    num_blocks, tensors = _get_kv_cache_config_deepseek_v4_main(
        _vllm_config_no_override(),
        [group],
        available_memory,
    )

    assert num_blocks == expected_num_blocks
    tensors_by_layers = {tuple(t.layers): t for t in tensors}
    assert set(tensors_by_layers) == {("small",), ("large",), ("model.mtp.layers.0.self_attn",)}
    tuple_stride = layout.bytes_per_tuple * expected_num_blocks
    backing_size = tuple_stride * layout.num_layer_tuples
    assert all(t.size == backing_size for t in tensors)

    small_tensor = tensors_by_layers[("small",)]
    assert small_tensor.offset == 0
    assert small_tensor.layer_stride == tuple_stride
    assert small_tensor.block_stride == 128

    large_tensor = tensors_by_layers[("large",)]
    assert large_tensor.offset == 128 * expected_num_blocks
    assert large_tensor.layer_stride == tuple_stride
    assert large_tensor.block_stride == 256

    mtp_tensor = tensors_by_layers[("model.mtp.layers.0.self_attn",)]
    # Trailing slot 1 (slot 0 is the one ordinary tuple), at the MTP page size's offset.
    assert mtp_tensor.offset == tuple_stride + 128 * expected_num_blocks
    assert mtp_tensor.layer_stride == 0
    assert mtp_tensor.block_stride == 256


# --- exact and insufficient memory budgets ------------------------------------


def test_exact_memory_budget_yields_requested_num_blocks():
    spec = _mla_spec(block_size=16, head_size=8)  # page 256
    group = _uniform_group({"attn.0": spec})
    layout = get_deepseek_v4_cache_layout([group])
    assert layout.pool_bytes_per_block == 256

    num_blocks, _ = _get_kv_cache_config_deepseek_v4_main(
        _vllm_config_no_override(),
        [group],
        available_memory=256 * 9,
    )
    assert num_blocks == 9


def test_insufficient_memory_budget_floors_to_zero_blocks():
    spec = _mla_spec(block_size=16, head_size=8)  # page 256
    group = _uniform_group({"attn.0": spec})

    num_blocks, tensors = _get_kv_cache_config_deepseek_v4_main(
        _vllm_config_no_override(),
        [group],
        available_memory=255,
    )

    assert num_blocks == 0
    # Geometry is still materialized at num_blocks=0: zero-sized descriptors,
    # not a crash or a silently dropped layer.
    assert [t.layers for t in tensors] == [["attn.0"]]
    assert all(t.size == 0 for t in tensors)


# --- deterministic ordering and aliasing between cache groups -----------------


def test_layout_is_deterministic_and_bucket_mapping_is_read_only():
    small = _mla_spec(block_size=16, head_size=4)
    large = _mla_spec(block_size=16, head_size=8)
    group = _uniform_group({"small": small, "large": large})

    layout_a = get_deepseek_v4_cache_layout([group])
    layout_b = get_deepseek_v4_cache_layout([group])

    assert layout_a == layout_b
    assert isinstance(layout_a, DeepseekV4CacheLayout)
    with pytest.raises(TypeError):
        layout_a.bucketed[0][256] = ("mutated",)  # type: ignore[index]


def test_aliased_groups_produce_identical_bucket_geometry():
    """Two cache groups built from the same layer tuple (an intentional
    alias, e.g. a resident copy of the main cache) must bucket identically,
    so their materialized descriptors land on the same tuple slots."""
    large = _mla_spec(block_size=16, head_size=8)  # page 256
    primary = _uniform_group({"attn.0": large})
    alias = _uniform_group({"attn.0_alias": large})

    layout = get_deepseek_v4_cache_layout([primary, alias])

    assert layout.bucketed == ({256: ("attn.0",)}, {256: ("attn.0_alias",)})
    assert layout.num_layer_tuples == 1


# --- padded physical pages (full group_and_unify -> uniform_groups pipeline) --


def test_padded_physical_pages_align_swa_bucket_to_full_mla_bucket():
    # Full-MLA group ("C4"): block_size=16, two page-size buckets.
    c4_small = _mla_spec(block_size=16, head_size=2)  # page 16*2*2=64
    c4_large = _mla_spec(block_size=16, head_size=4)  # page 16*4*2=128
    # "C128" group: different logical block_size, one layer matching the
    # c4 large bucket exactly (128) - no padding needed for this group.
    c128_attn = _mla_spec(block_size=64, head_size=4, tokens_per_state=4)  # num_states=16 -> page 128
    assert _page_size_bytes(block_size=16, head_size=2) == 64
    assert _page_size_bytes(block_size=16, head_size=4) == 128
    assert _page_size_bytes(block_size=64, head_size=4, tokens_per_state=4) == 128
    # SWA group: a physical page (96 bytes) strictly between the two full-MLA
    # buckets (64, 128), so it must be padded up to 128 - the *nearest
    # larger* bucket - not left at 96 and not rounded down to 64.
    swa = _swa_spec(block_size=16, head_size=3, sliding_window=32)  # page 16*3*2=96
    assert _page_size_bytes(block_size=16, head_size=3) == 96

    specs = {
        "c4.indexer": c4_small,
        "c4.attn": c4_large,
        "c128.attn": c128_attn,
        "swa.0": swa,
    }
    grouped_specs = group_and_unify_kv_cache_specs(specs)
    assert grouped_specs is not None
    assert [g.block_size for g in grouped_specs[:2]] == [16, 64]

    groups = _get_kv_cache_groups_uniform_groups(grouped_specs)

    # The SWA layer's spec was padded in place to the nearest larger
    # full-MLA bucket (128), not its own unpadded size (96).
    swa_group = next(g for g in groups if "swa.0" in g.layer_names)
    swa_spec_out = swa_group.kv_cache_spec.kv_cache_specs["swa.0"]
    assert swa_spec_out.page_size_padded == 128
    assert swa_spec_out.page_size_bytes == 128

    layout = get_deepseek_v4_cache_layout(groups)
    assert layout.page_sizes == (64, 128)
    # The SWA layer is bucketed under its padded size (128), aliasing the
    # c4.attn bucket, not under its original unpadded size (96) and not
    # under the smaller full-MLA bucket (64).
    swa_bucket = next(b for b in layout.bucketed if "swa.0" in b.get(128, ()))
    assert swa_bucket[128] == ("swa.0",)
    assert 64 not in swa_bucket


def test_kimi_k3_like_groups_are_not_classified_as_deepseek_v4():
    """Kimi-K3's mixed-group cache never sets model_version="deepseek_v4";
    this module must not claim it, since Kimi-K3 plans its own geometry in
    ``vllm_ascend.models.kimi_k3`` and never builds a DeepseekV4CacheLayout."""
    spec = MLAAttentionSpec(
        block_size=384,
        num_kv_heads=1,
        head_size=8,
        dtype=_DTYPE,
        tokens_per_state=1,
        model_version=None,
    )
    group = _uniform_group({"model.layers.0.self_attn.attn": spec})

    assert is_deepseek_v4_groups([group]) is False


@pytest.mark.parametrize(
    ("groups", "expected"),
    [
        pytest.param([], False, id="empty-groups"),
        pytest.param(
            [
                _uniform_group(
                    {"attn.0": _mla_spec(block_size=16, head_size=8)},
                )
            ],
            True,
            id="single-deepseek-v4-group",
        ),
    ],
)
def test_is_deepseek_v4_groups(groups, expected):
    assert is_deepseek_v4_groups(groups) is expected
