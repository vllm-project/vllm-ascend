# SPDX-License-Identifier: Apache-2.0
"""Ownership rules for uncompressed paired SFA caches."""

from dataclasses import replace

import pytest
import torch
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.core.paired_sfa_cache import align_paired_sfa_cache, is_paired_sfa_cache


def _specs():
    return {
        "model.layers.0.attn": AscendMLAAttentionSpec(
            block_size=768,
            num_kv_heads=1,
            head_size=656,
            dtype=torch.int8,
            cache_sparse_sfa_c8=True,
        ),
        "model.layers.0.indexer.k_cache": AscendSFAIndexerCacheSpec(
            block_size=768,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.int8,
            scale_dim=1,
            scale_dtype=torch.float32,
            cache_sparse_li_c8=True,
        ),
        "model.layers.1.linear_attn": MambaSpec(
            block_size=768,
            shapes=((3, 4608), (12, 128, 128)),
            dtypes=(torch.bfloat16, torch.float32),
            mamba_cache_mode="align",
        ),
    }


def test_paired_layout_requires_main_indexer_and_recurrent():
    specs = _specs()
    assert is_paired_sfa_cache(specs)
    assert not is_paired_sfa_cache({k: v for k, v in specs.items() if not isinstance(v, MambaSpec)})
    assert not is_paired_sfa_cache({k: v for k, v in specs.items() if isinstance(v, AscendMLAAttentionSpec)})


def test_alignment_pads_recurrent_state_and_keeps_physical_pages():
    specs = _specs()
    align_paired_sfa_cache(specs)
    main_page = specs["model.layers.0.attn"].real_page_size_bytes
    assert specs["model.layers.1.linear_attn"].page_size_padded == main_page
    # Metadata builders must keep addressing scheduler-allocated physical pages
    # even when a kernel selects a smaller block size.
    assert specs["model.layers.0.attn"].storage_block_size == 768
    assert specs["model.layers.0.indexer.k_cache"].storage_block_size == 768


def test_alignment_rejects_main_pages_with_different_geometry():
    specs = _specs()
    specs["model.layers.0.info"] = replace(specs["model.layers.0.attn"], head_size=512)
    with pytest.raises(ValueError, match="identical geometry"):
        align_paired_sfa_cache(specs)


def test_indexer_merge_preserves_storage_block_size():
    spec = _specs()["model.layers.0.indexer.k_cache"]
    merged = AscendSFAIndexerCacheSpec.merge([replace(spec, storage_block_size=768)])
    assert merged.storage_block_size == 768
    with pytest.raises(AssertionError, match="storage block size"):
        AscendSFAIndexerCacheSpec.merge([replace(spec, storage_block_size=768), replace(spec, storage_block_size=384)])


def test_main_merge_preserves_draft_flag_and_rejects_mixed_drafts():
    spec = _specs()["model.layers.0.attn"]
    merged = AscendMLAAttentionSpec.merge([spec, replace(spec)])
    assert merged.is_dspark_draft is False
    with pytest.raises(AssertionError, match="Ascend KV cache layout"):
        AscendMLAAttentionSpec.merge([spec, replace(spec, is_dspark_draft=True)])
