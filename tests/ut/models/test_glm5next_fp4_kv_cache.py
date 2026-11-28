# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the packed MXFP4 main-KV cache (GLM-5.3-Flash)."""

from dataclasses import replace

import pytest
import torch

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    register_ascend_kv_cache_specs,
)

# 512 four-bit values (256 B) plus 32 BF16 group scales (64 B).
FP4_ROW_BYTES = 320
FP4_BLOCK_SIZE = 3456


@pytest.fixture(autouse=True)
def _register_specs():
    register_ascend_kv_cache_specs()


def _fp4_spec() -> AscendMLAAttentionSpec:
    return AscendMLAAttentionSpec(
        block_size=FP4_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=FP4_ROW_BYTES,
        dtype=torch.uint8,
        cache_sparse_sfa_fp4=True,
    )


def test_fp4_main_cache_page_size():
    spec = _fp4_spec()
    assert spec.uses_packed_sfa_main_cache
    # The packed FP4 cache is exclusive with the C8 and TurboQuant layouts.
    assert not spec.cache_sparse_sfa_c8
    assert not spec.cache_sparse_sfa_turboquant
    assert spec.real_page_size_bytes == FP4_BLOCK_SIZE * FP4_ROW_BYTES
    assert spec.unpadded_page_size_bytes == spec.real_page_size_bytes
    assert spec.page_size_bytes == spec.real_page_size_bytes


def test_fp4_main_cache_merge_keeps_layout():
    spec = _fp4_spec()
    merged = AscendMLAAttentionSpec.merge([spec, replace(spec)])
    assert merged.cache_sparse_sfa_fp4
    assert merged.head_size == FP4_ROW_BYTES
    assert merged.dtype == torch.uint8
    assert merged.real_page_size_bytes == spec.real_page_size_bytes


def test_mxfp4_one_token_page_uses_packed_row():
    # The dtype handler teaches upstream block-size alignment that a 512-wide
    # MXFP4 token occupies 320 B, not the 512 B implied by uint8 * head_size.
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    import vllm_ascend.core.kv_cache_dtype_handlers  # noqa: F401

    spec = MLAAttentionSpec(
        block_size=1,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        cache_dtype_str="mxfp4",
    )
    assert spec.page_size_bytes == FP4_ROW_BYTES


def test_mxfp4_dtype_resolves_to_uint8_storage():
    from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
    from vllm.v1.kv_cache_interface import KVQuantMode

    from vllm_ascend.core.kv_cache_dtype_handlers import Mxfp4AscendHandler

    handler = Mxfp4AscendHandler()
    assert handler.name == "mxfp4"
    assert handler.torch_dtype() is torch.uint8
    assert handler.quant_mode() == KVQuantMode.CUSTOM
    # The launcher and worker resolve the CLI value through this table.
    assert STR_DTYPE_TO_TORCH_DTYPE["mxfp4"] is torch.uint8
