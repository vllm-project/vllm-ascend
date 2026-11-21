# SPDX-License-Identifier: Apache-2.0
"""Check physical page accounting and alias boundaries without an NPU."""

import pytest
import torch

from vllm_ascend.ops.triton.msa.msa_m3_a8c4_cache import a8c4_cache_views, dual_cache_scatter


@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize("prefix_bytes", [0, 256])
def test_dual_page_regions_do_not_overlap(heads, prefix_bytes):
    blocks, tokens = 3, 128
    plane_bytes = blocks * tokens * heads * 256
    storage = torch.full((prefix_bytes + 2 * plane_bytes + 256,), 0xCD, dtype=torch.uint8)
    planes = storage[prefix_bytes : prefix_bytes + 2 * plane_bytes].view(2, blocks, tokens, heads, 256)
    views = a8c4_cache_views(planes.view(torch.float8_e4m3fn))
    assert views is not None
    for plane, (fp8, packed, scales, decode) in enumerate(views):
        assert decode.data_ptr() == packed.data_ptr()
        assert fp8.shape == (blocks, tokens, heads, 128)
        assert packed.shape == (blocks, tokens, heads, 64)
        assert scales.shape == (blocks, tokens, heads, 8)
        assert scales.dtype == torch.bfloat16
        for block in range(blocks):
            fp8[block].view(torch.uint8).fill_(block + 1)
            packed[block].fill_(block + 11)
            scales[block].view(torch.uint8).fill_(block + 21)
            raw = planes[plane, block].reshape(-1)
            fp8_end = tokens * heads * 128
            packed_end = fp8_end + tokens * heads * 64
            scale_end = packed_end + tokens * heads * 16
            assert torch.all(raw[:fp8_end] == block + 1)
            assert torch.all(raw[fp8_end:packed_end] == block + 11)
            assert torch.all(raw[packed_end:scale_end] == block + 21)
            assert torch.all(raw[scale_end:] == 0xCD)
    assert torch.all(storage[:prefix_bytes] == 0xCD)
    assert torch.all(storage[-256:] == 0xCD)


def test_native_cache_has_no_a8c4_views():
    native = torch.empty((2, 3, 128, 1, 128), dtype=torch.float8_e4m3fn)
    assert a8c4_cache_views(native) is None


def test_reject_mismatched_planes():
    key = torch.empty((3, 128, 1, 256), dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="matching FP8"):
        a8c4_cache_views((key, key[:2]))


def test_reject_overlapping_pages():
    raw = torch.empty((2, 1, 128, 1, 256), dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="non-overlapping"):
        a8c4_cache_views(raw.expand(2, 3, 128, 1, 256))


def test_reject_noncontiguous_head_dimensions():
    raw = torch.empty((2, 3, 128, 2, 256), dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="contiguous"):
        a8c4_cache_views(raw[:, :, :, ::2])


@pytest.mark.parametrize("invalid", ["dtype", "slots_dtype", "slots_stride", "head_dim_stride", "slot_count"])
def test_scatter_rejects_unsupported_inputs_before_launch(invalid):
    pages = torch.empty((2, 1, 128, 1, 256), dtype=torch.float8_e4m3fn)
    key = torch.empty((2, 1, 128), dtype=torch.bfloat16)
    slots = torch.zeros(2, dtype=torch.int64)
    if invalid == "dtype":
        key = key.float()
    elif invalid == "slots_dtype":
        slots = slots.int()
    elif invalid == "slots_stride":
        slots = torch.zeros(4, dtype=torch.int64)[::2]
    elif invalid == "head_dim_stride":
        key = torch.empty((2, 1, 256), dtype=torch.bfloat16)[..., ::2]
    else:
        slots = torch.zeros(3, dtype=torch.int64)
    with pytest.raises(ValueError, match="A8C4 scatter requires"):
        dual_cache_scatter(key, key, pages, slots)
