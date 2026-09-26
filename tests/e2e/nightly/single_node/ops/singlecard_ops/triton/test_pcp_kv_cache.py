# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.pcp_kv_cache import copy_pcp_kv_cache


@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e4m3fn])
@pytest.mark.parametrize("case", ["empty", "padding", "mixed", "many", "large"])
@pytest.mark.parametrize("strided", [False, True])
def test_c8_cache_preserves_packed_bytes(dtype, case, strided):
    # Include every byte pattern, including FP8 NaNs: the RoPE and scale
    # sections are raw BF16/FP32 bytes, not FP8 values.
    shape = (20, 32, 1, 656)
    raw = (torch.arange(20 * 32 * 656, dtype=torch.int32) % 256 - 128).to(torch.int8).reshape(shape)
    source = raw.to("npu")
    if strided:
        raw = raw[::2]
        source = source[::2]
    source = source.view(dtype)
    indices = {
        "empty": [],
        "padding": [-1, -1, -1],
        "mixed": [0, 31, -1, 32, 319, -1],
        "many": list(range(257)) + [-1],
        "large": list(range(319)) * 2 + [-1],
    }[case]
    # Non-contiguous slot mappings are accepted too.
    slots = torch.tensor([[i, -1] for i in indices], dtype=torch.int64, device="npu").reshape(-1, 2)[:, 0]
    packed = copy_pcp_kv_cache((source,), slots)
    expected = torch.zeros((len(indices), 656), dtype=torch.int8)
    for row, slot in enumerate(indices):
        if slot >= 0:
            expected[row] = raw[slot // 32, slot % 32, 0]
    assert packed.dtype == torch.int8
    assert torch.equal(packed.cpu(), expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_slots", [5, 257, 513])
def test_separate_latent_and_rope_cache(dtype, num_slots):
    k = torch.arange(4 * 8 * 512, dtype=torch.float32).reshape(4, 8, 1, 512).to(dtype).to("npu")
    r = torch.arange(4 * 8 * 64, dtype=torch.float32).reshape(4, 8, 1, 64).to(dtype).to("npu")
    indices = (torch.arange(num_slots, dtype=torch.int64) * 7) % 32
    indices[1::5] = -1
    slots = indices.to("npu")
    packed = copy_pcp_kv_cache((k, r), slots)
    expected = torch.cat((k.cpu().reshape(32, 512), r.cpu().reshape(32, 64)), dim=-1)[indices.clamp_min(0)]
    expected[indices < 0] = 0
    assert torch.equal(packed.cpu(), expected)
