# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.triton.strided_mla_prefill_gather import try_gather_strided_prefill

PAGE_SIZE = 128
LATENT_DIM = 512
ROPE_DIM = 64


def _reference(cache, page_table, cumulative, starts):
    result = []
    for request, start in enumerate(starts):
        count = cumulative[request + 1] - cumulative[request]
        tokens = torch.arange(start, start + count)
        page_ids = page_table[request, tokens // PAGE_SIZE].long()
        valid = (page_ids >= 0) & (page_ids < cache.size(0))
        safe_pages = page_ids.masked_fill(~valid, 0)
        rows = cache[safe_pages, tokens % PAGE_SIZE]
        rows[~valid] = 0
        result.append(rows)
    return torch.cat(result)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_strided_mla_prefill_gather(dtype):
    if not get_current_hardware_profile().supports(HardwareCapability.STRIDED_MLA_PREFILL_GATHER):
        pytest.skip("Requires Ascend A3")
    torch.npu.set_device(0)

    pages = 6
    rng = torch.Generator().manual_seed(218)
    latent = torch.randn((pages, PAGE_SIZE, 1, LATENT_DIM), generator=rng).to(dtype)
    rotary = torch.randn((pages, PAGE_SIZE, 1, ROPE_DIM), generator=rng).to(dtype)
    padding = torch.full((pages, 64), float("nan"), dtype=dtype)
    backing = torch.cat((latent.flatten(1), rotary.flatten(1), padding), dim=1).npu()
    page_stride = backing.stride(0)
    latent_cache = backing.as_strided(latent.shape, (page_stride, LATENT_DIM, LATENT_DIM, 1))
    rotary_cache = backing.as_strided(rotary.shape, (page_stride, ROPE_DIM, ROPE_DIM, 1), PAGE_SIZE * LATENT_DIM)

    # Cross page boundaries from unaligned starts, leave a request empty,
    # and exercise both negative and out-of-range physical page IDs.
    starts = [125, 19, 5, 252]
    cumulative = [0, 15, 15, 285, 291]
    table_storage = torch.tensor(
        [[4, 1, -1, -1], [-1, -1, -1, -1], [2, -1, 3, -1], [0, pages, 5, -1]], dtype=torch.int32
    )
    page_table = table_storage.npu()[:, :3]
    cumulative_npu = torch.tensor(cumulative, dtype=torch.int32, device="npu")
    starts_npu = torch.tensor(starts, dtype=torch.int32, device="npu")
    key = torch.full((cumulative[-1], 1, LATENT_DIM), float("nan"), dtype=dtype, device="npu")
    rope = torch.full((cumulative[-1], 1, ROPE_DIM), float("nan"), dtype=dtype, device="npu")

    assert try_gather_strided_prefill(latent_cache, rotary_cache, page_table, cumulative_npu, starts_npu, key, rope)
    torch.testing.assert_close(key.cpu(), _reference(latent, table_storage, cumulative, starts), rtol=0, atol=0)
    torch.testing.assert_close(rope.cpu(), _reference(rotary, table_storage, cumulative, starts), rtol=0, atol=0)
