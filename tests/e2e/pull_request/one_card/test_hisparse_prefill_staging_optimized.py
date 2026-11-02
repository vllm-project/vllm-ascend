# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resident prefill plans must preserve each host page's physical row mapping."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.v1.hisparse.runtime import HiSparseRuntime, PagedCacheView, build_hisparse_prefill_staging_plan

from vllm_ascend.ops.hisparse_optimized import HiSparseNPUOps, allocate_mapped_host, scatter_packed_kv


@pytest.mark.parametrize("resident_block_size", [128, 64, 32])
@pytest.mark.parametrize("host_pages", [1, 2])
def test_resident_prefill_plan_shared_layers(resident_block_size, host_pages):
    if not torch.npu.is_available() or not torch.npu.get_device_name(0).startswith("Ascend950"):
        pytest.skip("HiSparse mapped-host staging currently targets Ascend 950")
    torch.npu.set_device(0)
    device = "npu:0"
    block_size, width, layers, blocks = 128, 576, 4, 16
    page_elements = block_size * width
    block_stride = 6 * page_elements
    raw = torch.zeros(blocks * block_stride * 2, dtype=torch.int8, device=device)
    # Match the shared HMA slab: create and retain every layer view before use.
    views = [
        PagedCacheView.bind(
            raw,
            dtype=torch.bfloat16,
            row_width=width,
            byte_offset=layer * resident_block_size * width * 2,
            block_stride=block_stride * 2,
            num_blocks=blocks,
            block_size=resident_block_size,
        )
        for layer in range(layers)
    ]
    host = allocate_mapped_host(((host_pages + 1) * block_size, width), torch.bfloat16)
    host.zero_()
    host[block_size:].fill_(-23)
    ops = HiSparseNPUOps(host._hisparse_registration)
    try:
        subpages = block_size // resident_block_size
        resident_ids = torch.arange(2, 2 + host_pages * subpages, dtype=torch.int32)
        if host_pages == 2:
            # The last subpage is a real host miss; other rows must stay on NPU.
            resident_ids[-1] = 0
        resident_table = resident_ids[None, :].to(device)
        host_table = torch.arange(1, host_pages + 1, dtype=torch.int32, device=device)[None, :]
        plan = build_hisparse_prefill_staging_plan(
            host_table, torch.tensor([host_pages * block_size - 7], device=device), block_size, host_pages
        )
        state = torch.tensor([0], dtype=torch.int32, device=device)
        logical_rows = torch.arange(host_pages * block_size)
        physical = resident_ids.long()[logical_rows // resident_block_size] * resident_block_size
        physical += logical_rows % resident_block_size
        hits = resident_ids[logical_rows // resident_block_size] > 0
        physical[~hits] = -1
        expected_gpu = torch.cat((torch.full((block_size,), -1), physical)).int()
        expected_mask = (expected_gpu < 0).int()
        original_rows = None
        for layer, view in enumerate(views):
            columns = torch.arange(width)[None, :] % 16
            payload = (layer * 16 + logical_rows[:, None].float() / 8 + columns / 16 + 1).bfloat16()
            on_device = payload.to(device)
            scatter_packed_kv(on_device[:, :512], on_device[:, 512:], view.cache, physical.int().to(device))
            plan.ensure_gpu_sources(resident_table, state, resident_block_size)
            torch.testing.assert_close(plan.gpu_row_ids.cpu().flatten(), expected_gpu, rtol=0, atol=0)
            torch.testing.assert_close(plan.miss_mask.cpu().flatten(), expected_mask, rtol=0, atol=0)
            if original_rows is None:
                original_rows = plan.gpu_row_ids
            else:
                # Layers in an index group reuse the same resolved page plan.
                assert plan.gpu_row_ids is original_rows
            actual = HiSparseRuntime.gather_prefill_cache(
                SimpleNamespace(cache_ops=ops), host.view(-1, block_size, width), plan, view.cache
            )
            expected = host.clone()
            expected[block_size:][hits] = payload[hits]
            torch.testing.assert_close(actual.cpu().flatten(0, 1), expected, rtol=0, atol=0)
    finally:
        torch.npu.synchronize()
        host._hisparse_registration.close()
