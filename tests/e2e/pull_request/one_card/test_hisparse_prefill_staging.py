# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resident prefill plans must preserve each host page's physical row mapping."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.v1.hisparse.runtime import HiSparseRuntime, PagedCacheView

from vllm_ascend.hisparse.backend import AscendHiSparseBackend
from vllm_ascend.hisparse.cache_ops import (
    HiSparseNPUOps,
    allocate_mapped_host,
    gather_valid_resident_rows,
    scatter_packed_kv,
)
from vllm_ascend.hisparse.prefill import build_hisparse_prefill_staging_plan


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
                SimpleNamespace(cache_ops=ops, backend=AscendHiSparseBackend()),
                host.view(-1, block_size, width),
                plan,
                view.cache,
            )
            expected = host.clone()
            expected[block_size:][hits] = payload[hits]
            torch.testing.assert_close(actual.cpu().flatten(0, 1), expected, rtol=0, atol=0)
    finally:
        torch.npu.synchronize()
        host._hisparse_registration.close()


@pytest.mark.parametrize("dtype,width", [(torch.bfloat16, 576), (torch.float8_e4m3fn, 656)])
def test_backend_resident_payload_graph_replay(dtype, width):
    if not torch.npu.is_available() or not torch.npu.get_device_name(0).startswith("Ascend950"):
        pytest.skip("HiSparse resident gather currently targets Ascend 950")
    torch.npu.set_device(0)
    stride = 3 * width
    raw = torch.arange(7 * width * dtype.itemsize, dtype=torch.int64).to(torch.uint8).to("npu:0")
    source = raw.view(dtype).as_strided((2, 2, width), (stride, width, 1), storage_offset=width)
    rows = torch.tensor([3, 0, 2, -1], dtype=torch.int32, device="npu:0")
    destination = torch.empty((4, width), dtype=dtype, device="npu:0")
    backend = AscendHiSparseBackend()
    backend.gather_resident_rows(source, destination, rows)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        backend.gather_resident_rows(source, destination, rows)
    for values in ([3, 0, 2, -1], [0, 3, 1, 2]):
        rows.copy_(torch.tensor(values, dtype=torch.int32, device="npu:0"))
        raw.bitwise_xor_(1)
        graph.replay()
        torch.npu.synchronize()
        safe = torch.tensor(values).clamp_min(0)
        expected = source.view(torch.uint8).cpu()[safe // 2, safe % 2]
        assert torch.equal(destination.view(torch.uint8).cpu(), expected)


@pytest.mark.parametrize("row_bytes", [37, 656])
@pytest.mark.parametrize("capture", [False, True])
def test_prefill_resident_patch_changing_extents(row_bytes, capture):
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    torch.npu.set_device(0)
    block = 4
    for rows, source_blocks in ((8, 4), (24, 6), (40, 8)):
        source_cpu = (torch.arange(source_blocks * 5 * block * row_bytes) % 251).to(torch.uint8)
        source_cpu = source_cpu.view(source_blocks, 5, block, row_bytes)
        source = source_cpu.npu()
        destination = torch.full((rows // block, 2, block, row_bytes), 7, dtype=torch.uint8, device="npu:0")
        indices = torch.empty(rows, dtype=torch.int32, device="npu:0")
        source_view, destination_view = source[:, 0], destination[:, 0]

        indices.fill_(-1)
        gather_valid_resident_rows(source_view, destination_view, indices)
        torch.npu.synchronize()
        graph = None
        if capture:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                gather_valid_resident_rows(source_view, destination_view, indices)
        for iteration in range(2):
            values = (torch.arange(rows, dtype=torch.int32) + iteration * 3) % (source_blocks * block + 2) - 1
            current_source = source_cpu + iteration * 13
            source.copy_(current_source)
            indices.copy_(values)
            destination.fill_(7)
            if graph is None:
                gather_valid_resident_rows(source_view, destination_view, indices)
            else:
                graph.replay()
            torch.npu.synchronize()
            expected = torch.full(destination.shape, 7, dtype=torch.uint8)
            for row, selected in enumerate(values.tolist()):
                if 0 <= selected < source_blocks * block:
                    expected[row // block, 0, row % block] = current_source[selected // block, 0, selected % block]
            assert torch.equal(destination.cpu(), expected)


def test_prefill_singleton_column_mapping_tracks_current_residency():
    if not torch.npu.is_available():
        pytest.skip("HiSparse prefill mapping requires an NPU")
    torch.npu.set_device(0)
    block = 128
    host_table = torch.tensor([[1, 2]], dtype=torch.int32, device="npu:0")
    lengths = torch.tensor([2 * block - 7], dtype=torch.int32, device="npu:0")
    resident = torch.tensor([[2, 0], [0, 5]], dtype=torch.int32, device="npu:0")
    state = torch.tensor([0], dtype=torch.int32, device="npu:0")

    def resolve():
        plan = build_hisparse_prefill_staging_plan(host_table, lengths, block, 2)
        plan.ensure_gpu_sources(resident, state, block)
        return plan.gpu_row_ids, plan.miss_mask

    # FULL_DECODE_ONLY builds Prefill metadata outside graph capture; Core's
    # representative-page scatter_reduce is not a capturable NPU operation.
    for state_row, pages in ((0, [2, 0]), (1, [0, 5]), (0, [2, 0])):
        state.fill_(state_row)
        gpu_rows, mask = resolve()
        expected = torch.full((3 * block,), -1, dtype=torch.int32)
        for logical, physical in enumerate(pages, start=1):
            if physical:
                expected[logical * block : (logical + 1) * block] = torch.arange(
                    physical * block, (physical + 1) * block, dtype=torch.int32
                )
        torch.testing.assert_close(gpu_rows.cpu().flatten(), expected, rtol=0, atol=0)
        torch.testing.assert_close(mask.cpu().flatten(), (expected < 0).int(), rtol=0, atol=0)
