# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resident prefill plans must preserve each host page's physical row mapping."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.v1.hisparse.runtime import HiSparseRuntime, PagedCacheView

from vllm_ascend.attention.sfa_hisparse import build_hisparse_prefill_staging_plan
from vllm_ascend.hisparse.cache_ops import (
    HiSparseNPUOps,
    scatter_packed_kv,
)
from vllm_ascend.hisparse.host_pool import allocate_mapped_host

pytest.importorskip("vllm.v1.hisparse.backend")
from vllm_ascend.hisparse.backend import AscendHiSparseBackend  # noqa: E402


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
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("num_rows", [0, 1, 4, 9])
def test_backend_resident_payload_graph_replay(dtype, width, index_dtype, num_rows):
    if not torch.npu.is_available() or not torch.npu.get_device_name(0).startswith("Ascend950"):
        pytest.skip("HiSparse resident gather currently targets Ascend 950")
    torch.npu.set_device(0)
    stride = 3 * width
    raw = torch.arange(7 * width * dtype.itemsize, dtype=torch.int64).to(torch.uint8).to("npu:0")
    source = raw.view(dtype).as_strided((2, 2, width), (stride, width, 1), storage_offset=width)
    rows = torch.empty(num_rows, dtype=index_dtype, device="npu:0")
    destination = torch.empty((num_rows, width), dtype=dtype, device="npu:0")
    rows.copy_((torch.arange(num_rows, dtype=index_dtype, device="npu:0") % 5) - 1)
    backend = AscendHiSparseBackend()
    backend.gather_resident_rows(source, destination, rows)
    torch.npu.synchronize()
    if not num_rows:
        assert destination.numel() == 0
        return
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph):
            backend.gather_resident_rows(source, destination, rows)
        for offset in (0, 2):
            values = (torch.arange(num_rows, dtype=index_dtype) + offset) % 5 - 1
            if index_dtype == torch.int64:
                values[0] = -(1 << 34)
            rows.copy_(values.to("npu:0"))
            raw.bitwise_xor_(1)
            graph.replay()
            torch.npu.synchronize()
            safe = values.long().clamp_min(0)
            expected = source.view(torch.uint8).cpu()[safe // 2, safe % 2]
            assert torch.equal(destination.view(torch.uint8).cpu(), expected)
    finally:
        torch.npu.synchronize()
        graph.reset()


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


@pytest.mark.parametrize("pages,stride_rows", [(2, 128), (3, 136)])
def test_profiled_workspace_rebind_preserves_prior_length_graphs(pages, stride_rows):
    from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

    from vllm_ascend.hisparse.backend import prepare_hisparse_workspaces as actual_prepare

    prepare_hisparse_workspaces = torch.inference_mode()(actual_prepare)

    if not torch.npu.is_available() or not torch.npu.get_device_name(0).startswith("Ascend950"):
        pytest.skip("HiSparse mapped-host owners currently target Ascend 950")
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    reset_workspace_manager()
    init_workspace_manager(device)
    backend = AscendHiSparseBackend()
    backend.check_available()
    host = allocate_mapped_host((4096,), torch.uint8)
    registration = host._hisparse_registration
    graphs = []

    def runtime():
        group = SimpleNamespace(
            copy_stream=torch.Stream(device=device),
            shared_topk=SimpleNamespace(device_topk_rows=torch.empty((4, 64), dtype=torch.int32, device=device)),
        )
        owner = SimpleNamespace(
            index_group=group,
            max_swap_rows=4,
            region_stride=128,
            max_union_rows=64,
            device=device,
            kv_dtype=torch.float8_e4m3fn,
            registered_host_pool=host,
        )
        group.leader = owner
        return owner

    def bind(owner, count):
        raw = torch.empty(count * stride_rows * 656, dtype=torch.float8_e4m3fn, device=device)
        cache = raw.as_strided((count, 128, 656), (stride_rows * 656, 656, 1))
        owner.hot = SimpleNamespace(cache=cache, attention_cache=cache, block_size=128)
        owner.cache_ops = backend.create_cache_ops(owner)
        return owner.cache_ops

    def capture_lengths(ops):
        query, key = ops.attention_lengths
        q_out, k_out = torch.empty_like(query), torch.empty_like(key)
        q_out.copy_(query)
        k_out.copy_(key)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        graphs.append(graph)
        with torch.npu.graph(graph):
            q_out.copy_(query)
            k_out.copy_(key)
        return graph, q_out, k_out

    try:
        first = runtime()
        layers = {"target": SimpleNamespace(hisparse_cache=SimpleNamespace(runtime=first))}
        prepare_hisparse_workspaces(layers)
        pending = first._ascend_profiled_cache_ops
        workspace = pending[0]
        prepare_hisparse_workspaces(layers)
        assert first._ascend_profiled_cache_ops is pending
        for tensor in (workspace.first_refs, workspace.errors, *pending[1]):
            assert not torch.is_inference(tensor)
        old = bind(first, pages)
        assert old.workspace is workspace and not hasattr(first, "_ascend_profiled_cache_ops")
        captured = [capture_lengths(old)]
        prepare_hisparse_workspaces(layers)
        new = bind(first, pages + 1)
        assert new.workspace is workspace
        for tensor in new.attention_lengths:
            assert not torch.is_inference(tensor)
        assert new.attention_lengths[1].data_ptr() != old.attention_lengths[1].data_ptr()
        captured.append(capture_lengths(new))
        prepare_hisparse_workspaces(layers)
        third = bind(first, pages + 2)
        assert third.workspace is workspace
        for tensor in third.attention_lengths:
            assert not torch.is_inference(tensor)
        for index, (graph, q_out, k_out) in enumerate(captured):
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(q_out.cpu(), torch.arange(5, dtype=torch.int32))
            torch.testing.assert_close(
                k_out.cpu(), torch.tensor([(pages + index - 1) * stride_rows + 128], dtype=torch.int32)
            )
    finally:
        torch.npu.synchronize()
        for graph in graphs:
            graph.reset()
        registration.close()
        reset_workspace_manager()
