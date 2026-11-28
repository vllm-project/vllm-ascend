# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare TileLang commit with the framework's Triton implementation."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "align,pattern", [(False, "none"), (True, "none"), (True, "mixed"), (True, "cross"), (True, "exact")]
)
@pytest.mark.parametrize("compact_backend", ["tilelang", "triton"])
@pytest.mark.parametrize("split_boundary", [False, True])
@pytest.mark.parametrize("mapped", [False, True])
def test_commit_and_compact(state_dtype, align, pattern, mapped, compact_backend, split_boundary):
    pytest.importorskip("torch_npu")
    pytest.importorskip("tilelang.ascend.language")
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

    from vllm_ascend.attention.recoverssm_gdn import GDNRecoverSSMCommitContext
    from vllm_ascend.device.device_config import get_ascend_device_type
    from vllm_ascend.device.hardware import AscendDeviceType

    if get_ascend_device_type() != AscendDeviceType.A5:
        pytest.skip("requires an A5 NPU")
    device = "npu"
    torch.manual_seed(22552)
    batch, pages, heads, window, channels = 4, 9, 12, 4, 2560
    state = torch.randn((pages, heads, 128, 128), device=device, dtype=state_dtype) * 0.02
    replay = torch.randn((pages, heads, window, 257), device=device, dtype=torch.float32) * 0.02
    replay[..., 256] = 0.9
    conv = torch.randn((pages, 6, channels), device=device, dtype=torch.bfloat16).transpose(1, 2)
    accepted = torch.tensor([0, 1, 3, 4], device=device, dtype=torch.int32)
    indices = torch.tensor([0, 2, 1, 3], device=device, dtype=torch.int32)
    offsets = torch.tensor([0, 0, 4, 7, 11], device=device, dtype=torch.int32)
    table = torch.stack((indices, indices + batch), dim=1) if align else None
    computed = torch.full((batch,), 1023, device=device, dtype=torch.int32) if align else None
    if align and pattern == "none":
        computed.zero_()
    elif align and pattern == "mixed":
        computed[::2] = 0
    elif align and pattern == "exact":
        computed.copy_(1024 - accepted)
    mapping = torch.tensor([2, 0, 3, 1], device=device, dtype=torch.int32) if mapped else None
    if mapped:
        reordered = torch.empty_like(accepted)
        reordered[mapping.long()] = accepted
        accepted = reordered
        if align:
            reordered_table = torch.empty_like(table)
            reordered_table[mapping.long()] = table
            table = reordered_table
            reordered_computed = torch.empty_like(computed)
            reordered_computed[mapping.long()] = computed
            computed = reordered_computed

    expected = None
    for backend in ("triton", "tilelang"):
        layers = []
        for _ in range(2):
            c = conv.clone()
            layers.append(
                SimpleNamespace(
                    kv_cache=[c if is_conv_state_dim_first() else c.transpose(-1, -2), state.clone(), replay.clone()],
                    num_k_heads=4,
                    tp_size=1,
                    gdn_recoverssm_backend=backend,
                )
            )
        ctx = GDNRecoverSSMCommitContext.create(layers, spec_query_len=window, max_num_reqs=8)

        def commit(ctx=ctx, backend=backend):
            if backend == "tilelang":
                from vllm_ascend.ops.tilelang_recoverssm.full_backend import commit_step

                commit_step(
                    ctx,
                    accepted,
                    indices,
                    offsets,
                    mapping,
                    table,
                    computed,
                    1024 if align else None,
                    compact_backend=compact_backend,
                    split_boundary=split_boundary,
                )
            else:
                ctx.commit(accepted, indices, offsets, mapping, table, computed, 1024 if align else None)

        commit()
        torch.npu.synchronize()
        actual = [t.cpu().clone() for cache in ctx.layer_caches for t in cache.all_pages()]
        if expected is None:
            expected = actual
        for i, (a, e) in enumerate(zip(actual, expected)):
            tol = 8e-3 if i % 3 == 1 else 0
            torch.testing.assert_close(a.float(), e.float(), atol=tol, rtol=tol)
        for cache in ctx.layer_caches:
            torch.testing.assert_close(cache.replay, replay, atol=0, rtol=0)
            cache.conv.copy_(conv)
            cache.checkpoint.copy_(state)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            commit()
        graph.replay()
        torch.npu.synchronize()
        for i, (a, e) in enumerate(zip([t.cpu() for c in ctx.layer_caches for t in c.all_pages()], expected)):
            tol = 8e-3 if i % 3 == 1 else 0
            torch.testing.assert_close(a.float(), e.float(), atol=tol, rtol=tol)


@pytest.mark.parametrize("align", [False, True])
@pytest.mark.parametrize("value_tile", [64, 128])
@pytest.mark.parametrize("row_group", [2, 4])
def test_state_pipeline_empty_rows_reuse_and_tail(align, value_tile, row_group):
    """Exercise both UB slots repeatedly, skipped waves and pooled page strides."""
    pytest.importorskip("torch_npu")
    tilelang = pytest.importorskip("tilelang")
    pytest.importorskip("tilelang.ascend.language")
    from vllm_ascend.device.device_config import get_ascend_device_type
    from vllm_ascend.device.hardware import AscendDeviceType
    from vllm_ascend.ops.tilelang_recoverssm.state_commit import state_commit_kernel

    if get_ascend_device_type() != AscendDeviceType.A5:
        pytest.skip("requires an A5 NPU")
    torch.manual_seed(2255201)
    batch, layers, heads, window, capacity = 9, 2, 3, 4, 12
    pages = 2 * batch + 1
    # Page pitch includes an unused head, as in a larger pooled allocation.
    initial = torch.randn(pages, heads + 1, 128, 128).to(torch.bfloat16) * 0.02
    # Distinct neighboring K lanes detect a missing even/odd key permutation.
    initial += torch.arange(128).to(torch.bfloat16)[None, None, None, :] / 128
    replay = torch.randn(pages, heads, window, 257) * 0.2
    replay[..., 256] = 0.9
    counts = [0, 1, 2, 3, 4, 2, 4, 1, 3]
    srcs = [0, 3, 1, 2, 5, 0, 4, 7, 6]
    destinations = [0, 3, 10, 11, 5, 0, 12, 7, 13]
    mids = [0, 0, 1, 14, 15, 0, 12, 0, 16] if align else [0] * batch
    cuts = [0, 0, 1, 2, 3, 0, 2, 0, 1] if align else [0] * batch
    for src, count in zip(srcs, counts):
        if src:
            replay[src, :, count:] = float("nan")
    expected = initial.clone()
    for src, dst, mid, count, cut in zip(srcs, destinations, mids, counts, cuts):
        if not count or not src or not dst:
            continue
        state = initial[src, :heads].float()
        for pos in range(count):
            record = replay[src, :, pos]
            state = state * record[:, 256, None, None] + record[:, :128, None] * record[:, None, 128:256]
            if align and 0 < mid < pages and mid != dst and pos + 1 == cut:
                expected[mid, :heads] = state.to(torch.bfloat16)
        expected[dst, :heads] = state.to(torch.bfloat16)
    states = [initial.to("npu") for _ in range(layers)]
    records = [replay.to("npu") for _ in range(layers)]
    pointers = torch.tensor(
        [t.data_ptr() for t in states]
        + [t.stride(0) for t in states]
        + [t.data_ptr() for t in records]
        + [t.stride(0) for t in records],
        dtype=torch.int64,
        device="npu",
    )
    indices = torch.zeros(batch * 2, dtype=torch.int32)
    indices[::2] = torch.tensor(srcs, dtype=torch.int32)
    indices = indices.to("npu")
    plan = torch.zeros(6, capacity, dtype=torch.int32)
    plan[0, :batch] = torch.tensor(counts, dtype=torch.int32)
    plan[3, :batch] = torch.tensor(destinations, dtype=torch.int32)
    plan[4, :batch] = torch.tensor(mids, dtype=torch.int32)
    plan[5, :batch] = torch.tensor(cuts, dtype=torch.int32)
    plan = plan.flatten().to("npu")
    unused_conv = torch.zeros(4 * layers, dtype=torch.int64, device="npu")
    config = (
        batch,
        layers,
        pages,
        heads,
        window,
        1,
        3,
        capacity,
        (batch * 2, (2,), "int32"),
        "bfloat16",
        "bfloat16",
        states[0].stride()[1:],
        records[0].stride()[1:],
        align,
        False,
    )
    kernel = tilelang.compile(
        state_commit_kernel(*config, value_tile=value_tile, cap=7, row_group=row_group),
        target="ascend",
        pass_configs={"tl.enable_auto_schedule": False},
    )

    def run():
        kernel(pointers, unused_conv, indices, plan)

    run()
    torch.npu.synchronize()
    for state in states:
        torch.testing.assert_close(state.cpu(), expected, atol=0, rtol=0)
        state.copy_(initial)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    # Reset before each replay: commit intentionally updates some pages in place.
    for _ in range(3):
        for state in states:
            state.copy_(initial)
        graph.replay()
        torch.npu.synchronize()
        for state in states:
            torch.testing.assert_close(state.cpu(), expected, atol=0, rtol=0)
    plan.zero_()
    for state in states:
        state.copy_(initial)
    run()
    torch.npu.synchronize()
    for state in states:
        torch.testing.assert_close(state.cpu(), initial, atol=0, rtol=0)


@pytest.mark.parametrize("conv_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("channels", [19, 2560, 5120, 8192])
def test_compact_whole_row_pool_stride_and_alias(conv_dtype, channels):
    """Check real DMA/tail copies and slot reuse against independent CPU slices."""
    pytest.importorskip("torch_npu")
    tilelang = pytest.importorskip("tilelang")
    from vllm_ascend.ops.tilelang_recoverssm.compact import compact_kernel

    torch.manual_seed(2255202)
    batch, layers, pages, capacity = 9, 2, 19, 12
    counts = [0, 1, 2, 3, 4, 2, 4, 1, 3]
    srcs = [0, 3, 1, 2, 5, 0, 4, 7, 6]
    destinations = [0, 3, 10, 11, 5, 0, 12, 7, 13]

    for align in (False, True):
        mids = [0, 0, 1, 14, 15, 0, 12, 0, 16] if align else [0] * batch
        cuts = [0, 0, 1, 2, 3, 0, 2, 0, 1] if align else [0] * batch
        for storage_layout in ("contiguous_channel", "pooled_offset", "contiguous_time"):
            if storage_layout == "contiguous_time":
                initial = torch.randn(pages, channels, 6).to(conv_dtype)

                def view(tensor):
                    return tensor
            elif storage_layout == "pooled_offset":
                initial = torch.randn(pages, 6, channels + 17).to(conv_dtype)

                def view(tensor):
                    return tensor[:, :, 7 : 7 + channels].transpose(1, 2)
            else:
                initial = torch.randn(pages, 6, channels).to(conv_dtype)

                def view(tensor):
                    return tensor.transpose(1, 2)

            expected = initial.clone()
            source = view(initial)
            target = view(expected)
            for src, dst, mid, count, cut in zip(srcs, destinations, mids, counts, cuts):
                if not count or not src or not dst:
                    continue
                final = source[src, :, count - 1 : count + 2].clone()
                if align and 0 < mid < pages and mid != dst and 0 < cut <= count:
                    target[mid, :, :3] = source[src, :, cut - 1 : cut + 2]
                target[dst, :, :3] = final
            allocations = [initial.to("npu") for _ in range(layers)]
            caches = [view(tensor) for tensor in allocations]
            table = torch.tensor(
                [t.data_ptr() for t in caches]
                + [t.stride(0) for t in caches]
                + [t.stride(1) for t in caches]
                + [t.stride(2) for t in caches],
                dtype=torch.int64,
                device="npu",
            )
            index = torch.zeros(batch * 2, dtype=torch.int32)
            index[::2] = torch.tensor(srcs, dtype=torch.int32)
            index = index.to("npu")
            plan = torch.zeros(6, capacity, dtype=torch.int32)
            for row, values in ((0, counts), (3, destinations), (4, mids), (5, cuts)):
                plan[row, :batch] = torch.tensor(values, dtype=torch.int32)
            plan = plan.flatten().to("npu")
            config = (
                batch,
                layers,
                pages,
                3,
                4,
                channels,
                3,
                capacity,
                (batch * 2, (2,), "int32"),
                "bfloat16",
                str(conv_dtype).removeprefix("torch."),
                (16384, 128, 1),
                (1028, 257, 1),
                align,
                True,
            )
            kernel = tilelang.compile(
                compact_kernel(*config, cap=7, shared_dim_stride=caches[0].stride(1)),
                target="ascend",
                pass_configs={"tl.enable_auto_schedule": False},
            )
            unused = torch.zeros(4 * layers, dtype=torch.int64, device="npu")

            def run(kernel=kernel, unused=unused, table=table, index=index, plan=plan):
                kernel(unused, table, index, plan)

            run()
            torch.npu.synchronize()
            for allocation in allocations:
                torch.testing.assert_close(allocation.cpu(), expected, atol=0, rtol=0)
                allocation.copy_(initial)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                run()
            for _ in range(3):
                for allocation in allocations:
                    allocation.copy_(initial)
                graph.replay()
                torch.npu.synchronize()
                for allocation in allocations:
                    torch.testing.assert_close(allocation.cpu(), expected, atol=0, rtol=0)
            plan.zero_()
            for allocation in allocations:
                allocation.copy_(initial)
            run()
            torch.npu.synchronize()
            for allocation in allocations:
                torch.testing.assert_close(allocation.cpu(), initial, atol=0, rtol=0)
