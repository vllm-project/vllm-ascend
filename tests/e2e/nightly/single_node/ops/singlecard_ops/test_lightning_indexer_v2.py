# SPDX-License-Identifier: Apache-2.0
"""Run on an NPU with CANN LightningIndexerV2 and V2Metadata installed."""

import pytest
import torch

pytest.importorskip("torch_npu")
import vllm_ascend.vllm_ascend_C  # noqa: E402,F401
from vllm_ascend.ops.lightning_indexer_v2 import (  # noqa: E402
    lightning_indexer_v2,
    lightning_indexer_v2_metadata,
)


@pytest.mark.parametrize("return_value", [0, 1])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("cmp_ratio", [1, 4])
def test_paged_values_and_causal_mask(return_value, strided, cmp_ratio):
    # Four independent requests exercise different compression residuals.
    # One nonzero head/dimension yields exact, tie-free positive scores.
    device = "npu:0"
    lengths = torch.tensor([32, 33, 34, 35], dtype=torch.int32)
    query_lengths = 4
    groups = lengths // cmp_ratio
    q = torch.zeros((16, 4, 128), dtype=torch.bfloat16, device=device)
    q[:, 0, 0] = 1
    w = torch.zeros((16, 4), dtype=torch.float32, device=device)
    w[:, 0] = 1
    storage = torch.zeros((8, 16, 1, 128), dtype=torch.bfloat16, device=device)
    cache = storage[::2] if strided else storage[:4]
    cache[..., 0] = torch.arange(1, 65, device=device).reshape(4, 16, 1)
    # Each request shares the same logical sequence; pages need not be unique.
    table = torch.arange(4, dtype=torch.int32, device=device).expand(4, -1).contiguous()
    cu_q = torch.arange(0, 17, 4, dtype=torch.int32, device=device)
    used_k = groups.to(device)
    residual = (lengths % cmp_ratio).to(device) if cmp_ratio != 1 else None
    common = dict(cu_seqlens_q=cu_q, seqused_k=used_k,
                  cmp_residual_k=residual, layout_q="TND", layout_k="PA_BBND",
                  mask_mode=3, cmp_ratio=cmp_ratio)
    metadata = lightning_indexer_v2_metadata(4, 1, 128, 16, batch_size=4, **common)
    assert metadata.shape == (2048,)
    assert metadata.dtype == torch.int32
    indices, values = lightning_indexer_v2(
        q, cache, w, 16, metadata=metadata, block_table=table,
        return_value=return_value, **common)
    actual = indices.cpu().squeeze(1)
    assert actual.shape == (16, 16)
    assert actual.dtype == torch.int32
    assert values.dtype == torch.float32
    for request, length in enumerate(lengths.tolist()):
        for row in range(query_lengths):
            visible = (length - query_lengths + row + 1) // cmp_ratio
            count = min(16, visible)
            selected = actual[request * query_lengths + row]
            expected = set(range(visible - count, visible))
            assert set(selected[selected >= 0].tolist()) == expected
            assert (selected == -1).sum().item() == 16 - count
            if return_value:
                scores = values.cpu()[request * query_lengths + row, 0]
                torch.testing.assert_close(scores[selected >= 0],
                                           selected[selected >= 0].float() + 1)
                assert torch.isneginf(scores[selected < 0]).all()
    if not return_value:
        assert values.shape == (0,)



@pytest.mark.parametrize("layout_q,layout_k", [("TND", "PA_BBND"), ("TND", "TND"), ("BSND", "BSND")])
@pytest.mark.parametrize("return_value", [0, 1])
def test_meta_outputs(layout_q, layout_k, return_value):
    qshape = (7, 4, 128) if layout_q == "TND" else (2, 7, 4, 128)
    kshape = (32, 1, 128) if layout_k == "TND" else (2, 32, 1, 128)
    q = torch.empty(qshape, device="meta", dtype=torch.bfloat16)
    k = torch.empty(kshape, device="meta", dtype=torch.bfloat16)
    w = torch.empty(qshape[:-1], device="meta", dtype=torch.float32)
    indices, values = lightning_indexer_v2(q, k, w, 16, layout_q=layout_q,
                                          layout_k=layout_k, return_value=return_value)
    expected = (7, 1, 16) if layout_q == "TND" else (2, 7, 1, 16)
    assert indices.shape == expected
    assert indices.dtype == torch.int32
    assert values.shape == (expected if return_value else (0,))
    assert values.dtype == torch.float32
    metadata = lightning_indexer_v2_metadata(4, 1, 128, 16, seqused_k=torch.empty(2, device="meta", dtype=torch.int32))
    assert metadata.device.type == "meta"
    assert metadata.shape == (2048,)


@pytest.mark.parametrize("layout", ["BSND", "TND"])
def test_dense_layout_and_graph_replay(layout):
    q = torch.zeros((1, 2, 4, 128), dtype=torch.bfloat16, device="npu:0")
    q[..., 0, 0] = 1
    k = torch.zeros((1, 32, 1, 128), dtype=torch.bfloat16, device=q.device)
    # Include negative scores to exercise ReLU before head reduction.
    k[0, :, 0, 0] = torch.arange(-16, 16, device=q.device)
    w = torch.zeros((1, 2, 4), dtype=torch.float32, device=q.device)
    w[..., 0] = 1
    cu_q = torch.tensor([0, 2], dtype=torch.int32, device=q.device) if layout == "TND" else None
    cu_k = torch.tensor([0, 32], dtype=torch.int32, device=q.device) if layout == "TND" else None
    if layout == "TND":
        q, k, w = q.flatten(0, 1), k.flatten(0, 1), w.flatten(0, 1)
    # BSND has no cumulative lengths, but metadata still needs an NPU tensor
    # for device dispatch. Explicit K usage preserves the full 32-token range.
    used_k = torch.tensor([32], dtype=torch.int32, device=q.device) if layout == "BSND" else None
    common = dict(cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, seqused_k=used_k,
                  layout_q=layout, layout_k=layout)

    def call():
        metadata = lightning_indexer_v2_metadata(4, 1, 128, 8, batch_size=1,
                                                max_seqlen_q=2, max_seqlen_k=32, **common)
        return lightning_indexer_v2(q, k, w, 8, metadata=metadata, return_value=1, **common)

    def check(result, scale):
        indices, values = (tensor.cpu().reshape(2, 8) for tensor in result)
        assert all(set(row.tolist()) == set(range(24, 32)) for row in indices)
        torch.testing.assert_close(values, (indices.float() - 16) * scale)

    check(call(), 1)
    compiled = torch.compile(call, backend="npugraph_ex", fullgraph=True, dynamic=False)
    check(compiled(), 1)
    call()  # Warm up before graph capture.
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        captured = call()
    graph.replay()
    torch.npu.synchronize()
    check(captured, 1)
    w.mul_(2)
    graph.replay()
    torch.npu.synchronize()
    check(captured, 2)


@pytest.mark.parametrize("case", ["empty_query", "invalid_layout", "invalid_return"])
def test_wrapper_rejects_invalid_inputs(case):
    q = torch.zeros((1, 4, 128), dtype=torch.bfloat16, device="npu:0")
    k = torch.zeros((1, 16, 1, 128), dtype=torch.bfloat16, device=q.device)
    w = torch.ones((1, 4), dtype=torch.float32, device=q.device)
    options = dict(layout_q="TND", layout_k="PA_BBND")
    if case == "empty_query":
        q = q[:0]
        w = w[:0]
        message = "Tensor q is empty"
    elif case == "invalid_layout":
        options["layout_q"] = "INVALID"
        message = "Unsupported query layout"
    else:
        options["return_value"] = 2
        message = "return_value must be 0 or 1"
    with pytest.raises(RuntimeError, match=message):
        lightning_indexer_v2(q, k, w, 16, **options)
