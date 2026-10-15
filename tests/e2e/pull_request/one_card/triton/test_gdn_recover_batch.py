# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.gdn_recover import GDNRecoverContext
from vllm_ascend.ops.triton.mamba.gdn_recover import (
    gdn_commit_conv_history,
    gdn_recover_commit,
    gdn_recover_verify,
    gdn_save_conv_history,
)
from vllm_ascend.ops.triton.mamba.gdn_recover_batch import GDNRecoverBatchLayout


def _page(heads, kd, vd, history, channels, window=4):
    conv = (history + window - 1) * channels * 2
    conv_aligned = (conv + 31) // 32 * 32
    state = heads * vd * kd * 4
    record = heads * window * (vd + kd + 1) * 4
    saved = history * channels * 2
    payload = conv_aligned + state + record + saved
    return (payload + 31) // 32 * 32 + 32, (0, conv_aligned, conv_aligned + state, conv_aligned + state + record)


def _views(raw, *, blocks, physical_layers, page, field_offsets, geometry, base_offset=0):
    heads, kd, vd, history, channels = geometry
    shapes = (
        (blocks, history + 3, channels),
        (blocks, heads, vd, kd),
        (blocks, heads, 4, vd + kd + 1),
        (blocks, history, channels),
    )
    result = []
    for layer in range(physical_layers):
        cache = []
        for shape, dtype, offset in zip(
            shapes, (torch.bfloat16, torch.float32, torch.float32, torch.bfloat16), field_offsets
        ):
            width = torch.empty((), dtype=dtype).element_size()
            strides = [physical_layers * page // width]
            for axis in range(1, len(shape)):
                product = 1
                for dim in shape[axis + 1 :]:
                    product *= dim
                strides.append(product)
            cache.append(raw.view(dtype).as_strided(shape, strides, (base_offset + layer * page + offset) // width))
        result.append(tuple(cache))
    return result


def _verify(cache, ids, offsets, heads, kd, vd):
    tokens = int(offsets[-1].cpu())
    q = torch.nn.functional.normalize(torch.randn(tokens, 1, kd, device="npu"), dim=-1).bfloat16()
    k = torch.nn.functional.normalize(torch.randn_like(q.float()), dim=-1).bfloat16()
    v = (torch.randn(tokens, heads, vd, device="npu") * 0.1).bfloat16()
    g = torch.full((tokens, heads), -0.1, device="npu")
    beta = torch.full_like(g, 0.5)
    gdn_save_conv_history(cache[0], cache[3], ids, offsets)
    gdn_recover_verify(q, k, v, g, beta, cache[1], cache[2], ids, offsets)
    # Independently emulate expanded conv writes; saved pre-verify history stays unchanged.
    cache[0][ids.long()] = torch.randn(ids.numel(), *cache[0].shape[1:], device="npu").bfloat16()


@pytest.mark.parametrize("geometry", [(6, 33, 17, 3, 257), (2, 256, 65, 5, 600), (48, 128, 128, 3, 10240)])
def test_fused_commit_matches_per_layer_across_shared_groups(geometry):
    torch.manual_seed(83)
    heads, kd, vd, history, channels = geometry
    page, fields = _page(*geometry)
    blocks, physical_layers, groups = 12, 2, 3
    raw = torch.full((blocks * physical_layers * page,), 0x6A, dtype=torch.uint8, device="npu")
    physical = _views(
        raw, blocks=blocks, physical_layers=physical_layers, page=page, field_offsets=fields, geometry=geometry
    )
    caches = physical * groups
    for cache in physical:
        cache[1].normal_(0, 0.03)
        cache[0].normal_()
    layout = GDNRecoverBatchLayout.try_create(caches)
    assert layout is not None
    for batch in (1, 3, 2):
        metadata = []
        for group in range(groups):
            ids = torch.arange(1 + group * 3, 1 + group * 3 + batch, dtype=torch.int32, device="npu")
            lengths = [4 if row % 2 == 0 else 2 for row in range(batch)]
            offsets = torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(), dtype=torch.int32, device="npu")
            counts = torch.tensor([(row + group) % 5 for row in range(batch)], dtype=torch.int32, device="npu")
            metadata.append((ids, offsets, counts))
        for layer, cache in enumerate(caches):
            ids, offsets, _ = metadata[layer // physical_layers]
            _verify(cache, ids, offsets, heads, kd, vd)
        golden = raw.clone()
        golden_physical = _views(
            golden, blocks=blocks, physical_layers=physical_layers, page=page, field_offsets=fields, geometry=geometry
        )
        golden_caches = golden_physical * groups
        entries = []
        for layer, (cache, expected) in enumerate(zip(caches, golden_caches)):
            ids, offsets, counts = metadata[layer // physical_layers]
            gdn_recover_commit(expected[1], expected[2], ids, offsets, counts)
            gdn_commit_conv_history(expected[0], expected[3], ids, offsets, counts)
            entries.append((cache, ids, offsets, counts))
        layout.commit(entries)
        assert torch.equal(raw.cpu(), golden.cpu()), "state/conv/records/unselected pages/padding differ"


def test_heterogeneous_allocations_use_per_layer_fallback():
    geometry = (2, 33, 17, 3, 234)
    page, fields = _page(*geometry)
    raw_a = torch.empty(3 * page, dtype=torch.uint8, device="npu")
    raw_b = torch.empty_like(raw_a)
    a = _views(raw_a, blocks=3, physical_layers=1, page=page, field_offsets=fields, geometry=geometry)[0]
    b = _views(raw_b, blocks=3, physical_layers=1, page=page, field_offsets=fields, geometry=geometry)[0]
    assert GDNRecoverBatchLayout.try_create([a, b]) is None


@pytest.mark.parametrize("shared_allocation", [False, True])
def test_context_commits_device_sampled_tokens_in_forward_row_order(shared_allocation):
    geometry = (2, 33, 17, 3, 234)
    heads, kd, vd, _, _ = geometry
    page, fields = _page(*geometry)
    raw = torch.full((4 * 2 * page,), 0x6A, dtype=torch.uint8, device="npu")
    caches = _views(raw, blocks=4, physical_layers=2, page=page, field_offsets=fields, geometry=geometry)
    if not shared_allocation:
        other = raw.clone()
        caches[1] = _views(other, blocks=4, physical_layers=2, page=page, field_offsets=fields, geometry=geometry)[1]
    indices = torch.tensor([1, 3], dtype=torch.int32, device="npu")
    offsets = torch.tensor([0, 4, 8], dtype=torch.int32, device="npu")
    rows = torch.tensor([2, 0], dtype=torch.int64, device="npu")
    sampled = torch.tensor([[11, 12, -1, -1], [21, 22, 23, 24], [31, -1, -1, -1]], device="npu")
    assert sampled.device == rows.device
    expected_counts = torch.tensor([1, 2], dtype=torch.int32, device="npu")
    context = GDNRecoverContext()
    context.begin()
    expected = []
    for layer, cache in enumerate(caches):
        cache[1].normal_(0, 0.03)
        cache[0].normal_()
        _verify(cache, indices, offsets, heads, kd, vd)
        golden = tuple(t.clone() for t in cache)
        gdn_recover_commit(golden[1], golden[2], indices, offsets, expected_counts)
        gdn_commit_conv_history(golden[0], golden[3], indices, offsets, expected_counts)
        expected.append(golden)
        context.enqueue(
            f"target.layer.{layer}",
            cache,
            SimpleNamespace(recover_state_indices=indices, recover_query_offsets=offsets, recover_request_rows=rows),
        )
    context.commit(sampled)
    assert (context._layout is not None) == shared_allocation
    context.commit(sampled)  # No second replay of the same records.
    context.begin()  # The next forward waits for completion before slot reuse.
    context.reset()
    for cache, golden in zip(caches, expected):
        for actual, reference in zip(cache, golden):
            assert torch.equal(actual.cpu(), reference.cpu())


def test_batch_layer_offsets_beyond_eight_gib():
    torch.npu.empty_cache()
    geometry = (2, 33, 17, 3, 234)
    heads, kd, vd, _, _ = geometry
    page, fields = _page(*geometry)
    high = 8 * 1024**3 + 32
    raw = torch.empty(high + 2 * page + 32, dtype=torch.uint8, device="npu")
    raw[: 2 * page].fill_(0x6A)
    raw[high:].fill_(0x6A)
    low = _views(raw, blocks=2, physical_layers=1, page=page, field_offsets=fields, geometry=geometry)[0]
    upper = _views(
        raw, blocks=2, physical_layers=1, page=page, field_offsets=fields, geometry=geometry, base_offset=high
    )[0]
    ids = torch.tensor([1], dtype=torch.int64, device="npu")
    offsets = torch.tensor([0, 4], dtype=torch.int64, device="npu")
    counts = torch.tensor([2], dtype=torch.int64, device="npu")
    for cache in (low, upper):
        cache[1].normal_(0, 0.03)
        cache[0].normal_()
        _verify(cache, ids, offsets, heads, kd, vd)
    golden = torch.cat((raw[: 2 * page], raw[high : high + 2 * page]))
    expected = []
    for base in (0, 2 * page):
        cache = _views(
            golden, blocks=2, physical_layers=1, page=page, field_offsets=fields, geometry=geometry, base_offset=base
        )[0]
        gdn_recover_commit(cache[1], cache[2], ids, offsets, counts)
        gdn_commit_conv_history(cache[0], cache[3], ids, offsets, counts)
        expected.append(cache)
    layout = GDNRecoverBatchLayout.try_create([low, upper])
    assert layout is not None and layout.offsets.dtype == torch.int64
    layout.commit([(low, ids, offsets, counts), (upper, ids, offsets, counts)])
    assert torch.equal(raw[: 2 * page].cpu(), golden[: 2 * page].cpu())
    assert torch.equal(raw[high : high + 2 * page].cpu(), golden[2 * page :].cpu())
    assert torch.equal(raw[high + 2 * page :].cpu(), torch.full((32,), 0x6A, dtype=torch.uint8))
