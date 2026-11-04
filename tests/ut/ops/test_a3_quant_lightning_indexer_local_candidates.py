# SPDX-License-Identifier: Apache-2.0
"""Native INT8 owner-local QLI contract; requires a rebuilt A3 custom-op package.

These tests validate native comparator values, not a dequantized attention
fallback. Keep the A5 contract in test_quant_lightning_indexer_local_candidates.py
unchanged and run it independently on A5 hardware after changes to shared code.
Current A3 metadata never enables LD; long-sequence cases exercise ordinary
accumulation. The private LD score-output path remains hardware-unverified.
"""

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(torch.npu.is_available() is not True, reason="requires an A3 NPU")


def _ops():
    return torch.ops._C_ascend


def test_a3_local_candidates_meta_and_dtype_validation():
    q = torch.empty(3, 32, 128, device="meta", dtype=torch.int8)
    key = torch.empty(8, 128, 1, 128, device="meta", dtype=torch.int8)
    weights = torch.empty(3, 32, device="meta", dtype=torch.float16)
    scale = torch.empty(8, 128, 1, device="meta", dtype=torch.float16)
    cu_q = torch.empty(2, device="meta", dtype=torch.int32)
    used_k = torch.empty(1, device="meta", dtype=torch.int32)
    blocks = torch.empty(1, 8, device="meta", dtype=torch.int32)
    visible = torch.empty(3, 1, device="meta", dtype=torch.int32)
    metadata = torch.empty(1024, device="meta", dtype=torch.int32)
    args = (q, key, weights, weights, scale, cu_q, used_k, blocks, visible, metadata, 2048)
    indices, scores = _ops().npu_quant_lightning_indexer_local_candidates(*args)
    assert indices.shape == scores.shape == (3, 1, 2048)
    assert indices.dtype == torch.int32 and scores.dtype == torch.float32
    with pytest.raises(RuntimeError, match="FP16"):
        _ops().npu_quant_lightning_indexer_local_candidates(*args[:2], weights.float(), *args[3:])
    with pytest.raises(RuntimeError, match="2048"):
        _ops().npu_quant_lightning_indexer_local_candidates(*args[:-1], 4096)


def _metadata(cu_q, used_k, heads, topk):
    return _ops().npu_quant_lightning_indexer_v2_metadata(
        heads,
        1,
        128,
        topk,
        2,
        cu_seqlens_q=cu_q,
        seqused_k=used_k,
        batch_size=used_k.numel(),
        max_seqlen_q=-1,
        max_seqlen_k=-1,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=0,
        cmp_ratio=1,
    )


def _case(kv_len, visible, *, heads=32, strided=False, ties=False, topk=2048):
    block_size = 128
    blocks = max(1, (kv_len + block_size - 1) // block_size)
    rows = len(visible)
    token_values = ((torch.arange(blocks * block_size) * 17) % 13 - 4).to(torch.int8)
    if ties:
        token_values.fill_(1)
    key_cpu = token_values[:, None].expand(-1, 128).reshape(blocks, block_size, 1, 128).contiguous()
    scales_cpu = torch.ones(blocks, block_size, 1, dtype=torch.float16)
    key, scales = key_cpu.npu(), scales_cpu.npu()
    if strided == "paired":
        # INT8 key + FP16 scale share a 16640-byte page, with independent
        # leading strides measured in their respective element types.
        backing = torch.empty(blocks, block_size * 130, device="npu", dtype=torch.uint8)
        key = backing[:, : block_size * 128].view(torch.int8).reshape(blocks, block_size, 1, 128)
        scales = backing[:, block_size * 128 :].view(torch.float16).reshape(blocks, block_size, 1)
        key.copy_(key_cpu.npu())
        scales.copy_(scales_cpu.npu())
    elif strided:
        key_backing = torch.empty(blocks * 2, block_size, 1, 128, device="npu", dtype=torch.int8)
        scale_backing = torch.empty(blocks * 3, block_size, 1, device="npu", dtype=torch.float16)
        key, scales = key_backing[::2], scale_backing[::3]
        key.copy_(key_cpu.npu())
        scales.copy_(scales_cpu.npu())
    q = torch.ones(rows, heads, 128, device="npu", dtype=torch.int8)
    weights = torch.ones(rows, heads, device="npu", dtype=torch.float16)
    cu_q = torch.tensor([0, rows], device="npu", dtype=torch.int32)
    used_k = torch.tensor([kv_len], device="npu", dtype=torch.int32)
    args = [
        q,
        key,
        weights,
        weights.clone(),
        scales,
        cu_q,
        used_k,
        torch.arange(blocks, device="npu", dtype=torch.int32).view(1, -1),
        torch.tensor(visible, device="npu", dtype=torch.int32).view(rows, 1),
        _metadata(cu_q, used_k, heads, topk),
        topk,
    ]
    # The inherited A3 Cube normalizes INT32 QK by 1/1024 before FP16
    # ReLU and weight reduction. These fixtures are exact in every stage.
    reference = token_values.float().clamp_min(0) * (heads / 8)
    return args, reference


def _check(outputs, reference, visible, kv_len, topk=2048):
    indices, scores = (tensor.cpu() for tensor in outputs)
    assert indices.dtype == torch.int32 and scores.dtype == torch.float32
    assert indices.shape == scores.shape == (len(visible), 1, topk)
    for row, count in enumerate(visible):
        count = max(0, min(count, kv_len))
        ids, values = indices[row, 0], scores[row, 0]
        valid = ids >= 0
        assert valid.sum().item() == min(count, topk)
        assert (ids[~valid] == -1).all()
        assert torch.isneginf(values[~valid]).all()
        assert ((ids[valid] >= 0) & (ids[valid] < count)).all()
        expected = torch.argsort(reference[:count], descending=True, stable=True)[:topk]
        # This is also the explicit A3 boundary-tie validation gate: a local
        # kernel must not discard smaller IDs before the distributed merge.
        assert torch.equal(ids[valid].sort().values.long(), expected.sort().values)
        assert torch.equal(values[valid], reference[ids[valid].long()])


@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize(
    "kv_len,visible,strided,ties",
    [
        (0, [0], False, False),
        (129, [-3, 0, 1, 127, 128, 129, 1000], False, False),
        (4097, [0, 1, 767, 768, 769, 2047, 2048, 2049, 4097], "paired", False),
        (32769, [0, 1, 2049, 32769], True, False),
        (32769, [32769], False, True),
    ],
)
def test_a3_local_candidates_eager(heads, kv_len, visible, strided, ties):
    args, reference = _case(kv_len, visible, heads=heads, strided=strided, ties=ties)
    _check(_ops().npu_quant_lightning_indexer_local_candidates(*args), reference, visible, kv_len)


@pytest.mark.parametrize("kv_len", [1024, 32769])
def test_a3_scores_preserve_fp32_comparator_bits(kv_len):
    visible = [kv_len]
    args, _ = _case(kv_len, visible, ties=True)
    scale_cpu = (1 + (torch.arange(args[4].numel()) % 1024).float() / 1024).half().view(args[4].shape)
    args[4].copy_(scale_cpu.npu())
    reference = scale_cpu.flatten().float() * 4
    assert (reference != reference.bfloat16().float()).any()
    _check(_ops().npu_quant_lightning_indexer_local_candidates(*args), reference, visible, kv_len)


def test_a3_many_query_rows_cover_non_ld_prefill_output():
    visible = [2049] * 192
    args, reference = _case(2049, visible, strided="paired")
    _check(_ops().npu_quant_lightning_indexer_local_candidates(*args), reference, visible, 2049)


@pytest.mark.parametrize("kv_len", [2049, 4097])
def test_a3_non_ld_equal_scores_keep_smallest_local_ids(kv_len):
    # Enough S1 blocks give each A3 core complete rows, so this exercises
    # accumulation across S2 tiles before the ordinary (non-LD) output.
    args, _ = _case(kv_len, [kv_len] * 192, strided="paired", ties=True)
    indices, scores = (tensor.cpu() for tensor in _ops().npu_quant_lightning_indexer_local_candidates(*args))
    expected = torch.arange(2048, dtype=torch.int32).view(1, 1, -1).expand(192, 1, -1)
    # Newer equal-score tiles must never displace any of IDs 0..2047.
    assert torch.equal(indices.sort(dim=-1).values, expected)
    assert torch.equal(scores, torch.full((192, 1, 2048), 4.0, dtype=torch.float32))


def test_a3_sort32_equal_scores_keep_smallest_local_ids():
    # Truncate inside a single 32-element Sort32 group, independently of
    # cross-tile and LD merge priorities. Scores remain exactly 4.0.
    args, _ = _case(32, [32], ties=True, topk=16)
    indices, scores = (tensor.cpu() for tensor in _ops().npu_quant_lightning_indexer_local_candidates(*args))
    assert torch.equal(indices.flatten().sort().values, torch.arange(16, dtype=torch.int32))
    assert torch.equal(scores, torch.full((1, 1, 16), 4.0, dtype=torch.float32))


def test_a3_future_scores_are_masked_before_topk():
    args, _ = _case(4097, [2049], strided="paired")
    key_cpu = torch.zeros(args[1].shape[0] * 128, 128, dtype=torch.int8)
    key_cpu[2049:].fill_(127)
    args[1].copy_(key_cpu.reshape(args[1].shape).npu())
    _check(_ops().npu_quant_lightning_indexer_local_candidates(*args), torch.zeros(4097), [2049], 4097)


def test_a3_graph_visibility_changes_and_unused_tnd_rows():
    args, reference = _case(4097, [769, 2049, 4097, 4097], strided="paired")
    args[5] = torch.tensor([0, 2], device="npu", dtype=torch.int32)
    args[9] = _metadata(args[5], args[6], 32, 2048)
    for _ in range(3):
        _ops().npu_quant_lightning_indexer_local_candidates(*args)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = _ops().npu_quant_lightning_indexer_local_candidates(*args)
    graph.replay()
    _check(outputs, reference, [769, 2049, 0, 0], 4097)
    args[8].zero_()
    graph.replay()
    _check(outputs, reference, [0, 0, 0, 0], 4097)


def test_a3_varlen_empty_requests_and_local_offset_slot():
    visible = [0, 0, 2049, 1, 7, 13]
    args, reference = _case(4097, visible, strided="paired")
    args[5] = torch.tensor([0, 2, 3, 6], device="npu", dtype=torch.int32)
    args[6] = torch.tensor([0, 4097, 13], device="npu", dtype=torch.int32)
    args[7] = args[7].expand(3, -1).contiguous()
    args[9] = _metadata(args[5], args[6], 32, 2048)
    _check(_ops().npu_quant_lightning_indexer_local_candidates(*args), reference, visible, 4097)


@pytest.mark.parametrize("negative_weights", [False, True])
def test_a3_complete_head_scales(negative_weights):
    visible = [2049, 4097]
    args, _ = _case(4097, visible, strided=True)
    weights = (2.0 ** (torch.arange(32) % 3 - 1)).half()
    q_scales = (2.0 ** (torch.arange(32) % 2 - 1)).half()
    if negative_weights:
        weights.neg_()
    args[2].copy_(weights.expand(2, -1).npu())
    args[3].copy_(q_scales.expand(2, -1).npu())
    tokens = args[1].cpu().reshape(-1, 128)[:, 0].float().clamp_min(0)
    reference = tokens / 8 * (weights * q_scales).float().sum()
    _check(_ops().npu_quant_lightning_indexer_local_candidates(*args), reference, visible, 4097)


@pytest.mark.parametrize("visible", [1, 65537])
def test_a3_sixteen_owner_candidate_union_matches_global_topk(visible):
    # Exercise the native operator on all sixteen owner shards using one NPU.
    # Actual multi-process HCCL/graph integration remains a separate DCP16 gate.
    world, interleave, topk = 16, 768, 2048
    global_tokens = ((torch.arange(65537) * 17) % 13 - 4).to(torch.int8)
    candidates, values = [], []
    for rank in range(world):
        owned = torch.arange(global_tokens.numel())
        owned = owned[(owned // interleave) % world == rank]
        local_visible = int((owned < visible).sum())
        args, _ = _case(owned.numel(), [local_visible], strided="paired")
        local_keys = torch.zeros(args[1].shape[0] * 128, 128, dtype=torch.int8)
        local_keys[: owned.numel()] = global_tokens[owned, None]
        args[1].copy_(local_keys.reshape(args[1].shape).npu())
        ids, scores = (tensor.cpu().flatten() for tensor in _ops().npu_quant_lightning_indexer_local_candidates(*args))
        mapped = (ids // interleave) * (world * interleave) + rank * interleave + ids % interleave
        candidates.append(torch.where(ids >= 0, mapped, -1))
        values.append(scores)
    all_ids, all_scores = torch.cat(candidates), torch.cat(values)
    valid = all_ids >= 0
    all_ids, all_scores = all_ids[valid], all_scores[valid]
    order = torch.argsort(all_ids, stable=True)
    order = order[torch.argsort(all_scores[order], descending=True, stable=True)[:topk]]
    reference = global_tokens[:visible].float().clamp_min(0) * 4
    expected = torch.argsort(reference, descending=True, stable=True)[:topk]
    assert torch.equal(all_ids[order].long(), expected)
