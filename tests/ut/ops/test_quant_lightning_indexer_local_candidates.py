# SPDX-License-Identifier: Apache-2.0
"""A5 QLI local-candidate contract, independently scored on CPU.

Run on Ascend 950 after building the repository custom QLI package and binding.
The standalone validation bridge registers the same _C_ascend schema in its
isolated test process; normal validation uses the production plugin binding.
"""

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")


def _ops():
    return torch.ops._C_ascend


def _qli_comparator_values(values):
    # The current vector kernel uses CAST_ROUND, whose BF16 midpoint rule is
    # ties away from zero. torch.bfloat16 uses RNE and is a different oracle
    # at exact midpoints (e.g. 12320 -> 12352 versus 12288).
    absolute = values.double().abs()
    _, exponent = torch.frexp(absolute)
    step = torch.ldexp(torch.ones_like(absolute), exponent - 8)
    rounded = torch.floor(absolute / step + 0.5) * step
    return torch.copysign(rounded, values.double()).float()


def test_local_candidates_meta():
    q = torch.empty(3, 32, 128, device="meta", dtype=torch.float8_e4m3fn)
    key = torch.empty(8, 128, 1, 128, device="meta", dtype=torch.float8_e4m3fn)
    weights = torch.empty(3, 32, device="meta", dtype=torch.float32)
    key_scale = torch.empty(8, 128, 1, device="meta", dtype=torch.float32)
    cu_q = torch.empty(2, device="meta", dtype=torch.int32)
    used_k = torch.empty(1, device="meta", dtype=torch.int32)
    blocks = torch.empty(1, 8, device="meta", dtype=torch.int32)
    visible = torch.empty(3, 1, device="meta", dtype=torch.int32)
    metadata = torch.empty(1024, device="meta", dtype=torch.int32)
    args = (q, key, weights, weights, key_scale, cu_q, used_k, blocks, visible, metadata, 2048)
    indices, scores = _ops().npu_quant_lightning_indexer_local_candidates(*args)
    assert indices.shape == scores.shape == (3, 1, 2048)
    assert indices.dtype == torch.int32 and scores.dtype == torch.bfloat16
    with pytest.raises(RuntimeError, match="INT32"):
        _ops().npu_quant_lightning_indexer_local_candidates(*args[:8], visible.float(), *args[9:])


def _case(kv_len, visible, strided=False, all_ties=False, topk=2048):
    block_size = 128
    num_blocks = max(1, (kv_len + block_size - 1) // block_size)
    t = len(visible)
    # Exact binary fractions make the independently evaluated comparison key
    # unambiguous while including negative, zero and highly tied scores.
    token_values = (
        torch.ones(num_blocks * block_size)
        if all_ties
        else ((torch.arange(num_blocks * block_size) * 17 % 33 - 8).float() / 8)
    )
    key_cpu = token_values[:, None].expand(-1, 128).reshape(num_blocks, block_size, 1, 128)
    key_cpu = key_cpu.to(torch.float8_e4m3fn)
    token_values = key_cpu.float().reshape(-1, 128)[:, 0]
    key_scale_cpu = (2.0 ** (torch.arange(num_blocks * block_size) % 3 - 1)).reshape(num_blocks, block_size, 1)
    if all_ties:
        key_scale_cpu.fill_(1)
    q_cpu = torch.ones(t, 32, 128, dtype=torch.float32).to(torch.float8_e4m3fn)
    q_scale_cpu = torch.ones(t, 32)
    weights_cpu = torch.ones(t, 32)
    key = key_cpu.npu()
    key_scale = key_scale_cpu.npu()
    if strided == "paired":
        backing = torch.empty(num_blocks, block_size * 132, device="npu", dtype=torch.uint8)
        key_view = backing[:, : block_size * 128].view(torch.float8_e4m3fn).reshape(num_blocks, block_size, 1, 128)
        scale_view = backing[:, block_size * 128 :].view(torch.float32).reshape(num_blocks, block_size, 1)
        key_view.copy_(key)
        scale_view.copy_(key_scale)
        key, key_scale = key_view, scale_view
    elif strided:
        key_backing = torch.empty(num_blocks * 2, block_size, 1, 128, device="npu", dtype=key.dtype)
        scale_backing = torch.empty(num_blocks * 2, block_size, 1, device="npu", dtype=torch.float32)
        key_backing[::2].copy_(key)
        scale_backing[::2].copy_(key_scale)
        key, key_scale = key_backing[::2], scale_backing[::2]
    cu_q = torch.tensor([0, t], dtype=torch.int32, device="npu")
    used_k = torch.tensor([kv_len], dtype=torch.int32, device="npu")
    block_table = torch.arange(num_blocks, dtype=torch.int32, device="npu").reshape(1, -1)
    lengths = torch.tensor(visible, dtype=torch.int32, device="npu").reshape(t, 1)
    metadata = _ops().npu_quant_lightning_indexer_v2_metadata(
        32,
        1,
        128,
        topk,
        1,
        cu_seqlens_q=cu_q,
        seqused_k=used_k,
        batch_size=1,
        max_seqlen_q=-1,
        max_seqlen_k=-1,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=0,
        cmp_ratio=1,
    )
    args = (
        q_cpu.npu(),
        key,
        weights_cpu.npu(),
        q_scale_cpu.npu(),
        key_scale,
        cu_q,
        used_k,
        block_table,
        lengths,
        metadata,
        topk,
    )
    # All heads are complete. Dot products and all intermediate values are
    # exactly represented; selection uses the existing BF16 comparator.
    scores = _qli_comparator_values((token_values * 4096 * key_scale_cpu.flatten()).clamp_min(0))
    return args, scores


def _check(outputs, reference, visible, kv_len, topk=2048):
    indices, values = (v.cpu() for v in outputs)
    assert indices.dtype == torch.int32 and values.dtype == torch.bfloat16
    assert indices.shape == values.shape == (len(visible), 1, topk)
    for row, count in enumerate(visible):
        count = max(0, min(count, kv_len))
        ids, scores = indices[row, 0], values[row, 0]
        valid = ids >= 0
        assert int(valid.sum()) == min(count, topk)
        assert torch.isneginf(scores[~valid]).all()
        expected = torch.argsort(reference[:count], descending=True, stable=True)[:topk]
        assert torch.equal(ids[valid].sort().values.long(), expected.sort().values)
        expected_scores = reference[ids[valid].long()]
        if not torch.equal(scores[valid], expected_scores):
            bad = scores[valid] != expected_scores
            raise AssertionError(
                {
                    "row": row,
                    "count": count,
                    "bad_scores": int(bad.sum()),
                    "ids": ids[valid][bad][:8].tolist(),
                    "actual": scores[valid][bad][:8].tolist(),
                    "reference": expected_scores[bad][:8].tolist(),
                }
            )


@pytest.mark.parametrize(
    "kv_len,visible,strided,ties",
    [
        (0, [0], False, False),
        (4097, [0, 1, 767, 768, 769, 2047, 2048, 2049, 4097], False, False),
        (4097, [1, 767, 2049, 4097], True, False),
        (32769, [32769], False, True),
        (32769, [2049], True, False),
        (32769, [32769], False, False),
        (4097, [0, 767, 768, 769, 2049, 4097], "paired", False),
    ],
)
def test_local_candidates_eager(kv_len, visible, strided, ties):
    args, reference = _case(kv_len, visible, strided, ties)
    _check(_ops().npu_quant_lightning_indexer_local_candidates(*args), reference, visible, kv_len)


def test_local_candidates_graph_visibility_is_dynamic():
    args, reference = _case(4097, [769])
    for _ in range(3):
        _ops().npu_quant_lightning_indexer_local_candidates(*args)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = _ops().npu_quant_lightning_indexer_local_candidates(*args)
    graph.replay()
    _check(outputs, reference, [769], 4097)
    args[8].zero_()
    graph.replay()
    _check(outputs, reference, [0], 4097)


def test_local_candidates_varlen_requests():
    visible = [0, 0, 2049, 1, 7, 13]
    args, reference = _case(4097, visible, "paired")
    args = list(args)
    args[5] = torch.tensor([0, 2, 3, 6], dtype=torch.int32, device="npu")
    args[6] = torch.tensor([0, 4097, 13], dtype=torch.int32, device="npu")
    args[7] = args[7].expand(3, -1).contiguous()
    args[9] = _ops().npu_quant_lightning_indexer_v2_metadata(
        32,
        1,
        128,
        2048,
        1,
        cu_seqlens_q=args[5],
        seqused_k=args[6],
        batch_size=3,
        max_seqlen_q=-1,
        max_seqlen_k=-1,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=0,
        cmp_ratio=1,
    )
    _check(_ops().npu_quant_lightning_indexer_local_candidates(*args), reference, visible, 4097)


@pytest.mark.parametrize("negative_weights", [False, True])
def test_local_candidates_complete_head_scales(negative_weights):
    visible = [2049, 4097]
    args, _ = _case(4097, visible, "paired")
    args = list(args)
    weights = 2.0 ** (torch.arange(32) % 3 - 1)
    q_scales = 2.0 ** (torch.arange(32) % 2 - 1)
    if negative_weights:
        weights = -weights
    args[2].copy_(weights.expand(2, -1))
    args[3].copy_(q_scales.expand(2, -1))
    key_values = args[1].cpu().float().reshape(-1, 128)[:, 0]
    key_scales = args[4].cpu().flatten()
    reference = _qli_comparator_values(key_values.clamp_min(0) * 128 * (weights * q_scales).sum() * key_scales)
    _check(_ops().npu_quant_lightning_indexer_local_candidates(*args), reference, visible, 4097)


def test_local_candidates_future_scores_cannot_displace_past():
    args, _ = _case(4097, [2049], "paired")
    key = args[1]
    # Huge future scores would completely displace visible zero-score keys if
    # causality were applied only after Top-K.
    future_values = torch.zeros(key.shape[0] * key.shape[1], 128)
    future_values[2049:].fill_(8)
    key.copy_(future_values.reshape(key.shape).to(key.dtype).npu())
    reference = torch.zeros(4097)
    _check(_ops().npu_quant_lightning_indexer_local_candidates(*args), reference, [2049], 4097)
