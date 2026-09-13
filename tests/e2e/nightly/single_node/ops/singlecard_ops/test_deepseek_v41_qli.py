# SPDX-License-Identifier: Apache-2.0
"""Aurora QLI/candidate correctness on A3, including real paged cache views."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from tests.deepseek_v41_cache_utils import allocate_cache_views, make_cache_config
from vllm_ascend.utils import enable_custom_op, is_950

enable_custom_op()

HEADS = 32
WIDTH = 128
PAGE = 64
# Fixed before NPU validation. Values at the TopK cutoff may tie or differ by
# FP16 intermediate rounding; every selected score must meet this cutoff.
RTOL = 1e-3
ATOL = 1e-5


def _paged(value, *, strided=True, page_size=PAGE):
    pages = (value.shape[0] + page_size - 1) // page_size
    shape = (pages, page_size, *value.shape[1:])
    dense = torch.zeros(shape, dtype=value.dtype)
    order = torch.arange(pages - 1, -1, -1)
    padded = F.pad(value.flatten(1), (0, 0, 0, pages * page_size - value.shape[0])).reshape(shape)
    dense[order] = padded
    dense = dense.npu()
    if strided:
        stride = (dense.stride(0) * 2, *dense.stride()[1:])
        backing = torch.zeros(pages * stride[0] + 128, dtype=value.dtype, device="npu")
        view = backing.as_strided(shape, stride, 128)
        view.copy_(dense)
        dense = view
    return dense, order.to(torch.int32).unsqueeze(0).npu()


def _data(length, qlen=5, heads=HEADS, seed=41, strided=True, page_size=PAGE):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randint(-8, 9, (qlen, heads, WIDTH), dtype=torch.int8, generator=gen)
    k = torch.randint(-8, 9, (length, 1, WIDTH), dtype=torch.int8, generator=gen)
    w = (torch.rand(qlen, heads, generator=gen) * 2 - 0.5).half()
    qs = (torch.rand(qlen, heads, generator=gen) / 32 + 0.01).half()
    ks = (torch.rand(length, 1, generator=gen) / 16 + 0.01).half()
    key, table = _paged(k, strided=strided, page_size=page_size)
    scale, _ = _paged(ks, strided=strided, page_size=page_size)
    return q, k, w, qs, ks, key, scale, table


def _scores(q, k, w, qs, ks, ratio, residual, mask=3):
    # Independent CPU form of the vendor INT8 golden's two matmuls. Preserve
    # FP16 QK/1024 and w*qscale rounding before the FP32 head reduction.
    qk = torch.einsum("qhd,kd->qhk", q.float(), k[:, 0].float())
    relu = (qk / 1024).clamp_min(0).half().float()
    score = (relu * (w * qs).float().unsqueeze(-1)).sum(1) * ks[:, 0].float()
    visible = torch.full((q.shape[0],), k.shape[0], dtype=torch.long)
    if mask == 3:
        visible = (k.shape[0] * ratio + residual - q.shape[0] + torch.arange(1, q.shape[0] + 1)) // ratio
        visible.clamp_min_(0)
    score.masked_fill_(torch.arange(k.shape[0])[None, :] >= visible[:, None], -torch.inf)
    return score, visible


def _mask_non_candidates(score, membership):
    # A2/A3 retain reachable non-candidates as low-priority TopK fillers.
    # Causal padding must remain -inf; the A5 kernel is unchanged.
    penalty = -torch.inf if is_950() else -1e30
    score.masked_fill_(~membership & torch.isfinite(score), penalty)


def _check_topk(score, indices, count):
    indices = indices.cpu().reshape(score.shape[0], -1)
    for row, idx in zip(score, indices):
        valid = idx[idx >= 0].long()
        expected_count = min(count, int(torch.isfinite(row).sum()))
        assert valid.numel() == expected_count
        assert valid.unique().numel() == valid.numel()
        assert bool((valid < row.numel()).all())
        if expected_count:
            selected = row[valid]
            assert bool(torch.isfinite(selected).all())
            cutoff = row.topk(expected_count).values[-1]
            assert selected.min() >= cutoff - (ATOL + RTOL * cutoff.abs())


def _check_candidates(score, visible, candidates, blocks):
    block_score = F.pad(score, (0, -score.shape[-1] % 8), value=-torch.inf).unflatten(-1, (-1, 8)).amax(-1)
    for i, n in enumerate(visible):
        if n > 0:
            block_score[i, (n - 1) // 8] = torch.inf
    indices = candidates.cpu().reshape(score.shape[0], -1)
    for row, n, idx in zip(block_score, visible, indices):
        valid = idx[idx >= 0].long()
        reachable = int((n + 7) // 8)
        assert valid.numel() == min(blocks, reachable)
        assert valid.unique().numel() == valid.numel()
        if not reachable:
            continue
        assert int((n - 1) // 8) in valid.tolist()
        cutoff = row.topk(min(blocks, reachable)).values[-1]
        if not torch.isinf(cutoff):
            assert row[valid].min() >= cutoff - (ATOL + RTOL * cutoff.abs())
    return indices


def _invoke(data, layout, ratio, mode, *, candidates=None, blocks=64, mask=3, residual=None, topk=None):
    q, k, w, qs, ks, key, scale, table = data
    if residual is None:
        residual = ratio - 1
    topk = min(128, k.shape[0]) if topk is None else topk
    cu = torch.tensor([0, q.shape[0]], dtype=torch.int32, device="npu")
    used = torch.tensor([k.shape[0]], dtype=torch.int32, device="npu")
    res = torch.tensor([residual], dtype=torch.int32, device="npu")
    common = dict(
        cu_seqlens_q=cu if layout == "TND" else None,
        seqused_k=used,
        cmp_residual_k=res if ratio != 1 and mask != 0 else None,
        max_seqlen_q=q.shape[0],
        layout_q=layout,
        layout_k="PA_BBND",
        mask_mode=mask,
        cmp_ratio=ratio,
    )
    metadata = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
        q.shape[1], 1, WIDTH, topk, 2, batch_size=1, max_seqlen_k=k.shape[0], **common
    )
    query, weights, query_scale = q.npu(), w.npu(), qs.npu()
    if layout == "BSND":
        query, weights, query_scale = (x.unsqueeze(0) for x in (query, weights, query_scale))
    print(
        {
            "layout": layout,
            "ratio": ratio,
            "candidate_mode": mode,
            "q_shape": list(query.shape),
            "k_shape": list(key.shape),
            "k_stride": list(key.stride()),
            "scale_stride": list(scale.stride()),
            "k_offset": key.storage_offset(),
            "topk": topk,
            "candidate_blocks": blocks,
        },
        flush=True,
    )
    output, values, candidate_out = torch.ops._C_ascend.npu_quant_lightning_indexer_v3(
        query,
        key,
        weights,
        query_scale,
        scale,
        topk,
        2,
        candidate_topk_index=candidates,
        block_table=table,
        metadata=metadata,
        candidate_mode=mode,
        candidate_topk_blocks=blocks,
        candidate_block_size=8,
        **common,
    )
    torch.npu.synchronize()
    assert values.numel() == 0
    return output, candidate_out


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("length", [7, 513, 1025])
@pytest.mark.parametrize("mode", [1, 2, 3])
def test_native_qli_candidate(ratio, length, mode):
    layout = "TND"
    data = _data(length)
    score, visible = _scores(data[0], data[1], data[2], data[3], data[4], ratio, ratio - 1)
    candidate_in = None
    if mode == 2:
        _, candidate_in = _invoke(data, layout, ratio, 1)
        _check_candidates(score, visible, candidate_in, 64)
        # A different consumer query must rerank within source blocks.
        other = _data(length, seed=17)
        data = (other[0], data[1], other[2], other[3], *data[4:])
        score, visible = _scores(data[0], data[1], data[2], data[3], data[4], ratio, ratio - 1)
        block_ids = candidate_in.cpu().reshape(data[0].shape[0], -1)
        membership = (torch.arange(length)[None, None, :] // 8 == block_ids[:, :, None]).any(1)
        _mask_non_candidates(score, membership)
    output, candidates = _invoke(data, layout, ratio, mode, candidates=candidate_in)
    _check_topk(score, output, min(128, length))
    if mode == 1:
        _check_candidates(score, visible, candidates, 64)
    else:
        assert candidates.numel() == 0


def test_native_qli_supports_bsnd_query_layout():
    data = _data(513)
    score, _ = _scores(data[0], data[1], data[2], data[3], data[4], 1, 0)
    output, _ = _invoke(data, "BSND", 1, 3)
    _check_topk(score, output, 128)


@pytest.mark.parametrize(
    "ratio,residual,qlen,length,blocks",
    [
        (2, 0, 2, 1, 64),  # First query has no completed compressed key.
        (2, 0, 1, 513, 64),
        (1, 0, 2, 17001, 2048),  # Production candidate width, actual filtering.
    ],
)
def test_candidate_boundaries(ratio, residual, qlen, length, blocks):
    data = _data(length, qlen=qlen)
    score, visible = _scores(data[0], data[1], data[2], data[3], data[4], ratio, residual)
    output, candidates = _invoke(data, "TND", ratio, 1, residual=residual, blocks=blocks)
    _check_topk(score, output, min(128, length))
    _check_candidates(score, visible, candidates, blocks)


@pytest.mark.parametrize("mode", [1, 3])
def test_fake_qli_shapes(mode):
    q = torch.empty(3, 32, 128, device="meta", dtype=torch.int8)
    k = torch.empty(16, 64, 1, 128, device="meta", dtype=torch.int8)
    w = torch.empty(3, 32, device="meta", dtype=torch.float16)
    ks = torch.empty(16, 64, 1, device="meta", dtype=torch.float16)
    out, values, candidates = torch.ops._C_ascend.npu_quant_lightning_indexer_v3(
        q, k, w, w, ks, 128, 2, candidate_mode=mode, candidate_topk_blocks=64
    )
    assert out.shape == (3, 1, 128) and out.dtype == torch.int32
    assert candidates.shape == ((3, 1, 64) if mode == 1 else (0,))
    assert values.shape == (0,)


def _indexer(ratio):
    from vllm_ascend.models.deepseek_v41.indexer import DeepseekV41Indexer

    obj = DeepseekV41Indexer.__new__(DeepseekV41Indexer)
    torch.nn.Module.__init__(obj)
    obj.width, obj.n_heads, obj.index_topk, obj.compress_ratio = WIDTH, HEADS, 128, ratio
    return obj


def _quant_query_reference(query):
    scale = (query.float().abs().amax(-1) / 127).half().clamp_min(2.0**-24)
    quant = (query.float() / scale.float().unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)
    return quant, scale


@pytest.mark.parametrize("ratio,zero_first", [(1, False), (2, False), (2, True)])
def test_model_indexer_mixed_batch(ratio, zero_first):
    page_size = 128 // ratio
    parts = [_data(7, qlen=2, seed=11, page_size=page_size), _data(1025, qlen=3, seed=29, page_size=page_size)]
    key = torch.cat([d[5] for d in parts])
    scale = torch.cat([d[6] for d in parts]).unsqueeze(-1)
    config = make_cache_config(key.shape[0] + 1)
    _, caches = allocate_cache_views(config, "npu")
    source = 2 if ratio == 2 else 20
    slot_key, slot_scale = caches[f"model.layers.{source}.self_attn.indexer.k_cache"]
    slot_key[1:].copy_(key)
    slot_scale[1:].copy_(scale)
    key, scale = slot_key, slot_scale
    assert key.shape[1] == page_size
    assert not key.is_contiguous() and not scale.is_contiguous()
    assert not key[0].any() and not scale[0].any()
    table = torch.zeros(2, parts[1][7].shape[1], dtype=torch.int32, device="npu")
    table[0, : parts[0][7].shape[1]] = parts[0][7][0] + 1
    table[1] = parts[1][7][0] + parts[0][5].shape[0] + 1
    lengths = [0 if zero_first else 7, 1025]
    original = [lengths[0] * ratio + (ratio - 1), 1025 * ratio + (ratio - 1)]
    # A zero-key request must still represent a valid original token range.
    if zero_first:
        parts[0] = _data(7, qlen=1, seed=11)
        original[0] = 1
    qlens = [d[0].shape[0] for d in parts]
    starts = [0, qlens[0], sum(qlens)]
    query = torch.cat([d[0].float() * 0.03125 for d in parts]).bfloat16()
    weights = torch.cat([d[2] for d in parts])
    positions = torch.cat([torch.arange(n - q, n) for n, q in zip(original, qlens)])
    metadata = SimpleNamespace(
        max_cache_seq_len=1025,
        max_query_len=max(qlens),
        query_start_loc=torch.tensor(starts, dtype=torch.int32, device="npu"),
        cache_seq_lens=torch.tensor(lengths, dtype=torch.int32, device="npu"),
        seq_lens=torch.tensor(original, dtype=torch.int32, device="npu"),
        block_table=table,
    )
    metadata.cmp_residual = (
        torch.tensor([value % ratio for value in original], dtype=torch.int32, device="npu") if ratio != 1 else None
    )
    metadata.qli_metadata = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
        HEADS,
        1,
        WIDTH,
        128,
        2,
        cu_seqlens_q=metadata.query_start_loc,
        seqused_k=metadata.cache_seq_lens,
        cmp_residual_k=metadata.cmp_residual,
        batch_size=2,
        max_seqlen_q=metadata.max_query_len,
        max_seqlen_k=metadata.max_cache_seq_len,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=ratio,
    )
    obj = _indexer(ratio)
    options = dict(candidate_topk_blocks=64, candidate_block_size=8)
    out, candidates = obj.select_projected(
        query.npu(),
        weights.npu(),
        positions.npu(),
        (key, scale),
        metadata,
        is_candidate_source=True,
        uses_candidate_filter=False,
        candidates=None,
        **options,
    )
    assert out.shape == (sum(qlens), 128)
    assert candidates.shape == (sum(qlens), 1, 64)
    for i, (start, end) in enumerate(zip(starts[:-1], starts[1:])):
        q, qs = _quant_query_reference(query[start:end])
        k = parts[i][1][: lengths[i]]
        ks = parts[i][4][: lengths[i]]
        score, visible = _scores(q, k, weights[start:end], qs, ks, ratio, original[i] % ratio)
        _check_topk(score, out[start:end], 128)
        if lengths[i]:
            _check_candidates(score, visible, candidates[start:end], 64)
        else:
            assert bool((candidates[start:end] == -1).all())
    # Same shared candidate tensor, different consumer projection.
    consumer_query = -query
    consumer_out, retained = obj.select_projected(
        consumer_query.npu(),
        weights.npu(),
        positions.npu(),
        (key, scale),
        metadata,
        is_candidate_source=False,
        uses_candidate_filter=True,
        candidates=candidates,
        **options,
    )
    assert retained is candidates
    for i, (start, end) in enumerate(zip(starts[:-1], starts[1:])):
        q, qs = _quant_query_reference(consumer_query[start:end])
        score, _ = _scores(
            q, parts[i][1][: lengths[i]], weights[start:end], qs, parts[i][4][: lengths[i]], ratio, original[i] % ratio
        )
        block_ids = candidates[start:end].cpu().reshape(end - start, -1)
        keep = (torch.arange(lengths[i])[None, None, :] // 8 == block_ids[:, :, None]).any(1)
        _mask_non_candidates(score, keep)
        _check_topk(score, consumer_out[start:end], 128)
        valid = consumer_out[start:end].cpu()
        valid = torch.where(valid < 0, torch.iinfo(torch.int32).max, valid)
        assert bool((valid[:, 1:] >= valid[:, :-1]).all())


def test_missing_candidate_rejected():
    obj = _indexer(1)
    with pytest.raises(RuntimeError, match="before its source"):
        obj.select_projected(
            None,
            None,
            None,
            None,
            None,
            is_candidate_source=False,
            uses_candidate_filter=True,
            candidate_topk_blocks=64,
            candidate_block_size=8,
            candidates=None,
        )


def test_empty_compressed_cache():
    obj = _indexer(2)
    q = torch.empty(1, 32, 128, device="npu", dtype=torch.bfloat16)
    out, candidates = obj.select_projected(
        q,
        None,
        None,
        None,
        SimpleNamespace(max_cache_seq_len=0),
        is_candidate_source=True,
        uses_candidate_filter=False,
        candidate_topk_blocks=64,
        candidate_block_size=8,
        candidates=None,
    )
    assert out.shape == (1, 0) and candidates.shape == (1, 1, 64)
    assert bool((candidates == -1).all())


@pytest.mark.parametrize("ratio", [1, 2])
def test_64_head_candidate_consumer(ratio):
    data = _data(1025, qlen=1, heads=64)
    score, visible = _scores(data[0], data[1], data[2], data[3], data[4], ratio, ratio - 1)
    output, candidates = _invoke(data, "TND", ratio, 1)
    _check_topk(score, output, 128)
    _check_candidates(score, visible, candidates, 64)
    block_ids = candidates.cpu().reshape(1, -1)
    keep = (torch.arange(1025)[None, None, :] // 8 == block_ids[:, :, None]).any(1)
    _mask_non_candidates(score, keep)
    output, _ = _invoke(data, "TND", ratio, 2, candidates=candidates)
    _check_topk(score, output, 128)


@pytest.mark.parametrize("topk", [512, 2048])
def test_position_topk_width(topk):
    data = _data(4097, qlen=1)
    score, visible = _scores(data[0], data[1], data[2], data[3], data[4], 2, 1)
    output, candidates = _invoke(data, "TND", 2, 1, topk=topk)
    _check_topk(score, output, topk)
    _check_candidates(score, visible, candidates, 64)
    block_ids = candidates.cpu().reshape(1, -1)
    keep = (torch.arange(4097)[None, None, :] // 8 == block_ids[:, :, None]).any(1)
    _mask_non_candidates(score, keep)
    output, _ = _invoke(data, "TND", 2, 2, candidates=candidates, topk=topk)
    _check_topk(score, output, topk)


@pytest.mark.skipif(is_950(), reason="Candidate fill semantics changed only on A2/A3")
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("length,qlen", [(91, 5), (4096, 1)])
def test_candidate_shortfall_keeps_reachable_indices(length, qlen, strided):
    topk = 2048
    data = _data(length, qlen=qlen, strided=strided)
    score, visible = _scores(data[0], data[1], data[2], data[3], data[4], 1, 0)
    block_ids = torch.arange(64, dtype=torch.int32).repeat(qlen, 1)
    block_ids[:, 9:11] = -1
    block_ids[block_ids >= (length + 7) // 8] = -1
    candidates = block_ids.unsqueeze(1).npu()
    membership = (torch.arange(length)[None, None, :] // 8 == block_ids[:, :, None]).any(1)
    _mask_non_candidates(score, membership)

    output, _ = _invoke(data, "TND", 1, 2, candidates=candidates, topk=topk)
    _check_topk(score, output, topk)
    # Missing candidate blocks must not turn reachable tokens into -1 slots.
    for row, indices in enumerate(output.cpu().reshape(qlen, topk)):
        valid = indices[indices >= 0].long()
        reachable = torch.arange(int(visible[row]))
        candidate_positions = reachable[membership[row, : reachable.numel()]]
        assert torch.isin(candidate_positions, valid).all()
        outside_count = (~membership[row, valid]).sum().item()
        assert outside_count == min(topk, reachable.numel()) - candidate_positions.numel()
        if reachable.numel() <= topk:
            assert torch.equal(valid.sort().values, reachable)
