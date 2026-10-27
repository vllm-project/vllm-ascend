# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the DeepSeek V4.1 prefill indexer TP row shard.

These mirror the contracts of vllm#54951 and the V4 sibling vllm-ascend#18799
but exercise the V4.1 ratio-1/2 plan + rebased metadata + padded all-gather
on CPU only: the QLI operator itself is mocked at the gather boundary.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.attention.dsa_v41_indexer_shard import (
    MIN_DSV41_PREFILL_INDEXER_TP_SHARD_ROWS_PER_RANK,
    balanced_prefill_row_shard,
    build_prefill_row_shard_tensors,
    dsv41_prefill_indexer_tp_shard_supported,
    gather_prefill_row_shard_rows,
    pack_prefill_row_shard_rows,
    plan_dsv41_prefill_row_shard,
)


# ---------------------------------------------------------------------------
# balanced_prefill_row_shard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_balanced_split_covers_every_row_and_balances_cost(tp_size):
    min_rows = MIN_DSV41_PREFILL_INDEXER_TP_SHARD_ROWS_PER_RANK
    # A long single request: the causal ramp tail-loads the last rank under
    # a naive row-count split; cost-balancing must stay within a few percent.
    n = 8 * min_rows
    seq = torch.tensor([n], dtype=torch.int32)
    qlen = torch.tensor([n], dtype=torch.int32)
    sizes = balanced_prefill_row_shard(seq, qlen, compress_ratio=2, tp_size=tp_size, min_rows=min_rows)
    assert sizes is not None and len(sizes) == tp_size
    assert sum(sizes) == n
    assert min(sizes) >= 1
    # Reproduce the per-row cost the splitter used and confirm no rank exceeds
    # 1.05x the average cost (the V4 PR used a 1.02 bound on heavier workloads).
    first_key = torch.repeat_interleave(seq - qlen + 1, qlen)
    row = torch.arange(n) - torch.repeat_interleave(torch.cumsum(qlen, 0) - qlen, qlen)
    cost = torch.cumsum((first_key + row) // 2, 0)
    total = int(cost[-1].item())
    avg = total / tp_size
    bounds = [0, *torch.cumsum(torch.tensor(sizes), 0).tolist()[:-1], n]
    rank_costs = [int(cost[bounds[i + 1] - 1].item() - (cost[bounds[i] - 1].item() if bounds[i] else 0)) for i in range(tp_size)]
    assert max(rank_costs) <= 1.05 * avg


def test_balanced_split_declines_small_batch():
    min_rows = MIN_DSV41_PREFILL_INDEXER_TP_SHARD_ROWS_PER_RANK
    seq = torch.tensor([min_rows - 1], dtype=torch.int32)
    qlen = torch.tensor([min_rows - 1], dtype=torch.int32)
    assert balanced_prefill_row_shard(seq, qlen, 2, tp_size=4, min_rows=min_rows) is None
    assert balanced_prefill_row_shard(seq, qlen, 2, tp_size=1, min_rows=min_rows) is None


def test_balanced_split_uneven_requests():
    # Mirrors the V4 PR's mixed-depth workload.
    min_rows = MIN_DSV41_PREFILL_INDEXER_TP_SHARD_ROWS_PER_RANK
    qlen = torch.tensor([2 * min_rows, 9007, 4 * min_rows, min_rows // 3 + 5], dtype=torch.int32)
    seq = torch.tensor([2 * min_rows, 9007, 4 * min_rows + 500, min_rows // 3 + 60005], dtype=torch.int32)
    sizes = balanced_prefill_row_shard(seq, qlen, 4, tp_size=4, min_rows=min_rows)
    assert sizes is not None
    assert sum(sizes) == int(qlen.sum().item())
    assert min(sizes) >= 1


# ---------------------------------------------------------------------------
# plan_dsv41_prefill_row_shard (rebase + cross-request windows)
# ---------------------------------------------------------------------------


def _cpu_meta(num_reqs, qlens, slens, prefill_flags):
    qsl = torch.zeros(num_reqs + 1, dtype=torch.int32)
    qsl[1:] = torch.tensor(qlens, dtype=torch.int32).cumsum(0)
    is_pf = torch.tensor(prefill_flags, dtype=torch.bool)
    return qsl, torch.tensor(slens, dtype=torch.int32), is_pf


def test_plan_rebases_cross_request_metadata_pure_prefill():
    # Two requests (3, 7 rows), ratio 2, tp2. Cost-balanced split is (6, 4);
    # rank1 owns the last 4 rows, all inside request 1, so request 0 is
    # inactive and only the rebased request 1 survives.
    qsl, seq, is_pf = _cpu_meta(2, [3, 7], [4, 10], [True, True])
    plan = plan_dsv41_prefill_row_shard(
        query_start_loc_cpu=qsl,
        seq_lens_cpu=seq,
        is_prefilling=is_pf,
        num_actual_tokens=10,
        compress_ratio=2,
        tp_size=2,
        tp_rank=1,
        min_rows=1,
    )
    assert plan is not None
    assert plan.sizes == (6, 4)
    assert plan.start == 6 and plan.stop == 10
    # Rebased seq = seq - qlen + local_stop = 10 - 7 + 7 = 10 (right end kept).
    assert plan.local_query_lens == (4,)
    assert plan.local_seq_lens == (10,)
    # Request indices skip the inactive request 0.
    assert plan.request_indices == (1,)


def test_plan_preserves_decode_rows_in_mixed_batch():
    # 2 decode reqs (3 tokens) leading, then 1 prefill req of 10 tokens, tp2.
    qsl, seq, is_pf = _cpu_meta(3, [2, 1, 10], [5, 6, 14], [False, False, True])
    plan = plan_dsv41_prefill_row_shard(
        query_start_loc_cpu=qsl,
        seq_lens_cpu=seq,
        is_prefilling=is_pf,
        num_actual_tokens=13,
        compress_ratio=2,
        tp_size=2,
        tp_rank=1,
        min_rows=1,
    )
    assert plan is not None
    assert plan.num_decode_tokens == 3
    # Decode rows stay outside the prefill shard window.
    assert plan.start == 9 and plan.stop == 13
    # Local query lens keep both decode requests then the rebased prefill.
    assert plan.local_query_lens[0:2] == (2, 1)
    assert plan.local_query_lens == (2, 1, 4)
    assert plan.local_seq_lens == (5, 6, 14)
    # Decode request indices lead the block-table selection.
    assert plan.request_indices[0:2] == (0, 1)
    assert plan.request_indices == (0, 1, 2)


def test_plan_declines_non_leading_decode_layout():
    # Decode request placed after the prefill -> not vLLM's prefix layout.
    qsl, seq, is_pf = _cpu_meta(2, [4, 4], [4, 8], [True, False])
    plan = plan_dsv41_prefill_row_shard(
        query_start_loc_cpu=qsl,
        seq_lens_cpu=seq,
        is_prefilling=is_pf,
        num_actual_tokens=8,
        compress_ratio=2,
        tp_size=2,
        tp_rank=0,
        min_rows=1,
    )
    assert plan is None


def test_plan_declines_missing_cpu_lengths():
    plan = plan_dsv41_prefill_row_shard(
        query_start_loc_cpu=torch.tensor([0, 4, 8], dtype=torch.int32),
        seq_lens_cpu=torch.zeros(0, dtype=torch.int32),
        is_prefilling=torch.tensor([True, True], dtype=torch.bool),
        num_actual_tokens=8,
        compress_ratio=2,
        tp_size=2,
        tp_rank=0,
    )
    assert plan is None


# ---------------------------------------------------------------------------
# build_prefill_row_shard_tensors (ratio 1 vs 2 residual contract)
# ---------------------------------------------------------------------------


def test_shard_tensors_ratio1_has_no_residual():
    # Single 8-row request, ratio 1, tp2: cost-balanced split is (5, 3).
    qsl, seq, is_pf = _cpu_meta(1, [8], [8], [True])
    plan = plan_dsv41_prefill_row_shard(
        query_start_loc_cpu=qsl, seq_lens_cpu=seq, is_prefilling=is_pf,
        num_actual_tokens=8, compress_ratio=1, tp_size=2, tp_rank=0, min_rows=1,
    )
    assert plan is not None
    block_table = torch.zeros((1, 4), dtype=torch.int32)
    shard = build_prefill_row_shard_tensors(plan, block_table, compress_ratio=1)
    assert shard.qli_cmp_residual_k is None
    # Rebased seq = 8 - 8 + stop(5) = 5; ratio 1 keeps it unchanged.
    assert torch.equal(shard.qli_seqused_k, torch.tensor([5], dtype=torch.int32))
    assert torch.equal(shard.qli_cu_seqlens_q, torch.tensor([0, 5], dtype=torch.int32))


def test_shard_tensors_ratio2_derives_compressed_lens():
    qsl, seq, is_pf = _cpu_meta(1, [8], [8], [True])
    plan = plan_dsv41_prefill_row_shard(
        query_start_loc_cpu=qsl, seq_lens_cpu=seq, is_prefilling=is_pf,
        num_actual_tokens=8, compress_ratio=2, tp_size=2, tp_rank=0, min_rows=1,
    )
    assert plan is not None
    block_table = torch.zeros((1, 4), dtype=torch.int32)
    shard = build_prefill_row_shard_tensors(plan, block_table, compress_ratio=2)
    # Rebased seq = 5 -> seqused_k = 2, residual = 1.
    assert torch.equal(shard.qli_seqused_k, torch.tensor([2], dtype=torch.int32))
    assert torch.equal(shard.qli_cmp_residual_k, torch.tensor([1], dtype=torch.int32))
    assert torch.equal(shard.qli_cu_seqlens_q, torch.tensor([0, 5], dtype=torch.int32))
    # The block table index_select keeps the active request rows.
    assert shard.block_table.shape[0] == 1


# ---------------------------------------------------------------------------
# gather_prefill_row_shard_rows (padded all-gather reassembly)
# ---------------------------------------------------------------------------


class _FakeTPGroup:
    def __init__(self, rank_outputs):
        # rank_outputs[r] is the tensor rank r contributes.
        self._rank_outputs = rank_outputs
        self.calls = 0

    def all_gather(self, padded, dim=0):
        self.calls += 1
        assert dim == 0
        return torch.cat(self._rank_outputs, dim=0)


def test_gather_pads_trims_and_prepends_decode_rows():
    # rank0 owns 3 rows, rank1 owns 1 row (padded to 3 with -1), decode = 2.
    leading = torch.tensor([[100], [101]], dtype=torch.int32)
    sizes = (3, 1)
    r0 = torch.arange(3, dtype=torch.int32).view(-1, 1)
    r1 = torch.tensor([[10]], dtype=torch.int32)
    fake = _FakeTPGroup([r0, r1])
    with patch("vllm.distributed.get_tp_group", lambda: fake):
        out = gather_prefill_row_shard_rows(r0, sizes, leading=leading)
    assert fake.calls == 1
    # Decode prefix stays, then rank0 rows then rank1 row, padding trimmed.
    assert torch.equal(out, torch.tensor([[100], [101], [0], [1], [2], [10]], dtype=torch.int32))


def test_gather_pure_prefill_no_leading():
    r0 = torch.tensor([[0], [1]], dtype=torch.int32)
    r1 = torch.tensor([[2]], dtype=torch.int32)
    fake = _FakeTPGroup([r0, r1])
    with patch("vllm.distributed.get_tp_group", lambda: fake):
        out = gather_prefill_row_shard_rows(r0, (2, 1), leading=None)
    assert torch.equal(out, torch.tensor([[0], [1], [2]], dtype=torch.int32))


def test_gather_rejects_local_rows_beyond_capacity():
    with pytest.raises(ValueError):
        gather_prefill_row_shard_rows(torch.zeros((4, 1), dtype=torch.int32), sizes=(2, 2), leading=None)


# ---------------------------------------------------------------------------
# pack_prefill_row_shard_rows
# ---------------------------------------------------------------------------


def test_pack_pure_prefill_is_zero_copy_slice():
    shard = SimpleNamespace(num_decode_tokens=0, start=2, stop=5)
    t = torch.arange(10)
    packed = pack_prefill_row_shard_rows(t, shard)
    assert packed.data_ptr() == t[2:5].data_ptr()


def test_pack_mixed_batch_cats_decode_prefix():
    shard = SimpleNamespace(num_decode_tokens=2, start=5, stop=8)
    t = torch.arange(10)
    packed = pack_prefill_row_shard_rows(t, shard)
    assert torch.equal(packed, torch.tensor([0, 1, 5, 6, 7]))


def test_pack_none_passthrough():
    assert pack_prefill_row_shard_rows(None, SimpleNamespace(num_decode_tokens=0, start=0, stop=0)) is None


# ---------------------------------------------------------------------------
# dsv41_prefill_indexer_tp_shard_supported (gate)
# ---------------------------------------------------------------------------


def _cfg(tp=4, pcp=1, dcp=1, cudagraph="PIECEWISE", model_type="deepseek_v41_text", dsa_cp=False):
    text = SimpleNamespace(model_type=model_type)
    model = SimpleNamespace(hf_text_config=text)
    parallel = SimpleNamespace(
        tensor_parallel_size=tp, prefill_context_parallel_size=pcp, decode_context_parallel_size=dcp
    )
    compilation = SimpleNamespace(cudagraph_mode=getattr(__import__("vllm.config", fromlist=["CUDAGraphMode"]).CUDAGraphMode, cudagraph))
    cfg = SimpleNamespace(model_config=model, parallel_config=parallel, compilation_config=compilation)
    return cfg, dsa_cp


def test_gate_enables_with_defaults(monkeypatch):
    cfg, dsa_cp = _cfg()
    monkeypatch.setenv("VLLM_ASCEND_DSV41_PREFILL_INDEXER_TP_SHARD", "1")
    with patch("vllm_ascend.ascend_config.get_ascend_config", lambda: SimpleNamespace(enable_dsa_cp=dsa_cp)):
        assert dsv41_prefill_indexer_tp_shard_supported(cfg)


@pytest.mark.parametrize("kw", [
    {"tp": 1},
    {"pcp": 2},
    {"dcp": 2},
    {"cudagraph": "FULL"},
    {"model_type": "deepseek_v4"},
])
def test_gate_rejects_unsupported(kw, monkeypatch):
    cfg, dsa_cp = _cfg(**kw)
    monkeypatch.setenv("VLLM_ASCEND_DSV41_PREFILL_INDEXER_TP_SHARD", "1")
    with patch("vllm_ascend.ascend_config.get_ascend_config", lambda: SimpleNamespace(enable_dsa_cp=dsa_cp)):
        assert not dsv41_prefill_indexer_tp_shard_supported(cfg)


def test_gate_defaults_off(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_DSV41_PREFILL_INDEXER_TP_SHARD", raising=False)
    cfg, dsa_cp = _cfg()
    with patch("vllm_ascend.ascend_config.get_ascend_config", lambda: SimpleNamespace(enable_dsa_cp=dsa_cp)):
        assert not dsv41_prefill_indexer_tp_shard_supported(cfg)
