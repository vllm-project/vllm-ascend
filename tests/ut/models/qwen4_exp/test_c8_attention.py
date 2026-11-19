# SPDX-License-Identifier: Apache-2.0
import math
from types import SimpleNamespace

import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import override_forward_context

from tests.ut.models.qwen4_exp.test_gbsa_preparation import load_cpu_preparation
from vllm_ascend.models.qwen4_exp import c8_attention, lightning_indexer
from vllm_ascend.models.qwen4_exp.attention_dispatch import build_qsa_request_plan
from vllm_ascend.models.qwen4_exp.quantization import qsa_c8_indexer_cache_views, quantize_qsa_rows


def cpu_gbsa(query, keys, values, indices, counts, block_shape, **kwargs):
    assert query.dtype == keys.dtype == values.dtype == torch.int8
    assert kwargs["quant_mode"] == 7 and kwargs["attention_out_dtype"] == torch.bfloat16
    for key in ("q_dequant_scale", "k_dequant_scale", "v_dequant_scale", "p_quant_scale"):
        assert kwargs[key].dtype == torch.float32
    assert kwargs["p_quant_scale"].tolist() == [127.0]
    output = torch.zeros_like(query, dtype=kwargs["attention_out_dtype"])
    for row in range(query.shape[0]):
        for head in range(query.shape[1]):
            kv_head = head // (query.shape[1] // keys.shape[2])
            logical = [
                group * 4 + offset
                for group in indices[kv_head, row, : int(counts[kv_head, row])].tolist()
                for offset in range(4)
                if group * 4 + offset < int(kwargs["seqused_kv"][row])
            ]
            if logical:
                token = torch.tensor(logical)
                page = kwargs["block_table"][row, token // 128].long()
                k = (
                    keys[page, token % 128, kv_head].float()
                    * kwargs["k_dequant_scale"][page, kv_head, token % 128, None]
                )
                v = values[page, token % 128, kv_head].float() * kwargs["v_dequant_scale"][kv_head]
                q = query[row, head].float() * kwargs["q_dequant_scale"][row, head]
                scores = k @ q * kwargs["softmax_scale"]
                exponent = (scores - scores.max()).exp()
                output[row, head] = ((exponent * 127).round() / 127) @ v / exponent.sum()
    return output, torch.empty(0)


@pytest.mark.parametrize(
    "starts,lengths,prefilling,tokens,expected",
    [
        ([0, 512], [10240], [True], 512, 10240),
        ([0, 4096], [4096], [True], 4096, 4096),
        ([0, 511], [4096], [True], 511, None),
        ([0, 512], [10241], [True], 512, None),
        ([0, 512], [4096], [False], 512, None),
        ([0, 512, 516], [4096, 2048], [True, False], 516, None),
        ([0, 512], [4096], [True], 516, None),
    ],
)
def test_c8_prefill_grouped_schedule_bounds(starts, lengths, prefilling, tokens, expected):
    plan = build_qsa_request_plan(torch.tensor(starts), torch.tensor(lengths), tokens, torch.tensor(prefilling))
    assert c8_attention.c8_prefill_kv_length(plan, tokens) == expected
    assert c8_attention.c8_prefill_kv_length(None, tokens) is None


def test_c8_grouped_prefill_uses_request_bounds_and_preserves_sparse_rows(monkeypatch):
    monkeypatch.setattr(c8_attention, "prepare_qsa_gbsa_inputs", load_cpu_preparation())
    rows = 512
    query = torch.zeros(rows, 3, 128, dtype=torch.bfloat16)
    cache = torch.zeros(9, 128, 1, 128, dtype=torch.int8)
    seq_lens = torch.tensor([1023], dtype=torch.int32)
    positions = torch.arange(511, 1023, dtype=torch.int32)
    table = torch.tensor([[6, 2, 0, 3, 4, 1, 5, 7, 8]], dtype=torch.int32)
    compact = torch.tensor([[0, 32, 128, -1, -1, -1]], dtype=torch.int32).expand(rows, -1).clone()
    captured = []

    def metadata(indices, counts, *args, **kwargs):
        assert kwargs["quant_mode"] == 7 and kwargs["mask_mode"] == 1
        assert kwargs["max_seqlen_q"] == rows and kwargs["max_seqlen_kv"] == 1024
        assert kwargs["cu_seqlens_q"].tolist() == [0, rows]
        assert kwargs["seqused_kv"] is seq_lens
        captured.append((indices, counts))
        return torch.empty(0)

    def attention(q, k, v, indices, counts, *args, **kwargs):
        assert kwargs["block_table"].tolist() == [[6, 2, 0, 3, 4, 1, 5, 7]]
        assert kwargs["cu_seqlens_q"].tolist() == [0, rows] and kwargs["seqused_kv"] is seq_lens
        assert captured == [(indices, counts)]
        assert indices[0, 0, :2].tolist() == [0, 32] and counts[0, 0] == 2
        assert indices[0, -1].tolist() == [0, 32, 128, 255] and counts[0, -1] == 4
        return torch.ones_like(q, dtype=torch.bfloat16), torch.empty(0)

    monkeypatch.setattr(c8_attention, "generic_block_sparse_attention_metadata", metadata)
    monkeypatch.setattr(c8_attention, "generic_block_sparse_attention", attention)
    result = c8_attention.qsa_c8_attention(
        query,
        cache,
        cache,
        compact,
        table,
        torch.zeros(rows, dtype=torch.int32),
        positions,
        seq_lens,
        torch.empty_like(query),
        torch.ones(1, 128),
        torch.tensor([127.0]),
        key_scales=torch.ones(9, 1, 128),
        prefill_kv_length=1024,
    )
    torch.testing.assert_close(result, torch.ones_like(result))


@pytest.mark.parametrize("tail", [0, 1, 2, 3])
@pytest.mark.parametrize("row_count", [1, 11, 129])
def test_c8_sparse_attention_matches_dense_quantized_reference(monkeypatch, tail, row_count):
    torch.manual_seed(123 + tail)
    monkeypatch.setattr(c8_attention, "prepare_qsa_gbsa_inputs", load_cpu_preparation())
    calls = []

    def metadata(indices, counts, *args, **kwargs):
        assert kwargs["quant_mode"] == 7
        assert kwargs["max_seqlen_kv"] == 3 * 128
        calls.append(counts.shape[1])
        return torch.empty(0)

    monkeypatch.setattr(c8_attention, "generic_block_sparse_attention_metadata", metadata)
    monkeypatch.setattr(c8_attention, "generic_block_sparse_attention", cpu_gbsa)
    cache = torch.empty(6, 2, 128, 2, 128, dtype=torch.int8)
    key_cache, value_cache = cache[:, 0], cache[:, 1]
    v_scale = torch.rand(2, 128) * 0.01 + 0.01
    p_scale = torch.tensor([127.0])
    key_codes, scales = quantize_qsa_rows(torch.randn(6 * 128, 2, 128, dtype=torch.bfloat16))
    key_scales = scales.view(6, 128, 2).transpose(1, 2).contiguous()
    key_cache.copy_(key_codes.view_as(key_cache))
    value_cache.random_(-127, 128)
    query = torch.randn(row_count, 4, 128, dtype=torch.bfloat16)
    # Two requests, nonconsecutive physical pages, cross-page selected groups,
    # holes in top-k, and a partial final group.
    table = torch.tensor([[5, 1, 3], [2, 0, 4]], dtype=torch.int32)
    requests = torch.arange(row_count, dtype=torch.int32) % 2
    positions = torch.full((row_count,), 259 + tail, dtype=torch.int32)
    lengths = torch.tensor([260 + tail, 260 + tail], dtype=torch.int32)
    compact = torch.tensor([[32, -1, 0, 64, -1, -1, -1]], dtype=torch.int32).expand(row_count, -1).clone()
    # Last row may be an empty causal padding row.
    if row_count > 1:
        positions[-1] = -1
    output = torch.empty_like(query)
    c8_attention.qsa_c8_attention(
        query,
        key_cache,
        value_cache,
        compact,
        table,
        requests,
        positions,
        lengths,
        output,
        v_scale,
        p_scale,
        key_scales=key_scales,
    )
    qcodes, qscales = quantize_qsa_rows(query)
    qdq = qcodes.float() * qscales.unsqueeze(-1)
    expected = torch.zeros_like(query)
    for row in range(row_count):
        if positions[row] < 0:
            continue
        selected = list(range(128, 132)) + list(range(4)) + list(range(256, 260)) + list(range(260, 260 + tail))
        token = torch.tensor(selected)
        page = table[requests[row], token // 128].long()
        key_scale = key_scales.transpose(1, 2)[page, token % 128].unsqueeze(-1)
        keys = (key_cache[page, token % 128].float() * key_scale).repeat_interleave(2, 1)
        values = (value_cache[page, token % 128].float() * v_scale).repeat_interleave(2, 1)
        score = torch.einsum("hd,shd->hs", qdq[row], keys) / math.sqrt(128)
        exponent = (score - score.amax(-1, keepdim=True)).exp()
        expected[row] = torch.einsum("hs,shd->hd", (exponent * 127).round() / 127, values) / exponent.sum(
            -1, keepdim=True
        )
    torch.testing.assert_close(output, expected)
    assert calls == [row_count]


@pytest.mark.parametrize(
    "mode, reuse",
    [(None, False), (CUDAGraphMode.PIECEWISE, False), (CUDAGraphMode.NONE, True), (CUDAGraphMode.FULL, True)],
)
def test_quantized_indexer_metadata_lifetime(monkeypatch, mode, reuse):
    calls = []

    def build_metadata(**kwargs):
        calls.append(kwargs["seqused_k"].tolist())
        return torch.empty(0)

    monkeypatch.setattr(torch.ops._C_ascend, "npu_quant_lightning_indexer_v2_metadata", build_metadata, raising=False)
    table = torch.tensor([[7, 9]], dtype=torch.int32)
    positions = torch.tensor([3, 7])
    requests = torch.tensor([0, 0])
    lengths = torch.tensor([8])

    def prepare(page_table=table):
        return lightning_indexer._prepare_quantized_indexer_metadata(
            page_table, requests, positions, lengths, 512, 4, 256
        )

    def new_context():
        if mode is None:
            return override_forward_context(None)
        return override_forward_context(SimpleNamespace(cudagraph_runtime_mode=mode, additional_kwargs={}))

    with new_context():
        first = prepare()
        assert (prepare() is first) == reuse
        other_table = torch.tensor([[2, 3]], dtype=torch.int32)
        other = prepare(other_table)
        assert other is not first
        torch.testing.assert_close(other[2], other_table.expand(2, -1))
        assert (prepare(other_table) is other) == reuse
    positions.copy_(torch.tensor([0, 3]))
    table.copy_(torch.tensor([[4, 5]]))
    with new_context():
        second = prepare()
        assert (prepare() is second) == reuse
    assert first is not second
    torch.testing.assert_close(second[2], table.expand(2, -1))
    calls_per_input = 1 if reuse else 2
    assert calls == [[1, 2]] * 2 * calls_per_input + [[0, 1]] * calls_per_input


def test_quantized_indexer_native_abi_and_visibility(monkeypatch):
    query = torch.randn(3, 4, 128, dtype=torch.bfloat16)
    cache = torch.zeros(4, 32, 1, 132, dtype=torch.int8)
    key, scales = qsa_c8_indexer_cache_views(cache)
    key.fill_(2)
    scales.fill_(0.125)
    calls = []

    def metadata(**kw):
        assert kw["quant_mode"] == 2 and kw["mask_mode"] == 0 and kw["cmp_ratio"] == 1
        assert kw["cu_seqlens_q"].tolist() == [0, 1, 2, 3]
        assert kw["seqused_k"].tolist() == [0, 32, 33]
        calls.append(kw)
        return torch.empty(0)

    def operator(**kw):
        assert kw["query"].shape == (3, 64, 128)
        assert kw["query"].dtype == kw["key"].dtype == torch.int8
        for name in ("weights", "query_dequant_scale", "key_dequant_scale"):
            assert kw[name].dtype == torch.float16
        assert not kw["weights"][:, 4:].any()
        assert kw["key"].stride()[1:] == (128, 128, 1)
        assert kw["block_table"].tolist() == [[1, 3], [1, 3], [0, 2]]
        return torch.full((3, 1, 512), -1, dtype=torch.int32), torch.empty(0)

    monkeypatch.setattr(torch.ops._C_ascend, "npu_quant_lightning_indexer_v2_metadata", metadata, raising=False)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_quant_lightning_indexer_v2", operator, raising=False)
    packed = []
    monkeypatch.setattr(lightning_indexer, "pack_qsa_group_indices_npu", lambda *args: packed.append(args))
    out = lightning_indexer.qsa_select_paged_tokens_quantized(
        query,
        cache,
        torch.tensor([[1, 3], [0, 2]], dtype=torch.int32),
        torch.tensor([0, 0, 1]),
        torch.tensor([0, 127, 131]),
        torch.tensor([128, 132]),
        torch.tensor([0, 2, 3]),
        2048,
        4,
    )
    assert len(calls) == len(packed) == 1
    assert out.shape == (3, 2051) and (out == -1).all()
    assert packed[0][-1].shape == (3, 515)


def test_c8_full_topk_replaces_last_group_with_residual(monkeypatch):
    monkeypatch.setattr(c8_attention, "prepare_qsa_gbsa_inputs", load_cpu_preparation())
    seen = []

    def metadata(indices, counts, *args, **kw):
        assert indices.shape == (1, 1, 512)
        assert counts.tolist() == [[512]]
        assert kw["seqused_kv"].tolist() == [2051]
        assert kw["max_seqlen_kv"] == 17 * 128
        seen.append(True)
        return torch.empty(0)

    monkeypatch.setattr(c8_attention, "generic_block_sparse_attention_metadata", metadata)
    monkeypatch.setattr(c8_attention, "generic_block_sparse_attention", cpu_gbsa)
    key = torch.zeros(17, 128, 1, 128, dtype=torch.int8)
    value = torch.ones_like(key)
    # Excluded group zero has a large value, so an off-by-one is observable.
    value[0, :4].fill_(100)
    query = torch.zeros(1, 1, 128, dtype=torch.bfloat16)
    compact = torch.cat((torch.arange(511, -1, -1), torch.full((3,), -1))).int().view(1, -1)
    output = torch.empty_like(query)
    c8_attention.qsa_c8_attention(
        query,
        key,
        value,
        compact,
        torch.arange(17).int().view(1, -1),
        torch.tensor([0]),
        torch.tensor([2050]),
        torch.tensor([2051]),
        output,
        torch.ones(1, 128),
        torch.tensor([127.0]),
        key_scales=torch.ones(17, 1, 128),
    )
    torch.testing.assert_close(output, torch.ones_like(output))
    assert seen == [True]


def test_c8_empty_batch_does_not_invoke_native_attention(monkeypatch):
    monkeypatch.setattr(
        c8_attention, "generic_block_sparse_attention_metadata", lambda *a, **k: pytest.fail("native call")
    )
    cache = torch.zeros(1, 128, 1, 128, dtype=torch.int8)
    output = torch.empty(0, 2, 128, dtype=torch.bfloat16)
    assert (
        c8_attention.qsa_c8_attention(
            output,
            cache,
            cache,
            torch.empty(0, 515, dtype=torch.int32),
            torch.tensor([[0]]),
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
            torch.tensor([0]),
            output,
            torch.ones(1, 128),
            torch.tensor([127.0]),
            key_scales=torch.ones(1, 1, 128),
        )
        is output
    )


@pytest.mark.parametrize("bad_dtype,bad_width", [(torch.bfloat16, 132), (torch.int8, 131), (torch.int8, 4)])
def test_c8_invalid_storage_is_rejected(bad_dtype, bad_width):
    with pytest.raises(ValueError, match="QSA C8"):
        qsa_c8_indexer_cache_views(torch.empty(1, 128, 1, bad_width, dtype=bad_dtype))
