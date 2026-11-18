# SPDX-License-Identifier: Apache-2.0
import math

import pytest
import torch

from tests.ut.models.qwen4_exp.test_gbsa_preparation import load_cpu_preparation
from vllm_ascend.models.qwen4_exp import c8_attention, lightning_indexer
from vllm_ascend.models.qwen4_exp.quantization import qsa_c8_cache_views, quantize_qsa_rows


def cpu_pack(cache, groups, counts, visible, table, output, output_scales=None):
    """Independent logical-to-physical page reference for the device gather."""
    codes, scales = qsa_c8_cache_views(cache)
    output.zero_()
    if output_scales is not None:
        output_scales.zero_()
    for row in range(output.shape[0]):
        for head in range(output.shape[2]):
            for slot in range(int(counts[head, row])):
                logical_group = int(groups[head, row, slot])
                for offset in range(4):
                    logical = logical_group * 4 + offset
                    if logical >= int(visible[row]) or logical < 0:
                        continue
                    page = int(table[row, logical // 128])
                    if 0 <= page < cache.shape[0]:
                        packed_token = slot * 4 + offset
                        output[row, packed_token, head] = codes[page, logical % 128, head]
                        if output_scales is not None:
                            packed_page = row * (output.shape[1] // 128) + packed_token // 128
                            output_scales[packed_page, head, packed_token % 128] = scales[page, logical % 128, head]


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


@pytest.mark.parametrize("tail", [0, 1, 2, 3])
@pytest.mark.parametrize("row_count", [1, 11, 129])
@pytest.mark.parametrize("tile_rows", [8, 128])
def test_c8_sparse_attention_matches_dense_quantized_reference(monkeypatch, tail, row_count, tile_rows):
    torch.manual_seed(123 + tail)
    monkeypatch.setattr(c8_attention, "C8_QUERY_TILE", tile_rows)
    monkeypatch.setattr(c8_attention, "prepare_qsa_gbsa_inputs", load_cpu_preparation())
    monkeypatch.setattr(c8_attention, "pack_qsa_c8_rows", cpu_pack)
    calls = []

    def metadata(indices, counts, *args, **kwargs):
        assert kwargs["quant_mode"] == 7
        assert kwargs["max_seqlen_kv"] <= 2048
        calls.append(counts.shape[1])
        return torch.empty(0)

    monkeypatch.setattr(c8_attention, "generic_block_sparse_attention_metadata", metadata)
    monkeypatch.setattr(c8_attention, "generic_block_sparse_attention", cpu_gbsa)
    key_cache = torch.zeros(6, 128, 2, 132, dtype=torch.int8)
    value_cache = torch.zeros_like(key_cache)
    v_scale = torch.rand(2, 128) * 0.01 + 0.01
    p_scale = torch.tensor([127.0])
    for cache in (key_cache, value_cache):
        codes, scales = qsa_c8_cache_views(cache)
        q, s = quantize_qsa_rows(torch.randn(6 * 128, 2, 128, dtype=torch.bfloat16))
        codes.copy_(q.view_as(codes))
        scales.copy_(s.view_as(scales))
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
        query, key_cache, value_cache, compact, table, requests, positions, lengths, output, v_scale, p_scale
    )
    qcodes, qscales = quantize_qsa_rows(query)
    qdq = qcodes.float() * qscales.unsqueeze(-1)
    expected = torch.zeros_like(query)
    kc, ks = qsa_c8_cache_views(key_cache)
    vc, vs = qsa_c8_cache_views(value_cache)
    for row in range(row_count):
        if positions[row] < 0:
            continue
        selected = list(range(128, 132)) + list(range(4)) + list(range(256, 260)) + list(range(260, 260 + tail))
        token = torch.tensor(selected)
        page = table[requests[row], token // 128].long()
        keys = (kc[page, token % 128].float() * ks[page, token % 128].unsqueeze(-1)).repeat_interleave(2, 1)
        values = (vc[page, token % 128].float() * v_scale).repeat_interleave(2, 1)
        score = torch.einsum("hd,shd->hs", qdq[row], keys) / math.sqrt(128)
        exponent = (score - score.amax(-1, keepdim=True)).exp()
        expected[row] = torch.einsum("hs,shd->hd", (exponent * 127).round() / 127, values) / exponent.sum(
            -1, keepdim=True
        )
    torch.testing.assert_close(output, expected)
    assert sum(calls) == row_count
    assert max(calls) <= tile_rows
    assert len(calls) == math.ceil(row_count / tile_rows)


def test_quantized_indexer_native_abi_and_visibility(monkeypatch):
    query = torch.randn(3, 4, 128, dtype=torch.bfloat16)
    cache = torch.zeros(4, 32, 1, 132, dtype=torch.int8)
    key, scales = qsa_c8_cache_views(cache, indexer=True)
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
    monkeypatch.setattr(c8_attention, "pack_qsa_c8_rows", cpu_pack)
    seen = []

    def metadata(indices, counts, *args, **kw):
        assert indices.shape == (1, 1, 512)
        assert counts.tolist() == [[512]]
        assert kw["seqused_kv"].tolist() == [2047]
        assert kw["max_seqlen_kv"] == 2048
        seen.append(True)
        return torch.empty(0)

    monkeypatch.setattr(c8_attention, "generic_block_sparse_attention_metadata", metadata)
    monkeypatch.setattr(c8_attention, "generic_block_sparse_attention", cpu_gbsa)
    key = torch.zeros(17, 128, 1, 132, dtype=torch.int8)
    value = torch.zeros_like(key)
    codes, scales = qsa_c8_cache_views(value)
    codes.fill_(1)
    scales.fill_(1)
    # Excluded group zero has a large value, so an off-by-one is observable.
    codes[0, :4].fill_(100)
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
    )
    torch.testing.assert_close(output, torch.ones_like(output))
    assert seen == [True]


def test_c8_empty_batch_does_not_invoke_native_attention(monkeypatch):
    monkeypatch.setattr(
        c8_attention, "generic_block_sparse_attention_metadata", lambda *a, **k: pytest.fail("native call")
    )
    cache = torch.zeros(1, 128, 1, 132, dtype=torch.int8)
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
        )
        is output
    )


@pytest.mark.parametrize("bad_dtype,bad_width", [(torch.bfloat16, 132), (torch.int8, 131), (torch.int8, 4)])
def test_c8_invalid_storage_is_rejected(bad_dtype, bad_width):
    with pytest.raises(ValueError, match="QSA C8"):
        qsa_c8_cache_views(torch.empty(1, 128, 1, bad_width, dtype=bad_dtype))
