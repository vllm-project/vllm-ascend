# SPDX-License-Identifier: Apache-2.0
"""Contracts for the optional Union adapter (native operators are mocked)."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.models.minimax_m3.ops import msa_m3_union as union


def inputs(tokens=9):
    q = torch.zeros((tokens, 16, 128), dtype=torch.bfloat16)
    k = torch.zeros((4, 128, 1, 128), dtype=torch.float8_e4m3fn)
    return dict(
        q=q,
        kv_cache=(k, k.clone()),
        topk_idx=torch.tensor([[[3, 1, -1]]], dtype=torch.int32).expand(1, tokens, 3),
        block_table=torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, tokens], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 512], dtype=torch.int32),
        seq_lens=torch.tensor([512], dtype=torch.int32),
        num_union_groups=(tokens + 7) // 8,
        max_kv_blocks=4,
        max_query_len=tokens,
        max_seq_len=512,
        sm_scale=128**-0.5,
        output=torch.empty_like(q),
    )


@pytest.mark.parametrize("tokens", [1, 7, 8, 9, 16])
@pytest.mark.parametrize("length_dtype", [torch.int32, torch.int64])
def test_union_preserves_exact_mask_and_static_capacity(monkeypatch, tokens, length_dtype):
    args = inputs(tokens)
    for field in ("cu_seqlens_q", "cu_seqlens_kv", "seq_lens"):
        args[field] = args[field].to(length_dtype)
    args["q"][0, 0, 0] = 1000
    idx, count, mask = (torch.empty(0, dtype=torch.int32) for _ in range(3))
    gen = Mock(return_value=(idx, count, mask))
    metadata = Mock(return_value=torch.empty(1024, dtype=torch.int32))
    expected = torch.ones_like(args["q"])
    attn = Mock(return_value=(expected, None))
    monkeypatch.setattr(
        union,
        "load_union_extensions",
        lambda: (
            SimpleNamespace(union_sparse_gen_launch=gen),
            SimpleNamespace(generic_block_sparse_attention_metadata=metadata, generic_block_sparse_attention=attn),
        ),
    )
    original = args["topk_idx"].clone()
    union.minimax_m3_sparse_attn_union(**args)
    torch.testing.assert_close(args["output"], expected)
    torch.testing.assert_close(args["topk_idx"], original)
    torch.testing.assert_close(gen.call_args.args[0], original.sort(dim=-1).values.roll(-1, dims=-1).contiguous())
    assert gen.call_args.args[-2:] == ((tokens + 7) // 8, 4)
    for position in (2, 3, 5):
        assert gen.call_args.args[position].dtype == torch.int32
    assert attn.call_args.args[0].dtype == torch.float8_e4m3fn
    assert float(attn.call_args.args[0][0, 0, 0]) == 448
    assert attn.call_args.args[3] is idx
    assert metadata.call_args.args[0] is idx
    assert attn.call_args.kwargs["attn_mask"] is mask
    assert attn.call_args.kwargs["metadata"] is metadata.return_value
    for op in (attn, metadata):
        assert op.call_args.kwargs["mask_mode"] == 3
        assert op.call_args.kwargs["softmax_precision"] == 0
        assert op.call_args.kwargs["quant_mode"] == 5
        assert op.call_args.kwargs["cu_seqlens_q"].dtype == torch.int64
        assert op.call_args.kwargs["seqused_kv"].dtype == torch.int32


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("q", torch.zeros((9, 16, 128)), "BF16"),
        ("output", torch.zeros((9, 16, 128)), "BF16"),
        ("q", torch.zeros((9, 16, 64), dtype=torch.bfloat16), "head size"),
        ("num_union_groups", -1, "nonnegative"),
        ("num_union_groups", 0, "positive"),
        ("max_kv_blocks", 0, "positive"),
    ],
)
def test_union_rejects_unsupported_inputs(field, value, error):
    args = inputs()
    args[field] = value
    with pytest.raises(ValueError, match=error):
        union.minimax_m3_sparse_attn_union(**args)


def test_union_empty_input_does_not_load_extensions(monkeypatch):
    loader = Mock(side_effect=AssertionError("must not load"))
    monkeypatch.setattr(union, "load_union_extensions", loader)
    union.minimax_m3_sparse_attn_union(**inputs(0))
    loader.assert_not_called()


def test_missing_extension_has_actionable_error(monkeypatch):
    union.load_union_extensions.cache_clear()
    monkeypatch.setattr(union, "import_module", Mock(side_effect=ImportError("missing")))
    with pytest.raises(RuntimeError, match="matching UnionSparseGen and GBSA exactMask"):
        union.load_union_extensions()
    union.load_union_extensions.cache_clear()


def test_union_respects_producer_count_with_stale_positive_tail(monkeypatch):
    args = inputs(1)
    args["topk_idx"] = torch.tensor([[[2, 0, 99]]], dtype=torch.int32)
    args["select_num_idx"] = torch.tensor([[2]], dtype=torch.int32)
    gen = Mock(return_value=(torch.empty(0), torch.empty(0), torch.empty(0)))
    monkeypatch.setattr(
        union,
        "load_union_extensions",
        lambda: (
            SimpleNamespace(union_sparse_gen_launch=gen),
            SimpleNamespace(
                generic_block_sparse_attention_metadata=Mock(),
                generic_block_sparse_attention=Mock(return_value=(torch.zeros_like(args["q"]), None)),
            ),
        ),
    )
    union.minimax_m3_sparse_attn_union(**args)
    torch.testing.assert_close(gen.call_args.args[0], torch.tensor([[[0, 2, -1]]], dtype=torch.int32))
    torch.testing.assert_close(gen.call_args.args[1], args["select_num_idx"])
