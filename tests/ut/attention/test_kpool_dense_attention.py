# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.attention.kpool_dense_attention import execute_kpool_hybrid_attention, supports_dense_kpool_attention
from vllm_ascend.attention.kpool_query_plan import build_kpool_query_plan


def test_dense_prefix_dequantizes_pages_and_restores_ragged_rows():
    common = SimpleNamespace(
        num_reqs=3,
        num_actual_tokens=1553,
        query_start_loc_cpu=torch.tensor([0, 0, 1536, 1553]),
        seq_lens_cpu=torch.tensor([0, 2560, 65553]),
        is_prefilling=torch.tensor([True, True, False]),
    )
    plan = build_kpool_query_plan(common, 2048, torch.device("cpu"), plan_dense_attention=True)
    query = torch.randn(1553, 8, 512, dtype=torch.bfloat16)
    cache = torch.ones(16, 128, 1, 528, dtype=torch.int8)
    cache[..., :512] = torch.arange(1, 17, dtype=torch.int8)[:, None, None, None]
    scales = torch.arange(1, 5, dtype=torch.float32).expand(16, 128, 1, 4)
    cache[..., 512:] = scales.contiguous().view(torch.int8)
    before = cache.clone()
    table = torch.arange(15, -1, -1, dtype=torch.int32).expand(3, 16)
    metadata = SimpleNamespace(block_size=128, block_table=table, attn_mask=torch.ones(2048, 2048, dtype=torch.int8))
    key_lengths = torch.tensor([0, 2560, 65553], dtype=torch.int32)
    indices = torch.zeros(1553, 1, 2051, dtype=torch.int32)

    def dense(q, key, value, **kwargs):
        assert kwargs["actual_seq_lengths"] == [1024]
        assert kwargs["actual_seq_lengths_kv"] == [2048]
        assert kwargs["scale"] == 0.0625
        assert not kwargs["query_rope"].count_nonzero()
        assert not kwargs["key_rope"].count_nonzero()
        torch.testing.assert_close(key[0, 0, 0], (16 * torch.arange(1, 5)).repeat_interleave(128).bfloat16())
        assert key is value
        return (q + 2,)

    def sparse(q, qrope, kv, selected, meta, ends, lengths):
        assert meta is metadata and lengths is key_lengths
        assert ends.tolist() == [0, 512, 529]
        torch.testing.assert_close(q, query[1024:])
        assert selected.shape[0] == 529
        return q + 3

    with patch(
        "vllm_ascend.attention.kpool_dense_attention.torch_npu.npu_fused_infer_attention_score", side_effect=dense
    ):
        output = execute_kpool_hybrid_attention(query, cache, indices, metadata, key_lengths, plan, 0.0625, sparse)
    assert output is not None
    torch.testing.assert_close(output[:1024], query[:1024] + 2)
    torch.testing.assert_close(output[1024:], query[1024:] + 3)
    assert torch.equal(cache, before)


@pytest.mark.parametrize("reason", ["no_plan", "query_dtype", "cache_dtype", "padded_query", "kernel_block", "mask"])
def test_ineligible_attention_keeps_normal_sfa(reason):
    common = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=1024,
        query_start_loc_cpu=torch.tensor([0, 1024]),
        seq_lens_cpu=torch.tensor([2048]),
        is_prefilling=torch.tensor([True]),
    )
    plan = build_kpool_query_plan(common, 2048, torch.device("cpu"), plan_dense_attention=True)
    assert plan is not None and plan.dense is not None
    query = torch.empty(1025 if reason == "padded_query" else 1024, 8, 512, dtype=torch.bfloat16)
    cache = torch.empty(8, 128, 1, 528, dtype=torch.int8)
    if reason == "query_dtype":
        query = query.float()
    if reason == "cache_dtype":
        cache = cache.bfloat16()
    metadata = SimpleNamespace(
        block_size=32 if reason == "kernel_block" else 128,
        attn_mask=torch.ones(2, 2) if reason == "mask" else torch.ones(2048, 2048, dtype=torch.bool),
    )
    sparse = Mock(side_effect=AssertionError("fallback belongs to the caller"))
    with patch("vllm_ascend.attention.kpool_dense_attention.torch_npu.npu_fused_infer_attention_score") as dense:
        assert (
            execute_kpool_hybrid_attention(
                query,
                cache,
                torch.empty(0),
                metadata,
                torch.empty(0),
                None if reason == "no_plan" else plan,
                0.0625,
                sparse,
            )
            is None
        )
    dense.assert_not_called()
    sparse.assert_not_called()


def test_whole_full_visible_batch_needs_no_sparse_attention():
    common = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=2048,
        query_start_loc_cpu=torch.tensor([0, 2048]),
        seq_lens_cpu=torch.tensor([2048]),
        is_prefilling=torch.tensor([True]),
    )
    plan = build_kpool_query_plan(common, 2048, torch.device("cpu"), plan_dense_attention=True)
    query = torch.zeros(2048, 8, 512, dtype=torch.bfloat16)
    cache = torch.zeros(16, 128, 1, 528, dtype=torch.int8)
    metadata = SimpleNamespace(
        block_size=128,
        block_table=torch.arange(16, dtype=torch.int32).unsqueeze(0),
        attn_mask=torch.ones(2048, 2048, dtype=torch.bool),
    )
    sparse = Mock(side_effect=AssertionError("unexpected sparse attention"))
    with patch(
        "vllm_ascend.attention.kpool_dense_attention.torch_npu.npu_fused_infer_attention_score", return_value=(query + 2,)
    ):
        output = execute_kpool_hybrid_attention(
            query,
            cache,
            torch.empty(2048, 1, 2051, dtype=torch.int32),
            metadata,
            torch.tensor([2048]),
            plan,
            0.0625,
            sparse,
        )
    assert output is not None
    torch.testing.assert_close(output, query + 2)
    sparse.assert_not_called()


@pytest.mark.parametrize("variant", ["supported", "fp8", "pcp", "dcp", "tp1", "rope", "hardware"])
def test_dense_attention_admission_is_limited_to_validated_contract(variant):
    model = SimpleNamespace(num_attention_heads=64, kv_lora_rank=512, qk_rope_head_dim=0, index_topk=2048)
    parallel = SimpleNamespace(tensor_parallel_size=8, prefill_context_parallel_size=1, decode_context_parallel_size=1)
    config = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype="fp8" if variant == "fp8" else "int8"),
        model_config=SimpleNamespace(dtype=torch.bfloat16, hf_text_config=model),
        parallel_config=parallel,
    )
    if variant == "pcp":
        parallel.prefill_context_parallel_size = 8
    if variant == "dcp":
        parallel.decode_context_parallel_size = 2
    if variant == "tp1":
        parallel.tensor_parallel_size = 1
    if variant == "rope":
        model.qk_rope_head_dim = 64
    with patch("vllm_ascend.attention.kpool_dense_attention.get_current_hardware_profile") as profile:
        profile.return_value.supports.return_value = variant != "hardware"
        assert supports_dense_kpool_attention(config) is (variant == "supported")
