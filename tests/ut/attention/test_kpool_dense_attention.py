# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode

import vllm_ascend.attention.sfa_v1 as sparse_mla
from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFAPCPImpl
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl, PreprocessType

from vllm_ascend.attention.kpool_dense_attention import (
    can_execute_projected_kpool_attention,
    execute_kpool_hybrid_attention,
    execute_projected_kpool_attention,
    supports_dense_kpool_attention,
    supports_projected_kpool_attention,
)
from vllm_ascend.attention.kpool_query_plan import build_kpool_mha_plan, build_kpool_query_plan


def test_projected_mha_preserves_c8_pages_request_boundaries_and_causal_selection():
    queries, lengths = [1024, 1024], [1137, 1137]
    common = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=2048,
        query_start_loc_cpu=torch.tensor([0, 1024, 2048]),
        seq_lens_cpu_upper_bound=torch.tensor(lengths),
        is_prefilling=torch.ones(2, dtype=torch.bool),
    )
    plan = build_kpool_mha_plan(common, torch.device("cpu"))
    assert plan is not None
    cache = torch.zeros(18, 128, 1, 528, dtype=torch.int8)
    cache[..., :512] = torch.arange(1, 19, dtype=torch.int8)[:, None, None, None]
    scales = torch.ones(18, 128, 1, 4)
    cache[..., 512:] = scales.view(torch.int8)
    before = cache.clone()
    table = torch.arange(17, -1, -1, dtype=torch.int32).view(2, 9)
    metadata = SimpleNamespace(block_size=128, block_table=table)
    assert can_execute_projected_kpool_attention(cache, metadata, plan)
    assert not can_execute_projected_kpool_attention(cache.float(), metadata, plan)
    query = torch.randn(2048, 8, 256, dtype=torch.bfloat16)
    indices = torch.full((2048, 1, 2051), -1, dtype=torch.int32)
    indices[:, 0, 0] = 0
    indices[:, 0, 1] = plan.visible_key_lengths - 1
    indices[:, 0, 2] = plan.visible_key_lengths
    weight_k = torch.zeros(8, 256, 512, dtype=torch.bfloat16)
    weight_v = torch.zeros(8, 512, 256, dtype=torch.bfloat16)
    weight_k[:, :, :256] = torch.eye(256, dtype=torch.bfloat16)
    weight_v[:, 256:, :] = torch.eye(256, dtype=torch.bfloat16)
    expected_latent = torch.cat([cache[table[i].long(), ..., :512].flatten(0, 2)[:length] for i, length in enumerate(lengths)])

    def attention(q, key, value, heads, **kwargs):
        assert q is query and heads == 8
        assert kwargs["actual_seq_qlen"] == [1024, 2048]
        assert kwargs["actual_seq_kvlen"] == [1137, 2274]
        expected = expected_latent[:, None, :256].expand(-1, 8, -1).bfloat16()
        torch.testing.assert_close(key, expected)
        torch.testing.assert_close(value, expected)
        mask = kwargs["atten_mask"]
        assert mask.shape == (2, 1, 1024, 1137)
        expected_mask = torch.ones_like(mask)
        for request in range(2):
            expected_mask[request, 0, :, 0] = False
            rows = torch.arange(1024)
            expected_mask[request, 0, rows, plan.visible_key_lengths[request * 1024 : (request + 1) * 1024] - 1] = False
        assert torch.equal(mask, expected_mask)
        return (q + 1,)

    with patch("vllm_ascend.attention.kpool_dense_attention.torch_npu.npu_fusion_attention", side_effect=attention):
        actual = execute_projected_kpool_attention(query, cache, indices, metadata, plan, weight_k, weight_v, 0.0625)
    torch.testing.assert_close(actual, query + 1)
    assert torch.equal(cache, before)


@pytest.mark.parametrize("pcp,dcp,nope,v_dim,supported", [(1, 1, 256, 256, True), (8, 1, 256, 256, False), (1, 2, 256, 256, False), (1, 1, 192, 256, False), (1, 1, 256, 128, False)])
def test_projected_mha_gate_keeps_parallel_and_other_geometry_on_sfa(pcp, dcp, nope, v_dim, supported):
    config = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype="int8"),
        model_config=SimpleNamespace(dtype=torch.bfloat16, hf_text_config=SimpleNamespace(num_attention_heads=64, kv_lora_rank=512, qk_rope_head_dim=0, qk_nope_head_dim=nope, v_head_dim=v_dim, index_topk=2048, index_kpool=4)),
        parallel_config=SimpleNamespace(tensor_parallel_size=8, prefill_context_parallel_size=pcp, decode_context_parallel_size=dcp),
    )
    with patch("vllm_ascend.attention.kpool_dense_attention.get_current_hardware_profile", return_value=SimpleNamespace(supports=lambda _: True)):
        assert supports_projected_kpool_attention(config) == supported


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
        "vllm_ascend.attention.kpool_dense_attention.torch_npu.npu_fused_infer_attention_score",
        return_value=(query + 2,),
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


def test_mha_query_projection_does_not_absorb_key_weights():
    hidden, weight = torch.randn(3, 4), torch.randn(4, 16)
    projection = Mock(return_value=(hidden @ weight,))
    impl: Any = object.__new__(AscendSFAImpl)
    impl.q_proj = projection
    impl.local_num_heads, impl.qk_head_dim, impl.qk_nope_head_dim, impl.qk_rope_head_dim = 2, 8, 8, 0
    with patch.object(torch, "bmm", side_effect=AssertionError("MHA must not absorb Q")):
        query, rope = impl._q_proj_and_k_up_proj(hidden, absorb=False)
    torch.testing.assert_close(query, (hidden @ weight).view(3, 2, 8))
    assert rope.shape == (3, 2, 0) and projection.call_count == 1


@pytest.mark.parametrize("impl_cls,graph", [(AscendSFAImpl, False), (AscendSFAImpl, True), (AscendSFAPCPImpl, False)])
def test_projected_mha_dispatch_preserves_cache_and_indexer_work(impl_cls, graph):
    impl: Any = object.__new__(impl_cls)
    impl.qk_rope_head_dim = 0
    impl.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(prefill_context_parallel_size=1))
    impl.enable_sparse_sfa_c8 = impl.enable_sparse_sfa_turboquant = False
    impl.q_lora_rank = impl.kv_lora_rank = impl.v_head_dim = 2
    impl.local_num_heads = 1
    impl.preprocess_type = PreprocessType.NATIVE
    impl.g_proj = impl.layerwise_kv_cache_hook = None
    impl.has_indexer = True
    impl.skip_topk = impl.use_index_cache = False
    impl.layer_name = "layer"
    impl.scale = 0.0625
    impl._is_mtp_layer = False
    impl.W_UK_T = impl.W_UV = torch.ones(1, 2, 2)
    hidden = torch.ones(3, 2)
    metadata = SimpleNamespace(cos=None, sin=None, num_actual_tokens=3, num_input_tokens=3,
        slot_mapping=torch.arange(3), prefill_mha=object())
    impl._compose_sfa_kv_cache = MagicMock(return_value=(torch.empty(1),))
    impl._get_sfa_kv_slot_mapping = MagicMock(return_value=metadata.slot_mapping)
    impl._get_indexer_attn_metadata = MagicMock(return_value=object())
    impl._get_parallel_forward_context = MagicMock(return_value=SimpleNamespace(
        actual_seq_lengths_query=torch.tensor([3]), actual_seq_lengths_key=torch.tensor([3]),
        kv_slot_mapping=metadata.slot_mapping, topk_num_tokens=3))
    impl.fused_qkv_a_proj = MagicMock(return_value=(torch.zeros(3, 4),))
    impl.q_a_layernorm = lambda value: value
    impl.exec_kv = MagicMock(return_value=(None, torch.zeros(3, 2)))
    impl._store_parallel_kv = MagicMock(return_value=(None, None))
    impl._q_proj_and_k_up_proj = MagicMock(return_value=(torch.zeros(3, 1, 2), None))
    impl._record_query_gather_context = MagicMock()
    impl._prepare_indexer_metadata = MagicMock()
    impl.indexer = MagicMock(return_value=torch.zeros(3, 1, 1, dtype=torch.int32))
    impl._execute_sparse_flash_attention_process = MagicMock(return_value=torch.ones(3, 2))
    impl._v_up_proj = MagicMock(return_value=torch.full((3, 2), 9.0))
    impl._finalize_o_proj = MagicMock(side_effect=lambda value, output: output.copy_(value))
    with (
        patch.object(sparse_mla, "get_forward_context", return_value=SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL if graph else CUDAGraphMode.NONE)),
        patch.object(sparse_mla, "can_execute_projected_kpool_attention", return_value=True),
        patch.object(sparse_mla, "execute_projected_kpool_attention", return_value=torch.full((3, 1, 2), 5.0)) as mha,
        patch.object(sparse_mla, "wait_for_kv_layer_from_connector"),
        patch.object(sparse_mla, "notify_kv_cache_written"),
        patch.object(sparse_mla, "attention_transfer_window"),
        patch.object(sparse_mla, "maybe_save_kv_layer_to_connector"),
    ):
        output = torch.empty_like(hidden)
        forward: Any = AscendSFAImpl.forward
        assert forward(impl, "layer", hidden, (), metadata, output) is output
    routed = impl_cls is AscendSFAImpl and not graph
    impl.exec_kv.assert_called_once()
    impl._store_parallel_kv.assert_called_once()
    impl.indexer.assert_called_once()
    assert impl.indexer.call_args.kwargs["compute_topk"]
    assert mha.call_count == int(routed)
    assert impl._execute_sparse_flash_attention_process.call_count == int(not routed)
    assert impl._v_up_proj.call_count == int(not routed)
    assert impl._q_proj_and_k_up_proj.call_args.kwargs == ({"absorb": False} if routed else {})
    torch.testing.assert_close(output, torch.full_like(output, 5.0 if routed else 9.0))
