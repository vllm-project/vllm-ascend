# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

with patch.dict(sys.modules, {"flash_attn_npu_3": MagicMock()}):
    from vllm_ascend.attention import flash_attention_v3 as fa3

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.device.hardware_profile import AttentionBackendFamily
from vllm_ascend.platform import NPUPlatform


@pytest.fixture
def builder():
    result = object.__new__(fa3.AscendFlashAttentionMetadataBuilder)
    result.device = torch.device("cpu")
    result.model_runner_type = "generate"
    result.max_num_reqs = 5
    result.graph_buffers = {}
    result.scheduler_buffers = {}
    result.scheduler_specs = set()
    result.capture_sizes = {2, 6}
    result.block_size = 128
    return result


def common_metadata(query_lens, seq_lens, num_actual_tokens=None):
    offsets = torch.tensor([0, *query_lens], dtype=torch.int32).cumsum(0, dtype=torch.int32)
    return SimpleNamespace(
        num_reqs=len(query_lens),
        num_input_tokens=0,
        num_actual_tokens=sum(query_lens) if num_actual_tokens is None else num_actual_tokens,
        query_start_loc=offsets,
        query_start_loc_cpu=offsets,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        block_table_tensor=torch.arange(len(seq_lens) * 3, dtype=torch.int32).view(len(seq_lens), 3),
        slot_mapping=torch.arange(sum(query_lens)),
        max_query_len=max(query_lens),
        attn_state=AscendAttentionState.ChunkedPrefill,
        causal=True,
    )


def test_metadata_uses_device_lengths_and_preserves_mixed_offsets(builder):
    common = common_metadata([3, 1, 7], [130, 259, 7])
    common.seq_lens_cpu = torch.tensor([999, 999, 999])
    with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("device synchronization")):
        metadata = builder.build(128, common)
    assert metadata.query_start_loc.tolist() == [0, 3, 4, 11, 11, 11]
    assert metadata.seq_lens.tolist() == [130, 259, 7, 0, 0]
    assert metadata.block_tables[:3].equal(common.block_table_tensor)
    assert metadata.causal


def test_graph_buffers_are_stable_and_clear_removed_requests(builder):
    common = common_metadata([3, 1, 7], [130, 259, 7])
    first = builder.build(0, common)
    addresses = [x.data_ptr() for x in (first.query_start_loc, first.seq_lens, first.block_tables)]
    common.num_reqs = 1
    common.num_actual_tokens = 2
    common.query_start_loc[1] = 2
    common.seq_lens[0] = 131
    second = builder.build(0, common)
    assert addresses == [x.data_ptr() for x in (second.query_start_loc, second.seq_lens, second.block_tables)]
    assert second.query_start_loc.tolist() == [0, 2, 2, 2, 2, 2]
    assert second.seq_lens.tolist() == [131, 0, 0, 0, 0]
    assert torch.count_nonzero(second.block_tables[1:]) == 0
    assert first.scheduler_metadata is not second.scheduler_metadata


def test_padding_request_does_not_read_beyond_kv_metadata(builder):
    common = common_metadata([1, 1, 1, 1, 4], [10, 20, 30, 40], num_actual_tokens=4)
    metadata = builder.build(0, common)
    assert metadata.query_start_loc.tolist() == [0, 1, 2, 3, 4, 4]
    assert metadata.seq_lens.tolist() == [10, 20, 30, 40, 0]
    assert metadata.slot_mapping.numel() == 4


def test_draft_step_buffers_are_independent(builder):
    first = builder.build(0, common_metadata([2, 3], [20, 30]))
    second = builder.build(0, common_metadata([1, 1], [21, 31]))
    assert first.seq_lens.data_ptr() != second.seq_lens.data_ptr()
    assert first.query_start_loc.tolist() == [0, 2, 5, 5, 5, 5]


@pytest.fixture
def impl():
    result = object.__new__(fa3.AscendFlashAttentionImpl)
    result.num_heads = 4
    result.num_kv_heads = 2
    result.head_size = 8
    result.scale = 0.123
    result.logits_soft_cap = 0.0
    result.key_cache = torch.zeros(9, 128, 2, 8)
    result.value_cache = torch.zeros_like(result.key_cache)
    return result


@pytest.mark.parametrize("state", [AscendAttentionState.ChunkedPrefill, AscendAttentionState.SpecDecoding])
def test_paged_call_keeps_causal_multi_token_queries(builder, impl, state):
    metadata = builder.build(0, common_metadata([3, 1, 7], [130, 259, 7]))
    metadata.attn_state = state
    tiling_tensor = torch.empty(1)
    metadata.scheduler_metadata[(4, 2, 8, torch.float32, 0.123, 0.0)] = tiling_tensor
    query = torch.randn(11, 4, 8)
    output = torch.empty_like(query)
    with (
        patch.object(fa3, "get_scheduler_metadata", return_value=torch.empty(1)) as tiling,
        patch.object(fa3, "flash_attn_with_kvcache", return_value=query) as kernel,
        patch.object(fa3, "record_attention_compute_start"),
    ):
        impl.forward_impl(query, None, None, (), metadata, output)
        impl.forward_impl(query, None, None, (), metadata, output)
    assert output.equal(query)
    tiling.assert_not_called()
    assert kernel.call_args.kwargs["causal"] is True
    assert kernel.call_args.kwargs["softmax_scale"] == impl.scale
    assert kernel.call_args.kwargs["cu_seqlens_q"] is metadata.query_start_loc
    assert kernel.call_args.kwargs["page_table"] is metadata.block_tables
    assert kernel.call_args.kwargs["scheduler_metadata"] is tiling_tensor


@pytest.mark.parametrize(
    "state,capturing",
    [
        (AscendAttentionState.PrefillNoCache, False),
        (AscendAttentionState.PrefillNoCache, True),
        (AscendAttentionState.PrefillCacheHit, False),
        (AscendAttentionState.ChunkedPrefill, False),
        (AscendAttentionState.DecodeOnly, False),
        (AscendAttentionState.DecodeOnly, True),
        (AscendAttentionState.SpecDecoding, False),
        (AscendAttentionState.SpecDecoding, True),
    ],
)
def test_full_forward_never_falls_back_to_fia(builder, impl, state, capturing):
    query_lens = [1, 1] if state == AscendAttentionState.DecodeOnly else [2, 3]
    metadata = builder.build(0, common_metadata(query_lens, [10, 20]))
    metadata.attn_state = state
    metadata.scheduler_metadata[(4, 2, 8, torch.float32, 0.123, 0.0)] = torch.empty(1)
    query = torch.randn(sum(query_lens), 4, 8)
    output = torch.empty_like(query)
    expected = torch.full_like(query, 7)
    impl.pcp_enabled = False
    impl.attn_type = "decoder"
    impl._use_layer_aware_fia_graph_replay = False
    impl.use_bnsd_kv_cache = False
    impl.kv_sharing_target_layer_name = None
    layer = SimpleNamespace(layer_name="model.layers.0.self_attn.attn", _k_scale_float=1.0, _v_scale_float=1.0)
    with (
        patch.object(fa3, "flash_attn_with_kvcache", return_value=expected) as paged,
        patch.object(fa3, "record_attention_compute_start"),
        patch("vllm_ascend.attention.attention_v1.DeviceOperator.reshape_and_cache") as cache_write,
        patch("vllm_ascend.attention.attention_v1.notify_kv_cache_written"),
        patch.object(fa3.AscendAttentionBackendImpl, "forward_impl", side_effect=AssertionError("FIA fallback")),
        patch.object(
            fa3.AscendAttentionBackendImpl, "forward_fused_infer_attention", side_effect=AssertionError("FIA")
        ),
        patch.object(fa3.AscendAttentionBackendImpl, "forward_paged_attention", side_effect=AssertionError("PA")),
    ):
        result = impl.forward(
            layer, query, query[:, :2], query[:, :2], (impl.key_cache, impl.value_cache), metadata, output
        )
    assert result is output
    assert output.equal(expected)
    cache_write.assert_called_once()
    paged.assert_called_once()
    assert paged.call_args.kwargs["num_splits"] == 0


def test_tiling_prepared_before_capture_and_refreshed_in_place(builder, impl):
    spec = (4, 2, 8, torch.float32, 0.123, 0.0)
    builder.scheduler_specs = {spec}
    common = common_metadata([1, 1], [10, 20])
    query = torch.randn(2, 4, 8)
    output = torch.empty_like(query)
    with (
        patch.object(fa3, "get_scheduler_metadata", side_effect=[torch.tensor([1]), torch.tensor([2])]) as tiling,
        patch.object(fa3, "flash_attn_with_kvcache", return_value=query),
        patch.object(fa3, "record_attention_compute_start"),
    ):
        metadata = builder.build(0, common)
        address = metadata.scheduler_metadata[spec].data_ptr()
        impl.forward_impl(query, None, None, (), metadata, output)
        assert tiling.call_count == 1
        common.seq_lens.add_(1)
        updated = builder.build(0, common)
    assert tiling.call_count == 2
    assert updated.scheduler_metadata[spec].data_ptr() == address
    assert metadata.scheduler_metadata[spec].tolist() == [2]


@pytest.mark.parametrize("c8,cache_dtype", [(True, "auto"), (False, "int8"), (False, "fp8")])
def test_quantized_cache_rejected(c8, cache_dtype):
    def initialize(self, *args, **kwargs):
        self.enable_c8_quant = c8
        self.kv_cache_dtype = cache_dtype

    with (
        patch.object(fa3.AscendAttentionBackendImpl, "__init__", initialize),
        pytest.raises(ValueError, match="C8"),
    ):
        fa3.AscendFlashAttentionImpl(4, 128, 0.1, 2, None, None, cache_dtype, None, "decoder", None)


@pytest.mark.parametrize(
    "architecture,installed,c8,cache_dtype,pcp,dcp,compatibility,rl,expected",
    [
        ("Qwen3ForCausalLM", True, False, "auto", 1, 1, False, False, "fa3"),
        ("Qwen3MoeForCausalLM", True, False, "bfloat16", 1, 1, False, False, "fa3"),
        ("Qwen3ForCausalLM", False, False, "auto", 1, 1, False, False, "fia"),
        ("Qwen3MoeForCausalLM", False, False, "auto", 1, 1, False, False, "fia"),
        ("LlamaForCausalLM", True, False, "auto", 1, 1, False, False, "fia"),
        ("Qwen3ForCausalLM", True, True, "auto", 1, 1, False, False, "fia"),
        ("Qwen3ForCausalLM", True, False, "int8", 1, 1, False, False, "fia"),
        ("Qwen3ForCausalLM", True, False, "auto", 2, 1, False, False, "fia"),
        ("Qwen3ForCausalLM", True, False, "auto", 1, 2, False, False, "fia"),
        ("Qwen3ForCausalLM", True, False, "auto", 1, 1, True, False, "310p"),
        ("Qwen3ForCausalLM", True, False, "auto", 1, 1, False, True, "rl"),
    ],
)
def test_platform_selects_fa3_by_model(architecture, installed, c8, cache_dtype, pcp, dcp, compatibility, rl, expected):
    selector = SimpleNamespace(use_mla=False, use_sparse=False, use_compress=False, use_pcp=pcp > 1)
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=[architecture])),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp, decode_context_parallel_size=dcp),
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
        quant_config=SimpleNamespace(enable_c8_quant=c8),
    )
    family = AttentionBackendFamily.COMPATIBILITY if compatibility else AttentionBackendFamily.STANDARD
    backends = {
        "fa3": "vllm_ascend.attention.flash_attention_v3.AscendFlashAttentionBackend",
        "fia": "vllm_ascend.attention.attention_v1.AscendAttentionBackend",
        "310p": "vllm_ascend._310p.attention.attention_v1.AscendAttentionBackend310",
        "rl": "vllm_ascend.attention.fa3_v1.AscendFABackend",
    }
    with (
        patch("vllm.config.get_current_vllm_config_or_none", return_value=config),
        patch("vllm_ascend.platform._validate_fa3_backend", return_value=rl),
        patch(
            "vllm_ascend.platform.get_current_hardware_profile",
            return_value=SimpleNamespace(attention_backend_family=family),
        ),
        patch("vllm_ascend.platform.util.find_spec", return_value=object() if installed else None),
    ):
        assert NPUPlatform.get_attn_backend_cls(None, selector) == backends[expected]


def test_paged_tiling_shared_but_attention_computed_for_each_layer(builder, impl):
    spec = (4, 2, 8, torch.float32, 0.123, 0.0)
    builder.scheduler_specs = {spec}
    common = common_metadata([3, 7], [3, 7])
    common.attn_state = AscendAttentionState.PrefillNoCache
    shared_tiling = torch.tensor([1], dtype=torch.uint8)
    query = torch.randn(10, 4, 8)
    output = torch.empty_like(query)
    with (
        patch.object(fa3, "get_scheduler_metadata", return_value=shared_tiling) as tiling,
        patch.object(fa3, "flash_attn_with_kvcache", side_effect=[query, query + 1]) as kernel,
        patch.object(fa3, "record_attention_compute_start"),
    ):
        metadata = builder.build(0, common)
        impl.forward_impl(query, query, query, (), metadata, output)
        assert output.equal(query)
        impl.forward_impl(query + 1, query + 1, query + 1, (), metadata, output)
        assert output.equal(query + 1)
    tiling.assert_called_once()
    assert tiling.call_args.kwargs["page_size"] == 128
    assert tiling.call_args.kwargs["num_splits"] == 0
    assert tiling.call_args.kwargs["max_seqlen_k"] == 384
    assert tiling.call_args.kwargs["cache_seqlens"].tolist() == [3, 7, 0, 0, 0]
    assert kernel.call_count == 2
    assert all(call.kwargs["scheduler_metadata"] is shared_tiling for call in kernel.call_args_list)
    assert metadata.scheduler_metadata[spec] is shared_tiling
    assert all(call.kwargs["num_splits"] == 0 for call in kernel.call_args_list)


@pytest.mark.parametrize(
    "query_lens,seq_lens,actual_tokens,expected_batch",
    [
        ([1], [4097], 1, 1),
        ([1, 0, 0], [4097, 0, 0], 1, 1),
        ([1, 1, 2], [4097, 4097], 2, 2),
        ([3, 1, 0], [130, 4097, 0], 4, 2),
        ([0], [0], 0, 1),
    ],
)
def test_metadata_excludes_padding_from_active_batch(builder, query_lens, seq_lens, actual_tokens, expected_batch):
    builder.max_num_reqs = 17
    builder.capture_sizes = {1, 2, 4}
    builder.scheduler_specs = {(4, 2, 8, torch.float32, 0.123, 0.0)}
    common = common_metadata(query_lens, seq_lens, num_actual_tokens=actual_tokens)
    with (
        patch.object(fa3, "get_scheduler_metadata", return_value=torch.empty(1)) as tiling,
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("device synchronization")),
    ):
        metadata = builder.build(0, common)
    assert tiling.call_args.kwargs["batch_size"] == expected_batch
    assert tiling.call_args.kwargs["num_splits"] == 0
    assert metadata.seq_lens.shape == (17,)
    assert metadata.query_start_loc.shape == (18,)
