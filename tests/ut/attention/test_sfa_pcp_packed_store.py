# SPDX-License-Identifier: Apache-2.0
"""PCP gathered packed KV must retain its matching compact cache addresses."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode

import vllm_ascend.attention.sfa_v1 as sparse_mla
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.context_parallel.sfa_cp import (
    AscendSFADCPMetadata,
    AscendSFAPCPDCPImpl,
    AscendSFAPCPImpl,
)
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl, PreprocessType


@pytest.mark.parametrize("impl_cls", [AscendSFAPCPImpl, AscendSFAPCPDCPImpl])
@pytest.mark.parametrize("num_decode_tokens", [0, 2, 5])
@pytest.mark.parametrize("cache_mode,rope_dim", [("c8", 0), ("c8", 2), ("tq4", 2)])
def test_native_pcp_packed_kv_preserves_gathered_payload_addresses(impl_cls, num_decode_tokens, cache_mode, rope_dim):
    impl = impl_cls.__new__(impl_cls)
    impl.is_pcp_decode_sharded = False
    impl.qk_rope_head_dim = rope_dim
    impl.kv_lora_rank = 3
    impl.num_kv_heads = 1
    impl.enable_sparse_sfa_c8 = cache_mode == "c8"
    impl.enable_sparse_sfa_turboquant = cache_mode == "tq4"
    impl.turboquant = Mock()
    impl.c8_cache_dtype = torch.int8
    impl.sfa_qsfa_tile_size = 32
    impl.sfa_qsfa_packed_kv_head_dim = 4 + rope_dim
    impl.kv_a_layernorm = SimpleNamespace(weight=torch.ones(3), variance_epsilon=1e-6)
    impl._record_dcp_kv_gather_context = Mock()
    local = torch.arange(1, 5 * (3 + rope_dim) + 1, dtype=torch.float32).reshape(5, 3 + rope_dim)
    peer_prefill = local[num_decode_tokens:] + 10
    expected_payload = torch.cat((local, peer_prefill)).to(torch.int8)
    expected_payload = torch.cat((expected_payload, torch.ones(len(expected_payload), 1, dtype=torch.int8)), dim=-1)
    expanded_slots = torch.tensor(
        list(range(5)) + [-1] * num_decode_tokens + list(range(5, len(expected_payload))), dtype=torch.int32
    )
    metadata = AscendSFADCPMetadata.__new__(AscendSFADCPMetadata)
    metadata.num_decode_tokens = num_decode_tokens
    metadata.num_prefills = int(num_decode_tokens < len(local))
    metadata.pcp_has_global_prefill = True
    metadata.uses_pcp_sharded_layout = True
    metadata.num_input_tokens = metadata.num_actual_tokens = len(local)
    metadata.slot_mapping = expanded_slots
    metadata.pcp_slot_mapping = expanded_slots
    metadata.cos = metadata.sin = torch.empty(len(local), rope_dim) if rope_dim else None
    metadata.attn_state = AscendAttentionState.ChunkedPrefill
    metadata.dcp_context = SimpleNamespace(slot_mapping=expanded_slots)
    group = SimpleNamespace(
        world_size=2,
        rank_in_group=0,
        all_gather=Mock(side_effect=lambda values, dim: torch.cat((values, values + 10), dim)),
    )
    cache = torch.full((1, 12, 1, 4 + rope_dim), -99, dtype=torch.int8)
    hidden = torch.zeros(len(local), 2)
    impl.q_lora_rank = 2
    impl.preprocess_type = PreprocessType.NATIVE
    impl.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(prefill_context_parallel_size=2))
    impl.g_proj = impl.layerwise_kv_cache_hook = None
    impl.has_indexer = True
    impl._is_mtp_layer = impl.skip_topk = impl.use_index_cache = False
    impl.layer_name = "layer"
    impl._compose_sfa_kv_cache = Mock(return_value=(cache,))
    impl._get_indexer_attn_metadata = Mock(return_value=object())
    impl._get_parallel_forward_context = Mock(
        return_value=SimpleNamespace(
            actual_seq_lengths_query=[len(local)],
            actual_seq_lengths_key=[len(local)],
            kv_slot_mapping=expanded_slots,
            topk_num_tokens=len(local),
            gather_full_o_proj=False,
        )
    )
    impl.fused_qkv_a_proj = Mock(return_value=(torch.cat((hidden, local), dim=-1),))
    impl.q_a_layernorm = lambda values: values
    impl._q_proj_and_k_up_proj = Mock(return_value=(hidden.unsqueeze(1), hidden.unsqueeze(1)))
    impl.rope_single = lambda values, *_args: values
    impl._record_query_gather_context = Mock()
    impl._prepare_indexer_metadata = Mock()
    impl.indexer = Mock(return_value=torch.zeros(len(local), 1, 1, dtype=torch.int32))
    impl._execute_sparse_flash_attention_process = Mock(return_value=hidden)
    impl._v_up_proj = lambda values: values
    impl._finalize_o_proj = lambda values, output, gather_full_o_proj: output.copy_(values)

    def quantize(values, *_args, **_kwargs):
        rows = len(values)
        return (
            values[..., 3:].to(torch.int8),
            values[..., :3].to(torch.int8),
            torch.ones(rows, 1, 1, 1, dtype=torch.int8),
        )

    def scatter(target, indices, values):
        indices = indices.flatten().long()
        valid = indices >= 0
        target.index_copy_(0, indices[valid], values[valid])

    # Run the real forward entry, PCP gather and cache store. Distributed and
    # device kernels are mocked; unrelated Q/indexer/attention paths use fixtures.
    with (
        patch("vllm.v1.attention.ops.pcp.get_pcp_group", return_value=group),
        patch("vllm_ascend.attention.sfa_v1.custom_kv_rmsnorm_rope", side_effect=quantize),
        patch("vllm_ascend.attention.sfa_v1.turboquant_kv_rmsnorm_rope", side_effect=quantize),
        patch("vllm_ascend.attention.sfa_v1.DeviceOperator.scatter_cache", side_effect=scatter),
        patch.object(
            sparse_mla, "get_forward_context", return_value=SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE)
        ),
        patch.object(sparse_mla, "wait_for_kv_layer_from_connector"),
        patch.object(sparse_mla, "notify_kv_cache_written"),
        patch.object(sparse_mla, "attention_transfer_window", side_effect=nullcontext),
        patch.object(sparse_mla, "maybe_save_kv_layer_to_connector"),
    ):
        AscendSFAImpl.forward(impl, "layer", hidden, (cache,), metadata, torch.empty_like(hidden))

    expected = torch.full_like(cache.view(-1, 4 + rope_dim), -99)
    expected[: len(expected_payload)] = expected_payload
    torch.testing.assert_close(cache.view(-1, 4 + rope_dim), expected, rtol=0, atol=0)
