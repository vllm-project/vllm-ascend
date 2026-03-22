from types import SimpleNamespace

import torch
import pytest
from vllm.triton_utils import triton
from vllm.forward_context import override_forward_context

from tests.ut.base import PytestBase
from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule
from vllm_ascend.ops.triton.fla.prefill_precompute import (
    GDN_PREFILL_CHUNK_SIZE,
    SOLVE_TRIL_LARGE_BLOCK_T,
    VALIDATE_GDN_PREFILL_PRECOMPUTE_ENV,
    build_gdn_prefill_precomputed,
)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.ops.triton.fla.utils import prepare_chunk_indices, prepare_chunk_offsets


def _make_cu_seqlens(lengths: list[int]) -> torch.LongTensor:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return torch.tensor(offsets, dtype=torch.long).npu()


class TestChunkGatedDeltaRulePrefillPrecompute(PytestBase):
    @pytest.mark.parametrize(
        ("lengths", "num_heads"),
        [
            ([129, 63, 255], 8),
            ([64, 1, 130, 17], 16),
        ],
    )
    def test_prefill_precomputed_matches_prepare_helpers(
        self,
        lengths,
        num_heads,
        monkeypatch,
    ):
        init_device_properties_triton()
        cu_seqlens = _make_cu_seqlens(lengths)
        monkeypatch.setenv(VALIDATE_GDN_PREFILL_PRECOMPUTE_ENV, "1")

        precomputed = build_gdn_prefill_precomputed(cu_seqlens, num_heads)

        expected_cumsum_block_size = triton.next_power_of_2(
            (2**18) // (num_heads * GDN_PREFILL_CHUNK_SIZE)
        )
        assert torch.equal(
            precomputed.chunk_size_64_indices,
            prepare_chunk_indices(cu_seqlens, GDN_PREFILL_CHUNK_SIZE),
        )
        assert torch.equal(
            precomputed.chunk_size_64_offsets,
            prepare_chunk_offsets(cu_seqlens, GDN_PREFILL_CHUNK_SIZE),
        )
        assert precomputed.solve_tril_large_block_t == SOLVE_TRIL_LARGE_BLOCK_T
        assert torch.equal(
            precomputed.solve_tril_large_block_indices,
            prepare_chunk_indices(cu_seqlens, SOLVE_TRIL_LARGE_BLOCK_T),
        )
        assert precomputed.cumsum_optim_block_size == expected_cumsum_block_size
        assert torch.equal(
            precomputed.cumsum_block_indices,
            prepare_chunk_indices(cu_seqlens, expected_cumsum_block_size),
        )

    def test_chunk_gated_delta_rule_prefill_precomputed_matches_legacy_path(self, monkeypatch):
        torch.manual_seed(0)
        init_device_properties_triton()

        lengths = [65, 129, 31]
        total_tokens = sum(lengths)
        num_sequences = len(lengths)
        num_k_heads = 4
        num_v_heads = 8
        head_dim = 128

        cu_seqlens = _make_cu_seqlens(lengths)
        q = torch.randn(1, total_tokens, num_k_heads, head_dim, dtype=torch.bfloat16).npu()
        k = torch.randn(1, total_tokens, num_k_heads, head_dim, dtype=torch.bfloat16).npu()
        v = torch.randn(1, total_tokens, num_v_heads, head_dim, dtype=torch.bfloat16).npu()
        g = torch.nn.functional.logsigmoid(
            torch.randn(1, total_tokens, num_v_heads, dtype=torch.float32).npu()
        )
        beta = torch.rand(1, total_tokens, num_v_heads, dtype=torch.bfloat16).npu()
        initial_state = torch.randn(
            num_sequences,
            num_v_heads,
            head_dim,
            head_dim,
            dtype=torch.bfloat16,
        ).npu()

        monkeypatch.setattr(
            "vllm_ascend.ops.triton.fla.chunk.get_pcp_group",
            lambda: SimpleNamespace(world_size=1, rank_in_group=0),
        )
        forward_context = SimpleNamespace(
            attn_metadata=SimpleNamespace(
                num_prefills=num_sequences,
                num_decodes=0,
            )
        )
        with override_forward_context(forward_context):
            legacy_out, legacy_final_state = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=initial_state.clone(),
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )
        precomputed = build_gdn_prefill_precomputed(cu_seqlens, num_v_heads)
        with override_forward_context(forward_context):
            cached_out, cached_final_state = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=initial_state.clone(),
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
                prefill_precomputed=precomputed,
            )

        assert torch.equal(legacy_out, cached_out)
        assert torch.equal(legacy_final_state, cached_final_state)
