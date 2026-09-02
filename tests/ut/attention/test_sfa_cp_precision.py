# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADCPImpl


def _make_impl(rank: int, interleave_size: int = 2) -> AscendSFADCPImpl:
    impl = AscendSFADCPImpl.__new__(AscendSFADCPImpl)
    impl.dcp_size = 2
    impl.dcp_rank = rank
    impl._dcp_interleave_size = interleave_size
    impl._dcp_index_topk = 8
    impl._remap_order = torch.arange(8, dtype=torch.float32)
    impl._remap_invalid_index = torch.tensor(-1.0)
    return impl


def test_sfa_dcp_sparse_indices_are_compacted_per_owner_rank() -> None:
    replicated_indices = torch.tensor([[0, 2, 1, 3, 4, 6, -1, 5]], dtype=torch.int32)

    rank0 = _make_impl(0)._remap_sparse_indices(replicated_indices)
    rank1 = _make_impl(1)._remap_sparse_indices(replicated_indices)

    torch.testing.assert_close(
        rank0,
        torch.tensor([[0, 1, 2, 3, -1, -1, -1, -1]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        rank1,
        torch.tensor([[0, 1, 2, -1, -1, -1, -1, -1]], dtype=torch.int32),
    )


@patch("torch.ops.vllm.sfa_dcp_a2a_fused")
def test_sfa_dcp_routes_native_output_merge_to_custom_op(fused_a2a) -> None:
    impl = _make_impl(rank=1)
    impl.dcp_group = SimpleNamespace(unique_name="dcp:0")
    output = torch.empty(3, 4, 8)
    lse = torch.empty(3, 4, 1, dtype=torch.float32)
    expected = torch.empty(3, 2, 8)
    fused_a2a.return_value = expected

    actual = impl._merge_dcp_outputs(output, lse)

    assert actual is expected
    fused_a2a.assert_called_once_with(output, lse, 2, 1, "dcp:0")


@patch("torch.ops.vllm.sfa_dcp_a2a_fused")
def test_sfa_dsa_dcp_routes_token_scatter_to_custom_op(fused_a2a) -> None:
    impl = _make_impl(rank=1)
    impl.dcp_group = SimpleNamespace(unique_name="dcp:0")
    output = torch.empty(4, 2, 8)
    lse = torch.empty(4, 2, 1, dtype=torch.float32)
    expected = torch.empty(2, 2, 8)
    fused_a2a.return_value = expected
    dsa_cp_context = SimpleNamespace(
        num_tokens_pad=4,
        local_start=2,
        local_end_with_pad=4,
    )

    actual = impl._merge_dcp_outputs(output, lse, dsa_cp_context)

    assert actual is expected
    fused_a2a.assert_called_once_with(output, lse, 2, 0, "dcp:0")


@patch("torch.ops.vllm.sfa_dcp_a2a_fused_max_sum")
def test_sfa_dcp_routes_native_max_sum_to_pack_custom_op(fused_a2a) -> None:
    impl = _make_impl(rank=1)
    impl.dcp_group = SimpleNamespace(unique_name="dcp:0")
    output = torch.empty(3, 4, 8)
    softmax_max = torch.empty(1, 3, 4, dtype=torch.float32)
    softmax_sum = torch.empty(1, 3, 4, dtype=torch.float32)
    expected = torch.empty(3, 2, 8)
    fused_a2a.return_value = expected

    actual = impl._merge_dcp_outputs_max_sum(output, softmax_max, softmax_sum)

    assert actual is expected
    fused_a2a.assert_called_once_with(output, softmax_max, softmax_sum, 2, 1, "dcp:0")


@patch("torch.ops.vllm.sfa_dcp_a2a_fused_max_sum")
def test_sfa_dsa_dcp_routes_native_max_sum_with_token_scatter(fused_a2a) -> None:
    impl = _make_impl(rank=1)
    impl.dcp_group = SimpleNamespace(unique_name="dcp:0")
    output = torch.empty(4, 2, 8)
    softmax_max = torch.empty(1, 4, 2, dtype=torch.float32)
    softmax_sum = torch.empty(1, 4, 2, dtype=torch.float32)
    expected = torch.empty(2, 2, 8)
    fused_a2a.return_value = expected
    dsa_cp_context = SimpleNamespace(
        num_tokens_pad=4,
        local_start=2,
        local_end_with_pad=4,
    )

    actual = impl._merge_dcp_outputs_max_sum(output, softmax_max, softmax_sum, dsa_cp_context)

    assert actual is expected
    fused_a2a.assert_called_once_with(output, softmax_max, softmax_sum, 2, 0, "dcp:0")


def test_sfa_dcp_passes_native_max_sum_from_attention_to_pack() -> None:
    impl = _make_impl(rank=0)
    impl.dcp_group = SimpleNamespace(unique_name="dcp:0")
    impl._has_prefill = MagicMock(return_value=False)
    impl._finish_dcp_gather = MagicMock()
    impl._merge_dcp_outputs_max_sum = MagicMock()

    ql_nope = torch.randn(3, 4, 8)
    q_pe = torch.randn(3, 4, 2)
    impl._finish_dcp_gather.return_value = (ql_nope, q_pe)
    kv_cache = (torch.empty(1), torch.empty(1))
    topk_indices = torch.zeros(3, 2, dtype=torch.int32)
    impl._remap_sparse_indices = MagicMock(return_value=topk_indices)
    dcp_context = SimpleNamespace(
        gather_context=object(),
        seq_lens=torch.tensor([3], dtype=torch.int32),
        block_table=torch.zeros(1, 1, dtype=torch.int32),
    )
    attn_metadata = SimpleNamespace(dcp_context=dcp_context, dsa_cp_context=None)
    output = torch.randn(3, 4, 8)
    softmax_max = torch.randn(1, 3, 4, dtype=torch.float32)
    softmax_sum = torch.rand(1, 3, 4, dtype=torch.float32) + 0.25
    merged = torch.randn_like(output)
    impl._merge_dcp_outputs_max_sum.return_value = merged

    with patch(
        "vllm_ascend.attention.context_parallel.sfa_cp.DeviceOperator.execute_sparse_flash_attention_process",
        return_value=(output, softmax_max, softmax_sum),
    ) as execute_sfa:
        actual = impl._execute_sparse_flash_attention_process(
            ql_nope,
            q_pe,
            kv_cache,
            topk_indices,
            attn_metadata,
            torch.tensor([3], dtype=torch.int32),
            torch.tensor([3], dtype=torch.int32),
        )

    assert actual is merged
    assert execute_sfa.call_args.kwargs["return_lse"] is True
    impl._remap_sparse_indices.assert_called_once_with(topk_indices)
    impl._merge_dcp_outputs_max_sum.assert_called_once_with(output, softmax_max, softmax_sum, None)
