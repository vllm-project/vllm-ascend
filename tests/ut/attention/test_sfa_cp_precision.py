# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

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


def test_sfa_dcp_torch_merge_handles_invalid_lse() -> None:
    output = torch.tensor(
        [
            [[[1.0]], [[3.0]]],
            [[[5.0]], [[7.0]]],
        ]
    )
    lse = torch.tensor(
        [
            [[0.0], [float("-inf")]],
            [[0.0], [0.0]],
        ]
    )

    merged = AscendSFADCPImpl._merge_dcp_outputs_with_torch(output, lse, token_dim=2)

    torch.testing.assert_close(merged, torch.tensor([[[3.0], [7.0]]]))

    dsa_merged = AscendSFADCPImpl._merge_dcp_outputs_with_torch(output, lse, token_dim=1)
    torch.testing.assert_close(dsa_merged, torch.tensor([[[3.0]], [[7.0]]]))
