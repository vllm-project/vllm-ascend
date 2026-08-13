# SPDX-License-Identifier: Apache-2.0

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


def test_sfa_dcp_sparse_indices_respect_local_bounds() -> None:
    replicated_indices = torch.tensor(
        [
            [0, 2, 1, 3, 4, 6, -1, 5],
            [8, 10, 9, 11, 12, 14, -1, 13],
        ],
        dtype=torch.int32,
    )
    local_bounds = torch.tensor([[3], [6]], dtype=torch.int32)

    rank0 = _make_impl(0)._remap_sparse_indices(
        replicated_indices,
        max_local_indices=local_bounds,
    )
    rank1 = _make_impl(1)._remap_sparse_indices(
        replicated_indices,
        max_local_indices=local_bounds,
    )

    # Remapped indices >= the row bound become -1; valid entries stay compacted.
    torch.testing.assert_close(
        rank0,
        torch.tensor(
            [
                [0, 1, 2, -1, -1, -1, -1, -1],
                [4, 5, -1, -1, -1, -1, -1, -1],
            ],
            dtype=torch.int32,
        ),
    )
    torch.testing.assert_close(
        rank1,
        torch.tensor(
            [
                [0, 1, 2, -1, -1, -1, -1, -1],
                [4, 5, -1, -1, -1, -1, -1, -1],
            ],
            dtype=torch.int32,
        ),
    )


def test_sfa_dcp_sparse_indices_without_bounds_matches_baseline() -> None:
    replicated_indices = torch.tensor([[0, 2, 1, 3, 4, 6, -1, 5]], dtype=torch.int32)

    rank0 = _make_impl(0)._remap_sparse_indices(replicated_indices, max_local_indices=None)

    torch.testing.assert_close(
        rank0,
        torch.tensor([[0, 1, 2, 3, -1, -1, -1, -1]], dtype=torch.int32),
    )


def test_sfa_dcp_torch_merge_ignores_nan_lse_rows() -> None:
    output = torch.tensor(
        [
            [[[float("nan")]], [[float("nan")]]],
            [[[5.0]], [[7.0]]],
        ]
    )
    lse = torch.tensor(
        [
            [[float("nan")], [float("nan")]],
            [[0.0], [0.0]],
        ]
    )

    # NaN lse rows get zero weight and must not poison the merge.
    merged = AscendSFADCPImpl._merge_dcp_outputs_with_torch(output, lse, token_dim=2)
    torch.testing.assert_close(merged, torch.tensor([[[5.0], [7.0]]]))

    dsa_merged = AscendSFADCPImpl._merge_dcp_outputs_with_torch(output, lse, token_dim=1)
    torch.testing.assert_close(dsa_merged, torch.tensor([[[5.0]], [[7.0]]]))
