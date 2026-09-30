# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADCPImpl


@pytest.mark.parametrize("prefill", [False, True])
def test_dcp_decode_passes_compact_indices_with_existing_operator_api(prefill):
    impl = AscendSFADCPImpl.__new__(AscendSFADCPImpl)
    impl.dcp_group = Mock()
    q, rope = torch.randn(2, 4, 512), torch.randn(2, 4, 64)
    indices = torch.tensor([[[7, -1, 0, 2]], [[-1, -1, -1, -1]]], dtype=torch.int32)
    compact = torch.tensor([[[7, 0, 2, -1]], [[-1, -1, -1, -1]]], dtype=torch.int32)
    kv = (torch.randn(2, 128, 1, 512), torch.randn(2, 128, 1, 64))
    lengths = torch.tensor([128, 0], dtype=torch.int32)
    blocks = torch.tensor([[0], [1]], dtype=torch.int32)
    context = SimpleNamespace(
        gather_context=object(), kv_gather_block_table=blocks, seq_lens=lengths, block_table=blocks
    )
    metadata = SimpleNamespace(dcp_context=context)
    maximum, total = torch.zeros(4, 2, 1), torch.ones(4, 2, 1)
    total[:, 1] = 0
    op_result = q if prefill else (q, maximum, total)
    with (
        patch.object(impl, "_has_prefill", return_value=prefill),
        patch.object(impl, "_finish_dcp_gather", return_value=kv if prefill else (q, rope)),
        patch.object(impl, "_remap_sparse_indices", return_value=compact) as remap,
        patch.object(impl, "_merge_dcp_outputs", return_value=q) as merge,
        patch("vllm_ascend.attention.context_parallel.sfa_cp.enable_sfa_dcp_force_tmajor_restore", return_value=False),
        patch(
            "vllm_ascend.attention.context_parallel.sfa_cp.DeviceOperator.execute_sparse_flash_attention_process",
            return_value=op_result,
        ) as execute,
    ):
        actual = impl._execute_sparse_flash_attention_process(q, rope, kv, indices, metadata, lengths, lengths)
    assert actual is q
    kwargs = execute.call_args.kwargs
    assert kwargs["sparse_mode"] == (3 if prefill else 0)
    assert kwargs["return_lse"] is (not prefill)
    assert execute.call_args.args[7] is lengths
    if prefill:
        remap.assert_not_called()
        merge.assert_not_called()
    else:
        remap.assert_called_once_with(indices)
        assert execute.call_args.args[4] is compact
        lse = merge.call_args.args[1]
        assert torch.isneginf(lse[1]).all()


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("interleave", [1, 128])
def test_valid_count_torch_remap_has_stable_valid_prefix(rank, interleave):
    impl = AscendSFADCPImpl.__new__(AscendSFADCPImpl)
    impl.dcp_size, impl.dcp_rank = 4, rank
    impl._dcp_interleave_size = interleave
    impl._dcp_index_topk = 8
    impl._remap_order = torch.arange(8, dtype=torch.float32)
    impl._remap_invalid_index = torch.tensor(-1.0)
    indices = torch.tensor([[130, -1, 0, 511, 257, 1, 129, -1], [-1] * 8], dtype=torch.int32)
    with patch("vllm_ascend.attention.context_parallel.sfa_cp.HAS_TRITON", False):
        actual = impl._remap_sparse_indices(indices)
    for row, result in zip(indices.tolist(), actual.tolist()):
        owned = [x for x in row if x >= 0 and (x // interleave) % 4 == rank]
        expected = [(x // (4 * interleave)) * interleave + x % interleave for x in owned]
        assert result == expected + [-1] * (8 - len(expected))
