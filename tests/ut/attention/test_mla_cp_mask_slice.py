# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.attention.context_parallel.mla_cp import (
    AscendMLADCPDecodeMetadata,
    AscendMlaDCPMetadataBuilder,
)
from vllm_ascend.attention.mla_v1 import AscendMLAMetadataBuilder
from vllm_ascend.attention.utils import AscendDCPMetadata


@pytest.mark.parametrize("dcp_rank", [0, 1])
@patch.object(AscendMLAMetadataBuilder, "build_decode_metadata")
def test_mla_dcp_decode_metadata_slices_lengths_to_decode_batch(mock_build, dcp_rank: int) -> None:
    decode = AscendMLADCPDecodeMetadata(
        input_positions=torch.arange(4),
        block_table=torch.ones((1, 2), dtype=torch.int32),
        seq_lens=torch.tensor([20]),
        max_seq_lens=20,
        seq_lens_list=[20],
        actual_seq_lengths_q=[4],
    )
    mock_build.return_value = decode

    mtp_mask = torch.zeros((2, 8, 32), dtype=torch.bool)
    local_lengths = torch.tensor([[12, 8], [16, 12]], dtype=torch.int32)
    dcp_metadata = AscendDCPMetadata(
        num_computed_tokens_of_dcp=local_lengths.numpy(),
        query_lens_cpu=torch.tensor([4, 8], dtype=torch.int32),
        max_query_len=8,
        draft_cp_seq_len=local_lengths[:, dcp_rank],
        dcp_mtp_attn_mask=mtp_mask,
    )
    builder = AscendMlaDCPMetadataBuilder.__new__(AscendMlaDCPMetadataBuilder)
    builder.num_decodes = 1
    builder.dcp_size = 2
    builder.dcp_rank = dcp_rank
    builder.cp_local_block_size = 4
    builder.query_lens = dcp_metadata.query_lens_cpu

    result = builder.build_decode_metadata(
        common_prefix_len=0,
        common_attn_metadata=SimpleNamespace(context_parallel_metadata=dcp_metadata),
    )

    assert result.cp_seq_len.tolist() == [12 if dcp_rank == 0 else 8]
    # Only the decode request contributes: 20 total tokens - 4 current tokens.
    assert result.cp_history_seq_len == [8]
    assert result.actual_seq_lengths_q == [4]
    assert result.dcp_mtp_attn_mask is None
    assert dcp_metadata.dcp_mtp_attn_mask is mtp_mask
