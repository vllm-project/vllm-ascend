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


@pytest.mark.parametrize("dcp_rank", [0, 1])
@patch.object(AscendMLAMetadataBuilder, "build_decode_metadata")
def test_mla_dcp_decode_metadata_slices_lengths_to_decode_batch(mock_build, dcp_rank: int) -> None:
    mtp_mask = torch.zeros((2, 8, 32), dtype=torch.bool)
    decode = AscendMLADCPDecodeMetadata(
        dcp_mtp_attn_mask=mtp_mask,
        input_positions=torch.arange(4),
        block_table=torch.ones((1, 2), dtype=torch.int32),
        seq_lens=torch.tensor([20]),
        max_seq_lens=20,
        seq_lens_list=[20],
        actual_seq_lengths_q=[4],
    )
    mock_build.return_value = decode

    local_lengths = torch.tensor([[12, 8], [16, 12]], dtype=torch.int32)
    builder = AscendMlaDCPMetadataBuilder.__new__(AscendMlaDCPMetadataBuilder)
    builder.num_decodes = 1
    builder.dcp_size = 2
    builder.dcp_rank = dcp_rank
    builder.cp_local_block_size = 4
    builder.query_lens = torch.tensor([4, 8], dtype=torch.int32)
    builder.seq_lens = torch.tensor([20, 28], dtype=torch.int32)

    result = builder.build_decode_metadata(
        common_prefix_len=0,
        common_attn_metadata=SimpleNamespace(
            dcp_local_seq_lens_cpu=local_lengths[:, dcp_rank],
        ),
    )

    assert result.cp_seq_len == [12 if dcp_rank == 0 else 8]
    # Only the decode request contributes: 20 total tokens - 4 current tokens.
    assert result.cp_history_seq_len == [8]
    assert result.actual_seq_lengths_q == [4]
    assert result.dcp_mtp_attn_mask is None
    assert not mtp_mask.any()
    torch.testing.assert_close(local_lengths, torch.tensor([[12, 8], [16, 12]], dtype=torch.int32))
