# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.attention.context_parallel.mla_cp import (
    AscendMLADCPDecodeMetadata,
    AscendMlaDCPMetadataBuilder,
)
from vllm_ascend.attention.mla_v1 import AscendMLAMetadataBuilder


@patch.object(AscendMLAMetadataBuilder, "build_decode_metadata")
def test_mla_dcp_decode_metadata_slices_mtp_mask_to_decode_batch(mock_build) -> None:
    decode = AscendMLADCPDecodeMetadata(
        input_positions=torch.arange(4),
        block_table=torch.ones((1, 2), dtype=torch.int32),
        seq_lens=torch.tensor([20]),
        max_seq_lens=20,
        seq_lens_list=[20],
    )
    mock_build.return_value = decode

    mtp_mask = torch.zeros((2, 8, 32), dtype=torch.bool)
    dcp_metadata = SimpleNamespace(
        draft_cp_seq_len=torch.tensor([10, 11], dtype=torch.int32),
        dcp_mtp_attn_mask=mtp_mask,
    )
    builder = AscendMlaDCPMetadataBuilder.__new__(AscendMlaDCPMetadataBuilder)
    builder.num_decodes = 1
    builder._require_dcp_metadata = lambda _metadata: dcp_metadata

    result = builder.build_decode_metadata(
        common_prefix_len=0,
        common_attn_metadata=SimpleNamespace(),
    )

    assert result.cp_seq_len.tolist() == [10]
    assert result.dcp_mtp_attn_mask.shape == (1, 8, 32)
    assert result.dcp_mtp_attn_mask.data_ptr() == mtp_mask.data_ptr()
