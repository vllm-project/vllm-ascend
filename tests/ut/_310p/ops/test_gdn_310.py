#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID

from vllm_ascend._310p.ops.fla.gdn_310 import (
    AscendGatedDeltaNetAttention310,
    _mask_padded_recurrent_accepted_tokens,
    _zero_padded_tokens,
)
from vllm_ascend._310p.ops.gdn_attn_builder_310 import (
    AscendGDNAttentionBackend310,
    AscendGDNAttentionMetadataBuilder310,
)
from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionMetadataBuilder


def test_ascend_gdn_attention_310_uses_310p_backend():
    assert AscendGatedDeltaNetAttention310.get_attn_backend(object()) is AscendGDNAttentionBackend310
    assert AscendGDNAttentionBackend310.get_builder_cls() is AscendGDNAttentionMetadataBuilder310


def test_builder310_reuses_common_graph_materialization():
    assert AscendGDNAttentionMetadataBuilder310.build is AscendGDNAttentionMetadataBuilder.build
    assert (
        AscendGDNAttentionMetadataBuilder310._pad_spec_decode_metadata
        is AscendGDNAttentionMetadataBuilder._pad_spec_decode_metadata
    )
    assert (
        AscendGDNAttentionMetadataBuilder310._pad_decode_metadata
        is AscendGDNAttentionMetadataBuilder._pad_decode_metadata
    )


def test_zero_padded_tokens_masks_only_padded_token_positions():
    tensor = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)

    masked = _zero_padded_tokens(tensor, torch.tensor(2), token_dim=1)

    torch.testing.assert_close(masked[:, :2], tensor[:, :2])
    assert torch.count_nonzero(masked[:, 2:]) == 0


@pytest.mark.parametrize(
    "requests,tokens,expected",
    [(4, 4, True), (4, 16, True), (5, 16, False), (4, 17, False)],
)
def test_builder310_spec_padding_checks_request_and_token_capacities(requests, tokens, expected):
    builder = object.__new__(AscendGDNAttentionMetadataBuilder310)
    builder.decode_cudagraph_max_bs = 4
    builder.spec_token_indx = torch.empty(16, dtype=torch.int32)
    assert builder._can_pad_spec_decode(requests, tokens) is expected


def test_common_spec_padding_keeps_existing_token_limit():
    builder = object.__new__(AscendGDNAttentionMetadataBuilder)
    builder.decode_cudagraph_max_bs = 4
    assert builder._can_pad_spec_decode(4, 4)
    assert not builder._can_pad_spec_decode(4, 16)


def test_mask_padded_recurrent_accepted_tokens_zeros_dummy_requests():
    accepted_tokens = torch.tensor([2, 3, 4], dtype=torch.int64)
    actual_seq_lengths = torch.tensor([4, 0, 1], dtype=torch.int32)

    masked = _mask_padded_recurrent_accepted_tokens(
        accepted_tokens,
        actual_seq_lengths,
    )

    assert masked.dtype == torch.int32
    assert masked.tolist() == [2, 0, 4]


@pytest.mark.parametrize("overflow", [None, "spec", "non_spec"])
def test_builder310_pads_spec_decode_metadata_with_dummy_requests(overflow):
    builder = object.__new__(AscendGDNAttentionMetadataBuilder310)
    builder.spec_state_indices_tensor = torch.full((4, 2), -1, dtype=torch.int32)
    builder.spec_sequence_masks = torch.empty(4, dtype=torch.bool)
    builder.non_spec_token_indx = torch.empty(0, dtype=torch.int32)
    builder.spec_token_indx = torch.empty(8, dtype=torch.int32)
    builder.spec_query_start_loc = torch.empty(5, dtype=torch.int32)
    builder.num_accepted_tokens = torch.empty(4, dtype=torch.int32)
    builder.spec_actual_seq_lengths = torch.empty(5, dtype=torch.int32)
    builder.use_full_cuda_graph = True
    attn_metadata = SimpleNamespace(
        num_prefills=0,
        num_decodes=0,
        num_spec_decodes=2,
        spec_state_indices_tensor=torch.tensor(
            [[3, 30], [4, 40]],
            dtype=torch.int32,
        ),
        spec_sequence_masks=torch.tensor([True, True]),
        spec_query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        num_accepted_tokens=torch.tensor([2, 3], dtype=torch.int32),
        non_spec_token_indx=torch.empty(0, dtype=torch.int32),
        spec_token_indx=torch.arange(8, dtype=torch.int32),
    )

    if overflow is not None:
        if overflow == "spec":
            attn_metadata.spec_token_indx = torch.arange(9, dtype=torch.int32)
        else:
            attn_metadata.non_spec_token_indx = torch.arange(1, dtype=torch.int32)
        before = builder.spec_state_indices_tensor.clone()
        with pytest.raises(ValueError, match="graph token buffer capacity"):
            builder._pad_spec_decode_metadata(attn_metadata, graph_request_count=4)
        torch.testing.assert_close(builder.spec_state_indices_tensor, before)
        return

    builder._pad_spec_decode_metadata(attn_metadata, graph_request_count=4)

    # 310P pads with PAD_SLOT_ID (-1), not NULL_BLOCK_ID (0), so FULL replay
    # does not write into mamba block 0. Pad accepted tokens stay 1 (not 0).
    assert attn_metadata.spec_state_indices_tensor.tolist() == [
        [3, 30],
        [4, 40],
        [PAD_SLOT_ID, PAD_SLOT_ID],
        [PAD_SLOT_ID, PAD_SLOT_ID],
    ]
    assert attn_metadata.spec_sequence_masks.tolist() == [True, True, False, False]
    assert attn_metadata.spec_query_start_loc.tolist() == [0, 4, 8, 8, 8]
    assert attn_metadata.num_accepted_tokens.tolist() == [2, 3, 1, 1]
    spec_meta = attn_metadata.spec_decode_metadata.spec_causal_conv1d
    assert spec_meta.query_start_loc.data_ptr() == attn_metadata.spec_query_start_loc.data_ptr()
    assert spec_meta.cache_indices.data_ptr() == attn_metadata.spec_state_indices_tensor.data_ptr()
    assert spec_meta.num_accepted_tokens.data_ptr() == attn_metadata.num_accepted_tokens.data_ptr()
    assert attn_metadata.spec_decode_metadata.actual_seq_lengths.tolist() == [0, 4, 4, 0, 0]


def test_builder310_pads_non_spec_decode_metadata_with_dummy_requests():
    builder = object.__new__(AscendGDNAttentionMetadataBuilder310)
    builder.non_spec_state_indices_tensor = torch.empty(4, dtype=torch.int32)
    builder.non_spec_query_start_loc = torch.empty(5, dtype=torch.int32)
    builder.non_spec_actual_seq_lengths = torch.empty(5, dtype=torch.int32)
    builder.use_full_cuda_graph = True
    attn_metadata = SimpleNamespace(
        num_prefills=0,
        num_decodes=2,
        num_decode_tokens=2,
        num_spec_decodes=0,
        non_spec_state_indices_tensor=torch.tensor([3, 4], dtype=torch.int32),
        non_spec_query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
    )

    builder._pad_decode_metadata(attn_metadata, graph_request_count=4)

    assert attn_metadata.non_spec_state_indices_tensor.tolist() == [
        3,
        4,
        NULL_BLOCK_ID,
        NULL_BLOCK_ID,
    ]
    assert attn_metadata.non_spec_query_start_loc.tolist() == [0, 1, 2, 2, 2]
    assert attn_metadata.non_spec_decode_metadata.actual_seq_lengths.tolist() == [0, 1, 1, 0, 0]
