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

import torch
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID

from vllm_ascend._310p.ops.fla.gdn_310 import (
    AscendGatedDeltaNetAttention310,
    _mask_padded_recurrent_accepted_tokens,
    _pad_spec_conv1d_host_args_shape_consistent_dummy_310p,
    _zero_padded_tokens,
)
from vllm_ascend._310p.ops.gdn_attn_builder_310 import (
    AscendGDNAttentionBackend310,
    AscendGDNAttentionMetadataBuilder310,
)


def test_ascend_gdn_attention_310_uses_310p_backend():
    assert AscendGatedDeltaNetAttention310.get_attn_backend(object()) is AscendGDNAttentionBackend310
    assert AscendGDNAttentionBackend310.get_builder_cls() is AscendGDNAttentionMetadataBuilder310


def test_zero_padded_tokens_masks_only_padded_token_positions():
    tensor = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)

    masked = _zero_padded_tokens(tensor, torch.tensor(2), token_dim=1)

    torch.testing.assert_close(masked[:, :2], tensor[:, :2])
    assert torch.count_nonzero(masked[:, 2:]) == 0


def test_mask_padded_recurrent_accepted_tokens_zeros_dummy_requests():
    accepted_tokens = torch.tensor([2, 3, 4], dtype=torch.int64)
    actual_seq_lengths = torch.tensor([4, 0, 1], dtype=torch.int32)

    masked = _mask_padded_recurrent_accepted_tokens(
        accepted_tokens,
        actual_seq_lengths,
    )

    assert masked.dtype == torch.int32
    assert masked.tolist() == [2, 0, 4]


def test_pad_spec_conv1d_host_args_adds_shape_consistent_dummy_requests():
    qsl_host, cidx_host, accepted_host = _pad_spec_conv1d_host_args_shape_consistent_dummy_310p(
        qsl_host=(0, 4, 8),
        cidx_host=(11,),
        num_accepted_host=(2,),
        cap_x_dim0=14,
        q_per_seq=4,
    )

    assert qsl_host == (0, 4, 8, 12, 14)
    assert cidx_host == (11, PAD_SLOT_ID, PAD_SLOT_ID, PAD_SLOT_ID)
    assert accepted_host == (2, 0, 0, 0)


def test_builder310_pads_spec_decode_metadata_with_dummy_requests():
    builder = object.__new__(AscendGDNAttentionMetadataBuilder310)
    builder.spec_state_indices_tensor = torch.full((4, 2), -1, dtype=torch.int32)
    builder.spec_sequence_masks = torch.empty(4, dtype=torch.bool)
    builder.non_spec_token_indx = torch.empty(0, dtype=torch.int32)
    builder.spec_token_indx = torch.empty(8, dtype=torch.int32)
    builder.spec_query_start_loc = torch.empty(5, dtype=torch.int32)
    builder.num_accepted_tokens = torch.empty(4, dtype=torch.int32)
    attn_metadata = SimpleNamespace(
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

    builder._pad_spec_decode_metadata(attn_metadata, graph_batch_size=4)

    assert attn_metadata.spec_state_indices_tensor.tolist() == [
        [3, 30],
        [4, 40],
        [NULL_BLOCK_ID, NULL_BLOCK_ID],
        [NULL_BLOCK_ID, NULL_BLOCK_ID],
    ]
    assert attn_metadata.spec_sequence_masks.tolist() == [True, True, False, False]
    assert attn_metadata.spec_query_start_loc.tolist() == [0, 4, 8, 8, 8]
    assert attn_metadata.num_accepted_tokens.tolist() == [2, 3, 0, 0]
