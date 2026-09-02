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
from unittest.mock import patch

import torch
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from vllm_ascend._310p.ops import gdn_attn_builder_310
from vllm_ascend._310p.ops.fla.gdn_310 import (
    AscendGatedDeltaNetAttention310,
    _mask_padded_recurrent_accepted_tokens,
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


def test_builder310_pads_spec_decode_metadata_with_dummy_requests():
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

    builder._pad_spec_decode_metadata(
        attn_metadata,
        graph_batch_size=4,
        graph_num_tokens=8,
    )

    assert attn_metadata.spec_state_indices_tensor.tolist() == [
        [3, 30],
        [4, 40],
        [NULL_BLOCK_ID, NULL_BLOCK_ID],
        [NULL_BLOCK_ID, NULL_BLOCK_ID],
    ]
    assert attn_metadata.spec_sequence_masks.tolist() == [True, True, False, False]
    assert attn_metadata.spec_query_start_loc.tolist() == [0, 4, 8, 8, 8]
    assert attn_metadata.num_accepted_tokens.tolist() == [2, 3, 0, 0]
    spec_meta = attn_metadata.spec_decode_metadata.spec_causal_conv1d
    assert spec_meta.query_start_loc.data_ptr() == attn_metadata.spec_query_start_loc.data_ptr()
    assert spec_meta.cache_indices.data_ptr() == attn_metadata.spec_state_indices_tensor.data_ptr()
    assert spec_meta.num_accepted_tokens.data_ptr() == attn_metadata.num_accepted_tokens.data_ptr()
    assert attn_metadata.spec_decode_metadata.actual_seq_lengths.tolist() == [0, 4, 4, 0, 0]


def test_builder310_keeps_spec_token_indices_at_graph_descriptor_extent():
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
        num_spec_decodes=3,
        spec_state_indices_tensor=torch.tensor(
            [[3, 30], [4, 40], [5, 50]],
            dtype=torch.int32,
        ),
        spec_sequence_masks=torch.tensor([True, True, True]),
        spec_query_start_loc=torch.tensor([0, 2, 4, 6], dtype=torch.int32),
        num_accepted_tokens=torch.tensor([2, 2, 2], dtype=torch.int32),
        non_spec_token_indx=torch.empty(0, dtype=torch.int32),
        spec_token_indx=torch.arange(6, dtype=torch.int32),
    )

    builder._pad_spec_decode_metadata(
        attn_metadata,
        graph_batch_size=4,
        graph_num_tokens=8,
        pad_to_graph_descriptor=True,
    )

    assert attn_metadata.spec_token_indx.data_ptr() == (builder.spec_token_indx.data_ptr())
    assert attn_metadata.spec_token_indx.tolist() == list(range(8))


def test_hybrid_full_uses_descriptor_extent_for_persistent_spec_token_view():
    with (
        patch.object(
            gdn_attn_builder_310,
            "is_310p_dflash_full_and_piecewise",
            return_value=True,
        ),
        patch.object(
            gdn_attn_builder_310,
            "is_310p_dflash_full_decode_only",
            return_value=False,
        ),
    ):
        assert gdn_attn_builder_310._should_pad_spec_tokens_to_graph_descriptor(
            SimpleNamespace(),
            use_full_graph=True,
        )


def test_builder310_replay_updates_logical_spec_metadata_at_stable_addresses():
    builder = object.__new__(AscendGDNAttentionMetadataBuilder310)
    builder.spec_state_indices_tensor = torch.full((10, 2), -1, dtype=torch.int32)
    builder.spec_sequence_masks = torch.empty(10, dtype=torch.bool)
    builder.non_spec_token_indx = torch.empty(0, dtype=torch.int32)
    builder.spec_token_indx = torch.empty(90, dtype=torch.int32)
    builder.spec_query_start_loc = torch.empty(11, dtype=torch.int32)
    builder.num_accepted_tokens = torch.empty(10, dtype=torch.int32)
    builder.spec_actual_seq_lengths = torch.empty(11, dtype=torch.int32)
    builder.use_full_cuda_graph = True

    def make_metadata(num_reqs: int, width: int, state_base: int):
        logical_tokens = num_reqs * width
        return SimpleNamespace(
            num_prefills=0,
            num_decodes=0,
            num_spec_decodes=num_reqs,
            spec_state_indices_tensor=torch.stack(
                (
                    torch.arange(state_base, state_base + num_reqs),
                    torch.arange(state_base + 100, state_base + 100 + num_reqs),
                ),
                dim=1,
            ).to(torch.int32),
            spec_sequence_masks=torch.ones(num_reqs, dtype=torch.bool),
            spec_query_start_loc=torch.arange(
                0,
                logical_tokens + 1,
                width,
                dtype=torch.int32,
            ),
            num_accepted_tokens=torch.full((num_reqs,), width, dtype=torch.int32),
            non_spec_token_indx=torch.empty(0, dtype=torch.int32),
            spec_token_indx=torch.arange(logical_tokens, dtype=torch.int32),
        )

    first = make_metadata(num_reqs=7, width=9, state_base=10)
    builder._pad_spec_decode_metadata(
        first,
        graph_batch_size=10,
        graph_num_tokens=90,
        pad_to_graph_descriptor=True,
    )
    addresses = {
        "states": first.spec_state_indices_tensor.data_ptr(),
        "masks": first.spec_sequence_masks.data_ptr(),
        "starts": first.spec_query_start_loc.data_ptr(),
        "tokens": first.spec_token_indx.data_ptr(),
        "accepted": first.num_accepted_tokens.data_ptr(),
    }

    second = make_metadata(num_reqs=2, width=2, state_base=40)
    builder._pad_spec_decode_metadata(
        second,
        graph_batch_size=10,
        graph_num_tokens=90,
        pad_to_graph_descriptor=True,
    )

    assert second.spec_state_indices_tensor.data_ptr() == addresses["states"]
    assert second.spec_sequence_masks.data_ptr() == addresses["masks"]
    assert second.spec_query_start_loc.data_ptr() == addresses["starts"]
    assert second.spec_token_indx.data_ptr() == addresses["tokens"]
    assert second.num_accepted_tokens.data_ptr() == addresses["accepted"]
    assert second.spec_query_start_loc.tolist() == [0, 2, 4] + [4] * 8
    assert second.spec_sequence_masks.tolist() == [True, True] + [False] * 8
    assert second.spec_state_indices_tensor[:2].tolist() == [[40, 140], [41, 141]]
    assert torch.all(second.spec_state_indices_tensor[2:] == NULL_BLOCK_ID)
    assert second.num_accepted_tokens.tolist() == [2, 2] + [0] * 8
    assert second.spec_token_indx.tolist() == list(range(90))


def test_builder310_keeps_legacy_spec_token_extent_outside_dflash_fdo():
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
        num_spec_decodes=3,
        spec_state_indices_tensor=torch.tensor([[3, 30], [4, 40], [5, 50]], dtype=torch.int32),
        spec_sequence_masks=torch.tensor([True, True, True]),
        spec_query_start_loc=torch.tensor([0, 2, 4, 6], dtype=torch.int32),
        num_accepted_tokens=torch.tensor([2, 2, 2], dtype=torch.int32),
        non_spec_token_indx=torch.empty(0, dtype=torch.int32),
        spec_token_indx=torch.arange(6, dtype=torch.int32),
    )

    builder._pad_spec_decode_metadata(
        attn_metadata,
        graph_batch_size=4,
        graph_num_tokens=8,
        pad_to_graph_descriptor=False,
    )

    assert attn_metadata.spec_token_indx.tolist() == list(range(6))
