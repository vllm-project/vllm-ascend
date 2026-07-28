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

from unittest.mock import patch

import pytest
import torch

from vllm_ascend.attention.mla_v1 import (
    _broadcast_mla_fia_tensor,
    _merge_mla_fia_splits,
    _mla_fia_num_splits,
    _normalize_mla_fia_output,
    _split_mla_fia_block_table,
    _split_mla_fia_seq_lens,
)


@pytest.mark.parametrize(
    ("batch_size", "expected"),
    [
        (1, 16),
        (2, 16),
        (4, 8),
        (8, 4),
        (16, 2),
        (32, 1),
        (64, 1),
    ],
)
def test_mla_fia_num_splits(batch_size: int, expected: int):
    assert _mla_fia_num_splits(batch_size) == expected


def test_mla_fia_num_splits_rejects_empty_batch():
    with pytest.raises(ValueError, match="batch_size must be positive"):
        _mla_fia_num_splits(0)


def test_split_mla_fia_block_table_non_divisible_batch_major_order():
    block_table = torch.tensor(
        [
            [11, 12, 13, 14, 15],
            [21, 22, 23, 24, 25],
        ],
        dtype=torch.int32,
    )

    split_table, blocks_per_split = _split_mla_fia_block_table(block_table, 3)

    assert blocks_per_split == 2
    torch.testing.assert_close(
        split_table,
        torch.tensor(
            [
                [11, 12],
                [13, 14],
                [15, 0],
                [21, 22],
                [23, 24],
                [25, 0],
            ],
            dtype=torch.int32,
        ),
    )
    assert split_table.is_contiguous()


def test_split_mla_fia_seq_lens_mixed_and_empty_requests():
    assert _split_mla_fia_seq_lens(
        [257, 64],
        batch_size=3,
        num_splits=3,
        blocks_per_split=1,
        block_size=128,
    ) == [
        128,
        128,
        1,
        64,
        0,
        0,
        0,
        0,
        0,
    ]


def test_broadcast_mla_fia_tensor_matches_block_table_order():
    query = torch.tensor(
        [
            [[1.0, 2.0]],
            [[3.0, 4.0]],
        ]
    )

    broadcast = _broadcast_mla_fia_tensor(query, 3)

    torch.testing.assert_close(
        broadcast,
        torch.tensor(
            [
                [[1.0, 2.0]],
                [[1.0, 2.0]],
                [[1.0, 2.0]],
                [[3.0, 4.0]],
                [[3.0, 4.0]],
                [[3.0, 4.0]],
            ]
        ),
    )
    assert broadcast.is_contiguous()


def test_broadcast_mla_fia_quant_scale_preserves_batch_head_order():
    scale = torch.tensor(
        [
            [[11.0], [12.0]],
            [[21.0], [22.0]],
        ]
    )

    broadcast = _broadcast_mla_fia_tensor(scale, 3)

    assert tuple(broadcast.shape) == (6, 2, 1)
    torch.testing.assert_close(
        broadcast[:, :, 0],
        torch.tensor(
            [
                [11.0, 12.0],
                [11.0, 12.0],
                [11.0, 12.0],
                [21.0, 22.0],
                [21.0, 22.0],
                [21.0, 22.0],
            ]
        ),
    )
    assert broadcast.is_contiguous()


def test_normalize_mla_fia_output_bnsd_preserves_batch_head_order():
    batch_size = 3
    num_heads = 2
    head_dim = 4
    output = torch.arange(
        batch_size * num_heads * head_dim,
        dtype=torch.float32,
    ).view(batch_size, num_heads, 1, head_dim)

    normalized = _normalize_mla_fia_output(
        output,
        input_layout="BNSD",
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
    )

    torch.testing.assert_close(normalized, output[:, :, 0, :])
    head_major = normalized.permute(1, 0, 2).contiguous()
    assert tuple(head_major.shape) == (num_heads, batch_size, head_dim)
    torch.testing.assert_close(head_major[:, 1], output[1, :, 0, :])


def _attention_update_reference(lse_list, output_list, update_type):
    assert update_type == 0
    lse = torch.stack(lse_list)
    output = torch.stack(output_list)
    weights = torch.softmax(lse, dim=0)
    merged_lse = torch.logsumexp(lse, dim=0)
    merged_output = (weights[..., None] * output).sum(dim=0)
    return merged_output, merged_lse


@patch(
    "vllm_ascend.attention.mla_v1.torch_npu.npu_attention_update",
    side_effect=_attention_update_reference,
)
@pytest.mark.parametrize("input_layout", ["BNSD_NBSD", "BSND_NBSD"])
def test_merge_mla_fia_splits_masks_empty_splits_and_padding(
    mock_update,
    input_layout,
):
    batch_size = 2
    num_splits = 2
    num_heads = 2
    head_dim = 2

    # Canonical local output is [B, S, H, D]. Convert it to FIA's NBSD
    # output layout [H, B*S, 1, D].
    canonical_output = torch.tensor(
        [
            [
                [[1.0, 10.0], [2.0, 20.0]],
                [[3.0, 30.0], [4.0, 40.0]],
            ],
            [
                [[100.0, 100.0], [100.0, 100.0]],
                [[200.0, 200.0], [200.0, 200.0]],
            ],
        ]
    )
    # Model an FIA implementation that leaves empty-shard output undefined.
    canonical_output[1] = float("nan")
    local_output = (
        canonical_output.reshape(batch_size * num_splits, num_heads, head_dim)
        .permute(1, 0, 2)
        .unsqueeze(2)
        .contiguous()
    )
    local_lse = torch.zeros(batch_size * num_splits, num_heads, 1, 1, dtype=torch.float32)

    merged = _merge_mla_fia_splits(
        local_output,
        local_lse,
        input_layout=input_layout,
        batch_size=batch_size,
        num_splits=num_splits,
        num_heads=num_heads,
        head_dim=head_dim,
        seq_lens_device=torch.tensor([128, 0], dtype=torch.int32),
        chunk_tokens=64,
    )

    # Request 0 has two equally weighted splits; request 1 is graph padding
    # and must be zero even though every Local Output contains nonzero data.
    expected_batch_major = torch.tensor(
        [
            [[2.0, 20.0], [3.0, 30.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]
    )
    torch.testing.assert_close(merged, expected_batch_major.permute(1, 0, 2))
    mock_update.assert_called_once()


def test_merge_mla_fia_splits_rejects_short_device_lengths():
    with pytest.raises(ValueError, match="one device-side sequence length"):
        _merge_mla_fia_splits(
            torch.zeros(2, 2, 1, 2),
            torch.zeros(2, 2, 1, 1),
            input_layout="BNSD",
            batch_size=2,
            num_splits=1,
            num_heads=2,
            head_dim=2,
            seq_lens_device=torch.tensor([1], dtype=torch.int32),
            chunk_tokens=128,
        )
