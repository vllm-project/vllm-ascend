#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import torch

from vllm_ascend.attention.utils import (
    filter_chunked_req_indices,
    get_or_register_attention_buffer,
    transdata,
)


def test_get_or_register_attention_buffer() -> None:
    module_a = torch.nn.Module()
    module_b = torch.nn.Module()
    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={
                "layer.a": module_a,
                "layer.b": module_b,
            }
        )
    )
    factory_call_count = 0

    def factory() -> torch.Tensor:
        nonlocal factory_call_count
        factory_call_count += 1
        return torch.tensor([1, 2, 3])

    buffer = get_or_register_attention_buffer(
        vllm_config,
        ["layer.a", "layer.b"],
        "_test_buffer",
        factory,
    )

    assert factory_call_count == 1
    assert module_a._buffers["_test_buffer"] is buffer
    assert module_b._buffers["_test_buffer"] is buffer
    assert "_test_buffer" not in module_a.state_dict()
    assert "_test_buffer" not in module_b.state_dict()


def test_filter_chunked_req_indices_empty_mask() -> None:
    indices = filter_chunked_req_indices(
        torch.tensor([2, 1, 3]),
        [False, False, False],
    )

    torch.testing.assert_close(indices, torch.empty(0, dtype=torch.long))


def test_filter_chunked_req_indices_mixed_mask() -> None:
    indices = filter_chunked_req_indices(
        torch.tensor([2, 1, 3]),
        [True, False, True],
    )

    torch.testing.assert_close(indices, torch.tensor([0, 1, 3, 4, 5]))


def test_transdata_pads_rows_and_columns_to_block_alignment() -> None:
    # Regression test: F.pad 4-tuple order is (col_left, col_right, row_top,
    # row_bottom). A swapped (r_pad, c_pad) would pad the wrong axes, making
    # the subsequent block reshape fail (numel mismatch) or silently corrupt
    # the NZ layout. Input (5, 6) with block (4, 4) requires row_pad=3,
    # col_pad=2: the swapped order would yield (7, 9) instead of (8, 8).
    m = torch.arange(5 * 6, dtype=torch.float32).reshape(5, 6)

    nz = transdata(m, block_size=(4, 4))

    expected = torch.tensor(
        [
            # block_col 0: block(0,0) then block(1,0), rows zero-padded
            [0, 1, 2, 3],
            [6, 7, 8, 9],
            [12, 13, 14, 15],
            [18, 19, 20, 21],
            [24, 25, 26, 27],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            # block_col 1: block(0,1) then block(1,1), cols zero-padded
            [4, 5, 0, 0],
            [10, 11, 0, 0],
            [16, 17, 0, 0],
            [22, 23, 0, 0],
            [28, 29, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=torch.float32,
    ).reshape(2, 8, 4)
    torch.testing.assert_close(nz, expected)


def test_transdata_aligned_input_is_unchanged_by_padding() -> None:
    # Aligned inputs (the shapes used by the current sfa_v1 call sites) need
    # no padding; the fix must keep that path byte-for-byte identical.
    m = torch.arange(8 * 8, dtype=torch.float32).reshape(8, 8)

    nz = transdata(m, block_size=(4, 4))

    expected = torch.tensor(
        [
            # block_col 0
            [0, 1, 2, 3],
            [8, 9, 10, 11],
            [16, 17, 18, 19],
            [24, 25, 26, 27],
            [32, 33, 34, 35],
            [40, 41, 42, 43],
            [48, 49, 50, 51],
            [56, 57, 58, 59],
            # block_col 1
            [4, 5, 6, 7],
            [12, 13, 14, 15],
            [20, 21, 22, 23],
            [28, 29, 30, 31],
            [36, 37, 38, 39],
            [44, 45, 46, 47],
            [52, 53, 54, 55],
            [60, 61, 62, 63],
        ],
        dtype=torch.float32,
    ).reshape(2, 8, 4)
    torch.testing.assert_close(nz, expected)
