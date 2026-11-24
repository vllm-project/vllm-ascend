# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_ascend.ops.triton.qwen4_exp.prepare_qsa_visible_lengths_triton import prepare_qsa_visible_lengths


@pytest.mark.parametrize("rows,kv_length", [(0, 4096), (1, 4096), (4095, 131071), (4096, 131072)])
def test_prefill_visible_lengths(rows, kv_length):
    positions = torch.arange(kv_length - rows, kv_length, dtype=torch.int64)
    output = torch.full((rows + 2,), -99, dtype=torch.int32, device="npu")
    prepare_qsa_visible_lengths(
        positions.to("npu"),
        torch.tensor([kv_length], dtype=torch.int32, device="npu"),
        torch.zeros(rows, dtype=torch.int32, device="npu"),
        output,
        rows,
    )
    expected = torch.cat((positions.int() + 1, torch.full((2,), -99, dtype=torch.int32)))
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


def test_multi_request_strides_and_boundaries():
    def strided(values, dtype):
        return torch.tensor(values, dtype=dtype, device="npu").repeat_interleave(2)[::2]

    output = torch.empty(8, dtype=torch.int32, device="npu")
    prepare_qsa_visible_lengths(
        strided([3, 5, -1, 99, 0, 4, 8, 6], torch.int32),
        strided([4, 7], torch.int32),
        strided([0, 1, 1, -1, -2, -3, 2, 1], torch.int64),
        output,
        8,
    )
    torch.testing.assert_close(output.cpu(), torch.tensor([4, 6, 0, 7, 1, 0, 0, 7], dtype=torch.int32), rtol=0, atol=0)
