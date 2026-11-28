# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.models.qwen4_exp.gbsa_union import build_qsa_union_inputs_from_compact
from vllm_ascend.utils import enable_custom_op

pytestmark = pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A3/A5",
)


def test_qsa_union_inputs_meta_rejects_non_power_of_two_width():
    enable_custom_op()
    compact = torch.empty((8, 11), device="meta", dtype=torch.int32)
    visible = torch.empty(8, device="meta", dtype=torch.int32)
    row_map = torch.empty((1, 8), device="meta", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="positive power of two"):
        build_qsa_union_inputs_from_compact(compact, visible, row_map)


def test_qsa_union_inputs_meta_reserves_tail_beyond_saturated_topk():
    enable_custom_op()
    compact = torch.empty((8, 515), device="meta", dtype=torch.int32)
    visible = torch.empty(8, device="meta", dtype=torch.int32)
    row_map = torch.empty((1, 8), device="meta", dtype=torch.int32)
    indices, counts, mask = build_qsa_union_inputs_from_compact(compact, visible, row_map)
    assert indices.shape == (1, 1, 4104)
    assert counts.shape == (1, 1)
    assert mask.shape == (1, 1, 4104 * 4)


@pytest.mark.parametrize("input_groups", [7, 511])
def test_qsa_union_inputs_exact_membership(input_groups):
    # Lane bits restart at zero in the second request. Blocks contain four tokens.
    base = (1 << 21) - 4
    compact = torch.full((3, input_groups + 3), -1, dtype=torch.int32)
    compact[0, :3] = torch.tensor([base, base, base + 3])
    compact[1, :2] = torch.tensor([base + 1, base])
    visible = torch.tensor([4 * base + 9, 4 * base + 8, 4 * base + 11], dtype=torch.int32)
    row_map = torch.tensor([[0, 1, -1, -1, -1, -1, -1, -1], [2, -1, -1, -1, -1, -1, -1, -1]], dtype=torch.int32)
    width = 8 * (input_groups + 1)
    indices = torch.full((1, 2, width), -1, dtype=torch.int32)
    indices[0, 0, :3] = torch.tensor([base, base + 1, base + 2])
    indices[0, 1, 0] = base + 2
    mask = torch.zeros((1, 2, width * 4), dtype=torch.int32)
    mask[0, 0, :9] = torch.tensor([3, 3, 3, 3, 2, 2, 2, 2, 1])
    mask[0, 1, :3] = 1
    actual = torch.ops._C_ascend.qsa_union_inputs(compact.npu(), visible.npu(), row_map.npu())
    for output, expected in zip(actual, (indices, torch.tensor([[3, 1]], dtype=torch.int32), mask), strict=True):
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


def test_full_chunk_union_with_saturated_topk():
    rows = 4096
    compact = torch.full((rows, 515), -1, dtype=torch.int32)
    compact[:, :384] = torch.arange(384)
    private = 512 + torch.arange(8)[:, None] * 128 + torch.arange(128)
    compact[:, 384:512] = private.repeat(rows // 8, 1)
    # The tail is additional to all 512 complete groups in every lane.
    tail = 32767
    visible = torch.full((rows,), tail * 4 + 1, dtype=torch.int32)
    row_map = torch.arange(rows, dtype=torch.int32).reshape(-1, 8)
    blocks = torch.cat((torch.arange(384), private.flatten(), torch.tensor([tail])))
    count = blocks.numel()
    indices = torch.full((4104,), -1, dtype=torch.int32)
    indices[:count] = blocks
    mask = torch.zeros((4104, 4), dtype=torch.int32)
    mask[:384] = 255
    mask[384 : count - 1] = (1 << torch.arange(8)).repeat_interleave(128)[:, None]
    mask[count - 1, 0] = 255
    expected = (
        indices.expand(1, rows // 8, -1),
        torch.full((1, rows // 8), count, dtype=torch.int32),
        mask.flatten().expand(1, rows // 8, -1),
    )
    actual = torch.ops._C_ascend.qsa_union_inputs(compact.npu(), visible.npu(), row_map.npu())
    for output, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(output.cpu(), reference, rtol=0, atol=0)


def test_union_tail_merges_with_another_lane_full_group():
    compact = torch.full((2, 10), -1, dtype=torch.int32)
    compact[0, 0] = 0
    compact[1, :3] = torch.tensor([0, 1, 2])
    visible = torch.tensor([9, 12], dtype=torch.int32)
    row_map = torch.tensor([[0, 1, -1, -1, -1, -1, -1, -1]], dtype=torch.int32)
    indices, counts, mask = torch.ops._C_ascend.qsa_union_inputs(compact.npu(), visible.npu(), row_map.npu())
    assert counts.cpu().tolist() == [[3]]
    assert indices.cpu()[0, 0, :3].tolist() == [0, 1, 2]
    assert mask.cpu()[0, 0, :12].reshape(3, 4).tolist() == [
        [3, 3, 3, 3],
        [2, 2, 2, 2],
        [3, 2, 2, 2],
    ]


def test_union_unmatched_tail_is_merged_in_block_order():
    compact = torch.full((8, 515), -1, dtype=torch.int32)
    compact[:7, :512] = torch.arange(512)
    compact[7, :511] = torch.arange(511)
    compact[7, 511] = 1025
    visible = torch.arange(4097, 4105, dtype=torch.int32)
    row_map = torch.arange(8, dtype=torch.int32).reshape(1, 8)
    indices, counts, mask = torch.ops._C_ascend.qsa_union_inputs(compact.npu(), visible.npu(), row_map.npu())
    assert counts.cpu().tolist() == [[514]]
    actual_indices = indices.cpu()[0, 0, :514]
    expected_indices = torch.cat((torch.arange(512, dtype=torch.int32), torch.tensor([1024, 1025], dtype=torch.int32)))
    torch.testing.assert_close(actual_indices, expected_indices, rtol=0, atol=0)
    assert torch.all(actual_indices[1:] > actual_indices[:-1]).item()
    actual_mask = mask.cpu()[0, 0, : 514 * 4].reshape(514, 4)
    assert actual_mask[511].tolist() == [127, 127, 127, 127]
    assert actual_mask[512].tolist() == [7, 6, 4, 0]
    assert actual_mask[513].tolist() == [240, 224, 192, 128]


@pytest.mark.parametrize("groups", [1021, 1022, 1023, 1024])
@torch.inference_mode()
def test_qsa_union_inputs_count_writeback(groups):
    rows = groups * 8
    compact = torch.full((rows, 515), -1, dtype=torch.int32, device="npu")
    compact[:, :128] = torch.arange(128, dtype=torch.int32, device="npu")
    visible = torch.full((rows,), 131072, dtype=torch.int32, device="npu")
    row_map = torch.arange(rows, dtype=torch.int32, device="npu").reshape(groups, 8)
    expected = torch.full((1, groups), 128, dtype=torch.int32)
    for _ in range(32):
        _, counts, _ = torch.ops._C_ascend.qsa_union_inputs(compact, visible, row_map)
        torch.testing.assert_close(counts.cpu(), expected, rtol=0, atol=0)
