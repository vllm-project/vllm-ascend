# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tests.e2e.ops.test_qwen4_exp_union import _reference_compact
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.models.qwen4_exp.attention_dispatch import build_qsa_union_group_row_map
from vllm_ascend.models.qwen4_exp.gbsa_union import build_qsa_union_inputs_from_compact

pytestmark = pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A3/A5",
)


@pytest.mark.parametrize("rows", [1, 7, 8, 15, 16])
@pytest.mark.parametrize("input_groups", [7, 511, 512])
@torch.inference_mode()
def test_qsa_union_inputs_matches_torch(rows, input_groups):
    generator = torch.Generator().manual_seed(rows * 1000 + input_groups)
    compact = torch.randint(-8, 2048, (rows, input_groups + 3), generator=generator, dtype=torch.int32)
    compact[:, 1] = compact[:, 0]
    visible = torch.randint(1, 8192, (rows,), generator=generator, dtype=torch.int32)
    row_map = torch.tensor(build_qsa_union_group_row_map((0, rows)), dtype=torch.int32)
    expected = _reference_compact(compact, visible, row_map)
    actual = build_qsa_union_inputs_from_compact(compact.npu(), visible.npu(), row_map.npu())
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor.cpu(), expected_tensor, rtol=0, atol=0)


def test_qsa_union_inputs_rejects_non_power_of_two_width():
    with pytest.raises(ValueError, match="positive power of two"):
        build_qsa_union_inputs_from_compact(torch.zeros(8, 11), torch.zeros(8), torch.zeros(1, 8))


@torch.inference_mode()
def test_qsa_union_inputs_direct_compact_matches_reference():
    generator = torch.Generator().manual_seed(20261117)
    compact = torch.randint(-8, 2048, (12, 515), generator=generator, dtype=torch.int32)
    compact[:, 1] = compact[:, 0]
    visible = torch.randint(1, 8192, (12,), generator=generator, dtype=torch.int32)
    row_map = torch.tensor(build_qsa_union_group_row_map((0, 5, 12)), dtype=torch.int32)
    expected = _reference_compact(compact, visible, row_map)
    actual = build_qsa_union_inputs_from_compact(compact.npu(), visible.npu(), row_map.npu())
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor.cpu(), expected_tensor, rtol=0, atol=0)
