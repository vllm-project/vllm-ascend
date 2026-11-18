# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.models.qwen4_exp.gbsa_union import (
    _build_qsa_union_inputs_torch,
)
from vllm_ascend.ops.triton.qwen4_exp.gbsa_union import (
    build_qsa_union_inputs_from_compact_triton,
    build_qsa_union_inputs_triton,
)

pytestmark = [
    pytest.mark.skipif(
        get_ascend_device_type() not in (AscendDeviceType.A3, AscendDeviceType.A5),
        reason="Requires Ascend A3/A5",
    ),
    pytest.mark.skipif(not HAS_TRITON, reason="Triton is not available"),
]


@pytest.mark.parametrize("rows", [1, 7, 8, 15, 16])
@pytest.mark.parametrize("topk", [8, 512])
@torch.inference_mode()
def test_qsa_union_inputs_matches_torch(rows: int, topk: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(rows * 1000 + topk)
    counts = torch.randint(0, topk + 1, (1, rows), generator=generator, dtype=torch.int32)
    sparse = torch.randint(0, 2048, (1, rows, topk), generator=generator, dtype=torch.int32)
    columns = torch.arange(topk).reshape(1, 1, topk)
    sparse.masked_fill_(columns >= counts.unsqueeze(-1), -1)
    # Exercise duplicate selections and the partial causal tail path.
    if topk >= 2:
        sparse[:, :, 1] = sparse[:, :, 0]
    visible = torch.randint(1, 8192, (rows,), generator=generator, dtype=torch.int32)
    sparse = sparse.npu()
    counts = counts.npu()
    visible = visible.npu()

    expected = _build_qsa_union_inputs_torch(sparse, counts, visible)
    actual = build_qsa_union_inputs_triton(sparse, counts, visible)
    torch.npu.synchronize()

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor.cpu(), expected_tensor.cpu(), rtol=0, atol=0)


@torch.inference_mode()
def test_qsa_union_inputs_rejects_non_power_of_two_width() -> None:
    sparse = torch.zeros((1, 8, 7), dtype=torch.int32, device="npu")
    counts = torch.full((1, 8), 7, dtype=torch.int32, device="npu")
    visible = torch.full((8,), 28, dtype=torch.int32, device="npu")
    with pytest.raises(ValueError, match="positive power of two"):
        build_qsa_union_inputs_triton(sparse, counts, visible)


@torch.inference_mode()
def test_qsa_union_inputs_direct_compact_matches_staged_path() -> None:
    """Cover per-request padding, holes, residual tails and 512 replacement."""
    rows = 12
    generator = torch.Generator(device="cpu").manual_seed(20261117)
    compact = torch.randint(-8, 2048, (rows, 515), generator=generator, dtype=torch.int32)
    compact[:, 1] = compact[:, 0]
    visible = torch.randint(1, 8192, (rows,), generator=generator, dtype=torch.int32)
    sparse = torch.full((1, 16, 512), -1, dtype=torch.int32)
    counts = torch.zeros((1, 16), dtype=torch.int32)
    padded_visible = torch.zeros(16, dtype=torch.int32)
    row_map = torch.tensor(
        [[0, 1, 2, 3, 4, -1, -1, -1], [5, 6, 7, 8, 9, 10, 11, -1]],
        dtype=torch.int32,
    )
    for group, mapped_rows in enumerate(row_map.tolist()):
        for lane, row in enumerate(mapped_rows):
            if row < 0:
                continue
            selected = compact[row, :512]
            selected = selected[(selected >= 0) & (selected < visible[row] // 4)]
            has_tail = int(visible[row] % 4 != 0)
            selected = selected[: 512 - has_tail]
            if has_tail:
                selected = torch.cat((selected, visible[row].div(4, rounding_mode="floor").reshape(1)))
            padded_row = group * 8 + lane
            sparse[0, padded_row, : selected.numel()] = selected
            counts[0, padded_row] = selected.numel()
            padded_visible[padded_row] = visible[row]

    compact = compact.npu()
    visible = visible.npu()
    row_map = row_map.npu()
    expected = _build_qsa_union_inputs_torch(sparse.npu(), counts.npu(), padded_visible.npu())
    actual = build_qsa_union_inputs_from_compact_triton(compact, visible, row_map)
    torch.npu.synchronize()
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor.cpu(), expected_tensor.cpu(), rtol=0, atol=0)
