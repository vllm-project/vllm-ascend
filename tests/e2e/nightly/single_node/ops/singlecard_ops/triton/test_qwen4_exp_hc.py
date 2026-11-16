# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import pytest
import torch
from vllm.triton_utils import HAS_TRITON, triton

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.triton.qwen4_exp.hc import (
    _HC_GATE_MIX_BLOCK_SIZE,
    _HC_GATE_MIX_ROWS_PER_BLOCK,
    _hc_gate_mix_kernel,
    _hc_gate_mix_row_block_kernel,
    hc_gate_mix,
)

pytestmark = [
    pytest.mark.skipif(not HAS_TRITON, reason="Triton is not available"),
    pytest.mark.skipif(
        not get_current_hardware_profile().supports(HardwareCapability.QWEN4EXP_HC_GATE_MIX_ROW_BLOCK),
        reason="Qwen4Exp HC row-block kernel is not enabled for this hardware profile",
    ),
]

HC_COUNT = 4
GROUP_DIM = 2560
HYPER_HIDDEN_SIZE = HC_COUNT * GROUP_DIM
DTYPE = torch.bfloat16


def _make_input(rows: int, width: int, *, row_strided: bool) -> torch.Tensor:
    if not row_strided:
        return torch.randn((rows, width), device="npu", dtype=DTYPE) * 0.5
    backing = torch.randn((rows, width + 16), device="npu", dtype=DTYPE) * 0.5
    result = backing[:, :width]
    assert result.stride(1) == 1 and not result.is_contiguous()
    return result


def _reference_gate_mix(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    grouped_x = x.float().unflatten(-1, (HC_COUNT, GROUP_DIM))
    grouped_gate = torch.sigmoid(gate.float()).unflatten(-1, (HC_COUNT, GROUP_DIM))
    return (grouped_x * grouped_gate).mean(dim=-2).to(x.dtype)


def _legacy_gate_mix(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    output = x.new_empty((x.shape[0], GROUP_DIM))
    _hc_gate_mix_kernel[(x.shape[0], triton.cdiv(GROUP_DIM, _HC_GATE_MIX_BLOCK_SIZE))](
        x,
        gate,
        output,
        x.stride(0),
        gate.stride(0),
        output.stride(0),
        GROUP_DIM=GROUP_DIM,
        HC_COUNT=HC_COUNT,
        BLOCK_SIZE=_HC_GATE_MIX_BLOCK_SIZE,
    )
    return output


def _wide_row_block_gate_mix(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    output = x.new_empty((x.shape[0], GROUP_DIM))
    _hc_gate_mix_row_block_kernel[
        (
            triton.cdiv(x.shape[0], _HC_GATE_MIX_ROWS_PER_BLOCK),
            triton.cdiv(GROUP_DIM, _HC_GATE_MIX_BLOCK_SIZE),
        )
    ](
        x,
        gate,
        output,
        x.shape[0],
        x.stride(0),
        gate.stride(0),
        output.stride(0),
        GROUP_DIM=GROUP_DIM,
        HC_COUNT=HC_COUNT,
        ROWS_PER_BLOCK=_HC_GATE_MIX_ROWS_PER_BLOCK,
        BLOCK_SIZE=_HC_GATE_MIX_BLOCK_SIZE,
        EVEN_ROWS=x.shape[0] % _HC_GATE_MIX_ROWS_PER_BLOCK == 0,
        EVEN_BLOCK=GROUP_DIM % _HC_GATE_MIX_BLOCK_SIZE == 0,
        WIDE_OFFSETS=True,
    )
    return output


@pytest.mark.parametrize(
    ("rows", "row_strided"),
    (
        (1023, False),
        (1024, False),
        (1025, True),
    ),
)
@torch.inference_mode()
def test_hc_gate_mix_row_block_matches_legacy_kernel(rows: int, row_strided: bool) -> None:
    x = _make_input(rows, HYPER_HIDDEN_SIZE, row_strided=row_strided)
    gate = _make_input(rows, HYPER_HIDDEN_SIZE, row_strided=row_strided)

    actual = hc_gate_mix(x, gate, HC_COUNT)
    legacy = _legacy_gate_mix(x, gate)
    expected = _reference_gate_mix(x, gate)
    torch.npu.synchronize()

    torch.testing.assert_close(actual, legacy, atol=0, rtol=0)
    torch.testing.assert_close(actual.float(), expected.float(), atol=3e-2, rtol=3e-2)


@torch.inference_mode()
def test_hc_gate_mix_row_block_npugraph_replay() -> None:
    rows = 1025
    x = _make_input(rows, HYPER_HIDDEN_SIZE, row_strided=True)
    gate = _make_input(rows, HYPER_HIDDEN_SIZE, row_strided=True)
    for _ in range(3):
        hc_gate_mix(x, gate, HC_COUNT)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        captured = hc_gate_mix(x, gate, HC_COUNT)

    for _ in range(2):
        x.copy_(_make_input(rows, HYPER_HIDDEN_SIZE, row_strided=True))
        gate.copy_(_make_input(rows, HYPER_HIDDEN_SIZE, row_strided=True))
        expected = hc_gate_mix(x, gate, HC_COUNT)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(captured, expected, atol=0, rtol=0)


@torch.inference_mode()
def test_hc_gate_mix_row_block_wide_offsets_matches_legacy() -> None:
    rows = 1025
    x = _make_input(rows, HYPER_HIDDEN_SIZE, row_strided=True)
    gate = _make_input(rows, HYPER_HIDDEN_SIZE, row_strided=True)

    actual = _wide_row_block_gate_mix(x, gate)
    legacy = _legacy_gate_mix(x, gate)
    torch.npu.synchronize()

    torch.testing.assert_close(actual, legacy, atol=0, rtol=0)
