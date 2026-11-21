# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import pytest
import torch
from vllm.triton_utils import HAS_TRITON, triton

from tests.e2e.nightly.single_node.ops.singlecard_ops.triton.dcp_test_utils import assert_dcp_close
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.triton.qwen4_exp import hc as hc_module
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


def _reference_gate_mix(x: torch.Tensor, gate: torch.Tensor, group_dim: int) -> torch.Tensor:
    grouped_x = x.float().unflatten(-1, (HC_COUNT, group_dim))
    grouped_gate = torch.sigmoid(gate.float()).unflatten(-1, (HC_COUNT, group_dim))
    return (grouped_x * grouped_gate).mean(dim=-2).to(x.dtype)


def _legacy_gate_mix(x: torch.Tensor, gate: torch.Tensor, group_dim: int) -> torch.Tensor:
    output = x.new_empty((x.shape[0], group_dim))
    _hc_gate_mix_kernel[(x.shape[0], triton.cdiv(group_dim, _HC_GATE_MIX_BLOCK_SIZE))](
        x,
        gate,
        output,
        x.stride(0),
        gate.stride(0),
        output.stride(0),
        GROUP_DIM=group_dim,
        HC_COUNT=HC_COUNT,
        BLOCK_SIZE=_HC_GATE_MIX_BLOCK_SIZE,
    )
    return output


def _row_block_gate_mix(
    x: torch.Tensor,
    gate: torch.Tensor,
    group_dim: int,
    *,
    wide_offsets: bool = False,
) -> torch.Tensor:
    output = x.new_empty((x.shape[0], group_dim))
    _hc_gate_mix_row_block_kernel[
        (
            triton.cdiv(x.shape[0], _HC_GATE_MIX_ROWS_PER_BLOCK),
            triton.cdiv(group_dim, _HC_GATE_MIX_BLOCK_SIZE),
        )
    ](
        x,
        gate,
        output,
        x.shape[0],
        x.stride(0),
        gate.stride(0),
        output.stride(0),
        GROUP_DIM=group_dim,
        HC_COUNT=HC_COUNT,
        ROWS_PER_BLOCK=_HC_GATE_MIX_ROWS_PER_BLOCK,
        BLOCK_SIZE=_HC_GATE_MIX_BLOCK_SIZE,
        WIDE_OFFSETS=wide_offsets,
    )
    return output


@pytest.mark.parametrize(
    ("rows", "row_strided"),
    (
        (2047, False),
        (2048, False),
        (2049, True),
    ),
)
@torch.inference_mode()
def test_hc_gate_mix_row_block_matches_legacy_kernel(rows: int, row_strided: bool) -> None:
    x = _make_input(rows, HYPER_HIDDEN_SIZE, row_strided=row_strided)
    gate = _make_input(rows, HYPER_HIDDEN_SIZE, row_strided=row_strided)

    actual = hc_gate_mix(x, gate, HC_COUNT)
    legacy = _legacy_gate_mix(x, gate, GROUP_DIM)
    expected = _reference_gate_mix(x, gate, GROUP_DIM)
    torch.npu.synchronize()

    torch.testing.assert_close(actual, legacy, atol=0, rtol=0)
    assert_dcp_close(actual, expected)


@pytest.mark.parametrize("rows", (1024, 1025))
@pytest.mark.parametrize("row_strided", (False, True))
@torch.inference_mode()
def test_hc_gate_mix_row_block_combined_tails(rows: int, row_strided: bool) -> None:
    group_dim = 513
    x = _make_input(rows, HC_COUNT * group_dim, row_strided=row_strided)
    gate = _make_input(rows, HC_COUNT * group_dim, row_strided=row_strided)

    actual = _row_block_gate_mix(x, gate, group_dim)
    legacy = _legacy_gate_mix(x, gate, group_dim)
    expected = _reference_gate_mix(x, gate, group_dim)
    torch.npu.synchronize()

    torch.testing.assert_close(actual, legacy, atol=0, rtol=0)
    assert_dcp_close(actual, expected)


@torch.inference_mode()
def test_hc_gate_mix_row_block_npugraph_replay() -> None:
    rows = 2049
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

    actual = _row_block_gate_mix(x, gate, GROUP_DIM, wide_offsets=True)
    legacy = _legacy_gate_mix(x, gate, GROUP_DIM)
    torch.npu.synchronize()

    torch.testing.assert_close(actual, legacy, atol=0, rtol=0)


@torch.inference_mode()
def test_hc_gate_mix_row_counts_reuse_compilation(monkeypatch: pytest.MonkeyPatch) -> None:
    max_rows = 2057
    x_backing = _make_input(max_rows, HYPER_HIDDEN_SIZE, row_strided=True)
    gate_backing = _make_input(max_rows, HYPER_HIDDEN_SIZE, row_strided=True)

    isolated_kernel = triton.jit(
        _hc_gate_mix_row_block_kernel.fn,
        do_not_specialize=_hc_gate_mix_row_block_kernel.do_not_specialize,
    )
    compile_events = []
    original_compile = isolated_kernel._do_compile

    def record_compile(*args, **kwargs):
        compile_events.append(args[0])
        return original_compile(*args, **kwargs)

    monkeypatch.setattr(isolated_kernel, "_do_compile", record_compile)
    monkeypatch.setattr(hc_module, "_hc_gate_mix_row_block_kernel", isolated_kernel)

    for rows in (2048, 2049, 2056, 2057):
        x = x_backing[:rows]
        gate = gate_backing[:rows]
        hc_gate_mix(x, gate, HC_COUNT)
        torch.npu.synchronize()

        assert len(compile_events) == 1, (rows, compile_events)
