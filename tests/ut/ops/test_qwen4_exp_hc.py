# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.device.hardware_profile import HardwareCapability
from vllm_ascend.ops.triton.qwen4_exp import hc


class _KernelRecorder:
    def __init__(self) -> None:
        self.launches: list[tuple[tuple[int, ...], tuple[object, ...], dict[str, object]]] = []
        self._grid: tuple[int, ...] = ()

    def __getitem__(self, grid: tuple[int, ...]) -> "_KernelRecorder":
        self._grid = grid
        return self

    def __call__(self, *args: object, **kwargs: object) -> None:
        self.launches.append((self._grid, args, kwargs))


@pytest.mark.parametrize(
    ("rows", "row_block_supported", "min_rows", "expected_kernel"),
    (
        (2047, True, 2048, "legacy"),
        (2048, True, 2048, "row_block"),
        (1024, True, 1024, "row_block"),
        (1024, False, 1024, "legacy"),
    ),
)
def test_hc_gate_mix_row_block_dispatch_is_scoped_by_capability(
    rows: int,
    row_block_supported: bool,
    min_rows: int,
    expected_kernel: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = _KernelRecorder()
    row_block = _KernelRecorder()
    requested_capabilities: list[HardwareCapability] = []

    def supports(capability: HardwareCapability) -> bool:
        requested_capabilities.append(capability)
        return row_block_supported

    profile = SimpleNamespace(
        supports=supports,
        qwen4exp_hc_gate_mix_row_block_min_rows=min_rows,
    )
    monkeypatch.setattr(hc, "_hc_gate_mix_kernel", legacy)
    monkeypatch.setattr(hc, "_hc_gate_mix_row_block_kernel", row_block)
    monkeypatch.setattr(hc, "get_current_hardware_profile", lambda: profile)

    x = torch.empty((rows, 40), dtype=torch.bfloat16)
    gate = torch.empty_like(x)
    output = hc._hc_gate_mix(x, gate, 4)

    assert output.shape == (rows, 10)
    if rows < min_rows:
        assert requested_capabilities == []
    else:
        assert requested_capabilities == [HardwareCapability.QWEN4EXP_HC_GATE_MIX_ROW_BLOCK]

    expected_recorder, other_recorder = (row_block, legacy) if expected_kernel == "row_block" else (legacy, row_block)
    assert len(expected_recorder.launches) == 1
    assert other_recorder.launches == []
    grid, _, kwargs = expected_recorder.launches[0]
    if expected_kernel == "row_block":
        assert grid == ((rows + hc._HC_GATE_MIX_ROWS_PER_BLOCK - 1) // hc._HC_GATE_MIX_ROWS_PER_BLOCK, 1)
        assert kwargs["WIDE_OFFSETS"] is False
    else:
        assert grid == (rows, 1)


@pytest.mark.parametrize(
    ("rows", "stride", "expected"),
    (
        (209_715, 10_240, False),
        (209_716, 10_240, True),
    ),
)
def test_hc_gate_mix_int32_offset_boundary(rows: int, stride: int, expected: bool) -> None:
    assert hc._row_major_offset_exceeds_int32(rows, 10_240, stride) is expected


@pytest.mark.parametrize(
    ("stride_x_row", "stride_gate_row", "stride_output_row", "expected"),
    (
        (10_240, 10_240, 2_560, False),
        (hc._INT32_MAX, 10_240, 2_560, True),
        (10_240, hc._INT32_MAX, 2_560, True),
        (10_240, 10_240, hc._INT32_MAX, True),
    ),
)
def test_hc_gate_mix_detects_wide_offsets_for_all_tensors(
    stride_x_row: int,
    stride_gate_row: int,
    stride_output_row: int,
    expected: bool,
) -> None:
    assert (
        hc._hc_gate_mix_needs_wide_offsets(
            2,
            10_240,
            2_560,
            stride_x_row,
            stride_gate_row,
            stride_output_row,
        )
        is expected
    )


def test_hc_gate_mix_passes_wide_offset_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    row_block = _KernelRecorder()
    profile = SimpleNamespace(
        supports=lambda _: True,
        qwen4exp_hc_gate_mix_row_block_min_rows=2048,
    )
    monkeypatch.setattr(hc, "_hc_gate_mix_row_block_kernel", row_block)
    monkeypatch.setattr(hc, "_hc_gate_mix_needs_wide_offsets", lambda *_: True)
    monkeypatch.setattr(hc, "get_current_hardware_profile", lambda: profile)

    x = torch.empty((2048, 40), dtype=torch.bfloat16)
    hc._hc_gate_mix(x, torch.empty_like(x), 4)

    _, _, kwargs = row_block.launches[0]
    assert kwargs["WIDE_OFFSETS"] is True


@pytest.mark.parametrize(
    ("rows", "row_wise_min_rows", "weight_shared", "expected_kernel"),
    (
        (511, 512, False, "legacy"),
        (512, 512, False, "row_wise"),
        (512, None, False, "legacy"),
        (4096, 512, True, "legacy"),
    ),
)
def test_hc_combine_norm_row_wise_dispatch(
    rows: int,
    row_wise_min_rows: int | None,
    weight_shared: bool,
    expected_kernel: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = _KernelRecorder()
    row_wise = _KernelRecorder()
    profile = SimpleNamespace(qwen4exp_hc_combine_norm_row_wise_min_rows=row_wise_min_rows)
    monkeypatch.setattr(hc, "_hc_combine_norm_kernel", legacy)
    monkeypatch.setattr(hc, "_hc_combine_norm_row_wise_kernel", row_wise)
    monkeypatch.setattr(hc, "get_current_hardware_profile", lambda: profile)

    residual = torch.empty((rows, 40), dtype=torch.bfloat16)
    block_output = torch.empty((rows, 10), dtype=torch.bfloat16)
    injection = torch.empty((rows, 4), dtype=torch.bfloat16)
    norm_weight = torch.empty(10 if weight_shared else 40, dtype=torch.bfloat16)
    combined, normalized = hc._hc_combine_norm(residual, block_output, injection, norm_weight, 1e-6, 4)

    assert combined.shape == residual.shape
    assert normalized.shape == residual.shape
    expected_recorder, other_recorder = (row_wise, legacy) if expected_kernel == "row_wise" else (legacy, row_wise)
    assert len(expected_recorder.launches) == 1
    assert other_recorder.launches == []
    grid, args, kwargs = expected_recorder.launches[0]
    assert grid == ((rows,) if expected_kernel == "row_wise" else (rows * 4,))
    assert args[2] is injection
    assert kwargs["HAS_INJECTION"] is True
    assert kwargs["WEIGHT_SHARED"] is weight_shared
