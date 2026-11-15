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
    ("rows", "row_block_supported", "expected_kernel"),
    (
        (1023, True, "legacy"),
        (1024, True, "row_block"),
        (1025, True, "row_block"),
        (1024, False, "legacy"),
    ),
)
def test_hc_gate_mix_row_block_dispatch_is_scoped_by_capability(
    rows: int,
    row_block_supported: bool,
    expected_kernel: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = _KernelRecorder()
    row_block = _KernelRecorder()
    requested_capabilities: list[HardwareCapability] = []

    def supports(capability: HardwareCapability) -> bool:
        requested_capabilities.append(capability)
        return row_block_supported

    profile = SimpleNamespace(supports=supports)
    monkeypatch.setattr(hc, "_hc_gate_mix_kernel", legacy)
    monkeypatch.setattr(hc, "_hc_gate_mix_row_block_kernel", row_block)
    monkeypatch.setattr(hc, "get_current_hardware_profile", lambda: profile)

    x = torch.empty((rows, 40), dtype=torch.bfloat16)
    gate = torch.empty_like(x)
    output = hc._hc_gate_mix(x, gate, 4)

    assert output.shape == (rows, 10)
    if rows < hc._HC_GATE_MIX_ROW_BLOCK_MIN_ROWS:
        assert requested_capabilities == []
    else:
        assert requested_capabilities == [HardwareCapability.QWEN4EXP_HC_GATE_MIX_ROW_BLOCK]

    expected_recorder, other_recorder = (row_block, legacy) if expected_kernel == "row_block" else (legacy, row_block)
    assert len(expected_recorder.launches) == 1
    assert other_recorder.launches == []
    grid, _, kwargs = expected_recorder.launches[0]
    if expected_kernel == "row_block":
        assert grid == ((rows + hc._HC_GATE_MIX_ROWS_PER_BLOCK - 1) // hc._HC_GATE_MIX_ROWS_PER_BLOCK, 1)
        assert kwargs["EVEN_ROWS"] is (rows % hc._HC_GATE_MIX_ROWS_PER_BLOCK == 0)
        assert kwargs["EVEN_BLOCK"] is False
    else:
        assert grid == (rows, 1)
