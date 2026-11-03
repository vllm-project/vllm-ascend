# SPDX-License-Identifier: Apache-2.0
"""Submission ordering for the optional Qwen4Exp HC injection stream."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.models.qwen4_exp import hc_dual_stream as hc


class _Tensor:
    def __init__(self, name: str, log: list[str]) -> None:
        self.name = name
        self.log = log

    def __getitem__(self, index: slice) -> "_Tensor":
        self.log.append(f"slice:{self.name}:{index.start}:{index.stop}")
        return self

    def record_stream(self, stream: object) -> None:
        self.log.append(f"record_stream:{self.name}:{stream.name}")


class _Stream:
    def __init__(self, name: str, log: list[str]) -> None:
        self.name = name
        self.log = log

    def record_event(self) -> object:
        self.log.append(f"event:{self.name}")
        return "gate_ready"

    def wait_event(self, event: object) -> None:
        self.log.append(f"wait_event:{self.name}:{event}")

    def wait_stream(self, stream: "_Stream") -> None:
        self.log.append(f"wait_stream:{self.name}:{stream.name}")


def test_gate_is_queued_before_injection() -> None:
    log: list[str] = []
    xn = SimpleNamespace(device=SimpleNamespace(type="npu"))
    weight = _Tensor("weight", log)
    mixer = SimpleNamespace(
        use_combine=True,
        dual_stream=True,
        input_mix_weight_down_block_inject=SimpleNamespace(weight=weight),
        input_mix_weight_up=lambda value: log.append("up") or value,
        hc_lowrank=320,
        hc_count=4,
    )

    with (
        patch.object(hc.F, "linear", side_effect=lambda *args: log.append("lowrank") or xn),
        patch.object(hc, "hc_silu", side_effect=lambda *args: log.append("silu") or xn),
        patch.object(hc, "hc_gate_mix", side_effect=lambda *args: log.append("gate_mix") or xn),
        patch.object(
            hc.torch.ops.vllm,
            "qwen4_exp_hc_launch_injection_projection",
            side_effect=lambda *args: log.append("injection") or xn,
        ),
    ):
        hc.AscendGatedResidual._project_from_normalized(mixer, xn)

    assert log == ["slice:weight:None:320", "lowrank", "silu", "up", "gate_mix", "injection"]


def test_injection_launch_waits_for_queued_gate_and_main_does_not_wait() -> None:
    log: list[str] = []
    main = _Stream("main", log)
    auxiliary = _Stream("aux", log)
    x = _Tensor("input", log)
    weight = _Tensor("weight", log)
    injection = _Tensor("injection", log)

    @contextmanager
    def switch_stream(stream: _Stream):
        log.append(f"enter:{stream.name}")
        yield
        log.append(f"exit:{stream.name}")

    def linear(input_tensor: _Tensor, sliced_weight: _Tensor) -> _Tensor:
        del input_tensor, sliced_weight
        log.append("linear:injection")
        return injection

    with (
        patch.object(hc.torch.npu, "current_stream", return_value=main),
        patch.object(hc.torch.npu, "stream", side_effect=switch_stream),
        patch.object(hc, "qwen4_exp_hc_injection_stream", return_value=auxiliary),
        patch.object(hc.F, "linear", side_effect=linear),
    ):
        assert hc._launch_injection_projection(x, weight, 320, 4) is injection

    assert log == [
        "event:main",
        "enter:aux",
        "wait_event:aux:gate_ready",
        "slice:weight:320:324",
        "linear:injection",
        "exit:aux",
        "record_stream:input:aux",
        "record_stream:injection:main",
    ]


def test_combine_waits_for_injection_before_reading_output() -> None:
    log: list[str] = []
    main = _Stream("main", log)
    auxiliary = _Stream("aux", log)
    with (
        patch.object(hc.torch.npu, "current_stream", return_value=main),
        patch.object(hc, "qwen4_exp_hc_injection_stream", return_value=auxiliary),
        patch.object(hc, "_hc_combine", side_effect=lambda *args: log.append("combine")),
        patch.object(hc, "_hc_combine_norm", side_effect=lambda *args: log.append("combine_norm")),
    ):
        hc._combine_after_injection(None, None, None, 4)
        hc._combine_norm_after_injection(None, None, None, None, 1e-6, 4)
    assert log == [
        "wait_stream:main:aux",
        "combine",
        "wait_stream:main:aux",
        "combine_norm",
    ]


def test_fake_projection_preserves_empty_and_decode_batch_shapes() -> None:
    weight = torch.empty((336, 10240))
    for rows in (0, 1, 48):
        injection = hc._launch_injection_projection_fake(torch.empty((rows, 10240)), weight, 320, 4)
        assert injection.shape == (rows, 4)
