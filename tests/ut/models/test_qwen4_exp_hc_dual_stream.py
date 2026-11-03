# SPDX-License-Identifier: Apache-2.0
"""Submission ordering for the optional Qwen4Exp HC injection stream."""

from contextlib import contextmanager
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
        return "input_ready"

    def wait_event(self, event: object) -> None:
        self.log.append(f"wait_event:{self.name}:{event}")

    def wait_stream(self, stream: "_Stream") -> None:
        self.log.append(f"wait_stream:{self.name}:{stream.name}")


def test_injection_launch_waits_for_input_and_main_gate_does_not_wait() -> None:
    log: list[str] = []
    main = _Stream("main", log)
    auxiliary = _Stream("aux", log)
    x = _Tensor("input", log)
    weight = _Tensor("weight", log)
    injection = _Tensor("injection", log)
    lowrank = _Tensor("lowrank", log)

    @contextmanager
    def switch_stream(stream: _Stream):
        log.append(f"enter:{stream.name}")
        yield
        log.append(f"exit:{stream.name}")

    def linear(input_tensor: _Tensor, sliced_weight: _Tensor) -> _Tensor:
        del input_tensor, sliced_weight
        if "exit:aux" not in log:
            log.append("linear:injection")
            return injection
        log.append("linear:lowrank")
        return lowrank

    with (
        patch.object(hc.torch.npu, "current_stream", return_value=main),
        patch.object(hc.torch.npu, "stream", side_effect=switch_stream),
        patch.object(hc, "qwen4_exp_hc_injection_stream", return_value=auxiliary),
        patch.object(hc.F, "linear", side_effect=linear),
    ):
        assert hc._split_down_projection(x, weight, 320, 4) == (lowrank, injection)

    assert log == [
        "event:main",
        "enter:aux",
        "wait_event:aux:input_ready",
        "slice:weight:320:324",
        "linear:injection",
        "exit:aux",
        "record_stream:input:aux",
        "record_stream:injection:main",
        "slice:weight:None:320",
        "linear:lowrank",
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
        lowrank, injection = hc._split_down_projection_fake(torch.empty((rows, 10240)), weight, 320, 4)
        assert lowrank.shape == (rows, 320)
        assert injection.shape == (rows, 4)
