# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from vllm_ascend.ops.triton.v2 import apply_grammar_bitmask as grammar


@pytest.fixture
def launcher(monkeypatch):
    events = []
    stream = Mock()
    stream.synchronize.side_effect = lambda: events.append("sync")
    npu = Mock()
    npu.is_current_stream_capturing.return_value = False
    npu.current_stream.return_value = stream
    monkeypatch.setattr(grammar, "torch", SimpleNamespace(npu=npu))
    monkeypatch.setattr(grammar, "init_device_properties_triton", Mock())
    monkeypatch.setattr(grammar, "get_vectorcore_num", lambda: 48)
    result = object()

    def submit(*args, **kwargs):
        events.append("launch")
        return result

    kernel = MagicMock()
    kernel.__getitem__.return_value.side_effect = submit
    monkeypatch.setattr(grammar, "_apply_grammar_bitmask_kernel_impl", kernel)
    launch = grammar._ApplyGrammarBitmaskKernelLauncher()[(3, 20)]
    args = (object(), 163840, object(), object(), object(), 5120, 163840)
    return SimpleNamespace(
        events=events,
        stream=stream,
        npu=npu,
        kernel=kernel,
        launch=launch,
        args=args,
        result=result,
    )


def test_eager_launch_is_bracketed_by_stream_synchronization(launcher):
    result = launcher.launch(*launcher.args, MASK_STRIDE=8, BLOCK_SIZE=8192)

    assert result is launcher.result
    assert launcher.events == ["sync", "launch", "sync"]
    launcher.npu.current_stream.assert_called_once_with()
    launcher.kernel.__getitem__.assert_called_once_with((48,))
    launcher.kernel.__getitem__.return_value.assert_called_once_with(
        *launcher.args,
        60,
        NUM_PROGRAMS=48,
        NUM_VOCAB_BLOCKS=20,
        MASK_STRIDE=8,
        BLOCK_SIZE=8192,
        multibuffer=False,
    )


def test_graph_capture_keeps_existing_launch_path(launcher):
    launcher.npu.is_current_stream_capturing.return_value = True

    result = launcher.launch(*launcher.args, MASK_STRIDE=8, BLOCK_SIZE=8192)

    assert result is launcher.result
    assert launcher.events == ["launch"]
    launcher.npu.current_stream.assert_not_called()


def test_single_program_launch_does_not_synchronize(launcher, monkeypatch):
    monkeypatch.setattr(grammar, "get_vectorcore_num", lambda: 1)
    launch = grammar._ApplyGrammarBitmaskKernelLauncher()[(3, 20)]

    result = launch(*launcher.args, MASK_STRIDE=8, BLOCK_SIZE=8192)

    assert result is launcher.result
    assert launcher.events == ["launch"]
    launcher.npu.is_current_stream_capturing.assert_not_called()
    launcher.npu.current_stream.assert_not_called()
