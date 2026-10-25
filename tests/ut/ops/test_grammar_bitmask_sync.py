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
        events=events, stream=stream, npu=npu, kernel=kernel, launch=launch, args=args, result=result
    )


@pytest.mark.parametrize("setting", [None, "0"])
def test_default_submission_remains_asynchronous(monkeypatch, launcher, setting):
    if setting is None:
        monkeypatch.delenv("VLLM_ASCEND_SYNCHRONIZE_GRAMMAR_BITMASK", raising=False)
    else:
        monkeypatch.setenv("VLLM_ASCEND_SYNCHRONIZE_GRAMMAR_BITMASK", setting)
    result = launcher.launch(*launcher.args, MASK_STRIDE=8, BLOCK_SIZE=8192)
    assert result is launcher.result
    assert launcher.events == ["launch"]
    launcher.npu.is_current_stream_capturing.assert_not_called()
    launcher.npu.current_stream.assert_not_called()


def test_synchronization_brackets_submission_and_preserves_arguments(monkeypatch, launcher):
    monkeypatch.setenv("VLLM_ASCEND_SYNCHRONIZE_GRAMMAR_BITMASK", "1")
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


def test_capture_is_rejected_before_sync_or_launch(monkeypatch, launcher):
    monkeypatch.setenv("VLLM_ASCEND_SYNCHRONIZE_GRAMMAR_BITMASK", "1")
    launcher.npu.is_current_stream_capturing.return_value = True
    with pytest.raises(RuntimeError, match="does not support graph capture"):
        launcher.launch(*launcher.args, MASK_STRIDE=8, BLOCK_SIZE=8192)
    assert launcher.events == []
    launcher.npu.current_stream.assert_not_called()
    launcher.kernel.__getitem__.assert_not_called()


@pytest.mark.parametrize("failure", ["before", "launch", "after"])
def test_errors_propagate_without_retry(monkeypatch, launcher, failure):
    monkeypatch.setenv("VLLM_ASCEND_SYNCHRONIZE_GRAMMAR_BITMASK", "1")
    error = RuntimeError("submission failed")
    sync_count = 0

    def synchronize():
        nonlocal sync_count
        sync_count += 1
        launcher.events.append("sync")
        if (failure == "before" and sync_count == 1) or (failure == "after" and sync_count == 2):
            raise error

    def submit(*args, **kwargs):
        launcher.events.append("launch")
        if failure == "launch":
            raise error

    launcher.stream.synchronize.side_effect = synchronize
    launcher.kernel.__getitem__.return_value.side_effect = submit
    with pytest.raises(RuntimeError) as caught:
        launcher.launch(*launcher.args, MASK_STRIDE=8, BLOCK_SIZE=8192)
    assert caught.value is error
    assert (
        launcher.events
        == {
            "before": ["sync"],
            "launch": ["sync", "launch"],
            "after": ["sync", "launch", "sync"],
        }[failure]
    )
