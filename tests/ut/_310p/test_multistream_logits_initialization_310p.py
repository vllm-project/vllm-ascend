# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from vllm_ascend._310p.worker.v2 import model_runner as model_runner_310p
from vllm_ascend._310p.worker.v2.model_runner import NPUModelRunner310V2
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


def _runner(*, multistream=True, tp_size=4, full=True, graph_count=2):
    runner = object.__new__(NPUModelRunner310V2)
    runner.ascend_config = SimpleNamespace(multistream_overlap_shared_expert=multistream)
    runner.parallel_config = SimpleNamespace(tensor_parallel_size=tp_size)
    runner.cudagraph_manager = SimpleNamespace(
        cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: full),
        graphs={i: object() for i in range(graph_count)},
    )
    runner._multistream_logits_initialized = False
    return runner


@pytest.mark.parametrize(
    "options",
    [dict(multistream=False), dict(tp_size=1), dict(full=False), dict(graph_count=0), dict(graph_count=1)],
)
def test_logits_sync_skips_unaffected_paths(options):
    runner = _runner(**options)
    with (
        patch.object(model_runner_310p.torch.npu, "current_stream") as current_stream,
        patch("vllm_ascend._310p.worker.v2.model_runner.shared_experts_calculation_stream") as shared_stream,
    ):
        runner._synchronize_multistream_moe_before_logits()
    current_stream.assert_not_called()
    shared_stream.assert_not_called()
    assert not runner._multistream_logits_initialized


@pytest.mark.parametrize("missing_attribute", [False, True])
def test_logits_sync_skips_uninitialized_graph_manager(missing_attribute):
    runner = _runner()
    if missing_attribute:
        del runner.cudagraph_manager
    else:
        runner.cudagraph_manager = None
    with (
        patch.object(model_runner_310p.torch.npu, "current_stream") as current_stream,
        patch("vllm_ascend._310p.worker.v2.model_runner.shared_experts_calculation_stream") as shared_stream,
    ):
        runner._synchronize_multistream_moe_before_logits()
    current_stream.assert_not_called()
    shared_stream.assert_not_called()
    assert not runner._multistream_logits_initialized


def test_logits_sync_drains_both_streams_only_before_first_real_sample():
    runner = _runner()
    calls = MagicMock()
    with (
        patch.object(model_runner_310p.torch.npu, "current_stream", return_value=calls.main),
        patch("vllm_ascend._310p.worker.v2.model_runner.shared_experts_calculation_stream", return_value=calls.shared),
    ):
        for _ in range(10):
            runner._synchronize_multistream_moe_before_logits()
    assert calls.mock_calls == [
        call.main.synchronize(),
        call.shared.synchronize(),
    ]
    assert runner._multistream_logits_initialized


def test_logits_sync_failure_does_not_disable_protection():
    runner = _runner()
    with (
        patch.object(model_runner_310p.torch.npu, "current_stream") as current_stream,
        patch("vllm_ascend._310p.worker.v2.model_runner.shared_experts_calculation_stream") as shared_stream,
    ):
        shared_stream.return_value.synchronize.side_effect = RuntimeError("stream failure")
        with pytest.raises(RuntimeError, match="stream failure"):
            runner._synchronize_multistream_moe_before_logits()
        assert not runner._multistream_logits_initialized
        shared_stream.return_value.synchronize.side_effect = None
        runner._synchronize_multistream_moe_before_logits()
        assert current_stream.return_value.synchronize.call_count == 2
        assert shared_stream.return_value.synchronize.call_count == 2


@pytest.mark.parametrize("multistream", [False, True])
def test_load_model_initializes_shared_stream_after_parent(multistream):
    runner = _runner(multistream=multistream)
    events = []
    with (
        patch.object(NPUModelRunner, "load_model", side_effect=lambda *a, **kw: events.append("loaded")) as load,
        patch(
            "vllm_ascend._310p.worker.v2.model_runner.shared_experts_calculation_stream",
            side_effect=lambda: events.append("stream"),
        ),
    ):
        runner.load_model(load_dummy_weights=True)
    load.assert_called_once_with(True)
    assert events == (["loaded", "stream"] if multistream else ["loaded"])


def test_logits_sync_main_failure_is_retryable():
    runner = _runner()
    with (
        patch.object(model_runner_310p.torch.npu, "current_stream") as main,
        patch("vllm_ascend._310p.worker.v2.model_runner.shared_experts_calculation_stream") as shared,
    ):
        main.return_value.synchronize.side_effect = RuntimeError("main failure")
        with pytest.raises(RuntimeError, match="main failure"):
            runner._synchronize_multistream_moe_before_logits()
        shared.assert_not_called()
        assert not runner._multistream_logits_initialized
        main.return_value.synchronize.side_effect = None
        runner._synchronize_multistream_moe_before_logits()
        assert runner._multistream_logits_initialized
