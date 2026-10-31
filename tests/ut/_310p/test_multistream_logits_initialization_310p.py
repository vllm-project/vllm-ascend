# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend._310p.worker.v2 import model_runner as model_runner_310p
from vllm_ascend._310p.worker.v2.model_runner import NPUModelRunner310V2
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


def _runner(*, multistream=True, tp_size=4, full=True, sizes=None):
    runner = object.__new__(NPUModelRunner310V2)
    runner.ascend_config = SimpleNamespace(multistream_overlap_shared_expert=multistream)
    runner.parallel_config = SimpleNamespace(tensor_parallel_size=tp_size)
    runner.compilation_config = SimpleNamespace(
        cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: full),
        cudagraph_capture_sizes=sizes,
    )
    runner.model = MagicMock()
    return runner


@pytest.mark.parametrize("options", [dict(multistream=False), dict(tp_size=1), dict(full=False)])
def test_logits_warmup_skips_unaffected_paths(options):
    runner = _runner(**options)
    hidden = MagicMock(shape=(8, 16))
    with patch.object(NPUModelRunner, "_dummy_sampler_run") as parent:
        runner._dummy_sampler_run(hidden)
    parent.assert_called_once_with(hidden)
    runner.model.compute_logits.assert_not_called()


@pytest.mark.parametrize(
    "sizes,profile_rows,expected",
    [
        ([4, 2, 2, 0, -1, 8, 16], 8, [1, 2, 4]),
        ([2, 4], 8, [1, 2, 4]),
        ([2], 8, [1, 2]),
        (None, 8, [1]),
        ([], 8, [1]),
        ([1, 2], 1, []),
    ],
)
def test_logits_warmup_profiles_unique_small_batches_after_parent(sizes, profile_rows, expected):
    runner = _runner(sizes=sizes)
    hidden = MagicMock(shape=(profile_rows, 16))
    hidden.__getitem__.side_effect = lambda index: SimpleNamespace(rows=index.stop)
    events = []
    runner.model.compute_logits.side_effect = lambda batch: events.append(batch.rows)
    with (
        patch.object(NPUModelRunner, "_dummy_sampler_run", side_effect=lambda x: events.append("parent")) as parent,
        patch.object(model_runner_310p.torch.npu, "current_stream") as current,
        patch.object(model_runner_310p, "shared_experts_calculation_stream") as shared,
    ):
        runner._dummy_sampler_run(hidden)
    parent.assert_called_once_with(hidden)
    assert events == ["parent", *expected]
    assert [entry.args[0] for entry in hidden.__getitem__.call_args_list] == [slice(None, n) for n in expected]
    current.assert_not_called()
    shared.assert_not_called()


def test_logits_warmup_does_not_hide_parent_failure():
    runner = _runner(sizes=[1, 2])
    with (
        patch.object(NPUModelRunner, "_dummy_sampler_run", side_effect=RuntimeError("profile failure")),
        pytest.raises(RuntimeError, match="profile failure"),
    ):
        runner._dummy_sampler_run(MagicMock(shape=(8, 16)))
    runner.model.compute_logits.assert_not_called()


def test_logits_warmup_does_not_hide_communication_failure():
    runner = _runner(sizes=[1, 2])
    runner.model.compute_logits.side_effect = RuntimeError("communication failure")
    with (
        patch.object(NPUModelRunner, "_dummy_sampler_run"),
        pytest.raises(RuntimeError, match="communication failure"),
    ):
        runner._dummy_sampler_run(MagicMock(shape=(8, 16)))
    assert runner.model.compute_logits.call_count == 1


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
