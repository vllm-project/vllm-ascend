# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The E2E observer must reject graph or runtime-allocation replacement."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.e2e.common.rlhf.weight_transfer.reload_probe import (
    ReloadProbeWorkerExtension,
    assert_reload_reuses_runtime,
    reload_snapshots,
)


@pytest.fixture
def observer():
    module = torch.nn.Linear(2, 2, bias=False)
    module.w13_weight_list = [torch.ones(2)]
    module._reload_runtime_buffers = {"projection": torch.ones(2)}
    manager = SimpleNamespace(graphs={"decode": object()}, capture=MagicMock(), run_fullgraph=MagicMock())
    extension = ReloadProbeWorkerExtension()
    extension.model_runner = SimpleNamespace(model=module, cudagraph_manager=manager)
    return extension, module, manager


def test_probe_accepts_in_place_refill_and_observes_replay(observer):
    extension, module, manager = observer
    before = extension.direct_reload_snapshot()
    with torch.no_grad():
        module.weight.fill_(3)
        module.w13_weight_list[0].fill_(4)
        module._reload_runtime_buffers["projection"].fill_(5)
    manager.run_fullgraph("decode")
    after = extension.direct_reload_snapshot()
    assert_reload_reuses_runtime([before], [after], execution_mode="graph")


@pytest.mark.parametrize("replacement", ["parameter", "runtime", "list", "graph", "capture", "no_replay"])
def test_probe_rejects_replacement_recapture_or_missing_replay(observer, replacement):
    extension, module, manager = observer
    before = extension.direct_reload_snapshot()
    if replacement == "parameter":
        module.weight = torch.nn.Parameter(module.weight.detach().clone())
    elif replacement == "runtime":
        module._reload_runtime_buffers["projection"] = torch.ones(2)
    elif replacement == "list":
        module.w13_weight_list = list(module.w13_weight_list)
    elif replacement == "graph":
        manager.graphs["decode"] = object()
    elif replacement == "capture":
        manager.capture()
    if replacement != "no_replay":
        manager.run_fullgraph("decode")
    after = extension.direct_reload_snapshot()
    with pytest.raises(AssertionError):
        assert_reload_reuses_runtime([before], [after], execution_mode="graph")


def test_eager_probe_does_not_require_graph_replay(observer):
    extension, _, manager = observer
    manager.graphs.clear()
    before = extension.direct_reload_snapshot()
    after = extension.direct_reload_snapshot()
    assert_reload_reuses_runtime([before], [after], execution_mode="eager")


def test_probe_reports_worker_direct_mode_and_recorded_checkpoint(observer):
    extension, module, _ = observer
    module._ckpt_param_names = {"weight"}
    with patch("vllm.model_executor.model_loader.reload.direct.direct_reload_enabled", return_value=True):
        assert extension.direct_reload_configuration() == {"enabled": True, "checkpoint_tensor_count": 1}


@pytest.mark.parametrize(
    "configuration",
    [{"enabled": False, "checkpoint_tensor_count": 1}, {"enabled": True, "checkpoint_tensor_count": 0}],
)
def test_snapshot_refuses_disabled_direct_or_missing_record(configuration):
    response = MagicMock()
    response.json.return_value = {"results": [configuration]}
    with (
        patch("tests.e2e.common.rlhf.weight_transfer.reload_probe.post_control", return_value=response),
        pytest.raises(AssertionError),
    ):
        reload_snapshots(MagicMock(), execution_mode="eager")
