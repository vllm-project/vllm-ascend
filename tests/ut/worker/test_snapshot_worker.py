# SPDX-License-Identifier: Apache-2.0

import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def npu_worker_cls():
    torch_npu = sys.modules.setdefault("torch_npu", types.ModuleType("torch_npu"))
    atb_ops = types.ModuleType("torch_npu.op_plugin.atb._atb_ops")
    atb_ops._register_atb_extensions = MagicMock()  # type: ignore[attr-defined]
    op_plugin = types.ModuleType("torch_npu.op_plugin")
    op_plugin.atb = types.SimpleNamespace(_atb_ops=atb_ops)  # type: ignore[attr-defined]
    profiler = types.ModuleType("torch_npu.profiler")
    profiler.dynamic_profile = MagicMock()  # type: ignore[attr-defined]
    sys.modules["torch_npu.op_plugin"] = op_plugin
    sys.modules["torch_npu.op_plugin.atb"] = op_plugin.atb  # type: ignore[assignment]
    sys.modules["torch_npu.op_plugin.atb._atb_ops"] = atb_ops
    sys.modules["torch_npu.profiler"] = profiler
    torch_npu.op_plugin = op_plugin  # type: ignore[attr-defined]
    torch_npu.profiler = profiler  # type: ignore[attr-defined]

    sys.modules.pop("vllm_ascend.worker.worker", None)
    from vllm_ascend.worker.worker import NPUWorker

    return NPUWorker


@pytest.fixture
def worker(npu_worker_cls):
    with patch.object(npu_worker_cls, "__init__", lambda self, **kwargs: None):
        worker = npu_worker_cls()
    worker.rank = 0
    worker.model_runner = MagicMock()
    worker.model_config = SimpleNamespace(enforce_eager=False)
    worker.vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector="MooncakeLayerwiseConnector",
            is_kv_producer=False,
            is_kv_consumer=True,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_master_ip="1.1.1.1",
            _snapshot_data_parallel_port_list=[29502],
        ),
        snapshot_config=object(),
    )
    worker.distributed_init_method = "tcp://127.0.0.1:29500"
    return worker


def test_get_acl_rt_lib_uses_cached_instance():
    from vllm_ascend.snapshot import worker_lifecycle as snapshot_worker

    cached = MagicMock()
    snapshot_worker._ACL_RT_LIB = cached

    assert snapshot_worker._get_acl_rt_lib() is cached


def test_worker_snapshot_methods_delegate(worker):
    with (
        patch("vllm_ascend.worker.worker.suspend_worker") as suspend,
        patch("vllm_ascend.worker.worker.resume_worker") as resume,
        patch("vllm_ascend.worker.worker.unlock_worker") as unlock,
    ):
        worker.suspend("/tmp/model")
        worker.resume("10.0.0.2", "10.0.0.3", "/tmp/model", "engine-id")
        worker.device_unlock()

    suspend.assert_called_once_with(worker, "/tmp/model")
    resume.assert_called_once_with(worker, "10.0.0.2", "10.0.0.3", "/tmp/model", "engine-id")
    unlock.assert_called_once_with(worker)


def test_snapshot_suspend_runs_npu_snapshot_sequence(worker):
    with (
        patch("vllm_ascend.snapshot.worker_lifecycle.gc.collect") as collect,
        patch("vllm_ascend.snapshot.worker_lifecycle._call_aclrt_snapshot_api") as call_aclrt,
        patch("vllm_ascend.snapshot.worker_lifecycle.dump_model_runner") as dump_model,
    ):
        from vllm_ascend.snapshot.worker_lifecycle import suspend_worker

        suspend_worker(worker, "/tmp/model")

    dump_model.assert_called_once_with(worker.model_runner, "/tmp/model")
    collect.assert_called_once_with()
    assert [call.args[1] for call in call_aclrt.call_args_list] == [
        "aclrtSnapShotProcessLock",
        "aclrtSnapShotProcessBackup",
    ]


def test_snapshot_resume_runs_npu_restore_phases(worker):
    with (
        patch("vllm_ascend.snapshot.worker_lifecycle._call_aclrt_snapshot_api") as call_aclrt,
        patch("vllm_ascend.snapshot.worker_lifecycle._update_worker_info") as update_worker,
        patch("vllm_ascend.snapshot.worker_lifecycle._rebuild_parallel_groups") as rebuild_parallel,
        patch("vllm_ascend.snapshot.worker_lifecycle.restore_model_runner") as restore_model,
        patch("vllm_ascend.snapshot.worker_lifecycle._recapture_graph") as recapture_graph,
        patch("vllm_ascend.snapshot.worker_lifecycle._rebuild_kv_transfer_engine") as rebuild_kv,
    ):
        from vllm_ascend.snapshot.worker_lifecycle import resume_worker

        resume_worker(worker, "10.0.0.2", "10.0.0.3", "/tmp/model", "engine-id")

    assert [call.args[1] for call in call_aclrt.call_args_list] == [
        "aclrtSnapShotProcessRestore",
        "aclrtSnapShotProcessUnlock",
    ]
    update_worker.assert_called_once_with(worker, "10.0.0.2", "10.0.0.3")
    rebuild_parallel.assert_called_once_with(worker)
    restore_model.assert_called_once_with(worker.model_runner, "/tmp/model")
    recapture_graph.assert_called_once_with(worker)
    rebuild_kv.assert_called_once_with(worker, "10.0.0.2", "engine-id")


def test_call_aclrt_snapshot_api_invokes_aclrt_library(worker):
    from vllm_ascend.snapshot.worker_lifecycle import _call_aclrt_snapshot_api

    mock_api = MagicMock(return_value=0)
    mock_lib = MagicMock()
    mock_lib.aclrtSnapShotProcessLock = mock_api

    with patch("vllm_ascend.snapshot.worker_lifecycle._get_acl_rt_lib", return_value=mock_lib):
        _call_aclrt_snapshot_api(worker, "aclrtSnapShotProcessLock")

    mock_api.assert_called_once()


@pytest.mark.parametrize("snapshot_config", [object(), None])
def test_parallel_group_clean_up_destroys_parallel_and_dist_env(worker, snapshot_config):
    from vllm_ascend.snapshot.worker_lifecycle import _parallel_group_cleanup

    worker.vllm_config.snapshot_config = snapshot_config
    with (
        patch("vllm_ascend.snapshot.worker_lifecycle.destroy_ascend_model_parallel") as mock_destroy,
        patch("vllm_ascend.snapshot.worker_lifecycle.cleanup_dist_env_for_snapshot") as mock_cleanup,
        patch("vllm_ascend.snapshot.worker_lifecycle.snapshot_hccl_teardown") as teardown,
    ):
        _parallel_group_cleanup(worker)

    teardown.assert_called_once_with(snapshot_config is not None)
    mock_destroy.assert_called_once()
    mock_cleanup.assert_called_once()


def test_rebuild_parallel_group_after_resume_updates_init_method(worker):
    from vllm_ascend.snapshot.worker_lifecycle import _rebuild_parallel_groups

    worker.vllm_config.parallel_config.data_parallel_master_ip = "10.0.0.1"
    tp_group = object()
    dp_group = object()
    ep_group = object()
    mc2_group = object()
    moe_config = SimpleNamespace(ep_size=2)
    dispatcher = SimpleNamespace(
        refresh_hccl_group=MagicMock(),
        reset_snapshot_runtime_state=MagicMock(),
    )
    comm_method = SimpleNamespace(moe_config=moe_config, token_dispatcher=dispatcher)

    with (
        patch("torch.distributed.set_debug_level"),
        patch("vllm_ascend.snapshot.worker_lifecycle._parallel_group_cleanup"),
        patch.object(worker, "_init_worker_distributed_environment") as mock_init,
        patch("vllm_ascend.snapshot.worker_lifecycle.set_current_vllm_config") as mock_ctx,
        patch.dict(
            "vllm_ascend.ops.fused_moe.moe_comm_method._MoECommMethods",
            {"fused_mc2": comm_method},
            clear=True,
        ),
        patch("vllm_ascend.snapshot.worker_lifecycle.get_tp_group", return_value=tp_group),
        patch("vllm.distributed.parallel_state.get_dp_group", return_value=dp_group),
        patch("vllm.distributed.parallel_state.get_ep_group", return_value=ep_group),
        patch("vllm_ascend.distributed.parallel_state.get_mc2_group", return_value=mc2_group),
    ):
        mock_ctx.return_value.__enter__ = MagicMock()
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
        calls = []
        mock_init.side_effect = lambda: calls.append("init")
        _rebuild_parallel_groups(worker)

    assert worker.distributed_init_method == "tcp://10.0.0.1:29502"
    assert worker.vllm_config.parallel_config._snapshot_data_parallel_port_list == [29502]
    mock_init.assert_called_once_with()
    assert calls == ["init"]
    assert moe_config.tp_group is tp_group
    assert moe_config.dp_group is dp_group
    assert moe_config.ep_group is ep_group
    assert moe_config.mc2_group is mc2_group
    dispatcher.refresh_hccl_group.assert_called_once_with()
    dispatcher.reset_snapshot_runtime_state.assert_called_once_with()


def test_update_worker_info_after_resume_updates_env_and_master_ip(worker, monkeypatch):
    from vllm_ascend.snapshot.worker_lifecycle import _update_worker_info

    monkeypatch.delenv("HCCL_IF_IP", raising=False)
    _update_worker_info(worker, "10.0.0.8", "10.0.0.9")

    assert os.environ["HCCL_IF_IP"] == "10.0.0.8"
    assert worker.vllm_config.parallel_config.data_parallel_master_ip == "10.0.0.9"


def test_rebuild_kv_transfer_engine_after_resume_delegates_to_connector(worker):
    from vllm_ascend.snapshot.worker_lifecycle import _rebuild_kv_transfer_engine

    rebuild = MagicMock()
    connector_worker = SimpleNamespace(rebuild_kv_transfer_endpoint=rebuild)
    kv_group = SimpleNamespace(connector_worker=connector_worker)

    with (
        patch("vllm_ascend.snapshot.worker_lifecycle.has_kv_transfer_group", return_value=True),
        patch("vllm_ascend.snapshot.worker_lifecycle.get_kv_transfer_group", return_value=kv_group),
    ):
        _rebuild_kv_transfer_engine(worker, "10.0.0.8", None)

    rebuild.assert_called_once_with("10.0.0.8", None)


def test_rebuild_kv_transfer_engine_after_resume_delegates_to_hybrid_connector(worker):
    from vllm_ascend.snapshot.worker_lifecycle import _rebuild_kv_transfer_engine

    worker.vllm_config.kv_transfer_config.kv_connector = "MooncakeHybridConnector"
    rebuild = MagicMock()
    connector_worker = SimpleNamespace(rebuild_kv_transfer_endpoint=rebuild)
    kv_group = SimpleNamespace(connector_worker=connector_worker)

    with (
        patch("vllm_ascend.snapshot.worker_lifecycle.has_kv_transfer_group", return_value=True),
        patch("vllm_ascend.snapshot.worker_lifecycle.get_kv_transfer_group", return_value=kv_group),
    ):
        _rebuild_kv_transfer_engine(worker, "10.0.0.8", None)

    rebuild.assert_called_once_with("10.0.0.8", None)


def test_recapture_graph_clears_and_recaptures(worker):
    from vllm_ascend.snapshot.worker_lifecycle import _recapture_graph

    with (
        patch("vllm_ascend.compilation.acl_graph.clear_all_aclgraph_entries") as mock_clear_entries,
        patch("vllm_ascend.compilation.acl_graph.clear_graph_params_for_recapture") as mock_clear_params,
        patch("vllm_ascend.snapshot.worker_lifecycle.restore_drafter_runtime_buffers") as restore_drafter,
        patch("vllm_ascend.snapshot.worker_lifecycle._warm_up_atb"),
    ):
        _recapture_graph(worker)

    mock_clear_entries.assert_called_once()
    mock_clear_params.assert_called_once()
    worker.model_runner.capture_model.assert_called_once()
    restore_drafter.assert_called_once_with(worker.model_runner)
