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

    NPUWorker._acl_rt_lib = None
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
        parallel_config=SimpleNamespace(data_parallel_master_ip="1.1.1.1"),
    )
    worker.distributed_init_method = "tcp://127.0.0.1:29500"
    return worker


def test_get_acl_rt_lib_uses_cached_instance(npu_worker_cls):
    cached = MagicMock()
    npu_worker_cls._acl_rt_lib = cached

    assert npu_worker_cls._get_acl_rt_lib() is cached


@pytest.mark.parametrize(
    ("method_name", "api_name"),
    [
        pytest.param("snapshot_process_lock", "aclrtSnapShotProcessLock", id="lock"),
        pytest.param("snapshot_process_backup", "aclrtSnapShotProcessBackup", id="backup"),
        pytest.param("snapshot_process_restore", "aclrtSnapShotProcessRestore", id="restore"),
        pytest.param("snapshot_process_unlock", "aclrtSnapShotProcessUnlock", id="unlock"),
    ],
)
def test_aclrt_snapshot_wrappers_call_expected_api(worker, method_name, api_name):
    with patch.object(worker, "_call_aclrt_snapshot_api") as mock_call:
        getattr(worker, method_name)()

    mock_call.assert_called_once_with(api_name)


def test_snapshot_suspend_runs_npu_snapshot_sequence(worker):
    with (
        patch.object(worker, "dump_model") as dump_model,
        patch("vllm_ascend.worker.worker.gc.collect") as collect,
        patch.object(worker, "snapshot_process_lock") as process_lock,
        patch.object(worker, "snapshot_process_backup") as process_backup,
    ):
        worker.suspend("/tmp/model")

    dump_model.assert_called_once_with("/tmp/model")
    collect.assert_called_once_with()
    process_lock.assert_called_once_with()
    process_backup.assert_called_once_with()


def test_snapshot_resume_runs_npu_restore_phases(worker):
    with (
        patch.object(worker, "snapshot_process_restore") as process_restore,
        patch.object(worker, "snapshot_process_unlock") as process_unlock,
        patch.object(worker, "update_worker_info_after_resume") as update_worker,
        patch.object(worker, "rebuild_parallel_group_after_resume") as rebuild_parallel,
        patch.object(worker, "re_load_weights") as reload_weights,
        patch.object(worker, "recapture_graph") as recapture_graph,
        patch.object(worker, "rebuild_kv_transfer_engine_after_resume") as rebuild_kv,
    ):
        worker.resume("10.0.0.2", "10.0.0.3", "/tmp/model", "engine-id")

    process_restore.assert_called_once_with()
    process_unlock.assert_called_once_with()
    update_worker.assert_called_once_with("10.0.0.2", "10.0.0.3")
    rebuild_parallel.assert_called_once_with()
    reload_weights.assert_called_once_with("/tmp/model")
    recapture_graph.assert_called_once_with()
    rebuild_kv.assert_called_once_with("10.0.0.2", "engine-id")


def test_call_aclrt_snapshot_api_invokes_aclrt_library(worker):
    mock_api = MagicMock(return_value=0)
    mock_lib = MagicMock()
    mock_lib.aclrtSnapShotProcessLock = mock_api

    with patch.object(worker, "_get_acl_rt_lib", return_value=mock_lib):
        worker._call_aclrt_snapshot_api("aclrtSnapShotProcessLock")

    mock_api.assert_called_once()


def test_dump_model_delegates_to_model_runner(worker):
    worker.dump_model(model_save_path="/tmp/model")

    worker.model_runner.dump_model.assert_called_once_with(path="/tmp/model")


def test_re_load_weights_delegates_to_model_runner(worker):
    worker.re_load_weights(model_path="/tmp/model")

    worker.model_runner.restore_model.assert_called_once_with(path="/tmp/model")


def test_parallel_group_clean_up_destroys_parallel_and_dist_env(worker):
    with (
        patch("vllm_ascend.worker.worker.destroy_ascend_model_parallel") as mock_destroy,
        patch("vllm_ascend.worker.worker.cleanup_dist_env_for_snapshot") as mock_cleanup,
    ):
        worker.parallel_group_clean_up()

    mock_destroy.assert_called_once()
    mock_cleanup.assert_called_once()


def test_rebuild_parallel_group_after_resume_updates_init_method(worker):
    worker.vllm_config.parallel_config.data_parallel_master_ip = "10.0.0.1"
    tp_group = object()
    dp_group = object()
    ep_group = object()
    mc2_group = object()
    moe_config = SimpleNamespace(ep_size=2)
    dispatcher = SimpleNamespace(refresh_hccl_group=MagicMock())
    comm_method = SimpleNamespace(moe_config=moe_config, token_dispatcher=dispatcher)

    with (
        patch("torch.distributed.set_debug_level"),
        patch.object(worker, "parallel_group_clean_up"),
        patch.object(worker, "_init_worker_distributed_environment") as mock_init,
        patch("vllm_ascend.worker.worker.set_current_vllm_config") as mock_ctx,
        patch.dict(
            "vllm_ascend.ops.fused_moe.moe_comm_method._MoECommMethods",
            {"fused_mc2": comm_method},
            clear=True,
        ),
        patch("vllm_ascend.worker.worker.get_tp_group", return_value=tp_group),
        patch("vllm.distributed.parallel_state.get_dp_group", return_value=dp_group),
        patch("vllm.distributed.parallel_state.get_ep_group", return_value=ep_group),
        patch("vllm_ascend.distributed.parallel_state.get_mc2_group", return_value=mc2_group),
    ):
        mock_ctx.return_value.__enter__ = MagicMock()
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
        calls = []
        mock_init.side_effect = lambda: calls.append("init")
        worker.rebuild_parallel_group_after_resume()

    assert worker.distributed_init_method == "tcp://10.0.0.1:29501"
    mock_init.assert_called_once()
    assert calls == ["init"]
    assert moe_config.tp_group is tp_group
    assert moe_config.dp_group is dp_group
    assert moe_config.ep_group is ep_group
    assert moe_config.mc2_group is mc2_group
    dispatcher.refresh_hccl_group.assert_called_once_with()


def test_update_worker_info_after_resume_updates_env_and_master_ip(worker, monkeypatch):
    monkeypatch.delenv("HCCL_IF_IP", raising=False)
    worker.update_worker_info_after_resume("10.0.0.8", "10.0.0.9")

    assert os.environ["HCCL_IF_IP"] == "10.0.0.8"
    assert worker.vllm_config.parallel_config.data_parallel_master_ip == "10.0.0.9"


def test_rebuild_kv_transfer_engine_after_resume_delegates_to_connector(worker):
    rebuild = MagicMock()
    connector_worker = SimpleNamespace(rebuild_kv_transfer_endpoint=rebuild)
    kv_group = SimpleNamespace(connector_worker=connector_worker)

    with (
        patch("vllm_ascend.worker.worker.has_kv_transfer_group", return_value=True),
        patch("vllm_ascend.worker.worker.get_kv_transfer_group", return_value=kv_group),
    ):
        worker.rebuild_kv_transfer_engine_after_resume("10.0.0.8", None)

    rebuild.assert_called_once_with("10.0.0.8", None)


def test_rebuild_kv_transfer_engine_after_resume_delegates_to_hybrid_connector(worker):
    worker.vllm_config.kv_transfer_config.kv_connector = "MooncakeHybridConnector"
    rebuild = MagicMock()
    connector_worker = SimpleNamespace(rebuild_kv_transfer_endpoint=rebuild)
    kv_group = SimpleNamespace(connector_worker=connector_worker)

    with (
        patch("vllm_ascend.worker.worker.has_kv_transfer_group", return_value=True),
        patch("vllm_ascend.worker.worker.get_kv_transfer_group", return_value=kv_group),
    ):
        worker.rebuild_kv_transfer_engine_after_resume("10.0.0.8", None)

    rebuild.assert_called_once_with("10.0.0.8", None)


def test_recapture_graph_clears_and_recaptures(worker):
    with (
        patch("vllm_ascend.compilation.acl_graph.clear_all_aclgraph_entries") as mock_clear_entries,
        patch("vllm_ascend.compilation.acl_graph.clear_graph_params_for_recapture") as mock_clear_params,
    ):
        worker.recapture_graph()

    mock_clear_entries.assert_called_once()
    mock_clear_params.assert_called_once()
    worker.model_runner.capture_model.assert_called_once()
