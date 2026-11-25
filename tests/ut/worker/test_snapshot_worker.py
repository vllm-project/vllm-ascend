# SPDX-License-Identifier: Apache-2.0

import os
import sys
import types
from collections import defaultdict
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
        use_v2_model_runner=True,
        kv_transfer_config=SimpleNamespace(
            kv_connector="MooncakeLayerwiseConnector",
            is_kv_producer=False,
            is_kv_consumer=True,
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
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
        patch("vllm_ascend.snapshot.worker_lifecycle.suspend_worker") as suspend,
        patch("vllm_ascend.snapshot.worker_lifecycle.resume_worker") as resume,
        patch("vllm_ascend.snapshot.worker_lifecycle.unlock_worker") as unlock,
    ):
        worker.suspend("/tmp/model")
        worker.resume("10.0.0.3", "/tmp/model", "engine-id")
        worker.device_unlock()

    suspend.assert_called_once_with(worker, "/tmp/model")
    resume.assert_called_once_with(worker, "10.0.0.3", "/tmp/model", "engine-id")
    unlock.assert_called_once_with(worker)


def test_snapshot_suspend_runs_npu_snapshot_sequence(worker):
    events = []
    with (
        patch(
            "vllm_ascend.snapshot.worker_lifecycle.dump_model_runner",
            side_effect=lambda *_: events.append("dump"),
        ) as dump_model,
        patch(
            "vllm_ascend.snapshot.worker_lifecycle.torch.npu.synchronize",
            side_effect=lambda: events.append("synchronize"),
        ) as synchronize,
        patch(
            "vllm_ascend.snapshot.worker_lifecycle.gc.collect",
            side_effect=lambda: events.append("gc"),
        ) as collect,
        patch(
            "vllm_ascend.snapshot.worker_lifecycle._call_aclrt_snapshot_api",
            side_effect=lambda _, api: events.append(api),
        ) as call_aclrt,
    ):
        from vllm_ascend.snapshot.worker_lifecycle import suspend_worker

        suspend_worker(worker, "/tmp/model")

    dump_model.assert_called_once_with(worker.model_runner, "/tmp/model")
    synchronize.assert_called_once_with()
    collect.assert_called_once_with()
    assert [call.args[1] for call in call_aclrt.call_args_list] == [
        "aclrtSnapShotProcessLock",
        "aclrtSnapShotProcessBackup",
    ]
    assert events == [
        "dump",
        "synchronize",
        "gc",
        "aclrtSnapShotProcessLock",
        "aclrtSnapShotProcessBackup",
    ]


def test_snapshot_suspend_stops_when_npu_synchronize_fails(worker):
    with (
        patch("vllm_ascend.snapshot.worker_lifecycle.dump_model_runner"),
        patch(
            "vllm_ascend.snapshot.worker_lifecycle.torch.npu.synchronize",
            side_effect=RuntimeError("synchronize failed"),
        ),
        patch("vllm_ascend.snapshot.worker_lifecycle.gc.collect") as collect,
        patch("vllm_ascend.snapshot.worker_lifecycle._call_aclrt_snapshot_api") as call_aclrt,
        pytest.raises(RuntimeError, match="synchronize failed"),
    ):
        from vllm_ascend.snapshot.worker_lifecycle import suspend_worker

        suspend_worker(worker, "/tmp/model")

    collect.assert_not_called()
    call_aclrt.assert_not_called()


def test_snapshot_resume_runs_npu_restore_phases(worker):
    with (
        patch("vllm_ascend.snapshot.worker_lifecycle.get_ip", return_value="10.0.0.2") as get_ip,
        patch("vllm_ascend.snapshot.worker_lifecycle._call_aclrt_snapshot_api") as call_aclrt,
        patch("vllm_ascend.snapshot.worker_lifecycle._reset_triton_kernel_caches") as reset_kernel_caches,
        patch("vllm_ascend.snapshot.worker_lifecycle._update_worker_info") as update_worker,
        patch("vllm_ascend.snapshot.worker_lifecycle._parallel_group_cleanup") as cleanup_parallel,
        patch("vllm_ascend.snapshot.worker_lifecycle._prepare_kv_transfer_for_snapshot_restore") as prepare_kv,
        patch("vllm_ascend.snapshot.worker_lifecycle._rebuild_parallel_groups") as rebuild_parallel,
        patch("vllm_ascend.snapshot.worker_lifecycle.restore_model_runner") as restore_model,
        patch("vllm_ascend.snapshot.worker_lifecycle._recapture_graph") as recapture_graph,
        patch("vllm_ascend.snapshot.worker_lifecycle._rebuild_kv_transfer_engine") as rebuild_kv,
    ):
        from vllm_ascend.snapshot.worker_lifecycle import resume_worker

        resume_worker(worker, "10.0.0.3", "/tmp/model", "engine-id")

    get_ip.assert_called_once_with(force=True)
    assert [call.args[1] for call in call_aclrt.call_args_list] == [
        "aclrtSnapShotProcessRestore",
        "aclrtSnapShotProcessUnlock",
    ]
    reset_kernel_caches.assert_called_once_with()
    update_worker.assert_called_once_with(worker, "10.0.0.2", "10.0.0.3")
    cleanup_parallel.assert_called_once_with(worker)
    prepare_kv.assert_called_once_with(worker)
    rebuild_parallel.assert_called_once_with(worker)
    restore_model.assert_called_once_with(worker.model_runner, "/tmp/model")
    recapture_graph.assert_called_once_with(worker)
    rebuild_kv.assert_called_once_with(worker, "10.0.0.2", "engine-id")


@pytest.mark.parametrize("cache_field", ["cache", "device_caches"])
def test_reset_triton_kernel_caches(cache_field):
    from vllm_ascend.snapshot.worker_lifecycle import _reset_triton_kernel_caches

    class FakeJITFunction:
        def __init__(self):
            setattr(self, cache_field, defaultdict(dict, {0: {"compiled": object()}}))

    class FakeAutotuner:
        def __init__(self):
            self.cache = MagicMock()
            self.fn = FakeJITFunction()

    class FakeHeuristics:
        def __init__(self):
            self.fn = FakeJITFunction()

    kernel = FakeJITFunction()
    tuned_kernel = FakeAutotuner()
    heuristic_kernel = FakeHeuristics()
    nested_kernel = FakeHeuristics()
    nested_kernel.fn = FakeAutotuner()
    upstream_module = types.ModuleType("vllm.test_snapshot_kernels")
    upstream_module.kernel = nested_kernel
    module = types.ModuleType("vllm_ascend.test_snapshot_kernels")
    module.kernel = kernel
    module.kernel_alias = kernel
    module.tuned_kernel = tuned_kernel
    module.heuristic_kernel = heuristic_kernel
    triton_runtime = types.ModuleType("triton.runtime")
    triton_runtime.JITFunction = FakeJITFunction
    triton_runtime.Autotuner = FakeAutotuner
    triton_runtime.Heuristics = FakeHeuristics
    jit_kernels = [kernel, tuned_kernel.fn, heuristic_kernel.fn, nested_kernel.fn.fn]
    caches = [getattr(item, cache_field) for item in jit_kernels]

    with patch.dict(
        sys.modules,
        {"triton.runtime": triton_runtime, module.__name__: module, upstream_module.__name__: upstream_module},
    ):
        _reset_triton_kernel_caches()
        _reset_triton_kernel_caches()

    for item, cache in zip(jit_kernels, caches):
        assert getattr(item, cache_field) is cache
        assert not cache
        assert cache[0] == {}  # Device cache can be initialized again after restore.
    assert tuned_kernel.cache.clear.call_count == 2
    assert nested_kernel.fn.cache.clear.call_count == 2


def test_reset_triton_kernel_caches_rejects_unknown_layout():
    from vllm_ascend.snapshot.worker_lifecycle import _reset_triton_kernel_caches

    class UnknownJITFunction:
        pass

    module = types.ModuleType("vllm_ascend.test_snapshot_kernels")
    module.kernel = UnknownJITFunction()
    triton_runtime = types.ModuleType("triton.runtime")
    triton_runtime.JITFunction = UnknownJITFunction
    triton_runtime.Autotuner = type("Autotuner", (), {})
    triton_runtime.Heuristics = type("Heuristics", (), {})
    with (
        patch.dict(sys.modules, {"triton.runtime": triton_runtime, module.__name__: module}),
        pytest.raises(RuntimeError, match="Unsupported Triton JIT cache layout"),
    ):
        _reset_triton_kernel_caches()


def test_call_aclrt_snapshot_api_invokes_aclrt_library(worker):
    from vllm_ascend.snapshot.worker_lifecycle import _call_aclrt_snapshot_api

    mock_api = MagicMock(return_value=0)
    mock_lib = MagicMock()
    mock_lib.aclrtSnapShotProcessLock = mock_api

    with patch("vllm_ascend.snapshot.worker_lifecycle._get_acl_rt_lib", return_value=mock_lib):
        _call_aclrt_snapshot_api(worker, "aclrtSnapShotProcessLock")

    mock_api.assert_called_once()


def test_call_aclrt_snapshot_api_raises_on_runtime_failure(worker):
    from vllm_ascend.snapshot.worker_lifecycle import _call_aclrt_snapshot_api

    mock_lib = MagicMock()
    mock_lib.aclrtSnapShotProcessLock = MagicMock(return_value=1)

    with (
        patch("vllm_ascend.snapshot.worker_lifecycle._get_acl_rt_lib", return_value=mock_lib),
        pytest.raises(RuntimeError, match="aclrtSnapShotProcessLock status=1"),
    ):
        _call_aclrt_snapshot_api(worker, "aclrtSnapShotProcessLock")


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


def test_rebuild_parallel_group_after_snapshot_restore_updates_init_method(worker):
    from vllm_ascend.snapshot.worker_lifecycle import _rebuild_parallel_groups

    worker.vllm_config.parallel_config.data_parallel_master_ip = "10.0.0.1"
    tp_group = object()
    dp_group = object()
    ep_group = object()
    mc2_group = object()
    moe_config = SimpleNamespace(ep_size=2)
    dispatcher = SimpleNamespace(
        refresh_hccl_group=MagicMock(),
        reset_runtime_state_after_snapshot_restore=MagicMock(),
    )
    comm_method = SimpleNamespace(moe_config=moe_config, token_dispatcher=dispatcher)

    with (
        patch("torch.distributed.set_debug_level"),
        patch("vllm_ascend.snapshot.worker_lifecycle._parallel_group_cleanup") as cleanup_parallel,
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
    cleanup_parallel.assert_not_called()
    assert calls == ["init"]
    assert moe_config.tp_group is tp_group
    assert moe_config.dp_group is dp_group
    assert moe_config.ep_group is ep_group
    assert moe_config.mc2_group is mc2_group
    dispatcher.refresh_hccl_group.assert_called_once_with()
    dispatcher.reset_runtime_state_after_snapshot_restore.assert_called_once_with()


def test_update_worker_info_after_snapshot_restore_updates_env_and_master_ip(worker, monkeypatch):
    from vllm_ascend.snapshot.worker_lifecycle import _update_worker_info

    monkeypatch.delenv("HCCL_IF_IP", raising=False)
    _update_worker_info(worker, "10.0.0.8", "10.0.0.9")

    assert os.environ["HCCL_IF_IP"] == "10.0.0.8"
    assert worker.vllm_config.parallel_config.data_parallel_master_ip == "10.0.0.9"


def test_rebuild_kv_transfer_engine_after_snapshot_restore_delegates_to_connector(worker):
    from vllm_ascend.snapshot.worker_lifecycle import _rebuild_kv_transfer_engine

    rebuild = MagicMock()
    kv_group = SimpleNamespace(rebuild_kv_transfer_endpoint=rebuild)

    with (
        patch("vllm_ascend.snapshot.worker_lifecycle.has_kv_transfer_group", return_value=True),
        patch("vllm_ascend.snapshot.worker_lifecycle.get_kv_transfer_group", return_value=kv_group),
    ):
        _rebuild_kv_transfer_engine(worker, "10.0.0.8", None)

    rebuild.assert_called_once_with("10.0.0.8", None)


def test_rebuild_kv_transfer_engine_after_snapshot_restore_delegates_to_hybrid_connector(worker):
    from vllm_ascend.snapshot.worker_lifecycle import _rebuild_kv_transfer_engine

    worker.vllm_config.kv_transfer_config.kv_connector = "MooncakeHybridConnector"
    rebuild = MagicMock()
    kv_group = SimpleNamespace(rebuild_kv_transfer_endpoint=rebuild)

    with (
        patch("vllm_ascend.snapshot.worker_lifecycle.has_kv_transfer_group", return_value=True),
        patch("vllm_ascend.snapshot.worker_lifecycle.get_kv_transfer_group", return_value=kv_group),
    ):
        _rebuild_kv_transfer_engine(worker, "10.0.0.8", None)

    rebuild.assert_called_once_with("10.0.0.8", None)


def test_recapture_graph_clears_and_recaptures(worker):
    from vllm_ascend.snapshot.worker_lifecycle import _recapture_graph

    calls = MagicMock()
    with (
        patch("vllm_ascend.compilation.acl_graph.clear_all_aclgraph_entries") as mock_clear_entries,
        patch("vllm_ascend.compilation.acl_graph.clear_graph_params_for_recapture") as mock_clear_params,
        patch("vllm_ascend.snapshot.model_runner_lifecycle.restore.reset_graph_managers") as reset_graph_managers,
    ):
        calls.attach_mock(mock_clear_entries, "clear_entries")
        calls.attach_mock(mock_clear_params, "clear_params")
        calls.attach_mock(reset_graph_managers, "reset_managers")
        calls.attach_mock(worker.model_runner.capture_model, "capture")
        _recapture_graph(worker)

    mock_clear_entries.assert_called_once()
    mock_clear_params.assert_called_once()
    reset_graph_managers.assert_called_once_with(worker.model_runner)
    assert [call[0] for call in calls.mock_calls] == [
        "clear_entries",
        "clear_params",
        "reset_managers",
        "capture",
    ]


def test_recapture_graph_eager_skips_manager_reset(worker):
    from vllm_ascend.snapshot.worker_lifecycle import _recapture_graph

    worker.model_config.enforce_eager = True
    with (
        patch("vllm_ascend.compilation.acl_graph.clear_all_aclgraph_entries") as clear_entries,
        patch("vllm_ascend.snapshot.model_runner_lifecycle.restore.reset_graph_managers") as reset,
    ):
        _recapture_graph(worker)
    clear_entries.assert_not_called()
    reset.assert_not_called()
    worker.model_runner.capture_model.assert_not_called()


def test_recapture_graph_v1_skips_v2_graph_manager_reset(worker):
    from vllm_ascend.snapshot.worker_lifecycle import _recapture_graph

    worker.vllm_config.use_v2_model_runner = False
    with (
        patch("vllm_ascend.compilation.acl_graph.clear_all_aclgraph_entries"),
        patch("vllm_ascend.compilation.acl_graph.clear_graph_params_for_recapture"),
        patch("vllm_ascend.snapshot.model_runner_lifecycle.restore.reset_graph_managers") as reset_graph_managers,
    ):
        _recapture_graph(worker)

    reset_graph_managers.assert_not_called()
    worker.model_runner.capture_model.assert_called_once_with()
