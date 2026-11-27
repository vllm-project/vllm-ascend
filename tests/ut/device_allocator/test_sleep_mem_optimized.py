# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Exercise the real sleep managers with device/control-plane dependencies mocked.

The isolated loader also allows these tests to run without installing vLLM or
CANN, using --confcutdir=tests/ut/device_allocator on a CPU-only workstation.
"""

import sys
from contextlib import nullcontext
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call

import pytest


@pytest.fixture
def env(monkeypatch):
    torch = ModuleType("torch")
    torch.npu = SimpleNamespace(
        synchronize=MagicMock(),
        current_device=MagicMock(return_value=0),
        mem_get_info=MagicMock(return_value=(100, 200)),
    )
    dist = ModuleType("torch.distributed")
    dist.is_available = MagicMock(return_value=True)
    dist.is_initialized = MagicMock(return_value=True)
    dist.get_world_size = MagicMock(return_value=2)
    dist.barrier = MagicMock()
    dist.all_gather_object = MagicMock(
        side_effect=lambda output, value, **kw: output.__setitem__(slice(None), [value] * len(output))
    )
    torch.distributed = dist
    parallel = ModuleType("vllm.distributed.parallel_state")
    business = SimpleNamespace(
        unique_name="tp:0",
        backend="hccl",
        group_ranks=[[0, 1]],
        device_group=object(),
        _hccl_sleep_adapter=SimpleNamespace(validate=MagicMock()),
    )
    parallel._groups = {"tp:0": lambda: business}
    world = SimpleNamespace(cpu_group=object(), local_rank=0)
    parallel.get_world_group = MagicMock(return_value=world)
    parallel.suspend_device_comms = MagicMock()
    parallel.resume_device_comms = MagicMock()

    def create_anchor(*args, **kwargs):
        anchor = SimpleNamespace(device_group=object())
        anchor.restore_hccl = MagicMock(side_effect=lambda: setattr(anchor, "device_group", object()) or True)
        anchor.destroy_hccl = MagicMock(side_effect=lambda: setattr(anchor, "device_group", None) or True)
        return anchor

    parallel.init_model_parallel_group = MagicMock(side_effect=create_anchor)
    vllm = ModuleType("vllm")
    distributed = ModuleType("vllm.distributed")
    distributed.parallel_state = parallel
    vllm.distributed = distributed
    config = ModuleType("vllm.config")
    config.VllmConfig = object
    config.set_current_vllm_config = lambda _: nullcontext()
    logger = ModuleType("vllm.logger")
    logger.logger = MagicMock()
    constants = ModuleType("vllm.utils.mem_constants")
    constants.GiB_bytes = 1024**3
    graph = ModuleType("vllm_ascend.compilation.acl_graph")
    graph._graph_params = graph._draft_graph_params = graph._draft_graph_prefill_params = None
    graph._acl_graph_wrappers = []
    compilation = ModuleType("vllm_ascend.compilation")
    compilation.acl_graph = graph
    utils = ModuleType("vllm_ascend.utils")
    utils.SLEEP_LIFECYCLE_ANCHOR_GROUP_NAME = "sleep_lifecycle_anchor"
    moe = ModuleType("vllm_ascend.ops.fused_moe.moe_comm_method")
    moe._MoECommMethods = {}
    modules = {
        "torch": torch,
        "torch.distributed": dist,
        "vllm": vllm,
        "vllm.config": config,
        "vllm.distributed": distributed,
        "vllm.distributed.parallel_state": parallel,
        "vllm.logger": logger,
        "vllm.utils.mem_constants": constants,
        "vllm_ascend.compilation": compilation,
        "vllm_ascend.compilation.acl_graph": graph,
        "vllm_ascend.utils": utils,
        "vllm_ascend.ops.fused_moe.moe_comm_method": moe,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/device_allocator/sleep_mem_optimized.py"
    spec = spec_from_file_location("_sleep_managers_under_test", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return SimpleNamespace(module=module, torch=torch, dist=dist, parallel=parallel, graph=graph, moe=moe)


@pytest.fixture
def manager(env):
    runner = SimpleNamespace(use_aclgraph=True, capture_model=MagicMock(), cudagraph_manager=None)
    worker = SimpleNamespace(_pp_send_work=[])
    result = env.module.SleepWakeupManager(object(), worker, lambda: runner)
    result.runner = runner
    return result


@dataclass
class DummyGraphParams:
    events: dict
    workspaces: dict
    metadata: dict


def test_graph_reset_preserves_metadata_but_invalidates_entries(env, manager):
    workspace = object()
    params = DummyGraphParams({1: ["event"]}, {1: workspace}, {1: ("keep",)})
    env.graph._graph_params = params
    wrapper = SimpleNamespace(concrete_aclgraph_entries={"old": object()}, first_run_finished=True)
    env.graph._acl_graph_wrappers = [wrapper]
    graph_manager = SimpleNamespace(graphs={"old": object()}, _graphs_captured=True)
    manager.runner.cudagraph_manager = graph_manager
    manager.sleep()
    assert params.events == {1: []}
    assert params.workspaces == {1: None}
    assert params.metadata == {1: ("keep",)}
    assert wrapper.concrete_aclgraph_entries == {}
    assert wrapper.first_run_finished is False
    assert graph_manager.graphs == {}
    assert graph_manager._graphs_captured is False
    manager.wakeup()
    manager.runner.capture_model.assert_called_once_with()


def test_sleep_orders_drain_anchor_graph_moe_and_upstream_contract(env, manager):
    order = []
    handle = MagicMock()
    handle.wait.side_effect = lambda: order.append("drain")
    manager.hccl.worker._pp_send_work = [handle]
    env.torch.npu.synchronize.side_effect = lambda: order.append("sync")
    env.dist.barrier.side_effect = lambda **kw: order.append("anchor")
    manager.acl_graph.sleep = MagicMock(side_effect=lambda: order.append("graph"))
    manager.hccl.prepare_moe_hccl_teardown = MagicMock(side_effect=lambda: order.append("moe") or True)
    env.parallel.suspend_device_comms.side_effect = lambda: order.append("suspend")
    manager.sleep()
    assert order == ["drain", "sync", "anchor", "graph", "moe", "suspend"]
    assert manager.hccl.worker._pp_send_work == []


def test_wake_orders_all_pg_restore_moe_graph_and_anchor(env, manager):
    order = []
    manager.sleep()
    env.parallel.resume_device_comms.side_effect = lambda: order.append("resume")
    manager.hccl.refresh_moe_hccl_groups = MagicMock(side_effect=lambda: order.append("dispatcher"))
    manager.hccl.refresh_moe_hccl_runtime_state = MagicMock(side_effect=lambda: order.append("runtime"))
    manager.runner.capture_model.side_effect = lambda: order.append("capture")
    env.dist.barrier.side_effect = lambda **kw: order.append("anchor")
    manager.wakeup()
    assert order == ["resume", "dispatcher", "runtime", "capture", "anchor"]


def test_two_cycles_reuse_anchor_and_recapture_once_per_cycle(env, manager):
    for _ in range(2):
        manager.sleep()
        manager.sleep()
        manager.wakeup()
        manager.wakeup()
    env.parallel.init_model_parallel_group.assert_called_once_with(
        [[0, 1]], 0, "hccl", use_device_communicator=False, group_name="sleep_lifecycle_anchor"
    )
    assert env.parallel.suspend_device_comms.call_count == 2
    assert env.parallel.resume_device_comms.call_count == 2
    assert manager.runner.capture_model.call_count == 2
    manager.hccl._lifecycle_anchor_group.restore_hccl.assert_called_once_with()
    assert manager.hccl._lifecycle_anchor_group.destroy_hccl.call_count == 2


def test_staged_wake_keeps_anchor_and_delays_capture(env, manager):
    manager.sleep()
    anchor = manager.hccl._lifecycle_anchor_group
    manager.wakeup(["weights"])
    manager.runner.capture_model.assert_not_called()
    anchor.destroy_hccl.assert_not_called()
    manager.wakeup(["weights"])
    manager.wakeup(["kv_cache"])
    manager.wakeup(["kv_cache"])
    manager.runner.capture_model.assert_called_once_with()
    anchor.destroy_hccl.assert_called_once_with()
    env.parallel.resume_device_comms.assert_called_once_with()


def test_cannot_sleep_again_halfway_through_staged_wake(env, manager):
    manager.sleep()
    manager.wakeup(["weights"])
    with pytest.raises(RuntimeError, match="Finish the staged"):
        manager.validate_sleep()
    manager.wakeup(["kv_cache"])
    assert manager.validate_sleep() is True


def test_kv_first_is_rejected_before_recovery_but_correct_order_can_retry(env, manager):
    manager.sleep()
    with pytest.raises(ValueError, match="Wake weights"):
        manager.validate_wakeup(["kv_cache"])
    env.parallel.resume_device_comms.assert_not_called()
    manager.wakeup(["weights"])
    manager.wakeup(["kv_cache"])
    manager.runner.capture_model.assert_called_once_with()


def test_wake_both_tags_is_complete(env, manager):
    manager.sleep()
    manager.wakeup(["weights", "kv_cache"])
    assert manager._sleeping is False
    manager.runner.capture_model.assert_called_once_with()


def test_eager_never_captures_graph(env, manager):
    manager.runner.use_aclgraph = False
    manager.acl_graph.sleep = MagicMock()
    manager.sleep()
    manager.wakeup()
    manager.acl_graph.sleep.assert_not_called()
    manager.runner.capture_model.assert_not_called()
    env.parallel.resume_device_comms.assert_called_once_with()


def test_tp_one_does_not_use_native_suspend_or_create_anchor(env, manager):
    env.dist.get_world_size.return_value = 1
    manager.sleep()
    manager.wakeup()
    env.parallel.init_model_parallel_group.assert_not_called()
    env.parallel.suspend_device_comms.assert_not_called()
    env.parallel.resume_device_comms.assert_not_called()
    manager.runner.capture_model.assert_not_called()


def test_uninitialized_distributed_does_not_invoke_collectives(env, manager):
    env.dist.is_initialized.return_value = False
    manager.sleep()
    manager.wakeup()
    env.dist.all_gather_object.assert_not_called()
    env.dist.barrier.assert_not_called()
    manager.runner.capture_model.assert_not_called()


def test_graph_rebuild_decision_is_recomputed_each_cycle(env, manager):
    for world_size, captures in ((1, 0), (2, 1), (1, 1)):
        env.dist.get_world_size.return_value = world_size
        manager.sleep()
        manager.wakeup()
        assert manager.runner.capture_model.call_count == captures
        assert manager.acl_graph._needs_recapture is False
    env.parallel.suspend_device_comms.assert_called_once_with()
    env.parallel.resume_device_comms.assert_called_once_with()


@pytest.mark.parametrize("path", ["single_rank", "uninitialized", "no_hccl", "anchor_only", "non_hccl"])
@pytest.mark.parametrize("staged", [False, True])
def test_no_hccl_teardown_preserves_graphs_and_workspaces_across_cycles(env, manager, path, staged):
    if path == "single_rank":
        env.dist.get_world_size.return_value = 1
    elif path == "uninitialized":
        env.dist.is_initialized.return_value = False
    elif path == "no_hccl":
        env.parallel._groups.clear()
    elif path == "anchor_only":
        anchor = SimpleNamespace(device_group=object())
        manager.hccl._lifecycle_anchor_group = anchor
        env.parallel._groups = {"sleep_lifecycle_anchor:0": lambda: anchor}
        # The module keeps the original group-registry reference.
        manager.hccl.iter_alive_group_coordinators = lambda: [anchor]
    else:
        env.parallel._groups["tp:0"]().backend = "gloo"

    workspace = object()
    events = [object()]
    params = DummyGraphParams({1: events}, {1: workspace}, {1: ("keep",)})
    env.graph._graph_params = params
    entry = object()
    wrapper = SimpleNamespace(concrete_aclgraph_entries={"keep": entry}, first_run_finished=True)
    env.graph._acl_graph_wrappers = [wrapper]
    runner_graph = object()
    graph_manager = SimpleNamespace(graphs={"keep": runner_graph}, _graphs_captured=True)
    manager.runner.cudagraph_manager = graph_manager
    # No business teardown also means no graph-workspace cleanup or MoE work.
    clear_workspaces = MagicMock(wraps=manager.acl_graph.clear_all_attention_workspaces)
    manager.acl_graph.clear_all_attention_workspaces = clear_workspaces
    manager.hccl.prepare_moe_hccl_teardown = MagicMock()
    manager.hccl.release_lifecycle_anchor = MagicMock()
    for _ in range(2):
        manager.sleep()
        assert manager.acl_graph._needs_recapture is False
        if staged:
            manager.wakeup(["weights"])
            manager.wakeup(["kv_cache"])
        else:
            manager.wakeup()
        assert wrapper.concrete_aclgraph_entries == {"keep": entry}
        assert wrapper.first_run_finished is True
        assert params.workspaces[1] is workspace
        assert params.events[1] is events
        assert graph_manager.graphs == {"keep": runner_graph}
        assert graph_manager._graphs_captured is True
    clear_workspaces.assert_not_called()
    manager.runner.capture_model.assert_not_called()
    manager.hccl.prepare_moe_hccl_teardown.assert_not_called()
    env.parallel.suspend_device_comms.assert_not_called()
    env.parallel.resume_device_comms.assert_not_called()
    env.parallel.init_model_parallel_group.assert_not_called()


@pytest.mark.parametrize("failure", ["resume", "dispatcher", "runtime", "capture"])
def test_wake_failure_propagates_and_never_releases_anchor(env, manager, failure):
    manager.sleep()
    anchor = manager.hccl._lifecycle_anchor_group
    callbacks = {
        "resume": env.parallel.resume_device_comms,
        "dispatcher": MagicMock(),
        "runtime": MagicMock(),
        "capture": manager.runner.capture_model,
    }
    manager.hccl.refresh_moe_hccl_groups = callbacks["dispatcher"]
    manager.hccl.refresh_moe_hccl_runtime_state = callbacks["runtime"]
    callbacks[failure].side_effect = RuntimeError("wake failed")
    with pytest.raises(RuntimeError, match="wake failed"):
        manager.wakeup()
    anchor.destroy_hccl.assert_not_called()
    with pytest.raises(RuntimeError, match="restart"):
        manager.wakeup()
    if failure != "capture":
        manager.runner.capture_model.assert_not_called()
    else:
        assert manager.acl_graph._needs_recapture is True


@pytest.mark.parametrize("failure", ["anchor", "moe", "graph", "suspend"])
def test_sleep_failure_propagates_and_does_not_claim_asleep(env, manager, failure):
    if failure == "anchor":
        manager.hccl._ensure_lifecycle_anchor = MagicMock(return_value=False)
    elif failure == "moe":
        manager.hccl.prepare_moe_hccl_teardown = MagicMock(return_value=False)
    elif failure == "graph":
        manager.acl_graph.sleep = MagicMock(side_effect=RuntimeError("graph failed"))
    else:
        env.parallel.suspend_device_comms.side_effect = RuntimeError("suspend failed")
    with pytest.raises(RuntimeError):
        manager.sleep()
    assert manager._sleeping is False
    with pytest.raises(RuntimeError, match="restart"):
        manager.sleep()
    if failure != "suspend":
        env.parallel.suspend_device_comms.assert_not_called()


@pytest.mark.parametrize(
    "failure", ["mismatch", "remote_error", "missing_adapter", "pg_state_mismatch", "missing_live_pg"]
)
def test_preflight_refuses_inconsistent_or_unsupported_pg_owners(env, manager, failure):
    if failure == "missing_adapter":
        group = SimpleNamespace(unique_name="tp:0", backend="hccl", group_ranks=[[0, 1]])
        env.parallel._groups["tp:0"] = lambda: group
    elif failure == "missing_live_pg":
        env.parallel._groups["tp:0"]().device_group = None
    elif failure == "mismatch":
        env.dist.all_gather_object.side_effect = lambda out, value, **kw: out.__setitem__(
            slice(None), [value, (["other"], None)]
        )
    elif failure == "remote_error":
        env.dist.all_gather_object.side_effect = lambda out, value, **kw: out.__setitem__(
            slice(None), [value, (value[0], "remote unavailable")]
        )
    else:
        env.dist.all_gather_object.side_effect = lambda out, value, **kw: out.__setitem__(
            slice(None), [value, ([("tp:0", "hccl", [[0, 1]], False)], None)]
        )
    with pytest.raises(RuntimeError, match="preflight"):
        manager.sleep()
    env.parallel.suspend_device_comms.assert_not_called()
    env.parallel.init_model_parallel_group.assert_not_called()


def test_preflight_validates_registered_owners_over_cpu_group(env, manager):
    adapter = SimpleNamespace(validate=MagicMock())
    group = SimpleNamespace(
        unique_name="tp:0", backend="hccl", group_ranks=[[0, 1]], device_group=object(), _hccl_sleep_adapter=adapter
    )
    env.parallel._groups["tp:0"] = lambda: group
    env.parallel._groups["expired"] = lambda: None
    manager.sleep()
    assert adapter.validate.call_count == 2
    expected = ([("tp:0", "hccl", [[0, 1]], True)], None)
    assert env.dist.all_gather_object.call_args.args[1] == expected
    assert env.dist.all_gather_object.call_args.kwargs["group"] is env.parallel.get_world_group().cpu_group


def test_moe_hooks_are_preserved(env, manager):
    method = SimpleNamespace(
        prepare_hccl_teardown=MagicMock(),
        token_dispatcher=SimpleNamespace(refresh_hccl_group=MagicMock()),
        refresh_hccl_runtime_state=MagicMock(),
    )
    env.moe._MoECommMethods["mc2"] = method
    manager.sleep()
    manager.wakeup()
    method.prepare_hccl_teardown.assert_called_once_with()
    method.token_dispatcher.refresh_hccl_group.assert_called_once_with()
    method.refresh_hccl_runtime_state.assert_called_once_with()


def test_moe_prepare_exception_refuses_teardown_on_all_ranks(env, manager):
    env.moe._MoECommMethods["mc2"] = SimpleNamespace(
        prepare_hccl_teardown=MagicMock(side_effect=RuntimeError("no update_group"))
    )
    with pytest.raises(RuntimeError, match="preflight"):
        manager.sleep()
    env.parallel.suspend_device_comms.assert_not_called()


@pytest.mark.parametrize("phase", ["prepare", "dispatcher", "runtime"])
def test_moe_explicit_failure_is_not_treated_as_success(env, manager, phase):
    method = SimpleNamespace(
        prepare_hccl_teardown=MagicMock(return_value=True),
        token_dispatcher=SimpleNamespace(refresh_hccl_group=MagicMock()),
        refresh_hccl_runtime_state=MagicMock(return_value=True),
    )
    env.moe._MoECommMethods["mc2"] = method
    if phase == "prepare":
        method.prepare_hccl_teardown.return_value = False
        with pytest.raises(RuntimeError, match="preflight"):
            manager.sleep()
        env.parallel.suspend_device_comms.assert_not_called()
    else:
        manager.sleep()
        hook = (
            method.token_dispatcher.refresh_hccl_group if phase == "dispatcher" else method.refresh_hccl_runtime_state
        )
        hook.return_value = False
        with pytest.raises(RuntimeError, match="refused"):
            manager.wakeup()
        manager.hccl._lifecycle_anchor_group.destroy_hccl.assert_not_called()


def test_graph_wake_without_sleep_is_noop(env, manager):
    manager.acl_graph.wakeup()
    manager.wakeup()
    manager.runner.capture_model.assert_not_called()
    env.parallel.resume_device_comms.assert_not_called()


def test_graph_reset_only_clears_list_fields(env):
    workspace = object()
    params = DummyGraphParams({1: ["event"]}, {1: workspace}, {1: ("keep",)})
    env.module.AclGraphSleepWakeupManager.reset_graph_params(params)
    assert params.events == {1: []}
    assert params.workspaces == {1: workspace}
    assert params.metadata == {1: ("keep",)}


def test_anchor_initialization_failure_is_logged_and_propagated(env, manager):
    env.parallel.init_model_parallel_group.side_effect = RuntimeError("new_group failed")
    with pytest.raises(RuntimeError, match="anchor"):
        manager.sleep()
    assert manager.hccl._lifecycle_anchor_group is None
    env.parallel.suspend_device_comms.assert_not_called()


def test_anchor_barrier_uses_only_its_live_group(env, manager):
    manager.sleep()
    anchor = manager.hccl._lifecycle_anchor_group
    old_pg = anchor.device_group
    manager.wakeup()
    assert env.dist.barrier.call_args_list == [call(group=old_pg, device_ids=[0]), call(group=old_pg, device_ids=[0])]
