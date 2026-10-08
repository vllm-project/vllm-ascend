# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run production device selection, stopping before any NPU initialization."""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


class DeviceSelected(Exception):
    pass


@pytest.fixture
def bind_device(monkeypatch):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/worker.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    worker = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUWorker")
    method = next(node for node in worker.body if isinstance(node, ast.FunctionDef) and node.name == "_init_device")
    assigned = []
    interface = ModuleType("vllm.platforms.interface")
    interface.set_assigned_physical_gpu_ids = lambda ids: assigned.__setitem__(slice(None), ids)
    monkeypatch.setitem(sys.modules, interface.__name__, interface)

    def selected(device):
        raise DeviceSelected(device)

    scope = {
        "torch": SimpleNamespace(
            npu=SimpleNamespace(is_available=lambda: True, device_count=lambda: 16, set_device=selected),
            device=lambda name: name,
        ),
        "current_platform": SimpleNamespace(
            device_type="npu",
            logical_device_id_to_visible_device_id=lambda index: assigned[index] if assigned else index,
        ),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)

    def bind(local_rank, config):
        obj = SimpleNamespace(local_rank=local_rank, parallel_config=config)
        assigned.clear()
        with pytest.raises(DeviceSelected) as selection:
            scope["_init_device"](obj)
        return int(selection.value.args[0].split(":")[1])

    return bind


def topology(dp_rank, width, **overrides):
    values = dict(
        distributed_executor_backend="mp",
        data_parallel_backend="mp",
        nnodes_within_dp=1,
        assigned_physical_gpu_ids=None,
        data_parallel_rank_local=dp_rank,
        data_parallel_index=dp_rank,
        world_size=width,
        local_world_size=width,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("width,dp", [(4, 4), (8, 2), (16, 1)])
def test_all_dp_replicas_bind_disjoint_devices(bind_device, width, dp):
    # Width 4: TP4/PCP1 baseline; width 8: TP4/PCP2 (or TP2/PP2/PCP2).
    groups = [[bind_device(local, topology(rank, width)) for local in range(width)] for rank in range(dp)]
    assert [device for group in groups for device in group] == list(range(16))


def test_dp_local_rank_takes_precedence_over_global_rank(bind_device):
    assert bind_device(3, topology(1, 8, data_parallel_index=7)) == 11


def test_fallback_uses_global_dp_index_when_local_rank_unspecified(bind_device):
    assert bind_device(7, topology(None, 8, data_parallel_index=1)) == 15


@pytest.mark.parametrize("backend", ["ray", "external_launcher"])
def test_external_worker_assignment_is_not_shifted_twice(bind_device, backend):
    assert bind_device(11, topology(1, 8, distributed_executor_backend=backend)) == 11


def test_ray_dp_assignment_is_not_shifted_twice(bind_device):
    assert bind_device(11, topology(1, 8, data_parallel_backend="ray")) == 11


def test_multinode_replica_preserves_node_local_rank(bind_device):
    assert bind_device(3, topology(1, 16, nnodes_within_dp=2)) == 3


def test_explicit_physical_device_assignment_is_not_shifted(bind_device):
    config = topology(1, 8, assigned_physical_gpu_ids=[15, 13, 11, 9, 7, 5, 3, 1])
    assert bind_device(0, config) == 15
    assert bind_device(7, config) == 1


def test_excess_replica_size_fails_before_device_initialization(bind_device):
    with pytest.raises(AssertionError, match="out of bounds"):
        bind_device(0, topology(2, 8))
