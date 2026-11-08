#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.device_allocator.sleep_mode_backend import SleepModeBackend, SleepModeBackendFactory

from vllm_ascend.device_allocator import sleep_mode_backend as backend_module
from vllm_ascend.distributed.kv_transfer.utils import mooncake_transfer_engine
from vllm_ascend.worker import worker as worker_module


@pytest.fixture
def calls():
    """One timeline for everything the sleep path touches."""
    return []


@pytest.fixture
def make_worker(monkeypatch, calls):
    """Build an ``__init__``-less worker whose sleep collaborators log to ``calls``."""

    def make(*, extra_cleanup: bool, legacy: bool = False, rl_enabled: bool | None = None):
        monkeypatch.setattr(backend_module, "current_platform", SimpleNamespace(device_type="npu"))

        worker = worker_module.NPUWorker.__new__(worker_module.NPUWorker)
        worker.model_config = SimpleNamespace(
            sleep_preserve_parameter_names=[],
            sleep_mode_backend=backend_module.UPSTREAM_DEFAULT_SLEEP_MODE_BACKEND,
        )
        worker.model_runner = SimpleNamespace(model=torch.nn.Linear(2, 2))
        worker.model_runner.model.register_buffer("scale", torch.tensor([2.0]))
        worker._sleep_saved_parameters = {}
        worker._sleep_saved_buffers = {}
        worker._sleep_mode_backend = None

        def log(name):
            return lambda *args, **kwargs: calls.append(name)

        worker.sleep_wakeup_manager = SimpleNamespace(sleep=log("manager.sleep"), wakeup=log("manager.wakeup"))
        worker.allocator = MagicMock()
        worker.allocator.sleep.side_effect = log("allocator.sleep")
        worker.allocator.wake_up.side_effect = log("allocator.wake_up")
        transfer_engine = SimpleNamespace(
            unregister_buffer=log("te.unregister"),
            reregister_buffer=log("te.reregister"),
        )
        save_model_state = worker._save_sleep_model_state
        restore_model_state = worker._restore_sleep_model_state
        monkeypatch.setattr(
            worker, "_save_sleep_model_state", lambda level: (calls.append("state.save"), save_model_state(level))
        )
        monkeypatch.setattr(
            worker,
            "_restore_sleep_model_state",
            lambda tags: (calls.append("state.restore"), restore_model_state(tags)),
        )
        monkeypatch.setattr(worker, "synchronize_device", log("worker.sync"))
        monkeypatch.setattr(worker_module.CaMemAllocator, "get_instance", lambda: worker.allocator)
        monkeypatch.setattr(mooncake_transfer_engine, "global_te", transfer_engine)
        monkeypatch.setattr(
            torch,
            "npu",
            SimpleNamespace(synchronize=log("device.sync"), mem_get_info=lambda: (100, 200)),
            raising=False,
        )
        monkeypatch.setattr(
            worker_module,
            "get_ascend_config",
            lambda: SimpleNamespace(
                weight_nz_mode=0,
                rl_config=SimpleNamespace(
                    enabled=(extra_cleanup or legacy) if rl_enabled is None else rl_enabled,
                    sleep_mode_extra_cleanup=extra_cleanup,
                    sleep_mode_legacy=legacy,
                ),
            ),
        )
        return worker

    return make


@pytest.mark.parametrize(
    ("extra_cleanup", "phase_names"),
    [
        (True, ("hccl_and_acl_graph", "model_state", "device_memory")),
        (False, ("model_state", "device_memory")),
    ],
)
def test_worker_hands_its_resources_to_the_backend_as_phases(make_worker, extra_cleanup, phase_names):
    worker = make_worker(extra_cleanup=extra_cleanup)

    backend = worker.sleep_mode_backend

    assert tuple(phase.name for phase in backend.phases) == phase_names
    assert worker.sleep_mode_backend is backend


def test_sleep_and_wake_up_keep_the_resource_order(make_worker, calls):
    worker = make_worker(extra_cleanup=True)

    worker.sleep(level=2)

    # Model state is copied out before anything is released.
    assert calls == [
        "worker.sync",
        "state.save",
        "manager.sleep",
        "device.sync",
        "te.unregister",
        "allocator.sleep",
    ]

    calls.clear()
    worker.wake_up()

    # Model state is back before the manager captures ACL graphs again.
    assert calls == ["allocator.wake_up", "te.reregister", "state.restore", "manager.wakeup", "worker.sync"]


def test_failed_unmap_restores_what_the_sleep_already_released(make_worker, calls):
    worker = make_worker(extra_cleanup=True)
    worker.model_config.sleep_preserve_parameter_names = ["weight"]
    worker.allocator.sleep.side_effect = RuntimeError("unmap failed")

    with pytest.raises(RuntimeError, match="unmap failed"):
        worker.sleep(level=2)

    assert calls[-4:] == ["allocator.wake_up", "te.reregister", "state.restore", "manager.wakeup"]
    worker.allocator.wake_up.assert_called_once_with(tags=None)
    assert worker._sleep_saved_buffers == {}
    assert worker.sleep_mode_backend.state() == "RUNNING"


@pytest.mark.parametrize("vllm_v0_33", [True, False])
def test_third_party_backend_frees_memory_while_the_worker_keeps_its_own_resources(
    make_worker, calls, monkeypatch, vllm_v0_33
):
    class ForeignBackend(SleepModeBackend):
        def suspend(self, discardable=()):
            calls.append(("suspend", discardable))

        def resume(self, tags=None):
            calls.append(("resume", tags))

    worker = make_worker(extra_cleanup=True)
    worker.model_config.sleep_mode_backend = "third_party"
    monkeypatch.setattr(
        SleepModeBackendFactory,
        "create_backend",
        lambda model_config: ForeignBackend() if model_config.sleep_mode_backend == "third_party" else None,
    )
    monkeypatch.setattr(worker_module, "vllm_version_is", lambda version: vllm_v0_33)

    worker.sleep(level=2)

    # Later vLLM passes the parameter storages instead of the level (vLLM #66010).
    parameters = {param.untyped_storage().data_ptr() for param in worker.model_runner.model.parameters()}
    expected = 2 if vllm_v0_33 else parameters
    assert calls == ["worker.sync", "state.save", "manager.sleep", "worker.sync", ("suspend", expected)]

    calls.clear()
    worker.wake_up(tags=["weights"])

    assert calls == [("resume", ["weights"]), "state.restore", "manager.wakeup", "worker.sync"]


@pytest.mark.parametrize("vllm_v0_33", [True, False])
def test_legacy_backend_keeps_the_previous_in_worker_sequence(make_worker, calls, monkeypatch, vllm_v0_33):
    worker = make_worker(extra_cleanup=True, legacy=True)
    monkeypatch.setattr(worker_module, "vllm_version_is", lambda version: vllm_v0_33)

    worker.sleep(level=2)

    assert isinstance(worker.sleep_mode_backend, backend_module.LegacyCaMemSleepBackend)
    assert calls == [
        "worker.sync",
        "state.save",
        "manager.sleep",
        "worker.sync",
        "device.sync",
        "te.unregister",
        "allocator.sleep",
    ]
    worker.allocator.sleep.assert_called_once_with(offload_tags=())

    calls.clear()
    worker.wake_up(tags=["kv_cache"])

    assert calls == ["allocator.wake_up", "te.reregister", "state.restore", "manager.wakeup", "worker.sync"]


def test_legacy_key_is_ignored_unless_rl_config_is_enabled(make_worker):
    # Like every other rl_config sub-field, the fallback only counts with ``enabled``.
    worker = make_worker(extra_cleanup=False, legacy=True, rl_enabled=False)

    assert type(worker.sleep_mode_backend) is backend_module.CaMemSleepBackend


def test_default_name_is_rejected_off_npu(make_worker, monkeypatch):
    worker = make_worker(extra_cleanup=False)
    monkeypatch.setattr(backend_module, "current_platform", SimpleNamespace(device_type="cuda"))

    with pytest.raises(ValueError, match="'cumem' is not supported on this platform"):
        _ = worker.sleep_mode_backend


def test_legacy_backend_does_not_roll_back_a_failed_sleep(make_worker, calls):
    worker = make_worker(extra_cleanup=True, legacy=True)
    worker.allocator.sleep.side_effect = RuntimeError("unmap failed")

    with pytest.raises(RuntimeError, match="unmap failed"):
        worker.sleep(level=1)

    assert "manager.wakeup" not in calls
    worker.allocator.wake_up.assert_not_called()


def test_shutdown_closes_the_backend(make_worker, monkeypatch):
    worker = make_worker(extra_cleanup=True)
    backend = worker.sleep_mode_backend
    worker.profiler = None
    monkeypatch.setattr(worker_module, "ensure_kv_transfer_shutdown", MagicMock())
    monkeypatch.setattr(worker_module, "ensure_ec_transfer_shutdown", MagicMock())

    worker.shutdown()

    assert tuple(phase.name for phase in backend.phases) == ("device_memory",)
    with pytest.raises(RuntimeError, match="closed"):
        backend.suspend()


@pytest.mark.parametrize("vllm_v0_33", [True, False])
@pytest.mark.parametrize("level", [1, 2])
def test_ascend_backend_sleeps_at_the_requested_level_on_either_vllm(make_worker, monkeypatch, vllm_v0_33, level):
    worker = make_worker(extra_cleanup=False)
    monkeypatch.setattr(worker_module, "vllm_version_is", lambda version: vllm_v0_33)

    worker.sleep(level=level)

    worker.allocator.sleep.assert_called_once_with(offload_tags=("weights",) if level == 1 else ())
    # Level 2 backs up the buffers; level 1 keeps them on the device.
    assert bool(worker._sleep_saved_buffers) is (level == 2)
