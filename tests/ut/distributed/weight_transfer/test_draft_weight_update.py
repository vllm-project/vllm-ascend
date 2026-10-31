# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Exercise draft sessions through the worker and NPU IPC receiver on CPU."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import get_current_vllm_config, set_current_vllm_config
from vllm.distributed.weight_transfer.base import WeightTransferEngine
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

from tests.ut.distributed.weight_transfer.test_npu_ipc_engine import (
    _patch_rebuild_npu_tensor,
    _patch_reload_module,
)
from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import NPUIPCWeightTransferEngine
from vllm_ascend.worker.v2.model_runner import NPUModelRunner
from vllm_ascend.worker.worker import NPUWorker

_IPC_MODULE = "vllm_ascend.distributed.weight_transfer.npu_ipc_engine"


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(3))

    def load_weights(self, weights):
        with torch.no_grad():
            for name, value in weights:
                self.get_parameter(name).copy_(value)


@pytest.fixture
def worker():
    worker = object.__new__(NPUWorker)
    worker.use_v2_model_runner = True
    target, draft = _Model(), _Model()
    worker.vllm_config = SimpleNamespace(model_config=object(), parallel_config=SimpleNamespace())
    worker.speculative_config = SimpleNamespace(draft_model_config=object())
    runner = object.__new__(NPUModelRunner)
    runner.model = target
    runner.speculator = MagicMock(spec=DraftModelSpeculator)
    runner.speculator.model = draft
    runner.reset_lora_state = MagicMock()
    worker.model_runner = runner
    engine = object.__new__(NPUIPCWeightTransferEngine)
    WeightTransferEngine.__init__(engine, SimpleNamespace(), worker.vllm_config, SimpleNamespace(index=0), target)
    engine._packed_importer = MagicMock()
    worker.weight_transfer_engine = engine
    worker._weight_update_active = False
    worker._weight_update_is_draft = False
    worker._check_nz_disabled = MagicMock()
    return worker


@pytest.mark.parametrize("packed", [False, True])
def test_ipc_draft_then_target_session(worker, packed):
    engine = worker.weight_transfer_engine
    engine.packed = packed
    target = worker.model_runner.model
    draft = worker.get_draft_model()
    draft_config = worker.speculative_config.draft_model_config
    target_config = worker.vllm_config.model_config
    original_storage = (target.weight.data_ptr(), draft.weight.data_ptr())
    initialized, finalized = [], []

    def initialize(model):
        if model is draft:
            assert get_current_vllm_config() is worker.vllm_config
        initialized.append(model)

    def finalize(model, config):
        if model is draft:
            assert get_current_vllm_config() is worker.vllm_config
        finalized.append((model, config))

    with (
        _patch_reload_module(initialize=initialize, finalize=finalize),
        patch(f"{_IPC_MODULE}.npu_generate_uuid", return_value="node-0"),
        patch("vllm.distributed.weight_transfer.base.torch.accelerator.synchronize"),
    ):
        for is_draft, value in [(True, 2.0), (False, 5.0), (True, 7.0)]:
            tensor = torch.full((3,), value)
            payload = dict(names=["weight"], dtype_names=["float32"], shapes=[[3]])
            if packed:
                payload.update(ipc_handles={"node-0": ("handle",)}, tensor_sizes=[12])
                receive_patch = patch(
                    f"{_IPC_MODULE}.packed_npu_ipc_consumer",
                    side_effect=lambda tensor=tensor, **kwargs: iter([("weight", tensor)]),
                )
            else:
                payload.update(ipc_handles=[{"node-0": (None,) * 6 + (999,)}])
                receive_patch = _patch_rebuild_npu_tensor(lambda *args, tensor=tensor: tensor)

            (worker.start_draft_weight_update if is_draft else worker.start_weight_update)()
            with receive_patch:
                # More than one chunk must keep the same target until FINISH.
                tensor.fill_(value / 2)
                worker.update_weights(payload)
                torch.testing.assert_close((draft if is_draft else target).weight, tensor)
                tensor.fill_(value)
                worker.update_weights(payload)
            worker.finish_weight_update()
            assert engine.model is target
            assert engine.model_config is target_config
            assert not worker._weight_update_active
            assert not worker._weight_update_is_draft
            assert (target.weight.data_ptr(), draft.weight.data_ptr()) == original_storage

            torch.testing.assert_close(draft.weight, torch.full((3,), value if is_draft else 2.0))
            torch.testing.assert_close(target.weight, torch.full((3,), 0.0 if len(finalized) == 1 else 5.0))

    assert initialized == [draft, target, draft]
    assert finalized == [(draft, draft_config), (target, target_config), (draft, draft_config)]
    worker.model_runner.reset_lora_state.assert_called_once_with()


@pytest.mark.parametrize("stage", ["start_weight_update", "update_weights"])
def test_failed_draft_session_restores_target(worker, stage):
    engine = worker.weight_transfer_engine
    target = engine.model
    with _patch_reload_module():
        if stage == "update_weights":
            worker.start_draft_weight_update()
        with (
            patch.object(engine, stage, side_effect=ValueError("failed")),
            pytest.raises(ValueError, match="failed"),
        ):
            if stage == "start_weight_update":
                worker.start_draft_weight_update()
            else:
                worker.update_weights({})
        assert not worker._weight_update_active
        assert not worker._weight_update_is_draft
        assert engine.model is target
        assert engine.model_config is worker.vllm_config.model_config
        worker.start_weight_update()
        worker.finish_weight_update()
    worker.model_runner.reset_lora_state.assert_called_once_with()


@pytest.mark.parametrize("missing", ["engine", "model", "config", "draft_config", "support"])
def test_draft_availability_and_start_validation(worker, missing):
    engine = worker.weight_transfer_engine
    target = engine.model
    assert worker.supports_draft_weight_updates()
    if missing == "engine":
        worker.weight_transfer_engine = None
    elif missing == "model":
        worker.model_runner.speculator = None
    elif missing == "config":
        worker.speculative_config = None
    elif missing == "draft_config":
        worker.speculative_config.draft_model_config = None
    else:
        engine.supports_draft_weight_update = False
    assert not worker.supports_draft_weight_updates()
    with pytest.raises(RuntimeError):
        worker.start_draft_weight_update()
    assert not worker._weight_update_active
    assert engine.model is target


@pytest.mark.parametrize("next_is_draft", [False, True])
def test_draft_session_rejects_nested_start_without_resetting_target(worker, next_is_draft):
    with _patch_reload_module():
        worker.start_draft_weight_update()
        with pytest.raises(RuntimeError, match="already active"):
            (worker.start_draft_weight_update if next_is_draft else worker.start_weight_update)()
        assert worker.weight_transfer_engine.model is worker.get_draft_model()
        assert worker._weight_update_active
        worker.finish_weight_update()


def test_draft_update_rejects_nz(worker):
    worker._check_nz_disabled.side_effect = ValueError("NZ enabled")
    with pytest.raises(ValueError, match="NZ enabled"):
        worker.start_draft_weight_update()
    assert not worker._weight_update_active
    assert worker.weight_transfer_engine.model is worker.model_runner.model


def test_failed_finish_can_be_retried_for_the_same_draft(worker):
    engine = worker.weight_transfer_engine
    with _patch_reload_module():
        worker.start_draft_weight_update()
        with (
            patch.object(engine, "finish_weight_update", side_effect=ValueError("finalize failed")),
            pytest.raises(ValueError, match="finalize failed"),
        ):
            worker.finish_weight_update()
        assert worker._weight_update_active
        assert worker._weight_update_is_draft
        assert engine.model is worker.get_draft_model()
        worker.finish_weight_update()
    assert not worker._weight_update_active
    assert not worker._weight_update_is_draft
    assert engine.model is worker.model_runner.model
    worker.model_runner.reset_lora_state.assert_not_called()


@pytest.mark.parametrize("speculator", [None, SimpleNamespace()])
def test_v2_get_draft_model_without_model_based_speculation(worker, speculator):
    worker.model_runner.speculator = speculator
    assert worker.get_draft_model() is None
    assert not worker.supports_draft_weight_updates()
    with pytest.raises(RuntimeError, match="no draft model"):
        worker.start_draft_weight_update()


def test_v2_get_draft_model_uses_speculator(worker):
    assert worker.get_draft_model() is worker.model_runner.speculator.model
    assert worker.supports_draft_weight_updates()


def test_shared_draft_parameter_keeps_its_target_reference(worker):
    draft = worker.get_draft_model()
    draft.weight = worker.model_runner.model.weight
    shared = draft.weight
    with _patch_reload_module():
        worker.start_draft_weight_update()
        worker.weight_transfer_engine.model.load_weights([("weight", torch.ones(3))])
        worker.finish_weight_update()
    assert draft.weight is shared
    assert worker.model_runner.model.weight is shared
    torch.testing.assert_close(shared, torch.ones(3))


def test_v1_draft_getter_cannot_enable_draft_updates(worker):
    getter = MagicMock(return_value=_Model())
    worker.use_v2_model_runner = False
    worker.model_runner = SimpleNamespace(get_draft_model=getter)
    assert worker.get_draft_model() is None
    assert not worker.supports_draft_weight_updates()
    with (
        patch.object(worker.weight_transfer_engine, "start_weight_update") as start,
        pytest.raises(RuntimeError, match="require Model Runner V2"),
    ):
        worker.start_draft_weight_update()
    getter.assert_not_called()
    start.assert_not_called()
    assert not worker._weight_update_active
    assert not worker._weight_update_is_draft


@pytest.mark.parametrize("runner", [None, SimpleNamespace(), SimpleNamespace(get_draft_model=None)])
def test_get_draft_model_handles_runners_without_a_getter(worker, runner):
    worker.model_runner = runner
    assert worker.get_draft_model() is None
    assert not worker.supports_draft_weight_updates()


def test_target_session_preserves_the_callers_configuration_context(worker):
    outer_config = SimpleNamespace()
    seen = []

    def record(*args):
        seen.append(get_current_vllm_config())

    with (
        set_current_vllm_config(outer_config),
        patch.object(worker.weight_transfer_engine, "start_weight_update", side_effect=record),
        patch.object(worker.weight_transfer_engine, "update_weights", side_effect=record),
        patch.object(worker.weight_transfer_engine, "finish_weight_update", side_effect=record),
    ):
        worker.start_weight_update()
        worker.update_weights({})
        worker.finish_weight_update()
    assert seen == [outer_config] * 3
    assert not worker._weight_update_is_draft


def test_draft_session_sets_worker_configuration_only_for_each_phase(worker):
    outer_config = SimpleNamespace()
    seen = []

    def record(*args):
        seen.append(get_current_vllm_config())
        assert worker.weight_transfer_engine.model is worker.get_draft_model()

    with (
        set_current_vllm_config(outer_config),
        patch.object(worker.weight_transfer_engine, "start_weight_update", side_effect=record),
        patch.object(worker.weight_transfer_engine, "update_weights", side_effect=record),
        patch.object(worker.weight_transfer_engine, "finish_weight_update", side_effect=record),
    ):
        worker.start_draft_weight_update()
        assert get_current_vllm_config() is outer_config
        worker.update_weights({})
        assert get_current_vllm_config() is outer_config
        worker.finish_weight_update()
        assert get_current_vllm_config() is outer_config
    assert seen == [worker.vllm_config] * 3
    assert not worker._weight_update_is_draft
