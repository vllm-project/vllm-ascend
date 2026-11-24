# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture
def worker(monkeypatch):
    from vllm_ascend.worker import worker as worker_module

    worker = worker_module.NPUWorker.__new__(worker_module.NPUWorker)
    worker.model_config = SimpleNamespace(
        sleep_preserve_parameter_names=["weight"],
        sleep_mode_backend="camem",
    )
    worker.vllm_config = SimpleNamespace(model_config=worker.model_config)
    worker.model_runner = SimpleNamespace(
        model=torch.nn.Linear(2, 2),
        get_draft_model=lambda: None,
    )
    from vllm_ascend.device_allocator.sleep_mode_backend import CaMemBackend

    worker._sleep_mode_backend = MagicMock(spec=CaMemBackend)
    worker.sleep_wakeup_manager = MagicMock()
    monkeypatch.setattr(worker, "synchronize_device", MagicMock())
    monkeypatch.setattr(worker_module.global_te, "unregister_buffer", MagicMock())
    monkeypatch.setattr(
        worker_module.torch.npu,
        "mem_get_info",
        MagicMock(side_effect=[(100, 200), (150, 200)]),
    )
    monkeypatch.setattr(
        worker_module,
        "get_ascend_config",
        lambda: SimpleNamespace(
            weight_nz_mode=0,
            rl_config=SimpleNamespace(enabled=False, sleep_mode_extra_cleanup=False),
        ),
    )
    return worker


def _storage_ptr(parameter: torch.nn.Parameter) -> int:
    return parameter.untyped_storage().data_ptr()


def test_discardable_excludes_preserved_storage(worker):
    model = worker.model_runner.model

    discardable = worker._discardable_parameters(model)

    assert _storage_ptr(model.weight) not in discardable
    assert _storage_ptr(model.bias) in discardable


def test_discardable_contains_all_parameters_without_preserve(worker):
    worker.model_config.sleep_preserve_parameter_names = []
    model = worker.model_runner.model

    discardable = worker._discardable_parameters(model)

    assert discardable == {_storage_ptr(model.weight), _storage_ptr(model.bias)}


def test_unmatched_pattern_fails_before_sleep(worker):
    worker.model_config.sleep_preserve_parameter_names = ["missing.*"]

    with pytest.raises(ValueError, match="No parameter matches sleep retention: missing"):
        worker.sleep(level=2)

    worker.sleep_mode_backend.suspend.assert_not_called()


def test_draft_pattern_is_allowed_without_draft(worker):
    worker.model_config.sleep_preserve_parameter_names = ["draft.*"]

    discardable = worker._discardable_parameters(worker.model_runner.model)

    assert _storage_ptr(worker.model_runner.model.weight) in discardable


def test_level_one_passes_empty_discardable(worker):
    worker.sleep(level=1)

    worker.sleep_mode_backend.suspend.assert_called_once_with(set())


def test_level_two_passes_discardable_storages(worker):
    model = worker.model_runner.model
    expected = worker._discardable_parameters(model)

    worker.sleep(level=2)

    worker.sleep_mode_backend.suspend.assert_called_once_with(expected)


def test_worker_no_longer_keeps_python_sleep_snapshots(worker):
    assert not hasattr(worker, "_sleep_saved_parameters")
    assert not hasattr(worker, "_sleep_saved_buffers")


def test_get_sleep_parameters_returns_target_parameters(worker):
    parameters = worker._get_sleep_parameters(worker.model_runner.model)

    assert set(parameters) == {"weight", "bias"}


def test_get_sleep_parameters_includes_draft_parameters(worker):
    target = worker.model_runner.model
    draft = torch.nn.Linear(2, 2)
    worker.model_runner.get_draft_model = lambda: draft

    parameters = worker._get_sleep_parameters(target)

    assert "draft.weight" in parameters
    assert "draft.bias" in parameters
    assert parameters["draft.weight"] is draft.weight
    assert parameters["draft.bias"] is draft.bias


def test_get_sleep_parameters_skips_shared_draft_parameters(worker):
    target = worker.model_runner.model
    draft = torch.nn.Module()
    draft.weight = target.weight
    worker.model_runner.get_draft_model = lambda: draft

    parameters = worker._get_sleep_parameters(target)

    assert "draft.weight" not in parameters
    assert parameters["weight"] is target.weight


def test_discardable_parameters_include_draft_parameters(worker):
    target = worker.model_runner.model
    draft = torch.nn.Linear(2, 2)
    worker.model_runner.get_draft_model = lambda: draft

    discardable = worker._discardable_parameters(target)

    assert draft.weight.untyped_storage().data_ptr() in discardable
    assert draft.bias.untyped_storage().data_ptr() in discardable
    assert target.weight.untyped_storage().data_ptr() not in discardable


def test_preserved_draft_parameters_are_not_discardable(worker):
    target = worker.model_runner.model
    draft = torch.nn.Linear(2, 2)
    worker.model_runner.get_draft_model = lambda: draft
    worker.model_config.sleep_preserve_parameter_names = ["draft.*"]

    discardable = worker._discardable_parameters(target)

    assert draft.weight.untyped_storage().data_ptr() not in discardable
    assert draft.bias.untyped_storage().data_ptr() not in discardable
