from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.worker.worker import NPUWorker


def _worker(draft_model=None, patterns=()):
    worker = object.__new__(NPUWorker)
    worker.model_config = SimpleNamespace(sleep_preserve_parameter_names=list(patterns))
    worker.model_runner = SimpleNamespace(get_draft_model=lambda: draft_model)
    worker._sleep_saved_parameters = {}
    return worker


def test_sleep_parameters_include_private_draft_and_skip_shared_parameters():
    target = torch.nn.Linear(2, 2, bias=False)
    draft = torch.nn.Module()
    draft.private = torch.nn.Linear(2, 2, bias=False)
    draft.shared = target.weight

    parameters = _worker(draft)._get_sleep_parameters(target)

    assert set(parameters) == {"weight", "draft.private.weight"}
    assert parameters["weight"] is target.weight
    assert parameters["draft.private.weight"] is draft.private.weight


def test_draft_wildcard_is_allowed_when_no_draft_parameters_exist():
    worker = _worker(patterns=("draft.*",))

    worker._save_sleep_parameters(torch.nn.Linear(2, 2, bias=False))

    assert worker._sleep_saved_parameters == {}


def test_unmatched_target_pattern_is_rejected():
    worker = _worker(patterns=("missing.weight",))

    with pytest.raises(ValueError, match="No parameter matches sleep retention"):
        worker._save_sleep_parameters(torch.nn.Linear(2, 2, bias=False))


def test_restore_keeps_draft_storage_and_does_not_overwrite_shared_target():
    target = torch.nn.Linear(2, 2, bias=False)
    draft = torch.nn.Module()
    draft.private = torch.nn.Linear(2, 2, bias=False)
    draft.shared = target.weight
    worker = _worker(draft)

    expected = draft.private.weight.detach().clone()
    pointer = draft.private.weight.data_ptr()
    worker._sleep_saved_parameters = {"draft.private.weight": expected}

    with torch.no_grad():
        target.weight.fill_(3.0)
        draft.private.weight.zero_()

    worker._restore_sleep_parameters(target)

    torch.testing.assert_close(draft.private.weight, expected)
    assert draft.private.weight.data_ptr() == pointer
    assert torch.all(target.weight == 3.0)
    assert worker._sleep_saved_parameters == {"draft.private.weight": expected}


@patch("vllm_ascend.worker.worker.get_ascend_config")
@patch("vllm_ascend.worker.worker.CaMemAllocator")
def test_wake_only_restores_snapshots_for_weights(mock_allocator_class, mock_get_config):
    target = torch.nn.Linear(2, 2, bias=False)
    target.register_buffer("target_buffer", torch.ones(2))
    draft = torch.nn.Module()
    draft.private = torch.nn.Linear(2, 2, bias=False)
    draft.register_buffer("draft_buffer", torch.ones(2))

    worker = _worker(draft)
    worker.model_runner.model = target
    worker.model_runner.get_model = lambda: target
    worker._sleep_saved_buffers = {"target_buffer": torch.zeros(2)}
    worker._sleep_saved_draft_buffers = {"draft_buffer": torch.zeros(2)}
    worker.sleep_wakeup_manager = MagicMock()

    mock_get_config.return_value = SimpleNamespace(
        weight_nz_mode=0,
        rl_config=SimpleNamespace(enabled=False, sleep_mode_extra_cleanup=False),
    )
    mock_allocator_class.get_instance.return_value = MagicMock()

    worker.wake_up(tags=["kv_cache"])

    assert torch.all(target.target_buffer == 1)
    assert torch.all(draft.draft_buffer == 1)
    assert worker._sleep_saved_buffers
    assert worker._sleep_saved_draft_buffers

    worker.wake_up(tags=["weights"])

    assert torch.all(target.target_buffer == 0)
    assert torch.all(draft.draft_buffer == 0)
    assert worker._sleep_saved_buffers == {}
    assert worker._sleep_saved_draft_buffers == {}


@patch("vllm_ascend.worker.worker.get_ascend_config")
@patch("vllm_ascend.worker.worker.CaMemAllocator")
def test_wake_failure_keeps_all_snapshots_for_retry(mock_allocator_class, mock_get_config):
    target = torch.nn.Linear(2, 2, bias=False)
    draft = torch.nn.Module()
    draft.private = torch.nn.Linear(2, 2, bias=False)
    draft.register_buffer("draft_buffer", torch.ones(2))

    worker = _worker(draft, patterns=("weight",))
    worker.model_runner.model = target
    worker._sleep_saved_parameters = {"weight": target.weight.detach().clone()}
    worker._sleep_saved_buffers = {"missing_buffer": torch.zeros(2)}
    worker._sleep_saved_draft_buffers = {"draft_buffer": torch.zeros(2)}
    worker.sleep_wakeup_manager = MagicMock()

    mock_get_config.return_value = SimpleNamespace(
        weight_nz_mode=0,
        rl_config=SimpleNamespace(enabled=False, sleep_mode_extra_cleanup=False),
    )
    mock_allocator_class.get_instance.return_value = MagicMock()

    with pytest.raises(RuntimeError, match="Saved model buffers are missing"):
        worker.wake_up(tags=["weights"])

    assert set(worker._sleep_saved_parameters) == {"weight"}
    assert set(worker._sleep_saved_buffers) == {"missing_buffer"}
    assert set(worker._sleep_saved_draft_buffers) == {"draft_buffer"}


@patch("vllm_ascend.worker.worker.get_ascend_config")
@patch("vllm_ascend.worker.worker.CaMemAllocator")
@patch("vllm_ascend.worker.worker.torch.npu.mem_get_info", return_value=(100, 1000))
def test_level_one_sleep_does_not_drop_level_two_snapshots(mock_mem_get_info, mock_allocator_class, mock_get_config):
    worker = _worker()
    worker._sleep_saved_buffers = {"target_buffer": torch.zeros(1)}
    worker._sleep_saved_parameters = {"weight": torch.zeros(1)}
    worker._sleep_saved_draft_buffers = {"draft_buffer": torch.zeros(1)}
    worker.sleep_wakeup_manager = MagicMock()
    mock_get_config.return_value = SimpleNamespace(
        rl_config=SimpleNamespace(enabled=False, sleep_mode_extra_cleanup=False),
    )
    mock_allocator_class.get_instance.return_value = MagicMock()

    worker.sleep(level=1)

    assert worker._sleep_saved_buffers
    assert worker._sleep_saved_parameters
    assert worker._sleep_saved_draft_buffers
