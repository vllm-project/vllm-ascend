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
    worker.model_config = SimpleNamespace(sleep_preserve_parameter_names=["weight"])
    worker.model_runner = SimpleNamespace(model=torch.nn.Linear(2, 2))
    worker.model_runner.model.register_buffer("scale", torch.tensor([2.0]))
    worker._sleep_saved_parameters = {}
    worker._sleep_saved_buffers = {}
    worker._sleep_saved_draft_buffers = {}
    worker.use_v2_model_runner = False
    worker.sleep_wakeup_manager = MagicMock()
    worker.allocator = MagicMock()
    monkeypatch.setattr(worker, "synchronize_device", MagicMock())
    monkeypatch.setattr(worker_module.CaMemAllocator, "get_instance", lambda: worker.allocator)
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


def device_parameters(model, monkeypatch):
    # CPU UTs simulate only the device-to-CPU boundary; comparisons/copies use real tensors.
    parameters = list(model.named_parameters())
    mocks = []
    for name, parameter in parameters:
        mock = MagicMock()
        mock.is_cpu = False
        mock.detach.return_value.to.side_effect = lambda _, parameter=parameter: parameter.detach().clone()
        mock.copy_.side_effect = parameter.copy_
        mocks.append((name, mock))
    monkeypatch.setattr(model, "named_parameters", lambda: iter(mocks))
    return dict(mocks)


def test_save_matches_globs_and_deduplicates_parameters(worker, monkeypatch):
    model = worker.model_runner.model
    parameters = device_parameters(model, monkeypatch)
    worker.model_config.sleep_preserve_parameter_names = ["w*", "weight"]

    worker._save_sleep_parameters(model)

    assert set(worker._sleep_saved_parameters) == {"weight"}
    assert torch.equal(worker._sleep_saved_parameters["weight"], model.weight)
    assert worker._sleep_saved_parameters["weight"].data_ptr() != model.weight.data_ptr()
    parameters["weight"].detach.return_value.to.assert_called_once_with("cpu")
    parameters["bias"].detach.assert_not_called()


def test_cpu_parameters_need_no_backup(worker):
    worker._save_sleep_parameters(worker.model_runner.model)

    assert worker._sleep_saved_parameters == {}


@pytest.mark.parametrize("config", [SimpleNamespace(), SimpleNamespace(sleep_preserve_parameter_names=[])])
def test_empty_or_older_configuration_clears_saved_parameters(worker, config):
    worker.model_config = config
    worker._sleep_saved_parameters = {"stale": torch.ones(1)}

    worker._save_sleep_parameters(worker.model_runner.model)

    assert worker._sleep_saved_parameters == {}


def test_unmatched_pattern_fails_before_discarding_memory(worker):
    worker.model_config.sleep_preserve_parameter_names = ["missing.*"]

    with pytest.raises(ValueError, match="No parameter matches sleep retention: missing"):
        worker.sleep(level=2)

    worker.allocator.sleep.assert_not_called()


def test_level_two_retains_parameters_before_allocator_discards_them(worker, monkeypatch):
    model = worker.model_runner.model
    expected = model.weight.detach().clone()
    device_parameters(model, monkeypatch)

    def discard(offload_tags):
        assert offload_tags == ()
        assert torch.equal(worker._sleep_saved_parameters["weight"], expected)
        with torch.no_grad():
            model.weight.zero_()
            model.scale.zero_()

    worker.allocator.sleep.side_effect = discard
    worker.sleep(level=2)
    worker.wake_up(tags=["weights"])

    assert torch.equal(model.weight, expected)
    assert torch.equal(model.scale, torch.tensor([2.0]))
    assert worker._sleep_saved_parameters == {}
    assert worker._sleep_saved_buffers == {}
    assert worker.synchronize_device.call_count == 3


def test_level_one_keeps_allocator_offload_and_clears_old_backups(worker, monkeypatch):
    worker._sleep_saved_parameters = {"weight": torch.ones(2, 2)}
    worker._sleep_saved_buffers = {"scale": torch.ones(1)}
    worker._sleep_saved_draft_buffers = {"metadata": torch.ones(1)}
    save = MagicMock()
    monkeypatch.setattr(worker, "_save_sleep_parameters", save)

    worker.sleep(level=1)

    save.assert_not_called()
    worker.allocator.sleep.assert_called_once_with(offload_tags=("weights",))
    assert worker._sleep_saved_parameters == {}
    assert worker._sleep_saved_buffers == {}
    assert worker._sleep_saved_draft_buffers == {}


@pytest.mark.parametrize("tags", [None, ["weights"], ["weights", "kv_cache"]])
def test_weight_wake_restores_in_place_after_mapping(worker, tags):
    model = worker.model_runner.model
    pointer = model.weight.data_ptr()
    expected = torch.full_like(model.weight, 3.0)
    worker._sleep_saved_parameters = {"weight": expected.clone()}
    worker._sleep_saved_buffers = {"scale": torch.tensor([4.0])}
    worker.allocator.wake_up.side_effect = lambda tags: model.weight.data.zero_()

    worker.wake_up(tags=tags)

    assert model.weight.data_ptr() == pointer
    assert model.weight.requires_grad
    assert torch.equal(model.weight, expected)
    assert torch.equal(model.scale, torch.tensor([4.0]))
    assert worker._sleep_saved_parameters == {}
    assert worker._sleep_saved_buffers == {}
    worker.allocator.wake_up.assert_called_once_with(tags=tags)
    worker.synchronize_device.assert_called_once_with()


@pytest.mark.parametrize("tags", [[], ["kv_cache"], ["other"]])
def test_non_weight_wake_preserves_backups_without_touching_parameters(worker, tags):
    model = worker.model_runner.model
    original = model.weight.detach().clone()
    backup = torch.full_like(model.weight, 3.0)
    worker._sleep_saved_parameters = {"weight": backup}
    worker._sleep_saved_buffers = {"scale": torch.tensor([4.0])}

    worker.wake_up(tags=tags)

    assert torch.equal(model.weight, original)
    assert worker._sleep_saved_parameters["weight"] is backup
    assert set(worker._sleep_saved_buffers) == {"scale"}


def test_repeated_sleep_uses_latest_parameter_values(worker, monkeypatch):
    model = worker.model_runner.model
    device_parameters(model, monkeypatch)

    for value in (3.0, 7.0):
        with torch.no_grad():
            model.weight.fill_(value)
        worker._save_sleep_parameters(model)
        with torch.no_grad():
            model.weight.zero_()
        worker._restore_sleep_parameters(model)

        assert torch.equal(model.weight, torch.full_like(model.weight, value))
        assert worker._sleep_saved_parameters == {}


@pytest.fixture
def draft(worker):
    model = torch.nn.Linear(2, 2)
    model.register_buffer("metadata", torch.tensor([3.0]), persistent=False)
    model.register_buffer("fused_kv", torch.arange(6.0).reshape(2, 3), persistent=False)
    model.fused_kv_view = model.fused_kv.view(-1)
    worker.use_v2_model_runner = True
    worker.model_runner.get_draft_model = lambda: model
    return model


@pytest.mark.parametrize("tags", [None, ["weights"], ["weights", "kv_cache"]])
def test_level_two_restores_draft_buffers_after_mapping(worker, draft, tags):
    worker.model_config.sleep_preserve_parameter_names = []
    pointer = draft.metadata.data_ptr()
    fused_pointer = draft.fused_kv.data_ptr()

    def discard(offload_tags):
        assert torch.equal(worker._sleep_saved_draft_buffers["metadata"], torch.tensor([3.0]))
        draft.metadata.zero_()
        draft.fused_kv.zero_()

    worker.allocator.sleep.side_effect = discard
    worker.sleep(level=2)
    worker.wake_up(tags=tags)
    assert draft.metadata.data_ptr() == pointer
    torch.testing.assert_close(draft.metadata, torch.tensor([3.0]))
    assert draft.fused_kv.data_ptr() == fused_pointer
    torch.testing.assert_close(draft.fused_kv_view, torch.arange(6.0))
    assert worker._sleep_saved_draft_buffers == {}


def test_kv_only_wake_keeps_draft_backup_for_the_weights_phase(worker, draft):
    worker._sleep_saved_draft_buffers = {"metadata": torch.tensor([8.0])}
    worker.wake_up(tags=["kv_cache"])
    torch.testing.assert_close(draft.metadata, torch.tensor([3.0]))
    assert "metadata" in worker._sleep_saved_draft_buffers
    worker.wake_up(tags=["weights"])
    torch.testing.assert_close(draft.metadata, torch.tensor([8.0]))
    assert worker._sleep_saved_draft_buffers == {}


def test_level_two_retains_selected_draft_parameters(worker, draft, monkeypatch):
    expected = draft.weight.detach().clone()
    worker.model_config.sleep_preserve_parameter_names = ["draft.weight"]
    device_parameters(draft, monkeypatch)

    def discard(offload_tags):
        assert set(worker._sleep_saved_parameters) == {"draft.weight"}
        draft.weight.data.zero_()

    worker.allocator.sleep.side_effect = discard
    worker.sleep(level=2)
    worker.wake_up(tags=["weights"])
    torch.testing.assert_close(draft.weight, expected)
    assert worker._sleep_saved_parameters == {}


def test_sleep_parameter_names_deduplicate_shared_target_and_draft_parameters(worker, draft):
    draft.weight = worker.model_runner.model.weight
    parameters = worker._get_sleep_parameters(worker.model_runner.model)
    assert "weight" in parameters
    assert "draft.weight" not in parameters
    assert parameters["draft.bias"] is draft.bias


def test_optional_draft_wildcard_allows_a_target_only_worker(worker):
    worker.model_config.sleep_preserve_parameter_names = ["draft.*"]
    worker._save_sleep_parameters(worker.model_runner.model)
    assert worker._sleep_saved_parameters == {}


def test_missing_draft_on_wake_preserves_backup_and_reports_error(worker):
    worker._sleep_saved_draft_buffers = {"metadata": torch.tensor([8.0])}
    with pytest.raises(RuntimeError, match="Cannot restore saved draft buffers"):
        worker.wake_up(tags=["weights"])
    assert "metadata" in worker._sleep_saved_draft_buffers


def test_repeated_level_two_sleep_uses_latest_draft_buffers(worker, draft, monkeypatch):
    worker.model_config.sleep_preserve_parameter_names = []
    monkeypatch.setattr("vllm_ascend.worker.worker.torch.npu.mem_get_info", lambda: (100, 200))
    worker.allocator.sleep.side_effect = lambda offload_tags: draft.metadata.zero_()
    for value in (4.0, 9.0):
        draft.metadata.fill_(value)
        worker.sleep(level=2)
        worker.wake_up(tags=["weights"])
        torch.testing.assert_close(draft.metadata, torch.tensor([value]))
