# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch_npu

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_ascend.worker.v2.model_runner import NPUModelRunner
from vllm_ascend.ec_manager.score_encoder_runner import AscendScoreEncoderRunner
from vllm_ascend.ec_manager.score_encoder_cache import ScoreEncoderCache


def _make_runner() -> AscendScoreEncoderRunner:
    runner = AscendScoreEncoderRunner.__new__(AscendScoreEncoderRunner)
    runner.encoder_cache = SimpleNamespace(encoder_outputs={})
    runner.device = torch.device("cpu")
    runner.storage = ScoreEncoderCache(runner.encoder_cache, runner.device)
    return runner


def _metadata(npu_freed=(), cpu_freed=(), promoting=(), cpu_get=()):
    return SimpleNamespace(
        promoting_mm_hashes=list(promoting),
        cpu_get_encoder_mm_hashes=list(cpu_get),
        npu_freed=list(npu_freed),
        cpu_freed=list(cpu_freed),
    )


def _mm_feature(mm_hash: str) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier=mm_hash,
        mm_position=PlaceholderRange(offset=0, length=4),
    )


def _tensor(value: float) -> torch.Tensor:
    return torch.tensor([value])


def test_track_and_remove_refcounts_evict_tmp():
    runner = _make_runner()
    runner.storage.tmp_encoder_cache = {
        "h1": _tensor(1),
        "h2": _tensor(2),
        "h3": _tensor(3),
    }

    runner.track_request("req-a", [_mm_feature("h1"), _mm_feature("h2")])
    runner.track_request("req-b", [_mm_feature("h1"), _mm_feature("h3")])
    assert runner.storage._req_refs == {
        "h1": {"req-a", "req-b"},
        "h2": {"req-a"},
        "h3": {"req-b"},
    }

    runner.on_request_removed("req-unknown")

    runner.on_request_removed("req-a")
    assert "h1" in runner.storage.tmp_encoder_cache
    assert "h2" not in runner.storage.tmp_encoder_cache

    runner.on_request_removed("req-b")
    assert runner.storage.tmp_encoder_cache == {}
    assert runner.storage._req_refs == {}
    assert runner.storage._req_hash_index == {}


def test_cache_encoder_output_routes_by_metadata():
    runner = _make_runner()
    resident, freed, cold = _tensor(1), _tensor(2), _tensor(3)
    runner.storage._current_step_metadata = _metadata(
        promoting=["resident"],
        npu_freed=["freed-and-promoted"],
    )

    runner._cache_encoder_output("resident", resident)
    runner._cache_encoder_output("freed-and-promoted", freed)
    runner.storage._current_step_metadata = None
    runner._cache_encoder_output("cold", cold)

    assert runner.encoder_cache.encoder_outputs == {"resident": resident}
    assert set(runner.storage.tmp_encoder_cache) == {"freed-and-promoted", "cold"}

    assert set(runner.storage.cpu_encoder_cache) == {"resident", "freed-and-promoted", "cold"}
    assert torch.equal(runner.storage.cpu_encoder_cache["resident"], resident)


def test_process_scheduler_metadata_consumes_four_lists():
    runner = _make_runner()
    runner.encoder_cache.encoder_outputs = {"evicted": _tensor(1)}
    runner.storage.tmp_encoder_cache = {"already-staged": _tensor(5)}
    runner.storage.cpu_encoder_cache = {
        "evicted-cpu": _tensor(4),
        "to-promote": _tensor(2),
        "to-stage": _tensor(3),
    }

    copies = MagicMock(return_value=_tensor(9))
    polled = MagicMock()
    runner.storage._copy_cpu_encoder_cache_to_device = copies
    runner.storage._clear_finished_encoder_cache_copies = polled

    metadata = _metadata(
        npu_freed=["evicted"],
        cpu_freed=["evicted-cpu"],
        promoting=["to-promote", "missing-on-cpu"],
        cpu_get=["to-stage", "already-staged", "missing-on-cpu"],
    )
    runner.process_scheduler_metadata(metadata)

    polled.assert_called_once_with()
    assert runner.storage._current_step_metadata is metadata
    assert "evicted" not in runner.encoder_cache.encoder_outputs
    assert runner.encoder_cache.encoder_outputs["to-promote"] is copies.return_value
    assert "evicted-cpu" not in runner.storage.cpu_encoder_cache
    assert runner.storage.tmp_encoder_cache["to-stage"] is copies.return_value
    assert copies.call_count == 2


def test_get_encoder_output_three_tier_priority():
    runner = _make_runner()
    resident, tmp, cpu = _tensor(1), _tensor(2), _tensor(3)

    runner.encoder_cache.encoder_outputs = {"mm": resident}
    assert runner._get_encoder_output_from_cache("mm") is resident
    assert runner._has_cached_encoder_output("mm")

    runner.encoder_cache.encoder_outputs = {}
    runner.storage.tmp_encoder_cache = {"mm": tmp}
    assert runner._get_encoder_output_from_cache("mm") is tmp
    assert runner._has_cached_encoder_output("mm")

    runner.storage.tmp_encoder_cache = {}
    runner.storage.cpu_encoder_cache = {"mm": cpu}
    assert runner._has_cached_encoder_output("mm")
    staged = runner._get_encoder_output_from_cache("mm")
    assert runner.storage.tmp_encoder_cache["mm"] is staged
    assert torch.equal(staged, cpu)

    runner.storage.tmp_encoder_cache = {}
    runner.storage.cpu_encoder_cache = {}
    assert not runner._has_cached_encoder_output("mm")
    assert runner._get_encoder_output_from_cache("mm") is None


def test_reset_encoder_cache_clears_all_tiers(monkeypatch):
    runner = _make_runner()
    runner.encoder_cache.encoder_outputs = {"mm": _tensor(1)}
    runner.storage.tmp_encoder_cache = {"mm2": _tensor(2)}
    runner.storage.cpu_encoder_cache = {"mm3": _tensor(3)}
    runner.storage._current_step_metadata = _metadata()
    runner.storage._req_refs = {"mm2": {"req-a"}}
    runner.storage._req_hash_index = {"req-a": {"mm2"}}
    runner.storage._pending_encoder_cache_copies = deque([(_tensor(4), MagicMock())])

    sync = MagicMock()
    monkeypatch.setattr(torch.npu, "synchronize", sync)

    runner.reset_encoder_cache()
    sync.assert_called_once_with()
    assert runner.encoder_cache.encoder_outputs == {}
    assert runner.storage.tmp_encoder_cache == {}
    assert runner.storage.cpu_encoder_cache == {}
    assert runner.storage._req_refs == {}
    assert runner.storage._req_hash_index == {}
    assert runner.storage._current_step_metadata is None
    assert len(runner.storage._pending_encoder_cache_copies) == 0

    runner.reset_encoder_cache()
    sync.assert_called_once_with()


def test_cache_encoder_outputs_batches_per_item():
    runner = _make_runner()
    hook = MagicMock()
    runner._cache_encoder_output = hook
    a, b = _tensor(1), _tensor(2)

    runner.cache_encoder_outputs(["mm-a", "mm-b"], [a, b])

    assert hook.call_count == 2
    assert hook.call_args_list[0].args == ("mm-a", a)
    assert hook.call_args_list[1].args == ("mm-b", b)


def _make_model_runner(flag: bool) -> NPUModelRunner:
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.use_score_encoder_cache = flag
    runner.model_state = SimpleNamespace(encoder_runner=MagicMock())
    return runner


def test_add_requests_tracks_new_mm_requests(monkeypatch):
    parent = MagicMock()
    monkeypatch.setattr(GPUModelRunner, "add_requests", parent)
    with_mm = SimpleNamespace(req_id="req-a", mm_features=[_mm_feature("h1")])
    without_mm = SimpleNamespace(req_id="req-b", mm_features=[])
    scheduler_output = SimpleNamespace(scheduled_new_reqs=[with_mm, without_mm])

    runner = _make_model_runner(flag=True)
    runner.add_requests(scheduler_output)

    parent.assert_called_once_with(scheduler_output)
    runner.model_state.encoder_runner.track_request.assert_called_once_with(
        "req-a", with_mm.mm_features)

    parent.reset_mock()
    runner = _make_model_runner(flag=False)
    runner.add_requests(scheduler_output)
    parent.assert_called_once_with(scheduler_output)
    runner.model_state.encoder_runner.track_request.assert_not_called()


def test_process_encoder_cache_scheduler_output_branches(monkeypatch):
    parent = MagicMock()
    monkeypatch.setattr(
        GPUModelRunner, "_process_encoder_cache_scheduler_output", parent)
    metadata = _metadata()
    delegated = SimpleNamespace(ec_manager_metadata=metadata)
    legacy = SimpleNamespace(ec_manager_metadata=None)

    runner = _make_model_runner(flag=False)
    runner._process_encoder_cache_scheduler_output(delegated)
    parent.assert_called_once_with(delegated)
    runner.model_state.encoder_runner.process_scheduler_metadata.assert_not_called()

    parent.reset_mock()
    runner = _make_model_runner(flag=True)
    runner._process_encoder_cache_scheduler_output(legacy)
    parent.assert_called_once_with(legacy)
    runner.model_state.encoder_runner.process_scheduler_metadata.assert_not_called()

    parent.reset_mock()
    runner = _make_model_runner(flag=True)
    runner._process_encoder_cache_scheduler_output(delegated)
    parent.assert_not_called()
    runner.model_state.encoder_runner.process_scheduler_metadata.assert_called_once_with(
        metadata)


def test_hook1_and_reset_delegate_when_enabled(monkeypatch):
    runner = _make_model_runner(flag=True)
    runner._on_request_state_removed("req-a")
    runner.model_state.encoder_runner.on_request_removed.assert_called_once_with(
        "req-a")

    parent_reset = MagicMock()
    monkeypatch.setattr(GPUModelRunner, "reset_encoder_cache", parent_reset)
    runner.reset_encoder_cache()
    parent_reset.assert_called_once_with()
    runner.model_state.encoder_runner.reset_encoder_cache.assert_called_once_with()

    runner = _make_model_runner(flag=False)
    runner._on_request_state_removed("req-b")
    runner.model_state.encoder_runner.on_request_removed.assert_not_called()
    runner.reset_encoder_cache()
    assert parent_reset.call_count == 2
    runner.model_state.encoder_runner.reset_encoder_cache.assert_not_called()
