# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from vllm import SamplingParams
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import FinishReason
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from tests.ut.kv_offload.utils import create_model_runner_output, create_scheduler, create_vllm_config
from vllm_ascend.ec_manager.score_ec_manager import ScoreEncoderCacheManager


def _make_scheduler(npu_slots: int = 8) -> Scheduler:
    config = create_vllm_config(max_num_batched_tokens=32, block_size=16)
    config.kv_transfer_config = None
    config.cache_config.enable_prefix_caching = False
    scheduler = create_scheduler(config, num_blocks=32)
    # Supply encoder metadata directly: these CPU tests exercise the real
    # scheduler without loading a multimodal model or running its processor.
    config.model_config.hf_config.vision_config = SimpleNamespace(num_heads=1, hidden_size=1, intermediate_size=1)
    config.ec_manager_config.manager_config = {"cpu_cache_slots": 8}
    scheduler.encoder_cache_manager = ScoreEncoderCacheManager(npu_slots, config)
    scheduler.max_num_encoder_input_tokens = 32
    return scheduler


def _request(request_id: str, items: list[tuple[str, list[bool]]]) -> Request:
    features = []
    offset = 0
    for identifier, is_embed in items:
        features.append(
            MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier=identifier,
                mm_position=PlaceholderRange(offset=offset, length=len(is_embed), is_embed=torch.tensor(is_embed)),
            )
        )
        offset += len(is_embed)
    return Request(
        request_id=request_id,
        prompt_token_ids=[1] * max(offset, 1),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        mm_features=features,
    )


def _seed(manager: ScoreEncoderCacheManager, identifier: str, count: int) -> None:
    owner = _request(f"owner-{identifier}", [(identifier, [True] * count + [False])])
    assert manager.can_allocate(owner, 0, 32, 0)
    manager.allocate(owner, 0)
    manager.free(owner)


def _seed_residency(scheduler: Scheduler, residency: str) -> None:
    manager = scheduler.encoder_cache_manager
    _seed(manager, "shared", 3)
    if residency in ("cpu-npu", "npu"):
        promoter = _request("promoter", [("shared", [True] * 3)])
        assert manager.check_and_update_cache(promoter, 0)
        manager.free(promoter)
        assert "shared" in manager.npu_cache
    if residency == "npu":
        replacement = _request("replacement", [("replacement", [True] * 8)])
        assert manager.can_allocate(replacement, 0, 32, 0)
        assert "shared" not in manager.cpu_cache
    manager.get_manager_metadata()
    manager._check_invariant()


def _capacity(manager: ScoreEncoderCacheManager) -> tuple[int, int, int, int]:
    return (
        manager.cpu_num_free_slots,
        manager.cpu_num_freeable_slots,
        manager.npu_num_free_slots,
        manager.npu_num_freeable_slots,
    )


def _assert_error_and_finish(scheduler: Scheduler, request: Request, scheduler_output) -> None:
    assert scheduler_output.total_num_scheduled_tokens == 0
    assert not scheduler_output.scheduled_encoder_inputs
    outputs = scheduler.update_from_output(scheduler_output, ModelRunnerOutput(req_ids=[], req_id_to_index={}))
    assert request.status == RequestStatus.FINISHED_ERROR
    assert request.request_id not in scheduler.requests
    assert request.request_id not in scheduler.encoder_cache_manager.request_cached_ids
    assert not scheduler.running
    assert not scheduler.waiting
    output = outputs[request.client_index].outputs[0]
    assert output.request_id == request.request_id
    assert output.finish_reason == FinishReason.ERROR


def _assert_success_and_finish(scheduler: Scheduler, request: Request, scheduler_output) -> None:
    assert scheduler_output.num_scheduled_tokens[request.request_id] == request.num_prompt_tokens
    outputs = scheduler.update_from_output(scheduler_output, create_model_runner_output([request]))
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert request.request_id not in scheduler.requests
    assert request.request_id not in scheduler.encoder_cache_manager.request_cached_ids
    assert outputs[request.client_index].outputs[0].finish_reason == FinishReason.LENGTH
    scheduler.encoder_cache_manager._check_invariant()


@pytest.mark.parametrize("residency", ["cpu", "cpu-no-promotion", "cpu-npu", "npu"])
@pytest.mark.parametrize("requested_count", [2, 4])
def test_mismatched_cache_hit_is_request_error(residency: str, requested_count: int):
    scheduler = _make_scheduler(npu_slots=1 if residency == "cpu-no-promotion" else 8)
    _seed_residency(scheduler, residency)
    manager = scheduler.encoder_cache_manager
    capacity_before = _capacity(manager)
    request = _request("mismatched", [("shared", [True] * requested_count + [False])])
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()

    _assert_error_and_finish(scheduler, request, scheduler_output)
    assert manager.cached["shared"] == set()
    assert _capacity(manager) == capacity_before
    assert not scheduler_output.ec_manager_metadata.promoting_mm_hashes
    assert not scheduler_output.ec_manager_metadata.cpu_get_encoder_mm_hashes
    manager._check_invariant()

    # The rejected UUID does not poison later requests, including reuse of the
    # original entry with the correct embedding count.
    healthy = _request("healthy", [("shared", [True, False, True, False, True, False, False])])
    scheduler.add_request(healthy)
    next_output = scheduler.schedule()
    assert not next_output.scheduled_encoder_inputs
    _assert_success_and_finish(scheduler, healthy, next_output)
    assert manager.cached["shared"] == set()


@pytest.mark.parametrize("residency", ["cpu", "cpu-no-promotion", "cpu-npu", "npu"])
def test_matching_embed_count_accepts_different_placeholder_layout(residency: str):
    scheduler = _make_scheduler(npu_slots=1 if residency == "cpu-no-promotion" else 8)
    _seed_residency(scheduler, residency)
    request = _request("compatible", [("shared", [True, False, True, False, True, False, False])])
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()

    assert not scheduler_output.scheduled_encoder_inputs
    metadata = scheduler_output.ec_manager_metadata
    if residency == "cpu":
        assert metadata.promoting_mm_hashes == ["shared"]
    elif residency == "cpu-no-promotion":
        assert metadata.cpu_get_encoder_mm_hashes == ["shared"]
        assert "shared" not in scheduler.encoder_cache_manager.npu_cache
    _assert_success_and_finish(scheduler, request, scheduler_output)


def test_mismatch_preserves_another_request_reference():
    scheduler = _make_scheduler()
    manager = scheduler.encoder_cache_manager
    owner = _request("active-owner", [("shared", [True] * 3)])
    assert manager.can_allocate(owner, 0, 32, 0)
    manager.allocate(owner, 0)
    capacity_before = _capacity(manager)
    request = _request("mismatched", [("shared", [True] * 4)])
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()

    _assert_error_and_finish(scheduler, request, scheduler_output)
    assert manager.cached["shared"] == {owner.request_id}
    assert manager.request_cached_ids == {owner.request_id: {0}}
    assert _capacity(manager) == capacity_before
    manager._check_invariant()
    manager.free(owner)
    manager._check_invariant()


@pytest.mark.parametrize("npu_slots", [1, 8])
def test_mismatch_releases_an_earlier_cache_hit(npu_slots: int):
    scheduler = _make_scheduler(npu_slots=npu_slots)
    manager = scheduler.encoder_cache_manager
    _seed(manager, "valid", 2)
    _seed(manager, "mismatch", 3)
    capacity_before = _capacity(manager)
    request = _request("partial-hit", [("valid", [True, True, False]), ("mismatch", [True] * 4)])
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()

    assert manager.cached["valid"] == {request.request_id}
    assert manager.cpu_num_freeable_slots == capacity_before[1] - 2
    _assert_error_and_finish(scheduler, request, scheduler_output)
    assert manager.cached["valid"] == set()
    assert manager.cached["mismatch"] == set()
    assert _capacity(manager)[:2] == capacity_before[:2]
    assert manager.npu_num_freeable_slots == capacity_before[3]
    metadata = scheduler_output.ec_manager_metadata
    if npu_slots == 8:
        # A valid prior hit can promote before the later input is rejected.
        # Keep that reusable entry while restoring its reclaimable capacity.
        assert metadata.promoting_mm_hashes == ["valid"]
        assert manager.npu_num_free_slots == capacity_before[2] - 2
        assert manager.npu_freeable["valid"].num_embeds == 2
    else:
        assert metadata.cpu_get_encoder_mm_hashes == ["valid"]
        assert manager.npu_num_free_slots == capacity_before[2]
    manager._check_invariant()


def test_same_request_duplicate_identifier_rejects_different_embed_counts():
    scheduler = _make_scheduler()
    manager = scheduler.encoder_cache_manager
    capacity_before = _capacity(manager)
    request = _request("duplicate", [("shared", [True, True, False]), ("shared", [True, True, True])])
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()

    _assert_error_and_finish(scheduler, request, scheduler_output)
    assert not manager.cached
    assert _capacity(manager) == capacity_before
    manager._check_invariant()
