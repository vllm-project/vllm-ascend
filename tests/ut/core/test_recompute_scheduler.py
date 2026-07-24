# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.engine import EngineCoreOutput, FinishReason
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm.v1.request import Request, RequestStatus

from vllm_ascend.core.recompute_scheduler import (
    RecomputeReqInfo,
    RecomputeScheduler,
)


def _make_connector_lookup_scheduler(
    monkeypatch: pytest.MonkeyPatch,
    per_group_hits: tuple[int, int],
    *,
    is_v0251: bool = False,
) -> tuple[RecomputeScheduler, SimpleNamespace]:
    monkeypatch.setattr(
        "vllm_ascend.core.recompute_scheduler.vllm_version_is",
        lambda version: is_v0251 and version == "0.25.1",
    )
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    coordinator = HybridKVCacheCoordinator.__new__(HybridKVCacheCoordinator)
    full_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=64,
        dtype=torch.float16,
    )
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=((1,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="none",
    )
    coordinator.attention_groups = [
        (full_spec, [0], MagicMock(), False),
        (mamba_spec, [1], MagicMock(), False),
    ]
    coordinator.dcp_world_size = 1
    coordinator.pcp_world_size = 1
    coordinator.find_longest_cache_hit_per_group = MagicMock(return_value=((["fa"], ["mamba"]), per_group_hits))
    scheduler.kv_cache_manager = SimpleNamespace(
        coordinator=coordinator,
        kv_cache_config=SimpleNamespace(has_mamba_layers=True),
        enable_caching=True,
        empty_kv_cache_blocks=(),
        log_stats=False,
        create_kv_cache_blocks=MagicMock(side_effect=lambda blocks: blocks),
        get_computed_blocks=MagicMock(return_value=((["common"], []), 16) if is_v0251 else ((["common"], []), 16, 8)),
    )
    request = SimpleNamespace(
        block_hashes=[],
        num_tokens=129,
        num_preemptions=0,
        skip_reading_prefix_cache=False,
    )
    return scheduler, request


def test_connector_lookup_uses_full_attention_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler, request = _make_connector_lookup_scheduler(monkeypatch, (64, 32))

    computed, local_tokens, shared_prefix_boundary, hit_diverged = scheduler.get_computed_blocks_for_connector(request)

    assert computed == (["fa"], ["mamba"])
    assert local_tokens == 64
    assert shared_prefix_boundary == 0
    assert hit_diverged is True
    assert scheduler.kv_cache_manager.coordinator.num_uncached_common_prefix_tokens == 0
    scheduler.kv_cache_manager.get_computed_blocks.assert_not_called()


def test_connector_lookup_reconciles_deeper_sparse_group(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler, request = _make_connector_lookup_scheduler(monkeypatch, (32, 64))
    scheduler.kv_cache_manager.get_computed_blocks.return_value = ((["common"], []), 32, 24)

    computed, local_tokens, shared_prefix_boundary, hit_diverged = scheduler.get_computed_blocks_for_connector(request)

    assert computed == (["common"], [])
    assert local_tokens == 32
    assert shared_prefix_boundary == 24
    assert hit_diverged is False
    scheduler.kv_cache_manager.get_computed_blocks.assert_called_once_with(request)


def test_connector_lookup_falls_back_without_full_attention_group(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler, request = _make_connector_lookup_scheduler(monkeypatch, (16, 16))
    scheduler.kv_cache_manager.coordinator.attention_groups = [
        scheduler.kv_cache_manager.coordinator.attention_groups[1]
    ]

    computed, local_tokens, shared_prefix_boundary, hit_diverged = scheduler.get_computed_blocks_for_connector(request)

    assert computed == (["common"], [])
    assert local_tokens == 16
    assert shared_prefix_boundary == 8
    assert hit_diverged is False
    scheduler.kv_cache_manager.get_computed_blocks.assert_called_once_with(request)


def test_connector_lookup_falls_back_for_hybrid_context_parallelism(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler, request = _make_connector_lookup_scheduler(monkeypatch, (64, 32))
    scheduler.kv_cache_manager.coordinator.pcp_world_size = 2

    computed, local_tokens, shared_prefix_boundary, hit_diverged = scheduler.get_computed_blocks_for_connector(request)

    assert computed == (["common"], [])
    assert local_tokens == 16
    assert shared_prefix_boundary == 8
    assert hit_diverged is False
    scheduler.kv_cache_manager.get_computed_blocks.assert_called_once_with(request)
    scheduler.kv_cache_manager.coordinator.find_longest_cache_hit_per_group.assert_not_called()


def test_reconciled_lookup_normalizes_v0251_result(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler, request = _make_connector_lookup_scheduler(monkeypatch, (16, 16), is_v0251=True)

    computed, local_tokens, shared_prefix_boundary = scheduler._get_reconciled_computed_blocks(request)

    assert computed == (["common"], [])
    assert local_tokens == 16
    assert shared_prefix_boundary == 0


def test_add_request_does_not_inject_placeholder_spec_tokens():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.requests = {}
    scheduler.log_stats = False
    scheduler.connector = None

    enqueued_requests = []

    def enqueue_waiting_request(self, request):
        enqueued_requests.append(request)

    scheduler._enqueue_waiting_request = MethodType(enqueue_waiting_request, scheduler)

    request = Request(
        request_id="pd-consumer-first-step",
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
    )

    scheduler.add_request(request)

    assert enqueued_requests == [request]
    assert scheduler.requests[request.request_id] is request
    assert request.spec_token_ids == []
    assert request.num_tokens_with_spec == request.num_tokens


def test_recompute_notification_precedes_regular_output():
    scheduler_output = SimpleNamespace(
        recomputed_reqs=[
            RecomputeReqInfo(
                request_id="recomputed-request",
                output_token_ids=[],
                client_index=0,
            )
        ]
    )
    outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)

    RecomputeScheduler._add_recomputed_outputs(scheduler_output, outputs)
    outputs[0].append(
        EngineCoreOutput(
            request_id="regular-request",
            new_token_ids=[1],
        )
    )

    output = outputs[0][0]
    assert output.request_id == "recomputed-request"
    assert output.finish_reason == FinishReason.STOP
    assert output.stop_reason == "recomputed"
    assert outputs[0][1].request_id == "regular-request"


def test_finish_recomputed_request_uses_normal_abort_cleanup():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    request = Request(
        request_id="fallback-recomputed-request",
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
    )
    request.status = RequestStatus.RUNNING

    # The fallback victim has already been popped from the running queue.
    scheduler.requests = {request.request_id: request}
    scheduler.running = []
    scheduler.waiting = MagicMock()
    scheduler.skipped_waiting = MagicMock()
    scheduler._inflight_prefills = {request}
    scheduler._connector_finished = MagicMock(return_value=(False, None))
    scheduler.encoder_cache_manager = MagicMock()
    scheduler.ec_connector = None
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler._free_request_blocks = MagicMock()

    recomputed_reqs: list[RecomputeReqInfo] = []
    scheduler._finish_recomputed_request(request, recomputed_reqs)

    assert request.status == RequestStatus.FINISHED_ABORTED
    assert request not in scheduler._inflight_prefills
    assert request.request_id not in scheduler.requests
    assert request.request_id in scheduler.finished_req_ids
    scheduler._connector_finished.assert_called_once_with(request)
    scheduler.encoder_cache_manager.free.assert_called_once_with(request)
    scheduler._free_request_blocks.assert_called_once_with(request)
    assert recomputed_reqs == [
        RecomputeReqInfo(
            request_id=request.request_id,
            output_token_ids=request.output_token_ids,
            client_index=request.client_index,
        )
    ]
