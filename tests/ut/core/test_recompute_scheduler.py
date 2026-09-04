# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_group_id,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.engine import EngineCoreOutput, FinishReason
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)
from vllm.v1.metrics.stats import RequestSpecDecodeMetrics
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from vllm_ascend.core.recompute_scheduler import (
    RecomputeReqInfo,
    RecomputeScheduler,
)
from vllm_ascend.utils import vllm_version_is


def test_add_request_does_not_inject_placeholder_spec_tokens():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.requests = {}
    scheduler.log_stats = False
    scheduler.connector = None
    if not vllm_version_is("0.27.1"):
        # vllm main: Scheduler.add_request reads spec_decode_metrics_level.
        scheduler.spec_decode_metrics_level = "none"

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


def test_update_from_output_returns_per_request_spec_decode_metrics():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.defer_block_free = False
    scheduler.perf_metrics = None
    scheduler.log_stats = False
    scheduler.num_sampled_tokens_per_step = 1
    scheduler.spec_decode_metrics_level = "detailed"
    scheduler.enable_return_routed_experts = False
    scheduler.connector = None
    scheduler.grammar_compile_error_reqs = []
    scheduler.recompute_kv_load_failures = True
    scheduler.finished_req_ids_dict = None
    scheduler.structured_output_manager = MagicMock()
    scheduler.structured_output_manager.should_advance.return_value = False
    scheduler.kv_cache_manager = MagicMock()
    scheduler.kv_cache_manager.take_events.return_value = None
    scheduler.kv_cache_manager.estimate_cached_tokens.return_value = 0
    scheduler.kv_event_publisher = MagicMock()
    scheduler.make_stats = MagicMock(return_value=None)
    scheduler._handle_stopped_request = MagicMock(return_value=False)

    request = Request(
        request_id="spec-request",
        prompt_token_ids=[1],
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
    )
    request.status = RequestStatus.RUNNING
    request.num_in_flight_tokens = 4
    request.num_computed_tokens = 1
    request.spec_decode_metrics = RequestSpecDecodeMetrics.new(3)
    scheduler.requests = {request.request_id: request}
    scheduler.running = [request]
    scheduler.waiting = MagicMock()
    scheduler.skipped_waiting = MagicMock()

    def finish_request(req, token_ids, *, is_stale):
        req.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return token_ids, True

    scheduler._update_request_with_output = MagicMock(side_effect=finish_request)

    scheduler_output = SimpleNamespace(
        recomputed_reqs=None,
        num_scheduled_tokens={request.request_id: 4},
        total_num_scheduled_tokens=4,
        scheduled_spec_decode_tokens={request.request_id: [11, 12, 13]},
        num_invalid_spec_tokens={request.request_id: 1},
    )
    model_runner_output = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[[11, 12, 99]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    outputs = scheduler.update_from_output(scheduler_output, model_runner_output)

    output = outputs[0].outputs[0]
    assert output.spec_decode_metrics is request.spec_decode_metrics
    assert output.spec_decode_metrics.to_dict() == {
        "mean_acceptance_length": 3.0,
        "draft_acceptance_rate": 1.0,
        "acceptance_histogram": [0, 0, 1, 0],
        "num_spec_steps": 1,
        "num_accepted_draft_tokens": 2,
        "num_draft_tokens": 2,
        "num_spec_tokens": 3,
        "per_step_accepted": [2],
        "per_step_drafted": [2],
    }


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


def test_truncate_computed_blocks_supports_legacy_short_mamba_group():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    mamba_block = MagicMock()
    attention_blocks = [MagicMock(), MagicMock()]
    blocks = SimpleNamespace(blocks=([mamba_block], attention_blocks))
    kv_cache_manager = SimpleNamespace(
        truncate_computed_blocks=MagicMock(),
        coordinator=SimpleNamespace(
            single_type_managers=[
                SimpleNamespace(block_size=4),
                SimpleNamespace(block_size=4),
            ]
        ),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=MagicMock(spec=MambaSpec)),
                SimpleNamespace(kv_cache_spec=MagicMock()),
            ]
        ),
        create_kv_cache_blocks=MagicMock(side_effect=lambda value: value),
    )
    scheduler.kv_cache_manager = kv_cache_manager

    truncated = scheduler._truncate_computed_blocks_for_connector(blocks, 8)

    assert truncated == ([mamba_block], attention_blocks)
    kv_cache_manager.truncate_computed_blocks.assert_not_called()


def test_dsv4_decode_node_observes_real_dense_local_cache_hit():
    register_all_kvcache_specs(MagicMock())
    init_none_hash(sha256)
    hash_block_size = 4
    kv_cache_config = KVCacheConfig(
        num_blocks=800,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["dense_mla"],
                MLAAttentionSpec(
                    block_size=256,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.uint8,
                    compress_ratio=4,
                    model_version="deepseek_v4",
                ),
            ),
            KVCacheGroupSpec(
                ["swa_tail"],
                SlidingWindowMLASpec(
                    block_size=64,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.uint8,
                    sliding_window=128,
                    model_version="deepseek_v4",
                ),
            ),
            KVCacheGroupSpec(
                ["c4_state"],
                SlidingWindowMLASpec(
                    block_size=4,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.uint8,
                    sliding_window=8,
                ),
            ),
            KVCacheGroupSpec(
                ["c128_state"],
                SlidingWindowMLASpec(
                    block_size=8,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.uint8,
                    sliding_window=128,
                ),
            ),
        ],
    )
    manager = KVCacheManager(
        kv_cache_config,
        max_model_len=8192,
        scheduler_block_size=256,
        hash_block_size=hash_block_size,
        enable_caching=True,
    )
    block_hasher = get_request_block_hasher(hash_block_size, sha256)

    def make_request(request_id: str, num_tokens: int) -> Request:
        return Request(
            request_id=request_id,
            prompt_token_ids=[0] * num_tokens,
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            block_hasher=block_hasher,
        )

    fill = make_request("fill", 1024)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(fill)
    assert num_computed == 0
    assert (
        manager.allocate_slots(
            fill,
            fill.num_tokens,
            num_new_computed_tokens=0,
            new_computed_blocks=computed_blocks,
        )
        is not None
    )
    manager.free(fill)

    non_dense_block_ids = {
        block.block_id
        for block in manager.block_pool.blocks
        if block.block_hash is not None and get_group_id(block.block_hash) in {1, 2, 3}
    }
    assert non_dense_block_ids
    manager.evict_blocks(non_dense_block_ids)

    replay = make_request("replay", 1280)
    replay.kv_transfer_params = {"do_remote_prefill": True}
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.kv_cache_manager = manager

    _, num_local, shared_prefix_boundary, hit_diverged = scheduler._get_computed_blocks_for_connector(replay)

    assert num_local == 1024
    assert shared_prefix_boundary == 0
    assert hit_diverged
