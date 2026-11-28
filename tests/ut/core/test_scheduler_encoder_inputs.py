# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm import SamplingParams
from vllm.distributed.ec_transfer.ec_connector.mooncake.metadata import ECMooncakeLoadSpec
from vllm.distributed.ec_transfer.ec_connector.mooncake.state import SchedulerTransferTable
from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalKwargsItem, PlaceholderRange
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

from tests.ut.core.test_dyntra_lb_scheduler import create_dyntra_lb_scheduler, make_dyntra_test_config
from tests.ut.kv_offload.utils import EOS_TOKEN_ID, create_model_runner_output
from vllm_ascend.core.dyntra_lb_scheduler import AsyncDyntraLBScheduler, DyntraLBScheduler
from vllm_ascend.core.recompute_scheduler import (
    AsyncDyntraLBRecomputeScheduler,
    AsyncRecomputeScheduler,
    DyntraLBRecomputeScheduler,
    RecomputeScheduler,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.connector import AscendECMooncakeConnector
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.scheduler import AscendECMooncakeScheduler
from vllm_ascend.patch.platform.patch_balance_schedule import BalanceScheduler

PROMPT_TOKENS = 32
MAX_NUM_BATCHED_TOKENS = 64
ENCODER_BUDGET = 16
BLOCK_SIZE = 16

# Every scheduler that keeps a local copy of upstream schedule().
SCHEDULER_CLASSES = [
    DyntraLBScheduler,
    AsyncDyntraLBScheduler,
    RecomputeScheduler,
    AsyncRecomputeScheduler,
    DyntraLBRecomputeScheduler,
    AsyncDyntraLBRecomputeScheduler,
    BalanceScheduler,
]


def _make_scheduler(scheduler_cls: type[Scheduler]) -> Scheduler:
    config = make_dyntra_test_config(max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS, block_size=BLOCK_SIZE)
    config.kv_transfer_config = None
    config.cache_config.enable_prefix_caching = False
    config.scheduler_config.async_scheduling = issubclass(scheduler_cls, AsyncScheduler)
    scheduler = create_dyntra_lb_scheduler(config, scheduler_cls=scheduler_cls)
    # Supply the encoder budget directly: these CPU tests exercise the real
    # scheduler without loading a multimodal model or running its processor.
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=ENCODER_BUDGET)
    scheduler.max_num_encoder_input_tokens = ENCODER_BUDGET
    if isinstance(scheduler, BalanceScheduler):
        # Exercise the local schedule implementation, not its super fallback.
        scheduler._balance_enabled = True
        scheduler.balance_queue = [torch.tensor([0], dtype=torch.int)]
    return scheduler


def _image_request(images: list[tuple[str, int]], request_id: str = "image-request") -> Request:
    """Build a request whose prompt starts with the given (identifier, length) images."""
    sampling_params = SamplingParams(max_tokens=4)
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    features = []
    offset = 0
    for identifier, length in images:
        features.append(
            MultiModalFeatureSpec(
                data=MultiModalKwargsItem.dummy(),
                modality="image",
                identifier=identifier,
                mm_position=PlaceholderRange(offset=offset, length=length),
            )
        )
        offset += length
    return Request(
        request_id=request_id,
        prompt_token_ids=[1] * PROMPT_TOKENS,
        sampling_params=sampling_params,
        pooling_params=None,
        mm_features=features,
    )


def _mooncake_consumer() -> tuple[AscendECMooncakeConnector, AscendECMooncakeScheduler]:
    """Build the real scheduler-side Mooncake consumer without a control plane."""
    consumer = AscendECMooncakeScheduler.__new__(AscendECMooncakeScheduler)
    consumer._is_producer = False
    consumer._is_consumer = True
    consumer._local_cache = set()
    consumer._failed_saves = set()
    consumer._store_candidates = []
    consumer._cross_encoder_cache = False
    consumer._pushes_to_prepare = {}
    consumer._prepared_push_transfer_ids = set()
    consumer._scheduler_pending_work = False
    consumer._push_wait_timeout = 5.0
    consumer._transfers = SchedulerTransferTable(1 << 20, 300)
    consumer._drain_pending = False
    consumer._event_ready_shards = OrderedDict()
    consumer._pending_cancels = {}
    consumer._control_executor = MagicMock()
    consumer._control_addr = ""
    consumer._drain_push_notifications = lambda: None
    consumer._note_awaiting_push = lambda *args, **kwargs: None
    consumer._store_get_enabled = False
    consumer._store_get_states = {}
    consumer._store_get_sizes = {}
    consumer._store_get_waiters = {}
    consumer._store_get_to_dispatch = {}
    consumer._store_get_to_consume = set()
    consumer._store_get_to_recompute = set()
    consumer._store_get_to_discard = set()
    connector = AscendECMooncakeConnector.__new__(AscendECMooncakeConnector)
    connector._scheduler = consumer
    connector._worker = None
    connector._closed = False
    return connector, consumer


def _offer_from_encoder(consumer: AscendECMooncakeScheduler, request: Request) -> None:
    """Record that the Encoder pushed every image of the request and can serve a load."""
    request.ec_transfer_params = {
        "ec_items": [
            {"mm_hash": feature.identifier, "transfer_id": f"xfer-{feature.identifier}"}
            for feature in request.mm_features
        ]
    }
    for feature in request.mm_features:
        consumer._transfers.observe_ready(
            ECMooncakeLoadSpec(
                mm_hash=feature.identifier,
                nbytes=64,
                shape=(feature.mm_position.length, 4),
                dtype="float16",
                transfer_id=f"xfer-{feature.identifier}",
            ),
            deadline=1e18,
        )


def _finish_prefill_chunk(scheduler: Scheduler, request: Request, scheduler_output) -> None:
    runner_output = create_model_runner_output([request])
    runner_output.sampled_token_ids = [[]]
    scheduler.update_from_output(scheduler_output, runner_output)


@pytest.mark.parametrize("scheduler_cls", SCHEDULER_CLASSES)
def test_waiting_duplicate_encoder_input_keeps_cache_reference(scheduler_cls):
    scheduler = _make_scheduler(scheduler_cls)
    request = _image_request([("shared", 4), ("shared", 4)])
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.scheduled_encoder_inputs == {request.request_id: [0]}
    manager = scheduler.encoder_cache_manager
    assert manager.get_cached_input_ids(request) == {0, 1}
    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    assert manager.get_cached_input_ids(request) == set()


@pytest.mark.parametrize("scheduler_cls", SCHEDULER_CLASSES)
def test_running_duplicate_encoder_input_keeps_cache_reference(scheduler_cls):
    scheduler = _make_scheduler(scheduler_cls)
    scheduler.max_num_scheduled_tokens = 2
    request = _image_request([("first", 2), ("shared", 2), ("shared", 2)])
    scheduler.add_request(request)
    first = scheduler.schedule()
    assert first.scheduled_encoder_inputs == {request.request_id: [0]}
    _finish_prefill_chunk(scheduler, request, first)
    assert request in scheduler.running

    scheduler.max_num_scheduled_tokens = 4
    with patch.object(scheduler, "_allocate_encoder_inputs", wraps=scheduler._allocate_encoder_inputs) as allocate:
        output = scheduler.schedule()

    assert output.scheduled_encoder_inputs == {request.request_id: [1]}
    allocate.assert_called_once_with(request, [1], [2])
    assert {1, 2} <= scheduler.encoder_cache_manager.get_cached_input_ids(request)


@pytest.mark.parametrize("scheduler_cls", SCHEDULER_CLASSES)
def test_consumed_cpu_encoder_input_is_released(scheduler_cls):
    """The worker drops its pixels once the placeholder range is confirmed."""
    scheduler = _make_scheduler(scheduler_cls)
    request = _image_request([("image", 8)])
    payload = request.mm_features[0].data
    scheduler.add_request(request)

    prefill = scheduler.schedule()
    assert prefill.num_scheduled_tokens == {request.request_id: PROMPT_TOKENS}
    assert prefill.scheduled_encoder_inputs == {request.request_id: [0]}
    # The placeholder range is still in flight, so the worker keeps its payload.
    assert prefill.encoder_input_updates == {}

    scheduler.update_from_output(prefill, create_model_runner_output([request]))
    decode = scheduler.schedule()
    assert decode.encoder_input_updates == {request.request_id: {0: None}}
    # EngineCore keeps the original payload for replay.
    assert request.mm_features[0].data is payload

    scheduler.update_from_output(decode, create_model_runner_output([request]))
    assert scheduler.schedule().encoder_input_updates == {}


@pytest.mark.parametrize("scheduler_cls", SCHEDULER_CLASSES)
def test_preempted_request_reinstalls_cpu_encoder_input(scheduler_cls):
    """A resumed request gets its pixels back and releases them again."""
    scheduler = _make_scheduler(scheduler_cls)
    request = _image_request([("image", 8)])
    payload = request.mm_features[0].data
    scheduler.add_request(request)

    prefill = scheduler.schedule()
    scheduler.update_from_output(prefill, create_model_runner_output([request]))
    decode = scheduler.schedule()
    assert decode.encoder_input_updates == {request.request_id: {0: None}}
    scheduler.update_from_output(decode, create_model_runner_output([request]))

    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    resumed = scheduler.schedule()
    assert resumed.scheduled_new_reqs[0].mm_features[0].data is payload
    assert resumed.encoder_input_updates == {}

    scheduler.update_from_output(resumed, create_model_runner_output([request]))
    assert scheduler.schedule().encoder_input_updates == {request.request_id: {0: None}}


def _loads(scheduler_output) -> list[str]:
    return [spec.mm_hash for spec in scheduler_output.ec_connector_metadata.loads]


@pytest.mark.parametrize("scheduler_cls", SCHEDULER_CLASSES)
def test_running_request_reloads_evicted_encoder_output(scheduler_cls):
    """An output evicted between prefill chunks is loaded before the chunk uses it.

    Mooncake still reports the request's own reservation as available, so
    without the running check the chunk is scheduled as an external load that
    no worker load serves, and the model runner raises "Encoder cache miss".
    """
    scheduler = _make_scheduler(scheduler_cls)
    connector, consumer = _mooncake_consumer()
    scheduler.ec_connector = connector
    images = [("first", 2), ("second", 2)]
    # An earlier request left both outputs in the local encoder cache.
    earlier = _image_request(images, request_id="earlier-request")
    for input_id in range(len(images)):
        scheduler.encoder_cache_manager.allocate(earlier, input_id)
    scheduler.encoder_cache_manager.free(earlier)
    consumer._local_cache.update(identifier for identifier, _ in images)
    request = _image_request(images)
    _offer_from_encoder(consumer, request)
    scheduler.add_request(request)

    scheduler.max_num_scheduled_tokens = 2
    first = scheduler.schedule()
    assert first.num_scheduled_tokens == {request.request_id: 2}
    assert _loads(first) == []
    _finish_prefill_chunk(scheduler, request, first)

    # Another allocation evicts "second" before the next chunk reaches it.
    assert scheduler.encoder_cache_manager.can_allocate(
        _image_request([("other", ENCODER_BUDGET - 2)], request_id="other-request"), 0, ENCODER_BUDGET, 0
    )
    connector.build_connector_meta(
        SimpleNamespace(
            free_encoder_mm_hashes=scheduler.encoder_cache_manager.get_freed_mm_hashes(),
            preempted_req_ids=set(),
        )
    )

    scheduler.max_num_scheduled_tokens = 4
    deferred = scheduler.schedule()
    assert request.request_id not in deferred.num_scheduled_tokens
    assert _loads(deferred) == ["second"]

    consumer._transfers.complete_load("second")
    resumed = scheduler.schedule()
    assert resumed.num_scheduled_tokens == {request.request_id: 4}
    assert request.request_id not in resumed.scheduled_encoder_inputs


@pytest.mark.parametrize("scheduler_cls", SCHEDULER_CLASSES)
def test_resumed_waiting_request_loads_encoder_output_first(scheduler_cls):
    """A waiting request with computed tokens, as after a streaming-input chunk."""
    scheduler = _make_scheduler(scheduler_cls)
    connector, consumer = _mooncake_consumer()
    scheduler.ec_connector = connector
    request = _image_request([("first", 2), ("second", 2)])
    _offer_from_encoder(consumer, request)
    scheduler.add_request(request)
    request.num_computed_tokens = 2

    deferred = scheduler.schedule()
    assert request.request_id not in deferred.num_scheduled_tokens
    assert request.status == RequestStatus.WAITING
    assert _loads(deferred) == ["second"]

    consumer._transfers.complete_load("second")
    resumed = scheduler.schedule()
    assert resumed.num_scheduled_tokens == {request.request_id: PROMPT_TOKENS - 2}
    assert request.request_id not in resumed.scheduled_encoder_inputs
