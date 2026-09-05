from unittest.mock import MagicMock, patch

import pytest

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    AscendStoreKVConnectorWorkerMetadata,
    ReqMeta,
    RequestTracker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.pool.scheduler import MPKVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.registration import (
    SchedulerIdentity,
    SchedulerRegistration,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.scheduler_view import (
    BlocksView,
    CachedReqsView,
    ConnectorOutputView,
    RequestIdView,
    RequestView,
    ScheduledNewReqPayload,
    SchedulerOutputView,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler

# isort: on

POOL_SCHEDULER_MODULE = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler"


@pytest.fixture(autouse=True)
def _patch_pool_scheduler_importlib():
    """KVPoolScheduler.__init__ loads the backend module dynamically; point
    importlib at a MagicMock so no real backend is imported."""
    with patch(f"{POOL_SCHEDULER_MODULE}.importlib") as mock_importlib:
        mock_importlib.import_module.return_value = MagicMock()
        yield


def _make_config(kv_role="kv_producer", extra_config=None, block_size=16):
    config = MagicMock()
    config.kv_transfer_config.kv_role = kv_role
    config.kv_transfer_config.engine_id = "engine-0"
    config.kv_transfer_config.kv_connector = "AscendStoreConnector"
    config.kv_transfer_config.kv_connector_extra_config = extra_config or {}
    config.kv_transfer_config.get_from_extra_config.return_value = True
    config.parallel_config.data_parallel_rank = 0
    config.parallel_config.data_parallel_index = 0
    config.parallel_config.data_parallel_size = 1
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.decode_context_parallel_size = 1
    config.parallel_config.tensor_parallel_size = 1
    config.parallel_config.pipeline_parallel_size = 1
    config.parallel_config.rank = 0
    config.parallel_config.world_size = 1
    config.cache_config.block_size = block_size
    config.cache_config.hash_block_size = block_size
    config.cache_config.prefix_match_unit = None
    config.model_config.model = "org/llama-7b"
    config.model_config.max_model_len = 1024
    config.model_config.use_mla = False
    config.model_config.hf_text_config = MagicMock(spec=[])
    config.model_config.hf_config = config.model_config.hf_text_config
    config.model_config.get_total_num_kv_heads.return_value = 1
    config.model_config.get_num_layers.return_value = 2
    config.scheduler_config.disable_hybrid_kv_cache_manager = False
    config.speculative_config = None
    config.kv_events_config = None
    return config


def _make_scheduler(extra_config=None, kv_role="kv_producer", block_size=16) -> tuple[MPKVPoolScheduler, MagicMock]:
    lookup_handler = MagicMock(return_value=0)
    registration = SchedulerRegistration.create(_make_config(kv_role, extra_config, block_size), None, 0)
    return MPKVPoolScheduler(registration, lookup_handler), lookup_handler


def _make_request(num_tokens=64, num_computed=0) -> MagicMock:
    request = MagicMock()
    request.prompt_token_ids = list(range(num_tokens))
    request.num_tokens = num_tokens
    request.request_id = "r1"
    request.block_hashes = [bytes([index]) * 32 for index in range(num_tokens // 16)]
    return request


def test_mp_scheduler_reuses_original_business_method() -> None:
    assert MPKVPoolScheduler.get_num_new_matched_tokens is KVPoolScheduler.get_num_new_matched_tokens


def test_mp_scheduler_reads_use_layerwise_from_extra_config() -> None:
    scheduler, _ = _make_scheduler(extra_config={"backend": "mooncake", "use_layerwise": True})
    assert scheduler.use_layerwise is True
    assert scheduler.use_layerwise_transfer is False


def test_mp_scheduler_defaults_to_non_layerwise() -> None:
    scheduler, _ = _make_scheduler()
    assert scheduler.use_layerwise is False


def test_mp_scheduler_consumer_no_load_skips_lookup() -> None:
    scheduler, lookup_handler = _make_scheduler(kv_role="kv_consumer")
    assert scheduler.get_num_new_matched_tokens(_make_request(), 0) == (0, False)
    lookup_handler.assert_not_called()


def test_mp_scheduler_too_short_prompt_skips_lookup() -> None:
    scheduler, lookup_handler = _make_scheduler(block_size=64)
    request = _make_request(num_tokens=32)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    lookup_handler.assert_not_called()


def test_mp_scheduler_full_hbm_hit_skips_external_lookup() -> None:
    scheduler, lookup_handler = _make_scheduler()
    request = _make_request()
    assert scheduler.get_num_new_matched_tokens(request, 64) == (0, False)
    lookup_handler.assert_not_called()


def test_mp_scheduler_hit_returns_need_to_allocate_and_records_load_spec() -> None:
    scheduler, lookup_handler = _make_scheduler()
    lookup_handler.return_value = 48
    request = _make_request()

    need, is_async = scheduler.get_num_new_matched_tokens(request, 16)

    assert (need, is_async) == (32, False)
    load_spec = scheduler.load_specs["r1"]
    assert load_spec.vllm_cached_tokens == 16
    assert load_spec.kvpool_cached_tokens == 48
    assert load_spec.kvpool_store_skip_tokens == 48
    # The bridge hides the zmq client: the original client interface is served
    # by the in-process lookup handler with the same arguments.
    lookup_handler.assert_called_once_with(
        SchedulerIdentity("engine-0", 0),
        64,
        request.block_hashes,
        [0],
        False,
        16,
    )


def test_mp_scheduler_full_external_hit_returns_all_but_one_token() -> None:
    scheduler, lookup_handler = _make_scheduler()
    lookup_handler.return_value = 64

    need, _ = scheduler.get_num_new_matched_tokens(_make_request(), 0)

    assert need == 63
    assert scheduler.load_specs["r1"].kvpool_cached_tokens == 63
    assert scheduler.load_specs["r1"].kvpool_store_skip_tokens == 64


def test_mp_scheduler_hit_below_computed_tokens_allocates_nothing() -> None:
    scheduler, lookup_handler = _make_scheduler()
    lookup_handler.return_value = 8

    assert scheduler.get_num_new_matched_tokens(_make_request(), 16) == (0, False)


def test_mp_scheduler_async_hit_reports_async_load() -> None:
    scheduler, lookup_handler = _make_scheduler(extra_config={"backend": "mooncake", "load_async": True})
    lookup_handler.return_value = 48

    need, is_async = scheduler.get_num_new_matched_tokens(_make_request(), 16)

    assert (need, is_async) == (32, True)


def test_mp_scheduler_layerwise_queries_store_scheduler_directly() -> None:
    scheduler, lookup_handler = _make_scheduler(extra_config={"backend": "mooncake", "use_layerwise": True})
    scheduler.store_scheduler.batch_is_exist = MagicMock(side_effect=lambda keys: [1] * len(keys))
    request = _make_request()

    need, _ = scheduler.get_num_new_matched_tokens(request, 0)

    # Every block hits across all layers: 64 tokens, reduced by one for scheduling.
    assert need == 63
    scheduler.store_scheduler.batch_is_exist.assert_called_once()
    lookup_handler.assert_not_called()


def test_mp_scheduler_layerwise_partial_layer_miss_stops_at_last_full_block() -> None:
    scheduler, _ = _make_scheduler(extra_config={"backend": "mooncake", "use_layerwise": True})
    # Each block spreads over 2 layer keys (num_layers=2); dropping the last
    # key misses one layer of the final block, leaving 3 full blocks = 48 tokens.
    scheduler.store_scheduler.batch_is_exist = MagicMock(side_effect=lambda keys: [1] * (len(keys) - 1) + [0])
    request = _make_request()

    need, _ = scheduler.get_num_new_matched_tokens(request, 0)

    assert need == 48


def _make_view(request: MagicMock) -> RequestView:
    return RequestView(
        request_id=request.request_id,
        prompt_token_ids=list(request.prompt_token_ids),
        block_hashes=list(request.block_hashes),
        num_prompt_tokens=len(request.prompt_token_ids),
        num_tokens=request.num_tokens,
        all_token_ids=list(request.prompt_token_ids),
    )


def test_mp_scheduler_update_state_after_alloc_flips_can_load_and_registers_view() -> None:
    scheduler, lookup_handler = _make_scheduler()
    lookup_handler.return_value = 48
    request = _make_request()
    scheduler.get_num_new_matched_tokens(request, 16)

    view = _make_view(request)
    scheduler.update_state_after_alloc(view, BlocksView(block_ids_by_group=[[7, 8]]), 32)

    assert scheduler.load_specs["r1"].can_load is True
    stored_request, stored_blocks = scheduler._unfinished_requests["r1"]
    assert stored_request is view
    assert stored_blocks == [[7, 8]]


def test_mp_scheduler_update_state_after_alloc_without_load_spec_only_registers() -> None:
    scheduler, _ = _make_scheduler()
    request = _make_request()

    view = _make_view(request)
    scheduler.update_state_after_alloc(view, BlocksView(block_ids_by_group=[[7]]), 16)

    assert "r1" not in scheduler.load_specs
    assert scheduler._unfinished_requests["r1"] == (view, [[7]])


def test_mp_scheduler_update_state_after_alloc_zero_external_keeps_load_unloadable() -> None:
    scheduler, lookup_handler = _make_scheduler()
    lookup_handler.return_value = 48
    request = _make_request()
    scheduler.get_num_new_matched_tokens(request, 16)

    scheduler.update_state_after_alloc(_make_view(request), BlocksView(block_ids_by_group=[]), 0)

    # Non-layerwise requests with zero allocated blocks cannot load.
    assert scheduler.load_specs["r1"].can_load is False
    assert scheduler._unfinished_requests["r1"][1] == [[]]


def test_mp_scheduler_update_state_after_alloc_rejects_mismatched_allocation() -> None:
    scheduler, lookup_handler = _make_scheduler()
    lookup_handler.return_value = 48
    request = _make_request()
    scheduler.get_num_new_matched_tokens(request, 16)

    with pytest.raises(AssertionError, match="Mismatch in number of tokens"):
        scheduler.update_state_after_alloc(_make_view(request), BlocksView(block_ids_by_group=[[7]]), 31)


def _make_output_view(
    new_reqs: list[ScheduledNewReqPayload] | None = None,
    cached: CachedReqsView | None = None,
    num_scheduled_tokens: dict | None = None,
    finished: set[str] | None = None,
    preempted: set[str] | None = None,
) -> SchedulerOutputView:
    empty_cached = CachedReqsView(req_ids=[], new_block_ids=[], num_computed_tokens=[], new_token_ids={})
    return SchedulerOutputView(
        finished_req_ids=finished or set(),
        preempted_req_ids=preempted or set(),
        num_scheduled_tokens=num_scheduled_tokens or {},
        scheduled_new_reqs=new_reqs or [],
        scheduled_cached_reqs=cached or empty_cached,
    )


def test_mp_scheduler_build_meta_new_request_produces_req_meta() -> None:
    scheduler, lookup_handler = _make_scheduler()
    lookup_handler.return_value = 48
    request = _make_request()
    scheduler.get_num_new_matched_tokens(request, 16)
    view = _make_view(request)
    scheduler.update_state_after_alloc(view, BlocksView(block_ids_by_group=[[7, 8]]), 32)

    output = _make_output_view(
        new_reqs=[ScheduledNewReqPayload(req_id="r1", num_computed_tokens=16, block_ids_by_group=[[7, 8]])],
        num_scheduled_tokens={"r1": 48},
    )
    metadata = scheduler.build_connector_meta(output)

    assert len(metadata.requests) == 1
    req_meta = metadata.requests[0]
    assert req_meta.req_id == "r1"
    assert req_meta.target_token_len == 64
    assert req_meta.block_ids_by_group == [[7, 8]]
    assert req_meta.load_spec is not None and req_meta.load_spec.can_load
    assert "r1" in scheduler._request_trackers


def test_mp_scheduler_build_meta_refreshes_dynamic_fields_for_decode() -> None:
    scheduler, lookup_handler = _make_scheduler()
    lookup_handler.return_value = 48
    request = _make_request()
    scheduler.get_num_new_matched_tokens(request, 16)
    scheduler.update_state_after_alloc(_make_view(request), BlocksView(block_ids_by_group=[[7, 8]]), 32)
    scheduler.build_connector_meta(
        _make_output_view(
            new_reqs=[ScheduledNewReqPayload(req_id="r1", num_computed_tokens=16, block_ids_by_group=[[7, 8]])],
            num_scheduled_tokens={"r1": 48},
        )
    )

    # Second step: the request is fully computed (decode phase) with one new
    # generated token; without save_decode_cache the inherited logic skips it.
    output = _make_output_view(
        cached=CachedReqsView(
            req_ids=["r1"],
            new_block_ids=[[[10]]],
            num_computed_tokens=[64],
            new_token_ids={"r1": [101]},
        ),
        num_scheduled_tokens={"r1": 1},
    )
    metadata = scheduler.build_connector_meta(output)

    assert metadata.requests == []
    view = scheduler._unfinished_requests["r1"][0]
    assert view.num_computed_tokens == 64
    assert view.all_token_ids[-1] == 101
    assert len(view.all_token_ids) == 65


def test_mp_scheduler_build_meta_async_load_uses_registered_view() -> None:
    scheduler, lookup_handler = _make_scheduler(extra_config={"backend": "mooncake", "load_async": True})
    lookup_handler.return_value = 48
    request = _make_request()
    scheduler.get_num_new_matched_tokens(request, 0)
    scheduler.update_state_after_alloc(_make_view(request), BlocksView(block_ids_by_group=[[7, 8, 9]]), 48)

    # The request is not part of this scheduling step; the async path must
    # still produce a load ReqMeta from the registered view.
    metadata = scheduler.build_connector_meta(_make_output_view())

    assert len(metadata.requests) == 1
    req_meta = metadata.requests[0]
    assert req_meta.req_id == "r1"
    assert req_meta.load_spec is not None and req_meta.load_spec.can_load
    assert "r1" in scheduler._loading_req_ids


def test_mp_scheduler_records_block_pool_touch_commands() -> None:
    scheduler, _ = _make_scheduler()
    scheduler.use_hybrid = True
    scheduler.mamba_group_ids = [0]
    req_meta = ReqMeta(req_id="r1", token_len_chunk=32, block_ids_by_group=[[5, 8]], block_hashes=[], can_save=True)

    scheduler.touch_sending_mamba_blocks(req_meta)

    assert req_meta.event_id is not None
    assert scheduler.take_block_pool_commands() == [5, 8]
    assert scheduler.take_block_pool_commands() == []
    assert scheduler.sending_blocks[req_meta.event_id] == [5, 8]


def test_mp_scheduler_request_finished_consumer_no_put() -> None:
    scheduler, _ = _make_scheduler(kv_role="kv_consumer")
    scheduler._delayed_free_req_ids.add("r1")

    assert scheduler.request_finished(RequestIdView("r1"), [7]) == (False, None)
    assert "r1" not in scheduler._delayed_free_req_ids


def test_mp_scheduler_request_finished_without_saved_tokens() -> None:
    scheduler, _ = _make_scheduler()
    tracker = RequestTracker(req_id="r1", token_len=64, allocated_block_ids_by_group=[[7]])
    scheduler._request_trackers["r1"] = tracker

    assert scheduler.request_finished(RequestIdView("r1"), [7]) == (False, None)
    assert "r1" not in scheduler._delayed_free_req_ids


def test_mp_scheduler_request_finished_with_saved_tokens_delays_free() -> None:
    scheduler, _ = _make_scheduler()
    tracker = RequestTracker(req_id="r1", token_len=64, allocated_block_ids_by_group=[[7]])
    tracker.num_saved_tokens = 64
    scheduler._request_trackers["r1"] = tracker

    assert scheduler.request_finished(RequestIdView("r1"), [7]) == (True, None)
    assert "r1" in scheduler._delayed_free_req_ids


def _make_connector_output(completed_events: dict[int, int]) -> ConnectorOutputView:
    return ConnectorOutputView(kv_connector_worker_meta=AscendStoreKVConnectorWorkerMetadata(completed_events))


def test_mp_scheduler_update_connector_output_completes_event_and_frees_blocks() -> None:
    scheduler, _ = _make_scheduler()
    # world_size is 1 in the test config, so a single worker report completes the event.
    scheduler.sending_events[7] = 0
    scheduler.sending_blocks[7] = [5, 8]

    scheduler.update_connector_output(_make_connector_output({7: 1}))

    assert scheduler.take_free_block_commands() == [5, 8]
    assert scheduler.take_free_block_commands() == []
    assert 7 not in scheduler.sending_events
    assert 7 not in scheduler.sending_blocks


def test_mp_scheduler_update_connector_output_accumulates_partial_counts() -> None:
    scheduler, _ = _make_scheduler()
    scheduler._expected_worker_count = 2
    scheduler.sending_events[7] = 0
    scheduler.sending_blocks[7] = [5, 8]

    scheduler.update_connector_output(_make_connector_output({7: 1}))

    assert scheduler.take_free_block_commands() == []
    assert scheduler.sending_events[7] == 1
    assert scheduler.sending_blocks[7] == [5, 8]


def test_mp_scheduler_update_connector_output_ignores_unknown_event() -> None:
    scheduler, _ = _make_scheduler()

    scheduler.update_connector_output(_make_connector_output({999: 1}))

    assert scheduler.take_free_block_commands() == []
    assert scheduler.sending_events == {}
