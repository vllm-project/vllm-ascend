import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.request import RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_ascend.core.laps_scheduler import AsyncLAPSScheduler, LAPSScheduler


@pytest.mark.cpu_test
def test_laps_long_prefill_cap_limits_single_long_request(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LAPS_LONG_PREFILL_CAP", "256")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_SHORT_RESERVED_RATIO", "0")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_THRESHOLD", "128")

    base_scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        enable_chunked_prefill=True,
    )
    scheduler = LAPSScheduler(
        vllm_config=base_scheduler.vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        block_size=base_scheduler.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base_scheduler.vllm_config),
    )

    request = create_requests(num_requests=1, num_tokens=800)[0]
    scheduler.add_request(request)
    output = scheduler.schedule()

    assert output.num_scheduled_tokens[request.request_id] == 256
    waiting = scheduler.waiting
    assert waiting._last_long_capped_count == 1
    assert waiting._last_long_actual_used_tokens == 256
    assert waiting._last_short_reserved_tokens == 0


@pytest.mark.cpu_test
def test_laps_short_reserved_budget_reduces_long_prefill_share(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LAPS_LONG_PREFILL_CAP", "0")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_SHORT_RESERVED_RATIO", "0.25")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_THRESHOLD", "128")

    base_scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        enable_chunked_prefill=True,
    )
    scheduler = LAPSScheduler(
        vllm_config=base_scheduler.vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        block_size=base_scheduler.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base_scheduler.vllm_config),
    )

    short_request = create_requests(
        num_requests=1,
        num_tokens=64,
        req_ids=["short"],
    )[0]
    long_request = create_requests(
        num_requests=1,
        num_tokens=800,
        req_ids=["long"],
    )[0]

    scheduler.add_request(short_request)
    scheduler.add_request(long_request)
    output = scheduler.schedule()

    assert output.num_scheduled_tokens[short_request.request_id] == 64
    assert output.num_scheduled_tokens[long_request.request_id] == 768
    waiting = scheduler.waiting
    assert waiting._last_short_reserved_tokens == 256
    assert waiting._last_short_actual_used_tokens == 64
    assert waiting._last_long_actual_used_tokens == 768


@pytest.mark.cpu_test
def test_laps_long_prefill_cap_does_not_limit_short_prefill(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LAPS_LONG_PREFILL_CAP", "128")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_SHORT_RESERVED_RATIO", "0")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_THRESHOLD", "256")

    base_scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        enable_chunked_prefill=True,
    )
    scheduler = LAPSScheduler(
        vllm_config=base_scheduler.vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        block_size=base_scheduler.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base_scheduler.vllm_config),
    )

    short_request = create_requests(
        num_requests=1,
        num_tokens=200,
        req_ids=["short-200"],
    )[0]

    scheduler.add_request(short_request)
    output = scheduler.schedule()

    assert output.num_scheduled_tokens[short_request.request_id] == 200
    waiting = scheduler.waiting
    assert waiting._last_long_capped_count == 0
    assert waiting._last_short_actual_used_tokens == 200
    assert waiting._last_long_actual_used_tokens == 0


@pytest.mark.cpu_test
def test_laps_long_cap_count_only_tracks_scheduled_requests(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LAPS_LONG_PREFILL_CAP", "256")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_SHORT_RESERVED_RATIO", "0")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_THRESHOLD", "128")

    base_scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        enable_chunked_prefill=True,
    )
    scheduler = LAPSScheduler(
        vllm_config=base_scheduler.vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        block_size=base_scheduler.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base_scheduler.vllm_config),
    )

    request_1 = create_requests(num_requests=1, num_tokens=800)[0]
    request_2 = create_requests(num_requests=1, num_tokens=800, req_ids=["long-2"])[0]
    scheduler.add_request(request_1)
    scheduler.add_request(request_2)

    request_1.status = RequestStatus.RUNNING
    request_2.status = RequestStatus.RUNNING
    scheduler.running = [request_1, request_2]
    scheduler.waiting.remove_requests([request_1, request_2])
    scheduler.kv_cache_manager.allocate_slots = lambda *args, **kwargs: None

    output = scheduler.schedule()

    assert output.preempted_req_ids
    assert scheduler.waiting._last_long_capped_count == 0


@pytest.mark.cpu_test
def test_laps_preempted_request_is_requeued_immediately(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LAPS_THRESHOLD", "256")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_WAIT_WINDOW_MS", "10")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_WAIT_MAX_BATCH", "4")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_LONG_PREFILL_CAP", "0")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_SHORT_RESERVED_RATIO", "0")

    base_scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        enable_chunked_prefill=True,
    )
    scheduler = LAPSScheduler(
        vllm_config=base_scheduler.vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        block_size=base_scheduler.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base_scheduler.vllm_config),
    )

    request = create_requests(num_requests=1, num_tokens=64)[0]
    request.status = RequestStatus.RUNNING

    scheduler._preempt_request(request, 0.0)

    selected_queue = scheduler._select_waiting_queue_for_scheduling()
    assert selected_queue is not None
    assert selected_queue.peek_request() is request


@pytest.mark.cpu_test
def test_laps_non_budget_path_records_short_usage(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LAPS_THRESHOLD", "256")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_WAIT_WINDOW_MS", "0")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_WAIT_MAX_BATCH", "4")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_LONG_PREFILL_CAP", "0")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_SHORT_RESERVED_RATIO", "0")

    base_scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        enable_chunked_prefill=True,
    )
    scheduler = LAPSScheduler(
        vllm_config=base_scheduler.vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        block_size=base_scheduler.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base_scheduler.vllm_config),
    )

    short_request = create_requests(
        num_requests=1,
        num_tokens=64,
        req_ids=["short-stats"],
    )[0]

    scheduler.add_request(short_request)
    output = scheduler.schedule()

    assert output.num_scheduled_tokens[short_request.request_id] == 64
    waiting = scheduler.waiting
    assert waiting._last_short_actual_used_tokens == 64
    assert waiting._last_long_actual_used_tokens == 0


@pytest.mark.cpu_test
def test_async_laps_scheduler_installs_laps_waiting_queue(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LAPS_THRESHOLD", "256")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_WAIT_WINDOW_MS", "10")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_WAIT_MAX_BATCH", "4")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_LONG_PREFILL_CAP", "0")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_SHORT_RESERVED_RATIO", "0")

    base_scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        enable_chunked_prefill=True,
    )
    base_scheduler.vllm_config.scheduler_config.async_scheduling = True
    scheduler = AsyncLAPSScheduler(
        vllm_config=base_scheduler.vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        block_size=base_scheduler.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base_scheduler.vllm_config),
    )

    request = create_requests(num_requests=1, num_tokens=64)[0]
    scheduler.add_request(request)

    assert scheduler.waiting.__class__.__name__ == "LAPSRequestQueue"
