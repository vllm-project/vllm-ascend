"""CPU regression tests adapted from vLLM PRs #48245 and #50297.

The upstream tests use vLLM's in-tree ``tests/v1/core`` helpers, which are not
shipped in its wheel. These tests cover the same compatibility-boundary state
transitions without requiring an NPU model runner.
"""

import inspect
from types import SimpleNamespace

import pytest

from vllm_ascend.patch.platform import patch_async_scheduler as async_backport
from vllm_ascend.patch.platform import patch_balance_schedule as backport


class _Queue:
    def __init__(self):
        self.requests = []

    def prepend_request(self, request):
        self.requests.insert(0, request)


class _Request(SimpleNamespace):
    # Request instances are identity-hashable and are stored in scheduler sets.
    # typeshed marks SimpleNamespace.__hash__ as None, so this intentional test
    # double override needs a narrow assignment suppression.
    __hash__ = object.__hash__  # type: ignore[assignment]


def _request(
    request_id="handoff",
    *,
    in_flight=4,
    placeholders=4,
    stale=0,
    drop_stale=False,
):
    return _Request(
        request_id=request_id,
        status=backport.RequestStatus.RUNNING,
        num_in_flight_tokens=in_flight,
        num_output_placeholders=placeholders,
        num_stale_output_tokens=stale,
        drop_stale_output=drop_stale,
        spec_token_ids=[],
        num_computed_tokens=32,
        num_preemptions=0,
    )


def _scheduler():
    scheduler = backport.BalanceScheduler.__new__(backport.BalanceScheduler)
    scheduler._free_request_blocks = lambda _request: None
    scheduler.encoder_cache_manager = SimpleNamespace(free=lambda _request: None)
    scheduler._inflight_prefills = set()
    scheduler.log_stats = False
    scheduler.waiting = _Queue()
    scheduler.reset_preempted_req_ids = set()
    return scheduler


@pytest.mark.parametrize(
    ("is_producer", "expected"),
    [(True, True), (False, False)],
)
def test_requires_kv_delivery_defaults_to_producer_role(is_producer, expected):
    connector = SimpleNamespace(_kv_transfer_config=SimpleNamespace(is_kv_producer=is_producer))

    assert backport._requires_kv_delivery(connector) is expected


@pytest.mark.parametrize(
    ("children", "expected"),
    [
        ([False, False], False),
        ([False, True], True),
        ([True, False], True),
        ([], False),
    ],
)
def test_multi_connector_aggregates_child_delivery_requirements(children, expected):
    connector = SimpleNamespace(_connectors=[SimpleNamespace(requires_kv_delivery=value) for value in children])

    assert backport._multi_requires_kv_delivery(connector) is expected


def test_best_effort_offload_cache_does_not_require_delivery():
    assert backport._best_effort_cache_requires_kv_delivery(SimpleNamespace()) is False


def test_request_stale_output_state_has_neutral_defaults():
    assert backport.Request.num_stale_output_tokens == 0
    assert backport.Request.drop_stale_output is False


def test_async_scheduler_inherits_existing_balance_patch():
    assert issubclass(async_backport.AsyncScheduler, backport.BalanceScheduler)


@pytest.mark.parametrize("drop_stale_output", [False, True])
def test_preempt_marks_all_inflight_output_stale(drop_stale_output):
    """#48245 records the complete in-flight token share, not one frame."""
    request = _request(in_flight=12, placeholders=4)
    scheduler = _scheduler()

    backport.BalanceScheduler._preempt_request(
        scheduler,
        request,
        0.0,
        drop_stale_output=drop_stale_output,
    )

    assert request.status == backport.RequestStatus.PREEMPTED
    assert request.num_stale_output_tokens == 12
    assert request.num_output_placeholders == 0
    assert request.drop_stale_output is drop_stale_output
    assert scheduler.waiting.requests == [request]


def test_repreempt_keeps_an_undrained_drop_share_dropped():
    request = _request(in_flight=8, stale=4, drop_stale=True)
    scheduler = _scheduler()

    backport.BalanceScheduler._preempt_request(scheduler, request, 0.0)

    assert request.num_stale_output_tokens == 8
    assert request.drop_stale_output is True


def test_reset_prefix_cache_uses_drop_mode_for_same_step_resume():
    """#48245 replaces v0.26's placeholder-count discard bookkeeping."""
    request = _request(in_flight=8, placeholders=4)
    scheduler = _scheduler()
    scheduler.running = [request]
    scheduler.prev_step_scheduled_req_ids = {request.request_id}
    scheduler.kv_cache_manager = SimpleNamespace(reset_prefix_cache=lambda: True)
    scheduler.connector = None

    assert backport.BalanceScheduler.reset_prefix_cache(scheduler, reset_running_requests=True) is True

    assert request.num_stale_output_tokens == 8
    assert request.num_output_placeholders == 0
    assert request.drop_stale_output is True
    assert scheduler.prev_step_scheduled_req_ids == set()


@pytest.mark.parametrize(
    ("is_stale", "expected_placeholders"),
    [(True, 0), (False, 2)],
)
def test_async_placeholder_update_skips_stale_delivery(monkeypatch, is_stale, expected_placeholders):
    """#48245 stale output is delivered without decrementing reset counters."""
    request = _request(placeholders=4)
    scheduler = async_backport.AsyncScheduler.__new__(async_backport.AsyncScheduler)
    scheduler.kv_cache_manager = SimpleNamespace(cache_blocks=lambda *_args: None)

    def update_original(_scheduler, _request, new_token_ids, is_stale=False):
        return new_token_ids, False

    monkeypatch.setattr(backport.BalanceScheduler, "_update_request_with_output", update_original)
    if is_stale:
        request.num_output_placeholders = 0

    new_token_ids, stopped = async_backport._update_request_with_output(
        scheduler,
        request,
        [10, 11],
        is_stale=is_stale,
    )

    assert new_token_ids == [10, 11]
    assert stopped is False
    assert request.num_output_placeholders == expected_placeholders


def test_50297_pressure_preemption_uses_connector_delivery_requirement():
    """The existing schedule copy carries #50297 at the original call site."""
    source = inspect.getsource(backport.BalanceScheduler.schedule)

    assert "drop_stale_output=self.requires_kv_delivery" in source


def test_48245_waits_for_deliverable_stale_output_before_resume():
    source = inspect.getsource(backport.BalanceScheduler.schedule)

    assert "request.num_stale_output_tokens > 0" in source
    assert "not request.drop_stale_output" in source


def test_48245_preemption_keeps_upstream_operation_order():
    source = inspect.getsource(backport.BalanceScheduler._preempt_request)
    markers = [
        "self._free_request_blocks(request)",
        "request.num_computed_tokens = 0",
        "request.drop_stale_output =",
        "request.num_stale_output_tokens = request.num_in_flight_tokens",
        "request.num_preemptions += 1",
        "self.waiting.prepend_request(request)",
    ]

    assert [source.index(marker) for marker in markers] == sorted(source.index(marker) for marker in markers)


def test_48245_output_update_keeps_stale_checks_in_original_flow():
    source = inspect.getsource(inspect.unwrap(backport.BalanceScheduler.update_from_output))
    markers = [
        "request.num_in_flight_tokens -= num_tokens_scheduled",
        "request.num_stale_output_tokens -= num_tokens_scheduled",
        "if failed_kv_load_req_ids",
        "if request is None or request.is_finished()",
        "if output_is_stale and request.drop_stale_output",
        "req_index = model_runner_output.req_id_to_index[req_id]",
    ]

    assert [source.index(marker) for marker in markers] == sorted(source.index(marker) for marker in markers)
