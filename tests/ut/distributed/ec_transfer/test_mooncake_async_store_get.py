from concurrent.futures import Future
from functools import reduce
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from weakref import ref

import pytest
import torch
from vllm.distributed.ec_transfer.ec_connector.mooncake.metadata import (
    ECMooncakeConnectorMetadata,
    ECMooncakeWorkerMetadata,
    TensorSpec,
)

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake import scheduler as module
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.async_store_get import (
    AscendECMooncakeWorkerMetadata,
    AsyncStoreGetExecutor,
    _PendingStoreGet,
    _StoreGetResult,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.scheduler import (
    AscendECMooncakeScheduler,
)


@pytest.fixture(autouse=True)
def encoder_output_spec():
    """Test GET state transitions independently of upstream model configuration."""
    with patch.object(
        module.ECMooncakeScheduler,
        "_encoder_output_spec",
        return_value=TensorSpec((8, 4), str(torch.float32), 128),
    ):
        yield


def _scheduler(max_bytes: int = 1024) -> AscendECMooncakeScheduler:
    scheduler = object.__new__(AscendECMooncakeScheduler)
    scheduler._store_get_enabled = True
    scheduler._store_get_max_bytes = max_bytes
    scheduler._store_get_bytes = 0
    scheduler._store_get_states = {}
    scheduler._store_get_sizes = {}
    scheduler._store_get_waiters = {}
    scheduler._store_get_to_dispatch = {}
    scheduler._store_get_to_consume = set()
    scheduler._store_get_to_recompute = set()
    scheduler._store_get_to_discard = set()
    scheduler._is_producer = True
    scheduler._local_cache = set()
    return scheduler


def _request(identifier: str = "image") -> SimpleNamespace:
    position = SimpleNamespace(offset=0, length=4)
    feature = SimpleNamespace(
        identifier=identifier,
        modality="image",
        mm_position=position,
    )
    request = SimpleNamespace(request_id="request", mm_features=[feature])
    request.get_num_encoder_embeds = MagicMock(return_value=8)
    return request


def test_async_store_get_defers_until_worker_reports_terminal_hit():
    scheduler = _scheduler()
    request = _request()
    with patch.object(
        module.ECMooncakeScheduler,
        "ensure_cache_available",
        return_value=True,
    ) as upstream:
        assert not scheduler.ensure_cache_available(request, 0)
        assert not scheduler.ensure_cache_available(request, 0)
    # Upstream continues preparing the P2P side while Store GET is pending;
    # the worker's late-hit binding reconnects its source afterward.
    assert upstream.call_count == 2

    assert list(scheduler._store_get_to_dispatch) == ["image"]
    assert scheduler._store_get_bytes == 128
    with patch.object(module.ECMooncakeScheduler, "build_connector_meta", return_value=ECMooncakeConnectorMetadata()):
        scheduler.build_connector_meta(MagicMock())

    output = SimpleNamespace(ec_connector_worker_meta=AscendECMooncakeWorkerMetadata(store_get_hits={"image"}))
    with patch.object(module.ECMooncakeScheduler, "update_connector_output"):
        scheduler.update_connector_output(output)
    with patch.object(
        module.ECMooncakeScheduler,
        "ensure_cache_available",
        return_value=True,
    ):
        assert scheduler.ensure_cache_available(request, 0)

    with patch.object(module.ECMooncakeScheduler, "update_state_after_alloc"):
        scheduler.update_state_after_alloc(request, 0)
    assert scheduler._store_get_bytes == 0
    assert scheduler._store_get_to_consume == {"image"}
    with (
        patch.object(module.ECMooncakeScheduler, "has_pending_push_work", return_value=False),
        patch.object(module.ECMooncakeScheduler, "build_connector_meta", return_value=ECMooncakeConnectorMetadata()),
    ):
        assert scheduler.has_pending_push_work()
        scheduler.build_connector_meta(MagicMock())
        assert not scheduler.has_pending_push_work()


def test_async_store_get_capacity_overflow_keeps_existing_synchronous_path():
    scheduler = _scheduler(max_bytes=64)
    request = _request()
    with patch.object(module.ECMooncakeScheduler, "ensure_cache_available", return_value=True):
        assert scheduler.ensure_cache_available(request, 0)
    assert scheduler._store_get_states == {}
    assert scheduler._store_get_to_dispatch == {}


def test_async_store_get_dispatches_once_and_preserves_upstream_fields():
    scheduler = _scheduler()
    request = _request()
    with patch.object(module.ECMooncakeScheduler, "ensure_cache_available", return_value=True):
        assert not scheduler.ensure_cache_available(request, 0)
    base = ECMooncakeConnectorMetadata(store_candidates={"existing": MagicMock()})
    with patch.object(module.ECMooncakeScheduler, "build_connector_meta", return_value=base):
        first = scheduler.build_connector_meta(MagicMock())
        second = scheduler.build_connector_meta(MagicMock())
    assert set(first.store_get_requests) == {"image"}
    assert first.store_candidates == base.store_candidates
    assert second.store_get_requests == {}


def test_cancelled_async_store_get_is_discarded_after_worker_completion():
    scheduler = _scheduler()
    worker = AsyncStoreGetExecutor(enabled=False, device="cpu")
    request = _request()
    request.mm_features *= 2
    with (
        patch.object(module.ECMooncakeScheduler, "ensure_cache_available", return_value=True),
        patch.object(module.ECMooncakeScheduler, "request_finished", return_value=(False, None)),
        patch.object(module.ECMooncakeScheduler, "update_connector_output"),
        patch.object(module.ECMooncakeScheduler, "build_connector_meta", return_value=ECMooncakeConnectorMetadata()),
        patch.object(module.ECMooncakeScheduler, "has_pending_push_work", return_value=False),
    ):
        assert not scheduler.ensure_cache_available(request, 0)
        dispatched = scheduler.build_connector_meta(MagicMock())
        completion: Future[_StoreGetResult] = Future()
        # Control native GET completion deterministically; keep real worker
        # polling, outcome reporting, command delivery and result disposal.
        worker._pending.append(_PendingStoreGet(set(dispatched.store_get_requests), completion))
        scheduler.request_finished(request)
        assert scheduler._store_get_states["image"] is module._StoreGetState.ABANDONED
        assert scheduler._store_get_bytes == 128
        assert scheduler._store_get_to_discard == set()
        assert scheduler.has_pending_push_work()

        target = torch.ones((8, 4))
        target_ref = ref(target)
        completion.set_result(_StoreGetResult({"image": target}, set()))
        output = SimpleNamespace(ec_connector_worker_meta=worker.extend_worker_metadata(ECMooncakeWorkerMetadata()))
        del target, completion
        scheduler.update_connector_output(output)
        assert scheduler._store_get_to_discard == {"image"}
        assert scheduler._store_get_bytes == 0
        assert scheduler._store_get_states == {}
        assert "image" in worker._ready and target_ref() is not None
        assert scheduler.has_pending_push_work()  # Discard is now the only work.

        metadata = scheduler.build_connector_meta(MagicMock())
        assert metadata.store_get_discard == {"image"}
        cache: dict[str, torch.Tensor] = {}
        assert worker.consume(metadata, cache) == set()
        assert worker._ready == {} and cache == {}
        assert target_ref() is None
        assert not scheduler.has_pending_push_work()
        assert scheduler.build_connector_meta(MagicMock()).store_get_discard == set()


@pytest.mark.parametrize(
    "cancel_new_waiter,outcome", [(False, "hits"), (False, "misses"), (False, "failures"), (True, "hits")]
)
def test_new_waiter_reuses_abandoned_get_until_terminal(cancel_new_waiter, outcome):
    scheduler = _scheduler()
    first, second = _request(), _request()
    second.request_id = "second"
    with (
        patch.object(module.ECMooncakeScheduler, "ensure_cache_available", return_value=True),
        patch.object(module.ECMooncakeScheduler, "request_finished", return_value=(False, None)),
        patch.object(module.ECMooncakeScheduler, "update_connector_output"),
        patch.object(module.ECMooncakeScheduler, "update_state_after_alloc"),
    ):
        assert not scheduler.ensure_cache_available(first, 0)
        scheduler._store_get_to_dispatch = {}
        scheduler.request_finished(first)
        assert not scheduler.ensure_cache_available(second, 0)
        assert scheduler._store_get_to_dispatch == {}
        assert scheduler._store_get_bytes == 128
        if cancel_new_waiter:
            scheduler.request_finished(second)
            assert scheduler._store_get_states["image"] is module._StoreGetState.ABANDONED
            assert scheduler._store_get_bytes == 128
            assert scheduler._store_get_to_discard == set()

        metadata = AscendECMooncakeWorkerMetadata(**{f"store_get_{outcome}": {"image"}})
        scheduler.update_connector_output(SimpleNamespace(ec_connector_worker_meta=metadata))
        if cancel_new_waiter:
            assert scheduler._store_get_to_discard == {"image"}
        else:
            assert scheduler.ensure_cache_available(second, 0)
            scheduler.update_state_after_alloc(second, 0)
            assert scheduler._store_get_to_consume == {"image"}
        assert scheduler._store_get_bytes == 0
        assert scheduler._store_get_states == {}


@pytest.mark.parametrize(
    "outcomes",
    [
        ("hit", "hit"),
        ("miss", "hit"),
        ("miss", "failure"),
        ("hit", "hit", "hit"),
        ("miss", "hit", "hit"),
        ("hit", "failure", "miss"),
    ],
)
@pytest.mark.parametrize("cancelled", [False, True])
def test_rank_outcomes_persist_across_steps_until_all_terminal_and_acknowledged(outcomes, cancelled):
    scheduler, request = _scheduler(), _request()
    workers = [AsyncStoreGetExecutor(enabled=False, device="cpu") for _ in outcomes]
    completions: list[Future[_StoreGetResult]] = [Future() for _ in outcomes]
    with (
        patch.object(module.ECMooncakeScheduler, "ensure_cache_available", return_value=True),
        patch.object(module.ECMooncakeScheduler, "request_finished", return_value=(False, None)),
        patch.object(module.ECMooncakeScheduler, "update_connector_output"),
        patch.object(module.ECMooncakeScheduler, "update_state_after_alloc"),
        patch.object(module.ECMooncakeScheduler, "build_connector_meta", return_value=ECMooncakeConnectorMetadata()),
        patch.object(module.ECMooncakeScheduler, "has_pending_push_work", return_value=False),
    ):
        assert not scheduler.ensure_cache_available(request, 0)
        scheduler.build_connector_meta(MagicMock())
        for worker, completion in zip(workers, completions, strict=True):
            worker._pending.append(_PendingStoreGet({"image"}, completion))
        if cancelled:
            scheduler.request_finished(request)
        for step, (outcome, completion) in enumerate(zip(outcomes, completions, strict=True)):
            tensor = torch.ones((8, 4)) if outcome == "hit" else None
            completion.set_result(_StoreGetResult({"image": tensor}, {"image"} if outcome == "failure" else set()))
            reports = [worker.extend_worker_metadata(ECMooncakeWorkerMetadata()) for worker in workers]
            combined = reduce(lambda left, right: left.aggregate(right), reports)
            scheduler.update_connector_output(SimpleNamespace(ec_connector_worker_meta=combined))
            if step < len(outcomes) - 1:
                assert not (combined.store_get_hits | combined.store_get_misses | combined.store_get_failures)
                assert scheduler._store_get_states["image"] is (
                    module._StoreGetState.ABANDONED if cancelled else module._StoreGetState.PENDING
                )
                assert scheduler._store_get_bytes == 128
                assert scheduler._store_get_to_discard == set()
        all_hit = all(outcome == "hit" for outcome in outcomes)
        if not cancelled:
            state = (
                module._StoreGetState.HIT
                if all_hit
                else module._StoreGetState.FAILED
                if "failure" in outcomes
                else module._StoreGetState.MISS
            )
            assert scheduler._store_get_states["image"] is state
            scheduler.update_state_after_alloc(request, 0)
        command = scheduler.build_connector_meta(MagicMock())
        assert command.store_get_discard == ({"image"} if cancelled else set())
        assert command.store_get_recompute == ({"image"} if not cancelled and not all_hit else set())
        for worker, report in zip(workers, reports, strict=True):
            before = (set(report.store_get_hits), set(report.store_get_misses), set(report.store_get_failures))
            cache: dict[str, torch.Tensor] = {}
            worker.consume(command, cache)
            assert set(cache) == ({"image"} if not cancelled and all_hit else set())
            assert not worker._ready
            cleared = worker.extend_worker_metadata(ECMooncakeWorkerMetadata())
            assert not (cleared.store_get_hits | cleared.store_get_misses | cleared.store_get_failures)
            assert before == (report.store_get_hits, report.store_get_misses, report.store_get_failures)
        assert not scheduler.has_pending_push_work()
        assert scheduler._store_get_bytes == 0
