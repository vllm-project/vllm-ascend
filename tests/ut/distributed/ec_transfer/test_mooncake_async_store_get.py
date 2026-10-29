from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.distributed.ec_transfer.ec_connector.mooncake.metadata import (
    ECMooncakeConnectorMetadata,
    TensorSpec,
)

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake import scheduler as module
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.async_store_get import (
    AscendECMooncakeWorkerMetadata,
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
    request = _request()
    request.mm_features *= 2
    with patch.object(module.ECMooncakeScheduler, "ensure_cache_available", return_value=True):
        assert not scheduler.ensure_cache_available(request, 0)
    scheduler._store_get_to_dispatch = {}
    with patch.object(module.ECMooncakeScheduler, "request_finished", return_value=(False, None)):
        scheduler.request_finished(request)
    assert scheduler._store_get_states["image"] is module._StoreGetState.ABANDONED
    assert scheduler._store_get_bytes == 128
    assert scheduler._store_get_to_discard == set()
    output = SimpleNamespace(ec_connector_worker_meta=AscendECMooncakeWorkerMetadata(store_get_hits={"image"}))
    with patch.object(module.ECMooncakeScheduler, "update_connector_output"):
        scheduler.update_connector_output(output)
    assert scheduler._store_get_to_discard == {"image"}
    assert scheduler._store_get_bytes == 0


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
