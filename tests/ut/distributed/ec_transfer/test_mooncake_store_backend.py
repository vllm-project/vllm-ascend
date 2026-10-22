from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake import store_backend as module
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.store_backend import (
    AscendDeferredStoreBackend,
    PreparedStorePut,
)


def _tensor(storage_ptr: int = 1000, storage_bytes: int = 128) -> MagicMock:
    tensor = MagicMock()
    tensor.is_contiguous.return_value = True
    tensor.device.type = "npu"
    storage = tensor.untyped_storage.return_value
    storage.data_ptr.return_value = storage_ptr
    storage.nbytes.return_value = storage_bytes
    return tensor


def _backend(*, max_items: int = 32, max_bytes: int = 2 * 1024**3):
    client = MagicMock()
    client.exists.return_value = False
    backend = AscendDeferredStoreBackend(
        client,
        "namespace",
        max_pending_items=max_items,
        max_pending_bytes=max_bytes,
    )
    backend._step_candidates = {"first", "second"}
    return backend, client


def test_deferred_store_admission_and_exists_rejection_do_not_submit_put():
    backend, client = _backend()
    tensor = _tensor()
    client.exists.return_value = True

    backend.save_output("first", tensor)
    assert backend.prepare_save("first") is None

    client.exists.assert_called_once()
    client.put_tensor.assert_not_called()
    client.put_prepared_tensor.assert_not_called()
    assert backend._pending == {}
    assert backend._pending_bytes == 0
    backend.shutdown()


def test_deferred_store_pending_budget_rejection_leaves_only_admitted_save():
    backend, _ = _backend(max_items=1)

    backend.save_output("first", _tensor(1000))
    backend.save_output("second", _tensor(2000))

    assert list(backend._pending) == ["first"]
    backend.shutdown()


def test_prepared_store_put_runs_on_single_publisher_and_reaps_storage():
    backend, client = _backend()
    tensor = _tensor()
    backend.save_output("first", tensor)
    ticket = backend.prepare_save("first")
    assert ticket is not None

    future = backend.submit_prepared(
        PreparedStorePut(
            ticket=ticket,
            addresses=[10, 20, 30],
            sizes=[24, 32, 72],
            owners=[tensor],
        )
    )
    future.result(timeout=1)
    backend.reap()

    client.put_prepared_tensor.assert_called_once_with(
        ticket.pool_key,
        [10, 20, 30],
        [24, 32, 72],
        [tensor],
    )
    assert backend._pending == {}
    assert backend._pending_bytes == 0
    backend.shutdown()


def test_pending_views_charge_storage_until_last_save_is_reaped():
    backend, client = _backend(max_bytes=128)
    first = _tensor(1000)
    second = _tensor(1000)
    second.untyped_storage.return_value = first.untyped_storage()
    backend.save_output("first", first)
    backend.save_output("second", second)
    assert backend._pending_bytes == 128
    assert len(backend._pending) == 2

    for identifier in ("first", "second"):
        ticket = backend.prepare_save(identifier)
        future = backend.submit_prepared(PreparedStorePut(ticket, [10], [128], [ticket.tensor]))
        future.result(timeout=1)
        backend.reap()
        assert backend._pending_bytes == (128 if identifier == "first" else 0)
    assert backend._pending == {}
    backend.shutdown()


def test_shutdown_drains_unselected_deferred_save_through_original_path():
    backend, client = _backend()
    tensor = _tensor()
    backend.save_output("first", tensor)

    backend.shutdown()

    client.exists.assert_called_once()
    pool_key = client.exists.call_args.args[0]
    client.put_tensor.assert_called_once_with(pool_key, tensor)
    client.close.assert_called_once_with()


def test_admission_records_npu_event_and_publisher_waits_before_reading():
    backend, client = _backend()
    tensor = _tensor()
    event = MagicMock()
    with patch.object(module.torch.npu, "Event", return_value=event):
        backend.save_output("first", tensor)
    event.record.assert_called_once_with(module.torch.npu.current_stream(tensor.device))
    client.put_tensor.side_effect = lambda *_: event.synchronize.assert_called_once_with()
    backend.shutdown()
    client.put_tensor.assert_called_once()


def test_prepared_submission_failure_leaves_ticket_abortable():
    backend, _ = _backend()
    backend.save_output("first", _tensor())
    ticket = backend.prepare_save("first")
    put = PreparedStorePut(ticket, [10], [128], [ticket.tensor])
    with (
        patch.object(backend._executor, "submit", side_effect=RuntimeError("executor stopped")),
        pytest.raises(RuntimeError, match="executor stopped"),
    ):
        backend.submit_prepared(put)
    backend.abort_prepared(ticket)
    assert backend._pending == {}
    assert backend._pending_bytes == 0
    backend.shutdown()
