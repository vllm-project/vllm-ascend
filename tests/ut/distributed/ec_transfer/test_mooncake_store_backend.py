from unittest.mock import MagicMock

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
    assert backend.prepare_fallback("first") is None

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
    ticket = backend.prepare_fallback("first")
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


def test_shutdown_drains_unselected_deferred_save_through_original_path():
    backend, client = _backend()
    tensor = _tensor()
    backend.save_output("first", tensor)

    backend.shutdown()

    client.exists.assert_called_once()
    pool_key = client.exists.call_args.args[0]
    client.put_tensor.assert_called_once_with(pool_key, tensor)
    client.close.assert_called_once_with()
