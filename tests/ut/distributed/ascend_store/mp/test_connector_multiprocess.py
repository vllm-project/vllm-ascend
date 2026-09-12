from unittest.mock import MagicMock

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import (
    AscendStoreConnector,
)


def _connector() -> tuple[AscendStoreConnector, MagicMock]:
    connector = AscendStoreConnector.__new__(AscendStoreConnector)
    connector.use_multiprocess = True
    connector.use_layerwise = True
    connector.kv_role = "kv_producer"
    connector.consumer_is_to_put = False
    worker = MagicMock()
    connector.connector_worker = worker
    connector._layerwise_step_prepared = False
    connector._current_step_has_real_forward = False
    connector._mamba_copy_bufs = None
    return connector, worker


def test_process_connector_prepares_each_layerwise_step_once():
    connector, worker = _connector()
    call_order = []
    worker.start_load_kv.side_effect = lambda _metadata: call_order.append("start")
    worker.wait_for_layer_load.side_effect = lambda: call_order.append("wait")
    worker.save_kv_layer.side_effect = lambda _metadata: call_order.append("save")

    first_metadata = MagicMock()
    connector.bind_connector_metadata(first_metadata)
    connector.wait_for_layer_load("layer_0")
    connector.start_load_kv(MagicMock())
    connector.wait_for_save()

    assert call_order == ["start", "wait"]
    worker.start_load_kv.assert_called_once_with(first_metadata)
    worker.wait_for_save.assert_called_once_with(first_metadata)

    second_metadata = MagicMock()
    connector.bind_connector_metadata(second_metadata)
    connector.save_kv_layer("layer_0", MagicMock(), MagicMock())

    assert call_order == ["start", "wait", "start", "save"]
    worker.start_load_kv.assert_called_with(second_metadata)


def test_process_connector_shutdown_closes_worker():
    connector, worker = _connector()

    connector.shutdown()

    worker.close.assert_called_once_with()
