from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm_ascend.distributed.kv_transfer.ascend_multi_connector import (
    AscendMultiConnector,
)


def test_rebuild_kv_transfer_endpoint_forwards_to_supported_sub_connectors():
    rebuild = MagicMock()
    connector = object.__new__(AscendMultiConnector)
    connector._connectors = [
        SimpleNamespace(rebuild_kv_transfer_endpoint=rebuild),
        object(),
    ]

    connector.rebuild_kv_transfer_endpoint("10.0.0.8", "engine-new")

    rebuild.assert_called_once_with("10.0.0.8", "engine-new")
