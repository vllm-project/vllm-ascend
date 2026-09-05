# SPDX-License-Identifier: Apache-2.0
from vllm_ascend.distributed.kv_transfer.kv_pool import ucm_connector
from vllm_ascend.distributed.kv_transfer.kv_pool.ucm_connector.connector import UCMConnectorV1


def test_package_exports_connector_class():
    assert ucm_connector.__all__ == ["UCMConnectorV1"]
    assert ucm_connector.UCMConnectorV1 is UCMConnectorV1
