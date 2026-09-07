from unittest.mock import patch

import pytest
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

from vllm_ascend.distributed.kv_transfer import register_connector


@pytest.mark.parametrize(
    ("multiprocess", "module_name", "class_name"),
    [
        (
            "0",
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector",
            "AscendStoreConnector",
        ),
        (
            "1",
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_mp_connector",
            "AscendStoreMPConnector",
        ),
    ],
)
def test_ascend_store_registration_selects_process_mode(
    monkeypatch,
    multiprocess: str,
    module_name: str,
    class_name: str,
) -> None:
    monkeypatch.setenv("VLLM_ASCEND_STORE_MULTIPROCESS", multiprocess)

    with (
        patch.object(KVConnectorFactory, "_registry", {}),
        patch.object(KVConnectorFactory, "register_connector") as register,
    ):
        register_connector()

    register.assert_any_call("AscendStoreConnector", module_name, class_name)
    register.assert_any_call("MooncakeConnectorStoreV1", module_name, class_name)


def test_explicit_mp_connector_remains_registered(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_ASCEND_STORE_MULTIPROCESS", raising=False)

    with (
        patch.object(KVConnectorFactory, "_registry", {}),
        patch.object(KVConnectorFactory, "register_connector") as register,
    ):
        register_connector()

    register.assert_any_call(
        "AscendStoreMPConnector",
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_mp_connector",
        "AscendStoreMPConnector",
    )
