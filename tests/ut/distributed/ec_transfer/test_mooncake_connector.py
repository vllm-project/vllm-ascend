from unittest.mock import patch

from vllm.distributed.ec_transfer.ec_connector.factory import (
    ECConnectorFactory,
)

from vllm_ascend.distributed import ec_transfer as ec_transfer_module
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake import (
    worker as worker_module,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.connector import (
    AscendECMooncakeConnector,
)


def test_register_connector_replaces_upstream_mooncake():
    registry = {"ECMooncakeConnector": object}

    with (
        patch.object(
            ec_transfer_module.importlib,
            "import_module",
        ) as import_module_mock,
        patch.object(ECConnectorFactory, "_registry", registry),
        patch.object(
            ECConnectorFactory,
            "register_connector",
        ) as register_connector,
    ):
        ec_transfer_module.register_connector()

    import_module_mock.assert_called_once_with("vllm.distributed.ec_transfer.ec_connector.mooncake_ec_connector")
    assert "ECMooncakeConnector" not in registry
    register_connector.assert_called_once_with(
        "ECMooncakeConnector",
        "vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.connector",
        "AscendECMooncakeConnector",
    )


def test_register_connector_skips_when_upstream_mooncake_not_found():
    upstream_module = "vllm.distributed.ec_transfer.ec_connector.mooncake_ec_connector"
    missing_module = ModuleNotFoundError(
        f"No module named '{upstream_module}'",
        name=upstream_module,
    )

    with (
        patch.object(
            ec_transfer_module.importlib,
            "import_module",
            side_effect=missing_module,
        ) as import_module_mock,
        patch.object(
            ECConnectorFactory,
            "register_connector",
        ) as register_connector,
    ):
        ec_transfer_module.register_connector()

    import_module_mock.assert_called_once_with(upstream_module)
    register_connector.assert_not_called()


def test_make_worker_uses_ascend_worker():
    connector = object.__new__(AscendECMooncakeConnector)
    vllm_config = object()
    expected_worker = object()

    with patch.object(
        worker_module,
        "AscendECMooncakeWorker",
        return_value=expected_worker,
    ) as worker_class:
        result = connector._make_worker(vllm_config)

    worker_class.assert_called_once_with(vllm_config)
    assert result is expected_worker
