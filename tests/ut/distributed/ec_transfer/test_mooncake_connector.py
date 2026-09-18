from unittest.mock import MagicMock, patch

from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.distributed.ec_transfer.ec_connector.factory import (
    ECConnectorFactory,
)

from vllm_ascend.distributed import ec_transfer as ec_transfer_module
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake import connector as connector_module
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


def test_connector_constructs_ascend_worker_without_upstream_factory_hooks():
    vllm_config = MagicMock()
    expected_worker = object()

    with (
        patch.object(
            connector_module.ECConnectorBase,
            "__init__",
            return_value=None,
        ) as base_init,
        patch.object(
            connector_module,
            "AscendECMooncakeWorker",
            return_value=expected_worker,
        ) as worker_class,
    ):
        connector = AscendECMooncakeConnector(
            vllm_config,
            ECConnectorRole.WORKER,
        )

    base_init.assert_called_once_with(
        connector,
        vllm_config=vllm_config,
        role=ECConnectorRole.WORKER,
    )
    worker_class.assert_called_once_with(vllm_config)
    assert connector._worker is expected_worker
    assert connector._scheduler is None


def test_connector_constructs_upstream_scheduler():
    vllm_config = MagicMock()
    expected_scheduler = object()

    with (
        patch.object(
            connector_module.ECConnectorBase,
            "__init__",
            return_value=None,
        ),
        patch.object(
            connector_module,
            "ECMooncakeScheduler",
            return_value=expected_scheduler,
        ) as scheduler_class,
    ):
        connector = AscendECMooncakeConnector(
            vllm_config,
            ECConnectorRole.SCHEDULER,
        )

    scheduler_class.assert_called_once_with(vllm_config)
    assert connector._scheduler is expected_scheduler
    assert connector._worker is None
