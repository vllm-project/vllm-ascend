from unittest.mock import MagicMock, patch

import pytest
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory

from vllm_ascend.distributed import ec_transfer as ec_transfer_module
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake import connector as connector_module
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.connector import AscendECMooncakeConnector


@pytest.mark.parametrize("registered", [False, True])
def test_register_connector_replaces_only_existing_mooncake_entry(registered):
    registry = {"ECMooncakeConnector": object} if registered else {}
    with (
        patch.object(ec_transfer_module, "_replace_cpu_connector", return_value=True),
        patch("importlib.import_module") as import_module_mock,
        patch.object(ECConnectorFactory, "_registry", registry),
        patch.object(ECConnectorFactory, "register_connector") as register,
    ):
        ec_transfer_module.register_connector()
    import_module_mock.assert_not_called()
    if registered:
        assert "ECMooncakeConnector" not in registry
        register.assert_called_once_with(
            "ECMooncakeConnector",
            "vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.connector",
            "AscendECMooncakeConnector",
        )
    else:
        register.assert_not_called()


@pytest.mark.parametrize(
    "role,factory_name,active,inactive",
    [
        (ECConnectorRole.WORKER, "AscendECMooncakeWorker", "_worker", "_scheduler"),
        (ECConnectorRole.SCHEDULER, "AscendECMooncakeScheduler", "_scheduler", "_worker"),
    ],
)
def test_connector_constructs_only_the_requested_ascend_role(role, factory_name, active, inactive):
    config, expected = MagicMock(), object()
    with (
        patch.object(connector_module.ECConnectorBase, "__init__", return_value=None) as base_init,
        patch.object(connector_module, factory_name, return_value=expected) as factory,
    ):
        connector = AscendECMooncakeConnector(config, role)
    base_init.assert_called_once_with(connector, vllm_config=config, role=role)
    factory.assert_called_once_with(config)
    assert getattr(connector, active) is expected
    assert getattr(connector, inactive) is None
