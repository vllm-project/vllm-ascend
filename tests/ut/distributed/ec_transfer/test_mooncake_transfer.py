import os
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.transfer import (
    AscendMooncakeTransfer,
)


@pytest.mark.parametrize(
    "selectors",
    [
        {},
        {"HCCL_INTRA_PCIE_ENABLE": "0", "HCCL_INTRA_ROCE_ENABLE": "1"},
        {"HCCL_INTRA_PCIE_ENABLE": "1", "HCCL_INTRA_ROCE_ENABLE": "0"},
    ],
)
def test_transfer_lazily_reuses_process_ascend_engine(selectors):
    transfer = AscendMooncakeTransfer("producer-host", 3)
    engine = MagicMock()
    engine.batch_register_memory.return_value = 0
    environment = {
        "ASCEND_GLOBAL_RESOURCE_CONFIG": '{"comm_resource_config":{"protocol_desc":["hccs:device"]}}',
        **selectors,
    }

    with (
        patch.dict(os.environ, environment, clear=True),
        patch(
            "vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.transfer.global_te.get_transfer_engine",
            return_value=engine,
        ) as get_transfer_engine,
        patch("vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.transfer.torch.npu.set_device") as set_device,
    ):
        transfer.ensure_initialized()
        transfer.ensure_initialized()
        assert dict(os.environ) == environment
        tensor = MagicMock(nbytes=128)
        tensor.data_ptr.return_value = 2097152
        assert transfer.register_memory(tensor) == 0
        assert dict(os.environ) == environment

    set_device.assert_called_once_with(3)
    get_transfer_engine.assert_called_once_with(
        "producer-host",
        device_name=None,
    )


def test_transient_sources_are_rejected_before_engine_initialization():
    transfer = AscendMooncakeTransfer("host", 0)
    with patch.object(transfer, "_ensure_engine") as engine, pytest.raises(RuntimeError):
        transfer.acquire_sources([])
    engine.assert_not_called()


def test_fixed_registrations_never_deregister_on_pool_or_transfer_close():
    transfer = AscendMooncakeTransfer("host", 0)
    engine = MagicMock()
    engine.batch_register_memory.return_value = 0
    tensor = MagicMock(nbytes=128)
    tensor.data_ptr.return_value = 2097152
    transfer._engine = engine
    assert transfer.register_memory(tensor) == 0
    assert not transfer.unregister_memory(tensor)
    transfer.close()
    engine.unregister_memory.assert_not_called()
    engine.batch_unregister_memory.assert_not_called()
    engine.close.assert_not_called()
    assert any(owner is tensor for owner in transfer._retained_owners)
