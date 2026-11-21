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
        assert transfer._ensure_engine() is engine
        assert transfer._ensure_engine() is engine
        assert dict(os.environ) == environment
        tensor = MagicMock(nbytes=128)
        tensor.data_ptr.return_value = 2097152
        assert transfer.register_memory(tensor) == 0
        assert any(owner is tensor for owner in transfer._retained_owners)
        assert dict(os.environ) == environment

    set_device.assert_called_once_with(3)
    get_transfer_engine.assert_called_once_with(
        "producer-host",
        device_name=None,
    )
