from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.transfer import (
    AscendMooncakeTransfer,
)


def test_transfer_lazily_reuses_process_ascend_engine():
    transfer = AscendMooncakeTransfer("producer-host", 3)
    engine = MagicMock()

    with (
        patch(
            "vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.transfer.global_te.get_transfer_engine",
            return_value=engine,
        ) as get_transfer_engine,
        patch("vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.transfer.torch.npu.set_device") as set_device,
    ):
        assert transfer._ensure_engine() is engine
        assert transfer._ensure_engine() is engine

    set_device.assert_called_once_with(3)
    get_transfer_engine.assert_called_once_with(
        "producer-host",
        device_name=None,
    )


@pytest.mark.parametrize("method,args", [("acquire_sources", [[]]), ("acquire_registration_ranges", [()])])
def test_transient_registrations_are_rejected_before_engine_use(method, args):
    transfer = AscendMooncakeTransfer("host", 0)
    with patch.object(transfer, "_ensure_engine") as engine, pytest.raises(RuntimeError):
        getattr(transfer, method)(*args)
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
