from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake import (
    transfer as transfer_module,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.transfer import (
    AscendMooncakeTransfer,
)


def test_initialize_engine_binds_npu_and_uses_ascend_protocol():
    transfer = AscendMooncakeTransfer("producer-host", 3)
    engine = MagicMock()
    calls = []

    def initialize(*args):
        calls.append(("initialize", args))
        return 0

    engine.initialize.side_effect = initialize

    with patch.object(
        transfer_module.torch.npu,
        "set_device",
        side_effect=lambda device: calls.append(("set_device", device)),
    ) as set_device:
        result = transfer._initialize_engine(engine)

    assert result == 0

    set_device.assert_called_once_with(3)
    engine.initialize.assert_called_once_with(
        "producer-host",
        "P2PHANDSHAKE",
        "ascend",
        "",
    )
    assert calls == [
        ("set_device", 3),
        (
            "initialize",
            ("producer-host", "P2PHANDSHAKE", "ascend", ""),
        ),
    ]


def test_acquire_sources_merges_views_into_aligned_storage_region():
    alignment = 2 * 1024 * 1024
    raw = torch.empty(alignment + 1024, dtype=torch.uint8)
    offset = (-raw.data_ptr()) % alignment
    aligned = raw.narrow(0, offset, 512)
    source_a = aligned.narrow(0, 128, 64)
    source_b = aligned.narrow(0, 256, 64)

    transfer = AscendMooncakeTransfer("producer-host", 0)
    engine = MagicMock()
    engine.batch_register_memory.return_value = 0

    with patch.object(transfer, "_ensure_engine", return_value=engine):
        addresses = transfer.acquire_sources([source_a, source_b])

    assert addresses == [aligned.data_ptr()]
    engine.batch_register_memory.assert_called_once_with(
        [aligned.data_ptr()],
        [raw.untyped_storage().nbytes() - offset],
    )
