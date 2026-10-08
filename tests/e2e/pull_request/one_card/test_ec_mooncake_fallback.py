# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""One-card NPU check for Mooncake encoder-output fallback memory.

The two-card EPD test owns real Mooncake integration coverage. Keeping this
test focused on the NPU bounce copy avoids constructing the process-wide
native TransferEngine, whose library teardown is unsafe in the pytest process.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("torch_npu")

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.memory import (  # noqa: E402
    ASCEND_DIRECT_MEMORY_ALIGNMENT,
    AscendProducerAllocator,
    AscendProducerMemoryPool,
)

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(),
    reason="Ascend NPU is unavailable",
)


class _RegistrationStub:
    """Provide only the allocator registration API used by this test."""

    def __init__(self) -> None:
        self.registered: set[int] = set()

    def register_memory(self, tensor: torch.Tensor) -> int:
        self.registered.add(tensor.data_ptr())
        return 0

    def unregister_memory(self, tensor: torch.Tensor) -> bool:
        self.registered.remove(tensor.data_ptr())
        return True


def _npu_device() -> tuple[torch.device, int]:
    device_index = torch.npu.current_device()
    torch.npu.set_device(device_index)
    return torch.device(f"npu:{device_index}"), device_index


def _byte_pattern(nbytes: int, device: torch.device) -> torch.Tensor:
    return torch.arange(nbytes, dtype=torch.int32, device=device).remainder_(251).to(torch.uint8)


def test_npu_bounce_copy_is_byte_tight_and_visible():
    device, _ = _npu_device()
    allocator = AscendProducerAllocator(
        staging_capacity=4096,
        bounce_capacity=ASCEND_DIRECT_MEMORY_ALIGNMENT,
    )
    transfer = _RegistrationStub()
    pool = AscendProducerMemoryPool(4096, transfer, allocator)  # type: ignore[arg-type]
    lease = None

    try:
        allocator.prepare(device, transfer)
        bounce = pool.bounce_tensor
        assert bounce is not None
        bounce.zero_()

        first = _byte_pattern(300, device)
        second = _byte_pattern(500, device).flip(0).contiguous()
        lease = pool.acquire_bounce(800)
        assert lease is not None

        address = pool.copy_to_bounce(
            lease,
            [(first, 0, first.nbytes), (second, first.nbytes, second.nbytes)],
        )

        expected = torch.cat((first, second))
        actual = bounce.narrow(0, lease.offset, expected.nbytes)
        assert address == bounce.data_ptr() + lease.offset
        assert torch.equal(actual.cpu(), expected.cpu())
        assert int(bounce[lease.offset + expected.nbytes].cpu()) == 0
    finally:
        if lease is not None:
            pool.release_bounce(lease)
        assert allocator.close(transfer)  # type: ignore[arg-type]
        assert not transfer.registered
