# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""One-card NPU check for Mooncake encoder-output fallback memory.

The two-card EPD test owns real Mooncake integration coverage. Keeping this
test focused on the NPU bounce copy avoids loading the native TransferEngine
extension in the pytest process, whose library teardown is unsafe there.
"""

from __future__ import annotations

import multiprocessing
import sys
from types import ModuleType

import pytest
import torch

pytest.importorskip("torch_npu")

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


def _npu_device() -> torch.device:
    device_index = torch.npu.current_device()
    torch.npu.set_device(device_index)
    return torch.device(f"npu:{device_index}")


def _byte_pattern(nbytes: int, device: torch.device) -> torch.Tensor:
    return torch.arange(nbytes, dtype=torch.int32, device=device).remainder_(251).to(torch.uint8)


def _run_npu_bounce_copy() -> None:
    mooncake = ModuleType("mooncake")
    mooncake.__path__ = []  # type: ignore[attr-defined]
    mooncake_engine = ModuleType("mooncake.engine")
    mooncake_engine.TransferEngine = object  # type: ignore[attr-defined]
    sys.modules["mooncake"] = mooncake
    sys.modules["mooncake.engine"] = mooncake_engine

    from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.memory import (
        ASCEND_DIRECT_MEMORY_ALIGNMENT,
        AscendProducerAllocator,
        AscendProducerMemoryPool,
    )

    device = _npu_device()
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


def test_npu_bounce_copy_is_byte_tight_and_visible():
    process = multiprocessing.get_context("spawn").Process(
        target=_run_npu_bounce_copy,
    )
    process.start()
    process.join(timeout=60)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("NPU bounce-copy subprocess timed out")
    assert process.exitcode == 0
