# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""Ascend hardware checks for Mooncake encoder-output fallback memory.

The NPU-only test runs wherever torch_npu is available. The real Mooncake
transfer test additionally requires ``VLLM_ASCEND_MOONCAKE_TEST_HOST`` to name
the local host address advertised by two in-process TransferEngine sessions.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.memory import (  # noqa: E402
    ASCEND_DIRECT_MEMORY_ALIGNMENT,
    AscendContiguousAllocator,
    AscendProducerAllocator,
    AscendProducerMemoryPool,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.transfer import (  # noqa: E402
    AscendMooncakeTransfer,
)

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(),
    reason="Ascend NPU is unavailable",
)


def _npu_device() -> tuple[torch.device, int]:
    device_index = torch.npu.current_device()
    torch.npu.set_device(device_index)
    return torch.device(f"npu:{device_index}"), device_index


def _byte_pattern(nbytes: int, device: torch.device) -> torch.Tensor:
    return torch.arange(nbytes, dtype=torch.int32, device=device).remainder_(251).to(torch.uint8)


def test_real_npu_bounce_copy_is_byte_tight_and_visible():
    device, device_index = _npu_device()
    allocator = AscendProducerAllocator(
        staging_capacity=4096,
        bounce_capacity=ASCEND_DIRECT_MEMORY_ALIGNMENT,
    )
    transfer = AscendMooncakeTransfer("127.0.0.1", device_index)
    pool = AscendProducerMemoryPool(4096, transfer, allocator)
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
        allocator.close(transfer)
        transfer.close()


def test_real_mooncake_bounce_prefix_and_registered_interior_suffix():
    hostname = os.environ.get("VLLM_ASCEND_MOONCAKE_TEST_HOST")
    if not hostname:
        pytest.skip("set VLLM_ASCEND_MOONCAKE_TEST_HOST for Mooncake hardware test")
    pytest.importorskip("mooncake.engine")

    device, device_index = _npu_device()
    producer_transfer = AscendMooncakeTransfer(hostname, device_index)
    consumer_transfer = AscendMooncakeTransfer(hostname, device_index)
    producer_allocator = AscendProducerAllocator(
        staging_capacity=ASCEND_DIRECT_MEMORY_ALIGNMENT,
        bounce_capacity=ASCEND_DIRECT_MEMORY_ALIGNMENT,
    )
    consumer_allocator = AscendContiguousAllocator(8192)
    producer_pool = AscendProducerMemoryPool(
        ASCEND_DIRECT_MEMORY_ALIGNMENT,
        producer_transfer,
        producer_allocator,
    )
    lease = None

    try:
        producer_allocator.prepare(device, producer_transfer)
        consumer_allocator.prepare(device, consumer_transfer)
        assert producer_allocator.tensor is not None
        assert consumer_allocator.tensor is not None

        source = producer_allocator.tensor.narrow(0, 123, 4096)
        source.copy_(_byte_pattern(source.nbytes, device))
        destination = consumer_allocator.tensor.narrow(0, 257, source.nbytes)
        destination.zero_()

        prefix_nbytes = 300
        lease = producer_pool.acquire_bounce(prefix_nbytes)
        assert lease is not None
        bounce_address = producer_pool.copy_to_bounce(
            lease,
            [(source, 0, prefix_nbytes)],
        )

        producer_transfer.write(
            consumer_transfer.local_session(),
            [bounce_address, source.data_ptr() + prefix_nbytes],
            [destination.data_ptr(), destination.data_ptr() + prefix_nbytes],
            [prefix_nbytes, source.nbytes - prefix_nbytes],
        )
        torch.npu.synchronize()

        assert torch.equal(destination.cpu(), source.cpu())
    finally:
        if lease is not None:
            producer_pool.release_bounce(lease)
        producer_allocator.close(producer_transfer)
        consumer_allocator.close(consumer_transfer)
        producer_transfer.close()
        consumer_transfer.close()
