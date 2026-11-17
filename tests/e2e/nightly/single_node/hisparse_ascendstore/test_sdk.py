# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Opt-in registration test for MemCache SDK 1.3.0 and a dedicated host_tcp pool.

Start a dedicated MemCache metadata service and set MMC_LOCAL_CONFIG_PATH as
for the serving instance. Pass --hisparse-sdk to run on NPU 0.
"""

from importlib.metadata import version

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.config import ParallelConfig

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import (
    MemcacheBackend,
    require_hisparse_host_tcp,
)


def test_real_memcache_registration_rollback_and_retry(pytestconfig):
    if not pytestconfig.getoption("--hisparse-sdk"):
        pytest.skip("Pass --hisparse-sdk with a dedicated MemCache host_tcp service")
    assert version("memcache_hybrid") == "1.3.0"
    assert torch.npu.is_available()
    require_hisparse_host_tcp()
    torch.npu.set_device(0)
    alignment = 2 * 1024 * 1024
    allocations = [torch.empty(3 * alignment, dtype=torch.uint8, device="npu:0") for _ in range(2)]
    first, second = [tensor.data_ptr() + (-tensor.data_ptr() % alignment) for tensor in allocations]
    backend = MemcacheBackend(ParallelConfig(), device_id=0)
    try:
        sdk = backend.store
        assert sdk is not None
        backend.register_buffer([first], [alignment], check_result=True)
        assert sdk.register_buffer(first, alignment) == 0
        assert sdk.register_buffer(first, 2 * alignment) == -1
        # Use a valid address inside the second allocation, but break the
        # device registration's 2 MiB alignment on the final entry. The
        # rollback must preserve the first range from a prior registration.
        with pytest.raises(RuntimeError, match="MemCache buffer registration failed"):
            backend.register_buffer([first, second, second + 1], [alignment] * 3, check_result=True)
        assert backend._pending_buffers is None
        assert sdk.register_buffer(first, 2 * alignment) == -1
        backend.register_buffer([second], [alignment], check_result=True)
        backend.unregister_buffer([second], [alignment])
        backend.register_buffer([second], [2 * alignment], check_result=True)
        backend.unregister_buffer([first, second], [alignment, 2 * alignment])
    finally:
        backend.close()
