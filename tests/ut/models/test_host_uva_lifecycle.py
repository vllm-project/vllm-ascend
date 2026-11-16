# SPDX-License-Identifier: Apache-2.0
"""Host allocation failures must preserve registration and ownership."""

import ctypes
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.models.deepseek_v41.engram import npu


@pytest.mark.parametrize("stage", ["view", "register", "device_pointer", "pointer_table"])
@pytest.mark.parametrize("cleanup", ["success", "unregister", "free"])
def test_constructor_cleanup_is_ordered_and_retryable(monkeypatch, stage, cleanup):
    backing = ctypes.create_string_buffer(131072)

    def malloc(pointer, size, flags):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.addressof(backing)
        return 0

    def device_pointer(pointer, address, flags):
        ctypes.cast(address, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.addressof(backing)
        return 1 if stage == "device_pointer" else 0

    lib = SimpleNamespace(
        aclrtMallocHost=Mock(side_effect=malloc),
        aclrtHostRegisterV2=Mock(return_value=1 if stage == "register" else 0),
        aclrtHostGetDevicePointer=Mock(side_effect=device_pointer),
        aclrtHostUnregister=Mock(side_effect=[1, 0] if cleanup == "unregister" else [0]),
        aclrtFreeHost=Mock(side_effect=[1, 0] if cleanup == "free" else [0]),
    )
    monkeypatch.setattr(npu, "_host_library", lambda: lib)
    monkeypatch.setattr(npu.DeviceOperator, "host_register_flags", lambda: 0)
    if stage == "view":
        monkeypatch.setattr(torch, "frombuffer", Mock(side_effect=ValueError("view failed")))

    if stage == "pointer_table":
        monkeypatch.setattr(torch, "tensor", Mock(side_effect=RuntimeError("pointer table failed")))

    registered = stage in ("device_pointer", "pointer_table")
    cleanup_fails = cleanup == "free" or (cleanup == "unregister" and registered)
    if cleanup_fails:
        with pytest.raises(npu.HostUvaBufferCleanupError) as caught:
            npu.HostUvaBuffer((4, 160), torch.bfloat16, "cpu")
        owner = caught.value.host_buffer
        assert owner.pointer.value
        assert owner._registered is (registered and cleanup == "unregister")
        if cleanup == "unregister":
            lib.aclrtFreeHost.assert_not_called()
        owner.close()
        owner.close()
        assert not owner.pointer.value
        assert owner.tensor is None and owner.ptrs is None
    else:
        error = ValueError if stage == "view" else RuntimeError
        with pytest.raises(error) as caught:
            npu.HostUvaBuffer((4, 160), torch.bfloat16, "cpu")
        assert not isinstance(caught.value, npu.HostUvaBufferCleanupError)

    expected_unregisters = (2 if cleanup == "unregister" else 1) if registered else 0
    assert lib.aclrtHostUnregister.call_count == expected_unregisters
    assert lib.aclrtFreeHost.call_count == (2 if cleanup == "free" else 1)


@pytest.mark.parametrize("failure", [None, "malloc", "null", "register", "device_pointer"])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("offset", [0, 4096, 65535])
def test_host_uva_aligns_registration_and_frees_original_allocation(monkeypatch, failure, dtype, offset):
    alignment, size = 65536, 4096

    class Library:
        def __init__(self):
            self.storage = ctypes.create_string_buffer(size + 4 * alignment)
            base = ctypes.addressof(self.storage)
            self.original = (base + alignment - 1) // alignment * alignment + offset
            self.registered = self.unregistered = self.freed = None
            self.free_count = self.unregister_count = 0

        def aclrtMallocHost(self, output, count, flags):
            assert count == size + alignment - 1
            if failure == "malloc":
                return 1
            ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p)).contents.value = (
                None if failure == "null" else self.original
            )
            return 0

        def aclrtHostRegisterV2(self, pointer, count, flags):
            assert count == size and pointer.value % alignment == 0
            self.registered = pointer.value
            return 1 if failure == "register" else 0

        def aclrtHostGetDevicePointer(self, pointer, output, flags):
            assert pointer.value == self.registered
            ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p)).contents.value = 0x100000000
            return 1 if failure == "device_pointer" else 0

        def aclrtHostUnregister(self, pointer):
            self.unregistered = pointer.value
            self.unregister_count += 1
            return 0

        def aclrtFreeHost(self, pointer):
            self.freed = pointer.value
            self.free_count += 1
            return 0

    lib = Library()
    monkeypatch.setattr(npu, "_host_library", lambda: lib)
    shape = (size // torch.empty((), dtype=dtype).element_size(),)
    if failure:
        message = "NULL pointer" if failure == "null" else "failed: rc=1"
        with pytest.raises(RuntimeError, match=message):
            npu.HostUvaBuffer(shape, dtype, "cpu")
    else:
        buffer = npu.HostUvaBuffer(shape, dtype, "cpu")
        assert buffer.tensor.data_ptr() == lib.registered
        assert lib.original <= lib.registered and lib.registered + size <= lib.original + size + alignment - 1
        buffer.tensor.fill_(17)
        value = torch.frombuffer(
            lib.storage, dtype=dtype, count=1, offset=lib.registered - ctypes.addressof(lib.storage)
        )
        assert value.item() == 17
        buffer.close()
        buffer.close()
    if failure in ("malloc", "null"):
        assert lib.registered is lib.unregistered is lib.freed is None
        assert lib.free_count == lib.unregister_count == 0
    elif failure == "register":
        assert lib.unregistered is None
        assert lib.unregister_count == 0
        assert lib.freed == lib.original and lib.free_count == 1
    else:
        assert lib.unregistered == lib.registered
        assert lib.freed == lib.original
        assert lib.free_count == lib.unregister_count == 1
