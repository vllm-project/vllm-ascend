# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import ctypes

import acl  # type: ignore[import-not-found, import-untyped]
import pytest
import torch
import torch_npu

from vllm_ascend.ops.fused_moe.routed_experts import _empty_like_preserving_npu_format

ACL_MEMCPY_DEVICE_TO_HOST = 2


def nz_tensor(source, *, fp4=False, semantic_fp4=False):
    tensor = source.to("npu")
    options = {"customize_dtype": torch.float8_e4m3fn}
    if fp4:
        options["input_dtype"] = torch_npu.float4_e2m1fn_x2
    if semantic_fp4:
        tensor = tensor.view(torch.float4_e2m1fn_x2)
        options["input_dtype"] = torch.float4_e2m1fn_x2
    return torch_npu.npu_format_cast(tensor, 29, **options)


def storage_bytes(tensor):
    torch.npu.synchronize()
    storage = tensor.untyped_storage()
    host = (ctypes.c_ubyte * storage.nbytes())()
    assert (
        acl.rt.memcpy(ctypes.addressof(host), len(host), storage.data_ptr(), len(host), ACL_MEMCPY_DEVICE_TO_HOST) == 0
    )
    return bytes(host)


def require_native_nz_copy(source):
    # Probe ordinary allocations before compaction. Old torch_npu releases lack
    # native NZ copy_ even though storage compaction itself is supported.
    destination = _empty_like_preserving_npu_format(source)
    assert source.storage_offset() == destination.storage_offset() == 0
    assert source.shape == destination.shape
    assert source.stride() == destination.stride()
    assert source.dtype == destination.dtype
    assert source.device == destination.device
    assert source.untyped_storage().nbytes() == destination.untyped_storage().nbytes()
    assert torch_npu.get_npu_format(source) == torch_npu.get_npu_format(destination)
    assert torch_npu.get_npu_format(source) in (29, 50)
    expected = storage_bytes(source)
    try:
        destination.copy_(source)
    except RuntimeError as error:
        message = str(error)
        if "NPUNativeFunctions::copy_ do not support internal format" in message and "ERR00007" in message:
            pytest.skip("requires torch_npu native-NZ copy_ support")
        raise
    assert storage_bytes(destination) == expected
