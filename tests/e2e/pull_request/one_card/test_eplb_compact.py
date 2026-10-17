# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import ctypes
import gc
from importlib import import_module

import acl  # type: ignore[import-not-found, import-untyped]
import pytest
import torch
import torch_npu

from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily, get_current_hardware_profile
from vllm_ascend.ops.fused_moe.routed_experts import _empty_like_preserving_npu_format

ACL_MEMCPY_DEVICE_TO_HOST = 2
pytestmark = pytest.mark.skipif(
    get_current_hardware_profile().device_adaptor_family != DeviceAdaptorFamily.FP8_OPTIMIZED,
    reason="NZ MXFP8/MXFP4 compaction requires A5",
)


@pytest.mark.parametrize("fp4", [False, True])
def test_compact_nz_aliases_copy_and_graph(fp4):
    torch.npu.set_device(0)
    import_module("vllm_ascend.vllm_ascend_C")
    if fp4:
        source = torch.randint(0, 256, (128, 64), dtype=torch.uint8, device="npu")
        weight = torch_npu.npu_format_cast(
            source, 29, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
        )
    else:
        source = torch.randint(-3, 4, (128, 128), dtype=torch.int8).to(torch.float8_e4m3fn).to("npu")
        weight = torch_npu.npu_format_cast(source, 29, customize_dtype=torch.float8_e4m3fn)
    parameter = torch.nn.Parameter(weight, requires_grad=False)
    alias = parameter.detach().transpose(0, 1)
    options = {"customize_dtype": torch.float8_e4m3fn}
    if fp4:
        options["input_dtype"] = torch_npu.float4_e2m1fn_x2
    buffer = torch_npu.npu_format_cast(
        source.clone(),
        29,
        **options,
    )
    replacement = torch_npu.npu_format_cast(source.cpu().flip(0).contiguous().to("npu"), 29, **options)
    original = parameter.cpu().view(torch.uint8)
    old_pointer = parameter.data_ptr()
    storage_id = parameter.untyped_storage()._cdata
    metadata = (parameter.shape, parameter.stride(), torch_npu.get_npu_format(parameter))
    base, capacity, payload, count = torch.ops._C_ascend.compact_eplb_weights([parameter, alias, weight, buffer])
    assert count == 2
    assert capacity - payload < 2 * 1024**2 + count * 512
    assert base <= parameter.data_ptr() < base + capacity
    assert parameter.data_ptr() != old_pointer
    assert parameter.untyped_storage()._cdata == storage_id
    assert parameter.data_ptr() == weight.data_ptr() == alias.data_ptr()
    assert metadata == (parameter.shape, parameter.stride(), torch_npu.get_npu_format(parameter))
    torch.testing.assert_close(parameter.cpu().view(torch.uint8), original, rtol=0, atol=0)

    # Capture NZ copy_ with external storage, then clear the ordinary allocator
    # and replay after changing the source. Graph addresses must stay valid.
    buffer.copy_(parameter)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        buffer.copy_(parameter)
    parameter.copy_(replacement)
    gc.collect()
    torch.npu.empty_cache()
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(buffer.cpu().view(torch.uint8), replacement.cpu().view(torch.uint8), rtol=0, atol=0)
    assert base <= buffer.data_ptr() < base + capacity


def test_compact_rejects_cpu_before_moving_storage():
    import_module("vllm_ascend.vllm_ascend_C")
    tensor = torch.ones(16)
    address = tensor.data_ptr()
    with pytest.raises(RuntimeError, match="NPU tensors"):
        torch.ops._C_ascend.compact_eplb_weights([tensor])
    assert tensor.data_ptr() == address


def test_compact_waits_for_pending_nz_operations():
    torch.npu.set_device(0)
    import_module("vllm_ascend.vllm_ascend_C")
    expected = torch.arange(128 * 128, dtype=torch.float32).reshape(128, 128).remainder(7).to(torch.bfloat16)
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        source = expected.to("npu")
        weight = torch_npu.npu_format_cast(source.to(torch.float8_e4m3fn), 29, customize_dtype=torch.float8_e4m3fn)
        for _ in range(32):
            source.add_(1)
            weight.copy_(
                torch_npu.npu_format_cast(source.to(torch.float8_e4m3fn), 29, customize_dtype=torch.float8_e4m3fn)
            )
    # No CPU read or stream synchronization before moving the live storage.
    torch.ops._C_ascend.compact_eplb_weights([weight])
    with torch.npu.stream(stream):
        result = torch_npu.npu_format_cast(weight, 2)
    del weight
    gc.collect()
    torch.testing.assert_close(
        result.cpu().view(torch.uint8), (expected + 32).to(torch.float8_e4m3fn).view(torch.uint8), rtol=0, atol=0
    )


def test_compact_semantic_fp4_eplb_buffer_and_graph():
    # Kimi SiTU w13 uses the semantic FP4 dtype rather than a uint8 carrier.
    torch.npu.set_device(0)
    import_module("vllm_ascend.vllm_ascend_C")
    raw = torch.randint(0, 256, (128, 64), dtype=torch.uint8)

    def nz(source):
        return torch_npu.npu_format_cast(
            source.to("npu").view(torch.float4_e2m1fn_x2),
            29,
            customize_dtype=torch.float8_e4m3fn,
            input_dtype=torch.float4_e2m1fn_x2,
        )

    def storage_bytes(tensor):
        torch.npu.synchronize()
        storage = tensor.untyped_storage()
        host = (ctypes.c_ubyte * storage.nbytes())()
        assert (
            acl.rt.memcpy(ctypes.addressof(host), len(host), storage.data_ptr(), len(host), ACL_MEMCPY_DEVICE_TO_HOST)
            == 0
        )
        return bytes(host)

    weight = nz(raw)
    replacement = nz(raw.flip(0).contiguous())
    buffer = _empty_like_preserving_npu_format(weight)
    assert buffer.dtype == weight.dtype == torch.float4_e2m1fn_x2
    assert buffer.untyped_storage().nbytes() == weight.untyped_storage().nbytes()
    expected = storage_bytes(weight)
    torch.ops._C_ascend.compact_eplb_weights([weight, buffer])
    assert storage_bytes(weight) == expected
    buffer.copy_(weight)
    torch.npu.synchronize()
    assert storage_bytes(buffer) == expected
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        buffer.copy_(weight)
    weight.copy_(replacement)
    torch.npu.empty_cache()
    graph.replay()
    torch.npu.synchronize()
    assert storage_bytes(buffer) == storage_bytes(replacement)
