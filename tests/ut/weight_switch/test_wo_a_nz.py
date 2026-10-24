# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Grouped wo_a NZ layout and ND communication regressions (CPU mocks)."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch_npu
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import AscendW8A8MXFP8DSDynamicLinearMethod
from vllm_ascend.weight_switch.linear import WeightSwitchConfig, _is_nz_batched_weight


@pytest.fixture
def nz_runtime(monkeypatch):
    # Track storage so detach/data/set_ preserve the simulated layout. Reject
    # generic copies/clones of NZ, as the Ascend 950 runtime does.
    layouts = {}
    original_copy, original_clone = torch.Tensor.copy_, torch.Tensor.clone

    def get_format(tensor):
        return layouts.get(tensor.untyped_storage().data_ptr(), 2)

    def format_cast(tensor, fmt, **kwargs):
        if get_format(tensor) == fmt:
            return tensor
        result = original_clone(tensor)
        layouts[result.untyped_storage().data_ptr()] = fmt
        return result

    def checked_copy(dst, src):
        assert get_format(dst) == get_format(src) == 2
        return original_copy(dst, src)

    def checked_clone(tensor, *args, **kwargs):
        assert get_format(tensor) == 2
        return original_clone(tensor, *args, **kwargs)

    def copy_memory(dst, src, *, non_blocking):
        assert non_blocking and get_format(dst) == get_format(src) == 29
        assert dst.shape == src.shape and dst.dtype == src.dtype
        return original_copy(dst, src)

    def all_gather(tensor, group, *, output, async_op):
        assert get_format(tensor) == get_format(output) == 2
        assert tensor.is_contiguous() and output.is_contiguous()
        chunks = [tensor.float() + rank for rank in range(group.world_size)]
        original_copy(output, torch.cat(chunks).to(output.dtype))
        return output, Mock() if async_op else None

    config = SimpleNamespace(weight_nz_mode=2)
    monkeypatch.setattr("vllm_ascend.utils.get_ascend_config", lambda: config)
    monkeypatch.setattr("vllm_ascend.utils.get_current_hardware_profile", Mock(return_value=Mock()))
    monkeypatch.setattr(torch_npu, "get_npu_format", get_format, raising=False)
    monkeypatch.setattr(torch_npu, "npu_format_cast", format_cast)
    monkeypatch.setattr(torch.Tensor, "copy_", checked_copy)
    monkeypatch.setattr(torch.Tensor, "clone", checked_clone)
    monkeypatch.setattr(torch.ops.npu, "copy_memory_", copy_memory, raising=False)
    monkeypatch.setattr("vllm_ascend.distributed.utils.all_gather_async", all_gather)
    return SimpleNamespace(config=config, get_format=get_format, cast=format_cast, copy=original_copy)


def make_layer(dtype=torch.bfloat16, groups=2, k=64, n=32):
    layer = torch.nn.Module()
    layer.prefix = "model.layers.0.attn.wo_a"
    weight = (torch.arange(groups * k * n).reshape(groups, k, n) % 4).to(dtype)
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.n_local_groups, layer.o_lora_rank = groups, n
    layer.skip_weight_nz_conversion = True
    layer.input_size = layer.input_size_per_partition = k
    layer.output_size, layer.output_size_per_partition = groups * n * 2, groups * n
    if dtype == torch.float8_e4m3fn:
        layer.weight_scale = torch.nn.Parameter(torch.ones(groups, 1, n, 2, dtype=torch.uint8), requires_grad=False)
    return layer


@pytest.mark.parametrize("mx_fusion", [False, True], ids=["A3", "A5"])
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("already_grouped", [False, True])
def test_bf16_post_load(nz_runtime, mx_fusion, mode, already_grouped):
    layer = make_layer()
    expected = layer.weight.float().clone()
    if not already_grouped:
        layer.weight.data = layer.weight.transpose(1, 2).reshape(-1, 64).contiguous()
    nz_runtime.config.weight_nz_mode = mode
    profile = Mock()
    profile.supports.return_value = mx_fusion
    with (
        patch.object(UnquantizedLinearMethod, "process_weights_after_loading"),
        patch("vllm_ascend.ops.linear.get_current_hardware_profile", return_value=profile),
    ):
        method = AscendUnquantizedLinearMethod.__new__(AscendUnquantizedLinearMethod)
        method.process_weights_after_loading(layer)
    torch.testing.assert_close(layer.weight.float(), expected)
    assert nz_runtime.get_format(layer.weight) == (29 if mode == 2 else 2)


@pytest.mark.parametrize("k,n", [(17, 32), (64, 17), (1, 32), (64, 1)])
def test_bf16_unaligned_weight_stays_nd(nz_runtime, k, n):
    layer = make_layer(k=k, n=n)
    with patch.object(UnquantizedLinearMethod, "process_weights_after_loading"):
        AscendUnquantizedLinearMethod.__new__(AscendUnquantizedLinearMethod).process_weights_after_loading(layer)
    assert nz_runtime.get_format(layer.weight) == 2


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("block", [32, 128])
@pytest.mark.parametrize("scale_dtype", [torch.float32, torch.float8_e8m0fnu])
def test_mxfp8_post_load_preserves_scale_ownership(nz_runtime, mode, block, scale_dtype):
    method = AscendW8A8MXFP8DSDynamicLinearMethod.__new__(AscendW8A8MXFP8DSDynamicLinearMethod)
    method.block_size, method.group_size, method.o_lora_rank = block, 32, 128
    method.n_local_groups = 8  # Loaded OTP/CP/PCP shards determine the group count.
    layer = make_layer(torch.float8_e4m3fn, k=256, n=128)
    expected = layer.weight.float().clone()
    layer.weight.data = layer.weight.transpose(1, 2).reshape(-1, 256).contiguous()
    codes = (torch.arange(256 // block * (256 // block)) % 3 + 126).to(torch.uint8).reshape(256 // block, -1)
    scales = codes.view(scale_dtype) if scale_dtype == torch.float8_e8m0fnu else torch.pow(2.0, codes.float() - 127)
    layer.weight_scale.data = scales
    nz_runtime.config.weight_nz_mode = mode
    method.process_weights_after_loading(layer)
    torch.testing.assert_close(layer.weight.float(), expected)
    assert nz_runtime.get_format(layer.weight) == (29 if mode else 2)
    assert nz_runtime.get_format(layer.weight_scale) == 2
    assert layer.weight.is_contiguous() and layer.weight_scale.is_contiguous()
    expected_scales = codes.repeat_interleave(block, 0).repeat_interleave(block // 32, 1)
    expected_scales = expected_scales.reshape(2, 128, 4, 2).transpose(1, 2).contiguous()
    torch.testing.assert_close(layer.weight_scale, expected_scales)


@pytest.fixture
def simulated_npu(nz_runtime, monkeypatch):
    # CPU tensors cannot carry a real NPU device; bypass only the device guard.
    monkeypatch.setattr(
        "vllm_ascend.weight_switch.linear._is_nz_batched_weight",
        lambda layer, attr, tensor: attr == "weight"
        and layer.prefix.endswith("wo_a")
        and tensor.ndim == 3
        and nz_runtime.get_format(tensor) != 2,
    )
    return nz_runtime


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("clone", [False, True])
@pytest.mark.parametrize("async_op", [False, True])
@pytest.mark.parametrize("world_size", [2, 4])
def test_weight_switch_values_formats_and_addresses(simulated_npu, dtype, clone, async_op, world_size):
    runtime = simulated_npu
    layer = make_layer(dtype)
    layer.weight.data = runtime.cast(layer.weight, 29)
    original_pointer = layer.weight.data_ptr()
    cls = AscendUnquantizedLinearMethod if dtype == torch.bfloat16 else AscendW8A8MXFP8DSDynamicLinearMethod
    method = cls.__new__(cls)
    config = WeightSwitchConfig(group=SimpleNamespace(world_size=world_size), world_size=world_size, rank=0)
    state = method.enable_weight_switch(layer, config, clone_local_tensors=clone)
    part = state.gather_parts["weight"]
    assert (part.local_tensor.data_ptr() != original_pointer) == clone
    pointers = (part.local_tensor.data_ptr(), part.full_tensor.data_ptr())
    assert part.is_nz_weight
    for value in (1, 4, 2):
        runtime.copy(part.local_tensor, torch.full_like(part.local_tensor, value))
        method.all_gather_weight(state, config, async_op=async_op)
        method.wait_weight_all_gather(state)
        method.switch_weight(layer, state, use_full_weight=True)
        expected = torch.cat([torch.full((2, 64, 32), value + rank) for rank in range(world_size)])
        torch.testing.assert_close(layer.weight.float(), expected.float())
        assert runtime.get_format(layer.weight) == 29 and layer.weight.data_ptr() == pointers[1]
        if dtype == torch.float8_e4m3fn:
            expected_scale = torch.cat(
                [torch.full((2, 1, 32, 2), 1 + rank, dtype=torch.uint8) for rank in range(world_size)]
            )
            torch.testing.assert_close(layer.weight_scale, expected_scale)
            assert runtime.get_format(layer.weight_scale) == 2
        method.switch_weight(layer, state, use_full_weight=False)
        torch.testing.assert_close(layer.weight.float(), torch.full((2, 64, 32), float(value)))
        assert runtime.get_format(layer.weight) == 29 and layer.weight.data_ptr() == pointers[0]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_shared_pool_and_pending_gather(simulated_npu, dtype):
    runtime, pool = simulated_npu, {}
    cls = AscendUnquantizedLinearMethod if dtype == torch.bfloat16 else AscendW8A8MXFP8DSDynamicLinearMethod
    method = cls.__new__(cls)
    config = WeightSwitchConfig(group=SimpleNamespace(world_size=2), world_size=2, rank=0)
    states = []
    for _ in range(2):
        layer = make_layer(dtype)
        layer.weight.data = runtime.cast(layer.weight, 29)
        states.append(method.enable_weight_switch(layer, config, pool=pool, pool_key_prefix="cp"))
    first, second = [state.gather_parts["weight"] for state in states]
    assert first.gather_output.data_ptr() == second.gather_output.data_ptr()
    assert first.full_tensor.data_ptr() == second.full_tensor.data_ptr()
    method.all_gather_weight(states[0], config)
    with pytest.raises(RuntimeError, match="still pending"):
        method.all_gather_weight(states[0], config)
    method.wait_weight_all_gather(states[0])


def test_nd_weight_uses_original_gather_buffer(nz_runtime):
    layer = make_layer()
    method = AscendUnquantizedLinearMethod.__new__(AscendUnquantizedLinearMethod)
    config = WeightSwitchConfig(group=SimpleNamespace(world_size=2), world_size=2, rank=0)
    state = method.enable_weight_switch(layer, config)
    part = state.gather_parts["weight"]
    assert not part.is_nz_weight and part.full_tensor is part.gather_output


def test_nz_detection_does_not_inspect_cpu_or_meta_storage(nz_runtime):
    layer = make_layer()
    with patch.object(torch_npu, "get_npu_format", side_effect=AssertionError("unexpected storage inspection")):
        assert not _is_nz_batched_weight(layer, "weight", layer.weight)
        assert not _is_nz_batched_weight(layer, "weight", torch.empty(2, 64, 32, device="meta"))
    tensor = Mock(ndim=3, device=SimpleNamespace(type="npu"))
    with patch.object(torch_npu, "get_npu_format", return_value=29):
        assert _is_nz_batched_weight(layer, "weight", tensor)
        assert not _is_nz_batched_weight(layer, "weight_scale", tensor)
        layer.prefix = "model.layers.0.attn.wo_b"
        assert not _is_nz_batched_weight(layer, "weight", tensor)
