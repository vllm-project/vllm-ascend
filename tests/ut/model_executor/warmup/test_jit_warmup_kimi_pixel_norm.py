# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""CPU tests for pixel-normalization registration, compile keys and launches."""

import inspect
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.model_executor.warmup.jit_warmup_triton_helper import TritonWarmupTensor

from tests.ut.model_executor.warmup.helpers import make_registry
from vllm_ascend.ops import kimi_pixel_norm as norm_ops
from vllm_ascend.ops.triton import kimi_pixel_norm as pixel_ops


def _make_norm(dtype):
    reference = torch.linspace(-1, 1, 256, device="cpu").reshape(256, 1).expand(-1, 3)
    return norm_ops.KimiPixelNormLut(reference, dtype=dtype, device=torch.device("cpu"))


@pytest.fixture(params=[False, True], ids=["configured-kernel", "plain-function"])
def kernel_owner(request, monkeypatch):
    if request.param:
        kernel = pixel_ops.KimiPixelNormKernel.kernel
        # Without Triton, its placeholder decorator leaves a plain function.
        monkeypatch.setattr(pixel_ops.KimiPixelNormKernel, "kernel", getattr(kernel, "fn", kernel))
    owner = pixel_ops.KimiPixelNormKernel()
    kernel = Mock()
    # Class access avoids binding away `pixels` when the kernel is a function.
    kernel_definition = type(owner).kernel
    kernel.arg_names = list(inspect.signature(getattr(kernel_definition, "fn", kernel_definition)).parameters)
    kernel.__getitem__ = Mock(return_value=Mock())
    owner.kernel = kernel
    return owner


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_registration_is_scoped_deduplicated_and_uses_final_dtype(dtype):
    norm = _make_norm(torch.bfloat16)
    owner = pixel_ops.KimiPixelNormKernel()
    registry = make_registry()
    with (
        patch.object(norm_ops, "_KIMI_PIXEL_NORM_KERNEL", owner),
        patch.object(owner, "get_warmup_keys", wraps=owner.get_warmup_keys) as expand_keys,
        patch.object(torch, "empty", side_effect=AssertionError("registration must not allocate payloads")),
    ):
        norm.register_warmup()
        assert len(registry) == 0
        with registry.activate():
            norm.register_warmup()
            norm.register_warmup()
        assert len(registry) == 1
        expand_keys.assert_not_called()

    # Weight loading may change dtype after registration. Expansion must restore
    # the table and use that final dtype, rather than the registration-time dtype.
    norm.to(dtype=dtype)
    expected = norm.lut.clone()
    norm.lut.zero_()
    ((args, kwargs),) = registry._registrations[owner]
    keys = owner.get_warmup_keys(*args, **kwargs)
    assert len(keys) == 3
    assert set(keys) == {owner.CompileKey(dtype, block) for block in (1024, 4096, 8192)}
    assert torch.equal(norm.lut.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_compile_uses_metadata_only_and_does_not_launch(kernel_owner, dtype):
    owner = kernel_owner
    keys = owner.get_warmup_keys(_make_norm(dtype))
    with patch.object(torch, "empty", side_effect=AssertionError("compile must not allocate payloads")):
        for key in keys:
            inputs = owner.warmup_inputs(key)
            for name, tensor_dtype in (("pixels", torch.uint8), ("lut", dtype), ("output", dtype)):
                assert isinstance(inputs[name], TritonWarmupTensor)
                assert inputs[name].dtype == tensor_dtype
                assert inputs[name].data_ptr() == 0
            assert inputs["lut"].shape == (256,)
            assert inputs["BLOCK"] == key.block
            owner.compile(key)

    kernel = owner.kernel
    assert kernel.warmup.call_count == 3
    kernel.__getitem__.assert_not_called()
    assert {call.kwargs["BLOCK"] for call in kernel.warmup.call_args_list} == {1024, 4096, 8192}
    for call in kernel.warmup.call_args_list:
        assert call.kwargs["grid"] == (1,)
        assert call.kwargs["count"] == 2
        assert "programs" not in call.kwargs


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "count,block,programs",
    [
        (0, None, 0),
        (1, 1024, 1),
        (196607, 1024, 24),
        (196608, 1024, 24),
        (196609, 4096, 24),
        (2408447, 4096, 24),
        (2408448, 4096, 24),
        (2408449, 8192, 24),
        (2**31 - 2, 8192, 24),
    ],
)
def test_warmup_covers_runtime_dispatch_boundaries(kernel_owner, dtype, count, block, programs):
    owner = kernel_owner
    for key in owner.get_warmup_keys(_make_norm(dtype)):
        owner.compile(key)
    warmed = {(call.kwargs["lut"].dtype, call.kwargs["BLOCK"]) for call in owner.kernel.warmup.call_args_list}
    device = SimpleNamespace(type="npu")
    pixels = SimpleNamespace(device=device, dtype=torch.uint8, numel=lambda: count, is_contiguous=lambda: True)
    lut = SimpleNamespace(device=device, dtype=dtype)
    output = object()
    with (
        patch.object(pixel_ops, "_KIMI_PIXEL_NORM_KERNEL", owner),
        patch.object(torch, "empty_like", return_value=output),
        patch.object(torch.npu, "device", return_value=nullcontext()),
    ):
        assert pixel_ops.kimi_pixel_norm_lut(pixels, lut, vector_cores=24) is output

    if count == 0:
        owner.kernel.__getitem__.assert_not_called()
        return
    owner.kernel.__getitem__.assert_called_once_with((programs,))
    launch = owner.kernel.__getitem__.return_value
    launch.assert_called_once_with(pixels=pixels, lut=lut, output=output, count=count, BLOCK=block)
    assert (dtype, launch.call_args.kwargs["BLOCK"]) in warmed
    assert owner.kernel.warmup.call_count == 3
