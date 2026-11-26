# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native LoRA layout checks with one request selected at a time."""
# ruff: noqa: E402

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
if not torch.npu.is_available():
    pytest.skip("LoRA layout acceptance requires a real NPU", allow_module_level=True)

from vllm.config import LoRAConfig
from vllm.lora.layers import LoRAMapping

from vllm_ascend.lora.punica_npu import PunicaWrapperNPU
from vllm_ascend.utils import enable_custom_op


@pytest.fixture(scope="module", autouse=True)
def initialize_npu_operators():
    torch.npu.set_device(0)
    assert enable_custom_op(), "GLM LoRA projection acceptance requires Ascend custom operators"


@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("zero", [False, True])
@pytest.mark.parametrize("slices,width", [((64, 64, 64, 32, 32, 32), 64), ((512, 512, 512, 4, 128, 128), 4096)])
def test_six_unequal_slices_switch_adapters_between_single_requests(prefill, zero, slices, width):
    torch.npu.set_device(0)
    torch.manual_seed(53)
    device, rank = torch.device("npu:0"), 8
    config = LoRAConfig(max_loras=2, max_lora_rank=rank, lora_dtype=torch.bfloat16)
    punica = PunicaWrapperNPU(8, 8, device, lora_config=config)
    count = 5 if prefill else 1
    x = torch.randn(count, width, device=device, dtype=torch.bfloat16) * 0.1
    a = tuple(torch.randn(2, 1, rank, width, device=device, dtype=torch.bfloat16) * 0.1 for _ in slices)
    b = tuple(torch.randn(2, 1, size, rank, device=device, dtype=torch.bfloat16) * 0.1 for size in slices)
    if zero:
        for factor in b:
            factor.zero_()
    for adapter_id in (1, 0, 2, 1):
        punica.update_metadata(LoRAMapping((adapter_id,) * count, (adapter_id,), is_prefill=prefill), [1, 2], 3, 128)
        output = torch.zeros(count, sum(slices), device=device, dtype=torch.bfloat16)
        punica.add_lora_linear(output, x, a, b, 1.0, slices)
        torch.npu.synchronize()
        expected = torch.zeros_like(output, device="cpu", dtype=torch.float64)
        if adapter_id:
            start = 0
            for size, aa, bb in zip(slices, a, b):
                expected[:, start : start + size] = (x.cpu().double() @ aa[adapter_id - 1, 0].cpu().double().T) @ bb[
                    adapter_id - 1, 0
                ].cpu().double().T
                start += size
        torch.testing.assert_close(output.cpu().double(), expected, atol=5e-4, rtol=3e-2)
        if not adapter_id or zero:
            assert not output.count_nonzero().item()
        else:
            assert expected.abs().max().item() > 5e-3


@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("offset", [1540, 1668])
def test_kda_split_view_shrink_matches_contiguous_input(prefill, offset):
    torch.manual_seed(5312)
    device, rank, width = torch.device("npu:0"), 8, 128
    config = LoRAConfig(max_loras=2, max_lora_rank=rank, lora_dtype=torch.bfloat16)
    punica = PunicaWrapperNPU(32, 32, device, lora_config=config)
    rows = 5 if prefill else 1
    punica.update_metadata(LoRAMapping((1,) * rows, (1,), is_prefill=prefill), [1, 2], 3, 154880)
    # TP16 KDA local outputs are [512, 512, 512, 4, 128, 128].
    backing = torch.randn(rows, 1796, device=device, dtype=torch.bfloat16) * 0.1
    inputs = backing[:, offset : offset + width]
    assert inputs.stride() == (1796, 1)
    a = torch.randn(2, 1, rank, width, device=device, dtype=torch.bfloat16) * 0.1
    outputs = []
    for value in (inputs, inputs.contiguous()):
        output = torch.zeros(1, rows, rank, device=device, dtype=torch.float32)
        punica.add_shrink(output, value, (a,), 1.0)
        outputs.append(output[0])
    torch.npu.synchronize()
    expected = inputs.cpu().double() @ a[0, 0].cpu().double().T
    torch.testing.assert_close(outputs[0].cpu().double(), expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
