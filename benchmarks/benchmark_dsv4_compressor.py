# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Operator-only DeepSeek V4 compressor comparison; run in the prepared NPU checkout."""

import json
import time
from types import SimpleNamespace

import torch
from torch import nn
from vllm.config import LoRAConfig
from vllm.lora.layers import LoRAMapping
from vllm.lora.utils import from_layer

import vllm_ascend.ops  # noqa: F401 - Initialize registered operators before the Punica helpers.
from vllm_ascend.lora.deepseek_v4 import AscendCompressorLinearWithLoRA
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU
from vllm_ascend.models.deepseek_v4.compressor import AscendCompressorLinear, AscendCompressorMetadata, Compressor
from vllm_ascend.utils import enable_custom_op


def projection(weight):
    layer = AscendCompressorLinear.__new__(AscendCompressorLinear)
    nn.Module.__init__(layer)
    layer.input_size, layer.output_size = weight.shape[1], weight.shape[0]
    layer.tp_size, layer.tp_rank = 1, 0
    layer.skip_bias_add, layer.return_bias = False, False
    layer.bias = None
    layer.weight = nn.Parameter(weight, requires_grad=False)
    return layer


def latency(function, warmup=10, iterations=30):
    for _ in range(warmup):
        function()
    torch.npu.synchronize()
    elapsed = []
    for _ in range(iterations):
        start = time.perf_counter()
        function()
        torch.npu.synchronize()
        elapsed.append((time.perf_counter() - start) * 1000)
    return {"mean_ms": sum(elapsed) / len(elapsed), "median_ms": sorted(elapsed)[len(elapsed) // 2]}


@torch.inference_mode()
def benchmark(ratio, head_dim):
    generator = torch.Generator().manual_seed(64880)
    requests, length, hidden, rank = 3, 128, 4096, 8
    tokens, width = requests * length, (2 if ratio == 4 else 1) * head_dim
    x = torch.randn(tokens, hidden, generator=generator).to(device="npu", dtype=torch.bfloat16)
    wkv = (torch.randn(width, hidden, generator=generator) * 0.02).to(device="npu", dtype=torch.bfloat16)
    wgate = (torch.randn(width, hidden, generator=generator) * 0.02).to(device="npu", dtype=torch.bfloat16)
    ape = (torch.randn(ratio, width, generator=generator) * 0.1).npu()
    norm = (1 + torch.randn(head_dim, generator=generator) * 0.1).npu()
    valid_outputs = tokens // ratio
    output_capacity = min(tokens, valid_outputs + requests)
    angles = torch.randn(output_capacity, 32, generator=generator).repeat_interleave(2, -1)
    sin, cos = angles.sin().npu(), angles.cos().npu()
    block_size = 8 if ratio == 4 else 32
    table = torch.arange(1, 1 + requests * length // block_size, dtype=torch.int32, device="npu").reshape(requests, -1)
    cumulative = torch.arange(requests + 1, dtype=torch.int32, device="npu") * length
    starts = torch.zeros(requests, dtype=torch.int32, device="npu")
    state_shape = (1 + requests * length // block_size, block_size, 2 * width)
    fused_state = torch.zeros(state_shape, device="npu")
    projected_state = torch.zeros_like(fused_state)
    slots = torch.arange(output_capacity, dtype=torch.int32, device="npu")
    metadata = AscendCompressorMetadata(
        cache=SimpleNamespace(req_metadata=SimpleNamespace(query_start_loc=cumulative, start_pos=starts)),
        state=SimpleNamespace(req_metadata=SimpleNamespace(block_table=table)),
    )
    compressor = Compressor.__new__(Compressor)
    nn.Module.__init__(compressor)
    compressor.wkv, compressor.wgate = projection(wkv), projection(wgate)
    compressor.compress_ratio, compressor.overlap = ratio, ratio == 4
    compressor.rope_head_dim, compressor.norm_eps = 64, 1e-6
    compressor.ape = nn.Parameter(ape)
    compressor.norm = nn.Module()
    compressor.norm.weight = nn.Parameter(norm)
    compressor._compute_metadata = lambda _: (cos, sin, slots)

    def fused():
        return torch.ops._C_ascend.compressor(
            x,
            wkv,
            wgate,
            fused_state,
            ape,
            norm,
            sin,
            cos,
            state_block_table=table,
            cu_seqlens=cumulative,
            seqused=None,
            start_pos=starts,
            rope_head_dim=64,
            cmp_ratio=ratio,
            coff=2 if ratio == 4 else 1,
            norm_eps=1e-6,
            rotary_mode=2,
            cache_mode=1,
        )

    def model():
        return compressor(x, projected_state.unsqueeze(-2), metadata)[0]

    expected_base = fused().clone()
    actual_no_adapter_module = model().clone()
    # The native fused kernel leaves capacity padding unspecified.
    torch.testing.assert_close(actual_no_adapter_module[:valid_outputs], expected_base[:valid_outputs], rtol=0, atol=0)
    fused_latency = latency(fused)
    model_no_lora_latency = latency(model)

    config = LoRAConfig(max_loras=2, max_lora_rank=rank, lora_dtype=torch.bfloat16)
    punica = PunicaWrapperNPU(tokens, tokens, "npu", lora_config=config)
    compressor.wkv = from_layer(compressor.wkv, config.max_loras, config, [])
    assert isinstance(compressor.wkv, AscendCompressorLinearWithLoRA)
    compressor.wkv.set_mapping(punica)
    adapter_ids = [0] * length + [1] * length + [2] * length
    punica.update_metadata(LoRAMapping(adapter_ids, adapter_ids, is_prefill=True), [1, 2], 3, 128)
    # Initial adapter buffers are zero: check the unfused base is equivalent.
    zero_adapter = model().clone()
    torch.testing.assert_close(zero_adapter[:valid_outputs], expected_base[:valid_outputs], rtol=2**-6, atol=2**-6)
    state_difference = (projected_state - fused_state).abs().max().item()
    zero_adapter_latency = latency(model)
    for slot in range(2):
        a = torch.zeros(rank, hidden, dtype=torch.bfloat16, device="npu")
        a[:, :rank] = torch.eye(rank, dtype=torch.bfloat16, device="npu") * (slot + 1) * 0.5
        b = (torch.randn(width, rank, generator=generator) * 0.125).to(device="npu", dtype=torch.bfloat16)
        compressor.wkv.set_lora(slot, a, b)
    adapted = model().clone()
    effect = (adapted - zero_adapter).abs().max().item()
    assert effect > 0.01, effect
    torch.testing.assert_close(adapted[: length // ratio], zero_adapter[: length // ratio], rtol=0, atol=0)
    lora_latency = latency(model)
    return {
        "ratio": ratio,
        "head_dim": head_dim,
        "hidden": hidden,
        "requests": requests,
        "length": length,
        "dtype": "bfloat16",
        "tp": 1,
        "warmup": 10,
        "iterations": 30,
        "measurement": "wall clock with NPU synchronize after every iteration",
        "valid_outputs": valid_outputs,
        "output_capacity": output_capacity,
        "no_lora_model_valid_rows_bit_exact": True,
        "zero_adapter_base_max_error": (zero_adapter[:valid_outputs] - expected_base[:valid_outputs])
        .abs()
        .max()
        .item(),
        "zero_adapter_state_max_error": state_difference,
        "nonzero_adapter_max_effect": effect,
        "original_fused": fused_latency,
        "model_without_lora": model_no_lora_latency,
        "projected_zero_adapter": zero_adapter_latency,
        "projected_mixed_adapter": lora_latency,
    }


if __name__ == "__main__":
    torch.npu.set_device(0)
    enable_custom_op()
    print(json.dumps({"torch": torch.__version__, "device": torch.npu.get_device_name(0)}), flush=True)
    for ratio, head_dim in ((4, 128), (4, 512), (128, 512)):
        print(json.dumps(benchmark(ratio, head_dim)), flush=True)
