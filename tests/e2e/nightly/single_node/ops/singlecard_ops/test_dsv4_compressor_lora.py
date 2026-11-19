# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real NPU Punica adapters through the main/indexer compressor boundary."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.config import LoRAConfig
from vllm.lora.layers import LoRAMapping
from vllm.lora.utils import from_layer

from tests.e2e.nightly.single_node.ops.singlecard_ops.triton.test_dsv4_compressor import projected_reference
from vllm_ascend.lora.deepseek_v4 import AscendCompressorLinearWithLoRA
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU
from vllm_ascend.models.deepseek_v4.compressor import AscendCompressorLinear, AscendCompressorMetadata, Compressor
from vllm_ascend.utils import enable_custom_op

HIDDEN_SIZE = 1024
LORA_RANK = 8


def _projection(weight):
    layer = AscendCompressorLinear.__new__(AscendCompressorLinear)
    nn.Module.__init__(layer)
    layer.input_size, layer.output_size = weight.shape[1], weight.shape[0]
    layer.tp_size, layer.tp_rank = 1, 0
    layer.skip_bias_add, layer.return_bias = False, False
    layer.bias = None
    layer.weight = nn.Parameter(weight.npu(), requires_grad=False)
    return layer


@pytest.mark.parametrize("ratio,head_dim", [(4, 128), (4, 512), (128, 512)])
@pytest.mark.parametrize("target", ["wkv", "wgate"])
@pytest.mark.parametrize("use_graph", [False, True])
@torch.inference_mode()
def test_compressor_mixed_adapters(ratio, head_dim, target, use_graph):
    torch.npu.set_device(0)
    enable_custom_op()
    generator = torch.Generator().manual_seed(64880)
    length, requests = max(8, ratio), 3
    tokens, width = length * requests, head_dim * (2 if ratio == 4 else 1)
    hidden = torch.randn(tokens, HIDDEN_SIZE, generator=generator).bfloat16()
    weight_kv = (torch.randn(width, HIDDEN_SIZE, generator=generator) * 0.02).bfloat16()
    weight_gate = (torch.randn(width, HIDDEN_SIZE, generator=generator) * 0.02).bfloat16()
    compressor = Compressor.__new__(Compressor)
    nn.Module.__init__(compressor)
    compressor.wkv, compressor.wgate = _projection(weight_kv), _projection(weight_gate)
    compressor.compress_ratio, compressor.norm_eps = ratio, 1e-6
    ape = torch.randn(ratio, width, generator=generator) * 0.1
    norm = torch.ones(head_dim)
    compressor.ape = nn.Parameter(ape.npu())
    compressor.norm = nn.Module()
    compressor.norm.weight = nn.Parameter(norm.npu())
    config = LoRAConfig(max_loras=2, max_lora_rank=LORA_RANK, lora_dtype=torch.bfloat16)
    punica = PunicaWrapperNPU(tokens, tokens, "npu", lora_config=config)
    wrapper = from_layer(getattr(compressor, target), config.max_loras, config, [])
    assert isinstance(wrapper, AscendCompressorLinearWithLoRA)
    setattr(compressor, target, wrapper)
    wrapper.set_mapping(punica)
    adapter_a, adapter_b = [], []
    for slot in range(2):
        # Sparse A makes the intermediate exactly representable while B remains
        # nonzero and different across output features and adapter slots.
        a = torch.zeros(LORA_RANK, HIDDEN_SIZE, dtype=torch.bfloat16)
        a[:, :LORA_RANK] = torch.eye(LORA_RANK, dtype=torch.bfloat16) * (slot + 1) * 0.5
        b = (torch.randn(width, LORA_RANK, generator=generator) * 0.125).bfloat16()
        adapter_a.append(a)
        adapter_b.append(b)
        wrapper.set_lora(slot, a.npu(), b.npu())
    block_size = 8 if ratio == 4 else 32
    pages_per_request = length // block_size
    table = torch.arange(1, 1 + requests * pages_per_request, dtype=torch.int32).reshape(requests, -1)
    state = torch.zeros(1 + requests * pages_per_request, block_size, 2 * width)
    device_state = state.npu()
    cumulative = torch.arange(requests + 1, dtype=torch.int32) * length
    starts = torch.zeros(requests, dtype=torch.int32)
    num_outputs = tokens // ratio
    angles = torch.randn(num_outputs, 32, generator=generator).repeat_interleave(2, -1)
    sin, cos = angles.sin(), angles.cos()
    slots = torch.arange(num_outputs, dtype=torch.int32, device="npu")
    device_sin, device_cos = sin.npu(), cos.npu()
    compressor._compute_metadata = lambda _: (device_cos, device_sin, slots)
    metadata = AscendCompressorMetadata(
        cache=SimpleNamespace(req_metadata=SimpleNamespace(query_start_loc=cumulative.npu(), start_pos=starts.npu())),
        state=SimpleNamespace(req_metadata=SimpleNamespace(block_table=table.npu())),
    )
    device_hidden = hidden.npu()

    def mapping(ids):
        punica.update_metadata(LoRAMapping(ids, ids, is_prefill=not use_graph), [1, 2], 3, 128)

    adapter_ids = [0] * length + [1] * length + [2] * length
    mapping(adapter_ids)
    if use_graph:
        for _ in range(3):
            compressor(device_hidden, device_state.unsqueeze(-2), metadata)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual, _ = compressor(device_hidden, device_state.unsqueeze(-2), metadata)
        adapter_ids = [2] * length + [0] * length + [1] * length
        mapping(adapter_ids)
        device_state.zero_()
        graph.replay()
    else:
        actual, _ = compressor(device_hidden, device_state.unsqueeze(-2), metadata)
    kv = hidden.float() @ weight_kv.float().T
    gate = hidden.float() @ weight_gate.float().T
    base_kv, base_gate = kv.clone(), gate.clone()
    selected = kv if target == "wkv" else gate
    for row, adapter_id in enumerate(adapter_ids):
        if adapter_id:
            delta = hidden[row].float() @ adapter_a[adapter_id - 1].float().T @ adapter_b[adapter_id - 1].float().T
            selected[row] += delta.bfloat16().float()
    expected, expected_state = projected_reference(
        kv, gate, state, ape, norm, sin, cos, table, cumulative, starts, ratio, torch.bfloat16
    )
    baseline, _ = projected_reference(
        base_kv, base_gate, state, ape, norm, sin, cos, table, cumulative, starts, ratio, torch.bfloat16
    )
    torch.testing.assert_close(actual.cpu(), expected, atol=2**-6, rtol=2**-6)
    torch.testing.assert_close(device_state.cpu(), expected_state, atol=2**-16, rtol=2**-10)
    assert (expected - baseline).abs().max().item() > 0.01
    no_adapter_request = adapter_ids.index(0) // length
    output_slice = slice(no_adapter_request * length // ratio, (no_adapter_request + 1) * length // ratio)
    torch.testing.assert_close(actual.cpu()[output_slice], baseline[output_slice], atol=2**-6, rtol=2**-6)


@pytest.mark.parametrize("use_graph", [False, True])
@torch.inference_mode()
def test_projection_scratch_reuse_across_layers_and_batch_sizes(use_graph):
    torch.npu.set_device(0)
    enable_custom_op()
    generator = torch.Generator().manual_seed(64880)
    capacity, width = 16, 128
    config = LoRAConfig(max_loras=2, max_lora_rank=LORA_RANK, lora_dtype=torch.bfloat16)
    punica = PunicaWrapperNPU(capacity, capacity, "npu", lora_config=config)
    weights = [(torch.randn(width, HIDDEN_SIZE, generator=generator) * 0.02).bfloat16() for _ in range(2)]
    layers = [from_layer(_projection(weight), config.max_loras, config, []) for weight in weights]
    a = torch.zeros(LORA_RANK, HIDDEN_SIZE, dtype=torch.bfloat16)
    a[:, :LORA_RANK] = torch.eye(LORA_RANK, dtype=torch.bfloat16) * 0.5
    adapter_b = [(torch.randn(width, LORA_RANK, generator=generator) * 0.125).bfloat16() for _ in layers]
    for layer, b in zip(layers, adapter_b):
        layer.set_mapping(punica)
        layer.set_lora(0, a.npu(), b.npu())
        layer.set_lora(1, a.npu(), (-b).npu())

    cases = []
    # Capture the small batch first; later sizes must not invalidate its scratch.
    for tokens in (4, capacity):
        hidden = torch.randn(tokens, HIDDEN_SIZE, generator=generator).bfloat16()
        device_hidden = hidden.npu()
        ids = [1, 0, 2, 1] * (tokens // 4)
        punica.update_metadata(LoRAMapping(ids, ids, is_prefill=not use_graph), [1, 2], 3, 128)
        graph = None
        if use_graph:
            for _ in range(3):
                for layer in layers:
                    layer(device_hidden)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                outputs = [layer(device_hidden) for layer in layers]
        else:
            outputs = []
        cases.append((hidden, device_hidden, graph, outputs))

    # Alternate graph sizes and mappings, including an all-base replay after
    # nonzero adapters. Both projections share scratch but own their outputs.
    for case_index, pattern in ((0, [1, 0, 2, 1]), (1, [2, 1, 0, 2]), (0, [0, 0, 0, 0]), (1, [1, 0, 2, 1])):
        hidden, device_hidden, graph, outputs = cases[case_index]
        ids = pattern * (hidden.shape[0] // 4)
        punica.update_metadata(LoRAMapping(ids, ids, is_prefill=not use_graph), [1, 2], 3, 128)
        if graph is not None:
            graph.replay()
        else:
            outputs = [layer(device_hidden) for layer in layers]
        torch.npu.synchronize()
        for output, weight, b in zip(outputs, weights, adapter_b):
            expected = hidden.float() @ weight.float().T
            delta = ((hidden.float() @ a.float().T) @ b.float().T).bfloat16().float()
            for token, adapter_id in enumerate(ids):
                if adapter_id:
                    expected[token] += delta[token] * (1 if adapter_id == 1 else -1)
            assert output.dtype == torch.float32
            torch.testing.assert_close(output.cpu(), expected, atol=2**-16, rtol=2**-10)
