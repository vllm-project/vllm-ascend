# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""NPU grouped wo_a regression, including local TP shards and ACLGraph.

The TP cases exercise the real adapter row slicing on each rank's local
projection. They do not stand in for a distributed model inference test.
"""

import pytest
import torch
from torch import nn
from vllm.config import LoRAConfig
from vllm.lora.layers import LoRAMapping
from vllm.lora.utils import from_layer

from vllm_ascend.attention.dsa_v1 import AscendDSAImpl
from vllm_ascend.device.device_config import is_950
from vllm_ascend.lora.deepseek_v4 import AscendDeepseekV4GroupedWoAWithLoRA
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU
from vllm_ascend.models.deepseek_v4.grouped_wo_a import AscendDeepseekV4GroupedWoA
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import AscendW8A8MXFP8DSDynamicLinearMethod
from vllm_ascend.utils import enable_custom_op

NUM_GROUPS = 4
GROUP_INPUT_SIZE = 128
GROUP_OUTPUT_SIZE = 64
NUM_TOKENS = 16
LORA_RANK = 8
TORCH_BACKEND_MAX_LORA_RANK = 128


def _make_case(tp_size, tp_rank, fully_sharded, num_tokens=NUM_TOKENS, max_lora_rank=LORA_RANK):
    torch.npu.set_device(0)
    enable_custom_op()
    generator = torch.Generator().manual_seed(64880)
    base = AscendDeepseekV4GroupedWoA.__new__(AscendDeepseekV4GroupedWoA)
    nn.Module.__init__(base)
    base.input_size = GROUP_INPUT_SIZE
    base.output_size = NUM_GROUPS * GROUP_OUTPUT_SIZE
    base.output_size_per_partition = base.output_size // tp_size
    base.tp_size, base.tp_rank = tp_size, tp_rank
    base.n_local_groups = NUM_GROUPS // tp_size
    base.support_fp8_attention = False
    full_weight = torch.randn(NUM_GROUPS, GROUP_INPUT_SIZE, GROUP_OUTPUT_SIZE, generator=generator) * 0.1
    start = tp_rank * base.n_local_groups
    base.weight = nn.Parameter(
        full_weight[start : start + base.n_local_groups].to(device="npu", dtype=torch.bfloat16), requires_grad=False
    )
    config = LoRAConfig(
        max_loras=2, max_lora_rank=max_lora_rank, lora_dtype=torch.bfloat16, fully_sharded_loras=fully_sharded
    )
    punica = PunicaWrapperNPU(num_tokens, num_tokens, "npu", lora_config=config)
    layer = from_layer(base, config.max_loras, config, [])
    assert isinstance(layer, AscendDeepseekV4GroupedWoAWithLoRA)
    layer.set_mapping(punica)
    adapter_a, adapter_b = [], []
    for slot in range(config.max_loras):
        a = (torch.randn(LORA_RANK, GROUP_INPUT_SIZE, generator=generator) * 0.1).to(torch.bfloat16)
        b = (torch.randn(NUM_GROUPS * GROUP_OUTPUT_SIZE, LORA_RANK, generator=generator) * 0.1).to(torch.bfloat16)
        adapter_a.append(a)
        adapter_b.append(b)
        layer.set_lora(slot, a.npu(), b.npu())
    x = torch.randn(num_tokens, base.n_local_groups, GROUP_INPUT_SIZE, generator=generator).to(
        device="npu", dtype=torch.bfloat16
    )
    impl = AscendDSAImpl.__new__(AscendDSAImpl)
    impl.wo_a = layer
    return impl, punica, x, adapter_a, adapter_b


def _set_mapping(punica, adapter_ids, *, is_prefill=False):
    punica.update_metadata(LoRAMapping(adapter_ids, adapter_ids, is_prefill=is_prefill), [1, 2], 3, 128)


def _reference(impl, x, adapter_ids, adapter_a, adapter_b):
    base = impl.wo_a.base_layer
    inputs = x.cpu().float()
    output = torch.einsum("tgi,gio->tgo", inputs, base.weight.cpu().float()).to(torch.bfloat16).float()
    for token, adapter_id in enumerate(adapter_ids):
        if adapter_id == 0:
            continue
        for group in range(base.n_local_groups):
            global_group = base.tp_rank * base.n_local_groups + group
            b = adapter_b[adapter_id - 1][global_group * GROUP_OUTPUT_SIZE : (global_group + 1) * GROUP_OUTPUT_SIZE]
            output[token, group] += inputs[token, group] @ adapter_a[adapter_id - 1].float().T @ b.float().T
    return output.to(torch.bfloat16)


@pytest.mark.parametrize("num_tokens", [1, 7])
@torch.inference_mode()
def test_grouped_lora_strided_input_preserves_token_order(num_tokens):
    impl, punica, x, adapter_a, adapter_b = _make_case(1, 0, fully_sharded=False, num_tokens=num_tokens)
    storage = torch.empty(num_tokens, NUM_GROUPS, GROUP_INPUT_SIZE * 2, dtype=x.dtype, device=x.device)
    strided_input = storage[..., ::2]
    strided_input.copy_(x)
    assert not strided_input.is_contiguous()
    adapter_ids = ([1, 0, 2, 1] * num_tokens)[:num_tokens]
    _set_mapping(punica, adapter_ids)
    actual = impl._wo_a_bmm(strided_input)
    expected = _reference(impl, x, adapter_ids, adapter_a, adapter_b)
    assert actual.is_contiguous()
    torch.testing.assert_close(actual.cpu(), expected, atol=0.02, rtol=0.02)


@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 0), (2, 1), (4, 3)])
@pytest.mark.parametrize("fully_sharded", [False, True])
@pytest.mark.parametrize("is_prefill", [False, True])
@torch.inference_mode()
def test_grouped_lora_mixed_adapters_local_tp_shard(tp_size, tp_rank, fully_sharded, is_prefill):
    impl, punica, x, adapter_a, adapter_b = _make_case(tp_size, tp_rank, fully_sharded)
    adapter_ids = [1, 0, 2, 1] * (NUM_TOKENS // 4)
    _set_mapping(punica, adapter_ids, is_prefill=is_prefill)
    actual = impl._wo_a_bmm(x)
    assert actual.is_contiguous()
    expected = _reference(impl, x, adapter_ids, adapter_a, adapter_b)
    torch.testing.assert_close(actual.cpu(), expected, atol=0.02, rtol=0.02)
    base_only = impl.wo_a.base_layer.grouped_forward(x)
    assert (actual[::4] - base_only[::4]).abs().max().item() > 0.01
    torch.testing.assert_close(actual[1::4], base_only[1::4], atol=0, rtol=0)

    impl.wo_a.reset_lora(0)
    adapter_b[0].zero_()
    actual = impl._wo_a_bmm(x)
    expected = _reference(impl, x, adapter_ids, adapter_a, adapter_b)
    torch.testing.assert_close(actual.cpu(), expected, atol=0.02, rtol=0.02)


@pytest.mark.skipif(not is_950(), reason="Requires the Ascend 950 MXFP8 grouped matmul")
@pytest.mark.parametrize("is_prefill", [False, True])
@torch.inference_mode()
def test_grouped_mxfp8_lora_preserves_bf16_adapter_input(is_prefill):
    impl, punica, x, adapter_a, adapter_b = _make_case(1, 0, fully_sharded=False)
    base = impl.wo_a.base_layer
    base.support_fp8_attention = True
    base.prefix = "model.layers.2.self_attn.wo_a"
    # Two nonzero columns in each checkpoint row, with one scale per 128x128 tile.
    weight = torch.zeros(base.output_size, GROUP_INPUT_SIZE)
    weight[:, :2] = 1
    block_size = 128
    scale = 0.25
    base.weight = nn.Parameter(weight.to(torch.float8_e4m3fn).npu(), requires_grad=False)
    scale_shape = (base.output_size // block_size, GROUP_INPUT_SIZE // block_size)
    base.weight_scale = nn.Parameter(torch.full(scale_shape, scale, dtype=torch.float32, device="npu"), False)
    method = AscendW8A8MXFP8DSDynamicLinearMethod.__new__(AscendW8A8MXFP8DSDynamicLinearMethod)
    method.block_size, method.group_size, method.o_lora_rank = block_size, 32, GROUP_OUTPUT_SIZE
    method.process_weights_after_loading(base)

    # Both values round to 1 in E4M3. Their BF16 difference survives only in the adapter.
    x.fill_(2)
    x[..., 0], x[..., 1] = 1.03125, 1.0
    gains = (8, -8)
    for slot, (a, b, gain) in enumerate(zip(adapter_a, adapter_b, gains)):
        a.zero_()
        a[0, 0], a[0, 1] = 1, -1
        b.zero_()
        b[:, 0] = gain
        impl.wo_a.set_lora(slot, a.npu(), b.npu())

    adapter_ids = [1, 0, 2, 1] * (NUM_TOKENS // 4)
    _set_mapping(punica, adapter_ids, is_prefill=is_prefill)
    actual = impl._wo_a_bmm(x).cpu()
    base_only = torch.full((NUM_TOKENS, NUM_GROUPS, GROUP_OUTPUT_SIZE), 2 * scale, dtype=torch.bfloat16)
    delta = (1.03125 - 1.0) * torch.tensor((0, *gains), dtype=torch.bfloat16)[adapter_ids]
    torch.testing.assert_close(actual, base_only + delta[:, None, None], atol=0, rtol=0)


@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 0), (2, 1), (4, 3)])
def test_grouped_lora_aclgraph_replays_changed_adapter_mapping(tp_size, tp_rank):
    _check_aclgraph_replay(tp_size, tp_rank, is_prefill=False)


@pytest.mark.parametrize(
    "max_lora_rank",
    [
        TORCH_BACKEND_MAX_LORA_RANK,
        pytest.param(
            LORA_RANK,
            marks=pytest.mark.skipif(not is_950(), reason="Requires the Ascend 950 small-rank PyTorch backend"),
        ),
    ],
)
def test_grouped_lora_prefill_aclgraph_replays_changed_adapter_mapping(max_lora_rank):
    _check_aclgraph_replay(1, 0, is_prefill=True, max_lora_rank=max_lora_rank)


@torch.inference_mode()
def _check_aclgraph_replay(tp_size, tp_rank, *, is_prefill, max_lora_rank=LORA_RANK):
    impl, punica, x, adapter_a, adapter_b = _make_case(
        tp_size, tp_rank, fully_sharded=False, max_lora_rank=max_lora_rank
    )
    initial_ids = [1, 0, 2, 1] * (NUM_TOKENS // 4)
    _set_mapping(punica, initial_ids, is_prefill=is_prefill)
    for _ in range(3):
        impl._wo_a_bmm(x)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = impl._wo_a_bmm(x)
    for adapter_ids in (
        [2, 1, 0, 2] * (NUM_TOKENS // 4),
        [0] * NUM_TOKENS,
        initial_ids,
    ):
        _set_mapping(punica, adapter_ids, is_prefill=is_prefill)
        graph.replay()
        torch.npu.synchronize()
        expected = _reference(impl, x, adapter_ids, adapter_a, adapter_b)
        torch.testing.assert_close(actual.cpu(), expected, atol=0.02, rtol=0.02)
