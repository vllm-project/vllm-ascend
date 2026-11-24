# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import builtins
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.lora.utils import MiniMaxM3QKVParallelLinearWithLoRA
from vllm_ascend.models.minimax_m3.msa_m3 import AscendMiniMaxM3QKVParallelLinearWithIndexer


def make_layer():
    layer = AscendMiniMaxM3QKVParallelLinearWithIndexer.__new__(AscendMiniMaxM3QKVParallelLinearWithIndexer)
    torch.nn.Module.__init__(layer)
    for key, value in dict(
        input_size=4,
        output_size_per_partition=12,
        tp_size=1,
        tp_rank=0,
        output_sizes=[4, 2, 2, 2, 2],
        num_heads=2,
        num_kv_heads=1,
        head_size=2,
        num_kv_head_replicas=1,
    ).items():
        setattr(layer, key, value)
    layer.weight = torch.nn.Parameter(torch.zeros(12, 4))
    return layer


@pytest.mark.parametrize("fully_sharded,packed,expected", [(False, 3, True), (True, 3, False), (False, 1, False)])
def test_selects_only_supported_m3_qkv(fully_sharded, packed, expected):
    assert (
        MiniMaxM3QKVParallelLinearWithLoRA.can_replace_layer(
            source_layer=make_layer(),
            lora_config=SimpleNamespace(fully_sharded_loras=fully_sharded),
            packed_modules_list=["target"] * packed,
            model_config=None,
        )
        is expected
    )
    assert not MiniMaxM3QKVParallelLinearWithLoRA.can_replace_layer(
        source_layer=torch.nn.Linear(4, 12),
        lora_config=SimpleNamespace(fully_sharded_loras=False),
        packed_modules_list=["q", "k", "v"],
        model_config=None,
    )


def test_non_m3_selection_does_not_import_m3_dependencies(monkeypatch):
    source = torch.nn.Linear(4, 12)
    original = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("vllm_ascend.models.minimax_m3"):
            raise AssertionError("Non-M3 LoRA selection imported M3")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert not MiniMaxM3QKVParallelLinearWithLoRA.can_replace_layer(
        source_layer=source,
        lora_config=SimpleNamespace(fully_sharded_loras=False),
        packed_modules_list=["q", "k", "v"],
        model_config=None,
    )


@pytest.mark.parametrize("shape", [(3, 12), (1, 3, 12)])
def test_three_adapter_slices_preserve_indexer_and_clear_previous_adapter(shape):
    wrapper = MiniMaxM3QKVParallelLinearWithLoRA(make_layer())
    config = SimpleNamespace(max_lora_rank=2, lora_dtype=torch.float32, fully_sharded_loras=False)
    wrapper.create_lora_weights(1, config)
    assert len(wrapper.lora_a_stacked) == len(wrapper.lora_b_stacked) == 3
    assert wrapper.output_sizes == [4, 2, 2]
    a = [torch.ones(2, 4) for _ in range(3)]
    b = [torch.ones(size, 2) * (i + 1) for i, size in enumerate(wrapper.output_slices)]
    wrapper.set_lora(0, a, b)

    def add(y, x, aa, bb, scale, slices, *, packed_lora_a, packed_lora_b):
        assert y.is_contiguous() and y.shape[-1] == 8
        flat, values = y.reshape(-1, 8), x.reshape(-1, 4)
        flat += scale * (values @ packed_lora_a[0, 0].T) @ packed_lora_b[0, 0]

    wrapper.punica_wrapper = SimpleNamespace(add_lora_linear=add)
    x = torch.ones((*shape[:-1], 4))
    original = torch.arange(torch.tensor(shape).prod()).float().reshape(shape)
    expected = original.clone()
    expected[..., :4] += 8
    expected[..., 4:6] += 16
    expected[..., 6:8] += 24
    result = wrapper._apply_lora_to_output(x, original.clone())
    assert torch.equal(result, expected)
    assert torch.equal(result[..., 8:], original[..., 8:])
    wrapper.set_lora(0, [a[0], None, None], [b[0], None, None])
    result = wrapper._apply_lora_to_output(x, original.clone())
    assert torch.equal(result[..., 4:], original[..., 4:])
    wrapper.reset_lora(0)
    assert torch.equal(wrapper._apply_lora_to_output(x, original.clone()), original)
