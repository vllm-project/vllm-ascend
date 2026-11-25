# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import builtins
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.lora.utils import MiniMaxM3QKVIndexerWithLoRA
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


@pytest.mark.parametrize(
    "fully_sharded,packed,expected", [(False, 5, True), (True, 5, False), (False, 1, False), (False, 3, False)]
)
def test_selects_only_supported_m3_qkv(fully_sharded, packed, expected):
    assert (
        MiniMaxM3QKVIndexerWithLoRA.can_replace_layer(
            source_layer=make_layer(),
            lora_config=SimpleNamespace(fully_sharded_loras=fully_sharded),
            packed_modules_list=["q_proj", "k_proj", "v_proj", "index_q_proj", "index_k_proj"][:packed],
            model_config=None,
        )
        is expected
    )
    assert not MiniMaxM3QKVIndexerWithLoRA.can_replace_layer(
        source_layer=torch.nn.Linear(4, 12),
        lora_config=SimpleNamespace(fully_sharded_loras=False),
        packed_modules_list=["q_proj", "k_proj", "v_proj", "index_q_proj", "index_k_proj"],
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
    assert not MiniMaxM3QKVIndexerWithLoRA.can_replace_layer(
        source_layer=source,
        lora_config=SimpleNamespace(fully_sharded_loras=False),
        packed_modules_list=["q_proj", "k_proj", "v_proj", "index_q_proj", "index_k_proj"],
        model_config=None,
    )


@pytest.mark.parametrize("shape", [(3, 12), (1, 3, 12)])
def test_five_adapter_slices_and_clear_previous_adapter(shape):
    wrapper = MiniMaxM3QKVIndexerWithLoRA(make_layer())
    config = SimpleNamespace(max_lora_rank=2, lora_dtype=torch.float32, fully_sharded_loras=False)
    wrapper.create_lora_weights(1, config)
    assert len(wrapper.lora_a_stacked) == len(wrapper.lora_b_stacked) == 5
    a = [torch.ones(2, 4) for _ in range(5)]
    b = [torch.ones(size, 2) * (i + 1) for i, size in enumerate(wrapper.output_slices)]
    wrapper.set_lora(0, a, b)

    def add(y, x, aa, bb, scale, slices, *, packed_lora_a, packed_lora_b):
        assert y.is_contiguous() and y.shape[-1] == 12
        flat, values = y.reshape(-1, 12), x.reshape(-1, 4)
        flat += scale * (values @ packed_lora_a[0, 0].T) @ packed_lora_b[0, 0]

    wrapper.punica_wrapper = SimpleNamespace(add_lora_linear=add)
    x = torch.ones((*shape[:-1], 4))
    original = torch.arange(torch.tensor(shape).prod()).float().reshape(shape)
    expected = original.clone()
    expected[..., :4] += 8
    expected[..., 4:6] += 16
    expected[..., 6:8] += 24
    expected[..., 8:10] += 32
    expected[..., 10:12] += 40
    result = wrapper._apply_lora_to_output(x, original.clone())
    assert torch.equal(result, expected)
    wrapper.set_lora(0, [a[0], None, None, None, None], [b[0], None, None, None, None])
    result = wrapper._apply_lora_to_output(x, original.clone())
    assert torch.equal(result[..., 4:], original[..., 4:])
    wrapper.reset_lora(0)
    assert torch.equal(wrapper._apply_lora_to_output(x, original.clone()), original)


@pytest.mark.parametrize("tp,rank", [(1, 0), (2, 1), (8, 7)])
def test_indexer_five_slices_follow_kv_replicas_and_replicated_key(tp, rank):
    from vllm_ascend.lora.utils import MiniMaxM3QKVIndexerWithLoRA

    base = make_layer()
    base.tp_size, base.tp_rank = tp, rank
    base.num_kv_head_replicas = max(tp // 2, 1)
    base.num_heads = 8 // tp
    base.num_kv_heads = max(2 // tp, 1)
    base.output_sizes = [16, max(4, tp * 2), max(4, tp * 2), max(4, tp * 2), 2 * tp]
    wrapper = MiniMaxM3QKVIndexerWithLoRA(base)
    config = SimpleNamespace(max_lora_rank=2, lora_dtype=torch.float32, fully_sharded_loras=False)
    wrapper.create_lora_weights(2, config)
    aa = [torch.ones(2, 4) for _ in range(5)]
    bb = [torch.arange(size * 2).reshape(size, 2).float() for size in (16, 4, 4, 4, 2)]
    wrapper.set_lora(1, aa, bb)
    for index in range(5):
        width = wrapper.output_slices[index]
        shard = rank if index == 0 else rank // base.num_kv_head_replicas if index < 4 else 0
        assert torch.equal(wrapper.lora_b_stacked[index][1, 0, :width], bb[index][shard * width : (shard + 1) * width])
        assert torch.count_nonzero(wrapper.lora_b_stacked[index][0]) == 0
    wrapper.reset_lora(1)
    assert torch.count_nonzero(wrapper.lora_a_packed) == 0
    assert torch.count_nonzero(wrapper.lora_b_packed) == 0


def test_dense_five_target_mapping_rejects_indexer_weights():
    from vllm_ascend.lora.utils import MiniMaxM3DenseQKVWithLoRA

    base = make_layer()
    base.output_sizes = base.output_sizes[:3]
    base.weight = torch.nn.Parameter(torch.zeros(8, 4))
    wrapper = MiniMaxM3DenseQKVWithLoRA(base)
    config = SimpleNamespace(max_lora_rank=2, lora_dtype=torch.float32, fully_sharded_loras=False)
    wrapper.create_lora_weights(2, config)
    aa = [torch.ones(2, 4), None, None, None, None]
    bb = [torch.ones(4, 2), None, None, None, None]
    wrapper.set_lora(0, aa, bb)
    assert torch.count_nonzero(wrapper.lora_b_stacked[0][0]) > 0
    with pytest.raises(ValueError, match="no indexer projections"):
        wrapper.set_lora(1, [*aa[:3], aa[0], None], bb)


def test_five_slice_wrappers_do_not_claim_unrelated_packed_mappings():
    from vllm_ascend.lora.utils import MiniMaxM3DenseQKVWithLoRA, MiniMaxM3QKVIndexerWithLoRA
    from vllm_ascend.ops.linear import AscendQKVParallelLinear

    dense = AscendQKVParallelLinear.__new__(AscendQKVParallelLinear)
    torch.nn.Module.__init__(dense)
    config = SimpleNamespace(fully_sharded_loras=False)
    for wrapper, source in ((MiniMaxM3DenseQKVWithLoRA, dense), (MiniMaxM3QKVIndexerWithLoRA, make_layer())):
        assert wrapper.can_replace_layer(
            source_layer=source,
            lora_config=config,
            packed_modules_list=["q_proj", "k_proj", "v_proj", "index_q_proj", "index_k_proj"],
        )
        assert not wrapper.can_replace_layer(
            source_layer=source, lora_config=config, packed_modules_list=["q_proj", "k_proj", "v_proj", "other", "tail"]
        )
