# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration with installed vLLM; CPU UT mocks do not establish NPU support."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from vllm.lora.model_manager import LoRAModelManager
from vllm.lora.utils import get_supported_lora_modules
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

from vllm_ascend.attention.indexer_kpool import (
    AscendIndexerKPoolMetadata,
    AscendIndexerKPoolRowShard,
    AscendIndexerKPoolTailMetadata,
    Glm5NextKPoolIndexerBackend,
)
from vllm_ascend.lora.utils import (
    AscendColumnParallelLinearWithLoRA,
    AscendMergedColumnParallelLinearWithLoRA,
)
from vllm_ascend.models.glm5next.model import Glm5NextForCausalLM, Glm5NextForConditionalGeneration, Glm5NextModel


@pytest.mark.parametrize(
    "option,error",
    [
        ("fully_sharded_loras", "fully sharded"),
        ("use_sequence_parallel_moe", "sequence parallelism"),
        ("prefill_context_parallel_size", "PCP=DCP=1"),
        ("decode_context_parallel_size", "PCP=DCP=1"),
    ],
)
def test_unsupported_token_layouts_fail_before_model_allocation(option, error):
    lora = SimpleNamespace(fully_sharded_loras=False)
    parallel = SimpleNamespace(
        use_sequence_parallel_moe=False, prefill_context_parallel_size=1, decode_context_parallel_size=1
    )
    target = lora if option == "fully_sharded_loras" else parallel
    setattr(target, option, True if option in {"fully_sharded_loras", "use_sequence_parallel_moe"} else 2)
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace()), lora_config=lora, parallel_config=parallel
    )
    with (
        patch("vllm_ascend.models.glm5next.model._validate_supported_config"),
        pytest.raises(NotImplementedError, match=error),
    ):
        Glm5NextModel(vllm_config=config)


def test_glm53_exposes_packed_targets_and_both_vocabulary_modules():
    for cls in (Glm5NextForCausalLM, Glm5NextForConditionalGeneration):
        assert cls.supports_lora
        model = nn.Module()
        model.embedding_modules = cls.embedding_modules
        embedding = object.__new__(VocabParallelEmbedding)
        nn.Module.__init__(embedding)
        model.add_module("embed_tokens", embedding)
        supported = get_supported_lora_modules(model)
        assert {"embed_tokens", "lm_head"} <= set(supported)
        assert cls.packed_modules_mapping["gate_up_proj"] == ["gate_proj", "up_proj"]
        assert cls.packed_modules_mapping["in_proj_qkvbfg_a"] == [
            "q_proj",
            "k_proj",
            "v_proj",
            "b_proj",
            "f_a_proj",
            "g_a_proj",
        ]
        assert cls.packed_modules_mapping["wk_weights_proj"] == ["wk", "weights_proj"]


def test_multimodal_language_prefix_routes_the_lm_head_to_the_same_punica():
    model = object.__new__(Glm5NextForConditionalGeneration)
    mapping = model.get_mm_mapping()
    assert mapping.language_model == ["language_model"]
    punica = object()
    manager = SimpleNamespace(
        supports_mm=True, supported_modules_to_save=set(), punica_wrapper_mapping={mapping.language_model[0]: punica}
    )
    for name in ["language_model.model.embed_tokens", "language_model.lm_head"]:
        assert LoRAModelManager._get_punica_wrapper(manager, name) is punica


class TupleIdentity(nn.Module):
    def forward(self, value):
        return value, None


class TupleLinear(nn.Linear):
    def forward(self, inputs):
        return super().forward(inputs), None


def make_lora_projection(a, b):
    projection = object.__new__(AscendColumnParallelLinearWithLoRA)
    nn.Module.__init__(projection)
    projection.lora_a_stacked = (a,)
    projection.lora_b_stacked = (b,)
    projection.punica_wrapper = SimpleNamespace()
    return projection


@pytest.mark.parametrize("zero", [False, True])
def test_indexer_key_and_fp32_head_preserve_base_precision(zero):
    torch.manual_seed(531)
    dim, heads, width = 4, 3, 5
    weight = torch.randn(dim + heads, width)
    head_weight = weight[dim:].clone() + 0.00013
    a, b = torch.randn(2, width), torch.randn(dim + heads, 2)
    if zero:
        b.zero_()
    projection = make_lora_projection(a[None, None], b[None, None])
    projection.base_layer = nn.Linear(width, dim + heads, bias=False)
    projection.base_layer.weight.data.copy_(weight)
    projection.output_slices = (dim + heads,)

    def add_lora(output, inputs, aa, bb, scale, slices):
        indices = projection.punica_wrapper.token_lora_indices
        output.add_(torch.where((indices == 0)[:, None], (inputs @ aa[0][0, 0].T) @ bb[0][0, 0].T, 0))
        # The real Punica contract is in-place and returns None.

    projection.punica_wrapper.add_lora_linear = add_lora
    backend = object.__new__(Glm5NextKPoolIndexerBackend)
    nn.Module.__init__(backend)
    backend.wk_weights_proj = projection
    backend.head_dim, backend.n_head = dim, heads
    backend._head_weight_f32 = head_weight
    backend.k_norm = nn.LayerNorm(dim)
    inputs = torch.randn(3, width)
    for slot in (0, -1, 0):
        projection.punica_wrapper.token_lora_indices = torch.full((3,), slot)
        key, gate = backend._project_lora(inputs)
        delta = F.linear(inputs, b @ a) if slot == 0 else torch.zeros(3, dim + heads)
        base_key = F.linear(inputs, weight[:dim])
        torch.testing.assert_close(key, backend.k_norm(base_key + delta[:, :dim]) - backend.k_norm(base_key))
        torch.testing.assert_close(gate, F.linear(inputs, head_weight) + delta[:, dim:])
        assert gate.dtype == torch.float32
        unused, local_gate = backend._project_lora(inputs, project_keys=False, head_rows=(1, 3))
        assert unused is None
        torch.testing.assert_close(local_gate, gate[1:3])


@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn, torch.int8])
@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("compute_topk", [False, True])
@pytest.mark.parametrize("target_slot,request_slot", [(None, 0), (0, None), (0, 1), (0, 0)])
@pytest.mark.parametrize("target_slices", [(0,), (1,), (0, 1)])
def test_indexer_selects_only_live_key_adapters_and_preserves_cache_and_tp(
    cache_dtype, sharded, compute_topk, target_slot, request_slot, target_slices
):
    # All tokens belong to one prompt; only its requested adapter changes.
    count, heads, dim = 4, 3, 2
    hidden, query = torch.randn(count, heads * dim), torch.randn(count, heads * dim)
    keys, head_weights = torch.randn(count, dim), torch.randn(count, heads)
    cache = torch.empty(1, 4, 1, dim, dtype=cache_dtype)
    scale = (
        None
        if cache_dtype == torch.bfloat16
        else torch.ones(1, 4, 1, 1, dtype=torch.float32 if cache_dtype == torch.float8_e4m3fn else torch.float16)
    )
    tail = torch.zeros(1, 4, dim * 2)
    metadata = AscendIndexerKPoolMetadata(
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        slot_mapping=torch.tensor([-1, -1, -1, 0]),
        seq_lens=torch.tensor([1]),
        positions=torch.arange(count),
        block_size=4,
        compress_ratio=4,
        cum_query_lens=torch.tensor([count]),
        raw_seq_lens=torch.tensor([count]),
        num_actual_tokens=count,
    )
    start, stop = (1, count) if sharded else (0, count)
    if sharded:
        metadata.indexer_row_shard = AscendIndexerKPoolRowShard(
            start,
            stop,
            (1, count - 1),
            metadata.positions[start:stop],
            torch.tensor([count - 1]),
            metadata.seq_lens,
            torch.zeros(1, dim),
            metadata.block_table,
        )
    tail_metadata = AscendIndexerKPoolTailMetadata(
        slot_mapping=torch.arange(count),
        block_table=torch.zeros(1, 1),
        block_size=4,
    )
    backend = object.__new__(Glm5NextKPoolIndexerBackend)
    nn.Module.__init__(backend)
    projection = object.__new__(AscendMergedColumnParallelLinearWithLoRA)
    nn.Module.__init__(projection)
    projection.lora_slots_by_slice = [set(), set()]
    if target_slot is not None:
        for target_slice in target_slices:
            projection.lora_slots_by_slice[target_slice].add(target_slot)
    projection.punica_wrapper = SimpleNamespace(
        active_lora_slots=frozenset() if request_slot is None else frozenset({request_slot})
    )
    backend.wk_weights_proj = projection
    backend.wq_b = TupleIdentity()
    backend.head_dim, backend.n_head, backend.softmax_scale = dim, heads, 0.5
    backend._head_weight_f32 = torch.ones(heads, hidden.shape[1])
    backend.k_cache = SimpleNamespace(kv_cache=(cache,) if scale is None else (cache, scale))
    backend.tail_cache = SimpleNamespace(kv_cache=(tail,), prefix="tail")
    backend._project_lora = Mock(
        side_effect=lambda hidden, project_keys, head_rows: (keys if project_keys else None, head_weights[start:stop])
    )
    backend._forward_cann = Mock(return_value="original" if compute_topk else None)
    with patch(
        "vllm_ascend.attention.indexer_kpool.get_forward_context",
        return_value=SimpleNamespace(attn_metadata={"tail": tail_metadata}),
    ):
        result = backend(hidden, query, hidden, metadata, compute_topk)
    active_key = target_slot is not None and target_slot == request_slot
    assert result == ("original" if compute_topk else None)
    assert backend._forward_cann.call_args.kwargs["key_delta"] is (keys if active_key and 0 in target_slices else None)
    assert backend._forward_cann.call_args.kwargs["indexer_scale_cache"] is scale
    if active_key and (0 in target_slices or (1 in target_slices and compute_topk)):
        backend._project_lora.assert_called_once()
    else:
        backend._project_lora.assert_not_called()
    if compute_topk:
        args = backend._forward_cann.call_args.args
        torch.testing.assert_close(args[1], query[start:stop].reshape(-1, heads, dim))
        expected_head = (
            head_weights[start:stop]
            if active_key and 1 in target_slices
            else F.linear(hidden[start:stop], backend._head_weight_f32)
        )
        torch.testing.assert_close(args[2], expected_head * (0.5 * heads**-0.5))


def test_sharded_query_projection_uses_row_offset_without_changing_punica():
    torch.manual_seed(5317)
    inputs = torch.randn(5, 6)
    a, b = torch.randn(1, 1, 2, 6), torch.randn(1, 1, 8, 2)
    projection = make_lora_projection(a, b)
    projection.base_layer = TupleLinear(6, 8, bias=False)
    indices = torch.zeros(5, dtype=torch.long)
    seen = []

    def shrink(x, weights, output, row_indices, scale):
        seen.append(row_indices)
        output.copy_(x @ weights[0, 0].T)

    def expand(x, weights, output, row_indices, offset, width, add_inputs):
        seen.append(row_indices)
        output.add_(x @ weights[0, 0].T)

    projection.punica_wrapper = SimpleNamespace(
        token_lora_indices=indices, bgmv_shrink=shrink, bgmv_expand_slice=expand
    )
    backend = object.__new__(Glm5NextKPoolIndexerBackend)
    nn.Module.__init__(backend)
    backend.wq_b = projection
    actual = backend._project_queries(inputs, 2, 5)
    expected = F.linear(inputs[2:5], projection.base_layer.weight) + (inputs[2:5] @ a[0, 0].T) @ b[0, 0].T
    torch.testing.assert_close(actual, expected)
    assert projection.punica_wrapper.token_lora_indices is indices
    assert all(value.storage_offset() == indices.storage_offset() + 2 and value.numel() == 3 for value in seen)
