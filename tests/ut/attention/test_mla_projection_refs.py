# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression coverage for projection replacement in existing MLA calls."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn
from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.lora.model_manager import LoRAModelManager

from vllm_ascend.attention.mla_projections import MLA_PROJECTION_NAMES, MLAProjectionMixin
from vllm_ascend.attention.mla_v1 import AscendMLAImpl
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
from vllm_ascend.ops import mla as mla_ops


class TupleLinear(nn.Linear):
    def forward(self, inputs):
        return super().forward(inputs), None


class AdapterProjection(BaseLayerWithLoRA):
    """CPU fixture with two nonzero adapters and an explicit base selection."""

    def __init__(self, base):
        super().__init__()
        self.base_layer = base
        self.register_buffer("a", torch.randn(2, 2, base.in_features) * 0.2)
        self.register_buffer("b", torch.randn(2, base.out_features, 2) * 0.2)
        self.slot = -1

    def forward(self, inputs):
        output = nn.functional.linear(inputs, self.base_layer.weight)
        if self.slot >= 0:
            output = output + (inputs @ self.a[self.slot].T) @ self.b[self.slot].T
        return output, None


def replace_projection(model, projection_name):
    """Run vLLM's alias-aware replacement; only the layer factory is a fixture."""
    manager = SimpleNamespace(
        model=model,
        supported_modules_to_save=set(),
        supported_lora_modules={projection_name},
        lora_config=SimpleNamespace(target_modules=None),
        _get_punica_wrapper=lambda name: object(),
        _is_non_gated_moe=False,
        packed_modules_mapping={},
        lora_slots=2,
        register_module=Mock(),
        _register_packed_modules=Mock(),
    )
    with (
        patch(
            "vllm.lora.model_manager.from_layer", side_effect=lambda layer, *args, **kwargs: AdapterProjection(layer)
        ) as factory,
        patch(
            "vllm.lora.model_manager.is_supported_lora_module",
            side_effect=lambda path, *args: path.endswith(projection_name),
        ),
    ):
        LoRAModelManager._create_lora_modules(manager)
    factory.assert_called_once()
    manager.register_module.assert_called_once_with(projection_name, getattr(model, projection_name))
    return getattr(model, projection_name)


@pytest.mark.parametrize("backend_type", [AscendMLAImpl, AscendSFAImpl])
@pytest.mark.parametrize("name", MLA_PROJECTION_NAMES)
def test_replacement_reaches_backend_without_duplicate_adapter_registration(backend_type, name):
    torch.manual_seed(41)
    original = TupleLinear(5, 4, bias=False)
    model = nn.Module()
    model.config = SimpleNamespace()
    model.add_module(name, original)
    model.mla = nn.Module()
    model.mla.projections = nn.ModuleDict({name: original})
    backend = object.__new__(backend_type)
    backend._query_projection_name = name if name in ("q_proj", "q_b_proj") else "q_proj"
    backend_name = "q_proj" if name == "q_b_proj" else name
    setattr(backend, backend_name, original)
    backend.projections = model.mla.projections

    wrapped = replace_projection(model, name)
    inputs = torch.randn(3, 5)
    for slot in (0, -1, 1, 0):
        wrapped.slot = slot
        weight = original.weight if slot < 0 else original.weight + wrapped.b[slot] @ wrapped.a[slot]
        actual = getattr(backend, backend_name)(inputs)[0]
        torch.testing.assert_close(actual, inputs @ weight.T, atol=1e-6, rtol=1e-5)
    assert model.mla.projections[name] is wrapped
    assert getattr(backend, backend_name) is wrapped


@pytest.mark.parametrize("backend_type", [AscendMLAImpl, AscendSFAImpl])
@pytest.mark.parametrize("query_name", ["q_proj", "q_b_proj"])
@pytest.mark.parametrize("num_tokens", [0, 1, 3])
def test_existing_query_bmm_observes_nonzero_projection_adapter(backend_type, query_name, num_tokens):
    torch.manual_seed(53)
    heads, key_dim, rope_dim, latent = 2, 3, 1, 2
    original = TupleLinear(5, heads * (key_dim + rope_dim), bias=False)
    model = nn.Module()
    model.config = SimpleNamespace()
    model.add_module(query_name, original)
    model.mla = nn.Module()
    model.mla.projections = nn.ModuleDict({query_name: original})
    backend = object.__new__(backend_type)
    backend._query_projection_name = query_name
    backend.q_proj = original
    backend.projections = model.mla.projections
    backend.num_heads = backend.local_num_heads = backend.q_projection_heads = heads
    backend.qk_head_dim = key_dim + rope_dim
    backend.qk_nope_head_dim = key_dim
    backend.qk_rope_head_dim = rope_dim
    backend.W_UK_T = torch.randn(heads, key_dim, latent)
    wrapped = replace_projection(model, query_name)
    inputs = torch.randn(num_tokens, 5)

    for slot in (0, -1, 1, 0):
        wrapped.slot = slot
        actual_key, actual_rope = backend._q_proj_and_k_up_proj(inputs)
        weight = original.weight if slot < 0 else original.weight + wrapped.b[slot] @ wrapped.a[slot]
        projected = (inputs @ weight.T).view(num_tokens, heads, key_dim + rope_dim)
        expected_key = torch.einsum("thd,hdl->thl", projected[:, :, :key_dim], backend.W_UK_T)
        torch.testing.assert_close(actual_key, expected_key, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(actual_rope, projected[:, :, key_dim:], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("q_lora_rank", [None, 2])
@pytest.mark.parametrize("shared", [False, True])
def test_projection_initialization_and_assignments(q_lora_rank, shared):
    query_name = "q_proj" if q_lora_rank is None else "q_b_proj"
    original, replacement = TupleLinear(3, 4), TupleLinear(3, 4)
    layers = {name: None for name in MLA_PROJECTION_NAMES}
    layers[query_name] = original
    if shared:
        layers["projections"] = nn.ModuleDict({query_name: original})
    backend = MLAProjectionMixin()
    backend.q_lora_rank = q_lora_rank
    backend._init_projections(layers)
    assert backend.q_proj is original
    backend.q_proj = replacement
    assert backend.q_proj is replacement
    if shared:
        assert layers["projections"][query_name] is replacement
    assert "kv_b_proj" not in backend.projections


@pytest.mark.parametrize("lora_enabled", [False, True])
def test_model_wrapper_shares_existing_projections_only_when_lora_enabled(lora_enabled):
    layers = {name: TupleLinear(5, 8, bias=False) for name in MLA_PROJECTION_NAMES}
    layers.update(
        kv_b_proj=TupleLinear(2, 6, bias=False),
        indexer=None,
        index_group_builder=None,
        is_sparse=False,
        rotary_emb=None,
        q_a_layernorm=None,
        kv_a_layernorm=None,
    )
    config = SimpleNamespace(
        lora_config=object() if lora_enabled else None,
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(num_hidden_layers=1)),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    with (
        patch.object(mla_ops, "get_current_vllm_config", return_value=config),
        patch.object(mla_ops, "get_tensor_model_parallel_world_size", return_value=1),
        patch.object(mla_ops, "AscendMLAAttention") as factory,
    ):
        factory.return_value.impl._fused_preprocess_type.return_value = None
        wrapper = mla_ops.AscendMultiHeadLatentAttention(
            hidden_size=5,
            num_heads=2,
            scale=1.0,
            qk_nope_head_dim=3,
            qk_rope_head_dim=1,
            v_head_dim=3,
            q_lora_rank=None,
            kv_lora_rank=2,
            mla_modules=SimpleNamespace(**layers),
        )
    assert factory.call_args.kwargs["kv_b_proj"] is layers["kv_b_proj"]
    if lora_enabled:
        assert factory.call_args.kwargs["projections"] is wrapper.projections
        for name in MLA_PROJECTION_NAMES:
            assert wrapper.projections[name] is layers[name]
        assert "kv_b_proj" not in wrapper.projections
    else:
        assert factory.call_args.kwargs["projections"] is None
        assert not hasattr(wrapper, "projections")
