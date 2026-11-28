# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration with installed vLLM; CPU UT mocks do not establish NPU support."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn
from vllm.lora.model_manager import LoRAModelManager

from vllm_ascend.attention.mla_projections import MLA_PROJECTION_NAMES
from vllm_ascend.attention.mla_v1 import AscendMLAImpl
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
from vllm_ascend.lora.utils import (
    AscendColumnParallelLinearWithLoRA,
)


@pytest.mark.parametrize("name", MLA_PROJECTION_NAMES)
def test_mla_projection_alias_uses_the_same_wrapper_without_registering_twice(name):
    # The backend reads the exact container updated by the real LoRA manager.
    projection = nn.Linear(5, 7, bias=False)
    model = nn.Module()
    model.config = SimpleNamespace()
    model.add_module(name, projection)
    model.mla = nn.Module()
    backend = object.__new__(AscendSFAImpl)
    backend.q_lora_rank = None if name == "q_proj" else 2
    model.mla.projections = nn.ModuleDict({name: projection})
    backend._init_projections({"projections": model.mla.projections}, lora_enabled=True)
    wrapper = object.__new__(AscendColumnParallelLinearWithLoRA)
    nn.Module.__init__(wrapper)
    wrapper.base_layer = projection
    punica = object()
    manager = SimpleNamespace(
        model=model,
        supported_modules_to_save=set(),
        supported_lora_modules={name},
        lora_config=SimpleNamespace(target_modules=None),
        _get_punica_wrapper=lambda name: punica,
        _is_non_gated_moe=False,
        packed_modules_mapping={},
        lora_slots=1,
        register_module=Mock(),
        _register_packed_modules=Mock(),
    )
    with (
        patch("vllm.lora.model_manager.from_layer", return_value=wrapper) as factory,
        patch(
            "vllm.lora.model_manager.is_supported_lora_module",
            side_effect=lambda path, *args: path.endswith(name),
        ),
        patch.object(AscendColumnParallelLinearWithLoRA, "set_mapping"),
    ):
        LoRAModelManager._create_lora_modules(manager)
    assert getattr(model, name) is model.mla.projections[name] is wrapper
    assert getattr(backend, "q_proj" if name == "q_b_proj" else name) is wrapper
    factory.assert_called_once()
    manager.register_module.assert_called_once_with(name, wrapper)


class TupleIdentity(nn.Module):
    def forward(self, value):
        return value, None


def make_backend(cls, projections, **attributes):
    backend = object.__new__(cls)
    backend.q_lora_rank = attributes.get("q_lora_rank")
    backend._init_projections(projections, lora_enabled=True)
    for name, value in attributes.items():
        setattr(backend, name, value)
    return backend


def make_lora_projection(a, b):
    projection = object.__new__(AscendColumnParallelLinearWithLoRA)
    nn.Module.__init__(projection)
    projection.lora_a_stacked = (a,)
    projection.lora_b_stacked = (b,)
    projection.lora_config = SimpleNamespace(fully_sharded_loras=False)
    projection.tp_size = 1
    projection.punica_wrapper = SimpleNamespace()
    return projection


@pytest.mark.parametrize("backend_type", [AscendMLAImpl, AscendSFAImpl])
@pytest.mark.parametrize("q_lora_rank", [None, 2])
def test_projection_assignment_and_manager_replacement_share_one_container(backend_type, q_lora_rank):
    query_name = "q_proj" if q_lora_rank is None else "q_b_proj"
    original, replacement = nn.Linear(5, 7), nn.Linear(5, 7)
    projections = nn.ModuleDict({query_name: original, "kv_b_proj": original})
    backend = make_backend(backend_type, {"projections": projections}, q_lora_rank=q_lora_rank)
    assert backend.q_proj is original
    projections[query_name] = replacement
    assert backend.q_proj is replacement
    backend.kv_b_proj = replacement
    assert projections["kv_b_proj"] is replacement


def transpose_bmm(x, w, perm_x1=(0, 1, 2), perm_x2=(0, 1, 2), perm_y=(0, 1, 2)):
    return torch.bmm(x.permute(perm_x1), w.permute(perm_x2)).permute(perm_y)


@pytest.mark.parametrize("backend_type", [AscendMLAImpl, AscendSFAImpl])
@pytest.mark.parametrize("zero", [False, True])
@pytest.mark.parametrize("value_path", ["head", "batch_fused", "batch_fallback"])
def test_absorbed_backend_paths_match_merged_weights(backend_type, zero, value_path):
    torch.manual_seed(534)
    heads, qdim, vdim, latent = 2, 4, 3, 5
    a, b = torch.randn(2, 1, 2, latent), torch.randn(2, 1, heads * (qdim + vdim), 2) * 0.1
    if zero:
        b.zero_()
    projection = make_lora_projection(a, b)
    base = torch.randn(heads, qdim + vdim, latent)
    backend = make_backend(
        backend_type,
        {"q_proj": TupleIdentity(), "kv_b_proj": projection},
        q_lora_rank=None,
        num_heads=heads,
        local_num_heads=heads,
        q_projection_heads=heads,
        qk_head_dim=qdim,
        qk_nope_head_dim=qdim,
        qk_rope_head_dim=0,
        kv_lora_rank=latent,
        v_head_dim=vdim,
        W_UK_T=base[:, :qdim],
        W_UV=base[:, qdim:].transpose(1, 2),
    )
    backend._supports_fused_v_up_proj_size = lambda: value_path == "batch_fused"
    q, x = torch.randn(3, heads * qdim), torch.randn(3, heads, latent)
    module = "vllm_ascend.attention.mla_v1" if backend_type is AscendMLAImpl else "vllm_ascend.attention.sfa_v1"
    with patch(module + ".torch_npu.npu_transpose_batchmatmul", side_effect=transpose_bmm):
        for slot in (0, -1, 1, 0):
            projection.punica_wrapper.token_lora_indices = torch.full((3,), slot)
            actual_q, rope = backend._q_proj_and_k_up_proj(q)
            if backend_type is AscendMLAImpl:
                actual_v = (
                    backend._v_up_proj(x.transpose(0, 1).contiguous())
                    if value_path == "head"
                    else backend._v_up_proj_batch_major(x)
                )
            else:
                actual_v = backend._v_up_proj(x)
            weight = base if slot < 0 else base + b[slot, 0].view(heads, qdim + vdim, 2) @ a[slot, 0]
            expected_q = torch.einsum("bhd,hdc->bhc", q.view(3, heads, qdim), weight[:, :qdim])
            expected_v = torch.einsum("thc,hvc->thv", x, weight[:, qdim:])
            torch.testing.assert_close(actual_q, expected_q, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(actual_v.reshape_as(expected_v), expected_v, atol=2e-6, rtol=2e-5)
            assert rope.shape[-1] == 0


@pytest.mark.parametrize("backend_type", [AscendMLAImpl, AscendSFAImpl])
def test_unwrapped_kv_projection_leaves_absorbed_output_unchanged(backend_type):
    backend = make_backend(backend_type, {"kv_b_proj": nn.Linear(5, 14)}, qk_nope_head_dim=4)
    output = torch.randn(3, 2, 3)
    expected = output.clone()
    backend._add_mla_lora(torch.randn(2, 3, 5), output, is_query=False)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    assert backend._fused_preprocess_type() is None


@pytest.mark.parametrize("is_query", [False, True])
def test_absorbed_fully_sharded_factors_gather_rank_before_projection(is_query):
    a, b = torch.ones(2, 1, 4, 5), torch.ones(2, 1, 14, 4)
    projection = make_lora_projection(a[:, :, :2], b)
    projection.lora_config.fully_sharded_loras = True
    projection.tp_size = 2
    projection.punica_wrapper.token_lora_indices = torch.tensor([0])
    backend = make_backend(AscendMLAImpl, {"kv_b_proj": projection}, qk_nope_head_dim=4)
    inputs = torch.ones(2, 1, 4 if is_query else 5)
    output = torch.zeros(1, 2, 5 if is_query else 3)
    with patch("vllm_ascend.attention.mla_projections.tensor_model_parallel_all_gather", return_value=a) as gather:
        backend._add_mla_lora(inputs, output, is_query=is_query)
    gather.assert_called_once_with(projection.lora_a_stacked[0], dim=2)
    torch.testing.assert_close(output, torch.full_like(output, (4 if is_query else 5) * 4))


def test_replicated_dcp_query_gathers_key_heads_without_gathering_value_heads():
    a, local_b = torch.ones(2, 1, 2, 5), torch.ones(2, 1, 7, 2)
    projection = make_lora_projection(a, local_b)
    projection.punica_wrapper.token_lora_indices = torch.tensor([0])
    backend = make_backend(AscendMLAImpl, {"kv_b_proj": projection}, qk_nope_head_dim=4, dcp_q_replicate_enabled=True)
    group = SimpleNamespace(all_gather=Mock(return_value=local_b.repeat(1, 1, 2, 1)))
    with patch("vllm_ascend.attention.mla_projections.get_dcp_group", return_value=group):
        query = torch.zeros(1, 2, 5)
        backend._add_mla_lora(torch.ones(2, 1, 4), query, is_query=True)
        group.all_gather.assert_called_once_with(local_b, dim=2)
        value = torch.zeros(1, 1, 3)
        backend._add_mla_lora(torch.ones(1, 1, 5), value, is_query=False)
    torch.testing.assert_close(query, torch.full_like(query, 8))
    torch.testing.assert_close(value, torch.full_like(value, 10))
    assert group.all_gather.call_count == 1
