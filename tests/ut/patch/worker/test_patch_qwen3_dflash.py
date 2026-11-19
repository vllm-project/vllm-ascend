# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model

import vllm_ascend.patch.worker.patch_qwen3_dflash  # noqa: F401

try:
    from vllm.model_executor.models.ricursive_dflash2 import RicursiveDFlash2Model, _NoRope
except ModuleNotFoundError as exc:
    if exc.name != "vllm.model_executor.models.ricursive_dflash2":
        raise
    # The paired vLLM release can precede upstream Ripple model support.
    RicursiveDFlash2Model = None
    _NoRope = None


@pytest.mark.parametrize(
    "model_cls,no_rope_layers",
    [
        pytest.param(DFlashQwen3Model, [], id="qwen3"),
        pytest.param(RicursiveDFlash2Model, [], id="ricursive-all-rope"),
        pytest.param(RicursiveDFlash2Model, [1], id="ricursive-middle-nope"),
        pytest.param(RicursiveDFlash2Model, [0], id="ricursive-first-nope"),
        pytest.param(RicursiveDFlash2Model, [0, 1, 2], id="ricursive-all-nope"),
    ],
)
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("slot_mode", ["shared", "per-layer", "none"])
def test_context_cache_respects_model_rope_hook(model_cls, no_rope_layers, inplace, slot_mode):
    """The worker patch must retain Ricursive's real NoPE restoration hook."""
    if model_cls is None:
        pytest.skip("Ricursive DFlash2 is unavailable in this upstream vLLM revision.")
    num_layers, num_tokens, num_heads, head_dim = 3, 3, 2, 4
    norm_eps = 1e-6
    generator = torch.Generator().manual_seed(7)
    keys = torch.randn(num_layers, num_tokens, num_heads, head_dim, generator=generator)
    values = torch.randn(keys.shape, generator=generator)
    positions = torch.tensor([2, 4, 7])

    def rotate(pos, query, key):
        assert key is None
        original = query.view(-1, num_heads, head_dim)
        left, right = original.chunk(2, dim=-1)
        angles = pos[:, None, None].float()
        rotated = torch.cat(
            (left * angles.cos() - right * angles.sin(), right * angles.cos() + left * angles.sin()), dim=-1
        ).view_as(query)
        return (query.copy_(rotated) if inplace else rotated), None

    rotary = MagicMock(head_size=head_dim, is_neox_style=True, cos_sin_cache=None, side_effect=rotate)
    model = model_cls.__new__(model_cls)
    torch.nn.Module.__init__(model)
    # Ordinary Qwen3 configs do not have the Ricursive-specific field.
    model.config = (
        SimpleNamespace(no_rope_layers=no_rope_layers) if model_cls is RicursiveDFlash2Model else SimpleNamespace()
    )
    model._num_attn_layers = num_layers
    model._num_kv_heads = num_heads
    model._head_dim = head_dim
    model._kv_size = num_heads * head_dim
    model._project_context_kv = MagicMock(side_effect=lambda *args: (keys.clone(), values.clone()))
    model.layers = [
        SimpleNamespace(
            self_attn=SimpleNamespace(
                k_norm=torch.nn.RMSNorm(head_dim, eps=norm_eps, elementwise_affine=False),
                rotary_emb=_NoRope(rotary) if layer in no_rope_layers else rotary,
            )
        )
        for layer in range(num_layers)
    ]
    model._attn_layers = [SimpleNamespace(impl=MagicMock(), kv_cache=object()) for _ in range(num_layers)]
    mapping = torch.arange(num_tokens, dtype=torch.int32)
    slots = {"shared": mapping, "per-layer": [mapping, None, mapping.flip(0)], "none": None}[slot_mode]

    model.precompute_and_store_context_kv(torch.zeros(num_tokens, num_heads * head_dim), positions, slots)

    expected = keys * torch.rsqrt(keys.square().mean(dim=-1, keepdim=True) + norm_eps)
    angles = positions[None, :, None, None].float()
    left, right = expected.chunk(2, dim=-1)
    rotated = expected * angles.cos() + torch.cat((-right, left), dim=-1) * angles.sin()
    for layer, attn in enumerate(model._attn_layers):
        slot_mapping = slots[layer] if isinstance(slots, list) else slots
        if slot_mapping is None:
            attn.impl.do_kv_cache_update.assert_not_called()
            continue
        attn.impl.do_kv_cache_update.assert_called_once()
        cached_attn, cached_k, cached_v, kv_cache, cached_slots = attn.impl.do_kv_cache_update.call_args.args
        assert cached_attn is attn
        assert kv_cache is attn.kv_cache
        assert cached_slots is slot_mapping
        torch.testing.assert_close(cached_k, expected[layer] if layer in no_rope_layers else rotated[layer])
        torch.testing.assert_close(cached_v, values[layer])

    if len(no_rope_layers) == num_layers:
        rotary.assert_not_called()
    else:
        rotary.assert_called_once()
