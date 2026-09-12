# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the non-interleaved GQA ``in_proj_qkvz`` weight-split path.

``AscendGatedDeltaNetAttention.forward`` (vllm_ascend/ops/gdn.py) projects the
hidden states through the fused ``in_proj_qkvz`` and then splits the result into
``mixed_qkv`` and ``z``. The split is now done on the *weight* matrix up front
(two independent ``torch.nn.functional.linear`` calls) instead of on the fused
*output* tensor. These tests exercise that branch end-to-end and lock in
numerical equivalence with the previous fused-projection-then-split approach.

They mirror the lightweight ``_GDNForwardWrapper`` style from
``test_gdn_layerwise_kv.py``: the real production ``forward`` is bound onto a
minimal module whose downstream ops (``qwen_gdn_attention_core``, ``norm``,
``out_proj``) are stubbed, so the projection logic runs on CPU without an NPU.
"""

from unittest.mock import patch

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.attention import is_deferred_attention_layer
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention

from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention
from vllm_ascend.patch.worker.patch_qwen3_5 import _GDN_PATCH_TARGET


class _QkvzLinear(nn.Module):
    """Stand-in for the fused ``in_proj_qkvz`` projection.

    Mirrors a bias-less column-parallel linear: ``forward`` returns
    ``(output, None)`` (the tuple the production code unpacks as ``_, _``) and
    ``weight`` is stored as ``[out_features, in_features]`` so that slicing it
    reproduces the per-rank shard that ``functional.linear`` consumes.
    """

    def __init__(self, in_features: int, out_features: int, *, seed: int = 0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.weight = nn.Parameter(torch.randn(out_features, in_features, generator=generator))
        self.quant_method = UnquantizedLinearMethod()

    def forward(self, hidden_states: torch.Tensor):
        return torch.nn.functional.linear(hidden_states, self.weight), None


class _QuantizedQkvzLinear(nn.Module):
    """Quantized stand-in whose packed weight cannot be used by F.linear."""

    def __init__(self, in_features: int, out_features: int, *, seed: int = 0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.weight = nn.Parameter(
            torch.randint(-8, 8, (out_features, in_features), dtype=torch.int8, generator=generator),
            requires_grad=False,
        )
        self.register_buffer("weight_scale", torch.rand(out_features, 1, generator=generator) + 0.5)
        self.quant_method = object()
        self.forward_calls = 0

    def forward(self, hidden_states: torch.Tensor):
        self.forward_calls += 1
        dequantized_weight = self.weight.to(hidden_states.dtype) * self.weight_scale.to(hidden_states.dtype)
        return torch.nn.functional.linear(hidden_states, dequantized_weight), None


class _Linear(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = output_size

    def forward(self, hidden_states: torch.Tensor):
        return hidden_states[:, : self.output_size], None


class _CapturingNorm(nn.Module):
    """``norm`` that sums the core output with ``z`` and records ``z``.

    Summing keeps the final output a function of both projections, so a change
    to either ``mixed_qkv`` or ``z`` surfaces in the asserted output. The
    reshaped ``z`` received by ``norm`` is captured for direct comparison with
    the reference projection.
    """

    def __init__(self):
        super().__init__()
        self.last_z = None

    def forward(self, hidden_states: torch.Tensor, z: torch.Tensor):
        self.last_z = z.detach().clone()
        return hidden_states + z


class _OutputProjection(nn.Module):
    def forward(self, hidden_states: torch.Tensor):
        return hidden_states, None


class _GDNQkvzForwardWrapper(nn.Module):
    """Minimal module bound to the real production ``forward``.

    Configured for the non-interleaved GQA layout (``gqa_interleaved_layout =
    False``) with the fused ``in_proj_qkvz`` projection -- exactly the branch
    touched by the weight-split refactor.
    """

    forward = AscendGatedDeltaNetAttention.forward
    _split_ba_for_tp = AscendGatedDeltaNetAttention._split_ba_for_tp
    process_weights_after_loading = AscendGatedDeltaNetAttention.process_weights_after_loading

    def __init__(
        self,
        *,
        num_k_heads: int,
        head_k_dim: int,
        num_v_heads: int,
        head_v_dim: int,
        hidden_dim: int,
        tp_size: int = 1,
        seed: int = 0,
    ):
        super().__init__()
        self.gqa_interleaved_layout = False
        self.quant_config: object = None
        self.tp_size = tp_size
        self.num_k_heads = num_k_heads
        self.head_k_dim = head_k_dim
        self.key_dim = num_k_heads * head_k_dim
        self.num_v_heads = num_v_heads
        self.head_v_dim = head_v_dim
        self.value_dim = num_v_heads * head_v_dim
        self.activation = None
        self.prefix = "layers.0.linear_attn"

        qkv_size = (self.key_dim * 2 + self.value_dim) // tp_size
        z_size = self.value_dim // tp_size
        self.in_proj_qkvz = _QkvzLinear(hidden_dim, qkv_size + z_size, seed=seed)
        self.in_proj_ba = _Linear(2 * z_size)
        self.norm = _CapturingNorm()
        self.out_proj = _OutputProjection()

    def split_ba(self, ba: torch.Tensor):
        # Avoid torch.chunk: torch-npu's aten.split fallback conflicts with
        # PyTorch's decomposition check on some CANN builds (mirrors the
        # existing _GDNForwardWrapper rationale).
        midpoint = ba.shape[-1] // 2
        return ba[..., :midpoint], ba[..., midpoint:]


def _reference_projection(wrapper: _GDNQkvzForwardWrapper, hidden_states: torch.Tensor):
    """The previous implementation: fused projection then split the output."""
    mixed_qkvz, _ = wrapper.in_proj_qkvz(hidden_states)
    qkv_size = (wrapper.key_dim * 2 + wrapper.value_dim) // wrapper.tp_size
    z_size = wrapper.value_dim // wrapper.tp_size
    mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
    return mixed_qkv, z


def _run_forward(wrapper: _GDNQkvzForwardWrapper, hidden_states: torch.Tensor):
    """Run the real production forward with the core op stubbed out.

    The stub fills ``core_attn_out`` from the first ``z_size`` columns of the
    projected ``mixed_qkv`` (always available since ``qkv_size > z_size``), so
    the final output is a closed-form function of the two projections. It also
    captures the raw ``mixed_qkv`` for direct comparison with the reference.
    """
    captured = {}
    z_size = wrapper.value_dim // wrapper.tp_size
    num_v_heads_per_rank = wrapper.num_v_heads // wrapper.tp_size

    def core_op(mixed_qkv, b, a, core_attn_out, layer_name, flag):
        del b, a, layer_name, flag
        captured["mixed_qkv"] = mixed_qkv.detach().clone()
        view = mixed_qkv[:, :z_size].reshape(-1, num_v_heads_per_rank, wrapper.head_v_dim)
        core_attn_out.copy_(view)

    with patch.object(
        torch.ops.vllm,
        "qwen_gdn_attention_core",
        side_effect=core_op,
        create=True,
    ):
        output = wrapper(hidden_states)

    return output, captured


# (num_k_heads, head_k_dim, num_v_heads, head_v_dim, hidden_dim, tp_size)
_CONFIGS = [
    pytest.param(2, 4, 2, 4, 8, 1, id="small_tp1"),
    pytest.param(4, 8, 2, 16, 16, 1, id="asymmetric_v_tp1"),
    pytest.param(4, 8, 4, 8, 16, 2, id="tp2_even_heads"),
    pytest.param(8, 16, 8, 16, 32, 2, id="tp2_large_heads"),
]


@pytest.mark.parametrize(
    ("num_k_heads", "head_k_dim", "num_v_heads", "head_v_dim", "hidden_dim", "tp_size"),
    _CONFIGS,
)
def test_qkvz_weight_split_matches_fused_projection(
    num_k_heads,
    head_k_dim,
    num_v_heads,
    head_v_dim,
    hidden_dim,
    tp_size,
):
    wrapper = _GDNQkvzForwardWrapper(
        num_k_heads=num_k_heads,
        head_k_dim=head_k_dim,
        num_v_heads=num_v_heads,
        head_v_dim=head_v_dim,
        hidden_dim=hidden_dim,
        tp_size=tp_size,
    )
    num_tokens = 5
    generator = torch.Generator().manual_seed(123)
    hidden_states = torch.randn(num_tokens, hidden_dim, generator=generator)

    ref_mixed_qkv, ref_z = _reference_projection(wrapper, hidden_states)
    output, captured = _run_forward(wrapper, hidden_states)

    # The production weight-split projection must reproduce the fused output.
    torch.testing.assert_close(captured["mixed_qkv"], ref_mixed_qkv)
    torch.testing.assert_close(
        wrapper.norm.last_z,
        ref_z.reshape(-1, num_v_heads // tp_size, head_v_dim).reshape(-1, head_v_dim),
    )

    # End-to-end output is mixed_qkv[:, :z_size] + z under the stubbed core.
    z_size = wrapper.value_dim // wrapper.tp_size
    expected = ref_mixed_qkv[:, :z_size] + ref_z
    torch.testing.assert_close(output, expected)


def test_qkvz_weight_split_uses_independent_linears_not_module_forward(monkeypatch):
    """Guards the refactor's intent: the module ``forward`` must not run.

    The new path reads ``in_proj_qkvz.weight`` directly instead of calling
    ``in_proj_qkvz(hidden_states)``. Asserting the stub is never invoked proves
    the production code took the weight-slice branch (and lets us catch a
    regression that re-fuses the projection).
    """
    wrapper = _GDNQkvzForwardWrapper(
        num_k_heads=2,
        head_k_dim=4,
        num_v_heads=2,
        head_v_dim=4,
        hidden_dim=8,
        tp_size=1,
    )
    hidden_states = torch.randn(3, 8)

    calls = {"count": 0}
    original_forward = wrapper.in_proj_qkvz.forward

    def spy(hidden_states):
        calls["count"] += 1
        return original_forward(hidden_states)

    monkeypatch.setattr(wrapper.in_proj_qkvz, "forward", spy)

    _run_forward(wrapper, hidden_states)

    assert calls["count"] == 0, (
        "expected in_proj_qkvz.forward to be bypassed by the weight-split path, but it was invoked"
    )


def test_quantized_qkvz_projection_uses_module_forward():
    """Packed quantized weights must go through their quantization method."""
    wrapper = _GDNQkvzForwardWrapper(
        num_k_heads=2,
        head_k_dim=4,
        num_v_heads=2,
        head_v_dim=4,
        hidden_dim=8,
        tp_size=1,
    )
    qkv_size = wrapper.key_dim * 2 + wrapper.value_dim
    z_size = wrapper.value_dim
    wrapper.in_proj_qkvz = _QuantizedQkvzLinear(8, qkv_size + z_size)
    hidden_states = torch.randn(3, 8)

    projected_qkvz, _ = wrapper.in_proj_qkvz(hidden_states)
    ref_mixed_qkv, ref_z = projected_qkvz.split([qkv_size, z_size], dim=-1)
    wrapper.in_proj_qkvz.forward_calls = 0

    output, captured = _run_forward(wrapper, hidden_states)

    assert wrapper.in_proj_qkvz.forward_calls == 1
    torch.testing.assert_close(captured["mixed_qkv"], ref_mixed_qkv)
    torch.testing.assert_close(wrapper.norm.last_z, ref_z.reshape(-1, wrapper.head_v_dim))
    torch.testing.assert_close(output, ref_mixed_qkv[:, :z_size] + ref_z)


def test_quantized_model_with_float_nd_qkvz_uses_weight_split(monkeypatch):
    """A model quant config does not prevent splitting a standard ND weight."""
    wrapper = _GDNQkvzForwardWrapper(
        num_k_heads=2,
        head_k_dim=4,
        num_v_heads=2,
        head_v_dim=4,
        hidden_dim=8,
        tp_size=1,
    )
    wrapper.quant_config = object()
    hidden_states = torch.randn(3, 8)
    ref_mixed_qkv, ref_z = _reference_projection(wrapper, hidden_states)
    calls = {"count": 0}
    original_forward = wrapper.in_proj_qkvz.forward

    def spy(states):
        calls["count"] += 1
        return original_forward(states)

    monkeypatch.setattr(wrapper.in_proj_qkvz, "forward", spy)
    output, captured = _run_forward(wrapper, hidden_states)

    assert calls["count"] == 0
    torch.testing.assert_close(captured["mixed_qkv"], ref_mixed_qkv)
    torch.testing.assert_close(wrapper.norm.last_z, ref_z.reshape(-1, wrapper.head_v_dim))
    torch.testing.assert_close(output, ref_mixed_qkv[:, : wrapper.value_dim] + ref_z)


def test_non_nd_unquantized_qkvz_uses_module_forward(monkeypatch):
    """Packed or unknown layouts must not be passed directly to F.linear."""
    wrapper = _GDNQkvzForwardWrapper(
        num_k_heads=2,
        head_k_dim=4,
        num_v_heads=2,
        head_v_dim=4,
        hidden_dim=8,
        tp_size=1,
    )
    wrapper.quant_config = object()
    hidden_states = torch.randn(3, 8)
    calls = {"count": 0}
    original_forward = wrapper.in_proj_qkvz.forward

    def spy(states):
        calls["count"] += 1
        return original_forward(states)

    monkeypatch.setattr(wrapper.in_proj_qkvz, "forward", spy)
    monkeypatch.setattr("vllm_ascend.ops.gdn._is_standard_nd_weight", lambda _: False)

    _run_forward(wrapper, hidden_states)

    assert calls["count"] == 1


def test_post_load_hook_enables_standard_nd_float_qkvz():
    """A finalized unquantized ND projection can use direct weight slices."""
    wrapper = _GDNQkvzForwardWrapper(
        num_k_heads=2,
        head_k_dim=4,
        num_v_heads=2,
        head_v_dim=4,
        hidden_dim=8,
    )

    wrapper.process_weights_after_loading(torch.float16)

    assert wrapper._use_weight_split is True
    assert type(wrapper._use_weight_split) is bool


def test_post_load_hook_disables_packed_or_unknown_layout(monkeypatch):
    """Packed/NZ/unknown storage must stay on the projection module path."""
    wrapper = _GDNQkvzForwardWrapper(
        num_k_heads=2,
        head_k_dim=4,
        num_v_heads=2,
        head_v_dim=4,
        hidden_dim=8,
    )
    monkeypatch.setattr("vllm_ascend.ops.gdn._is_standard_nd_weight", lambda _: False)

    wrapper.process_weights_after_loading(torch.bfloat16)

    assert wrapper._use_weight_split is False
    assert type(wrapper._use_weight_split) is bool


def test_post_load_hook_disables_quantized_projection():
    """Quantized packed weights must use their module's quantization method."""
    wrapper = _GDNQkvzForwardWrapper(
        num_k_heads=2,
        head_k_dim=4,
        num_v_heads=2,
        head_v_dim=4,
        hidden_dim=8,
    )
    qkvz_size = wrapper.key_dim * 2 + wrapper.value_dim
    wrapper.in_proj_qkvz = _QuantizedQkvzLinear(8, qkvz_size)

    wrapper.process_weights_after_loading(torch.bfloat16)

    assert wrapper._use_weight_split is False
    assert type(wrapper._use_weight_split) is bool


def test_post_load_hook_handles_missing_qkvz_projection():
    """LoRA's split qkv/z projection layout must not raise in the hook."""
    wrapper = _GDNQkvzForwardWrapper(
        num_k_heads=2,
        head_k_dim=4,
        num_v_heads=2,
        head_v_dim=4,
        hidden_dim=8,
    )
    del wrapper.in_proj_qkvz

    wrapper.process_weights_after_loading(torch.float32)

    assert wrapper._use_weight_split is False
    assert type(wrapper._use_weight_split) is bool


def test_qwen_gdn_is_deferred_attention_layer_and_hook_is_bound():
    """Qwen's upstream GDN class receives the deferred post-load hook."""
    qwen_gdn_layer = object.__new__(QwenGatedDeltaNetAttention)

    assert _GDN_PATCH_TARGET is QwenGatedDeltaNetAttention
    assert is_deferred_attention_layer(qwen_gdn_layer) is True
    # The uninitialized stand-in has no fused projection, exercising the
    # missing-``in_proj_qkvz`` guard through the same bound hook used by the
    # loader.
    qwen_gdn_layer.process_weights_after_loading(torch.float16)
    assert qwen_gdn_layer._use_weight_split is False
