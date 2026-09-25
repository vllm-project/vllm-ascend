# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Kimi K3 two-phase Online Softmax AttnRes backend.

The two-phase backend lives in the vLLM Ascend adapter
(``vllm_ascend.models.kimi_k3``) and is selected through ``attn_res_mode``.
These tests verify numerical equivalence against the original single-pass
residual flow, the CANNBot DSL dispatch, and the DSpark aux-capture plumbing.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from vllm_ascend.models import kimi_k3
from vllm_ascend.models.kimi_k3 import (
    AscendKimiK3ForConditionalGeneration,
    AscendKimiLinearForCausalLM,
    AscendKimiLinearModel,
)


def test_two_phase_phase1_phase2_matches_single_softmax():
    """The two-phase Online Softmax helpers are numerically identical to the
    single-pass softmax over ``[completed blocks..., running partial]``."""

    def single_pass(rows, partial_vec, query, eps):
        values = torch.cat((rows, partial_vec.unsqueeze(1)), dim=1).float()
        q = query.float()
        inv_rms = torch.rsqrt(values.square().mean(-1) + eps)
        logits = torch.matmul(values, q) * inv_rms
        weights = logits.softmax(-1)
        return torch.matmul(weights.unsqueeze(1), values).squeeze(1)

    torch.manual_seed(3)
    token_count, num_blocks, hidden_size = 2, 3, 6
    eps = 1e-6
    block_residual = torch.randn(token_count, num_blocks, hidden_size)
    queries = torch.randn(4, hidden_size)

    phase1 = kimi_k3._prepare_attn_res_phase1(block_residual, queries, eps)
    slot = kimi_k3.AttnResPhase2Slot(
        queries[0],
        phase1.inter_numerator[0],
        phase1.inter_max[0],
        phase1.inter_exp_sum[0],
    )

    partial = torch.randn(token_count, hidden_size)
    expected = single_pass(block_residual, partial, queries[0], eps)
    torch.testing.assert_close(
        kimi_k3._merge_attn_res_partial(partial, slot, eps),
        expected,
        atol=1e-4,
        rtol=1e-4,
    )

    delta = torch.randn(token_count, hidden_size)
    updated = partial + delta
    torch.testing.assert_close(
        kimi_k3._update_attn_res_phase2(partial.clone(), delta, slot, eps),
        single_pass(block_residual, updated, queries[0], eps),
        atol=1e-4,
        rtol=1e-4,
    )

    torch.testing.assert_close(
        kimi_k3._merge_attn_res_slot(updated.float(), slot, eps),
        single_pass(block_residual, updated, queries[0], eps),
        atol=1e-4,
        rtol=1e-4,
    )


def test_two_phase_matches_original_text_model_flow(monkeypatch: pytest.MonkeyPatch):
    """Running the text model with the two-phase backend reproduces the
    original single-pass residual flow (and both DSpark aux capture modes)."""

    def single_pass(prefix_sum, block_residual, projection, norm, num_valid_blocks):
        values = torch.cat(
            (block_residual[:, :num_valid_blocks], prefix_sum.unsqueeze(1)),
            dim=1,
        ).float()
        weights = projection.weight.squeeze(0).float() * norm.weight.float()
        inv_rms = torch.rsqrt(values.square().mean(-1) + norm.variance_epsilon)
        logits = torch.matmul(values, weights) * inv_rms
        probabilities = logits.softmax(-1)
        return torch.matmul(probabilities.unsqueeze(1), values).squeeze(1).to(prefix_sum.dtype)

    class FakeLinear:
        def __init__(self, hidden_size):
            self.weight = nn.Parameter(torch.randn(1, hidden_size))

    class FakeLayer(nn.Module):
        def __init__(self, layer_idx, block_size, hidden_size):
            super().__init__()
            self.layer_idx = layer_idx
            self.block_size = block_size
            self.is_block_write_layer = layer_idx % block_size == 0
            self.block_write_idx = layer_idx // block_size
            self.prev_valid_blocks = (layer_idx + block_size - 1) // block_size
            self.self_attention_res_norm = SimpleNamespace(
                weight=nn.Parameter(torch.randn(hidden_size)),
                variance_epsilon=1e-6,
            )
            self.mlp_res_norm = SimpleNamespace(
                weight=nn.Parameter(torch.randn(hidden_size)),
                variance_epsilon=1e-6,
            )
            self.self_attention_res_proj = FakeLinear(hidden_size)
            self.mlp_res_proj = FakeLinear(hidden_size)

        def _attention(self, positions, attention_input):
            del positions
            return attention_input * 0.5

        def _mlp(self, mlp_input):
            return mlp_input + 10.0

        def forward(self, *, positions, hidden_states, residual):
            prefix_sum = hidden_states
            hidden_states = kimi_k3._apply_ascend_attn_res(
                prefix_sum,
                residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                self.prev_valid_blocks,
            )
            if self.is_block_write_layer:
                residual[:, self.block_write_idx, :].copy_(prefix_sum)
                prefix_sum = None
            attention_output = self._attention(positions, hidden_states)
            prefix_sum = attention_output if prefix_sum is None else prefix_sum + attention_output
            mlp_valid_blocks = self.prev_valid_blocks + (1 if self.is_block_write_layer else 0)
            hidden_states = kimi_k3._apply_ascend_attn_res(
                prefix_sum,
                residual,
                self.mlp_res_proj,
                self.mlp_res_norm,
                mlp_valid_blocks,
            )
            hidden_states = self._mlp(hidden_states)
            return prefix_sum + hidden_states, residual

    monkeypatch.setattr(kimi_k3, "_apply_ascend_attn_res", single_pass)
    monkeypatch.setattr(
        kimi_k3,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    hidden_size, num_layers, block_size = 4, 3, 2

    def build_model(mode: str) -> AscendKimiLinearModel:
        torch.manual_seed(11)
        model = AscendKimiLinearModel.__new__(AscendKimiLinearModel)
        nn.Module.__init__(model)
        model.start_layer = 0
        model.end_layer = num_layers
        model.config = SimpleNamespace(
            hidden_size=hidden_size,
            rms_norm_eps=1e-6,
            attn_res_block_size=block_size,
        )
        model.layers = nn.ModuleList([FakeLayer(layer_idx, block_size, hidden_size) for layer_idx in range(num_layers)])
        model.output_attn_res_proj = FakeLinear(hidden_size)
        model.output_attn_res_norm = SimpleNamespace(
            weight=nn.Parameter(torch.randn(hidden_size)),
            variance_epsilon=1e-6,
        )
        model.use_sequence_parallel = False
        model.dspark_aux_capture_materialized = True
        model.attn_res_mode = mode
        model.attn_res_effective_queries = None
        model._set_aux_hidden_state_layers((0, 2))
        return model

    original = build_model("original")
    two_phase = build_model("two_phase")
    embeds = torch.randn(1, hidden_size)
    positions = torch.tensor([0])

    orig_hidden, orig_aux = original(None, positions, None, inputs_embeds=embeds)
    two_hidden, two_aux = two_phase(None, positions, None, inputs_embeds=embeds)

    torch.testing.assert_close(two_hidden, orig_hidden, atol=1e-4, rtol=1e-4)
    assert len(two_aux) == len(orig_aux)
    for captured_two, captured_orig in zip(two_aux, orig_aux):
        torch.testing.assert_close(captured_two, captured_orig, atol=1e-4, rtol=1e-4)

    for model in (original, two_phase):
        model.dspark_aux_capture_materialized = False
        model._set_aux_hidden_state_layers((1, 3))

    orig_hidden, orig_aux = original(None, positions, None, inputs_embeds=embeds)
    two_hidden, two_aux = two_phase(None, positions, None, inputs_embeds=embeds)

    torch.testing.assert_close(two_hidden, orig_hidden, atol=1e-4, rtol=1e-4)
    assert len(two_aux) == len(orig_aux)
    for captured_two, captured_orig in zip(two_aux, orig_aux):
        torch.testing.assert_close(captured_two, captured_orig, atol=1e-4, rtol=1e-4)


def test_kimi_k3_attn_res_fused_availability_gating(monkeypatch: pytest.MonkeyPatch):
    """The fused AttnRes backend requires both the CANNBot DSL operators and
    the opt-in environment variable."""
    monkeypatch.setattr(kimi_k3, "_block_attn_res_prepare_fused", object())
    monkeypatch.setattr(kimi_k3, "_block_attn_res_update_fused", object())
    monkeypatch.setenv("VLLM_ASCEND_KIMI_K3_ATTNRES_FUSED_ENABLED", "1")
    assert kimi_k3._use_fused_attn_res() is True

    monkeypatch.setenv("VLLM_ASCEND_KIMI_K3_ATTNRES_FUSED_ENABLED", "0")
    assert kimi_k3._use_fused_attn_res() is False

    monkeypatch.setattr(kimi_k3, "_block_attn_res_prepare_fused", None)
    monkeypatch.setenv("VLLM_ASCEND_KIMI_K3_ATTNRES_FUSED_ENABLED", "1")
    assert kimi_k3._use_fused_attn_res() is False


class _AttnResFusedDispatchStub(AscendKimiLinearModel):
    """Minimal model shim exercising the fused dispatch methods."""

    attn_res_mode = "fused"

    def __init__(self) -> None:
        pass


def test_fused_phase1_and_phase2_dispatch_to_cannbot_dsl(monkeypatch: pytest.MonkeyPatch):
    """The fused Phase-1/Phase-2 dispatch forwards the reference operator
    interfaces (valid_blocks scalar, fp32 partial, bf16 delta) and consumes
    the returned ``(h, partial_blocks)`` buffers correctly."""
    token_count, num_blocks, hidden_size = 2, 3, 6
    eps = 1e-6
    torch.manual_seed(7)
    v = torch.randn(token_count, num_blocks, hidden_size).contiguous()
    queries = torch.randn(4, hidden_size).contiguous()
    prepare_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]] = []
    update_calls: list[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]
    ] = []

    def fused_prepare(v_arg, eq_arg, valid_blocks_arg, eps=1e-6):
        prepare_calls.append((v_arg, eq_arg, valid_blocks_arg, eps))
        assert v_arg.dtype == torch.float32 and v_arg.is_contiguous()
        assert eq_arg.dtype == torch.float32 and eq_arg.dim() == 2
        assert valid_blocks_arg.dtype == torch.int64 and valid_blocks_arg.numel() == 1
        assert int(valid_blocks_arg) == v_arg.shape[1]
        phase1 = kimi_k3._prepare_attn_res_phase1(v_arg, eq_arg, eps)
        return phase1.inter_numerator, phase1.inter_max, phase1.inter_exp_sum

    monkeypatch.setattr(kimi_k3, "_block_attn_res_prepare_fused", fused_prepare)

    stub = _AttnResFusedDispatchStub()
    phase1 = stub._run_attn_res_phase1(v, queries, eps)
    assert len(prepare_calls) == 1
    assert int(prepare_calls[0][2]) == num_blocks
    expected = kimi_k3._prepare_attn_res_phase1(v, queries, eps)
    for actual, ref in zip(phase1, expected):
        torch.testing.assert_close(actual, ref, atol=1e-4, rtol=1e-4)

    partial = torch.randn(token_count, hidden_size, dtype=torch.float32).contiguous()
    delta = torch.randn(token_count, hidden_size, dtype=torch.bfloat16).contiguous()
    slot = kimi_k3.AttnResPhase2Slot(
        queries[0],
        phase1.inter_numerator[0],
        phase1.inter_max[0],
        phase1.inter_exp_sum[0],
    )

    def fused_update(
        partial_block,
        partial_delta,
        effective_query,
        inter_max,
        inter_exp_sum,
        inter_numerator,
        epsilon,
    ):
        update_calls.append(
            (partial_block, partial_delta, effective_query, inter_max, inter_exp_sum, inter_numerator, epsilon)
        )
        assert partial_block.dtype == torch.float32 and partial_block.is_contiguous()
        assert partial_delta.dtype in (torch.bfloat16, torch.float16)
        assert partial_delta.shape == partial_block.shape
        assert effective_query.dtype == torch.float32 and effective_query.shape == (hidden_size,)
        assert inter_max.dtype == torch.float32 and inter_max.shape == (token_count,)
        assert inter_exp_sum.dtype == torch.float32 and inter_exp_sum.shape == (token_count,)
        assert inter_numerator.dtype == torch.float32 and inter_numerator.shape == (token_count, hidden_size)
        updated = partial_block.clone().float() + partial_delta.float()
        merged = kimi_k3._update_attn_res_phase2(
            partial_block,
            partial_delta,
            kimi_k3.AttnResPhase2Slot(effective_query, inter_numerator, inter_max, inter_exp_sum),
            epsilon,
        )
        return merged.to(partial_delta.dtype), updated

    monkeypatch.setattr(kimi_k3, "_block_attn_res_update_fused", fused_update)

    partial_before = partial.clone()
    output = stub._run_attn_res_phase2(partial, delta, slot, eps)

    assert len(update_calls) == 1
    assert torch.equal((partial_before.float() + delta.float()).to(torch.float32), partial)
    expected_merged = kimi_k3._update_attn_res_phase2(partial_before.clone(), delta, slot, eps)
    torch.testing.assert_close(output, expected_merged.to(torch.bfloat16), atol=1e-2, rtol=1e-2)


def test_fused_mode_text_model_flow_matches_two_phase(monkeypatch: pytest.MonkeyPatch):
    """With the CANNBot DSL operators installed, the fused backend follows the
    same dynamic block flow as the pure-Torch two-phase backend and produces
    matching results (DSpark aux capture included)."""

    def single_pass(prefix_sum, block_residual, projection, norm, num_valid_blocks):
        values = torch.cat(
            (block_residual[:, :num_valid_blocks], prefix_sum.unsqueeze(1)),
            dim=1,
        ).float()
        weights = projection.weight.squeeze(0).float() * norm.weight.float()
        inv_rms = torch.rsqrt(values.square().mean(-1) + norm.variance_epsilon)
        logits = torch.matmul(values, weights) * inv_rms
        probabilities = logits.softmax(-1)
        return torch.matmul(probabilities.unsqueeze(1), values).squeeze(1).to(prefix_sum.dtype)

    class FakeLinear:
        def __init__(self, hidden_size):
            self.weight = nn.Parameter(torch.randn(1, hidden_size))

    class FakeLayer(nn.Module):
        def __init__(self, layer_idx, block_size, hidden_size):
            super().__init__()
            self.layer_idx = layer_idx
            self.block_size = block_size
            self.is_block_write_layer = layer_idx % block_size == 0
            self.block_write_idx = layer_idx // block_size
            self.prev_valid_blocks = (layer_idx + block_size - 1) // block_size
            self.self_attention_res_norm = SimpleNamespace(
                weight=nn.Parameter(torch.randn(hidden_size)),
                variance_epsilon=1e-6,
            )
            self.mlp_res_norm = SimpleNamespace(
                weight=nn.Parameter(torch.randn(hidden_size)),
                variance_epsilon=1e-6,
            )
            self.self_attention_res_proj = FakeLinear(hidden_size)
            self.mlp_res_proj = FakeLinear(hidden_size)

        def _attention(self, positions, attention_input):
            del positions
            return attention_input * 0.5

        def _mlp(self, mlp_input):
            return mlp_input + 10.0

        def forward(self, *, positions, hidden_states, residual):
            prefix_sum = hidden_states
            hidden_states = kimi_k3._apply_ascend_attn_res(
                prefix_sum,
                residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                self.prev_valid_blocks,
            )
            if self.is_block_write_layer:
                residual[:, self.block_write_idx, :].copy_(prefix_sum)
                prefix_sum = None
            attention_output = self._attention(positions, hidden_states)
            prefix_sum = attention_output if prefix_sum is None else prefix_sum + attention_output
            mlp_valid_blocks = self.prev_valid_blocks + (1 if self.is_block_write_layer else 0)
            hidden_states = kimi_k3._apply_ascend_attn_res(
                prefix_sum,
                residual,
                self.mlp_res_proj,
                self.mlp_res_norm,
                mlp_valid_blocks,
            )
            hidden_states = self._mlp(hidden_states)
            return prefix_sum + hidden_states, residual

    monkeypatch.setattr(kimi_k3, "_apply_ascend_attn_res", single_pass)
    monkeypatch.setattr(
        kimi_k3,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    hidden_size, num_layers, block_size = 4, 3, 2
    prepare_calls: list[int] = []
    update_calls: list[int] = []

    def fused_prepare(v, effective_queries, valid_blocks, eps=1e-6):
        prepare_calls.append(int(valid_blocks))
        assert v.dtype == torch.float32 and int(valid_blocks) == v.shape[1]
        phase1 = kimi_k3._prepare_attn_res_phase1(v, effective_queries, eps)
        return phase1.inter_numerator, phase1.inter_max, phase1.inter_exp_sum

    def fused_update(partial_block, partial_delta, effective_query, inter_max, inter_exp_sum, inter_numerator, epsilon):
        update_calls.append(partial_delta.shape[0])
        assert partial_block.dtype == torch.float32
        merged = kimi_k3._update_attn_res_phase2(
            partial_block,
            partial_delta,
            kimi_k3.AttnResPhase2Slot(effective_query, inter_numerator, inter_max, inter_exp_sum),
            epsilon,
        )
        return merged, partial_block

    monkeypatch.setattr(kimi_k3, "_block_attn_res_prepare_fused", fused_prepare)
    monkeypatch.setattr(kimi_k3, "_block_attn_res_update_fused", fused_update)

    def build_model() -> AscendKimiLinearModel:
        model = AscendKimiLinearModel.__new__(AscendKimiLinearModel)
        nn.Module.__init__(model)
        model.start_layer = 0
        model.end_layer = num_layers
        model.config = SimpleNamespace(
            hidden_size=hidden_size,
            rms_norm_eps=1e-6,
            attn_res_block_size=block_size,
        )
        model.layers = nn.ModuleList([FakeLayer(layer_idx, block_size, hidden_size) for layer_idx in range(num_layers)])
        model.output_attn_res_proj = FakeLinear(hidden_size)
        model.output_attn_res_norm = SimpleNamespace(
            weight=nn.Parameter(torch.randn(hidden_size)),
            variance_epsilon=1e-6,
        )
        model.use_sequence_parallel = False
        model.dspark_aux_capture_materialized = True
        model.attn_res_effective_queries = None
        model._set_aux_hidden_state_layers((0, 2))
        return model

    torch.manual_seed(11)
    two_phase = build_model()
    two_phase.attn_res_mode = "two_phase"
    torch.manual_seed(11)
    fused = build_model()
    fused.attn_res_mode = "fused"

    embeds = torch.randn(1, hidden_size)
    positions = torch.tensor([0])

    two_hidden, two_aux = two_phase(None, positions, None, inputs_embeds=embeds)
    fused_hidden, fused_aux = fused(None, positions, None, inputs_embeds=embeds)

    torch.testing.assert_close(fused_hidden, two_hidden, atol=1e-4, rtol=1e-4)
    assert len(fused_aux) == len(two_aux)
    for captured_fused, captured_two in zip(fused_aux, two_aux):
        torch.testing.assert_close(captured_fused, captured_two, atol=1e-4, rtol=1e-4)
    assert prepare_calls == [1, 2]
    assert len(update_calls) == 4


def test_kimi_k3_dspark_aux_capture_mode_is_forwarded():
    causal_model = AscendKimiLinearForCausalLM.__new__(AscendKimiLinearForCausalLM)
    nn.Module.__init__(causal_model)
    causal_model.model = SimpleNamespace(dspark_aux_capture_materialized=False)

    causal_model.set_dspark_aux_capture_materialized(True)

    assert causal_model.model.dspark_aux_capture_materialized is True

    wrapper = AscendKimiK3ForConditionalGeneration.__new__(AscendKimiK3ForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper.language_model = MagicMock()

    wrapper.set_dspark_aux_capture_materialized(True)

    wrapper.language_model.set_dspark_aux_capture_materialized.assert_called_once_with(True)
