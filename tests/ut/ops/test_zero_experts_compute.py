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
# This file is a part of the vllm-ascend project.
#
"""Unit tests for ``zero_experts_compute`` and the ``GPUInputBatch`` import fix.

PR: [FusedMoE] Support zero-compute expert type and fix GPUInputBatch import
"""

import importlib

import pytest
import torch

from vllm_ascend.ops.fused_moe.routed_experts import zero_experts_compute

# ---------------------------------------------------------------------------
# zero_experts_compute — "zero" type
# ---------------------------------------------------------------------------


def test_zero_type_returns_none():
    """The "zero" type must return ``None`` as the third tuple element."""
    topk_ids = torch.tensor([[0, 64], [1, 65]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.5, 0.5], [0.3, 0.7]], dtype=torch.float32)
    hidden = torch.randn(2, 8, dtype=torch.float32)

    _, _, result = zero_experts_compute(
        expert_indices=topk_ids.clone(),
        expert_scales=topk_weights.clone(),
        num_experts=64,
        zero_expert_type="zero",
        hidden_states=hidden,
    )

    assert result is None


def test_zero_type_remaps_zero_expert_slots_to_expert_0():
    """Zero-expert slots (>= num_experts) must be remapped to expert 0 with weight 0."""
    topk_ids = torch.tensor([[0, 64], [3, 65], [0, 1]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.5, 0.5], [0.3, 0.7], [0.4, 0.6]], dtype=torch.float32)
    hidden = torch.randn(3, 8, dtype=torch.float32)

    ids, weights, _ = zero_experts_compute(
        expert_indices=topk_ids.clone(),
        expert_scales=topk_weights.clone(),
        num_experts=64,
        zero_expert_type="zero",
        hidden_states=hidden,
    )

    expected_ids = torch.tensor([[0, 0], [3, 0], [0, 1]], dtype=torch.int32)
    expected_weights = torch.tensor([[0.5, 0.0], [0.3, 0.0], [0.4, 0.6]], dtype=torch.float32)

    assert torch.equal(ids, expected_ids)
    assert torch.equal(weights, expected_weights)


def test_zero_type_preserves_real_expert_slots():
    """Real expert slots (< num_experts) must keep their original indices and weights."""
    topk_ids = torch.tensor([[10, 20], [0, 63]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.6, 0.4], [0.5, 0.5]], dtype=torch.float32)
    hidden = torch.randn(2, 8, dtype=torch.float32)

    ids, weights, _ = zero_experts_compute(
        expert_indices=topk_ids.clone(),
        expert_scales=topk_weights.clone(),
        num_experts=64,
        zero_expert_type="zero",
        hidden_states=hidden,
    )

    # All slots are real experts — no remapping should occur
    assert torch.equal(ids, topk_ids)
    assert torch.equal(weights, topk_weights)


# ---------------------------------------------------------------------------
# zero_experts_compute — "identity" type
# ---------------------------------------------------------------------------


def test_identity_type_returns_weighted_hidden_sum():
    """The "identity" type must return hidden_states * sum(zero_expert_weights)."""
    topk_ids = torch.tensor([[0, 64], [1, 65]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.5, 0.5], [0.3, 0.7]], dtype=torch.float32)
    hidden = torch.tensor([[1.0] * 4, [2.0] * 4], dtype=torch.float32)

    _, _, result = zero_experts_compute(
        expert_indices=topk_ids.clone(),
        expert_scales=topk_weights.clone(),
        num_experts=64,
        zero_expert_type="identity",
        hidden_states=hidden,
    )

    # Token 0: hidden * (0.5) = [0.5, 0.5, 0.5, 0.5]
    # Token 1: hidden * (0.7) = [1.4, 1.4, 1.4, 1.4]
    expected = torch.tensor([[0.5, 0.5, 0.5, 0.5], [1.4, 1.4, 1.4, 1.4]], dtype=torch.float32)
    assert torch.allclose(result, expected)


def test_identity_type_remaps_zero_expert_slots():
    """Identity type must also remap zero-expert slots to expert 0 / weight 0."""
    topk_ids = torch.tensor([[0, 64], [3, 65]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.5, 0.5], [0.3, 0.7]], dtype=torch.float32)
    hidden = torch.randn(2, 8, dtype=torch.float32)

    ids, weights, _ = zero_experts_compute(
        expert_indices=topk_ids.clone(),
        expert_scales=topk_weights.clone(),
        num_experts=64,
        zero_expert_type="identity",
        hidden_states=hidden,
    )

    expected_ids = torch.tensor([[0, 0], [3, 0]], dtype=torch.int32)
    expected_weights = torch.tensor([[0.5, 0.0], [0.3, 0.0]], dtype=torch.float32)
    assert torch.equal(ids, expected_ids)
    assert torch.equal(weights, expected_weights)


def test_identity_type_zero_real_expert_scales_in_result():
    """The identity result must exclude real-expert contributions (their scales are zeroed)."""
    topk_ids = torch.tensor([[5, 64]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.8, 0.2]], dtype=torch.float32)
    hidden = torch.tensor([[3.0, 4.0]], dtype=torch.float32)

    _, _, result = zero_experts_compute(
        expert_indices=topk_ids.clone(),
        expert_scales=topk_weights.clone(),
        num_experts=64,
        zero_expert_type="identity",
        hidden_states=hidden,
    )

    # Only zero-expert weight (0.2) contributes: [3*0.2, 4*0.2] = [0.6, 0.8]
    expected = torch.tensor([[0.6, 0.8]], dtype=torch.float32)
    assert torch.allclose(result, expected)


def test_identity_type_all_real_experts_returns_zero_result():
    """When all selected experts are real, the identity result must be all zeros."""
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    hidden = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)

    _, _, result = zero_experts_compute(
        expert_indices=topk_ids.clone(),
        expert_scales=topk_weights.clone(),
        num_experts=64,
        zero_expert_type="identity",
        hidden_states=hidden,
    )

    # No zero-expert slots → all scales zeroed → result is zero
    assert torch.allclose(result, torch.zeros(1, 3, dtype=torch.float32))


# ---------------------------------------------------------------------------
# zero_experts_compute — error handling
# ---------------------------------------------------------------------------


def test_unsupported_type_raises_value_error():
    topk_ids = torch.tensor([[0, 64]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    hidden = torch.randn(1, 8, dtype=torch.float32)

    with pytest.raises(ValueError, match="Unsupported zero_expert_type"):
        zero_experts_compute(
            expert_indices=topk_ids,
            expert_scales=topk_weights,
            num_experts=64,
            zero_expert_type="unknown",
            hidden_states=hidden,
        )


# ---------------------------------------------------------------------------
# zero_experts_compute — boundary / edge cases
# ---------------------------------------------------------------------------


def test_all_zero_expert_slots():
    """All slots are zero-expert → all remapped to expert 0, all weights to 0."""
    topk_ids = torch.tensor([[64, 65], [66, 67]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.5, 0.5], [0.3, 0.7]], dtype=torch.float32)
    hidden = torch.randn(2, 8, dtype=torch.float32)

    ids, weights, result = zero_experts_compute(
        expert_indices=topk_ids.clone(),
        expert_scales=topk_weights.clone(),
        num_experts=64,
        zero_expert_type="zero",
        hidden_states=hidden,
    )

    assert torch.equal(ids, torch.zeros_like(topk_ids))
    assert torch.equal(weights, torch.zeros_like(topk_weights))
    assert result is None


def test_num_experts_boundary():
    """Index exactly equal to num_experts is a zero-expert slot."""
    topk_ids = torch.tensor([[63, 64]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    hidden = torch.randn(1, 8, dtype=torch.float32)

    ids, weights, _ = zero_experts_compute(
        expert_indices=topk_ids.clone(),
        expert_scales=topk_weights.clone(),
        num_experts=64,
        zero_expert_type="zero",
        hidden_states=hidden,
    )

    # 63 < 64 → real, stays; 64 >= 64 → zero-expert, remapped
    expected_ids = torch.tensor([[63, 0]], dtype=torch.int32)
    expected_weights = torch.tensor([[0.5, 0.0]], dtype=torch.float32)
    assert torch.equal(ids, expected_ids)
    assert torch.equal(weights, expected_weights)


def test_does_not_mutate_inputs():
    """The function must not mutate the input tensors (it works on clones internally)."""
    topk_ids = torch.tensor([[0, 64]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    hidden = torch.randn(1, 8, dtype=torch.float32)
    ids_copy = topk_ids.clone()
    weights_copy = topk_weights.clone()

    zero_experts_compute(
        expert_indices=topk_ids,
        expert_scales=topk_weights,
        num_experts=64,
        zero_expert_type="zero",
        hidden_states=hidden,
    )

    assert torch.equal(topk_ids, ids_copy)
    assert torch.equal(topk_weights, weights_copy)


# ---------------------------------------------------------------------------
# GPUInputBatch import fix
# ---------------------------------------------------------------------------


def test_patch_mamba_utils_imports_gpu_input_batch():
    """``patch_mamba_utils.py`` must import ``InputBatch`` as ``GPUInputBatch``
    from ``vllm.v1.worker.gpu_input_batch`` (the new v0.29 path).

    On v0.28.0 the name is still re-exported from ``lora_model_runner_mixin``,
    but the canonical source is ``gpu_input_batch``.
    """
    mamba_utils_mod = importlib.import_module("vllm_ascend.patch.worker.patch_mamba_utils")
    assert hasattr(mamba_utils_mod, "GPUInputBatch")
    # The name must resolve to vllm's InputBatch class, not a stale alias
    from vllm.v1.worker.gpu_input_batch import InputBatch

    assert mamba_utils_mod.GPUInputBatch is InputBatch
