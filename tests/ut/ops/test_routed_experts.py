# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts


def _routed_experts(weight_views):
    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    routed_experts.local_num_experts = 2
    routed_experts.quant_method = SimpleNamespace(
        get_eplb_weight_views=lambda layer: weight_views,
    )
    return routed_experts


def test_get_expert_weights_flattens_layout_aware_views():
    weights = [torch.randn(2, 3, 4), torch.randn(2, 5)]

    views = list(_routed_experts(weights).get_expert_weights())

    assert [view.shape for view in views] == [torch.Size([2, 12]), torch.Size([2, 5])]
    assert views[0].untyped_storage().data_ptr() == weights[0].untyped_storage().data_ptr()


def test_get_expert_weights_rejects_unsupported_quantization():
    with pytest.raises(NotImplementedError, match="weight views are not defined"):
        list(_routed_experts([]).get_expert_weights())


def test_get_expert_weights_rejects_missing_weight_view_contract():
    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    routed_experts.local_num_experts = 2
    routed_experts.quant_method = SimpleNamespace()

    with pytest.raises(NotImplementedError, match="must implement get_eplb_weight_views"):
        list(routed_experts.get_expert_weights())


def test_get_expert_weights_rejects_non_expert_first_dimension():
    with pytest.raises(ValueError, match="first dimension"):
        list(_routed_experts([torch.randn(3, 4)]).get_expert_weights())


def test_get_expert_weights_rejects_non_contiguous_view():
    with pytest.raises(ValueError, match="flattenable without a copy"):
        list(_routed_experts([torch.randn(2, 3, 4).transpose(1, 2)]).get_expert_weights())
