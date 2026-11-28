# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""The MTP draft loader must map EPLB redundant expert slots like the target."""

from types import SimpleNamespace

import torch
from torch import nn
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping as _real_mapping,
)

from vllm_ascend.models.glm5next import mtp as mtp_module

_MTP_LAYER = 45
_NUM_LOGICAL = 4


def _draft(num_redundant_experts: int) -> nn.Module:
    model = mtp_module.Glm5NextMTP.__new__(mtp_module.Glm5NextMTP)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_hidden_layers=_MTP_LAYER,
        num_nextn_predict_layers=1,
        n_routed_experts=_NUM_LOGICAL,
    )
    model.is_rot_weight_used = False
    # Set by DeepseekV2MixtureOfExperts.extract_moe_parameters() from the draft
    # MoE's own allocation, so it is exactly the value the loader must forward.
    model.num_redundant_experts = num_redundant_experts
    model.model = nn.Module()
    model.model.mtp_start_layer_idx = _MTP_LAYER
    model.model.num_mtp_layers = 1
    layer = nn.Module()
    layer.enorm = nn.LayerNorm(4, bias=False)
    model.model.layers = nn.ModuleDict({str(_MTP_LAYER): layer})
    return model


def _load_and_capture_mapping(monkeypatch, num_redundant_experts: int):
    calls = []

    def spy(*args, **kwargs):
        mapping = _real_mapping(*args, **kwargs)
        calls.append((kwargs, mapping))
        return mapping

    monkeypatch.setattr(mtp_module, "fused_moe_make_expert_params_mapping", spy)
    model = _draft(num_redundant_experts)
    model.load_weights([(f"model.layers.{_MTP_LAYER}.enorm.weight", torch.ones(4))])
    assert len(calls) == 1, "MTP loader must build exactly one expert mapping"
    return calls[0]


def test_mtp_loader_forwards_draft_redundant_expert_count(monkeypatch):
    kwargs, mapping = _load_and_capture_mapping(monkeypatch, 2)
    assert kwargs["num_experts"] == _NUM_LOGICAL
    assert kwargs["num_redundant_experts"] == 2
    # Redundant physical slots have to be in the mapping, otherwise they are
    # never populated from the checkpoint and the draft runs on garbage.
    assert {expert_id for _, _, expert_id, _ in mapping} == set(range(_NUM_LOGICAL + 2))


def test_mtp_loader_maps_logical_experts_only_without_eplb(monkeypatch):
    kwargs, mapping = _load_and_capture_mapping(monkeypatch, 0)
    assert kwargs["num_redundant_experts"] == 0
    assert {expert_id for _, _, expert_id, _ in mapping} == set(range(_NUM_LOGICAL))
