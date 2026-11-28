# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared projection ownership and absorbed LoRA for MLA and SFA backends."""

import torch
from torch import nn
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import get_dcp_group
from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA

MLA_PROJECTION_NAMES = (
    "fused_qkv_a_proj",
    "q_b_proj",
    "q_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "o_proj",
    "g_proj",
)


# TODO: Once the vLLM pin includes MLAKVBProjWithLoRA in
# vllm/lora/layers/custom_layer.py, delegate to its
# apply_mla_absorbed_kv_b_lora method and remove this helper.
# The current c410543d pin does not expose that API.
def add_mla_lora(
    inputs: torch.Tensor,
    output: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    indices: torch.Tensor,
    qk_nope_head_dim: int,
    *,
    is_query: bool,
) -> None:
    """Add absorbed KV-B LoRA; inputs are head-major, output token-major.

    Query absorption computes (q @ B_k) @ A. Value expansion computes
    (v @ A.T) @ B_v.T. B retains the TP-local [head, key + value, rank]
    layout; A is shared by the heads. Adapter scaling is already folded in B.
    """
    heads = inputs.shape[0]
    indices = indices[: inputs.shape[1]]
    # Keep factors per adapter: gathering A/B for every token makes workspace
    # grow as tokens * heads * head_dim * rank during a long prefill.
    for slot in range(lora_a.shape[0]):
        a = lora_a[slot, 0]
        b = lora_b[slot, 0].view(heads, -1, lora_b.shape[-1])
        b = b[:, :qk_nope_head_dim] if is_query else b[:, qk_nope_head_dim:]
        if is_query:
            delta = torch.bmm(inputs, b) @ a
        else:
            delta = torch.bmm(inputs @ a.T, b.transpose(1, 2))
        output.add_(torch.where((indices == slot)[:, None, None], delta.transpose(0, 1), 0))


class MLAProjectionMixin:
    """Backends read the same module container that the LoRA manager updates.

    Attention implementations are not nn.Modules. The model-facing wrapper
    registers this container when LoRA is enabled, exposing canonical projection
    suffixes to vLLM's existing alias replacement. No forward-time rebinding is
    needed. Without a wrapper-provided container, the backend owns local refs.
    """

    lora_enabled: bool = False
    _query_projection_name: str = "q_proj"

    def _init_projections(self, layers: dict, *, lora_enabled: bool) -> None:
        self.lora_enabled = lora_enabled
        self._query_projection_name = "q_proj" if self.q_lora_rank is None else "q_b_proj"
        self.projections = layers.get("projections")
        if self.projections is None:
            self.projections = {name: layers.get(name) for name in MLA_PROJECTION_NAMES}

    def _set_projection(self, name: str, layer: nn.Module | None) -> None:
        # Keep the backend's writable projection interface. Assignments and
        # LoRA-manager replacements must update the same source of truth.
        if "projections" not in self.__dict__:
            self.projections = {}
        self.projections[name] = layer

    @property
    def q_proj(self) -> nn.Module:
        return self.projections[self._query_projection_name]

    @q_proj.setter
    def q_proj(self, layer: nn.Module) -> None:
        self._set_projection(self._query_projection_name, layer)

    @property
    def fused_qkv_a_proj(self) -> nn.Module | None:
        return self.projections["fused_qkv_a_proj"]

    @fused_qkv_a_proj.setter
    def fused_qkv_a_proj(self, layer: nn.Module | None) -> None:
        self._set_projection("fused_qkv_a_proj", layer)

    @property
    def kv_a_proj_with_mqa(self) -> nn.Module | None:
        return self.projections["kv_a_proj_with_mqa"]

    @kv_a_proj_with_mqa.setter
    def kv_a_proj_with_mqa(self, layer: nn.Module | None) -> None:
        self._set_projection("kv_a_proj_with_mqa", layer)

    @property
    def kv_b_proj(self) -> nn.Module:
        return self.projections["kv_b_proj"]

    @kv_b_proj.setter
    def kv_b_proj(self, layer: nn.Module) -> None:
        self._set_projection("kv_b_proj", layer)

    @property
    def o_proj(self) -> nn.Module:
        return self.projections["o_proj"]

    @o_proj.setter
    def o_proj(self, layer: nn.Module) -> None:
        self._set_projection("o_proj", layer)

    @property
    def g_proj(self) -> nn.Module | None:
        return self.projections["g_proj"]

    @g_proj.setter
    def g_proj(self, layer: nn.Module | None) -> None:
        self._set_projection("g_proj", layer)

    def _add_mla_lora(self, inputs: torch.Tensor, output: torch.Tensor, *, is_query: bool) -> None:
        # On the vLLM upgrade, select the upstream MLA KV-B LoRA wrapper during
        # layer replacement, then delegate absorbed increments to its API.
        # Retain the shared projection container so Ascend MLA/SFA backends
        # continue to see module replacements made by the LoRA manager.
        if not self.lora_enabled:
            return
        projection = self.kv_b_proj
        if isinstance(projection, BaseLinearLayerWithLoRA):
            a, b = projection.lora_a_stacked[0], projection.lora_b_stacked[0]
            if projection.lora_config.fully_sharded_loras and projection.tp_size > 1:
                a = tensor_model_parallel_all_gather(a, dim=2)
            if is_query and getattr(self, "dcp_q_replicate_enabled", False):
                b = get_dcp_group().all_gather(b, dim=2)
            add_mla_lora(
                inputs,
                output,
                a,
                b,
                projection.punica_wrapper.token_lora_indices,
                self.qk_nope_head_dim,
                is_query=is_query,
            )
