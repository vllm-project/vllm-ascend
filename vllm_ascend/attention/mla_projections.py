# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep existing MLA projection calls bound to the model's live modules."""

from torch import nn

MLA_PROJECTION_NAMES = (
    "fused_qkv_a_proj",
    "q_b_proj",
    "q_proj",
    "kv_a_proj_with_mqa",
    "o_proj",
    "g_proj",
)


class MLAProjectionMixin:
    """Read the aliases updated by vLLM's LoRA manager.

    Attention implementations are not nn.Modules, so references cached directly
    on them are invisible to module replacement. The model-facing wrapper owns
    the shared ModuleDict. Existing unfused projection calls read that container
    without forward-time rebinding. KV-B absorption remains backend-owned.
    """

    _query_projection_name: str = "q_proj"

    def _init_projections(self, layers: dict) -> None:
        self._query_projection_name = "q_proj" if self.q_lora_rank is None else "q_b_proj"
        self.projections = layers.get("projections")
        if self.projections is None:
            self.projections = {name: layers.get(name) for name in MLA_PROJECTION_NAMES}

    def _set_projection(self, name: str, layer: nn.Module | None) -> None:
        # Preserve writable projection attributes used during weight processing.
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
