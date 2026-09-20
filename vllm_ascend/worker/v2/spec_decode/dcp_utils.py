# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replicated GQA draft support for the v2 model runner."""

import copy
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config, set_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
)
from vllm_ascend.attention.utils import enable_dcp

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
    from vllm.v1.worker.utils import AttentionGroup


def uses_dcp_replicated_gqa_draft(config: VllmConfig) -> bool:
    spec_config = config.speculative_config
    if spec_config is None:
        return False
    target_model_config = config.model_config
    target_architectures = {
        *(getattr(target_model_config, "architectures", ()) or ()),
        *(getattr(target_model_config.hf_config, "architectures", ()) or ()),
    }
    target_architecture = getattr(target_model_config, "architecture", None)
    if target_architecture:
        target_architectures.add(target_architecture)
    draft_hf_config = spec_config.draft_model_config.hf_config
    draft_architectures = {
        *(getattr(spec_config.draft_model_config, "architectures", ()) or ()),
        *(getattr(draft_hf_config, "architectures", ()) or ()),
    }
    return (
        (
            getattr(target_model_config.hf_config, "model_type", None) == "kimi_k3"
            or any("KimiK3" in architecture for architecture in target_architectures)
        )
        and getattr(draft_hf_config, "model_type", None) == "qwen3"
        and any(architecture in {"DSparkDraftModel", "Qwen3DSparkModel"} for architecture in draft_architectures)
    )


def draft_additional_config(additional_config: dict | None) -> dict:
    """Isolate the model-only draft from target PD scheduler options."""
    result = copy.deepcopy(additional_config or {})
    if "recompute_scheduler_enable" in result:
        result["recompute_scheduler_enable"] = False
    draft_scheduler_config = result.get("scheduler_config")
    if draft_scheduler_config is None:
        draft_scheduler_config = {}
        result["scheduler_config"] = draft_scheduler_config
    draft_scheduler_config["recompute_scheduler_enable"] = False
    return result


class ReplicatedDraftAttentionBackend(AscendAttentionBackend):
    """Keep replicated draft groups local when target DCP builds metadata."""

    @staticmethod
    def get_impl_cls() -> type[AscendAttentionBackendImpl]:
        return AscendAttentionBackendImpl

    @staticmethod
    def get_builder_cls() -> type[AscendAttentionMetadataBuilder]:
        return AscendAttentionMetadataBuilder


class DCPDraftReplicatedMixin:
    """Add local GQA draft KV replication before the upstream speculator in the MRO.

    Upstream owns per-group KV tables and DCP sharding. This mixin isolates
    Ascend PD settings and selects the local draft attention backend.
    """

    # State supplied by the speculator hosting this cooperative mixin.
    vllm_config: VllmConfig
    attn_vllm_config: VllmConfig
    device: torch.device
    draft_kv_cache_group_ids: list[int]
    attn_groups: list[list["AttentionGroup"]]

    def _prepare_dcp_draft_config(self, vllm_config: VllmConfig) -> VllmConfig:
        self.target_vllm_config = vllm_config
        self.replicated_draft_kv = uses_dcp_replicated_gqa_draft(vllm_config)
        if self.replicated_draft_kv:
            parallel_config = copy.copy(vllm_config.parallel_config)
            parallel_config.decode_context_parallel_size = 1
            vllm_config = replace(
                vllm_config,
                parallel_config=parallel_config,
                cache_config=copy.deepcopy(vllm_config.cache_config),
                additional_config=draft_additional_config(vllm_config.additional_config),
                kv_transfer_config=None,
            )
        return vllm_config

    def load_draft_model(self, target_model: torch.nn.Module, target_attn_layer_names: set[str]) -> torch.nn.Module:
        with self._draft_dcp_context():
            model = cast("DSparkSpeculator", super()).load_draft_model(target_model, target_attn_layer_names)
        if self.replicated_draft_kv:
            # Keep identical draft grouping on P (DCP=1) and D (DCP>1).
            layers = get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase)
            for name, layer in layers.items():
                if name not in target_attn_layer_names:
                    layer._ascend_dcp_replicated_draft = True
                    layer.attn_backend = ReplicatedDraftAttentionBackend
        return model

    def set_attn(
        self,
        model_state: Any,
        kv_cache_config: Any,
        block_tables: Any,
        target_input_buffers: Any,
        target_attn_groups: Any,
    ) -> None:
        with self._draft_dcp_context():
            cast("DSparkSpeculator", super()).set_attn(
                model_state, kv_cache_config, block_tables, target_input_buffers, target_attn_groups
            )

    @contextmanager
    def _draft_dcp_context(self):
        if not self.replicated_draft_kv:
            yield
            return
        enable_dcp.cache_clear()
        try:
            with set_current_vllm_config(self.attn_vllm_config):
                yield
        finally:
            enable_dcp.cache_clear()
            with set_current_vllm_config(self.target_vllm_config):
                enable_dcp()
