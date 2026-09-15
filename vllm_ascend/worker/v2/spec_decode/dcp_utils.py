# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replicated GQA draft support for the v2 model runner."""

import copy
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config, set_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.worker.gpu.input_batch import InputBatch

from vllm_ascend.attention.context_parallel.common_cp import expand_dcp_replicated_block_table
from vllm_ascend.attention.utils import enable_dcp
from vllm_ascend.core.kv_cache_interface import AscendDCPReplicatedDraftAttentionSpec


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


class DCPDraftReplicatedMixin:
    """Add local GQA draft KV replication before the upstream speculator in the MRO.

    The host prepares its DCP config before PCP setup and refreshes draft tables
    before proposing. Model loading and attention setup delegate through super().
    """

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
            model = super().load_draft_model(target_model, target_attn_layer_names)
        if self.replicated_draft_kv:
            # The upstream load_model sets draft_attn_layer_names after this hook.
            layers = get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase)
            for name, layer in layers.items():
                if name not in target_attn_layer_names:
                    layer._ascend_dcp_replicated_draft = True
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
            super().set_attn(model_state, kv_cache_config, block_tables, target_input_buffers, target_attn_groups)

        if self.replicated_draft_kv:
            self._target_block_tables = block_tables
            self.block_tables = copy.copy(block_tables)
            self.block_tables.cp_size = 1
            self.block_tables.cp_rank = 0
            self.block_tables.cp_interleave = 1
            # Query graphs retain these pointers. Never replace or mutate the
            # target runner's tables/slots while preparing the draft.
            self.block_tables.input_block_tables = [
                torch.zeros_like(table) for table in block_tables.input_block_tables
            ]
            self.block_tables.slot_mappings = torch.full_like(block_tables.slot_mappings, -1)
            self._replicated_columns = {}
            self._replicated_specs = {}
            for gid in self.draft_kv_cache_group_ids:
                spec = self.attn_groups[gid][0].kv_cache_spec
                if not isinstance(spec, AscendDCPReplicatedDraftAttentionSpec):
                    raise TypeError("GQA draft with target DCP requires replicated KV cache specs.")
                table = block_tables.input_block_tables[gid]
                cols = table.shape[1] * spec.dcp_replication_size
                self.block_tables.input_block_tables[gid] = torch.zeros(
                    (table.shape[0], cols), dtype=table.dtype, device=self.device
                )
                self._replicated_columns[gid] = torch.arange(cols, dtype=torch.int32, device=self.device)
                self._replicated_specs[gid] = spec

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

    def _refresh_replicated_block_tables(self, input_batch: InputBatch) -> None:
        for gid, spec in self._replicated_specs.items():
            table = self.block_tables.input_block_tables[gid]
            table.zero_()
            num_reqs = input_batch.num_reqs
            expanded = expand_dcp_replicated_block_table(
                self._target_block_tables.input_block_tables[gid][:num_reqs],
                spec.block_size,
                self.block_tables.kernel_block_sizes[gid],
                spec.dcp_replication_size,
                self._replicated_columns[gid],
            )
            table[:num_reqs].copy_(torch.where(input_batch.seq_lens[:num_reqs, None] > 0, expanded, 0))

    def _prepare_dcp_draft_batch(self, input_batch: InputBatch, dummy_run: bool, skip_attn_for_dummy_run: bool) -> None:
        if self.replicated_draft_kv and not (dummy_run and skip_attn_for_dummy_run):
            self._refresh_replicated_block_tables(input_batch)
