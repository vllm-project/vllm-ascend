# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Ascend adaptation of the upstream MambaAttnHybridSpeculator, following the
# same pattern as AscendDSparkSpeculator: bind the ACL-graph manager refs,
# cast slot mappings to int32, and build draft attention metadata through the
# Ascend attention-utils wrapper.
#
from typing import Any, cast

import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.mamba_attn_hybrid.speculator import (
    MambaAttnHybridSpeculator,
)

from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata_wrapper


class AscendMambaAttnHybridSpeculator(MambaAttnHybridSpeculator):
    _speculator_name = "MambaAttnHybrid"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self.input_batch: InputBatch | None = None

    def init_cudagraph_manager(self, cudagraph_mode) -> None:
        super().init_cudagraph_manager(cudagraph_mode)
        # The Ascend graph manager is created by
        # super().init_cudagraph_manager without a speculator ref. It needs
        # the speculator to update full-graph params, so set it here.
        self.query_cudagraph_manager.speculator = self
        self.query_cudagraph_manager.update_stream = self.update_stream

    def set_attn(
        self,
        model_state,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
        target_input_buffers=None,
        target_attn_groups=None,
    ) -> None:
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )
        self._context_slot_mappings = self._context_slot_mappings.to(torch.int32)  # type: ignore[has-type]
        # NPU needs attn_backends to update full graph params in run_fullgraph.
        attn_backends: dict[str, type[AttentionBackend]] = {}
        active_layer_names = self.draft_attn_layer_names
        for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
            layer_names = kv_cache_group_spec.layer_names
            if active_layer_names is not None:
                layer_names = list(active_layer_names.intersection(layer_names))

            layer_type = cast(type[Any], AttentionLayerBase)
            attn_layers = get_layers_from_vllm_config(
                self.vllm_config, layer_type, layer_names
            )

            for layer_name in layer_names:
                attn_backends[layer_name] = attn_layers[layer_name].get_attn_backend()

        self.attn_backends = attn_backends

    if vllm_version_is("0.26.0"):

        def build_draft_attn_metadatas(self, num_reqs_padded, seq_lens_cpu_upper_bound):
            num_tokens_padded = num_reqs_padded * self.num_query_per_req
            assert self.input_batch is not None
            with build_attn_metadata_wrapper():
                attn_metadata = self._build_draft_attn_metadata(
                    num_reqs=self.input_batch.num_reqs,
                    num_reqs_padded=num_reqs_padded,
                    num_tokens_padded=num_tokens_padded,
                    causal=self.dflash_causal,
                )
            return [attn_metadata]
    else:

        def build_draft_attn_metadatas(self, num_reqs_padded, seq_lens_cpu_upper_bound):
            num_tokens_padded = num_reqs_padded * self.num_query_per_req
            assert self.input_batch is not None
            with build_attn_metadata_wrapper():
                attn_metadata = self._build_draft_attn_metadata(
                    num_reqs=self.input_batch.num_reqs,
                    num_reqs_padded=num_reqs_padded,
                    num_tokens_padded=num_tokens_padded,
                    seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                    step=self.num_query_per_req,
                    causal=self.dflash_causal,
                )
            return [attn_metadata]

    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        self.input_batch = input_batch
        with build_attn_metadata_wrapper():
            return super().propose(
                input_batch,
                attn_metadata,
                slot_mappings,
                last_hidden_states,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                temperature,
                seeds,
                num_tokens_across_dp,
                dummy_run,
                skip_attn_for_dummy_run,
                mm_inputs,
                is_profile=is_profile,
            )
