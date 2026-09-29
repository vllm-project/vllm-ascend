# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Ascend adaptation of the upstream MambaAttnHybridSpeculator (H-Spec).
#
# Design: subclass AscendDSparkSpeculator and linearize the upstream
# MambaAttnHybridSpeculator into the MRO, so every MRV2/ACL-graph contract
# (enforce-eager gating, BatchExecutionDescriptor-based uniform attention
# metadata, padded FULL-graph query lengths, DCP/PCP preparation, and the
# order-preserving cache-group walk) is inherited from the DSpark adaptation
# instead of being re-implemented here. Only the H-Spec specifics live here:
# the latent-seed prefill in propose(), and mirroring the drafter's
# sliding/full decision (dflash_causal) into the group-causal slot consumed
# by the DSpark metadata builders.
#
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.mamba_attn_hybrid.speculator import (
    MambaAttnHybridSpeculator,
)

from vllm_ascend.worker.v2.spec_decode.dspark.speculator import (
    AscendDSparkSpeculator,
)


class AscendMambaAttnHybridSpeculator(
    AscendDSparkSpeculator, MambaAttnHybridSpeculator
):
    _speculator_name = "MambaAttnHybrid"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        parallel_config = vllm_config.parallel_config
        if (
            getattr(parallel_config, "decode_context_parallel_size", 1) > 1
            or getattr(parallel_config, "prefill_context_parallel_size", 1) > 1
        ):
            # The upstream drafter rejects PP != 1 and the DSpark DCP/PCP
            # metadata preparation assumes DFlash-style drafting; fail fast
            # instead of producing silently misaligned draft metadata.
            raise NotImplementedError(
                "mamba_attn_hybrid does not support DCP/PCP yet; run with "
                "decode_context_parallel_size=1 and prefill_context_parallel_size=1."
            )
        super().__init__(vllm_config, device)

    def set_attn(
        self,
        model_state: Any,
        kv_cache_config: Any,
        block_tables: Any,
        target_input_buffers: Any = None,
        target_attn_groups: Any = None,
    ) -> None:
        # MRO: AscendDSparkSpeculator.set_attn (int32 slot mappings,
        # order-preserving backend walk, attn_architecture detection) then
        # MambaAttnHybridSpeculator.set_attn (sliding/full recipe validation,
        # sets dflash_causal).
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )
        # The DSpark metadata builders take the causal decision from
        # _group_causal; the hybrid drafter's is dflash_causal (True when all
        # attention sub-layers are sliding-window, i.e. in-fill drafting).
        self._group_causal = self.dflash_causal

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
        dp_sync: Any = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        # Latent-seed prefill from the verifier's fusion-layer hidden states,
        # ported from MambaAttnHybridSpeculator.propose whose signature
        # predates the dp_sync proposal contract (this skips that override:
        # calling it would forward dp_sync under the old num_tokens_across_dp
        # keyword and break the DSpark proposal path).
        if aux_hidden_states:
            num_reqs = input_batch.num_reqs
            fused = torch.cat(aux_hidden_states, dim=-1)
            anchor = (
                input_batch.query_start_loc[1 : num_reqs + 1]
                - 1
                - num_rejected[:num_reqs]
            )
            self.latent_seed[:num_reqs].copy_(fused[anchor])
        else:
            last_hidden_states = last_hidden_states.new_zeros(
                last_hidden_states.shape[0], self.hidden_states.shape[1]
            )

        return AscendDSparkSpeculator.propose(
            self,
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
            dp_sync,
            dummy_run,
            skip_attn_for_dummy_run,
            mm_inputs,
            is_profile=is_profile,
        )
