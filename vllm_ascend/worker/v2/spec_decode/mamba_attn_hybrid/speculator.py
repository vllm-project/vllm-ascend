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
from vllm.v1.worker.gpu.spec_decode.mamba_attn_hybrid.speculator import (  # type: ignore[import-not-found]
    MambaAttnHybridSpeculator,
)

from vllm_ascend.worker.v2.spec_decode.dspark.speculator import (
    AscendDSparkSpeculator,
)


class AscendMambaAttnHybridSpeculator(AscendDSparkSpeculator, MambaAttnHybridSpeculator):
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
