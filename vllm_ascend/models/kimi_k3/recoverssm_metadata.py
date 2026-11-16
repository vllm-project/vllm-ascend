# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bind Kimi's request-indexed RecoverSSM records to Ascend commit kernels."""

from copy import copy

import torch
from vllm.distributed import get_pcp_group

from vllm_ascend.models.glm5next.recoverssm_metadata import AscendKDARecoverSSMMetadataBuilder
from vllm_ascend.ops.kda_pcp import KimiKDAHostMetadataBuilder
from vllm_ascend.ops.triton.kda.recoverssm import KDARecoverSSMCommitContext


class AscendKimiKDARecoverSSMMetadataBuilder(AscendKDARecoverSSMMetadataBuilder):
    recoverssm_context: KDARecoverSSMCommitContext | None

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
        if pcp_size > 1:
            vllm_config = copy(vllm_config)
            vllm_config.compilation_config = copy(vllm_config.compilation_config)
            capture_size = vllm_config.compilation_config.max_cudagraph_capture_size
            if capture_size is not None:
                vllm_config.compilation_config.max_cudagraph_capture_size = capture_size * pcp_size
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.pcp_size = pcp_size
        self.pcp_rank = get_pcp_group().rank_in_group if pcp_size > 1 else 0
        capacity = max(vllm_config.scheduler_config.max_num_batched_tokens, self.decode_cudagraph_max_bs)
        self._pcp_restore_idx = torch.zeros(capacity * pcp_size, dtype=torch.long, device=device)
        self._pcp_restore_mask = torch.zeros(capacity * pcp_size, dtype=torch.bool, device=device)
        self._pcp_local_idx = torch.zeros(capacity, dtype=torch.long, device=device)
        self._pcp_local_mask = torch.zeros(capacity, dtype=torch.bool, device=device)

    def build(self, *args, **kwargs):
        metadata = super().build(*args, **kwargs)
        return KimiKDAHostMetadataBuilder._attach_pcp(self, metadata, metadata.pcp_context)

    def _get_recoverssm_context(self) -> KDARecoverSSMCommitContext:
        forward_context = self.vllm_config.compilation_config.static_forward_context
        layers = [forward_context[name] for name in self.layer_names]
        context = self.recoverssm_context
        if context is None or not context.is_bound_to(layers):
            context = KDARecoverSSMCommitContext.create(
                layers,
                spec_query_len=1 + self.vllm_config.num_speculative_tokens,
                max_num_reqs=self.vllm_config.scheduler_config.max_num_seqs,
            )
            self.recoverssm_context = context
        return context
