# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch
from vllm.config import VllmConfig
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import CandidateSampler
from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache
from vllm.v1.worker.gpu.spec_decode.dspine.speculator import DSpineSpeculator

from vllm_ascend.ops.triton.spec_decode.dspine import sample_candidates
from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import AscendDFlash2Speculator


class AscendDSpineCandidateSampler(CandidateSampler):
    """Reuse upstream buffers with NPU walk and proposal-cache kernels."""

    def sample(
        self,
        candidate_ids: torch.Tensor,
        scores: torch.Tensor,
        num_reqs: int,
        sample_pos: torch.Tensor,
        idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_logits: DraftLogitsCache | None,
        use_fp64: bool,
    ) -> None:
        if use_fp64:
            raise NotImplementedError("DSpine FP64 Gumbel sampling is not supported on Ascend.")
        if draft_logits is not None and draft_logits.row_mapping is not None:
            raise NotImplementedError("DSpine batch-sharded draft logits are not supported on Ascend.")
        sample_candidates(
            candidate_ids,
            scores,
            num_reqs,
            sample_pos,
            idx_mapping,
            temperature,
            seeds,
            draft_tokens,
            self.scores,
            self.cached_candidate_ids,
            draft_logits.values if draft_logits is not None else None,
        )


class AscendDSpineSpeculator(DSpineSpeculator, AscendDFlash2Speculator):
    """Keep upstream DSpine candidates and reuse the Ascend DFlash2 runtime.

    DSpineSpeculator supplies transition_top_k, tuple-output handling and exact
    transition scores through _prepare_candidates. Cooperative inheritance
    retains Ascend attention metadata, KV preparation, ACL graph management and
    LM head TP guards. An inherited CandidateSampler uses explicit NPU kernels
    for both the walk and realized proposal-cache writes.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        if self.use_fp64_gumbel:
            raise NotImplementedError("DSpine FP64 Gumbel sampling is not supported on Ascend.")
        self.candidate_sampler = AscendDSpineCandidateSampler(
            self.max_num_reqs, self.num_speculative_steps, self.top_k, device
        )
