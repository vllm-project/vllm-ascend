# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch
from vllm.config import VllmConfig
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import CandidateSampler
from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache
from vllm.v1.worker.gpu.spec_decode.dspine.speculator import DSpineSpeculator

from vllm_ascend.ops.triton.spec_decode.dspine import cache_draft_logits
from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import AscendDFlash2Speculator


class AscendDSpineCandidateSampler(CandidateSampler):
    """Reuse the Ascend DFlash2 walk with ordered proposal-cache writes."""

    def _cache_draft_logits(
        self,
        candidate_ids: torch.Tensor,
        num_sample: int,
        idx_mapping: torch.Tensor,
        draft_logits: DraftLogitsCache,
    ) -> None:
        if draft_logits.row_mapping is not None:
            raise NotImplementedError("DSpine batch-sharded draft logits are not supported on Ascend.")
        cache_draft_logits(
            candidate_ids,
            num_sample,
            idx_mapping,
            self.scores,
            self.cached_candidate_ids,
            draft_logits.values,
        )


class AscendDSpineSpeculator(DSpineSpeculator, AscendDFlash2Speculator):
    """Use upstream DSpine candidate preparation with the Ascend DFlash2 runtime."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        spec = vllm_config.speculative_config
        assert spec is not None
        if spec.enable_adaptive_verification:
            raise NotImplementedError("DSpine adaptive verification is not supported on Ascend.")
        if spec.draft_tensor_parallel_size != vllm_config.parallel_config.tensor_parallel_size:
            raise ValueError("DSpine draft tensor parallel size must match the target tensor parallel size.")
        super().__init__(vllm_config, device)
        if self.use_fp64_gumbel:
            raise NotImplementedError("DSpine FP64 Gumbel sampling is not supported on Ascend.")
        self.candidate_sampler = AscendDSpineCandidateSampler(
            self.max_num_reqs, self.num_speculative_steps, self.top_k, device
        )

    def supports_batch_sharded_draft_logits(self) -> bool:
        # The ordered cache writes are keyed by persistent request state.
        return False
