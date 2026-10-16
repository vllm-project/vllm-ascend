# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.lilicorr.speculator import LiLiCorrSpeculator

import vllm_ascend.patch.worker.patch_lilicorr  # noqa: F401
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator
from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import AscendDFlash2Speculator


class AscendLiLiCorrSpeculator(LiLiCorrSpeculator, AscendDFlash2Speculator):
    """LiLiCorr's context anchor and correlator with Ascend candidate sampling.

    LiLiCorr supplies model loading, anchor extraction and lattice scoring.
    AscendDFlash2 supplies the LM-head TP guard; AscendDFlash supplies NPU
    attention metadata, context KV preparation and ACL graph manager wiring.
    The shared CandidateSampler uses the Ascend selector-walk worker patch
    for both greedy paths and probabilistic proposal logits.
    """

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        if not self.speculative_config.enforce_eager:
            raise NotImplementedError("LiLiCorr on Ascend currently requires speculative enforce_eager=True.")
        AscendDFlashSpeculator.init_cudagraph_manager(self, CUDAGraphMode.NONE)
