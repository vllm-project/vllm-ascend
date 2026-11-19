# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch
from vllm.config import VllmConfig
from vllm.v1.worker.gpu.spec_decode.dspine.speculator import DSpineSpeculator

from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import AscendDFlash2Speculator


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
