# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""MRV2 model state for Ascend 310P."""

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache

from vllm_ascend.worker.v2.model_states.default import AscendModelState

from .sampler import Ascend310PSampler


class Ascend310PModelState(AscendModelState):
    """Model state with the Triton-free 310P sampler."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        if encoder_cache is not None:
            # TODO: Support multimodal encoder state in the next 310P MRV2 iteration.
            raise NotImplementedError("Multimodal encoder state is not supported by model runner v2 on 310P.")
        # Plain-text Qwen3 uses ordinary 1D RoPE, for which upstream returns
        # no RopeState and therefore does not launch its Triton position kernel.
        super().__init__(vllm_config, model, encoder_cache, device)

    def custom_sampler(self, sampler):
        del sampler
        return Ascend310PSampler(), None
