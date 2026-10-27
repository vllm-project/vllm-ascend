# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
import torch_npu
from torch import nn
from vllm.config import VllmConfig

from vllm_ascend.device.hardware_profile import AttentionBackendFamily, get_current_hardware_profile
from vllm_ascend.spec_decode.eagle4 import validate_eagle4_runtime
from vllm_ascend.utils import lmhead_tp_enable
from vllm_ascend.worker.v2.spec_decode.eagle.speculator import AscendEagleSpeculator


class AscendEagle4Speculator(AscendEagleSpeculator):
    """Keep Ascend's chain attention, rejection sampling and ACL graph loop."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        validate_eagle4_runtime(vllm_config)
        if (
            get_current_hardware_profile().attention_backend_family is AttentionBackendFamily.COMPATIBILITY
            or lmhead_tp_enable()
        ):
            raise NotImplementedError("Ascend EAGLE-4 does not support 310P or separate lmhead TP groups.")
        super().__init__(vllm_config, device)

    def load_draft_model(self, target_model: nn.Module, target_attn_layer_names: set[str]) -> nn.Module:
        draft = super().load_draft_model(target_model, target_attn_layer_names)
        weight = draft.lm_head.weight.detach()
        # Raw Triton loads need ND storage. If upstream shares an identical
        # target head stored in FRACTAL_NZ, normalize once before graph capture.
        # Do not change the target's parameter or cast a full head per token.
        draft.candidate_weight = torch_npu.npu_format_cast(weight, 2)
        return draft
