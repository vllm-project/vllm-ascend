#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Adapted from vllm/model_executor/models/qwen2_5_vl.py
# Copyright 2023 The vLLM team.
#
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import \
    Qwen2_5OmniThinkerConfig
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniThinkerDummyInputsBuilder,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniThinkerMultiModalProcessor, Qwen2_5OmniThinkerProcessingInfo)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_ascend.models.qwen2_5_vl import AscendQwen2_5_VisionTransformer


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5OmniThinkerMultiModalProcessor,
    info=Qwen2_5OmniThinkerProcessingInfo,
    dummy_inputs=Qwen2_5OmniThinkerDummyInputsBuilder)
class AscendQwen2_5OmniThinkerForConditionalGeneration(
        Qwen2_5OmniThinkerForConditionalGeneration):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):

        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config: Qwen2_5OmniThinkerConfig = vllm_config.model_config.hf_config.thinker_config
        quant_config = vllm_config.quant_config
        # The following code reuse AscendQwen2_5_VisionTransformer from Qwen2_5_VL,
        # which does not import any model strcut difference. And will not impact
        # the modeling files removing.
        self.visual = AscendQwen2_5_VisionTransformer(
            vision_config=config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "visual"),
        )
