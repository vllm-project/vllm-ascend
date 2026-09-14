#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from vllm.config.model import ModelConfig
from vllm.transformers_utils.config import _CONFIG_REGISTRY

from vllm_ascend.models.glm5next.config import (
    Glm5NextConfig,
    Glm5NextTextConfig,
    Glm5NextVisionConfig,
)

# GLM-5.3-Flash config classes live in vllm-ascend while the architecture is
# maintained downstream, so the model_type lookup has to be taught about them.
# `_CONFIG_REGISTRY` is a LazyConfigDict whose values may be either a module
# attribute name or the class itself, so the classes can be inserted directly.
_CONFIG_REGISTRY["glm5_next"] = Glm5NextConfig
_CONFIG_REGISTRY["glm5_next_text"] = Glm5NextTextConfig
_CONFIG_REGISTRY["glm5_next_vision"] = Glm5NextVisionConfig

# GLM-5.3-Flash uses MLA, but upstream's `is_deepseek_mla` decides that from a
# hard-coded model_type tuple that cannot know about a downstream architecture.
# Answering False there would route the model down the non-MLA KV cache and
# quantization paths.
_GLM5_NEXT_MLA_MODEL_TYPES = frozenset({"glm5_next", "glm5_next_text"})

_original_is_deepseek_mla = ModelConfig.is_deepseek_mla.fget  # type: ignore[attr-defined]


def _is_deepseek_mla(self: ModelConfig) -> bool:
    if _original_is_deepseek_mla(self):
        return True
    if getattr(self.hf_text_config, "model_type", None) not in _GLM5_NEXT_MLA_MODEL_TYPES:
        return False
    return getattr(self.hf_text_config, "kv_lora_rank", None) is not None


ModelConfig.is_deepseek_mla = property(_is_deepseek_mla)  # type: ignore[method-assign,assignment]
