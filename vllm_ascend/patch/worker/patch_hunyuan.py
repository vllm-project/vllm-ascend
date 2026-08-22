#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#

from vllm.model_executor.models.hunyuan_vision import HunYuanVLForConditionalGeneration
from vllm.model_executor.models.utils import WeightsMapper

from vllm_ascend.utils import vllm_version_is

# Upstream "Fix weight tying" (#51665) removed the conditional
# `skip_prefixes=["lm_head."]` from HunYuanVLForConditionalGeneration.load_weights
# and relies on AutoWeightsLoader's tied-embedding alias detection to skip the
# tied lm_head. That detection only recognizes vLLM module names, but checkpoints
# (e.g. vllm-ascend/HunyuanOCR) store the tied head as a top-level
# `lm_head.weight`. Without a rename, AutoWeightsLoader raises "no module or
# parameter named 'lm_head'". Translate the checkpoint name to
# `language_model.lm_head.weight`, which the alias logic then skips (or loads
# when the checkpoint ships an untied lm_head).
if not vllm_version_is("0.27.1"):
    HunYuanVLForConditionalGeneration.hf_to_vllm_mapper = (
        HunYuanVLForConditionalGeneration.hf_to_vllm_mapper
        | WeightsMapper(orig_to_new_prefix={"lm_head.": "language_model.lm_head."})
    )
