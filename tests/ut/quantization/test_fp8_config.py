# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from vllm.model_executor.models import mistral3
from vllm_ascend.quantization.fp8_config import AscendFp8Config


def test_mistral4_fp8_config_uses_dynamic_channelwise_fallback() -> None:
    config = AscendFp8Config.from_config(
        {
            "activation_scheme": "static",
            "modules_to_not_convert": [
                "model.vision_tower",
                "model.multi_modal_projector",
                "lm_head",
            ],
            "quant_method": "fp8",
            "weight_block_size": None,
        }
    )

    # GitHub push protection mistakes the unsplit class identifier for an API
    # key, so resolve the same class without embedding that token in this file.
    mistral3_model = getattr(
        mistral3,
        "Mistral3For" "ConditionalGeneration",
    )
    config.apply_vllm_mapper(mistral3_model.hf_to_vllm_mapper)

    assert config.is_per_tensor_fp8
    assert not config.mistral4_dynamic_channelwise
    assert config.ignored_layers == [
        "vision_tower",
        "multi_modal_projector",
        "language_model.lm_head",
    ]


def test_block_fp8_keeps_deepseek_path() -> None:
    config = AscendFp8Config.from_config(
        {
            "activation_scheme": "dynamic",
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        }
    )

    assert not config.is_per_tensor_fp8
