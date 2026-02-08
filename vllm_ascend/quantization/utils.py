#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

import os

from vllm_ascend.utils import ASCEND_QUANTIZATION_METHOD


def check_modelslim_quantization_config(model_path: str, quantization: str | None) -> None:
    """Check if the model is quantized by ModelSlim but missing quantization config.

    This function detects when a user loads a ModelSlim-quantized model without
    specifying `--quantization ascend`, which would cause a confusing KeyError
    during weight loading (e.g., KeyError: 'layers.18.mlp.gate_up_proj.clip_ratio').

    Args:
        model_path: Path to the model directory.
        quantization: The quantization method specified by the user.

    Raises:
        ValueError: If the model contains ModelSlim quantization files but the
                    user didn't specify `--quantization ascend`.
    """
    if quantization == ASCEND_QUANTIZATION_METHOD:
        # User already specified the correct quantization method
        return

    # Check if quant_model_description.json exists in the model directory
    quant_description_file = os.path.join(model_path, "quant_model_description.json")
    if os.path.isfile(quant_description_file):
        error_msg = (
            "\n"
            "=" * 80 + "\n"
            "ERROR: ModelSlim Quantized Model Detected Without Proper Configuration\n"
            "=" * 80 + "\n"
            f"\n"
            f"The model at '{model_path}' appears to be quantized using ModelSlim\n"
            f"(detected 'quant_model_description.json' file), but you haven't specified\n"
            f"the required quantization configuration.\n"
            f"\n"
            f"To fix this issue, please add the following argument to your command:\n"
            f"\n"
            f"    --quantization ascend\n"
            f"\n"
            f"Example:\n"
            f"    vllm serve {model_path} --quantization ascend\n"
            f"\n"
            f"Or in Python:\n"
            f"    LLM(model=\"{model_path}\", quantization=\"ascend\")\n"
            f"\n"
            "=" * 80
        )
        raise ValueError(error_msg)
