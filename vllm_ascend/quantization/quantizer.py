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
#

import importlib
from typing import Any, Dict, List, Optional

CUSTOMIZED_QUANTIZER_TYPE: List[str] = []


class AscendQuantizer:
    """An interface to different quantization implementations for ascend hardwares."""

    @classmethod
    def get_quantizer(cls,
                      quant_config: Dict[str, Any],
                      prefix: str,
                      packed_modules_mapping: Optional[Dict[str,
                                                            Any]] = dict()):
        # TODO: Need a param to choose quantization algorithms.
        quantization_algorithm = ''

        if quantization_algorithm in CUSTOMIZED_QUANTIZER_TYPE:
            return

        try:
            module = importlib.import_module("mindie_turbo")
            MindIETurboQuantizer = module.MindIETurboQuantizer
        except ImportError:
            raise NotImplementedError(
                "There is no available ascend quantizer.")

        return MindIETurboQuantizer.get_quantizer(quant_config, prefix,
                                                  packed_modules_mapping)

    def build_linear_method(self):
        raise NotImplementedError

    def build_moe_method(self):
        raise NotImplementedError

    def build_attention_method(self):
        raise NotImplementedError
