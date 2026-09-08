# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import vllm.lora.utils as lora_utils

from vllm_ascend.lora.fused_moe import AscendFusedMoE3DWithLoRA, AscendFusedMoEWithLoRA
from vllm_ascend.lora.utils import refresh_all_lora_classes


def test_refresh_all_lora_classes_prepends_ascend_wrappers() -> None:
    sentinel = object()
    original = lora_utils._all_lora_classes
    try:
        lora_utils._all_lora_classes = (sentinel,)
        refresh_all_lora_classes()
        classes = list(lora_utils._all_lora_classes)
        assert classes[0] is AscendFusedMoEWithLoRA
        assert classes[1] is AscendFusedMoE3DWithLoRA
        assert classes[-1] is sentinel
        # Upstream model_manager still matches the GPU class names.
        assert AscendFusedMoEWithLoRA.__name__ == "FusedMoEWithLoRA"
        assert AscendFusedMoE3DWithLoRA.__name__ == "FusedMoE3DWithLoRA"
    finally:
        lora_utils._all_lora_classes = original
