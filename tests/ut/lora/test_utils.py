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

import torch
import vllm.lora.utils as lora_utils
from vllm.config import LoRAConfig
from vllm.model_executor.layers.linear import ReplicatedLinear

from vllm_ascend.lora.fused_moe import AscendFusedMoE3DWithLoRA, AscendFusedMoEWithLoRA
from vllm_ascend.lora.utils import (
    AscendColumnParallelLinearWithLoRA,
    AscendMergedColumnParallelLinearWithLoRA,
    AscendMergedQKVParallelLinearWithLoRA,
    AscendReplicatedLinearWithLoRA,
    AscendRowParallelLinearWithLoRA,
    refresh_all_lora_classes,
)


def test_refresh_all_lora_classes_prepends_ascend_wrappers() -> None:
    sentinel = object()
    original = lora_utils._all_lora_classes
    try:
        lora_utils._all_lora_classes = (sentinel,)
        refresh_all_lora_classes()
        classes = list(lora_utils._all_lora_classes)
        expected_prefix = [
            AscendReplicatedLinearWithLoRA,
            AscendRowParallelLinearWithLoRA,
            AscendColumnParallelLinearWithLoRA,
            AscendMergedColumnParallelLinearWithLoRA,
            AscendMergedQKVParallelLinearWithLoRA,
            AscendFusedMoEWithLoRA,
            AscendFusedMoE3DWithLoRA,
        ]
        assert classes[: len(expected_prefix)] == expected_prefix
        assert classes[-1] is sentinel
        # Upstream model_manager still matches the GPU class names.
        assert AscendFusedMoEWithLoRA.__name__ == "FusedMoEWithLoRA"
        assert AscendFusedMoE3DWithLoRA.__name__ == "FusedMoE3DWithLoRA"
    finally:
        lora_utils._all_lora_classes = original


def test_single_linear_wrapper_tracks_loaded_lora_slots() -> None:
    layer = object.__new__(AscendReplicatedLinearWithLoRA)
    torch.nn.Module.__init__(layer)
    layer.base_layer = object.__new__(ReplicatedLinear)
    layer.n_slices = 1
    layer.input_size = 4
    layer.output_size = 6
    layer.tp_size = 1
    layer.device = torch.device("cpu")

    lora_config = LoRAConfig(max_loras=1, max_lora_rank=8, lora_dtype=torch.bfloat16)
    layer.create_lora_weights(max_loras=1, lora_config=lora_config)
    assert layer._lora_slots == set()

    layer.set_lora(
        0,
        torch.ones(8, 4, dtype=torch.bfloat16),
        torch.ones(6, 8, dtype=torch.bfloat16),
    )
    assert layer._lora_slots == {0}

    layer.reset_lora(0)
    assert layer._lora_slots == set()
