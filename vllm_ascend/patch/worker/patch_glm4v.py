#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#

from vllm.model_executor.models.glm4v import Glm4vForConditionalGeneration

_original_forward = Glm4vForConditionalGeneration.forward

def _patched_forward(self, *args, **kwargs):
    if "pixel_values" in kwargs and kwargs["pixel_values"] is not None:
        kwargs["pixel_values"] = kwargs["pixel_values"].contiguous()
    return _original_forward(self, *args, **kwargs)

Glm4vForConditionalGeneration.forward = _patched_forward
