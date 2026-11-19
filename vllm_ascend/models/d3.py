# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from vllm.model_executor.models.d3 import D3ForSequenceClassification, D3ProcessingInfo
from vllm.model_executor.models.qwen3_vl import Qwen3VLDummyInputsBuilder
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_ascend.models.qwen3_5_vl import (
    AscendQwen3_5VLProcessor,
    _AscendProcessorMixin,
    _AscendVLPreprocessMixin,
)


class AscendD3ProcessingInfo(_AscendProcessorMixin, D3ProcessingInfo):
    pass


@MULTIMODAL_REGISTRY.register_processor(
    AscendQwen3_5VLProcessor,
    info=AscendD3ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class AscendD3ForSequenceClassification(_AscendVLPreprocessMixin, D3ForSequenceClassification):
    """D3 on the Ascend device-side Qwen3.5-VL preprocessing path.

    Reuses upstream `D3ForSequenceClassification` entirely (the `as_seq_cls_model`
    pooling wrapper, `hf_to_vllm_mapper`, the `readout.safetensors` secondary-weight
    source, `D3ProcessingInfo`'s image/video size overrides) and mixes in only the
    Ascend-specific device-side image/video preprocessing, the same way
    `AscendQwen3_5ForConditionalGeneration` does (see `qwen3_5_vl.py`): D3's backbone
    is a Qwen3.5-VL model, so it needs the same NPU-side resize/patchify/normalize
    path, not a reimplementation of it.
    """

    # `is_unused_checkpoint_weight` must stay directly callable on the class (see
    # `tests/ut/models/test_d3_registration.py`); upstream defines it as a plain
    # instance method, so redeclare it as `staticmethod` here rather than
    # reusing `super()`'s version, which needs a bound instance.
    @staticmethod
    def is_unused_checkpoint_weight(name: str) -> bool:
        return name == "weight"
