# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config import VllmConfig
from vllm.model_executor.models.clef_flash import ClefFlashForJointSchema
from vllm.model_executor.models.qwen3_vl import Qwen3VLDummyInputsBuilder
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_ascend.models.qwen3_5_vl import (
    AscendQwen3_5ProcessingInfo,
    AscendQwen3_5VLProcessor,
    _AscendVLPreprocessMixin,
)


@MULTIMODAL_REGISTRY.register_processor(
    AscendQwen3_5VLProcessor,
    info=AscendQwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class AscendClefFlashForJointSchema(_AscendVLPreprocessMixin, ClefFlashForJointSchema):
    """Reuse the upstream joint-schema model with Ascend Qwen3.5 preprocessing."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
