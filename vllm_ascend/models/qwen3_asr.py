# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Expose Qwen3-ASR's EAGLE3 interface on supported upstream versions."""

from vllm.model_executor.models.interfaces import SupportsEagle3
from vllm.model_executor.models.qwen3_asr import Qwen3ASRForConditionalGeneration


class AscendQwen3ASRForConditionalGeneration(Qwen3ASRForConditionalGeneration, SupportsEagle3):
    """Reuse the ASR model and EAGLE3 auxiliary-state delegation from vLLM.

    This covers vLLM refs predating upstream #52824. The inherited decoder
    forward already preserves the auxiliary hidden-state return value.
    """
