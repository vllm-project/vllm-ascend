# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from vllm.model_executor.models.dbrx import DbrxExperts
from vllm.model_executor.models.gpt_oss import GptOssRoutedExperts
from vllm.model_executor.models.transformers.moe import TransformersMoERunner, TransformersRoutedExperts

from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts


class AscendDbrxExperts(DbrxExperts, AscendRoutedExperts):
    """Preserve DBRX constructor state and its stacked checkpoint loader."""


class AscendGptOssRoutedExperts(GptOssRoutedExperts, AscendRoutedExperts):
    """Preserve GPT-OSS checkpoint loading and Ascend expert execution."""


class AscendTransformersRoutedExperts(TransformersRoutedExperts, AscendRoutedExperts):
    """Preserve Transformers loading and its externally supplied routing."""


class AscendTransformersMoERunner(TransformersMoERunner, AscendMoERunner):
    """Preserve Transformers' forward signature and Ascend's execution path."""
