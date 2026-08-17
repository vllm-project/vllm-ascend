# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ascend implementation of the standalone Qwen3-Omni DSpark drafter."""

from vllm.model_executor.models.qwen3_omni_dspark import Qwen3OmniDSparkModel

from vllm_ascend.models.qwen3_dspark import AscendQwen3DSparkForCausalLM


class AscendQwen3OmniDSparkForCausalLM(AscendQwen3DSparkForCausalLM):
    """Qwen3-Omni DSpark with Ascend draft-weight preprocessing."""

    model_cls = Qwen3OmniDSparkModel
