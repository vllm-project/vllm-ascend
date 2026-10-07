# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/spec_decode/eagle.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
from vllm.config import VllmConfig, replace
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator

from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import (
    AscendAutoRegressiveSpeculator,
)
from vllm_ascend.worker.v2.spec_decode.pcp_utils import (
    replace_replicated_pcp_parallel_config,
)


class AscendEagleSpeculator(AscendAutoRegressiveSpeculator, EagleSpeculator):
    """Ascend Eagle speculator: the NPU loop from AscendAutoRegressiveSpeculator
    layered on upstream EagleSpeculator (flat/GQA attention)."""

    def _create_draft_vllm_config(self) -> VllmConfig:
        # EAGLE draft models are dense even when the target is an MoE model.
        # The base swaps in the draft model config without re-validating it;
        # the draft-only settings on top turn EP/EPLB off (no experts).
        draft_vllm_config = super()._create_draft_vllm_config()
        parallel_config = draft_vllm_config.parallel_config
        changes = {
            "prefill_context_parallel_size": 1,
            "enable_expert_parallel": False,
            "enable_eplb": False,
        }
        if self.replicated_pcp:
            parallel_config = replace_replicated_pcp_parallel_config(parallel_config, **changes)
        else:
            parallel_config = replace(parallel_config, **changes)
        draft_vllm_config.parallel_config = parallel_config
        return draft_vllm_config
