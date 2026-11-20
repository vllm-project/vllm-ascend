# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
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
# AscendMTPSpeculator layers the shared Ascend draft loop
# (AscendTargetDependentARSpeculator) on upstream MTPSpeculator. All MLA-specific
# handling lives in the base, gated by is_mla.
from vllm.config import VllmConfig
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

from vllm_ascend.worker.v2.spec_decode.target_dependent_ar.speculator import (
    AscendTargetDependentARSpeculator,
    _align_mtp_draft_position_cache,
)


class AscendMTPSpeculator(AscendTargetDependentARSpeculator, MTPSpeculator):
    """Ascend MTP speculator (MLA draft). All MLA handling is in the base
    (AscendTargetDependentARSpeculator)"""

    def _create_draft_vllm_config(self) -> VllmConfig:
        # MTP consumes positions from the target model. Dict-style HF overrides
        # are target-only in vLLM, so a long-context target can otherwise leave
        # the draft with the checkpoint's shorter RoPE cache and graph replay
        # will index that cache out of bounds. Align before constructing the
        # runtime draft config so every downstream consumer sees the fix.
        _align_mtp_draft_position_cache(self.vllm_config.model_config, self.draft_model_config)
        return super()._create_draft_vllm_config()
