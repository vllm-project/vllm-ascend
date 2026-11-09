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
from contextlib import nullcontext
from typing import Any

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.spec_decode.target_dependent_ar.speculator import AscendTargetDependentARSpeculator


class AscendMTPSpeculator(AscendTargetDependentARSpeculator, MTPSpeculator):
    """Ascend MTP speculator (MLA draft). All MLA handling is in the base
    (AscendTargetDependentARSpeculator)"""

    pass


class AscendGlm5NextMTPSpeculator(AscendMTPSpeculator):
    def _prefill(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        skip_sample: bool = False,
    ) -> None:
        cache_only = (
            skip_sample
            and cudagraph_runtime_mode == CUDAGraphMode.NONE
            and num_reqs == 1
            and self.input_batch is not None
            and bool(self.input_batch.is_prefilling_np[0])
            and not vllm_version_is("0.32.0")
        )
        with self.model.prefill_cache_only() if cache_only else nullcontext():
            super()._prefill(
                num_reqs,
                num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
                mm_inputs,
                skip_sample,
            )
