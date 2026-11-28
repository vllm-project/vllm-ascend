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
from copy import copy

from vllm.config import ParallelConfig, VllmConfig, replace


def get_mtp_draft_parallel_config(vllm_config: VllmConfig) -> ParallelConfig:
    """Retain the target worker's topology and use the draft's resolved EP setting."""
    target = vllm_config.parallel_config
    spec = vllm_config.speculative_config
    assert spec is not None and spec.draft_parallel_config is not None
    draft = spec.draft_parallel_config
    if target.enable_expert_parallel and not draft.enable_expert_parallel:
        # A dense draft has no experts to balance. A target's redundant expert
        # count or elastic EP flag would also fail ParallelConfig validation.
        return replace(
            target,
            pipeline_parallel_size=1,
            enable_expert_parallel=False,
            enable_eplb=False,
            enable_elastic_ep=False,
            eplb_config=copy(draft.eplb_config),
        )
    return replace(target, pipeline_parallel_size=1)
