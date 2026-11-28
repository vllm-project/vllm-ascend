# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
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
import vllm.v1.worker.gpu.spec_decode.dflash.cudagraph as cudagraph_module
import vllm.v1.worker.gpu.spec_decode.dflash.speculator as speculator_module
import vllm.v1.worker.gpu.spec_decode.dflash2.speculator as speculator2_module
import vllm.v1.worker.gpu.spec_decode.dspark.utils as dspark_utils
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import get_pcp_group

from vllm_ascend.worker.v2.attn_utils import build_attn_metadata
from vllm_ascend.worker.v2.spec_decode.dflash.aclgraph import DFlashAclGraphManager
from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import (
    _selector_walk_kernel_ascend,
)

_upstream_replace = speculator_module.replace


def _replace_draft_parallel_config(config, **changes):
    if isinstance(config, VllmConfig):
        parallel = changes.get("parallel_config", config.parallel_config)
        if (
            parallel.prefill_context_parallel_size == 1
            and parallel.tensor_parallel_size % parallel.decode_context_parallel_size
        ):
            pcp = get_pcp_group().world_size
            if pcp > 1:
                # VllmConfig revalidates its nested ParallelConfig on cloning.
                # Validate the physical topology, then restore draft execution
                # settings before handing the config to the model loader.
                physical = _upstream_replace(parallel, prefill_context_parallel_size=pcp)
                draft = _upstream_replace(config, **{**changes, "parallel_config": physical})
                draft.parallel_config = parallel
                return draft
    if (
        isinstance(config, ParallelConfig)
        and changes.get("prefill_context_parallel_size", config.prefill_context_parallel_size) == 1
    ):
        dcp = changes.get("decode_context_parallel_size", config.decode_context_parallel_size)
        tp = changes.get("tensor_parallel_size", config.tensor_parallel_size)
        pcp = config.prefill_context_parallel_size
        if tp % dcp and pcp == 1:
            # Draft loading clones the already PCP=1 config again. The worker
            # group still records the PCP axis that supplies its DCP ranks.
            pcp = get_pcp_group().world_size
        if pcp > 1 and tp % dcp:
            # DCP reuses the target's PCP ranks. Validate against that worker
            # topology, then disable only draft PCP execution; keep real DCP
            # before the base speculator caches its size and allocates state.
            draft = _upstream_replace(config, **{**changes, "prefill_context_parallel_size": pcp})
            draft.prefill_context_parallel_size = 1
            draft.world_size //= pcp
            return draft
    return _upstream_replace(config, **changes)


# Cover construction and the second parallel-config clone in DSpark loading.
# Other config copies and global config validation stay upstream.
speculator_module.replace = _replace_draft_parallel_config
dspark_utils.replace = _replace_draft_parallel_config

cudagraph_module.build_attn_metadata = build_attn_metadata
speculator_module.DFlashCudaGraphManager = DFlashAclGraphManager

# Use Ascend's int32 random path with the upstream stable Gumbel transform
# and draft noise salt; keep the shared candidate sampler unchanged.
speculator2_module._selector_walk_kernel = _selector_walk_kernel_ascend
