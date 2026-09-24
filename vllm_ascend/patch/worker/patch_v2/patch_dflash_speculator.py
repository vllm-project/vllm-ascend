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
import vllm.v1.worker.gpu.spec_decode.dflash.utils as dflash_utils
import vllm.v1.worker.gpu.spec_decode.dflash2.speculator as speculator2_module

from vllm_ascend.draft_config_context import draft_config_loading
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata
from vllm_ascend.worker.v2.spec_decode.dflash.aclgraph import DFlashAclGraphManager
from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import (
    _selector_walk_kernel_ascend,
)

_original_load_dflash_model = dflash_utils.load_dflash_model


def _load_dflash_model_with_draft_context(target_model, vllm_config):
    # DFlash retains the target model_config and passes the draft model_config
    # separately to get_model. Mark its VllmConfig reconstruction explicitly
    # so config validation does not mistake the draft for the target.
    with draft_config_loading("dflash"):
        return _original_load_dflash_model(target_model, vllm_config)


cudagraph_module.build_attn_metadata = build_attn_metadata
speculator_module.DFlashCudaGraphManager = DFlashAclGraphManager
dflash_utils.load_dflash_model = _load_dflash_model_with_draft_context
speculator_module.load_dflash_model = _load_dflash_model_with_draft_context

# triton-ascend cannot lower tldevice.log1p in the upstream selector walk;
# swap in the algebraically equivalent log(1 - u) variant.
speculator2_module._selector_walk_kernel = _selector_walk_kernel_ascend
