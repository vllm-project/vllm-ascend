# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Route V4.1 runtime-NONE steps around the compiled model wrapper.

V4.1's Python reference compressor/indexer path is correctness-safe in eager
mode, while only uniform decode is prepared for a full ACL graph.
FULL_DECODE_ONLY dispatches prefills and unsupported decode shapes as runtime
NONE; upstream's ``skip_compiled`` only covers encoder-decoder steps
(``vllm/v1/worker/gpu/model_runner.py`` passes ``skip_compiled=has_encoder_input``),
so wrap the module-level ``set_forward_context`` consumed by
``GPUModelRunner.execute_model`` to force eager for those V4.1 calls. The
wrapper checks the model type on every call, so non-V4.1 runners are
unaffected. Runner V1 resolves ``set_forward_context`` through
``ascend_forward_context.set_ascend_forward_context`` and is not intercepted.
"""

from contextlib import contextmanager

from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu import model_runner as vllm_model_runner

from vllm_ascend.utils import is_deepseek_v41

_original_set_forward_context = vllm_model_runner.set_forward_context


@contextmanager
def v41_aware_set_forward_context(*args, **kwargs):
    vllm_config = args[1] if len(args) > 1 else kwargs.get("vllm_config")
    if (
        vllm_config is not None
        and is_deepseek_v41(vllm_config.model_config.hf_config)
        and kwargs.get("cudagraph_runtime_mode", CUDAGraphMode.NONE) == CUDAGraphMode.NONE
    ):
        kwargs["skip_compiled"] = True
    with _original_set_forward_context(*args, **kwargs):
        yield


vllm_model_runner.set_forward_context = v41_aware_set_forward_context
