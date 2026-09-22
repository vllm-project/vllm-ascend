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

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as vllm_speculator_module
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph import AutoRegressiveAclGraphManager

vllm_speculator_module.SpeculatorCudaGraphManager = AutoRegressiveAclGraphManager

if not vllm_version_is("0.29.0"):

    def _compat_build_draft_attn_metadata(
        self,
        num_reqs: int,
        num_reqs_padded: int,
        num_tokens_padded: int,
        seq_lens_cpu_upper_bound,
        step: int,
        num_query_per_req: int = 1,
        causal=True,
        query_start_loc_np=None,
        dcp_local_seq_lens=None,
    ):
        """Adapter for the pre-refactor draft metadata API.

        vLLM main split ``DraftModelSpeculator._build_draft_attn_metadata``
        into ``_build_attn_metadata`` + ``_build_uniform_attn_metadata`` and
        dropped the DFlash override. vllm-ascend's draft speculators still call
        the old entry point internally, so restore it as a thin adapter that
        delegates to the new API. The old callers always passed an explicit
        padded token count (FULL-graph shape), so replay it through
        ``batch_desc.num_tokens``.
        """
        batch_desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.FULL,
            num_tokens=num_tokens_padded,
            num_reqs=num_reqs_padded,
        )
        if query_start_loc_np is not None:
            return self._build_attn_metadata(
                num_reqs=num_reqs,
                batch_desc=batch_desc,
                query_start_loc_np=query_start_loc_np,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                step=step,
                causal=causal,
                dcp_local_seq_lens=dcp_local_seq_lens,
            )
        return self._build_uniform_attn_metadata(
            batch_desc=batch_desc,
            num_reqs=num_reqs,
            num_query_per_req=num_query_per_req,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            step=step,
            causal=causal,
            dcp_local_seq_lens=dcp_local_seq_lens,
        )

    DraftModelSpeculator._build_draft_attn_metadata = (  # type: ignore[method-assign,attr-defined,assignment]
        _compat_build_draft_attn_metadata
    )
