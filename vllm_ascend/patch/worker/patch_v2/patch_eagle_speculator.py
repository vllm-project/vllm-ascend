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

from typing import Any

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as vllm_speculator_module
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph import AutoRegressiveAclGraphManager

vllm_speculator_module.SpeculatorCudaGraphManager = AutoRegressiveAclGraphManager


if not vllm_version_is("0.28.0"):
    # Upstream renamed `_build_draft_attn_metadata` to `_build_attn_metadata`
    # and added the `_build_uniform_attn_metadata` wrapper. Ascend speculator
    # subclasses and their UTs still call the old name, so keep it working on
    # main by translating to the new entry point.
    def _build_draft_attn_metadata_compat(
        self: Any,
        num_reqs: int,
        num_reqs_padded: int,
        num_tokens_padded: int,
        seq_lens_cpu_upper_bound: Any,
        step: int,
        num_query_per_req: int | None = None,
        causal: Any = True,
        query_start_loc_np: Any = None,
        dcp_local_seq_lens: Any = None,
    ) -> Any:
        if num_query_per_req is None:
            num_query_per_req = getattr(self, "num_query_per_req", 1)
        if query_start_loc_np is None:
            query_start_loc_np = self.arange_np[: num_reqs + 1] * num_query_per_req
        # The legacy `_build_draft_attn_metadata` always handed the caller's
        # `num_tokens_padded` to `build_attn_metadata`. The new
        # `_build_attn_metadata` only reads `batch_desc.num_tokens` when
        # `cg_mode == FULL` (PIECEWISE derives it from `query_start_loc_np[-1]`,
        # which shrinks the token count under request padding). Describe the
        # descriptor as FULL so the padded layout captured by the draft graph is
        # preserved for every legacy caller.
        batch_desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.FULL,
            num_tokens=num_tokens_padded,
            num_reqs=num_reqs_padded,
        )
        return self._build_attn_metadata(
            num_reqs=num_reqs,
            batch_desc=batch_desc,
            query_start_loc_np=query_start_loc_np,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            step=step,
            causal=causal,
            dcp_local_seq_lens=dcp_local_seq_lens,
        )

    DraftModelSpeculator._build_draft_attn_metadata = _build_draft_attn_metadata_compat  # type: ignore[attr-defined]
