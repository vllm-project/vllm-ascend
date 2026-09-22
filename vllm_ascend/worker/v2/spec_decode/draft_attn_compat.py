#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

"""Restore upstream's renamed MRV2 draft-attention-metadata entry point.

Upstream main renamed ``DraftModelSpeculator._build_draft_attn_metadata`` to
``_build_attn_metadata`` (adding ``_build_uniform_attn_metadata``).  The Ascend
speculators still express their draft-metadata wrap as the old method name, so
after importing this module ``DraftModelSpeculator`` again exposes an
old-signature ``_build_draft_attn_metadata`` that forwards to the new API.
On the release tag the method already exists and nothing is patched.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

from vllm_ascend.utils import vllm_version_is


def _build_draft_attn_metadata_compat(
    self: DraftModelSpeculator,
    num_reqs: int,
    num_reqs_padded: int,
    num_tokens_padded: int,
    seq_lens_cpu_upper_bound: torch.Tensor,
    step: int,
    num_query_per_req: int = 1,
    causal: bool = True,
    query_start_loc_np: np.ndarray | None = None,
    dcp_local_seq_lens: torch.Tensor | None = None,
) -> dict[str, Any] | None:
    if num_query_per_req is None:
        num_query_per_req = self.num_query_per_req
    batch_desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=num_tokens_padded,
        num_reqs=num_reqs_padded,
    )
    if query_start_loc_np is None:
        return self._build_uniform_attn_metadata(  # type: ignore[attr-defined]
            batch_desc=batch_desc,
            num_reqs=num_reqs,
            num_query_per_req=num_query_per_req,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            step=step,
            causal=causal,
            dcp_local_seq_lens=dcp_local_seq_lens,
        )
    return self._build_attn_metadata(  # type: ignore[attr-defined]
        num_reqs=num_reqs,
        batch_desc=batch_desc,
        query_start_loc_np=query_start_loc_np,
        seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
        step=step,
        causal=causal,
        dcp_local_seq_lens=dcp_local_seq_lens,
    )


if not vllm_version_is("0.29.0") and not hasattr(DraftModelSpeculator, "_build_draft_attn_metadata"):
    DraftModelSpeculator._build_draft_attn_metadata = _build_draft_attn_metadata_compat
