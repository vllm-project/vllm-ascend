# SPDX-License-Identifier: Apache-2.0
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
"""Shared Ascend adapter for the draft attention metadata builder.

Upstream renamed ``DraftModelSpeculator._build_draft_attn_metadata`` to
``_build_attn_metadata`` / ``_build_uniform_attn_metadata`` and changed how the
token count is derived (FULL graph: ``batch_desc.num_tokens``; otherwise
``query_start_loc_np[-1]``). Ascend keeps the old signature as a stable internal
API and routes both spellings through the version-appropriate base method while
applying its rotary-position wrapper and DecodeOnly attention state.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.attn_utils import build_draft_attn_metadata_factory


class AscendDraftAttnMetadataMixin:
    """Ascend draft-attention-metadata builder shared by all draft speculators."""

    input_batch: Any
    input_buffers: Any

    @staticmethod
    def _force_decode_only(attn_metadata: dict[str, Any] | None) -> None:
        if attn_metadata is None:
            return
        for metadata in attn_metadata.values():
            if metadata is None:
                continue
            # Ascend-specific: force DecodeOnly attention state for the draft model.
            metadata.attn_state = AscendAttentionState.DecodeOnly

    if vllm_version_is("0.28.0"):

        def _build_draft_attn_metadata(
            self,
            num_reqs: int,
            num_reqs_padded: int,
            num_tokens_padded: int,
            seq_lens_cpu_upper_bound: torch.Tensor,
            step: int,
            num_query_per_req: int = 1,
            causal: bool = True,
            query_start_loc_np: np.ndarray | None = None,
        ) -> dict[str, Any] | None:
            assert self.input_batch is not None
            with build_draft_attn_metadata_factory(
                self.input_buffers.positions,
                num_tokens_padded,
                torch.from_numpy(self.input_batch.is_prefilling_np),
            ):
                attn_metadata = DraftModelSpeculator._build_draft_attn_metadata(  # type: ignore[attr-defined]
                    self,
                    num_reqs,
                    num_reqs_padded,
                    num_tokens_padded,
                    seq_lens_cpu_upper_bound,
                    step,
                    num_query_per_req,
                    causal,
                    query_start_loc_np=query_start_loc_np,
                )
            self._force_decode_only(attn_metadata)
            return attn_metadata

    else:

        def _build_attn_metadata(  # type: ignore[misc]
            self,
            num_reqs: int,
            batch_desc: BatchExecutionDescriptor,
            query_start_loc_np: np.ndarray,
            seq_lens_cpu_upper_bound: torch.Tensor,
            step: int,
            causal: bool | Mapping[int, bool] = True,
            dcp_local_seq_lens: torch.Tensor | None = None,
        ) -> dict[str, Any] | None:
            assert self.input_batch is not None
            num_tokens = (
                batch_desc.num_tokens if batch_desc.cg_mode == CUDAGraphMode.FULL else int(query_start_loc_np[-1])
            )
            with build_draft_attn_metadata_factory(
                self.input_buffers.positions,
                num_tokens,
                torch.from_numpy(self.input_batch.is_prefilling_np),
            ):
                attn_metadata = super()._build_attn_metadata(  # type: ignore[misc,attr-defined]
                    num_reqs=num_reqs,
                    batch_desc=batch_desc,
                    query_start_loc_np=query_start_loc_np,
                    seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                    step=step,
                    causal=causal,
                    dcp_local_seq_lens=dcp_local_seq_lens,
                )
            self._force_decode_only(attn_metadata)
            return attn_metadata

        def _build_draft_attn_metadata(  # type: ignore[misc]
            self,
            num_reqs: int,
            num_reqs_padded: int,
            num_tokens_padded: int,
            seq_lens_cpu_upper_bound: torch.Tensor,
            step: int,
            num_query_per_req: int = 1,
            causal: bool = True,
            query_start_loc_np: np.ndarray | None = None,
        ) -> dict[str, Any] | None:
            """Ascend-internal stable API kept from the pre-rename upstream."""
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
                )
            return self._build_uniform_attn_metadata(  # type: ignore[attr-defined]
                batch_desc=batch_desc,
                num_reqs=num_reqs,
                num_query_per_req=num_query_per_req,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                step=step,
                causal=causal,
            )
