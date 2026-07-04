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

"""310P RC GDN metadata builder: Ascend builder with RC-safe prefill metadata."""

from __future__ import annotations

import torch
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm_ascend._310p.ops.fla.cumpute_causal_conv1d_metadata_310 import (
    compute_causal_conv1d_metadata,
)
from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionMetadataBuilder


class GDNAttentionMetadataBuilder310(AscendGDNAttentionMetadataBuilder):
    """RC-only overrides on top of :class:`AscendGDNAttentionMetadataBuilder`.

    310P does not support Triton, so fallback metadata attachment is skipped.
    """

    def _build_prefill_has_initial_state_and_causal_conv1d_meta(
        self,
        *,
        common_attn_metadata: CommonAttentionMetadata,
        context_lens_tensor: torch.Tensor,
        num_prefills: int,
        spec_sequence_masks_cpu: torch.Tensor | None,
        non_spec_sequence_indices: torch.Tensor | None,
        non_spec_query_start_loc_cpu: torch.Tensor | None,
        query_start_loc: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        dict[int, dict[str, object]] | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        del non_spec_sequence_indices, num_prefills
        assert non_spec_query_start_loc_cpu is not None

        m = common_attn_metadata
        # 310P RC: compute has_initial_state on CPU, single H2D upload.
        # NOTE: This must mirror the mainline device path
        # ``has_initial_state = context_lens_tensor > 0`` where
        # ``context_lens_tensor = m.compute_num_computed_tokens()`` is the
        # number of *already computed* tokens (context length), NOT the total
        # ``seq_lens``. Using ``seq_lens`` would flag fresh prefills (0
        # computed tokens) as having an initial recurrent state and read
        # uninitialized SSM state -> garbage output.
        num_computed_tokens_cpu = getattr(m, "_num_computed_tokens_cpu", None)
        if num_computed_tokens_cpu is None:
            num_computed_tokens_cpu = getattr(m, "num_computed_tokens_cpu", None)
        if num_computed_tokens_cpu is not None:
            context_lens_cpu = num_computed_tokens_cpu
        else:
            context_lens_cpu = context_lens_tensor.detach().cpu()

        has_initial_state_cpu = context_lens_cpu > 0
        if spec_sequence_masks_cpu is not None:
            has_initial_state_cpu = has_initial_state_cpu[~spec_sequence_masks_cpu]

        has_initial_state = has_initial_state_cpu.to(
            query_start_loc.device,
            non_blocking=True,
        )
        nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
            non_spec_query_start_loc_cpu,
            device=query_start_loc.device,
        )
        return has_initial_state, nums_dict, batch_ptr, token_chunk_offset_ptr

    def _attach_non_spec_prefill_fallback_meta(
        self,
        attn_metadata: GDNAttentionMetadata,
        common_attn_metadata: CommonAttentionMetadata,
        non_spec_query_start_loc_cpu: torch.Tensor | None,
    ) -> GDNAttentionMetadata:
        del common_attn_metadata, non_spec_query_start_loc_cpu
        return attn_metadata

    def _attach_spec_decode_fallback_meta(
        self,
        attn_metadata: GDNAttentionMetadata,
        common_attn_metadata: CommonAttentionMetadata,
        num_decode_draft_tokens_cpu: torch.Tensor | None,
    ) -> GDNAttentionMetadata:
        del common_attn_metadata, num_decode_draft_tokens_cpu
        return attn_metadata

    def _attach_non_spec_decode_fallback_meta(
        self,
        attn_metadata: GDNAttentionMetadata,
        common_attn_metadata: CommonAttentionMetadata,
        num_decode_draft_tokens_cpu: torch.Tensor | None,
    ) -> GDNAttentionMetadata:
        del common_attn_metadata, num_decode_draft_tokens_cpu
        return attn_metadata
