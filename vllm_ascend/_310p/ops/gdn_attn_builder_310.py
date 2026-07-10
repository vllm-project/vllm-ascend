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

"""310P GDN metadata builder overrides."""

from __future__ import annotations

import torch
from vllm.v1.attention.backend import CommonAttentionMetadata

from vllm_ascend._310p.ops.fla.cumpute_causal_conv1d_metadata_310 import (
    compute_causal_conv1d_metadata,
)
from vllm_ascend.ops.gdn_attn_builder import (
    AscendGDNAttentionBackend,
    AscendGDNAttentionMetadataBuilder,
)


class GDNAttentionMetadataBuilder310(AscendGDNAttentionMetadataBuilder):
    """310P-specific metadata policies and prefill preparation."""

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
        del common_attn_metadata, num_prefills
        assert non_spec_query_start_loc_cpu is not None

        has_initial_state = context_lens_tensor > 0
        if spec_sequence_masks_cpu is not None:
            assert non_spec_sequence_indices is not None
            has_initial_state = torch.index_select(
                has_initial_state,
                0,
                non_spec_sequence_indices,
            )
        nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
            non_spec_query_start_loc_cpu,
            device=query_start_loc.device,
        )
        return (
            has_initial_state,
            nums_dict,
            batch_ptr,
            token_chunk_offset_ptr,
        )


# Keep the name introduced by the 310P ACL graph padding patch so existing
# imports and tests from that patch continue to work after rebasing onto
# upstream/main, whose class name is GDNAttentionMetadataBuilder310.
AscendGDNAttentionMetadataBuilder310 = GDNAttentionMetadataBuilder310


class AscendGDNAttentionBackend310(AscendGDNAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[AscendGDNAttentionMetadataBuilder310]:
        return AscendGDNAttentionMetadataBuilder310
