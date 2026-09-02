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

"""310P RC GDN metadata builder.

This 310P-specific builder keeps the upstream RC-safe prefill metadata path
and adds ACL graph replay padding for decode / speculative decode metadata.
"""

from __future__ import annotations

import torch
from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from vllm_ascend._310p.dflash_full import is_310p_dflash_full
from vllm_ascend._310p.dflash_full_and_piecewise import (
    is_310p_dflash_full_and_piecewise,
)
from vllm_ascend._310p.dflash_full_decode_only import (
    is_310p_dflash_full_decode_only,
)
from vllm_ascend._310p.ops.fla.cumpute_causal_conv1d_metadata_310 import (
    compute_causal_conv1d_metadata,
)
from vllm_ascend.ops.gdn_attn_builder import (
    AscendGDNAttentionBackend,
    AscendGDNAttentionMetadataBuilder,
)


def _should_pad_spec_tokens_to_graph_descriptor(
    vllm_config,
    *,
    use_full_graph: bool,
) -> bool:
    """Keep Hybrid/FDO FULL spec-token views at descriptor extent."""
    return use_full_graph and (
        is_310p_dflash_full_decode_only(vllm_config)
        or is_310p_dflash_full_and_piecewise(vllm_config)
    )


class GDNAttentionMetadataBuilder310(AscendGDNAttentionMetadataBuilder):
    """310P overrides on top of :class:`AscendGDNAttentionMetadataBuilder`.

    310P does not support Triton, so fallback metadata attachment is skipped.
    For ACL graph replay, decode metadata is padded into fixed graph buffers.
    """

    use_full_cuda_graph: bool

    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
        if is_310p_dflash_full(vllm_config):
            return AttentionCGSupport.ALWAYS
        return super().get_cudagraph_support(vllm_config, kv_cache_spec)

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

    def _pad_spec_decode_metadata(
        self,
        attn_metadata: GDNAttentionMetadata,
        graph_batch_size: int,
        graph_num_tokens: int,
        pad_to_graph_descriptor: bool = False,
    ) -> None:
        num_spec_decodes = attn_metadata.num_spec_decodes
        spec_state_indices = attn_metadata.spec_state_indices_tensor
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        num_accepted_tokens = attn_metadata.num_accepted_tokens
        assert spec_state_indices is not None
        assert spec_sequence_masks is not None
        assert spec_query_start_loc is not None
        assert num_accepted_tokens is not None

        self.spec_state_indices_tensor[:num_spec_decodes].copy_(
            spec_state_indices,
            non_blocking=True,
        )
        attn_metadata.spec_state_indices_tensor = self.spec_state_indices_tensor[:graph_batch_size]
        attn_metadata.spec_state_indices_tensor[num_spec_decodes:].fill_(NULL_BLOCK_ID)

        self.spec_sequence_masks[:num_spec_decodes].copy_(
            spec_sequence_masks[:num_spec_decodes],
            non_blocking=True,
        )
        attn_metadata.spec_sequence_masks = self.spec_sequence_masks[:graph_batch_size]
        attn_metadata.spec_sequence_masks[num_spec_decodes:].fill_(False)

        assert attn_metadata.non_spec_token_indx is not None
        assert attn_metadata.spec_token_indx is not None
        non_spec_tokens = attn_metadata.non_spec_token_indx
        spec_tokens = attn_metadata.spec_token_indx
        self.non_spec_token_indx[: non_spec_tokens.size(0)].copy_(
            non_spec_tokens,
            non_blocking=True,
        )
        self.spec_token_indx[: spec_tokens.size(0)].copy_(
            spec_tokens,
            non_blocking=True,
        )
        num_spec_tokens = spec_tokens.size(0)
        descriptor_tokens = graph_num_tokens if pad_to_graph_descriptor else num_spec_tokens
        if not num_spec_tokens <= descriptor_tokens <= self.spec_token_indx.size(0):
            raise RuntimeError(
                "310P GDN FULL spec token descriptor is outside its "
                "persistent buffer: "
                f"actual={num_spec_tokens}, descriptor={descriptor_tokens}, "
                f"capacity={self.spec_token_indx.size(0)}"
            )
        if num_spec_tokens < descriptor_tokens:
            self.spec_token_indx[num_spec_tokens:descriptor_tokens].copy_(
                torch.arange(
                    num_spec_tokens,
                    descriptor_tokens,
                    dtype=self.spec_token_indx.dtype,
                    device=self.spec_token_indx.device,
                ),
                non_blocking=True,
            )
        attn_metadata.non_spec_token_indx = self.non_spec_token_indx[: non_spec_tokens.size(0)]
        attn_metadata.spec_token_indx = self.spec_token_indx[:descriptor_tokens]

        self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
            spec_query_start_loc,
            non_blocking=True,
        )
        attn_metadata.spec_query_start_loc = self.spec_query_start_loc[: graph_batch_size + 1]
        query_padding = attn_metadata.spec_query_start_loc[num_spec_decodes + 1 :]
        if query_padding.numel() > 0:
            query_padding.copy_(
                spec_query_start_loc[-1].expand_as(query_padding),
                non_blocking=True,
            )

        self.num_accepted_tokens[:num_spec_decodes].copy_(
            num_accepted_tokens,
            non_blocking=True,
        )
        attn_metadata.num_accepted_tokens = self.num_accepted_tokens[:graph_batch_size]
        attn_metadata.num_accepted_tokens[num_spec_decodes:].fill_(0)
        self._attach_spec_decode_metadata(attn_metadata)

    def _pad_decode_metadata(
        self,
        attn_metadata: GDNAttentionMetadata,
        graph_batch_size: int,
    ) -> None:
        num_decodes = attn_metadata.num_decodes
        state_indices = attn_metadata.non_spec_state_indices_tensor
        query_start_loc = attn_metadata.non_spec_query_start_loc
        assert state_indices is not None
        assert query_start_loc is not None

        self.non_spec_state_indices_tensor[:num_decodes].copy_(
            state_indices,
            non_blocking=True,
        )
        attn_metadata.non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[:graph_batch_size]
        attn_metadata.non_spec_state_indices_tensor[num_decodes:].fill_(NULL_BLOCK_ID)

        self.non_spec_query_start_loc[: num_decodes + 1].copy_(
            query_start_loc,
            non_blocking=True,
        )
        attn_metadata.non_spec_query_start_loc = self.non_spec_query_start_loc[: graph_batch_size + 1]
        query_padding = attn_metadata.non_spec_query_start_loc[num_decodes + 1 :]
        if query_padding.numel() > 0:
            query_padding.copy_(
                query_start_loc[-1].expand_as(query_padding),
                non_blocking=True,
            )
        self._attach_non_spec_decode_metadata(
            attn_metadata,
            attn_metadata.non_spec_state_indices_tensor,
        )

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        use_full_graph = self.use_full_cuda_graph
        self.use_full_cuda_graph = False
        try:
            attn_metadata = super().build(
                common_prefix_len,
                common_attn_metadata,
                num_accepted_tokens,
                num_decode_draft_tokens_cpu,
                fast_build,
            )
        finally:
            self.use_full_cuda_graph = use_full_graph

        if not use_full_graph:
            return attn_metadata

        graph_batch_size = common_attn_metadata.num_reqs
        if (
            attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes == 0
            and attn_metadata.num_spec_decodes <= self.decode_cudagraph_max_bs
            and attn_metadata.num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        ):
            self._pad_spec_decode_metadata(
                attn_metadata,
                graph_batch_size,
                common_attn_metadata.num_input_tokens,
                pad_to_graph_descriptor=(
                    _should_pad_spec_tokens_to_graph_descriptor(
                        self.vllm_config,
                        use_full_graph=use_full_graph,
                    )
                ),
            )
        elif (
            attn_metadata.num_prefills == 0
            and attn_metadata.num_spec_decodes == 0
            and attn_metadata.num_decodes <= self.decode_cudagraph_max_bs
        ):
            self._pad_decode_metadata(attn_metadata, graph_batch_size)
        return attn_metadata


# Keep the name introduced by the 310P ACL graph padding patch so existing
# imports and tests from that patch continue to work after rebasing onto
# upstream/main, whose class name is GDNAttentionMetadataBuilder310.
AscendGDNAttentionMetadataBuilder310 = GDNAttentionMetadataBuilder310


class AscendGDNAttentionBackend310(AscendGDNAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[AscendGDNAttentionMetadataBuilder310]:
        return AscendGDNAttentionMetadataBuilder310
