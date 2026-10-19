# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.v1.attention.backend import CommonAttentionMetadata

from vllm_ascend.models.glm5next.ops.metadata import aligned_mamba_block_table
from vllm_ascend.ops.gdn_attn_builder import AscendGDNHostMetadataBackend, AscendGDNHostMetadataBuilder


class Glm5NextKDAMetadataBuilder(AscendGDNHostMetadataBuilder):
    """Use GLM KDA's request order without changing other GDN builders."""

    _USE_SPEC_SEQUENCE_VIEWS = True

    def _get_mamba_block_table_tensor(self, common_attn_metadata: CommonAttentionMetadata) -> torch.Tensor:
        if (
            common_attn_metadata.block_table_tensor.device.type == "npu"
            and self.vllm_config.cache_config.mamba_cache_mode == "align"
        ):
            # Fuse the seven small kernels while retaining live device lengths.
            return aligned_mamba_block_table(
                common_attn_metadata.block_table_tensor,
                common_attn_metadata.seq_lens,
                self.kv_cache_spec.block_size,
                self.kv_cache_spec.num_speculative_blocks,
            )
        return super()._get_mamba_block_table_tensor(common_attn_metadata)


class Glm5NextKDABackend(AscendGDNHostMetadataBackend):
    @staticmethod
    def get_builder_cls() -> type[Glm5NextKDAMetadataBuilder]:
        return Glm5NextKDAMetadataBuilder
