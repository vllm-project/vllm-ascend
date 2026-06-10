#
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

from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend._310p.attention.attention_mask import (
    AttentionMaskBuilder310,
    is_compressed_mask_supported,
)
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata


class AscendAttentionMetadataBuilder310(AscendAttentionMetadataBuilder):
    """
    Metadata builder specialized for the Huawei Ascend 310P NPU.

    This class extends the base Ascend attention metadata builder to use
    the 310P-specific attention mask builder, ensuring that masks are
    generated in the correct format (FRACTAL_NZ) and logic required by
    the 310P hardware.
    """

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        """
        Initializes the metadata builder and the 310P-specific mask builder.

        Args:
            kv_cache_spec (AttentionSpec): Specification for the KV cache (block size, etc.).
            layer_names (list[str]): List of layer names in the model.
            vllm_config (VllmConfig): Global vLLM configuration object.
            device (torch.device): The device (NPU) to run operations on.
        """
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        # Override the mask builder with the 310P-specific version
        max_model_len = vllm_config.model_config.max_model_len
        self.attn_mask_builder: Any = AttentionMaskBuilder310(self.device, max_model_len)

        self._query_lens_cpu_pool: list[torch.Tensor] = []
        self._query_lens_cpu_pool_idx = -1
        if device.type != "cpu":
            max_num_seqs = vllm_config.scheduler_config.max_num_seqs
            self._query_lens_cpu_pool = [
                torch.empty(max_num_seqs, dtype=torch.int32, device="cpu", pin_memory=True),
                torch.empty(max_num_seqs, dtype=torch.int32, device="cpu", pin_memory=True),
            ]

    def _fill_query_lens_cpu(
        self,
        num_reqs: int,
        query_start_loc_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """Pinned CPU per-request query lengths for ATB splitfuse (host qLensTensor)."""
        query_lens_cpu = query_start_loc_cpu[1 : num_reqs + 1] - query_start_loc_cpu[:num_reqs]
        if not self._query_lens_cpu_pool:
            return query_lens_cpu.contiguous()

        self._query_lens_cpu_pool_idx = (self._query_lens_cpu_pool_idx + 1) % len(self._query_lens_cpu_pool)
        buffer = self._query_lens_cpu_pool[self._query_lens_cpu_pool_idx]
        buffer[:num_reqs].copy_(query_lens_cpu)
        return buffer[:num_reqs]

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
    ):
        attn_metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)

        num_reqs = common_attn_metadata.num_reqs

        splitfuse_states = (
            AscendAttentionState.SpecDecoding,
            AscendAttentionState.ChunkedPrefill,
        )
        if attn_metadata.attn_state not in splitfuse_states:
            return attn_metadata

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]
        # ATB splitfuse qLensTensor must be host; filled here (outside graph forward).
        attn_metadata.query_lens_cpu = self._fill_query_lens_cpu(num_reqs, query_start_loc_cpu)

        # Bind device-side views for in-place graph replay updates.
        attn_metadata.seq_lens = common_attn_metadata.seq_lens[:num_reqs]
        attn_metadata.query_start_loc = common_attn_metadata.query_start_loc[: num_reqs + 1]

        if is_compressed_mask_supported():
            attn_metadata.attn_mask = AttentionMaskBuilder310.get_compressed_splitfuse_mask(self.device)

        return attn_metadata
