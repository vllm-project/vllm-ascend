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
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend._310p.attention.attention_mask import (
    AttentionMaskBuilder310,
    is_compressed_mask_supported,
)
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
    AscendMetadata,
)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata

QUERY_LENS_CPU_ATTR = "query_lens_cpu"
SPLITFUSE_MASK_NZ_ATTR = "splitfuse_mask_nz"

# Batches at or below this token count may be graph-captured (spec-decode /
# decode sizes); their mask must live in a stable-address buffer the builder
# refreshes before every replay. Larger (eager-only) batches skip the copy.
_MASK_PERSISTENT_MAX_TOKENS = 64


def set_query_lens_cpu(attn_metadata: AscendMetadata, query_lens_cpu: torch.Tensor) -> None:
    """Attach host qLens for ATB splitfuse without extending upstream AscendMetadata."""
    setattr(attn_metadata, QUERY_LENS_CPU_ATTR, query_lens_cpu)


def get_splitfuse_mask_nz(attn_metadata: AscendMetadata) -> torch.Tensor | None:
    return getattr(attn_metadata, SPLITFUSE_MASK_NZ_ATTR, None)


def get_query_lens_cpu(attn_metadata: AscendMetadata) -> torch.Tensor | None:
    value = getattr(attn_metadata, QUERY_LENS_CPU_ATTR, None)
    if value is None:
        return None
    return value


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

        self._query_lens_cpu_buffer: torch.Tensor | None = None
        if device.type != "cpu":
            max_num_seqs = vllm_config.scheduler_config.max_num_seqs
            self._query_lens_cpu_buffer = torch.empty(max_num_seqs, dtype=torch.int32, device="cpu", pin_memory=True)
        # Stable-address NZ mask buffers, one per captured batch size (see build()).
        self._splitfuse_mask_bufs: dict[int, torch.Tensor] = {}

    def _fill_query_lens_cpu(
        self, num_reqs: int, query_start_loc_cpu: torch.Tensor, is_drafting: bool = False
    ) -> torch.Tensor:
        """Pinned CPU per-request query lengths for ATB splitfuse (host qLensTensor)."""
        if self._query_lens_cpu_buffer is None:
            return (query_start_loc_cpu[1 : num_reqs + 1] - query_start_loc_cpu[:num_reqs]).contiguous()
        if is_drafting:
            # We are using the same buffer for multi step drafting,
            # so we have to clone the buffer or the q lens of step 0
            # will be overwritten by the following steps.
            buffer = self._query_lens_cpu_buffer[:num_reqs].clone()
        else:
            buffer = self._query_lens_cpu_buffer[:num_reqs]
        torch.sub(
            query_start_loc_cpu[1 : num_reqs + 1],
            query_start_loc_cpu[:num_reqs],
            out=buffer,
        )
        return buffer

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
        is_drafting: bool = False,
    ) -> AscendMetadata:
        attn_metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)

        num_reqs = common_attn_metadata.num_reqs

        splitfuse_states = (
            AscendAttentionState.SpecDecoding,
            AscendAttentionState.ChunkedPrefill,
        )

        # Paged and splitfuse attention consume device-side context lengths.
        # Bind the persistent input buffers before graph capture so their
        # forward paths do not trigger a pageable host-to-device copy.
        device_metadata_states = (
            AscendAttentionState.DecodeOnly,
            *splitfuse_states,
        )
        if attn_metadata.attn_state in device_metadata_states:
            attn_metadata.seq_lens = common_attn_metadata.seq_lens[:num_reqs]
            attn_metadata.query_start_loc = common_attn_metadata.query_start_loc[: num_reqs + 1]

        if attn_metadata.attn_state not in splitfuse_states:
            return attn_metadata

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]
        # ATB splitfuse qLensTensor must be host; filled here (outside graph forward).
        set_query_lens_cpu(
            attn_metadata,
            self._fill_query_lens_cpu(num_reqs, query_start_loc_cpu, is_drafting),
        )

        if is_compressed_mask_supported():
            attn_metadata.attn_mask = AttentionMaskBuilder310.get_compressed_splitfuse_mask(self.device)
        else:
            # Build the per-step splitfuse mask here, outside the forward: the
            # forward-time get_splitfuse_mask does sync D2H/H2D copies that abort
            # an ACL graph capture. For capture-sized batches the mask content is
            # copied into a stable-address buffer so replays see fresh values.
            q_list = get_query_lens_cpu(attn_metadata).tolist()
            num_tokens = int(sum(q_list))
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu
            if seq_lens_cpu is None:
                # Capture dummy_run path: build() runs before the captured region,
                # so this D2H is legal; mask content is refreshed by real builds.
                seq_lens_cpu = attn_metadata.seq_lens.cpu()
            c_list = seq_lens_cpu[:num_reqs].tolist()
            mask_nz = AttentionMaskBuilder310.build_splitfuse_mask_nz_from_host(q_list, c_list, self.device)
            if num_tokens <= _MASK_PERSISTENT_MAX_TOKENS:
                buf = self._splitfuse_mask_bufs.get(num_tokens)
                if buf is None:
                    buf = mask_nz
                    self._splitfuse_mask_bufs[num_tokens] = buf
                else:
                    buf.copy_(mask_nz)
                mask_nz = buf
            setattr(attn_metadata, SPLITFUSE_MASK_NZ_ATTR, mask_nz)

        return attn_metadata

    def build_for_drafting(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int,
    ):
        # override build_for_drafting for passing status.
        return self.build(
            common_prefix_len=0, common_attn_metadata=common_attn_metadata, fast_build=True, is_drafting=True
        )
