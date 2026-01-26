#
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

import torch
import torch_npu

from vllm_ascend._310p.attention.attention_mask import AttentionMaskBuilder, build_splitfuse_attn_mask_310p
from vllm_ascend._310p.attention.metadata_builder import AscendAttentionMetadataBuilder310P
from vllm_ascend.attention.attention_v1 import AscendAttentionBackend as _BaseBackend
from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl as _BaseImpl
from vllm_ascend.attention.attention_v1 import AscendAttentionMetadataBuilder, AscendAttentionState, AscendMetadata
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, nd_to_nz_2d


class AscendAttentionBackend310(_BaseBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_mask_builder = AttentionMaskBuilder(self.device)

    @staticmethod
    def get_kv_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int, head_size: int):
        # Align to a multiple of 16, as required by the 310P device.
        return (2, num_blocks, (num_kv_heads * head_size) // 16, block_size, 16)

    @staticmethod
    def get_impl_cls():
        return AscendAttentionBackendImpl310

    @staticmethod
    def get_builder_cls() -> type["AscendAttentionMetadataBuilder"]:
        return AscendAttentionMetadataBuilder310P


class AscendAttentionBackendImpl310(_BaseImpl):
    def forward_paged_attention(
        self,
        query: Any,
        attn_metadata: AscendMetadata,
        output: Any | None = None,
    ) -> Any:
        if attn_metadata.seq_lens.device != query.device:
            attn_metadata.seq_lens = attn_metadata.seq_lens.to(
                device=query.device,
                non_blocking=True,
            )
        return super().forward_paged_attention(query, attn_metadata, output)

    def _forward_prefill_310p_fallback(self, query, key, value, attn_metadata, output):
        real_tokens = int(attn_metadata.seq_lens.sum().item())

        seq_len = attn_metadata.seq_lens
        if seq_len.dtype != torch.int32:
            seq_len = seq_len.to(torch.int32)

        aligned_tokens = int(query.shape[0])
        delta = aligned_tokens - real_tokens
        if delta:
            seq_len = seq_len.clone()
            seq_len[-1] += delta

        mask = attn_metadata.attn_mask
        if mask is not None and mask.dim() == 2:
            max_len = int(seq_len.max().item())
            aligned_len = ((max_len + 15) // 16) * 16

            mask2d = mask[:aligned_len, :aligned_len].contiguous()
            mask2d = mask2d.to(torch.float16)
            mask_nz = nd_to_nz_2d(mask2d).contiguous()

            bsz = int(seq_len.numel())
            if bsz > 1:
                mask_nz = mask_nz.repeat(bsz, 1, 1, 1).contiguous()

            mask = torch_npu.npu_format_cast(mask_nz, ACL_FORMAT_FRACTAL_NZ)

        torch_npu._npu_flash_attention(
            query=query,
            key=key,
            value=value,
            mask=mask,
            seq_len=seq_len,
            scale_value=self.scale,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            out=output,
        )

        out_real = output[:real_tokens, :, :]
        return out_real

    def _forward_chunked_prefill_310p(self, query, attn_metadata, output):
        assert attn_metadata is not None

        if query.dtype == torch.float32:
            query = query.to(torch.float16)

        qsl_cpu = attn_metadata.query_start_loc.detach().to("cpu", dtype=torch.int32)
        qlens = (qsl_cpu[1:] - qsl_cpu[:-1]).to(torch.int32)

        context_lens = attn_metadata.seq_lens
        if context_lens.dtype != torch.int32:
            context_lens = context_lens.to(torch.int32)

        block_table = attn_metadata.block_tables.detach()
        if block_table.dtype != torch.int32:
            block_table = block_table.to(torch.int32)

        if not hasattr(self, "_sf_full_mask_cache"):
            self._sf_full_mask_cache = None
            self._sf_full_mask_cache_len = 0

        mask, self._sf_full_mask_cache, self._sf_full_mask_cache_len = build_splitfuse_attn_mask_310p(
            attn_metadata,
            query.device,
            full_mask_cache=self._sf_full_mask_cache,
            full_mask_cache_len=int(self._sf_full_mask_cache_len),
        )

        if qlens.device.type != "cpu":
            qlens = qlens.to("cpu")
        if context_lens.device != query.device:
            context_lens = context_lens.to(query.device, non_blocking=True)

        torch_npu._npu_paged_attention_splitfuse(
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            mask=mask,
            block_table=block_table,
            seq_len=qlens,
            context_lens=context_lens,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            out=output,
        )

    def forward_impl(self, query, key, value, kv_cache, attn_metadata, output):
        state = attn_metadata.attn_state

        if state == AscendAttentionState.DecodeOnly:
            return self.forward_paged_attention(query, attn_metadata, output)

        if state == AscendAttentionState.PrefillNoCache:
            num_tokens = query.shape[0]
            q = query[:num_tokens]
            k = key[:num_tokens]
            v = value[:num_tokens]
            out = self._forward_prefill_310p_fallback(q, k, v, attn_metadata, output)
            output[:num_tokens] = out
            return output

        if state == AscendAttentionState.ChunkedPrefill:
            self._forward_chunked_prefill_310p(query, attn_metadata, output)
            return output

        raise NotImplementedError(
            f"{self.__class__.__name__}.forward_impl: 310P only supports "
            f"{AscendAttentionState.DecodeOnly.name}, "
            f"{AscendAttentionState.PrefillNoCache.name}, "
            f"{AscendAttentionState.ChunkedPrefill.name}, "
            f"got {state!r}."
        )
