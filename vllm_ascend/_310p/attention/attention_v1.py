import torch
import torch_npu

from vllm_ascend._310p.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend._310p.attention.metadata_builder import \
    AscendAttentionMetadataBuilder310P
from vllm_ascend.attention.attention_v1 import \
    AscendAttentionBackend as _BaseBackend
from vllm_ascend.attention.attention_v1 import \
    AscendAttentionBackendImpl as _BaseImpl
from vllm_ascend.attention.attention_v1 import (AscendAttentionMetadataBuilder,
                                                AscendAttentionState)
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, aligned_16, nd_to_nz_2d


class AscendAttentionBackend310(_BaseBackend):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_mask_builder = AttentionMaskBuilder(self.device)

    @staticmethod
    def get_kv_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int,
                           head_size: int):
        return (2, num_blocks, (num_kv_heads * head_size) // 16, block_size,
                16)

    @staticmethod
    def get_impl_cls():
        return AscendAttentionBackendImpl310

    @staticmethod
    def get_builder_cls() -> type["AscendAttentionMetadataBuilder"]:
        return AscendAttentionMetadataBuilder310P


class AscendMLABackend310(AscendAttentionBackend310):
    pass


class AscendSFABackend310(AscendAttentionBackend310):
    pass


class AscendAttentionBackendImpl310(_BaseImpl):

    def forward_paged_attention(self, query, attn_metadata, output):
        if attn_metadata.seq_lens.device != query.device:
            attn_metadata.seq_lens = attn_metadata.seq_lens.to(
                device=query.device, non_blocking=True)
        return super().forward_paged_attention(query, attn_metadata, output)

    def _forward_prefill_310p_fallback(self, query, key, value, attn_metadata,
                                       output):
        real_tokens = int(attn_metadata.seq_lens.sum().item())

        query, key, value, output = (aligned_16(t)
                                     for t in (query, key, value, output))

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

    def forward_impl(self, query, key, value, kv_cache, attn_metadata, output):
        if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            output = self.forward_paged_attention(query, attn_metadata, output)

        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            num_tokens = query.shape[0]
            q = query[:num_tokens]
            k = key[:num_tokens]
            v = value[:num_tokens]
            out = self._forward_prefill_310p_fallback(q, k, v, attn_metadata,
                                                      output)
            output[:num_tokens] = out

        return output
