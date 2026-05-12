import torch
import vllm.envs as envs_vllm
from flash_attn_v3 import flash_attn_with_kvcache as _fa3_fn  # type: ignore[import-not-found]
from vllm.v1.attention.backend import AttentionBackend  # type: ignore

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
)


class AscendFABackend(AttentionBackend):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_name() -> str:
        return "CUSTOM" if not envs_vllm.VLLM_USE_V2_MODEL_RUNNER else "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["AscendFAImpl"]:
        return AscendFAImpl

    @staticmethod
    def get_builder_cls() -> type["AscendAttentionMetadataBuilder"]:
        return AscendAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_type: str = "",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [128]


class AscendFAImpl(AscendAttentionBackendImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.sliding_window is not None:
            raise ValueError(
                "AscendFAImpl does not support sliding window attention. "
                "Please disable sliding window or use the default FIA backend."
            )
        if not self.vllm_config.model_config.enforce_eager:
            from vllm.config.compilation import CUDAGraphMode

            cudagraph_mode = self.vllm_config.compilation_config.cudagraph_mode
            if cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY:
                raise ValueError(
                    "AscendFAImpl does not support ACL graph capture with "
                    "FULL_DECODE_ONLY mode. Please set enforce_eager=True or "
                    "not set FULL_DECODE_ONLY or use the default FIA backend."
                )

    def _flash_attn_with_kvcache(
        self,
        query: torch.Tensor,
        block_table: torch.Tensor,
        actual_seq_lengths_fa: torch.Tensor,
        seq_lens_list_qa: list,
        is_causal: bool,
        max_seq_len: int,
    ):
        num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
        key_fa_blk = self.key_cache.view(  # type: ignore
            num_block, block_size, self.num_kv_heads, self.head_size
        )
        value_fa_blk = self.value_cache.view(  # type: ignore
            num_block, block_size, self.num_kv_heads, self.head_size
        )

        kv_seqlen_list = torch.tensor(seq_lens_list_qa, dtype=torch.int32).npu()

        attn_output = _fa3_fn(
            query,
            key_fa_blk,
            value_fa_blk,
            cache_seqlens=kv_seqlen_list,  # kv sequence length for each individual request (NOT cumulative)
            page_table=block_table,  #  must match the block table for the corresponding q
            cu_seqlens_q=actual_seq_lengths_fa,  # cumulative sequence length for q
            max_seqlen_q=max_seq_len,
            causal=is_causal,
            window_size=[-1, -1],
            rotary_interleaved=False,
            num_splits=1,
            softcap=0.0,
            attention_chunk=0,
            sm_margin=0,
            return_softmax_lse=False,
        )

        return attn_output

    def forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata,
        output: torch.Tensor,
    ):
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        query = query[:num_tokens]

        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefills = attn_metadata.num_prefills
        outputs = []

        if num_decodes > 0:
            outputs.append(
                self._flash_attn_with_kvcache(
                    query[:num_decode_tokens],
                    attn_metadata.block_tables[:num_decodes, :],
                    attn_metadata.query_start_loc[: num_decodes + 1],
                    attn_metadata.seq_lens_list[:num_decodes],
                    False,
                    max(attn_metadata.seq_lens_list[:num_decodes]),
                )
            )

        if num_prefills > 0:
            outputs.append(
                self._flash_attn_with_kvcache(
                    query[num_decode_tokens:],
                    attn_metadata.block_tables[num_decode_tokens:, :],
                    attn_metadata.query_start_loc[num_decodes:],
                    attn_metadata.seq_lens_list[num_decodes:],
                    True,  # enable causal for prefill
                    max(attn_metadata.seq_lens_list[num_decodes:]),
                )
            )

        if not outputs:
            raise ValueError("No attention output available")

        attn_output_fa = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
        output[:num_tokens] = attn_output_fa[:num_tokens]
        return output
