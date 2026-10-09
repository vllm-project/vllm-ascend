import torch
from flash_attn_npu_v3 import flash_attn_with_kvcache as _fa3_fn  # type: ignore[import-not-found]
from vllm.v1.attention.backend import AttentionBackend  # type: ignore

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
)


def _as_fa3_paged_cache(cache: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Expose strided physical pages to FA3 without copying the KV cache.

    FA3 addresses pages as contiguous storage, ignoring tensor block strides.
    Ascend can interleave K/V pages, or pad pages, so expose the intervening
    storage as unused pages and scale the page table by the physical stride.
    The final view ends at the last valid page, not the end of its padding.
    """
    num_blocks, block_size, num_heads, head_size = cache.shape
    page_elements = block_size * num_heads * head_size
    inner_strides = (num_heads * head_size, head_size, 1)
    page_step, remainder = divmod(cache.stride(0), page_elements)
    if remainder or page_step < 1 or cache.stride()[1:] != inner_strides:
        raise ValueError("FA3 requires dense inner pages and a block stride divisible by the page size")
    if page_step == 1:
        return cache, page_step
    shape = ((num_blocks - 1) * page_step + 1, block_size, num_heads, head_size)
    return cache.as_strided(shape, (page_elements, *inner_strides)), page_step


class AscendFABackend(AttentionBackend):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

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
        actual_seq_lengths: torch.Tensor,
        seq_lens: torch.Tensor,
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

        key_fa_blk, key_page_step = _as_fa3_paged_cache(key_fa_blk)
        value_fa_blk, value_page_step = _as_fa3_paged_cache(value_fa_blk)
        if key_page_step != value_page_step:
            raise ValueError("FA3 requires matching K and V physical page strides")
        if key_page_step != 1:
            block_table = block_table * key_page_step

        attn_output = _fa3_fn(
            query,
            key_fa_blk,
            value_fa_blk,
            cache_seqlens=seq_lens,  # kv sequence length for each individual request (NOT cumulative)
            page_table=block_table,  # must match the block table for the corresponding q
            cu_seqlens_q=actual_seq_lengths,  # cumulative sequence length for q
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
                    attn_metadata.seq_lens[:num_decodes].npu(),
                    False,
                    max(attn_metadata.seq_lens[:num_decodes]),
                )
            )

        if num_prefills > 0:
            outputs.append(
                self._flash_attn_with_kvcache(
                    query[num_decode_tokens:],
                    attn_metadata.block_tables[num_decode_tokens:, :],
                    attn_metadata.query_start_loc[num_decodes:],
                    attn_metadata.seq_lens[num_decodes:].npu(),
                    True,  # enable causal for prefill
                    max(attn_metadata.seq_lens[num_decodes:]),
                )
            )

        if not outputs:
            raise ValueError("No attention output available")

        attn_output_fa = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
        output[:num_tokens] = attn_output_fa[:num_tokens]
        return output
