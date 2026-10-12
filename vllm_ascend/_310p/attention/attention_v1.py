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
import torch_npu
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.registry import (  # type: ignore
    AttentionBackendEnum,
    register_backend,
)

from vllm_ascend._310p.attention.attention_mask import (
    AttentionMaskBuilder310,
    is_compressed_mask_supported,
)
from vllm_ascend._310p.attention.metadata_builder import (
    AscendAttentionMetadataBuilder310,
    get_query_lens_cpu,
)
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
    AscendMetadata,
)

MASK_TYPE_NORM_COMPRESS_SELF_ATTENTION = 3
MASK_TYPE_NORM_COMPRESS_PAGED_ATTENTION = 5


@register_backend(AttentionBackendEnum.CUSTOM, "ASCEND")
class AscendAttentionBackend310(AscendAttentionBackend):
    def __init__(self, *args, **kwargs):
        """
        Initializes the 310P backend and sets up the device-specific mask builder.
        """
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_type: str = "",
    ):
        """
        Determines the shape of the Key-Value (KV) cache tensor.

        The 310P hardware requires specific memory alignment for optimal performance.
        This method defines a 5D tensor shape where the head size dimension is
        split to ensure alignment to multiples of 16.

        Args:
            num_blocks (int): Number of memory blocks.
            block_size (int): Size of each block.
            num_kv_heads (int): Number of KV heads.
            head_size (int): Dimension size of each head.

        Returns:
            tuple: The specific 5D shape required by the hardware
                   (2, num_blocks, hidden_dim_aligned, block_size, 16).
        """
        # Align to a multiple of 16, as required by the 310P device.
        return (2, num_blocks, (num_kv_heads * head_size) // 16, block_size, 16)

    @staticmethod
    def get_impl_cls():
        """
        Returns the implementation class for the attention operations.
        """
        return AscendAttentionBackendImpl310

    @staticmethod
    def get_builder_cls() -> type["AscendAttentionMetadataBuilder"]:
        """
        Returns the metadata builder class specifically for 310P.
        """
        return AscendAttentionMetadataBuilder310

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [128, 64]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        # Keep the 310P default at 128 rather than selecting the smallest
        # supported size (64), which regresses paged decode performance.
        if cls.supports_block_size(default_block_size):
            return default_block_size
        return 128


class AscendAttentionBackendImpl310(AscendAttentionBackendImpl):
    """
    Implementation of attention operations (Prefill, Decode, Chunked Prefill)
    optimized for the Ascend 310P architecture.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.support_compressed_mask = is_compressed_mask_supported()

    def _flash_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        seq_len: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if not self.support_compressed_mask:
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
            return output

        torch_npu._npu_flash_attention_v3(
            query=query,
            key=key,
            value=value,
            mask=mask,
            seq_len=seq_len,
            scale_value=self.scale,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            mask_type=MASK_TYPE_NORM_COMPRESS_SELF_ATTENTION,
            out=output,
        )
        return output

    def _get_encoder_band_mask(self, num_tokens: int, head_size: int, device) -> torch.Tensor | None:
        """Additive bidirectional sliding-window mask ([T, T], fp32, -inf outside).

        EmbeddingGemma2 sliding layers use |q-k| <= text_config.sliding_window
        (HF semantics); full-attention layers are identified at runtime by
        their per-layer head size (512 vs 256) and return no mask. The mask is
        cached and rebuilt in chunks as the sequence grows.
        """
        cached = getattr(self, "_enc_band_mask", None)
        cached_len = getattr(self, "_enc_band_mask_len", 0)
        if cached is not None and num_tokens <= cached_len:
            return cached[:num_tokens, :num_tokens]
        import math

        round_len = max(num_tokens, cached_len, 512)
        round_len = 1 << max(9, int(math.ceil(math.log2(round_len))))  # >=512, pow2
        try:
            text_cfg = self.vllm_config.model_config.hf_config.text_config
            window = text_cfg.sliding_window
            layer_types = getattr(text_cfg, "layer_types", None) or []
            plc = text_cfg.per_layer_config
            full_dims = set()
            for i, lt in enumerate(layer_types):
                if lt == "full_attention":
                    d = getattr(plc[i], "head_dim", None)
                    if d is not None:
                        full_dims.add(int(d))
        except Exception:
            return None
        if window is None or head_size in full_dims:
            return None
        idx = torch.arange(round_len, dtype=torch.int32, device=device)
        band = (idx[:, None] - idx[None, :]).abs() <= window
        mask = torch.zeros(round_len, round_len, dtype=torch.float32, device=device)
        mask.masked_fill_(~band, float("-inf"))
        self._enc_band_mask = mask
        self._enc_band_mask_len = round_len
        return mask[:num_tokens, :num_tokens]

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        # The ATB SelfAttentionOperation behind torch_npu._npu_flash_attention
        # has a first-call allocator defect on 310P (aclrtAllocatorGetByStream)
        # that silently corrupts device memory on subsequent calls. Encoder-only
        # pooling attention (bidirectional, no KV cache) is inexpensive, so
        # compute it with native torch ops instead.
        num_tokens = query.shape[0]
        num_heads = self.num_heads
        num_kv_heads = self.num_kv_heads
        head_size = query.shape[-1]

        # Pooling batches pack multiple sequences into one prefill; each
        # sequence must attend only within itself (the ATB path relied on
        # seq_lens for this). Build the block-diagonal keep-mask on device
        # and combine it with the bidirectional sliding-window band.
        keep = None
        seq_lens = getattr(attn_metadata, "seq_lens", None)
        if seq_lens is not None and seq_lens.numel() > 1:
            bounds = torch.cumsum(seq_lens, 0)[:-1].to(query.device)
            seg = torch.bucketize(torch.arange(num_tokens, device=query.device), bounds, right=True)
            keep = seg[:, None] == seg[None, :]
        if keep is None:
            keep = torch.ones(num_tokens, num_tokens, dtype=torch.bool, device=query.device)
        band = self._get_encoder_band_mask(num_tokens, head_size, query.device)
        if band is not None:
            keep = keep & (band > float("-inf"))

        def _attend(qs, ks, vs, keep_mask):
            q = qs.float().permute(1, 0, 2)
            k = ks.float().permute(1, 0, 2)
            v = vs.float().permute(1, 0, 2)
            if num_kv_heads != num_heads:
                rep = num_heads // num_kv_heads
                k = k.repeat_interleave(rep, dim=0)
                v = v.repeat_interleave(rep, dim=0)
            scores = torch.bmm(q, k.transpose(1, 2)) * self.scale
            scores = scores.masked_fill(~keep_mask, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            return torch.bmm(probs, v).permute(1, 0, 2).to(query.dtype)

        if num_tokens * num_tokens <= (8192 * 8192) or seq_lens is None or seq_lens.numel() <= 1:
            result = _attend(query, key, value, keep)
            output[:num_tokens] = result
            return result

        # Extremely large totals: per-sequence attention to bound the score
        # tensor (the host sync is acceptable on this rare path).
        lens = seq_lens.long().cpu()
        results = []
        offset = 0
        for length in lens.tolist():
            results.append(
                _attend(
                    query[offset : offset + length],
                    key[offset : offset + length],
                    value[offset : offset + length],
                    keep[offset : offset + length, offset : offset + length],
                )
            )
            offset += length
        result = torch.cat(results, dim=0)
        output[:num_tokens] = result
        return result

    def forward_paged_attention(
        self,
        query: Any,
        attn_metadata: AscendMetadata,
        output: Any | None = None,
    ) -> Any:
        """
        Executes Paged Attention (typically for the decode phase).

        Ensures that the sequence length metadata is on the correct device
        before invoking the base implementation.

        Args:
            query (Any): The query tensor.
            attn_metadata (AscendMetadata): Metadata associated with the attention request.
            output (Any | None): Optional output tensor.

        Returns:
            Any: The result of the attention operation.
        """
        if attn_metadata.seq_lens.device != query.device:
            # Pageable H2D is illegal under NPU GLOBAL ACLGraph capture.
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError(
                    "310P paged attention: seq_lens must already be on-device before "
                    "ACLGraph capture; move it outside torch.npu.graph()."
                )
            attn_metadata.seq_lens = attn_metadata.seq_lens.to(
                device=query.device,
                non_blocking=True,
            )

        torch_npu._npu_paged_attention(
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            block_table=attn_metadata.block_tables,
            context_lens=attn_metadata.seq_lens,
            out=output,
        )
        return output

    def forward_prefill_310(self, query, key, value, attn_metadata, output):
        """
        Executes Flash Attention for the prefill phase on 310P.

        This method handles memory alignment padding. If the query shape implies
        padding (aligned_tokens > real_tokens), it adjusts the sequence length
        of the last request to account for the delta, ensuring the NPU operator
        processes the data correctly.

        Args:
            query, key, value: Input tensors.
            attn_metadata (AscendMetadata): Attention metadata containing masks and seq_lens.
            output: Output tensor.

        Returns:
            The output tensor after flash attention.
        """
        real_tokens = int(attn_metadata.seq_lens.sum().item())
        seq_len = attn_metadata.seq_lens
        aligned_tokens = int(query.shape[0])
        delta = aligned_tokens - real_tokens

        # Adjust sequence length if padding (alignment) was applied to the inputs
        if delta:
            seq_len = seq_len.clone()
            seq_len[-1] += delta

        mask = attn_metadata.attn_mask
        return self._flash_attention(query, key, value, mask, seq_len, output)

    def forward_chunked_prefill_310(self, query, attn_metadata, output):
        """
        Executes SplitFuse (Chunked Prefill) attention on 310P.

        This handles scenarios where the prefill is split into chunks. It prepares
        the necessary metadata (query lengths, block tables) and generates the
        specific splitfuse mask before calling the NPU operator.

        Args:
            query: The query tensor.
            attn_metadata (AscendMetadata): Metadata containing start locations and block tables.
            output: The output tensor.
        """
        num_actual_tokens = int(attn_metadata.num_actual_tokens)
        query = query[:num_actual_tokens]
        output_slice = output[:num_actual_tokens]

        # Host qLens filled in AscendAttentionMetadataBuilder310.build(); eager fallback only.
        qlens = get_query_lens_cpu(attn_metadata)
        if qlens is None:
            from vllm_ascend.ascend_forward_context import _EXTRA_CTX

            if _EXTRA_CTX.capturing:
                raise RuntimeError(
                    "310P splitfuse requires attn_metadata.query_lens_cpu during graph capture; "
                    "ensure AscendAttentionMetadataBuilder310.build() ran before forward."
                )
            qsl_cpu = attn_metadata.query_start_loc.cpu()
            qlens = qsl_cpu[1:] - qsl_cpu[:-1]

        block_table = attn_metadata.block_tables

        if attn_metadata.seq_lens.device != query.device:
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError("310P splitfuse: seq_lens must already be on-device before ACLGraph capture.")
            attn_metadata.seq_lens = attn_metadata.seq_lens.to(
                device=query.device,
                non_blocking=True,
            )

        if self.support_compressed_mask:
            # splitfuse_v2 requires fixed ND [2048, 2048]; parent build() may set FRACTAL_NZ mask.
            mask = AttentionMaskBuilder310.get_compressed_splitfuse_mask(query.device)
            torch_npu._npu_paged_attention_splitfuse_v2(
                query=query,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                mask=mask,
                block_table=block_table,
                seq_len=qlens,
                context_lens=attn_metadata.seq_lens,
                num_kv_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale_value=self.scale,
                mask_type=MASK_TYPE_NORM_COMPRESS_PAGED_ATTENTION,
                out=output_slice,
            )
            return output

        # Generate the specific mask for splitfuse
        mask = AttentionMaskBuilder310.get_splitfuse_mask(attn_metadata, query.device)
        torch_npu._npu_paged_attention_splitfuse(
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            mask=mask,
            block_table=block_table,
            seq_len=qlens,
            context_lens=attn_metadata.seq_lens,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            out=output_slice,
        )

        return output

    def forward_impl(self, query, key, value, kv_cache, attn_metadata, output):
        """
        Main dispatch method for attention operations.

        Routes the execution to Decode, Prefill, or Chunked Prefill methods
        based on the current attention state found in metadata.

        Args:
            query, key, value: Input tensors (Key/Value usually empty for decode/chunked).
            kv_cache: The KV cache structure.
            attn_metadata: Metadata determining the state (Prefill vs Decode).
            output: Tensor to write results to.

        Returns:
            The output tensor.

        Raises:
            NotImplementedError: If the attention state is not supported on 310P.
        """
        state = attn_metadata.attn_state
        # Encoder-only (pooling) layers use the native torch path; see
        # _forward_encoder_attention for why the ATB op is avoided here.
        if getattr(self, "attn_type", None) == AttentionType.ENCODER_ONLY:
            return self._forward_encoder_attention(query, key, value, attn_metadata, output)
        # Condition for PrefillNoCache: No previous tokens have been processed yet
        if state == AscendAttentionState.PrefillNoCache:
            output = self.forward_prefill_310(query, key, value, attn_metadata, output)
        # Condition for DecodeOnly: Pure decoding phase where each request generates one token
        elif state == AscendAttentionState.DecodeOnly:
            output = self.forward_paged_attention(query, attn_metadata, output)
        # ChunkedPrefill / PrefillCacheHit: chunked prefill or mixed batches.
        # SpecDecoding: MTP uniform spec verify (splitfuse on 310P).
        elif (
            state in [AscendAttentionState.ChunkedPrefill, AscendAttentionState.PrefillCacheHit]
            or state == AscendAttentionState.SpecDecoding
        ):
            output = self.forward_chunked_prefill_310(query, attn_metadata, output)
        else:
            raise NotImplementedError(f"AscendAttentionState: {state} is not supported for 310P currently.")
        return output
