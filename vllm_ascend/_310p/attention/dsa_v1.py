# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4 attention fallback for the experimental Ascend 310P backend."""

import torch
from vllm.forward_context import get_forward_context
from vllm.logger import logger

from vllm_ascend.attention.dsa_v1 import AscendDSABackend, AscendDSAImpl, DSAMetadataList
from vllm_ascend.attention.utils import (
    maybe_save_kv_layer_to_connector,
    notify_kv_cache_written,
    wait_for_kv_layer_from_connector,
)
from vllm_ascend.memcache_comm_fence import record_attention_compute_start

from .dense_dsa import (
    apply_interleaved_rope,
    dense_causal_current_attention,
    dense_causal_swa_attention,
    dense_decode_swa_attention,
    dense_dspark_swa_attention,
    infer_blocks_per_phys_block,
    infer_blocks_per_phys_block_from_shape,
    normalize_swa_cache,
    write_paged_swa_cache,
)


def _linear_output(value):
    return value[0] if isinstance(value, tuple) else value


def _can_use_uniform_decode(metadata, num_query_tokens: int) -> bool:
    query_len = getattr(metadata, "uniform_query_len", None)
    num_reqs = len(getattr(metadata, "seq_lens_list", ()))
    return query_len is not None and num_query_tokens == query_len * num_reqs


class AscendDSAImpl310(AscendDSAImpl):
    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "Using the experimental Ascend 310P DeepSeek Sparse Attention implementation. "
            "Unsupported fused operators are being replaced incrementally."
        )
        super().__init__(*args, **kwargs)
        self._blocks_per_phys_block: int | None = None
        self._max_model_len = self.vllm_config.model_config.max_model_len
        if self._max_model_len > self.window_size:
            raise ValueError(
                "The Ascend 310P dense DeepSeek V4 fallback is exact only while "
                "max_model_len does not exceed sliding_window, got "
                f"{self._max_model_len} and {self.window_size}."
            )

    def _project_q_kv(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_a = _linear_output(self.wq_a(hidden_states))
        q = _linear_output(self.wq_b(self.q_norm(q_a))).unflatten(
            -1,
            (self.n_local_heads, self.head_dim),
        )
        q_dtype = q.dtype
        q_fp32 = q.to(torch.float32)
        q = (q_fp32 * torch.rsqrt(q_fp32.square().mean(-1, keepdim=True) + self.eps)).to(q_dtype)

        kv = self.kv_norm(_linear_output(self.wkv(hidden_states))).view(-1, 1, self.head_dim)
        q = apply_interleaved_rope(q, cos, sin, self.rope_head_dim or 0)
        kv = apply_interleaved_rope(kv, cos, sin, self.rope_head_dim or 0)
        return q, kv

    def _forward_dense_segment(
        self,
        layer_name: str,
        hidden_states: torch.Tensor,
        swa_cache: torch.Tensor,
        metadata,
        *,
        decode: bool = False,
    ) -> torch.Tensor:
        cos = metadata.cos[layer_name]
        sin = metadata.sin[layer_name]
        q, kv = self._project_q_kv(hidden_states, cos, sin)
        if self._blocks_per_phys_block is None:
            try:
                self._blocks_per_phys_block = infer_blocks_per_phys_block_from_shape(
                    metadata.block_table,
                    metadata.block_size,
                    self._max_model_len,
                )
            except ValueError:
                self._blocks_per_phys_block = infer_blocks_per_phys_block(
                    metadata.block_table,
                    metadata.slot_mapping,
                    metadata.input_positions,
                    metadata.query_start_loc,
                    metadata.block_size,
                )
        blocks_per_phys_block = self._blocks_per_phys_block
        write_paged_swa_cache(
            swa_cache,
            kv,
            metadata.slot_mapping,
            metadata.block_size,
        )
        notify_kv_cache_written(layer_name)
        record_attention_compute_start()
        if decode and metadata.dspark_swa_indices is not None:
            attention = dense_dspark_swa_attention(
                q,
                swa_cache,
                metadata.dspark_swa_indices,
                softmax_scale=self.softmax_scale,
                sinks=self.attn_sink,
            )
        elif decode and _can_use_uniform_decode(metadata, q.shape[0]):
            attention = dense_decode_swa_attention(
                q,
                swa_cache,
                metadata.block_table,
                metadata.seq_lens,
                block_size=metadata.block_size,
                blocks_per_phys_block=blocks_per_phys_block,
                window_size=self.window_size,
                softmax_scale=self.softmax_scale,
                sinks=self.attn_sink,
            )
        elif decode:
            attention = dense_causal_swa_attention(
                q,
                swa_cache,
                metadata.block_table,
                metadata.seq_lens,
                metadata.query_start_loc,
                block_size=metadata.block_size,
                blocks_per_phys_block=blocks_per_phys_block,
                window_size=self.window_size,
                softmax_scale=self.softmax_scale,
                sinks=self.attn_sink,
            )
        else:
            if metadata.fresh_prefill:
                attention = dense_causal_current_attention(
                    q,
                    kv,
                    metadata.query_start_loc,
                    window_size=self.window_size,
                    softmax_scale=self.softmax_scale,
                    sinks=self.attn_sink,
                )
            else:
                attention = dense_causal_swa_attention(
                    q,
                    swa_cache,
                    metadata.block_table,
                    metadata.seq_lens,
                    metadata.query_start_loc,
                    block_size=metadata.block_size,
                    blocks_per_phys_block=blocks_per_phys_block,
                    window_size=self.window_size,
                    softmax_scale=self.softmax_scale,
                    sinks=self.attn_sink,
                )
        attention = apply_interleaved_rope(
            attention,
            cos,
            sin,
            self.rope_head_dim or 0,
            inverse=True,
        )
        return attention

    def _forward_o_proj_310p(
        self,
        o_proj_input: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Run the DeepSeek V4 O-LoRA projection without grouped batch matmul.

        The current 8-way TP deployment has eight O-LoRA groups, so each rank
        owns exactly one group.  In that layout the grouped projection is
        exactly equivalent to flattening the local heads and applying the two
        existing parallel linear modules normally.  The generic Ascend path
        uses ``npu_transpose_batchmatmul`` and assumes a 3-D packed ``wo_a``
        weight, while the 310P conversion keeps the weight 2-D.
        """
        if self.n_local_groups != 1:
            raise ValueError(
                "The Ascend 310P DeepSeek V4 O-LoRA fallback currently "
                "requires one local O-LoRA group per TP rank, got "
                f"{self.n_local_groups}."
            )

        num_tokens = o_proj_input.shape[0]
        local_hidden = o_proj_input.reshape(num_tokens, -1)
        o_lora = _linear_output(self.wo_a(local_hidden))
        output[...] = _linear_output(self.wo_b(o_lora))
        return output

    def forward(  # type: ignore[override]
        self,
        layer_name,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...] | None,
        attn_metadata: DSAMetadataList | None,
        need_gather_q_kv: bool = False,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Preserve the already validated zero-output profiling path.
        if attn_metadata is None:
            return super().forward(
                layer_name,
                hidden_states,
                kv_cache,
                attn_metadata,
                need_gather_q_kv,
                output,
            )

        if output is None:
            raise ValueError("Output tensor must be provided for DeepSeek V4 attention")
        if kv_cache is None:
            raise ValueError("KV cache must be provided for DeepSeek V4 attention")
        if not isinstance(attn_metadata, list):
            attn_metadata = [attn_metadata]

        common_metadata = attn_metadata[0]
        swa_metadata = attn_metadata[-1]
        hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(hidden_states, need_gather_q_kv)
        actual_tokens = common_metadata.num_actual_tokens
        decode_tokens = common_metadata.num_decode_tokens
        hidden_states = hidden_states[:actual_tokens]

        forward_context = get_forward_context()
        o_proj_input = torch.zeros(
            (forward_context.num_tokens, self.n_local_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        swa_cache = normalize_swa_cache(kv_cache[1])
        wait_for_kv_layer_from_connector(layer_name)

        if common_metadata.num_prefills > 0:
            if swa_metadata.prefill is None:
                raise ValueError("Missing SWA prefill metadata")
            o_proj_input[decode_tokens:actual_tokens] = self._forward_dense_segment(
                layer_name,
                hidden_states[decode_tokens:actual_tokens],
                swa_cache,
                swa_metadata.prefill,
            )

        if common_metadata.num_decodes > 0:
            if swa_metadata.decode is None:
                raise ValueError("Missing SWA decode metadata")
            o_proj_input[:decode_tokens] = self._forward_dense_segment(
                layer_name,
                hidden_states[:decode_tokens],
                swa_cache,
                swa_metadata.decode,
                decode=True,
            )

        self._forward_o_proj_310p(o_proj_input, output)
        maybe_save_kv_layer_to_connector(layer_name, list(kv_cache))
        return output


class AscendDSABackend310(AscendDSABackend):
    @staticmethod
    def get_name() -> str:
        return "ASCEND_DSA_310P"

    @staticmethod
    def get_impl_cls() -> type[AscendDSAImpl310]:
        return AscendDSAImpl310
