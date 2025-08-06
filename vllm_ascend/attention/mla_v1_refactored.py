from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple, Type, TypeVar

import numpy as np
import torch
import torch_npu
from vllm.attention.backends.abstract import (AttentionBackend, AttentionLayer,
                                              AttentionMetadata,
                                              MLAAttentionImpl)
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from vllm.utils import cdiv, round_down

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.multistream.base import MSAttentionMetadataSplitConfig
from vllm_ascend.multistream.context import get_multistream_comm_context
from vllm_ascend.multistream.ms_split import model_input_split_v1_mla_attn
from vllm_ascend.ops.attention import vanilla_chunked_prefill_mla
from vllm_ascend.torchair.utils import npu_stream_switch, npu_wait_tensor
from vllm_ascend.attention.mla_v1 import AscendMLABackend, AscendMLAMetadata

M = TypeVar("M", bound=AscendMLAMetadata)

class AscendRefactoredMLABackend(AscendMLABackend):
    
    @staticmethod
    def get_impl_cls() -> Type["MLAAttentionImpl"]:
        return AscendMLAImplRefactored

class AscendMLAImplRefactored(MLAAttentionImpl):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[dict[str, Any]],
        logits_soft_cap: Optional[float],
        attn_type: str,
        kv_sharing_target_layer_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        # MLA Args
        self.q_lora_rank = kwargs['q_lora_rank']
        self.kv_lora_rank = kwargs['kv_lora_rank']
        self.qk_nope_head_dim = kwargs['qk_nope_head_dim']
        self.qk_rope_head_dim = kwargs['qk_rope_head_dim']
        self.qk_head_dim = kwargs['qk_head_dim']
        self.v_head_dim = kwargs['v_head_dim']
        self.rotary_emb = kwargs['rotary_emb']
        self.q_proj = kwargs['q_proj']
        self.kv_b_proj = kwargs['kv_b_proj']
        self.o_proj = kwargs['o_proj']
        self.kv_a_proj_with_mqa = kwargs.get('kv_a_proj_with_mqa', None)
        self.kv_a_layernorm = kwargs.get('kv_a_layernorm', None)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        ascend_config = get_ascend_config()
        self.torchair_graph_enabled = ascend_config.torchair_graph_config.enabled
        self.enable_kv_nz = ascend_config.torchair_graph_config.enable_kv_nz
        self.enable_multistream_mla = \
            ascend_config.torchair_graph_config.enable_multistream_mla

        # Adapt torch air graph mode with spec decoding.
        speculative_config = get_current_vllm_config().speculative_config
        if speculative_config is not None:
            self.spec_token_num = speculative_config.num_speculative_tokens
            assert self.spec_token_num > 0

    def _v_up_proj_and_o_proj(self, x):
        # Convert from (B, N, L) to (N, B, L)
        x = x.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)
        # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
        x = torch.bmm(x, self.W_UV)
        # Convert from (N, B, V) to (B, N * V)
        x = x.transpose(0, 1).reshape(-1, self.num_heads * self.v_head_dim)
        return self.o_proj(x)[0]

    # Return `ql_nope`, `q_pe`
    def _q_proj_and_k_up_proj(self, x):
        q_nope, q_pe = self.q_proj(x)[0]\
            .view(-1, self.num_heads, self.qk_head_dim)\
            .split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # Convert from (B, N, P) to (N, B, P)
        q_nope = q_nope.transpose(0, 1)
        # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
        ql_nope = torch.bmm(q_nope, self.W_UK_T)
        # Convert from (N, B, L) to (B, N, L)
        return ql_nope.transpose(0, 1), q_pe

    def process_weights_after_loading(self, act_dtype: torch.dtype):

        def get_layer_weight(layer):
            WEIGHT_NAMES = ("weight", "qweight", "weight_packed")
            for attr in WEIGHT_NAMES:
                if hasattr(layer, attr):
                    return getattr(layer, attr)
            raise AttributeError(
                f"Layer '{layer}' has no recognized weight attribute:"
                f" {WEIGHT_NAMES}.")

        def get_and_maybe_dequant_weights(layer: LinearBase):
            if not isinstance(layer.quant_method, UnquantizedLinearMethod):
                # NOTE: This should only be used offline, since it's O(N^3)
                eye = torch.eye(layer.input_size_per_partition,
                                dtype=act_dtype,
                                device=get_layer_weight(layer).device)
                dequant_weights = layer.quant_method.apply(layer,
                                                           eye,
                                                           bias=None)
                del eye
                # standardize to (output, input)
                return dequant_weights.T
            return layer.weight

        # we currently do not have quantized bmm's which are needed for
        # `W_UV` and `W_UK_T`, we we just store fp16/bf16 copies and perform
        # the bmm's in 16-bit, the extra memory overhead of this is fairly low
        kv_b_proj_weight = get_and_maybe_dequant_weights(self.kv_b_proj).T
        assert kv_b_proj_weight.shape == (
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim)), (
                f"{kv_b_proj_weight.shape=}, "
                f"{self.kv_lora_rank=}, "
                f"{self.num_heads=}, "
                f"{self.qk_nope_head_dim=}, "
                f"{self.v_head_dim=}")
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        W_UK, W_UV = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Convert from (L, N, V) to (N, L, V)
        self.W_UV = W_UV.transpose(0, 1).contiguous()
        # Convert from (L, N, P) to (N, P, L)
        self.W_UK_T = W_UK.permute(1, 2, 0).contiguous()

        # Waiting for BMM NZ support
        # self.W_UV.data = torch_npu.npu_format_cast(self.W_UV.data, 29)
        # self.W_UK_T.data = torch_npu.npu_format_cast(self.W_UK_T.data, 29)

    def _compute_prefill_context(
        self,
        query: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        rope_dim: int,
        attn_metadata: AscendMLAMetadata,
        prefix_output: torch.Tensor,
        prefix_lse: torch.Tensor,
    ):
        prefill_metadata = attn_metadata.prefill
        if prefill_metadata is None or prefill_metadata.chunked_context is None:
            return prefix_output, prefix_lse

        iters = len(prefill_metadata.chunked_context.seq_tot)
        q_pe = query[..., self.qk_nope_head_dim:]
        q_nope = query[..., :self.qk_nope_head_dim]

        seq_len1 = torch.tensor(prefill_metadata.query_lens, dtype=torch.int32)
        latent_kv_dim = kv_c_and_k_pe_cache[0].size(-1) - rope_dim
        cache_kv_c = kv_c_and_k_pe_cache[0]
        cache_k_pe = kv_c_and_k_pe_cache[1]
        for i in range(iters):
            toks = prefill_metadata.chunked_context.seq_tot[i]

            seq_len2 = prefill_metadata.chunked_context.chunk_seq_lens[i]
            seq_len = torch.stack([seq_len1, seq_len2])
            kv_c_normed = torch.empty(toks,
                                      kv_c_and_k_pe_cache[0].size(2),
                                      latent_kv_dim,
                                      dtype=query.dtype,
                                      device=query.device)
            k_pe = torch.empty(toks,
                               kv_c_and_k_pe_cache[1].size(2),
                               rope_dim,
                               dtype=query.dtype,
                               device=query.device)

            torch_npu.atb.npu_paged_cache_load(
                cache_kv_c,
                cache_k_pe,
                prefill_metadata.block_table,
                seq_len2.to(query.device),
                seq_starts=prefill_metadata.chunked_context.starts[i],
                key=kv_c_normed,
                value=k_pe,
            )

            kv_c_normed = kv_c_normed.squeeze()
            kv_nope = self.kv_b_proj(kv_c_normed)[0].view( \
                -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = kv_nope\
                .split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k_pe = k_pe.expand((*k_nope.shape[:-1], -1))
            mask = torch.triu(
                torch.ones(512, 512, device=query.device, dtype=query.dtype),
                1)
            torch_npu.atb.npu_ring_mla(
                q_nope=q_nope,
                q_rope=q_pe,
                k_nope=k_nope,
                k_rope=k_pe,
                value=v,
                mask=mask,
                seqlen=seq_len,
                head_num=self.num_heads,
                kv_head_num=self.num_heads,
                pre_out=prefix_output,
                prev_lse=prefix_lse,
                qk_scale=self.scale,
                kernel_type="kernel_type_high_precision",
                mask_type="no_mask",
                input_layout="type_bsnd",
                calc_type="calc_type_default",
                output=prefix_output,
                softmax_lse=prefix_lse)
        return prefix_output, prefix_lse

    def _forward_prefill(
        self,
        query: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: AscendMLAMetadata,
    ) -> torch.Tensor:
        assert attn_metadata.prefill is not None
        num_tokens = query.size(0)
        attn_output = torch.empty(num_tokens,
                                  self.num_heads,
                                  self.v_head_dim,
                                  dtype=query.dtype,
                                  device=query.device)
        k_nope, value = self.kv_b_proj(kv_c_normed)[0].view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).split(
                [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_pe = k_pe.expand((*k_nope.shape[:-1], -1))
        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            key = torch.cat((k_nope, k_pe), dim=-1)
            torch_npu._npu_flash_attention(
                query=query,
                key=key,
                value=value,
                mask=attn_metadata.attn_mask,
                seq_len=attn_metadata.prefill.context_lens,
                scale_value=self.scale,
                num_heads=self.num_heads,
                num_kv_heads=self.num_heads,
                out=attn_output)
        else:
            attn_lse = torch.empty(self.num_heads,
                                    num_tokens,
                                    dtype=torch.float32,
                                    device=query.device)
            q_pe = query[..., self.qk_nope_head_dim:]
            q_nope = query[..., :self.qk_nope_head_dim]
            mask = torch.triu(
                torch.ones(512, 512, device=query.device, dtype=query.dtype),
                1)  # 512: mask only support 512
            if attn_metadata.num_prefills > 1:
                mask = mask.unsqueeze(0).repeat(attn_metadata.num_prefills, 1,
                                                1)
            torch_npu.atb.npu_ring_mla(
                q_nope=q_nope,
                q_rope=q_pe,
                k_nope=k_nope,
                k_rope=k_pe,
                value=value,
                mask=mask,
                seqlen=torch.tensor(attn_metadata.prefill.query_lens,
                                    dtype=torch.int32),
                head_num=self.num_heads,
                kv_head_num=self.num_heads,
                pre_out=None,
                prev_lse=None,
                qk_scale=self.scale,
                kernel_type="kernel_type_high_precision",
                mask_type="mask_type_triu",
                input_layout="type_bsnd",
                calc_type="calc_type_first_ring",
                output=attn_output,
                softmax_lse=attn_lse)
            attn_output, attn_lse = self._compute_prefill_context( \
                query, kv_c_and_k_pe_cache, self.qk_rope_head_dim, attn_metadata, attn_output, attn_lse)

        attn_output = attn_output.reshape(
            [num_tokens, self.num_heads * self.v_head_dim])

        current_ms_metadata = get_multistream_comm_context()
        if current_ms_metadata is None:
            return self.o_proj(attn_output)[0]
        else:
            current_ms_metadata.before_comm_event.record()
            with torch.npu.stream(current_ms_metadata.comm_stream):
                current_ms_metadata.before_comm_event.wait()
                return self.o_proj(attn_output)[0]

    def exec_kv(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Tuple,
        slots: torch.Tensor,
    ):

        B = hidden_states.shape[0]
        N = self.num_kv_heads
        S = 1
        kv = self.kv_a_proj_with_mqa(hidden_states)[0]
        # npu_kv_rmsnorm_rope_cache needs [B, N, S, D]
        kv = kv.view(B, N, S, self.kv_lora_rank + self.qk_rope_head_dim)
        cache_mode = "PA_NZ" if self.enable_kv_nz else "PA"
        with npu_stream_switch("mla_secondary",
                               0,
                               enabled=self.enable_multistream_mla):
            k_pe, k_nope, _, _ = torch_npu.npu_kv_rmsnorm_rope_cache(
                kv,
                self.kv_a_layernorm.weight,
                cos,
                sin,
                slots.to(torch.int64),
                kv_cache[1],
                kv_cache[0],
                epsilon=self.kv_a_layernorm.variance_epsilon,
                cache_mode=cache_mode,
            )
        return k_pe, k_nope

    def exec_kv_prefill(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Tuple,
        slots: torch.Tensor,
    ):
        B = hidden_states.shape[0]
        N = self.num_kv_heads
        S = 1
        kv = self.kv_a_proj_with_mqa(hidden_states)[0]
        # npu_kv_rmsnorm_rope_cache needs [B, N, S, D]
        kv = kv.view(B, N, S, self.kv_lora_rank + self.qk_rope_head_dim)
        cache_mode = "PA_BLK_NZ" if self.enable_kv_nz else "PA"
        _, _, k_pe, k_nope = torch_npu.npu_kv_rmsnorm_rope_cache(
            kv,
            self.kv_a_layernorm.weight,
            cos,
            sin,
            slots.to(torch.int64),
            kv_cache[1],
            kv_cache[0],
            epsilon=self.kv_a_layernorm.variance_epsilon,
            cache_mode=cache_mode,
            is_output_kv=True,
        )
        return k_pe, k_nope
    
    def select_sin_cos(
        self,
        dtype: torch.dtype,
        positions: torch.Tensor,
    ):
        if self.rotary_emb.cos_cached.dtype != dtype:
            self.rotary_emb.cos_cached = self.rotary_emb.cos_cached.to(
                dtype=dtype)
            self.rotary_emb.sin_cached = self.rotary_emb.sin_cached.to(
                dtype=dtype)
        cos = self.rotary_emb.cos_cached[positions]
        sin = self.rotary_emb.sin_cached[positions]
        return cos[:, None, None, :], sin[:, None, None, :]

    def rope_single(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, N, D = x.shape
        S = 1
        x = x.view(B, N, S, D)
        x = torch_npu.npu_interleave_rope(x, cos, sin)
        return x.view(B, N, D)

    def _forward_decode(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        block_size: int,
        attn_metadata: AscendMLAMetadata,
    ) -> torch.Tensor:
        decode_meta = attn_metadata.decode
        assert decode_meta is not None
        num_tokens = ql_nope.size(0)
        attn_output = torch.empty(
                [num_tokens, self.num_heads, self.kv_lora_rank],
                dtype=ql_nope.dtype,
                device=ql_nope.device)
        # TorchAir's shape is [bs, num_heads_per_rank, q_seq_len, dim]
        if attn_metadata.attn_state == AscendAttentionState.SpecDecoding:
            assert num_tokens % self.spec_token_num == 0
            q_nope = ql_nope.view(num_tokens // (self.spec_token_num + 1),
                                    self.spec_token_num + 1, self.num_heads,
                                    -1)
            q_pe = q_pe.view(num_tokens // (self.spec_token_num + 1),
                                self.spec_token_num + 1, self.num_heads, -1)
            if not self.enable_kv_nz:
                q_nope = q_nope.transpose(1, 2).contiguous()
                q_pe = q_pe.transpose(1, 2).contiguous()
            sparse_mode = 3
            spec_attn_mask = attn_metadata.decode.attn_mask  # type:ignore
        else:
            if self.enable_kv_nz:
                q_nope = ql_nope.view(num_tokens, 1, self.num_heads, -1)
                q_pe = q_pe.view(num_tokens, 1, self.num_heads, -1)
            else:
                q_nope = ql_nope.view(num_tokens, self.num_heads, 1, -1)
                q_pe = q_pe.view(num_tokens, self.num_heads, 1, -1)
            sparse_mode = 0
            spec_attn_mask = None
        # shape of knope/k_pe for FIA should be:
        # [num_blocks, num_kv_heads, block_size, self.kv_lora_rank/self.qk_rope_head_dim]
        if self.enable_kv_nz:
            k_nope = k_nope.view(-1, self.num_kv_heads,
                                    self.kv_lora_rank // 16, block_size, 16)
            k_pe = k_pe.view(-1, self.num_kv_heads,
                                self.qk_rope_head_dim // 16, block_size, 16)
            input_layout = "BSND"
        else:
            k_nope = k_nope.view(-1, self.num_kv_heads, block_size,
                                    self.kv_lora_rank)
            k_pe = k_pe.view(-1, self.num_kv_heads, block_size,
                                self.qk_rope_head_dim)
            input_layout = "BNSD"

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            q_nope,
            k_nope,
            k_nope,
            query_rope=q_pe,
            key_rope=k_pe,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout=input_layout,
            atten_mask=spec_attn_mask,
            sparse_mode=sparse_mode,
            scale=self.scale,
            antiquant_mode=0,
            antiquant_scale=None,
            block_table=decode_meta.block_table,
            block_size=block_size,
            actual_seq_lengths_kv=decode_meta.seq_lens_list,
        )
        current_ms_metadata = get_multistream_comm_context()
        if current_ms_metadata is None:
            return self._v_up_proj_and_o_proj(attn_output)
        else:
            current_ms_metadata.before_comm_event.record()
            with torch.npu.stream(current_ms_metadata.comm_stream):
                current_ms_metadata.before_comm_event.wait()
                return self._v_up_proj_and_o_proj(attn_output)

    def forward(
        self,
        layer: AttentionLayer,
        q_c: torch.Tensor,  # query in unified attn
        hidden_states: torch.Tensor,  # key in unified attn
        k_pe: torch.Tensor, # value in unified attn
        kv_cache: torch.Tensor,
        attn_metadata: M,
        output: Optional[torch.Tensor] = None,
        enable_multistream_mla: bool = False,
        ckq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:
            # Profiling run.
            return output
        assert attn_metadata.num_decodes is not None and \
        attn_metadata.num_prefills is not None and \
        attn_metadata.num_decode_tokens is not None
        num_actual_tokens = attn_metadata.num_actual_tokens
        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0
        self.running_in_graph = self.torchair_graph_enabled and not has_prefill
        num_decode_tokens = attn_metadata.num_decode_tokens
        if has_decode:
            if self.running_in_graph:
                num_decode_tokens = hidden_states.size(0)
            # MLA Preprocess for decoding
            decode_hs = hidden_states[:num_decode_tokens]
            decode_q_c = q_c[:num_decode_tokens]
            cos, sin = self.select_sin_cos(decode_q_c.dtype, attn_metadata.decode.input_positions)
            # Without explicitly controlling the order, IndexByTensor operations
            # would be placed after `matmul W_KV_T` hindering the overlapping of
            # KvRmsNormRopeCache and SingleRope.
            npu_wait_tensor(decode_q_c,
                            cos,
                            enabled=self.enable_multistream_mla)
            npu_wait_tensor(decode_q_c,
                            sin,
                            enabled=self.enable_multistream_mla)
            decode_ql_nope, decode_q_pe = \
                self._q_proj_and_k_up_proj(decode_q_c)
            decode_slots = attn_metadata.slot_mapping[:num_decode_tokens]
            with npu_stream_switch("mla_secondary",
                                    0,
                                    enabled=enable_multistream_mla):
                npu_wait_tensor(hidden_states,
                                ckq,
                                enabled=enable_multistream_mla)
                decode_k_pe, decode_k_nope = self.exec_kv(
                    decode_hs, cos, sin, kv_cache,
                    decode_slots)
                npu_wait_tensor(decode_q_pe,
                                decode_k_pe,
                                enabled=enable_multistream_mla)
                decode_q_pe = self.rope_single(decode_q_pe, cos, sin)

            output_decode = self._forward_decode(decode_ql_nope,
                                                    decode_q_pe,
                                                    decode_k_nope,
                                                    decode_k_pe, kv_cache[0].shape[1],
                                                    attn_metadata)
            current_ms_metadata = get_multistream_comm_context()
            if current_ms_metadata is not None:
                with torch.npu.stream(current_ms_metadata.comm_stream):
                    output[:num_decode_tokens] = output_decode
                    current_ms_metadata.after_comm_event.record()
            else:
                output[:num_decode_tokens] = output_decode

        if has_prefill:
            prefill_hs = hidden_states[num_decode_tokens:num_actual_tokens]
            prefill_q_c = q_c[num_decode_tokens:num_actual_tokens]
            prefill_q = self.q_proj(prefill_q_c)[0]\
                .view(-1, self.num_heads, self.qk_head_dim)
            prefill_q_pe = prefill_q[..., self.qk_nope_head_dim:]
            prefill_q_nope = prefill_q[..., :self.qk_nope_head_dim]
            num_tokens = prefill_q_c.shape[0]
            cos, sin = self.select_sin_cos(prefill_q_pe.dtype, attn_metadata.prefill.input_positions)
            prefill_slots = attn_metadata.slot_mapping[num_decode_tokens:num_actual_tokens]
            prefill_q_pe = self.rope_single(prefill_q_pe, cos, sin)
            prefill_k_pe, prefill_k_c_normed = self.exec_kv_prefill(
                prefill_hs, cos, sin, kv_cache,
                prefill_slots)
            prefill_k_pe = prefill_k_pe.view(num_tokens, self.num_kv_heads,
                                                -1)
            prefill_q = torch.cat([prefill_q_nope, prefill_q_pe], dim=-1)
            # FIX: aicore move should be also placed on the comm stream in dbo,
            # otherwise it may affect the accuracy
            # TODO: use an elegant way to overlap
            output_prefill = self._forward_prefill(prefill_q,
                                                   prefill_k_c_normed,
                                                   prefill_k_pe, kv_cache,
                                                   attn_metadata)
            current_ms_metadata = get_multistream_comm_context()
            if current_ms_metadata is not None:
                with torch.npu.stream(current_ms_metadata.comm_stream):
                    output[num_decode_tokens:num_actual_tokens] = output_prefill
                    current_ms_metadata.after_comm_event.record()
            else:
                output[num_decode_tokens:num_actual_tokens] = output_prefill

        return output
