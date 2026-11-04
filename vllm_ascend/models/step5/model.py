# SPDX-License-Identifier: Apache-2.0
"""Ascend Step5 model for vLLM 0.33 / MRV2.

The model skeleton (projections, MoE wiring, weight loading, MTP metadata,
the decoder-layer residual flow) is inherited from the upstream
implementation shipped in vLLM 0.33 (``vllm.models.step5.nvidia.model``);
only the compute that upstream pins to CUDA is swapped:

* DSA (full-attention) layers run ``AscendStep5DSAAttention``: the fused
  Triton QKNorm+RoPE / indexer norm+rope kernels proven on Ascend MRV1, the
  aclnn region scorer and block-sparse attention operators, and CSA state
  in the upstream deterministic layout -- ``Step5CSATailRing`` per request
  plus a per-block FP8 ``Step5SparseSummaryCacheLayer`` (both layer classes
  are reused verbatim from upstream; only the update kernels are Ascend).
* Sliding-window (dense) layers run ``AscendStep5Attention`` on the
  platform-selected Ascend attention backend with the same fused
  QKNorm+RoPE kernel.
* The decoder layer forwards through the upstream residual flow; Step5RMSNorm
  falls back to its eager path on NPU automatically.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from copy import copy
from typing import Any

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    maybe_prefix,
)
from vllm.models.step5.nvidia.config import (
    Step5SparseConfig,
    checkpoint_has_step5_sparse_config,
    get_step5_sparse_config,
)
from vllm.models.step5.nvidia.layernorm import (
    Step5IndexerLayerNorm,
    get_step5_norm_dtype,
    make_step5_rms_norm,
)
from vllm.models.step5.nvidia.model import (
    Step5Attention,
    Step5DecoderLayer,
    Step5ForCausalLM,
    Step5FusedQKVIndexerLinear,
    Step5MLP,
    Step5Model,
    Step5MoE,
    Step5SparseIndexerIndexTPLinear,
    _get_step5_layer_types,
    _get_step5_moe_layer_indices,
    _is_step5_full_attention_layer,
    _per_layer_value,
    _set_step5_moe_protocol_metadata,
    _step5_use_sequence_parallel,
    _validate_step5_dsa_parallel_geometry,
    extract_layer_index,
    make_layers,
)
from vllm.models.step5.nvidia.csa_tail_ring import Step5CSATailRing
from vllm.models.step5.nvidia.sparse_summary_cache import (
    Step5SparseSummaryCacheLayer,
)
from vllm.distributed import (
    get_dp_group,
    get_pp_group,
    get_tp_group,
)
from vllm.sequence import IntermediateTensors
from vllm.model_executor.models.utils import make_empty_intermediate_tensors_factory
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.models.step5.dsa_attention import (
    AscendStep5DSABackend,
    AscendStep5DSAImpl,
    AscendStep5SparseSummaryCacheLayer,
)
from vllm_ascend.models.step5.fused_qknorm_rope import (
    fused_indexer_qkz_norm_rope,
    fused_qknorm_rope,
)

logger = init_logger(__name__)


def _kv_cache_torch_dtype(cache_config, vllm_config) -> torch.dtype:
    cache_dtype = cache_config.cache_dtype if cache_config is not None else "auto"
    if cache_dtype in ("auto", None, ""):
        return vllm_config.model_config.dtype
    if cache_dtype in ("bfloat16", "bf16"):
        return torch.bfloat16
    if cache_dtype in ("float16", "fp16", "half"):
        return torch.float16
    raise ValueError(f"Unsupported step5-ascend KV cache dtype: {cache_dtype}")


class AscendStep5DSAAttention(nn.Module, AttentionLayerBase):
    """Step5 DSA layer on Ascend over the upstream ring/summary layout."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float | list[float] | None = 10000,
        cache_config: Any | None = None,
        quant_config: Any | None = None,
        rope_scaling: dict[str, Any] | None = None,
        sliding_window: int | None = None,
        prefix: str = "",
        use_head_wise_attn_gate: bool = False,
        layer_types: list | None = None,
        use_rope_layers: list | None = None,
        yarn_only_types: list | None = None,
        swa_num_attention_heads: int | None = None,
        partial_rotary_factor: float = 1.0,
        zero_centered: bool = True,
        vllm_config: VllmConfig | None = None,
        sparse_config: Step5SparseConfig | None = None,
        norm_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert sparse_config is not None and vllm_config is not None
        # Both are sliding-layer options the shared kwargs carry; DSA layers
        # are full-attention so they are accepted and ignored.
        del sliding_window, swa_num_attention_heads
        if qkv_bias:
            raise ValueError(
                "Step5 DSA layers fuse qkv with the sparse indexer into one "
                "bias-free GEMM, but attention_bias is enabled."
            )
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        self.prefix = prefix
        self.layer_idx = extract_layer_index(prefix)
        tp_size = get_tensor_model_parallel_world_size()
        self.tp_size = tp_size

        if isinstance(rope_theta, list):
            rope_theta = _per_layer_value(
                rope_theta, self.layer_idx, name="rope_theta", default=None
            )

        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.partial_rotary_factor = float(partial_rotary_factor)
        self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.zero_centered = zero_centered
        self.use_head_wise_attn_gate = use_head_wise_attn_gate
        self.use_rope = bool(
            _per_layer_value(
                use_rope_layers, self.layer_idx, name="use_rope_layers", default=True
            )
        )
        if norm_dtype is None:
            norm_dtype = get_step5_norm_dtype(vllm_config.model_config.hf_config)
        # The fused qkv+indexer GEMM stays unquantized (upstream behaviour:
        # DSA projections load the checkpoint's bf16 tensors).
        linear_quant_config = quant_config

        # --- upstream DSA geometry validation -------------------------------
        if self.num_kv_heads != 1:
            raise ValueError(
                f"Step5 DSA requires exactly one local KV head, got {self.num_kv_heads}"
            )
        if self.head_dim not in (128, 192):
            raise ValueError(
                f"Step5 DSA head_dim must be 128/192, got {self.head_dim}"
            )
        _validate_step5_dsa_parallel_geometry(
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            indexer_num_heads=int(sparse_config.sparse_indexer_num_heads),
            index_tp_size=int(sparse_config.index_tp_size),
            tp_size=tp_size,
        )

        # --- projections -----------------------------------------------------
        proxy_dim = int(sparse_config.proxy_dim)
        self.sparse_config = sparse_config
        self.qkv_indexer_proj = Step5FusedQKVIndexerLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            indexer_q_output_size=(
                int(sparse_config.sparse_indexer_num_heads) * proxy_dim
            ),
            indexer_kv_output_size=(
                int(sparse_config.sparse_indexer_num_k_heads) * proxy_dim
            ),
            gate_output_size=(
                self.total_num_heads if use_head_wise_attn_gate else None
            ),
            proxy_dim=proxy_dim,
            index_tp_size=int(sparse_config.index_tp_size),
            quant_config=linear_quant_config,
            prefix=f"{prefix}.qkv_indexer_proj",
        )
        self.params_dtype = self.qkv_indexer_proj.params_dtype
        self.sparse_indexer_w = Step5SparseIndexerIndexTPLinear(
            hidden_size,
            int(sparse_config.sparse_indexer_num_heads),
            params_dtype=self.params_dtype,
            index_tp_size=int(sparse_config.index_tp_size),
            prefix=f"{prefix}.sparse_indexer_w",
        )
        # Dormant in the deployed scorer; registered for explicit loading.
        from torch.nn.parameter import Parameter

        self.ssmax_s = Parameter(
            torch.zeros(self.total_num_heads, dtype=torch.float32),
            requires_grad=False,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=linear_quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # --- main attention norms + rope -------------------------------------
        rope_parameters: dict[str, Any] = (
            dict(rope_scaling) if rope_scaling is not None else {}
        )
        rope_parameters.setdefault("rope_type", "default")
        if self.rope_theta is not None:
            rope_parameters["rope_theta"] = self.rope_theta
        rope_parameters["partial_rotary_factor"] = partial_rotary_factor

        self._sparse_indexer_base_rope_parameters = dict(rope_parameters)
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
            dtype=self.params_dtype,
        )
        self.q_norm = make_step5_rms_norm(
            self.head_dim,
            eps=rms_norm_eps,
            zero_centered=zero_centered,
            dtype=norm_dtype,
        )
        self.k_norm = make_step5_rms_norm(
            self.head_dim,
            eps=rms_norm_eps,
            zero_centered=zero_centered,
            dtype=norm_dtype,
        )

        # --- indexer norms + rope --------------------------------------------
        sparse_params = dict(self._sparse_indexer_base_rope_parameters)
        if str(sparse_params.get("rope_type", "default")).strip().lower() == "none":
            sparse_params["rope_type"] = "default"
        sparse_params["partial_rotary_factor"] = float(
            int(sparse_config.sparse_indexer_rope_dim) / proxy_dim
        )
        self.sparse_indexer_rotary_emb = get_rope(
            head_size=proxy_dim,
            max_position=max_position,
            rope_parameters=sparse_params,
            dtype=self.params_dtype,
        )
        self.sparse_indexer_q_norm = make_step5_rms_norm(
            proxy_dim,
            eps=rms_norm_eps,
            zero_centered=zero_centered,
        )
        self.sparse_indexer_k_norm = Step5IndexerLayerNorm(
            proxy_dim,
            eps=rms_norm_eps,
        )

        # --- indexer fused norm+rope switch --------------------------------
        indexer_rope_dim = int(sparse_config.sparse_indexer_rope_dim)
        self.use_fused_indexer_norm_rope = (
            os.environ.get("VLLM_STEP4_FUSE_INDEXER_NORM_ROPE", "1") == "1"
            and indexer_rope_dim > 0
            and indexer_rope_dim < proxy_dim
        )
        if self.use_fused_indexer_norm_rope:
            self._indexer_rotary_dim = indexer_rope_dim
            self._indexer_num_q_heads = int(
                sparse_config.sparse_indexer_num_heads
            ) // int(sparse_config.index_tp_size)
            self._indexer_num_k_heads = int(
                sparse_config.sparse_indexer_num_k_heads
            )
            self._indexer_zero_centered = zero_centered
            indexer_rotary_cache = self.sparse_indexer_rotary_emb.cos_sin_cache
            self._indexer_rope_cos, self._indexer_rope_sin = (
                indexer_rotary_cache.chunk(2, dim=-1)
            )

        # --- vLLM attention-layer plumbing -----------------------------------
        self.layer_name = f"{prefix}.attn"
        self.kv_cache_dtype = (
            cache_config.cache_dtype if cache_config is not None else "auto"
        )
        self.kv_cache_torch_dtype = _kv_cache_torch_dtype(cache_config, vllm_config)
        self.attn_backend = AscendStep5DSABackend
        # The MTP speculative token count is handed to the DSA impl so verify
        # batches (K+1 rows per request) take the ring update path.
        num_speculative_tokens = 0
        spec_config = vllm_config.speculative_config
        if spec_config is not None and spec_config.method == "mtp":
            num_speculative_tokens = int(
                getattr(spec_config, "num_speculative_tokens", 0) or 0
            )
        self.impl = AscendStep5DSAImpl(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            kv_cache_dtype=self.kv_cache_dtype,
            topk_regions=int(sparse_config.topk),
            region_size=int(sparse_config.region_block_size),
            proxy_dim=proxy_dim,
            max_num_seqs=int(vllm_config.scheduler_config.max_num_seqs),
            max_model_len=int(vllm_config.model_config.max_model_len),
            block_size=int(vllm_config.cache_config.block_size),
            decode_out_dtype=vllm_config.model_config.dtype,
            num_speculative_tokens=num_speculative_tokens,
        )
        compilation_config = vllm_config.compilation_config
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        compilation_config.static_forward_context[self.layer_name] = self
        self.kv_cache = torch.tensor([])
        self.max_position_embeddings = max_position

        self.use_fused_qknorm_rope = (
            os.environ.get("VLLM_STEP4_FUSE_QK_NORM_ROPE", "1") == "1"
            and self.use_rope
            and self.head_dim == 192
        )
        if self.use_fused_qknorm_rope:
            self._fused_rotary_pairs = self.rotary_dim // 2
            rotary_cache = self.rotary_emb.cos_sin_cache
            self.rope_cos, self.rope_sin = rotary_cache.chunk(2, dim=-1)
        else:
            self._fused_rotary_pairs = 0
            self.rope_cos = None
            self.rope_sin = None

        # Upstream ring/summary layers, reused verbatim: the summary page
        # (one FP8 summary per closed region, sharing this layer's block
        # ids) and the per-request CSA tail ring (deterministic slot
        # addressing derived from the block table).
        self.sparse_summary_cache = AscendStep5SparseSummaryCacheLayer(
            prefix=f"{prefix}.attn.sparse_summary_cache",
            target_impl=self.impl,
            static_forward_context=compilation_config.static_forward_context,
        )
        self.csa_tail_ring = Step5CSATailRing(
            prefix=f"{prefix}.attn.csa_tail_ring",
            main_layer_name=self.layer_name,
            num_speculative_tokens=num_speculative_tokens,
            static_forward_context=compilation_config.static_forward_context,
        )
        self.impl.csa_ring = self.csa_tail_ring

    def get_attn_backend(self) -> type[AscendStep5DSABackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
        )

    def _apply_sparse_indexer_rope(
        self,
        positions: torch.Tensor,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        proxy_dim = int(index_q.shape[-1])
        positions = positions.reshape(-1).to(device=index_q.device, dtype=torch.long)
        q_shape = tuple(index_q.shape)
        k_shape = tuple(index_k.shape)
        q = index_q.reshape(q_shape[0], -1, proxy_dim)
        k = index_k.reshape(k_shape[0], -1, proxy_dim)
        q, k = self.sparse_indexer_rotary_emb(positions, q, k)
        return q.reshape(q_shape), k.reshape(k_shape)

    def _finish_indexer(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        index_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Indexer norm+rope and head scores, all local shapes.

        Returns index_q [t, groups, heads_per_group, proxy] (float8_e4m3fn
        on the fused path -- the kernel quantizes at the store and the
        scoring op consumes it directly; bf16 on the non-fused fallback),
        index_k [t, 1, proxy], index_z [t, 1, proxy],
        weights [t, groups, heads_per_group] fp32.
        """
        num_tokens = int(hidden_states.shape[0])
        proxy_dim = int(self.sparse_config.proxy_dim)
        weights, _ = self.sparse_indexer_w(hidden_states)

        num_index_q_heads = int(index_q.shape[-1]) // proxy_dim
        num_index_k_heads = int(index_k.shape[-1]) // proxy_dim
        heads_per_group = num_index_q_heads // self.num_kv_heads

        if self.use_fused_indexer_norm_rope:
            index_q, index_k, index_z = fused_indexer_qkz_norm_rope(
                index_q,
                index_k,
                index_z,
                self.sparse_indexer_q_norm.weight,
                self.sparse_indexer_k_norm.weight,
                self._indexer_rope_cos,
                self._indexer_rope_sin,
                positions,
                q_norm_bias=None,
                k_norm_bias=self.sparse_indexer_k_norm.bias,
                head_dim=proxy_dim,
                num_q_heads=num_index_q_heads,
                num_k_heads=num_index_k_heads,
                rotary_dim=self._indexer_rotary_dim,
                eps=self.sparse_indexer_q_norm.variance_epsilon,
                q_norm_weight_bias=1.0 if self._indexer_zero_centered else 0.0,
                k_norm_weight_bias=0.0,
                subtract_mean_q=False,
                subtract_mean_k=True,
            )
            index_q = index_q.reshape(
                num_tokens, self.num_kv_heads, heads_per_group, proxy_dim
            )
            index_k = index_k.reshape(num_tokens, num_index_k_heads, proxy_dim)
            index_z = index_z.reshape(num_tokens, num_index_k_heads, proxy_dim)
        else:
            index_k = self.sparse_indexer_k_norm(index_k.contiguous())
            index_q = index_q.view(
                num_tokens, self.num_kv_heads, heads_per_group, proxy_dim
            ).contiguous()
            index_q = self.sparse_indexer_q_norm(index_q)
            index_k = index_k.view(
                num_tokens, num_index_k_heads, proxy_dim
            ).contiguous()
            index_q, index_k = self._apply_sparse_indexer_rope(
                positions,
                index_q.reshape(num_tokens, -1, proxy_dim),
                index_k,
            )
            index_q = index_q.reshape(
                num_tokens, self.num_kv_heads, heads_per_group, proxy_dim
            )
            index_z = index_z.view(num_tokens, num_index_k_heads, proxy_dim).contiguous()

        weights = weights.to(dtype=torch.float32)
        weights = weights.view(num_tokens, self.num_kv_heads, heads_per_group)
        weights = weights * (float(heads_per_group) ** -0.5)
        return index_q, index_k, index_z, weights.contiguous()

    def _write_kv(self, key: torch.Tensor, value: torch.Tensor) -> None:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return
        md = attn_metadata[self.layer_name]
        num_tokens = md.num_actual_tokens
        k = key[:num_tokens].view(-1, self.num_kv_heads, self.head_dim)
        v = value[:num_tokens].view(-1, self.num_kv_heads, self.head_dim)
        key_cache, value_cache = self.kv_cache[0], self.kv_cache[1]
        DeviceOperator.reshape_and_cache(
            k, v, key_cache, value_cache, md.slot_mapping[:num_tokens]
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        reduce_scatter_output: bool = False,
    ) -> torch.Tensor:
        qkv, qkzg = self.qkv_indexer_proj(hidden_states)

        if self.use_fused_qknorm_rope:
            q, k, v = fused_qknorm_rope(
                qkv,
                self.q_norm.weight,
                self.k_norm.weight,
                self.rope_cos,
                self.rope_sin,
                positions,
                head_dim=self.head_dim,
                num_q_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                rotary_pairs=self._fused_rotary_pairs,
                eps=self.q_norm.variance_epsilon,
                norm_weight_bias=1.0 if self.zero_centered else 0.0,
            )
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q_by_head = q.view(*q.shape[:-1], self.num_heads, self.head_dim)
            q = self.q_norm(q_by_head.contiguous()).view(q.shape)
            k_by_head = k.view(*k.shape[:-1], self.num_kv_heads, self.head_dim)
            k = self.k_norm(k_by_head.contiguous()).view(k.shape)
            if self.use_rope:
                q, k = self.rotary_emb(positions, q, k)

        index_q_raw, index_k_raw, index_z_raw, gate = (
            self.qkv_indexer_proj.split_indexer(qkzg)
        )
        index_q, index_k, index_z, weights = self._finish_indexer(
            positions, hidden_states, index_q_raw, index_k_raw, index_z_raw
        )

        num_tokens = int(hidden_states.shape[0])
        query = q.view(num_tokens, self.num_heads, self.head_dim)
        output = torch.empty(
            (num_tokens, self.num_heads, self.head_dim),
            dtype=q.dtype,
            device=q.device,
        )
        self._write_kv(k, v)
        self.impl.forward(
            self,
            query,
            self.kv_cache,
            index_q,
            index_k,
            index_z,
            weights,
            output,
            positions,
        )
        attn_output = output.view(num_tokens, self.num_heads * self.head_dim)
        if gate is not None:
            attn_output = (
                attn_output.view(num_tokens, self.num_heads, self.head_dim)
                * gate.unsqueeze(-1).sigmoid()
            ).view(num_tokens, -1)
        if reduce_scatter_output:
            # Sequence-parallel o_proj: pad rows to a TP multiple before the
            # reduce-scatter.
            sp_pad = (-attn_output.shape[0]) % self.tp_size
            if sp_pad:
                attn_output = torch.nn.functional.pad(attn_output, (0, 0, 0, sp_pad))
            output_tensor, _ = self.o_proj(
                attn_output, reduce_scatter_results=True, reduce_scatter_dim=0
            )
        else:
            output_tensor, _ = self.o_proj(attn_output)
        return output_tensor


class AscendStep5Attention(nn.Module):
    """Step5 sliding-window (dense) layer on Ascend.

    Built instead of the upstream ``Step5Attention`` so the dense layers run
    on the platform-selected Ascend attention backend (upstream pins a CUDA
    FlashAttention backend and a CUDA fused QKNorm+RoPE cache op). The
    projections, norms and RoPE mirror the upstream construction; the
    QKNorm+RoPE runs through the Ascend SIMD Triton kernel for head_dim=192.
    The KV cache spec comes from the inner platform ``Attention`` layer
    (registered as ``{prefix}.attn``), so this wrapper is a plain module.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float | list[float] | None = 10000,
        cache_config: Any | None = None,
        quant_config: Any | None = None,
        rope_scaling: dict[str, Any] | None = None,
        sliding_window: int | None = None,
        prefix: str = "",
        use_head_wise_attn_gate: bool = False,
        layer_types: list | None = None,
        use_rope_layers: list | None = None,
        yarn_only_types: list | None = None,
        swa_num_attention_heads: int | None = None,
        partial_rotary_factor: float = 1.0,
        zero_centered: bool = True,
        vllm_config: VllmConfig | None = None,
        sparse_config: Step5SparseConfig | None = None,
        norm_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert vllm_config is not None
        assert sparse_config is None, "AscendStep5Attention serves dense layers only"
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        self.prefix = prefix
        self.layer_idx = extract_layer_index(prefix)
        tp_size = get_tensor_model_parallel_world_size()

        if isinstance(rope_theta, list):
            rope_theta = _per_layer_value(
                rope_theta, self.layer_idx, name="rope_theta", default=None
            )

        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.partial_rotary_factor = float(partial_rotary_factor)
        self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.zero_centered = zero_centered
        self.use_head_wise_attn_gate = use_head_wise_attn_gate
        self.use_rope = bool(
            _per_layer_value(
                use_rope_layers, self.layer_idx, name="use_rope_layers", default=True
            )
        )
        if norm_dtype is None:
            norm_dtype = get_step5_norm_dtype(vllm_config.model_config.hf_config)

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.params_dtype = self.qkv_proj.params_dtype
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        if use_head_wise_attn_gate:
            self.g_proj = ColumnParallelLinear(
                hidden_size,
                self.total_num_heads,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
            )

        rope_parameters: dict[str, Any] = (
            dict(rope_scaling) if rope_scaling is not None else {}
        )
        rope_parameters.setdefault("rope_type", "default")
        if rope_theta is not None:
            rope_parameters["rope_theta"] = rope_theta
        rope_parameters["partial_rotary_factor"] = partial_rotary_factor
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
            dtype=self.params_dtype,
        )
        self.q_norm = make_step5_rms_norm(
            self.head_dim,
            eps=rms_norm_eps,
            zero_centered=zero_centered,
            dtype=norm_dtype,
        )
        self.k_norm = make_step5_rms_norm(
            self.head_dim,
            eps=rms_norm_eps,
            zero_centered=zero_centered,
            dtype=norm_dtype,
        )

        # The vLLM Attention layer resolves the backend through the platform
        # selector (AscendFlashAttentionBackend for GQA/SWA on NPU).
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=None,
            prefix=f"{prefix}.attn",
            per_layer_sliding_window=sliding_window,
        )

        self.use_fused_qknorm_rope = (
            os.environ.get("VLLM_STEP4_FUSE_QK_NORM_ROPE", "1") == "1"
            and self.use_rope
            and self.head_dim == 192
        )
        if self.use_fused_qknorm_rope:
            self._fused_rotary_pairs = self.rotary_dim // 2
            rotary_cache = self.rotary_emb.cos_sin_cache
            self.rope_cos, self.rope_sin = rotary_cache.chunk(2, dim=-1)
        else:
            self._fused_rotary_pairs = 0
            self.rope_cos = None
            self.rope_sin = None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        reduce_scatter_output: bool = False,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        if self.use_fused_qknorm_rope:
            q, k, v = fused_qknorm_rope(
                qkv,
                self.q_norm.weight,
                self.k_norm.weight,
                self.rope_cos,
                self.rope_sin,
                positions,
                head_dim=self.head_dim,
                num_q_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                rotary_pairs=self._fused_rotary_pairs,
                eps=self.q_norm.variance_epsilon,
                norm_weight_bias=1.0 if self.zero_centered else 0.0,
            )
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q_by_head = q.view(*q.shape[:-1], self.num_heads, self.head_dim)
            q = self.q_norm(q_by_head.contiguous()).view(q.shape)
            k_by_head = k.view(*k.shape[:-1], self.num_kv_heads, self.head_dim)
            k = self.k_norm(k_by_head.contiguous()).view(k.shape)
            if self.use_rope:
                q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)
        extra_dims = None
        if self.use_head_wise_attn_gate:
            extra_dims, _ = self.g_proj(hidden_states)
            attn_output = (
                attn_output.view(*attn_output.shape[:-1], self.num_heads, self.head_dim)
                * extra_dims.unsqueeze(-1).sigmoid()
            ).view(*attn_output.shape)
        if reduce_scatter_output:
            sp_pad = (-attn_output.shape[0]) % self.tp_size
            if sp_pad:
                attn_output = torch.nn.functional.pad(attn_output, (0, 0, 0, sp_pad))
            output, _ = self.o_proj(
                attn_output, reduce_scatter_results=True, reduce_scatter_dim=0
            )
        else:
            output, _ = self.o_proj(attn_output)
        return output


class AscendStep5DecoderLayer(Step5DecoderLayer):
    """Upstream decoder layer with the attention swapped per layer type.

    DSA (full-attention) layers get the Ascend DSA module; sliding layers get
    the dense Ascend module on the platform backend. The forward pass is
    inherited from upstream (the standard residual flow).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size
        self.fp32_residual_connection = config.fp32_residual_connection
        layer_idx = extract_layer_index(prefix)
        self.layer_idx = layer_idx
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        sliding_window = getattr(config, "sliding_window", None)
        if sliding_window is None:
            sliding_window = getattr(
                vllm_config.model_config.hf_text_config, "sliding_window", None
            )
        if sliding_window is None and cache_config is not None:
            sliding_window = cache_config.sliding_window
        if cache_config is not None:
            cache_config = copy(cache_config)
            cache_config.sliding_window = None

        sparse_config = get_step5_sparse_config(config)
        use_dsa_for_layer = (
            sparse_config is not None
            and _is_step5_full_attention_layer(config, layer_idx, sparse_config)
        )
        use_dsa_for_layer = use_dsa_for_layer and (
            os.environ.get("VLLM_STEP4_SPARSE", "1") != "0"
        )

        norm_dtype = get_step5_norm_dtype(config)
        layer_types = _get_step5_layer_types(config)
        partial_rotary_factors = getattr(config, "partial_rotary_factors", [])
        partial_rotary_factor = float(
            _per_layer_value(
                partial_rotary_factors,
                layer_idx,
                name="partial_rotary_factors",
                default=1.0,
            )
        )
        max_position = getattr(config, "max_position_embeddings", None)
        if max_position is None:
            max_position = vllm_config.model_config.max_model_len

        # Both layer families share 64 heads / 4 kv groups / head_dim 192 in
        # this checkpoint (num_attention_heads == num_sliding_attention_heads),
        # so config defaults resolve correctly for both.
        attn_kwargs = dict(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=max_position,
            num_kv_heads=config.num_attention_groups,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=getattr(config, "rope_scaling", None),
            sliding_window=sliding_window,
            use_head_wise_attn_gate=getattr(config, "use_head_wise_attn_gate", False),
            layer_types=layer_types,
            use_rope_layers=getattr(config, "use_rope_layers", []),
            yarn_only_types=getattr(config, "yarn_only_types", []),
            swa_num_attention_heads=getattr(
                config, "swa_num_attention_heads", None
            ),
            partial_rotary_factor=partial_rotary_factor,
            prefix=f"{prefix}.self_attn",
            zero_centered=config.zero_centered,
            vllm_config=vllm_config,
            norm_dtype=norm_dtype,
        )
        if use_dsa_for_layer:
            self.self_attn = AscendStep5DSAAttention(
                sparse_config=sparse_config,
                **attn_kwargs,
            )
        else:
            self.self_attn = AscendStep5Attention(
                sparse_config=None,
                **attn_kwargs,
            )

        self.use_sequence_parallel = False
        # Only the sequence-parallel forward compensates a skipped o_proj
        # reduction (sp_reduce_scatter after self_attn); with sequence
        # parallel off the RowParallelLinear must keep its TP all-reduce, or
        # every layer's attention output stays a per-rank partial sum.
        self.use_moe = False
        self.tp_group = get_tp_group()
        self.use_fused_all_reduce = (
            get_tensor_model_parallel_world_size() > 1
            and get_dp_group().world_size == 1
            and not self.use_sequence_parallel
        )

        moe_layers_idx = _get_step5_moe_layer_indices(config)
        if layer_idx in moe_layers_idx:
            self.moe = Step5MoE(
                vllm_config,
                prefix=f"{prefix}.moe",
            )
            self.use_moe = True
        else:
            self.mlp = Step5MLP(
                config=config,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act="silu",
                quant_config=quant_config,
                reduce_results=True,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = make_step5_rms_norm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            zero_centered=config.zero_centered,
            dtype=norm_dtype,
        )
        self.post_attention_layernorm = make_step5_rms_norm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            zero_centered=config.zero_centered,
            dtype=norm_dtype,
        )

    def _forward_ffn(
        self,
        hidden_states: torch.Tensor,
        padding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_moe:
            shared_output, moe_output = self.moe(hidden_states, padding=padding)
            if shared_output is None:
                return moe_output
            if (
                self.moe.fuse_all_reduce
                and AscendStep5DecoderLayer._ascend_moe_routed_reduced()
            ):
                # The Ascend MoE runner already all-reduced the routed output
                # for this dispatch; reduce the still-partial shared-expert
                # output to match and skip the upstream combined all-reduce,
                # which would count the routed contribution once per rank.
                shared_output = tensor_model_parallel_all_reduce(shared_output)
                return self._add_for_residual(moe_output, shared_output)
            combined = self._add_for_residual(moe_output, shared_output)
            if self.moe.fuse_all_reduce:
                if self.use_fused_all_reduce:
                    combined = self.tp_group.all_reduce(combined)
                else:
                    combined = tensor_model_parallel_all_reduce(combined)
            return combined
        return self.mlp(hidden_states)

    @staticmethod
    def _ascend_moe_routed_reduced() -> bool:
        try:
            from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
        except ImportError:
            return False
        return _EXTRA_CTX.moe_comm_type in {
            MoECommType.ALLTOALL,
            MoECommType.MC2,
            MoECommType.FUSED_MC2,
        }


class AscendStep5Model(Step5Model):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        # The DSA modules hold python-side per-layer state; stay eager.
        self.do_not_compile = True
        self.vllm_config = vllm_config
        # Read by the inherited Step5Model.forward (SP padding/all-gather).
        self.use_sequence_parallel = _step5_use_sequence_parallel(vllm_config)
        config = vllm_config.model_config.hf_config
        self.vocab_size = config.vocab_size
        self.config = config
        self.fp32_residual_connection = config.fp32_residual_connection
        self.moe_num_experts = config.moe_num_experts
        self.parallel_config = vllm_config.parallel_config

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: AscendStep5DecoderLayer(
                vllm_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            norm_dtype = get_step5_norm_dtype(config)
            self.norm = make_step5_rms_norm(
                config.hidden_size,
                eps=config.rms_norm_eps,
                zero_centered=config.zero_centered,
                dtype=norm_dtype,
            )
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
        )


# ModelSlim MXFP8 checkpoints store the routed-expert projections fused
# across all experts as 2D "[num_experts * out_features, in_features]"
# tensors, while the upstream Step5 loader requires 3D stacked
# "[num_experts, out_features, in_features]" tensors. The suffixes mirror
# the upstream expert_params_mapping weight names; ".moe.share_expert.*"
# and the dense-MLP projections must not match.
_STEP5_FUSED_EXPERT_2D_SUFFIXES = (
    ".moe.gate_proj.weight",
    ".moe.up_proj.weight",
    ".moe.down_proj.weight",
    ".moe.gate_proj.weight_scale",
    ".moe.up_proj.weight_scale",
    ".moe.down_proj.weight_scale",
    ".moe.gate_proj.weight_scale_2",
    ".moe.up_proj.weight_scale_2",
    ".moe.down_proj.weight_scale_2",
    ".moe.gate_proj.input_scale",
    ".moe.up_proj.input_scale",
    ".moe.down_proj.input_scale",
)


def _reshape_step5_fused_expert_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    num_experts: int,
) -> Iterable[tuple[str, torch.Tensor]]:
    """Reshape 2D fused-expert checkpoint tensors to the 3D stacked layout.

    Only tensors with an expert-projection suffix, a 2D shape, and a leading
    dimension that is a non-trivial multiple of ``num_experts`` are reshaped;
    everything else passes through untouched (the reshape is a view).
    """
    for name, tensor in weights:
        if (
            tensor.ndim == 2
            and name.endswith(_STEP5_FUSED_EXPERT_2D_SUFFIXES)
            and tensor.shape[0] != num_experts
            and tensor.shape[0] % num_experts == 0
        ):
            tensor = tensor.reshape(
                num_experts, tensor.shape[0] // num_experts, tensor.shape[1]
            )
        yield name, tensor


class AscendStep5ForCausalLM(Step5ForCausalLM):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        nn.Module.__init__(self)
        self.vllm_config = vllm_config
        model_config = vllm_config.model_config
        config = model_config.hf_config
        self.config = config
        self.fp32_residual_connection = config.fp32_residual_connection
        self.model = AscendStep5Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
            self.logits_processor = LogitsProcessor(config.vocab_size)
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters
        self.moe_layers: list[Any] = []
        example_layer: Step5MoE | None = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if hasattr(layer, "moe") and isinstance(layer.moe, Step5MoE):
                example_layer = layer.moe
                self.moe_layers.append(layer.moe.experts)
        _set_step5_moe_protocol_metadata(self, example_layer)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Adapt 2D fused-expert tensors (ModelSlim MXFP8 layout) to the 3D
        # stacked layout the upstream loader expects.
        return super().load_weights(
            _reshape_step5_fused_expert_weights(
                weights, self.model.moe_num_experts
            )
        )
