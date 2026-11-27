# SPDX-License-Identifier: Apache-2.0
"""Ascend Step5 model.

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

from collections.abc import Iterable
from copy import copy
from typing import Any

import torch
from torch import nn
from torch.nn.parameter import Parameter
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, GateLinear
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)
from vllm.models.step5.common.config import (  # type: ignore[import-not-found]
    Step5SparseConfig,
    get_step5_sparse_config,
)
from vllm.models.step5.nvidia.layernorm import (
    Step5IndexerLayerNorm,
    Step5RMSNorm,
    get_step5_norm_dtype,
)
from vllm.models.step5.nvidia.model import (
    Step5DecoderLayer,
    Step5ForCausalLM,
    Step5FusedQKVIndexerLinear,
    Step5MLP,
    Step5Model,
    Step5SparseIndexerIndexTPLinear,
    _get_step5_layer_types,
    _get_step5_moe_layer_indices,
    _is_step5_full_attention_layer,
    _per_layer_value,
    _set_step5_moe_protocol_metadata,
    _step5_moe_reduce_policy,
    _step5_sp_padding,
    _validate_step5_dsa_parallel_geometry,
    extract_layer_index,
    make_layers,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

from vllm_ascend.attention.utils import maybe_save_kv_layer_to_connector, wait_for_kv_layer_from_connector
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_reduce_scatter,
    sp_shard,
)
from vllm_ascend.models.step5.sparse_attention import (
    AscendStep5CSATailRing,
    AscendStep5DSABackend,
    AscendStep5DSAImpl,
    AscendStep5SparseSummaryCacheLayer,
)
from vllm_ascend.ops.triton.step5.fused_add_rms_norm import (
    can_use_fused_add_rms_norm_cast,
    fused_add_rms_norm_cast,
)
from vllm_ascend.ops.triton.step5.fused_qknorm_rope import (
    fused_qknorm_rope,
)
from vllm_ascend.ops.triton.step5.gated_scale import (
    gated_scale,
)
from vllm_ascend.ops.triton.step5.indexer_qkz_norm_rope import (
    fused_indexer_qkz_norm_rope,
)


def _kv_cache_torch_dtype(cache_config, vllm_config) -> torch.dtype:
    cache_dtype = cache_config.cache_dtype if cache_config is not None else "auto"
    if cache_dtype in ("auto", None, ""):
        return vllm_config.model_config.dtype
    if cache_dtype in ("bfloat16", "bf16"):
        return torch.bfloat16
    if cache_dtype in ("float16", "fp16", "half"):
        return torch.float16
    raise ValueError(f"Unsupported step5-ascend KV cache dtype: {cache_dtype}")


@eager_break_during_capture
def step5_sparse_attention_forward(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    index_q_raw: torch.Tensor,
    index_k_raw: torch.Tensor,
    index_z_raw: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Eager sparse compute behind an opaque op boundary (torch.compile-safe).

    Everything downstream of the projections owns python-side per-layer
    state: the indexer norm+rope+scoring (``_finish_indexer``), the
    KV-connector handoff, the KV cache write and the region-sparse
    attention kernels. The layer is resolved at runtime from the forward
    context -- ``no_compile_layers`` is the static forward context the
    module registered itself into.
    """
    layer = get_forward_context().no_compile_layers[layer_name]
    index_q, index_k, index_z, weights = layer._finish_indexer(
        positions, hidden_states, index_q_raw, index_k_raw, index_z_raw
    )
    wait_for_kv_layer_from_connector(layer.layer_name)
    layer._write_kv(key, value)
    layer.impl.forward(
        layer,
        query,
        layer.kv_cache,
        index_q,
        index_k,
        index_z,
        weights,
        output,
        positions,
    )
    maybe_save_kv_layer_to_connector(layer.layer_name, list(layer.kv_cache))


def step5_sparse_attention_forward_fake(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    index_q_raw: torch.Tensor,
    index_k_raw: torch.Tensor,
    index_z_raw: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="step5_sparse_attention_forward",
    op_func=step5_sparse_attention_forward,
    mutates_args=["output"],
    fake_impl=step5_sparse_attention_forward_fake,
    dispatch_key="PrivateUse1",
)


class AscendStep5RMSNorm(Step5RMSNorm):
    """Step5 RMSNorm on the fused NPU Gemma RMSNorm op.

    Step5 norm weights are stored zero-centered and npu_gemma_rms_norm
    applies ``weight + 1`` internally, so zero-centered norms map 1:1 onto
    the fused op instead of the eager fp32 pow/mean/rsqrt/mul small-op
    chain. The op requires input and weight to share a dtype and keeps the
    compute dtype on its output, so the forward casts across residual
    domains (e.g. a bf16 input with fp32 weights) and restores the caller's
    dtype -- the exact Step5RMSNorm contract (fp32 compute, output in the
    input/out dtype). Everything else (weight layout, module interface) is
    inherited unchanged, so checkpoints load and callers see a plain
    Step5RMSNorm.
    """

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        output: torch.Tensor | None = None,
        fp16_out: bool = False,
        *,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if not self.zero_centered or residual is not None or output is not None or fp16_out:
            return super().forward(
                x,
                residual=residual,
                output=output,
                fp16_out=fp16_out,
                out_dtype=out_dtype,
            )
        out_dtype = out_dtype or x.dtype
        if x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        out = DeviceOperator.npu_gemma_rms_norm(x, self.weight, self.variance_epsilon)
        return out.to(out_dtype) if out.dtype != out_dtype else out


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
            rope_theta = _per_layer_value(rope_theta, self.layer_idx, name="rope_theta", default=None)

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
        self.use_rope = bool(_per_layer_value(use_rope_layers, self.layer_idx, name="use_rope_layers", default=True))
        if norm_dtype is None:
            norm_dtype = get_step5_norm_dtype(vllm_config.model_config.hf_config)
        # The fused qkv+indexer GEMM stays unquantized (upstream behaviour:
        # DSA projections load the checkpoint's bf16 tensors).
        linear_quant_config = quant_config

        if self.num_kv_heads != 1:
            raise ValueError(f"Step5 DSA requires exactly one local KV head, got {self.num_kv_heads}")
        if self.head_dim not in (128, 192):
            raise ValueError(f"Step5 DSA head_dim must be 128/192, got {self.head_dim}")
        _validate_step5_dsa_parallel_geometry(
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            indexer_num_heads=int(sparse_config.sparse_indexer_num_heads),
            index_tp_size=int(sparse_config.index_tp_size),
            tp_size=tp_size,
        )

        proxy_dim = int(sparse_config.proxy_dim)
        self.sparse_config = sparse_config
        self.qkv_indexer_proj = Step5FusedQKVIndexerLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            indexer_q_output_size=(int(sparse_config.sparse_indexer_num_heads) * proxy_dim),
            indexer_kv_output_size=(int(sparse_config.sparse_indexer_num_k_heads) * proxy_dim),
            gate_output_size=(self.total_num_heads if use_head_wise_attn_gate else None),
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
        self._gate_col_offset = (
            self.qkv_indexer_proj.qkv_size + self.qkv_indexer_proj.indexer_size - self.qkv_indexer_proj.local_gate_size
        )

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

        rope_parameters: dict[str, Any] = dict(rope_scaling) if rope_scaling is not None else {}
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
        self.q_norm = AscendStep5RMSNorm(
            self.head_dim,
            eps=rms_norm_eps,
            zero_centered=zero_centered,
            dtype=norm_dtype,
        )
        self.k_norm = AscendStep5RMSNorm(
            self.head_dim,
            eps=rms_norm_eps,
            zero_centered=zero_centered,
            dtype=norm_dtype,
        )

        sparse_params = dict(self._sparse_indexer_base_rope_parameters)
        if str(sparse_params.get("rope_type", "default")).strip().lower() == "none":
            sparse_params["rope_type"] = "default"
        sparse_params["partial_rotary_factor"] = float(int(sparse_config.sparse_indexer_rope_dim) / proxy_dim)
        self.sparse_indexer_rotary_emb = get_rope(
            head_size=proxy_dim,
            max_position=max_position,
            rope_parameters=sparse_params,
            dtype=self.params_dtype,
        )
        self.sparse_indexer_q_norm = AscendStep5RMSNorm(
            proxy_dim,
            eps=rms_norm_eps,
            zero_centered=zero_centered,
        )
        self.sparse_indexer_k_norm = Step5IndexerLayerNorm(
            proxy_dim,
            eps=rms_norm_eps,
        )

        self.layer_name = f"{prefix}.attn"
        self.kv_cache_dtype = cache_config.cache_dtype if cache_config is not None else "auto"
        self.kv_cache_torch_dtype = _kv_cache_torch_dtype(cache_config, vllm_config)
        self.attn_backend = AscendStep5DSABackend
        # The MTP speculative token count is handed to the DSA impl so verify
        # batches (K+1 rows per request) take the ring update path.
        num_speculative_tokens = 0
        spec_config = vllm_config.speculative_config
        if spec_config is not None and spec_config.method == "mtp":
            num_speculative_tokens = int(getattr(spec_config, "num_speculative_tokens", 0) or 0)
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

        self.use_fused_qknorm_rope = self.use_rope and self.head_dim == 192
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
        self.csa_tail_ring = AscendStep5CSATailRing(
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

    def _finish_indexer(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        index_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Indexer norm+rope and head scores, all local shapes.

        Returns index_q [t, groups, heads_per_group, proxy] (float8_e4m3fn;
        the kernel quantizes at the store and the scoring op consumes it
        directly), index_k [t, 1, proxy], index_z [t, 1, proxy],
        weights [t, groups, heads_per_group] fp32.
        """
        num_tokens = int(hidden_states.shape[0])
        proxy_dim = int(self.sparse_config.proxy_dim)
        weights, _ = self.sparse_indexer_w(hidden_states)

        num_index_q_heads = int(index_q.shape[-1]) // proxy_dim
        num_index_k_heads = int(index_k.shape[-1]) // proxy_dim
        heads_per_group = num_index_q_heads // self.num_kv_heads

        # One kernel for the whole indexer chain: per-head RMSNorm on
        # index_q, per-head LayerNorm on index_k, NeoX RoPE on both and
        # the Z passthrough, reading the strided split_indexer views in
        # place (no materializing copy on any of the three inputs) and
        # quantizing index_q to e4m3 at the store.
        index_q, index_k, index_z = fused_indexer_qkz_norm_rope(
            index_q,
            index_k,
            index_z,
            self.sparse_indexer_q_norm.weight,
            self.sparse_indexer_k_norm.weight,
            self.sparse_indexer_rotary_emb.cos_sin_cache,
            positions,
            k_norm_bias=self.sparse_indexer_k_norm.bias,
            head_dim=proxy_dim,
            num_q_heads=num_index_q_heads,
            num_k_heads=num_index_k_heads,
            rotary_dim=self.sparse_indexer_rotary_emb.rotary_dim,
            eps=self.sparse_indexer_q_norm.variance_epsilon,
            q_norm_weight_bias=1.0 if self.sparse_indexer_q_norm.zero_centered else 0.0,
        )
        index_q = index_q.reshape(num_tokens, self.num_kv_heads, heads_per_group, proxy_dim)
        index_k = index_k.reshape(num_tokens, num_index_k_heads, proxy_dim)
        index_z = index_z.reshape(num_tokens, num_index_k_heads, proxy_dim)
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
        DeviceOperator.reshape_and_cache(k, v, key_cache, value_cache, md.slot_mapping[:num_tokens])

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
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

        index_q_raw, index_k_raw, index_z_raw, gate = self.qkv_indexer_proj.split_indexer(qkzg)
        query = q.view(-1, self.num_heads, self.head_dim)
        output = q.new_empty((q.shape[0], self.num_heads, self.head_dim))
        torch.ops.vllm.step5_sparse_attention_forward(
            hidden_states,
            positions,
            query,
            k,
            v,
            index_q_raw,
            index_k_raw,
            index_z_raw,
            output,
            self.layer_name,
        )

        attn_output = output.view(-1, self.num_heads * self.head_dim)
        if gate is not None:
            # The gate is the tail of the fused qkv+indexer GEMM row. Address
            # it through the offset-0 qkv view of the same storage so no
            # mid-storage view crosses the op boundary (the dispatch layer
            # would materialize it first: TensorMove).
            attn_output = gated_scale(
                attn_output,
                qkv,
                self._gate_col_offset,
                gate.shape[-1],
            )
        output_tensor, _ = self.o_proj(attn_output)
        return output_tensor


class AscendStep5Attention(nn.Module):
    """Step5 sliding-window layer on Ascend.

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
            rope_theta = _per_layer_value(rope_theta, self.layer_idx, name="rope_theta", default=None)

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
        self.use_rope = bool(_per_layer_value(use_rope_layers, self.layer_idx, name="use_rope_layers", default=True))
        if norm_dtype is None:
            norm_dtype = get_step5_norm_dtype(vllm_config.model_config.hf_config)

        # Pack Q/K/V and the per-head gate into one column-parallel GEMM when
        # KV heads are partitioned (not replicated) across TP ranks, mirroring
        # the upstream ``Step5Attention`` construction. The local output layout
        # is [Q, K, V, gate]. ``MergedColumnParallelLinear`` cannot replicate
        # KV heads, so packing stays off when ``total_num_kv_heads < tp_size``
        # and the layer falls back to ``qkv_proj`` + ``g_proj``.
        packed_quant_ok = True
        if use_head_wise_attn_gate and quant_config is not None:
            tp_rank = get_tensor_model_parallel_rank()
            packed_quant_ok = all(
                isinstance(
                    LinearBase(
                        hidden_size,
                        width,
                        quant_config=quant_config,
                        prefix=f"{prefix}.{name}",
                        tp_rank=tp_rank,
                        tp_size=tp_size,
                    ).quant_method,
                    UnquantizedLinearMethod,
                )
                for name, width in (
                    (
                        "qkv_proj",
                        (self.total_num_heads + 2 * num_kv_heads) * self.head_dim,
                    ),
                    ("g_proj", self.total_num_heads),
                )
            )
        self.fuse_qkv_gate = (
            use_head_wise_attn_gate and not qkv_bias and self.total_num_kv_heads >= tp_size and packed_quant_ok
        )
        if self.fuse_qkv_gate:
            # Keep the qkv_proj prefix so quantization configuration and
            # checkpoint weight remapping that target qkv_proj continue to
            # apply to the fused module (same contract as upstream).
            self.qkvg_proj = MergedColumnParallelLinear(
                hidden_size,
                [
                    self.total_num_heads * self.head_dim,
                    self.total_num_kv_heads * self.head_dim,
                    self.total_num_kv_heads * self.head_dim,
                    self.total_num_heads,
                ],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
            self.params_dtype = self.qkvg_proj.params_dtype
        else:
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
        if use_head_wise_attn_gate and not self.fuse_qkv_gate:
            self.g_proj = ColumnParallelLinear(
                hidden_size,
                self.total_num_heads,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
            )

        rope_parameters: dict[str, Any] = dict(rope_scaling) if rope_scaling is not None else {}
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
        self.q_norm = AscendStep5RMSNorm(
            self.head_dim,
            eps=rms_norm_eps,
            zero_centered=zero_centered,
            dtype=norm_dtype,
        )
        self.k_norm = AscendStep5RMSNorm(
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

        self.use_fused_qknorm_rope = self.use_rope and self.head_dim == 192
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
    ) -> torch.Tensor:
        qkvg = None
        if self.fuse_qkv_gate:
            qkvg, _ = self.qkvg_proj(hidden_states)
            qkv = qkvg[..., : self.q_size + 2 * self.kv_size]
        else:
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
        if qkvg is not None:
            # Packed layout: the gate is the tail of the qkvg row and anchors
            # at the packed GEMM output (offset-0), so no mid-storage view
            # crosses the op boundary (the dispatch layer would materialize
            # it first: TensorMove).
            gate_base = qkvg
            gate_col_offset = self.q_size + 2 * self.kv_size
        elif self.use_head_wise_attn_gate:
            # Standalone layout: the g_proj output anchors itself with offset 0.
            gate_base, _ = self.g_proj(hidden_states)
            gate_col_offset = 0
        else:
            gate_base = None
        if gate_base is not None:
            attn_output = gated_scale(
                attn_output,
                gate_base,
                gate_col_offset,
                self.num_heads,
            )
        output, _ = self.o_proj(attn_output)
        return output


class AscendStep5MoE(nn.Module):
    """Step5 MoE block wired to the vllm-ascend fused MoE runner.

    A standalone nn.Module assembled in a single pass, MiniMaxM3MoE /
    AscendKimiMoE / Glm5NextMoE style -- the upstream ``Step5MoE.__init__``
    is not reused: it would allocate a complete routed-experts runner
    through the patched ``FusedMoEFactory`` that this block then discards
    and rebuilds, so every MoE layer's expert weights would be allocated
    twice. Every attribute the moe-collection / EPLB gates read (the n_*
    metadata family, gate, router_bias, share_expert, experts) is set here
    with upstream-compatible names and prefixes; the gates in
    ``AscendStep5ForCausalLM`` dispatch on this class instead of
    ``Step5MoE``.

    Differences vs the upstream block (mirroring the proven MRV1 wiring):

    - Shared expert: built with ``reduce_results=False`` and handed to
      ``FusedMoEFactory(shared_experts=...)`` so ``AscendMoERunner`` owns its
      execution: the shared/routed sum and the TP reductions happen inside
      the runner, and the shared expert can overlap the routed experts on a
      separate stream via ``multistream_overlap_shared_expert``. The block's
      forward returns a single complete tensor instead of a
      (shared, routed) tuple.
    - MoE sequence parallelism: ``is_sequence_parallel=enable_sp_moe`` drives
      the runner's EP all-gather/reduce-scatter path so the MoE consumes the
      sequence shard directly (the AG-attn-RS-MoE(shard) decoder loop).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        self.tp_size = get_tensor_model_parallel_world_size()
        self.layer_idx = extract_layer_index(prefix)

        ep_size = get_ep_group().device_group.size()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        self.enable_eplb = parallel_config.enable_eplb
        self.n_routed_experts = config.moe_num_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = parallel_config.eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // ep_size

        if self.tp_size > config.moe_num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than the number of experts {config.moe_num_experts}."
            )
        self.gate = GateLinear(
            config.hidden_size,
            config.moe_num_experts,
            bias=False,
            out_dtype=torch.float32,
            params_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        if not config.use_moe_router_bias:
            raise ValueError("Step5 MoE currently requires use_moe_router_bias=true.")
        self.router_bias = nn.Parameter(
            torch.zeros(config.moe_num_experts, dtype=torch.float32),
            requires_grad=False,
        )
        if not config.need_fp32_gate:
            raise ValueError("Step5 MoE requires need_fp32_gate=true for stable router logits.")

        activation = "silu"
        swiglu_limits = config.swiglu_limits or []
        swiglu_limit = swiglu_limits[self.layer_idx] if self.layer_idx < len(swiglu_limits) else None
        if swiglu_limit not in (None, 0):
            activation = "swiglustep"

        enable_sp_moe = parallel_config.use_sequence_parallel_moe

        _, reduce_results = _step5_moe_reduce_policy(
            self.tp_size,
            get_dp_group().world_size,
        )

        self.share_expert = Step5MLP(
            config=config,
            hidden_size=self.hidden_size,
            intermediate_size=config.share_expert_dim,
            hidden_act="silu",
            reduce_results=False,
            quant_config=quant_config,
            prefix=f"{prefix}.share_expert",
        )

        self.experts = FusedMoEFactory(
            shared_experts=self.share_expert,
            num_experts=config.moe_num_experts,
            top_k=config.moe_top_k,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=reduce_results,
            scoring_func=getattr(config, "moe_router_activation", "sigmoid"),
            renormalize=config.norm_expert_weight,
            quant_config=quant_config,
            activation=activation,
            prefix=f"{prefix}.experts",
            e_score_correction_bias=self.router_bias,
            routed_scaling_factor=config.moe_router_scaling_factor,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=enable_sp_moe,
            router_logits_dtype=torch.float32,
        )
        # Ascend dispatch sizes the global physical expert space from this
        # metadata (mirrors the upstream block and MiniMaxM3MoE).
        self.experts.global_redundant_expert_num = self.n_redundant_experts
        self.experts.moe_config.global_redundant_expert_num = self.n_redundant_experts

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_is_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)
        # Sequence-parallel state is carried by the runner's FusedMoEConfig.
        if self.experts.moe_config.is_sequence_parallel and not input_is_sequence_parallel:
            hidden_states = sp_shard(hidden_states)

        if self.experts.is_internal_router:
            # Internal-router runners compute the gate themselves and the
            # runner sums the shared expert into the routed output.
            return self.experts(hidden_states=hidden_states, router_logits=hidden_states)
        router_logits, _ = self.gate(hidden_states)
        # Single complete tensor: AscendMoERunner sums the shared expert
        # output internally and applies the required TP reductions.
        return self.experts(hidden_states=hidden_states, router_logits=router_logits)


class AscendStep5MLP(Step5MLP):
    """Dense Step5 FFN under sequence parallelism (Kimi-K3 / MiniMax-M3 wiring).

    TP-sharded weights must see the same token rows before the row-parallel
    reduction, so the local sequence shard is gathered back to the padded
    full sequence, the down projection stays unreduced, and
    ``sp_reduce_scatter`` sums the TP partials straight back into the
    rank-local shard (an all-reduce + re-shard fused into one collective).
    """

    def __init__(
        self,
        *args,
        sequence_parallel: bool = False,
        **kwargs,
    ) -> None:
        # Under SP the down projection must not all-reduce: partial sums of
        # *different* token shards would be added together.
        if sequence_parallel:
            kwargs["reduce_results"] = False
        super().__init__(*args, **kwargs)
        self.sequence_parallel_mlp = sequence_parallel

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.sequence_parallel_mlp:
            hidden_states = sp_all_gather(hidden_states)
        output = super().forward(hidden_states)
        if self.sequence_parallel_mlp:
            output = sp_reduce_scatter(output)
        return output


class AscendStep5DecoderLayer(Step5DecoderLayer):
    """Upstream decoder layer with the attention swapped per layer type.

    DSA (full-attention) layers get the Ascend DSA module; sliding layers get
    the dense Ascend module on the platform backend. The forward mirrors the
    upstream Step5DecoderLayer flow (unfused add + gemma norms) including the
    sequence-parallel all-gather/reduce-scatter placement around attention.
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
            sliding_window = getattr(vllm_config.model_config.hf_text_config, "sliding_window", None)
        if sliding_window is None and cache_config is not None:
            sliding_window = cache_config.sliding_window
        if cache_config is not None:
            cache_config = copy(cache_config)
            cache_config.sliding_window = None

        sparse_config = get_step5_sparse_config(config)
        use_dsa_for_layer = sparse_config is not None and _is_step5_full_attention_layer(
            config, layer_idx, sparse_config
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
            swa_num_attention_heads=getattr(config, "swa_num_attention_heads", None),
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

        self.use_sequence_parallel = vllm_config.parallel_config.use_sequence_parallel_moe
        self.use_moe = False

        moe_layers_idx = _get_step5_moe_layer_indices(config)
        if layer_idx in moe_layers_idx:
            self.moe = AscendStep5MoE(
                vllm_config,
                prefix=f"{prefix}.moe",
            )
            self.use_moe = True
        else:
            self.mlp = AscendStep5MLP(
                config=config,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act="silu",
                quant_config=quant_config,
                reduce_results=True,
                prefix=f"{prefix}.mlp",
                sequence_parallel=self.use_sequence_parallel,
            )
        if self.use_sequence_parallel:
            # Upstream wiring: drop the o_proj all-reduce; the layer
            # reduce-scatters the partial attention output into sequence
            # shards right after the projection.
            self.self_attn.o_proj.reduce_results = False
        self.input_layernorm = AscendStep5RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            zero_centered=config.zero_centered,
            dtype=norm_dtype,
        )
        self.post_attention_layernorm = AscendStep5RMSNorm(
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
            # The next layer or the final residual add handles FP32 promotion.
            return self.moe(hidden_states, input_is_sequence_parallel=self.use_sequence_parallel)
        return self.mlp(hidden_states)

    def _add_rms_norm(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        norm: AscendStep5RMSNorm,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if can_use_fused_add_rms_norm_cast(hidden_states, residual, norm.weight, self.fp32_residual_connection):
            return fused_add_rms_norm_cast(
                hidden_states, residual, norm.weight, norm.variance_epsilon, norm.zero_centered
            )
        if residual is not None:
            hidden_states = self._add_for_residual(hidden_states, residual)
        return self._cast_for_param_op(norm(hidden_states)), hidden_states

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
        padding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self._add_rms_norm(hidden_states, residual, self.input_layernorm)

        if self.use_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)[: positions.shape[0]]
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        if self.use_sequence_parallel:
            hidden_states = sp_reduce_scatter(hidden_states)

        hidden_states, residual = self._add_rms_norm(hidden_states, residual, self.post_attention_layernorm)

        ffn_output = self._forward_ffn(hidden_states, padding=padding)
        return ffn_output, residual


@support_torch_compile
class AscendStep5Model(Step5Model):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.vllm_config = vllm_config
        # Switched like the other Ascend MoE models; drives the shard-at-
        # entry / gather-at-exit forward below and the per-layer SP wiring.
        self.use_sequence_parallel = vllm_config.parallel_config.use_sequence_parallel_moe
        config = vllm_config.model_config.hf_config
        self.vocab_size = config.vocab_size
        self.config = config
        self.fp32_residual_connection = config.fp32_residual_connection
        self.moe_num_experts = config.moe_num_experts
        self.parallel_config = vllm_config.parallel_config

        if get_pp_group().is_first_rank or (config.tie_word_embeddings and get_pp_group().is_last_rank):
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
            self.norm = AscendStep5RMSNorm(
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

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
        hidden_states = self._cast_for_residual(hidden_states)
        padding = None
        if self.use_sequence_parallel:
            padding = _step5_sp_padding(hidden_states)
            hidden_states = sp_shard(hidden_states)
        residual = None
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states, residual, padding=padding)
        if residual is not None:
            hidden_states = layer._add_for_residual(hidden_states, residual)
        if self.use_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)[: positions.shape[0]]

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                }
            )

        return hidden_states


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
            tensor = tensor.reshape(num_experts, tensor.shape[0] // num_experts, tensor.shape[1])
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
        self.model = AscendStep5Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.num_logical_experts: int
        self.num_physical_experts: int
        self.num_local_physical_experts: int
        self.num_redundant_experts: int
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

        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        # Set MoE hyperparameters
        self.moe_layers: list[Any] = []
        example_layer: AscendStep5MoE | None = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if hasattr(layer, "moe") and isinstance(layer.moe, AscendStep5MoE):
                example_layer = layer.moe
                self.moe_layers.append(layer.moe.experts)
        _set_step5_moe_protocol_metadata(self, example_layer)

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        if self.num_local_physical_experts != num_local_physical_experts:
            raise ValueError(
                "Step5 EPLB cannot change the number of local physical experts: "
                f"expected={self.num_local_physical_experts}, "
                f"got={num_local_physical_experts}."
            )
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            moe = getattr(layer, "moe", None)
            if not isinstance(moe, AscendStep5MoE):
                continue
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Adapt 2D fused-expert tensors (ModelSlim MXFP8 layout) to the 3D
        # stacked layout the upstream loader expects.
        return super().load_weights(_reshape_step5_fused_expert_weights(weights, self.model.moe_num_experts))
