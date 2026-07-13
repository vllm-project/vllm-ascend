# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3Config

from vllm import _custom_ops as ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.multimodal.inputs import NestedTensors
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.v1.attention.backend import AttentionType

from vllm.model_executor.models.qwen2 import Qwen2MLP as Qwen3MLP
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    get_draft_quant_config,
    maybe_prefix,
    process_eagle_weight,
)

from vllm_ascend.models.dspark_quarot import (
    load_quarot_rotation,
    resolve_quarot_rotation_path,
    transform_fc_weight_for_quarot,
)

logger = init_logger(__name__)


def _rms_norm_into(
    out: torch.Tensor,
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> None:
    try:
        ops.rms_norm(out, input_tensor, weight, epsilon)
    except AttributeError:
        variance = input_tensor.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        normalized = input_tensor * torch.rsqrt(variance.to(input_tensor.dtype) + epsilon)
        out.copy_(normalized * weight)


def _apply_rotary_native_inplace(
    x: torch.Tensor,
    positions: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox_style: bool,
) -> None:
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    cos_sin = cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)

    x_shape = x.shape
    x_view = x.view(num_tokens, -1, head_size)
    rotary_dim = cos.shape[-1] * 2
    x_rot = x_view[..., :rotary_dim]
    x_pass = x_view[..., rotary_dim:]

    cos = cos.unsqueeze(-2).to(x_rot.dtype)
    sin = sin.unsqueeze(-2).to(x_rot.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x_rot, 2, dim=-1)
    else:
        x1 = x_rot[..., ::2]
        x2 = x_rot[..., 1::2]

    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        out_rot = torch.cat((o1, o2), dim=-1)
    else:
        out_rot = torch.stack((o1, o2), dim=-1).flatten(-2)

    if x_pass.shape[-1] > 0:
        out = torch.cat((out_rot, x_pass), dim=-1)
    else:
        out = out_rot
    x.copy_(out.reshape(x_shape))


def _rotary_embedding_inplace(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox_style: bool,
) -> None:
    try:
        ops.rotary_embedding(
            positions,
            query,
            key,
            head_size,
            cos_sin_cache,
            is_neox_style,
        )
    except AttributeError:
        _apply_rotary_native_inplace(
            query,
            positions,
            head_size,
            cos_sin_cache,
            is_neox_style,
        )
        if key is not None:
            _apply_rotary_native_inplace(
                key,
                positions,
                head_size,
                cos_sin_cache,
                is_neox_style,
            )


class DSparkMarkovHead(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        markov_rank: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.markov_w1 = VocabParallelEmbedding(
            vocab_size,
            markov_rank,
            prefix=maybe_prefix(prefix, "markov_w1"),
        )
        self.markov_w2 = ParallelLMHead(
            vocab_size,
            markov_rank,
            prefix=maybe_prefix(prefix, "markov_w2"),
        )
        self.logits_processor = LogitsProcessor(vocab_size)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        markov_embed = self.markov_w1(token_ids.long())
        markov_logits = self.logits_processor(self.markov_w2, markov_embed)
        return markov_logits, markov_embed


class DSparkConfidenceHead(nn.Module):
    def __init__(
        self,
        input_size: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.proj = ReplicatedLinear(
            input_size,
            1,
            bias=True,
            quant_config=None,
            prefix=maybe_prefix(prefix, "proj"),
            return_bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        markov_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if markov_embeds is not None:
            hidden_states = torch.cat([hidden_states, markov_embeds], dim=-1)
        logits = self.proj(hidden_states)
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.squeeze(-1)


class DFlashQwen3Attention(nn.Module):
    """Attention for DFlash speculative decoding.

    Context KVs are pre-inserted into the KV cache before the forward pass.
    This layer handles only query tokens via standard attention.
    Adapted from Qwen3Attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        attention_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()
        self.layer_name = prefix
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,  # DFlash has o_proj bias when using attention bias
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        rope_interleave = os.getenv("DSPARK_ROPE_INTERLEAVE", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            is_neox_style=not rope_interleave,
            rope_parameters=rope_parameters,
        )
        if rope_interleave:
            logger.warning_once(
                "[spec_decode/dspark] using interleaved draft RoPE "
                "(is_neox_style=False)"
            )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """DFlash attention assumes that the KV cache is already populated
        with the context K/V from the target model's hidden states. This forward op
        computes attention for the query tokens only.
        See also: precompute_and_store_context_kv"""
        qkv = F.linear(hidden_states, self.qkv_proj.weight, self.qkv_proj.bias)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Per-head RMSNorm
        q_shape, k_shape = q.shape, k.shape
        q = self.q_norm(
            q.view(*q_shape[:-1], q_shape[-1] // self.head_dim, self.head_dim)
        ).view(q_shape)
        k = self.k_norm(
            k.view(*k_shape[:-1], k_shape[-1] // self.head_dim, self.head_dim)
        ).view(k_shape)

        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class DFlashQwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config: Qwen3Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=1000000)
        attn_type = AttentionType.DECODER

        self.self_attn = DFlashQwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        else:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class DFlashQwen3Model(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.vocab_size = self.config.vocab_size
        self.quant_config = get_draft_quant_config(vllm_config)

        drafter_config = getattr(self.config, "eagle_config", {})
        drafter_config.update(getattr(self.config, "dflash_config", {}))

        if drafter_config is not None and "use_aux_hidden_state" in drafter_config:
            self.use_aux_hidden_state = drafter_config["use_aux_hidden_state"]
        else:
            self.use_aux_hidden_state = True

        current_vllm_config = get_current_vllm_config()

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        self.layers = nn.ModuleList(
            [
                DFlashQwen3DecoderLayer(
                    current_vllm_config,
                    config=self.config,
                    cache_config=current_vllm_config.cache_config,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, f"layers.{layer_idx + start_layer_id}"),
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        if self.use_aux_hidden_state:
            num_features_to_use = self.config.num_hidden_layers
            if "target_layer_ids" in drafter_config:
                num_features_to_use = len(drafter_config["target_layer_ids"])
            elif "layer_ids" in drafter_config:
                num_features_to_use = len(drafter_config["layer_ids"])
            if hasattr(self.config, "target_hidden_size"):
                fc_input_size = self.config.target_hidden_size * num_features_to_use
            else:
                fc_input_size = self.config.hidden_size * num_features_to_use
            self.fc = ReplicatedLinear(
                input_size=fc_input_size,
                output_size=self.config.hidden_size,
                bias=False,
                params_dtype=vllm_config.model_config.dtype,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "fc"),
                return_bias=False,
            )
        self.hidden_norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _build_fused_kv_buffers(self) -> None:
        """Build fused weight buffers for precompute_and_store_context_kv.

        Must be called after weights are loaded. Stacks the KV-projection
        weights, K-norm weights, and RoPE parameters from every attention
        layer so that precompute_and_store_context_kv can run one fused
        GEMM for all layers at once. Also aliases the weight of the hidden_norm.
        """
        layers_attn = [layer.self_attn for layer in self.layers]
        attn0 = layers_attn[0]
        has_bias = attn0.qkv_proj.bias is not None

        self._hidden_norm_weight = self.hidden_norm.weight.data

        # KV projection weights: [num_layers * 2 * kv_size, hidden_size]
        kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
        self._fused_kv_weight = torch.cat(kv_weights, dim=0)
        if has_bias:
            kv_biases = [a.qkv_proj.bias[a.q_size :] for a in layers_attn]
            self._fused_kv_bias: torch.Tensor | None = torch.cat(kv_biases, dim=0)
        else:
            self._fused_kv_bias = None

        # K-norm weights: list of [head_dim] tensors, one per layer.
        self._k_norm_weights = [a.k_norm.weight.data for a in layers_attn]

        # RoPE parameters
        self._rope_head_size = attn0.rotary_emb.head_size
        self._rope_cos_sin_cache = attn0.rotary_emb.cos_sin_cache
        self._rope_is_neox = attn0.rotary_emb.is_neox_style
        # Validation that RoPE params are the same across all layers
        for attn in layers_attn[1:]:
            assert (
                attn.rotary_emb.head_size == self._rope_head_size
                and attn.rotary_emb.is_neox_style == self._rope_is_neox
            ), "All layers must have the same RoPE parameters for DFlash precomputation"

        # Layer metadata
        self._num_attn_layers = len(layers_attn)
        self._kv_size = attn0.kv_size
        self._head_dim = attn0.head_dim
        self._num_kv_heads = attn0.num_kv_heads
        self._rms_norm_eps = attn0.q_norm.variance_epsilon
        # Validation that all layers have the same attention config
        for attn in layers_attn[1:]:
            assert (
                attn.kv_size == self._kv_size
                and attn.head_dim == self._head_dim
                and attn.num_kv_heads == self._num_kv_heads
                and attn.q_norm.variance_epsilon == self._rms_norm_eps
            ), "All layers must have the same attn config for DFlash precomputation"

        # References to inner Attention layers for direct cache writes
        self._attn_layers = [layer.self_attn.attn for layer in self.layers]

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None = None,
    ) -> None:
        """Precompute K/V for context states write them into each layer's KV cache.

        Input context states are projected to K/V, normed, and have RoPE applied.
        Since the context shape is different than the query shape, we can't rely on the
        regular forward pass to apply torch.compile and CUDA graphs to this section.
        As such, this function is optimized to minimize the number of torch ops present:
        we use fused vLLM kernels for RMSNorm and RoPE, fuse the GEMM into one
        large projection, and avoid cloning buffers (with .contiguous()) where possible.

        When context_slot_mapping is None (e.g. during dummy_run) only
        the computation runs, and no K/V is written to cache.
        """
        if not hasattr(self, "_num_attn_layers"):
            logger.warning_once(
                "DFlash buffer initialization was skipped. If dummy weights are not "
                "in use, this may indicate an error in weight loading."
            )
            self._build_fused_kv_buffers()

        num_ctx = context_states.shape[0]
        L = self._num_attn_layers
        kv = self._kv_size
        hd = self._head_dim
        nkv = self._num_kv_heads

        # --- Fused KV projection (one GEMM for all layers) ---
        normed_context_states = torch.empty_like(context_states)
        _rms_norm_into(
            normed_context_states,
            context_states,
            self._hidden_norm_weight,
            self._rms_norm_eps,
        )
        all_kv_flat = F.linear(
            normed_context_states, self._fused_kv_weight, self._fused_kv_bias
        )
        # Single contiguous copy that separates K/V and transposes to
        # layer-major layout.  Result: [2, L, num_ctx, nkv, hd] contiguous.
        # Indexing dim-0 gives contiguous [L, num_ctx, nkv, hd] for K and V.
        all_kv = (
            all_kv_flat.view(num_ctx, L, 2, nkv, hd).permute(2, 1, 0, 3, 4).contiguous()
        )
        all_k = all_kv[0]  # [L, num_ctx, nkv, hd], contiguous
        all_v = all_kv[1]  # [L, num_ctx, nkv, hd], contiguous

        # --- Per-layer RMSNorm K (3D: [num_ctx, nkv, hd] per layer) ---
        all_k_normed = torch.empty_like(all_k)
        for i in range(L):
            _rms_norm_into(
                all_k_normed[i],
                all_k[i],
                self._k_norm_weights[i],
                self._rms_norm_eps,
            )

        # --- Fused RoPE across all layers ---
        # View as [L * num_ctx, kv] so RoPE sees one big batch (no copy).
        # In-place RoPE: pass K as the "query" arg with key=None.
        all_k_flat = all_k_normed.view(L * num_ctx, kv)
        positions_repeated = context_positions.repeat(L)
        cos_sin_cache = self._rope_cos_sin_cache
        if (
            cos_sin_cache.device != all_k_flat.device
            or cos_sin_cache.dtype != all_k_flat.dtype
        ):
            cos_sin_cache = cos_sin_cache.to(
                device=all_k_flat.device,
                dtype=all_k_flat.dtype,
            )
        _rotary_embedding_inplace(
            positions_repeated,
            all_k_flat,
            None,
            self._rope_head_size,
            cos_sin_cache,
            self._rope_is_neox,
        )

        if context_slot_mapping is None:
            return

        # --- Per-layer cache insert ---
        all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)
        if (
            os.getenv("DSPARK_LOGITS_DEBUG", "0").lower() in ("1", "true", "yes", "on")
            and getattr(self, "_dspark_context_kv_debug_count", 0) < 4
        ):
            if context_slot_mapping.ndim == 1:
                valid_context = context_slot_mapping >= 0
            else:
                valid_context = (context_slot_mapping >= 0).all(dim=-1)
            self._dspark_context_kv_debug_count = (
                getattr(self, "_dspark_context_kv_debug_count", 0) + 1
            )
            logger.warning(
                "[spec_decode/dspark] context kv debug #%d: rows=%d valid_rows=%d "
                "slot_tail=%s",
                self._dspark_context_kv_debug_count,
                int(context_slot_mapping.shape[0]),
                int(valid_context.detach().to(torch.int32).sum().cpu().item()),
                context_slot_mapping[-8:].detach().cpu().tolist(),
            )

        # Keep the cache-update inputs fixed-shape for ACL graph capture.
        # DeviceOperator.reshape_and_cache treats negative slot ids as padding,
        # matching the upstream DFlash graph path.
        context_slot_mapping = context_slot_mapping.to(torch.int32).contiguous()
        for i in range(L):
            attn = self._attn_layers[i]
            kv_cache = attn.kv_cache
            attn.impl.do_kv_cache_update(
                attn,
                all_k_final[i].contiguous(),
                all_v[i].contiguous(),
                kv_cache,
                context_slot_mapping,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)

        hidden_states = input_embeds

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "midlayer." in name:
                name = name.replace("midlayer.", "layers.0.")
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = DFlashQwen3Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        else:
            self.draft_id_to_target_id = None

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if self.draft_id_to_target_id is None:
            return logits

        base = torch.arange(self.config.draft_vocab_size, device=logits.device)
        targets = base + self.draft_id_to_target_id
        logits_new = logits.new_full(
            (logits.shape[0], self.config.vocab_size),
            float("-inf"),
        )
        logits_new[:, targets] = logits
        return logits_new

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None = None,
    ) -> None:
        """Precompute projected + RoPE'd K/V and write to cache."""
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mapping
        )

    def combine_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if not self.model.use_aux_hidden_state:
            return hidden_states
        needs_squeeze = hidden_states.dim() == 1
        if needs_squeeze:
            hidden_states = hidden_states.unsqueeze(0)
        result = self.model.fc(hidden_states)
        if needs_squeeze:
            result = result.squeeze(0)
        return result

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = {}
        includes_draft_id_mapping = False
        includes_embed_tokens = False
        for name, loaded_weight in weights:
            assert "mask_hidden" not in name, (
                "DFlash should use mask_token_id to embed the padding hidden state"
            )
            if "t2d" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif "lm_head" not in name:
                name = "model." + name
            if "embed_tokens" in name:
                includes_embed_tokens = True
            model_weights[name] = loaded_weight
            process_eagle_weight(self, name)

        skip_substrs = []
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        if not self.model.use_aux_hidden_state:
            skip_substrs.append("fc.")
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=None,
            skip_substrs=skip_substrs,
        )
        loader.load_weights(model_weights.items())
        self.model._build_fused_kv_buffers()


class DSparkQwen3ForCausalLM(DFlashQwen3ForCausalLM):
    """DSpark speculators draft model on top of the DFlash Qwen3 backbone.

    The GLM-5.2 DSpark checkpoint uses the speculators DSpark format:
    DFlash backbone weights plus Markov and confidence heads.  vLLM-Ascend's
    DFlash proposer already builds the parallel query hidden states; this class
    adds the DSpark token refinement step.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        target_model_config = (
            getattr(
                vllm_config.speculative_config,
                "target_model_config",
                None,
            )
            or vllm_config.model_config
        )
        target_model_path = getattr(target_model_config, "model", "")
        self._quarot_rotation_path = resolve_quarot_rotation_path(target_model_path)
        if self._quarot_rotation_path is not None:
            logger.info(
                "[spec_decode/dspark] detected target QuaRot rotation: %s",
                self._quarot_rotation_path,
            )
        markov_rank = int(
            getattr(
                self.config,
                "markov_rank",
                getattr(self.config, "dspark_markov_rank", 256),
            )
        )
        self.confidence_head_with_markov = bool(
            getattr(self.config, "confidence_head_with_markov", True)
        )
        self.enable_confidence_head = bool(
            getattr(self.config, "enable_confidence_head", True)
        )
        self.markov_head = DSparkMarkovHead(
            self.config.vocab_size,
            markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
        )
        if self.enable_confidence_head:
            confidence_input_size = self.config.hidden_size
            if self.confidence_head_with_markov:
                confidence_input_size += markov_rank
            self.confidence_head = DSparkConfidenceHead(
                confidence_input_size,
                prefix=maybe_prefix(prefix, "confidence_head"),
            )
        else:
            self.confidence_head = None
        self.last_confidence_logits: torch.Tensor | None = None

    def compute_dspark_draft_tokens(
        self,
        anchor_token_ids: torch.Tensor,
        proposal_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_speculative_tokens, hidden_size = proposal_hidden_states.shape
        base_logits = self.compute_logits(
            proposal_hidden_states.reshape(batch_size * num_speculative_tokens, hidden_size)
        ).view(batch_size, num_speculative_tokens, -1)

        prev_token_ids = anchor_token_ids.long()
        draft_token_ids: list[torch.Tensor] = []
        markov_embeds: list[torch.Tensor] = []
        disable_markov_bias = os.getenv("DSPARK_DISABLE_MARKOV_BIAS", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        debug_logits = (
            os.getenv("DSPARK_LOGITS_DEBUG", "0").lower() in ("1", "true", "yes", "on")
            and get_tensor_model_parallel_rank() == 0
            and getattr(self, "_dspark_logits_debug_count", 0) < 2
            and batch_size > 0
        )
        debug_steps: list[dict[str, object]] = []
        for step_idx in range(num_speculative_tokens):
            markov_logits, markov_embed = self.markov_head(prev_token_ids)
            if disable_markov_bias:
                markov_logits = torch.zeros_like(markov_logits)
            step_logits = base_logits[:, step_idx, :] + markov_logits
            next_token_ids = step_logits.argmax(dim=-1)
            draft_token_ids.append(next_token_ids)
            markov_embeds.append(markov_embed)
            if debug_logits:
                k = min(5, int(step_logits.shape[-1]))
                base_vals, base_ids = torch.topk(base_logits[0, step_idx].float(), k=k)
                markov_vals, markov_ids = torch.topk(markov_logits[0].float(), k=k)
                final_vals, final_ids = torch.topk(step_logits[0].float(), k=k)
                debug_steps.append(
                    {
                        "step": int(step_idx),
                        "prev": int(prev_token_ids[0].detach().cpu().item()),
                        "sampled": int(next_token_ids[0].detach().cpu().item()),
                        "base_ids": [int(x) for x in base_ids.detach().cpu().tolist()],
                        "base_vals": [round(float(x), 4) for x in base_vals.detach().cpu().tolist()],
                        "markov_ids": [int(x) for x in markov_ids.detach().cpu().tolist()],
                        "markov_vals": [round(float(x), 4) for x in markov_vals.detach().cpu().tolist()],
                        "final_ids": [int(x) for x in final_ids.detach().cpu().tolist()],
                        "final_vals": [round(float(x), 4) for x in final_vals.detach().cpu().tolist()],
                    }
                )
            prev_token_ids = next_token_ids

        if self.confidence_head is not None:
            markov_embed_tensor = torch.stack(markov_embeds, dim=1)
            confidence_markov = (
                markov_embed_tensor if self.confidence_head_with_markov else None
            )
            self.last_confidence_logits = self.confidence_head(
                proposal_hidden_states,
                confidence_markov,
            )

        if debug_logits:
            self._dspark_logits_debug_count = getattr(self, "_dspark_logits_debug_count", 0) + 1
            logger.warning(
                "[spec_decode/dspark] logits debug #%d: anchor=%s hidden_shape=%s "
                "base_shape=%s hidden_abs_mean=%.6f disable_markov_bias=%s "
                "draft_first=%s steps=%s",
                self._dspark_logits_debug_count,
                int(anchor_token_ids[0].detach().cpu().item()),
                tuple(proposal_hidden_states.shape),
                tuple(base_logits.shape),
                float(proposal_hidden_states.detach().float().abs().mean().cpu().item()),
                disable_markov_bias,
                [int(x) for x in torch.stack(draft_token_ids, dim=1)[0].detach().cpu().tolist()],
                debug_steps,
            )

        return torch.stack(draft_token_ids, dim=1)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = {}
        includes_draft_id_mapping = False
        includes_embed_tokens = False
        rotation = None
        transformed_quarot_fc = False
        for name, loaded_weight in weights:
            assert "mask_hidden" not in name, (
                "DSpark should use mask_token_id to embed the padding hidden state"
            )
            if name == "fc.weight" and self._quarot_rotation_path is not None:
                if rotation is None:
                    rotation = load_quarot_rotation(self._quarot_rotation_path)
                loaded_weight = transform_fc_weight_for_quarot(
                    loaded_weight,
                    rotation,
                    target_device=self.model.fc.weight.device,
                )
                transformed_quarot_fc = True
                logger.info(
                    "[spec_decode/dspark] transformed fc.weight for QuaRot: "
                    "rotation=%s fc_shape=%s blocks=%d device=%s",
                    self._quarot_rotation_path,
                    tuple(loaded_weight.shape),
                    loaded_weight.shape[1] // rotation.shape[0],
                    loaded_weight.device,
                )
            if "t2d" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif name.startswith(("markov_head.", "confidence_head.")):
                pass
            elif "lm_head" not in name:
                name = "model." + name
            if "embed_tokens" in name:
                includes_embed_tokens = True
            model_weights[name] = loaded_weight
            process_eagle_weight(self, name)

        if self._quarot_rotation_path is not None and not transformed_quarot_fc:
            raise ValueError(
                "Target model uses QuaRot, but the DSpark checkpoint did not "
                "provide fc.weight for the required load-time transformation"
            )

        skip_substrs = []
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        if not self.model.use_aux_hidden_state:
            skip_substrs.append("fc.")
        if not self.enable_confidence_head:
            skip_substrs.append("confidence_head.")
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=None,
            skip_substrs=skip_substrs,
        )
        loader.load_weights(model_weights.items())
        self.model._build_fused_kv_buffers()
