# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 dense MLA draft model for DSpark speculative decoding (Ascend).

The draft mirrors Kimi-K3's own MLA attention geometry (576-element latent KV
per token) but flips the two K3-target constraints: the draft is trained WITH
YaRN RoPE on the positional slice (``use_mla_rope=True``) and WITHOUT an
output gate (``use_output_gate=False``, no ``g_proj``).

Execution shape (v1 ``AscendDSparkProposer``):
  * ``combine_hidden_states`` projects the target's concatenated aux hidden
    states (5 x 7168) into the draft width (``context_proj`` + ``context_norm``).
  * ``precompute_and_store_context_kv`` writes the projected context straight
    into each draft layer's latent KV cache through the Ascend impl's
    ``exec_kv_prefill`` (fused kv_a_layernorm + YaRN RoPE + paged-cache
    insert). No attention is computed for context tokens.
  * The (per request) 7-token draft block then runs one bidirectional forward:
    the draft attention metadata's ``causal=False`` drives ``sparse_mode=0``
    in ``mla_v1.py``, and the draft attention-group metadata builders are
    flipped to ``use_mla_rope=True`` so the block's q_pe/k_pe receive the same
    YaRN rotations as the precomputed context KV.
"""

import math
from collections.abc import Iterable

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.deepseek_v2 import DeepSeekV2FusedQkvAProjLinear
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper, maybe_prefix

from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_mla
from vllm_ascend.transformers_utils.configs.kimi_k3 import (
    K3_DSPARK_HIDDEN_ACT,
    K3_DSPARK_USE_MLA_ROPE,
    K3DSparkConfig,
)


class K3DSparkMLAAttention(nn.Module):
    """K3 MLA for the DSpark draft: RoPE enabled, output gate disabled.

    Mirrors ``KimiK3MLAAttention`` (vllm_ascend/models/kimi_k3.py) but the
    draft is trained with YaRN RoPE on the q/k positional slice and without
    an output gate. All projection / RoPE / cache / attention work is
    delegated to ``AscendMLAImpl`` through ``MultiHeadLatentAttentionWrapper``
    (OOT-overridden by ``AscendMultiHeadLatentAttention``).
    """

    def __init__(
        self,
        config: K3DSparkConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        cache_config,
        quant_config,
        prefix: str,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        if num_heads % tp_size:
            raise ValueError("num_attention_heads must be divisible by tensor parallel size")
        self.num_local_heads = num_heads // tp_size
        self.scaling = self.qk_head_dim**-0.5

        # Fused q-down + kv-down projection; checkpoint keys ``q_a_proj`` and
        # ``kv_a_proj_with_mqa`` map onto shards 0 and 1 (see WeightsMapper).
        self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
            hidden_size,
            [q_lora_rank, kv_lora_rank + qk_rope_head_dim],
            quant_config=quant_config,
            prefix=f"{prefix}.fused_qkv_a_proj",
        )
        self.q_a_layernorm = RMSNorm(q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            q_lora_rank,
            num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        # No output gate (draft: mla_use_output_gate=False).
        self.o_proj = RowParallelLinear(
            num_heads * v_head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # YaRN RoPE on the positional slice (the draft is trained with RoPE,
        # unlike the NoPE K3 target). Constructing the rotary emb instantiates
        # AscendDeepseekScalingRotaryEmbedding, which records its cos/sin cache
        # globally; get_cos_and_sin_mla reads it during metadata builds and the
        # context-KV precompute. Keep the model default dtype (bf16) exactly as
        # the in-tree DeepSeek MLA path does: the fused
        # npu_kv_rmsnorm_rope_cache / npu_interleave_rope ops are validated
        # against that dtype contract.
        rope_parameters = dict(config.rope_parameters)
        if rope_parameters["rope_type"] == "yarn":
            # Route to DeepseekScalingRotaryEmbedding: the plain YaRN class
            # does not record the MLA cos/sin cache on Ascend.
            rope_parameters["rope_type"] = "deepseek_yarn"
        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=rope_parameters,
            is_neox_style=False,
        )
        if rope_parameters["rope_type"] == "deepseek_yarn":
            scaling_factor = float(rope_parameters["factor"])
            mscale_all_dim = float(rope_parameters["mscale_all_dim"])
            if scaling_factor > 1 and mscale_all_dim:
                mscale = 0.1 * mscale_all_dim * math.log(scaling_factor) + 1.0
                self.scaling = self.scaling * mscale * mscale

        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj,
            kv_a_proj_with_mqa=None,
            q_a_layernorm=self.q_a_layernorm,
            q_b_proj=self.q_b_proj,
            q_proj=None,
            indexer=None,
            is_sparse=False,
            topk_indices_buffer=None,
        )
        # MLAModules is an upstream dataclass without these fields; the Ascend
        # OOT wrapper reads them as dynamic attributes (see KimiK3MLAAttention).
        mla_modules.use_output_gate = False
        mla_modules.use_mla_rope = K3_DSPARK_USE_MLA_ROPE

        self.mla_attn = MultiHeadLatentAttentionWrapper(
            hidden_size,
            self.num_local_heads,
            self.scaling,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            q_lora_rank,
            kv_lora_rank,
            mla_modules,
            cache_config,
            quant_config,
            prefix,
        )
        # ``mla_attn`` is AscendMultiHeadLatentAttention; its ``.mla_attn`` is
        # the inner MLAAttention (owns .impl / .kv_cache / .layer_name).
        self.attn = self.mla_attn.mla_attn

    @property
    def layer_name(self) -> str:
        return self.attn.layer_name

    @property
    def impl(self):
        return self.attn.impl

    @property
    def kv_cache(self):
        return self.attn.kv_cache

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.mla_attn(positions, hidden_states)


class K3DSparkMLP(nn.Module):
    """Dense SwiGLU MLP for the K3 DSpark draft."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config,
        prefix: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported K3 dspark activation: {hidden_act}. Only 'silu' is supported.")
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class K3DSparkDecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: K3DSparkConfig,
        layer_idx: int,
        start_layer_id: int,
        prefix: str,
    ) -> None:
        super().__init__()
        # The K3 dspark draft checkpoint is bf16 (no quantization_config).
        # vllm_config.quant_config carries the target's w4a8 scheme, whose
        # layer descriptions do not cover the draft; inheriting it would break
        # the draft's process_weights_after_loading on bf16 weights. The draft
        # is always bf16, so force quant_config=None.
        quant_config = None
        self.self_attn = K3DSparkMLAAttention(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, f"layers.{start_layer_id + layer_idx}.self_attn"),
        )
        self.mlp = K3DSparkMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=K3_DSPARK_HIDDEN_ACT,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, f"layers.{start_layer_id + layer_idx}.mlp"),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class K3DSparkModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int,
        prefix: str,
    ) -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        draft_model_config = vllm_config.speculative_config.draft_model_config
        if draft_model_config is None:
            raise ValueError("K3 DSpark requires a draft model config.")
        draft_hf_config = draft_model_config.hf_config
        if not isinstance(draft_hf_config, K3DSparkConfig):
            raise TypeError(f"K3DSparkModel requires K3DSparkConfig, got {type(draft_hf_config).__name__}.")
        self.config = draft_hf_config
        # Draft is bf16 (see K3DSparkDecoderLayer); never inherit target quant.
        self.quant_config = None

        # Aliased to the target's embedding by the proposer's
        # _maybe_share_embeddings (has_own_embed_tokens=False).
        self.embed_tokens: nn.Module | None = None

        self.context_proj = ReplicatedLinear(
            self.config.target_hidden_size * self.config.num_target_layers,
            self.config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "context_proj"),
        )
        self.context_norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

        self.layers = nn.ModuleList(
            [
                K3DSparkDecoderLayer(
                    vllm_config=vllm_config,
                    config=self.config,
                    layer_idx=layer_idx,
                    start_layer_id=start_layer_id,
                    prefix=prefix,
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        self.final_norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.markov_head = DSparkMarkovHead(
            self.config.vocab_size,
            self.config.vocab_size,
            self.config.markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
        )

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """main_x = context_norm(context_proj(concat of target aux hidden states)).

        ``hidden_states`` is [T, target_hidden_size * num_target_layers].
        """
        return self.context_norm(self.context_proj(hidden_states))

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        """Embed draft input ids via the shared target embedding.

        The v1 dspark proposer runs text-only speculative decoding even when the
        K3 target is multimodal (it logs "Proceeding with text-only speculative
        decoding" and passes multimodal_embeddings=is_multimodal=None), so the
        SupportsMultiModal kwargs are accepted only to satisfy the call contract.
        ``embed_tokens`` is aliased from the target by _maybe_share_embeddings.
        """
        assert self.embed_tokens is not None
        return self.embed_tokens(input_ids)

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: (
            torch.Tensor | list[torch.Tensor | None] | tuple[torch.Tensor | None, ...] | None
        ) = None,
    ) -> None:
        """Project target-derived context into each draft layer's latent cache.

        Writes through the Ascend impl's ``exec_kv_prefill`` (fused
        kv_a_layernorm + YaRN RoPE + paged-cache insert), using the draft's
        recorded YaRN cos/sin. ``context_slot_mapping`` is a per-layer list
        (the hybrid manager may place draft layers in different cache groups);
        a ``None`` entry (or ``None`` overall, e.g. profiling) writes nothing.
        """
        if context_states.numel() == 0:
            return
        if isinstance(context_slot_mapping, (list, tuple)):
            slot_mappings = tuple(context_slot_mapping)
        else:
            slot_mappings = (context_slot_mapping,) * len(self.layers)
        if len(slot_mappings) != len(self.layers):
            raise ValueError(
                "context_slot_mapping must contain one entry per draft layer: "
                f"got {len(slot_mappings)} entries for {len(self.layers)} layers"
            )
        cos, sin = get_cos_and_sin_mla(context_positions)
        for layer_idx, layer in enumerate(self.layers):
            attn = layer.self_attn
            slot_mapping = slot_mappings[layer_idx]
            if slot_mapping is None:
                continue
            qkv_lora = attn.fused_qkv_a_proj(context_states)[0]
            # kv part only (drop the q_lora slice); raw, pre-norm -- the Ascend
            # impl applies kv_a_layernorm + RoPE inside exec_kv_prefill.
            kv_no_split = qkv_lora[..., attn.q_lora_rank :].contiguous()
            attn.impl.exec_kv_prefill(
                kv_no_split,
                cos,
                sin,
                attn.kv_cache,
                slot_mapping,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            assert self.embed_tokens is not None
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        hidden_states, _ = self.final_norm(hidden_states, residual)
        return hidden_states

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        # Inner MLAAttention.layer_name (e.g. "model.layers.N.self_attn.attn")
        # -- the key used by the proposer for all_attn_layers / kv-cache groups
        # / draft attn_metadata, and by mla_forward's attn_metadata lookup.
        return [layer.self_attn.attn.layer_name for layer in self.layers]

    def get_draft_attn_causal(self) -> list[bool]:
        # Older checkpoints omit the flag and use bidirectional attention.
        causal = getattr(self.config, "full_attention_causal", False)
        dflash_config = getattr(self.config, "dflash_config", None)
        if dflash_config is not None:
            causal = dflash_config.get("causal", causal)
        return [bool(causal)] * len(self.layers)

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.markov_head.bias(markov_embed, self.logits_processor)


class K3DSparkForCausalLM(nn.Module):
    # Draft shares the target's embedding / lm_head: the checkpoint carries a
    # frozen copy of the target embedding and no lm_head at all.
    has_own_embed_tokens = False
    has_own_lm_head = False
    # Full-vocab draft: draft ids are target ids, no remapping.
    draft_id_to_target_id = None
    # confidence_head is training-only; embed_tokens / lm_head are shared from
    # the target, so skip any unloaded copies in the checkpoint.
    checkpoint_skip_substrs = ("confidence_head", "embed_tokens", "lm_head")

    # Checkpoint keys are in the model namespace without the ``model.`` prefix
    # and with separate q_a_proj / kv_a_proj_with_mqa / gate_proj / up_proj;
    # remap to the fused module names and prefix with ``model.``.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"": "model."},
        orig_to_new_stacked={
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
            ".q_a_proj": (".fused_qkv_a_proj", 0),
            ".kv_a_proj_with_mqa": (".fused_qkv_a_proj", 1),
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        if self.draft_model_config is None:
            raise ValueError("K3 DSpark requires a draft model config.")
        draft_hf_config = self.draft_model_config.hf_config
        if not isinstance(draft_hf_config, K3DSparkConfig):
            raise TypeError(f"K3DSparkForCausalLM requires K3DSparkConfig, got {type(draft_hf_config).__name__}.")
        self.config = draft_hf_config
        # Start draft layer names after the target's (93 for K3) so the MLA
        # attention layers register under non-colliding static_forward_context
        # keys.
        target_layer_num = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        self.model = K3DSparkModel(
            vllm_config=vllm_config,
            start_layer_id=target_layer_num,
            prefix=maybe_prefix(prefix, "model"),
        )

        # Aliased from the target by _maybe_share_lm_head; keep no placeholder
        # module to avoid a transient full-vocabulary allocation (163k vocab).
        self.lm_head: nn.Module | None = None
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        # The Markov head's bias() needs the logits_processor; thread it through.
        self.model.logits_processor = self.logits_processor

    # --- Hooks used by the v1 dspark proposer ------------------------------

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(hidden_states)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids, multimodal_embeddings, is_multimodal)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return self.model.get_draft_kv_cache_layer_names()

    def get_draft_attn_causal(self) -> list[bool]:
        return self.model.get_draft_attn_causal()

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: (
            torch.Tensor | list[torch.Tensor | None] | tuple[torch.Tensor | None, ...] | None
        ) = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(context_states, context_positions, context_slot_mapping)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.lm_head is not None
        return self.logits_processor(self.lm_head, hidden_states)

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_bias(markov_embed)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_substrs=list(self.checkpoint_skip_substrs),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
