# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 MLA DSpark draft model for Ascend."""

from collections.abc import Iterable

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    _merge_multimodal_embeddings,
    get_draft_quant_config,
    maybe_prefix,
)
from vllm.models.kimi_k3.amd.linear import KimiMLP
from vllm.models.kimi_k3.nvidia.dspark_mla import (
    K3DSparkDecoderLayer as UpstreamK3DSparkDecoderLayer,
)
from vllm.models.kimi_k3.nvidia.dspark_mla import (
    K3DSparkForCausalLM as UpstreamK3DSparkForCausalLM,
)
from vllm.models.kimi_k3.nvidia.dspark_mla import (
    K3DSparkModel as UpstreamK3DSparkModel,
)

from vllm_ascend.models.dspark_aux import (
    DSparkAuxHiddenContract,
    build_k3_mla_aux_hidden_contract,
)
from vllm_ascend.models.kimi_k3 import (
    AscendKimiMLAAttention,
)
from vllm_ascend.models.llama_eagle3 import load_quarot_target_layer
from vllm_ascend.models.qwen3_dspark import (
    TARGET_EMBED_WEIGHT_NAMES,
    TARGET_LM_HEAD_WEIGHT_NAMES,
    process_weight,
)
from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_mla
from vllm_ascend.utils import (
    get_rotation_matrix,
    get_rotation_path,
    vllm_version_is,
)


def _uses_causal_draft_attention(config) -> bool:
    dflash_config = getattr(config, "dflash_config", None)
    if isinstance(dflash_config, dict) and "causal" in dflash_config:
        return bool(dflash_config["causal"])
    return bool(getattr(config, "full_attention_causal", False))


class AscendK3DSparkDecoderLayer(UpstreamK3DSparkDecoderLayer):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config,
        layer_idx: int,
        start_layer_id: int,
        prefix: str,
    ) -> None:
        # The upstream constructor hard-codes NVIDIA attention and MLP
        # components. Keep its class contract while constructing the Ascend
        # equivalents below.
        nn.Module.__init__(self)
        quant_config = get_draft_quant_config(vllm_config)
        layer_prefix = maybe_prefix(
            prefix,
            f"layers.{start_layer_id + layer_idx}",
        )
        self.self_attn = AscendKimiMLAAttention(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            use_output_gate=False,
            use_rope=True,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=f"{layer_prefix}.self_attn",
            non_causal_multi_token_decode=not _uses_causal_draft_attention(config),
            disable_mlapo=True,
        )
        self.mlp = KimiMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{layer_prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

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
            hidden_states, residual = self.input_layernorm(
                hidden_states,
                residual,
            )
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states,
            residual,
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class AscendK3DSparkModel(UpstreamK3DSparkModel):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int,
        prefix: str,
    ) -> None:
        # The upstream constructor hard-codes CUDA/NVIDIA attention and decoder
        # classes.  Initialize only the module base, then keep the upstream
        # model methods while constructing the Ascend-specific components.
        nn.Module.__init__(self)
        assert vllm_config.speculative_config is not None
        draft_model_config = vllm_config.speculative_config.draft_model_config
        assert draft_model_config is not None
        self.config = draft_model_config.hf_config
        self.quant_config = get_draft_quant_config(vllm_config)
        self.embed_tokens: nn.Module | None = None

        self.context_proj = ReplicatedLinear(
            self.config.target_hidden_size * self.config.num_target_layers,
            self.config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "context_proj"),
        )
        self.context_norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )
        self.layers = nn.ModuleList(
            [
                AscendK3DSparkDecoderLayer(
                    vllm_config=vllm_config,
                    config=self.config,
                    layer_idx=layer_idx,
                    start_layer_id=start_layer_id,
                    prefix=prefix,
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        self.final_norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )
        self.markov_head = DSparkMarkovHead(
            self.config.vocab_size,
            self.config.draft_vocab_size,
            self.config.markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
        )

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: (
            torch.Tensor | list[torch.Tensor | None] | tuple[torch.Tensor | None, ...] | None
        ) = None,
    ) -> None:
        if context_slot_mapping is None or context_states.numel() == 0:
            return
        per_layer_slot_mapping = isinstance(context_slot_mapping, (list, tuple))
        cos, sin = get_cos_and_sin_mla(context_positions)
        for layer_idx, layer in enumerate(self.layers):
            attn = layer.self_attn
            assert attn.fused_qkv_a_proj is not None
            assert attn.q_lora_rank is not None
            qkv_lora = attn.fused_qkv_a_proj(context_states)[0]
            kv_no_split = qkv_lora[..., attn.q_lora_rank :].contiguous()
            slots = context_slot_mapping[layer_idx] if per_layer_slot_mapping else context_slot_mapping
            if slots is None:
                continue
            attn.impl.exec_kv_prefill(
                kv_no_split,
                cos,
                sin,
                attn.kv_cache,
                slots,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
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


class AscendK3DSparkForCausalLM(UpstreamK3DSparkForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        assert vllm_config.speculative_config is not None
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        assert self.draft_model_config is not None
        self.config = self.draft_model_config.hf_config
        self._aux_hidden_contract = build_k3_mla_aux_hidden_contract(
            self.config,
            vllm_config.model_config.dtype,
        )
        target_layer_num = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        self.model = AscendK3DSparkModel(
            vllm_config=vllm_config,
            start_layer_id=target_layer_num,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head: nn.Module | None = None
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size,
            scale=getattr(self.config, "logit_scale", 1.0),
        )
        self.rotation_path = get_rotation_path(vllm_config)
        self.target_model_path = vllm_config.model_config.model
        if self.rotation_path is not None:
            target_config = vllm_config.model_config.hf_text_config
            model_prefix = maybe_prefix(prefix, "model")
            self.model.embed_tokens = VocabParallelEmbedding(
                target_config.vocab_size,
                target_config.hidden_size,
                prefix=maybe_prefix(model_prefix, "embed_tokens"),
            )
            self.lm_head = ParallelLMHead(
                target_config.vocab_size,
                target_config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )

    def get_draft_attn_causal(self) -> list[bool]:
        causal = _uses_causal_draft_attention(self.config)
        return [causal] * len(self.model.layers)

    def get_required_dspark_aux_hidden_state_contract(
        self,
    ) -> DSparkAuxHiddenContract:
        return self._aux_hidden_contract

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """Load the per-layer KV projections used by the Ascend draft model.

        Upstream additionally duplicates these weights into a CUDA-specific
        cross-layer ``context_kv_proj``.  Ascend deliberately retains the
        quantization-aware per-layer projections, so use vLLM's public loader
        interface without creating that extra packed parameter.
        """
        if vllm_version_is("0.27.1"):
            loader = AutoWeightsLoader(
                self,
                skip_substrs=list(self.checkpoint_skip_substrs),
            )
        else:
            # Current vLLM drops the training-only and shared checkpoint
            # weights in hf_to_vllm_mapper instead of AutoWeightsLoader.
            loader = AutoWeightsLoader(self)
        rotation_weight = None
        if self.rotation_path is not None:
            rotation_weight = get_rotation_matrix(self.rotation_path)
            weights = (
                (
                    name,
                    process_weight(loaded_weight, rotation_weight) if "context_proj." in name else loaded_weight,
                )
                for name, loaded_weight in weights
            )
        loaded_weights = loader.load_weights(
            weights,
            mapper=self.hf_to_vllm_mapper,
        )
        if rotation_weight is not None:
            assert self.model.embed_tokens is not None
            assert self.lm_head is not None
            load_quarot_target_layer(
                self.model.embed_tokens,
                self.target_model_path,
                TARGET_EMBED_WEIGHT_NAMES,
                rotation_weight,
                "draft embed_tokens.weight",
            )
            load_quarot_target_layer(
                self.lm_head,
                self.target_model_path,
                TARGET_LM_HEAD_WEIGHT_NAMES,
                rotation_weight,
                "draft lm_head.weight",
            )
            self.has_own_embed_tokens = True
            self.has_own_lm_head = True
        return loaded_weights

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed draft tokens and replace multimodal placeholder positions.

        vLLM 0.27 passes the target model's precomputed multimodal embeddings
        through the speculative proposer.  K3 DSpark shares the target token
        embedding but upstream still exposes the older text-only method
        signature, so adapt that interface without duplicating the vision
        tower in the draft model.
        """
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0 or is_multimodal is None:
            return self.model.embed_input_ids(input_ids)

        # Placeholder ids are overwritten below.  Mask them before the shared
        # vocabulary lookup so out-of-vocabulary multimodal ids are safe too.
        text_input_ids = input_ids.masked_fill(
            is_multimodal.to(device=input_ids.device, non_blocking=True),
            0,
        )
        inputs_embeds = self.model.embed_input_ids(text_input_ids)
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )
