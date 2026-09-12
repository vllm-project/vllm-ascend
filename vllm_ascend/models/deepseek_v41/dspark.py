# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aurora / DeepSeek-V4.1 dSPark draft model for Ascend."""

import torch
import vllm.envs as envs
from vllm.compilation.decorators import support_torch_compile
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.utils import maybe_prefix

from vllm_ascend.attention.context_parallel.dsa_v41_cp import get_v41_cp_classes
from vllm_ascend.attention.dsa_v41 import DeepseekV41CacheBackend, scatter_cache_sk
from vllm_ascend.core.deepseek_v41 import DeepseekV41DraftSWASpec, validate_cache_runtime
from vllm_ascend.models.deepseek_v4.dspark import (
    DeepseekV4DSparkModel,
    DSparkConfidenceHead,
    DSparkDeepseekV4ForCausalLM,
    DSparkMarkovHead,
    _get_dspark_num_mtp_layers,
)
from vllm_ascend.models.deepseek_v4.model import AscendDeepseekV4SWACache, DeepseekV4Attention
from vllm_ascend.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_shard,
)
from vllm_ascend.models.deepseek_v41.model import (
    DeepseekV41Attention, DeepseekV41DecoderLayer, DeepseekV41LayerRole,
)


class DeepseekV41DSparkSWACache(AscendDeepseekV4SWACache):
    def get_kv_cache_spec(self, vllm_config):
        spec = super().get_kv_cache_spec(vllm_config)
        return DeepseekV41DraftSWASpec(
            block_size=spec.block_size,
            num_kv_heads=spec.num_kv_heads,
            head_size=spec.head_size,
            dtype=spec.dtype,
            sliding_window=spec.sliding_window,
            cache_dtype_str=spec.cache_dtype_str,
            model_version=spec.model_version,
        )

    def get_attn_backend(self):
        return DeepseekV41CacheBackend


class DeepseekV41DSparkAttention(DeepseekV4Attention):
    swa_cache_cls = DeepseekV41DSparkSWACache

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.compress_ratio != 0:
            raise ValueError("Aurora DSpark supports only uncompressed draft SWA layers")
        # V4.1 applies Q LoRA RMSNorm only, without a second per-head Q norm.
        self.dsa_attn.dsa_attn.impl.apply_q_norm = False
        self.softmax_scale = self.scale
        self.shared_state = None
        prefix = kwargs["prefix"]
        self.v41_impl = get_v41_cp_classes()[1](
            prefix=prefix,
            role=DeepseekV41LayerRole(
                layer_idx=int(prefix.split(".")[-2]), compress_ratio=0,
                kv_source_layer=None, index_source_layer=None,
                is_kv_source=False, is_index_source=False,
                is_candidate_source=False, uses_candidate_filter=False, engram_slot=None,
            ),
            topology=None, long_kv_source_prefix=None, index_k_source_prefix=None,
        )
        self.v41_layer_name = f"{prefix}.v41_attn"
        context = kwargs["vllm_config"].compilation_config.static_forward_context
        if self.v41_layer_name in context:
            raise ValueError(f"Duplicate V4.1 attention layer: {self.v41_layer_name}")
        context[self.v41_layer_name] = self

    forward = DeepseekV41Attention.forward


class DeepseekV41DSparkDecoderLayer(DeepseekV41DecoderLayer):
    """V4.1 delayed-mHC block with a draft-only SWA attention backend."""

    attention_cls = DeepseekV41DSparkAttention


class DeepseekV41DSparkModel(DeepseekV4DSparkModel):
    """Three serial draft blocks matching the checkpoint's ``mtp.*`` tree."""

    def __init__(self, *, vllm_config, prefix="") -> None:
        # Deliberately do not call the V4 dSPark constructor: V4.1 has delayed
        # mHC state between blocks and no terminal hc_head parameters.
        torch.nn.Module.__init__(self)
        assert vllm_config.speculative_config is not None
        self.vllm_config = vllm_config
        validate_cache_runtime(vllm_config)
        draft_model_config = vllm_config.speculative_config.draft_model_config
        config = draft_model_config.hf_text_config
        self.config = config
        self.hc_mult = config.hc_mult
        self.hidden_size = config.hidden_size
        self.block_size = int(config.dspark_block_size)
        self.target_layer_ids = list(config.dspark_target_layer_ids)
        self.num_dspark_layers = _get_dspark_num_mtp_layers(config)
        if self.num_dspark_layers != 3:
            raise ValueError("Aurora's DSpark cache group requires exactly three draft layers")
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.use_sequence_parallel = (
            vllm_config.parallel_config.use_sequence_parallel_moe
        )

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.layers = torch.nn.ModuleDict(
            {
                str(self.mtp_start_layer_idx + idx): DeepseekV41DSparkDecoderLayer(
                    vllm_config,
                    prefix=f"mtp.{idx}",
                    config=config,
                    is_draft_layer=True,
                )
                for idx in range(self.num_dspark_layers)
            }
        )

        self.needs_moe_input_ids = any(
            layer.mlp.gate.tid2eid is not None or layer.mlp.gate.bias_vl is not None for layer in self.layers.values()
        )
        first_layer = self.layers[str(self.mtp_start_layer_idx)]
        self.main_proj = ColumnParallelLinear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=None,  # Aurora stores this projection in BF16.
            prefix=maybe_prefix(prefix, f"layers.{self.mtp_start_layer_idx}.main_proj"),
            gather_output=True,
        )
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        first_layer.main_proj = self.main_proj
        first_layer.main_norm = self.main_norm

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        last_layer_idx = self.mtp_start_layer_idx + self.num_dspark_layers - 1
        self.markov_head = DSparkMarkovHead(config, maybe_prefix(prefix, f"layers.{last_layer_idx}.markov_head"))
        self.confidence_head = DSparkConfidenceHead(config, maybe_prefix(prefix, "confidence_head"))
        last_layer = self.layers[str(last_layer_idx)]
        last_layer.norm = self.norm
        last_layer.markov_head = self.markov_head

    def _store_standard_swa_kv(self, shared_kv, slot_mapping, attn=None):
        if slot_mapping is None or slot_mapping.numel() == 0:
            return
        cache = attn.dsa_attn.swa_cache_layer
        if slot_mapping.ndim == 1:
            valid = slot_mapping >= 0
            physical = slot_mapping.clamp_min(0)
            slot_mapping = torch.stack(
                (physical // cache.block_size, physical % cache.block_size), dim=-1
            ).to(torch.int32)
            slot_mapping.masked_fill_(~valid.unsqueeze(-1), -1)
        scatter_cache_sk(cache.kv_cache[0], slot_mapping, shared_kv.squeeze(1))

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids).unsqueeze(-2).repeat(1, self.hc_mult, 1)
        full_num_tokens = positions.shape[0]
        if self.use_sequence_parallel:
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                forward_context = get_forward_context()
                forward_context.is_padding = sp_padding_mask(
                    forward_context.is_padding,
                    hidden_states,
                )
            hidden_states = sp_shard(hidden_states)
            input_ids = sp_shard(input_ids)
        pre_mix = hidden_states.new_zeros(hidden_states.shape[0], self.hc_mult, dtype=torch.float32)
        pre_mix[:, 0] = 1.0
        last_layer = None
        moe_input_ids = input_ids
        if self.needs_moe_input_ids:
            moe_input_ids = torch.where(input_ids == -1, 0, input_ids)
        for layer in self.layers.values():
            last_layer = layer
            hidden_states, pre_mix = layer(
                positions,
                hidden_states,
                pre_mix,
                llama_4_scaling=None,
                input_ids=moe_input_ids,
            )
        assert last_layer is not None
        hidden_states = last_layer.hc_collapse(hidden_states, pre_mix)
        if self.use_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
        return hidden_states


@support_torch_compile
class DSparkDeepseekV41ForCausalLM(DSparkDeepseekV4ForCausalLM):
    def __init__(self, *, vllm_config, prefix="") -> None:
        torch.nn.Module.__init__(self)
        assert vllm_config.speculative_config is not None
        self.config = vllm_config.speculative_config.draft_model_config.hf_text_config

        from vllm_ascend.utils import get_rotation_path

        self.rotation_path = get_rotation_path(vllm_config) if vllm_config.quant_config is not None else None
        self.model = DeepseekV41DSparkModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.set_moe_parameters()

    def _remap_dspark_name(self, name: str) -> str | None:
        mapped = super()._remap_dspark_name(name)
        if mapped is None:
            return None
        # Aurora names the low-rank Markov matrices after their operations,
        # while the runtime uses explicit embedding/projection parameter names.
        mapped = mapped.replace(".markov_head.embed.weight", ".markov_head.markov_w1.weight")
        mapped = mapped.replace(".markov_head.head.weight", ".markov_head.markov_w2.weight")
        mapped = mapped.replace("model.confidence_head.weight", "model.confidence_head.proj.weight")
        return mapped
