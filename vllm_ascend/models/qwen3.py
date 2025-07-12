from collections.abc import Iterable
from typing import Optional, Union

import torch
from torch import nn
from transformers import Qwen3Config
from vllm.attention import AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.qwen3 import Qwen3Attention, Qwen3MLP
from vllm.model_executor.models.utils import (AutoWeightsLoader,
                                              PPMissingLayer, maybe_prefix)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from vllm_ascend.ops.layernorm import AddRMSNormW8A8Quant


class CustomQwen3Attention(Qwen3Attention):

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 max_position: int = 4096 * 32,
                 head_dim: Optional[int] = None,
                 rms_norm_eps: float = 1e-06,
                 qkv_bias: bool = False,
                 rope_theta: float = 10000,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 rope_scaling: Optional[tuple] = None,
                 prefix: str = "",
                 attn_type: str = AttentionType.DECODER) -> None:
        super().__init__(hidden_size=hidden_size,
                         num_heads=num_heads,
                         num_kv_heads=num_kv_heads,
                         max_position=max_position,
                         head_dim=head_dim,
                         rms_norm_eps=rms_norm_eps,
                         qkv_bias=qkv_bias,
                         rope_theta=rope_theta,
                         cache_config=cache_config,
                         quant_config=quant_config,
                         rope_scaling=rope_scaling,
                         prefix=prefix,
                         attn_type=attn_type)

    def forward(
        self,
        positions: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Add qk-norm
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim,
                           self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim,
                           self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions,
                               q,
                               k,
                               cos=cos,
                               sin=sin,
                               skip_index_select=True)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class CustomQwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)

        # By default, Qwen3 uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. Alibaba-NLP/gte-Qwen3-7B-instruct)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = CustomQwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=rope_scaling,
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
        if quant_config is None:
            self.input_layernorm = RMSNorm(config.hidden_size,
                                           eps=config.rms_norm_eps)
            self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                    eps=config.rms_norm_eps)
        else:
            from vllm_ascend.quantization.quant_config import AscendQuantConfig
            from vllm_ascend.quantization.w8a8 import AscendW8A8LinearMethod

            assert isinstance(quant_config, AscendQuantConfig), \
                "Expected quant_config to be an instance of AscendQuantConfig"

            if isinstance(self.self_attn.qkv_proj.quant_method.quant_method,
                          AscendW8A8LinearMethod):
                self.input_layernorm = AddRMSNormW8A8Quant(
                    config.hidden_size,
                    layer=self.self_attn.qkv_proj,
                    eps=config.rms_norm_eps)
            else:
                self.input_layernorm = RMSNorm(config.hidden_size,
                                               eps=config.rms_norm_eps)
            if isinstance(self.mlp.gate_up_proj.quant_method.quant_method,
                          AscendW8A8LinearMethod):
                self.post_attention_layernorm = AddRMSNormW8A8Quant(
                    config.hidden_size,
                    layer=self.mlp.gate_up_proj,
                    eps=config.rms_norm_eps)
            else:
                self.post_attention_layernorm = RMSNorm(
                    config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            cos=cos,
            sin=sin,
            hidden_states=hidden_states,
        )
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "attention": CustomQwen3DecoderLayer,
}


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    })
class CustomQwen3Model(Qwen2Model):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config,
                         prefix=prefix,
                         decoder_layer_type=CustomQwen3DecoderLayer)
        self.cos_sin_cache = self.layers[0].self_attn.rotary_emb.cos_sin_cache

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        cos_sin = self.cos_sin_cache.index_select(0, positions)
        last_dim = cos_sin.size()[-1]
        cos, sin = cos_sin.reshape(-1, 2,
                                   last_dim // 2).repeat(1, 1, 2).chunk(2,
                                                                        dim=-2)
        # BSNH
        cos, sin = cos.view(1, -1, 1, last_dim).contiguous(), sin.view(
            1, -1, 1, last_dim).contiguous()

        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states, residual = layer(
                positions,
                cos,
                sin,
                hidden_states,
                residual,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class CustomQwen3ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    # add `CustomQwen3Model` to init self.model
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.model = CustomQwen3Model(vllm_config=vllm_config,
                                      prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(config.vocab_size,
                                              config.hidden_size,
                                              quant_config=quant_config,
                                              prefix=maybe_prefix(
                                                  prefix, "lm_head"))
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
