from collections.abc import Iterable
from typing import Optional, Union

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers import Qwen3Config
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.attention import Attention, AttentionType
from vllm.distributed import (get_pp_group, 
                              get_tensor_model_parallel_world_size,
                              get_tensor_model_parallel_rank,
                              tensor_model_parallel_all_gather)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.layers.linear import RowParallelLinear, ReplicatedLinear
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.qwen2 import Qwen2MLP as Qwen3MLP
from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer, Qwen3Attention
from vllm.model_executor.models.utils import (AutoWeightsLoader,
                                              PPMissingLayer, maybe_prefix)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from vllm_ascend import envs
from vllm_ascend.ops.layernorm import AddRMSNormW8A8Quant


def pad(tensor, x):
    length = tensor.size(0)
    pad_size = (x - (length % x)) % x
    if pad_size > 0:
        return F.pad(tensor, (0, 0, 0, pad_size)), pad_size
    return tensor, pad_size

def unpad(tensor, pad_size):
    if pad_size > 0:
        return tensor[:-pad_size, :]
    return tensor


class CustomQwen3MLP(Qwen3MLP):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(hidden_size=hidden_size,
                         intermediate_size=intermediate_size,
                         hidden_act=hidden_act,
                         quant_config=quant_config,
                         prefix=prefix)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.enable_fc = envs.VLLM_ENABLE_FC
        if self.enable_fc:
            # if flashcomm2 enbaled, replace Linear+AllReduce with All2All+Linear
            self.down_proj = ReplicatedLinear(
                intermediate_size,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.down_proj",
            )
        else:
            self.down_proj = RowParallelLinear(
                intermediate_size,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.down_proj",
            )

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        pad_size = 0
        if self.enable_fc:
            # pad input because AllGather requires token_num to be divisible by tp_size
            x, pad_size = pad(x, self.tp_size)
            output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
            dist.all_to_all_single(output, x)
            x = output.reshape(self.tp_size, -1, output.size(-1)) \
                        .transpose(0, 1) \
                        .reshape(-1, output.size(-1)*self.tp_size)
        x, _ = self.down_proj(x)
        return x, pad_size


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
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.enable_fc = envs.VLLM_ENABLE_FC
        if self.enable_fc:
            self.o_proj = ReplicatedLinear(
                self.total_num_heads * self.head_dim,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.o_proj",
            )
        else:
            self.o_proj = RowParallelLinear(
                self.total_num_heads * self.head_dim,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.o_proj",
            )

    def forward(
        self,
        positions: torch.Tensor,
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
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        pad_size = 0
        if self.enable_fc:
            # pad input because AllGather requires token_num to be divisible by tp_size
            attn_output, pad_size = pad(attn_output, self.tp_size)
            output = torch.empty(attn_output.shape, dtype=attn_output.dtype, device=attn_output.device)
            dist.all_to_all_single(output, attn_output)
            attn_output = output.reshape(self.tp_size, -1, output.size(-1)) \
                                .transpose(0, 1) \
                                .reshape(-1, output.size(-1)*self.tp_size)
        output, _ = self.o_proj(attn_output)
        return output, pad_size


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
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.enable_fc = envs.VLLM_ENABLE_FC
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
        self.mlp = CustomQwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        if quant_config is None:
            return

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
        if isinstance(self.mlp.gate_up_proj.quant_method.quant_method,
                      AscendW8A8LinearMethod):
            self.post_attention_layernorm = AddRMSNormW8A8Quant(
                config.hidden_size,
                layer=self.mlp.gate_up_proj,
                eps=config.rms_norm_eps)

    def pre_attention_process(self, hidden_states, residual, pad_size=0):
        hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
        hidden_states = unpad(hidden_states, pad_size)
        return hidden_states, residual
    
    def pre_mlp_process(self, hidden_states, residual, pad_size=0):
        token_num = hidden_states.size(0)
        if token_num != residual.size(0):
            if pad_size > 0:
                residual = F.pad(residual, (0, 0, 0, pad_size))
            split_size_list = [token_num] * self.tp_size
            residual = torch.split(residual, split_size_list)[self.tp_rank]

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
        hidden_states = unpad(hidden_states, pad_size)
        return hidden_states, residual

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        pad_size: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            if self.enable_fc:
                hidden_states, residual = self.pre_attention_process(hidden_states, residual, pad_size)
            else:
                hidden_states, residual = self.input_layernorm(
                    hidden_states, residual)
        hidden_states, pad_size = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        if self.enable_fc:
            hidden_states, residual = self.pre_mlp_process(hidden_states, residual, pad_size)
        else:
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)
        hidden_states, pad_size = self.mlp(hidden_states)
        return hidden_states, residual, pad_size
    

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
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.enable_fc = envs.VLLM_ENABLE_FC

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
        pad_size = 0
        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states, residual, pad_size = layer(
                positions,
                hidden_states,
                residual,
                pad_size
            )
        if self.enable_fc:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            residual = tensor_model_parallel_all_gather(residual, 0)
            if pad_size > 0:
                hidden_states = hidden_states[:-pad_size]
                residual = residual[:-pad_size]
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
