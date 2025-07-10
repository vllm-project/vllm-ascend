from collections.abc import Iterable
from typing import Any, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F
from transformers import Qwen2Config

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (AutoWeightsLoader, PPMissingLayer, maybe_prefix)

from vllm.model_executor.models.qwen2 import Qwen2Model, Qwen2DecoderLayer
from vllm.distributed import (
                            get_pp_group,
                            get_tensor_model_parallel_world_size,
                            get_tensor_model_parallel_rank,
                            tensor_model_parallel_all_gather,
                            tensor_model_parallel_all_reduce,
                            tensor_model_parallel_reduce_scatter)
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm.forward_context import get_forward_context
import vllm_ascend.envs as ascend_envs


def all_gather_and_maybe_unpad(
    hidden_states: torch.Tensor,
    pad_size: int,
) -> torch.Tensor:
    hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
    if pad_size > 0:
        return hidden_states[:-pad_size, :]
    return hidden_states

def maybe_pad_and_reduce_scatter(
    hidden_states: torch.Tensor,
    pad_size: int,
) -> torch.Tensor:
    if pad_size > 0:
        hidden_states = F.pad(hidden_states, (0, 0, 0, pad_size))
    hidden_states = tensor_model_parallel_reduce_scatter(hidden_states, 0)
    return hidden_states

class CustomQwen2DecoderLayer(Qwen2DecoderLayer):

    def __init__(
        self,
        config: Qwen2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config,
                         cache_config=cache_config,
                         quant_config=quant_config,
                         prefix=prefix)
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.self_attn.o_proj.reduce_results=False
        self.mlp.down_proj.reduce_results=False
        
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        flashcomm_v1_enabled: bool,
        pad_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            if flashcomm_v1_enabled:
                if pad_size > 0:
                    residual = F.pad(residual, (0, 0, 0, pad_size))
                residual = torch.chunk(residual, self.tp_size, dim=0)[self.tp_rank]
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
            if flashcomm_v1_enabled:
                hidden_states = all_gather_and_maybe_unpad(hidden_states, pad_size)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        if flashcomm_v1_enabled:
            hidden_states = maybe_pad_and_reduce_scatter(hidden_states, pad_size)
        else:
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        if flashcomm_v1_enabled:
            hidden_states = all_gather_and_maybe_unpad(hidden_states, pad_size)
        hidden_states = self.mlp(hidden_states)
        if flashcomm_v1_enabled:
            hidden_states = maybe_pad_and_reduce_scatter(hidden_states, pad_size)
        else:
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    })
class CustomQwen2Model(Qwen2Model):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 decoder_layer_type: type[nn.Module] = CustomQwen2DecoderLayer):
        super().__init__(vllm_config=vllm_config,
                         prefix=prefix,
                         decoder_layer_type=decoder_layer_type)
        self.tp_size = get_tensor_model_parallel_world_size()

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
        flashcomm_v1_enabled = False
        attn_metadata = get_forward_context().attn_metadata
        if ascend_envs.VLLM_ASCEND_ENABLE_FLASHCOMM == 1 and \
            attn_metadata is not None and \
            attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
            flashcomm_v1_enabled = True
        if flashcomm_v1_enabled:
            num_tokens = hidden_states.size(0)
            pad_size = (self.tp_size - (num_tokens % self.tp_size)) % self.tp_size
        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                flashcomm_v1_enabled,
                pad_size,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        if flashcomm_v1_enabled:
            hidden_states = all_gather_and_maybe_unpad(hidden_states, pad_size)
        return hidden_states


class CustomQwen2ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    # add `CustomQwen2Model` to init self.model
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
        self.model = CustomQwen2Model(vllm_config=vllm_config,
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
