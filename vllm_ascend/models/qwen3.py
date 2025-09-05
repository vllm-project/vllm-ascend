from collections.abc import Iterable
from typing import Optional, Union

import torch
from torch import nn
from transformers import Qwen3Config
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
import torch.distributed as dist
from vllm.distributed import (get_pp_group,
                              get_context_model_parallel_world_size,
                              get_cp_group)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer
from vllm.model_executor.models.utils import (AutoWeightsLoader, PPMissingLayer, extract_layer_index,
                    is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from vllm_ascend.ops.layernorm import AddRMSNormW8A8Quant
from vllm_ascend.ops.sequence_parallel import init_metadata_for_sp, MetadataForPadding


class CustomQwen3DecoderLayer(Qwen3DecoderLayer):

    def __init__(
        self,
        config: Qwen3Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        vllm_config: Optional[VllmConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config,
                         cache_config=cache_config,
                         quant_config=quant_config,
                         prefix=prefix)
        self.enable_sequence_parallelism = vllm_config.compilation_config.pass_config.enable_sequence_parallelism
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

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            residual: Optional[torch.Tensor],
            _metadata_for_padding: Optional[MetadataForPadding] = None,
    ) -> torch.Tensor:
        # To prevent precision issues during the decoder phase when only prefilling enables SP
        if not self.enable_sequence_parallelism:
            self.self_attn.o_proj.reduce_results = True
            self.mlp.down_proj.reduce_results = True
        else:
            self.self_attn.o_proj.reduce_results = \
                not _metadata_for_padding.not_dummy_and_is_prefill \
                    if _metadata_for_padding is not None else True
            self.mlp.down_proj.reduce_results = \
                not _metadata_for_padding.not_dummy_and_is_prefill \
                    if _metadata_for_padding is not None else True

            # Self Attention
        if residual is None:
            residual = hidden_states
            if _metadata_for_padding and _metadata_for_padding.not_dummy_and_is_prefill:
                residual = _metadata_for_padding.padding_slice(residual)

            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
            if _metadata_for_padding and _metadata_for_padding.not_dummy_and_is_prefill:
                hidden_states = _metadata_for_padding.allgather_unpadding_aligned(
                    hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        if _metadata_for_padding and _metadata_for_padding.not_dummy_and_is_prefill:
            hidden_states = _metadata_for_padding.padding_aligned_reduce_scatter(
                hidden_states)
        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        if _metadata_for_padding and _metadata_for_padding.not_dummy_and_is_prefill:
            hidden_states = _metadata_for_padding.allgather_unpadding_aligned(
                hidden_states)
        hidden_states = self.mlp(hidden_states)
        if _metadata_for_padding and _metadata_for_padding.not_dummy_and_is_prefill:
            hidden_states = _metadata_for_padding.padding_aligned_reduce_scatter(
                hidden_states)

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
        nn.Module.__init__(self)
        self.config = vllm_config.model_config.hf_config
        self.cache_config = vllm_config.cache_config
        self.quant_config = vllm_config.quant_config
        self.vocab_size = self.config.vocab_size

        self.cp_size = get_context_model_parallel_world_size()
        self.cp_group = get_cp_group().device_group

        if get_pp_group().is_first_rank or (self.config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                self.config.hidden_size,
                quant_config=self.quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            self.config.num_hidden_layers,
            lambda prefix: CustomQwen3DecoderLayer(config=self.config,
                                                   cache_config=vllm_config.cache_config,
                                                   quant_config=self.quant_config,
                                                   prefix=prefix,
                                                   vllm_config=vllm_config),
            prefix=f"{prefix}.layers",
        )

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], self.config.hidden_size))
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.aux_hidden_state_layers: tuple[int] = tuple()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            intermediate_tensors: Optional[IntermediateTensors] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            _metadata_for_padding: Optional[MetadataForPadding] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        is_prefill = 0
        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata:
            is_prefill = attn_metadata.num_prefills
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

        aux_hidden_states = []
        for idx, layer in enumerate(
                self.layers[self.start_layer:self.end_layer]):
            if idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states + residual)
            hidden_states, residual = layer(positions,
                                            hidden_states,
                                            residual,
                                            _metadata_for_padding=_metadata_for_padding)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)

        if _metadata_for_padding and _metadata_for_padding.not_dummy_and_is_prefill:
            hidden_states = _metadata_for_padding.allgather_unpadding_aligned(
                hidden_states)

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states

        if self.cp_size > 1 and is_prefill:
            chunk_hidden_states = [torch.empty_like(hidden_states) for _ in range(self.cp_size)]
            dist.all_gather(list(chunk_hidden_states), hidden_states, self.cp_group)
            hidden_states = torch.cat(chunk_hidden_states, dim=0)
            hidden_states = torch.index_select(hidden_states, 0, attn_metadata.prefill.cp_metadata.cp_kv_recover_idx)

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
        self.enable_sequence_parallelism = (
            vllm_config.compilation_config.pass_config.
            enable_sequence_parallelism if vllm_config is not None else False)

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
        _metadata_for_padding = init_metadata_for_sp(input_ids, self.enable_sequence_parallelism)
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
