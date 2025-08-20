from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn
from transformers import Gemma3TextConfig
from vllm_ascend.ops.layernorm import AddRMSNormW8A8Quant, AscendRMSNorm
from vllm_ascend.quantization.w8a8 import AscendW8A8LinearMethod

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import \
    VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.models.gemma3 import (Gemma3Attention,
                                               Gemma3DecoderLayer,
                                               Gemma3ForCausalLM, Gemma3MLP,
                                               Gemma3Model)
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (
    is_pp_missing_parameter, make_empty_intermediate_tensors_factory,
    make_layers, maybe_prefix)


class AscendGemma3DecoderLayer(Gemma3DecoderLayer):
    def __init__(
        self,
        config: Gemma3TextConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = Gemma3Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            attn_logits_soft_cap=None,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Gemma3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_activation=config.hidden_activation,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        self.input_layernorm = AscendRMSNorm(config.hidden_size,
                                             eps=config.rms_norm_eps)
        self.post_attention_layernorm = AscendRMSNorm(config.hidden_size,
                                                      eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = AscendRMSNorm(config.hidden_size,
                                                       eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = AscendRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)

        if isinstance(self.self_attn.qkv_proj.quant_method.quant_method,
                      AscendW8A8LinearMethod):
            self.input_layernorm = AddRMSNormW8A8Quant(
                config.hidden_size,
                layer=self.self_attn.qkv_proj,
                eps=config.rms_norm_eps)
        if isinstance(self.mlp.gate_up_proj.quant_method.quant_method,
                      AscendW8A8LinearMethod):
            self.pre_feedforward_layernorm = AddRMSNormW8A8Quant(
                config.hidden_size,
                layer=self.mlp.gate_up_proj,
                eps=config.rms_norm_eps)


@support_torch_compile
class AscendGemma3Model(Gemma3Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        cache_config = vllm_config.cache_config
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            self.config.num_hidden_layers,
            lambda prefix: AscendGemma3DecoderLayer(
                self.config, cache_config, self.quant_config, prefix=prefix),
            prefix=f"{prefix}.layers")
        self.norm = AscendRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

        # Normalize the embedding by sqrt(hidden_size)
        # The normalizer's data type should be downcasted to the model's
        # data type such as bfloat16, not float32.
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = self.config.hidden_size**0.5
        self.register_buffer("normalizer",
                             torch.tensor(normalizer),
                             persistent=False)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], self.config.hidden_size))

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            # Add 1 to GemmaRMSNorm weights after loading
            if "norm.weight" in name or "layernorm.weight" in name:
                loaded_weight = loaded_weight + 1.0
            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                # Loading kv cache scales for compressed-tensors quantization
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = loaded_weight[0]
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            for (param_name, shard_name, shard_id) in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params


class AscendGemma3ForCausalLM(Gemma3ForCausalLM):
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
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        nn.Module.__init__(self)
        SupportsLoRA.__init__(self)
        SupportsPP.__init__(self)
        # currently all existing Gemma models have `tie_word_embeddings` enabled
        assert self.config.tie_word_embeddings
        self.model = AscendGemma3Model(vllm_config=vllm_config,
                                       prefix=maybe_prefix(prefix, "model"))
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size,
            soft_cap=self.config.final_logit_softcapping)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)
