# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import LlamaConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.llama import LlamaForCausalLM, LlamaDecoderLayer
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM, LlamaModel
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.multimodal.inputs import NestedTensors
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    get_draft_quant_config,
    maybe_prefix,
    process_eagle_weight,
)

logger = init_logger(__name__)

class PreVwnLayerV0(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        config: LlamaConfig | None = None,
        quant_config: QuantizationConfig = None,
    ) -> None:
        super().__init__()
        config = config or vllm_config.model_config.hf_config
        hidden_size = config.hidden_size
        m = getattr(config, "vwn_m", 1)
        expanded_factor = getattr(config, "vwn_r", 1)
        wider_dim = int(hidden_size * expanded_factor)

        self.wider_dim = wider_dim
        
        self.upward = ReplicatedLinear(
            input_size=hidden_size,
            output_size=wider_dim,
            bias=False,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "upward"),
            return_bias=False,
        )
        extend_size = 2 * wider_dim
        self.extend = ReplicatedLinear(
            input_size=wider_dim,
            output_size=extend_size,
            bias=False,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "extend"),
            return_bias=False,
        )
        self.input_layernorm = RMSNorm(wider_dim, eps=config.rms_norm_eps)
        self.hidden_norm = RMSNorm(wider_dim, eps=config.rms_norm_eps)
        self.fc = ReplicatedLinear(
            input_size=extend_size,
            output_size=wider_dim,
            bias=False,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "fc"),
            return_bias=False,
        )
        self.full_connected = ReplicatedLinear(
            input_size=wider_dim,
            output_size=wider_dim,
            bias=False,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "extend"),
            return_bias=False,
        )

    def forward(
        self,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor
    ):
        wider_embeds = self.upward(embeds)
        wider_hidden_states = self.upward(hidden_states)

        extended_hidden_states = self.extend(wider_hidden_states)
        hidden_states_to_cat, hidden_residual = torch.split(
            extended_hidden_states,
            [self.wider_dim, extended_hidden_states.shape[-1] - self.wider_dim],
            dim=-1
        )
        norm_wider_embeds = self.input_layernorm(wider_embeds)
        hidden_states_to_cat = self.hidden_norm(hidden_states_to_cat)
        hidden_to_fc = torch.cat([norm_wider_embeds, hidden_states_to_cat], dim=-1)
        hidden_after_fc = self.fc(hidden_to_fc)
        hidden_before_residual = self.full_connected(hidden_after_fc)
        wider_hidden_states = hidden_before_residual + hidden_residual

        return wider_hidden_states

class PreVwnLayerV1(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        config: LlamaConfig | None = None,
        quant_config: QuantizationConfig = None,
    ) -> None:
        super().__init__()
        config = config or vllm_config.model_config.hf_config
        hidden_size = config.hidden_size
        m = getattr(config, "vwn_m", 1)
        expanded_factor = getattr(config, "vwn_r", 1)
        wider_dim = int(hidden_size * expanded_factor)
        self.wider_dim = wider_dim

        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        fc_input_size = 2 * hidden_size
        self.fc = ReplicatedLinear(
            input_size=fc_input_size,
            output_size=hidden_size,
            bias=False,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "fc"),
            return_bias=False,
        )
        upward_input_size = hidden_size // m
        upward_output_size = wider_dim // m
        self.upward = ReplicatedLinear(
            input_size=upward_input_size,
            output_size=upward_output_size,
            bias=False,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "upward"),
            return_bias=False,
        )
        self.m = m
        self.hidden_size = hidden_size
        self.wider_dim = wider_dim

    def forward(
        self,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor
    ):
        norm_embeds = self.input_layernorm(embeds)
        norm_hidden = self.hidden_norm(hidden_states)
        hidden_to_fc = torch.cat([norm_embeds, norm_hidden], dim=-1)
        hidden_after_fc = self.fc(hidden_to_fc)
        hidden_after_fc_new = hidden_after_fc.view(-1 , self.hidden_size // self.m)
        wider_hidden_states_tmp = self.upward(hidden_after_fc_new)
        wider_hidden_states = wider_hidden_states_tmp.view(-1 , self.wider_dim)

        return wider_hidden_states

class PreVwnLayerV2(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        config: LlamaConfig | None = None,
        quant_config: QuantizationConfig = None,
    ) -> None:
        super().__init__()
        config = config or vllm_config.model_config.hf_config
        hidden_size = config.hidden_size

        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        fc_input_size = 2 * hidden_size
        self.fc = ReplicatedLinear(
            input_size=fc_input_size,
            output_size=hidden_size,
            bias=False,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "fc"),
            return_bias=False,
        )

    def forward(
        self,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor
    ):
        norm_embeds = self.input_layernorm(embeds)
        norm_hidden = self.hidden_norm(hidden_states)
        hidden_to_fc = torch.cat([norm_embeds, norm_hidden], dim=-1)
        hidden_after_fc = self.fc(hidden_to_fc)
        return hidden_after_fc


class VwnLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        config: LlamaConfig | None = None,
        layer_idx: int = 0,
    ) -> None:
        super().__init__(vllm_config, prefix=prefix, config=config)

        config = config or vllm_config.model_config.hf_config
        quant_config = self.get_quant_config(vllm_config)
        m = getattr(config, "vwn_m", 1)
        expanded_factor = getattr(config, "vwn_r", 1)
        n = int(expanded_factor * m)
        wider_dim = int(self.hidden_size * expanded_factor)
        self.wider_dim = wider_dim
        self.m = m
        upward_input_size = self.hidden_size // m
        upward_output_size = wider_dim // m
        pre_vwn_version = getattr(config, "pre_vwn_version", 0)
        self.pre_vwn_version = pre_vwn_version
        if pre_vwn_version == 0:
            self.pre_vwn_layer = PreVwnLayerV0(
                vllm_config,
                prefix=maybe_prefix(prefix, f"layers.pre_vwn_layer"),
                config=config,
                quant_config=quant_config
            )
        elif pre_vwn_version == 1:
            self.pre_vwn_layer = PreVwnLayerV1(
                vllm_config,
                prefix=maybe_prefix(prefix, f"layers.pre_vwn_layer"),
                config=config,
                quant_config=quant_config
            )
        else:
            self.pre_vwn_layer = PreVwnLayerV2(
                vllm_config,
                prefix=maybe_prefix(prefix, f"layers.pre_vwn_layer"),
                config=config,
                quant_config=quant_config
            )

        self.downward_and_forgot = ReplicatedLinear(
            input_size=wider_dim,
            output_size=self.hidden_size + wider_dim,
            bias=False,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "downward_and_forgot"),
            return_bias=False,
        )
        
        self.pre_attention_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.upward_after_attn = ReplicatedLinear(
            input_size=upward_input_size,
            output_size=upward_output_size,
            bias=False,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "upward_after_attn"),
            return_bias=False,
        )
        self.downward_and_forgot_after_attn = ReplicatedLinear(
            input_size=wider_dim,
            output_size=self.hidden_size + wider_dim,
            bias=False,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "downward_and_forgot"),
            return_bias=False,
        )
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.upward_after_mlp = ReplicatedLinear(
            input_size=upward_input_size,
            output_size=upward_output_size,
            bias=False,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "upward_after_mlp"),
            return_bias=False,
        )
        self.downward = ReplicatedLinear(
            input_size=upward_output_size,
            output_size=upward_input_size,
            bias=False,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "downward"),
            return_bias=False,
        )

        self.layer_idx = layer_idx

    def get_quant_config(self, vllm_config: VllmConfig) -> QuantizationConfig | None:
        """Use drafter's quantization config instead of verifier's."""
        return get_draft_quant_config(vllm_config)

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.layer_idx == 0:
            # First layer: preVwn + vwn layer
            wider_hidden_states = self.pre_vwn_layer(embeds, hidden_states)

            # attn
            total_hidden_states = self.downward_and_forgot(wider_hidden_states)
            hidden_states, hidden_residual = torch.split(
                total_hidden_states,
                [self.hidden_size, total_hidden_states.shape[-1] - self.hidden_size],
                dim=-1
            )
            hidden_states = self.pre_attention_layernorm(hidden_states)
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
            )
            hidden_states_view = hidden_states.view(-1, self.hidden_size // self.m)
            upward_hidden_states_tmp = self.upward_after_attn(hidden_states_view)
            upward_hidden_states = upward_hidden_states_tmp.view(-1, self.wider_dim)
            wider_hidden_states = upward_hidden_states + hidden_residual

            # mlp
            total_hidden_states = self.downward_and_forgot_after_attn(wider_hidden_states)
            hidden_states, hidden_residual = torch.split(
                total_hidden_states,
                [self.hidden_size, total_hidden_states.shape[-1] - self.hidden_size],
                dim=-1
            )
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states_view = hidden_states.view(-1, self.hidden_size // self.m)
            upward_hidden_states_tmp = self.upward_after_mlp(hidden_states_view)
            upward_hidden_states = upward_hidden_states_tmp.view(-1, self.wider_dim)
            hidden_states = upward_hidden_states + hidden_residual

            # downward
            if self.pre_vwn_version != 2:
                wider_hidden_states_view = hidden_states.view(-1, self.wider_dim // self.m)
                hidden_state_tmp = self.downward(wider_hidden_states_view)
                hidden_states = hidden_state_tmp.view(-1, self.hidden_size)

        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "hidden_states": 0,
        "input_embeds": 0,
    }
)
class VwnLlamaModel(nn.Module):
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

        # Get drafter's quantization config
        self.quant_config = get_draft_quant_config(vllm_config)

        eagle_config = getattr(self.config, "eagle_config", None)
        if eagle_config is not None and "use_aux_hidden_state" in eagle_config:
            self.use_aux_hidden_state = eagle_config["use_aux_hidden_state"]
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
                VwnLlamaDecoderLayer(
                    current_vllm_config,
                    prefix=maybe_prefix(prefix, f"layers.{layer_idx + start_layer_id}"),
                    config=self.config,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        if self.use_aux_hidden_state:
            if hasattr(self.config, "target_hidden_size"):
                fc_input_size = self.config.target_hidden_size * 3
            else:
                fc_input_size = self.config.hidden_size * 3
            self.fc = ReplicatedLinear(
                input_size=fc_input_size,
                output_size=self.config.hidden_size,
                bias=False,
                params_dtype=vllm_config.model_config.dtype,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "fc"),
                return_bias=False,
            )
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)
        assert hidden_states.shape[-1] == input_embeds.shape[-1]

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                embeds=input_embeds,
                hidden_states=hidden_states,
                residual=residual,
            )
        hidden_prenorm = hidden_states
        hidden_states = self.norm(hidden_states, residual)
        return hidden_states, hidden_prenorm

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
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
            # Handle kv cache quantization scales
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = (
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            # Remapping the name FP8 kv-scale or zero point.
            if "scale" in name or "zero_point" in name:
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


class Eagle3VwnLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        # Ensure draft_vocab_size is set
        # default to the base vocab size when absent
        if getattr(self.config, "draft_vocab_size", None) is None:
            base_vocab_size = getattr(self.config, "vocab_size", None)
            self.config.draft_vocab_size = base_vocab_size
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )

        # Store target layer count in draft config for
        # proper layer_types indexing in draft models
        self.config.target_layer_count = target_layer_num
        self.model = VwnLlamaModel(
            vllm_config=vllm_config, prefix="model", start_layer_id=target_layer_num
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
        self.draft_id_to_target_id = nn.Parameter(
            torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
            requires_grad=False,
        )

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
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(input_ids, positions, hidden_states, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if self.draft_id_to_target_id is None:
            assert logits.shape[1] == self.config.vocab_size, (
                "Expected logits to have shape "
                f"(*, {self.config.vocab_size}), but got {logits.shape}"
            )
            return logits

        base = torch.arange(self.config.draft_vocab_size, device=logits.device)
        targets = base + self.draft_id_to_target_id
        logits_new = logits.new_full(
            (
                logits.shape[0],
                self.config.vocab_size,
            ),
            float("-inf"),
        )
        logits_new[:, targets] = logits
        return logits_new

    def combine_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if not self.model.use_aux_hidden_state:
            return hidden_states
        # combine multiple auxiliary hidden states returned by eagle3
        return self.model.fc(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = {}
        includes_draft_id_mapping = False
        includes_embed_tokens = False
        for name, loaded_weight in weights:
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
