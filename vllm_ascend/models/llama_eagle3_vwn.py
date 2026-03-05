# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import LlamaConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM, LlamaDecoderLayer

from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    maybe_prefix,
    process_eagle_weight,
)

logger = init_logger(__name__)

class PreVwnLayerV0(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        config: LlamaConfig | None = None
    ) -> None:
        super().__init__()
        config = config or vllm_config.model_config.hf_config
        quant_config = self.get_quant_config(vllm_config)
        hidden_size = config.hidden_size
        m = getattr(config, "m_size", 1)
        expanded_factor = getattr(config, "expanded_factor", 1)
        wider_dim = hidden_size * expanded_factor

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
        config: LlamaConfig | None = None
    ) -> None:
        super().__init__()
        config = config or vllm_config.model_config.hf_config
        quant_config = self.get_quant_config(vllm_config)
        hidden_size = config.hidden_size
        m = getattr(config, "m_size", 1)
        expanded_factor = getattr(config, "expanded_factor", 1)
        wider_dim = hidden_size * expanded_factor
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
        self.upward = ReplicatedLinear(
            input_size=hidden_size,
            output_size=wider_dim,
            bias=False,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "upward"),
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
        wider_hidden_states = self.upward(hidden_after_fc)

        return wider_hidden_states

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
        m = getattr(config, "m_size", 1)
        expanded_factor = getattr(config, "expanded_factor", 1)
        self.hidden_size = config.hiddensize
        wider_dim = self.hidden_size * expanded_factor
        self.wider_dim = wider_dim

        if getattr(config, "eable_pre_vmn", True):
            self.pre_vwn_layer = PreVwnLayerV0(
                vllm_config,
                prefix=maybe_prefix(prefix, f"layers.pre_vwn_layer"),
                config=self.config,
                layer_idx=layer_idx,
            )
        else:
            self.pre_vwn_layer = PreVwnLayerV1(
                vllm_config,
                prefix=maybe_prefix(prefix, f"layers.pre_vwn_layer"),
                config=self.config,
                layer_idx=layer_idx,
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
            input_size=self.hidden_size,
            output_size=wider_dim,
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
            input_size=self.hidden_size,
            output_size=wider_dim,
            bias=False,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "upward_after_mlp"),
            return_bias=False,
        )
        self.downward = ReplicatedLinear(
            input_size=wider_dim,
            output_size=self.hidden_size,
            bias=False,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "downward"),
            return_bias=False,
        )

        self.layer_idx = layer_idx

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
            upward_hidden_states = self.upward_after_attn(hidden_states)
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
            upward_hidden_states = self.upward_after_mlp(hidden_states)
            wider_hidden_states = upward_hidden_states + hidden_residual

            # downward
            hidden_states = self.downward(wider_hidden_states)

        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "hidden_states": 0,
        "input_embeds": 0,
    }
)
class LlamaModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        current_vllm_config = get_current_vllm_config()
        # overwrite layers
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
        hidden_states, hidden_prenorm = self.norm(hidden_states, residual)
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


class Eagle3VwnLlamaForCausalLM(Eagle3LlamaForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)

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
