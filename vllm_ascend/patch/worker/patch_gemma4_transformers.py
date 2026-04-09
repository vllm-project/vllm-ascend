#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from functools import wraps

import torch
from transformers.models.gemma4 import modeling_gemma4
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.transformers import base as transformers_base
from vllm.model_executor.models.transformers import causal as transformers_causal
from vllm.model_executor.models.transformers import multimodal as transformers_multimodal


def _is_gemma4(model_config) -> bool:
    hf_text_config = getattr(model_config, "hf_text_config", None)
    return getattr(hf_text_config, "model_type", None) in {"gemma4", "gemma4_text"}


_orig_create_attention_instances = transformers_base.Base.create_attention_instances
_orig_causal_init = transformers_causal.CausalMixin.__init__
_orig_gemma4_text_attention_forward = modeling_gemma4.Gemma4TextAttention.forward
_orig_gemma4_model_forward = modeling_gemma4.Gemma4Model.forward
_orig_gemma4_text_model_forward = modeling_gemma4.Gemma4TextModel.forward
_orig_init = transformers_base.Base.__init__


def _patch_config(self):
    self.text_config._attn_implementation = "vllm"
    self.config.dtype = torch.get_default_dtype()
    for sub_config_name in getattr(self.config, "sub_configs", {}):
        sub_config = getattr(self.config, sub_config_name, None)
        if sub_config is None:
            continue
        if sub_config.dtype != (dtype := self.config.dtype):
            sub_config.dtype = dtype


def _gemma4_embed_forward(orig_forward, scale: float):
    def forward(input_ids: torch.Tensor):
        outputs = orig_forward(input_ids)
        return outputs * outputs.new_tensor(scale)

    return forward


def _forward_gemma4_text_attention_for_vllm_shared_kv(
    self,
    hidden_states: torch.Tensor,
    *args,
    **kwargs,
):
    position_embeddings = kwargs.pop("position_embeddings", None)
    attention_mask = kwargs.pop("attention_mask", None)
    past_key_values = kwargs.get("past_key_values")
    if position_embeddings is None and len(args) > 0:
        position_embeddings = args[0]
    if attention_mask is None and len(args) > 1:
        attention_mask = args[1]
    if past_key_values is None and len(args) > 2:
        past_key_values = args[2]

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    cos, sin = position_embeddings

    query_states = self.q_proj(hidden_states).view(hidden_shape)
    query_states = self.q_norm(query_states)
    query_states = modeling_gemma4.apply_rotary_pos_emb(
        query_states,
        cos,
        sin,
        unsqueeze_dim=2,
    )
    query_states = query_states.transpose(1, 2)

    key_states = self.k_proj(hidden_states).view(hidden_shape)
    value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states
    key_states = self.k_norm(key_states)
    key_states = modeling_gemma4.apply_rotary_pos_emb(
        key_states,
        cos,
        sin,
        unsqueeze_dim=2,
    )
    value_states = self.v_norm(value_states)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    attention_interface = modeling_gemma4.eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = modeling_gemma4.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=self.attention_dropout if self.training else 0.0,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


@wraps(_orig_gemma4_text_attention_forward)
def _gemma4_text_attention_forward(self, hidden_states, *args, **kwargs):
    if (
        getattr(self, "is_kv_shared_layer", False)
        and kwargs.get("past_key_values") is None
        and getattr(self.config, "_attn_implementation", None) == "vllm"
    ):
        return _forward_gemma4_text_attention_for_vllm_shared_kv(
            self,
            hidden_states,
            *args,
            **kwargs,
        )
    return _orig_gemma4_text_attention_forward(self, hidden_states, *args, **kwargs)


@wraps(_orig_gemma4_text_model_forward)
def _gemma4_text_model_forward(self, *args, **kwargs):
    if kwargs.get("use_cache") is False:
        kwargs["use_cache"] = True
    return _orig_gemma4_text_model_forward(self, *args, **kwargs)


@wraps(_orig_gemma4_model_forward)
def _gemma4_model_forward(
    self,
    input_ids=None,
    pixel_values=None,
    pixel_values_videos=None,
    input_features=None,
    attention_mask=None,
    input_features_mask=None,
    position_ids=None,
    past_key_values=None,
    mm_token_type_ids=None,
    inputs_embeds=None,
    use_cache=None,
    image_position_ids=None,
    video_position_ids=None,
    **kwargs,
):
    if (
        input_ids is not None
        and inputs_embeds is not None
        and pixel_values is None
        and pixel_values_videos is None
        and input_features is None
    ):
        kwargs = dict(kwargs)
        kwargs.pop("return_dict", None)
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        image_mask, video_mask, audio_mask = self.get_placeholder_mask(input_ids=input_ids)
        multimodal_mask = image_mask | video_mask | audio_mask

        if self.config.get_text_config().hidden_size_per_layer_input:
            llm_input_ids = input_ids.clone()
            llm_input_ids[multimodal_mask] = self.config.text_config.pad_token_id
            pad_embedding = self.language_model.embed_tokens.weight[self.config.text_config.pad_token_id, :]
            llm_inputs_embeds = torch.where(
                multimodal_mask[..., None],
                pad_embedding.view(1, 1, -1),
                inputs_embeds,
            )
            per_layer_inputs = self.language_model.get_per_layer_inputs(
                llm_input_ids,
                llm_inputs_embeds,
            )
        else:
            per_layer_inputs = None

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            if self.config.get_text_config().use_bidirectional_attention == "vision":
                causal_mask_mapping = modeling_gemma4.create_causal_mask_mapping(
                    self.config,
                    inputs_embeds,
                    attention_mask,
                    past_key_values,
                    position_ids,
                    mm_token_type_ids,
                    pixel_values,
                    is_training=self.training,
                )
            else:
                causal_mask_mapping = modeling_gemma4.create_masks_for_generate(
                    self.config,
                    inputs_embeds,
                    attention_mask,
                    past_key_values,
                    position_ids,
                )

        outputs = self.language_model(
            per_layer_inputs=per_layer_inputs,
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )

        return modeling_gemma4.Gemma4ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=None,
            audio_hidden_states=None,
        )

    return _orig_gemma4_model_forward(
        self,
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        input_features=input_features,
        attention_mask=attention_mask,
        input_features_mask=input_features_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        mm_token_type_ids=mm_token_type_ids,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        image_position_ids=image_position_ids,
        video_position_ids=video_position_ids,
        **kwargs,
    )


def _init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    if not _is_gemma4(self.model_config):
        return

    # Let the Ascend attention backend own Gemma4 KV sharing. The generic
    # transformers-side shared-layer path does not preserve Gemma4's
    # semantics on Ascend, but the backend can now reuse the correct cache.
    for module in self.model.modules():
        if getattr(module, "is_kv_shared_layer", False):
            module.is_kv_shared_layer = False

    self.requires_raw_input_tokens = True

    input_embeddings = self.model.get_input_embeddings()
    if not isinstance(input_embeddings, VocabParallelEmbedding):
        return

    embed_scale = getattr(self, "embed_scale", None)
    if embed_scale is None or getattr(input_embeddings, "_vllm_ascend_gemma4_scaled", False):
        return

    scale = float(embed_scale.item()) if torch.is_tensor(embed_scale) else float(embed_scale)
    input_embeddings.forward = _gemma4_embed_forward(input_embeddings.forward, scale)
    input_embeddings._vllm_ascend_gemma4_scaled = True
    self.embed_scale = None


def _causal_init(self, *args, **kwargs):
    _orig_causal_init(self, *args, **kwargs)
    if not _is_gemma4(self.model_config):
        return

    logits_processor = getattr(self, "logits_processor", None)
    if logits_processor is None or getattr(logits_processor, "soft_cap", None) is not None:
        return

    logits_processor.soft_cap = getattr(self.text_config, "final_logit_softcapping", None)


def _multimodal_forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors=None,
    inputs_embeds: torch.Tensor | None = None,
    **kwargs,
):
    if _is_gemma4(self.model_config):
        kwargs = {key: value for key, value in kwargs.items() if key in {"token_type_ids", "mm_token_type_ids"}}
        if "token_type_ids" in kwargs and "mm_token_type_ids" not in kwargs:
            kwargs["mm_token_type_ids"] = kwargs["token_type_ids"]
    else:
        kwargs = {key: value for key, value in kwargs.items() if key == "token_type_ids"}

    if self.model_config.uses_mrope:
        positions = positions[:, None]

    return transformers_base.Base.forward(
        self,
        input_ids,
        positions,
        intermediate_tensors,
        inputs_embeds,
        **kwargs,
    )


def _create_attention_instances(self):
    if not _is_gemma4(self.model_config):
        return _orig_create_attention_instances(self)

    text_config = self.text_config
    num_heads = self.model_config.get_num_attention_heads(self.parallel_config)
    logits_soft_cap = getattr(
        text_config,
        "attn_logit_softcapping",
        getattr(text_config, "attention_logit_cap", None),
    )

    is_encoder = lambda module: not getattr(module, "is_causal", True)
    has_encoder = lambda model: any(is_encoder(module) for module in model.modules())
    is_multimodal = lambda config: config != config.get_text_config()
    if has_encoder(self.model) and not is_multimodal(self.config):
        self.check_version("5.0.0", "encoder models support")
        attn_type = transformers_base.AttentionType.ENCODER_ONLY
    else:
        attn_type = transformers_base.AttentionType.DECODER

    layer_types = getattr(self.config, "layer_types", None)
    if layer_types is None:
        layer_types = getattr(text_config, "layer_types", None)
    if layer_types is None:
        raise ValueError("Gemma4 layer_types are required to build heterogeneous attention instances.")

    pp_rank = self.pp_group.rank_in_group
    pp_size = self.pp_group.world_size
    start, end = transformers_base.get_pp_indices(text_config.num_hidden_layers, pp_rank, pp_size)
    num_kv_shared_layers = getattr(text_config, "num_kv_shared_layers", 0)
    first_kv_shared_layer_idx = text_config.num_hidden_layers - num_kv_shared_layers

    attention_instances = {}
    for index in range(start, end):
        per_layer_sliding_window = None
        if layer_types[index] == "sliding_attention":
            per_layer_head_size = text_config.head_dim
            per_layer_sliding_window = text_config.sliding_window
            per_layer_total_num_kv_heads = text_config.num_key_value_heads
        else:
            per_layer_head_size = getattr(text_config, "global_head_dim", None) or text_config.head_dim
            if getattr(text_config, "attention_k_eq_v", False):
                per_layer_total_num_kv_heads = getattr(
                    text_config,
                    "num_global_key_value_heads",
                    text_config.num_key_value_heads,
                )
            else:
                per_layer_total_num_kv_heads = text_config.num_key_value_heads

        per_layer_num_kv_heads = max(
            1,
            per_layer_total_num_kv_heads // self.parallel_config.tensor_parallel_size,
        )

        kv_sharing_target_layer_name = None
        if index >= first_kv_shared_layer_idx > 0:
            prev_layers = layer_types[:first_kv_shared_layer_idx]
            current_layer_type = layer_types[index]
            kv_shared_layer_index = len(prev_layers) - 1 - prev_layers[::-1].index(current_layer_type)
            if kv_shared_layer_index >= 0:
                kv_sharing_target_layer_name = f"{kv_shared_layer_index}.attn"

        attn_cls = (
            transformers_base.EncoderOnlyAttention
            if attn_type == transformers_base.AttentionType.ENCODER_ONLY
            else transformers_base.Attention
        )
        attention_instances[index] = attn_cls(
            num_heads=num_heads,
            head_size=per_layer_head_size,
            # Gemma4 relies on Q/K normalization instead of the usual
            # 1/sqrt(head_dim) attention scaling.
            scale=1.0,
            num_kv_heads=per_layer_num_kv_heads,
            cache_config=self.cache_config,
            quant_config=self.quant_config,
            logits_soft_cap=logits_soft_cap,
            per_layer_sliding_window=per_layer_sliding_window,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            prefix=f"{index}.attn",
            attn_type=attn_type,
        )
    return attention_instances


transformers_base.Base._patch_config = _patch_config
transformers_base.Base.__init__ = _init
transformers_causal.CausalMixin.__init__ = _causal_init
transformers_multimodal.MultiModalMixin.forward = _multimodal_forward
transformers_base.Base.create_attention_instances = _create_attention_instances
modeling_gemma4.Gemma4Model.forward = _gemma4_model_forward
modeling_gemma4.Gemma4TextAttention.forward = _gemma4_text_attention_forward
modeling_gemma4.Gemma4TextModel.forward = _gemma4_text_model_forward
