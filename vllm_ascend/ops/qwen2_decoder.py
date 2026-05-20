from collections.abc import Callable

import torch
import torch.nn as nn
import torch_npu
from transformers import Qwen2Config
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2Model,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from vllm.model_executor.models.deepencoder2 import CustomQwen2Decoder


class AscendCustomQwen2Decoder(CustomQwen2Decoder):
    def __init__(
        self,
        decoder_layer: int = 24,
        max_position_embeddings: int = 131072,
        hidden_dimension: int = 896,
        num_attention_heads: int = 14,
        num_key_value_heads: int = 2,
        intermediate_size: int = 4864,
        vocab_size: int = 151936,
        attn_implementation: str = "sdpa",
        rms_norm_eps: float = 1e-06,
        rope_theta: float = 1000000.0,
        attention_dropout: float = 0.0,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
    ):
        super().__init__(
            decoder_layer,
            max_position_embeddings,
            hidden_dimension,
            num_attention_heads,
            num_key_value_heads,
            intermediate_size,
            vocab_size,
            attn_implementation,
            rms_norm_eps,
            rope_theta,
            attention_dropout,
            hidden_act,
            initializer_range,
        )
        # config
        config = Qwen2Config(
            hidden_size=hidden_dimension,
            num_hidden_layers=decoder_layer,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            vocab_size=vocab_size,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            attention_dropout=attention_dropout,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            _attn_implementation=attn_implementation,
        )
        self.model = self._create_optimized_custom_model(config)
        del self.model.embed_tokens

    def _create_optimized_custom_model(self, config):
        class CustomQwen2ModelInner(AscendQwen2Model):
            def forward(
                self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                use_cache=None,
                cache_position=None,
                token_type_ids=None,
                **kwargs: Unpack[TransformersKwargs],
            ):
                # token_type_ids
                self._current_token_type_ids = token_type_ids
                causal_mask_mapping = {
                    "full_attention": self._create_npu_optimized_mask(
                        attention_mask=attention_mask,
                        input_tensor=inputs_embeds,
                        token_type_ids=token_type_ids,
                    )
                }
                outputs = super().forward(
                    input_ids=input_ids,
                    attention_mask=causal_mask_mapping,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

                return outputs

            def _create_npu_optimized_mask(
                self,
                attention_mask,
                input_tensor,
                token_type_ids,
            ):
                """
                4D Mask generation optimized for NPU
                vector parallel implementation, replacing the original loop implementation
                """
                dtype, device = input_tensor.dtype, input_tensor.device
                min_dtype = torch.finfo(dtype).min
                batch_size, sequence_length = input_tensor.shape[0], input_tensor.shape[1]

                if token_type_ids is None:
                    return self._create_standard_causal_mask(batch_size, sequence_length, dtype, device, attention_mask)
                # ==========================================
                # NPU optimization: vectorized parallel mask generation (replacing loops)
                # ==========================================
                # 1. create image token position mask [batch, seq_len]
                is_image = (token_type_ids == 0).unsqueeze(-1).to(dtype=dtype, device=device)  # [batch, seq_len, 1]
                is_text = (token_type_ids == 1).unsqueeze(-1).to(dtype=dtype, device=device)  # [batch, seq_len, 1]
                # 2. Bidirectional attention (fully connected) between image tokens.
                # image_attention: [batch, seq_len, seq_len]
                image_attention = torch.bmm(is_image, is_image.transpose(1, 2))
                # 3. Visibility of text tokens to image tokens (full connection)
                text_to_image = torch.bmm(is_text, is_image.transpose(1, 2))
                # 4. Causal attention from text to text.
                # First, perform matrix multiplication to obtain the text-text relationship pairs of [B, L, L].
                text_to_text_base = torch.bmm(is_text, is_text.transpose(1, 2))
                # create casual triangular Lower
                causal_mask = torch.tril(torch.ones((sequence_length, sequence_length), device=device, dtype=dtype))
                text_to_text = text_to_text_base * causal_mask.unsqueeze(0)
                # 5. Merge all attention patterns
                combined_mask = image_attention + text_to_image + text_to_text
                combined_mask = (1 - combined_mask) * min_dtype  # reverse：0->min_dtype, 1->0
                # 6. Process Padding Mask (attention_mask)
                if attention_mask is not None:
                    # Ensure that padding_mask is on the same device
                    p_mask = attention_mask.to(device=device, dtype=dtype)
                    if p_mask.dim() == 2:
                        # Extended to [B, 1, 1, L] to adapt to 4D attention.
                        p_mask = (1.0 - p_mask[:, None, None, :]) * min_dtype
                        return combined_mask.unsqueeze(1) + p_mask
                return combined_mask.unsqueeze(1)

            def _create_standard_causal_mask(self, batch_size, seq_len, dtype, device, attention_mask):
                """Standard causal mask (when token_type_ids is None)"""
                min_dtype = torch.finfo(dtype).min
                mask = torch.triu(torch.full((seq_len, seq_len), min_dtype, dtype=dtype, device=device), diagonal=1)
                mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
                if attention_mask is not None and attention_mask.dim() == 2:
                    padding_mask = attention_mask[:, None, None, :].to(dtype=dtype)
                    padding_mask = (1.0 - padding_mask) * min_dtype
                    mask = mask + padding_mask
                return mask

        return CustomQwen2ModelInner(config)


class AscendQwen2Model(Qwen2Model):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [AscendQwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = AscendQwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class AscendQwen2DecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = AscendQwen2Attention(config=config, layer_idx=layer_idx)
        self.input_layernorm = AscendQwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = AscendQwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def optimized_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    return q_embed, k_embed


class AscendQwen2Attention(Qwen2Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = optimized_apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # main diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class AscendQwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]
