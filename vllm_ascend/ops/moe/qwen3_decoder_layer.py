
from collections.abc import Iterable
from typing import Optional, Union

import torch
from transformers import Qwen3Config
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer

from vllm_ascend.ops.comm_utils import get_sp_metadata_context
from vllm_ascend.ops.layernorm import AddRMSNormW8A8Quant


class AscendQwen3DecoderLayer(Qwen3DecoderLayer):

    def __init__(
        self,
        config: Qwen3Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config,
                         cache_config=cache_config,
                         quant_config=quant_config,
                         prefix=prefix)
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
            residual: Optional[torch.Tensor]
    ) -> torch.Tensor:
        # To prevent precision issues during the decoder phase when only prefilling enables SP
        sp_metadata, enable_sp = get_sp_metadata_context()
        if not enable_sp:
            self.self_attn.o_proj.reduce_results = True
            self.mlp.down_proj.reduce_results = True
        else:
            self.self_attn.o_proj.reduce_results = \
                not sp_metadata.metadata_for_padding.not_dummy_and_is_prefill \
                    if sp_metadata is not None else True
            self.mlp.down_proj.reduce_results = \
                not sp_metadata.metadata_for_padding.not_dummy_and_is_prefill \
                    if sp_metadata is not None else True

        hidden_states, residual = self._forward(hidden_states, positions, residual, sp_metadata)

        if sp_metadata and sp_metadata.metadata_for_padding.not_dummy_and_is_prefill:
            hidden_states = sp_metadata.metadata_for_padding.allgather_unpadding_aligned(
                hidden_states)
        hidden_states = self.mlp(hidden_states)
        if sp_metadata and sp_metadata.metadata_for_padding.not_dummy_and_is_prefill:
            hidden_states = sp_metadata.metadata_for_padding.padding_aligned_reduce_scatter(
                hidden_states)

        return hidden_states, residual

    def _forward(self, hidden_states, positions, residual, sp_metadata):
        # Self Attention
        if residual is None:
            residual = hidden_states
            if sp_metadata and sp_metadata.metadata_for_padding.not_dummy_and_is_prefill:
                residual = sp_metadata.metadata_for_padding.padding_slice(residual)
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
            if sp_metadata and sp_metadata.metadata_for_padding.not_dummy_and_is_prefill:
                hidden_states = sp_metadata.metadata_for_padding.allgather_unpadding_aligned(
                    hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        if sp_metadata and sp_metadata.metadata_for_padding.not_dummy_and_is_prefill:
            hidden_states = sp_metadata.metadata_for_padding.padding_aligned_reduce_scatter(
                hidden_states)
        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return hidden_states, residual