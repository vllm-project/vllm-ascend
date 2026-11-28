# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expose Kimi K3.1's nested DSA schema through the existing Kimi config."""

from vllm.config import VllmConfig
from vllm.model_executor.models.config import (
    MODELS_CONFIG_MAP,
    KimiK3ForConditionalGenerationConfig,
    VerifyAndUpdateConfig,
)
from vllm.transformers_utils.config import _CONFIG_REGISTRY
from vllm.transformers_utils.configs.kimi_k3 import KimiK3Config
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig


class AscendKimiModelConfig(KimiK3ForConditionalGenerationConfig):
    @classmethod
    def verify_and_update_config(cls, vllm_config: VllmConfig) -> None:
        # #66037's Kimi hook selects GPU sparse/dense MLA cache formats and
        # defaults MLA drafts to FLASHINFER_MLA + fp8, including on A2/A3.
        # Ascend owns backend/cache selection in NPUPlatform and its attention
        # layers. Keep explicit settings and the inherited quantization hook.
        VerifyAndUpdateConfig.verify_and_update_config(vllm_config)


for architecture in ("KimiK3ForConditionalGeneration", "KimiK3MTPModel", "KimiK31ForConditionalGeneration"):
    MODELS_CONFIG_MAP[architecture] = AscendKimiModelConfig


class KimiK31TextConfig(KimiLinearConfig):
    # Preserve K3.1 identity when with_hf_config enters the kimi_linear model.
    is_kimi_k31 = True

    def __init__(self, dsa_config: dict | None = None, **kwargs):
        # Native upstream owns the dataclass consumed by the composite config.
        super().__init__(dsa_config=dsa_config, **kwargs)
        dsa_config = super().to_dict().get("dsa_config")
        if dsa_config is None:
            return
        self.index_topk = dsa_config["index_topk"]
        self.index_n_heads = dsa_config["index_num_heads"]
        self.index_head_dim = dsa_config["index_head_dim"]
        self.index_rope_dim = dsa_config.get("index_rope_dim", 0)
        self.index_topk_freq = dsa_config.get("index_topk_freq", 1)
        self.index_skip_topk_offset = dsa_config.get("index_skip_topk_offset", 1)
        if dsa_config.get("index_k_norm_type", "layernorm") != "layernorm":
            raise ValueError("Kimi K3.1 DSA requires a LayerNorm indexer key projection")
        if not 0 <= self.index_rope_dim <= self.index_head_dim:
            raise ValueError("DSA index_rope_dim must be within the indexer head dimension")
        dsa_layers = set(dsa_config["dsa_layers"])
        if any(layer < 1 or layer > self.num_hidden_layers for layer in dsa_layers):
            raise ValueError("DSA layer indices are one-based and must be inside the backbone")
        if self.linear_attn_config is not None:
            if dsa_layers != set(self.linear_attn_config["full_attn_layers"]):
                raise ValueError("Kimi K3.1 DSA layers must match the full attention layers")
            if dsa_layers.intersection(self.linear_attn_config["kda_layers"]):
                raise ValueError("Kimi K3.1 KDA and DSA layers must be disjoint")

    def is_dsa_layer(self, layer_idx: int) -> bool:
        return super().is_dsa_layer(layer_idx)


class KimiK31Config(KimiK3Config):
    model_type = "kimi_k31"

    def __init__(self, text_config=None, **kwargs):
        if isinstance(text_config, dict):
            text_config = KimiK31TextConfig(**text_config)
        super().__init__(text_config=text_config, **kwargs)


_CONFIG_REGISTRY["kimi_k31"] = KimiK31Config
