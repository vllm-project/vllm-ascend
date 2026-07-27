# SPDX-License-Identifier: Apache-2.0
"""Configuration classes for Kimi K3.

Transformers does not yet ship Kimi K3.  Keep the configuration local to the
plugin so loading a K3 checkpoint never depends on executing model code from
the checkpoint repository.
"""

from typing import Any

from transformers.configuration_utils import PretrainedConfig
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig


class KimiK3TextConfig(KimiLinearConfig):
    """Kimi linear-text configuration with the K3 extensions."""

    model_type = "kimi_linear"

    def __init__(
        self,
        mla_use_output_gate: bool = False,
        mla_use_rope: bool = False,
        attn_res_block_size: int | None = None,
        latent_moe_use_norm: bool = False,
        activation_situ_beta: float | None = None,
        activation_situ_linear_beta: float | None = None,
        routed_expert_hidden_size: int | None = None,
        topk_method: str = "noaux_tc",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.mla_use_output_gate = mla_use_output_gate
        # K3 keeps the 64-dimensional q/k positional slice in the checkpoint
        # and in the attention scale, but intentionally does not rotate it.
        self.mla_use_rope = mla_use_rope
        self.attn_res_block_size = attn_res_block_size
        self.latent_moe_use_norm = latent_moe_use_norm
        self.activation_situ_beta = activation_situ_beta
        self.activation_situ_linear_beta = activation_situ_linear_beta
        self.routed_expert_hidden_size = routed_expert_hidden_size
        self.topk_method = topk_method


class KimiK3VisionConfig(PretrainedConfig):
    """MoonViT configuration used by Kimi K3.

    The checkpoint uses ``vt_*`` field names.  The aliases without that prefix
    let the vLLM vision layers consume the config without rewriting checkpoint
    metadata.
    """

    model_type = "kimi_k3_vision"

    def __init__(
        self,
        patch_size: int = 14,
        init_pos_emb_height: int = 64,
        init_pos_emb_width: int = 64,
        init_pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
        vt_num_attention_heads: int = 12,
        vt_num_hidden_layers: int = 27,
        vt_hidden_size: int = 1024,
        vt_intermediate_size: int = 4096,
        merge_kernel_size: tuple[int, int] | list[int] = (2, 2),
        merge_type: str = "sd2_tpool",
        qkv_hidden_size: int = 1536,
        norm_type: str = "rmsnorm",
        attn_bias: bool = False,
        patch_embed_proj_bias: bool = False,
        mlp_type: str = "mlp2",
        linear_bias: bool = False,
        activation_func: str = "gelu_pytorch_tanh",
        pos_emb_interpolation_mode: str = "bilinear",
        mm_projector_type: str = "patchmergerv2",
        mm_hidden_size: int | None = None,
        text_hidden_size: int = 7168,
        projector_hidden_act: str = "gelu",
        projector_ln_eps: float = 1e-5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        self.init_pos_emb_time = init_pos_emb_time
        self.pos_emb_type = pos_emb_type
        self.vt_num_attention_heads = vt_num_attention_heads
        self.vt_num_hidden_layers = vt_num_hidden_layers
        self.vt_hidden_size = vt_hidden_size
        self.vt_intermediate_size = vt_intermediate_size
        self.num_attention_heads = vt_num_attention_heads
        self.num_hidden_layers = vt_num_hidden_layers
        self.hidden_size = vt_hidden_size
        self.intermediate_size = vt_intermediate_size
        self.merge_kernel_size = tuple(merge_kernel_size)
        self.merge_type = merge_type
        self.qkv_hidden_size = qkv_hidden_size
        self.norm_type = norm_type
        self.attn_bias = attn_bias
        self.patch_embed_proj_bias = patch_embed_proj_bias
        self.mlp_type = mlp_type
        self.linear_bias = linear_bias
        self.activation_func = activation_func
        self.pos_emb_interpolation_mode = pos_emb_interpolation_mode
        self.mm_projector_type = mm_projector_type
        self.mm_hidden_size = vt_hidden_size if mm_hidden_size is None else mm_hidden_size
        self.text_hidden_size = text_hidden_size
        self.projector_hidden_act = projector_hidden_act
        self.projector_ln_eps = projector_ln_eps


class KimiK3Config(PretrainedConfig):
    """Top-level multimodal Kimi K3 configuration."""

    model_type = "kimi_k3"
    sub_configs = {
        "text_config": KimiK3TextConfig,
        "vision_config": KimiK3VisionConfig,
    }

    def __init__(
        self,
        text_config: dict[str, Any] | KimiK3TextConfig | None = None,
        vision_config: dict[str, Any] | KimiK3VisionConfig | None = None,
        ignore_index: int = -100,
        media_placeholder_token_id: int = 163605,
        pad_token_id: int = 0,
        use_unified_vision_chunk: bool = True,
        **kwargs: Any,
    ) -> None:
        if text_config is None:
            text_config = KimiK3TextConfig()
        elif isinstance(text_config, dict):
            text_config = KimiK3TextConfig(**text_config)
        if vision_config is None:
            vision_config = KimiK3VisionConfig(text_hidden_size=text_config.hidden_size)
        elif isinstance(vision_config, dict):
            vision_config = KimiK3VisionConfig(**vision_config)

        self.text_config = text_config
        self.vision_config = vision_config
        self.ignore_index = ignore_index
        self.media_placeholder_token_id = media_placeholder_token_id
        # vLLM's OpenAI renderer uses this flag to map standard image/video
        # content onto the unified ``vision_chunk`` modality registered below.
        self.use_unified_vision_chunk = use_unified_vision_chunk

        # Compressed-tensors configuration lives in the nested text config in
        # the released checkpoint, while vLLM discovers it on the top level.
        if getattr(text_config, "quantization_config", None) is not None:
            self.quantization_config = text_config.quantization_config

        super().__init__(pad_token_id=pad_token_id, **kwargs)

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size


def register_kimi_k3_config() -> None:
    """Register K3 with both vLLM's parser and Transformers AutoConfig."""
    from transformers import AutoConfig
    from vllm.transformers_utils import config as vllm_config_module

    vllm_config_module._CONFIG_REGISTRY["kimi_k3"] = KimiK3Config
    AutoConfig.register("kimi_k3", KimiK3Config, exist_ok=True)
