# SPDX-License-Identifier: Apache-2.0
"""Transformers config classes for the downstream DeepSeek V4.1 port."""

from typing import Any

from transformers.configuration_utils import PretrainedConfig


def _mirror_config_aliases(
    values: dict[str, Any],
    aliases: dict[str, str],
) -> dict[str, Any]:
    """Expose both released and pre-release field names, rejecting conflicts."""
    values = dict(values)
    for legacy_name, released_name in aliases.items():
        legacy = values.get(legacy_name)
        released = values.get(released_name)
        if legacy is not None and released is not None and legacy != released:
            raise ValueError(
                f"Conflicting DeepSeek V4.1 config fields: {legacy_name}="
                f"{legacy!r}, {released_name}={released!r}"
            )
        value = released if released is not None else legacy
        if value is not None:
            values[legacy_name] = value
            values[released_name] = value
    return values


class DeepseekV41TextConfig(PretrainedConfig):
    model_type = "deepseek_v41_text"
    base_config_key = "text_config"

    def __init__(self, model_type: str = "deepseek_v41_text", **kwargs: Any) -> None:
        kwargs = _mirror_config_aliases(
            kwargs,
            {
                "kv_source_layers": "kv_source_layer_ids",
                "index_source_layers": "index_source_layer_ids",
                "candidate_source_layer": "candidate_source_layer_id",
                "engram_pad_id": "engram_pad_token_id",
                "dspark_n_activated_experts": "dspark_num_experts_per_tok",
            },
        )
        for name, value in kwargs.items():
            setattr(self, name, value)

        rope = dict(kwargs.get("rope_scaling") or kwargs.get("rope_parameters") or {})
        rope.setdefault("factor", 1.0)
        rope.setdefault("beta_fast", 32)
        rope.setdefault("beta_slow", 1)
        rope.setdefault(
            "original_max_position_embeddings",
            kwargs.get("max_position_embeddings", 1048576),
        )
        rope.setdefault("rope_theta", kwargs.get("rope_theta", 10000.0))
        self.rope_parameters = rope

        base_kwargs = dict(kwargs)
        base_kwargs.pop("rope_scaling", None)
        base_kwargs.pop("rope_parameters", None)
        super().__init__(**base_kwargs)
        self.model_type = model_type

        self.num_hash_layers = int(kwargs.get("num_hash_layers", 0))
        self.n_group = int(kwargs.get("n_group", 1))
        self.topk_group = int(kwargs.get("topk_group", 1))
        self.first_k_dense_replace = int(kwargs.get("first_k_dense_replace", 0))
        self.moe_layer_freq = int(kwargs.get("moe_layer_freq", 1))


class DeepseekV41VisionConfig(PretrainedConfig):
    model_type = "deepseek_v41_vision"
    base_config_key = "vision_config"

    def __init__(self, model_type: str = "deepseek_v41_vision", **kwargs: Any) -> None:
        kwargs = _mirror_config_aliases(
            kwargs,
            {"max_num_tokens": "max_image_tokens"},
        )
        super().__init__(**kwargs)
        self.model_type = model_type
        for name, value in kwargs.items():
            setattr(self, name, value)


class DeepseekV41Config(PretrainedConfig):
    model_type = "deepseek_v41"
    sub_configs = {
        "text_config": DeepseekV41TextConfig,
        "vision_config": DeepseekV41VisionConfig,
    }

    def __init__(
        self,
        text_config: dict[str, Any] | DeepseekV41TextConfig | None = None,
        vision_config: dict[str, Any] | DeepseekV41VisionConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        text_config = text_config or {}
        vision_config = vision_config or {}
        text_field_names = set(text_config) if isinstance(text_config, dict) else set(vars(text_config))
        self.text_config = (
            text_config if isinstance(text_config, DeepseekV41TextConfig) else DeepseekV41TextConfig(**text_config)
        )
        self.vision_config = (
            vision_config
            if isinstance(vision_config, DeepseekV41VisionConfig)
            else DeepseekV41VisionConfig(**vision_config)
        )
        # Include mirrored release/pre-release aliases when flattening the
        # text config for model implementations that consume the root config.
        text_field_names.update(vars(self.text_config))

        text_field_names.update(
            {
                "rope_parameters",
                "num_hash_layers",
                "n_group",
                "topk_group",
                "first_k_dense_replace",
                "moe_layer_freq",
            }
        )
        for name in text_field_names:
            if not name.startswith("_") and name not in {"architectures", "model_type"}:
                setattr(self, name, getattr(self.text_config, name))

        vision_aliases = {
            "vision_n_layers": "num_hidden_layers",
            "vision_dim": "hidden_size",
            "vision_n_heads": "num_attention_heads",
            "vision_inter_dim": "intermediate_size",
            "vision_patch_size": "patch_size",
            "vision_rope_theta": "rope_theta",
            "vision_downsample_ratio": "downsample_ratio",
            "vision_max_n_token": "max_image_tokens",
            "vision_min_pixels": "min_pixels",
            "vision_max_wh_ratio": "max_wh_ratio",
        }
        for alias, source in vision_aliases.items():
            setattr(self, alias, getattr(self.vision_config, source, None))

        # The released checkpoint is still multimodal even though its
        # architecture was renamed from *ForConditionalGeneration to
        # DeepseekV41ForCausalLM. Presence of the populated vision config,
        # rather than the architecture suffix, is the capability signal.
        vision_enabled = bool(self.vision_n_layers)
        self.image_token_id = int(getattr(self, "image_token_id", 129264))
        # V4.1 uses one image token for every span role. The adjacent reserved
        # token is used only for vLLM's ratio-2 compressor-alignment row.
        self.image_sentinel_base_id = self.image_token_id
        self.image_pad_token_id = self.image_token_id + 1
        self.is_mm_prefix_lm = vision_enabled
        self.mm_prefix_clamp_sliding_window = vision_enabled
        self.mm_prefix_span_leading_pad_modulus = 2 if vision_enabled else 0

        # The released W8A8 checkpoint states the Engram basis contract
        # explicitly. The current gate restores the hidden stream to the
        # original basis and adds a value that is already globally rotated.
        # Fail closed for future layouts that require different runtime math.
        rotation = getattr(self, "engram_rotation_config", None)
        if rotation is None:
            rotation = {
                "value_projection_rotated": True,
                "value_basis": "quarot_global",
                "key_and_gate_basis": "original",
                "runtime_delta_rotation": False,
            }
        supported_rotation = {
            "value_projection_rotated": True,
            "value_basis": "quarot_global",
            "key_and_gate_basis": "original",
            "runtime_delta_rotation": False,
        }
        if any(rotation.get(name) != value for name, value in supported_rotation.items()):
            raise ValueError(
                "Unsupported DeepSeek V4.1 Engram rotation contract: "
                f"{rotation!r}"
            )
        self.engram_rotation_config = dict(rotation)
