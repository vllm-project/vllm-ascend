# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers.configuration_utils import PretrainedConfig


class IQuestQ1Config(PretrainedConfig):
    """Configuration for IQuestQ1's hybrid attention and MoE layers."""

    model_type = "iquest_q1"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 160000,
        hidden_size: int = 3072,
        intermediate_size: int = 1536,
        dense_intermediate_size: int = 12288,
        num_hidden_layers: int = 88,
        num_attention_heads: int = 48,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 524288,
        rms_norm_eps: float = 1e-6,
        num_experts: int = 256,
        num_experts_per_tok: int = 8,
        mlp_only_layers: list[int] | None = None,
        use_hybrid_layers: bool = True,
        first_layers_types: list[str] | None = None,
        hybrid_layers_types_block: list[str] | None = None,
        num_hybrid_layers_block: int = 21,
        last_layers_types: list[str] | None = None,
        sliding_window: int | None = 4096,
        use_sliding_window: bool = True,
        rope_theta: float = 1000000.0,
        swa_rope_theta: float = 10000.0,
        rotary_dim: int | None = 32,
        rope_parameters: dict | None = None,
        no_rope_layers: list[int] | None = None,
        enable_sink_attention: bool = True,
        use_over_encoding: bool = False,
        shared_kv_num_layers: int = 0,
        shared_kv_source_begin: int = 0,
        shared_kv_target_begin: int = 0,
        attn_out_scale: float = 1.0,
        ffn_out_scale: float = 0.53881590608,
        first_layer_attn_out_scale: float = 1.0,
        first_layer_ffn_out_scale: float = 1.0,
        softmax_scale: float | None = None,
        logit_scale: float = 1.0,
        moe_router_dtype: str = "fp32",
        enable_lm_head_fp32: bool = False,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dense_intermediate_size = dense_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.mlp_only_layers = [0] if mlp_only_layers is None else list(mlp_only_layers)
        self.use_hybrid_layers = use_hybrid_layers
        self.first_layers_types = (
            ["full_attention"]
            if first_layers_types is None
            else list(first_layers_types)
        )
        self.hybrid_layers_types_block = (
            [
                "full_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
            ]
            if hybrid_layers_types_block is None
            else list(hybrid_layers_types_block)
        )
        self.num_hybrid_layers_block = num_hybrid_layers_block
        self.last_layers_types = (
            ["full_attention"] * 3
            if last_layers_types is None
            else list(last_layers_types)
        )
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        if use_hybrid_layers:
            layer_types = (
                self.first_layers_types
                + self.hybrid_layers_types_block * num_hybrid_layers_block
                + self.last_layers_types
            )
            if len(layer_types) != num_hidden_layers:
                raise ValueError(
                    "IQuestQ1 hybrid layer pattern has "
                    f"{len(layer_types)} layers, expected {num_hidden_layers}"
                )
            if any(
                t not in ("full_attention", "sliding_attention") for t in layer_types
            ):
                raise ValueError(
                    "IQuestQ1 supports full_attention and sliding_attention layers"
                )
        else:
            layer_types = ["full_attention"] * num_hidden_layers
        if not use_sliding_window:
            layer_types = ["full_attention"] * num_hidden_layers
        self.layer_types = layer_types
        self.rope_theta = rope_theta
        self.swa_rope_theta = swa_rope_theta
        self.rotary_dim = rotary_dim
        self.rope_parameters = dict(rope_parameters or {})
        self.rope_parameters.setdefault("rope_type", "default")
        self.rope_parameters.setdefault("rope_theta", rope_theta)
        if rotary_dim is not None:
            if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
                raise ValueError(
                    "rotary_dim must be even, positive, and at most head_dim"
                )
            self.rope_parameters["partial_rotary_factor"] = rotary_dim / head_dim
        self.no_rope_layers = list(no_rope_layers or [])
        self.enable_sink_attention = enable_sink_attention
        self.use_over_encoding = use_over_encoding
        self.shared_kv_num_layers = shared_kv_num_layers
        self.shared_kv_source_begin = shared_kv_source_begin
        self.shared_kv_target_begin = shared_kv_target_begin
        self.attn_out_scale = attn_out_scale
        self.ffn_out_scale = ffn_out_scale
        self.first_layer_attn_out_scale = first_layer_attn_out_scale
        self.first_layer_ffn_out_scale = first_layer_ffn_out_scale
        self.softmax_scale = softmax_scale
        self.logit_scale = logit_scale
        # GateLinear selects weight/compute dtypes; this controls router logits.
        self.moe_router_dtype = {
            "fp32": "float32",
            "fp16": "float16",
            "bf16": "bfloat16",
        }.get(moe_router_dtype, moe_router_dtype)
        if self.moe_router_dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError(f"Unsupported moe_router_dtype: {moe_router_dtype!r}")
        self.enable_lm_head_fp32 = enable_lm_head_fp32
        kwargs.pop("layer_types", None)
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class IQuestQ1MTPConfig(IQuestQ1Config):
    """Standalone recursive draft with its own attention configuration."""

    model_type = "iquest_q1_mtp"

    def __init__(
        self,
        target_config: dict | None = None,
        num_draft_slots: int = 7,
        sliding_window: int | None = None,
        swa_rope_theta: float | None = None,
        fp32_residual_connection: bool = False,
        enable_lm_head_fp32: bool = False,
        draft_type: str = "eagle3",
        **kwargs,
    ):
        if draft_type != "eagle3":
            raise ValueError("Recursive MTP requires draft_type='eagle3'")
        if kwargs.get("dense_ffn", False) or kwargs.get("eagle3_shared_kv", False):
            raise ValueError("Recursive MTP requires MoE and independent draft KV")
        if kwargs.get("dflash_num_layers", 1) != 1:
            raise ValueError("Recursive MTP supports one physical draft layer")
        if num_draft_slots < 1:
            raise ValueError("num_draft_slots must be positive")
        window = sliding_window or None
        if window is not None and (window < 0 or not swa_rope_theta):
            raise ValueError("A positive draft window requires swa_rope_theta")
        config = dict(target_config or {})
        for key in ("model_type", "architectures", "auto_map", "_name_or_path"):
            config.pop(key, None)
        config.update(kwargs)
        config.update(
            enable_lm_head_fp32=enable_lm_head_fp32,
            num_hidden_layers=1,
            mlp_only_layers=[],
            shared_kv_num_layers=0,
            no_rope_layers=[],
            use_hybrid_layers=True,
            use_sliding_window=window is not None,
            sliding_window=window,
            first_layers_types=[
                "sliding_attention" if window is not None else "full_attention"
            ],
            hybrid_layers_types_block=[],
            num_hybrid_layers_block=0,
            last_layers_types=[],
        )
        if swa_rope_theta is not None:
            config["swa_rope_theta"] = swa_rope_theta
        super().__init__(**config)
        self.target_config = dict(target_config or {})
        self.num_draft_slots = num_draft_slots
        self.fp32_residual_connection = fp32_residual_connection
        self.draft_type = draft_type


__all__ = ["IQuestQ1Config", "IQuestQ1MTPConfig"]
