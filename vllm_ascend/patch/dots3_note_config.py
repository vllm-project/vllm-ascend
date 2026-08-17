# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any

from transformers import DeepseekV2Config, PretrainedConfig
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict


class Dots3NoteVisionConfig(PretrainedConfig):
    model_type = "dots3_note_vision"

    def __init__(
        self,
        embed_dim: int = 1536,
        hidden_size: int = 5120,
        intermediate_size: int = 4224,
        moe_intermediate_size: int = 2112,
        num_hidden_layers: int = 42,
        num_attention_heads: int = 24,
        num_channels: int = 3,
        patch_size: int = 14,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 1,
        rms_norm_eps: float = 1e-5,
        use_bias: bool = False,
        use_qk_norm: bool = True,
        attn_implementation: str = "flash_attention_2",
        post_norm: bool = True,
        pyramid_num_routed: list[int] | None = None,
        capacity_factor: float = 2.0,
        router_scoring_func: str = "sigmoid",
        router_scale: float = 1.0,
        adapter_in_dim: int = 1536,
        adapter_out_dim: int = 5120,
        adapter_merge_size: int = 2,
        adapter_type: str = "patch_merger",
        pre_pixel_shuffle: bool = True,
        enable_torch_compile: bool = False,
        enable_fp8_moe: bool = False,
        **kwargs: Any,
    ) -> None:
        if adapter_type != "patch_merger":
            raise ValueError(f"Unsupported Dots3 Note vision adapter_type: {adapter_type!r}")
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.rms_norm_eps = rms_norm_eps
        self.use_bias = use_bias
        self.use_qk_norm = use_qk_norm
        self.attn_implementation = attn_implementation
        self.post_norm = post_norm
        self.pyramid_num_routed = pyramid_num_routed or []
        self.capacity_factor = capacity_factor
        self.router_scoring_func = router_scoring_func
        self.router_scale = router_scale
        self.adapter_in_dim = adapter_in_dim
        self.adapter_out_dim = adapter_out_dim
        self.adapter_merge_size = adapter_merge_size
        self.adapter_type = adapter_type
        self.pre_pixel_shuffle = pre_pixel_shuffle
        self.enable_torch_compile = enable_torch_compile
        self.enable_fp8_moe = enable_fp8_moe


class Dots3NoteAudioConfig(PretrainedConfig):
    model_type = "dots3_note_audio"

    def __init__(
        self,
        processor_type: str = "omni_audio_processor",
        data_type: str = "base64",
        encoder_type: str = "dots",
        whisper_config: dict[str, Any] | None = None,
        use_conv2d_stem: bool = True,
        use_rope: bool = True,
        use_rms_norm: bool = True,
        use_causal: bool = False,
        downsample_hidden_size: int = 480,
        rope_parameters: dict[str, Any] | None = None,
        merge_factor: int = 1,
        chunk_seconds: int = 60,
        conv_bucket_step: float | None = None,
        conv_bucket_max_elements: int | None = None,
        whisper_adapter_in_dim: int = 1280,
        whisper_adapter_out_dim: int = 5120,
        sampling_rate: int = 16000,
        audio_comp_start: str = "<|audio_comp_start|>",
        audio_comp_span: str = "<|audio_comp_pad|>",
        audio_comp_end: str = "<|audio_comp_end|>",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.processor_type = processor_type
        self.data_type = data_type
        self.encoder_type = encoder_type
        self.whisper_config = whisper_config or {}
        self.use_conv2d_stem = use_conv2d_stem
        self.use_rope = use_rope
        self.use_rms_norm = use_rms_norm
        self.use_causal = use_causal
        self.downsample_hidden_size = downsample_hidden_size
        self.rope_parameters = rope_parameters or {
            "partial_rotary_factor": 0.5,
            "rope_theta": 10000.0,
        }
        self.merge_factor = merge_factor
        self.chunk_seconds = chunk_seconds
        self.conv_bucket_step = conv_bucket_step
        self.conv_bucket_max_elements = conv_bucket_max_elements
        self.whisper_adapter_in_dim = whisper_adapter_in_dim
        self.whisper_adapter_out_dim = whisper_adapter_out_dim
        self.sampling_rate = sampling_rate
        self.audio_comp_start = audio_comp_start
        self.audio_comp_span = audio_comp_span
        self.audio_comp_end = audio_comp_end


class Dots3NoteConfig(DeepseekV2Config):
    model_type = "dots3_note"

    def __init__(
        self,
        rope_scaling: dict[str, Any] | None = None,
        scoring_func: str = "noaux_tc",
        moe_layer_freq: int | list[int] = 1,
        qk_layernorm: bool = True,
        k_rope_only_layernorm: bool = True,
        apply_mla_qkv_lora_rescale: bool = True,
        attention_gate_type: str = "headwise",
        swa_num_attention_heads: int | None = None,
        swa_num_key_value_heads: int | None = None,
        swa_q_lora_rank: int | None = None,
        swa_kv_lora_rank: int | None = None,
        swa_qk_nope_head_dim: int | None = None,
        swa_qk_rope_head_dim: int | None = None,
        swa_v_head_dim: int | None = None,
        swa_attention_gate_type: str = "headwise",
        sliding_window_size: int = 512,
        moe_gating_fp32: bool = False,
        num_nextn_predict_layers: int = 1,
        mtp_use_moe: bool = False,
        mtp_head_sharing: str = "full",
        use_dedicated_mtp_embeddings: bool = False,
        layer_types: list[str] | None = None,
        vision_config: dict[str, Any] | Dots3NoteVisionConfig | None = None,
        audio_config: dict[str, Any] | Dots3NoteAudioConfig | None = None,
        image_token_id: int | None = None,
        image_start_token_id: int | None = None,
        image_end_token_id: int | None = None,
        audio_token_id: int | None = None,
        audio_start_token_id: int | None = None,
        audio_end_token_id: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Build the effective text and multimodal checkpoint configuration."""
        multimodal_token_ids: dict[str, int] = {}
        model_path = kwargs.get("name_or_path")
        if model_path and any(
            token_id is None
            for token_id in (
                image_start_token_id,
                image_token_id,
                image_end_token_id,
                audio_start_token_id,
                audio_token_id,
                audio_end_token_id,
            )
        ):
            revision = kwargs.get("_commit_hash")
            added_tokens = get_hf_file_to_dict("added_tokens.json", model_path, revision)
            if isinstance(added_tokens, dict):
                multimodal_token_ids.update(
                    {token: token_id for token, token_id in added_tokens.items() if isinstance(token_id, int)}
                )

            tokenizer_config = get_hf_file_to_dict("tokenizer_config.json", model_path, revision)
            if isinstance(tokenizer_config, dict):
                added_tokens_decoder = tokenizer_config.get("added_tokens_decoder", {})
                for token_id, token_config in added_tokens_decoder.items():
                    if not isinstance(token_config, dict):
                        continue
                    token = token_config.get("content")
                    if not isinstance(token, str):
                        continue
                    try:
                        multimodal_token_ids.setdefault(token, int(token_id))
                    except (TypeError, ValueError):
                        continue

        rope_parameters = kwargs.pop("rope_parameters", None) or rope_scaling
        if rope_parameters is None:
            rope_parameters = {
                "rope_theta": kwargs.get("rope_theta", 10000.0),
                "rope_type": "default",
            }
        else:
            rope_parameters = dict(rope_parameters)
            if "type" in rope_parameters and "rope_type" not in rope_parameters:
                rope_parameters["rope_type"] = rope_parameters.pop("type")
            rope_parameters.setdefault("rope_theta", kwargs.get("rope_theta", 10000.0))
            rope_parameters.setdefault("rope_type", "default")

        topk_method = kwargs.pop("topk_method", None)
        if topk_method is None and scoring_func == "noaux_tc":
            topk_method = scoring_func
        if scoring_func == "noaux_tc":
            scoring_func = "sigmoid"

        super().__init__(
            rope_parameters=rope_parameters,
            topk_method=topk_method,
            **kwargs,
        )

        self.rope_scaling = rope_parameters
        self.scoring_func = scoring_func
        if getattr(self, "n_group", None) is None:
            self.n_group = 1
        if getattr(self, "topk_group", None) is None:
            self.topk_group = 1
        self.moe_layer_freq = moe_layer_freq
        self.qk_layernorm = qk_layernorm
        self.k_rope_only_layernorm = k_rope_only_layernorm
        self.apply_mla_qkv_lora_rescale = apply_mla_qkv_lora_rescale
        self.attention_gate_type = attention_gate_type
        self.sdpa_gate_type = attention_gate_type
        self.swa_num_attention_heads = swa_num_attention_heads
        self.swa_num_key_value_heads = swa_num_key_value_heads
        self.swa_q_lora_rank = swa_q_lora_rank
        self.swa_kv_lora_rank = swa_kv_lora_rank
        self.swa_qk_nope_head_dim = swa_qk_nope_head_dim
        self.swa_qk_rope_head_dim = swa_qk_rope_head_dim
        self.swa_v_head_dim = swa_v_head_dim
        self.swa_attention_gate_type = swa_attention_gate_type
        self.sliding_window_size = sliding_window_size
        self.sliding_window = sliding_window_size
        self.use_sliding_window = bool(layer_types and "sliding_attention" in layer_types)
        self.moe_gating_fp32 = moe_gating_fp32
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.original_mtp_num_layers = num_nextn_predict_layers
        self.mtp_use_moe = mtp_use_moe
        self.mtp_head_sharing = mtp_head_sharing
        self.use_dedicated_mtp_embeddings = use_dedicated_mtp_embeddings
        self.layer_types = layer_types
        if isinstance(vision_config, Dots3NoteVisionConfig):
            self.vision_config = vision_config
        else:
            self.vision_config = Dots3NoteVisionConfig(**(vision_config or {}))
        if isinstance(audio_config, Dots3NoteAudioConfig):
            self.audio_config = audio_config
        else:
            self.audio_config = Dots3NoteAudioConfig(**(audio_config or {}))
        self.image_start_token_id = (
            image_start_token_id
            if image_start_token_id is not None
            else multimodal_token_ids.get("<|img|>", self.vocab_size - 3)
        )
        self.image_token_id = (
            image_token_id
            if image_token_id is not None
            else multimodal_token_ids.get("<|imgpad|>", self.vocab_size - 2)
        )
        self.image_end_token_id = (
            image_end_token_id
            if image_end_token_id is not None
            else multimodal_token_ids.get("<|endofimg|>", self.vocab_size - 1)
        )
        self.audio_start_token_id = (
            audio_start_token_id
            if audio_start_token_id is not None
            else multimodal_token_ids.get("<|audio_comp_start|>", self.vocab_size - 6)
        )
        self.audio_token_id = (
            audio_token_id
            if audio_token_id is not None
            else multimodal_token_ids.get("<|audio_comp_pad|>", self.vocab_size - 4)
        )
        self.audio_end_token_id = (
            audio_end_token_id
            if audio_end_token_id is not None
            else multimodal_token_ids.get("<|audio_comp_end|>", self.vocab_size - 5)
        )
