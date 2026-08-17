# SPDX-License-Identifier: Apache-2.0

import math
from collections.abc import Iterable, Iterator
from functools import wraps
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.model_executor.models.deepseek_v2 import DeepseekV3ForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper


_MISTRAL3_TEXT_ARCHITECTURES = {
    "mistral4": "Mistral4ForCausalLM",
}

_MISTRAL4_BASE_WEIGHTS_MAPPER = WeightsMapper(
    orig_to_new_suffix={
        ".activation_scale": ".input_scale",
        ".weight_scale_inv": ".weight_scale",
    },
)
_MISTRAL4_FUSED_QKV_A_WEIGHTS_MAPPER = WeightsMapper(
    orig_to_new_stacked={
        ".self_attn.q_a_proj.": (".self_attn.fused_qkv_a_proj.", 0),
        ".self_attn.kv_a_proj_with_mqa.": (
            ".self_attn.fused_qkv_a_proj.",
            1,
        ),
    }
) | _MISTRAL4_BASE_WEIGHTS_MAPPER


def _get_mistral3_text_architectures(text_config: object) -> list[str]:
    model_type = getattr(text_config, "model_type", None)
    try:
        return [_MISTRAL3_TEXT_ARCHITECTURES[model_type]]
    except KeyError as exc:
        supported = ", ".join(sorted(_MISTRAL3_TEXT_ARCHITECTURES))
        raise ValueError(
            "Unsupported Mistral3 text_config.model_type "
            f"{model_type!r}; expected one of: {supported}"
        ) from exc


def _prepare_llama4_scaling(config: object) -> None:
    if getattr(config, "llama_4_scaling", None) is not None:
        return
    rope_parameters = getattr(config, "rope_parameters", None)
    if not isinstance(rope_parameters, dict):
        return
    scaling_beta = rope_parameters.get("llama_4_scaling_beta")
    if scaling_beta is None:
        return
    original_max_position = rope_parameters.get(
        "original_max_position_embeddings"
    )
    if original_max_position is None:
        raise ValueError(
            "llama_4_scaling_beta requires original_max_position_embeddings"
        )
    config.llama_4_scaling = {  # type: ignore[attr-defined]
        "beta": scaling_beta,
        "original_max_position_embeddings": original_max_position,
    }


def patch_mistral3_text_model() -> None:
    """Resolve new nested Mistral text configs on the pinned vLLM baseline."""
    import vllm.model_executor.models.mistral3 as mistral3

    original = mistral3.init_vllm_registered_model
    if getattr(original, "_vllm_ascend_mistral_compat", False):
        return

    @wraps(original)
    def init_vllm_registered_model(
        vllm_config: VllmConfig,
        *,
        prefix: str = "",
        hf_config: Any | None = None,
        architectures: list[str] | None = None,
    ):
        if hf_config is not None and architectures is None:
            architectures = _get_mistral3_text_architectures(hf_config)
            _prepare_llama4_scaling(hf_config)
        return original(
            vllm_config,
            prefix=prefix,
            hf_config=hf_config,
            architectures=architectures,
        )

    init_vllm_registered_model._vllm_ascend_mistral_compat = True  # type: ignore[attr-defined]
    mistral3.init_vllm_registered_model = init_vllm_registered_model


def _prepare_mistral4_config(config: object) -> None:
    if getattr(config, "model_type", None) != "mistral4":
        raise ValueError("Mistral4ForCausalLM requires model_type='mistral4'")
    if not getattr(config, "rope_interleave", False):
        raise ValueError("Mistral4 requires interleaved RoPE")

    qk_nope_head_dim = getattr(config, "qk_nope_head_dim")
    qk_rope_head_dim = getattr(config, "qk_rope_head_dim")
    qk_head_dim = getattr(config, "qk_head_dim")
    if qk_nope_head_dim + qk_rope_head_dim != qk_head_dim:
        raise ValueError(
            "Mistral4 qk_head_dim must equal qk_nope_head_dim + "
            "qk_rope_head_dim"
        )

    rope_parameters = dict(getattr(config, "rope_parameters"))
    partial_rotary_factor = rope_parameters.pop("partial_rotary_factor", None)
    expected_partial_factor = qk_rope_head_dim / qk_head_dim
    if partial_rotary_factor is not None and not math.isclose(
        float(partial_rotary_factor), float(expected_partial_factor)
    ):
        raise ValueError(
            "Mistral4 partial_rotary_factor does not match the MLA rotary "
            "head dimension"
        )

    # DeepseekV2Attention already receives the rotary-only MLA head dimension.
    config.rope_parameters = rope_parameters  # type: ignore[attr-defined]
    _prepare_llama4_scaling(config)


def _mistral4_linear_scale_width(name: str, config: object) -> int | None:
    if ".q_a_proj." in name:
        return int(getattr(config, "q_lora_rank"))
    if ".kv_a_proj_with_mqa." in name:
        return int(getattr(config, "kv_lora_rank")) + int(
            getattr(config, "qk_rope_head_dim")
        )
    if ".q_b_proj." in name:
        return int(getattr(config, "num_attention_heads")) * int(
            getattr(config, "qk_head_dim")
        )
    if ".kv_b_proj." in name:
        return int(getattr(config, "num_attention_heads")) * (
            int(getattr(config, "qk_nope_head_dim"))
            + int(getattr(config, "v_head_dim"))
        )
    if ".self_attn.o_proj." in name:
        return int(getattr(config, "hidden_size"))
    if ".mlp.shared_experts.gate_proj." in name:
        return int(getattr(config, "moe_intermediate_size")) * int(
            getattr(config, "n_shared_experts", 1)
        )
    if ".mlp.shared_experts.up_proj." in name:
        return int(getattr(config, "moe_intermediate_size")) * int(
            getattr(config, "n_shared_experts", 1)
        )
    if ".mlp.shared_experts.down_proj." in name:
        return int(getattr(config, "hidden_size"))
    return None


def _expand_packed_expert_scalar(
    name: str,
    value: torch.Tensor,
    *,
    suffix: str,
    projection: str,
    destination: str,
    duplicate_gate_up: bool,
    output_width: int | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    values = value.reshape(value.shape[0], -1)
    if values.shape[1] != 1:
        raise ValueError(
            f"Expected one scalar per Mistral4 expert for {name}, got "
            f"shape {tuple(value.shape)}"
        )
    base = name[: -len(suffix)]
    projections = ("gate_proj", "up_proj") if duplicate_gate_up else (projection,)
    for expert_id, expert_value in enumerate(values[:, 0]):
        if output_width is not None:
            expert_value = expert_value.reshape(1, 1).expand(output_width, 1)
            expert_value = expert_value.contiguous()
        for target_projection in projections:
            yield (
                f"{base}{expert_id}.{target_projection}{destination}",
                expert_value,
            )


def _explode_packed_expert_weight(
    name: str,
    value: torch.Tensor,
    *,
    projection: str,
    config: object,
) -> Iterator[tuple[str, torch.Tensor]]:
    if value.ndim != 3:
        raise ValueError(
            f"Expected a 3D packed Mistral4 expert weight for {name}, got "
            f"shape {tuple(value.shape)}"
        )
    expected_experts = int(getattr(config, "n_routed_experts"))
    if value.shape[0] != expected_experts:
        raise ValueError(
            f"Expected {expected_experts} Mistral4 experts for {name}, got "
            f"{value.shape[0]}"
        )

    base = name[: -len(projection)]
    if projection == "gate_up_proj":
        intermediate_size = int(getattr(config, "moe_intermediate_size"))
        if value.shape[1] != 2 * intermediate_size:
            raise ValueError(
                f"Expected gate/up width {2 * intermediate_size} for {name}, "
                f"got {value.shape[1]}"
            )
        for expert_id, expert_weight in enumerate(value.unbind(0)):
            gate_weight, up_weight = expert_weight.split(intermediate_size, dim=0)
            yield f"{base}{expert_id}.gate_proj.weight", gate_weight
            yield f"{base}{expert_id}.up_proj.weight", up_weight
        return
    if projection == "down_proj":
        for expert_id, expert_weight in enumerate(value.unbind(0)):
            yield f"{base}{expert_id}.down_proj.weight", expert_weight
        return
    raise ValueError(f"Unsupported packed Mistral4 expert projection: {projection}")


def _prepare_mistral4_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    config: object,
    *,
    channelwise_fp8: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    expert_widths = {
        "gate_up_proj": 2 * int(getattr(config, "moe_intermediate_size")),
        "down_proj": int(getattr(config, "hidden_size")),
    }
    expert_metadata = {
        "gate_up_proj_scale_inv": ("gate_up_proj", ".weight_scale", True),
        "down_proj_scale_inv": ("down_proj", ".weight_scale", False),
        "gate_up_proj_activation_scale": ("gate_up_proj", ".input_scale", True),
        "down_proj_activation_scale": ("down_proj", ".input_scale", False),
    }

    for name, loaded_weight in weights:
        if ".mlp.experts." in name:
            projection = next(
                (
                    item
                    for item in ("gate_up_proj", "down_proj")
                    if name.endswith(item)
                ),
                None,
            )
            if projection is not None:
                yield from _explode_packed_expert_weight(
                    name, loaded_weight, projection=projection, config=config
                )
                continue

            matched = next(
                (
                    (suffix, metadata)
                    for suffix, metadata in expert_metadata.items()
                    if name.endswith(suffix)
                ),
                None,
            )
            if matched is not None:
                suffix, (projection, destination, duplicate_gate_up) = matched
                if channelwise_fp8:
                    if destination == ".input_scale":
                        continue
                    output_width = expert_widths[projection]
                    if duplicate_gate_up:
                        output_width //= 2
                    yield from _expand_packed_expert_scalar(
                        name,
                        loaded_weight,
                        suffix=suffix,
                        projection=projection,
                        destination=destination,
                        duplicate_gate_up=duplicate_gate_up,
                        output_width=output_width,
                    )
                else:
                    yield from _expand_packed_expert_scalar(
                        name,
                        loaded_weight,
                        suffix=suffix,
                        projection=projection,
                        destination=destination,
                        duplicate_gate_up=duplicate_gate_up,
                    )
                continue

        if channelwise_fp8 and name.endswith((".activation_scale", ".input_scale")):
            continue
        if channelwise_fp8 and name.endswith((".weight_scale_inv", ".weight_scale")):
            output_size = _mistral4_linear_scale_width(name, config)
            if output_size is not None:
                if loaded_weight.numel() != 1:
                    raise ValueError(
                        f"Expected a scalar Mistral4 linear scale for {name}, "
                        f"got shape {tuple(loaded_weight.shape)}"
                    )
                loaded_weight = loaded_weight.reshape(1, 1).expand(output_size, 1)
                loaded_weight = loaded_weight.contiguous()
        yield name, loaded_weight


def _get_mistral4_weights_mapper(parameter_names: Iterable[str]) -> WeightsMapper:
    if any(".fused_qkv_a_proj." in name for name in parameter_names):
        return _MISTRAL4_FUSED_QKV_A_WEIGHTS_MAPPER
    return _MISTRAL4_BASE_WEIGHTS_MAPPER


class Mistral4ForCausalLM(DeepseekV3ForCausalLM):
    hf_to_vllm_mapper = _MISTRAL4_FUSED_QKV_A_WEIGHTS_MAPPER

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        _prepare_mistral4_config(vllm_config.model_config.hf_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        channelwise_fp8 = bool(
            getattr(self.quant_config, "mistral4_dynamic_channelwise", False)
        )
        weights = _prepare_mistral4_weights(
            weights, self.config, channelwise_fp8=channelwise_fp8
        )
        mapper = _get_mistral4_weights_mapper(
            name for name, _ in self.named_parameters()
        )
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=mapper)
