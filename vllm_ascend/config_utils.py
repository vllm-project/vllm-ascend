# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dependency-light helpers for vLLM-compatible config dataclasses."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar, overload

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from pydantic.fields import Field as PydanticField
from typing_extensions import dataclass_transform

if TYPE_CHECKING:
    from _typeshed import DataclassInstance
else:
    DataclassInstance = Any

ConfigT = TypeVar("ConfigT", bound=DataclassInstance)


@overload
@dataclass_transform(field_specifiers=(PydanticField,))
def config(cls: type[ConfigT]) -> type[ConfigT]: ...


@overload
@dataclass_transform(field_specifiers=(PydanticField,))
def config(*, config: ConfigDict | None = None, **kwargs: Any) -> Callable[[type[ConfigT]], type[ConfigT]]: ...


@dataclass_transform(field_specifiers=(PydanticField,))
def config(
    cls: type[ConfigT] | None = None,
    *,
    config: ConfigDict | None = None,
    **kwargs: Any,
) -> type[ConfigT] | Callable[[type[ConfigT]], type[ConfigT]]:
    """Create a vLLM-compatible config dataclass without importing vllm.config.

    This mirrors ``vllm.config.utils.config`` while avoiding that module's
    package-initialization cycle during vLLM platform discovery.
    """
    merged_config = ConfigDict(extra="forbid")
    if config is not None:
        merged_config.update(config)

    def decorator(config_cls: type[ConfigT]) -> type[ConfigT]:
        return dataclass(config_cls, config=merged_config, **kwargs)  # type: ignore[return-value]

    return decorator if cls is None else decorator(cls)


def is_deepseek_v41(hf_config: Any) -> bool:
    """Identify released and legacy V4.1 configs at the model boundary."""
    model_types = ("deepseek_v41", "deepseek_v4.1", "deepseek_v41_text", "deepseek_v4.1_text")
    if isinstance(hf_config, dict):
        return hf_config.get("model_type") in model_types or is_deepseek_v41(hf_config.get("text_config"))
    # SpeculativeConfig may overwrite the instance model_type for DSpark.
    # The upstream flattened config class still identifies the V4.1 checkpoint.
    return (
        getattr(type(hf_config), "model_type", None) in model_types
        or getattr(hf_config, "model_type", None) in model_types
        or (getattr(hf_config, "text_config", None) is not None and is_deepseek_v41(hf_config.text_config))
    )


def normalize_deepseek_v41_config(hf_config: Any) -> Any:
    """Normalize legacy checkpoint fields without replacing upstream config classes."""
    aliases = {
        "kv_source_layers": "kv_source_layer_ids",
        "index_source_layers": "index_source_layer_ids",
        "candidate_source_layer": "candidate_source_layer_id",
        "engram_pad_id": "engram_pad_token_id",
        "dspark_n_activated_experts": "dspark_num_experts_per_tok",
    }
    for legacy_name, released_name in aliases.items():
        legacy = getattr(hf_config, legacy_name, None)
        released = getattr(hf_config, released_name, None)
        if legacy is not None and released is not None and legacy != released:
            raise ValueError(f"Conflicting DeepSeek V4.1 config fields: {legacy_name}, {released_name}")
        value = released if released is not None else legacy
        if value is not None:
            setattr(hf_config, legacy_name, value)
            setattr(hf_config, released_name, value)
    for name, default in (("num_hash_layers", 0), ("n_group", 1), ("topk_group", 1)):
        if not hasattr(hf_config, name):
            setattr(hf_config, name, default)
    rope = dict(getattr(hf_config, "rope_parameters", None) or {})
    for name, value in {
        "factor": 1.0,
        "beta_fast": 32,
        "beta_slow": 1,
        "original_max_position_embeddings": getattr(hf_config, "max_position_embeddings", 1048576),
        "rope_theta": getattr(hf_config, "rope_theta", 10000.0),
    }.items():
        rope.setdefault(name, value)
    hf_config.rope_parameters = rope
    hf_config.image_sentinel_base_id = getattr(hf_config, "image_token_id", 129264)
    hf_config.image_pad_token_id = hf_config.image_sentinel_base_id + 1
    supported_rotation = {
        "value_projection_rotated": True,
        "value_basis": "quarot_global",
        "key_and_gate_basis": "original",
        "runtime_delta_rotation": False,
    }
    rotation = getattr(hf_config, "engram_rotation_config", None) or supported_rotation
    if any(rotation.get(name) != value for name, value in supported_rotation.items()):
        raise ValueError(f"Unsupported DeepSeek V4.1 Engram rotation contract: {rotation!r}")
    hf_config.engram_rotation_config = dict(rotation)
    return hf_config
