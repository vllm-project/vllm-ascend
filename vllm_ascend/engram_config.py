# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram input normalization and deterministic runtime configuration.

This module must remain importable during platform discovery. Device detection,
upstream configuration and model-specific checks are lazy, and no runtime
resources are created here.
"""

from copy import copy
from dataclasses import dataclass, field, fields, replace
from typing import Literal

from pydantic import TypeAdapter
from pydantic.dataclasses import rebuild_dataclass
from pydantic.json_schema import SkipJsonSchema

from vllm_ascend.config_utils import config

LEGACY_ENGRAM_KEYS = (
    "enable_engram",
    "engram_offload_backend",
    "multistream_engram_overlap",
    "engram_tp_size",
)


@config(frozen=True)
class AscendEngramConfig:
    """Fields added to the native --engram-config; None means automatic."""

    enabled: bool = False
    backend: Literal["uva", "elastic"] = "uva"
    overlap: bool | None = None
    cpu_offload: bool | None = None
    dp_shared_memory: bool | None = None


@dataclass(frozen=True)
class ResolvedEngramConfig:
    """Read-only execution policy, also used to project upstream EngramConfig."""

    enabled: bool = False
    backend: Literal["uva", "elastic"] = "uva"
    overlap: bool = False
    cpu_offload: bool = False
    dp_shared_memory: bool = False
    group_size: int | None = None
    # HBM replicas need local lookups under V2, but do not share host storage.
    replicate_across_dp: bool = False


@dataclass
class _EngramMetadata:
    # A declared, non-input field: vLLM replace() inspects every __dict__ key.
    # Exclude provenance from the CLI schema, serialization and graph hash.
    _ascend_requested: SkipJsonSchema[AscendEngramConfig | None] = field(
        default=None, init=False, repr=False, compare=False, metadata={"exclude": True}
    )


def extend_native_engram_config(native) -> None:
    """Extend the existing class in place so cached CLI type references work."""
    if "enabled" in native.__dataclass_fields__:
        return
    for cls in (AscendEngramConfig, _EngramMetadata):
        for item in fields(cls):
            native.__annotations__[item.name] = item.type
            native.__dataclass_fields__[item.name] = copy(item)
            setattr(native, item.name, item.default)
    # Cross-field validation belongs after the enabled gate and automatic
    # placement resolution, rather than during raw CLI parsing.
    native.__pydantic_decorators__.model_validators["_validate_shared_memory"].func = lambda self: self
    rebuild_dataclass(native, force=True)


def _requested(native) -> AscendEngramConfig:
    requested = getattr(native, "_ascend_requested", None)
    if requested is not None:
        return requested
    return TypeAdapter(AscendEngramConfig).validate_python(
        {item.name: getattr(native, item.name) for item in fields(AscendEngramConfig)}
    )


def copy_engram_projection(engram_config, **changes):
    """Adapt the PCP draft without altering the target or adding dynamic keys."""
    from vllm.config import replace as replace_vllm_config

    copied = replace_vllm_config(engram_config, **changes)
    copied._ascend_requested = replace(_requested(engram_config), **changes)
    return copied


def default_engram_overlap() -> bool:
    from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily, get_current_hardware_profile

    return get_current_hardware_profile().device_adaptor_family == DeviceAdaptorFamily.FP8_OPTIMIZED


def _target_model(vllm_config):
    model = vllm_config.model_config
    spec = vllm_config.speculative_config
    if spec is not None and model is getattr(spec, "draft_model_config", None):
        model = spec.target_model_config
    return model


def _validate_model_and_parallel(vllm_config, model) -> None:
    if model is None or model.architecture != "DeepseekV41ForCausalLM":
        raise ValueError("Ascend Engram requires DeepSeek V4.1.")
    parallel = vllm_config.parallel_config
    if (
        parallel.enable_elastic_ep
        or parallel.tensor_parallel_size not in (1, 2, 4, 8)
        or parallel.pipeline_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
    ):
        raise ValueError("Ascend Engram requires TP=1/2/4/8 with PP=DCP=1.")
    if vllm_config.load_config.load_format not in ("auto", "safetensors", "dummy"):
        raise ValueError("Ascend Engram requires indexed safetensors (auto/safetensors), or dummy weights.")


def _validate_elastic(config, vllm_config) -> int:
    if not config.cpu_offload:
        raise ValueError("Engram backend='elastic' requires cpu_offload=True")
    if config.dp_shared_memory:
        raise ValueError("ElasticBuffer row sharding cannot be combined with dp_shared_memory")
    parallel = vllm_config.parallel_config
    if any(
        getattr(parallel, key, 1) != 1
        for key in ("pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size")
    ):
        raise ValueError("ElasticBuffer Engram currently requires PP=PCP=DCP=1")
    world = int(getattr(parallel, "world_size_across_dp", getattr(parallel, "world_size", 1)))
    local = int(getattr(parallel, "local_world_size", world)) * int(getattr(parallel, "data_parallel_size_local", 1))
    if world < 2 or world > local:
        raise ValueError("ElasticBuffer requires >=2 node-local participating ranks")
    if getattr(parallel, "use_sequence_parallel_moe", False):
        raise ValueError("ElasticBuffer Engram does not yet support MoE sequence parallelism")
    if vllm_config.load_config.load_format == "dummy":
        raise ValueError("ElasticBuffer requires a native MXFP8 checkpoint; dummy weights are unsupported")
    _validate_elastic_speculation(vllm_config.speculative_config)
    return world


def _validate_elastic_speculation(speculative_config) -> None:
    """Check the Elastic history capacity without changing proposer selection."""
    if speculative_config is None:
        return
    if getattr(speculative_config, "method", None) != "dspark":
        raise ValueError("ElasticBuffer Engram supports only method='dspark' or disabled speculative decoding")
    use_dspark = getattr(speculative_config, "use_dspark", None)
    if callable(use_dspark) and not use_dspark():
        raise ValueError("ElasticBuffer Engram method='dspark' must report use_dspark()=True")

    from vllm_ascend.models.deepseek_v41.cache_config import STATE_RING_ROWS

    maximum = getattr(speculative_config, "num_speculative_tokens", None)
    schedule = getattr(speculative_config, "num_speculative_tokens_per_batch_size", None) or ()
    lengths = schedule.values() if isinstance(schedule, dict) else (entry[2] for entry in schedule)
    if maximum is None or not 0 < maximum < STATE_RING_ROWS:
        raise ValueError(f"ElasticBuffer Engram DSpark requires 1..{STATE_RING_ROWS - 1} speculative tokens")
    if any(not 0 <= length < STATE_RING_ROWS for length in lengths):
        raise ValueError(f"ElasticBuffer Engram DSpark dynamic lengths must be in 0..{STATE_RING_ROWS - 1}")


def _publish(vllm_config, resolved, projection):
    from vllm.logger import logger

    vllm_config.engram_config = projection
    logger.info_once(
        "Engram resolved: enabled=%s backend=%s overlap=%s cpu_offload=%s dp_shared_memory=%s "
        "group_size=%s replicate_across_dp=%s",
        resolved.enabled,
        resolved.backend,
        resolved.overlap,
        resolved.cpu_offload,
        resolved.dp_shared_memory,
        resolved.group_size,
        resolved.replicate_across_dp,
    )
    return resolved


def resolve_engram_config(vllm_config) -> ResolvedEngramConfig:
    """Gate Engram first, then resolve the native config before runtime setup."""
    additional = vllm_config.additional_config or {}
    if any(key in additional for key in ("engram_config", *LEGACY_ENGRAM_KEYS)):
        raise ValueError(
            "Engram options have moved to --engram-config. Remove additional-config Engram keys; "
            'use --engram-config \'{"enabled":true,"backend":"elastic","overlap":false}\'.'
        )
    native = getattr(vllm_config, "engram_config", None)
    if isinstance(native, dict):
        from vllm.config import EngramConfig

        native = TypeAdapter(EngramConfig).validate_python(native)
    if native is None or not native.enabled:
        return _publish(vllm_config, ResolvedEngramConfig(), None)
    requested = _requested(native)

    model = _target_model(vllm_config)
    _validate_model_and_parallel(vllm_config, model)
    if not getattr(model.hf_text_config, "engram_layer_ids", None):
        raise ValueError("Enabled Engram requires a supported model with non-empty Engram layer ids")
    if native.embedding_across_dp:
        raise ValueError("Ascend Engram does not support embedding_across_dp")

    from vllm.config import EngramConfig, VllmConfig

    cpu_offload = True if requested.cpu_offload is None else requested.cpu_offload
    shared = requested.dp_shared_memory
    parallel = vllm_config.parallel_config
    peers = parallel.data_parallel_size > 1 or parallel.prefill_context_parallel_size > 1
    v2_local = vllm_config.use_v2_model_runner and peers and requested.backend != "elastic"
    replicate = v2_local and not cpu_offload
    if shared is None:
        shared = v2_local and cpu_offload
    elif v2_local and cpu_offload and not shared:
        raise ValueError("V2 Engram with DP/PCP requires dp_shared_memory=true; omit it for automatic selection")
    if shared and not cpu_offload:
        raise ValueError("dp_shared_memory requires cpu_offload=True")
    overlap = requested.overlap if requested.overlap is not None else default_engram_overlap()
    resolved = ResolvedEngramConfig(
        enabled=True,
        backend=requested.backend,
        overlap=overlap,
        cpu_offload=cpu_offload,
        dp_shared_memory=shared,
        replicate_across_dp=replicate,
    )
    if resolved.backend == "elastic":
        resolved = replace(resolved, group_size=_validate_elastic(resolved, vllm_config))
    projection = EngramConfig(
        enabled=True,
        backend=resolved.backend,
        overlap=overlap,
        cpu_offload=cpu_offload,
        dp_shared_memory=shared,
        embedding_across_dp=False,
    )
    projection._ascend_requested = requested
    previous = getattr(vllm_config, "engram_config", None)
    vllm_config.engram_config = projection
    try:
        VllmConfig._resolve_and_verify_engram_config(vllm_config)
    except Exception:
        vllm_config.engram_config = previous
        raise
    return _publish(vllm_config, resolved, projection)
