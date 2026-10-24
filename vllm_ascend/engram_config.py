# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram input normalization and deterministic runtime configuration.

This module must remain importable during platform discovery. Device detection,
upstream configuration and model-specific checks are lazy, and no runtime
resources are created here.
"""

from dataclasses import dataclass, replace
from typing import Any, Literal

from pydantic import TypeAdapter

from vllm_ascend.config_utils import config

LEGACY_ENGRAM_KEYS = (
    "enable_engram",
    "engram_offload_backend",
    "multistream_engram_overlap",
    "engram_tp_size",
)


@config(frozen=True)
class AscendEngramConfig:
    """The only public Ascend Engram input schema; None means automatic."""

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
    source: str = "default"


@dataclass(frozen=True)
class _EngramInput:
    # Preserve original upstream fields, not a subsequently derived projection.
    native: tuple[bool, bool, bool] | None = None
    shared_override: bool | None = None


@dataclass(frozen=True)
class _EngramState:
    inputs: _EngramInput
    projection: Any


def copy_engram_projection(engram_config, **changes):
    """Preserve input provenance when adapting the replicated PCP draft."""
    copied = replace(engram_config, **changes)
    inputs = getattr(engram_config, "_ascend_engram_input", None)
    if isinstance(inputs, _EngramInput):
        copied._ascend_engram_input = replace(inputs, shared_override=changes.get("dp_shared_memory"))
    return copied


def _original_input(vllm_config) -> _EngramInput:
    native = getattr(vllm_config, "engram_config", None)
    inputs = getattr(native, "_ascend_engram_input", None)
    if isinstance(inputs, _EngramInput):
        return inputs
    state = getattr(vllm_config, "_ascend_engram_state", None)
    if isinstance(state, _EngramState) and native is state.projection:
        return state.inputs
    if native is None:
        return _EngramInput()
    return _EngramInput(
        native=(
            bool(getattr(native, "cpu_offload", False)),
            bool(getattr(native, "dp_shared_memory", False)),
            bool(getattr(native, "embedding_across_dp", False)),
        )
    )


def _parse_input(vllm_config, inputs):
    additional = vllm_config.additional_config or {}
    legacy = {key: additional[key] for key in LEGACY_ENGRAM_KEYS if key in additional}
    if "engram_config" in additional:
        if legacy or inputs.native is not None:
            raise ValueError(
                "Do not mix additional_config.engram_config with legacy Engram keys or --engram-config; "
                "move all Engram options into additional_config.engram_config."
            )
        raw = additional["engram_config"]
        if not isinstance(raw, dict):
            raise ValueError("additional_config.engram_config must be an object")
        return TypeAdapter(AscendEngramConfig).validate_python(raw), None, "additional_config.engram_config"
    raw = {
        new: legacy[old]
        for old, new in (
            ("enable_engram", "enabled"),
            ("engram_offload_backend", "backend"),
            ("multistream_engram_overlap", "overlap"),
        )
        if old in legacy
    }
    if inputs.native is not None:
        raw["cpu_offload"], raw["dp_shared_memory"], _ = inputs.native
    size = TypeAdapter(int | None).validate_python(legacy.get("engram_tp_size"))
    source = "legacy" if legacy or inputs.native is not None else "default"
    return TypeAdapter(AscendEngramConfig).validate_python(raw), size, source


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


def _validate_elastic(config, vllm_config, legacy_size) -> int:
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
    if legacy_size is not None and legacy_size != world:
        raise ValueError("Legacy engram_tp_size must equal world_size_across_dp; group_size is now derived")
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


def _publish(vllm_config, resolved, inputs, projection):
    from vllm.logger import logger

    vllm_config.engram_config = projection
    if projection is not None:
        projection._ascend_engram_input = inputs
    vllm_config._ascend_engram_state = _EngramState(inputs, projection)
    if resolved.source == "legacy":
        logger.warning_once(
            "Legacy Engram options are deprecated; use only additional_config.engram_config "
            "with enabled/backend/overlap/cpu_offload/dp_shared_memory."
        )
    logger.info_once(
        "Engram resolved: enabled=%s backend=%s overlap=%s cpu_offload=%s dp_shared_memory=%s "
        "group_size=%s replicate_across_dp=%s source=%s",
        resolved.enabled,
        resolved.backend,
        resolved.overlap,
        resolved.cpu_offload,
        resolved.dp_shared_memory,
        resolved.group_size,
        resolved.replicate_across_dp,
        resolved.source,
    )
    return resolved


def resolve_engram_config(vllm_config) -> ResolvedEngramConfig:
    """Normalize once before model/group setup; safe to repeat after config copies."""
    inputs = _original_input(vllm_config)
    requested, legacy_size, source = _parse_input(vllm_config, inputs)
    if not requested.enabled:
        return _publish(vllm_config, ResolvedEngramConfig(backend=requested.backend, source=source), inputs, None)

    model = _target_model(vllm_config)
    if (
        model is None
        or model.architecture != "DeepseekV41ForCausalLM"
        or not getattr(model.hf_text_config, "engram_layer_ids", None)
    ):
        # Keep the legacy no-op on models without Engram, but reject an explicit
        # new request rather than reporting an enabled feature with no layers.
        if source == "additional_config.engram_config" or inputs.native is not None:
            raise ValueError("Enabled Engram requires a supported model with non-empty Engram layer ids")
        return _publish(vllm_config, ResolvedEngramConfig(backend=requested.backend, source=source), inputs, None)
    _validate_model_and_parallel(vllm_config, model)
    if inputs.native is not None and inputs.native[2]:
        raise ValueError("Ascend Engram does not support embedding_across_dp")

    from vllm.config import EngramConfig, VllmConfig

    cpu_offload = requested.cpu_offload
    shared = requested.dp_shared_memory
    if source == "legacy" and inputs.native is None:
        # Preserve the upstream environment default only for legacy input.
        defaults = EngramConfig()
        cpu_offload, shared = defaults.cpu_offload, defaults.dp_shared_memory
    if cpu_offload is None:
        cpu_offload = True
    parallel = vllm_config.parallel_config
    peers = parallel.data_parallel_size > 1 or parallel.prefill_context_parallel_size > 1
    v2_local = vllm_config.use_v2_model_runner and peers and requested.backend != "elastic"
    replicate = v2_local and not cpu_offload
    # The PCP draft override applies only while it has no DP/PCP peers.
    shared_override = inputs.shared_override if not peers else None
    if shared is None:
        shared = v2_local and cpu_offload
    elif v2_local and cpu_offload and not shared:
        if source != "legacy" and shared_override is None:
            raise ValueError("V2 Engram with DP/PCP requires dp_shared_memory=true; use null for automatic selection")
        shared = True
    if shared_override is not None:
        shared = shared_override
    if replicate and source == "legacy":
        shared = False
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
        source=source,
    )
    if resolved.backend == "elastic":
        resolved = replace(resolved, group_size=_validate_elastic(resolved, vllm_config, legacy_size))
    projection = EngramConfig(cpu_offload=cpu_offload, dp_shared_memory=shared, embedding_across_dp=False)
    previous = getattr(vllm_config, "engram_config", None)
    vllm_config.engram_config = projection
    try:
        VllmConfig._resolve_and_verify_engram_config(vllm_config)
    except Exception:
        # Do not leave an unmarked projection behind if validation fails.
        vllm_config.engram_config = previous
        raise
    return _publish(vllm_config, resolved, inputs, projection)
