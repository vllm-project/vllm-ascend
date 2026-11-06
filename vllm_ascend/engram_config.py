# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram input normalization and deterministic runtime configuration.

This module must remain importable during platform discovery. Device detection,
upstream configuration and model-specific checks are lazy, and no runtime
resources are created here.
"""

from dataclasses import dataclass, replace
from typing import Literal

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
    """Ascend-only switches in additional_config.engram_config."""

    enabled: bool = False
    backend: Literal["uva", "elastic"] = "uva"
    overlap: bool | None = None


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
    # DSpark reconstructs VllmConfig with the draft TP but retains the
    # target's Engram tables and communication domain.
    target_parallel = getattr(vllm_config.speculative_config, "target_parallel_config", None)
    parallel = target_parallel if target_parallel is not None else vllm_config.parallel_config
    if any(
        getattr(parallel, key, 1) != 1
        for key in ("pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size")
    ):
        raise ValueError("ElasticBuffer Engram currently requires PP=PCP=DCP=1")
    world = int(getattr(parallel, "world_size_across_dp", getattr(parallel, "world_size", 1)))
    local = int(getattr(parallel, "local_world_size", world)) * int(getattr(parallel, "data_parallel_size_local", 1))
    if world < 2:
        raise ValueError(f"ElasticBuffer requires >=2 participating ranks; world_size_across_dp={world}")
    # Each external DP server manages one replica, even when all replicas
    # share a host. Verify their physical locality after distributed startup.
    external_dp = bool(getattr(parallel, "data_parallel_external_lb", False))
    if not external_dp and world > local:
        raise ValueError(
            "ElasticBuffer requires all participating ranks on one node; "
            f"world_size_across_dp={world}, configured_local_ranks={local}"
        )
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
    if any(key in additional for key in LEGACY_ENGRAM_KEYS):
        raise ValueError("Move flat Engram switches into additional_config.engram_config: enabled/backend/overlap")
    raw = additional.get("engram_config", {})
    if not isinstance(raw, dict):
        raise ValueError("additional_config.engram_config must be an object")
    requested = TypeAdapter(AscendEngramConfig).validate_python(raw)
    if not requested.enabled:
        return _publish(vllm_config, ResolvedEngramConfig(), None)

    model = _target_model(vllm_config)
    _validate_model_and_parallel(vllm_config, model)
    if not getattr(model.hf_text_config, "engram_layer_ids", None):
        raise ValueError("Enabled Engram requires a supported model with non-empty Engram layer ids")
    from vllm.config import EngramConfig, VllmConfig

    native = getattr(vllm_config, "engram_config", None)
    if native is None:
        native = EngramConfig(cpu_offload=True)
    if native.embedding_across_dp:
        raise ValueError("Ascend Engram does not support embedding_across_dp")
    cpu_offload = native.cpu_offload
    shared = native.dp_shared_memory
    parallel = vllm_config.parallel_config
    peers = parallel.data_parallel_size > 1 or parallel.prefill_context_parallel_size > 1
    v2_local = vllm_config.use_v2_model_runner and peers and requested.backend != "elastic"
    replicate = v2_local and not cpu_offload
    # Keep the existing native V2 policy: host tables share storage across
    # DP/PCP peers; HBM tables use independent local replicas.
    if v2_local and cpu_offload:
        shared = True
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
        cpu_offload=cpu_offload,
        dp_shared_memory=shared,
        embedding_across_dp=False,
    )
    previous = getattr(vllm_config, "engram_config", None)
    vllm_config.engram_config = projection
    try:
        VllmConfig._resolve_and_verify_engram_config(vllm_config)
    except Exception:
        vllm_config.engram_config = previous
        raise
    if resolved.backend == "elastic":
        loader_extra = getattr(vllm_config.load_config, "model_loader_extra_config", None)
        if loader_extra and loader_extra.get("enable_multithread_load", False):
            from vllm.logger import logger

            logger.warning(
                "Disabling multithread checkpoint loading for Elastic Engram: "
                "complete shards would be materialized before Engram table filtering."
            )
            vllm_config.load_config.model_loader_extra_config = {
                **loader_extra,
                "enable_multithread_load": False,
            }
    return _publish(vllm_config, resolved, projection)
