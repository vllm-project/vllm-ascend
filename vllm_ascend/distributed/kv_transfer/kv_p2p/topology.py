# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Any

from vllm.config import VllmConfig


@dataclass(frozen=True)
class PDTopology:
    prefill_tp_size: int
    prefill_dp_size: int
    prefill_pp_size: int
    decode_tp_size: int
    decode_dp_size: int
    decode_pp_size: int
    prefill_pp_layer_partition: str | None


def _global_dp_size(vllm_config: VllmConfig) -> int:
    parallel_config = vllm_config.parallel_config
    return int(getattr(parallel_config, "data_parallel_size", 1))


def _get_extra_parallel_config(vllm_config: VllmConfig, key: str) -> dict[str, Any]:
    config = vllm_config.kv_transfer_config.get_from_extra_config(key, {})
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"kv_connector_extra_config.{key} must be a dict, got {type(config).__name__}.")
    return config


def _has_any_parallel_size(config: dict[str, Any]) -> bool:
    return any(key in config for key in ("tp_size", "dp_size"))


def _validate_complete_legacy_config(prefill_config: dict[str, Any], decode_config: dict[str, Any]) -> bool:
    has_prefill = _has_any_parallel_size(prefill_config)
    has_decode = _has_any_parallel_size(decode_config)
    if not has_prefill and not has_decode:
        return False

    missing = []
    for side, config in (("prefill", prefill_config), ("decode", decode_config)):
        for key in ("tp_size", "dp_size"):
            if key not in config:
                missing.append(f"{side}.{key}")
    if missing:
        raise ValueError(
            "Incomplete legacy kv_connector_extra_config PD topology. "
            f"Missing: {', '.join(missing)}. "
            "Either provide all prefill/decode dp_size/tp_size fields, "
            "or remove them all to enable automatic topology inference."
        )
    return True


def resolve_pd_topology(vllm_config: VllmConfig) -> PDTopology:
    """Resolve PD topology for Mooncake connectors.

    Legacy ``kv_connector_extra_config.prefill/decode`` values are honored when
    fully specified. When omitted, the local side is inferred from vLLM's
    parallel config and ``kv_role``; request metadata supplies remote topology
    for unequal P/D transfers.
    """
    prefill_config = _get_extra_parallel_config(vllm_config, "prefill")
    decode_config = _get_extra_parallel_config(vllm_config, "decode")

    if _validate_complete_legacy_config(prefill_config, decode_config):
        return PDTopology(
            prefill_tp_size=int(prefill_config["tp_size"]),
            prefill_dp_size=int(prefill_config["dp_size"]),
            prefill_pp_size=int(prefill_config.get("pp_size", 1)),
            decode_tp_size=int(decode_config["tp_size"]),
            decode_dp_size=int(decode_config["dp_size"]),
            decode_pp_size=int(decode_config.get("pp_size", 1)),
            prefill_pp_layer_partition=prefill_config.get("pp_layer_partition"),
        )

    parallel_config = vllm_config.parallel_config
    local_tp_size = int(parallel_config.tensor_parallel_size)
    local_dp_size = _global_dp_size(vllm_config)
    local_pp_size = int(getattr(parallel_config, "pipeline_parallel_size", 1))
    kv_role = vllm_config.kv_transfer_config.kv_role

    prefill_tp_size = local_tp_size
    prefill_dp_size = local_dp_size
    prefill_pp_size = local_pp_size if kv_role == "kv_producer" else 1
    decode_tp_size = local_tp_size
    decode_dp_size = local_dp_size
    decode_pp_size = local_pp_size if kv_role == "kv_consumer" else 1

    if prefill_config:
        prefill_pp_size = int(prefill_config.get("pp_size", prefill_pp_size))
    if decode_config:
        decode_pp_size = int(decode_config.get("pp_size", decode_pp_size))

    return PDTopology(
        prefill_tp_size=prefill_tp_size,
        prefill_dp_size=prefill_dp_size,
        prefill_pp_size=prefill_pp_size,
        decode_tp_size=decode_tp_size,
        decode_dp_size=decode_dp_size,
        decode_pp_size=decode_pp_size,
        prefill_pp_layer_partition=prefill_config.get("pp_layer_partition"),
    )


def validate_remote_prefill_tp_size(prefill_tp_size: int, decode_tp_size: int) -> None:
    if decode_tp_size <= 0:
        raise ValueError(f"decode_tp_size: {decode_tp_size} must be greater than 0.")
    if prefill_tp_size < decode_tp_size:
        raise ValueError(
            f"prefill_tp_size: {prefill_tp_size} must be greater than or equal to decode_tp_size: {decode_tp_size}."
        )
    if prefill_tp_size % decode_tp_size != 0:
        raise ValueError(f"prefill_tp_size: {prefill_tp_size} must be divisible by decode_tp_size: {decode_tp_size}.")
