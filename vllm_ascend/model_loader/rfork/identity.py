#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Build stable model compatibility fingerprints and shard seed keys."""

import hashlib
import json
import math
from enum import Enum
from typing import Any

import torch
from vllm.config import ModelConfig, VllmConfig

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.model_loader.rfork.types import RFORK_PROTOCOL_VERSION


def _canonicalize_fingerprint_value(value: Any, *, _active_ids: set[int] | None = None) -> Any:
    """Convert config values to deterministic JSON data and reject cycles."""

    if _active_ids is None:
        _active_ids = set()

    recursive_value = isinstance(value, (dict, list, tuple, set, frozenset)) or isinstance(
        getattr(value, "__dict__", None), dict
    )
    value_id = id(value)
    if recursive_value:
        if value_id in _active_ids:
            raise ValueError("RFork fingerprint value contains a recursive reference")
        _active_ids.add(value_id)

    try:
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else str(value)
        if isinstance(value, torch.dtype):
            return str(value)
        if isinstance(value, Enum):
            return value.name
        if isinstance(value, dict):
            return {
                str(key): _canonicalize_fingerprint_value(item, _active_ids=_active_ids)
                for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            }
        if isinstance(value, (list, tuple)):
            return [_canonicalize_fingerprint_value(item, _active_ids=_active_ids) for item in value]
        if isinstance(value, set | frozenset):
            canonical_values = [_canonicalize_fingerprint_value(item, _active_ids=_active_ids) for item in value]
            return sorted(canonical_values, key=lambda item: json.dumps(item, sort_keys=True, default=str))

        # Prefer stable config names because repr(value) may contain process-local addresses.
        name = getattr(value, "name", None)
        if isinstance(name, str):
            return name
        get_name = getattr(value, "get_name", None)
        if callable(get_name):
            try:
                named_value = get_name()
            except Exception:  # pragma: no cover - defensive for third-party configs
                named_value = None
            if isinstance(named_value, str):
                return named_value

        public_attributes = getattr(value, "__dict__", None)
        if isinstance(public_attributes, dict):
            attributes = {
                str(key): _canonicalize_fingerprint_value(item, _active_ids=_active_ids)
                for key, item in sorted(public_attributes.items(), key=lambda item: str(item[0]))
                if not str(key).startswith("_") and not callable(item)
            }
            if attributes:
                value_type = type(value)
                return {
                    "type": f"{value_type.__module__}.{value_type.__qualname__}",
                    "attributes": attributes,
                }

        value_type = type(value)
        return f"{value_type.__module__}.{value_type.__qualname__}"
    finally:
        if recursive_value:
            _active_ids.remove(value_id)


def _get_architectures_descriptor(model_config: ModelConfig) -> Any:
    """Capture the architecture list only; tensor structure covers the rest."""
    for attr in ("hf_config", "hf_text_config"):
        hf_config = getattr(model_config, attr, None)
        architectures = getattr(hf_config, "architectures", None)
        if architectures is not None:
            return _canonicalize_fingerprint_value(list(architectures))
    for attr in ("architecture", "architectures"):
        architectures = getattr(model_config, attr, None)
        if architectures is not None:
            return _canonicalize_fingerprint_value(architectures)
    return None


def _normalize_rope_config_value(value: Any) -> Any:
    """Normalize equivalent RoPE numeric values before hashing."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _normalize_rope_config_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_rope_config_value(item) for item in value]
    return value


def _get_rope_config_descriptor(model_config: ModelConfig) -> list[dict[str, Any]]:
    """Collect RoPE fields that affect cache values or extent."""
    descriptors: list[dict[str, Any]] = []
    for attr in ("hf_config", "hf_text_config"):
        hf_config = getattr(model_config, attr, None)
        if hf_config is None:
            continue
        descriptors.append(
            {
                "field": attr,
                "rope_theta": _normalize_rope_config_value(getattr(hf_config, "rope_theta", None)),
                "rope_scaling": _normalize_rope_config_value(getattr(hf_config, "rope_scaling", None)),
                "max_position_embeddings": _normalize_rope_config_value(
                    getattr(hf_config, "max_position_embeddings", None)
                ),
                # Nested fields can rewrite RoPE cache bytes without changing tensor shapes.
                "rope_parameters": _normalize_rope_config_value(getattr(hf_config, "rope_parameters", None)),
                "compress_rope_theta": _normalize_rope_config_value(getattr(hf_config, "compress_rope_theta", None)),
            }
        )
    return descriptors


def _get_model_revision(model_config: ModelConfig) -> Any:
    # Prefer the resolved commit because a branch or tag can move without changing its name.
    for attr in ("hf_config", "hf_text_config"):
        hf_config = getattr(model_config, attr, None)
        for revision_attr in ("_commit_hash", "commit_hash"):
            revision = getattr(hf_config, revision_attr, None)
            if isinstance(revision, str) and revision:
                return revision
    revision = getattr(model_config, "revision", None)
    if revision is None:
        revision = getattr(model_config, "model_revision", None)
    if revision is not None:
        return _canonicalize_fingerprint_value(revision)
    for attr in ("hf_config", "hf_text_config"):
        hf_config = getattr(model_config, attr, None)
        revision = getattr(hf_config, "revision", None)
        if revision is not None:
            return _canonicalize_fingerprint_value(revision)
    return None


def _get_effective_kv_role(vllm_config: VllmConfig) -> str | None:
    """Return a stable KV role across current and legacy vLLM configs."""
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer_config is None:
        return None

    kv_role = getattr(kv_transfer_config, "kv_role", None)
    if kv_role is not None:
        return str(kv_role)

    is_kv_producer = bool(getattr(kv_transfer_config, "is_kv_producer", False))
    is_kv_consumer = bool(getattr(kv_transfer_config, "is_kv_consumer", False))
    if is_kv_producer and is_kv_consumer:
        return "kv_both"
    if is_kv_producer:
        return "kv_producer"
    if is_kv_consumer:
        return "kv_consumer"
    return None


def _get_quantization_config_digest(model_config: ModelConfig) -> str | None:
    """Digest the effective quantization config so parameter changes re-key the fingerprint."""
    quantization_config = None
    try:
        quantization_config = getattr(model_config, "hf_quant_config", None)
    except Exception:  # pragma: no cover - property may raise before config load
        quantization_config = None
    if quantization_config is None:
        for attr in ("hf_config", "hf_text_config"):
            hf_config = getattr(model_config, attr, None)
            if hf_config is None:
                continue
            quantization_config = getattr(hf_config, "quantization_config", None)
            if quantization_config is not None:
                break
    if quantization_config is None:
        return None
    canonical_json = json.dumps(
        _canonicalize_fingerprint_value(quantization_config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical_json).hexdigest()


def build_compatibility_fingerprint(
    vllm_config: VllmConfig,
    model_config: ModelConfig,
    *,
    model_url: str,
    model_deploy_strategy_name: str,
) -> str:
    """Build the RFork compatibility identity and return its SHA256 digest."""

    try:
        ascend_config = get_ascend_config()
    except Exception:  # pragma: no cover - only reached before Ascend config setup
        ascend_config = None

    descriptor = {
        "rfork_protocol_version": RFORK_PROTOCOL_VERSION,
        "model_url": model_url,
        "model_deploy_strategy_name": model_deploy_strategy_name,
        "model_revision": _get_model_revision(model_config),
        "quantization_config": _get_quantization_config_digest(model_config),
        "architecture": _get_architectures_descriptor(model_config),
        "rope": _get_rope_config_descriptor(model_config),
        "weight_layout": {
            # P/D roles can materialize different SFA/MLA inference tensors.
            "kv_role": _get_effective_kv_role(vllm_config),
            # Mix placement may swap experts without changing any tensor shape.
            "mix_placement": _canonicalize_fingerprint_value(getattr(ascend_config, "mix_placement", None)),
        },
    }
    canonical_json = json.dumps(
        _canonicalize_fingerprint_value(descriptor),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical_json).hexdigest()


def _resolve_sharded_dp_rank(vllm_config: VllmConfig) -> int | None:
    """Return the DP rank only when weight content is sharded across DP.

    DP ranks hold identical replicated weights and share one seed pool, except
    when content is sharded across the DP dimension: fine-grained TP shards
    selected modules along DP, and MoE expert weights are sharded across the
    DPxTP plane with or without EP (with EP on, the ep_rank in the seed key
    already encodes the DP position). In those deployments the DP position
    picks content that identical shapes cannot distinguish. A false split only
    costs a seed miss, so an unknown is_moe_model conservatively partitions.
    """
    try:
        ascend_config = get_ascend_config()
    except Exception:  # pragma: no cover - only reached before Ascend config setup
        ascend_config = None

    parallel_config = getattr(vllm_config, "parallel_config", None)
    finegrained_tp_config = getattr(ascend_config, "finegrained_tp_config", None)
    finegrained_tp_sizes = (
        getattr(finegrained_tp_config, "oproj_tensor_parallel_size", 0),
        getattr(finegrained_tp_config, "lmhead_tensor_parallel_size", 0),
        getattr(finegrained_tp_config, "embedding_tensor_parallel_size", 0),
        getattr(finegrained_tp_config, "mlp_tensor_parallel_size", 0),
    )
    finegrained_tp_enabled = any(
        isinstance(size, int) and not isinstance(size, bool) and size > 0 for size in finegrained_tp_sizes
    )
    data_parallel_size = getattr(parallel_config, "data_parallel_size", None)
    content_sharded = finegrained_tp_enabled or (
        getattr(parallel_config, "is_moe_model", None) is not False
        and isinstance(data_parallel_size, int)
        and not isinstance(data_parallel_size, bool)
        and data_parallel_size > 1
    )
    if not content_sharded:
        return None
    return getattr(parallel_config, "data_parallel_rank", None)


def build_seed_key(
    tp_rank: int,
    model_url: str,
    model_deploy_strategy_name: str,
    compatibility_fingerprint: str,
    structural_digest: str,
    is_draft_model: bool = False,
    pp_rank: int | None = None,
    ep_rank: int | None = None,
    sharded_dp_rank: int | None = None,
) -> str:
    if not model_url or not model_deploy_strategy_name:
        raise RuntimeError(
            f"RFork seed key is not set: model_url={model_url!r}, "
            f"model_deploy_strategy_name={model_deploy_strategy_name!r}. "
            "Configure both values through model_loader_extra_config or "
            "MODEL_URL and MODEL_DEPLOY_STRATEGY_NAME."
        )
    if not isinstance(compatibility_fingerprint, str) or not compatibility_fingerprint:
        raise RuntimeError(
            "RFork requires a compatibility fingerprint for the seed key; "
            "build one with build_compatibility_fingerprint()."
        )
    if not isinstance(structural_digest, str) or not structural_digest:
        raise RuntimeError(
            "RFork requires a structural digest for the seed key; "
            "build one with build_structural_digest() over collect_transferable_tensors()."
        )

    descriptor = {
        "compatibility_fingerprint": str(compatibility_fingerprint),
        "structural_digest": structural_digest,
        "model_url": model_url,
        "model_deploy_strategy_name": model_deploy_strategy_name,
        "tp_rank": tp_rank,
        "pp_rank": pp_rank,
        "ep_rank": ep_rank,
        # Shard selector like the ranks above: None while DP ranks hold
        # identical replicated weights, set when content is sharded across DP.
        "sharded_dp_rank": sharded_dp_rank,
        "is_draft_model": bool(is_draft_model),
    }
    canonical_descriptor = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical_descriptor.encode("utf-8")).hexdigest()
