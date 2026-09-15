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
from vllm_ascend.device.hardware_profile import get_current_hardware_profile


def _canonicalize_fingerprint_value(value: Any) -> Any:
    """Convert a configuration value to deterministic JSON-compatible data."""

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
            str(key): _canonicalize_fingerprint_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize_fingerprint_value(item) for item in value]
    if isinstance(value, set | frozenset):
        canonical_values = [_canonicalize_fingerprint_value(item) for item in value]
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
            str(key): _canonicalize_fingerprint_value(item)
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


def _get_hf_config_descriptor(model_config: ModelConfig) -> list[dict[str, Any]]:
    descriptors: list[dict[str, Any]] = []
    for attr in ("hf_config", "hf_text_config"):
        hf_config = getattr(model_config, attr, None)
        if hf_config is None:
            continue
        architectures = getattr(hf_config, "architectures", None)
        if isinstance(architectures, str):
            architectures = [architectures]
        if not isinstance(architectures, (list, tuple)):
            architectures = []
        descriptors.append(
            {
                "field": attr,
                "model_type": _canonicalize_fingerprint_value(getattr(hf_config, "model_type", None)),
                "architectures": sorted(
                    str(architecture) for architecture in architectures if isinstance(architecture, str)
                ),
                "config_type": f"{type(hf_config).__module__}.{type(hf_config).__qualname__}",
            }
        )
    if descriptors:
        return descriptors

    architecture = getattr(model_config, "architecture", None)
    if architecture is None:
        architecture = getattr(model_config, "architectures", None)
    if architecture is not None:
        return [{"field": "model_config", "architectures": _canonicalize_fingerprint_value(architecture)}]

    return [
        {
            "field": "model_config",
            "config_type": f"{type(model_config).__module__}.{type(model_config).__qualname__}",
        }
    ]


def _get_model_revision(model_config: ModelConfig) -> Any:
    revision = getattr(model_config, "revision", None)
    if revision is None:
        revision = getattr(model_config, "model_revision", None)
    if revision is not None:
        return _canonicalize_fingerprint_value(revision)
    for attr in ("hf_config", "hf_text_config"):
        hf_config = getattr(model_config, attr, None)
        for revision_attr in ("_commit_hash", "commit_hash", "revision"):
            revision = getattr(hf_config, revision_attr, None)
            if revision is not None:
                return _canonicalize_fingerprint_value(revision)
    return None


def _get_parallel_world_size(parallel_config: object | None, *attrs: str, default: Any = None) -> Any:
    if parallel_config is None:
        return default
    for attr in attrs:
        value = getattr(parallel_config, attr, None)
        if value is not None:
            return _canonicalize_fingerprint_value(value)
    return default


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

    parallel_config = getattr(vllm_config, "parallel_config", None)

    try:
        ascend_config = get_ascend_config()
        weight_nz_mode = getattr(ascend_config, "weight_nz_mode", None)
    except Exception:  # pragma: no cover - only reached before Ascend config setup
        weight_nz_mode = None
    try:
        hardware_profile = get_current_hardware_profile()
        hardware_policy = getattr(hardware_profile, "weight_layout_policy", None)
    except Exception:  # pragma: no cover - only reached on unsupported test hardware
        hardware_policy = None

    descriptor = {
        "model_url": model_url,
        "model_deploy_strategy_name": model_deploy_strategy_name,
        "model_revision": _get_model_revision(model_config),
        "dtype": _canonicalize_fingerprint_value(getattr(model_config, "dtype", None)),
        "quantization": _canonicalize_fingerprint_value(getattr(model_config, "quantization", None)),
        "quantization_config": _get_quantization_config_digest(model_config),
        "architecture": _get_hf_config_descriptor(model_config),
        "parallel": {
            "tensor_world_size": _get_parallel_world_size(
                parallel_config, "tensor_parallel_size", "tp_size", default=1
            ),
            "pipeline_world_size": _get_parallel_world_size(
                parallel_config, "pipeline_parallel_size", "pp_size", default=1
            ),
            "expert_world_size": _get_parallel_world_size(
                parallel_config, "expert_parallel_size", "ep_size", default=None
            ),
            "data_world_size": _get_parallel_world_size(parallel_config, "data_parallel_size", "dp_size", default=None),
            "expert_parallel_enabled": bool(getattr(parallel_config, "enable_expert_parallel", False)),
            "is_moe_model": _canonicalize_fingerprint_value(getattr(parallel_config, "is_moe_model", None)),
        },
        "weight_layout": {
            "weight_nz_mode": _canonicalize_fingerprint_value(weight_nz_mode),
            "hardware_policy": _canonicalize_fingerprint_value(hardware_policy),
        },
    }
    canonical_json = json.dumps(
        _canonicalize_fingerprint_value(descriptor),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical_json).hexdigest()


def build_seed_key(
    tp_rank: int,
    model_url: str,
    model_deploy_strategy_name: str,
    compatibility_fingerprint: str,
    is_draft_model: bool = False,
    pp_rank: int | None = None,
    ep_rank: int | None = None,
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

    descriptor = {
        "compatibility_fingerprint": str(compatibility_fingerprint),
        "model_url": model_url,
        "model_deploy_strategy_name": model_deploy_strategy_name,
        "tp_rank": tp_rank,
        "pp_rank": pp_rank,
        "ep_rank": ep_rank,
        # Keep this wire-format key for compatibility with existing seed keys.
        "is_draft_worker": bool(is_draft_model),
    }
    canonical_descriptor = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical_descriptor.encode("utf-8")).hexdigest()
