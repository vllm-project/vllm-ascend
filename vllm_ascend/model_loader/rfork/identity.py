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
from collections.abc import Mapping
from enum import Enum
from typing import Any

import torch
from vllm.config import ModelConfig, VllmConfig

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.device.hardware_profile import get_current_hardware_profile
from vllm_ascend.model_loader.rfork.types import RFORK_PROTOCOL_VERSION

_FINEGRAINED_TP_LAYOUT_FIELDS = (
    "oproj_tensor_parallel_size",
    "lmhead_tensor_parallel_size",
    "embedding_tensor_parallel_size",
    "mlp_tensor_parallel_size",
    "olora_tensor_parallel_size",
)
_MULTIMODAL_LAYOUT_FIELDS = (
    "limit_per_prompt",
    "mm_encoder_tp_mode",
    "enable_multimodal_pruning",
)
_LORA_LAYOUT_FIELDS = (
    "max_lora_rank",
    "lora_dtype",
    "fully_sharded_loras",
    "lora_extra_vocab_size",
    "bias_enabled",
)
_PROMPT_ADAPTER_LAYOUT_FIELDS = (
    "max_prompt_adapter_token",
    "prompt_adapter_dtype",
)
_POOLER_LAYOUT_FIELDS = ("pooling_type",)
_QUANTIZATION_LAYOUT_FIELDS = (
    "quant_description",
    "model_type",
    "packed_modules_mapping",
    "enable_fa_quant",
    "enable_indexer_quant",
    "enable_c8_quant",
    "kvcache_quant_layers",
    "indexer_quant_layers",
    "c8_quant_layers",
)
_SPECULATIVE_LAYOUT_FIELDS = (
    "method",
    "num_speculative_tokens",
    "num_speculative_tokens_per_batch_size",
    "draft_tensor_parallel_size",
    "parallel_drafting",
    "disable_padded_drafter_batch",
    "enforce_eager",
    "draft_sample_method",
    "speculative_token_tree",
    "quantization",
    "use_local_argmax_reduction",
)


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


def _get_hf_config_descriptor(model_config: ModelConfig) -> list[dict[str, Any]]:
    descriptors: list[dict[str, Any]] = []
    for attr in ("hf_config", "hf_text_config"):
        hf_config = getattr(model_config, attr, None)
        if hf_config is None:
            continue
        to_dict = getattr(hf_config, "to_dict", None)
        config_values = to_dict() if callable(to_dict) else getattr(hf_config, "__dict__", {})
        if not isinstance(config_values, Mapping):
            config_values = {}
        descriptors.append(
            {
                "field": attr,
                "config_type": f"{type(hf_config).__module__}.{type(hf_config).__qualname__}",
                "config": _canonicalize_fingerprint_value(dict(config_values)),
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


def _get_parallel_world_size(parallel_config: object | None, *attrs: str, default: Any = None) -> Any:
    if parallel_config is None:
        return default
    for attr in attrs:
        value = getattr(parallel_config, attr, None)
        if value is not None:
            return _canonicalize_fingerprint_value(value)
    return default


def _get_config_fields(config: object | None, fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: _canonicalize_fingerprint_value(getattr(config, field, None)) for field in fields}


def _get_feature_layout_descriptor(config: object | None, fields: tuple[str, ...]) -> dict[str, Any] | None:
    """Describe only fields that can change a feature's model layout."""
    if config is None:
        return None
    return {
        "config_type": f"{type(config).__module__}.{type(config).__qualname__}",
        "layout": _get_config_fields(config, fields),
    }


def _get_multimodal_layout_descriptor(model_config: ModelConfig) -> dict[str, Any] | None:
    multimodal_config = getattr(model_config, "multimodal_config", None)
    descriptor = _get_feature_layout_descriptor(multimodal_config, _MULTIMODAL_LAYOUT_FIELDS)
    if descriptor is None:
        return None

    get_limit = getattr(multimodal_config, "get_limit_per_prompt", None)
    if callable(get_limit):
        effective_limits = {}
        for modality in ("image", "video", "audio"):
            try:
                effective_limits[modality] = _canonicalize_fingerprint_value(get_limit(modality))
            except Exception:  # pragma: no cover - unsupported modality or third-party config
                effective_limits[modality] = None
        descriptor["effective_limits"] = effective_limits
    is_pruning_enabled = getattr(multimodal_config, "is_multimodal_pruning_enabled", None)
    if callable(is_pruning_enabled):
        descriptor["effective_pruning"] = bool(is_pruning_enabled())
    return descriptor


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


def _get_effective_quantization_descriptor(vllm_config: VllmConfig) -> dict[str, Any] | None:
    """Capture quantization settings used to construct per-layer NPU weights."""
    quant_config = getattr(vllm_config, "quant_config", None)
    if quant_config is None:
        return None
    get_name = getattr(quant_config, "get_name", None)
    name = get_name() if callable(get_name) else None
    return {
        "config_type": f"{type(quant_config).__module__}.{type(quant_config).__qualname__}",
        "name": _canonicalize_fingerprint_value(name),
        "layout": _get_config_fields(quant_config, _QUANTIZATION_LAYOUT_FIELDS),
    }


def _get_model_layout_descriptor(model_config: ModelConfig | None) -> dict[str, Any] | None:
    if model_config is None:
        return None
    return {
        "revision": _get_model_revision(model_config),
        "dtype": _canonicalize_fingerprint_value(getattr(model_config, "dtype", None)),
        "quantization": _canonicalize_fingerprint_value(getattr(model_config, "quantization", None)),
        "quantization_config": _get_quantization_config_digest(model_config),
        "architecture": _get_hf_config_descriptor(model_config),
        "max_model_len": _canonicalize_fingerprint_value(getattr(model_config, "max_model_len", None)),
        "runner_type": _canonicalize_fingerprint_value(getattr(model_config, "runner_type", None)),
        "task": _canonicalize_fingerprint_value(getattr(model_config, "task", None)),
        "convert": _canonicalize_fingerprint_value(getattr(model_config, "convert", None)),
        "use_mla": _canonicalize_fingerprint_value(getattr(model_config, "use_mla", None)),
        "enforce_eager": bool(getattr(model_config, "enforce_eager", False)),
        "multimodal": _get_multimodal_layout_descriptor(model_config),
    }


def _get_speculative_layout_descriptor(vllm_config: VllmConfig) -> dict[str, Any] | None:
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if speculative_config is None:
        return None

    descriptor = _get_config_fields(speculative_config, _SPECULATIVE_LAYOUT_FIELDS)
    descriptor["draft_model"] = _get_model_layout_descriptor(getattr(speculative_config, "draft_model_config", None))
    descriptor["draft_parallel"] = _get_config_fields(
        getattr(speculative_config, "draft_parallel_config", None),
        (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "expert_parallel_size",
            "data_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
            "use_sequence_parallel_moe",
        ),
    )
    return descriptor


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
    except Exception:  # pragma: no cover - only reached before Ascend config setup
        ascend_config = None
    try:
        hardware_profile = get_current_hardware_profile()
        hardware_layout = _get_config_fields(
            hardware_profile,
            (
                "_device_type",
                "weight_layout_policy",
            ),
        )
    except Exception:  # pragma: no cover - only reached on unsupported test hardware
        hardware_layout = {}

    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    cache_config = getattr(vllm_config, "cache_config", None)
    finegrained_tp_config = getattr(ascend_config, "finegrained_tp_config", None)
    finegrained_tp_layout = _get_config_fields(finegrained_tp_config, _FINEGRAINED_TP_LAYOUT_FIELDS)
    finegrained_tp_enabled = any(
        isinstance(size, int) and not isinstance(size, bool) and size > 0 for size in finegrained_tp_layout.values()
    )
    dynamic_spec_config = getattr(ascend_config, "dynamic_spec_config", None)
    multimodal_layout = _get_multimodal_layout_descriptor(model_config)

    descriptor = {
        "rfork_protocol_version": RFORK_PROTOCOL_VERSION,
        "model_url": model_url,
        "model_deploy_strategy_name": model_deploy_strategy_name,
        "model_revision": _get_model_revision(model_config),
        "dtype": _canonicalize_fingerprint_value(getattr(model_config, "dtype", None)),
        "quantization": _canonicalize_fingerprint_value(getattr(model_config, "quantization", None)),
        "quantization_config": _get_quantization_config_digest(model_config),
        "effective_quantization": _get_effective_quantization_descriptor(vllm_config),
        "architecture": _get_hf_config_descriptor(model_config),
        "rope": _get_rope_config_descriptor(model_config),
        "model_runtime": {
            "max_model_len": _canonicalize_fingerprint_value(getattr(model_config, "max_model_len", None)),
            "runner_type": _canonicalize_fingerprint_value(getattr(model_config, "runner_type", None)),
            "task": _canonicalize_fingerprint_value(getattr(model_config, "task", None)),
            "convert": _canonicalize_fingerprint_value(getattr(model_config, "convert", None)),
            "use_mla": _canonicalize_fingerprint_value(getattr(model_config, "use_mla", None)),
            "enforce_eager": bool(getattr(model_config, "enforce_eager", False)),
            "use_v2_model_runner": bool(getattr(vllm_config, "use_v2_model_runner", False)),
        },
        "multimodal_layout": multimodal_layout,
        "optional_model_layout": {
            "lora": _get_feature_layout_descriptor(getattr(vllm_config, "lora_config", None), _LORA_LAYOUT_FIELDS),
            "prompt_adapter": _get_feature_layout_descriptor(
                getattr(vllm_config, "prompt_adapter_config", None), _PROMPT_ADAPTER_LAYOUT_FIELDS
            ),
            "pooler": _get_feature_layout_descriptor(
                getattr(vllm_config, "pooler_config", None), _POOLER_LAYOUT_FIELDS
            ),
        },
        "scheduler_layout": _get_config_fields(
            scheduler_config,
            ("max_num_batched_tokens",),
        ),
        "cache_layout": _get_config_fields(
            cache_config,
            ("block_size", "cache_dtype", "mamba_cache_dtype"),
        ),
        "speculative_layout": _get_speculative_layout_descriptor(vllm_config),
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
            "prefill_context_world_size": _get_parallel_world_size(
                parallel_config, "prefill_context_parallel_size", "pcp_size", default=1
            ),
            "decode_context_world_size": _get_parallel_world_size(
                parallel_config, "decode_context_parallel_size", "dcp_size", default=1
            ),
            "sequence_parallel_moe": bool(getattr(parallel_config, "use_sequence_parallel_moe", False)),
            # Fine-grained TP shards modules across the DP dimension. Include
            # the DP layout position only when those shards are active so
            # ordinary replicated DP ranks can continue sharing seeds.
            "finegrained_data_parallel_rank": (
                _canonicalize_fingerprint_value(getattr(parallel_config, "data_parallel_rank", None))
                if finegrained_tp_enabled
                else None
            ),
        },
        "weight_layout": {
            # P/D roles can materialize different SFA/MLA inference tensors.
            "kv_role": _get_effective_kv_role(vllm_config),
            "hardware": hardware_layout,
            "weight_nz_mode": _canonicalize_fingerprint_value(getattr(ascend_config, "weight_nz_mode", None)),
            # Fused MC2 rewrites MoE storage, so differently configured instances cannot share seeds.
            "enable_fused_mc2": _canonicalize_fingerprint_value(getattr(ascend_config, "enable_fused_mc2", None)),
            "enable_mlapo": _canonicalize_fingerprint_value(getattr(ascend_config, "enable_mlapo", None)),
            "mlapo_keep_prefill_weights": _canonicalize_fingerprint_value(
                getattr(ascend_config, "mlapo_keep_prefill_weights", None)
            ),
            "enable_sparse_sfa_c8": _canonicalize_fingerprint_value(
                getattr(ascend_config, "enable_sparse_sfa_c8", None)
            ),
            "enable_sparse_li_c8": _canonicalize_fingerprint_value(getattr(ascend_config, "enable_sparse_li_c8", None)),
            "c8_reshape_optim_enabled": _canonicalize_fingerprint_value(
                getattr(ascend_config, "c8_reshape_optim_enabled", None)
            ),
            "enable_dsa_cp": _canonicalize_fingerprint_value(getattr(ascend_config, "enable_dsa_cp", None)),
            "mix_placement": _canonicalize_fingerprint_value(getattr(ascend_config, "mix_placement", None)),
            "enable_shared_expert_dp": _canonicalize_fingerprint_value(
                getattr(ascend_config, "enable_shared_expert_dp", None)
            ),
            "enable_sp_by_pass": _canonicalize_fingerprint_value(getattr(ascend_config, "enable_sp_by_pass", None)),
            "pd_tp_ratio": _canonicalize_fingerprint_value(getattr(ascend_config, "pd_tp_ratio", None)),
            "pd_head_ratio": _canonicalize_fingerprint_value(getattr(ascend_config, "pd_head_ratio", None)),
            "num_head_replica": _canonicalize_fingerprint_value(getattr(ascend_config, "num_head_replica", None)),
            "finegrained_tp": finegrained_tp_layout,
            "draft_window_size": _canonicalize_fingerprint_value(getattr(ascend_config, "draft_window_size", None)),
            "dynamic_spec": _get_config_fields(dynamic_spec_config, ("method", "method_params")),
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
        "is_draft_model": bool(is_draft_model),
    }
    canonical_descriptor = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical_descriptor.encode("utf-8")).hexdigest()
