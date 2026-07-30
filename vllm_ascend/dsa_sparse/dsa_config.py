# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA 稀疏卸载的统一拉起参数解析入口。

本文件把用户在 ``additional_config["dsa_sparse_config"]`` 中传入的字段
规范化到 vLLM/vLLM-Ascend 运行时读取的动态 cache 属性、row-mode graph
开关和 trace 配置。解析发生在 worker、EngineCore 和 KV cache 规划使用这些
字段之前，并对未知字段、冲突字段及不兼容图配置做启动期校验。

本模块不参与请求状态推进和推理热路径。后续新增 DSA 拉起参数时应继续在此
集中声明默认值、公开名称和内部映射，避免配置入口再次散落。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from vllm.logger import init_logger

from vllm_ascend.dsa_sparse.dsa_graph_gate import (
    DSA_ROW_MODE_DECODE_GRAPH_CONFIG_KEY,
)
from vllm_ascend.dsa_sparse.dsa_trace import (
    DSA_TRACE_CONFIG_KEY,
    DSA_TRACE_DEFAULT_POINTS,
    DSA_TRACE_DEFAULT_RANKS,
    DSA_TRACE_PUBLIC_KEYS,
)
from vllm_ascend.dsa_sparse.dsa_types import (
    DSA_LIDU_OUTPUT_CAPACITY,
    DSA_LIDU_SUPPORTED_RESIDENT_BUDGETS,
    DSA_SFA_COMPUTE_TOPK,
)
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

logger = init_logger(__name__)

DSA_SPARSE_ADDITIONAL_CONFIG_KEY = "dsa_sparse_config"
_DSA_GRAPH_PUBLIC_CONFIG_KEY = "enable_row_mode_decode_graph"

_DSA_SPARSE_CONFIG_FIELD_MAPPINGS = (
    ("enabled", "enable_dsa_sparse_cache"),
    ("split_indexer_cache", "enable_dsa_split_indexer_cache"),
    ("indexer_mla_block_ratio", "dsa_indexer_mla_block_ratio"),
    ("max_active_reqs", "dsa_max_active_reqs"),
    ("hot_cpu_block_multiple", "dsa_hot_cpu_block_multiple"),
)
_DSA_SPARSE_ACTIVATION_CONFIG_KEY = "sparse_activation_tokens"
_DSA_PROMPT_BUDGET_THRESHOLDS_CONFIG_KEY = "prompt_budget_thresholds"
_DSA_RESIDENT_BUDGET_TOKENS_CONFIG_KEY = "resident_budget_tokens"
_DSA_SPARSE_DEFAULT_CACHE_ATTRS: dict[str, Any] = {
    "enable_dsa_sparse_cache": False,
    "enable_dsa_split_indexer_cache": False,
    "dsa_indexer_mla_block_ratio": 3,
    "dsa_sparse_activation_tokens": 6144,
    "dsa_prompt_budget_thresholds": (32768, 65536),
    "dsa_resident_budget_tokens": (6144, 10240, 12288),
    # 兼容现有 KV 容量规划代码的内部上界，不再是公开配置项。scheduler
    # 实际分配使用每请求冻结的 target resident budget。
    "dsa_hbm_sparse_budget": 12288,
    "dsa_max_active_reqs": 256,
    "dsa_hot_cpu_block_multiple": 3,
}
_DSA_SPARSE_PUBLIC_KEYS = frozenset(
    {public for public, _ in _DSA_SPARSE_CONFIG_FIELD_MAPPINGS}
    | {
        _DSA_SPARSE_ACTIVATION_CONFIG_KEY,
        _DSA_PROMPT_BUDGET_THRESHOLDS_CONFIG_KEY,
        _DSA_RESIDENT_BUDGET_TOKENS_CONFIG_KEY,
        _DSA_GRAPH_PUBLIC_CONFIG_KEY,
        "trace_points",
    }
)


def _normalize_positive_int_sequence(
    raw_value: Any,
    *,
    field_name: str,
) -> tuple[int, ...]:
    if (isinstance(raw_value, (str, bytes, bytearray))
            or not isinstance(raw_value, Sequence)):
        raise TypeError(
            f"dsa_sparse_config[{field_name!r}] must be a sequence of "
            f"positive integers, got {type(raw_value)!r}")
    values = tuple(int(value) for value in raw_value)
    if not values or any(value <= 0 for value in values):
        raise ValueError(
            f"dsa_sparse_config[{field_name!r}] must contain positive "
            f"integers, got {values}")
    return values


def _normalize_dsa_trace_points_config(trace_config: Any) -> dict[str, Any]:
    if isinstance(trace_config, bool):
        return {"enabled": trace_config}
    if not isinstance(trace_config, dict):
        raise TypeError(
            "dsa_sparse_config['trace_points'] must be a dict or bool, got "
            f"{type(trace_config)!r}")

    unknown = sorted(set(trace_config) - DSA_TRACE_PUBLIC_KEYS)
    if unknown:
        raise ValueError(
            "Unknown dsa_sparse_config['trace_points'] key(s): "
            f"{', '.join(unknown)}. Supported keys: "
            f"{sorted(DSA_TRACE_PUBLIC_KEYS)}")

    def normalize_sequence(value: Any) -> Any:
        if value is None or isinstance(value, (str, bytes, bytearray)):
            return value
        if isinstance(value, (set, frozenset)):
            return sorted(value)
        if isinstance(value, Sequence):
            return list(value)
        return value

    return {
        key: normalize_sequence(value)
        for key, value in trace_config.items()
    }


def _normalize_dsa_sparse_config(
    raw_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    unknown = sorted(set(raw_config) - _DSA_SPARSE_PUBLIC_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown dsa_sparse_config key(s): {', '.join(unknown)}. "
            f"Supported keys: {sorted(_DSA_SPARSE_PUBLIC_KEYS)}")

    for field_name in (
            "enabled",
            "split_indexer_cache",
            _DSA_GRAPH_PUBLIC_CONFIG_KEY):
        if (field_name in raw_config
                and not isinstance(raw_config[field_name], bool)):
            raise TypeError(
                f"dsa_sparse_config[{field_name!r}] must be a bool, got "
                f"{type(raw_config[field_name])!r}")

    cache_attrs = dict(_DSA_SPARSE_DEFAULT_CACHE_ATTRS)
    for public_name, cache_attr in _DSA_SPARSE_CONFIG_FIELD_MAPPINGS:
        if public_name in raw_config:
            cache_attrs[cache_attr] = raw_config[public_name]

    if _DSA_SPARSE_ACTIVATION_CONFIG_KEY in raw_config:
        cache_attrs["dsa_sparse_activation_tokens"] = int(
            raw_config[_DSA_SPARSE_ACTIVATION_CONFIG_KEY])
    if _DSA_PROMPT_BUDGET_THRESHOLDS_CONFIG_KEY in raw_config:
        cache_attrs["dsa_prompt_budget_thresholds"] = (
            _normalize_positive_int_sequence(
                raw_config[_DSA_PROMPT_BUDGET_THRESHOLDS_CONFIG_KEY],
                field_name=_DSA_PROMPT_BUDGET_THRESHOLDS_CONFIG_KEY,
            ))
    if _DSA_RESIDENT_BUDGET_TOKENS_CONFIG_KEY in raw_config:
        cache_attrs["dsa_resident_budget_tokens"] = (
            _normalize_positive_int_sequence(
                raw_config[_DSA_RESIDENT_BUDGET_TOKENS_CONFIG_KEY],
                field_name=_DSA_RESIDENT_BUDGET_TOKENS_CONFIG_KEY,
            ))

    activation_tokens = int(cache_attrs["dsa_sparse_activation_tokens"])
    thresholds = tuple(cache_attrs["dsa_prompt_budget_thresholds"])
    budgets = tuple(cache_attrs["dsa_resident_budget_tokens"])
    if activation_tokens <= 0:
        raise ValueError(
            "dsa_sparse_config['sparse_activation_tokens'] must be positive, "
            f"got {activation_tokens}")
    if len(budgets) != len(thresholds) + 1:
        raise ValueError(
            "dsa_sparse_config['resident_budget_tokens'] must contain exactly "
            "one more entry than 'prompt_budget_thresholds': "
            f"thresholds={thresholds}, budgets={budgets}")
    if any(left >= right for left, right in zip(thresholds, thresholds[1:])):
        raise ValueError(
            "dsa_sparse_config['prompt_budget_thresholds'] must be strictly "
            f"increasing, got {thresholds}")
    if any(left > right for left, right in zip(budgets, budgets[1:])):
        raise ValueError(
            "dsa_sparse_config['resident_budget_tokens'] must be "
            f"non-decreasing, got {budgets}")
    if budgets[0] < DSA_SFA_COMPUTE_TOPK:
        raise ValueError(
            "The smallest DSA resident budget must cover SFA-Offload topK: "
            f"budget={budgets[0]}, topk={DSA_SFA_COMPUTE_TOPK}")
    if budgets[-1] > DSA_LIDU_OUTPUT_CAPACITY:
        raise ValueError(
            "The largest DSA resident budget exceeds LIDU output capacity: "
            f"budget={budgets[-1]}, capacity={DSA_LIDU_OUTPUT_CAPACITY}")
    unsupported_budgets = tuple(
        budget for budget in budgets
        if budget not in DSA_LIDU_SUPPORTED_RESIDENT_BUDGETS)
    if unsupported_budgets:
        raise ValueError(
            "DSA resident budgets are not supported by the current LIDU "
            f"kernel: unsupported={unsupported_budgets}, supported="
            f"{DSA_LIDU_SUPPORTED_RESIDENT_BUDGETS}")
    if activation_tokens > budgets[0]:
        raise ValueError(
            "DSA sparse activation cannot exceed the smallest resident "
            f"budget: activation={activation_tokens}, budgets={budgets}")
    max_active_reqs = int(cache_attrs["dsa_max_active_reqs"])
    if max_active_reqs <= 0:
        raise ValueError(
            "dsa_sparse_config['max_active_reqs'] must be positive, got "
            f"{max_active_reqs}")
    cache_attrs["dsa_hbm_sparse_budget"] = max(budgets)

    if cache_attrs["enable_dsa_sparse_cache"]:
        if cache_attrs["enable_dsa_split_indexer_cache"] is False and (
                "split_indexer_cache" in raw_config):
            raise ValueError(
                "dsa_sparse_config['enabled']=True requires "
                "dsa_sparse_config['split_indexer_cache']=True")
        cache_attrs["enable_dsa_split_indexer_cache"] = True
    elif cache_attrs["enable_dsa_split_indexer_cache"]:
        raise ValueError(
            "dsa_sparse_config['split_indexer_cache']=True is only valid when "
            "dsa_sparse_config['enabled']=True. Disable both fields for a "
            "true dense-path A/B run.")

    additional_updates: dict[str, Any] = {}
    if _DSA_GRAPH_PUBLIC_CONFIG_KEY in raw_config:
        additional_updates[DSA_ROW_MODE_DECODE_GRAPH_CONFIG_KEY] = (
            raw_config[_DSA_GRAPH_PUBLIC_CONFIG_KEY])
    trace_config = (
        raw_config["trace_points"]
        if "trace_points" in raw_config
        else {
            "enabled": bool(cache_attrs["enable_dsa_sparse_cache"]),
            "points": list(DSA_TRACE_DEFAULT_POINTS),
            "ranks": list(DSA_TRACE_DEFAULT_RANKS),
        }
    )
    additional_updates[DSA_TRACE_CONFIG_KEY] = (
        _normalize_dsa_trace_points_config(trace_config))

    return cache_attrs, additional_updates


def attach_dsa_sparse_cache_attrs(vllm_config: Any) -> None:
    """Attach DSA cache knobs from ``additional_config`` onto CacheConfig.

    vLLM's core ``CacheConfig`` is backend-agnostic. Users pass DSA
    sparse-offload settings through ``additional_config["dsa_sparse_config"]``;
    vllm-ascend then materializes them as dynamic cache attributes before its
    platform checks and KV-cache allocation patches read those knobs.
    """
    additional_config = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional_config, dict):
        return

    cache_attrs = additional_config.get(DSA_SPARSE_ADDITIONAL_CONFIG_KEY)
    if cache_attrs is None:
        return
    if not isinstance(cache_attrs, dict):
        raise TypeError(
            f"additional_config[{DSA_SPARSE_ADDITIONAL_CONFIG_KEY!r}] must "
            f"be a dict, got {type(cache_attrs)!r}")

    merged_attrs, additional_updates = _normalize_dsa_sparse_config(cache_attrs)
    if merged_attrs["enable_dsa_sparse_cache"]:
        max_active_reqs = int(merged_attrs["dsa_max_active_reqs"])
        max_num_seqs = int(vllm_config.scheduler_config.max_num_seqs or 0)
        if max_num_seqs > max_active_reqs:
            raise ValueError(
                "DSA DRAM request-row capacity must cover scheduler "
                "max_num_seqs: "
                f"max_active_reqs={max_active_reqs}, "
                f"max_num_seqs={max_num_seqs}")
    block_size = int(getattr(vllm_config.cache_config, "block_size", 0) or 0)
    budgets = tuple(merged_attrs["dsa_resident_budget_tokens"])
    activation_tokens = int(merged_attrs["dsa_sparse_activation_tokens"])
    # This hook can run before AscendPlatform.refresh_block_size(). Defer
    # block-alignment checks until the backend has materialized a positive
    # block size; DSASparseBase repeats the same invariants unconditionally
    # when scheduler/worker state is constructed.
    if block_size > 0:
        if any(budget % block_size != 0 for budget in budgets):
            raise ValueError(
                "All DSA resident budgets must be aligned to block_size: "
                f"budgets={budgets}, block_size={block_size}")
        if activation_tokens % block_size != 0:
            raise ValueError(
                "DSA sparse activation must be aligned to block_size so the "
                "first SPARSE row has a complete candidate prefix: "
                f"activation={activation_tokens}, block_size={block_size}")
    for key, value in additional_updates.items():
        if key in additional_config and additional_config[key] != value:
            raise ValueError(
                "Conflicting DSA sparse-offload config for "
                f"additional_config[{key!r}]: {additional_config[key]!r} "
                f"vs {value!r}")
        additional_config[key] = value

    if bool(additional_updates.get(
            DSA_ROW_MODE_DECODE_GRAPH_CONFIG_KEY, False)):
        ascend_compile_config = additional_config.setdefault(
            "ascend_compilation_config", {})
        if not isinstance(ascend_compile_config, dict):
            raise TypeError(
                "additional_config['ascend_compilation_config'] must be a "
                f"dict when DSA graph is enabled, got "
                f"{type(ascend_compile_config)!r}")
        if ascend_compile_config.get("enable_npugraph_ex", False):
            raise ValueError(
                "DSA row-mode decode graph does not support "
                "ascend_compilation_config['enable_npugraph_ex']=True. "
                "DSA graph capture only targets row-mode decode; npugraph_ex "
                "also compiles profile/prefill paths and can fail inside "
                "MoE communication operators. Please set it to False or omit "
                "it.")
        # vllm-ascend defaults npugraph_ex to True. DSA graph mode needs the
        # ACL full-graph replay path, but does not want TorchAir to compile the
        # profiling prefill path before KV-cache split metadata is initialized.
        ascend_compile_config["enable_npugraph_ex"] = False

    for key, value in merged_attrs.items():
        object.__setattr__(vllm_config.cache_config, key, value)


def is_dsa_sparse_config_enabled(vllm_config: Any) -> bool:
    """Return whether DSA sparse offload is requested by user config.

    Some call sites run across vLLM multiprocessing/pydantic boundaries where
    dynamic CacheConfig attributes may not have been materialized yet. Treat
    ``additional_config["dsa_sparse_config"].enabled`` as the source of truth,
    while still accepting an already-attached cache flag.
    """
    if vllm_config is None:
        return False

    additional_config = getattr(vllm_config, "additional_config", None)
    if isinstance(additional_config, dict):
        dsa_config = additional_config.get(
            DSA_SPARSE_ADDITIONAL_CONFIG_KEY)
        if isinstance(dsa_config, dict) and "enabled" in dsa_config:
            # The public switch is the source of truth. In particular, an
            # explicit false must override a stale dynamic CacheConfig
            # attribute after config serialization or an in-process A/B run.
            return bool(dsa_config["enabled"])

    cache_config = getattr(vllm_config, "cache_config", None)
    return bool(
        cache_config is not None
        and getattr(cache_config, "enable_dsa_sparse_cache", False))


def validate_dsa_sparse_runtime_config(vllm_config: Any) -> None:
    """Validate and normalize the supported v0.23 sparse-offload envelope.

    The implementation intentionally has a narrow first-class envelope:
    GLM-5/5.1, decoder-only eager execution, and TP with DP=1 without speculative
    decoding. Options which change cache ownership (prefix cache, KV
    connectors), add speculative tokens, or shard the token domain
    (DP/DCP/PCP/PP) are rejected instead of silently producing an invalid
    resident map.
    """
    attach_dsa_sparse_cache_attrs(vllm_config)
    if not is_dsa_sparse_config_enabled(vllm_config):
        return

    architecture = getattr(vllm_config.model_config, "architecture", None)
    if architecture != "GlmMoeDsaForCausalLM":
        raise ValueError(
            "DSA sparse offload in this v0.23 adaptation is limited to "
            "GLM-5/5.1 (GlmMoeDsaForCausalLM), got "
            f"{architecture!r}")
    if bool(getattr(vllm_config.model_config, "is_encoder_decoder", False)):
        raise ValueError("DSA sparse offload supports decoder-only models")

    device_type = get_ascend_device_type()
    if device_type == AscendDeviceType._310P:
        raise ValueError(
            "DSA sparse offload supports Ascend A2, A3 and A5, not 310P")

    parallel_config = vllm_config.parallel_config
    data_parallel_size = int(
        getattr(parallel_config, "data_parallel_size", 1))
    if data_parallel_size != 1:
        raise ValueError(
            "DSA sparse offload currently supports TP with "
            "data_parallel_size=1 only. DSA request stages, resident rows, "
            "and hot-DRAM block tables are worker-local and are not included "
            "in v0.23 DP metadata synchronization; DP>1 can silently corrupt "
            "token accuracy.")
    incompatible_parallel = {
        "pipeline_parallel_size":
        int(getattr(parallel_config, "pipeline_parallel_size", 1)),
        "decode_context_parallel_size":
        int(getattr(parallel_config, "decode_context_parallel_size", 1)),
        "prefill_context_parallel_size":
        int(getattr(parallel_config, "prefill_context_parallel_size", 1)),
    }
    invalid_parallel = {
        name: value
        for name, value in incompatible_parallel.items() if value != 1
    }
    if invalid_parallel:
        raise ValueError(
            "DSA sparse offload currently supports TP but not PP/DCP/PCP; "
            f"got {invalid_parallel}")

    if getattr(vllm_config, "kv_transfer_config", None) is not None:
        raise ValueError(
            "DSA sparse offload owns the MLA cache and cannot be combined "
            "with a vLLM KV connector/offloading connector")
    if bool(getattr(vllm_config.cache_config, "enable_prefix_caching", False)):
        raise ValueError(
            "DSA sparse offload does not support prefix caching; set "
            "--no-enable-prefix-caching")
    block_size = int(
        getattr(vllm_config.cache_config, "block_size", 0) or 0
    )
    if block_size != 128:
        raise ValueError(
            "KvcacheScatterCopy and LightningIndexerDecodeUpdate preserve "
            "their source ABI and require block_size=128; got "
            f"{block_size}"
        )
    cache_dtype = str(
        getattr(vllm_config.cache_config, "cache_dtype", "auto")
    ).lower()
    if cache_dtype not in {
        "auto",
        "bfloat16",
        "bf16",
        "float16",
        "fp16",
    }:
        raise ValueError(
            "DSA sparse offload requires FP16/BF16 MLA and Indexer cache "
            f"planes; got cache_dtype={cache_dtype!r}"
        )
    model_dtype = str(
        getattr(vllm_config.model_config, "dtype", "")
    ).lower().replace("torch.", "")
    if model_dtype not in {
        "bfloat16",
        "bf16",
        "float16",
        "fp16",
        "half",
    }:
        raise ValueError(
            "KvcacheScatterCopy and LightningIndexerDecodeUpdate preserve "
            "their source FP16/BF16 ABI; got model dtype "
            f"{model_dtype!r}"
        )
    if bool(
            getattr(vllm_config.scheduler_config, "enable_chunked_prefill",
                    False)):
        raise ValueError(
            "DSA sparse offload requires non-chunked prefill; set "
            "--no-enable-chunked-prefill and size max_num_batched_tokens for "
            "the complete prompt")
    if bool(getattr(vllm_config.scheduler_config, "async_scheduling", False)):
        raise ValueError(
            "DSA sparse offload requires synchronous scheduling")

    additional_config = vllm_config.additional_config or {}
    dsa_cp_enabled = additional_config.get("enable_dsa_cp", False)
    if not isinstance(dsa_cp_enabled, bool):
        raise TypeError(
            "additional_config['enable_dsa_cp'] must be a bool when DSA "
            f"sparse offload is enabled, got {type(dsa_cp_enabled)!r}"
        )
    if dsa_cp_enabled:
        raise ValueError(
            "DSA sparse offload cannot be combined with SFA DSA-CP "
            "(additional_config.enable_dsa_cp=true). The two features use "
            "different token sharding, slot mappings, and Indexer/SFA tensor "
            "layouts; set enable_dsa_cp=false."
        )
    incompatible_cache_modes = {
        key: bool(additional_config.get(key, False))
        for key in ("enable_sparse_sfa_c8", "enable_sparse_li_c8")
        if bool(additional_config.get(key, False))
    }
    if incompatible_cache_modes:
        raise ValueError(
            "DSA sparse offload preserves the BF16/FP16 operator ABI and "
            "cannot use sparse C8 cache modes; disable "
            f"{sorted(incompatible_cache_modes)}"
        )
    if bool(additional_config.get(DSA_ROW_MODE_DECODE_GRAPH_CONFIG_KEY, False)):
        raise ValueError(
            "This v0.23 adaptation supports eager DSA sparse offload only; "
            "disable enable_row_mode_decode_graph")
    ascend_config = additional_config.get("ascend_compilation_config", {})
    if isinstance(ascend_config, dict) and bool(
            ascend_config.get("enable_npugraph_ex", False)):
        raise ValueError(
            "DSA sparse offload is eager-only; set "
            "ascend_compilation_config.enable_npugraph_ex=false")

    # The source DSA implementation does not integrate speculative/MTP cache
    # ownership. Letting a multi-token step enter this migrated sparse path can
    # return numerically corrupted tokens without raising a device exception,
    # so keep the unsupported combination fail-closed until it passes
    # device-side token-accuracy validation.
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if speculative_config is not None:
        raise ValueError(
            "DSA sparse offload does not support speculative/MTP decoding "
            "until device-side token accuracy is validated; remove "
            "--speculative-config")

    raw_dsa_config = additional_config.get(
        DSA_SPARSE_ADDITIONAL_CONFIG_KEY, {})
    budgets = tuple(
        int(value) for value in
        vllm_config.cache_config.dsa_resident_budget_tokens)
    if (device_type == AscendDeviceType.A5
            and _DSA_RESIDENT_BUDGET_TOKENS_CONFIG_KEY
            not in raw_dsa_config):
        # A5's native LI interface accepts at most 8192 selected entries.
        # Preserve the three prompt tiers while choosing portable A5 defaults.
        budgets = (6144, 8192, 8192)
        object.__setattr__(vllm_config.cache_config,
                           "dsa_resident_budget_tokens", budgets)
        object.__setattr__(vllm_config.cache_config,
                           "dsa_hbm_sparse_budget", max(budgets))
        logger.info(
            "Using A5-compatible default DSA resident budgets: %s", budgets)

    if device_type in (AscendDeviceType.A2, AscendDeviceType.A3):
        allowed_budgets = {6144, 10240, 12288}
    else:
        allowed_budgets = {6144, 8192}
    unsupported = tuple(
        value for value in budgets if value not in allowed_budgets)
    if unsupported:
        raise ValueError(
            f"DSA resident budgets {unsupported} are unsupported on "
            f"{device_type.name}; supported={sorted(allowed_budgets)}")

    # Apply eager before AscendPlatform snapshots model_config.enforce_eager.
    vllm_config.model_config.enforce_eager = True
