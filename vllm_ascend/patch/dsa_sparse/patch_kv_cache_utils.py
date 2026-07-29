# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Patch vLLM KV-cache config generation for DSA split cache groups."""

from __future__ import annotations

from collections import defaultdict

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.utils.mem_utils import format_gib
from vllm.v1.core import kv_cache_utils as kv_utils
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
)

from vllm_ascend.dsa_sparse.dsa_config import (
    attach_dsa_sparse_cache_attrs,
    is_dsa_sparse_config_enabled,
)
from vllm_ascend.dsa_sparse.dsa_model_support import (
    DSA_SPARSE_SUPPORTED_ARCHITECTURES,
    is_dsa_sparse_model_supported,
)
from vllm_ascend.dsa_sparse.dsa_spec_utils import (
    is_dsa_indexer_spec,
    is_dsa_mla_resident_spec,
)

# vLLM's default logging config attaches the stream handler to the "vllm"
# logger tree.  Keep the capacity report under that namespace so INFO-level
# reports are emitted by the normal vLLM handler instead of relying on Python's
# WARNING-level fallback behavior for third-party logger names.
report_logger = init_logger("vllm.dsa_sparse")
_DSA_KV_CONFIGS_WRAPPER_ATTR = "_vllm_ascend_dsa_kv_cache_configs_wrapper"

_ORIGINAL_GET_KV_CACHE_CONFIG_FROM_GROUPS = (
    kv_utils.get_kv_cache_config_from_groups)
_ORIGINAL_GET_KV_CACHE_GROUPS = kv_utils.get_kv_cache_groups
_ORIGINAL_REPORT_KV_CACHE_CONFIG = kv_utils._report_kv_cache_config
_ORIGINAL_MAX_MEMORY_USAGE_BYTES_FROM_GROUPS = (
    kv_utils._max_memory_usage_bytes_from_groups)
_ORIGINAL_GET_KV_CACHE_CONFIGS = kv_utils.get_kv_cache_configs


def _has_indexer_kv_group(kv_cache_groups: list[KVCacheGroupSpec]) -> bool:
    return any(
        is_dsa_indexer_spec(group.kv_cache_spec)
        for group in kv_cache_groups)


def _has_dsa_split_kv_groups(kv_cache_groups: list[KVCacheGroupSpec]) -> bool:
    has_indexer = any(
        is_dsa_indexer_spec(group.kv_cache_spec)
        for group in kv_cache_groups)
    has_mla = any(
        is_dsa_mla_resident_spec(group.kv_cache_spec)
        for group in kv_cache_groups)
    return has_indexer and has_mla


def _has_indexer_kv_spec(kv_cache_specs: dict[str, KVCacheSpec]) -> bool:
    return any(is_dsa_indexer_spec(spec) for spec in kv_cache_specs.values())


def _summarize_kv_cache_specs(
    kv_cache_specs: list[dict[str, KVCacheSpec]],
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(sorted({type(spec).__name__ for spec in specs.values()}))
        for specs in kv_cache_specs)


def _ensure_dsa_indexer_spec_present(
    vllm_config,
    kv_cache_specs: list[dict[str, KVCacheSpec]],
) -> None:
    if not is_dsa_sparse_config_enabled(vllm_config):
        return
    if not is_dsa_sparse_model_supported(vllm_config):
        raise RuntimeError(
            "DSA sparse-cache is enabled for an unsupported model "
            f"architecture={vllm_config.model_config.architecture!r}; "
            "supported_architectures="
            f"{sorted(DSA_SPARSE_SUPPORTED_ARCHITECTURES)}")
    if any(_has_indexer_kv_spec(specs) for specs in kv_cache_specs):
        return

    import vllm.v1.kv_cache_interface as kv_cache_interface
    from vllm_ascend.patch.dsa_sparse.patch_deepseek_v2 import (
        is_dsa_indexer_cache_spec_patch_installed,
    )

    raise RuntimeError(
        "DSA sparse-cache is enabled, but model KV-cache specs do not contain "
        "IndexerKVSpec. This means the shared DSA indexer-cache spec patch "
        "did not take effect before workers reported get_kv_cache_spec(). "
        f"spec_groups={_summarize_kv_cache_specs(kv_cache_specs)} "
        f"cache_flags={{'enable_dsa_sparse_cache': "
        f"{getattr(vllm_config.cache_config, 'enable_dsa_sparse_cache', None)}, "
        f"'enable_dsa_split_indexer_cache': "
        f"{getattr(vllm_config.cache_config, 'enable_dsa_split_indexer_cache', None)}, "
        f"'dsa_indexer_mla_block_ratio': "
        f"{getattr(vllm_config.cache_config, 'dsa_indexer_mla_block_ratio', None)}}} "
        "indexer_spec_patch_installed="
        f"{is_dsa_indexer_cache_spec_patch_installed()} "
        f"kv_interface_indexer_cls={getattr(kv_cache_interface, 'IndexerKVSpec', None)!r}"
    )


def _get_dsa_base_and_group_num_blocks(
    vllm_config,
    available_memory: int,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> tuple[int, list[int]]:
    ratio = int(getattr(vllm_config.cache_config,
                        "dsa_indexer_mla_block_ratio", 3) or 3)
    if ratio <= 0:
        raise ValueError(
            f"dsa_indexer_mla_block_ratio must be positive, got {ratio}")

    weighted_page_size = 0
    for group in kv_cache_groups:
        weight = ratio if is_dsa_indexer_spec(group.kv_cache_spec) else 1
        weighted_page_size += (group.kv_cache_spec.page_size_bytes *
                               len(group.layer_names) * weight)

    num_blocks = available_memory // weighted_page_size
    num_blocks = num_blocks // 128 * 128
    num_blocks = kv_utils.may_override_num_blocks(vllm_config, num_blocks)
    assert num_blocks > 0

    group_num_blocks = [
        num_blocks * ratio
        if is_dsa_indexer_spec(group.kv_cache_spec) else num_blocks
        for group in kv_cache_groups
    ]
    return num_blocks, group_num_blocks


def _get_kv_cache_groups_uniform_block_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    same_type_layers: dict[KVCacheSpec, list[str]] = defaultdict(list)
    _, first_spec = next(iter(kv_cache_spec.items()))
    block_size = first_spec.block_size
    for layer_name, layer_spec in kv_cache_spec.items():
        assert layer_spec.block_size == block_size, (
            "DSA split KV cache groups require a uniform block size.")
        same_type_layers[layer_spec].append(layer_name)
    return kv_utils.create_kv_cache_group_specs(
        kv_cache_spec, list(same_type_layers.values()))


def _get_kv_cache_config_from_groups(
    vllm_config,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    if not _has_indexer_kv_group(kv_cache_groups):
        return _ORIGINAL_GET_KV_CACHE_CONFIG_FROM_GROUPS(
            vllm_config, kv_cache_groups, available_memory)

    num_blocks, group_num_blocks = _get_dsa_base_and_group_num_blocks(
        vllm_config,
        available_memory,
        kv_cache_groups,
    )
    kv_cache_tensors = []
    for group, group_blocks in zip(kv_cache_groups, group_num_blocks):
        setattr(group, "dsa_num_blocks", int(group_blocks))
        for layer_name in group.layer_names:
            kv_cache_tensors.append(
                kv_utils.KVCacheTensor(
                    size=(group.kv_cache_spec.page_size_bytes *
                          int(group_blocks)),
                    shared_by=[layer_name],
                ))
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
    )


def _get_kv_cache_groups(vllm_config,
                         kv_cache_spec: dict[str, KVCacheSpec]):
    keep_dsa_split_groups = _has_indexer_kv_spec(kv_cache_spec)
    if not keep_dsa_split_groups:
        return _ORIGINAL_GET_KV_CACHE_GROUPS(vllm_config, kv_cache_spec)

    if (vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            and not keep_dsa_split_groups):
        kv_utils.unify_hybrid_kv_cache_specs(kv_cache_spec)

    if kv_utils.is_kv_cache_type_attention_free(kv_cache_spec):
        return []
    if kv_utils.is_kv_cache_spec_uniform(kv_cache_spec):
        return kv_utils._get_kv_cache_groups_uniform_spec(kv_cache_spec)
    return _get_kv_cache_groups_uniform_block_size(kv_cache_spec)


def _get_dsa_capacity_metrics(
    mla_num_blocks: int,
    mla_block_size: int,
    indexer_num_blocks: int,
    indexer_block_size: int,
    sparse_budget_tokens: int,
    max_model_len: int,
    max_num_seqs: int,
) -> dict[str, int | str]:
    mla_token_capacity = mla_num_blocks * mla_block_size
    indexer_token_capacity = indexer_num_blocks * indexer_block_size
    if mla_token_capacity <= indexer_token_capacity:
        prefill_token_capacity = mla_token_capacity
        prefill_limiting_plane = "MLA/full"
    else:
        prefill_token_capacity = indexer_token_capacity
        prefill_limiting_plane = "Indexer"

    rounded_sparse_budget = 0
    if sparse_budget_tokens > 0:
        rounded_sparse_budget = ((
            sparse_budget_tokens + mla_block_size - 1) // mla_block_size *
                                 mla_block_size)
    resident_slots_per_decode_request = (
        rounded_sparse_budget + mla_block_size
        if rounded_sparse_budget > 0 else 0)
    decode_requests_by_mla = (
        mla_token_capacity // resident_slots_per_decode_request
        if resident_slots_per_decode_request > 0 else 0)
    indexer_blocks_per_max_model_request = (
        (max_model_len + indexer_block_size - 1) // indexer_block_size
        if max_model_len > 0 else 0)
    decode_requests_by_indexer_at_max_model_len = (
        indexer_num_blocks // indexer_blocks_per_max_model_request
        if indexer_blocks_per_max_model_request > 0 else 0)
    decode_requests_at_max_model_len = min(
        decode_requests_by_mla,
        decode_requests_by_indexer_at_max_model_len,
        max_num_seqs,
    )
    return {
        "mla_token_capacity": mla_token_capacity,
        "indexer_token_capacity": indexer_token_capacity,
        "prefill_token_capacity": prefill_token_capacity,
        "prefill_limiting_plane": prefill_limiting_plane,
        "rounded_sparse_budget": rounded_sparse_budget,
        "resident_slots_per_decode_request":
        resident_slots_per_decode_request,
        "decode_requests_by_mla": decode_requests_by_mla,
        "indexer_blocks_per_max_model_request":
        indexer_blocks_per_max_model_request,
        "decode_requests_by_indexer_at_max_model_len":
        decode_requests_by_indexer_at_max_model_len,
        "decode_requests_at_max_model_len": decode_requests_at_max_model_len,
    }


def _report_dsa_kv_cache_config(vllm_config,
                                kv_cache_config: KVCacheConfig,
                                *,
                                force: bool = False) -> bool:
    if not is_dsa_sparse_config_enabled(vllm_config):
        return False
    if not force:
        if getattr(vllm_config.cache_config, "_dsa_capacity_reported", False):
            return True
        if getattr(kv_cache_config, "_dsa_capacity_reported", False):
            return True

    indexer_groups = [
        group for group in kv_cache_config.kv_cache_groups
        if is_dsa_indexer_spec(group.kv_cache_spec)
    ]
    mla_groups = [
        group for group in kv_cache_config.kv_cache_groups
        if is_dsa_mla_resident_spec(group.kv_cache_spec)
    ]
    if not indexer_groups or not mla_groups:
        return False

    def group_tokens(group: KVCacheGroupSpec) -> int:
        return (int(getattr(group, "dsa_num_blocks",
                            kv_cache_config.num_blocks)) *
                int(group.kv_cache_spec.block_size))

    mla_group = min(mla_groups, key=group_tokens)
    indexer_group = min(indexer_groups, key=group_tokens)
    mla_num_blocks = int(
        getattr(mla_group, "dsa_num_blocks", kv_cache_config.num_blocks))
    indexer_num_blocks = int(
        getattr(indexer_group, "dsa_num_blocks", kv_cache_config.num_blocks))
    mla_block_size = int(mla_group.kv_cache_spec.block_size)
    indexer_block_size = int(indexer_group.kv_cache_spec.block_size)
    sparse_budget_tokens = int(
        getattr(vllm_config.cache_config, "dsa_hbm_sparse_budget", 0) or 0)
    max_model_len = int(vllm_config.model_config.max_model_len)
    max_num_seqs = int(vllm_config.scheduler_config.max_num_seqs)
    ratio = int(getattr(vllm_config.cache_config,
                        "dsa_indexer_mla_block_ratio", 3) or 3)
    metrics = _get_dsa_capacity_metrics(
        mla_num_blocks=mla_num_blocks,
        mla_block_size=mla_block_size,
        indexer_num_blocks=indexer_num_blocks,
        indexer_block_size=indexer_block_size,
        sparse_budget_tokens=sparse_budget_tokens,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )

    total_bytes = 0
    group_lines: list[str] = []
    for idx, group in enumerate(kv_cache_config.kv_cache_groups):
        num_blocks = int(
            getattr(group, "dsa_num_blocks", kv_cache_config.num_blocks))
        block_size = int(group.kv_cache_spec.block_size)
        page_size = int(group.kv_cache_spec.page_size_bytes)
        group_bytes = page_size * num_blocks * len(group.layer_names)
        total_bytes += group_bytes
        plane = ("Indexer dense" if is_dsa_indexer_spec(group.kv_cache_spec) else
                 "MLA/full resident")
        group_lines.append(
            "  - group[%d] %-18s spec=%s layers=%d blocks=%s "
            "block=%s page=%s bytes=%s (%s GiB)" % (
                idx,
                plane,
                type(group.kv_cache_spec).__name__,
                len(group.layer_names),
                f"{num_blocks:,}",
                f"{block_size:,}",
                f"{page_size:,}",
                f"{group_bytes:,}",
                format_gib(group_bytes),
            ))

    report_logger.info(
        "\n"
        "================ DSA HBM CACHE CAPACITY REPORT ================\n"
        "  Split ratio             : indexer:mla = %s:1; base blocks = %s\n"
        "  Allocated HBM KV bytes  : %s bytes (%s GiB)\n"
        "  MLA/full resident plane : %s tokens (%s blocks x %s tokens)\n"
        "  Indexer dense plane     : %s tokens (%s blocks x %s tokens)\n"
        "  Batched prefill limit   : %s tokens (limited by %s; dense KV is "
        "required in both planes)\n"
        "  Sparse decode MLA limit : %s requests (%s resident slots/request "
        "= sparse budget %s + reserved tail capacity %s)\n"
        "  Dense Indexer limit     : total active decode context <= %s tokens; "
        "at max_model_len=%s (%s blocks/request), <= %s requests\n"
        "  Configured decode limit : <= %s requests at max_model_len "
        "(also capped by max_num_seqs=%s)\n"
        "  KV cache groups:\n%s\n"
        "  RISK: Long prefill admission while decode requests are resident "
        "can preempt/rebuild resident decode state; rebuild-cost-aware "
        "admission is not optimized yet.\n"
        "=================================================================",
        f"{ratio:,}",
        f"{int(kv_cache_config.num_blocks):,}",
        f"{total_bytes:,}",
        format_gib(total_bytes),
        f"{metrics['mla_token_capacity']:,}",
        f"{mla_num_blocks:,}",
        f"{mla_block_size:,}",
        f"{metrics['indexer_token_capacity']:,}",
        f"{indexer_num_blocks:,}",
        f"{indexer_block_size:,}",
        f"{metrics['prefill_token_capacity']:,}",
        metrics["prefill_limiting_plane"],
        f"{metrics['decode_requests_by_mla']:,}",
        f"{metrics['resident_slots_per_decode_request']:,}",
        f"{metrics['rounded_sparse_budget']:,}",
        f"{mla_block_size:,}",
        f"{metrics['indexer_token_capacity']:,}",
        f"{max_model_len:,}",
        f"{metrics['indexer_blocks_per_max_model_request']:,}",
        f"{metrics['decode_requests_by_indexer_at_max_model_len']:,}",
        f"{metrics['decode_requests_at_max_model_len']:,}",
        f"{max_num_seqs:,}",
        "\n".join(group_lines),
    )
    setattr(vllm_config.cache_config, "_dsa_capacity_reported", True)
    setattr(kv_cache_config, "_dsa_capacity_reported", True)
    return True


def _raise_missing_dsa_report(vllm_config,
                              kv_cache_config: KVCacheConfig) -> None:
    group_specs = tuple(
        type(group.kv_cache_spec).__name__
        for group in kv_cache_config.kv_cache_groups
    )
    raise RuntimeError(
        "DSA sparse-cache is enabled, but split DSA KV-cache groups were "
        "not generated. Expected at least one IndexerKVSpec group and one "
        "MLA/full resident group. "
        f"group_specs={group_specs} "
        f"split_indexer_cache={getattr(vllm_config.cache_config, 'enable_dsa_split_indexer_cache', None)} "
        f"ratio={getattr(vllm_config.cache_config, 'dsa_indexer_mla_block_ratio', None)} "
        f"hbm_sparse_budget={getattr(vllm_config.cache_config, 'dsa_hbm_sparse_budget', None)}"
    )


def _report_kv_cache_config(vllm_config,
                            kv_cache_config: KVCacheConfig) -> None:
    attach_dsa_sparse_cache_attrs(vllm_config)
    if is_dsa_sparse_config_enabled(vllm_config):
        has_unfinalized_split_groups = (
            _has_dsa_split_kv_groups(kv_cache_config.kv_cache_groups)
            and not all(
                hasattr(group, "dsa_num_blocks")
                for group in kv_cache_config.kv_cache_groups))
        if has_unfinalized_split_groups:
            # Upstream get_kv_cache_configs() calls _report_kv_cache_config()
            # before our wrapper has normalized per-group dsa_num_blocks.  Do
            # not emit a misleading capacity report from that intermediate
            # state; EngineCore reports once after the scheduler KV config is
            # finalized.
            return

    if _report_dsa_kv_cache_config(vllm_config, kv_cache_config):
        return
    if is_dsa_sparse_config_enabled(vllm_config):
        _raise_missing_dsa_report(vllm_config, kv_cache_config)
    _ORIGINAL_REPORT_KV_CACHE_CONFIG(vllm_config, kv_cache_config)


def report_dsa_kv_cache_config_or_raise(
        vllm_config,
        kv_cache_config: KVCacheConfig,
        *,
        force: bool = False,
) -> None:
    attach_dsa_sparse_cache_attrs(vllm_config)
    if not is_dsa_sparse_config_enabled(vllm_config):
        return
    if not _report_dsa_kv_cache_config(
            vllm_config, kv_cache_config, force=force):
        _raise_missing_dsa_report(vllm_config, kv_cache_config)


def _max_memory_usage_bytes_from_groups(
    vllm_config,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    if not _has_indexer_kv_group(kv_cache_groups):
        return _ORIGINAL_MAX_MEMORY_USAGE_BYTES_FROM_GROUPS(
            vllm_config, kv_cache_groups)

    sparse_budget = int(
        getattr(vllm_config.cache_config, "dsa_hbm_sparse_budget", 0) or 0)
    total = 0
    for group in kv_cache_groups:
        spec = group.kv_cache_spec
        if is_dsa_indexer_spec(spec):
            total += len(group.layer_names) * spec.max_memory_usage_bytes(
                vllm_config)
        elif is_dsa_mla_resident_spec(spec):
            resident_tokens = sparse_budget + spec.block_size
            resident_blocks = cdiv(resident_tokens, spec.block_size)
            total += (len(group.layer_names) * resident_blocks *
                      spec.page_size_bytes)
        else:
            total += len(group.layer_names) * spec.max_memory_usage_bytes(
                vllm_config)
    return total


def _fix_dsa_group_num_blocks(kv_cache_configs: list[KVCacheConfig],
                              vllm_config) -> None:
    if not any(_has_indexer_kv_group(config.kv_cache_groups)
               for config in kv_cache_configs):
        return
    min_num_blocks = min(config.num_blocks for config in kv_cache_configs)
    ratio = int(getattr(vllm_config.cache_config,
                        "dsa_indexer_mla_block_ratio", 3) or 3)
    for config in kv_cache_configs:
        for group in config.kv_cache_groups:
            group_blocks = (
                min_num_blocks * ratio
                if is_dsa_indexer_spec(group.kv_cache_spec)
                else min_num_blocks)
            setattr(group, "dsa_num_blocks", int(group_blocks))
        for tensor in config.kv_cache_tensors:
            owner_group = next(
                group for group in config.kv_cache_groups
                if tensor.shared_by[0] in group.layer_names)
            tensor.size = (owner_group.kv_cache_spec.page_size_bytes *
                           int(getattr(owner_group, "dsa_num_blocks",
                                       min_num_blocks)))


def _get_kv_cache_configs(vllm_config, kv_cache_specs, available_memory):
    attach_dsa_sparse_cache_attrs(vllm_config)
    _ensure_dsa_indexer_spec_present(vllm_config, kv_cache_specs)
    configs = _ORIGINAL_GET_KV_CACHE_CONFIGS(
        vllm_config, kv_cache_specs, available_memory)
    _fix_dsa_group_num_blocks(configs, vllm_config)
    if is_dsa_sparse_config_enabled(vllm_config):
        split_configs = [
            config for config in configs
            if _has_dsa_split_kv_groups(config.kv_cache_groups)
        ]
        if not split_configs and configs:
            _raise_missing_dsa_report(vllm_config, configs[0])
        if not configs:
            raise RuntimeError(
                "DSA sparse-cache is enabled, but KV-cache config generation "
                "returned no configs.")
        for config in split_configs:
            if not all(
                    hasattr(group, "dsa_num_blocks")
                    for group in config.kv_cache_groups):
                group_specs = tuple(
                    type(group.kv_cache_spec).__name__
                    for group in config.kv_cache_groups)
                raise RuntimeError(
                    "DSA sparse-cache split groups were generated, but "
                    "dsa_num_blocks was not attached after normalization. "
                    f"group_specs={group_specs}"
                )
    return configs


setattr(_get_kv_cache_configs, _DSA_KV_CONFIGS_WRAPPER_ATTR, True)


def is_dsa_get_kv_cache_configs_wrapper(fn) -> bool:
    return bool(getattr(fn, _DSA_KV_CONFIGS_WRAPPER_ATTR, False))


def describe_callable(fn) -> str:
    return (
        f"{getattr(fn, '__module__', None)}."
        f"{getattr(fn, '__qualname__', getattr(fn, '__name__', None))}"
        f"@{id(fn)}")


def install_dsa_kv_cache_utils_patch() -> None:
    if is_dsa_get_kv_cache_configs_wrapper(kv_utils.get_kv_cache_configs):
        kv_utils._dsa_kv_cache_utils_patched = True
        return
    kv_utils._has_indexer_kv_group = _has_indexer_kv_group
    kv_utils._has_indexer_kv_spec = _has_indexer_kv_spec
    kv_utils._get_dsa_base_and_group_num_blocks = (
        _get_dsa_base_and_group_num_blocks)
    kv_utils._get_kv_cache_groups_uniform_block_size = (
        _get_kv_cache_groups_uniform_block_size)
    kv_utils.get_kv_cache_config_from_groups = (
        _get_kv_cache_config_from_groups)
    kv_utils.get_kv_cache_groups = _get_kv_cache_groups
    kv_utils._report_kv_cache_config = _report_kv_cache_config
    kv_utils._max_memory_usage_bytes_from_groups = (
        _max_memory_usage_bytes_from_groups)
    kv_utils.get_kv_cache_configs = _get_kv_cache_configs
    kv_utils._dsa_kv_cache_utils_patched = True


install_dsa_kv_cache_utils_patch()
