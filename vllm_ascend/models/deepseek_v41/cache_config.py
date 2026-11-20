# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Framework-side V4.1 cache placements within a shared allocation."""

from dataclasses import dataclass, fields, replace

import torch
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv

# Both supported versions expose planning helpers in kv_cache_planning.
from vllm.v1.core.kv_cache_planning import may_override_num_blocks
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import (
    KV_CACHE_BUFFER_ALIGNMENT,
    AscendKVCacheConfig,
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
    declared_kwarg,
)
from vllm_ascend.core.kv_cache_pool_plan import CachePoolPlan, CachePoolSpec, RequestStatePoolPlan
from vllm_ascend.device.hardware_profile import (
    HardwareCapability,
    get_current_hardware_profile,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dsv41_state import validate_pd_config

STATE_RING_ROWS = 32
REQUEST_STATE_POOLS_CONFIG_KEY = "enable_deepseek_v41_request_state_pools"

# Fixed dimensions accepted by the packaged mixed-quant attention ABI.
# The actual head sizes come from the model config; these bounds reject
# unsupported layouts before allocating caches, rather than override config.
MLA_HEAD_DIM = 512
INDEX_HEAD_DIM = 128
INDEX_FOLD_ROWS = 8


def uses_a5_packed_cache() -> bool:
    """Select the V4.1 cache ABI from the detected hardware profile."""
    return get_current_hardware_profile().supports(HardwareCapability.DSV41_PACKED_CACHE)


def make_swa_cache_spec(*, block_size, window_size, head_size, dtype, cache_dtype, bounded_replay=False):
    if uses_a5_packed_cache():
        if head_size != MLA_HEAD_DIM:
            raise ValueError(f"A5 DeepSeek V4.1 requires head size {MLA_HEAD_DIM}, got {head_size}")
        # FP8 values followed by one BF16 scale per 32 values.
        head_size += (head_size // 32) * torch.bfloat16.itemsize
        dtype = torch.uint8
        cache_dtype = "a5_mxfp8_bf16_scale"
    return AscendSlidingWindowMLASpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=dtype,
        sliding_window=window_size,
        cache_dtype_str=cache_dtype,
        model_version="deepseek_v41",
        alignment=None,
        **declared_kwarg(AscendSlidingWindowMLASpec, "bounded_replay", bounded_replay),
    )


def make_mla_cache_spec(*, block_size, head_size, compress_ratio):
    if uses_a5_packed_cache():
        if head_size != MLA_HEAD_DIM:
            raise ValueError(f"A5 DeepSeek V4.1 requires head size {MLA_HEAD_DIM}, got {head_size}")
        # Two FP4 values per byte, followed by one BF16 scale per 16 values.
        head_size = head_size // 2 + (head_size // 16) * torch.bfloat16.itemsize
        dtype = torch.uint8
        scale_dtype = torch.bfloat16
    else:
        dtype = torch.bfloat16
        scale_dtype = torch.int8
    return AscendMLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=dtype,
        tokens_per_state=compress_ratio,
        model_version="deepseek_v41",
        # Both supported vLLM lanes (#57169) omit MLA `storage_block_size`.
        **{},
        scale_dtype=scale_dtype,
    )


def make_index_cache_spec(*, block_size, head_size, compress_ratio):
    if uses_a5_packed_cache():
        if head_size != INDEX_HEAD_DIM:
            raise ValueError(f"A5 DeepSeek V4.1 requires head size {INDEX_HEAD_DIM}, got {head_size}")
        scale_dim = head_size // 32
        head_size //= 2
        dtype = torch.uint8
        scale_dtype = torch.uint8
    else:
        dtype = torch.int8
        scale_dim = 1
        scale_dtype = torch.float16
    return AscendMLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=dtype,
        tokens_per_state=compress_ratio,
        model_version="deepseek_v41",
        # Both supported vLLM lanes (#57169) omit MLA `storage_block_size`.
        **{},
        scale_dim=scale_dim,
        scale_dtype=scale_dtype,
    )


def make_folded_index_cache_spec(*, block_size):
    """QSLI candidate-source view: eight K/scale rows per 544-byte row."""
    if block_size % INDEX_FOLD_ROWS:
        raise ValueError("A5 folded index page must contain full 8-token groups")
    return AscendMLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=INDEX_FOLD_ROWS * (INDEX_HEAD_DIM // 2 + INDEX_HEAD_DIM // 32),
        dtype=torch.uint8,
        tokens_per_state=INDEX_FOLD_ROWS,
        model_version="deepseek_v41",
        # Both supported vLLM lanes (#57169) omit MLA `storage_block_size`.
        **{},
    )


def is_deepseek_v41_cache(specs_or_groups):
    if isinstance(specs_or_groups, dict):
        specs = list(specs_or_groups.values())
    else:
        specs = []
        for item in specs_or_groups:
            spec = getattr(item, "kv_cache_spec", item)
            if isinstance(spec, UniformTypeKVCacheSpecs):
                specs.extend(spec.kv_cache_specs.values())
            else:
                specs.append(spec)
    return any(getattr(spec, "model_version", None) == "deepseek_v41" for spec in specs)


def _layer_number(name):
    return int(name.rsplit(".layers.", 1)[1].split(".", 1)[0])


def _draft_layer_number(name):
    return int(("." + name).rsplit(".mtp.", 1)[1].split(".", 1)[0])


def get_layer_tuples(specs):
    """Return DSV4-style ordered layer tuples and their physical page sizes."""
    mla = {name for name, spec in specs.items() if isinstance(spec, AscendMLAAttentionSpec)}
    state = sorted((name for name, spec in specs.items() if isinstance(spec, CircularBufferSpec)), key=_layer_number)
    swa = {name for name, spec in specs.items() if isinstance(spec, AscendSlidingWindowMLASpec)}

    full = sorted((name for name in mla if name.endswith(".long_kv_cache")), key=_layer_number)
    target_swa = sorted((name for name in swa if ".mtp." not in f".{name}"), key=_layer_number)
    draft_swa = sorted((name for name in swa if ".mtp." in f".{name}"), key=_draft_layer_number)

    # A reduced backbone may contain only the initial SWA layers. Without
    # full-context source slots, each SWA layer needs its own physical slot.
    if not full and len(swa) == len(specs):
        names = target_swa + draft_swa
        return [specs[name].unpadded_page_size_bytes for name in names], [(name,) for name in names]

    layer_tuples: list[tuple[str, ...]] = []
    page_sizes: list[int] = []
    for slot_idx, kv_name in enumerate(full):
        prefix = kv_name.rsplit(".", 1)[0]
        index_name = prefix + ".indexer.k_cache"
        folded_name = index_name + "_folded"
        folded_spec = specs.get(folded_name)
        index_spec = specs[index_name]
        kv_spec = specs[kv_name]
        aliases = ([state[slot_idx]] if slot_idx < len(state) else []) + target_swa[slot_idx :: len(full)]
        kv_bytes = kv_spec.unpadded_page_size_bytes
        index_bytes = index_spec.unpadded_page_size_bytes
        folded_bytes = folded_spec.unpadded_page_size_bytes if folded_spec is not None else 0
        if slot_idx < len(draft_swa):
            aliases.append(draft_swa[slot_idx])
        capacity = max(
            kv_bytes + index_bytes + folded_bytes,
            *(specs[name].unpadded_page_size_bytes for name in aliases),
        )
        layer_tuples.append((kv_name, index_name, *((folded_name,) if folded_spec is not None else ()), *aliases))
        page_sizes.append(capacity)
    return page_sizes, layer_tuples


def group_cache_specs(specs):
    """Merge full-context resources and pad layer tuples without mutating inputs."""
    if specs and all(isinstance(spec, AscendSlidingWindowMLASpec) for spec in specs.values()):
        return [UniformTypeKVCacheSpecs.from_specs({name: spec}) for name, spec in specs.items()]
    page_sizes, layer_tuples = get_layer_tuples(specs)
    padded = {}
    for page_size, layer_tuple in zip(page_sizes, layer_tuples):
        kv_name, index_name, *rest = layer_tuple
        folded_name = index_name + "_folded"
        has_folded = folded_name in rest
        aliases = rest[1:] if has_folded else rest
        kv_bytes = specs[kv_name].unpadded_page_size_bytes
        index_bytes = specs[index_name].unpadded_page_size_bytes
        padded[kv_name] = replace(specs[kv_name], page_size_padded=kv_bytes)
        padded[index_name] = replace(
            specs[index_name], page_size_padded=index_bytes if has_folded else page_size - kv_bytes
        )
        if has_folded:
            padded[folded_name] = replace(specs[folded_name], page_size_padded=page_size - kv_bytes - index_bytes)
        padded.update((name, replace(specs[name], page_size_padded=page_size)) for name in aliases)

    mla_names = [name for name, spec in padded.items() if isinstance(spec, AscendMLAAttentionSpec)]
    state_names = [name for name, spec in padded.items() if isinstance(spec, CircularBufferSpec)]
    groups = [
        UniformTypeKVCacheSpecs.from_specs({name: padded[name] for name in mla_names}),
        UniformTypeKVCacheSpecs.from_specs({name: padded[name] for name in state_names}),
    ]

    # Transpose the physical tuples. Each scheduler SWA group takes one layer
    # from every tuple, so its members use distinct slots at the same block ID.
    swa_columns = [
        [
            name
            for name in layer_tuple
            if isinstance(padded[name], AscendSlidingWindowMLASpec) and ".mtp." not in f".{name}"
        ]
        for layer_tuple in layer_tuples
    ]
    for names in zip(*swa_columns):
        groups.append(UniformTypeKVCacheSpecs.from_specs({name: padded[name] for name in names}))

    draft_names = [
        name
        for layer_tuple in layer_tuples
        for name in layer_tuple
        if isinstance(padded[name], AscendSlidingWindowMLASpec) and ".mtp." in f".{name}"
    ]
    if draft_names:
        groups.append(UniformTypeKVCacheSpecs.from_specs({name: padded[name] for name in draft_names}))
    return groups


def make_cache_groups(grouped_specs):
    return [KVCacheGroupSpec(layer_names=list(s.kv_cache_specs), kv_cache_spec=s) for s in grouped_specs]


def _specs_from_groups(groups):
    specs = {}
    for group in groups:
        for name in group.layer_names:
            group_spec = group.kv_cache_spec
            specs[name] = (
                group_spec.kv_cache_specs[name] if isinstance(group_spec, UniformTypeKVCacheSpecs) else group_spec
            )
    return specs


def get_deepseek_v41_pool_bytes_per_block(groups):
    page_sizes, _ = get_layer_tuples(_specs_from_groups(groups))
    return sum(page_sizes)


def get_deepseek_v41_kv_cache_config(
    vllm_config: VllmConfig,
    groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """Place resource pages in layer slots within one shared backing pool."""
    specs = _specs_from_groups(groups)
    page_sizes, layer_tuples = get_layer_tuples(specs)
    capacity = max(available_memory // sum(page_sizes), 0)
    num_blocks = may_override_num_blocks(vllm_config, capacity)
    backing_size = num_blocks * sum(page_sizes)
    tensors: list[KVCacheTensor] = []
    slot_offset = 0
    for page_size, layer_names in zip(page_sizes, layer_tuples):
        # Long KV, split indexer and optional folded indexer are consecutive
        # within a page. State/SWA groups reuse that page from its beginning.
        component_offset = 0
        for name in layer_names:
            spec = specs[name]
            is_mla = isinstance(spec, AscendMLAAttentionSpec)
            tensors.append(
                KVCacheTensor(
                    size=backing_size,
                    layers=[name],
                    offset=slot_offset + (component_offset if is_mla else 0),
                    layer_stride=num_blocks * page_size,
                    block_stride=page_size,
                )
            )
            if is_mla:
                component_offset += spec.unpadded_page_size_bytes
        slot_offset += num_blocks * page_size
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
        prefix_cache_retention_interval=vllm_config.cache_config.prefix_cache_retention_interval,
    )


# The first request-state layout targets the unpacked 128-token SWA kernel.
WINDOW = 128
ROW_BYTES = MLA_HEAD_DIM * torch.bfloat16.itemsize


@dataclass(frozen=True, kw_only=True)
class RequestWindowSpec(AscendSlidingWindowMLASpec):
    @property
    def uses_slot_mapping(self) -> bool:
        return False

    @property
    def block_table_token_alignment(self) -> int | None:
        return None

    @property
    def prefix_cacheable(self) -> bool:
        return False

    def max_num_blocks_per_req(self, vllm_config, max_len) -> int:
        return 1

    def max_admission_blocks_per_request(self, **kwargs) -> int:
        return 1

    def max_memory_usage_bytes(self, vllm_config) -> int:
        return self.page_size_bytes


def unwrap(spec):
    return spec.kv_cache_specs if isinstance(spec, UniformTypeKVCacheSpecs) else {"representative": spec}


def is_request_state_cache(groups: list[KVCacheGroupSpec]) -> bool:
    return any(isinstance(spec, RequestWindowSpec) for group in groups for spec in unwrap(group.kv_cache_spec).values())


def get_request_state_groups(vllm_config: VllmConfig, specs: dict[str, KVCacheSpec]) -> list[KVCacheGroupSpec] | None:
    """Select the opt-in three-pool layout or reject an invalid opt-in."""
    enabled = (getattr(vllm_config, "additional_config", None) or {}).get(REQUEST_STATE_POOLS_CONFIG_KEY, False)
    if type(enabled) is not bool:
        raise ValueError(f"{REQUEST_STATE_POOLS_CONFIG_KEY} must be a boolean")
    if not enabled:
        return None
    if vllm_config.model_config.architecture != "DeepseekV41ForCausalLM" or not is_deepseek_v41_cache(specs):
        raise ValueError(f"{REQUEST_STATE_POOLS_CONFIG_KEY} requires the DeepSeek V4.1 architecture and cache specs")
    parallel = vllm_config.parallel_config
    cache = vllm_config.cache_config
    if not vllm_config.use_v2_model_runner:
        raise ValueError(f"{REQUEST_STATE_POOLS_CONFIG_KEY} requires Model Runner V2")
    if (
        parallel.pipeline_parallel_size,
        parallel.prefill_context_parallel_size,
        parallel.decode_context_parallel_size,
    ) != (1, 1, 1):
        raise ValueError(f"{REQUEST_STATE_POOLS_CONFIG_KEY} requires PP=PCP=DCP=1")
    if (
        cache.enable_prefix_caching
        or getattr(cache, "enable_extensible_kv_cache", False)
        or getattr(cache, "swa_bounded_replay", False)
        or vllm_config.speculative_config is not None
        or vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
        or getattr(vllm_config.model_config, "enable_sleep_mode", False)
        or uses_a5_packed_cache()
    ):
        raise ValueError(
            f"{REQUEST_STATE_POOLS_CONFIG_KEY} requires hybrid cache management and BF16 cache geometry, "
            "with prefix caching, bounded replay, extensible KV, speculative decoding and sleep mode disabled"
        )
    if vllm_config.kv_transfer_config is not None:
        validate_pd_config(vllm_config)

    history, swa, compressor = {}, {}, {}
    for name, spec in specs.items():
        spec = replace(spec, page_size_padded=None)
        if isinstance(spec, AscendSlidingWindowMLASpec):
            if (spec.block_size, spec.sliding_window, spec.dtype, spec.head_size) != (
                WINDOW,
                WINDOW,
                torch.bfloat16,
                MLA_HEAD_DIM,
            ):
                raise ValueError(f"Unsupported request-state SWA cache geometry for {name}")
            attrs = {field.name: getattr(spec, field.name) for field in fields(spec) if field.init}
            swa[name] = RequestWindowSpec(**attrs)
        elif isinstance(spec, CircularBufferSpec):
            compressor[name] = spec
        elif isinstance(spec, AscendMLAAttentionSpec) and name.endswith((".long_kv_cache", ".indexer.k_cache")):
            history[name] = spec
        else:
            raise ValueError(f"Unsupported request-state cache spec for {name}")
    if [len(history), len(swa), len(compressor)] != [8, 40, 3]:
        raise ValueError("Request-state pools require the supported DSV4.1 Flash history/SWA/compressor geometry")
    groups = []
    for pool_specs in (history, swa, compressor):
        spec = UniformTypeKVCacheSpecs.from_specs(pool_specs)
        if spec is None:
            raise ValueError("Incompatible cache specs within a DSV4.1 request-state pool")
        groups.append(spec)
    return make_cache_groups(groups)


def get_window_staging_rows(max_tokens: int, max_requests: int) -> int:
    return cdiv(max_tokens + max_requests * 2 * WINDOW, WINDOW) * WINDOW


def get_request_state_pool_plan(
    vllm_config: VllmConfig, groups: list[KVCacheGroupSpec], num_blocks: int
) -> RequestStatePoolPlan:
    scheduler = vllm_config.scheduler_config
    specs = [
        CachePoolSpec(
            name,
            (group_id,),
            unit,
            sum(spec.unpadded_page_size_bytes for spec in unwrap(group.kv_cache_spec).values()),
        )
        for group_id, ((name, unit), group) in enumerate(
            zip(
                (("history", "token_block"), ("swa", "request_slot"), ("compressor", "request_slot")),
                groups,
                strict=True,
            )
        )
    ]
    rows = get_window_staging_rows(scheduler.max_num_batched_tokens, scheduler.max_num_seqs)
    return RequestStatePoolPlan(
        CachePoolPlan(specs[0], num_blocks),
        tuple(CachePoolPlan(spec, scheduler.max_num_seqs + 1) for spec in specs[1:]),
        workspace_bytes=rows * ROW_BYTES + 2**20,
    )


def get_request_state_pd_alignment_bytes(vllm_config: VllmConfig, groups: list[KVCacheGroupSpec]) -> int:
    # One aligned backing per layer, including both views of an indexer layer.
    if vllm_config.kv_transfer_config is None:
        return 0
    return sum(len(group.layer_names) for group in groups) * KV_CACHE_BUFFER_ALIGNMENT


def materialize_request_state_pools(
    vllm_config: VllmConfig, groups: list[KVCacheGroupSpec], num_blocks: int
) -> AscendKVCacheConfig:
    plan = get_request_state_pool_plan(vllm_config, groups, num_blocks)
    plan.validate_group_coverage(len(groups))
    tensors = [
        KVCacheTensor(
            size=plan.pool_for_group(group_id).capacity * spec.unpadded_page_size_bytes,
            layers=[name],
            offset=0,
            layer_stride=0,
            block_stride=spec.unpadded_page_size_bytes,
        )
        for group_id, group in enumerate(groups)
        for name, spec in unwrap(group.kv_cache_spec).items()
    ]
    if sum(tensor.size for tensor in tensors) + plan.workspace_bytes != plan.size_bytes:
        raise ValueError("Materialized tensor sizes differ from request-state pool budget")
    scheduler = vllm_config.scheduler_config
    return AscendKVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_groups=groups,
        kv_cache_tensors=tensors,
        request_state_plan=plan,
        staging_rows=get_window_staging_rows(scheduler.max_num_batched_tokens, scheduler.max_num_seqs),
        prefix_cache_retention_interval=vllm_config.cache_config.prefix_cache_retention_interval,
    )
