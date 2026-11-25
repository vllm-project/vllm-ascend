# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Ascend KV cache planning rules for vLLM's pluggable config builder.

vLLM plans the KV cache in ``vllm.v1.core.kv_cache_planning`` and resolves the
planner through ``Platform.get_kv_cache_config_builder_cls`` (see
``NPUPlatform.get_kv_cache_config_builder_cls``). Ascend-specific grouping,
pool accounting, and materialization live here, so the planning module itself
no longer needs a monkey patch for these decisions.
"""

from dataclasses import replace

from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_planning import (
    DefaultKVCacheConfigBuilder,
    may_override_num_blocks,
    unify_hybrid_kv_cache_specs,
)
from vllm.v1.core.kv_cache_planning import (
    _pool_bytes_per_block as _upstream_pool_bytes_per_block,
)
from vllm.v1.core.kv_cache_planning import (
    get_kv_cache_config_from_groups as _upstream_get_kv_cache_config_from_groups,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheSpecKind,
    KVCacheTensor,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
    get_kv_cache_spec_kind,
)

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cache_planner import (
    get_layerwise_offload_config,
    get_spec_reuse_rules,
    plan_layerwise_cache_reuse,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    LAYERWISE_REUSE_BACKING_BYTES,
    get_layerwise_reuse_backing_bytes,
)
from vllm_ascend.models.deepseek_v4.cache_config import (
    deepseek_v4_pool_bytes_per_block,
    get_deepseek_v4_cache_layout,
    materialize_deepseek_v4_tensors,
)
from vllm_ascend.models.deepseek_v4.cache_config import (
    is_deepseek_v4_groups as _is_deepseek_v4_groups,
)
from vllm_ascend.models.deepseek_v41.cache_config import (
    get_deepseek_v41_kv_cache_config,
    get_deepseek_v41_pool_bytes_per_block,
    group_cache_specs,
    is_deepseek_v41_cache,
    make_cache_groups,
)
from vllm_ascend.models.glm5next.cache_config import (
    _get_glm5_next_cache_layout,
    get_glm5_next_kv_cache_config,
    get_glm5_next_kv_cache_groups,
    get_glm5_next_pool_bytes_per_block,
)
from vllm_ascend.models.glm5next.kv_cache import is_glm5_next_cache_spec
from vllm_ascend.models.kimi_k3.cache_config import (
    _get_kimi_k3_dspark_mixed_kv_cache_groups,
)
from vllm_ascend.models.kimi_k3.paired_sfa_cache import (
    paired_sfa_cache_config,
    paired_sfa_cache_groups,
    paired_sfa_pool_bytes,
)
from vllm_ascend.models.qwen4_exp.cache_config import (
    get_qwen4_exp_kv_cache_config,
    get_qwen4_exp_kv_cache_groups,
    get_qwen4_exp_pool_bytes_per_block,
)
from vllm_ascend.models.step5.cache_config import (
    get_step5_kv_cache_config_from_groups,
    get_step5_kv_cache_groups,
    get_step5_pool_bytes_per_block,
    is_step5_groups,
)
from vllm_ascend.quantization.methods.kv_cache.turboquant import cache as tq_cache
from vllm_ascend.utils import vllm_version_is


class AscendKVCacheConfigBuilder(DefaultKVCacheConfigBuilder):
    """Build Ascend cache layouts with vLLM's cross-worker planning flow."""

    def get_kv_cache_groups(
        self,
        vllm_config: VllmConfig,
        kv_cache_spec: dict[str, KVCacheSpec],
    ) -> list[KVCacheGroupSpec]:
        """Group layers, preferring the Ascend layout when one applies.

        Precedence: Kimi-K3.1, Step5, GLM-Next, Qwen4-Exp MTP, DeepSeek V4.1,
        Kimi-K3 DSpark, then vLLM's model-owned or default grouping.
        """
        if vllm_config.model_config.hf_config.model_type == "kimi_k31":
            return paired_sfa_cache_groups(vllm_config, kv_cache_spec)
        step5_groups = get_step5_kv_cache_groups(vllm_config, kv_cache_spec, super().get_kv_cache_groups)
        if step5_groups is not None:
            return step5_groups
        if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
            # vLLM's generic grouping promotes local-attention specs before it
            # groups, so the Ascend layouts must see the same promoted specs.
            unify_hybrid_kv_cache_specs(kv_cache_spec)

        if any(is_glm5_next_cache_spec(spec) for spec in kv_cache_spec.values()):
            return get_glm5_next_kv_cache_groups(vllm_config, kv_cache_spec)

        qwen4_exp_groups = get_qwen4_exp_kv_cache_groups(vllm_config, kv_cache_spec, super().get_kv_cache_groups)
        if qwen4_exp_groups is not None:
            return qwen4_exp_groups

        if is_deepseek_v41_cache(kv_cache_spec):
            return make_cache_groups(group_cache_specs(kv_cache_spec))

        kimi_k3_groups = _get_kimi_k3_dspark_mixed_kv_cache_groups(kv_cache_spec)
        if kimi_k3_groups is not None:
            return kimi_k3_groups

        return super().get_kv_cache_groups(vllm_config, kv_cache_spec)

    def get_kv_cache_config_from_groups(
        self,
        vllm_config: VllmConfig,
        kv_cache_groups: list[KVCacheGroupSpec],
        num_blocks: int,
    ) -> KVCacheConfig:
        """Materialize the Ascend layouts for exactly ``num_blocks`` blocks.

        The default implementation derives the pool size from a one-block
        placement, which the shared-slot layouts here violate. Use the native
        pool cost so the model planners materialize exactly the requested block
        count before layerwise reuse is planned.
        """
        bytes_per_block = self._get_native_pool_bytes_per_block(vllm_config, kv_cache_groups)
        if bytes_per_block is None:
            return _attach_transfer_and_speculation_metadata(
                vllm_config,
                super().get_kv_cache_config_from_groups(vllm_config, kv_cache_groups, num_blocks),
            )
        return get_ascend_kv_cache_config_from_groups(
            vllm_config,
            kv_cache_groups,
            num_blocks * bytes_per_block,
        )

    def _get_pool_bytes_per_block(
        self,
        vllm_config: VllmConfig,
        kv_cache_groups: list[KVCacheGroupSpec],
    ) -> int | None:
        """Return the device memory cost per global block ID."""
        extra_config = get_layerwise_offload_config(getattr(vllm_config, "kv_transfer_config", None))
        if extra_config is not None and kv_cache_groups and get_spec_reuse_rules(vllm_config) is not None:
            probe = self.get_kv_cache_config_from_groups(vllm_config, kv_cache_groups, num_blocks=1)
            backing_bytes = get_layerwise_reuse_backing_bytes(probe)
            if backing_bytes is not None:
                return backing_bytes
        return self._get_native_pool_bytes_per_block(vllm_config, kv_cache_groups)

    def _get_native_pool_bytes_per_block(
        self,
        vllm_config: VllmConfig,
        kv_cache_groups: list[KVCacheGroupSpec],
    ) -> int | None:
        """Return the pool cost used to build the native descriptors."""
        if vllm_config.model_config.hf_config.model_type == "kimi_k31":
            return paired_sfa_pool_bytes(kv_cache_groups)
        pools = _get_replicated_draft_pools(kv_cache_groups)
        if pools is not None:
            return sum(_upstream_pool_bytes_per_block(vllm_config, pool) for pool in pools)
        if is_step5_groups(kv_cache_groups):
            return get_step5_pool_bytes_per_block(kv_cache_groups)
        if is_deepseek_v41_cache(kv_cache_groups):
            return get_deepseek_v41_pool_bytes_per_block(kv_cache_groups)
        if _get_glm5_next_cache_layout(kv_cache_groups) is not None:
            return get_glm5_next_pool_bytes_per_block(kv_cache_groups)
        qwen_bytes = get_qwen4_exp_pool_bytes_per_block(kv_cache_groups)
        if qwen_bytes is not None:
            return qwen_bytes
        if tq_cache.uses_turboquant_groups(kv_cache_groups):
            return tq_cache.pool_bytes_per_block(kv_cache_groups)
        if _is_deepseek_v4_groups(kv_cache_groups):
            return deepseek_v4_pool_bytes_per_block(kv_cache_groups)
        return super()._get_pool_bytes_per_block(vllm_config, kv_cache_groups)


def _get_replicated_draft_pools(
    groups: list[KVCacheGroupSpec],
) -> list[list[KVCacheGroupSpec]] | None:
    """Keep contiguous dense draft K/V outside the target/state alias pool."""
    if not any(get_kv_cache_spec_kind(group.kv_cache_spec) == KVCacheSpecKind.MAMBA for group in groups):
        return None
    target_groups = []
    draft_pools = []
    for group in groups:
        spec = group.kv_cache_spec
        specs = spec.kv_cache_specs.values() if isinstance(spec, UniformTypeKVCacheSpecs) else (spec,)
        if all(
            isinstance(s, FullAttentionSpec) and not isinstance(s, MLAAttentionSpec) and not s.dcp_sharded
            for s in specs
        ):
            draft_pools.append([group])
        else:
            target_groups.append(group)
    if target_groups and draft_pools:
        return [target_groups, *draft_pools]
    return None


def _get_kv_cache_config_deepseek_v4_main(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> tuple[int, list[KVCacheTensor]]:
    if tq_cache.uses_turboquant_groups(kv_cache_groups):
        return tq_cache.cache_config(vllm_config, kv_cache_groups, available_memory)
    layout = get_deepseek_v4_cache_layout(kv_cache_groups)
    num_blocks = may_override_num_blocks(vllm_config, available_memory // layout.pool_bytes_per_block)
    return num_blocks, materialize_deepseek_v4_tensors(layout, num_blocks)


def get_ascend_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """Materialize an Ascend layout from the memory budget Core resolved.

    The transfer and speculation metadata is attached by
    ``_attach_transfer_and_speculation_metadata`` before returning; see it for
    why the PD role must ride on the config object.
    """
    qwen_config = (
        None
        if vllm_version_is("0.28.0")
        else get_qwen4_exp_kv_cache_config(vllm_config, kv_cache_groups, available_memory)
    )
    pools = _get_replicated_draft_pools(kv_cache_groups)
    if vllm_config.model_config.hf_config.model_type == "kimi_k31":
        kv_cache_config = paired_sfa_cache_config(vllm_config, kv_cache_groups, available_memory)
    elif is_step5_groups(kv_cache_groups):
        kv_cache_config = get_step5_kv_cache_config_from_groups(vllm_config, kv_cache_groups, available_memory)
    elif pools is not None:
        # Every physical pool shares the scheduler's block count, but owns a
        # disjoint byte range. Use the same aggregate divisor as admission and
        # rank re-planning, then retain upstream's per-pool stride descriptors.
        pool_bytes = [_upstream_pool_bytes_per_block(vllm_config, pool) for pool in pools]
        bytes_per_block = sum(pool_bytes)
        num_blocks = may_override_num_blocks(vllm_config, available_memory // bytes_per_block)
        total_size = num_blocks * bytes_per_block
        kv_cache_tensors: list[KVCacheTensor] = []
        offset = 0
        for pool, pool_bytes_per_block in zip(pools, pool_bytes):
            pool_size = num_blocks * pool_bytes_per_block
            pool_config = _upstream_get_kv_cache_config_from_groups(vllm_config, pool, pool_size)
            assert pool_config.num_blocks == num_blocks
            kv_cache_tensors.extend(
                replace(tensor, size=total_size, offset=offset + tensor.offset)
                for tensor in pool_config.kv_cache_tensors
            )
            offset += pool_size
        kv_cache_config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_groups=kv_cache_groups,
            prefix_cache_retention_interval=vllm_config.cache_config.prefix_cache_retention_interval,
        )
    elif qwen_config is not None:
        kv_cache_config = qwen_config
    elif is_deepseek_v41_cache(kv_cache_groups):
        kv_cache_config = get_deepseek_v41_kv_cache_config(vllm_config, kv_cache_groups, available_memory)
    elif _get_glm5_next_cache_layout(kv_cache_groups) is not None:
        kv_cache_config = get_glm5_next_kv_cache_config(vllm_config, kv_cache_groups, available_memory)
    elif not _is_deepseek_v4_groups(kv_cache_groups):
        kv_cache_config = _upstream_get_kv_cache_config_from_groups(vllm_config, kv_cache_groups, available_memory)
    else:
        num_blocks, kv_cache_tensors = _get_kv_cache_config_deepseek_v4_main(
            vllm_config,
            kv_cache_groups,
            available_memory,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_groups=kv_cache_groups,
            prefix_cache_retention_interval=vllm_config.cache_config.prefix_cache_retention_interval,
        )
    return _attach_transfer_and_speculation_metadata(vllm_config, kv_cache_config)


def _attach_transfer_and_speculation_metadata(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
) -> KVCacheConfig:
    """Keep transfer and speculation metadata on every materialization path.

    Attach ``kv_transfer_config`` onto the built config: ``KVCacheConfig`` has
    no native field, and the PD prefill-producer role read back by the kv-cache
    coordinator (see ``patch_kv_cache_coordinator.py``) must ride on the config
    object - the coordinator factory receives ``KVCacheConfig`` but not
    ``VllmConfig``. The attribute survives the scheduler-side ``copy.deepcopy``
    in ``generate_scheduler_kv_cache_config`` and is dropped by worker pickle
    IPC, which never reads it. The hybrid cache coordinator treats a config
    without the role as standalone and would skip the EAGLE replay block drop
    used for external cache loads, so the upstream default materializer's
    result carries it too.
    """
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    kv_cache_config.kv_transfer_config = kv_transfer_config
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if speculative_config is not None and speculative_config.method == "dspark":
        from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_context import (
            resident_mla_context_group_ids,
        )
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.dspark_prefix_layout import (
            draft_only_group_ids,
        )

        # Upstream scheduler conversion keeps only the first spec in a uniform
        # group, losing both residency and per-layer draft ownership. Keep the
        # PD context and external prefix-store classifications separate: a
        # mixed group needs draft block IDs AND ordinary target hit keys.
        # Remove these snapshots once upstream preserves per-layer ownership.
        kv_cache_config.dspark_context_group_ids = resident_mla_context_group_ids(kv_cache_config.kv_cache_groups)
        kv_cache_config.dspark_draft_only_group_ids = draft_only_group_ids(kv_cache_config.kv_cache_groups)
    reuse_layout = plan_layerwise_cache_reuse(kv_cache_config, vllm_config)
    if reuse_layout is not None:
        assert kv_transfer_config is not None
        # Core budgets these bytes and the scheduler uses the value to detect reuse.
        kv_cache_config.kv_transfer_config = replace(
            kv_transfer_config,
            kv_connector_extra_config={
                **kv_transfer_config.kv_connector_extra_config,
                LAYERWISE_REUSE_BACKING_BYTES: reuse_layout.kv_cache_tensors[0].size,
            },
        )
    return kv_cache_config
