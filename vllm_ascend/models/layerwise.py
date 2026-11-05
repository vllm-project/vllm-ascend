# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Central model adapter selection and configuration/worker boundary."""

from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec

from vllm_ascend.ascend_config import KVPPConfig
from vllm_ascend.core.layerwise_cache_config import ModelKVCacheTensor, bind_model_plan, get_model_summary
from vllm_ascend.core.layerwise_cache_planner import materialize_layerwise_plan, plan_layerwise_cache
from vllm_ascend.models.qwen4_exp.cache_config import build_qwen4_kv_cache_layout
from vllm_ascend.models.qwen4_exp.layerwise import Qwen4LayerwiseAdapter


def resolve_cache_layout_adapter(groups):
    return Qwen4LayerwiseAdapter() if build_qwen4_kv_cache_layout(groups, 1) is not None else None


def uses_model_layerwise_physical_budget(kv_cache_spec):
    """Recognize adapters before scheduler groups exist, without scaling bytes.

    Group membership is immaterial for recognition. Actual capacity planning
    must still use the scheduler's groups and disjoint block-ID contract.
    """
    groups = [KVCacheGroupSpec([name], spec) for name, spec in kv_cache_spec.items()]
    return resolve_cache_layout_adapter(groups) is not None


def get_model_layerwise_context(vllm_config, groups):
    # The old public layout module delegates pure policy to core. Keep the
    # capability check at this integration boundary, avoiding an import cycle.
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
        get_layerwise_reuse_config,
    )

    extra = get_layerwise_reuse_config(getattr(vllm_config, "kv_transfer_config", None))
    if extra is None:
        return None
    adapter = resolve_cache_layout_adapter(groups)
    if adapter is None:
        return None
    policy = adapter.policy(groups, extra)
    probe = adapter.build_layout(groups, policy, 1)
    if not probe.has_reuse:
        return adapter, policy
    if not getattr(vllm_config, "use_v2_model_runner", False):
        raise ValueError("Qwen4 layerwise cache reuse requires model runner V2")
    if not getattr(vllm_config.model_config, "enforce_eager", False):
        raise ValueError("Qwen4 layerwise cache reuse currently requires --enforce-eager")
    if KVPPConfig.from_vllm_config(vllm_config).size > 1:
        raise ValueError("Qwen4 layerwise cache reuse has not been validated with KVPP")
    transfer = vllm_config.kv_transfer_config
    role = transfer.kv_role
    if transfer.kv_connector == "MultiConnector":
        role = next(
            (
                connector.get("kv_role", role)
                for connector in transfer.kv_connector_extra_config["connectors"]
                if connector.get("kv_connector_extra_config") == extra
            ),
            role,
        )
    if role == "kv_consumer" and not extra.get("consumer_is_to_put", False):
        raise ValueError("Qwen4 layerwise cache reuse requires saving every rotating owner")
    parallel = vllm_config.parallel_config
    if any(
        getattr(parallel, key, 1) != 1
        for key in ("pipeline_parallel_size", "decode_context_parallel_size", "prefill_context_parallel_size")
    ):
        raise ValueError("Qwen4 layerwise cache reuse has not been validated with PP/DCP/PCP")
    if getattr(vllm_config, "speculative_config", None) is not None:
        raise ValueError("Qwen4 layerwise cache reuse has not been validated with speculative decoding")
    return adapter, policy


def get_model_layerwise_cache_config(vllm_config, groups, memory_budget):
    context = get_model_layerwise_context(vllm_config, groups)
    if context is None:
        return None
    adapter, policy = context
    if not adapter.build_layout(groups, policy, 1).has_reuse:
        return None
    override = getattr(vllm_config.cache_config, "num_gpu_blocks_override", None)
    plan = plan_layerwise_cache(adapter, groups, policy, memory_budget, override)
    config = KVCacheConfig(
        num_blocks=plan.num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
        prefix_cache_retention_interval=vllm_config.cache_config.prefix_cache_retention_interval,
    )
    return adapter.export(config, plan)


def get_model_layerwise_pool_bytes(vllm_config, groups):
    context = get_model_layerwise_context(vllm_config, groups)
    if context is None:
        return None
    adapter, policy = context
    if not adapter.build_layout(groups, policy, 1).has_reuse:
        return None
    # Qwen4's pages are aligned at every integer N and have no fixed overhead.
    # This hook exists for upstream's linear pool-cost API; full planning uses
    # the exact cost function and validates the final allocation separately.
    one = adapter.build_layout(groups, policy, 1).required_bytes
    if adapter.build_layout(groups, policy, 2).required_bytes != 2 * one:
        raise ValueError("Upstream linear pool-cost hook cannot represent this model layout")
    return one


def rebuild_model_layerwise_plan(config, vllm_config=None, excluded_owners=()):
    summary = get_model_summary(config)
    if summary is None:
        if any(isinstance(tensor, ModelKVCacheTensor) or tensor.layer_stride < 0 for tensor in config.kv_cache_tensors):
            raise ValueError("Non-affine cache descriptors require their model layout summary")
        return None
    adapter = resolve_cache_layout_adapter(config.kv_cache_groups)
    if adapter is None or (adapter.adapter_id, adapter.version) != (summary.adapter_id, summary.version):
        raise ValueError("Layerwise model adapter/version mismatch")
    if set(excluded_owners) - set(summary.policy.excluded_owners):
        raise ValueError("Layerwise owner exclusions changed after capacity planning")
    if vllm_config is not None:
        context = get_model_layerwise_context(vllm_config, config.kv_cache_groups)
        if context is None or context[1] != summary.policy:
            raise ValueError("Layerwise policy changed after capacity planning")
    if not summary.has_reuse:
        if any(isinstance(tensor, ModelKVCacheTensor) or tensor.layer_stride < 0 for tensor in config.kv_cache_tensors):
            raise ValueError("Inactive model summary cannot describe a non-affine layout")
        probe = adapter.build_layout(config.kv_cache_groups, summary.policy, 1)
        if probe.group_signature != summary.group_signature or probe.owners != summary.owners:
            raise ValueError("Inactive layerwise groups or owners changed after planning")
        return None
    plan = materialize_layerwise_plan(
        adapter, config.kv_cache_groups, summary.policy, config.num_blocks, summary.memory_budget
    )
    if plan.signature != summary.signature:
        raise ValueError("Layerwise capacity, groups or backing signature mismatch")
    expected = adapter.export(config, plan).kv_cache_tensors
    if expected != config.kv_cache_tensors:
        raise ValueError("Layerwise descriptors do not match the model plan")
    return plan


def preserve_inactive_model_summary(config, vllm_config):
    context = get_model_layerwise_context(vllm_config, config.kv_cache_groups)
    if context is None:
        return config
    adapter, policy = context
    plan = adapter.build_layout(config.kv_cache_groups, policy, config.num_blocks)
    if plan.has_reuse:
        raise ValueError("Active layerwise layout missed configuration planning")
    return bind_model_plan(config, plan, config.kv_cache_tensors)


def normalize_model_layerwise_capacities(configs):
    """Rebuild normalized offsets while retaining each rank's reserved bytes."""
    minimum = min(config.num_blocks for config in configs)
    result = []
    for config in configs:
        summary = get_model_summary(config)
        if summary is None or not summary.has_reuse:
            result.append(config)
            continue
        adapter = resolve_cache_layout_adapter(config.kv_cache_groups)
        plan = materialize_layerwise_plan(
            adapter, config.kv_cache_groups, summary.policy, minimum, summary.memory_budget
        )
        config.num_blocks = minimum
        result.append(adapter.export(config, plan))
    return result
