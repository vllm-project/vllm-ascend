# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit, pickleable configuration boundary for model layout adapters."""

from dataclasses import dataclass, fields

from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor

from vllm_ascend.core.layerwise_cache_plan import (
    CacheOwnerDescription,
    LayerwisePolicy,
    PhysicalCachePlan,
    ReuseDependency,
)


@dataclass(frozen=True)
class LayerwiseRuntimeSummary:
    adapter_id: str
    version: int
    policy: LayerwisePolicy
    memory_budget: int
    signature: str
    group_signature: str
    owners: tuple[CacheOwnerDescription, ...]
    dependencies: tuple[ReuseDependency, ...]

    @classmethod
    def from_plan(cls, plan: PhysicalCachePlan) -> "LayerwiseRuntimeSummary":
        return cls(
            plan.adapter_id,
            plan.version,
            plan.policy,
            plan.required_bytes,
            plan.signature,
            plan.group_signature,
            plan.owners,
            plan.dependencies,
        )

    @property
    def has_reuse(self) -> bool:
        return bool(self.dependencies)


@dataclass
class LayerwiseKVCacheConfig(KVCacheConfig):
    # A declared field survives dataclass IPC and scheduler deepcopy. It is
    # deliberately a logical projection, not a device/runtime physical plan.
    layerwise_summary: LayerwiseRuntimeSummary | None = None


@dataclass
class ModelKVCacheTensor(KVCacheTensor):
    # The negative layer stride is a guard, not an affine placement. Only a
    # matching adapter plan can resolve these owner addresses.
    backing_id: str = ""
    layout_signature: str = ""


def bind_model_plan(
    config: KVCacheConfig, plan: PhysicalCachePlan, tensors: list[KVCacheTensor]
) -> LayerwiseKVCacheConfig:
    values = {field.name: getattr(config, field.name) for field in fields(KVCacheConfig)}
    values["kv_cache_tensors"] = tensors
    return LayerwiseKVCacheConfig(**values, layerwise_summary=LayerwiseRuntimeSummary.from_plan(plan))


def get_model_summary(config: KVCacheConfig | None) -> LayerwiseRuntimeSummary | None:
    return config.layerwise_summary if isinstance(config, LayerwiseKVCacheConfig) else None
