# SPDX-License-Identifier: Apache-2.0
"""Preserve layer specs for placement and project scalar geometry for vLLM consumers."""

from copy import deepcopy
from dataclasses import replace
from functools import wraps

from vllm.v1.core import kv_cache_coordinator, kv_cache_utils, single_type_kv_cache_manager
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import core as engine_core
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    MLAAttentionSpec,
    RSWASpec,
    SinkFullAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.layerwise_cache_config import get_scheduler_spec, scheduler_geometry_config
from vllm_ascend.utils import vllm_version_is

if vllm_version_is("0.32.0"):
    from vllm.v1.core import kv_cache_utils as planning
else:
    from vllm.v1.core import kv_cache_planning as planning


_orig_scheduler_config = kv_cache_utils.generate_scheduler_kv_cache_config
_orig_scheduler_init = Scheduler.__init__
_orig_manager = single_type_kv_cache_manager.get_manager_for_kv_cache_spec


def _merge_attention_geometry(original):
    @classmethod
    @wraps(original)
    def merge(cls, specs):
        # Attention merge reconstructs the spec from geometry and drops plugin fields.
        if not specs[0].reuse:
            return original(cls, specs)
        # v0.32 also compares every input field against the newly constructed spec.
        if vllm_version_is("0.32.0"):
            specs = [replace(spec, reuse=False) for spec in specs]
        return replace(original(cls, specs), reuse=True)

    return merge


for spec_cls in (FullAttentionSpec, MLAAttentionSpec, SinkFullAttentionSpec, RSWASpec, SlidingWindowMLASpec):
    spec_cls.merge = _merge_attention_geometry(spec_cls.merge.__func__)


def _members(spec):
    return spec.kv_cache_specs.values() if isinstance(spec, UniformTypeKVCacheSpecs) else (spec,)


@wraps(_orig_manager)
def _manager(kv_cache_spec, *args, **kwargs):
    return _orig_manager(get_scheduler_spec(kv_cache_spec), *args, **kwargs)


def generate_scheduler_kv_cache_config(kv_cache_configs):
    first = kv_cache_configs[0]
    if not any(isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs) for group in first.kv_cache_groups):
        return _orig_scheduler_config(kv_cache_configs)
    for config in kv_cache_configs[1:]:
        if config.num_blocks != first.num_blocks or getattr(config, "hisparse_host_num_blocks", None) != getattr(
            first, "hisparse_host_num_blocks", None
        ):
            raise ValueError("KV cache block counts differ across workers")
    return deepcopy(first)


@wraps(_orig_scheduler_init)
def _scheduler_init(self, vllm_config, kv_cache_config, *args, **kwargs):
    geometry = scheduler_geometry_config(kv_cache_config)
    if geometry is kv_cache_config:
        return _orig_scheduler_init(self, vllm_config, kv_cache_config, *args, **kwargs)
    # Core managers and third-party connectors take scalar group specs. A
    # layerwise store (including a MultiConnector child) needs the named specs.
    geometry.layerwise_scheduler_config = kv_cache_config
    _orig_scheduler_init(self, vllm_config, geometry, *args, **kwargs)
    self.kv_cache_config = kv_cache_config
    if hasattr(self, "mamba_prefill_checkpoint_alignment"):
        states = [
            spec
            for group in kv_cache_config.kv_cache_groups
            for spec in _members(group.kv_cache_spec)
            if isinstance(spec, MambaSpec)
        ]
        self.mamba_prefill_checkpoint_alignment = next((s.prefill_checkpoint_alignment for s in states), None)
        self.mamba_has_prefill_checkpoint_blocks = self.has_mamba_layers and all(
            s.num_prefill_checkpoint_blocks > 0 for s in states
        )


kv_cache_utils.generate_scheduler_kv_cache_config = generate_scheduler_kv_cache_config
engine_core.generate_scheduler_kv_cache_config = generate_scheduler_kv_cache_config
Scheduler.__init__ = _scheduler_init
single_type_kv_cache_manager.get_manager_for_kv_cache_spec = _manager
kv_cache_coordinator.get_manager_for_kv_cache_spec = _manager


def _check_rank_reuse(original):
    @wraps(original)
    def checked(vllm_config, kv_cache_specs, *args, **kwargs):
        seen = {}
        for rank_specs in kv_cache_specs:
            for name, spec in rank_specs.items():
                if seen.setdefault(name, spec.reuse) != spec.reuse:
                    raise ValueError(f"KV cache reuse intent differs across workers for {name}")
        return original(vllm_config, kv_cache_specs, *args, **kwargs)

    return checked


# v0.32 performs the cross-rank merge inside get_kv_cache_configs.
if hasattr(planning, "get_global_kv_cache_groups"):
    planning.get_global_kv_cache_groups = _check_rank_reuse(planning.get_global_kv_cache_groups)
    engine_core.get_global_kv_cache_groups = planning.get_global_kv_cache_groups
else:
    planning.get_kv_cache_configs = _check_rank_reuse(planning.get_kv_cache_configs)
    engine_core.get_kv_cache_configs = planning.get_kv_cache_configs

if hasattr(kv_cache_utils, "resolve_cache_hit_alignment_tokens"):
    _orig_alignment = kv_cache_utils.resolve_cache_hit_alignment_tokens

    @wraps(_orig_alignment)
    def _alignment(kv_cache_config, *args, **kwargs):
        return _orig_alignment(scheduler_geometry_config(kv_cache_config), *args, **kwargs)

    kv_cache_utils.resolve_cache_hit_alignment_tokens = _alignment
    engine_core.resolve_cache_hit_alignment_tokens = _alignment
