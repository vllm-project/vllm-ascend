# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup capacity search for Kimi's spec-driven packed cache."""

import math
from copy import copy
from dataclasses import replace

import torch.distributed as dist
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.abstract import get_mamba_prefill_checkpoint_alignment
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.models.kimi_k3.packed_sfa_cache import (
    KIMI_KERNEL_BLOCK_SIZE,
    packed_sfa_cache_groups,
    packed_sfa_max_memory,
    packed_sfa_row_bytes,
)

logger = init_logger(__name__)
MAX_AUTO_BLOCK_SIZE = 65536
SIZING_REFERENCE_TP = 16


def select_packed_sfa_block_size(config, stage_specs, alignment):
    """Minimize the worst PP stage's planner demand, including one null row.

    All stages use the union of actual specs for grouping and keep those
    global groups when projecting physical rows onto each PP stage.
    """
    specs = {}
    for local_specs in stage_specs:
        for name, spec in local_specs.items():
            if name in specs and specs[name] != spec:
                raise ValueError(f"Inconsistent packed Kimi cache spec across PP stages: {name}.")
            specs[name] = spec
    states = [s for s in specs.values() if isinstance(s, MambaSpec)]
    history = [s for s in specs.values() if not isinstance(s, MambaSpec)]
    main_count = sum(isinstance(s, AscendMLAAttentionSpec) and not s.is_draft for s in history)
    state_bytes = max(s.page_size_bytes for s in states)
    token_bytes = sum(s.page_size_bytes // s.block_size for s in history)
    states_per_row = cdiv(len(states), cdiv(len(states), main_count))
    tp_size = config.parallel_config.tensor_parallel_size
    reference_tp = math.lcm(SIZING_REFERENCE_TP, tp_size)
    minimum = (
        KIMI_KERNEL_BLOCK_SIZE
        * cdiv(states_per_row * state_bytes * tp_size, KIMI_KERNEL_BLOCK_SIZE * reference_tp * token_bytes)
        * (reference_tp // tp_size)
    )
    interleave = config.parallel_config.cp_kv_cache_interleave_size
    cp_size = config.parallel_config.prefill_context_parallel_size * config.parallel_config.decode_context_parallel_size
    if cp_size > 1 and (interleave <= 0 or interleave % KIMI_KERNEL_BLOCK_SIZE):
        raise ValueError("Kimi CP interleave must be a positive multiple of the 128-token kernel block.")
    step = math.lcm(KIMI_KERNEL_BLOCK_SIZE, alignment, interleave if cp_size > 1 else 1)
    prefix_unit = getattr(config.cache_config, "prefix_match_unit", None)
    if prefix_unit is not None:
        step = math.lcm(step, prefix_unit)
    best = None
    for block_size in range(cdiv(minimum, step) * step, MAX_AUTO_BLOCK_SIZE + 1, step):
        candidate = copy(config)
        candidate.cache_config = copy(config.cache_config)
        candidate.cache_config.block_size = block_size
        candidate_specs = {
            name: replace(spec, block_size=block_size)
            if not isinstance(spec, MambaSpec) or spec.mamba_cache_mode == "align"
            else spec
            for name, spec in specs.items()
        }
        groups = packed_sfa_cache_groups(candidate, candidate_specs)
        peak = 0
        for local_specs in stage_specs:
            local_groups = [
                replace(group, layer_names=[name for name in group.layer_names if name in local_specs])
                for group in groups
            ]
            row_bytes = packed_sfa_row_bytes(local_groups)
            peak = max(peak, packed_sfa_max_memory(candidate, local_groups) + row_bytes)
        score = (peak, block_size)
        if best is None or score < best:
            best = score
    if best is None:
        raise ValueError(
            f"No packed Kimi block size satisfies alignment {step} within {MAX_AUTO_BLOCK_SIZE}; "
            "set --block-size explicitly."
        )
    return best[1]


def auto_select_packed_sfa_block_size(config):
    """Resolve the omitted block size after target and draft layers are loaded."""
    cache = config.cache_config
    if cache.user_specified_block_size:
        return
    transfer = config.kv_transfer_config
    if transfer is not None and transfer.kv_connector not in ("MooncakeConnectorV2", "MooncakePullConnector"):
        raise ValueError(
            "Automatic packed Kimi block sizing supports MooncakeConnectorV2/MooncakePullConnector; "
            "set --block-size explicitly for other KV connectors."
        )

    # The worker binding uses this same builder. Import lazily to avoid the
    # platform/worker bootstrap cycle and use loaded layers' native dtypes.
    from vllm_ascend.worker.v2.attn_utils import get_kv_cache_spec

    alignment = get_mamba_prefill_checkpoint_alignment(config) or 1
    local_specs = get_kv_cache_spec(config)
    stages = [(local_specs, alignment)]
    if config.parallel_config.pipeline_parallel_size > 1:
        group = get_pp_group()
        stages = [({}, 1) for _ in range(group.world_size)]
        dist.all_gather_object(stages, (local_specs, alignment), group=group.cpu_group)
    block_size = select_packed_sfa_block_size(
        config, [specs for specs, _ in stages], math.lcm(*(value for _, value in stages))
    )
    logger.info("Automatically selected packed Kimi block_size=%d (previous=%d).", block_size, cache.block_size)
    cache.block_size = block_size
    if cache.mamba_cache_mode == "align":
        cache.mamba_block_size = block_size
