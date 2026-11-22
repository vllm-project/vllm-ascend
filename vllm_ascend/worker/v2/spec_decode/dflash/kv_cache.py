# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from typing import Any

import torch
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

from vllm_ascend.attention.context_parallel.sfa_dcp_utils import build_sfa_dcp_replicated_block_table
from vllm_ascend.core.kv_cache_interface import AscendReplicatedDraftCacheSpec


class ReplicatedDFlashBlockTables:
    """Draft-only full-token views of shared target scheduler block IDs.

    Target gathering and slot mapping keep their original tensors. The draft
    preparation kernel and attention metadata both read these persistent
    expanded tables, so context writes, query writes and reads agree.
    """

    def __init__(self, source: Any, replicas: dict[int, int]):
        self._source = source
        self._replicas = replicas
        self.input_block_tables = list(source.input_block_tables)
        self.slot_mappings = torch.full_like(source.slot_mappings, -1)
        self._columns = {}
        self._active_rows = {}
        for gid, factor in replicas.items():
            table = source.input_block_tables[gid]
            width = table.shape[1] * factor
            self.input_block_tables[gid] = table.new_zeros((table.shape[0], width))
            self._columns[gid] = torch.arange(width, dtype=torch.int32, device=table.device)
            self._active_rows[gid] = table.new_ones(table.shape[0])

    def __getattr__(self, name: str):
        return getattr(self._source, name)

    def refresh(self) -> None:
        for gid, factor in self._replicas.items():
            splits = self.block_sizes[gid] // self.kernel_block_sizes[gid]
            output = self.input_block_tables[gid]
            build_sfa_dcp_replicated_block_table(
                self._source.input_block_tables[gid],
                self._active_rows[gid],
                output,
                self._columns[gid],
                factor,
                splits,
            )
            # Manager block zero owns all of these subpages. DFlash's prep
            # kernel recognizes page ID zero as PAD, not its positive siblings.
            output.masked_fill_(output // (factor * splits) == 0, 0)


def draft_block_tables(source: Any, kv_cache_config: Any, draft_layer_names: set[str]) -> Any:
    if source is None:
        return source
    replicas = {}
    for gid, group in enumerate(kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        specs = (
            spec.kv_cache_specs if isinstance(spec, UniformTypeKVCacheSpecs) else dict.fromkeys(group.layer_names, spec)
        )
        draft_specs = [specs[name] for name in group.layer_names if name in draft_layer_names]
        if not draft_specs or not any(isinstance(spec, AscendReplicatedDraftCacheSpec) for spec in draft_specs):
            continue
        factors = {
            spec.replication_size if isinstance(spec, AscendReplicatedDraftCacheSpec) else 1 for spec in draft_specs
        }
        if len(factors) != 1:
            raise ValueError("DFlash layers sharing a block table must use the same physical replication factor")
        replicas[gid] = factors.pop()
    return ReplicatedDFlashBlockTables(source, replicas) if replicas else source
