"""Round-owned original-KV bindings for accepted-path compaction.

Host keys describe cache views and metadata groups, never live page contents.
Snapshot each binding once so later table-buffer reuse cannot change it.
"""

from __future__ import annotations

import torch
from vllm.v1.kv_cache_interface import CircularBufferSpec, MambaSpec

from vllm_ascend.worker.v2.spec_decode.tree.kv_layout import iter_unique_kv_cache_tensors


def _view_key(tensor):
    return (str(tensor.device), tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)


def _group_is_linear_token_cache(group) -> bool:
    """Only ordinary token caches participate in slot relocation."""
    if any("state_cache" in name for name in group.layer_names):
        return False
    spec = group.kv_cache_spec
    if isinstance(spec, (CircularBufferSpec, MambaSpec)) or getattr(spec, "cache_role", None):
        return False
    inner = getattr(spec, "kv_cache_specs", None)
    for item in inner.values() if inner else [spec]:
        if isinstance(item, (CircularBufferSpec, MambaSpec)) or getattr(item, "cache_role", None):
            return False
        ratio = getattr(item, "tokens_per_state", 1)
        if ratio is not None and ratio > 1:
            return False
    return True


def bind_linear_kv(compactor, num_reqs=None):
    """Bind target token caches once, excluding draft and recurrent states."""
    if compactor._groups is not None:
        return compactor._groups
    runner = compactor.runner
    ctx = runner.compilation_config.static_forward_context
    draft_names = set(runner.speculator.draft_attn_layer_names)
    groups, seen = [], set()
    for group_id, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        if not _group_is_linear_token_cache(group):
            continue
        caches, gathers = [], []
        block_size = runner.block_tables.kernel_block_sizes[group_id]
        for layer_name in group.layer_names:
            if layer_name in draft_names:
                continue
            layer = ctx.get(layer_name)
            if layer is None:
                continue
            for tensor in iter_unique_kv_cache_tensors(layer.kv_cache):
                key = (group_id, _view_key(tensor))
                if key in seen:
                    continue
                seen.add(key)
                if tensor.shape[1] != block_size:
                    raise RuntimeError("Target KV page size differs from block-table metadata")
                caches.append(tensor)
                gathers.append(
                    torch.empty(
                        (compactor.max_num_reqs * compactor.spec_len, *tensor.shape[2:]),
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )
                )
        if caches:
            groups.append((caches, runner.block_tables.block_tables[group_id].gpu, block_size, gathers))
    return groups
