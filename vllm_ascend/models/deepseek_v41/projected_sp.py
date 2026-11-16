# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token-local projections and dtype-preserving sequence gathers."""

from collections.abc import Callable

import torch


def projected_sp_supported(config, ascend_config, *, is_draft_layer: bool, batch_invariant: bool) -> bool:
    """Static eligibility; the decoder additionally checks SP and bucket size."""
    parallel = config.parallel_config
    speculative = config.speculative_config
    dspark = speculative is not None and speculative.use_dspark()
    return (
        ascend_config.dsv41_projected_sp
        and not is_draft_layer
        and (parallel.tensor_parallel_size > 1 or parallel.prefill_context_parallel_size > 1)
        and parallel.decode_context_parallel_size == 1
        and not parallel.use_ubatching
        and (speculative is None or dspark)
        and config.lora_config is None
        and not batch_invariant
    )


def project_shard(attn, hidden: torch.Tensor, *, project_qkv: Callable | None = None) -> dict[str, torch.Tensor]:
    q_a, kv = (attn.wq_a(hidden), attn.wkv(hidden)) if project_qkv is None else project_qkv(attn, hidden)
    local = {"q_a": q_a, "swa_kv": kv}
    if attn.role.is_kv_source:
        compressor = attn.compressor
        source = hidden.float() if attn.role.compress_ratio == 2 else hidden
        local["compressor_kv"] = compressor.wkv(source)
        if attn.role.compress_ratio == 2:
            local["compressor_score"] = compressor.wgate(source)
    if attn.role.is_index_source:
        local["indexer_weights"] = attn.indexer.project_weights(hidden)
    return local


def gather_projections(local: dict[str, torch.Tensor], gather: Callable) -> dict[str, torch.Tensor]:
    """Use the same collective order on every rank and preserve FP32 scores."""
    groups = {}
    for name, value in local.items():
        groups.setdefault(value.dtype, []).append(name)
    result = {}
    for names in groups.values():
        if len(names) == 1:
            name = names[0]
            result[name] = gather(local[name].contiguous()).contiguous()
            continue
        widths = [local[name].shape[-1] for name in names]
        packed = torch.cat([local[name] for name in names], dim=-1)
        gathered = gather(packed)
        for name, value in zip(names, gathered.split(widths, dim=-1)):
            # NPU projection and compressor kernels may require dense rows.
            result[name] = value.contiguous()
    return result


def project_and_gather(
    attn, hidden: torch.Tensor, num_tokens: int, gather: Callable, *, project_qkv: Callable | None = None
) -> dict[str, torch.Tensor]:
    local = project_shard(attn, hidden, project_qkv=project_qkv)
    return gather_projections(local, lambda packed: gather(packed)[:num_tokens])
