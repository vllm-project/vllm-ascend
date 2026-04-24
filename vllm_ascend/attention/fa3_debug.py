import os
import re
import time
from pathlib import Path
from typing import Any

import torch
from vllm.logger import logger

from vllm_ascend import envs as envs_ascend


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _maybe_get_dist_ranks() -> tuple[int, int]:
    dp_rank = tp_rank = -1
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        from vllm.distributed.parallel_state import get_dp_group

        tp_rank = get_tensor_model_parallel_rank()
        dp_group = get_dp_group()
        dp_rank = getattr(dp_group, "rank_in_group", -1)
    except Exception:
        pass
    return dp_rank, tp_rank


def _dump_limit_reached(dump_dir: Path, stage: str) -> bool:
    limit = envs_ascend.VLLM_ASCEND_FA3_DUMP_LIMIT
    if limit <= 0:
        return False
    return len(list(dump_dir.glob(f"fa3_{stage}_*.pt"))) >= limit


def should_dump_fa3(stage: str, layer_name: str, num_tokens: int) -> bool:
    dump_mode = envs_ascend.VLLM_ASCEND_FA3_DUMP_MODE.lower()
    dump_dir = envs_ascend.VLLM_ASCEND_FA3_DUMP_DIR
    layer_filter = envs_ascend.VLLM_ASCEND_FA3_DUMP_LAYER

    if not dump_dir or dump_mode == "off" or num_tokens <= 1:
        return False
    if dump_mode not in {stage, "both"}:
        return False
    if layer_filter and layer_filter not in layer_name:
        return False

    path = Path(dump_dir)
    path.mkdir(parents=True, exist_ok=True)
    return not _dump_limit_reached(path, stage)


def _detach_to_cpu(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().contiguous()
    return obj


def _take_first_dim(tensor: torch.Tensor, indices: list[int]) -> torch.Tensor:
    if not indices:
        return tensor[:0].detach().clone()
    if len(indices) == 1:
        return tensor.narrow(0, indices[0], 1).contiguous()
    return torch.cat([tensor.narrow(0, idx, 1) for idx in indices], dim=0).contiguous()


def _normalize_seq_lens(seq_lens: list[int] | torch.Tensor) -> list[int]:
    if isinstance(seq_lens, torch.Tensor):
        return [int(x) for x in seq_lens.detach().cpu().tolist()]
    return [int(x) for x in seq_lens]


def canonicalize_fa3_output(output: torch.Tensor, batch_size: int, num_heads: int) -> torch.Tensor:
    if output.dim() == 4:
        if output.shape[0] == batch_size and output.shape[1] == num_heads:
            return output.squeeze(2).contiguous()
        if output.shape[0] == num_heads and output.shape[1] == batch_size:
            return output.permute(1, 0, 2, 3).squeeze(2).contiguous()
    if output.dim() == 3:
        if output.shape[0] == batch_size and output.shape[1] == num_heads:
            return output.contiguous()
        if output.shape[0] == num_heads and output.shape[1] == batch_size:
            return output.permute(1, 0, 2).contiguous()
    raise ValueError(
        f"Unexpected FA3 output shape {tuple(output.shape)} for batch_size={batch_size}, num_heads={num_heads}"
    )


def _build_local_block_view(block_table: torch.Tensor, seq_lens: list[int], block_size: int) -> tuple[list[int], torch.Tensor, list[list[int]]]:
    local_block_table = torch.zeros_like(block_table)
    used_global_block_ids: list[int] = []
    global_to_local: dict[int, int] = {}
    per_req_local_block_ids: list[list[int]] = []

    for row_idx, seq_len in enumerate(seq_lens):
        num_blocks = (seq_len + block_size - 1) // block_size
        req_local_ids: list[int] = []
        for col_idx in range(num_blocks):
            global_block_id = int(block_table[row_idx, col_idx].item())
            local_block_id = global_to_local.get(global_block_id)
            if local_block_id is None:
                local_block_id = len(used_global_block_ids)
                global_to_local[global_block_id] = local_block_id
                used_global_block_ids.append(global_block_id)
            local_block_table[row_idx, col_idx] = local_block_id
            req_local_ids.append(local_block_id)
        per_req_local_block_ids.append(req_local_ids)

    return used_global_block_ids, local_block_table, per_req_local_block_ids


def make_fa3_forward_dump(
    *,
    layer_name: str,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    query_rope: torch.Tensor,
    key_rope: torch.Tensor,
    block_table: torch.Tensor,
    actual_seq_qlen: list[int] | torch.Tensor | None,
    actual_seq_kvlen: list[int] | torch.Tensor,
    dequant_scale_query: torch.Tensor,
    dequant_scale_key: torch.Tensor,
    dequant_scale_value: torch.Tensor,
    block_size: int,
    num_query_heads: int,
    num_key_value_heads: int,
    softmax_scale: float,
    input_layout: str,
    sparse_mode: int,
) -> dict[str, Any]:
    seq_lens = _normalize_seq_lens(actual_seq_kvlen)
    query_lens = _normalize_seq_lens(actual_seq_qlen) if actual_seq_qlen is not None else None
    block_table_cpu = _detach_to_cpu(block_table).to(torch.int32)
    used_global_block_ids, local_block_table, per_req_local_block_ids = _build_local_block_view(
        block_table_cpu, seq_lens, block_size
    )
    local_key_cache = _detach_to_cpu(_take_first_dim(key_cache, used_global_block_ids))
    local_key_rope = _detach_to_cpu(_take_first_dim(key_rope, used_global_block_ids))
    dp_rank, tp_rank = _maybe_get_dist_ranks()

    return {
        "meta": {
            "kind": "fa3_forward_decode",
            "layer_name": layer_name,
            "pid": os.getpid(),
            "timestamp_ns": time.time_ns(),
            "dp_rank": dp_rank,
            "tp_rank": tp_rank,
        },
        "inputs": {
            "query": _detach_to_cpu(query),
            "query_rope": _detach_to_cpu(query_rope),
            "key_cache": local_key_cache,
            "key_rope": local_key_rope,
            "block_table": local_block_table,
            "actual_seq_qlen": query_lens,
            "actual_seq_kvlen": seq_lens,
            "dequant_scale_query": _detach_to_cpu(dequant_scale_query),
            "dequant_scale_key": _detach_to_cpu(dequant_scale_key),
            "dequant_scale_value": _detach_to_cpu(dequant_scale_value),
        },
        "debug": {
            "used_global_block_ids": used_global_block_ids,
            "per_req_local_block_ids": per_req_local_block_ids,
        },
        "op_kwargs": {
            "block_size": int(block_size),
            "num_query_heads": int(num_query_heads),
            "num_key_value_heads": int(num_key_value_heads),
            "softmax_scale": float(softmax_scale),
            "input_layout": input_layout,
            "sparse_mode": int(sparse_mode),
        },
        "outputs": {},
    }


def make_fa3_cache_dump(
    *,
    layer_name: str,
    kv_no_split: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    slots: torch.Tensor,
    kv_cache_before: tuple[torch.Tensor, torch.Tensor],
    kv_cache_after: tuple[torch.Tensor, torch.Tensor],
    block_size: int,
    cache_mode: str,
    kv_scale: torch.Tensor,
) -> dict[str, Any]:
    slots_cpu = _detach_to_cpu(slots).to(torch.int64)
    valid_slots = slots_cpu[slots_cpu >= 0]
    touched_blocks = sorted({int(slot.item()) // block_size for slot in valid_slots})
    before_key, before_value = kv_cache_before
    after_key, after_value = kv_cache_after
    dp_rank, tp_rank = _maybe_get_dist_ranks()

    return {
        "meta": {
            "kind": "fa3_kv_decode",
            "layer_name": layer_name,
            "pid": os.getpid(),
            "timestamp_ns": time.time_ns(),
            "dp_rank": dp_rank,
            "tp_rank": tp_rank,
        },
        "inputs": {
            "kv_no_split": _detach_to_cpu(kv_no_split),
            "cos": _detach_to_cpu(cos),
            "sin": _detach_to_cpu(sin),
            "slots": slots_cpu,
            "touched_global_blocks": touched_blocks,
            "cache_mode": cache_mode,
            "kv_scale": _detach_to_cpu(kv_scale),
            "k_cache_before": _detach_to_cpu(before_key),
            "v_cache_before": _detach_to_cpu(before_value),
            "k_cache_after": _detach_to_cpu(after_key),
            "v_cache_after": _detach_to_cpu(after_value),
        },
    }


def snapshot_touched_cache_blocks(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slots: torch.Tensor,
    block_size: int,
) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    slots_cpu = _detach_to_cpu(slots).to(torch.int64)
    valid_slots = slots_cpu[slots_cpu >= 0]
    touched_blocks = sorted({int(slot.item()) // block_size for slot in valid_slots})
    return (
        touched_blocks,
        _detach_to_cpu(_take_first_dim(key_cache, touched_blocks)),
        _detach_to_cpu(_take_first_dim(value_cache, touched_blocks)),
    )


def persist_fa3_dump(payload: dict[str, Any], stage: str, status: str, error: str | None = None) -> Path:
    dump_dir = Path(envs_ascend.VLLM_ASCEND_FA3_DUMP_DIR)
    dump_dir.mkdir(parents=True, exist_ok=True)

    payload["meta"]["stage"] = stage
    payload["meta"]["status"] = status
    if error is not None:
        payload["meta"]["error"] = error

    meta = payload["meta"]
    file_name = (
        f"fa3_{stage}_{status}_"
        f"dp{meta['dp_rank']}_tp{meta['tp_rank']}_"
        f"pid{meta['pid']}_{_sanitize_name(meta['layer_name'])}_{meta['timestamp_ns']}.pt"
    )
    path = dump_dir / file_name
    torch.save(payload, path)
    logger.info("FA3 debug dump saved to %s", path)
    return path
