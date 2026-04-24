import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from vllm.logger import logger

from vllm_ascend import envs as envs_ascend

_CURRENT_CONTEXT: dict[str, Any] = {}


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _debug_dir() -> Path | None:
    debug_dir = envs_ascend.VLLM_ASCEND_IDENTICAL_REQ_DEBUG_DIR
    if not debug_dir:
        return None
    path = Path(debug_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _artifact_limit_reached() -> bool:
    path = _debug_dir()
    limit = envs_ascend.VLLM_ASCEND_IDENTICAL_REQ_DEBUG_LIMIT
    if path is None or limit <= 0:
        return False
    return len(list(path.glob("identical_req_*.pt"))) >= limit


def _mode_enabled(kind: str) -> bool:
    mode = envs_ascend.VLLM_ASCEND_IDENTICAL_REQ_DEBUG_MODE
    if mode == "off":
        return False
    if mode == "all":
        return True
    return mode == kind


def _is_model_level_name(name: str) -> bool:
    return name.startswith("model.")


def infer_stage_from_attn_metadata(attn_metadata: Any) -> str:
    if isinstance(attn_metadata, list):
        sample = next((metadata for metadata in attn_metadata if metadata), {})
        if sample:
            sample = next(iter(sample.values()))
    elif isinstance(attn_metadata, dict):
        sample = next(iter(attn_metadata.values())) if attn_metadata else None
    else:
        sample = attn_metadata

    if sample is None:
        return "unknown"

    num_prefills = getattr(sample, "num_prefills", 0) or 0
    num_decodes = getattr(sample, "num_decodes", 0) or 0
    if num_prefills > 0 and num_decodes == 0:
        return "prefill"
    if num_prefills == 0 and num_decodes > 0:
        return "decode"
    if num_prefills > 0 and num_decodes > 0:
        return "mixed"
    return "unknown"


def set_request_divergence_context(*, stage: str, decode_step: int | None, req_ids: list[str]) -> None:
    _CURRENT_CONTEXT.clear()
    _CURRENT_CONTEXT.update(
        {
            "stage": stage,
            "decode_step": decode_step,
            "req_ids": list(req_ids),
        }
    )


def get_request_divergence_context() -> dict[str, Any]:
    return dict(_CURRENT_CONTEXT)


def clear_request_divergence_context() -> None:
    _CURRENT_CONTEXT.clear()


def should_dump_logits(step_id: int | None = None) -> bool:
    if not _mode_enabled("logits"):
        return False
    if _debug_dir() is None or _artifact_limit_reached():
        return False
    target_step = envs_ascend.VLLM_ASCEND_IDENTICAL_REQ_DEBUG_STEP
    if target_step >= 0:
        if step_id is None:
            return False
        if target_step != step_id:
            return False
    return True


def should_dump_layer(layer_name: str, step_id: int | None = None) -> bool:
    if not _mode_enabled("layer"):
        return False
    if _debug_dir() is None or _artifact_limit_reached():
        return False
    layer_filter = envs_ascend.VLLM_ASCEND_IDENTICAL_REQ_DEBUG_LAYER
    if layer_filter and not _is_model_level_name(layer_name) and layer_filter not in layer_name:
        return False
    target_step = envs_ascend.VLLM_ASCEND_IDENTICAL_REQ_DEBUG_STEP
    if target_step >= 0:
        if step_id is None:
            return False
        if target_step != step_id:
            return False
    return True


def should_dump_tensor(layer_name: str, step_id: int | None = None) -> bool:
    if not _mode_enabled("tensor"):
        return False
    if _debug_dir() is None or _artifact_limit_reached():
        return False
    layer_filter = envs_ascend.VLLM_ASCEND_IDENTICAL_REQ_DEBUG_LAYER
    if layer_filter and not _is_model_level_name(layer_name) and layer_filter not in layer_name:
        return False
    target_step = envs_ascend.VLLM_ASCEND_IDENTICAL_REQ_DEBUG_STEP
    if target_step >= 0:
        if step_id is None:
            return False
        if target_step != step_id:
            return False
    return True


def _detach_to_cpu(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().contiguous()
    return obj


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


def _persist_artifact(kind: str, payload: dict[str, Any]) -> Path | None:
    debug_dir = _debug_dir()
    if debug_dir is None or _artifact_limit_reached():
        return None

    meta = payload.setdefault("meta", {})
    dp_rank, tp_rank = _maybe_get_dist_ranks()
    meta.setdefault("pid", os.getpid())
    meta.setdefault("dp_rank", dp_rank)
    meta.setdefault("tp_rank", tp_rank)
    meta.setdefault("timestamp_ns", time.time_ns())
    file_name = (
        f"identical_req_{kind}_{meta.get('stage', 'unknown')}_"
        f"step{meta.get('decode_step', 'na')}_dp{meta.get('dp_rank', 'na')}_tp{meta.get('tp_rank', 'na')}_"
        f"{_sanitize_name(meta.get('name', 'artifact'))}_"
        f"{meta['timestamp_ns']}.pt"
    )
    path = debug_dir / file_name
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=debug_dir, prefix=file_name + ".", suffix=".tmp", delete=False) as tmp:
            tmp_path = tmp.name
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
        logger.info("Identical-request debug artifact saved to %s", path)
        return path
    except Exception as exc:
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass
        logger.warning("Skip identical-request debug artifact %s due to dump failure: %s", path, exc)
        return None


def dump_logits_summary(logits: torch.Tensor, req_ids: list[str], stage: str, decode_step: int | None) -> Path | None:
    if not should_dump_logits(decode_step):
        return None
    if logits.ndim != 2 or logits.shape[0] < 2:
        return None

    num_reqs = min(len(req_ids), logits.shape[0])
    logits_cpu = logits[:num_reqs].detach().float().cpu()
    topk = min(envs_ascend.VLLM_ASCEND_IDENTICAL_REQ_DEBUG_TOPK, logits_cpu.shape[-1])
    topk_logits, topk_ids = torch.topk(logits_cpu, k=topk, dim=-1)
    ref = logits_cpu[0:1]
    cosine = torch.nn.functional.cosine_similarity(logits_cpu, ref.expand_as(logits_cpu), dim=-1)
    max_abs_diff = (logits_cpu - ref).abs().amax(dim=-1)
    argmax_ids = topk_ids[:, 0]
    diverged = bool(torch.any(topk_ids != topk_ids[0:1]) or torch.any(argmax_ids != argmax_ids[0]))

    payload = {
        "meta": {
            "kind": "logits",
            "name": "compute_logits",
            "stage": stage,
            "decode_step": decode_step,
            "diverged": diverged,
        },
        "req_ids": req_ids[:num_reqs],
        "summary": {
            "shape": list(logits_cpu.shape),
            "argmax_ids": argmax_ids.tolist(),
            "argmax_logits": topk_logits[:, 0].tolist(),
            "topk_ids": topk_ids.tolist(),
            "topk_logits": topk_logits.tolist(),
            "mean": logits_cpu.mean(dim=-1).tolist(),
            "max": logits_cpu.max(dim=-1).values.tolist(),
            "min": logits_cpu.min(dim=-1).values.tolist(),
            "norm": logits_cpu.norm(dim=-1).tolist(),
            "cosine_to_req0": cosine.tolist(),
            "max_abs_diff_to_req0": max_abs_diff.tolist(),
        },
    }
    if envs_ascend.VLLM_ASCEND_IDENTICAL_REQ_DEBUG_MODE == "all" and diverged:
        payload["full_logits"] = logits_cpu
    return _persist_artifact("logits", payload)


def dump_layer_summary(
    *,
    layer_name: str,
    tensor: torch.Tensor,
    query_lens: list[int] | None,
    tag: str,
) -> Path | None:
    ctx = get_request_divergence_context()
    stage = ctx.get("stage", "unknown")
    decode_step = ctx.get("decode_step")
    req_ids = ctx.get("req_ids", [])
    if not should_dump_layer(layer_name, decode_step):
        return None
    if query_lens is None or len(query_lens) < 2 or tensor.ndim < 2:
        return None

    total_tokens = sum(int(length) for length in query_lens)
    if total_tokens <= 0 or tensor.shape[0] < total_tokens:
        return None

    tensor_cpu = tensor[:total_tokens].detach().float().cpu()
    req_last_tokens = []
    end = 0
    for query_len in query_lens:
        end += int(query_len)
        req_last_tokens.append(tensor_cpu[end - 1])
    req_last_tokens_tensor = torch.stack(req_last_tokens, dim=0)
    ref = req_last_tokens_tensor[0:1]
    cosine = torch.nn.functional.cosine_similarity(
        req_last_tokens_tensor, ref.expand_as(req_last_tokens_tensor), dim=-1
    )
    max_abs_diff = (req_last_tokens_tensor - ref).abs().amax(dim=-1)
    diverged = bool(torch.any(max_abs_diff > 1e-4))

    payload = {
        "meta": {
            "kind": "layer",
            "name": f"{layer_name}.{tag}",
            "stage": stage,
            "decode_step": decode_step,
            "diverged": diverged,
        },
        "req_ids": req_ids[: len(query_lens)],
        "summary": {
            "query_lens": [int(length) for length in query_lens],
            "full_shape": list(tensor.shape),
            "last_token_shape": list(req_last_tokens_tensor.shape),
            "mean": req_last_tokens_tensor.mean(dim=-1).tolist(),
            "max": req_last_tokens_tensor.max(dim=-1).values.tolist(),
            "min": req_last_tokens_tensor.min(dim=-1).values.tolist(),
            "norm": req_last_tokens_tensor.norm(dim=-1).tolist(),
            "cosine_to_req0": cosine.tolist(),
            "max_abs_diff_to_req0": max_abs_diff.tolist(),
        },
    }
    if envs_ascend.VLLM_ASCEND_IDENTICAL_REQ_DEBUG_MODE == "all" and diverged:
        payload["last_token_tensor"] = req_last_tokens_tensor
    return _persist_artifact("layer", payload)


def dump_tensor_pack(
    *,
    name: str,
    tag: str,
    tensors: dict[str, Any],
) -> Path | None:
    ctx = get_request_divergence_context()
    stage = ctx.get("stage", "unknown")
    decode_step = ctx.get("decode_step")
    req_ids = ctx.get("req_ids", [])
    if not should_dump_tensor(name, decode_step):
        return None

    payload = {
        "meta": {
            "kind": "tensor",
            "name": f"{name}.{tag}",
            "stage": stage,
            "decode_step": decode_step,
        },
        "req_ids": list(req_ids),
        "tensors": {name: _detach_to_cpu(value) for name, value in tensors.items()},
    }
    return _persist_artifact("tensor", payload)


def dump_attention_tensor_pack(
    *,
    layer_name: str,
    tag: str,
    tensors: dict[str, Any],
) -> Path | None:
    return dump_tensor_pack(name=layer_name, tag=tag, tensors=tensors)
