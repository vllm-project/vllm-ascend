from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ENV_POC_DUMP_DIR = "VLLM_ASCEND_SPEC_SAMPLING_POC_DIR"
ENV_POC_DUMP_MAX_CASES = "VLLM_ASCEND_SPEC_SAMPLING_POC_MAX_CASES"

_dump_lock = threading.Lock()
_dump_case_count = 0


def _get_dump_dir() -> Path | None:
    dump_dir = os.getenv(ENV_POC_DUMP_DIR)
    if not dump_dir:
        return None
    return Path(dump_dir)


def _get_max_cases() -> int:
    raw = os.getenv(ENV_POC_DUMP_MAX_CASES, "1")
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def _allocate_case_dir() -> Path | None:
    global _dump_case_count
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return None

    with _dump_lock:
        if _dump_case_count >= _get_max_cases():
            return None
        case_id = _dump_case_count
        _dump_case_count += 1

    case_dir = dump_dir / f"case_{case_id:03d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def _move_tensors(obj: Any, device: str | torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().to(device)
    if isinstance(obj, np.ndarray):
        tensor = torch.from_numpy(obj.copy())
        return tensor.to(device) if device != "cpu" else tensor
    if is_dataclass(obj) and not isinstance(obj, type):
        kwargs = {field.name: _move_tensors(getattr(obj, field.name), device) for field in fields(obj)}
        return type(obj)(**kwargs)
    if hasattr(obj, "_fields") and hasattr(obj, "_asdict"):
        values = {name: _move_tensors(getattr(obj, name), device) for name in obj._fields}
        return type(obj)(**values)
    if isinstance(obj, dict):
        return {key: _move_tensors(value, device) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_move_tensors(item, device) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_move_tensors(item, device) for item in obj)
    return obj


def _tensor_shape(obj: Any) -> list[int] | None:
    if torch.is_tensor(obj):
        return list(obj.shape)
    if isinstance(obj, np.ndarray):
        return list(obj.shape)
    return None


def dump_spec_sampling_case(
    *,
    logits: torch.Tensor,
    sampling_metadata: Any,
    spec_decode_metadata: Any,
    sampler_output: Any,
    draft_probs: torch.Tensor | None,
    prepared_top_k: int | None,
    positions: torch.Tensor | None,
    slot_mapping: torch.Tensor | None,
    input_token_ids: np.ndarray | torch.Tensor | None,
) -> Path | None:
    case_dir = _allocate_case_dir()
    if case_dir is None:
        return None

    payload = {
        "timestamp": time.time(),
        "logits": _move_tensors(logits, "cpu"),
        "draft_probs": _move_tensors(draft_probs, "cpu"),
        "sampling_metadata": _move_tensors(sampling_metadata, "cpu"),
        "spec_decode_metadata": _move_tensors(spec_decode_metadata, "cpu"),
        "sampler_output": _move_tensors(sampler_output, "cpu"),
        "prepared_top_k": prepared_top_k,
        "positions": _move_tensors(positions, "cpu"),
        "slot_mapping": _move_tensors(slot_mapping, "cpu"),
        "input_token_ids": _move_tensors(input_token_ids, "cpu"),
    }

    torch.save(payload, case_dir / "case.pt")

    summary = {
        "timestamp": payload["timestamp"],
        "logits_shape": _tensor_shape(payload["logits"]),
        "draft_probs_shape": _tensor_shape(payload["draft_probs"]),
        "positions_shape": _tensor_shape(payload["positions"]),
        "slot_mapping_shape": _tensor_shape(payload["slot_mapping"]),
        "input_token_ids_shape": _tensor_shape(payload["input_token_ids"]),
        "sampled_token_ids_shape": _tensor_shape(payload["sampler_output"].sampled_token_ids),
        "prepared_top_k": prepared_top_k,
    }
    with open(case_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return case_dir


def load_spec_sampling_case(case_path: str | os.PathLike[str]) -> dict[str, Any]:
    case_path = Path(case_path)
    if case_path.is_dir():
        case_path = case_path / "case.pt"
    return torch.load(case_path, map_location="cpu")

