#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""
Simple TP dump comparator.

Only reports two metrics:
1) cosine similarity (torch.cosine_similarity, float64)
2) max-diff (max absolute difference, float64)

Supported dump formats:
- JSONL (`tp_row_dump_rank*.jsonl`)
- PT tensors (`*.pt`)
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LAYER_PATTERNS = (
    r"(?:^|\.)layers\.(\d+)\.",
    r"(?:^|\.)h\.(\d+)\.",
    r"(?:^|\.)blocks\.(\d+)\.",
)

PT_DUMP_PATTERN = re.compile(
    r"^step(?P<step>\d+)_rank(?P<rank>\d+)_(?:(?P<comm_mode>.+)_)?"
    r"(?P<proj_type>o_proj|down_proj|attn_norm_output|rmsnorm_input|rmsnorm_output|qkv_input|qkv_output|gate_up_input|gate_up_output|down_input)"
    r"_(?P<prefix>.+)\.pt$"
)


@dataclass
class DumpEntry:
    step: int
    layer_idx: int | None
    proj_type: str
    comm_mode: str
    prefix: str
    rank: int
    payload_type: str  # jsonl_sample | pt_tensor | tensor
    payload: Any


_TORCH = None


def get_torch():
    global _TORCH
    if _TORCH is not None:
        return _TORCH
    try:
        import torch
    except ImportError as e:
        raise ImportError("compare_tp_dump.py requires torch to compute cosine similarity.") from e
    _TORCH = torch
    return _TORCH


def parse_layer_ids(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    out: set[int] = set()
    for token in raw.strip().split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", maxsplit=1)
            try:
                lo = int(left.strip())
                hi = int(right.strip())
            except ValueError:
                continue
            if lo > hi:
                lo, hi = hi, lo
            out.update(range(lo, hi + 1))
            continue
        try:
            out.add(int(token))
        except ValueError:
            continue
    return out or None


def parse_csv_set(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    out = {x.strip() for x in raw.split(",") if x.strip()}
    return out or None


def get_layer_idx(prefix: str) -> int | None:
    for pattern in LAYER_PATTERNS:
        match = re.search(pattern, prefix)
        if match is not None:
            return int(match.group(1))
    return None


def normalize_rank(rank: int) -> int | None:
    return None if rank < 0 else rank


def make_key(
    entry: DumpEntry,
    *,
    ignore_comm_mode: bool,
) -> tuple[int, int | None, str, str, int] | tuple[int, int | None, str, str, str, int]:
    if ignore_comm_mode:
        return (entry.step, entry.layer_idx, entry.proj_type, entry.prefix, entry.rank)
    return (entry.step, entry.layer_idx, entry.proj_type, entry.comm_mode, entry.prefix, entry.rank)


def should_keep_entry(
    entry: DumpEntry,
    *,
    layer_ids: set[int] | None,
    proj_types: set[str] | None,
    comm_modes: set[str] | None,
    start_step: int,
    max_steps: int,
) -> bool:
    if entry.step < start_step:
        return False
    if max_steps >= 0 and entry.step >= (start_step + max_steps):
        return False
    if layer_ids is not None and entry.layer_idx not in layer_ids:
        return False
    if proj_types is not None and entry.proj_type not in proj_types:
        return False
    if comm_modes is not None and entry.comm_mode not in comm_modes:
        return False
    return True


def discover_jsonl_paths(path: Path, rank_filter: int | None) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix == ".jsonl" else []
    if not path.is_dir():
        return []
    if rank_filter is None:
        return sorted(path.glob("tp_row_dump_rank*.jsonl"))
    ranked = path / f"tp_row_dump_rank{rank_filter}.jsonl"
    return [ranked] if ranked.exists() else []


def discover_pt_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix == ".pt" else []
    if not path.is_dir():
        return []
    return sorted(path.glob("*.pt"))


def load_jsonl_entries(
    jsonl_paths: list[Path],
    *,
    rank_filter: int | None,
    layer_ids: set[int] | None,
    proj_types: set[str] | None,
    comm_modes: set[str] | None,
    start_step: int,
    max_steps: int,
    ignore_comm_mode: bool,
) -> dict[tuple[Any, ...], DumpEntry]:
    entries: dict[tuple[Any, ...], DumpEntry] = {}

    for path in jsonl_paths:
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                rank = int(record.get("rank", 0))
                if rank_filter is not None and rank != rank_filter:
                    continue

                prefix = str(record.get("prefix", ""))
                layer_idx = record.get("layer_idx", None)
                if layer_idx is None:
                    layer_idx = get_layer_idx(prefix)
                else:
                    layer_idx = int(layer_idx)

                sample = record.get("sample", [])
                if not isinstance(sample, list):
                    continue

                entry = DumpEntry(
                    step=int(record["decode_step"]),
                    layer_idx=layer_idx,
                    proj_type=str(record.get("proj_type", "")),
                    comm_mode=str(record.get("comm_mode", "unknown")),
                    prefix=prefix,
                    rank=rank,
                    payload_type="jsonl_sample",
                    payload=[float(x) for x in sample],
                )
                if not should_keep_entry(
                    entry,
                    layer_ids=layer_ids,
                    proj_types=proj_types,
                    comm_modes=comm_modes,
                    start_step=start_step,
                    max_steps=max_steps,
                ):
                    continue

                key = make_key(entry, ignore_comm_mode=ignore_comm_mode)
                entries[key] = entry

    return entries


def load_pt_entries(
    pt_paths: list[Path],
    *,
    rank_filter: int | None,
    layer_ids: set[int] | None,
    proj_types: set[str] | None,
    comm_modes: set[str] | None,
    start_step: int,
    max_steps: int,
    ignore_comm_mode: bool,
) -> dict[tuple[Any, ...], DumpEntry]:
    entries: dict[tuple[Any, ...], DumpEntry] = {}

    for path in pt_paths:
        match = PT_DUMP_PATTERN.match(path.name)
        if match is None:
            continue

        rank = int(match.group("rank"))
        if rank_filter is not None and rank != rank_filter:
            continue

        prefix = match.group("prefix")
        layer_idx = get_layer_idx(prefix)
        comm_mode = match.group("comm_mode") or "unknown"

        entry = DumpEntry(
            step=int(match.group("step")),
            layer_idx=layer_idx,
            proj_type=match.group("proj_type"),
            comm_mode=comm_mode,
            prefix=prefix,
            rank=rank,
            payload_type="pt_tensor",
            payload=path,
        )
        if not should_keep_entry(
            entry,
            layer_ids=layer_ids,
            proj_types=proj_types,
            comm_modes=comm_modes,
            start_step=start_step,
            max_steps=max_steps,
        ):
            continue

        key = make_key(entry, ignore_comm_mode=ignore_comm_mode)
        entries[key] = entry

    return entries


def load_entries(
    path: Path,
    *,
    mode: str,
    rank_filter: int | None,
    layer_ids: set[int] | None,
    proj_types: set[str] | None,
    comm_modes: set[str] | None,
    start_step: int,
    max_steps: int,
    ignore_comm_mode: bool,
) -> tuple[dict[tuple[Any, ...], DumpEntry], str]:
    jsonl_paths = discover_jsonl_paths(path, rank_filter)
    pt_paths = discover_pt_paths(path)

    if mode == "jsonl":
        if not jsonl_paths:
            raise FileNotFoundError(f"No jsonl dump found under: {path}")
        return (
            load_jsonl_entries(
                jsonl_paths,
                rank_filter=rank_filter,
                layer_ids=layer_ids,
                proj_types=proj_types,
                comm_modes=comm_modes,
                start_step=start_step,
                max_steps=max_steps,
                ignore_comm_mode=ignore_comm_mode,
            ),
            "jsonl",
        )

    if mode == "pt":
        if not pt_paths:
            raise FileNotFoundError(f"No pt dump found under: {path}")
        return (
            load_pt_entries(
                pt_paths,
                rank_filter=rank_filter,
                layer_ids=layer_ids,
                proj_types=proj_types,
                comm_modes=comm_modes,
                start_step=start_step,
                max_steps=max_steps,
                ignore_comm_mode=ignore_comm_mode,
            ),
            "pt",
        )

    if jsonl_paths:
        return (
            load_jsonl_entries(
                jsonl_paths,
                rank_filter=rank_filter,
                layer_ids=layer_ids,
                proj_types=proj_types,
                comm_modes=comm_modes,
                start_step=start_step,
                max_steps=max_steps,
                ignore_comm_mode=ignore_comm_mode,
            ),
            "jsonl",
        )

    if pt_paths:
        return (
            load_pt_entries(
                pt_paths,
                rank_filter=rank_filter,
                layer_ids=layer_ids,
                proj_types=proj_types,
                comm_modes=comm_modes,
                start_step=start_step,
                max_steps=max_steps,
                ignore_comm_mode=ignore_comm_mode,
            ),
            "pt",
        )

    raise FileNotFoundError(f"No dump files found under: {path}")


def entry_to_tensor(entry: DumpEntry, cache: dict[Path, Any]) -> Any:
    torch = get_torch()
    if entry.payload_type == "tensor":
        assert isinstance(entry.payload, torch.Tensor)
        return entry.payload.reshape(-1).to(torch.float64)

    if entry.payload_type == "jsonl_sample":
        assert isinstance(entry.payload, list)
        return torch.tensor(entry.payload, dtype=torch.float64)

    assert entry.payload_type == "pt_tensor"
    assert isinstance(entry.payload, Path)
    if entry.payload not in cache:
        tensor = torch.load(entry.payload, map_location="cpu")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Invalid pt dump (not tensor): {entry.payload}")
        cache[entry.payload] = tensor.detach().reshape(-1).to(torch.float64)
    return cache[entry.payload]


def aggregate_entries_across_ranks(
    entries: dict[tuple[Any, ...], DumpEntry],
    *,
    method: str,
    ignore_comm_mode: bool,
) -> dict[tuple[Any, ...], DumpEntry]:
    torch = get_torch()
    if method == "none":
        return entries

    grouped: dict[tuple[int, int | None, str, str, str], list[DumpEntry]] = defaultdict(list)
    for entry in entries.values():
        group_key = (entry.step, entry.layer_idx, entry.proj_type, entry.comm_mode, entry.prefix)
        grouped[group_key].append(entry)

    out: dict[tuple[Any, ...], DumpEntry] = {}
    cache: dict[Path, Any] = {}

    for group in grouped.values():
        group = sorted(group, key=lambda e: e.rank)
        tensors = [entry_to_tensor(e, cache) for e in group]
        if not tensors:
            continue
        min_len = min(int(t.numel()) for t in tensors)
        if min_len <= 0:
            continue
        stacked = torch.stack([t[:min_len] for t in tensors], dim=0)
        if method == "sum":
            agg = stacked.sum(dim=0)
        elif method == "mean":
            agg = stacked.mean(dim=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        template = group[0]
        entry = DumpEntry(
            step=template.step,
            layer_idx=template.layer_idx,
            proj_type=template.proj_type,
            comm_mode=template.comm_mode,
            prefix=template.prefix,
            rank=0,
            payload_type="tensor",
            payload=agg,
        )
        out[make_key(entry, ignore_comm_mode=ignore_comm_mode)] = entry

    return out


def cosine_similarity_float64(a, b) -> float:
    torch = get_torch()
    # Required by user: torch.cosine_similarity() + high precision.
    # Handle zero vectors explicitly to avoid NaN from 0/0.
    a_zero = bool(torch.count_nonzero(a).item() == 0)
    b_zero = bool(torch.count_nonzero(b).item() == 0)
    if a_zero and b_zero:
        return 1.0
    if a_zero or b_zero:
        return 0.0
    return float(torch.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).item())


def safe_float(v: float) -> str:
    return f"{v:.6e}"


def summarize_axes(entries: dict[tuple[Any, ...], DumpEntry]) -> dict[str, Any]:
    if not entries:
        return {
            "steps": (None, None, 0),
            "layers": [],
            "proj_types": [],
            "comm_modes": [],
            "ranks": [],
        }
    values = list(entries.values())
    steps = sorted({e.step for e in values})
    layers = sorted({e.layer_idx for e in values})
    proj_types = sorted({e.proj_type for e in values})
    comm_modes = sorted({e.comm_mode for e in values})
    ranks = sorted({e.rank for e in values})
    return {
        "steps": (steps[0], steps[-1], len(steps)),
        "layers": layers,
        "proj_types": proj_types,
        "comm_modes": comm_modes,
        "ranks": ranks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple TP dump comparator.")
    parser.add_argument("--base", type=Path, required=True, help="Baseline dump path")
    parser.add_argument("--cand", type=Path, required=True, help="Candidate dump path")
    parser.add_argument("--mode", choices=("auto", "jsonl", "pt"), default="auto", help="Input format")

    parser.add_argument("--rank", type=int, default=0, help="Rank filter. -1 means all ranks.")
    parser.add_argument("--base-rank", type=int, default=None, help="Base-only rank filter. -1 means all ranks.")
    parser.add_argument("--cand-rank", type=int, default=None, help="Cand-only rank filter. -1 means all ranks.")
    parser.add_argument(
        "--aggregate-base-ranks",
        choices=("none", "sum", "mean"),
        default="none",
        help="Aggregate base entries across ranks before compare",
    )
    parser.add_argument(
        "--aggregate-cand-ranks",
        choices=("none", "sum", "mean"),
        default="none",
        help="Aggregate cand entries across ranks before compare",
    )

    parser.add_argument("--layer-ids", type=str, default="", help="Layer filter, e.g. 0-4,8")
    parser.add_argument(
        "--proj",
        type=str,
        default="o_proj,rmsnorm_input,rmsnorm_output,qkv_input,qkv_output,gate_up_input,gate_up_output,down_proj",
        help="Proj filter csv",
    )
    parser.add_argument("--comm-mode", type=str, default="", help="Comm mode filter for both sides")
    parser.add_argument("--base-comm-mode", type=str, default="", help="Base comm mode filter")
    parser.add_argument("--cand-comm-mode", type=str, default="", help="Cand comm mode filter")
    parser.add_argument(
        "--ignore-comm-mode-in-key",
        action="store_true",
        help="Do not require comm_mode equality when matching keys",
    )
    parser.add_argument("--start-step", type=int, default=0, help="Start decode step (inclusive)")
    parser.add_argument("--max-steps", type=int, default=-1, help="Max decode steps")
    parser.add_argument("--print-details", action="store_true", help="Print per-prefix rows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    shared_rank_filter = normalize_rank(args.rank)
    base_rank_filter = normalize_rank(args.base_rank) if args.base_rank is not None else shared_rank_filter
    cand_rank_filter = normalize_rank(args.cand_rank) if args.cand_rank is not None else shared_rank_filter

    layer_ids = parse_layer_ids(args.layer_ids)
    proj_types = parse_csv_set(args.proj)
    shared_comm_modes = parse_csv_set(args.comm_mode)
    base_comm_modes = parse_csv_set(args.base_comm_mode) or shared_comm_modes
    cand_comm_modes = parse_csv_set(args.cand_comm_mode) or shared_comm_modes

    base_map, base_mode = load_entries(
        args.base,
        mode=args.mode,
        rank_filter=base_rank_filter,
        layer_ids=layer_ids,
        proj_types=proj_types,
        comm_modes=base_comm_modes,
        start_step=args.start_step,
        max_steps=args.max_steps,
        ignore_comm_mode=args.ignore_comm_mode_in_key,
    )
    cand_map, cand_mode = load_entries(
        args.cand,
        mode=args.mode,
        rank_filter=cand_rank_filter,
        layer_ids=layer_ids,
        proj_types=proj_types,
        comm_modes=cand_comm_modes,
        start_step=args.start_step,
        max_steps=args.max_steps,
        ignore_comm_mode=args.ignore_comm_mode_in_key,
    )

    if args.aggregate_base_ranks != "none":
        base_map = aggregate_entries_across_ranks(
            base_map,
            method=args.aggregate_base_ranks,
            ignore_comm_mode=args.ignore_comm_mode_in_key,
        )
    if args.aggregate_cand_ranks != "none":
        cand_map = aggregate_entries_across_ranks(
            cand_map,
            method=args.aggregate_cand_ranks,
            ignore_comm_mode=args.ignore_comm_mode_in_key,
        )

    if base_mode != cand_mode:
        raise ValueError(f"Input mode mismatch: base={base_mode}, cand={cand_mode}")

    base_keys = set(base_map.keys())
    cand_keys = set(cand_map.keys())
    common_keys = sorted(base_keys & cand_keys)

    print(f"[info] base_mode={base_mode}, cand_mode={cand_mode}")
    print(f"[info] base_records={len(base_map)}, cand_records={len(cand_map)}, common={len(common_keys)}")
    print(f"[info] only_base={len(base_keys - cand_keys)}, only_cand={len(cand_keys - base_keys)}")

    if not common_keys:
        print("[error] no common keys to compare")
        base_axes = summarize_axes(base_map)
        cand_axes = summarize_axes(cand_map)
        print(
            "[debug] base: "
            f"steps={base_axes['steps']} comm_modes={base_axes['comm_modes']} "
            f"proj_types={base_axes['proj_types']} ranks={base_axes['ranks']}"
        )
        print(
            "[debug] cand: "
            f"steps={cand_axes['steps']} comm_modes={cand_axes['comm_modes']} "
            f"proj_types={cand_axes['proj_types']} ranks={cand_axes['ranks']}"
        )
        return

    tensor_cache: dict[Path, torch.Tensor] = {}
    detail_rows: list[dict[str, Any]] = []

    for key in common_keys:
        base_entry = base_map[key]
        cand_entry = cand_map[key]
        a = entry_to_tensor(base_entry, tensor_cache)
        b = entry_to_tensor(cand_entry, tensor_cache)

        numel_cmp = min(int(a.numel()), int(b.numel()))
        if numel_cmp <= 0:
            continue
        a = a[:numel_cmp]
        b = b[:numel_cmp]
        diff = (a - b).abs()

        cosine = cosine_similarity_float64(a, b)
        max_diff = float(diff.max().item())
        comm_pair = (
            base_entry.comm_mode
            if base_entry.comm_mode == cand_entry.comm_mode
            else f"{base_entry.comm_mode}->{cand_entry.comm_mode}"
        )
        detail_rows.append(
            {
                "step": base_entry.step,
                "layer_idx": base_entry.layer_idx,
                "proj_type": base_entry.proj_type,
                "comm_mode_pair": comm_pair,
                "rank": base_entry.rank,
                "prefix": base_entry.prefix,
                "numel_cmp": numel_cmp,
                "cosine": cosine,
                "max_diff": max_diff,
            }
        )

    if not detail_rows:
        print("[error] no comparable rows after filtering")
        return

    summary_groups: dict[tuple[int, int | None, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        group_key = (row["step"], row["layer_idx"], row["proj_type"], row["comm_mode_pair"])
        summary_groups[group_key].append(row)

    print("\n[summary]")
    print("step layer proj_type comm_mode pairs cosine_mean cosine_min max_diff_max")
    for (step, layer_idx, proj_type, comm_pair), rows in sorted(summary_groups.items()):
        cosines = [float(r["cosine"]) for r in rows]
        max_diffs = [float(r["max_diff"]) for r in rows]
        print(
            f"{step:>4} "
            f"{str(layer_idx):>5} "
            f"{proj_type:<13} "
            f"{comm_pair:<24} "
            f"{len(rows):>5} "
            f"{safe_float(sum(cosines) / len(cosines))} "
            f"{safe_float(min(cosines))} "
            f"{safe_float(max(max_diffs))}"
        )

    if args.print_details:
        print("\n[details]")
        print("step layer proj_type comm_mode rank cosine max_diff numel_cmp prefix")
        for row in sorted(
            detail_rows,
            key=lambda x: (
                int(x["step"]),
                -1 if x["layer_idx"] is None else int(x["layer_idx"]),
                str(x["proj_type"]),
                str(x["comm_mode_pair"]),
                int(x["rank"]),
                str(x["prefix"]),
            ),
        ):
            print(
                f"{row['step']:>4} "
                f"{str(row['layer_idx']):>5} "
                f"{row['proj_type']:<13} "
                f"{row['comm_mode_pair']:<24} "
                f"{row['rank']:>4} "
                f"{safe_float(float(row['cosine']))} "
                f"{safe_float(float(row['max_diff']))} "
                f"{row['numel_cmp']:>8} "
                f"{row['prefix']}"
            )


if __name__ == "__main__":
    main()
