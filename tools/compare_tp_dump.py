#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""
Compare TP dump tensors from two runs.

Supports:
1) JSONL dump records (`sample` field) produced by TP dump hooks.
2) PT tensor dumps (`VLLM_ASCEND_TP_DUMP_SAVE_TENSOR=1`).

Examples:
  python tools/compare_tp_dump.py \
    --base /tmp/tp1 --cand /tmp/tp8 --rank 0 \
    --comm-mode pre_all_reduce,all_reduce --layer-ids 0-4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
    r"^step(?P<step>\d+)_rank(?P<rank>\d+)_(?:(?P<comm_mode>.+)_)?(?P<proj_type>o_proj|down_proj)_(?P<prefix>.+)\.pt$"
)


@dataclass
class DumpEntry:
    step: int
    layer_idx: int | None
    proj_type: str
    comm_mode: str
    prefix: str
    rank: int
    source: str
    payload_type: str  # "jsonl_sample" | "pt_tensor"
    payload: list[float] | Path
    shape: list[int] | None
    dtype: str | None


def parse_layer_ids(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    out: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            bounds = token.split("-", maxsplit=1)
            if len(bounds) != 2:
                continue
            try:
                start = int(bounds[0].strip())
                end = int(bounds[1].strip())
            except ValueError:
                continue
            lo = min(start, end)
            hi = max(start, end)
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
    items = {x.strip() for x in raw.split(",") if x.strip()}
    return items or None


def get_layer_idx(prefix: str) -> int | None:
    for pattern in LAYER_PATTERNS:
        match = re.search(pattern, prefix)
        if match is not None:
            return int(match.group(1))
    return None


def normalize_rank(rank: int) -> int | None:
    return None if rank < 0 else rank


def make_key(entry: DumpEntry) -> tuple[int, int | None, str, str, str, int]:
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


def load_jsonl_entries(
    jsonl_paths: list[Path],
    *,
    rank_filter: int | None,
    layer_ids: set[int] | None,
    proj_types: set[str] | None,
    comm_modes: set[str] | None,
    start_step: int,
    max_steps: int,
) -> tuple[dict[tuple[int, int | None, str, str, str, int], DumpEntry], int]:
    entries: dict[tuple[int, int | None, str, str, str, int], DumpEntry] = {}
    duplicate_count = 0

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
                    source=str(record.get("source", "jsonl")),
                    payload_type="jsonl_sample",
                    payload=[float(x) for x in sample],
                    shape=record.get("shape", None),
                    dtype=str(record.get("dtype", "unknown")),
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

                key = make_key(entry)
                if key in entries:
                    duplicate_count += 1
                entries[key] = entry

    return entries, duplicate_count


def load_pt_entries(
    pt_paths: list[Path],
    *,
    rank_filter: int | None,
    layer_ids: set[int] | None,
    proj_types: set[str] | None,
    comm_modes: set[str] | None,
    start_step: int,
    max_steps: int,
) -> tuple[dict[tuple[int, int | None, str, str, str, int], DumpEntry], int]:
    entries: dict[tuple[int, int | None, str, str, str, int], DumpEntry] = {}
    duplicate_count = 0

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
            source="pt",
            payload_type="pt_tensor",
            payload=path,
            shape=None,
            dtype=None,
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

        key = make_key(entry)
        if key in entries:
            duplicate_count += 1
        entries[key] = entry

    return entries, duplicate_count


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
) -> tuple[dict[tuple[int, int | None, str, str, str, int], DumpEntry], str, int]:
    jsonl_paths = discover_jsonl_paths(path, rank_filter)
    pt_paths = discover_pt_paths(path)

    if mode == "jsonl":
        if not jsonl_paths:
            raise FileNotFoundError(f"No jsonl dump found under: {path}")
        loaded, dup = load_jsonl_entries(
            jsonl_paths,
            rank_filter=rank_filter,
            layer_ids=layer_ids,
            proj_types=proj_types,
            comm_modes=comm_modes,
            start_step=start_step,
            max_steps=max_steps,
        )
        return loaded, "jsonl", dup

    if mode == "pt":
        if not pt_paths:
            raise FileNotFoundError(f"No pt dump found under: {path}")
        loaded, dup = load_pt_entries(
            pt_paths,
            rank_filter=rank_filter,
            layer_ids=layer_ids,
            proj_types=proj_types,
            comm_modes=comm_modes,
            start_step=start_step,
            max_steps=max_steps,
        )
        return loaded, "pt", dup

    # auto mode: prefer jsonl if exists, fallback to pt.
    if jsonl_paths:
        loaded, dup = load_jsonl_entries(
            jsonl_paths,
            rank_filter=rank_filter,
            layer_ids=layer_ids,
            proj_types=proj_types,
            comm_modes=comm_modes,
            start_step=start_step,
            max_steps=max_steps,
        )
        return loaded, "jsonl", dup
    if pt_paths:
        loaded, dup = load_pt_entries(
            pt_paths,
            rank_filter=rank_filter,
            layer_ids=layer_ids,
            proj_types=proj_types,
            comm_modes=comm_modes,
            start_step=start_step,
            max_steps=max_steps,
        )
        return loaded, "pt", dup

    raise FileNotFoundError(f"No dump files found under: {path}")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = 0.0
    a_sq = 0.0
    b_sq = 0.0
    for ai, bi in zip(a, b):
        dot += ai * bi
        a_sq += ai * ai
        b_sq += bi * bi

    a_norm = math.sqrt(a_sq)
    b_norm = math.sqrt(b_sq)
    if a_norm == 0.0 and b_norm == 0.0:
        return 1.0
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return dot / (a_norm * b_norm)


def _import_torch():
    try:
        import torch  # type: ignore
    except ImportError as e:
        raise ImportError("Reading .pt dumps requires torch. Please run in an environment with torch installed.") from e
    return torch


def entry_to_flat_values(entry: DumpEntry, cache: dict[Path, list[float]]) -> list[float]:
    if entry.payload_type == "jsonl_sample":
        payload = entry.payload
        assert isinstance(payload, list)
        return payload

    path = entry.payload
    assert isinstance(path, Path)
    if path not in cache:
        torch = _import_torch()
        tensor = torch.load(path, map_location="cpu")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Invalid pt dump (not tensor): {path}")
        cache[path] = tensor.detach().reshape(-1).to(torch.float64).tolist()
    return cache[path]


def safe_float(v: float) -> str:
    if math.isnan(v) or math.isinf(v):
        return "nan"
    return f"{v:.6e}"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare TP dump tensors from two runs.")
    parser.add_argument("--base", type=Path, required=True, help="Baseline dump file or directory")
    parser.add_argument("--cand", type=Path, required=True, help="Candidate dump file or directory")
    parser.add_argument(
        "--mode",
        choices=("auto", "jsonl", "pt"),
        default="auto",
        help="Input format selection (default: auto)",
    )
    parser.add_argument("--rank", type=int, default=0, help="Rank filter. Use -1 for all ranks. (default: 0)")
    parser.add_argument("--layer-ids", type=str, default="", help="Layer filter, e.g. '0-4,8,10-12'")
    parser.add_argument("--proj", type=str, default="o_proj,down_proj", help="Proj filter csv")
    parser.add_argument("--comm-mode", type=str, default="", help="Comm mode filter csv")
    parser.add_argument("--start-step", type=int, default=0, help="Start decode step (inclusive)")
    parser.add_argument("--max-steps", type=int, default=-1, help="Max decode steps to compare")
    parser.add_argument(
        "--print-details",
        action="store_true",
        help="Print per-prefix detailed rows (otherwise print summary only)",
    )
    parser.add_argument("--summary-csv", type=Path, default=None, help="Optional summary csv output path")
    parser.add_argument("--details-csv", type=Path, default=None, help="Optional details csv output path")
    parser.add_argument(
        "--warn-cos-threshold",
        type=float,
        default=0.999,
        help="Warn threshold for cosine similarity in summary rows",
    )
    parser.add_argument(
        "--warn-max-abs-threshold",
        type=float,
        default=1e-2,
        help="Warn threshold for max_abs in summary rows",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rank_filter = normalize_rank(args.rank)
    layer_ids = parse_layer_ids(args.layer_ids)
    proj_types = parse_csv_set(args.proj)
    comm_modes = parse_csv_set(args.comm_mode)

    base_map, base_mode, base_dup = load_entries(
        args.base,
        mode=args.mode,
        rank_filter=rank_filter,
        layer_ids=layer_ids,
        proj_types=proj_types,
        comm_modes=comm_modes,
        start_step=args.start_step,
        max_steps=args.max_steps,
    )
    cand_map, cand_mode, cand_dup = load_entries(
        args.cand,
        mode=args.mode,
        rank_filter=rank_filter,
        layer_ids=layer_ids,
        proj_types=proj_types,
        comm_modes=comm_modes,
        start_step=args.start_step,
        max_steps=args.max_steps,
    )

    if base_mode != cand_mode:
        raise ValueError(f"Input mode mismatch: base={base_mode}, cand={cand_mode}. Use same dump format.")

    base_keys = set(base_map.keys())
    cand_keys = set(cand_map.keys())
    common_keys = sorted(base_keys & cand_keys)
    only_base = len(base_keys - cand_keys)
    only_cand = len(cand_keys - base_keys)

    print(f"[info] base_mode={base_mode}, cand_mode={cand_mode}")
    print(f"[info] base_records={len(base_map)}, cand_records={len(cand_map)}, common={len(common_keys)}")
    print(f"[info] only_base={only_base}, only_cand={only_cand}")
    if base_dup > 0 or cand_dup > 0:
        print(f"[warn] duplicate_keys_overwritten: base={base_dup}, cand={cand_dup}")

    if not common_keys:
        print("[error] no common keys to compare")
        return

    tensor_cache: dict[Path, list[float]] = {}
    detail_rows: list[dict[str, Any]] = []

    for key in common_keys:
        base_entry = base_map[key]
        cand_entry = cand_map[key]
        a = entry_to_flat_values(base_entry, tensor_cache)
        b = entry_to_flat_values(cand_entry, tensor_cache)

        numel_base = len(a)
        numel_cand = len(b)
        numel_cmp = min(numel_base, numel_cand)
        if numel_cmp == 0:
            continue
        a_cmp = a[:numel_cmp]
        b_cmp = b[:numel_cmp]

        max_abs = 0.0
        abs_sum = 0.0
        sq_sum = 0.0
        for ai, bi in zip(a_cmp, b_cmp):
            d = ai - bi
            ad = abs(d)
            if ad > max_abs:
                max_abs = ad
            abs_sum += ad
            sq_sum += d * d
        mean_abs = abs_sum / numel_cmp
        rmse = math.sqrt(sq_sum / numel_cmp)
        l2 = math.sqrt(sq_sum)

        detail_rows.append(
            {
                "step": base_entry.step,
                "layer_idx": base_entry.layer_idx,
                "proj_type": base_entry.proj_type,
                "comm_mode": base_entry.comm_mode,
                "rank": base_entry.rank,
                "prefix": base_entry.prefix,
                "source_base": base_entry.source,
                "source_cand": cand_entry.source,
                "numel_cmp": numel_cmp,
                "numel_base": numel_base,
                "numel_cand": numel_cand,
                "cosine": cosine_similarity(a_cmp, b_cmp),
                "max_abs": max_abs,
                "mean_abs": mean_abs,
                "rmse": rmse,
                "l2": l2,
            }
        )

    if not detail_rows:
        print("[error] no comparable rows after filtering")
        return

    summary_groups: dict[tuple[int, int | None, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        group_key = (row["step"], row["layer_idx"], row["proj_type"], row["comm_mode"])
        summary_groups[group_key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (step, layer_idx, proj_type, comm_mode), rows in sorted(summary_groups.items()):
        cosines = [float(r["cosine"]) for r in rows]
        max_abs_list = [float(r["max_abs"]) for r in rows]
        mean_abs_list = [float(r["mean_abs"]) for r in rows]
        rmse_list = [float(r["rmse"]) for r in rows]
        l2_list = [float(r["l2"]) for r in rows]
        summary_rows.append(
            {
                "step": step,
                "layer_idx": layer_idx,
                "proj_type": proj_type,
                "comm_mode": comm_mode,
                "pairs": len(rows),
                "cosine_mean": sum(cosines) / len(cosines),
                "cosine_min": min(cosines),
                "max_abs_max": max(max_abs_list),
                "mean_abs_mean": sum(mean_abs_list) / len(mean_abs_list),
                "rmse_mean": sum(rmse_list) / len(rmse_list),
                "l2_mean": sum(l2_list) / len(l2_list),
            }
        )

    print("\n[summary] step/layer/proj/comm")
    print(
        "step layer proj_type comm_mode pairs cosine_mean cosine_min max_abs_max mean_abs_mean rmse_mean l2_mean"
    )
    for row in summary_rows:
        print(
            f"{row['step']:>4} "
            f"{str(row['layer_idx']):>5} "
            f"{row['proj_type']:<9} "
            f"{row['comm_mode']:<14} "
            f"{row['pairs']:>5} "
            f"{safe_float(row['cosine_mean'])} "
            f"{safe_float(row['cosine_min'])} "
            f"{safe_float(row['max_abs_max'])} "
            f"{safe_float(row['mean_abs_mean'])} "
            f"{safe_float(row['rmse_mean'])} "
            f"{safe_float(row['l2_mean'])}"
        )

    warn_rows = [
        r
        for r in summary_rows
        if (r["cosine_min"] < args.warn_cos_threshold or r["max_abs_max"] > args.warn_max_abs_threshold)
    ]
    if warn_rows:
        print(
            f"\n[warn] {len(warn_rows)} summary rows exceeded thresholds: "
            f"cosine_min<{args.warn_cos_threshold} or max_abs_max>{args.warn_max_abs_threshold}"
        )
        for row in warn_rows:
            print(
                f"  step={row['step']} layer={row['layer_idx']} proj={row['proj_type']} comm={row['comm_mode']} "
                f"cosine_min={row['cosine_min']:.6e} max_abs_max={row['max_abs_max']:.6e}"
            )

    if args.print_details:
        print("\n[details] matched keys")
        print("step layer proj_type comm_mode rank cosine max_abs mean_abs rmse l2 numel_cmp prefix")
        for row in sorted(
            detail_rows,
            key=lambda x: (
                int(x["step"]),
                -1 if x["layer_idx"] is None else int(x["layer_idx"]),
                str(x["proj_type"]),
                str(x["comm_mode"]),
                int(x["rank"]),
                str(x["prefix"]),
            ),
        ):
            print(
                f"{row['step']:>4} "
                f"{str(row['layer_idx']):>5} "
                f"{row['proj_type']:<9} "
                f"{row['comm_mode']:<14} "
                f"{row['rank']:>4} "
                f"{safe_float(row['cosine'])} "
                f"{safe_float(row['max_abs'])} "
                f"{safe_float(row['mean_abs'])} "
                f"{safe_float(row['rmse'])} "
                f"{safe_float(row['l2'])} "
                f"{row['numel_cmp']:>8} "
                f"{row['prefix']}"
            )

    if args.summary_csv is not None:
        write_csv(
            args.summary_csv,
            summary_rows,
            [
                "step",
                "layer_idx",
                "proj_type",
                "comm_mode",
                "pairs",
                "cosine_mean",
                "cosine_min",
                "max_abs_max",
                "mean_abs_mean",
                "rmse_mean",
                "l2_mean",
            ],
        )
        print(f"\n[info] summary csv written: {args.summary_csv}")

    if args.details_csv is not None:
        write_csv(
            args.details_csv,
            detail_rows,
            [
                "step",
                "layer_idx",
                "proj_type",
                "comm_mode",
                "rank",
                "prefix",
                "source_base",
                "source_cand",
                "numel_cmp",
                "numel_base",
                "numel_cand",
                "cosine",
                "max_abs",
                "mean_abs",
                "rmse",
                "l2",
            ],
        )
        print(f"[info] details csv written: {args.details_csv}")


if __name__ == "__main__":
    main()
