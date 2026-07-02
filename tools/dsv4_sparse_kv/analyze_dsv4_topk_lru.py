#!/usr/bin/env python3
import argparse
import json
import statistics
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze DSV4 top-k trace locality and simulate an LRU cache.")
    parser.add_argument("--trace", required=True, help="Path to topk_trace.jsonl")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--phase", default="decode", choices=["prefill", "decode", "all"])
    parser.add_argument("--capacities", default="512,1024,2048,4096,8192")
    parser.add_argument("--compress-ratio", type=int, default=4)
    parser.add_argument("--keep-negative", action="store_true", help="Keep negative sparse indices in the analysis.")
    return parser.parse_args()


def parse_capacities(value: str) -> list[int]:
    capacities = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not capacities:
        raise ValueError("at least one capacity is required")
    return capacities


def load_records(trace_path: Path, phase: str, drop_negative: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as trace_file:
        for line_no, line in enumerate(trace_file, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if phase != "all" and record.get("phase") != phase:
                continue
            topk = [int(item) for item in record.get("topk", [])]
            if drop_negative:
                topk = [item for item in topk if item >= 0]
            if not topk:
                continue
            record["topk"] = topk
            record["_line_no"] = line_no
            records.append(record)
    return records


def group_records(records: list[dict[str, Any]]) -> dict[tuple[int, str, int], list[dict[str, Any]]]:
    groups: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            int(record.get("pid", 0)),
            str(record.get("layer_name", "")),
            int(record.get("row_idx", 0)),
        )
        groups[key].append(record)
    for group in groups.values():
        group.sort(key=lambda item: int(item.get("trace_index", item.get("_line_no", 0))))
    return groups


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[index]


def consecutive_overlap(groups: dict[tuple[int, str, int], list[dict[str, Any]]]) -> dict[str, Any]:
    overlaps: list[float] = []
    jaccards: list[float] = []
    layer_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"overlap": [], "jaccard": []})

    for (_pid, layer_name, _row_idx), group in groups.items():
        for previous, current in zip(group, group[1:]):
            previous_set = set(previous["topk"])
            current_set = set(current["topk"])
            if not current_set:
                continue
            intersection = len(previous_set & current_set)
            union = len(previous_set | current_set)
            overlap = intersection / len(current_set)
            jaccard = intersection / union if union else 0.0
            overlaps.append(overlap)
            jaccards.append(jaccard)
            layer_values[layer_name]["overlap"].append(overlap)
            layer_values[layer_name]["jaccard"].append(jaccard)

    by_layer = {
        layer_name: {
            "pairs": len(values["overlap"]),
            "overlap_mean": mean_or_none(values["overlap"]),
            "overlap_p50": percentile(values["overlap"], 0.50),
            "jaccard_mean": mean_or_none(values["jaccard"]),
            "jaccard_p50": percentile(values["jaccard"], 0.50),
        }
        for layer_name, values in sorted(layer_values.items())
    }
    return {
        "pairs": len(overlaps),
        "overlap_mean": mean_or_none(overlaps),
        "overlap_p50": percentile(overlaps, 0.50),
        "overlap_p90": percentile(overlaps, 0.90),
        "jaccard_mean": mean_or_none(jaccards),
        "jaccard_p50": percentile(jaccards, 0.50),
        "by_layer": by_layer,
    }


def simulate_lru_for_group(group: list[dict[str, Any]], capacity: int) -> dict[str, int]:
    cache: OrderedDict[int, None] = OrderedDict()
    hits = 0
    misses = 0
    requests = 0
    unique_tokens: set[int] = set()

    for record in group:
        for token_idx in record["topk"]:
            requests += 1
            unique_tokens.add(token_idx)
            if token_idx in cache:
                hits += 1
                cache.move_to_end(token_idx)
            else:
                misses += 1
                cache[token_idx] = None
                if len(cache) > capacity:
                    cache.popitem(last=False)

    return {
        "requests": requests,
        "hits": hits,
        "misses": misses,
        "unique_tokens": len(unique_tokens),
        "final_cache_size": len(cache),
    }


def simulate_lru(
    groups: dict[tuple[int, str, int], list[dict[str, Any]]],
    capacities: list[int],
    compress_ratio: int,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for capacity in capacities:
        aggregate = {"requests": 0, "hits": 0, "misses": 0, "unique_tokens": 0}
        group_hit_rates: list[float] = []
        for group in groups.values():
            stats = simulate_lru_for_group(group, capacity)
            aggregate["requests"] += stats["requests"]
            aggregate["hits"] += stats["hits"]
            aggregate["misses"] += stats["misses"]
            aggregate["unique_tokens"] += stats["unique_tokens"]
            if stats["requests"]:
                group_hit_rates.append(stats["hits"] / stats["requests"])

        requests = aggregate["requests"]
        summaries.append(
            {
                "capacity_sparse_indices": capacity,
                "approx_capacity_original_tokens": capacity * compress_ratio,
                "requests": requests,
                "hits": aggregate["hits"],
                "misses": aggregate["misses"],
                "hit_rate": aggregate["hits"] / requests if requests else None,
                "miss_rate": aggregate["misses"] / requests if requests else None,
                "group_hit_rate_p50": percentile(group_hit_rates, 0.50),
                "group_hit_rate_p90": percentile(group_hit_rates, 0.90),
                "summed_group_unique_tokens": aggregate["unique_tokens"],
            }
        )
    return summaries


def write_markdown(out_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# DeepSeek V4 Top-k LRU Analysis",
        "",
        f"- trace: `{summary['trace']}`",
        f"- phase: `{summary['phase']}`",
        f"- records: {summary['records']}",
        f"- groups: {summary['groups']}",
        "",
        "## Consecutive Decode Locality",
        "",
        f"- pairs: {summary['consecutive_overlap']['pairs']}",
        f"- overlap_mean: {summary['consecutive_overlap']['overlap_mean']}",
        f"- overlap_p50: {summary['consecutive_overlap']['overlap_p50']}",
        f"- jaccard_mean: {summary['consecutive_overlap']['jaccard_mean']}",
        "",
        "## LRU Simulation",
        "",
        "| sparse capacity | approx original tokens | hit rate | miss rate | requests | misses |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["lru"]:
        hit_rate = item["hit_rate"]
        miss_rate = item["miss_rate"]
        row_template = (
            "| {capacity_sparse_indices} | {approx_capacity_original_tokens} | "
            "{hit_rate} | {miss_rate} | {requests} | {misses} |"
        )
        lines.append(
            row_template.format(
                **{
                    **item,
                    "hit_rate": f"{hit_rate:.6f}" if hit_rate is not None else "n/a",
                    "miss_rate": f"{miss_rate:.6f}" if miss_rate is not None else "n/a",
                }
            )
        )
    lines.append("")
    (out_dir / "topk_lru_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    trace_path = Path(args.trace)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    capacities = parse_capacities(args.capacities)
    records = load_records(trace_path, args.phase, not args.keep_negative)
    groups = group_records(records)

    summary: dict[str, Any] = {
        "trace": str(trace_path),
        "phase": args.phase,
        "records": len(records),
        "groups": len(groups),
        "capacities": capacities,
        "compress_ratio": args.compress_ratio,
        "consecutive_overlap": consecutive_overlap(groups),
        "lru": simulate_lru(groups, capacities, args.compress_ratio),
    }
    (out_dir / "topk_lru_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(out_dir, summary)
    print(f"Wrote {out_dir / 'topk_lru_summary.json'} and {out_dir / 'topk_lru_summary.md'}")


if __name__ == "__main__":
    main()
