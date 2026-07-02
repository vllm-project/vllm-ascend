#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any


LOG_PATTERNS = {
    "startup_complete": re.compile(r"Application startup complete", re.IGNORECASE),
    "kv_pool_load_spec": re.compile(r"KV pool load spec created", re.IGNORECASE),
    "kv_pool_backend_get": re.compile(r"backend get", re.IGNORECASE),
    "kv_pool_store": re.compile(r"Storing KV cache|store.*KV", re.IGNORECASE),
    "kv_load_failure": re.compile(r"Failed to load blocks|kv.*load.*fail", re.IGNORECASE),
    "traceback": re.compile(r"Traceback|RuntimeError|Exception|ERROR|OOM", re.IGNORECASE),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize one DSV4 sparse KV experiment output directory.")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def scan_log(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"exists": path.exists(), "path": str(path), "patterns": {}, "interesting_lines": []}
    if not path.exists():
        return result
    for key in LOG_PATTERNS:
        result["patterns"][key] = 0
    with path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line_no, line in enumerate(log_file, 1):
            for key, pattern in LOG_PATTERNS.items():
                if pattern.search(line):
                    result["patterns"][key] += 1
                    if key in {"kv_load_failure", "traceback"} and len(result["interesting_lines"]) < 80:
                        result["interesting_lines"].append({"line": line_no, "text": line.rstrip()[:1000]})
    return result


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_probe_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as results_file:
        for line in results_file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_markdown(out_dir: Path, summary: dict[str, Any]) -> None:
    probe = summary.get("probe_summary") or {}
    lru = summary.get("topk_lru_summary") or {}
    lines = [
        "# DeepSeek V4 Sparse KV Experiment Summary",
        "",
        f"- out_dir: `{summary['out_dir']}`",
        f"- serve_command_file: `{summary.get('serve_command_file')}`",
        "",
        "## Probe Requests",
        "",
        f"- total: {probe.get('total', 'n/a')}",
        f"- ok: {probe.get('ok', 'n/a')}",
        f"- failed: {probe.get('failed', 'n/a')}",
        f"- latency_mean_s: {probe.get('latency_mean_s', 'n/a')}",
        f"- latency_p90_s: {probe.get('latency_p90_s', 'n/a')}",
        "",
        "## vLLM Log Counters",
        "",
        "| counter | count |",
        "|---|---:|",
    ]
    for key, value in summary["run_dsv4_log"].get("patterns", {}).items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Mooncake Log Counters", "", "| counter | count |", "|---|---:|"])
    for key, value in summary["run_mooncake_log"].get("patterns", {}).items():
        lines.append(f"| {key} | {value} |")

    if lru:
        lines.extend(["", "## Top-k LRU", "", "| sparse capacity | approx original tokens | hit rate | miss rate |"])
        lines.append("|---:|---:|---:|---:|")
        for item in lru.get("lru", []):
            hit_rate = item.get("hit_rate")
            miss_rate = item.get("miss_rate")
            lines.append(
                "| {capacity} | {tokens} | {hit_rate} | {miss_rate} |".format(
                    capacity=item.get("capacity_sparse_indices"),
                    tokens=item.get("approx_capacity_original_tokens"),
                    hit_rate=f"{hit_rate:.6f}" if hit_rate is not None else "n/a",
                    miss_rate=f"{miss_rate:.6f}" if miss_rate is not None else "n/a",
                )
            )
        overlap = lru.get("consecutive_overlap", {})
        lines.extend(
            [
                "",
                f"- consecutive_pairs: {overlap.get('pairs')}",
                f"- overlap_mean: {overlap.get('overlap_mean')}",
                f"- jaccard_mean: {overlap.get('jaccard_mean')}",
            ]
        )

    interesting = summary["run_dsv4_log"].get("interesting_lines", [])
    if interesting:
        lines.extend(["", "## Interesting vLLM Lines", ""])
        for item in interesting[:30]:
            lines.append(f"- line {item['line']}: `{item['text']}`")

    (out_dir / "experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    summary: dict[str, Any] = {
        "out_dir": str(out_dir),
        "serve_command_file": str(out_dir / "serve_command.txt"),
        "run_dsv4_log": scan_log(out_dir / "run_dsv4.log"),
        "run_mooncake_log": scan_log(out_dir / "run_mooncake.log"),
        "probe_summary": load_json(out_dir / "probe_summary.json"),
        "probe_results_count": len(load_probe_results(out_dir / "probe_results.jsonl")),
        "topk_lru_summary": load_json(out_dir / "topk_lru_summary.json"),
    }
    (out_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(out_dir, summary)
    print(f"Wrote {out_dir / 'experiment_summary.json'} and {out_dir / 'experiment_summary.md'}")


if __name__ == "__main__":
    main()
