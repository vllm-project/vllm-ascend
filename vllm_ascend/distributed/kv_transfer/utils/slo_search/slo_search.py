#!/usr/bin/env python3
"""Search AISBench concurrency through aisbench_test.py's CLI only.

Example:
  python slo_search.py --project-dir /path/to/aisbench_auto_tools_prefix \
      --ttft-limit-ms 10000 --tpot-limit-ms 20 --percentile 90 \
      --max-concurrency 256 -- --input_len 2048 --output_len 1024

Only Python's standard library is required. No project modules are imported.
"""

import argparse
import csv
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
MANAGED_ARGS = {"--concurrency", "--request_rate", "--repeat", "--test_accuracy", "--data_num"}


class TrialError(RuntimeError):
    """An invalid trial must not be treated as an SLO failure."""


def percentile_name(value):
    if value.lower() in ("mean", "avg"):
        return "mean"
    value = value.upper().removeprefix("P")
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("use mean or a percentile such as 90, 95, 99")
    if not math.isfinite(number) or not 0 < number < 100:
        raise argparse.ArgumentTypeError("percentile must be between 0 and 100")
    return f"P{number:g}"


def parse_cache_summary(text, required=False):
    """Read the last ALL_PODS summary, not the buffered warmup summary.

    External KV cache retains its original name. Zero queries means an
    undefined rate, even when the dependency formats it as 0.00%.
    """
    result = {}
    for cache in ("hbm", "external"):
        result.update(
            {
                f"{cache}_hit_rate_pct": None,
                f"{cache}_hits": None,
                f"{cache}_queries": None,
                f"{cache}_cache_status": "not_reported",
            }
        )
    if "Prefix cache hit summary" not in text:
        if required:
            raise TrialError("missing full-workload Prefix cache hit summary")
        return result
    block = text.rsplit("Prefix cache hit summary", 1)[1]
    if "Skipped PODs:" in block:
        raise TrialError("prefix cache summary has skipped PODs; totals are incomplete")
    header = None
    for line in block.splitlines():
        cells = line.split()
        if cells and cells[0] == "scope":
            header = cells
        if not cells or cells[0] != "ALL_PODS":
            continue
        if header is None or len(cells) != len(header):
            raise TrialError("cache summary row/header column count mismatch")
        values = dict(zip(header, cells))
        for cache in ("hbm", "external"):
            rate_key, count_key = f"{cache}_hit_rate", f"{cache}(hit/query)"
            if rate_key not in values and count_key not in values:
                continue
            if rate_key not in values or count_key not in values:
                raise TrialError(f"incomplete {cache} cache columns")
            counts = re.fullmatch(f"({NUMBER})/({NUMBER})", values[count_key])
            rate = re.fullmatch(f"({NUMBER})%", values[rate_key])
            if counts is None or rate is None:
                raise TrialError(f"invalid {cache} cache counters or rate")
            hits, queries = float(counts[1]), float(counts[2])
            displayed = float(rate[1])
            if not all(math.isfinite(v) for v in (hits, queries, displayed)) or not 0 <= hits <= queries:
                raise TrialError(f"invalid {cache} cache counter values")
            if hits != int(hits) or queries != int(queries):
                raise TrialError(f"non-integral {cache} token counts")
            percentage = hits / queries * 100 if queries else None
            if (
                not 0 <= displayed <= 100
                or (queries == 0 and displayed != 0)
                or (percentage is not None and abs(displayed - percentage) > 0.0051)
            ):
                raise TrialError(f"{cache} cache rate disagrees with hit/query counts")
            result.update(
                {
                    f"{cache}_hit_rate_pct": percentage,
                    f"{cache}_hits": int(hits),
                    f"{cache}_queries": int(queries),
                    f"{cache}_cache_status": "reported" if queries else "no_queries",
                }
            )
        if required and any(result[f"{cache}_cache_status"] == "not_reported" for cache in ("hbm", "external")):
            raise TrialError("cache summary must contain HBM and external columns")
        return result
    raise TrialError("cache summary has no ALL_PODS row")


def parse_metrics(text, percentile, latency_unit, stage="total", require_cache=False, tpot_applicable=True):
    """Read named columns in AISBench's ASCII/Unicode summary tables.

    Select a stage explicitly and read units from cells or metric labels.
    Never count numeric substrings: Stage and N are not latency statistics.
    """
    text = ANSI.sub("", text)
    if re.search(r"Traceback \(most recent call last\)|\bERROR\b", text):
        raise TrialError("child output contains ERROR or a Python traceback; inspect console.log")
    # Prefix warmup and the full workload each print summaries. Only use the
    # full workload, even if its summary is incomplete.
    text = text.rsplit("[开始] 全量数据集测试", 1)[-1]
    header = None
    result = {}
    counters = {}
    labels = {
        "Output Token Throughput": ("output_tokens_per_s", "token/s"),
        "Total Token Throughput": ("total_tokens_per_s", "token/s"),
        "Prefill Token Throughput": ("prefill_tokens_per_s", "token/s"),
        "Request Throughput": ("requests_per_s", "req/s"),
        "Total Requests": ("total", None),
        "Success Requests": ("successful", None),
        "Successful Requests": ("successful", None),
        "Failed Requests": ("failed", None),
    }

    def quantity(cell):
        match = re.fullmatch(r"\s*(" + NUMBER + r")\s*([^\s]*)\s*", cell)
        if not match:
            raise TrialError(f"invalid metric value: {cell!r}")
        value = float(match[1])
        if not math.isfinite(value) or value < 0:
            raise TrialError(f"non-finite or negative metric: {cell!r}")
        return value, match[2]

    for line in text.splitlines():
        if not re.match(r"\s*[|│]", line):
            continue
        cells = [cell.strip() for cell in re.split(r"[|│]", line.strip())[1:-1]]
        if not cells:
            continue
        normalized = ["mean" if cell.lower() in ("average", "avg", "mean") else cell.upper() for cell in cells]
        if (
            "mean" in normalized and any(re.fullmatch(r"P\d+(?:\.\d+)?", cell) for cell in normalized)
        ) or "VALUE" in normalized:
            header = normalized
            continue
        metric = re.fullmatch(r"(.*?)\s*(?:\(([^()]*)\))?", cells[0])
        name, row_unit = metric[1].strip(), metric[2]
        if name == "TPOT" and not tpot_applicable:
            continue
        if name not in ("TTFT", "TPOT") and name not in labels:
            continue
        if header is None or len(header) != len(cells):
            raise TrialError(f"{name} row/header column count mismatch")
        if "STAGE" in header and cells[header.index("STAGE")] != stage:
            continue
        if name in ("TTFT", "TPOT"):
            latencies = {}
            for stat in dict.fromkeys(("mean", "P90", percentile)):
                if stat not in header:
                    raise TrialError(f"{name} table has no {stat} column")
                value, cell_unit = quantity(cells[header.index(stat)])
                if row_unit and cell_unit and row_unit != cell_unit:
                    raise TrialError(f"conflicting latency units: {row_unit}, {cell_unit}")
                unit = cell_unit or row_unit or latency_unit
                if unit is None:
                    raise TrialError("latency row has no unit; supply --latency-unit ms or s")
                scales = {"s": 1000, "ms": 1, "us": 0.001, "µs": 0.001}
                if unit not in scales:
                    raise TrialError(f"unsupported latency unit: {unit}")
                latencies[stat] = value * scales[unit]
            result.update(
                {
                    name.lower() + "_ms": latencies[percentile],
                    name.lower() + "_mean_ms": latencies["mean"],
                    name.lower() + "_p90_ms": latencies["P90"],
                }
            )
            continue
        if "VALUE" not in header:
            raise TrialError(f"{name} table has no Value column")
        value, cell_unit = quantity(cells[header.index("VALUE")])
        key, expected_unit = labels[name]
        if cell_unit and row_unit and cell_unit != row_unit:
            raise TrialError(f"conflicting units for {name}")
        unit = cell_unit or row_unit
        if unit != expected_unit:
            raise TrialError(f"{name} requires unit {expected_unit!r}, got {unit!r}")
        target = counters if key in ("total", "successful", "failed") else result
        target[key] = value
    required = {"ttft_ms", "prefill_tokens_per_s", "output_tokens_per_s", "requests_per_s"}
    if tpot_applicable:
        required.add("tpot_ms")
    missing = required - result.keys()
    if missing:
        raise TrialError(f"missing summary metrics for stage {stage!r}: {', '.join(sorted(missing))}")
    if "total" not in counters or "failed" not in counters:
        raise TrialError("summary must contain Total Requests and Failed Requests")
    total, failed = counters["total"], counters["failed"]
    if total <= 0 or not 0 <= failed <= total or total != int(total) or failed != int(failed):
        raise TrialError("invalid request counts")
    if "successful" in counters and counters["successful"] != total - failed:
        raise TrialError("inconsistent request counts")
    if any(not math.isfinite(v) or v < 0 for v in result.values()):
        raise TrialError("non-finite or negative metrics")
    result.update(total_requests=int(total), failed_requests=int(failed), success_rate=(total - failed) / total)
    result["tpot_status"] = "reported" if tpot_applicable else "not_applicable_single_token"
    if not tpot_applicable:
        result.update(tpot_ms=None, tpot_mean_ms=None, tpot_p90_ms=None)
    result.update(parse_cache_summary(text, required=require_cache))
    return result


def metric_text(value):
    return "N/A" if value is None else f"{value:g}"


def format_trial_metrics(row):
    headers = [
        "Stage",
        "TTFT mean (ms)",
        "TTFT P90 (ms)",
        "TPOT mean (ms)",
        "TPOT P90 (ms)",
        "Prefill (token/s)",
        "hbm_hit_rate (%)",
        "external_hit_rate (%)",
    ]
    values = [
        row["stage"],
        metric_text(row["ttft_mean_ms"]),
        metric_text(row["ttft_p90_ms"]),
        metric_text(row["tpot_mean_ms"]),
        metric_text(row["tpot_p90_ms"]),
        f"{row['prefill_tokens_per_s']:g}",
    ]
    for cache in ("hbm", "external"):
        value = row[f"{cache}_hit_rate_pct"]
        values.append(f"{value:.2f}" if value is not None else f"N/A ({row[f'{cache}_cache_status']})")
    if row["percentile"] not in ("mean", "P90"):
        headers.extend([f"SLO TTFT {row['percentile']} (ms)", f"SLO TPOT {row['percentile']} (ms)"])
        values.extend([metric_text(row["ttft_ms"]), metric_text(row["tpot_ms"])])
    table = "\n".join("| " + " | ".join(cells) + " |" for cells in (headers, ["---"] * len(headers), values))
    note = (
        "TPOT is not applicable with one output token.\n"
        if row.get("tpot_status") == "not_applicable_single_token"
        else ""
    )
    return (
        f"SLO trial metrics (SLO statistic: {row['percentile']}; cache interval: full workload)\n" + table + "\n" + note
    )


class Search:
    def __init__(self, args, forwarded, output):
        self.args, self.forwarded, self.output = args, forwarded, output
        self.history = []
        self.confirmed = {}

    def record(self, row):
        self.history.append(row)
        with (self.output / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        fields = [
            "trial",
            "phase",
            "concurrency",
            "prefix_num",
            "data_num",
            "status",
            "passed",
            "stage",
            "percentile",
            "ttft_limit_ms",
            "tpot_limit_ms",
            "min_success_rate",
            "ttft_ms",
            "tpot_ms",
            "ttft_mean_ms",
            "ttft_p90_ms",
            "tpot_mean_ms",
            "tpot_p90_ms",
            "tpot_status",
            "success_rate",
            "total_requests",
            "failed_requests",
            "prefill_tokens_per_s",
            "output_tokens_per_s",
            "total_tokens_per_s",
            "requests_per_s",
            *[
                f"{cache}_{field}"
                for cache in ("hbm", "external")
                for field in ("hit_rate_pct", "hits", "queries", "cache_status")
            ],
            "started_at",
            "returncode",
            "log",
            "error",
        ]
        with (self.output / "history.csv").open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            if len(self.history) == 1:
                writer.writeheader()
            writer.writerow(row)
        (self.output / f"trial_{row['trial']:04d}" / "result.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if row["status"] != "error":
            table = format_trial_metrics(row)
            (self.output / f"trial_{row['trial']:04d}" / "metrics.txt").write_text(table, encoding="utf-8")
            with Path(row["log"]).open("a", encoding="utf-8") as stream:
                stream.write("\n" + table)

    def trial(self, concurrency, phase):
        """Run a CLI workload, validate its SLO, and persist its metrics before reporting."""
        a = self.args
        trial = len(self.history) + 1
        folder = self.output / f"trial_{trial:04d}"
        folder.mkdir()
        log = folder / "console.log"
        count = a.data_num if a.data_num is not None else max(a.min_data_num, concurrency * a.requests_per_concurrency)
        command = [
            a.python,
            str(a.project_dir / "aisbench_test.py"),
            *self.forwarded,
            "--concurrency",
            str(concurrency),
            "--request_rate",
            "0",
            "--repeat",
            "1",
            "--data_num",
            str(count),
        ]
        if a.prefix_num_equals_concurrency:
            command.extend(["--prefix_num", str(concurrency)])
        row = dict(
            trial=trial,
            phase=phase,
            concurrency=concurrency,
            data_num=count,
            status="error",
            passed=False,
            percentile=a.percentile,
            stage=a.stage,
            ttft_limit_ms=a.ttft_limit_ms,
            tpot_limit_ms=a.tpot_limit_ms,
            min_success_rate=a.min_success_rate,
            started_at=datetime.now(timezone.utc).isoformat(),
            log=str(log),
            command=command,
        )
        if a.prefix_num_equals_concurrency:
            row["prefix_num"] = concurrency
        print(f"[{trial}] {phase}: concurrency={concurrency}, data_num={count}", flush=True)
        (folder / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
        process = None
        try:
            with log.open("wb") as stream:
                process = subprocess.Popen(
                    command,
                    cwd=a.project_dir,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=(os.name != "nt"),
                )
                try:
                    row["returncode"] = process.wait(timeout=a.timeout)
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    if os.name == "nt":
                        subprocess.run(
                            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    else:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise
            if row["returncode"] != 0:
                raise TrialError(f"child exited with code {row['returncode']}")
            row.update(
                parse_metrics(
                    log.read_text(encoding="utf-8", errors="replace"),
                    a.percentile,
                    a.latency_unit,
                    a.stage,
                    require_cache="--prefix_test" in self.forwarded,
                    tpot_applicable=not a.single_token_output,
                )
            )
            row["passed"] = (
                (a.ttft_limit_ms is None or row["ttft_ms"] < a.ttft_limit_ms)
                and (a.tpot_limit_ms is None or row["tpot_ms"] < a.tpot_limit_ms)
                and row["success_rate"] >= a.min_success_rate
            )
            row["status"] = "pass" if row["passed"] else "fail"
        except (OSError, TrialError, subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            row["error"] = str(exc) or type(exc).__name__
            self.record(row)
            raise TrialError(f"trial {trial}: {row['error']}; log: {log}") from exc
        self.record(row)

        def rate(cache):
            value = row[f"{cache}_hit_rate_pct"]
            return f"{value:.2f}%" if value is not None else f"N/A ({row[f'{cache}_cache_status']})"

        latency = ", ".join(
            f"{metric.upper()}({stat})={metric_text(row[f'{metric}_{stat.lower()}_ms'])} ms"
            for metric in ("ttft", "tpot")
            for stat in ("mean", "P90")
        )
        if a.percentile not in ("mean", "P90"):
            latency += (
                f", TTFT({a.percentile})={metric_text(row['ttft_ms'])} ms, "
                f"TPOT({a.percentile})={metric_text(row['tpot_ms'])} ms"
            )
        print(
            f"    {row['status'].upper()} (SLO: {a.percentile}): {latency}, "
            f"output={row['output_tokens_per_s']:g} token/s, "
            f"requests={row['requests_per_s']:g}/s, prefill={row['prefill_tokens_per_s']:g} token/s, "
            f"HBM={rate('hbm')}, external={rate('external')}",
            flush=True,
        )
        print(format_trial_metrics(row), flush=True)
        return row

    def run(self):
        a = self.args
        low, high = a.min_concurrency - 1, None
        current = a.start_concurrency
        while True:
            if self.trial(current, "expand")["passed"]:
                low = current
                if current == a.max_concurrency:
                    break
                current = min(current * 2, a.max_concurrency)
            else:
                high = current
                break
        if high is not None:
            while high - low > 1:
                mid = (low + high) // 2
                if self.trial(mid, "bisect")["passed"]:
                    low = mid
                else:
                    high = mid

        if a.confirm_runs:
            # Fresh repetitions at each neighbor. All repetitions must pass.
            upper = min(a.max_concurrency, max(low, a.min_concurrency) + a.boundary_radius)
            lower = max(a.min_concurrency, low - a.boundary_radius)
            for concurrency in range(upper, lower - 1, -1):
                self.confirm(concurrency)
            winner = self.winner()
            while winner is None and lower > a.min_concurrency:
                lower -= 1
                self.confirm(lower)
                winner = self.winner()
            rows = self.confirmed[winner] if winner is not None else []
        else:
            winner = low if low >= a.min_concurrency else None
            rows = [r for r in self.history if r["passed"] and r["concurrency"] == winner]
        summary = {
            "status": "complete",
            "percentile": a.percentile,
            "stage": a.stage,
            "ttft_limit_ms": a.ttft_limit_ms,
            "tpot_limit_ms": a.tpot_limit_ms,
            "active_slo_metrics": [
                name for name, limit in (("ttft", a.ttft_limit_ms), ("tpot", a.tpot_limit_ms)) if limit is not None
            ],
            "min_success_rate": a.min_success_rate,
            "highest_passing_concurrency": winner,
            "search_range": [a.min_concurrency, a.max_concurrency],
            "at_search_limit": winner == a.max_concurrency,
            "confirmation_runs": a.confirm_runs,
            "confirmation_performed": bool(a.confirm_runs),
            "result_source": "confirm" if a.confirm_runs else "search",
            "result_trial_ids": [r["trial"] for r in rows],
            "confirmation_trial_ids": [r["trial"] for r in rows] if a.confirm_runs else [],
            "confirmation_cache_metrics": [],
            "trials": len(self.history),
            "scope": (
                "Highest confirmed passing concurrency"
                if a.confirm_runs
                else "Highest passing concurrency without confirmation"
            )
            + " found under an approximately monotonic SLO assumption; not an exhaustive scan.",
        }
        if winner is not None:
            summary["throughput"] = {
                "output_tokens_per_s_mean": statistics.mean(r["output_tokens_per_s"] for r in rows),
                "requests_per_s_mean": statistics.mean(r["requests_per_s"] for r in rows),
                "prefill_tokens_per_s_mean": statistics.mean(r["prefill_tokens_per_s"] for r in rows),
            }
            summary["result_cache_metrics"] = [
                {
                    "trial": r["trial"],
                    **{key: value for key, value in r.items() if key.startswith(("hbm_", "external_"))},
                }
                for r in rows
            ]
            if a.confirm_runs:
                summary["confirmation_cache_metrics"] = summary["result_cache_metrics"]
            summary["worst_ttft_ms"] = max(r["ttft_ms"] for r in rows)
            tpot_values = [r["tpot_ms"] for r in rows if r["tpot_ms"] is not None]
            summary["worst_tpot_ms"] = max(tpot_values) if tpot_values else None
        return summary

    def confirm(self, concurrency):
        self.confirmed[concurrency] = [self.trial(concurrency, "confirm") for _ in range(self.args.confirm_runs)]

    def winner(self):
        passed = [c for c, rows in self.confirmed.items() if all(r["passed"] for r in rows)]
        return max(passed) if passed else None


def arguments(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    split = argv.index("--") if "--" in argv else len(argv)
    forwarded = argv[split + 1 :]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--python", default=sys.executable, help="Python executable for the existing project")
    parser.add_argument(
        "--ttft-limit-ms", type=float, help="TTFT upper bound in ms; omit to disable (at least one bound required)"
    )
    parser.add_argument(
        "--tpot-limit-ms", type=float, help="TPOT upper bound in ms; omit to disable (at least one bound required)"
    )
    parser.add_argument(
        "--percentile", type=percentile_name, default="P90", help="mean, 90, 95, 99, etc.; must exist in the log header"
    )
    parser.add_argument("--stage", default="total", help="exact AISBench Stage name to evaluate (default: total)")
    parser.add_argument("--latency-unit", choices=("ms", "s"), help="required only if log latency rows omit units")
    parser.add_argument("--min-success-rate", type=float, default=1.0)
    parser.add_argument("--min-concurrency", type=int, default=1)
    parser.add_argument("--start-concurrency", type=int, default=8)
    parser.add_argument("--max-concurrency", type=int, default=1024)
    parser.add_argument(
        "--confirm-runs", type=int, default=0, help="repetitions per boundary point; 0 skips confirmation (default)"
    )
    parser.add_argument("--boundary-radius", type=int, default=1)
    parser.add_argument("--data-num", type=int, help="fixed request count; otherwise scales with concurrency")
    parser.add_argument("--min-data-num", type=int, default=0, help="optional request-count floor (default: no floor)")
    parser.add_argument("--requests-per-concurrency", type=int, default=4)
    parser.add_argument(
        "--prefix-num-equals-concurrency",
        action="store_true",
        help="set --prefix_num to each trial's concurrency; requires --dataset_type prefix_cache",
    )
    parser.add_argument("--timeout", type=float, default=3600, help="seconds per child command")
    parser.add_argument("--output-dir", type=Path, help="new, non-existing directory for this search")
    a = parser.parse_args(argv[:split])
    for token in forwarded:
        if token.split("=", 1)[0] in MANAGED_ARGS:
            parser.error(f"{token} is managed by slo_search; do not pass it after --")
    if a.prefix_num_equals_concurrency:
        dataset_type = None
        for index, token in enumerate(forwarded):
            option = token.split("=", 1)[0]
            if option in ("--prefix_num", "--dataset"):
                parser.error(f"{option} cannot be combined with --prefix-num-equals-concurrency")
            if option == "--dataset_type":
                dataset_type = (
                    token.split("=", 1)[1]
                    if "=" in token
                    else (forwarded[index + 1] if index + 1 < len(forwarded) else None)
                )
        if dataset_type != "prefix_cache":
            parser.error("--prefix-num-equals-concurrency requires --dataset_type prefix_cache after --")
    if a.ttft_limit_ms is None and a.tpot_limit_ms is None:
        parser.error("specify at least one of --ttft-limit-ms or --tpot-limit-ms")
    output_len = None
    for index, token in enumerate(forwarded):
        if token.split("=", 1)[0] == "--output_len":
            output_len = (
                token.split("=", 1)[1]
                if "=" in token
                else (forwarded[index + 1] if index + 1 < len(forwarded) else None)
            )
    try:
        a.single_token_output = output_len is not None and int(output_len) == 1
    except ValueError:
        parser.error("--output_len must be an integer")
    if a.single_token_output and a.tpot_limit_ms is not None:
        parser.error("TPOT is undefined for --output_len 1; omit --tpot-limit-ms")
    for name in ("ttft_limit_ms", "tpot_limit_ms", "timeout"):
        value = getattr(a, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not 0 < a.min_success_rate <= 1:
        parser.error("--min-success-rate must be in (0, 1]")
    if not 1 <= a.min_concurrency <= a.start_concurrency <= a.max_concurrency:
        parser.error("require 1 <= min-concurrency <= start-concurrency <= max-concurrency")
    if a.requests_per_concurrency < 1 or min(a.confirm_runs, a.min_data_num, a.boundary_radius) < 0:
        parser.error(
            "requests-per-concurrency must be positive; "
            "confirm-runs, min-data-num and boundary-radius must be nonnegative"
        )
    if a.data_num is not None and a.data_num < a.max_concurrency:
        parser.error("--data-num must be >= max-concurrency to exercise the requested concurrency")
    a.project_dir = a.project_dir.resolve()
    if not (a.project_dir / "aisbench_test.py").is_file():
        parser.error("project-dir must contain aisbench_test.py")
    return a, forwarded


def main(argv=None):
    args, forwarded = arguments(argv)
    output = (
        args.output_dir or args.project_dir / "slo_results" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ).resolve()
    # Prevent overlapping searches from overwriting the project's shared files.
    lock = args.project_dir / ".slo_search.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(
            f"ERROR: another search holds {lock}; after an interrupted run, "
            "verify its process has stopped before removing it.",
            file=sys.stderr,
        )
        return 2
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(str(os.getpid()))
        output.mkdir(parents=True, exist_ok=False)
        settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        settings["forwarded_args"] = forwarded
        (output / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
        search = Search(args, forwarded, output)
        try:
            summary = search.run()
        except (TrialError, KeyboardInterrupt) as exc:
            summary = {"status": "aborted", "error": str(exc), "trials": len(search.history)}
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Results: {output}")
        return 2 if summary["status"] == "aborted" else (0 if summary["highest_passing_concurrency"] is not None else 1)
    except OSError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        lock.unlink()


if __name__ == "__main__":
    sys.exit(main())
