#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run and compare single-node SWA OFF/ON streaming benchmarks with vLLM."""

import argparse
import csv
import datetime
import html
import json
import math
import shutil
import statistics
import string
import subprocess
import urllib.request
import uuid
from pathlib import Path

METRICS = {
    **{
        f"{stat}_{metric}_ms": f"{metric.upper()} {stat} (ms)"
        for metric in ("ttft", "tpot", "itl", "e2el")
        for stat in ("mean", "p50", "p95", "p99")
    },
    "output_throughput": "Output tokens/s",
    "request_throughput": "Requests/s",
}
WORKLOAD_KEYS = (
    "model",
    "tokenizer",
    "tokenizer_mode",
    "input_lens",
    "concurrencies",
    "rounds",
    "num_prompts",
    "output_len",
    "warmups",
    "seed",
    "request_rate",
    "allow_prefix_caching",
)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def server_info(args):
    if args.skip_server_info:
        return {"status": "skipped"}, {}
    try:
        with urllib.request.urlopen(
            args.base_url.rstrip("/") + "/server_info?config_format=json", timeout=5
        ) as response:
            info = json.load(response)
    except (OSError, ValueError) as error:
        print(f"Server config unavailable; verify OFF/ON and prefix caching manually: {error}", flush=True)
        return {"status": "unavailable", "error": str(error)}, {}
    if not isinstance(info, dict):
        raise ValueError("Server info is not a JSON object")
    config = info.get("vllm_config", {})
    if isinstance(config, str):
        try:
            config = json.loads(config)
        except ValueError:
            config = {}
    if not isinstance(config, dict):
        config = {}
    cache = config.get("cache_config", {})
    if not isinstance(cache, dict):
        cache = {}
    actual = cache.get("swa_bounded_replay")
    if isinstance(actual, bool) and actual != (args.mode == "on"):
        raise ValueError(f"Server swa_bounded_replay={actual} disagrees with --mode {args.mode}")
    if cache.get("enable_prefix_caching") is True and not args.allow_prefix_caching:
        raise ValueError(
            "Cold-prefill comparison requires --no-enable-prefix-caching on the server. "
            "Use --allow-prefix-caching only for an explicitly separate experiment."
        )
    if not isinstance(actual, bool):
        print("Server does not expose the replay switch; --mode is a manual label.", flush=True)
    # Keep both full evidence and a stable subset for comparison. Runtime IDs,
    # compile cache paths and worker addresses must not create false mismatches.
    fields = {
        "model_config": ("model", "tokenizer", "dtype", "quantization", "max_model_len"),
        "cache_config": ("block_size", "cache_dtype", "gpu_memory_utilization", "enable_prefix_caching"),
        "parallel_config": (
            "tensor_parallel_size",
            "data_parallel_size",
            "enable_expert_parallel",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
        ),
        "scheduler_config": ("max_num_seqs", "max_num_batched_tokens", "async_scheduling"),
        "compilation_config": ("cudagraph_mode", "cudagraph_capture_sizes"),
    }
    signature = {
        group: {key: config[group].get(key) for key in keys}
        for group, keys in fields.items()
        if isinstance(config.get(group), dict)
    }
    signature.update({key: config.get(key) for key in ("additional_config", "engram_config")})
    signature["VLLM_USE_V2_MODEL_RUNNER"] = info.get("vllm_env", {}).get("VLLM_USE_V2_MODEL_RUNNER")
    max_len = signature.get("model_config", {}).get("max_model_len")
    if isinstance(max_len, int) and max(args.input_lens) + args.output_len > max_len:
        raise ValueError("Requested input + output exceeds the server's max_model_len")
    # Do not persist arbitrary system environment or model credentials returned
    # by /server_info. Only retain the benchmark-relevant fields above.
    return {
        "status": "available",
        "swa_bounded_replay": actual,
        "enable_prefix_caching": cache.get("enable_prefix_caching"),
        "configuration": signature,
    }, signature


def validate_result(result, args):
    if result.get("completed") != args.num_prompts or result.get("failed", 0) != 0:
        raise ValueError(f"Incomplete benchmark: completed={result.get('completed')} failed={result.get('failed')}")
    if any(result.get("errors", [])):
        raise ValueError("Benchmark contains request errors; inspect raw JSON/log")
    for key in ("input_lens", "output_lens", "ttfts", "itls"):
        if len(result.get(key, [])) != args.num_prompts:
            raise ValueError(f"Missing per-request detail: {key}; requires --save-detailed")
    if any(length != args.output_len for length in result["output_lens"]):
        raise ValueError("Actual output lengths differ from requested length despite ignore_eos; inspect raw JSON")
    for key in METRICS:
        value = result.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Missing or invalid metric: {key}")


def run(args):
    if args.output_len < 2:
        raise ValueError("--output-len must be at least 2 to measure TPOT and ITL")
    if args.num_prompts < max(args.concurrencies):
        raise ValueError("--num-prompts must be at least the maximum concurrency")
    if len(set(args.input_lens)) != len(args.input_lens) or len(set(args.concurrencies)) != len(args.concurrencies):
        raise ValueError("Input lengths and concurrencies must not contain duplicates")
    executable = shutil.which(args.vllm_bin)
    if not executable:
        raise ValueError(f"vLLM executable not found: {args.vllm_bin}")
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = args.output_dir / f"{args.mode}-{timestamp}-{uuid.uuid4().hex[:8]}"
    folder.mkdir(parents=True)
    manifest = {
        "schema": 1,
        "mode": args.mode,
        "status": "running",
        "base_url": args.base_url,
        "workload": {key: getattr(args, key) for key in WORKLOAD_KEYS},
        "cases": [],
    }
    print(f"Results: {folder.resolve()}\n--mode labels the existing server; it does not restart it.", flush=True)
    try:
        info, signature = server_info(args)
        save(folder / "server-info.json", info)
        manifest["server_signature"] = signature
        version = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=60)
        (folder / "client-version.txt").write_text(version.stdout + version.stderr)
        for round_index in range(1, args.rounds + 1):
            for input_len in args.input_lens:
                for concurrency in args.concurrencies:
                    name = f"i{input_len}-c{concurrency}-r{round_index}"
                    seed = args.seed + round_index - 1
                    command = [
                        executable,
                        "bench",
                        "serve",
                        "--backend",
                        "vllm",
                        "--endpoint",
                        "/v1/completions",
                        "--base-url",
                        args.base_url,
                        "--model",
                        args.model,
                        "--tokenizer",
                        args.tokenizer,
                        "--tokenizer-mode",
                        args.tokenizer_mode,
                        "--dataset-name",
                        "random",
                        "--random-input-len",
                        str(input_len),
                        "--random-output-len",
                        str(args.output_len),
                        "--random-range-ratio",
                        "0",
                        "--random-prefix-len",
                        "0",
                        "--num-prompts",
                        str(args.num_prompts),
                        "--max-concurrency",
                        str(concurrency),
                        "--request-rate",
                        args.request_rate,
                        "--seed",
                        str(seed),
                        "--ignore-eos",
                        "--temperature",
                        "0",
                        "--num-warmups",
                        str(args.warmups),
                        "--percentile-metrics",
                        "ttft,tpot,itl,e2el",
                        "--metric-percentiles",
                        "50,95,99",
                        "--disable-tqdm",
                        "--save-result",
                        "--save-detailed",
                        "--result-dir",
                        str(folder.resolve()),
                        "--result-filename",
                        name + ".json",
                    ]
                    save(folder / (name + ".command.json"), command)
                    case = {
                        "input_len": input_len,
                        "concurrency": concurrency,
                        "round": round_index,
                        "seed": seed,
                        "result": name + ".json",
                        "log": name + ".log",
                        "status": "running",
                    }
                    manifest["cases"].append(case)
                    save(folder / "manifest.json", manifest)
                    print(f"[{args.mode}] {name}: {args.num_prompts} requests; warmups={args.warmups}", flush=True)
                    with (folder / case["log"]).open("w") as log:
                        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                    if process.returncode:
                        raise ValueError(f"vllm bench exited {process.returncode}; see {folder / case['log']}")
                    result = json.loads((folder / case["result"]).read_text())
                    validate_result(result, args)
                    case["status"] = "complete"
                    print(
                        f"  TTFT p50={result['p50_ttft_ms']:.2f}ms; ITL p95={result['p95_itl_ms']:.2f}ms; "
                        f"TPOT p50={result['p50_tpot_ms']:.2f}ms; output={result['output_throughput']:.2f} tokens/s",
                        flush=True,
                    )
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        raise
    finally:
        save(folder / "manifest.json", manifest)
    print(f"DONE: {folder.resolve()}", flush=True)
    save(args.output_dir / f"latest-{args.mode}.json", {"directory": str(folder.resolve())})


def read_run(folder, mode):
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest.get("mode") != mode or manifest.get("status") != "complete":
        raise ValueError(f"{folder} must be a complete {mode} run")
    cases = {}
    for case in manifest["cases"]:
        key = (case["input_len"], case["concurrency"], case["round"], case["seed"])
        if case["status"] != "complete" or key in cases:
            raise ValueError("Incomplete or duplicate case")
        result = json.loads((folder / case["result"]).read_text())
        validate_result(result, argparse.Namespace(**manifest["workload"]))
        cases[key] = result
    workload = manifest["workload"]
    expected = len(set(workload["input_lens"])) * len(set(workload["concurrencies"])) * workload["rounds"]
    if not cases or len(cases) != expected:
        raise ValueError("Run does not contain its full test matrix")
    return manifest, cases


def compare(args):
    if bool(args.off_dir) != bool(args.on_dir):
        raise ValueError("--off-dir and --on-dir must be supplied together")
    if args.off_dir is None:
        args.off_dir = Path(json.loads((args.results_dir / "latest-off.json").read_text())["directory"])
        args.on_dir = Path(json.loads((args.results_dir / "latest-on.json").read_text())["directory"])
    if args.output_dir is None:
        args.output_dir = args.results_dir / "comparison"
    off, off_cases = read_run(args.off_dir, "off")
    on, on_cases = read_run(args.on_dir, "on")
    if off["workload"] != on["workload"] or off_cases.keys() != on_cases.keys():
        raise ValueError("OFF/ON workloads, seeds or test matrices differ")
    if off.get("server_signature") and on.get("server_signature") and off["server_signature"] != on["server_signature"]:
        raise ValueError("Server configurations differ beyond the replay switch; inspect server-info.json")
    for key in off_cases:
        for lengths in ("input_lens", "output_lens"):
            if off_cases[key][lengths] != on_cases[key][lengths]:
                raise ValueError(f"Actual {lengths} differ for {key}; comparison would mix different workloads")
    rows = []
    for input_len, concurrency in sorted({key[:2] for key in off_cases}):
        keys = sorted(key for key in off_cases if key[:2] == (input_len, concurrency))
        for metric, title in METRICS.items():
            before = [off_cases[key][metric] for key in keys]
            after = [on_cases[key][metric] for key in keys]
            b, a = statistics.median(before), statistics.median(after)
            rows.append(
                {
                    "input_len": input_len,
                    "concurrency": concurrency,
                    "metric": metric,
                    "title": title,
                    "rounds": len(keys),
                    "off_median": b,
                    "on_median": a,
                    "off_min": min(before),
                    "off_max": max(before),
                    "on_min": min(after),
                    "on_max": max(after),
                    "change_percent": (a / b - 1) * 100 if b else None,
                }
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save(
        args.output_dir / "comparison.json",
        {"off_dir": str(args.off_dir), "on_dir": str(args.on_dir), "workload": off["workload"], "rows": rows},
    )
    with (args.output_dir / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    body = []
    for row in rows:
        change = row["change_percent"]
        better = change is not None and (change < 0 if row["metric"].endswith("_ms") else change > 0)
        color = "good" if better else "bad" if change else ""
        cells = [
            str(row["input_len"]),
            str(row["concurrency"]),
            row["title"],
            f"{row['off_median']:.2f}",
            f"{row['on_median']:.2f}",
            f"{change:+.2f}%" if change is not None else "—",
            f"{row['off_min']:.2f}–{row['off_max']:.2f}",
            f"{row['on_min']:.2f}–{row['on_max']:.2f}",
        ]
        body.append(f'<tr class="{color}">' + "".join(f"<td>{html.escape(cell)}</td>" for cell in cells) + "</tr>")
    report = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>SWA OFF/ON benchmark</title>
<style>body{font:15px system-ui;max-width:1300px;margin:32px auto;padding:0 16px;color:#172238}
table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:right}
th{background:#eef2f8;position:sticky;top:0}.good td:nth-child(6){color:#08743c}.bad td:nth-child(6){color:#b22828}
code{background:#eef2f8;padding:3px}p{line-height:1.7}.scroll{overflow:auto}</style>
<h1>SWA replay OFF / ON</h1><p>OFF：$OFF_SOURCE<br>ON：$ON_SOURCE</p><p>$WORKLOAD</p>
<p>每格为各轮统计量的中位数，末两列显示轮间范围；不是合并所有请求后重新计算的百分位。
变化 = (ON / OFF − 1) × 100%；延迟负数更好，吞吐正数更好。需要结合轮间波动判断收益。</p>
<p>TTFT：首个流式输出到达时间；TPOT：(请求耗时 − TTFT) / (输出 token 数 − 1)。
ITL沿用vLLM的流式到达间隔；投机解码或多token合并发送时，可能是chunk间隔，并非严格逐token耗时。
不直接等同于NPU算子耗时。失败或输出长度不一致的实验不会生成有效对照。</p>
<div class="scroll"><table><thead><tr><th>输入</th><th>并发</th><th>指标</th><th>OFF</th><th>ON</th>
<th>变化</th><th>OFF轮间范围</th><th>ON轮间范围</th></tr></thead><tbody>$ROWS</tbody></table></div>
<p>运行标签由操作者设置；应确认服务重启后的replay状态、实际裁剪日志与相同启动配置。
该随机负载用于性能测试，不评估模型精度。</p></html>"""
    (args.output_dir / "report.html").write_text(
        string.Template(report).substitute(
            ROWS="".join(body),
            WORKLOAD=html.escape(json.dumps(off["workload"], ensure_ascii=False)),
            OFF_SOURCE=html.escape(str(args.off_dir.resolve())),
            ON_SOURCE=html.escape(str(args.on_dir.resolve())),
        )
    )
    print(f"Comparison: {(args.output_dir / 'report.html').resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    bench = commands.add_parser("run", help="Benchmark an already-running OFF or ON service")
    bench.add_argument("--mode", choices=("off", "on"), required=True)
    bench.add_argument("--base-url", default="http://127.0.0.1:8989")
    bench.add_argument("--model", default="dsv41")
    bench.add_argument("--tokenizer", required=True, help="Local checkpoint/tokenizer directory")
    bench.add_argument("--tokenizer-mode", default="deepseek_v41")
    bench.add_argument("--input-lens", nargs="+", type=positive, default=[16384])
    bench.add_argument("--concurrencies", nargs="+", type=positive, default=[1, 4])
    bench.add_argument("--rounds", type=positive, default=3)
    bench.add_argument("--num-prompts", type=positive, default=32)
    bench.add_argument("--output-len", type=positive, default=128)
    bench.add_argument("--warmups", type=positive, default=8)
    bench.add_argument("--seed", type=int, default=42)
    bench.add_argument("--request-rate", default="inf")
    bench.add_argument("--output-dir", type=Path, default=Path("/tmp/swa-bench"))
    bench.add_argument("--vllm-bin", default="vllm")
    bench.add_argument("--skip-server-info", action="store_true")
    bench.add_argument("--allow-prefix-caching", action="store_true")
    summary = commands.add_parser("compare", help="Validate two matched runs and create HTML/CSV/JSON")
    summary.add_argument(
        "--results-dir", type=Path, default=Path("/tmp/swa-bench"), help="Find latest complete OFF/ON runs"
    )
    summary.add_argument("--off-dir", type=Path)
    summary.add_argument("--on-dir", type=Path)
    summary.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "run":
            run(args)
        else:
            compare(args)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
