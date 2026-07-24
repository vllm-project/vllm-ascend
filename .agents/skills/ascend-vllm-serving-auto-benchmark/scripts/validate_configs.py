#!/usr/bin/env python3
"""Validate ascend-vllm-serving-auto-benchmark YAML configs.

Checks that required fields are present, values are in allowed ranges,
and parallel_levels / requests_per_level lists are the same length.
Does NOT launch any server.
"""

import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Run: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

REQUIRED_MODEL_KEYS = {"path", "name", "dtype", "max_model_len"}
REQUIRED_SERVER_KEYS = {"port", "tensor_parallel_size", "pipeline_parallel_size"}
REQUIRED_WORKLOAD_KEYS = {"input_tokens", "output_tokens", "dataset"}
REQUIRED_BENCHMARK_KEYS = {"parallel_levels", "requests_per_level", "warmup_requests"}

VALID_DTYPES = {"float16", "bfloat16", "float32", "auto"}
VALID_QUANT = {None, "w8a8", "w4a16", "gptq", "awq"}
VALID_DATASETS = {"random", "openqa", "longalpaca"}

ERRORS: list[str] = []
WARNINGS: list[str] = []


def _err(msg: str) -> None:
    ERRORS.append(msg)


def _warn(msg: str) -> None:
    WARNINGS.append(msg)


def _check_required(section: dict, keys: set[str], section_name: str) -> None:
    for k in keys:
        if k not in section:
            _err(f"[{section_name}] missing required key: '{k}'")


def validate_model(cfg: dict) -> None:
    section = cfg.get("model", {})
    _check_required(section, REQUIRED_MODEL_KEYS, "model")

    dtype = section.get("dtype", "")
    if dtype not in VALID_DTYPES:
        _err(f"[model] dtype '{dtype}' invalid; must be one of {VALID_DTYPES}")

    quant = section.get("quantization")
    if quant not in VALID_QUANT:
        _warn(f"[model] quantization '{quant}' is not in known values {VALID_QUANT}")

    max_len = section.get("max_model_len", 0)
    if not isinstance(max_len, int) or max_len <= 0:
        _err(f"[model] max_model_len must be a positive integer, got: {max_len!r}")

    model_path = section.get("path", "")
    if not model_path:
        _err("[model] path must not be empty")


def validate_server(cfg: dict) -> None:
    section = cfg.get("server", {})
    _check_required(section, REQUIRED_SERVER_KEYS, "server")

    tp = section.get("tensor_parallel_size", 1)
    pp = section.get("pipeline_parallel_size", 1)
    port = section.get("port", 8000)

    if not isinstance(tp, int) or tp < 1:
        _err(f"[server] tensor_parallel_size must be >= 1, got: {tp!r}")
    if not isinstance(pp, int) or pp < 1:
        _err(f"[server] pipeline_parallel_size must be >= 1, got: {pp!r}")
    if not isinstance(port, int) or not (1024 <= port <= 65535):
        _err(f"[server] port must be in [1024, 65535], got: {port!r}")

    env = section.get("env", {})
    visible = env.get("ASCEND_RT_VISIBLE_DEVICES", "")
    if visible:
        npu_ids = [x.strip() for x in str(visible).split(",") if x.strip()]
        if len(npu_ids) != tp:
            _warn(
                f"[server] ASCEND_RT_VISIBLE_DEVICES lists {len(npu_ids)} devices "
                f"but tensor_parallel_size={tp}"
            )


def validate_workload(cfg: dict) -> None:
    section = cfg.get("workload", {})
    _check_required(section, REQUIRED_WORKLOAD_KEYS, "workload")

    for key in ("input_tokens", "output_tokens"):
        val = section.get(key, 0)
        if not isinstance(val, int) or val <= 0:
            _err(f"[workload] {key} must be a positive integer, got: {val!r}")

    dataset = section.get("dataset", "")
    if dataset not in VALID_DATASETS:
        _warn(f"[workload] dataset '{dataset}' not in known values {VALID_DATASETS}")


def validate_benchmark(cfg: dict) -> None:
    section = cfg.get("benchmark", {})
    _check_required(section, REQUIRED_BENCHMARK_KEYS, "benchmark")

    pl_raw = section.get("parallel_levels", "")
    rpl_raw = section.get("requests_per_level", "")

    try:
        pl = [int(x) for x in str(pl_raw).split()]
    except ValueError:
        _err(f"[benchmark] parallel_levels must be space-separated integers, got: {pl_raw!r}")
        pl = []

    try:
        rpl = [int(x) for x in str(rpl_raw).split()]
    except ValueError:
        _err(f"[benchmark] requests_per_level must be space-separated integers, got: {rpl_raw!r}")
        rpl = []

    if pl and rpl and len(pl) != len(rpl):
        _err(
            f"[benchmark] parallel_levels ({len(pl)} items) and "
            f"requests_per_level ({len(rpl)} items) must have the same length"
        )

    for i, (p, r) in enumerate(zip(pl, rpl)):
        if p <= 0:
            _err(f"[benchmark] parallel_levels[{i}]={p} must be > 0")
        if r <= 0:
            _err(f"[benchmark] requests_per_level[{i}]={r} must be > 0")
        if r < p:
            _warn(
                f"[benchmark] requests_per_level[{i}]={r} < parallel_levels[{i}]={p}; "
                "each concurrency level may not complete a full batch"
            )

    warmup = section.get("warmup_requests", 0)
    if not isinstance(warmup, int) or warmup < 0:
        _err(f"[benchmark] warmup_requests must be >= 0, got: {warmup!r}")


def validate_sla(cfg: dict) -> None:
    section = cfg.get("sla", {})
    if not section:
        _warn("[sla] section missing; no SLA thresholds will be checked in the report")
        return

    for key in ("max_p99_ttft_ms", "min_success_rate", "min_output_token_throughput"):
        if key not in section:
            _warn(f"[sla] missing optional key '{key}'")

    success_rate = section.get("min_success_rate")
    if success_rate is not None and not (0.0 < success_rate <= 1.0):
        _err(f"[sla] min_success_rate must be in (0, 1], got: {success_rate!r}")


def validate_config(path: Path) -> bool:
    ERRORS.clear()
    WARNINGS.clear()

    try:
        with open(path) as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"YAML parse error in {path}: {e}", file=sys.stderr)
        return False

    if not isinstance(cfg, dict):
        print(f"ERROR: {path} must be a YAML mapping at the top level", file=sys.stderr)
        return False

    validate_model(cfg)
    validate_server(cfg)
    validate_workload(cfg)
    validate_benchmark(cfg)
    validate_sla(cfg)

    ok = True
    if WARNINGS:
        for w in WARNINGS:
            print(f"  WARN  {w}")
    if ERRORS:
        ok = False
        for e in ERRORS:
            print(f"  ERROR {e}")

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate ascend-vllm-serving-auto-benchmark YAML configs"
    )
    parser.add_argument("configs", nargs="+", type=Path, help="YAML config files to validate")
    args = parser.parse_args()

    all_ok = True
    for config_path in args.configs:
        print(f"\nValidating: {config_path}")
        result = validate_config(config_path)
        status = "OK" if result else "FAIL"
        print(f"  -> {status}")
        if not result:
            all_ok = False

    if all_ok:
        print("\nAll configs valid.")
        sys.exit(0)
    else:
        print("\nOne or more configs have errors.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
