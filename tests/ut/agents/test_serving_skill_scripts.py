import importlib.util
import json
import subprocess
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_module(module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dump_json_line(payload: dict) -> str:
    return json.dumps(payload, separators=(",", ":"))


def test_generate_report_parses_real_evalscope_v181_output() -> None:
    """Parse a real-shaped evalscope perf v1.8.1 output tree: nested under
    <timestamp>/<name>/parallel_<P>_number_<N>/ with paired
    benchmark_summary.json (flat dict) + benchmark_percentile.json (list)."""
    module = load_module(
        "generate_report_module",
        ".agents/skills/ascend-vllm-serving-auto-benchmark/scripts/generate_report.py",
    )

    fixture_dir = REPO_ROOT / "tests/ut/agents/fixtures/evalscope_v181/outputs/20250423_002442/TestModel"

    parsed = module._parse_evalscope_output_dir(fixture_dir)
    assert len(parsed) == 3
    assert [r.concurrency for r in parsed] == [1, 4, 8]

    c1 = parsed[0]
    # TTFT/TPOT are already ms in v1.8.1; read straight from summary.
    assert c1.ttft_avg_ms == 64.2
    assert c1.tpot_avg_ms == 10.27
    assert c1.output_token_throughput == 96.38
    # Avg Latency is in SECONDS in v1.8.1; parser must convert to ms.
    assert c1.latency_avg_ms == 5312.0
    # Percentiles come from benchmark_percentile.json (p99 row).
    assert c1.ttft_p99_ms == 68.5
    assert c1.tpot_p99_ms == 10.44
    assert c1.latency_p99_ms == 5400.0
    assert c1.success_requests == 20
    assert c1.failed_requests == 0


def test_generate_report_jsonl_fallback_uses_real_fields(tmp_path: Path) -> None:
    """When no parallel_* subdirs exist, fall back to JSONL where each line is
    a flat v1.8.1 summary dict (no paired percentile file)."""
    module = load_module(
        "generate_report_module",
        ".agents/skills/ascend-vllm-serving-auto-benchmark/scripts/generate_report.py",
    )

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    write_jsonl(
        results_dir / "results.jsonl",
        [
            dump_json_line(
                {
                    "Concurrency": 1,
                    "Total Requests": 10,
                    "Success Requests": 10,
                    "Output Throughput (tok/s)": 128.0,
                    "Avg Latency (s)": 0.5,
                    "TTFT (ms)": 100.0,
                    "TPOT (ms)": 20.0,
                }
            )
        ],
    )

    parsed = module._parse_evalscope_output_dir(results_dir)
    assert len(parsed) == 1
    assert parsed[0].concurrency == 1
    assert parsed[0].ttft_avg_ms == 100.0
    assert parsed[0].tpot_avg_ms == 20.0
    assert parsed[0].output_token_throughput == 128.0
    assert parsed[0].latency_avg_ms == 500.0  # 0.5 s -> 500 ms


def test_generate_tuning_report_preserves_failure_verdicts_and_metric_direction(
    tmp_path: Path,
) -> None:
    module = load_module(
        "generate_tuning_report_module",
        ".agents/skills/ascend-vllm-serving-tune-loop/scripts/generate_tuning_report.py",
    )

    work_dir = tmp_path / "tuning"
    baseline_dir = work_dir / "baseline"
    iter_01_dir = work_dir / "iter_01"
    baseline_dir.mkdir(parents=True)
    iter_01_dir.mkdir()

    write_jsonl(
        baseline_dir / "results.jsonl",
        [
            dump_json_line(
                {
                    "Concurrency": 1,
                    "TTFT (ms)": 100.0,
                    "TPOT (ms)": 30.0,
                    "Avg Latency (s)": 0.50,
                    "Output Throughput (tok/s)": 100.0,
                }
            ),
            dump_json_line(
                {
                    "Concurrency": 4,
                    "TTFT (ms)": 200.0,
                    "TPOT (ms)": 40.0,
                    "Avg Latency (s)": 0.80,
                    "Output Throughput (tok/s)": 250.0,
                }
            ),
        ],
    )
    write_jsonl(
        iter_01_dir / "results.jsonl",
        [
            dump_json_line(
                {
                    "Concurrency": 1,
                    "TTFT (ms)": 80.0,
                    "TPOT (ms)": 30.0,
                    "Avg Latency (s)": 0.45,
                    "Output Throughput (tok/s)": 110.0,
                }
            ),
            dump_json_line(
                {
                    "Concurrency": 4,
                    "TTFT (ms)": 180.0,
                    "TPOT (ms)": 40.0,
                    "Avg Latency (s)": 0.75,
                    "Output Throughput (tok/s)": 260.0,
                }
            ),
        ],
    )

    (work_dir / "ledger.md").write_text(
        textwrap.dedent(
            """
            ## Iteration 1 — 2.1 balance_scheduling

            **Hypothesis**: better low-concurrency scheduling
            **Change**: `--additional-config {"enable_balance_scheduling": true}`
            **Verdict**: WIN
            **Carry forward**: YES
            **Notes**: baseline improved

            ## Iteration 2 — 3.7 xlite_graph

            **Hypothesis**: graph replay can help
            **Change**: `--compilation-config {"xlite_graph": true}`
            **Verdict**: STARTUP_FAIL
            **Failure stage**: server_startup
            **Reason**: missing xlite package
            **Evidence**:
            - Exit code: 1
            - Log excerpt: ModuleNotFoundError: No module named 'xlite'
            **Carry forward**: NO
            **Notes**: skip until dependency is installed
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    baseline_metrics = module._load_json_metrics(baseline_dir)
    iterations = module._parse_ledger(work_dir / "ledger.md")
    for iteration in iterations:
        iter_dir = work_dir / f"iter_{iteration.iteration:02d}"
        metrics = module._load_json_metrics(iter_dir)
        if metrics:
            iteration.metrics = metrics

    run = module.TuningRun(
        model_name="Qwen3-32B",
        target_metric="ttft_avg",
        baseline_metrics=baseline_metrics,
        best_metrics=iterations[0].metrics,
        best_config=iterations[0].change_desc,
        iterations=iterations,
        winning_levers=[iterations[0].lever_name],
        failed_levers=[iterations[1].lever_name],
        timestamp="2026-07-12 12:00:00",
    )

    report = module.render_report(run)
    assert "| STARTUP_FAIL | `1` |" in report
    assert "`+20.0%`" in report
    assert "`3.7 xlite_graph` | `STARTUP_FAIL` | `server_startup`" in report


def test_auto_resume_wrapper_exits_on_sigint_code(tmp_path: Path) -> None:
    work_dir = tmp_path / "run"
    work_dir.mkdir()
    wrapper = REPO_ROOT / ".agents/skills/ascend-vllm-serving-tune-loop/scripts/auto_resume_wrapper.sh"

    result = subprocess.run(
        [
            "bash",
            str(wrapper),
            "--work-dir",
            str(work_dir),
            "--",
            "bash",
            "-lc",
            "exit 130",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 130
    assert "Wrapper exits without retry" in result.stdout


def _run_benchmark_dry_run(tmp_path: Path, *extra_args: str) -> str:
    """Run run_benchmark.sh in dry-run and return the generated server_cmd.txt."""
    output_dir = tmp_path / "out"
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / ".agents/skills/ascend-vllm-serving-auto-benchmark/scripts/run_benchmark.sh"),
            "--dry-run",
            "--model-path",
            "/tmp/fake-model",
            "--model-name",
            "FakeModel",
            "--output-dir",
            str(output_dir),
            *extra_args,
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    return (output_dir / "server_cmd.txt").read_text(encoding="utf-8")


def test_no_aclgraph_maps_to_enforce_eager(tmp_path: Path) -> None:
    """--no-aclgraph must disable ACLGraph via the supported vLLM flag
    --enforce-eager (the legacy VLLM_ASCEND_ENABLE_ACLGRAPH env was a no-op:
    never defined in vllm_ascend/envs.py, never read by runtime)."""
    server_cmd = _run_benchmark_dry_run(tmp_path, "--no-aclgraph")
    assert "--enforce-eager" in server_cmd

    # Without --no-aclgraph the graph stays enabled (no --enforce-eager).
    baseline_cmd = _run_benchmark_dry_run(tmp_path)
    assert "--enforce-eager" not in baseline_cmd
