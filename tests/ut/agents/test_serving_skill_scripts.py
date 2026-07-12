import importlib.util
import subprocess
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def load_module(module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_generate_report_parses_json_results(tmp_path: Path) -> None:
    module = load_module(
        "generate_report_module",
        ".agents/skills/ascend-vllm-serving-auto-benchmark/scripts/generate_report.py",
    )

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "results.json").write_text(
        """
        [
          {
            "concurrency": 1,
            "stats": {
              "total_requests": 10,
              "success_requests": 10,
              "success_rate": 1.0,
              "request_throughput": 2.0,
              "output_token_throughput": 128.0,
              "latency": {"mean": 0.5, "p50": 0.4, "p90": 0.6, "p99": 0.8},
              "ttft": {"mean": 0.1, "p50": 0.09, "p90": 0.12, "p99": 0.15},
              "tpot": {"mean": 0.02, "p50": 0.02, "p90": 0.03, "p99": 0.04}
            }
          }
        ]
        """,
        encoding="utf-8",
    )

    parsed = module._parse_evalscope_output_dir(results_dir)
    assert len(parsed) == 1
    assert parsed[0].concurrency == 1
    assert parsed[0].ttft_avg_ms == 100.0
    assert parsed[0].tpot_avg_ms == 20.0
    assert parsed[0].output_token_throughput == 128.0


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
            '{"concurrency": 1, "stats": {"ttft": {"mean": 0.10}, "tpot": {"mean": 0.03}, "latency": {"mean": 0.50}, "output_token_throughput": 100.0}}',
            '{"concurrency": 4, "stats": {"ttft": {"mean": 0.20}, "tpot": {"mean": 0.04}, "latency": {"mean": 0.80}, "output_token_throughput": 250.0}}',
        ],
    )
    write_jsonl(
        iter_01_dir / "results.jsonl",
        [
            '{"concurrency": 1, "stats": {"ttft": {"mean": 0.08}, "tpot": {"mean": 0.03}, "latency": {"mean": 0.45}, "output_token_throughput": 110.0}}',
            '{"concurrency": 4, "stats": {"ttft": {"mean": 0.18}, "tpot": {"mean": 0.04}, "latency": {"mean": 0.75}, "output_token_throughput": 260.0}}',
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
