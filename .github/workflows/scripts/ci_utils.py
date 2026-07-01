import os
import subprocess
import time
from dataclasses import dataclass


class _Color:
    HEADER = "\033[95m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"


@dataclass
class TestFile:
    name: str
    estimated_time: float = 60
    is_skipped: bool = False


@dataclass
class TestRecord:
    name: str
    passed: bool
    elapsed: float
    estimated: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "elapsed": self.elapsed,
            "estimated": self.estimated,
        }


def _escape_github_actions_value(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _print_github_actions_group_start(title: str) -> None:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::group::{_escape_github_actions_value(title)}", flush=True)


def _print_github_actions_group_end() -> None:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print("::endgroup::", flush=True)


def _print_github_actions_annotation(annotation: str, message: str) -> None:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::{annotation}::{_escape_github_actions_value(message)}", flush=True)


def run_tests(
    files: list[TestFile],
    continue_on_error: bool = False,
) -> tuple[int, list[TestRecord]]:
    """
    Run each TestFile with pytest and collect timing results.

    NOTE:
        The emitted START / PASSED / FAILED log lines are parsed by
        ci_log_summary.py to recover per-test invocation boundaries.
        Keep this output format stable, or update the corresponding
        regexes in those CI log summarizers together.

    Args:
        files: Tests to run (skipped entries should already be filtered out).
        continue_on_error: If True, keep running after a failure.
        report_path: If provided, write a Markdown timing report here.

    Returns:
        (exit_code, records) — exit_code is 0 on full success, -1 otherwise.
    """
    records: list[TestRecord] = []
    all_passed = True
    total_start = time.perf_counter()

    for i, test in enumerate(files):
        _print_github_actions_group_start(f"[{i + 1}/{len(files)}] {test.name}")
        try:
            print(f"\n{'.' * 60}", flush=True)
            # NOTE: ci_log_summary.py depend on this
            # START line format when splitting suite-level logs into test runs.
            print(
                f"{_Color.HEADER}[{i + 1}/{len(files)}] START  {test.name}{_Color.RESET}",
                flush=True,
            )

            start = time.perf_counter()
            result = subprocess.run(["pytest", "-sv", "--durations=0", "--color=yes", test.name])
            elapsed = time.perf_counter() - start
            passed = result.returncode == 0

            records.append(TestRecord(name=test.name, passed=passed, elapsed=elapsed, estimated=test.estimated_time))

            color = _Color.GREEN if passed else _Color.RED
            status = "PASSED" if passed else f"FAILED (exit code {result.returncode})"
            # NOTE: ci_log_summary.py depend on this
            # PASSED / FAILED (exit code X) line format for suite end detection.
            print(
                f"{color}[{i + 1}/{len(files)}] {status}  {test.name}  ({elapsed:.0f}s){_Color.RESET}",
                flush=True,
            )
        finally:
            _print_github_actions_group_end()

        if not passed:
            _print_github_actions_annotation(
                "error",
                f"[{i + 1}/{len(files)}] FAILED {test.name}. "
                "Please go to the Summary section to quickly review the error overview, "
                "or expand the logs to view the error details.",
            )
            all_passed = False
            if not continue_on_error:
                break

    total_elapsed = time.perf_counter() - total_start
    passed_count = sum(1 for r in records if r.passed)

    print(f"\n{'=' * 60}")
    color = _Color.GREEN if all_passed else _Color.RED
    print(f"{color}Summary: {passed_count}/{len(files)} passed  ({total_elapsed:.2f}s total){_Color.RESET}")
    print("=" * 60)
    for r in records:
        icon = f"{_Color.GREEN}✓{_Color.RESET}" if r.passed else f"{_Color.RED}✗{_Color.RESET}"
        print(f"  {icon} {r.name}  ({r.elapsed:.0f}s)")
    print(flush=True)

    return (0 if all_passed else -1), records
