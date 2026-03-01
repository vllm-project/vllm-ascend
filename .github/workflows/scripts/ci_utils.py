import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


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


def _write_timing_report(records: list[TestRecord], total_elapsed: float, path: Path) -> None:
    """Write a Markdown timing report."""
    lines = [
        "# Test Timing Report",
        "",
        f"**Total elapsed: {total_elapsed:.2f}s**",
        "",
        "| Test File | Status | Elapsed (s) | Estimated (s) | Diff (s) |",
        "|-----------|--------|------------:|---------------:|---------:|",
    ]
    for r in records:
        status = "PASSED" if r.passed else "FAILED"
        diff = r.elapsed - r.estimated
        diff_str = f"+{diff:.0f}" if diff >= 0 else f"{diff:.0f}"
        lines.append(f"| `{r.name}` | {status} | {r.elapsed:.0f} | {r.estimated:.0f} | {diff_str} |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"Timing report written to: {path}")


def run_tests(
    files: list[TestFile],
    continue_on_error: bool = False,
    report_path: Path | None = None,
) -> tuple[int, list[TestRecord]]:
    """
    Run each TestFile with pytest and collect timing results.

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
        print(f"\n{'.' * 60}", flush=True)
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
        print(
            f"{color}[{i + 1}/{len(files)}] {status}  {test.name}  ({elapsed:.0f}s){_Color.RESET}",
            flush=True,
        )

        if not passed:
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

    if report_path is not None:
        _write_timing_report(records, total_elapsed, report_path)

    return (0 if all_passed else -1), records
