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
    error_output: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "elapsed": self.elapsed,
            "estimated": self.estimated,
        }


def run_tests(
    files: list[TestFile],
    continue_on_error: bool = False,
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
        env = os.environ.copy()
        proc = subprocess.Popen(
            ["pytest", "-sv", "--durations=0", "--color=yes", "--tb=short", test.name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        captured_lines: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            captured_lines.append(line)
        proc.wait()
        elapsed = time.perf_counter() - start
        passed = proc.returncode == 0
        error_output = "" if passed else "".join(captured_lines)

        records.append(
            TestRecord(
                name=test.name, passed=passed, elapsed=elapsed, estimated=test.estimated_time, error_output=error_output
            )
        )

        color = _Color.GREEN if passed else _Color.RED
        status = "PASSED" if passed else f"FAILED (exit code {proc.returncode})"
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
    failed = [r for r in records if not r.passed and r.error_output]
    if failed:
        print(f"\n{_Color.RED}{'=' * 60}")
        print(f"FAILED TEST OUTPUT ({len(failed)} failure(s)){_Color.RESET}")
        for r in failed:
            print(f"\n{_Color.RED}{'─' * 60}")
            print(f"FAILED: {r.name}{_Color.RESET}")
            print(r.error_output, end="")
        print(f"{_Color.RED}{'=' * 60}{_Color.RESET}")
    print(flush=True)

    return (0 if all_passed else -1), records
