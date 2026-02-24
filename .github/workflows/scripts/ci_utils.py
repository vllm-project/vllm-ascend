import logging
import os
import subprocess
import time
from dataclasses import dataclass

# Configure logger to output to stdout
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


@dataclass
class TestFile:
    name: str
    estimated_time: float = 60
    is_skipped: bool = False


def _write_timing_report(
    timing_records: list[dict],
    elapsed_total: float,
    report_path: str,
) -> None:
    """Write a Markdown timing report for all test files."""
    lines = [
        "# Test Timing Report",
        "",
        f"**Total elapsed: {elapsed_total:.2f}s**",
        "",
        "| Test File | Status | Elapsed (s) | Estimated (s) | Diff (s) |",
        "|-----------|--------|------------:|---------------:|---------:|",
    ]
    for record in timing_records:
        name = record["name"]
        status = "PASSED" if record["passed"] else "FAILED"
        elapsed = record["elapsed"]
        estimated = record["estimated"]
        diff = elapsed - estimated
        diff_str = f"+{diff:.0f}" if diff >= 0 else f"{diff:.0f}"
        lines.append(f"| `{name}` | {status} | {elapsed:.0f} | {estimated:.0f} | {diff_str} |")

    report_dir = os.path.dirname(os.path.abspath(report_path))
    os.makedirs(report_dir, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Timing report written to: {report_path}")


def run_e2e_files(
    files: list[TestFile],
    continue_on_error: bool = False,
    report_path: str | None = None,
):
    """
    Run a list of test files.

    Args:
        files: List of TestFile objects to run
        continue_on_error: If True, continue running remaining tests even if one fails.
                          If False, stop at first failure (default behavior for PR tests).
        report_path: If provided, write a Markdown timing report to this path.
    """
    tic = time.perf_counter()
    success = True
    passed_tests = []
    failed_tests = []
    timing_records = []

    for i, file in enumerate(files):
        filename, estimated_time = file.name, file.estimated_time

        full_path = os.path.join(os.getcwd(), filename)
        logger.info(f".\n.\n{Colors.HEADER}Begin ({i}/{len(files)}):{Colors.ENDC}\npytest -sv {full_path}\n.\n.\n")
        file_tic = time.perf_counter()

        process = subprocess.Popen(
            ["pytest", "-sv", "--durations=0", "--color=yes", full_path],
            stdout=None,
            stderr=None,
            env=os.environ,
        )
        process.wait()

        elapsed = time.perf_counter() - file_tic
        ret_code = process.returncode

        logger.info(
            f".\n.\n{Colors.HEADER}End ({i}/{len(files)}):{Colors.ENDC}\n{filename=}, \
                {elapsed=:.0f}, {estimated_time=}\n.\n.\n"
        )

        if ret_code == 0:
            passed_tests.append(filename)
            timing_records.append({"name": filename, "passed": True, "elapsed": elapsed, "estimated": estimated_time})
        else:
            logger.info(f"\n{Colors.FAIL}✗ FAILED: {filename} returned exit code {ret_code}{Colors.ENDC}\n")
            failed_tests.append((filename, f"exit code {ret_code}"))
            timing_records.append({"name": filename, "passed": False, "elapsed": elapsed, "estimated": estimated_time})
            success = False
            if not continue_on_error:
                break

    elapsed_total = time.perf_counter() - tic

    if success:
        logger.info(f"{Colors.OKGREEN}Success. Time elapsed: {elapsed_total:.2f}s{Colors.ENDC}")
    else:
        logger.info(f"{Colors.FAIL}Fail. Time elapsed: {elapsed_total:.2f}s{Colors.ENDC}")

    # Print summary
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Test Summary: {Colors.OKGREEN}{len(passed_tests)}/{len(files)} passed{Colors.ENDC}")
    logger.info(f"{'=' * 60}")
    if passed_tests:
        logger.info(f"{Colors.OKGREEN}✓ PASSED:{Colors.ENDC}")
        for test in passed_tests:
            logger.info(f"  {test}")
    if failed_tests:
        logger.info(f"\n{Colors.FAIL}✗ FAILED:{Colors.ENDC}")
        for test, reason in failed_tests:
            logger.info(f"  {test} ({reason})")
    logger.info(f"{'=' * 60}\n")

    if report_path is not None:
        _write_timing_report(timing_records, elapsed_total, report_path)

    return (0 if success else -1), timing_records
