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


def run_e2e_files(
    files: list[TestFile],
    continue_on_error: bool = False,
):
    """
    Run a list of test files.

    Args:
        files: List of TestFile objects to run
        continue_on_error: If True, continue running remaining tests even if one fails.
                          If False, stop at first failure (default behavior for PR tests).
    """
    tic = time.perf_counter()
    success = True
    passed_tests = []
    failed_tests = []

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
        else:
            logger.info(f"\n{Colors.FAIL}✗ FAILED: {filename} returned exit code {ret_code}{Colors.ENDC}\n")
            failed_tests.append((filename, f"exit code {ret_code}"))
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

    return 0 if success else -1
