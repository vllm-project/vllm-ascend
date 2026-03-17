#!/usr/bin/env python3
"""
Extract failed test cases and root-cause errors from a GitHub Actions workflow run.

This script queries the GitHub API via `gh` CLI to find failed jobs, download
their logs (in chunks), parse out failed test identifiers, extract true root-cause
exceptions, deduplicate them, and output a structured JSON report.

Usage:
  # With a specific run ID:
  python3 scripts/extract_and_analyze.py --run-id 22490469887

  # Auto-find latest failed schedule_test_vllm_main run:
  python3 scripts/extract_and_analyze.py

  # Output to file:
  python3 scripts/extract_and_analyze.py --run-id 22490469887 -o analysis.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from typing import Any

REPO = "vllm-project/vllm-ascend"
WORKFLOW = "schedule_test_vllm_main.yaml"

# ─── Regex patterns ───────────────────────────────────────────────────────────

# Pytest / run_suite.py failure markers
_FAILED_INLINE_RE = re.compile(
    r"FAILED:\s+(tests/\S+\.py(?:::\S+)?)\s+.*(?:exit code|returned)",
)
_FAILED_SUMMARY_RE = re.compile(
    r"^\s+(tests/\S+\.py(?:::\S+)?)\s+\(exit code",
    re.MULTILINE,
)
_FAILED_PYTEST_RE = re.compile(
    r"FAILED\s+(tests/\S+\.py::\S+)",
)

# Core Python exceptions (the ones we care about for root-cause)
_CORE_ERROR_RE = re.compile(
    r"(TypeError|AttributeError|ImportError|ModuleNotFoundError"
    r"|KeyError|NotImplementedError|ValueError|OSError):\s*(.+)",
)

# Wrapper errors to skip (these hide the real cause)
_WRAPPER_PATTERNS = [
    "Engine core initialization failed",
    "Worker failed with error",
    "subprocess.CalledProcessError",
    "SystemExit",
]

# Downstream effect errors — these are symptoms, not root causes.
# E.g., KeyError: 'choices' happens when the engine crashes and returns an
# error response instead of a valid completion.
_DOWNSTREAM_PATTERNS = [
    r"KeyError:\s*'choices'",
    r"KeyError:\s*'message'",
    r"AssertionError:\s*assert.*response",
]

# Environment flake patterns (no code fix needed)
_ENV_FLAKE_PATTERNS = [
    r"OSError:.*Stale file handle",
    r"ConnectionResetError",
    r"filelock.*Lock",
    r"ConnectionRefusedError",
    r"TimeoutError",
    r"torch\.cuda\.OutOfMemoryError",
    r"OSError:.*No space left on device",
]

# Timestamp prefix in GitHub Actions logs
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z\s*")

# ANSI escape codes
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# vLLM / Ray generic logger prefixes like (EngineCore_DP0 pid=30130) ERROR 03-03 15:41:37 [core.py:1100]
_VLLM_LOG_PREFIX_RE = re.compile(
    r"^(?:\[.*?\]\s*:\s*)?(?:\(.*?\)\s*)*[A-Z]+\s+\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+\[.*?\]\s*"
)
# Profiler/modelscope prefixes like 2026-03-03 15:42:30,706 - 49378 - vllmProfiler - INFO -
_PROFILER_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d+\s+-\s+\d+\s+-\s+\S+\s+-\s+[A-Z]+\s+-\s*")

# vLLM version from pip/git describe: "vLLM 0.1.dev1+g<SHORT_SHA>.empty"
_VLLM_VERSION_RE = re.compile(r"vLLM\s+\S*\+g([0-9a-f]{7,12})\b")

# ─── gh CLI helpers ───────────────────────────────────────────────────────────


def gh_api_json(endpoint: str, **params) -> Any:
    """Call `gh api` and return parsed JSON."""
    url = endpoint
    if params:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{endpoint}?{qs}"
    try:
        r = subprocess.run(
            ["gh", "api", url],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        print("ERROR: 'gh' CLI not found. Install it or run 'gh auth login'.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: gh api {url} failed: {e.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return json.loads(r.stdout)


def gh_api_raw(endpoint: str) -> str:
    """Call `gh api` and return raw text."""
    try:
        r = subprocess.run(
            ["gh", "api", endpoint],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"WARNING: Failed to download {endpoint}: {e.stderr.strip()}", file=sys.stderr)
        return ""
    return r.stdout


# ─── Core logic ───────────────────────────────────────────────────────────────


def find_latest_failed_run() -> dict | None:
    """Find the most recent failed run of schedule_test_vllm_main."""
    data = gh_api_json(
        f"/repos/{REPO}/actions/workflows/{WORKFLOW}/runs",
        status="failure",
        per_page="5",
    )
    runs = data.get("workflow_runs", [])
    return runs[0] if runs else None


def get_failed_jobs(run_id: int) -> list[dict]:
    """List all failed jobs in a workflow run."""
    data = gh_api_json(
        f"/repos/{REPO}/actions/runs/{run_id}/jobs",
        per_page="100",
    )
    return [j for j in data.get("jobs", []) if j.get("conclusion") == "failure"]


def clean_line(line: str) -> str:
    """Strip timestamp prefix, ANSI codes, and routine log wrappers from a log line."""
    line = _TIMESTAMP_RE.sub("", line)
    line = _ANSI_RE.sub("", line)
    line = _VLLM_LOG_PREFIX_RE.sub("", line)
    line = _PROFILER_PREFIX_RE.sub("", line)
    return line


def extract_failed_tests(log_text: str) -> list[str]:
    """Extract failed test file paths/identifiers from a job log."""
    failed = set()
    for m in _FAILED_INLINE_RE.finditer(log_text):
        failed.add(m.group(1))
    for m in _FAILED_SUMMARY_RE.finditer(log_text):
        failed.add(m.group(1))
    for m in _FAILED_PYTEST_RE.finditer(log_text):
        failed.add(m.group(1))
    return sorted(failed)


def extract_test_sections(log_text: str) -> list[dict]:
    """Extract test execution sections with their line ranges.

    Returns a list of dicts with test_name, start_line, and end_line.
    This enables temporal correlation: errors are mapped to the test
    that was running when the error occurred.
    """
    lines = log_text.splitlines()
    sections = []
    current_test = None
    current_start = None

    # Pattern for pytest execution markers - match full path
    pytest_start_re = re.compile(r"pytest\s+-sv\s+[/\w\-]+/(tests/\S+)")

    for i, raw_line in enumerate(lines):
        line = clean_line(raw_line)

        # Detect test start: "pytest -sv tests/..."
        m = pytest_start_re.search(line)
        if m:
            # Save previous section if exists
            if current_test and current_start is not None:
                sections.append(
                    {
                        "test_name": current_test,
                        "start_line": current_start,
                        "end_line": i,
                    }
                )
            current_test = m.group(1)
            current_start = i
            continue

        # Detect FAILED markers that might indicate test end
        # These appear inline: "✗ FAILED: tests/... returned exit code 1"
        m_failed = _FAILED_INLINE_RE.search(line)
        if m_failed and current_test:
            test_name = m_failed.group(1)
            # If this FAILED matches our current test (or is closely related)
            if test_name.startswith(current_test.split("::")[0]):
                sections.append(
                    {
                        "test_name": current_test,
                        "start_line": current_start,
                        "end_line": i,
                    }
                )
                current_test = None
                current_start = None

    # Handle last section
    if current_test and current_start is not None:
        sections.append(
            {
                "test_name": current_test,
                "start_line": current_start,
                "end_line": len(lines),
            }
        )

    return sections


def extract_error_to_test_mapping(log_text: str) -> dict[str, list[str]]:
    """Extract direct error-to-test mappings from FAILED pytest lines.

    Pytest outputs lines like:
      FAILED tests/xxx.py::test_name - AttributeError: message
      FAILED tests/xxx.py - TypeError: message

    For errors embedded in assertion messages (like OSError in assertion text),
    we also extract from the assertion content.

    Returns: {error_signature: [test1, test2, ...]}
    """
    lines = log_text.splitlines()

    # Pattern: FAILED tests/... - ErrorType: message
    failed_pytest_re = re.compile(
        r"FAILED\s+(tests/\S+?)\s+-\s+(TypeError|AttributeError|ImportError|ModuleNotFoundError|KeyError|NotImplementedError|ValueError|OSError|RuntimeError|AssertionError):\s*(.+)"
    )

    error_to_tests = defaultdict(set)

    for raw_line in lines:
        line = clean_line(raw_line)
        m = failed_pytest_re.search(line)
        if m:
            test_name = m.group(1)
            error_type = m.group(2)
            error_msg = m.group(3).strip()

            # Extract base test name (without ::test_name[param] part for file-level mapping)
            base_test = test_name.split("::")[0]

            # Normalize error message for dedup (similar to error extraction)
            normalized = re.sub(r"pid=\d+", "pid=X", error_msg)
            normalized = re.sub(r"0x[0-9a-f]+", "0xXXX", normalized)
            normalized = re.sub(r"\[Errno \d+\]", "[Errno X]", normalized)
            sig = f"{error_type}:{normalized}"

            error_to_tests[sig].add(base_test)

            # Also check for embedded errors in assertion messages
            # E.g., "assert 'X' in 'OSError: [Errno 116] Stale file handle...'"
            if error_type == "AssertionError":
                # Look for OSError in the assertion text
                os_err_m = re.search(r"OSError:\s*\[Errno\s+\d+\]\s*(\S+(?:\s+\S+)?)", error_msg)
                if os_err_m:
                    os_err_msg = os_err_m.group(1)
                    os_normalized = re.sub(r"\[Errno \d+\]", "[Errno X]", f"[Errno X] {os_err_msg}")
                    os_sig = f"OSError:{os_normalized}"
                    error_to_tests[os_sig].add(base_test)

    # Convert sets to sorted lists
    return {sig: sorted(list(tests)) for sig, tests in error_to_tests.items()}


def extract_bad_commit(log_text: str) -> str | None:
    """Extract the vLLM commit hash from a job log.

    Strategy: Look for the vLLM version string printed by vllmProfiler,
    e.g. 'vLLM 0.1.dev1+g6d4f9d3ad.empty' -> '6d4f9d3ad'
    Then look up the full SHA via the GitHub API.
    """
    m = _VLLM_VERSION_RE.search(log_text)
    if m:
        short_sha = m.group(1)
        # Try to resolve to full SHA
        try:
            data = gh_api_json(f"/repos/vllm-project/vllm/commits/{short_sha}")
            return data.get("sha")
        except SystemExit:
            return short_sha
    return None


def extract_root_cause_errors(log_text: str) -> list[dict]:
    """Extract root-cause exceptions from a job log.

    Skips wrapper errors and downstream effects, deduplicates by normalized
    error signature, and detects environment flakes even when embedded in
    assertion messages.

    Returns a list of error dicts, each with line_number for temporal correlation.
    """
    errors = []
    lines = log_text.splitlines()

    # For deduplication within this log: signature -> list of (line_number, affected_tests)
    # We keep track of all occurrences, not just the first one
    sig_to_entries = {}

    # First pass: detect environment flakes embedded in assertion messages.
    # E.g., the full assertion output text may contain "OSError: ... Stale file handle"
    # even though it's not on its own line as a standalone exception.
    for i, raw_line in enumerate(lines):
        line = clean_line(raw_line)
        for pattern in _ENV_FLAKE_PATTERNS:
            if re.search(pattern, line):
                m_flake = re.search(
                    r"(OSError|ConnectionResetError|TimeoutError|ConnectionRefusedError):\s*(.+?)(?:\\n|$)", line
                )
                if m_flake:
                    error_type = m_flake.group(1)
                    error_msg = m_flake.group(2).strip()
                    # Clean trailing escape chars and quotes from embedded strings
                    error_msg = re.sub(r"(?:\\n|\\r|[\\'\"\n\r])+$", "", error_msg).strip()
                    error_msg = re.sub(r"\\n.*$", "", error_msg).strip()
                    # Normalize for dedup
                    normalized_msg = re.sub(r"\[Errno \d+\]", "[Errno X]", error_msg)
                    signature = f"{error_type}:{normalized_msg}"

                    if signature not in sig_to_entries:
                        sig_to_entries[signature] = {
                            "error_type": error_type,
                            "error_message": error_msg,
                            "category": "Environment Flake",
                            "line_numbers": [],
                            "contexts": [],
                        }

                    sig_to_entries[signature]["line_numbers"].append(i + 1)
                    sig_to_entries[signature]["contexts"].append(
                        [clean_line(lines[j]) for j in range(max(0, i - 2), min(len(lines), i + 3))]
                    )
                break  # Only match one flake pattern per line

    # Second pass: detect core exceptions on their own lines.
    for i, raw_line in enumerate(lines):
        line = clean_line(raw_line)

        # Skip wrapper errors
        if any(wp in line for wp in _WRAPPER_PATTERNS):
            continue

        m = _CORE_ERROR_RE.search(line)
        if not m:
            continue

        error_type = m.group(1)
        error_msg = m.group(2).strip()

        # Skip downstream effects (symptoms, not root causes)
        full_error = f"{error_type}: {error_msg}"
        if any(re.search(p, full_error) for p in _DOWNSTREAM_PATTERNS):
            continue

        # Classify as environment flake or code bug
        is_env_flake = any(re.search(p, full_error) for p in _ENV_FLAKE_PATTERNS)

        # Truncate embedded stack traces/outputs after newlines
        error_msg = re.sub(r"(\\n|\n).*$", "", error_msg)
        error_msg = re.sub(r"\\['\"]", "'", error_msg)
        error_msg = error_msg.strip()

        # Normalize for dedup: strip PIDs, timestamps, addresses, errno numbers,
        # trailing escape sequences and quotes
        normalized = re.sub(r"pid=\d+", "pid=X", error_msg)
        normalized = re.sub(r"0x[0-9a-f]+", "0xXXX", normalized)
        normalized = re.sub(r"\d{4}-\d{2}-\d{2}", "YYYY-MM-DD", normalized)
        normalized = re.sub(r"\[Errno \d+\]", "[Errno X]", normalized)
        normalized = re.sub(r"""(?:\\[nr]|['"])+$""", "", normalized).strip()
        signature = f"{error_type}:{normalized}"

        if signature not in sig_to_entries:
            sig_to_entries[signature] = {
                "error_type": error_type,
                "error_message": error_msg,
                "category": "Environment Flake" if is_env_flake else "Code Bug",
                "line_numbers": [],
                "contexts": [],
            }

        sig_to_entries[signature]["line_numbers"].append(i + 1)

        # Smart Traceback context extraction: looking back for the start of the exception block
        start_idx = max(0, i - 15)  # Default fallback depth

        # When looking for boundary, scan up to 100 lines for the test name header
        for j in range(i, max(-1, i - 100), -1):
            cleaned_line = clean_line(lines[j])
            if "Traceback (most recent call last):" in cleaned_line:
                start_idx = j
                break
            # Pytest test name boundary looks like: _______________________ test_name _______________________
            elif cleaned_line.startswith("__") and " test_" in cleaned_line and cleaned_line.endswith("__"):
                start_idx = max(0, j)
                break
            # Fallback Pytest traceback boundary looks like: _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
            elif cleaned_line.startswith("_ _ _ _ _") and len(cleaned_line.strip()) > 10:
                # include the line right above `_ _ _ _ _` because it's usually the file path
                start_idx = max(start_idx, j - 1)
                # Don't break here, keep searching upwards for the `____ test_name ____` header

        ctx_start = start_idx
        ctx_end = min(len(lines), i + 1)
        sig_to_entries[signature]["contexts"].append([clean_line(lines[j]) for j in range(ctx_start, ctx_end)])

    # Convert to list format, using the first occurrence's context
    # but keeping all line_numbers for correlation
    for sig, entry in sig_to_entries.items():
        errors.append(
            {
                "error_type": entry["error_type"],
                "error_message": entry["error_message"],
                "category": entry["category"],
                "context": entry["contexts"][0],  # Use first context
                "line_number": entry["line_numbers"][0],  # Primary line number
            }
        )

    return errors


def get_good_commit() -> str | None:
    """Extract the good (pinned) vLLM commit from workflow YAML files.

    Searches pr_test_full.yaml and pr_test_light.yaml for the vllm_version
    matrix field containing a commit hash.
    """
    commit_re = re.compile(r"^[0-9a-f]{7,40}$")
    yaml_files = [
        ".github/workflows/pr_test_full.yaml",
        ".github/workflows/pr_test_light.yaml",
    ]

    for yaml_rel in yaml_files:
        # Strategy 1: read from disk (current checkout)
        try:
            repo_root = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            import os

            disk_path = os.path.join(repo_root, yaml_rel)
            if os.path.exists(disk_path):
                with open(disk_path) as f:
                    content = f.read()
                m = re.search(r"vllm_version:\s*\[([^\]]+)\]", content)
                if m:
                    entries = [e.strip().strip("'\"") for e in m.group(1).split(",")]
                    for entry in entries:
                        if commit_re.match(entry):
                            return entry
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            pass

        # Strategy 2: read from origin/main via git show
        try:
            r = subprocess.run(
                ["git", "show", f"origin/main:{yaml_rel}"],
                capture_output=True,
                text=True,
                check=True,
            )
            m = re.search(r"vllm_version:\s*\[([^\]]+)\]", r.stdout)
            if m:
                entries = [e.strip().strip("'\"") for e in m.group(1).split(",")]
                for entry in entries:
                    if commit_re.match(entry):
                        return entry
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    return None


def process_run(run_id: int) -> dict:
    """Full pipeline: get jobs, download logs, extract failures and errors."""
    # Get run info
    run_info = gh_api_json(f"/repos/{REPO}/actions/runs/{run_id}")

    # Get all jobs (not just failed) for summary
    all_jobs_data = gh_api_json(
        f"/repos/{REPO}/actions/runs/{run_id}/jobs",
        per_page="100",
    )
    all_jobs = all_jobs_data.get("jobs", [])
    failed_jobs = [j for j in all_jobs if j.get("conclusion") == "failure"]

    # Get good commit
    good_commit = get_good_commit()

    bad_commit = None
    all_failed_tests = []
    all_errors = []
    job_results = []

    for job in failed_jobs:
        job_id = job["id"]
        job_name = job["name"]
        print(f"  Downloading log for {job_name} ({job_id})...", file=sys.stderr)

        log_text = gh_api_raw(f"/repos/{REPO}/actions/jobs/{job_id}/logs")
        if not log_text:
            job_results.append(
                {
                    "job_id": job_id,
                    "job_name": job_name,
                    "error": "Failed to download log",
                }
            )
            continue

        # Extract bad commit (only need once)
        if bad_commit is None:
            bad_commit = extract_bad_commit(log_text)

        # Extract failed tests
        tests = extract_failed_tests(log_text)
        all_failed_tests.extend(tests)

        # Extract direct error-to-test mapping from FAILED pytest lines
        error_to_test_map = extract_error_to_test_mapping(log_text)

        # Extract test sections for temporal correlation fallback
        test_sections = extract_test_sections(log_text)

        # Extract root-cause errors
        errors = extract_root_cause_errors(log_text)

        # Map each error to its affected tests
        for err in errors:
            error_type = err["error_type"]
            error_msg = err["error_message"]

            # Normalize error message to match the mapping format
            normalized = re.sub(r"pid=\d+", "pid=X", error_msg)
            normalized = re.sub(r"0x[0-9a-f]+", "0xXXX", normalized)
            normalized = re.sub(r"\[Errno \d+\]", "[Errno X]", normalized)
            sig = f"{error_type}:{normalized}"

            affected_tests_set = set()

            # Strategy 1: Use direct mapping from FAILED pytest lines
            if sig in error_to_test_map:
                affected_tests_set.update(error_to_test_map[sig])
            else:
                # Strategy 2: Try exact match
                exact_sig = f"{error_type}:{error_msg}"
                if exact_sig in error_to_test_map:
                    affected_tests_set.update(error_to_test_map[exact_sig])

            # Strategy 3: Temporal correlation by line_number
            # This matches errors to the test sections where they occurred
            err_line = err.get("line_number", 0)
            matched_test = None
            for section in test_sections:
                if section["start_line"] <= err_line < section["end_line"]:
                    matched_test = section["test_name"].split("::")[0]  # Use base test name
                    break
            if matched_test:
                affected_tests_set.add(matched_test)

            err["affected_tests"] = sorted(list(affected_tests_set))

        all_errors.extend(errors)

        job_results.append(
            {
                "job_id": job_id,
                "job_name": job_name,
                "failed_tests": tests,
                "errors": errors,
            }
        )

    # Deduplicate tests
    seen = set()
    unique_tests = []
    for t in all_failed_tests:
        if t not in seen:
            seen.add(t)
            unique_tests.append(t)

    # Deduplicate errors across jobs by signature, aggregating affected_tests
    seen_sigs = {}
    # sig -> {error, affected_tests_set}
    for err in all_errors:
        sig = f"{err['error_type']}:{err['error_message']}"
        if sig not in seen_sigs:
            seen_sigs[sig] = {
                "error": err,
                "affected_tests": set(),
            }
        # Aggregate affected_tests from all occurrences of this error
        for t in err.get("affected_tests", []):
            seen_sigs[sig]["affected_tests"].add(t)

    # Build unique_errors list with aggregated affected_tests
    unique_errors = []
    for sig, data in seen_sigs.items():
        err = data["error"]
        err["affected_tests"] = sorted(list(data["affected_tests"]))
        unique_errors.append(err)

    return {
        "run_id": run_id,
        "run_url": run_info.get("html_url"),
        "run_created_at": run_info.get("created_at"),
        "good_commit": good_commit,
        "bad_commit": bad_commit,
        "total_jobs": len(all_jobs),
        "failed_jobs_count": len(failed_jobs),
        "job_summary": [{"name": j["name"], "conclusion": j.get("conclusion", "unknown")} for j in all_jobs],
        "job_results": job_results,
        "failed_tests": unique_tests,
        "distinct_errors": unique_errors,
        "code_bugs": [e for e in unique_errors if e["category"] == "Code Bug"],
        "env_flakes": [e for e in unique_errors if e["category"] == "Environment Flake"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract and analyze failed tests from a vLLM-Ascend CI run.",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=None,
        help="Workflow run ID. If omitted, finds the latest failed run.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file path. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--llm-output",
        action="store_true",
        help="Generate a simplified, token-optimized JSON designed solely for LLM analysis.",
    )
    args = parser.parse_args()

    if args.run_id:
        run_id = args.run_id
    else:
        print("Finding latest failed run...", file=sys.stderr)
        run = find_latest_failed_run()
        if not run:
            print("No failed runs found.", file=sys.stderr)
            sys.exit(0)
        run_id = run["id"]
        print(f"Found run {run_id}: {run.get('html_url', '')}", file=sys.stderr)

    print(f"Analyzing run {run_id}...", file=sys.stderr)
    result = process_run(run_id)

    if args.llm_output:
        output_data = {
            "run_id": result["run_id"],
            "run_url": result["run_url"],
            "good_commit": result["good_commit"],
            "bad_commit": result["bad_commit"],
            "code_bugs": result["code_bugs"],
        }
    else:
        output_data = result

    output = json.dumps(output_data, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(output)

    # Summary to stderr
    print("\nSummary:", file=sys.stderr)
    print(f"  Good commit: {result['good_commit']}", file=sys.stderr)
    print(f"  Bad commit:  {result['bad_commit']}", file=sys.stderr)
    print(f"  Failed jobs: {result['failed_jobs_count']}/{result['total_jobs']}", file=sys.stderr)
    print(f"  Failed tests: {len(result['failed_tests'])}", file=sys.stderr)
    print(f"  Code bugs:   {len(result['code_bugs'])}", file=sys.stderr)
    print(f"  Env flakes:  {len(result['env_flakes'])}", file=sys.stderr)


if __name__ == "__main__":
    main()
