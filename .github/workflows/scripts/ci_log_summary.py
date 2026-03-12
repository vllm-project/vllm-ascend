#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = "vllm-project/vllm-ascend"

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
_PYTEST_START_RE = re.compile(r"pytest\s+-sv\s+(?:[/\w.\-]+/)?(tests/\S+)")
_PYTEST_FAILURE_HEADER_RE = re.compile(r"^_+\s+test_\S+.*_+$")

_CORE_ERROR_RE = re.compile(
    r"(TypeError|AttributeError|ImportError|ModuleNotFoundError"
    r"|KeyError|NotImplementedError|ValueError|OSError):\s*(.+)",
)

_WRAPPER_PATTERNS = [
    "Engine core initialization failed",
    "Worker failed with error",
    "subprocess.CalledProcessError",
    "SystemExit",
]

_DOWNSTREAM_PATTERNS = [
    r"KeyError:\s*'choices'",
    r"KeyError:\s*'message'",
    r"AssertionError:\s*assert.*response",
]

_ENV_FLAKE_PATTERNS = [
    r"OSError:.*Stale file handle",
    r"ConnectionResetError",
    r"filelock.*Lock",
    r"ConnectionRefusedError",
    r"TimeoutError",
    r"torch\.cuda\.OutOfMemoryError",
    r"OSError:.*No space left on device",
]

_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z\s*")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_GHA_LOG_PREFIX_RE = re.compile(r"^[^\t]+\t[^\t]+\t")
_VLLM_LOG_PREFIX_RE = re.compile(
    r"^(?:\[.*?\]\s*:\s*)?(?:\(.*?\)\s*)*[A-Z]+\s+\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+\[.*?\]\s*"
)
_PROFILER_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d+\s+-\s+\d+\s+-\s+\S+\s+-\s+[A-Z]+\s+-\s*")
_VLLM_VERSION_RE = re.compile(r"vLLM\s+\S*\+g([0-9a-f]{7,12})\b")


def gh_api_json(endpoint: str, **params) -> Any:
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


def clean_line(line: str) -> str:
    line = _GHA_LOG_PREFIX_RE.sub("", line)
    line = _TIMESTAMP_RE.sub("", line)
    line = _ANSI_RE.sub("", line)
    line = _VLLM_LOG_PREFIX_RE.sub("", line)
    line = _PROFILER_PREFIX_RE.sub("", line)
    return line


def _context_score(context: list[str]) -> tuple[int, int]:
    first_nonempty = next((line for line in context if line.strip()), "")
    if first_nonempty.startswith("Traceback (most recent call last):"):
        return (3, len(context))
    if _PYTEST_FAILURE_HEADER_RE.match(first_nonempty):
        return (2, len(context))
    return (1, len(context))


def _find_context_start(lines: list[str], error_index: int) -> int:
    fallback = max(0, error_index - 15)
    nearest_test_header = None

    for j in range(error_index, max(-1, error_index - 200), -1):
        cleaned_line = clean_line(lines[j])
        if "Traceback (most recent call last):" in cleaned_line:
            return j
        if _PYTEST_FAILURE_HEADER_RE.match(cleaned_line):
            nearest_test_header = j
            continue
        if cleaned_line.startswith("_ _ _ _ _") and len(cleaned_line.strip()) > 10:
            fallback = max(fallback, j - 1)

    if nearest_test_header is not None:
        return nearest_test_header
    return fallback


def extract_failed_test_files(log_text: str) -> list[str]:
    failed = set()
    cleaned_log_text = _ANSI_RE.sub("", log_text)

    for m in _FAILED_INLINE_RE.finditer(cleaned_log_text):
        failed.add(m.group(1).split("::")[0])
    for m in _FAILED_SUMMARY_RE.finditer(cleaned_log_text):
        failed.add(m.group(1).split("::")[0])
    for m in _FAILED_PYTEST_RE.finditer(cleaned_log_text):
        failed.add(m.group(1).split("::")[0])
    return sorted(failed)


def extract_failed_test_cases(log_text: str) -> list[str]:
    failed = set()
    cleaned_log_text = _ANSI_RE.sub("", log_text)

    for m in _FAILED_PYTEST_RE.finditer(cleaned_log_text):
        failed.add(m.group(1))

    return sorted(failed)


def extract_test_sections(log_text: str) -> list[dict]:
    lines = log_text.splitlines()
    sections = []
    current_test = None
    current_start = None

    for i, raw_line in enumerate(lines):
        line = clean_line(raw_line)
        m = _PYTEST_START_RE.search(line)
        if m:
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
    cleaned_log_text = _ANSI_RE.sub("", log_text)
    failed_pytest_re = re.compile(
        r"FAILED\s+(tests/\S+?)\s+-\s+(TypeError|AttributeError|ImportError|ModuleNotFoundError|KeyError|NotImplementedError|ValueError|OSError|RuntimeError|AssertionError):\s*(.+)"
    )

    error_to_tests = defaultdict(set)

    for m in failed_pytest_re.finditer(cleaned_log_text):
        test_name = m.group(1)
        error_type = m.group(2)
        error_msg = m.group(3).strip()
        base_test = test_name.split("::")[0]

        normalized = re.sub(r"pid=\d+", "pid=X", error_msg)
        normalized = re.sub(r"0x[0-9a-f]+", "0xXXX", normalized)
        normalized = re.sub(r"\[Errno \d+\]", "[Errno X]", normalized)
        sig = f"{error_type}:{normalized}"

        error_to_tests[sig].add(base_test)

        if error_type == "AssertionError":
            os_err_m = re.search(r"OSError:\s*\[Errno\s+\d+\]\s*(\S+(?:\s+\S+)?)", error_msg)
            if os_err_m:
                os_err_msg = os_err_m.group(1)
                os_normalized = re.sub(r"\[Errno \d+\]", "[Errno X]", f"[Errno X] {os_err_msg}")
                os_sig = f"OSError:{os_normalized}"
                error_to_tests[os_sig].add(base_test)

    return {sig: sorted(list(tests)) for sig, tests in error_to_tests.items()}


def extract_bad_commit(log_text: str) -> str | None:
    m = _VLLM_VERSION_RE.search(log_text)
    if m:
        short_sha = m.group(1)
        if shutil.which("gh") is None:
            return short_sha
        try:
            data = gh_api_json(f"/repos/vllm-project/vllm/commits/{short_sha}")
            return data.get("sha")
        except SystemExit:
            return short_sha
    return None


def extract_root_cause_errors(log_text: str) -> list[dict]:
    errors = []
    lines = log_text.splitlines()
    sig_to_entries = {}

    for i, raw_line in enumerate(lines):
        line = clean_line(raw_line)
        for pattern in _ENV_FLAKE_PATTERNS:
            if re.search(pattern, line):
                m_flake = re.search(
                    r"(OSError|ConnectionResetError|TimeoutError|ConnectionRefusedError):\s*(.+?)(?:\\n|$)",
                    line,
                )
                if m_flake:
                    error_type = m_flake.group(1)
                    error_msg = m_flake.group(2).strip()
                    error_msg = re.sub(r"(?:\\n|\\r|[\\'\"\n\r])+$", "", error_msg).strip()
                    error_msg = re.sub(r"\\n.*$", "", error_msg).strip()
                    normalized_msg = re.sub(r"\[Errno \d+\]", "[Errno X]", error_msg)
                    signature = f"{error_type}:{normalized_msg}"

                    if signature not in sig_to_entries:
                        sig_to_entries[signature] = {
                            "error_type": error_type,
                            "error_message": error_msg,
                            "category": "Environment Flake",
                            "line_numbers": [],
                            "best_context": None,
                            "best_context_score": None,
                        }

                    sig_to_entries[signature]["line_numbers"].append(i + 1)
                    context = [clean_line(lines[j]) for j in range(max(0, i - 2), min(len(lines), i + 3))]
                    score = _context_score(context)
                    if (
                        sig_to_entries[signature]["best_context"] is None
                        or score > sig_to_entries[signature]["best_context_score"]
                    ):
                        sig_to_entries[signature]["best_context"] = context
                        sig_to_entries[signature]["best_context_score"] = score
                break

    for i, raw_line in enumerate(lines):
        line = clean_line(raw_line)

        if any(wp in line for wp in _WRAPPER_PATTERNS):
            continue
        if line.startswith("FAILED tests/"):
            continue

        m = _CORE_ERROR_RE.search(line)
        if not m:
            continue

        error_type = m.group(1)
        error_msg = m.group(2).strip()
        full_error = f"{error_type}: {error_msg}"
        if any(re.search(p, full_error) for p in _DOWNSTREAM_PATTERNS):
            continue

        is_env_flake = any(re.search(p, full_error) for p in _ENV_FLAKE_PATTERNS)

        error_msg = re.sub(r"(\\n|\n).*$", "", error_msg)
        error_msg = re.sub(r"\\['\"]", "'", error_msg)
        error_msg = error_msg.strip()

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
                "best_context": None,
                "best_context_score": None,
            }

        sig_to_entries[signature]["line_numbers"].append(i)

        ctx_start = _find_context_start(lines, i)
        ctx_end = min(len(lines), i + 1)
        context = [clean_line(lines[j]) for j in range(ctx_start, ctx_end)]
        score = _context_score(context)
        if sig_to_entries[signature]["best_context"] is None or score > sig_to_entries[signature]["best_context_score"]:
            sig_to_entries[signature]["best_context"] = context
            sig_to_entries[signature]["best_context_score"] = score

    for entry in sig_to_entries.values():
        errors.append(
            {
                "error_type": entry["error_type"],
                "error_message": entry["error_message"],
                "category": entry["category"],
                "context": entry["best_context"] or [],
                "line_number": entry["line_numbers"][0],
            }
        )

    return errors


def get_good_commit() -> str | None:
    commit_re = re.compile(r"^[0-9a-f]{7,40}$")
    yaml_files = [
        ".github/workflows/pr_test_full.yaml",
        ".github/workflows/pr_test_light.yaml",
    ]

    for yaml_rel in yaml_files:
        try:
            repo_root = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            disk_path = Path(repo_root) / yaml_rel
            if disk_path.exists():
                content = disk_path.read_text()
                m = re.search(r"vllm_version:\s*\[([^\]]+)\]", content)
                if m:
                    entries = [e.strip().strip("'\"") for e in m.group(1).split(",")]
                    for entry in entries:
                        if commit_re.match(entry):
                            return entry
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            pass

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


def _attach_failed_tests(
    log_text: str,
    errors: list[dict],
    *,
    failed_test_files: list[str],
    failed_test_cases: list[str],
) -> None:
    test_sections = extract_test_sections(log_text)
    failed_files_set = set(failed_test_files)
    failed_cases_set = set(failed_test_cases)
    failed_cases_by_file: dict[str, set[str]] = defaultdict(set)
    for test_case in failed_test_cases:
        failed_cases_by_file[test_case.split("::")[0]].add(test_case)

    for err in errors:
        err_line = err.get("line_number", 0)
        matched_target = None
        for section in test_sections:
            if section["start_line"] <= err_line < section["end_line"]:
                matched_target = section["test_name"]
                break

        matched_files = set()
        matched_cases = set()
        if matched_target:
            matched_file = matched_target.split("::")[0]
            if matched_file in failed_files_set:
                matched_files.add(matched_file)
            if "::" in matched_target:
                if matched_target in failed_cases_set:
                    matched_cases.add(matched_target)
            else:
                matched_cases.update(failed_cases_by_file.get(matched_file, set()))

        err["failed_test_files"] = sorted(matched_files)
        err["failed_test_cases"] = sorted(matched_cases)


def _dedupe_errors(all_errors: list[dict]) -> list[dict]:
    seen_sigs = {}
    for err in all_errors:
        sig = f"{err['error_type']}:{err['error_message']}"
        if sig not in seen_sigs:
            seen_sigs[sig] = {
                "error": err,
                "failed_test_files": set(),
                "failed_test_cases": set(),
            }
        for test_file in err.get("failed_test_files", []):
            seen_sigs[sig]["failed_test_files"].add(test_file)
        for test_case in err.get("failed_test_cases", []):
            seen_sigs[sig]["failed_test_cases"].add(test_case)

    unique_errors = []
    for data in seen_sigs.values():
        err = data["error"]
        err["failed_test_files"] = sorted(list(data["failed_test_files"]))
        err["failed_test_cases"] = sorted(list(data["failed_test_cases"]))
        unique_errors.append(err)
    return unique_errors


def process_local_log(log_text: str, job_name: str = "local-log") -> dict:
    failed_test_files = extract_failed_test_files(log_text)
    failed_test_cases = extract_failed_test_cases(log_text)
    errors = extract_root_cause_errors(log_text)
    _attach_failed_tests(
        log_text,
        errors,
        failed_test_files=failed_test_files,
        failed_test_cases=failed_test_cases,
    )
    unique_errors = _dedupe_errors(errors)

    conclusion = "failure" if failed_test_files or failed_test_cases or unique_errors else "success"
    return {
        "run_id": None,
        "run_url": None,
        "run_created_at": None,
        "good_commit": get_good_commit(),
        "bad_commit": extract_bad_commit(log_text),
        "total_jobs": 1,
        "failed_jobs_count": 1 if conclusion == "failure" else 0,
        "job_summary": [{"name": job_name, "conclusion": conclusion}],
        "job_results": [
            {
                "job_id": None,
                "job_name": job_name,
                "failed_test_files": failed_test_files,
                "failed_test_cases": failed_test_cases,
                "errors": errors,
            }
        ],
        "failed_test_files": failed_test_files,
        "failed_test_cases": failed_test_cases,
        "distinct_errors": unique_errors,
        "code_bugs": [e for e in unique_errors if e["category"] == "Code Bug"],
        "env_flakes": [e for e in unique_errors if e["category"] == "Environment Flake"],
    }


def process_run(run_id: int, repo: str = REPO) -> dict:
    run_info = gh_api_json(f"/repos/{repo}/actions/runs/{run_id}")
    all_jobs_data = gh_api_json(
        f"/repos/{repo}/actions/runs/{run_id}/jobs",
        per_page="100",
    )
    all_jobs = all_jobs_data.get("jobs", [])
    failed_jobs = [j for j in all_jobs if j.get("conclusion") == "failure"]

    good_commit = get_good_commit()
    bad_commit = None
    all_failed_test_files = []
    all_failed_test_cases = []
    all_errors = []
    job_results = []

    for job in failed_jobs:
        job_id = job["id"]
        job_name = job["name"]
        log_text = gh_api_raw(f"/repos/{repo}/actions/jobs/{job_id}/logs")
        if not log_text:
            job_results.append(
                {
                    "job_id": job_id,
                    "job_name": job_name,
                    "error": "Failed to download log",
                }
            )
            continue

        if bad_commit is None:
            bad_commit = extract_bad_commit(log_text)

        failed_test_files = extract_failed_test_files(log_text)
        failed_test_cases = extract_failed_test_cases(log_text)
        all_failed_test_files.extend(failed_test_files)
        all_failed_test_cases.extend(failed_test_cases)

        errors = extract_root_cause_errors(log_text)
        _attach_failed_tests(
            log_text,
            errors,
            failed_test_files=failed_test_files,
            failed_test_cases=failed_test_cases,
        )
        all_errors.extend(errors)

        job_results.append(
            {
                "job_id": job_id,
                "job_name": job_name,
                "failed_test_files": failed_test_files,
                "failed_test_cases": failed_test_cases,
                "errors": errors,
            }
        )

    unique_failed_test_files = sorted(set(all_failed_test_files))
    unique_failed_test_cases = sorted(set(all_failed_test_cases))

    unique_errors = _dedupe_errors(all_errors)

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
        "failed_test_files": unique_failed_test_files,
        "failed_test_cases": unique_failed_test_cases,
        "distinct_errors": unique_errors,
        "code_bugs": [e for e in unique_errors if e["category"] == "Code Bug"],
        "env_flakes": [e for e in unique_errors if e["category"] == "Environment Flake"],
    }


def analyze_log(log_text: str, job_name: str = "local-log") -> dict:
    return process_local_log(log_text, job_name=job_name)


def _format_error_block(error: dict) -> str:
    lines = [
        f"- `{error['error_type']}`: {error['error_message']}",
    ]
    if error.get("failed_test_files"):
        failed_files = ", ".join(f"`{test}`" for test in error["failed_test_files"])
        lines.append(f"  Failed test files: {failed_files}")
    if error.get("failed_test_cases"):
        failed_cases = ", ".join(f"`{test}`" for test in error["failed_test_cases"])
        lines.append(f"  Failed test cases: {failed_cases}")
    context = "\n".join(error.get("context", [])).strip()
    if context:
        lines.append("  Context:")
        lines.append("  ```text")
        lines.append("\n".join(f"  {line}" for line in context.splitlines()))
        lines.append("  ```")
    return "\n".join(lines)


def render_summary(result: dict, *, step_name: str, mode: str) -> str:
    lines = [
        f"## Test Failure Summary: {step_name}",
        "",
        f"- Mode: `{mode}`",
        f"- Failed test files: `{len(result['failed_test_files'])}`",
        f"- Failed test cases: `{len(result['failed_test_cases'])}`",
        f"- Distinct issues: `{len(result['distinct_errors'])}`",
        f"- Code bugs: `{len(result['code_bugs'])}`",
        f"- Environment flakes: `{len(result['env_flakes'])}`",
    ]

    if result.get("run_id") is not None:
        lines.append(f"- Run ID: `{result['run_id']}`")
    if result.get("run_url"):
        lines.append(f"- Run URL: {result['run_url']}")
    if result.get("good_commit"):
        lines.append(f"- Good commit: `{result['good_commit']}`")
    if result.get("bad_commit"):
        lines.append(f"- Bad commit: `{result['bad_commit']}`")
    lines.append("")

    if result["failed_test_files"]:
        lines.extend(
            [
                "### Failed Test Files",
                "",
                *[f"- `{test}`" for test in result["failed_test_files"]],
                "",
            ]
        )

    if result["failed_test_cases"]:
        lines.extend(
            [
                "### Failed Test Cases",
                "",
                *[f"- `{test}`" for test in result["failed_test_cases"]],
                "",
            ]
        )

    if result["code_bugs"]:
        lines.extend(["### Code Bugs", ""])
        for error in result["code_bugs"]:
            lines.append(_format_error_block(error))
            lines.append("")

    if result["env_flakes"]:
        lines.extend(["### Environment Flakes", ""])
        for error in result["env_flakes"]:
            lines.append(_format_error_block(error))
            lines.append("")

    if not result["distinct_errors"]:
        lines.extend(
            [
                "### Notes",
                "",
                "- No root-cause exception was extracted from the input log.",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GitHub job summary from a local test log or workflow run.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--log-file", type=Path, help="Path to the local test log file.")
    source.add_argument("--run-id", type=int, help="GitHub Actions run ID to analyze through gh api.")
    parser.add_argument("--repo", default=REPO, help=f"GitHub repo for --run-id mode (default: {REPO}).")
    parser.add_argument("--mode", required=True, choices=("ut", "e2e"), help="Test mode for the summary.")
    parser.add_argument("--step-name", required=True, help="Workflow step name shown in the summary.")
    parser.add_argument(
        "--summary-file",
        default=None,
        help="Path to GitHub step summary file. If omitted, prints to stdout.",
    )
    args = parser.parse_args()

    if args.run_id is not None:
        result = process_run(args.run_id, repo=args.repo)
    else:
        log_text = args.log_file.read_text(encoding="utf-8", errors="replace")
        result = process_local_log(log_text, job_name=args.step_name)

    summary = render_summary(result, step_name=args.step_name, mode=args.mode)

    if args.summary_file:
        with open(args.summary_file, "a", encoding="utf-8") as f:
            f.write(summary)
    else:
        print(summary, end="")


if __name__ == "__main__":
    main()
