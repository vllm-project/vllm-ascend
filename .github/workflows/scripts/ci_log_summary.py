#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
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
_PYTEST_CASE_PROGRESS_RE = re.compile(r"^(tests/\S+\.py::\S+)\b")
_PYTEST_START_RE = re.compile(r"pytest\s+-sv\s+(?:[/\w.\-]+/)?(tests/\S+)")
_RUN_SUITE_START_RE = re.compile(r"\[\d+/\d+\]\s+START\s+(tests/\S+)")
_PYTEST_FAILURE_HEADER_RE = re.compile(r"^_+\s+test_\S+.*_+$")
_PYTEST_FAILURES_BANNER_RE = re.compile(r"^=+\s+FAILURES\s+=+$")
_PYTEST_SUMMARY_BANNER_RE = re.compile(r"^=+\s+short test summary info\s+=+$", re.IGNORECASE)
_PYTEST_SUMMARY_FAILED_RE = re.compile(r"^FAILED\s+(tests/\S+\.py::\S+)")

_CORE_ERROR_RE = re.compile(
    r"(TypeError|AttributeError|ImportError|ModuleNotFoundError"
    r"|KeyError|NotImplementedError|ValueError|OSError|AssertionError):\s*(.+)",
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

_WRAPPER_ASSERTION_PATTERNS = [
    r"function <function .* failed when called with args .* and kwargs .*",
    r"assert _exitcode == 0",
]

_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z\s*")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_GHA_LOG_PREFIX_RE = re.compile(r"^[^\t]+\t[^\t]+\t")
_VLLM_LOG_PREFIX_RE = re.compile(
    r"^(?:\[.*?\]\s*:\s*)?(?:\(.*?\)\s*)*[A-Z]+\s+\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+\[.*?\]\s*"
)
_PROFILER_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d+\s+-\s+\d+\s+-\s+\S+\s+-\s+[A-Z]+\s+-\s*")
_VLLM_VERSION_RE = re.compile(r"vLLM\s+\S*\+g([0-9a-f]{7,12})\b")
_WORKER_PID_PREFIX_RE = re.compile(r"^\([^)]*pid=\d+\)\s*")
_TEST_CASE_TOKEN_RE = re.compile(r"(tests/\S+\.py::\S+)")
_MAX_CONTEXT_LINES = 40


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


def _normalize_error_signature(error_type: str, error_message: str) -> str:
    normalized = re.sub(r"pid=\d+", "pid=X", error_message)
    normalized = re.sub(r"0x[0-9a-f]+", "0xXXX", normalized)
    normalized = re.sub(r"\[Errno \d+\]", "[Errno X]", normalized)
    return f"{error_type}:{normalized}"


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


def _strip_worker_prefix(line: str) -> str:
    return _WORKER_PID_PREFIX_RE.sub("", line)


def _extract_worker_prefix(line: str) -> str | None:
    match = _WORKER_PID_PREFIX_RE.match(line)
    if not match:
        return None
    return match.group(0).strip()


def _is_tracebackish_line(line: str) -> bool:
    stripped = _strip_worker_prefix(line)
    if not stripped:
        return True
    if stripped.startswith("Traceback (most recent call last):"):
        return True
    if stripped.startswith("During handling of the above exception"):
        return True
    if stripped.startswith("  File ") or stripped.startswith('File "'):
        return True
    if stripped.startswith(" ") or stripped.startswith("^"):
        return True
    return bool(_CORE_ERROR_RE.search(stripped))


def _extract_traceback_block(lines: list[str], error_index: int) -> list[str] | None:
    candidate_starts = []
    for j in range(error_index, max(-1, error_index - 200), -1):
        cleaned_line = clean_line(lines[j])
        if "Traceback (most recent call last):" in _strip_worker_prefix(cleaned_line):
            candidate_starts.append(j)

    for start in candidate_starts:
        block: list[str] = []
        start_cleaned = clean_line(lines[start])
        worker_prefix = _extract_worker_prefix(start_cleaned)
        valid = True

        for j in range(start, error_index + 1):
            cleaned_line = clean_line(lines[j])
            if not _is_tracebackish_line(cleaned_line):
                valid = False
                break

            current_prefix = _extract_worker_prefix(cleaned_line)
            if worker_prefix is not None and current_prefix is not None and current_prefix != worker_prefix:
                valid = False
                break

            block.append(cleaned_line)

        if valid and block:
            tracebackish = sum(1 for line in block if _is_tracebackish_line(line))
            if tracebackish / max(len(block), 1) < 0.8:
                continue
            return block

    return None


def _compress_context(context: list[str]) -> list[str]:
    if len(context) <= _MAX_CONTEXT_LINES:
        return context

    return context[:24] + ["..."] + context[-12:]


def _normalize_error_match(error_type: str, error_msg: str) -> tuple[str, str, str, bool]:
    full_error = f"{error_type}: {error_msg}"
    if any(re.search(p, full_error) for p in _DOWNSTREAM_PATTERNS):
        return ("", "", "", False)

    is_env_flake = any(re.search(p, full_error) for p in _ENV_FLAKE_PATTERNS)

    error_msg = re.sub(r"(\\n|\n).*$", "", error_msg)
    error_msg = re.sub(r"\\['\"]", "'", error_msg)
    error_msg = error_msg.strip()

    normalized = re.sub(r"pid=\d+", "pid=X", error_msg)
    normalized = re.sub(r"0x[0-9a-f]+", "0xXXX", normalized)
    normalized = re.sub(r"\d{4}-\d{2}-\d{2}", "YYYY-MM-DD", normalized)
    normalized = re.sub(r"\[Errno \d+\]", "[Errno X]", normalized)
    normalized = re.sub(r"""(?:\\[nr]|['"])+$""", "", normalized).strip()
    return error_msg, normalized, ("Environment Flake" if is_env_flake else "Code Bug"), True


def _extract_pytest_failure_blocks(lines: list[str]) -> list[dict[str, int]]:
    blocks: list[dict[str, int]] = []
    in_failures = False
    current_start = None
    current_has_terminal = False

    for i, raw_line in enumerate(lines):
        line = clean_line(raw_line)
        if _PYTEST_FAILURES_BANNER_RE.match(line):
            in_failures = True
            current_start = None
            current_has_terminal = False
            continue
        if not in_failures:
            continue
        if _PYTEST_SUMMARY_BANNER_RE.match(line):
            if current_start is not None:
                blocks.append({"start_line": current_start, "end_line": i})
            break
        if current_start is not None:
            if line.startswith("E ") or line.startswith("E       ") or re.search(r"tests/\S+\.py:\d+:", line):
                current_has_terminal = True
        if _PYTEST_FAILURE_HEADER_RE.match(line):
            if current_start is None:
                current_start = i
                current_has_terminal = False
                continue
            if current_has_terminal:
                blocks.append({"start_line": current_start, "end_line": i})
                current_start = i
                current_has_terminal = False

    return blocks


def extract_failed_test_files(log_text: str) -> list[str]:
    failed = set()
    cleaned_log_text = _ANSI_RE.sub("", log_text)
    failed_cases = extract_failed_test_cases(log_text)

    for m in _FAILED_INLINE_RE.finditer(cleaned_log_text):
        failed.add(m.group(1).split("::")[0])
    for m in _FAILED_SUMMARY_RE.finditer(cleaned_log_text):
        failed.add(m.group(1).split("::")[0])
    for test_case in failed_cases:
        failed.add(test_case.split("::")[0])
    return sorted(failed)


def extract_failed_test_cases(log_text: str) -> list[str]:
    lines = log_text.splitlines()
    failed = set()
    in_summary = False

    for raw_line in lines:
        line = clean_line(raw_line)
        if _PYTEST_SUMMARY_BANNER_RE.match(line):
            in_summary = True
            continue
        if in_summary and line.startswith("="):
            in_summary = False
        if not in_summary:
            continue
        m = _PYTEST_SUMMARY_FAILED_RE.match(line)
        if m:
            failed.add(m.group(1))

    if failed:
        return sorted(failed)

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
        m = _PYTEST_CASE_PROGRESS_RE.search(line)
        if not m:
            m = _PYTEST_START_RE.search(line)
        if not m:
            m = _RUN_SUITE_START_RE.search(line)
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

        sig = _normalize_error_signature(error_type, error_msg)

        error_to_tests[sig].add(test_name)

        if error_type == "AssertionError":
            os_err_m = re.search(r"OSError:\s*\[Errno\s+\d+\]\s*(\S+(?:\s+\S+)?)", error_msg)
            if os_err_m:
                os_err_msg = os_err_m.group(1)
                os_normalized = re.sub(r"\[Errno \d+\]", "[Errno X]", f"[Errno X] {os_err_msg}")
                os_sig = f"OSError:{os_normalized}"
                error_to_tests[os_sig].add(test_name)

    return {sig: sorted(list(tests)) for sig, tests in error_to_tests.items()}


def extract_failed_case_mentions(log_text: str, failed_test_cases: list[str]) -> dict[str, list[int]]:
    mentions: dict[str, list[int]] = {test_case: [] for test_case in failed_test_cases}
    if not mentions:
        return mentions

    lines = log_text.splitlines()
    for i, raw_line in enumerate(lines):
        line = clean_line(raw_line)
        for test_case in _TEST_CASE_TOKEN_RE.findall(line):
            if test_case in mentions:
                mentions[test_case].append(i)

    return mentions


def extract_bad_commit(log_text: str, *, resolve_remote: bool = True) -> str | None:
    m = _VLLM_VERSION_RE.search(log_text)
    if m:
        short_sha = m.group(1)
        if not resolve_remote or shutil.which("gh") is None:
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
    consumed_error_lines: set[int] = set()

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
                    context = [clean_line(lines[j]) for j in range(max(0, i - 2), min(len(lines), i + 3))]
                    errors.append(
                        {
                            "error_type": error_type,
                            "error_message": error_msg,
                            "category": "Environment Flake",
                            "context": context,
                            "line_number": i + 1,
                            "source": "environment",
                        }
                    )
                    consumed_error_lines.add(i)
                break

    for block in _extract_pytest_failure_blocks(lines):
        candidate: tuple[int, str, str, str] | None = None
        for i in range(block["start_line"], block["end_line"]):
            line = clean_line(lines[i])
            if any(wp in line for wp in _WRAPPER_PATTERNS):
                continue
            if line.startswith("FAILED tests/"):
                continue

            m = _CORE_ERROR_RE.search(line)
            if not m:
                continue

            error_type = m.group(1)
            error_msg, _, category, ok = _normalize_error_match(error_type, m.group(2).strip())
            if not ok:
                continue
            candidate = (i, error_type, error_msg, category)

        if candidate is None:
            continue

        line_index, error_type, error_msg, category = candidate
        context = [clean_line(lines[j]) for j in range(block["start_line"], line_index + 1)]
        errors.append(
            {
                "error_type": error_type,
                "error_message": error_msg,
                "category": category,
                "context": context,
                "line_number": line_index,
                "source": "pytest_failure_block",
            }
        )
        consumed_error_lines.update(range(block["start_line"], block["end_line"]))

    for i, raw_line in enumerate(lines):
        if i in consumed_error_lines:
            continue
        line = clean_line(raw_line)

        if any(wp in line for wp in _WRAPPER_PATTERNS):
            continue
        if line.startswith("FAILED tests/"):
            continue

        m = _CORE_ERROR_RE.search(line)
        if not m:
            continue

        error_type = m.group(1)
        error_msg, normalized, category, ok = _normalize_error_match(error_type, m.group(2).strip())
        if not ok:
            continue
        context = _extract_traceback_block(lines, i)
        if context is None:
            ctx_start = _find_context_start(lines, i)
            ctx_end = min(len(lines), i + 1)
            context = [clean_line(lines[j]) for j in range(ctx_start, ctx_end)]
        errors.append(
            {
                "error_type": error_type,
                "error_message": error_msg,
                "category": category,
                "context": context,
                "line_number": i,
                "source": "general",
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
    error_to_test_map = extract_error_to_test_mapping(log_text)
    test_sections = extract_test_sections(log_text)
    failed_case_mentions = extract_failed_case_mentions(log_text, failed_test_cases)
    failed_files_set = set(failed_test_files)
    failed_cases_set = set(failed_test_cases)
    failed_cases_by_file: dict[str, set[str]] = defaultdict(set)
    for test_case in failed_test_cases:
        failed_cases_by_file[test_case.split("::")[0]].add(test_case)

    for err in errors:
        sig = _normalize_error_signature(err["error_type"], err["error_message"])

        mapped_tests = set(error_to_test_map.get(sig, []))
        err_line = err.get("line_number", 0)
        matched_target = None
        for section in test_sections:
            if section["start_line"] <= err_line < section["end_line"]:
                matched_target = section["test_name"]
                break

        matched_files = set()
        matched_cases = set()
        if matched_target and "::" in matched_target and err.get("source") != "pytest_failure_block":
            matched_file = matched_target.split("::")[0]
            if matched_file in failed_files_set:
                matched_files.add(matched_file)
            if matched_target in failed_cases_set:
                matched_cases.add(matched_target)
        else:
            for test_name in mapped_tests:
                matched_files.add(test_name.split("::")[0])
                if "::" in test_name:
                    matched_cases.add(test_name)
            if matched_target and err.get("source") != "pytest_failure_block":
                matched_file = matched_target.split("::")[0]
                if matched_file in failed_files_set:
                    matched_files.add(matched_file)
                if "::" in matched_target:
                    if matched_target in failed_cases_set:
                        matched_cases.add(matched_target)
                else:
                    matched_cases.update(failed_cases_by_file.get(matched_file, set()))

        if not matched_cases and len(matched_files) == 1:
            matched_file = next(iter(matched_files))
            candidate_cases = sorted(failed_cases_by_file.get(matched_file, set()))
            if candidate_cases:
                nearest_case = min(
                    candidate_cases,
                    key=lambda test_case: min(
                        (abs(err_line - mention_line) for mention_line in failed_case_mentions.get(test_case, [])),
                        default=float("inf"),
                    ),
                )
                if failed_case_mentions.get(nearest_case):
                    matched_cases.add(nearest_case)

        err["failed_test_files"] = sorted(matched_files)
        err["failed_test_cases"] = sorted(matched_cases)
        err["context"] = _compress_context(err.get("context", []))


def _dedupe_errors(all_errors: list[dict]) -> list[dict]:
    seen_sigs = {}
    for err in all_errors:
        sig = f"{err['error_type']}:{err['error_message']}"
        if sig not in seen_sigs:
            seen_sigs[sig] = {
                "error": copy.deepcopy(err),
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


def _dedupe_errors_by_scope(errors: list[dict]) -> list[dict]:
    seen: dict[tuple[Any, ...], dict] = {}
    for error in errors:
        key = (
            error["error_type"],
            error["error_message"],
            tuple(error.get("failed_test_files", [])),
            tuple(error.get("failed_test_cases", [])),
        )
        if key not in seen:
            seen[key] = copy.deepcopy(error)
            continue

        if error.get("line_number", 0) < seen[key].get("line_number", 0):
            seen[key] = copy.deepcopy(error)

    return list(seen.values())


def _is_wrapper_assertion(error: dict) -> bool:
    if error.get("error_type") != "AssertionError":
        return False

    error_message = error.get("error_message", "")
    context = "\n".join(error.get("context", []))
    return any(
        re.search(pattern, error_message) or re.search(pattern, context) for pattern in _WRAPPER_ASSERTION_PATTERNS
    )


def _suppress_wrapper_assertions(errors: list[dict]) -> list[dict]:
    case_to_specific_errors: dict[str, set[str]] = defaultdict(set)
    file_to_specific_errors: dict[str, set[str]] = defaultdict(set)

    for error in errors:
        if _is_wrapper_assertion(error):
            continue
        signature = f"{error['error_type']}:{error['error_message']}"
        for test_case in error.get("failed_test_cases", []):
            case_to_specific_errors[test_case].add(signature)
        for test_file in error.get("failed_test_files", []):
            file_to_specific_errors[test_file].add(signature)

    filtered = []
    for error in errors:
        if not _is_wrapper_assertion(error):
            filtered.append(error)
            continue

        matched_specific = False
        for test_case in error.get("failed_test_cases", []):
            if case_to_specific_errors.get(test_case):
                matched_specific = True
                break
        if not matched_specific:
            for test_file in error.get("failed_test_files", []):
                if file_to_specific_errors.get(test_file):
                    matched_specific = True
                    break

        if not matched_specific:
            filtered.append(error)

    return filtered


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
    errors = _suppress_wrapper_assertions(errors)
    job_errors = _dedupe_errors_by_scope(errors)
    unique_errors = _dedupe_errors(job_errors)

    conclusion = "failure" if failed_test_files or failed_test_cases or unique_errors else "success"
    return {
        "run_id": None,
        "run_url": None,
        "run_created_at": None,
        "good_commit": get_good_commit(),
        "bad_commit": extract_bad_commit(log_text, resolve_remote=False),
        "total_jobs": 1,
        "failed_jobs_count": 1 if conclusion == "failure" else 0,
        "job_summary": [{"name": job_name, "conclusion": conclusion}],
        "job_results": [
            {
                "job_id": None,
                "job_name": job_name,
                "failed_test_files": failed_test_files,
                "failed_test_cases": failed_test_cases,
                "errors": job_errors,
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
        errors = _suppress_wrapper_assertions(errors)
        job_scoped_errors = _dedupe_errors_by_scope(errors)
        all_errors.extend(job_scoped_errors)

        job_results.append(
            {
                "job_id": job_id,
                "job_name": job_name,
                "failed_test_files": failed_test_files,
                "failed_test_cases": failed_test_cases,
                "errors": job_scoped_errors,
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


def _format_error_block(index: int, error: dict) -> list[str]:
    lines = [
        f"{index}. `{error['error_type']}`: {error['error_message']}",
        f"   Category: `{error['category']}`",
    ]

    failed_test_files = error.get("failed_test_files", [])
    if failed_test_files:
        lines.append("   Failed test files:")
        lines.extend(f"   - `{test}`" for test in failed_test_files)

    failed_test_cases = error.get("failed_test_cases", [])
    if failed_test_cases:
        lines.append("   Failed test cases:")
        lines.extend(f"   - `{test}`" for test in failed_test_cases)

    context = error.get("context", [])
    if context:
        lines.extend(
            [
                "   Context:",
                "   ```text",
                *[f"   {line}" for line in context],
                "   ```",
            ]
        )

    return lines


def render_json(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2) + "\n"


def render_llm_json(result: dict) -> str:
    output_data = {
        "run_id": result["run_id"],
        "run_url": result["run_url"],
        "good_commit": result["good_commit"],
        "bad_commit": result["bad_commit"],
        "failed_test_files_count": len(result["failed_test_files"]),
        "failed_test_cases_count": len(result["failed_test_cases"]),
        "failed_test_files": result["failed_test_files"],
        "failed_test_cases": result["failed_test_cases"],
        "code_bugs": result["code_bugs"],
        "env_flakes": result["env_flakes"],
    }
    return json.dumps(output_data, ensure_ascii=False, indent=2) + "\n"


def render_summary(result: dict, *, step_name: str, mode: str) -> str:
    lines = [
        f"## Test Failure Summary: {step_name}",
        "",
        "### Overview",
        "",
        f"- Mode: `{mode}`",
    ]

    if result.get("run_id") is not None:
        lines.append(f"- Run ID: `{result['run_id']}`")
    if result.get("run_url"):
        lines.append(f"- Run URL: {result['run_url']}")
    lines.extend(
        [
            f"- Failed test files: `{len(result['failed_test_files'])}`",
            f"- Failed test cases: `{len(result['failed_test_cases'])}`",
            f"- Distinct issues: `{len(result['distinct_errors'])}`",
            f"- Code bugs: `{len(result['code_bugs'])}`",
            f"- Environment flakes: `{len(result['env_flakes'])}`",
            "",
        ]
    )

    if result["failed_test_files"]:
        lines.extend(
            [
                "### Failed Tests",
                "",
                "Files:",
                "",
                *[f"- `{test}`" for test in result["failed_test_files"]],
                "",
            ]
        )

    if result["failed_test_cases"]:
        lines.extend(
            [
                "Cases:",
                "",
                *[f"- `{test}`" for test in result["failed_test_cases"]],
                "",
            ]
        )

    if result["distinct_errors"]:
        lines.extend(["### Distinct Issues", ""])
        for index, error in enumerate(result["distinct_errors"], start=1):
            lines.extend(_format_error_block(index, error))
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
    parser.add_argument(
        "--mode",
        default="e2e",
        choices=("ut", "e2e"),
        help="Test mode for the summary (default: e2e).",
    )
    parser.add_argument(
        "--step-name",
        default="Run test",
        help="Workflow step name shown in the summary (default: Run test).",
    )
    parser.add_argument(
        "--format",
        choices=("summary", "json", "llm-json"),
        default="summary",
        help="Output format (default: summary).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file path. If omitted, prints to stdout.",
    )
    args = parser.parse_args()

    if args.run_id is not None:
        result = process_run(args.run_id, repo=args.repo)
    else:
        log_text = args.log_file.read_text(encoding="utf-8", errors="replace")
        result = process_local_log(log_text, job_name=args.step_name)

    if args.format == "json":
        rendered_output = render_json(result)
    elif args.format == "llm-json":
        rendered_output = render_llm_json(result)
    else:
        rendered_output = render_summary(result, step_name=args.step_name, mode=args.mode)

    if args.output is not None:
        args.output.write_text(rendered_output, encoding="utf-8")
    else:
        print(rendered_output, end="")


if __name__ == "__main__":
    main()
