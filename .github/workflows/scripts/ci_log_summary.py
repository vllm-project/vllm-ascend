from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import regex as re

"""
Generate CI failure summaries from a local pytest log or a GitHub Actions run.
Examples:
    python3 .github/workflows/scripts/ci_log_summary.py --log-file /tmp/unit-test.log --mode ut --step-name "Unit test"
    python3 .github/workflows/scripts/ci_log_summary.py --run-id 23127187822 --format json
"""

REPO = "vllm-project/vllm-ascend"
_RUN_SUITE_START_RE = re.compile(r"\[\d+/\d+\]\s+START\s+(tests/\S+)")
_RUN_SUITE_END_RE = re.compile(r"\[\d+/\d+\]\s+(?:PASSED|FAILED \(exit code \d+\))\s+(tests/\S+)")
_PYTEST_FAILURE_HEADER_RE = re.compile(r"^_+\s+test_\S+.*_+$")
_PYTEST_FAILURES_BANNER_RE = re.compile(r"^=+\s+FAILURES\s+=+$")
_PYTEST_SUMMARY_BANNER_RE = re.compile(r"^=+\s+short test summary info\s+=+$", re.IGNORECASE)
_PYTEST_SUMMARY_FAILED_RE = re.compile(r"^FAILED\s+(tests/\S+\.py::\S+)")
_FAILED_SUMMARY_PAYLOAD_RE = re.compile(r"^FAILED\s+(tests/\S+\.py::\S+)\s+-\s+(.+)")
_EXTENDED_ERROR_RE = re.compile(r"((?:[A-Za-z_][\w]*\.)*[A-Za-z_][\w]*(?:Error|Exception)):\s*(.+)")
_SUMMARY_NAMED_ERROR_RE = re.compile(r"((?:[A-Za-z_][\w]*\.)*[A-Z][\w]+):\s*(.+)")

_ENV_FLAKE_PATTERNS = [
    r"OSError:.*Stale file handle",
    r"ConnectionResetError",
    r"filelock.*Lock",
    r"ConnectionRefusedError",
    r"TimeoutError",
    r"torch\.cuda\.OutOfMemoryError",
    r"OSError:.*No space left on device",
]

_WRAPPER_PATTERNS = [
    "Engine core initialization failed",
    "Worker failed with error",
    "subprocess.CalledProcessError",
    "SystemExit",
    "Server at 0.0.0.0 exited unexpectedly",
    "EngineCore encountered an issue",
    "See stack trace above",
    "NPUModelRunner init failed",
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
_MAX_CONTEXT_LINES = 50


def gh_api_json(endpoint: str, **params) -> Any:
    url = endpoint
    if params:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{endpoint}?{qs}"
    try:
        result = subprocess.run(["gh", "api", url], capture_output=True, text=True, check=True)
    except FileNotFoundError:
        print("ERROR: 'gh' CLI not found. Install it or run 'gh auth login'.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: gh api {url} failed: {exc.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return json.loads(result.stdout)


def gh_api_raw(endpoint: str) -> str:
    try:
        result = subprocess.run(["gh", "api", endpoint], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"WARNING: Failed to download {endpoint}: {exc.stderr.strip()}", file=sys.stderr)
        return ""
    return result.stdout


def clean_line(line: str) -> str:
    line = _GHA_LOG_PREFIX_RE.sub("", line)
    line = _TIMESTAMP_RE.sub("", line)
    line = _ANSI_RE.sub("", line)
    line = _VLLM_LOG_PREFIX_RE.sub("", line)
    line = _PROFILER_PREFIX_RE.sub("", line)
    return line


def _strip_worker_prefix(line: str) -> str:
    return _WORKER_PID_PREFIX_RE.sub("", line)


def _clean_context_line(line: str) -> str:
    return _strip_worker_prefix(clean_line(line))


def _compress_context(context: list[str]) -> list[str]:
    if len(context) <= _MAX_CONTEXT_LINES:
        return context
    return context[:10] + ["..."] + context[-38:]


def _normalize_error_match(error_type: str, error_msg: str) -> tuple[str, str]:
    full_error = f"{error_type}: {error_msg}"
    is_env_flake = any(re.search(pattern, full_error) for pattern in _ENV_FLAKE_PATTERNS)
    error_msg = re.sub(r"(\\n|\n).*$", "", error_msg)
    error_msg = re.sub(r"\\['\"]", "'", error_msg)
    error_msg = error_msg.strip()
    error_msg = re.sub(r"""(?:\\[nr]|['"])+$""", "", error_msg).strip()
    return error_msg, ("Environment Flake" if is_env_flake else "Code Bug")


def _is_wrapper_error(error_type: str, error_message: str) -> bool:
    haystack = f"{error_type}: {error_message}"
    return any(pattern in haystack for pattern in _WRAPPER_PATTERNS)


def _match_error_line(line: str) -> tuple[str, str] | None:
    for match in _EXTENDED_ERROR_RE.finditer(line):
        if match.start() > 0 and line[match.start() - 1] == "\\":
            continue
        return match.group(1), match.group(2).strip()
    return None


def _iter_payload_error_matches(payload: str) -> list[tuple[str, str]]:
    normalized_payload = payload.replace("\\n", "\n").replace("\\r", "\n")
    matches: list[tuple[str, str]] = []
    for match in _EXTENDED_ERROR_RE.finditer(normalized_payload):
        matches.append((match.group(1), match.group(2).strip()))
    return matches


def _iter_pytest_summary_lines(log_text: str) -> list[str]:
    lines = log_text.splitlines()
    summary_lines: list[str] = []
    in_summary = False
    for raw_line in lines:
        line = clean_line(raw_line)
        if _PYTEST_SUMMARY_BANNER_RE.match(line):
            in_summary = True
            continue
        if in_summary and line.startswith("="):
            in_summary = False
        if in_summary:
            summary_lines.append(line)
    return summary_lines


def extract_failed_test_cases(log_text: str) -> list[str]:
    failed = set()
    for line in _iter_pytest_summary_lines(log_text):
        match = _PYTEST_SUMMARY_FAILED_RE.match(line)
        if match:
            failed.add(match.group(1))
    return sorted(failed)


def _extract_named_summary_error(payload: str) -> tuple[str, str, str] | None:
    match = _SUMMARY_NAMED_ERROR_RE.search(payload)
    if not match:
        return None
    error_type = match.group(1).strip()
    raw_error_message = re.sub(r"""(?:\\[nr]|['"])+$""", "", match.group(2)).strip()
    error_msg, category = _normalize_error_match(error_type, raw_error_message)
    return error_type, error_msg, category


def _extract_summary_error_info(line: str) -> tuple[str, str, str, str] | None:
    summary_match = _FAILED_SUMMARY_PAYLOAD_RE.match(line)
    if not summary_match:
        return None

    test_name = summary_match.group(1)
    payload = summary_match.group(2).strip()
    named_error = _extract_named_summary_error(payload)
    if named_error is not None:
        error_type, error_msg, category = named_error
        return test_name, error_type, error_msg, category
    if ":" not in payload:
        return None

    error_type, raw_error_message = payload.split(":", 1)
    error_type = error_type.strip()
    raw_error_message = raw_error_message.strip()
    if not error_type or " " in error_type:
        return None

    error_msg, category = _normalize_error_match(error_type, raw_error_message)
    return test_name, error_type, error_msg, category


def _extract_pytest_failure_blocks(lines: list[str]) -> list[dict[str, int]]:
    blocks: list[dict[str, int]] = []
    in_failures = False
    current_start = None
    current_has_terminal = False

    for idx, raw_line in enumerate(lines):
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
                blocks.append({"start_line": current_start, "end_line": idx})
            break
        if current_start is not None:
            if line.startswith("E ") or line.startswith("E       ") or re.search(r"tests/\S+\.py:\d+:", line):
                current_has_terminal = True
        if _PYTEST_FAILURE_HEADER_RE.match(line):
            if current_start is None:
                current_start = idx
                current_has_terminal = False
                continue
            if current_has_terminal:
                blocks.append({"start_line": current_start, "end_line": idx})
                current_start = idx
                current_has_terminal = False

    return blocks


def _base_case_name(test_case: str) -> str:
    if "[" not in test_case:
        return test_case
    prefix, _, _suffix = test_case.partition("[")
    return prefix if "::" in prefix else test_case


def _header_matches_case(header_line: str, test_case: str) -> bool:
    full_target = test_case.split("::", 1)[-1]
    base_target = _base_case_name(test_case).split("::", 1)[-1]
    cleaned = clean_line(header_line).strip("_ ").strip()
    return cleaned in (full_target, base_target)


def _build_invocation_sections(log_text: str) -> list[dict[str, Any]]:
    lines = log_text.splitlines()
    sections: list[dict[str, Any]] = []
    current_name: str | None = None
    current_start: int | None = None

    for idx, raw_line in enumerate(lines):
        line = clean_line(raw_line)
        start_match = _RUN_SUITE_START_RE.search(line)
        if start_match:
            if current_name is not None and current_start is not None:
                sections.append({"test_name": current_name, "start_line": current_start, "end_line": idx})
            current_name = start_match.group(1)
            current_start = idx
            continue

        end_match = _RUN_SUITE_END_RE.search(line)
        if end_match and current_name is not None and current_start is not None:
            if end_match.group(1) == current_name:
                sections.append({"test_name": current_name, "start_line": current_start, "end_line": idx + 1})
                current_name = None
                current_start = None

    if current_name is not None and current_start is not None:
        sections.append({"test_name": current_name, "start_line": current_start, "end_line": len(lines)})

    return sections


def _find_section_for_case(
    sections: list[dict[str, Any]], total_lines: int, test_case: str
) -> tuple[dict[str, Any] | None, int, int]:
    base_case = _base_case_name(test_case)
    test_file = test_case.split("::")[0]

    for section in sections:
        if section["test_name"] == base_case:
            return section, section["start_line"], section["end_line"]
    for section in sections:
        if section["test_name"] == test_file:
            return section, section["start_line"], section["end_line"]

    return None, 0, total_lines


def _find_case_anchor(
    lines: list[str], test_case: str, section: dict[str, Any] | None, start: int, end: int
) -> int | None:
    if section is not None and "::" in section["test_name"]:
        return start
    full_hits: list[int] = []
    base_hits: list[int] = []
    base_case = _base_case_name(test_case)
    for idx in range(start, end):
        line = clean_line(lines[idx])
        if test_case in line:
            full_hits.append(idx)
        elif base_case in line:
            base_hits.append(idx)
    mentions = full_hits or base_hits
    if not mentions:
        return None
    return min(mentions)


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
    if _match_error_line(stripped) is not None:
        return True
    if _RUN_SUITE_START_RE.search(stripped) or _RUN_SUITE_END_RE.search(stripped):
        return False
    if _PYTEST_FAILURES_BANNER_RE.match(stripped) or _PYTEST_SUMMARY_BANNER_RE.match(stripped):
        return False
    if _PYTEST_FAILURE_HEADER_RE.match(stripped):
        return False
    return not (stripped.startswith("FAILED tests/") or stripped.startswith("tests/"))


def _iter_traceback_blocks(lines: list[str], start: int, end: int) -> list[tuple[int, int]]:
    blocks: list[tuple[int, int]] = []
    idx = start
    while idx < end:
        cleaned = _clean_context_line(lines[idx])
        if "Traceback (most recent call last):" not in cleaned:
            idx += 1
            continue
        block_end = idx + 1
        while block_end < end:
            next_cleaned = _clean_context_line(lines[block_end])
            if not _is_tracebackish_line(next_cleaned):
                break
            block_end += 1
        blocks.append((idx, block_end))
        idx = block_end
    return blocks


def _build_error(
    error_type: str,
    error_message: str,
    category: str,
    context: list[str],
    *,
    line_number: int,
    source: str,
    test_case: str,
) -> dict[str, Any]:
    return {
        "error_type": error_type,
        "error_message": error_message,
        "category": category,
        "context": _compress_context(context),
        "line_number": line_number,
        "source": source,
        "failed_test_files": [test_case.split("::")[0]],
        "failed_test_cases": [test_case],
    }


def _first_traceback_candidate(
    lines: list[str], block_start: int, block_end: int
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    wrapper_candidate = None
    for idx in range(block_start, block_end):
        line = _clean_context_line(lines[idx])
        matched = _match_error_line(line)
        if not matched:
            continue
        error_type, raw_error_message = matched
        error_message, category = _normalize_error_match(error_type, raw_error_message)
        context = [_clean_context_line(lines[j]) for j in range(block_start, idx + 1)]
        candidate = {
            "error_type": error_type,
            "error_message": error_message,
            "category": category,
            "context": context,
            "line_number": idx,
        }
        if _is_wrapper_error(error_type, error_message):
            if wrapper_candidate is None:
                wrapper_candidate = candidate
            continue
        return candidate, wrapper_candidate
    return None, wrapper_candidate


def _find_traceback_error_for_case(
    lines: list[str], test_case: str, section: dict[str, Any] | None, start: int, end: int
) -> dict[str, Any] | None:
    anchor = _find_case_anchor(lines, test_case, section, start, end)
    if anchor is None:
        return None

    wrapper_fallback = None
    for block_start, block_end in _iter_traceback_blocks(lines, anchor, end):
        candidate, block_wrapper = _first_traceback_candidate(lines, block_start, block_end)
        if candidate is not None:
            return _build_error(
                candidate["error_type"],
                candidate["error_message"],
                candidate["category"],
                candidate["context"],
                line_number=candidate["line_number"],
                source="case_traceback",
                test_case=test_case,
            )
        if wrapper_fallback is None and block_wrapper is not None:
            wrapper_fallback = block_wrapper

    if wrapper_fallback is None:
        return None
    return _build_error(
        wrapper_fallback["error_type"],
        wrapper_fallback["error_message"],
        wrapper_fallback["category"],
        wrapper_fallback["context"],
        line_number=wrapper_fallback["line_number"],
        source="case_traceback",
        test_case=test_case,
    )


def _find_failure_block_context_for_case(
    lines: list[str], test_case: str, start: int, end: int, error_type: str, error_message: str
) -> tuple[list[str], int] | None:
    sub_lines = lines[start:end]
    full_error = f"{error_type}: {error_message}"
    for block in _extract_pytest_failure_blocks(sub_lines):
        header_line = sub_lines[block["start_line"]]
        if not _header_matches_case(header_line, test_case):
            continue

        match_idx = None
        for rel_idx in range(block["start_line"], block["end_line"]):
            line = _clean_context_line(sub_lines[rel_idx])
            if full_error in line:
                match_idx = rel_idx
                break
            if match_idx is None and error_message in line:
                match_idx = rel_idx
            if match_idx is None and line.lstrip().startswith("E") and error_type in line:
                match_idx = rel_idx

        if match_idx is None:
            continue

        context = [_clean_context_line(sub_lines[j]) for j in range(block["start_line"], match_idx + 1)]
        return context, start + match_idx

    return None


def _summary_entry_map(log_text: str) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for line in _iter_pytest_summary_lines(log_text):
        match = _FAILED_SUMMARY_PAYLOAD_RE.match(line)
        if match is None:
            continue
        test_case = match.group(1)
        payloads[test_case] = {
            "line": line,
            "extracted": _extract_summary_error_info(line),
        }
    return payloads


def _find_summary_payload_error_for_case(
    test_case: str,
    entry: dict[str, Any] | None,
    lines: list[str],
    section: dict[str, Any] | None,
    start: int,
    end: int,
) -> dict[str, Any] | None:
    if entry is None:
        return None
    line = entry["line"]
    extracted = entry["extracted"]
    if extracted is None:
        return None
    _name, error_type, error_message, category = extracted
    context = [line]
    line_number = 0
    block_context = _find_failure_block_context_for_case(lines, test_case, start, end, error_type, error_message)
    if block_context is not None:
        context, line_number = block_context
    return _build_error(
        error_type,
        error_message,
        category,
        context,
        line_number=line_number,
        source="case_summary_payload",
        test_case=test_case,
    )


def _payload_traceback_error(payload: str) -> tuple[str, str, str] | None:
    payload_lines = payload.replace("\\n", "\n").replace("\\r", "\n").splitlines()
    for block_start, block_end in _iter_traceback_blocks(payload_lines, 0, len(payload_lines)):
        candidate, wrapper_candidate = _first_traceback_candidate(payload_lines, block_start, block_end)
        if candidate is not None:
            return candidate["error_type"], candidate["error_message"], candidate["category"]
        if wrapper_candidate is not None:
            return (
                wrapper_candidate["error_type"],
                wrapper_candidate["error_message"],
                wrapper_candidate["category"],
            )
    return None


def _find_summary_fallback_error_for_case(test_case: str, entry: dict[str, Any] | None) -> dict[str, Any] | None:
    if entry is None:
        return None
    line = entry["line"]

    payload_match = _FAILED_SUMMARY_PAYLOAD_RE.match(line)
    if payload_match is None:
        return None
    payload = payload_match.group(2).strip()

    payload_tb_error = _payload_traceback_error(payload)
    if payload_tb_error is not None:
        error_type, error_message, category = payload_tb_error
        return _build_error(
            error_type,
            error_message,
            category,
            [f"{error_type}: {error_message}"],
            line_number=0,
            source="case_summary_fallback",
            test_case=test_case,
        )

    payload_matches = _iter_payload_error_matches(payload)
    wrapper_candidate = None
    for error_type, raw_error_message in payload_matches:
        error_message, category = _normalize_error_match(error_type, raw_error_message)
        if _is_wrapper_error(error_type, error_message):
            if wrapper_candidate is None:
                wrapper_candidate = (error_type, error_message, category)
            continue
        return _build_error(
            error_type,
            error_message,
            category,
            [f"{error_type}: {error_message}"],
            line_number=0,
            source="case_summary_fallback",
            test_case=test_case,
        )

    if wrapper_candidate is not None and not payload.startswith("assert "):
        error_type, error_message, category = wrapper_candidate
        return _build_error(
            error_type,
            error_message,
            category,
            [f"{error_type}: {error_message}"],
            line_number=0,
            source="case_summary_fallback",
            test_case=test_case,
        )

    return _build_error(
        "SummaryFailure",
        payload[:1200],
        "Code Bug",
        [line],
        line_number=0,
        source="case_summary_fallback",
        test_case=test_case,
    )


def _extract_case_first_errors(log_text: str, failed_test_cases: list[str]) -> list[dict[str, Any]]:
    lines = log_text.splitlines()
    sections = _build_invocation_sections(log_text)
    summary_entries = _summary_entry_map(log_text)
    errors: list[dict[str, Any]] = []

    for test_case in failed_test_cases:
        section, start, end = _find_section_for_case(sections, len(lines), test_case)
        error = _find_traceback_error_for_case(lines, test_case, section, start, end)
        if error is None:
            error = _find_summary_payload_error_for_case(
                test_case, summary_entries.get(test_case), lines, section, start, end
            )
        if error is None:
            error = _find_summary_fallback_error_for_case(test_case, summary_entries.get(test_case))
        if error is not None:
            errors.append(error)

    return errors


def extract_bad_commit(log_text: str, *, resolve_remote: bool = True) -> str | None:
    match = _VLLM_VERSION_RE.search(log_text)
    if match:
        short_sha = match.group(1)
        if not resolve_remote or shutil.which("gh") is None:
            return short_sha
        try:
            data = gh_api_json(f"/repos/vllm-project/vllm/commits/{short_sha}")
            return data.get("sha")
        except SystemExit:
            return short_sha
    return None


def get_good_commit() -> str | None:
    commit_re = re.compile(r"^[0-9a-f]{7,40}$")
    yaml_files = [
        ".github/workflows/pr_test_full.yaml",
        ".github/workflows/pr_test_light.yaml",
    ]

    for yaml_rel in yaml_files:
        try:
            repo_root = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            disk_path = Path(repo_root) / yaml_rel
            if disk_path.exists():
                content = disk_path.read_text()
                match = re.search(r"vllm_version:\s*\[([^\]]+)\]", content)
                if match:
                    entries = [entry.strip().strip("'\"") for entry in match.group(1).split(",")]
                    for entry in entries:
                        if commit_re.match(entry):
                            return entry
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            pass

        try:
            result = subprocess.run(
                ["git", "show", f"origin/main:{yaml_rel}"], capture_output=True, text=True, check=True
            )
            match = re.search(r"vllm_version:\s*\[([^\]]+)\]", result.stdout)
            if match:
                entries = [entry.strip().strip("'\"") for entry in match.group(1).split(",")]
                for entry in entries:
                    if commit_re.match(entry):
                        return entry
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    return None


def _dedupe_errors_by_scope(errors: list[dict]) -> list[dict]:
    seen: dict[tuple[Any, ...], dict] = {}
    for error in errors:
        key = (
            error["error_type"],
            error["error_message"],
            tuple(error.get("failed_test_files", [])),
            tuple(error.get("failed_test_cases", [])),
        )
        if key not in seen or error.get("line_number", 0) < seen[key].get("line_number", 0):
            seen[key] = copy.deepcopy(error)

    deduped = list(seen.values())
    for error in deduped:
        error["error_failed_test_files_count"] = len(error.get("failed_test_files", []))
        error["error_failed_test_cases_count"] = len(error.get("failed_test_cases", []))
    return deduped


def _dedupe_errors(all_errors: list[dict]) -> list[dict]:
    seen_sigs = {}
    for error in all_errors:
        signature = f"{error['error_type']}:{error['error_message']}"
        if signature not in seen_sigs:
            seen_sigs[signature] = {
                "error": copy.deepcopy(error),
                "failed_test_files": set(),
                "failed_test_cases": set(),
            }
        for test_file in error.get("failed_test_files", []):
            seen_sigs[signature]["failed_test_files"].add(test_file)
        for test_case in error.get("failed_test_cases", []):
            seen_sigs[signature]["failed_test_cases"].add(test_case)

    unique_errors = []
    for data in seen_sigs.values():
        error = data["error"]
        error["failed_test_files"] = sorted(data["failed_test_files"])
        error["failed_test_cases"] = sorted(data["failed_test_cases"])
        error["error_failed_test_files_count"] = len(error["failed_test_files"])
        error["error_failed_test_cases_count"] = len(error["failed_test_cases"])
        unique_errors.append(error)
    return unique_errors


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

        matched_specific = any(
            case_to_specific_errors.get(test_case) for test_case in error.get("failed_test_cases", [])
        )
        if not matched_specific:
            matched_specific = any(
                file_to_specific_errors.get(test_file) for test_file in error.get("failed_test_files", [])
            )
        if not matched_specific:
            filtered.append(error)

    return filtered


def process_local_log(log_text: str, job_name: str = "local-log") -> dict:
    failed_test_cases = extract_failed_test_cases(log_text)
    failed_test_files = sorted({test_case.split("::")[0] for test_case in failed_test_cases})
    if failed_test_cases:
        errors = _extract_case_first_errors(log_text, failed_test_cases)
    else:
        errors = []

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
        "code_bugs": [error for error in unique_errors if error["category"] == "Code Bug"],
        "env_flakes": [error for error in unique_errors if error["category"] == "Environment Flake"],
    }


def process_run(run_id: int, repo: str = REPO) -> dict:
    run_info = gh_api_json(f"/repos/{repo}/actions/runs/{run_id}")
    all_jobs_data = gh_api_json(f"/repos/{repo}/actions/runs/{run_id}/jobs", per_page="100")
    all_jobs = all_jobs_data.get("jobs", [])
    candidate_jobs = [
        job for job in all_jobs if job.get("status") == "completed" and job.get("conclusion") != "skipped"
    ]

    good_commit = get_good_commit()
    bad_commit = None
    all_failed_test_files: list[str] = []
    all_failed_test_cases: list[str] = []
    all_errors: list[dict[str, Any]] = []
    job_results: list[dict[str, Any]] = []

    for job in candidate_jobs:
        job_id = job["id"]
        job_name = job["name"]
        log_text = gh_api_raw(f"/repos/{repo}/actions/jobs/{job_id}/logs")
        if not log_text:
            if job.get("conclusion") == "failure":
                job_results.append({"job_id": job_id, "job_name": job_name, "error": "Failed to download log"})
            continue

        if bad_commit is None:
            bad_commit = extract_bad_commit(log_text)

        local_result = process_local_log(log_text, job_name=job_name)
        job_scoped_errors = local_result["job_results"][0]["errors"]
        has_failure_signal = bool(
            local_result["failed_test_files"] or local_result["failed_test_cases"] or job_scoped_errors
        )
        if not has_failure_signal and job.get("conclusion") != "failure":
            continue

        all_failed_test_files.extend(local_result["failed_test_files"])
        all_failed_test_cases.extend(local_result["failed_test_cases"])
        all_errors.extend(job_scoped_errors)
        job_results.append(
            {
                "job_id": job_id,
                "job_name": job_name,
                "failed_test_files": local_result["failed_test_files"],
                "failed_test_cases": local_result["failed_test_cases"],
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
        "failed_jobs_count": len(job_results),
        "job_summary": [{"name": job["name"], "conclusion": job.get("conclusion", "unknown")} for job in all_jobs],
        "job_results": job_results,
        "failed_test_files": unique_failed_test_files,
        "failed_test_cases": unique_failed_test_cases,
        "distinct_errors": unique_errors,
        "code_bugs": [error for error in unique_errors if error["category"] == "Code Bug"],
        "env_flakes": [error for error in unique_errors if error["category"] == "Environment Flake"],
    }


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
        lines.extend(["   Context:", "   ```text", *[f"   {line}" for line in context], "   ```"])

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
            ["### Failed Tests", "", "Files:", "", *[f"- `{test}`" for test in result["failed_test_files"]], ""]
        )
    if result["failed_test_cases"]:
        lines.extend(["Cases:", "", *[f"- `{test}`" for test in result["failed_test_cases"]], ""])

    if result["distinct_errors"]:
        lines.extend(["### Distinct Issues", ""])
        for index, error in enumerate(result["distinct_errors"], start=1):
            lines.extend(_format_error_block(index, error))
            lines.append("")
    else:
        lines.extend(["### Notes", "", "- No root-cause exception was extracted from the input log.", ""])

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GitHub job summary from a local test log or workflow run.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--log-file", type=Path, help="Path to the local test log file.")
    source.add_argument("--run-id", type=int, help="GitHub Actions run ID to analyze through gh api.")
    parser.add_argument("--repo", default=REPO, help=f"GitHub repo for --run-id mode (default: {REPO}).")
    parser.add_argument(
        "--mode", default="e2e", choices=("ut", "e2e"), help="Test mode for the summary (default: e2e)."
    )
    parser.add_argument(
        "--step-name", default="Run test", help="Workflow step name shown in the summary (default: Run test)."
    )
    parser.add_argument(
        "--format", choices=("summary", "json", "llm-json"), default="summary", help="Output format (default: summary)."
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Optional output file path. If omitted, prints to stdout."
    )
    args = parser.parse_args()

    if args.run_id is not None:
        result = process_run(args.run_id, repo=args.repo)
    else:
        if not args.log_file.exists() or args.log_file.stat().st_size == 0:
            return
        log_text = args.log_file.read_text(encoding="utf-8", errors="replace")
        result = process_local_log(log_text, job_name=args.step_name)

    if args.format == "json":
        rendered_output = render_json(result)
    elif args.format == "llm-json":
        rendered_output = render_llm_json(result)
    else:
        if not (result["failed_test_files"] or result["failed_test_cases"] or result["distinct_errors"]):
            return
        rendered_output = render_summary(result, step_name=args.step_name, mode=args.mode)

    if args.output is not None:
        args.output.write_text(rendered_output, encoding="utf-8")
    else:
        print(rendered_output, end="")


if __name__ == "__main__":
    main()
