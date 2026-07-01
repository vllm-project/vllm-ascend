"""
CI AI diagnosis agent — multi-round agent-loop mode with function calling.

This script pre-filters the main CI log and presents the LLM with a
filtered overview plus a file index.  The LLM then explores additional
log files via function-calling tools (list_dir / read_file / search)
over multiple rounds, submitting a final diagnosis via a diagnose tool.

Modes:
- Disabled / missing config: emit skip note, exit 0.
- Live LLM call: multi-round agent diagnosis driven by function calling.
- The agent NEVER breaks CI: every failure path exits 0.

Configuration (env vars, prefixed by VLLM_ASCEND_CI_AI_DIAGNOSIS_*):

    VLLM_ASCEND_CI_AI_DIAGNOSIS_ENABLED            default 0
    VLLM_ASCEND_CI_AI_DIAGNOSIS_API_KEY            required to call LLM
    VLLM_ASCEND_CI_AI_DIAGNOSIS_BASE_URL           e.g. https://api.openai.com/v1
    VLLM_ASCEND_CI_AI_DIAGNOSIS_MODEL              required for LLM (repo variable)
    VLLM_ASCEND_CI_AI_DIAGNOSIS_BACKEND            default openai_compatible
    VLLM_ASCEND_CI_AI_DIAGNOSIS_TIMEOUT_S          default 120
    VLLM_ASCEND_CI_AI_DIAGNOSIS_MAX_INPUT_CHARS    default 120000
    VLLM_ASCEND_CI_AI_DIAGNOSIS_TAIL_CHARS         default 200000 (tail size for large files)
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import regex as re

# Ensure sibling scripts are importable when run as a CLI from the
# skill scripts directory.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from ci_log_filter_llm import build_llm_log_bundle, clip_text  # noqa: E402

SCHEMA_VERSION = "1.0"
ALLOWED_STAGES = {"setup", "collection", "startup", "runtime", "assertion", "teardown", "infra", "unknown"}
ALLOWED_LAYERS = {
    "ci_workflow",
    "pytest",
    "dependency",
    "vllm_engine",
    "vllm_ascend",
    "npu_runtime",
    "kubernetes",
    "network",
    "storage",
    "external_service",
    "unknown",
}
ALLOWED_CONF = {"high", "medium", "low"}
ALLOWED_CLASS = {"flake", "test_bug", "product_bug", "infra_issue", "unknown"}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    enabled: bool
    api_key: str
    base_url: str
    model: str
    backend: str
    max_rounds: int
    timeout_s: int
    max_input_chars: int
    tail_chars: int

    @classmethod
    def from_env(cls) -> AgentConfig:
        return cls(
            enabled=os.getenv("VLLM_ASCEND_CI_AI_DIAGNOSIS_ENABLED", "0").lower() in ("1", "true", "yes", "on"),
            api_key=os.getenv("VLLM_ASCEND_CI_AI_DIAGNOSIS_API_KEY", "").strip(),
            base_url=os.getenv("VLLM_ASCEND_CI_AI_DIAGNOSIS_BASE_URL", "").strip(),
            model=os.getenv("VLLM_ASCEND_CI_AI_DIAGNOSIS_MODEL", "").strip(),
            backend=os.getenv("VLLM_ASCEND_CI_AI_DIAGNOSIS_BACKEND", "openai_compatible").strip(),
            max_rounds=int(os.getenv("VLLM_ASCEND_CI_AI_DIAGNOSIS_MAX_ROUNDS", "3")),
            timeout_s=int(os.getenv("VLLM_ASCEND_CI_AI_DIAGNOSIS_TIMEOUT_S", "30")),
            max_input_chars=int(os.getenv("VLLM_ASCEND_CI_AI_DIAGNOSIS_MAX_INPUT_CHARS", "120000")),
            tail_chars=int(os.getenv("VLLM_ASCEND_CI_AI_DIAGNOSIS_TAIL_CHARS", "200000")),
        )

    def llm_ready(self) -> bool:
        """True when repo configuration is complete enough to call the LLM."""
        return (
            self.enabled
            and bool(self.api_key)
            and bool(self.base_url)
            and bool(self.model)
            and self.backend == "openai_compatible"
        )

    def missing_llm_config(self) -> list[str]:
        """Human-readable list of unset CI LLM settings."""
        missing: list[str] = []
        if not self.enabled:
            missing.append("VLLM_ASCEND_CI_AI_DIAGNOSIS_ENABLED (must be 1)")
        if not self.api_key:
            missing.append("VLLM_ASCEND_CI_AI_DIAGNOSIS_API_KEY (secret)")
        if not self.base_url:
            missing.append("VLLM_ASCEND_CI_AI_DIAGNOSIS_BASE_URL (secret)")
        if not self.model:
            missing.append("VLLM_ASCEND_CI_AI_DIAGNOSIS_MODEL (variable)")
        if self.backend != "openai_compatible":
            missing.append(f"VLLM_ASCEND_CI_AI_DIAGNOSIS_BACKEND (unsupported: {self.backend!r})")
        return missing


# ---------------------------------------------------------------------------
# LLM backend
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = (
    "You are a CI diagnosis agent for vllm-ascend (Huawei Ascend NPU + vLLM).\n"
    "You diagnose CI test failures by exploring log files interactively.\n\n"
    "## Workflow\n"
    "1. Start with the filtered main-log overview provided in the first user message.\n"
    "2. If the root cause is clear from the overview, call `diagnose` immediately.\n"
    "3. If not, use `list_dir` / `read_file` / `search` to explore additional log files.\n"
    "4. As soon as you have sufficient evidence, call `diagnose` to submit your conclusion.\n"
    "5. Do NOT explore aimlessly — every file read must have a clear purpose.\n\n"
    "## Rules\n"
    "- Pytest wrapper errors like 'Server exited unexpectedly' or 'FAILED tests/...'\n"
    "  are SYMPTOMS, never root causes. Trace back to the upstream error.\n"
    "- root_cause must reference specific file names and line numbers from the logs.\n"
    "- If you cannot identify a root cause, set confidence=low and needs_human_review=true.\n"
    "- DO NOT invent root causes when evidence is insufficient.\n"
    "- Reply in English for all output.\n\n"
    "## Output\n"
    "When calling `diagnose`, provide ALL of these fields:\n"
    "- failure_family: pytest|engine_startup|benchmark|infra|unknown\n"
    "- root_cause: one sentence describing the first non-wrapper cause\n"
    "- classification: flake|test_bug|product_bug|infra_issue|unknown\n"
    "- confidence: high|medium|low\n"
    "- failure_stage: setup|collection|startup|runtime|assertion|teardown|infra|unknown\n"
    "- failure_layer: ci_workflow|pytest|dependency|vllm_engine|vllm_ascend|npu_runtime|\n"
    "  kubernetes|network|storage|external_service|unknown\n"
    "- visible_failure: the outermost error in the main log\n"
    '- evidence: [{"file":"...", "line":N, "snippet":"...", "interpretation":"..."}, ...]\n'
    "- needs_human_review: true|false"
)


def _log(stage: str, msg: str) -> None:
    """Stage-tagged progress log. Goes to stderr so it is visible even
    when the main output is a JSON file. Keep it machine-greppable."""
    sys.stderr.write(f"[ci-ai-diagnosis] {stage}: {msg}\n")
    sys.stderr.flush()


def _chat_one(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    timeout_s: int,
    max_input_chars: int,
    stream: bool = False,
) -> dict[str, Any]:
    """Single LLM call returning the full chat completion response dict.

    When ``tools`` is provided, ``tool_choice: "auto"`` is set so the model
    may return ``tool_calls`` instead of a plain text message.

    When ``stream`` is True (only for simple/display uses, NOT for agent
    loop), the response is reconstructed from SSE chunks.  The return value
    is still a dict to match the non-stream path.
    """
    if not base_url:
        raise RuntimeError("base_url is empty")
    if not api_key:
        raise RuntimeError("api_key is empty")
    url = base_url.rstrip("/") + "/chat/completions"
    msg_count = len(messages)
    tool_count = len(tools) if tools else 0
    stream_flag = bool(stream)
    _log("chat", f"url={url} model={model} messages={msg_count} tools={tool_count} stream={stream_flag}")
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "stream": stream_flag,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    body = json.dumps(payload).encode("utf-8")
    body_chars = len(body)
    _log("chat", f"request body bytes={body_chars} cap={max_input_chars * 2}")
    if body_chars > max_input_chars * 2:
        raise RuntimeError(f"request body too large: {body_chars} chars")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Accept": "text/event-stream" if stream else "application/json",
        },
    )
    started = _dt.datetime.now(_dt.timezone.utc)
    try:
        resp_ctx = urllib.request.urlopen(req, timeout=timeout_s)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        elapsed = (_dt.datetime.now(_dt.timezone.utc) - started).total_seconds()
        _log("chat", f"HTTP {exc.code} after {elapsed:.2f}s: {detail[:200]}")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        elapsed = (_dt.datetime.now(_dt.timezone.utc) - started).total_seconds()
        _log("chat", f"URLError after {elapsed:.2f}s: {exc.reason}")
        raise RuntimeError(f"URLError from {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        elapsed = (_dt.datetime.now(_dt.timezone.utc) - started).total_seconds()
        _log("chat", f"timeout after {elapsed:.2f}s (limit={timeout_s}s)")
        raise RuntimeError(f"timeout after {timeout_s}s calling {url}") from exc

    try:
        with resp_ctx as resp:
            if stream:
                # NOTE: stream mode only supports delta.content, NOT
                # delta.tool_calls.  It is safe for display/logging calls
                # but MUST NOT be used during the agent loop.
                content_parts: list[str] = []
                first_byte_at: _dt.datetime | None = None
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                    if not line:
                        continue
                    if line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    payload_str = line[5:].strip()
                    if payload_str == "[DONE]":
                        break
                    if first_byte_at is None:
                        first_byte_at = _dt.datetime.now(_dt.timezone.utc)
                        elapsed = (first_byte_at - started).total_seconds()
                        _log("stream", f"first byte in {elapsed:.2f}s")
                    try:
                        obj = json.loads(payload_str)
                    except json.JSONDecodeError:
                        continue
                    for choice in obj.get("choices") or []:
                        delta = (choice.get("delta") or {}).get("content")
                        if delta:
                            content_parts.append(delta)
                content = "".join(content_parts)
                elapsed = (_dt.datetime.now(_dt.timezone.utc) - started).total_seconds()
                _log("chat", f"stream done chars={len(content)} elapsed={elapsed:.2f}s")
                return {"choices": [{"message": {"role": "assistant", "content": content}}]}

            raw = resp.read().decode("utf-8", errors="replace")
            elapsed = (_dt.datetime.now(_dt.timezone.utc) - started).total_seconds()
            resp_chars = len(raw)
            _log("chat", f"response chars={resp_chars} elapsed={elapsed:.2f}s")
    except OSError:
        raise
    try:
        result = json.loads(raw) or {}
        choices = result.get("choices") or []
        finish_reason = choices[0].get("finish_reason", "") if choices else ""
        has_tool_calls = bool((choices[0].get("message") or {}).get("tool_calls") if choices else False)
        _log("chat", f"parsed ok finish_reason={finish_reason} has_tool_calls={has_tool_calls}")
        return result
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"non-JSON response from {url}: {exc}") from exc


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _strip_text(s: str, cap: int) -> str:
    if len(s) <= cap:
        return s
    head = cap // 2
    tail = cap - head - 80
    return s[:head] + f"\n... [truncated at {cap} chars] ...\n" + s[-tail:]


def _parse_json_block(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a markdown-style fenced block.
    Falls back to the first balanced { ... } if no fence is present."""
    m = _JSON_FENCE.search(text)
    candidate = m.group(1) if m else None
    if candidate is None:
        # try to find a balanced JSON object
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    break
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _extract_json_blocks(text: str) -> list[dict[str, Any]]:
    """Extract all valid JSON objects from markdown fenced blocks.

    Uses balanced-bracket matching to correctly handle nested JSON
    (e.g. an array of hypothesis objects inside a HYPOTHESES block),
    unlike the non-greedy regex in ``_parse_json_block``.

    Returns successfully parsed dicts in order of appearance;
    non-dict JSON values are silently dropped.
    """
    blocks: list[dict[str, Any]] = []
    for m in re.finditer(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL):
        for obj_text in _balanced_json_objects(m.group(1)):
            try:
                obj = json.loads(obj_text)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                blocks.append(obj)
    return blocks


def _balanced_json_objects(text: str):
    """Yield balanced { ... } JSON object strings from *text*.
    Handles escaped quotes and nested braces."""
    i = 0
    n = len(text)
    while i < n:
        start = text.find("{", i)
        if start < 0:
            break
        depth = 0
        in_string = False
        escape = False
        for j in range(start, n):
            ch = text[j]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield text[start : j + 1]
                    i = j + 1
                    break
        else:
            break  # no closing brace for this object; stop


# ---------------------------------------------------------------------------
# Schema validation / sanitation
# ---------------------------------------------------------------------------


def _ensure_str_list(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    return [str(x)]


def _sanitize_diagnosis(raw: dict[str, Any], routing: dict[str, Any], failed_tests: list[str]) -> dict[str, Any]:
    classification = raw.get("classification") if raw.get("classification") in ALLOWED_CLASS else "unknown"
    confidence = raw.get("confidence") if raw.get("confidence") in ALLOWED_CONF else "low"
    # root_cause may arrive as a dict {type, description} — extract the human part.
    rc = raw.get("root_cause")
    if isinstance(rc, dict):
        rc = str(rc.get("description") or rc.get("hypothesis") or rc.get("detail") or rc.get("summary") or "")
    root_cause = str(rc or "").strip()
    return {
        "routing": routing,
        "failure_family": str(raw.get("failure_family") or "unknown"),
        "root_cause": root_cause,
        "classification": classification,
        "confidence": confidence,
        "evidence": _ensure_evidence_list(raw.get("evidence")),
        "counter_evidence": _ensure_evidence_list(raw.get("counter_evidence")),
        "wrapper_errors": routing.get("wrapper_errors", []),
        "failed_tests": failed_tests,
        "matched_playbooks": _ensure_str_list(raw.get("matched_playbooks")),
        "next_actions": _ensure_actions(raw.get("next_actions")),
        "needs_human_review": bool(raw.get("needs_human_review", True)) or confidence == "low",
    }


def _ensure_evidence_list(x: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(x, list):
        return out
    for e in x:
        if not isinstance(e, dict):
            continue
        out.append(
            {
                "file": str(e.get("file") or ""),
                "line": int(e.get("line") or 0),
                "snippet": str(e.get("snippet") or "")[:240],
                "interpretation": str(e.get("interpretation") or ""),
            }
        )
    return out


def _ensure_actions(x: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(x, list):
        return out
    for a in x:
        if not isinstance(a, dict):
            continue
        pr = a.get("priority") if a.get("priority") in {"P0", "P1", "P2"} else "P1"
        out.append(
            {
                "priority": pr,
                "action": str(a.get("action") or ""),
                "command": str(a.get("command") or ""),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Skip / fallback
# ---------------------------------------------------------------------------


def _skip_diagnosis(reason: str, step_name: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "step_name": step_name,
        "log_file": "",
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "routing": {
            "failure_stage": "unknown",
            "failure_layer": "unknown",
            "visible_failure": reason,
            "first_failure_signal": "",
            "wrapper_errors": [],
            "candidate_routes": [],
            "evidence_requests": [],
            "matched_playbooks": [],
        },
        "failure_family": "unknown",
        "root_cause": "diagnosis skipped",
        "classification": "unknown",
        "confidence": "low",
        "evidence": [],
        "counter_evidence": [],
        "wrapper_errors": [],
        "failed_tests": [],
        "matched_playbooks": [],
        "next_actions": [
            {"priority": "P0", "action": "Investigate manually", "command": ""},
        ],
        "needs_human_review": True,
    }


def _make_evidence_only_diagnosis(
    *,
    step_name: str,
    log_file: Path,
    git_context: dict[str, Any] | None,
    missing_config: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "type": "evidence_only",
        "step_name": step_name,
        "log_file": str(log_file),
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git_context": git_context,
        "llm_analysis_performed": False,
        "missing_llm_config": missing_config,
        "routing": {
            "failure_stage": "unknown",
            "failure_layer": "unknown",
            "visible_failure": "",
            "first_failure_signal": "",
            "wrapper_errors": [],
            "candidate_routes": [],
            "evidence_requests": [],
            "matched_playbooks": [],
        },
        "failure_family": "unknown",
        "root_cause": "LLM analysis not performed (CI secrets/variables not configured)",
        "classification": "unknown",
        "confidence": "low",
        "evidence": [],
        "counter_evidence": [],
        "wrapper_errors": [],
        "failed_tests": [],
        "matched_playbooks": [],
        "next_actions": [
            {
                "priority": "P0",
                "action": "Configure CI LLM settings and re-run",
                "command": "",
            },
        ],
        "needs_human_review": True,
    }


def render_evidence_summary(diag: dict[str, Any]) -> str:
    """Markdown for evidence-only mode (no LLM root-cause claim)."""
    missing = diag.get("missing_llm_config") or []
    lines: list[str] = []
    lines.append("## CI AI Diagnosis (LLM not called)")
    lines.append("")
    lines.append(f"**Step**: {diag.get('step_name') or ''}")
    lines.append(f"**Log**: `{diag.get('log_file') or ''}`")
    lines.append("")
    lines.append("### Reason for Skipping AI Analysis")
    if missing:
        for item in missing:
            lines.append(f"- `{item}` not configured")
    else:
        lines.append("- LLM configuration incomplete")
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Agent tools (function-calling schemas)
# ---------------------------------------------------------------------------

AGENT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and subdirectories in a given directory path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative or absolute path to the directory."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a log file with optional line offset and limit. "
                "Use this to drill into specific log files when the "
                "filtered overview is insufficient."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative or absolute path to the file."},
                    "offset": {
                        "type": "integer",
                        "description": "1-based line number to start reading from (default: 1).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read (default: 200, max: 500).",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search for a regex pattern in a log file. Returns matching lines with line numbers and context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative or absolute path to the file."},
                    "pattern": {
                        "type": "string",
                        "description": "Case-insensitive regex pattern to search for.",
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Number of context lines before and after each match (default: 3, max: 10).",
                    },
                },
                "required": ["path", "pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "diagnose",
            "description": (
                "Submit final diagnosis when root cause has been identified. "
                "Call this ONLY when you have sufficient evidence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "failure_family": {
                        "type": "string",
                        "description": "pytest|engine_startup|benchmark|infra|unknown",
                    },
                    "root_cause": {
                        "type": "string",
                        "description": "One sentence describing the first non-wrapper cause.",
                    },
                    "classification": {
                        "type": "string",
                        "description": "flake|test_bug|product_bug|infra_issue|unknown",
                    },
                    "confidence": {"type": "string", "description": "high|medium|low"},
                    "failure_stage": {
                        "type": "string",
                        "description": "setup|collection|startup|runtime|assertion|teardown|infra|unknown",
                    },
                    "failure_layer": {
                        "type": "string",
                        "description": (
                            "ci_workflow|pytest|dependency|vllm_engine|vllm_ascend|"
                            "npu_runtime|kubernetes|network|storage|external_service|unknown"
                        ),
                    },
                    "visible_failure": {
                        "type": "string",
                        "description": "The outermost error visible in the main log.",
                    },
                    "evidence": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "List of evidence items with file/line/snippet/interpretation.",
                    },
                    "needs_human_review": {
                        "type": "boolean",
                        "description": "True if diagnosis is uncertain and needs human review.",
                    },
                },
                "required": ["root_cause", "confidence"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Agent: log directory scanner
# ---------------------------------------------------------------------------

_FILE_INDEX_CHAR_CAP = 10_000


def _scan_log_dir(
    log_file: Path,
    artifact_dir: Path | None = None,
    k8s_dir: Path | None = None,
    benchmark_dir: Path | None = None,
) -> tuple[str, set[Path]]:
    """Walk known log directories and produce a human-readable file index.

    Returns (index_text, allowed_roots) where allowed_roots contains
    resolved parent directories used for path-sandbox validation.
    """
    _scan_start = _dt.datetime.now(_dt.timezone.utc)
    allowed_roots: set[Path] = set()
    lines: list[str] = ["## Available log files\n"]

    def _add_dir(label: str, root: Path | None) -> None:
        dir_start = _dt.datetime.now(_dt.timezone.utc)
        if root is None:
            _log("scan", f"dir={label} path=None (skipped)")
            return
        if not root.is_dir():
            _log("scan", f"dir={label} path={root} NOT_A_DIR (skipped)")
            return
        rp = root.resolve()
        allowed_roots.add(rp)
        _log("scan", f"dir={label} resolved={rp} scanning...")
        file_count = 0
        total_bytes = 0
        lines.append(f"\n### {label} (`{rp}`)\n")
        for f in sorted(rp.rglob("*")):
            if not f.is_file():
                continue
            file_count += 1
            try:
                size = f.stat().st_size
            except OSError:
                size = 0
            total_bytes += size
            rel = f.relative_to(rp)
            lines.append(f"  {rel}  ({_fmt_size(size)})")
        elapsed = (_dt.datetime.now(_dt.timezone.utc) - dir_start).total_seconds()
        _log(
            "scan",
            f"dir={label} files={file_count} total_size={_fmt_size(total_bytes)} elapsed={elapsed:.2f}s",
        )

    _add_dir("main-log", log_file.parent)
    _add_dir("artifact-logs", artifact_dir)
    _add_dir("k8s-diagnostics", k8s_dir)
    _add_dir("benchmark", benchmark_dir)

    text = "\n".join(lines)
    index_size = len(text)
    if index_size > _FILE_INDEX_CHAR_CAP:
        text = text[:_FILE_INDEX_CHAR_CAP] + f"\n... [truncated: {index_size} chars total]"
    _scan_total = (_dt.datetime.now(_dt.timezone.utc) - _scan_start).total_seconds()
    _log(
        "scan",
        f"done index_chars={min(index_size, _FILE_INDEX_CHAR_CAP)} "
        f"original_chars={index_size} "
        f"truncated={index_size > _FILE_INDEX_CHAR_CAP} "
        f"allowed_roots={len(allowed_roots)} "
        f"total_elapsed={_scan_total:.2f}s",
    )
    return text, allowed_roots


def _fmt_size(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} MB"
    if n >= 1_000:
        return f"{n / 1_000:.1f} KB"
    return f"{n} B"


# ---------------------------------------------------------------------------
# Agent: initial prompt builder
# ---------------------------------------------------------------------------


def _build_initial_user_prompt(
    step_name: str,
    filtered_log: str,
    file_index: str,
    user_hint: str | None = None,
    git_context: dict[str, Any] | None = None,
) -> str:
    """Assemble the first user message for the agent."""
    parts: list[str] = [f"## CI Run: {step_name}\n"]
    if git_context and git_context.get("commit"):
        c = git_context["commit"]
        parts.append(f"\n**Trigger commit**: {c.get('hash', '')[:8]} {c.get('subject', '')}\n")
        changed = git_context.get("changed_files") or []
        if changed:
            parts.append(f"**Changed files ({len(changed)})**: {', '.join(changed[:10])}\n")
    if user_hint:
        parts.append(f"\n**User hint**: {user_hint}\n")

    parts.append("\n---\n\n## Filtered Main-Log Overview (two-phase extraction)\n\n")
    parts.append(filtered_log)
    parts.append("\n---\n\n")
    parts.append(file_index)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Agent: tool executor + path sandbox
# ---------------------------------------------------------------------------

_TOOL_RESULT_CHAR_CAP = 20_000
_SEARCH_MAX_MATCHES = 50
_READ_MAX_LINES = 500
_READ_DEFAULT_LINES = 200


def _execute_tool_call(
    tool_call: dict[str, Any],
    allowed_roots: set[Path],
) -> str:
    """Execute a single tool call with path-sandbox enforcement.

    Returns a plain-text string suitable for a ``tool`` role message.
    """
    fn_name = (tool_call.get("function") or {}).get("name", "")
    try:
        args = json.loads((tool_call.get("function") or {}).get("arguments", "{}"))
    except json.JSONDecodeError:
        _log("tool", f"fn={fn_name} ERROR malformed args")
        return f"[ERROR] malformed tool arguments for {fn_name}"

    if fn_name == "diagnose":
        _log("tool", "fn=diagnose acknowledged")
        return "[diagnose acknowledged — agent loop will terminate]"

    if fn_name not in {"list_dir", "read_file", "search"}:
        _log("tool", f"fn={fn_name} ERROR unknown tool")
        return f"[ERROR] unknown tool: {fn_name}"

    raw_path = str(args.get("path") or "")
    resolved = _resolve_sandboxed(raw_path, allowed_roots)
    if resolved is None:
        _log("tool", f"fn={fn_name} raw_path={raw_path} RESOLVE_FAILED")
        return f"[ERROR] path not allowed or not resolvable within allowed roots: {raw_path}"

    if fn_name == "list_dir":
        _log("tool", f"fn=list_dir resolved={resolved}")
        return _tool_list_dir(resolved)

    if fn_name == "read_file":
        offset = int(args.get("offset") or 1)
        limit = int(args.get("limit") or _READ_DEFAULT_LINES)
        _log("tool", f"fn=read_file resolved={resolved} offset={offset} limit={limit}")
        return _tool_read_file(resolved, offset, limit)

    if fn_name == "search":
        pattern = str(args.get("pattern") or "")
        ctx = max(0, min(int(args.get("context_lines") or 3), 10))
        _log("tool", f"fn=search resolved={resolved} pattern={pattern[:80]} ctx={ctx}")
        return _tool_search(resolved, pattern, ctx)

    return "[ERROR] unreachable"


def _resolve_sandboxed(raw_path: str, allowed_roots: set[Path]) -> Path | None:
    """Resolve *raw_path* against *allowed_roots* and enforce the sandbox.

    - Relative paths are joined with the first root in which they exist.
    - Absolute paths are checked against every root.
    - ``..`` traversal is blocked via ``resolve()`` then prefix check.
    """
    # Absolute path
    if raw_path.startswith("/"):
        rp = Path(raw_path).resolve()
        for root in allowed_roots:
            try:
                rp.relative_to(root)
                ok = rp.is_file() or rp.is_dir()
                _log("sandbox", f"raw={raw_path} type=abs resolved={rp} root={root} ok={ok}")
                return rp if ok else None
            except ValueError:
                continue
        _log("sandbox", f"raw={raw_path} type=abs MATCHED_NO_ROOT")
        return None

    # Relative path — try each root
    for root in sorted(allowed_roots):
        candidate = (root / raw_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if candidate.is_file() or candidate.is_dir():
            _log("sandbox", f"raw={raw_path} type=rel resolved={candidate} root={root}")
            return candidate
    _log("sandbox", f"raw={raw_path} type=rel MATCHED_NO_ROOT (tried {len(allowed_roots)} roots)")
    return None


def _tool_list_dir(path: Path) -> str:
    if not path.is_dir():
        return f"[ERROR] not a directory: {path}"
    lines: list[str] = [f"## {path}\n"]
    for entry in sorted(path.iterdir()):
        kind = "DIR" if entry.is_dir() else "FILE"
        try:
            size = entry.stat().st_size if entry.is_file() else 0
        except OSError:
            size = 0
        lines.append(f"  [{kind}] {entry.name}  {_fmt_size(size)}")
    return _cap_result("\n".join(lines))


def _tool_read_file(path: Path, offset: int, limit: int) -> str:
    if not path.is_file():
        return f"[ERROR] not a file: {path}"
    limit = max(1, min(limit, _READ_MAX_LINES))
    offset = max(1, offset)
    try:
        file_bytes = path.stat().st_size
    except OSError:
        file_bytes = 0
    _log("tool", f"read_file path={path} file_size={_fmt_size(file_bytes)} offset={offset} limit={limit}")
    read_start = _dt.datetime.now(_dt.timezone.utc)
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            all_lines = fh.readlines()
    except OSError as exc:
        _log("tool", f"read_file path={path} ERROR {exc}")
        return f"[ERROR] cannot read {path}: {exc}"
    total = len(all_lines)
    start = offset - 1
    if start >= total:
        return f"[INFO] {path}: {total} lines total, offset {offset} is past EOF"
    end = min(start + limit, total)
    selected = all_lines[start:end]
    selected_chars = sum(len(line) for line in selected) + len(selected)
    header = f"## {path}  lines {start + 1}-{end} / {total} ({_fmt_size(selected_chars)})\n"
    body = "".join(f"  L{i + 1}:{line}" for i, line in enumerate(selected, start))
    result = _cap_result(header + body)
    read_elapsed = (_dt.datetime.now(_dt.timezone.utc) - read_start).total_seconds()
    _log(
        "tool",
        f"read_file path={path} total_lines={total} returned_chars={len(result)} elapsed={read_elapsed:.2f}s",
    )
    return result


def _tool_search(path: Path, pattern: str, context_lines: int) -> str:
    if not path.is_file():
        return f"[ERROR] not a file: {path}"
    if not pattern:
        return "[ERROR] empty search pattern"
    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        return f"[ERROR] invalid regex pattern: {exc}"
    try:
        file_bytes = path.stat().st_size
    except OSError:
        file_bytes = 0
    _log("tool", f"search path={path} file_size={_fmt_size(file_bytes)} pattern={pattern[:80]}")
    search_start = _dt.datetime.now(_dt.timezone.utc)
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            all_lines = fh.readlines()
    except OSError as exc:
        _log("tool", f"search path={path} ERROR {exc}")
        return f"[ERROR] cannot read {path}: {exc}"
    total = len(all_lines)
    matches: list[str] = []
    match_count = 0
    for i, line in enumerate(all_lines):
        if compiled.search(line):
            match_count += 1
            lo = max(0, i - context_lines)
            hi = min(total, i + context_lines + 1)
            block = "\n".join(f"  {'>>' if j == i else '  '} L{j + 1}:{all_lines[j].rstrip()}" for j in range(lo, hi))
            matches.append(f"--- match at L{i + 1} ---\n{block}")
            if len(matches) >= _SEARCH_MAX_MATCHES:
                matches.append(f"... [capped at {_SEARCH_MAX_MATCHES} matches]")
                break
    if not matches:
        return f"[INFO] no matches for '{pattern}' in {path}"
    result = _cap_result("\n\n".join(matches))
    search_elapsed = (_dt.datetime.now(_dt.timezone.utc) - search_start).total_seconds()
    _log(
        "tool",
        f"search path={path} total_lines={total} "
        f"total_matches={match_count} returned={len(matches) if matches else 0} "
        f"result_chars={len(result)} elapsed={search_elapsed:.2f}s",
    )
    return result


def _cap_result(text: str) -> str:
    if len(text) <= _TOOL_RESULT_CHAR_CAP:
        return text
    return text[:_TOOL_RESULT_CHAR_CAP] + (
        f"\n... [truncated at {_TOOL_RESULT_CHAR_CAP} chars, original {len(text)} chars]"
    )


# ---------------------------------------------------------------------------
# Agent: message budget management
# ---------------------------------------------------------------------------


def _estimate_chars(messages: list[dict[str, Any]]) -> int:
    return sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)


def _trim_history(
    messages: list[dict[str, Any]],
    keep_recent: int = 2,
) -> list[dict[str, Any]]:
    """Drop older assistant/tool turns, keeping system + first user + recent.

    Returns a new message list.  The system message and first user message
    are always preserved.
    """
    total_before = len(messages)
    chars_before = _estimate_chars(messages)
    if len(messages) <= 2:
        return list(messages)
    # system (index 0, if role == system) + first user (index 0 or 1)
    keep: list[dict[str, Any]] = []
    sys_idx: int | None = None
    first_user_idx: int | None = None
    for i, m in enumerate(messages):
        if m.get("role") == "system" and sys_idx is None:
            sys_idx = i
            keep.append(m)
        elif m.get("role") == "user" and first_user_idx is None:
            first_user_idx = i
            keep.append(m)
            break
    # recent turns (assistant + tool pairs from the end)
    tail: list[dict[str, Any]] = []
    turn_count = 0
    trimmed_turns = 0
    for m in reversed(messages):
        if m.get("role") in ("assistant", "tool"):
            tail.insert(0, m)
            if m.get("role") == "assistant":
                turn_count += 1
        else:
            break
        if turn_count >= keep_recent:
            break
    # Count how many assistant turns were dropped
    for m in messages:
        if m.get("role") == "assistant":
            trimmed_turns += 1
    trimmed_turns -= turn_count
    if trimmed_turns > 0:
        summary = {
            "role": "user",
            "content": (
                f"[CONTEXT TRIMMED: {trimmed_turns} earlier exploration turns were "
                f"removed to stay within the message budget. You already have the "
                f"filtered main-log overview. If you need results from trimmed "
                f"turns, re-read or re-search the relevant files. Otherwise, "
                f"call diagnose with what you have.]"
            ),
        }
        keep = keep + [summary]
    result = keep + tail
    chars_after = _estimate_chars(result)
    _log(
        "trim",
        f"messages={total_before}->{len(result)} "
        f"chars={chars_before}->{chars_after} "
        f"trimmed_turns={trimmed_turns} kept_turns={turn_count}",
    )
    return result


# ---------------------------------------------------------------------------
# Agent: diagnose-call detection
# ---------------------------------------------------------------------------


def _find_diagnose_call(tool_calls: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    """Return the first ``diagnose`` tool call arguments, or None."""
    if not tool_calls:
        return None
    for tc in tool_calls:
        fn = (tc.get("function") or {}).get("name", "")
        if fn == "diagnose":
            try:
                return json.loads((tc.get("function") or {}).get("arguments", "{}"))
            except json.JSONDecodeError:
                return {}
    return None


def _calls_equal(
    a: list[dict[str, Any]] | None,
    b: list[dict[str, Any]] | None,
) -> bool:
    """True when two tool_calls lists have identical function names and arguments."""
    if a is None or b is None:
        return a is b
    if len(a) != len(b):
        return False
    for ta, tb in zip(a, b):
        fa = ta.get("function") or {}
        fb = tb.get("function") or {}
        if fa.get("name") != fb.get("name"):
            return False
        if fa.get("arguments") != fb.get("arguments"):
            return False
    return True


# ---------------------------------------------------------------------------
# Agent diagnosis (multi-round, replaces single-round)
# ---------------------------------------------------------------------------

_AGENT_MAX_CHARS_SCALE = 0.8


def run_agent_diagnosis(
    cfg: AgentConfig,
    log_file: Path,
    step_name: str,
    user_hint: str | None = None,
    artifact_dir: Path | None = None,
    git_context: dict[str, Any] | None = None,
    k8s_dir: Path | None = None,
    benchmark_dir: Path | None = None,
) -> dict[str, Any]:
    """Multi-round agent loop: pre-filtered overview + file index, then LLM
    explores via function-calling tools until it submits a diagnosis or runs
    out of rounds."""
    if not cfg.llm_ready():
        missing = cfg.missing_llm_config()
        _log("skip", f"LLM not configured: {', '.join(missing)}")
        return _skip_diagnosis(
            "LLM analysis not performed (CI secrets/variables not configured: " + ", ".join(missing) + ")",
            step_name,
        )
    if cfg.backend != "openai_compatible":
        _log("skip", f"backend {cfg.backend!r} not supported")
        return _skip_diagnosis(f"backend {cfg.backend!r} not supported", step_name)
    if not log_file.exists() or log_file.stat().st_size == 0:
        _log("skip", f"log file missing or empty: {log_file}")
        return _skip_diagnosis("log file missing or empty", step_name)

    _log("init", f"log={log_file} size={log_file.stat().st_size} model={cfg.model}")
    _log(
        "init",
        f"artifact_dir={artifact_dir} k8s_dir={k8s_dir} "
        f"benchmark_dir={benchmark_dir} max_rounds={cfg.max_rounds} "
        f"max_input_chars={cfg.max_input_chars} budget_chars={int(cfg.max_input_chars * _AGENT_MAX_CHARS_SCALE)}",
    )

    # ---- pre-filter main log ----
    _log("filter", f"reading raw log from {log_file}")
    filter_start = _dt.datetime.now(_dt.timezone.utc)
    try:
        raw_log = log_file.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError) as exc:
        return _skip_diagnosis(f"cannot read log file: {exc}", step_name)
    _log("filter", f"raw log chars={len(raw_log)}")
    filtered_log = build_llm_log_bundle(raw_log)
    _log("filter", f"after build_llm_log_bundle chars={len(filtered_log)}")
    filtered_log = clip_text(filtered_log, max_chars=cfg.max_input_chars - 10_000)
    filter_elapsed = (_dt.datetime.now(_dt.timezone.utc) - filter_start).total_seconds()
    _log("filter", f"final filtered chars={len(filtered_log)} elapsed={filter_elapsed:.2f}s")

    # ---- file index + allowed roots ----
    file_index, allowed_roots = _scan_log_dir(log_file, artifact_dir, k8s_dir, benchmark_dir)
    allowed_roots.add(log_file.parent.resolve())

    # ---- system + first user message ----
    initial_user_content = _build_initial_user_prompt(step_name, filtered_log, file_index, user_hint, git_context)
    _log(
        "init",
        f"initial_prompt system_chars={len(AGENT_SYSTEM_PROMPT)} "
        f"user_chars={len(initial_user_content)} "
        f"total_chars={len(AGENT_SYSTEM_PROMPT) + len(initial_user_content)}",
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": initial_user_content},
    ]
    last_tool_calls: list[dict[str, Any]] | None = None
    budget_chars = int(cfg.max_input_chars * _AGENT_MAX_CHARS_SCALE)

    for round_idx in range(1, cfg.max_rounds + 1):
        # Last round: force no-tools call
        round_tools = None if round_idx == cfg.max_rounds else AGENT_TOOLS

        _log(
            "agent",
            f"round {round_idx}/{cfg.max_rounds} "
            f"msg_chars={_estimate_chars(messages)} "
            f"tools={'on' if round_tools else 'off'}",
        )

        try:
            reply = _chat_one(
                base_url=cfg.base_url,
                api_key=cfg.api_key,
                model=cfg.model,
                messages=messages,
                tools=round_tools,
                timeout_s=cfg.timeout_s,
                max_input_chars=cfg.max_input_chars,
                stream=False,
            )
        except RuntimeError as exc:
            _log("llm", f"failed: {type(exc).__name__}: {exc}")
            return _skip_diagnosis(
                f"LLM request failed at round {round_idx}: {type(exc).__name__}: {exc}",
                step_name,
            )

        msg: dict[str, Any] = (reply.get("choices") or [{}])[0].get("message") or {}
        messages.append(msg)

        tool_calls: list[dict[str, Any]] = msg.get("tool_calls") or []
        tc_names = [((tc.get("function") or {}).get("name", "?")) for tc in tool_calls]
        _log("agent", f"round {round_idx} tool_calls={len(tool_calls)} names={tc_names}")

        # -- termination 1: diagnose tool_call
        diag_args = _find_diagnose_call(tool_calls)
        if diag_args:
            _log("agent", f"diagnose called at round {round_idx}")
            return _agent_result_to_diagnosis(
                raw=diag_args,
                step_name=step_name,
                log_file=log_file,
            )

        # -- termination 2: no tool_calls (model finished)
        if not tool_calls:
            _log("agent", f"no tool_calls at round {round_idx}, extracting JSON from content")
            raw = _parse_json_block(msg.get("content") or "") or {}
            if raw:
                return _agent_result_to_diagnosis(
                    raw=raw,
                    step_name=step_name,
                    log_file=log_file,
                )
            # Content didn't contain a JSON block — force one more round
            _log("agent", "no JSON found in content, forcing no-tools call")
            try:
                reply2 = _chat_one(
                    base_url=cfg.base_url,
                    api_key=cfg.api_key,
                    model=cfg.model,
                    messages=messages,
                    timeout_s=cfg.timeout_s,
                    max_input_chars=cfg.max_input_chars,
                    stream=False,
                )
            except RuntimeError as exc:
                _log("llm", f"force-final failed: {exc}")
                return _skip_diagnosis(
                    f"LLM returned no tool_calls and force-final also failed: {exc}",
                    step_name,
                )
            msg2 = (reply2.get("choices") or [{}])[0].get("message") or {}
            raw2 = _parse_json_block(msg2.get("content") or "") or {}
            return _agent_result_to_diagnosis(
                raw=raw2,
                step_name=step_name,
                log_file=log_file,
            )

        # -- termination 3: duplicate tool_calls (detected loop)
        if _calls_equal(tool_calls, last_tool_calls):
            _log("agent", f"duplicate tool_calls at round {round_idx}, force no-tools")
            try:
                reply = _chat_one(
                    base_url=cfg.base_url,
                    api_key=cfg.api_key,
                    model=cfg.model,
                    messages=messages,
                    timeout_s=cfg.timeout_s,
                    max_input_chars=cfg.max_input_chars,
                    stream=False,
                )
            except RuntimeError as exc:
                _log("llm", f"force-final after loop failed: {exc}")
                return _skip_diagnosis(
                    f"duplicate tool_calls detected and force-final failed: {exc}",
                    step_name,
                )
            msg = (reply.get("choices") or [{}])[0].get("message") or {}
            raw = _parse_json_block(msg.get("content") or "") or {}
            return _agent_result_to_diagnosis(
                raw=raw,
                step_name=step_name,
                log_file=log_file,
            )
        last_tool_calls = tool_calls

        # -- execute tool calls
        for tc in tool_calls:
            result = _execute_tool_call(tc, allowed_roots)
            messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": result})

        # -- message budget check + trim
        current_chars = _estimate_chars(messages)
        _log("agent", f"round {round_idx} post-tool-exec msg_chars={current_chars} budget={budget_chars}")
        if current_chars > budget_chars:
            pre_trim = len(messages)
            _log("agent", f"round {round_idx} budget exceeded, trimming history")
            messages = _trim_history(messages, keep_recent=2)
            _log(
                "agent",
                f"round {round_idx} trimmed messages={pre_trim}->{len(messages)} new_chars={_estimate_chars(messages)}",
            )

    # max_rounds exhausted (last round was no-tools): produce
    # low-confidence diagnosis from whatever we have
    last_msg = messages[-1] if messages else {}
    raw = _parse_json_block(last_msg.get("content", "")) or {}
    _log("agent", "max_rounds exhausted, fallback diagnosis")
    return _agent_result_to_diagnosis(
        raw=raw,
        step_name=step_name,
        log_file=log_file,
    )


def _agent_result_to_diagnosis(
    raw: dict[str, Any],
    step_name: str,
    log_file: Path,
) -> dict[str, Any]:
    """Wrap raw diagnose arguments (or parsed JSON) into the standard
    diagnosis dict expected by downstream renderers."""
    _log(
        "result",
        f"raw keys={list(raw.keys())[:10]} "
        f"failure_family={raw.get('failure_family')} "
        f"confidence={raw.get('confidence')} "
        f"classification={raw.get('classification')}",
    )
    wrapper_errors: list[dict[str, Any]] = []
    for w in raw.get("wrapper_errors") or []:
        if isinstance(w, dict):
            wrapper_errors.append(
                {
                    "type": str(w.get("type") or "Unknown"),
                    "line": int(w.get("line") or 0),
                    "snippet": str(w.get("snippet") or "")[:240],
                }
            )
    routing = {
        "failure_stage": raw.get("failure_stage") if raw.get("failure_stage") in ALLOWED_STAGES else "unknown",
        "failure_layer": raw.get("failure_layer") if raw.get("failure_layer") in ALLOWED_LAYERS else "unknown",
        "visible_failure": str(raw.get("visible_failure") or ""),
        "first_failure_signal": "",
        "wrapper_errors": wrapper_errors,
        "candidate_routes": [],
        "evidence_requests": [],
        "matched_playbooks": [],
    }
    diagnosis = _sanitize_diagnosis(
        raw,
        routing,
        failed_tests=_ensure_str_list(raw.get("failed_tests")),
    )
    diagnosis.update(
        {
            "schema_version": SCHEMA_VERSION,
            "step_name": step_name,
            "log_file": str(log_file),
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        }
    )
    return diagnosis


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_markdown(diag: dict[str, Any]) -> str:
    routing = diag.get("routing") or {}
    lines: list[str] = []
    lines.append("## CI AI Diagnosis")
    lines.append("")
    lines.append(f"**Step**: {diag.get('step_name') or ''}")
    lines.append(f"**Log**: `{diag.get('log_file') or ''}`")
    lines.append(f"**Stage / Layer**: `{routing.get('failure_stage')}` / `{routing.get('failure_layer')}`")
    lines.append(f"**Classification**: `{diag.get('classification')}`")
    lines.append(f"**Confidence**: `{diag.get('confidence')}`")
    lines.append(f"**Needs human review**: `{diag.get('needs_human_review')}`")
    lines.append("")

    if routing.get("visible_failure"):
        lines.append(f"**Visible failure**: {routing['visible_failure']}")
    if routing.get("first_failure_signal"):
        lines.append(f"**First signal**: {routing['first_failure_signal']}")
    if routing.get("matched_playbooks"):
        lines.append(f"**Matched playbooks**: {', '.join(routing['matched_playbooks'])}")
    lines.append("")

    lines.append("### Root cause")
    lines.append(diag.get("root_cause") or "_(no claim)_")
    lines.append("")

    if diag.get("evidence"):
        lines.append("### Evidence")
        lines.append("| file | line | snippet | interpretation |")
        lines.append("|---|---|---|---|")
        for e in diag["evidence"]:
            lines.append(
                f"| {e.get('file', '')} | {e.get('line', '')} | "
                f"`{e.get('snippet', '')}` | "
                f"{e.get('interpretation', '')} |"
            )
        lines.append("")

    if diag.get("counter_evidence"):
        lines.append("### Counter-evidence")
        lines.append("| file | line | snippet | interpretation |")
        lines.append("|---|---|---|---|")
        for e in diag["counter_evidence"]:
            lines.append(
                f"| {e.get('file', '')} | {e.get('line', '')} | "
                f"`{e.get('snippet', '')}` | "
                f"{e.get('interpretation', '')} |"
            )
        lines.append("")

    if diag.get("next_actions"):
        lines.append("### Next actions")
        lines.append("| priority | action | command |")
        lines.append("|---|---|---|")
        for a in diag["next_actions"]:
            lines.append(f"| {a.get('priority', '')} | {a.get('action', '')} | `{a.get('command', '')}` |")
        lines.append("")

    if routing.get("wrapper_errors"):
        lines.append("### Wrapper errors (symptoms, not root cause)")
        for w in routing["wrapper_errors"]:
            lines.append(f"- L{w.get('line', '')} {w.get('type', '')}: `{w.get('snippet', '')}`")
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def _append_step_summary(text: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        sys.stdout.write(text)
        return
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(text)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CI AI diagnosis agent (vllm-ascend).")
    p.add_argument("--log-file", type=Path, required=True)
    p.add_argument("--step-name", default="tests")
    p.add_argument("--artifact-dir", type=Path, default=None)
    p.add_argument("--k8s-dir", type=Path, default=None)
    p.add_argument("--benchmark-dir", type=Path, default=None)
    p.add_argument("--user-hint", default=None)
    p.add_argument("--ref", default=None)
    p.add_argument("--sha", default=None)
    p.add_argument("--repo-dir", type=Path, default=None)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--write-summary", action="store_true", help="Write Markdown to GITHUB_STEP_SUMMARY.")
    return p.parse_args(argv)


def _build_git_context(
    repo_dir: Path | None,
    ref: str | None,
    sha: str | None,
) -> dict[str, Any]:
    ctx: dict[str, Any] = {"ref": ref or "", "sha": sha or ""}
    if repo_dir is None or not repo_dir.is_dir():
        _log("git", f"repo_dir={repo_dir} not a dir, skipping git context")
        return ctx
    _log("git", f"building context repo_dir={repo_dir} ref={ref} sha={sha}")
    try:
        log = subprocess.run(
            ["git", "-C", str(repo_dir), "log", "-1", "--format=%H%n%an%n%ad%n%s", sha or "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if log.returncode == 0 and log.stdout.strip():
            lines = log.stdout.strip().split("\n")
            ctx["commit"] = {
                "hash": lines[0] if len(lines) > 0 else "",
                "author": lines[1] if len(lines) > 1 else "",
                "date": lines[2] if len(lines) > 2 else "",
                "subject": lines[3] if len(lines) > 3 else "",
            }
            _log("git", f"commit={ctx['commit']['hash'][:8]} {ctx['commit']['subject'][:60]}")
        else:
            _log("git", f"git log failed rc={log.returncode}")
    except (subprocess.SubprocessError, OSError) as exc:
        _log("git", f"git log error: {exc}")
    try:
        parent = f"{sha}~1" if sha else "HEAD~1"
        diff = subprocess.run(
            ["git", "-C", str(repo_dir), "diff", "--name-only", parent, sha or "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if diff.returncode == 0:
            ctx["changed_files"] = [f for f in diff.stdout.strip().split("\n") if f][:50]
            _log("git", f"changed_files count={len(ctx['changed_files'])}")
        else:
            _log("git", f"git diff failed rc={diff.returncode}")
    except (subprocess.SubprocessError, OSError) as exc:
        _log("git", f"git diff error: {exc}")
    return ctx


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main_impl(argv: list[str] | None = None) -> int:
    _log("start", f"pid={os.getpid()} python={sys.version.split()[0]} argv={argv}")
    args = _parse_args(argv)
    _log(
        "args",
        f"log_file={args.log_file} step_name={args.step_name} "
        f"artifact_dir={args.artifact_dir} k8s_dir={args.k8s_dir} "
        f"benchmark_dir={args.benchmark_dir} "
        f"output_json={args.output_json} write_summary={args.write_summary}",
    )
    cfg_start = _dt.datetime.now(_dt.timezone.utc)
    cfg = AgentConfig.from_env()
    _log(
        "config",
        f"enabled={cfg.enabled} backend={cfg.backend} model={cfg.model} "
        f"max_rounds={cfg.max_rounds} timeout_s={cfg.timeout_s} "
        f"max_input_chars={cfg.max_input_chars} tail_chars={cfg.tail_chars} "
        f"elapsed={(_dt.datetime.now(_dt.timezone.utc) - cfg_start).total_seconds():.2f}s",
    )
    git_context = _build_git_context(
        repo_dir=args.repo_dir,
        ref=args.ref,
        sha=args.sha,
    )

    if not args.log_file.exists() or args.log_file.stat().st_size == 0:
        _log("skip", f"log file missing or empty: {args.log_file}")
        diag = _skip_diagnosis("log file missing or empty", args.step_name)
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(diag, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.write_summary:
            _append_step_summary(
                render_evidence_summary(
                    {
                        "step_name": args.step_name,
                        "log_file": str(args.log_file),
                        "index": {},
                        "missing_llm_config": ["log file missing or empty"],
                    }
                )
            )
        return 0

    _log("init", f"log={args.log_file} size={args.log_file.stat().st_size}")

    ctx = git_context if any(git_context.values()) else None

    if cfg.llm_ready():
        _log("llm", f"calling model={cfg.model}")
        diag = run_agent_diagnosis(
            cfg,
            log_file=args.log_file,
            step_name=args.step_name,
            user_hint=args.user_hint,
            artifact_dir=args.artifact_dir,
            git_context=ctx,
            k8s_dir=args.k8s_dir,
            benchmark_dir=args.benchmark_dir,
        )
        summary_md = render_markdown(diag)
    else:
        missing = cfg.missing_llm_config()
        _log("llm", f"skipped: {', '.join(missing)}")
        diag = _make_evidence_only_diagnosis(
            step_name=args.step_name,
            log_file=args.log_file,
            git_context=ctx,
            missing_config=missing,
        )
        summary_md = render_evidence_summary(diag)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(diag, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _log("output", f"wrote diagnosis JSON to {args.output_json}")
    if args.write_summary:
        _append_step_summary(summary_md)
        _log("output", f"wrote summary to GITHUB_STEP_SUMMARY ({len(summary_md)} chars)")
    else:
        sys.stdout.write(summary_md)
        _log("output", f"wrote summary to stdout ({len(summary_md)} chars)")
    _log("done", f"confidence={diag.get('confidence')} classification={diag.get('classification')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return _main_impl(argv)
    except Exception as exc:
        import traceback

        tb = traceback.format_exc()
        safe_type = type(exc).__name__
        safe_msg = str(exc)[:500]
        _log("fatal", f"{safe_type}: {safe_msg}")
        _log("fatal", tb[-2000:])
        md = (
            "## CI AI Diagnosis\n\n"
            "**Agent Exception**: The diagnosis script encountered an internal error and did not produce a result.\n\n"
            f"| Exception Type | Message |\n|---|---|\n"
            f"| `{safe_type}` | `{safe_msg}` |\n\n"
        )
        with contextlib.suppress(Exception):
            _append_step_summary(md)
        sys.stderr.write(f"[ci-ai-diagnosis] fatal: {safe_type}: {safe_msg}\n")
        sys.stderr.write(tb[-4000:])
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
