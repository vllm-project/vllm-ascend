"""Two-phase CI log shrinking for LLM diagnostics (see .agents/skills/ci-llm-ci-log-diagnostics/).

Phase A — high-signal lines (errors, warnings, failure markers, related INFO).
Phase B — fixed windows around anchors (tracebacks, FAILED lines, exceptions).
"""

from __future__ import annotations

import regex as re

# Lines matching any of these (case-insensitive) are kept in phase A.
_IMPORTANT_SUBSTRINGS = (
    "error",
    "warning",
    "fail",
    "exception",
    "traceback",
    "assert",
    "fatal",
    "timeout",
    "oom",
    "out of memory",
    "killed",
    "segfault",
    "signal",
    "core dumped",
    "acl",
    "ascend",
    "cann",
    "hccl",
    "kubernetes",
    "kubelet",
    "kubectl",
    "backoff",
    "crashloop",
    "imagepull",
    "errimagepull",
    "oomkilled",
    "leaderworkerset",
    "distributed",
    "errno",
    "cannot",
    "could not",
    "unable to",
)

# INFO lines must also match one of these to be included (reduces noise).
_INFO_EXTRA = (
    "fail",
    "error",
    "warn",
    "timeout",
    "oom",
    "assert",
    "traceback",
    "exception",
    "worker",
    "engine",
    "cuda",
    "npu",
    "acl",
    "ascend",
    "cann",
    "ray",
    "hccl",
    "kubectl",
    "kube",
    "pod",
    "backoff",
    "leader",
)

_ANCHOR_RES = (
    re.compile(r"^=+\s+FAILURES\s+=+$", re.IGNORECASE),
    re.compile(r"^=+\s+short test summary info\s+=+$", re.IGNORECASE),
    re.compile(r"^FAILED\s+tests/\S+", re.IGNORECASE),
    re.compile(r"^=== .+ ===$"),
    re.compile(r"Traceback \(most recent call last\):"),
    re.compile(r"The above exception was the direct cause of the following exception:", re.IGNORECASE),
    re.compile(
        r"\b(?:CrashLoopBackOff|ImagePullBackOff|ErrImagePull|OOMKilled|CreateContainerConfigError|RunContainerError)\b"
    ),
    re.compile(r"^\s*raise [\w.:]+"),
    re.compile(r"\b(?:Error|Exception)\s*:\s*\S+"),
    re.compile(r"E\s+(?:AssertionError|RuntimeError|OSError|ValueError|TypeError|KeyError)\b"),
    re.compile(r"(?:ERROR|CRITICAL)\s*[\]:]"),
)


def _is_interesting_info(line: str) -> bool:
    u = line.upper()
    if " INFO " not in u and not u.startswith("INFO"):
        return False
    low = line.lower()
    return any(x in low for x in _INFO_EXTRA)


def select_important_lines(lines: list[str], *, max_lines: int = 800) -> list[str]:
    """Phase A: high-signal lines (not necessarily contiguous)."""
    out: list[str] = []
    for i, raw in enumerate(lines):
        s = raw.rstrip("\n")
        low = s.lower()
        if any(k in low for k in _IMPORTANT_SUBSTRINGS):
            out.append(f"L{i + 1}:{s}")
            continue
        if _is_interesting_info(s):
            out.append(f"L{i + 1}:{s}")
    if len(out) > max_lines:
        out = out[: max_lines // 2] + [f"... [{len(out) - max_lines} lines omitted] ..."] + out[-(max_lines // 2) :]
    return out


def _anchor_indices(lines: list[str]) -> list[int]:
    idx: list[int] = []
    for i, raw in enumerate(lines):
        s = raw.rstrip("\n")
        if any(r.search(s) for r in _ANCHOR_RES):
            idx.append(i)
    return idx


def _merge_spans(spans: list[tuple[int, int]], n_lines: int) -> list[tuple[int, int]]:
    if not spans:
        return []
    spans = sorted(spans)
    merged: list[tuple[int, int]] = []
    cur_s, cur_e = spans[0]
    for s, e in spans[1:]:
        if s <= cur_e + 1:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return [(max(0, a), min(n_lines - 1, b)) for a, b in merged]


def extract_nearby_regions(
    lines: list[str],
    *,
    context_before: int = 40,
    context_after: int = 80,
) -> list[tuple[int, int, str]]:
    """Phase B: contiguous regions (start_line, end_line, content) around anchors."""
    anchors = _anchor_indices(lines)
    if not anchors:
        window = max(1, context_before + context_after)
        start = max(0, len(lines) - window)
        spans = [(start, len(lines) - 1)]
    else:
        spans = []
        for a in anchors:
            spans.append((max(0, a - context_before), min(len(lines) - 1, a + context_after)))
    merged = _merge_spans(spans, len(lines))
    regions = []
    for s, e in merged:
        chunk = "\n".join(f"L{i + 1}:{lines[i].rstrip()}" for i in range(s, e + 1))
        regions.append((s + 1, e + 1, chunk))
    return regions


def build_llm_log_bundle(
    log_text: str,
    *,
    context_before: int = 40,
    context_after: int = 80,
    max_important_lines: int = 800,
    max_regions_chars: int = 100_000,
) -> str:
    """Combine phase A + B into one markdown document for the model."""
    lines = log_text.splitlines()
    if not lines:
        return "(empty log)"

    important = select_important_lines(lines, max_lines=max_important_lines)
    regions = extract_nearby_regions(lines, context_before=context_before, context_after=context_after)

    parts = [
        "### Phase A — high-signal lines (filtered)",
        "These lines were selected by level/heuristic (errors, warnings, failures, select INFO).",
        "",
    ]
    parts.extend(important if important else ["(no lines matched phase-A heuristics)"])
    parts.extend(["", "### Phase B — local context around failure anchors", ""])
    total = 0
    for start, end, chunk in regions:
        header = f"--- lines {start}-{end} ---"
        if total + len(chunk) > max_regions_chars:
            parts.append(f"{header}\n(truncated: remaining regions omitted for size cap)")
            break
        parts.append(f"{header}\n{chunk}")
        total += len(chunk)
    return "\n".join(parts)


def clip_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    mid = f"\n\n... [{len(text) - max_chars} chars omitted] ...\n\n"
    if max_chars <= len(mid) + 2:
        return text[:max_chars]
    avail = max_chars - len(mid)
    head = avail // 2
    tail = avail - head
    return text[:head] + mid + text[-tail:]
