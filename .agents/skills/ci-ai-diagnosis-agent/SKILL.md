---
name: ci-ai-diagnosis-agent
description: >
  AI-driven CI failure diagnosis agent for vllm-ascend. Routes unknown CI
  failures to evidence retrieval, hypothesis verification, and structured
  diagnosis. Reuses log-diagnosis-{problem-id} playbooks as accelerators.
  Use when a CI step fails and the cause is not obvious.
---

# CI AI Diagnosis Agent

This skill is the **top-level diagnosis dispatcher** for unknown CI failures
in vllm-ascend. It is written for an LLM agent with evidence access, not for
a keyword classifier. The agent uses `.agents/skills/ci-ai-diagnosis-agent/scripts/ci_diagnosis_agent.py`
as its entry point and explores logs via function-calling tools (list_dir,
read_file, search, diagnose) built directly into the agent loop.

---

## 1. Location

This is a **CI failure diagnosis dispatcher**, not a log organizer or a
single-issue playbook. It dispatches across the full space of possible CI
failures: setup, collection, startup, runtime, assertion, teardown, infra.
It never treats a wrapper error as a root cause.

---

## 2. Input Contract

| Input | Required | Notes |
|---|---|---|
| Log file path | yes | UTF-8 text, provided via `--log-file` |
| Step name | recommended | Used in the diagnosis output |
| Artifact directory | optional | Ascend logs, misc artifacts |
| K8s directory | optional | Pods JSON, events, describe output |
| Benchmark directory | optional | Benchmark results JSON |
| Git context | optional | Branch, commit, changed files |
| User hint | optional | Bias, not verdict |

Failure mode: if the log file is missing, emit a skip note. Never fabricate.

---

## 3. Execution Protocol

The agent follows a multi-round function-calling loop driven by `ci_diagnosis_agent.py`:

### 3.1 Pre-filter & Index
The agent script pre-filters the raw CI log using `ci_log_filter_llm.py`
(two-phase: high-signal lines + context windows around anchors). It then
builds a file index of the main log directory and optional artifact/K8s/
benchmark directories.

### 3.2 Agent Loop
The LLM receives the filtered log + file index as its first message. It
explores via function-calling tools (`list_dir`, `read_file`, `search`).
When ready, it calls `diagnose` to submit a structured diagnosis.

### 3.3 Tool Set
All evidence tools are built into the agent loop (not separate scripts):

| Tool | Purpose |
|---|---|
| `list_dir(dir, recursive, pattern)` | List files in a directory |
| `read_file(path, offset, limit)` | Read section of a log file by line range |
| `search(path, pattern, context_lines)` | Regex search with line numbers and context |
| `diagnose(...)` | Submit final structured diagnosis JSON |

### 3.4 Termination
The loop ends when the LLM calls `diagnose`, returns no tool calls, or
exhausts `max_rounds` (default 3). On exhaustion, a force-final round
without tools runs to extract whatever diagnosis the model can produce.

---

## 4. Evidence Source Orchestration

CI diagnosis is a multi-source evidence problem. The agent must decide
which evidence source answers which question before declaring a root cause.

### 4.1 Source Roles

| Source | Role | Answers |
|---|---|---|
| Main CI log | Primary timeline and visible failure surface | Which step failed, first visible exception, pytest wrapper, final exit |
| Benchmark results | Authoritative benchmark outcome | Accuracy/performance pass-fail, metric value, baseline, threshold, failed task |
| K8s diagnostics | Environment and pod state | Scheduling, image pull, OOM, restarts, pod phase, warning events |
| Ascend/worker logs | Engine, rank, and NPU runtime facts | EngineCore crash, HCCL/ACL errors, rank-local failure, worker stdout not streamed to leader |
| Git context | Regression prior, never verdict | Whether changed files touch the failed surface |

### 4.2 Source Routing Matrix

| Failure signal | Required sources | Optional sources | Decisive evidence |
|---|---|---|---|
| `AssertionError: some aisbench cases failed` | Main CI log, benchmark results | Ascend/worker logs | Failed benchmark task with metric, baseline, threshold |
| Performance or accuracy regression | Benchmark results, main CI log | Git context, Ascend/worker logs | Benchmark JSON `pass_fail=fail` plus metric delta |
| EngineDeadError, EngineCoreFatal, 500 wrapper | Main CI log, Ascend/worker logs | K8s diagnostics | First non-wrapper engine/rank exception before the wrapper |
| HCCL, ACL, CANN, rank crash | Ascend/worker logs, main CI log | K8s diagnostics | Rank-local first error with file and line |
| Pod Pending, CrashLoopBackOff, OOMKilled, ImagePullBackOff | K8s diagnostics | Main CI log | Pod state, reason, restart count, warning event |
| Pytest collection/import error | Main CI log | Git context | Import/collection traceback before pytest summary |
| Missing benchmark result | Main CI log, benchmark artifact manifest | K8s diagnostics, Ascend/worker logs | Test reached benchmark phase but no result file, or failed before serialization |

### 4.3 Evidence Request Contract

Every `evidence_request` must include:

```json
{
  "source": "benchmark_results",
  "question": "Which benchmark task failed and by how much?",
  "tool": "get_benchmark_summary",
  "expected_fact": "failed task, metric, value, baseline, threshold",
  "fallback": "If benchmark results are missing, record missing_source and inspect main log around aisbench output"
}
```

Use source-specific tools before generic search:

1. Benchmark outcome: `get_benchmark_summary()` first, then `search()` for
   the same task or metric in the main log.
2. K8s state: `get_k8s_summary()` first, then `search_artifacts()` for the
   pod name or abnormal reason.
3. Engine/NPU/rank failures: `search_artifacts()` first for HCCL/ACL/Engine
   patterns, then `get_artifact_window()` or `get_window()` around matched
   lines.
4. Wrapper errors: `get_wrapper_upstream_context()` first, then source
   routing decides whether to inspect benchmark, K8s, or worker logs.

### 4.4 Cross-Source Rules

- Main CI log is the timeline anchor, not always the root-cause source.
- Benchmark JSON is authoritative for benchmark pass/fail. Main log copies
  of benchmark output are supporting evidence.
- K8s diagnostics are authoritative for pod state, restart count, OOM,
  scheduling, and image pull failures.
- Ascend/worker logs are authoritative for rank-local engine and NPU
  runtime failures that may not appear in the leader log.
- Git context is only a prior. It can raise or lower suspicion but cannot
  prove a root cause without log evidence.
- Missing sources must be listed in `missing_sources[]` with the diagnostic
  impact. Example: "benchmark_results missing, cannot distinguish benchmark
  threshold failure from benchmark execution failure".
- If sources conflict, list the conflict in `source_conflicts[]` and prefer
  the source that owns the fact. Example: K8s summary says pod Running, so
  a later HTTP 500 is not classified as pod startup failure without worker
  evidence.

### 4.5 Confidence Fusion

Use this confidence policy:

| Confidence | Required evidence |
|---|---|
| `high` | Direct evidence from the owning source, at least one independent corroborating source, and no strong counter-evidence |
| `medium` | Direct evidence from the owning source, but corroboration is missing or partial |
| `low` | Only wrapper evidence, missing owning source, conflicting sources, or generic pattern hits |

Set `needs_human_review=true` when the owning source is missing, all
available evidence is wrapper-level, or source conflicts remain unresolved.

---

## 5. Failure Routing

Routing uses two orthogonal dimensions:

| Dimension | Values |
|---|---|
| `failure_stage` | `setup`, `collection`, `startup`, `runtime`, `assertion`, `teardown`, `infra`, `unknown` |
| `failure_layer` | `ci_workflow`, `pytest`, `dependency`, `vllm_engine`, `vllm_ascend`, `npu_runtime`, `kubernetes`, `network`, `storage`, `external_service`, `unknown` |

**Wrapper error rule**: These are symptoms, never root causes unless proven
otherwise — `EngineDeadError`, `CalledProcessError`, `TimeoutError`,
`500 Internal Server Error`, `CrashLoopBackOff`, etc. For each wrapper,
request `get_wrapper_upstream_context`.

**Routing JSON schema**:

```json
{
  "failure_stage": "startup",
  "failure_layer": "vllm_engine",
  "visible_failure": "CI shows EngineDeadError",
  "first_failure_signal": "traceback in RPC call at line 3200",
  "wrapper_errors": [{"type": "EngineDeadError", "line": 4200, "snippet": "..."}],
  "evidence_source_plan": [
    {"source": "main_ci_log", "role": "timeline", "status": "available"},
    {"source": "ascend_worker_logs", "role": "owning_source_for_engine_crash", "status": "required"}
  ],
  "candidate_routes": [
    {"route": "vllm_engine_runtime", "playbook": "log-diagnosis-vllm-inference-timeout",
     "confidence": "medium", "supporting_evidence": [...], "missing_evidence": [...]}
  ],
  "evidence_requests": [
    {"source": "main_ci_log", "question": "Where is the first visible exception?", "tool": "get_first_exception_context"},
    {"source": "ascend_worker_logs", "question": "What is the first rank-local engine error?", "tool": "search_artifacts", "pattern": "EngineCore|HCCL|ACL|RuntimeError"}
  ],
  "missing_sources": [],
  "source_conflicts": [],
  "matched_playbooks": ["log-diagnosis-vllm-inference-timeout"]
}
```

If evidence is too sparse to pick a stage/layer, set both to `unknown`,
`candidate_routes` to `[]`, and `needs_human_review` to `true`. Never
invent a route you cannot justify.

---

## 6. Evidence Tools

All tools are built into the agent's function-calling loop within
`ci_diagnosis_agent.py`. The LLM accesses them as OpenAI-compatible
function calls:

| Tool | Purpose |
|---|---|
| `list_dir(dir, recursive, pattern)` | List files in a directory |
| `read_file(path, offset, limit)` | Read a section of a log file by line range (max 500 lines) |
| `search(path, pattern, context_lines)` | Regex search with line numbers and context (max 50 matches) |
| `diagnose(failure_family, root_cause, classification, ...)` | Submit the final structured diagnosis |

The agent must call these tools rather than guess. Every claim must
reference a `{file, line}` pair.

---

## 7. Playbook Reuse (Accelerators, not Authorities)

When routing aligns with a known failure family, reference the playbook's
decision tree for guidance. Playbooks are accelerators — the agent must
still verify every claim against the actual log.

Known playbooks:
- `log-diagnosis-vllm-inference-timeout`
- `log-diagnosis-pd-link-establishment`
- `log-diagnosis-large-ep-startup`
- `log-diagnosis-shrink-p-reserve-d`
- `log-diagnosis-controller-recovery-terminate`

When no playbook matches, fall back to generic hypothesis protocol. Never
fabricate a playbook match.

---

## 8. Git Context

When the index includes `git_context`, use it for code-aware diagnosis:

- **Narrow classification**: if `changed_files` touches the failure surface,
  favour `product_bug` or `test_bug` over `infra_issue`.
- **Correlate commits**: if the commit subject mentions the failing
  component, treat as a strong prior for code-introduced regression.
- **Rule out code**: if no file touches the failing domain, favour
  `infra_issue` or `flake`.

Git context is a prior, never a verdict. Cross-check against log evidence.

---

## 9. Output Format

The full output format specification is in `references/output-format.md`.
Summary:

1. **Diagnosis JSON** (machine-readable, source of truth):
   Key fields: `failure_family`, `root_cause`, `classification`
   (`flake|test_bug|product_bug|infra_issue|unknown`), `confidence`
   (`high|medium|low`), `evidence[]`, `counter_evidence[]`,
   `wrapper_errors[]`, `next_actions[]`, `matched_playbooks[]`,
   `evidence_sources[]`, `missing_sources[]`, `source_conflicts[]`,
   `needs_human_review`.

2. **Markdown report** (human-readable, 11 sections):
   诊断结论, 环境概要, 关键时间线, 证据链, 失败路径对比, 故障链路总览,
   根因分析, 排除项, 下一步行动, 修复建议, 关键日志检索命令.

See `references/output-format.md` for the detailed schema and section
requirements.

---

## 10. Failure Mode

- Never breaks CI. All failure paths exit 0.
- If LLM call is disabled or misconfigured, emit a skip note with
  `confidence=low`, `classification=unknown`, `needs_human_review=true`.
- The Python entry point (``ci_diagnosis_agent.py``) builds the log index,
  collects evidence, and calls the configured LLM when
  ``VLLM_ASCEND_CI_AI_DIAGNOSIS_ENABLED=1`` and API credentials are set.
  If the LLM call is disabled or misconfigured, it emits a skip note instead.

---

## 11. What This Skill Does NOT Do

- Does not commit, push, or change business code.
- Does not hard-code keyword-to-playbook mappings at the top level.
- Does not declare a root cause without log line evidence.
- Does not depend on log-diagnosis playbooks being present.
- Does not create pull requests or modify CI configuration.

---

## 12. Reference Implementation

| Component | Path | Role |
|---|---|---|
| Skill definition | `.agents/skills/ci-ai-diagnosis-agent/SKILL.md` | This file |
| Output format | `.agents/skills/ci-ai-diagnosis-agent/references/output-format.md` | Detailed output spec |
| Agent entry point | `.agents/skills/ci-ai-diagnosis-agent/scripts/ci_diagnosis_agent.py` | CLI entry, agent loop, evidence tools |
| Log filter | `.agents/skills/ci-ai-diagnosis-agent/scripts/ci_log_filter_llm.py` | Two-phase log pre-filtering for LLM input |
