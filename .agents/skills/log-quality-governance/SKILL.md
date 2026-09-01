---
name: log-quality-governance
description: >
  Phase-1 stock log-quality governance for vllm-ascend: periodic scan of
  existing logger calls, prioritized P0/P1/P2 findings, and rewrite candidates
  only. Does not block PRs. Use when asked to scan stock logs, batch-fix log
  quality, rewrite vague/missing ERROR logs, run phase1-stock governance, or
  produce a log-quality remediation backlog for vllm_ascend/.
---

# Log Quality Governance (Phase 1 Stock Layer)

This skill is the **third layer** of vLLM-Ascend Phase-1 log quality:

| Layer | Role | Lives in |
|---|---|---|
| `pre-commit` | Hard-fail incremental gate (LQ001–LQ005) | `tools/check_log_quality.py`, `tools/check_logger.sh` |
| Gemini Review | Context MUST/MUST NOT on PR diffs | `.gemini/styleguide.md` → Logging Quality |
| **This skill** | Stock inventory + rewrite candidates | `.agents/skills/log-quality-governance/` |

It does **not** replace pre-commit or Gemini. It does **not** auto-apply patches or commit.

Default mode is **`phase1-stock`** (minimum quality bar + stock scan standard).
Use **`full-standard`** only when the user explicitly asks for the full 8-rule ideal bar.

Authoritative references:
[references/phase1-stock-scan-standard.md](references/phase1-stock-scan-standard.md),
[references/rewrite-guide.md](references/rewrite-guide.md).

---

## When to use

- Periodic / batch scan of `vllm_ascend/**/*.py` logger quality
- Produce a remediation backlog (P0 → P1 → P2)
- Generate rewrite candidates / diffs for selected findings
- User says: stock log scan, phase1-stock, log quality governance, batch rewrite logs

## When NOT to use

- Reviewing a normal PR diff → use pre-commit + Gemini styleguide
- Diagnosing a CI failure root cause → `ci-ai-diagnosis-agent`
- User only wants the hard gate explained → point to `tools/log_quality_rules.toml`

---

## Input contract

| Input | Required | Notes |
|---|---|---|
| Repo root | yes | This checkout |
| Scope | optional | Default `vllm_ascend/`; may narrow to a module path |
| Mode | optional | Default `phase1-stock`; `full-standard` only if requested |
| Priority filter | optional | Default rewrite **P0+P1 only**; include P2 only if asked |
| Runtime logs | optional | CI/run log file for optional live-log validation |

---

## Workflow (must follow)

```
1. Load the Phase-1 stock scan standard (references/phase1-stock-scan-standard.md)
2. Scan code (logger extraction + checks) → findings with P0/P1/P2
3. Optional: scan a runtime log file with the same priority model
4. If user wants rewrite: only P0/P1 (unless P2 requested) → candidates + diff
5. Stop for human review — never apply, never git commit / merge
```

### Step 1 — Load口径

Read `references/phase1-stock-scan-standard.md`. Do **not** load the personal Hermes
`external-standard.md` as the default pass/fail bar for stock scans.
That full standard is `full-standard` mode only.

### Step 2 — Code scan

Extract logger calls under scope:

```bash
rg -n --glob 'vllm_ascend/**/*.py' \
  'logger\.(debug|info|warning|error|exception|critical)\(|self\.logger\.(debug|info|warning|error|exception|critical)\('
```

For each call, capture: path, line, level, message template, kwargs / format args,
surrounding loop/retry/`raise` context (read nearby lines; do not invent CFG).

Emit findings using the **nine coverage buckets** in
`phase1-stock-scan-standard.md`, each item tagged **P0 / P1 / P2**.

Also run a coarse raise-gap pass (see coverage bucket 9). Prefer precision over
recall: skip generated/tests helpers; do not mark every `raise` as P0.

**Out of scope for code scan**

- Ops retention / log shipping / disk aging (full-standard “运维管理”)
- Forcing `[repo/module]` component prefixes in source (runtime logger name)
- Forcing Chinese “可检查” text
- Forcing `event=` / fixed `reason=` as hard failures in `phase1-stock`

### Step 3 — Optional runtime log scan

If the user provides a `.log` / CI artifact, check the same P0/P1 themes
(privacy leak, vague ERROR, missing id on request path). Mark items that only
appear in runtime as `source=runtime`.

### Step 4 — Rewrite candidates

Follow `references/rewrite-guide.md`.

Rules:

- Match surrounding file style (English; `logger.*`; `%s` / f-string as in file)
- Prefer `req_id=` for new request-path fields; accept legacy `request_id=`
- Only change logger statements (+ optional rate-limit helpers if already used nearby)
- **MUST NOT** change retry/backoff, returns, thread loops, HTTP calls, or exception control flow
- Output Markdown before/after table + unified diff draft
- **Do not** apply the diff; **do not** `git commit` / `git merge`

### Step 5 — Human gate

End with a short backlog:

| Priority | Count | Suggested next batch |
|---|---|---|
| P0 | N | … |
| P1 | N | … |
| P2 | N | (defer unless asked) |

Ask which batch to rewrite next if not already specified.

---

## Output templates

### Scan report

```markdown
# vLLM-Ascend log quality stock scan (phase1-stock)

- Scope: …
- Mode: phase1-stock
- Logger calls scanned: N

## Summary by priority
| Priority | Count | Meaning |
|---|---|---|
| P0 | | Sensitive / empty ERROR / critical failure path silent |
| P1 | | Missing diagnostic / cross-component / final failure summary |
| P2 | | Naming / weak readability / incomplete closure |

## Findings
### P0
| ID | Bucket | Location | Evidence | Suggested fix direction |
|---|---|---|---|---|

### P1
…

### P2
…
```

### Rewrite pack

```markdown
# Rewrite candidates (P0/P1 only)

| ID | Before | After | Priority | Notes |
|---|---|---|---|---|

## Diff draft
```diff
…
```

Human review required before apply. No business-control-flow edits included.
```

---

## Hard boundaries

| Allowed | Forbidden |
|---|---|
| Scan + report + candidate diffs | Auto-apply patches |
| Narrow scope / priority | Bind this skill into PR/CI gate |
| `full-standard` when user asks | Treat full ideal bar as default stock gate |
| Point to pre-commit for incremental hard fails | Re-implement LQ001–LQ005 as a second blocker here |

---

## Relation to repo tools

| Artifact | Use with this skill |
|---|---|
| `tools/log_quality_rules.toml` | Same privacy / vague literals for P0 detection heuristics |
| `tools/check_log_quality.py` | Incremental only; may run on a file for cross-check, not as stock inventory |
| `.gemini/styleguide.md` Logging Quality | Field names (`req_id`, `peer_component`, …) for P1 rewrite guidance |
