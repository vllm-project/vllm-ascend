# Robot Review Bot — Test Report

Generated: 2026-06-27  
Model: deepseek-v4-flash  
Pipeline: `lib/review.py` (shared by production bot and test harness)

---

## 1. Case Selection

### Issue Dataset (`issues.csv`)

Issues collected from `ai-infra-develop/vllm-ascend` via the GitHub API using
state-based stratified sampling for balanced representation.

**Total: 84 rows** (59 evaluated, 25 skipped — ineligible title prefix)

Prefix distribution:

| Prefix | Count |
|--------|-------|
| `[Bug]` | 57 |
| `[Contribution]` | 5 |
| `[Installation]` | 4 |
| `[Doc]` | 3 |
| `[Feature]` | 3 |
| `[RFC]` | 2 |
| `[Misc]` | 2 |
| `[BugFix]` | 1 |
| `[Usage]` | 1 |
| `(none / ineligible)` | 6 |

Ineligible titles (skipped by both bot and test harness): `[Contribution]`,
`[RFC]`, `[BugFix]`, `[Bug][Upstream]:`, `[Bug][v0.20.2]:`, and titles with
no recognised `[Prefix]:` pattern per `extract_issue_type()` and
`should_review()`.

### PR Dataset (`prs.csv`)

PRs collected via time-based stratified sampling across recent PRs.

**Total: 61 rows** (61 evaluated, 0 skipped — all PRs are always reviewed)

Prefix distribution (informational only — PR type key is always `other`):

| Prefix | Count |
|--------|-------|
| `[BugFix]` / `[bugfix]` | 18 |
| `[CI]` | 15 |
| `[Doc]` | 7 |
| `[Feature]` | 6 |
| `(none)` | 6 |
| `[Misc]` / `[MISC]` | 3 |
| `[EPLB]` | 2 |
| other | 4 |

---

## 2. Evaluation Process

Each case is evaluated using the **same `review()` pipeline** as the production
`bot_issue_review.yaml` and `bot_pr_review.yaml` workflows:

```text
Title + Body
    │
    ▼
should_review(kind, title)        ← gate: mirrors workflow if: filter
    │
    ▼
resolve_type_key(kind, title)     ← derives prefix + type_key from title only
    │
    ▼
load_template(kind, prefix)       ← loads matching ISSUE_TEMPLATE/*.yml
    │
    ▼
build_review_prompt(...)          ← single source of truth in lib/review.py
    │
    ▼
call_llm(system_prompt, user_prompt)
    │
    ▼
validate_result(parse_json_output(raw))
    │
    ▼
{ ok, score, reasoning, missing_items, suggestions }
```

Results stored as:
- `deepseek_v4_flash_output` — raw LLM JSON string
- `expected_ok_dsv4` — parsed boolean (`true` / `false`)

Templates used per prefix:

| Prefix | Template |
|--------|----------|
| `[Bug]` | `400-bug-report.yml` |
| `[Installation]` | `200-installation.yml` |
| `[Usage]` | `300-usage.yml` |
| `[Doc]` | `100-documentation.yml` |
| `[Misc]` | `800-others.yml` |
| `[Feature]` | `500-feature-request.yml` |
| `[Perf]` | `700-performance-discussion.yml` |
| PR (any) | `PULL_REQUEST_TEMPLATE.md` |

---

## 3. Judge Process

Each evaluation is audited by the same LLM with a dedicated judge system prompt
across four dimensions:

| Column | Question |
|--------|----------|
| `ok_reasonable` | Is the ok=true/false consistent with actual content quality? |
| `reasoning_valid` | Is reasoning self-consistent with the ok/score decision? |
| `suggestions_valid` | Are suggestions specific, actionable, and free of mandatory language? |
| `judge_reasoning` | Overall audit note pointing out specific issues |

---

## 4. Results

### Issue Evaluation

| Metric | Value |
|--------|-------|
| Total rows | 84 |
| Ineligible (skipped) | 25 |
| Evaluated | 59 |
| ok=true (sufficient) | 33 |
| ok=false (insufficient) | 26 |
| Average score | 69.1 / 100 |
| Judged | 59 |

### Issue Judge Results

| Metric | Count |
|--------|-------|
| ok_reasonable=true | 53 |
| ok_reasonable=false | 6 |
| reasoning_valid=true | 46 |
| reasoning_valid=false | 13 |
| suggestions_valid=true | 52 |
| suggestions_valid=false | 7 |

**Confusion matrix:**

| | Judge: reasonable | Judge: not reasonable |
|---|---|---|
| **Eval: ok=true** (passed) | TN = 30 | FP = 3 |
| **Eval: ok=false** (flagged) | TP = 23 | FN = 3 |

- **Accuracy**: 89.8% (53/59)
- **Precision**: 88.5% (23/26) — of flagged issues, 88.5% were genuinely insufficient
- **Recall**: 88.5% (23/26) — of genuinely insufficient issues, 88.5% were correctly flagged

#### False Positives — bot too lenient (passed a bad description)

| # | Title | Judge note |
|---|-------|------------|
| 10599 | `[Bug]`: 模型kimi2.6正常运行一段时间，停掉服务后，重启拉起HCCL报错 | Lacks reproduction steps and explicit environment details |
| 10383 | `[Bug]`: Probabilistic empty outputs under multi-concurrency | No reproduction steps, sample logs, or clear characterization of the error |
| 10226 | `[Bug]`: DeepSeek V4 Flash PDD 64xNPU 2P1D proxy host config | Brief curl command and screenshot alone are insufficient; missing expected/actual behavior |

#### False Negatives — bot too strict (flagged a good description)

| # | Title | Judge note |
|---|-------|------------|
| 10724 | `[Bug]`: [v0.21.0rc1] Crash on Deepseek v4 Flash on 2\*A2 PD-Mix | Description includes complete stack trace and repro steps; bot wrongly claimed logs were missing |
| 10522 | `[Bug]`: 0.20.2rc1 GLM-5.1 PD分离部署，P节点偶现crash | "Describe the bug" section is present with detail; bot incorrectly flagged it as missing |
| 9871 | `[Bug]`: Low MTP acceptance rate for Qwen3.5-122B | Description includes explicit acceptance rate data; bot wrongly called it truncated |

---

### PR Evaluation

| Metric | Value |
|--------|-------|
| Total rows | 61 |
| Evaluated | 61 |
| ok=true (sufficient) | 34 |
| ok=false (insufficient) | 27 |
| Average score | 68.8 / 100 |
| Judged | 61 |

### PR Judge Results

| Metric | Count |
|--------|-------|
| ok_reasonable=true | 60 |
| ok_reasonable=false | 1 |
| reasoning_valid=true | 61 |
| reasoning_valid=false | 0 |
| suggestions_valid=true | 60 |
| suggestions_valid=false | 1 |

**Confusion matrix:**

| | Judge: reasonable | Judge: not reasonable |
|---|---|---|
| **Eval: ok=true** (passed) | TN = 33 | FP = 1 |
| **Eval: ok=false** (flagged) | TP = 27 | FN = 0 |

- **Accuracy**: 98.4% (60/61)
- **Precision**: 96.4% (27/28) — of flagged PRs, 96.4% were genuinely insufficient
- **Recall**: 100.0% (27/27) — all genuinely insufficient PRs were correctly flagged

#### False Positives — bot too lenient

| # | Title | Judge note |
|---|-------|------------|
| 10734 | `[Doc]` Translated Doc files 2026-06-19 | Description is a bare file list with no explanatory summary; lacks substantive content |

#### False Negatives

None.

---

## 5. Conclusion

The review bot pipeline achieves **89–98% accuracy** on description completeness
judgments across issues and PRs.

| Dataset | Total | Evaluated | Judged | Accuracy | Precision | Recall |
|---------|-------|-----------|--------|----------|-----------|--------|
| Issues | 84 | 59 | 59 | 89.8% | 88.5% | 88.5% |
| PRs | 61 | 61 | 61 | 98.4% | 96.4% | 100.0% |

**Key observations:**

- **PR evaluation is near-perfect** (98.4% accuracy, 0 false negatives). The PR
  template is straightforward and the bot reliably identifies incomplete descriptions.
- **Issue evaluation is solid at 89.8%** with 3 false positives (bot too lenient
  on incomplete bug reports, particularly Chinese-language ones missing repro steps)
  and 3 false negatives (bot too strict on descriptions with sufficient detail in
  non-standard format).
- **Reasoning quality**: 78% of issue reasoning and 100% of PR reasoning is
  self-consistent. The lower issue rate reflects borderline cases where the
  bot's judgment is correct but the explanation imprecise.
- **Suggestions quality**: 88% of issue suggestions and 98% of PR suggestions
  are specific, actionable, and free of mandatory language.
- **25 issue rows are ineligible** (e.g. `[Contribution]`, `[RFC]`,
  `[Bug][Upstream]:`) — filtered out by both the production workflow `if:`
  condition and `should_review()` in the test harness, ensuring the test
  accurately reflects production behaviour.
