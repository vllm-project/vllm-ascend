# Review Bot Scenarios

## Issue Bot — `bot_issue_review.yaml`

### Guard

```
state == open
AND (opened OR edited with title/body changes)
AND title prefix matches [Bug]: [Install]: [Doc]: [Misc]: [Feature]: [Perf]:
```

### Steps

```
Extract type → Load template → Load prompt → LLM check → Post/Delete comment → Manage label
```

### States

- **Clean**: No `need-detail-desc` label, no bot comment
- **Flagged**: Has `need-detail-desc` label + 1 bot comment

### Scenarios

| # | From | Trigger | Desc result | Comment | Label |
|---|------|---------|:--:|---------|-------|
| I1 | (new) | `opened` | PASS | — | — |
| I2 | (new) | `opened` | FAIL | Post | +need-detail-desc |
| I3 | Clean | `edited` | PASS | — | — |
| I4 | Clean | `edited` | FAIL | Post | +need-detail-desc |
| I5 | Flagged | `edited` | PASS | new post pass | −need-detail-desc |
| I6 | Flagged | `edited` | FAIL | new posted | (unchanged) |

### Concurrency

```yaml
concurrency:
  group: issue-review-${{ github.event.issue.number }}
  cancel-in-progress: false
```

**Why `cancel-in-progress: false`?**

When events fire rapidly on the same issue (e.g. user opens then immediately edits), `false` queues them:

```
opened  ──► run ████████████ done
edited  ──────────────── queued ──► run ████████ done
```

With `true`, the `opened` run would be **cancelled mid-LLM-call**, leaving the issue in an inconsistent state (partial labels, no comment). `false` guarantees each run completes with a consistent result.

**Why per-issue grouping?**

`issue-review-5` only serialises runs for issue #5. Issue #7 runs in parallel — different group key. No unnecessary blocking across different issues.

---

## PR Bot — `bot_pr_review.yaml`

### Guard

```
state == open (all PRs, no title prefix filter)
```

### Phase execution per trigger

| Trigger | Desc runs | Commit runs |
|---------|:--:|:--:|
| `opened` | Yes | Yes |
| `reopened` | Yes | Yes |
| `edited` + title/body changed | Yes | — |
| `synchronize` + body changed | Yes | Yes |
| `synchronize` + body unchanged | — | Yes |

When a phase is skipped, a placeholder JSON with `"executed": false` is written.

### Steps

```
Extract PR info → Load template → Load desc prompt → LLM desc → (skip placeholder)
→ Load commit prompt → Check commits → (skip placeholder) → Post/Delete comment → Manage labels
```

### States

- **Clean**: No labels, no bot comment
- **Desc-flagged**: `need-detail-desc`, 1 bot comment
- **Commit-flagged**: `need-commit-fix`, 1 bot comment
- **Both-flagged**: both labels, 1 bot comment

### Label logic

Only manages a label when that phase actually executed:

```
Desc executed + PASS  → remove need-detail-desc
Desc executed + FAIL  → add    need-detail-desc
Desc skipped          → (don't touch need-detail-desc)

Commit executed + PASS → remove need-commit-fix
Commit executed + FAIL → add    need-commit-fix
Commit skipped         → (don't touch need-commit-fix)
```

### Scenarios — opened / reopened

| # | From | Desc | Commit | Comment | Labels |
|---|------|:--:|:--:|---------|--------|
| P1 | (new) | PASS | PASS | — | — |
| P2 | (new) | FAIL | PASS | Post | +need-detail-desc |
| P3 | (new) | PASS | FAIL | Post | +need-commit-fix |
| P4 | (new) | FAIL | FAIL | Post | +both |

### Scenarios — edited (desc runs, commit skipped)

| # | From | Desc | Commit | Comment | Labels |
|---|------|:--:|:--:|---------|--------|
| P5 | Clean | PASS | skip | — | — |
| P6 | Clean | FAIL | skip | Post | +need-detail-desc |
| P7 | Desc-flagged | PASS | skip | new post pass | −need-detail-desc |
| P8 | Desc-flagged | FAIL | skip |  new posted | (unchanged) |
| P9 | Commit-flagged | PASS | skip | new post pass | (unchanged) |
| P10 | Commit-flagged | FAIL | skip |  new posted | +need-detail-desc |
| P11 | Both-flagged | PASS | skip |  new post pass | −need-detail-desc |
| P12 | Both-flagged | FAIL | skip |  new posted | (unchanged) |

**Key**: Commit phase skipped → commit label never touched. P9–P12 preserve the commit label correctly.

### Scenarios — synchronize (body unchanged, desc skipped, commit runs)

| # | From | Desc | Commit | Comment | Labels |
|---|------|:--:|:--:|---------|--------|
| P13 | Clean | skip | PASS | — | — |
| P14 | Clean | skip | FAIL | Post | +need-commit-fix |
| P15 | Commit-flagged | skip | PASS | new post pass | −need-commit-fix |
| P16 | Commit-flagged | skip | FAIL |  new posted | (unchanged) |
| P17 | Desc-flagged | skip | PASS | new post pass | (unchanged) |
| P18 | Desc-flagged | skip | FAIL |  new posted | +need-commit-fix |
| P19 | Both-flagged | skip | PASS | new post pass | −need-commit-fix |
| P20 | Both-flagged | skip | FAIL |  new posted | (unchanged) |

**Key**: Desc phase skipped → desc label never touched. P17–P20 preserve the desc label correctly.

### Scenarios — synchronize (body changed, both run)

Same as `opened`. Both phases execute, labels managed normally.

### Concurrency

```yaml
concurrency:
  group: pr-review-${{ github.event.pull_request.number }}
  cancel-in-progress: false
```

**Why `cancel-in-progress: false`?**

Force-push to a PR fires both `edited` and `synchronize` in quick succession. The correct sequence is `edited` first (cleans up old comment, updates desc label) → then `synchronize` (rechecks commits, updates commit label).

```
edited ──► run ████████████ done
sync   ──────────────── queued ──► run ████████ done
```

With `cancel-in-progress: true`, `edited` would be **killed mid-run** when `sync` arrives. The LLM call, label update, and comment cleanup would all be aborted — leaving stale labels and comments.

**Why per-PR grouping?**

`pr-review-42` only serialises runs for PR #42. PR #43 runs independently. Different group keys = parallel execution where safe.
