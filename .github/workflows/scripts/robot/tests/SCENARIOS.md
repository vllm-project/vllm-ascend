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
Extract type → Load template → Load prompt → LLM check → Post comment → Manage label
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

Rapid open+edit on the same issue. With `false`, they queue:

```
opened  ──► run ████████████ done
edited  ──────────────── queued ──► run ████████ done
```

With `true`, `opened` is cancelled mid-LLM-call → partial labels, no comment.
`false` guarantees each run completes with a consistent result.

Trade-off: `edited` waits for `opened`'s LLM call (~30s). Acceptable.

**Why per-issue grouping?**

`issue-review-5` serialises runs for issue #5 only. Issue #7 runs independently.

---

## PR Bot — `bot_pr_review.yaml`

### Guard

```
state == open (all PRs, no title prefix filter)
```

### Trigger execution

| Trigger | Desc runs |
|---------|:--:|
| `opened` | Yes |
| `reopened` | Yes |
| `edited` + title/body changed | Yes |
| `synchronize` + body changed | Yes |
| `synchronize` + body unchanged | — |

When desc is skipped, a placeholder JSON with `"executed": false` is written.

### Steps

```
Extract PR info → Load template → Load prompt → LLM desc → (skip placeholder)
→ Post comment → Manage label
```

### States

- **Clean**: No labels, no bot comment
- **Desc-flagged**: `need-detail-desc`, 1 bot comment

### Label logic

```
Desc executed + PASS  → remove need-detail-desc
Desc executed + FAIL  → add    need-detail-desc
Desc skipped          → (don't touch need-detail-desc)
```

### Scenarios — opened / reopened

| # | From | Desc | Comment | Labels |
|---|------|:--:|---------|--------|
| P1 | (new) | PASS | — | — |
| P2 | (new) | FAIL | Post | +need-detail-desc |

### Scenarios — edited (desc runs)

| # | From | Desc | Comment | Labels |
|---|------|:--:|---------|--------|
| P3 | Clean | PASS | — | — |
| P4 | Clean | FAIL | Post | +need-detail-desc |
| P5 | Desc-flagged | PASS | new post pass | −need-detail-desc |
| P6 | Desc-flagged | FAIL | new posted | (unchanged) |

### Scenarios — synchronize (body changed, desc runs)

| # | From | Desc | Comment | Labels |
|---|------|:--:|---------|--------|
| P7 | Clean | PASS | — | — |
| P8 | Clean | FAIL | Post | +need-detail-desc |
| P9 | Desc-flagged | PASS | new post pass | −need-detail-desc |
| P10 | Desc-flagged | FAIL | new posted | (unchanged) |

### Scenarios — synchronize (body unchanged, desc skipped)

| # | From | Desc | Comment | Labels |
|---|------|:--:|---------|--------|
| P11 | Clean | skip | — | — |
| P12 | Desc-flagged | skip | — | (unchanged) |

### Concurrency

```yaml
concurrency:
  group: pr-review-${{ github.event.pull_request.number }}
  cancel-in-progress: false
```

**Why `cancel-in-progress: false`?**

Force-push fires both `edited` + `synchronize`. Each does different work:

| Run | Desc |
|-----|:--:|
| `edited` | Runs (if body changed) |
| `synchronize` | Runs (if body changed) |

If `edited` is cancelled, the desc check is **lost** — `synchronize` may skip desc.
The desc label is never updated. With `false`, both complete in order:

```
edited ──► run ████████ done
sync   ──────────────── queued ──► run ████████ done
```

Trade-off: `sync` waits for `edited` (~5s for a no-body-change `edited`, ~30s if body changed). Acceptable — correctness over latency.

With `cancel-in-progress: true`, `edited` is killed mid-run, desc label is stale forever.

**Why per-PR grouping?**

`pr-review-42` serialises runs for PR #42 only. PR #43 runs independently — different group keys, no cross-PR blocking.
