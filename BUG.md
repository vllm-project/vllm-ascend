# Review Bot Bugs

Found by `test_scenarios.py` — E2E lifecycle test covering all 26 scenarios from
`.github/workflows/scripts/robot/tests/SCENARIOS.md`.

## Bug 1: Issue label management never applies
**File:** `.github/workflows/bot_issue_review.yaml:91` /
`.github/workflows/scripts/robot/handle_review.py:250`

`handle_review.py` writes `label_actions.json` format (`{"add":["need-detail-desc"],"remove":[]}`)
to `label_action.txt`. The workflow's JS step reads `label_action.txt` as plain text and checks
`action === 'add'` / `action === 'remove'`. The JSON object never matches, so **labels are
never managed for issues**.

**Fix options:**
1. Change the JS to `JSON.parse(fs.readFileSync(...))` (matching `bot_pr_review.yaml`)
2. Change the issue path in `handle_review.py` to write plain text (`"add"` / `"remove"` / `"none"`)

## Bug 2: PR commit check flags all commits (including good ones)
**File:** `.github/workflows/scripts/robot/commit_check.py:46`, `.github/workflows/bot_pr_review.yaml:97-98`

`hard_check()` returns `HARD_FAIL` for properly formatted, signed-off commits like
`feat(npu): add optimised memory allocator` with body containing `Signed-off-by: ...`.

Likely cause: `VALID_TYPES` or `REQUIRE_SIGNOFF` environment variables set via
`${{ vars.PR_COMMIT_CHECK_VALID_TYPES || '...' }}` may contain whitespace from repo
variables, causing `"feat"` to not match `" feat"` after `split(",")`.

## Bug 3: No "pass" comments on successful state transitions
**File:** `.github/workflows/scripts/robot/handle_review.py:217`

`needs_comment` logic only posts when `not desc_ok or (commit_executed and not commit_ok)`
— i.e., only on **failures**. SCENARIOS.md specifies "new post pass" comments when
flagged→clean transitions succeed (P5, P7, P9, P11, P13, P15, P17, P19, I5).

**Fix:** Add `or (desc_executed and desc_ok) or (commit_executed and commit_ok)` to
`needs_comment`, and post a "pass" variant comment.

## Bug 4: SCENARIOS.md "Pass" comment column is aspirational
**File:** `.github/workflows/scripts/robot/tests/SCENARIOS.md`

Related to Bug 3 — the SCENARIOS document specifies "new post pass" for scenarios where the
bot should acknowledge improvement. The implementation currently doesn't match this spec.
Either the spec or the code should be updated to align.
