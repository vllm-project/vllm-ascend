# AI Code Review

The AI review agent provides deep, on-demand code review for pull requests. It complements — never replaces — human review: first-pass screening (CODEOWNERS routing, gemini-code-assist comments) and CI run automatically on every PR, while the AI review agent handles the deep rounds that a reviewer explicitly requests. The agent can never approve or merge; its verdict is advisory input for the human reviewer.

The mechanism was proposed in [RFC #21451](https://github.com/vllm-project/vllm-ascend/issues/21451) and is currently in a **gray release** (limited reviewers, opt-in per PR).

## How it works, end to end

```text
reviewer applies "ai-review" label
        │
        ▼
workflow checks the reviewer's allowlist membership
        │
        ▼
orchestrator reads the previous summary and assembles the prompt
        │
        ▼
dsh agent (headless) ──► new findings posted inline where possible
        │
        ▼
summary comment updated (verdict, findings and saved state)
        │
        ▼
new top-level /ai-review comment ──► next round
new push ──► ai-review-passed revoked
```

### Triggering and the gray-release gate

- An operator listed in `.github/agent-review/allowlist.txt` can start a review by applying the `ai-review` label to a PR. Applying the label requires permission to manage PR labels on GitHub. If an unlisted person applies it, the bot posts a notice and removes the label.
- Once enabled, any listed operator can request another review, waive findings, or stop the review by commenting on the PR. The same list applies to PR authors and maintainers. To join the pilot, comment on RFC #21451.
- During the gray release the maintainer's bot account posts the review comments. A command gets one feedback line on the original comment, such as `[🤖 AI review bot]: ⏳ Review requested` with a **View run** link. A rejection or completed waiver updates that line.

### Review rounds

Each round is one agent invocation. The prompt includes the following information.

1. The path to the trusted project guidelines, the knowledge-base map and the domains the PR touches.
2. The PR title and description, target branch tip, merge base, head and `git diff --stat`.
3. The last published review's head, merge base, coverage status and summary, plus a separate list of unfinished static checks.
4. Open and resolved findings, waived findings and their reasons, the triggering comment (if any) and the expected review format.

The agent runs in a Git worktree at the reviewed head and uses Git to inspect changes from the merge base. It reads applicable checklists from the trusted checkout. After about ten minutes, it is asked to finish with the evidence it has. The workflow stops the agent after twelve minutes.

The review summary explains what the agent checked, what it concluded and which claims still need runtime evidence. A separate list names the static checks left for the next round. A round with unfinished checks cannot pass. An allowlisted operator can comment `/ai-review` to continue those checks and review any new changes. A completed review has an empty list and keeps its summary for the next round.

### Documentation changes

When the current PR diff changes either root README (`README.md` or `README.zh.md`), Markdown in `docs/` or `vllm_ascend/ops/triton/docs/`, or Chinese translations in `docs/source/locale/zh_CN/LC_MESSAGES/`, the agent reviews the documentation against internal standards and reports documentation findings alongside code findings in the same round.

If you withdraw a documentation change, include the reason and finding ID in the next review request. The agent can mark that finding `resolved` after verifying the affected content was withdrawn. If you moved the content and the issue persists, the finding stays open.

### Multi-round interaction and termination

Before the first review, the bot posts the summary title so the summary appears before the inline discussions. The heading changes to **Reviewing** when the agent step starts. Once the inline comments are ready, the bot updates the summary with the outcome and finding links. **Review passed** means no blocking findings remain. SHOULD-FIX and NIT items can still be open. During later rounds the last published result stays visible below the current activity. A failed review or publication shows **Review failed** with a **View logs** link. The hidden `ai-review-state` JSON block records the reviewed commit, published rounds and finding history. Only a published result or a notice that the code changed during review counts as a round.

Waivers, stop commands and discarded results can update the comment. The next review still receives the snapshot, summary and unfinished checks from the last round that published a review result.

| Event | Effect |
|---|---|
| Allowlisted operator comments `/ai-review` | Request one review of the current PR. Add a note such as `/ai-review F001 F002 fixed in this commit` to explain changes to several findings in one comment. The agent still reviews the PR as a whole. |
| Author pushes a commit | Revoke `ai-review-passed`. The next `/ai-review` reviews all changes together. |
| Allowlisted operator comments `/ai-review-waive F003 reason` or `/ai-review-waive all reason` | Permanently waive one or all open finding IDs on this PR. The original comment shows who waived the finding, the commit and the reason. The bot updates the verdict without calling the agent. |
| Someone removes the label on GitHub, or an allowlisted operator comments `/ai-review-stop` | Stop review and remove both labels. Wait for the Stop workflow to finish before reapplying `ai-review`. |
| Review covers the full diff, no MUST-FIX remains open and no human decision is pending | The orchestrator applies `ai-review-passed`. |
| Round cap reached (5 per commit) | Push changes and request review again, or ask an allowlisted reviewer to restart the session. |
| Session budget exhausted (15 published rounds) | An allowlisted reviewer can remove and reapply the label to reset both counts. Existing findings are retained. |

Waived IDs stay archived in the PR summary after a push. Before reporting a new problem, the agent consults the waiver list to avoid repeating a behavior and impact the reviewer already accepted. A distinct defect or an impact outside the waiver can receive a new ID, even on the same line.

The summary shows a collapsed group for each severity, including groups with zero open findings. Expand a group and follow a finding ID to its inline discussion. Resolved and waived findings appear in separate groups when present. If GitHub cannot attach a finding to a changed line, its details appear in the summary. Expand the review details to read the last round's conclusions.

Each new inline comment explains the problem and suggests a fix. A precise local fix includes a read-only diff against the reviewed commit. The original suggestion stays visible after a finding is resolved or waived. The outcome and its evidence appear below it.

Each PR allows one running review job and one pending job. When another authorized request enters a full queue, it replaces the pending job.

`ai-review-passed` is **not an approve**. Human review remains the merge gate.

## For PR authors

1. When you want a deep review, ask an allowlisted reviewer to apply the `ai-review` label to your PR.
2. Read the summary comment for the verdict and the full list of findings. Follow a finding ID to its inline discussion. The **Review commands** section at the bottom explains how to request another review, waive a finding, or stop AI review for the PR.
3. Fix what is marked `MUST-FIX` and push your changes. When ready, ask an allowlisted operator to post a **top-level PR comment** starting with `/ai-review`, or post it yourself if you are on the list. Inline replies do not start a round.
4. Everything is optional: without the label your PR flow is unchanged.

## Knowledge base maintenance

The review rules live in `.github/agent-review/` and follow the project's ownership model — the people who own a module own the review rules for it:

```text
.github/agent-review/
├── instructions.md        # output protocol (maintainers)
├── allowlist.txt          # operators who can start reviews and use bot commands
├── domains.yaml           # coarse path-glob → domain registry (maintainers)
├── L1/checklist.md        # cross-domain global rules (maintainers)
└── L2/<domain>/           # one directory per domain, owned by that
    └── checklist.md       #   domain's CODEOWNERS owners
```

- **Ownership is enforced by CODEOWNERS.** Each `L2/<domain>/` directory gets one CODEOWNERS entry naming that domain's owners (later entries win over the `/.github` wildcard). Editing a domain's rules then requires those owners' approval — the same gate as editing the code itself.
- **Updating rules** is an ordinary PR against the checklist file. Changes are effective for the next review round; there is no deployment step.
- **Adding a domain** takes three steps: create `L2/<domain>/checklist.md` with stable rule IDs (`L2-<domain>#slug`), add a coarse glob entry to `domains.yaml`, append the CODEOWNERS entry. Keep domains coarse — per-file directories are explicitly avoided.
- **Rule discipline**: rule IDs are stable API (findings cite them); every rule must be decidable from the diff and repository files (things needing test runs or hardware belong to human review); explicitly exclude what the agent should not report.

During the gray release, maintainer feedback on the agent's inline comments (agree/disagree/corrections) is the main input for checklist iteration. Collect it and land checklist updates in small weekly batches rather than one change per finding.

## Components

| Path | Role |
|---|---|
| `.github/workflows/ai_review.yml` | trigger orchestration (label / comment / push), gates, agent job on the self-hosted CPU pool |
| `.github/workflows/scripts/ai_review.py` | state machine, prompt assembly, finding parsing, inline comment posting |
| `.github/workflows/dockerfiles/Dockerfile.ai-review` | preinstalled agent image (Node + dsh CLI) |
| `.github/workflows/_schedule_ai_review_image_build.yaml` | image build workflow (Dockerfile changes or manual dispatch) |
| `.github/agent-review/` | instructions, knowledge base, allowlist, domain registry |

The agent engine is deliberately decoupled: the orchestrator invokes a configurable command (a repository variable), so the underlying engine can be replaced without workflow surgery.
