# AI Code Review

The AI review agent provides deep, on-demand code review for pull requests. It complements — never replaces — human review: first-pass screening (CODEOWNERS routing, gemini-code-assist comments) and CI run automatically on every PR, while the AI review agent handles the deep rounds that a reviewer explicitly requests. The agent can never approve or merge; its verdict is advisory input for the human reviewer.

The mechanism was proposed in [RFC #21451](https://github.com/vllm-project/vllm-ascend/issues/21451) and is currently in a **gray release** (limited reviewers, opt-in per PR).

## How it works, end to end

```text
reviewer applies "ai-review" label
        │
        ▼
workflow gate (write permission + reviewer in allowlist)
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

- A reviewer with write permission applies the `ai-review` label to a PR.
- The reviewer applying the label must be listed in `.github/agent-review/allowlist.txt`. If that reviewer is missing from the list, the agent posts a notice and removes the label. The gray release scales by editing this file; removing lines rolls it back with no code change.
- During the gray release the agent's PR comments are posted by the maintainer's bot account, with the summary headed **AI review · DeepSeek Harness headless · POC pilot**. The bot appends an Actions link to accepted command comments and explains rejections on the original comment.

### Review rounds

Each round is one agent invocation. The orchestrator assembles the prompt deterministically:

1. the output protocol (finding format, severities and updates),
2. a working-environment section (where the PR code, docs and guidelines live),
3. a knowledge-base map — file paths plus rule IDs, not full contents,
4. the list of knowledge-base domains the PR touches (resolved from `domains.yaml`),
5. the PR diff from the merge base to the current head, all previous findings, and the triggering comment, if any. A truncated diff cannot pass review.

The agent reads the referenced checklist files itself from the checked-out repository and cites only rules it has actually read (every finding has a file path, a rule ID and a severity, with a line number for inline findings).

### Multi-round interaction and termination

The bot updates one PR summary comment. Its hidden `ai-review-state` JSON block records the reviewed commit, published rounds and finding history. A round counts when the bot publishes its result or a notice that the code changed during review. If a run fails before publication, a rerun starts a fresh review from the last published summary.

| Event | Effect |
|---|---|
| PR author or reviewer with write permission comments `/ai-review` | Request one review of the current PR. Include explanations for multiple findings in the same top-level comment. |
| Author pushes a commit | Revoke `ai-review-passed`. The next `/ai-review` reviews all changes together. |
| Reviewer comments `/ai-review-waive F003 reason` or `/ai-review-waive all reason` | Waive one or all open findings for the reviewed commit. Existing inline comments show `[WAIVED by @user]`. The bot updates the verdict without calling the agent. |
| Reviewer removes the label or comments `/ai-review-stop` | Stop review and remove both labels. Wait for the Stop workflow to finish before reapplying `ai-review`. |
| Review covers the full diff, no MUST-FIX remains open and no human decision is pending | The orchestrator applies `ai-review-passed`. |
| Round cap reached (5 per commit) | Push changes and request review again, or ask an allowlisted reviewer to restart the session. |
| Session budget exhausted (15 published rounds) | An allowlisted reviewer can remove and reapply the label to reset both counts. Existing findings are retained. |

Each PR allows one running review job and one pending job. When another authorized request enters a full queue, it replaces the pending job.

`ai-review-passed` is **not an approve**. Human review remains the merge gate.

## For PR authors

1. When you want a deep review, ask an allowlisted reviewer to apply the `ai-review` label to your PR.
2. Read the summary comment for the verdict and the full list of findings. The agent also posts findings inline when it can attach them to a changed line.
3. Fix what is marked `MUST-FIX` and push your changes. When ready, add a new **top-level PR comment** starting with `/ai-review`. Inline replies do not start a round.
4. Everything is optional: without the label your PR flow is unchanged.

## Knowledge base maintenance

The review rules live in `.github/agent-review/` and follow the project's ownership model — the people who own a module own the review rules for it:

```text
.github/agent-review/
├── instructions.md        # output protocol (maintainers)
├── allowlist.txt          # gray-release reviewers (maintainers)
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
