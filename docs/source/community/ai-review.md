# AI Code Review

The AI review agent provides deep, on-demand code review for pull requests. It complements — never replaces — human review: first-pass screening (CODEOWNERS routing, gemini-code-assist comments) and CI run automatically on every PR, while the AI review agent handles the deep rounds that a reviewer explicitly requests. The agent can never approve or merge; its verdict is advisory input for the human reviewer.

The mechanism was proposed in [RFC #21451](https://github.com/vllm-project/vllm-ascend/issues/21451) and is currently in a **gray release** (limited authors, opt-in per PR).

## How it works, end to end

```text
reviewer applies "ai-review" label
        │
        ▼
workflow gate (write permission + label applier in allowlist)
        │
        ▼
orchestrator assembles prompt ──► dsh agent (headless) ──► findings
        │                                                      │
        ▼                                                      ▼
state restored from last                  one COMMENT-level review with
agent comment (signature block)           inline comments on code lines
        │                                                      │
        ▼                                                      ▼
        └────────────── summary comment (verdict + signature) ──┘
        │
        ▼
multi-round: author replies with /ai-review ──► next round
new push ──► ai-review-passed revoked ──► incremental re-review
```

### Triggering and the gray-release gate

- A reviewer with write permission applies the `ai-review` label to a PR.
- The person applying the label must be listed in `.github/agent-review/allowlist.txt` — the allowlist authorizes operators, not PRs. From anyone else the agent posts a single notice and removes the label. The gray release scales by editing this file; removing lines rolls it back with no code change.
- During the gray release the agent's PR comments are posted by the maintainer's bot account, with an "AI review" banner marking them as machine output and clarifying that `ai-review-passed` is not an approve.

### Review rounds

Each round is one agent invocation. The orchestrator assembles the prompt deterministically:

1. the output protocol (finding format, severities, verdict marker),
2. a working-environment section (where the PR code, docs and guidelines live),
3. a knowledge-base map — file paths plus rule IDs, not full contents,
4. the list of knowledge-base domains the PR touches (resolved from `domains.yaml`),
5. the PR diff, the agent's previous comment, and the author's reply.

The agent reads the referenced checklist files itself from the checked-out repository and cites only rules it has actually read (every finding must carry a `file:line`, a rule ID and a severity — findings without grounding are invalid).

### Multi-round interaction and termination

State is carried in a machine-readable signature block at the end of every agent comment, so nothing needs to be stored outside GitHub:

```text
<!-- ai-review: state=reviewing | round=2/5 | total=6 | head=abc123 -->
```

| Event | Effect |
|---|---|
| Author replies with the `/ai-review` command | next round (plain comments, CI bots and human conversations are ignored) |
| Author pushes a commit | `ai-review-passed` revoked, round budget reset, incremental re-review of the new diff |
| Reviewer comments `/ai-review-waive` on a PR comment | ALL open findings waived (marked on the inline comments), one re-evaluation round runs |
| Reviewer replies `/ai-review-waive` inside one finding's thread | that single finding is waived |
| Reviewer removes the label or comments `/ai-review-stop` | agent stops; a newer trigger always cancels the in-flight round (latest wins) |
| All MUST-FIX resolved | agent emits `AI-REVIEW-VERDICT: PASS`; the orchestrator applies `ai-review-passed` |
| Round cap reached (5 per push) | closing summary, state `waiting-human`, a human decides |
| Whole-PR budget exhausted (15 rounds) | agent goes silent until a reviewer re-applies the label |

`ai-review-passed` is **not an approve**. The agent posts issue comments and COMMENT-level reviews only — it has no permission to approve, merge or push, and human review remains the merge gate.

## For PR authors

1. Ask a reviewer to apply the `ai-review` label (only allowlisted operators can — see below), or become an operator yourself via RFC #21451.
2. When you want a deep review, ask a reviewer to apply the `ai-review` label to your PR.
3. The agent posts findings as inline comments on your diff plus one summary comment. To trigger the next round, add a **top-level PR comment** containing `/ai-review` — replies inside an inline finding thread also reach the agent, but a plain comment without the command never does.
4. Fix what is marked `MUST-FIX` and push. The agent re-reviews the new diff automatically; when nothing blocking remains it applies the `ai-review-passed` label and a human reviewer takes over.
5. Everything is optional: without the label your PR flow is unchanged.

## Knowledge base maintenance

The review rules live in `.github/agent-review/` and follow the project's ownership model — the people who own a module own the review rules for it:

```text
.github/agent-review/
├── instructions.md        # output protocol (maintainers)
├── allowlist.txt          # operators allowed to apply the label (maintainers)
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
| `.github/workflows/scripts/ai_review.py` | state machine, prompt assembly, finding parsing, inline-review posting |
| `.github/workflows/dockerfiles/Dockerfile.ai-review` | preinstalled agent image (Node + dsh CLI), rebuilt manually |
| `.github/workflows/_schedule_ai_review_image_build.yaml` | image build workflow (manual/deliberate, no schedule) |
| `.github/agent-review/` | instructions, knowledge base, allowlist, domain registry |

The agent engine is deliberately decoupled: the orchestrator invokes a configurable command (a repository variable), so the underlying engine can be replaced without workflow surgery.
