# vLLM Ascend skills

This directory contains the skills for vLLM Ascend.

Note: Please copy the skills directory `.agents/skills` to `.claude/skills` if you want to use the skills in this repo with Claude code.

## Table of Contents

- [vLLM Ascend skills](#vllm-ascend-skills)
  - [Table of Contents](#table-of-contents)
  - [vLLM Ascend Model Adapter Skill](#vllm-ascend-model-adapter-skill)
    - [What it does](#what-it-does)
    - [File layout](#file-layout)
    - [Quick start](#quick-start)
    - [Key constraints](#key-constraints)
    - [Two-stage validation](#two-stage-validation)
  - [vLLM Ascend main2main Skill](#vllm-ascend-main2main-skill)
    - [What it does](#what-it-does-1)
    - [Quick start](#quick-start-1)
  - [vLLM Ascend Release Note Writer Skill](#vllm-ascend-release-note-writer-skill)
    - [What it does](#what-it-does-2)
    - [File layout](#file-layout-1)
    - [Quick start](#quick-start-2)
    - [Key guidelines](#key-guidelines)


## vLLM Ascend Model Adapter Skill

Adapt and debug models for vLLM on Ascend NPU — covering both already-supported
architectures and new models not yet registered in vLLM.

### What it does

This skill guides an AI agent through a deterministic workflow to:

1. Triage a model checkpoint (architecture, quant type, multimodal capability).
2. Implement minimal code changes in `/vllm-workspace/vllm` and `/vllm-workspace/vllm-ascend`.
3. Validate via a two-stage gate (dummy fast gate + real-weight mandatory gate).
4. Deliver one signed commit with code, test config, and tutorial doc.

### File layout

| File | Purpose |
| ---- | ------- |
| `SKILL.md` | Skill definition, constraints, and execution playbook |
| `references/workflow-checklist.md` | Step-by-step commands and templates |
| `references/troubleshooting.md` | Symptom-action pairs for common failures |
| `references/fp8-on-npu-lessons.md` | FP8 checkpoint handling on Ascend |
| `references/multimodal-ep-aclgraph-lessons.md` | VL, EP, and ACLGraph patterns |
| `references/deliverables.md` | Required outputs and commit discipline |

### Quick start

1. Open a conversation with the AI agent inside the vllm-ascend dev container.
2. Invoke the skill (e.g. `/vllm-ascend-model-adapter`).
3. Provide the model path (default `/models/<model-name>`) and the originating issue number.
4. The agent follows the playbook in `SKILL.md` and produces a ready-to-merge commit.

### Key constraints

- Never upgrade `transformers`.
- Start `vllm serve` from `/workspace` (direct command, port 8000).
- Dummy-only evidence is not sufficient — real-weight validation is mandatory.
- Final delivery is exactly one signed commit in the current repo.

### Two-stage validation

- **Stage A (dummy)**: fast architecture / operator / API path check with `--load-format dummy`.
- **Stage B (real)**: real-weight loading, fp8/quant path, KV sharding, runtime stability.

Both stages require request-level verification (`/v1/models` + at least one chat request),
not just startup success.

## vLLM Ascend main2main Skill

Adapt vLLM-Ascend to upstream vLLM main branch evolution, either proactively before breakage lands or reactively when main2main CI is already red.

### What it does

This skill supports two workflows:

1. **Proactive upgrade**
   - Compare the currently adapted vLLM commit with a newer upstream commit.
   - Generate a prioritized change report for vLLM-Ascend-relevant upstream changes.
   - Apply Ascend-side adaptations before scheduled CI starts failing.
2. **CI failure diagnosis**
   - Analyze a failing main2main GitHub Actions run or run ID.
   - Use the repository summary tooling to extract failed tests, code bugs, and environment flakes.
   - Correlate failures with upstream vLLM commits, generate a diagnostic report, and apply adaptation fixes.

Both workflows produce code changes in the working tree and a structured summary for follow-up git/PR operations. The skill itself does not create branches, commits, or PRs.

### File layout

| File | Purpose |
| ---- | ------- |
| `SKILL.md` | Skill entrypoint and workflow routing rules |
| `proactive-upgrade.md` | Playbook for forward-looking upstream upgrade analysis |
| `error-analysis.md` | Playbook for diagnosing and fixing failing main2main CI |
| `reference/error-patterns.md` | Concrete upstream-change-to-fix patterns |

### Quick start

1. Open a conversation with the AI agent inside the vllm-ascend dev container.
2. Invoke the skill (e.g. `/main2main`).
3. Choose one of the two common entry modes:
   - **Upgrade mode**: ask the agent to analyze upstream vLLM changes and adapt vllm-ascend to a newer commit.
   - **Failure-analysis mode**: provide a GitHub Actions URL or run ID for a failing main2main / schedule test run.
4. The agent follows the corresponding playbook and produces:
   - code changes in the working tree
   - a structured summary of causes and fixes
   - when applicable, a diagnostic report such as `vllm_error_analyze.md`

### Key guidance

- Use this skill whenever the task is about upstream vLLM main branch drift, not ordinary feature development.
- If the request contains both “upgrade” and “CI is failing” signals, treat it as a CI failure diagnosis task first.
- For CI diagnosis, prefer running the repository summary script first instead of reading raw logs directly.
- Most compatibility fixes should preserve both pinned-release and main-branch behavior via `vllm_version_is()` guards.

## vLLM Ascend Release Note Writer Skill

You just need to say: `Please help me write a 0.13.0 release note based on commits from v0.11.0 and releases/v0.13.0`

### What it does

This skill guides you through a structured workflow to:

1. Fetch commits between two versions using the provided script.
2. Analyze and categorize each commit in a CSV workspace.
3. Draft highlights and write polished release notes.
4. Generate release notes organized by category (Features, Hardware Support, Performance, Dependencies, etc.).

### File layout

| File | Purpose |
| ---- | ------- |
| `SKILL.md` | Skill definition, workflow, and writing guidelines |
| `references/ref-past-release-notes-highlight.md` | Style and category reference for release notes |
| `scripts/fetch_commits-optimize.py` | Script to fetch commits between versions |

### Quick start

1. Open a conversation with the AI agent.
2. Invoke the skill (e.g. `/vllm-ascend-release-note-writer`).
3. Follow the workflow steps:
   - Fetch commits between versions
   - Analyze commits in CSV format
   - Draft and edit highlights
4. Output files are saved to `vllm-ascend-release-note/output/$version`

### Key guidelines

- Use one-level headings (###) for sections in a specific order: Highlights, Features, Hardware and Operator Support, Performance, Dependencies, Deprecation & Breaking Changes, Documentation, Others.
- Focus on user-facing impact and include context for practical usage.
- Verify details by checking linked PRs (use GitHub API for descriptions if needed).
- Keep notes concise and avoid unnecessary technical details.
