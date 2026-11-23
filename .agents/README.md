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
  - [vLLM Ascend Release Skill](#vllm-ascend-release-skill)
    - [What it does](#what-it-does-1)
    - [File layout](#file-layout-1)
    - [Getting started](#getting-started)
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

## vLLM Ascend Release Skill

Manage the complete release process for vLLM Ascend, from the release checklist issue to the
post-release announcement.

### What it does

This skill guides an AI agent through the end-to-end release workflow:

1. Create the release feedback issue and the release checklist issue.
2. Triage release-blocking bugs and identify must-merge PRs.
3. Check test coverage for new features and models, and scan Nightly CI status.
4. Assemble release notes from `release_notes/fragments/` and audit commits without fragments.
5. Update version references across the documentation.
6. Guide release execution and post-release steps, including the WeChat article.

### File layout

| File | Purpose |
| ---- | ------- |
| `SKILL.md` | Phases, commands, and human decision points for the full release process |
| `scripts/` | Release automation: checklist creation, bug/PR/test/nightly scans, commit fetch, release note aggregation, version updates, announcements |
| `templates/` | Release checklist and feedback issue templates |
| `references/version-files.yaml` | Files that need version updates and their update patterns |
| `references/ref-past-release-notes-highlight.md` | Style and category reference for past release notes |

### Getting started

1. Open a conversation with the AI agent.
2. Invoke the skill (e.g. `/vllm-ascend-release`).
3. Provide the release version, branch, target date, and release manager.
4. Work through the phases in `SKILL.md`; the skill pauses for human review at the key
   decision points (bug triage, functional testing, release execution).

### Key guidelines

- Release notes are assembled from `release_notes/fragments/`; the audit step lists commits
  without fragments so nothing user-facing is missed. The fragment principles live in
  `release_notes/README.md`.
- Human oversight is required for bug prioritization, test results, and the final release
  decision.
- Scripts are idempotent and safe to re-run; checklist updates replace a section instead of
  appending duplicates.