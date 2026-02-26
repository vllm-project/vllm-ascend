# vLLM Ascend skills

This directory contains the skills for vLLM Ascend.

Note: Please copy the skills directory `.agents/skills` to `.claude/skills` if you want to use the skills in this repo with Claude code.

## Table of Contents

- [vLLM Ascend Model Adapter Skill](#vllm-ascend-model-adapter-skill)
- [vLLM Ascend main2main Skill](#vllm-ascend-main2main-skill)
- [vLLM Ascend Release Note Writer Skill](#vllm-ascend-release-note-writer-skill)

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

Migrate changes from the main vLLM repository to the vLLM Ascend repository, ensuring compatibility and performance optimizations for Ascend NPUs.

### What it does

This skill facilitates the process of:

1. Identifying changes in the main vLLM repository.
2. Applying necessary modifications for Ascend support.
3. Validating the changes in an Ascend environment.
4. Delivering a ready-to-merge commit with optimized code and configurations.

### Quick start

1. Open a conversation with the AI agent inside the vllm-ascend dev container.
2. Invoke the skill (e.g. `/main2main`).
3. The agent follows the playbook and produces a ready-to-merge commit.

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
