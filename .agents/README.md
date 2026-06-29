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
  - [vLLM Ascend Release Note Writer Skill](#vllm-ascend-release-note-writer-skill)
    - [What it does](#what-it-does-2)
    - [File layout](#file-layout-1)
    - [Quick start](#quick-start-2)
    - [Key guidelines](#key-guidelines)
  - [vLLM Ascend Serving Auto Benchmark Skill](#vllm-ascend-serving-auto-benchmark-skill)
    - [What it does](#what-it-does-3)
    - [File layout](#file-layout-2)
    - [Quick start](#quick-start-3)
    - [Key guidelines](#key-guidelines-2)
  - [vLLM Ascend Serving Tune Loop Skill](#vllm-ascend-serving-tune-loop-skill)
    - [What it does](#what-it-does-4)
    - [File layout](#file-layout-3)
    - [Quick start](#quick-start-4)
    - [Key guidelines](#key-guidelines-3)


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

## vLLM Ascend Serving Auto Benchmark Skill

Automated benchmark skill for vLLM-Ascend serving on Huawei Ascend 910C NPU. Launches vLLM serving, runs evalscope perf stress tests, and produces a complete Markdown/CSV performance report.

### What it does

This skill runs an automated five-phase benchmark workflow:

1. Validate environment and config (NPU devices, model path, evalscope/vllm-ascend install).
2. Launch the vLLM-Ascend OpenAI-compatible server with the chosen model and parallelism.
3. Run evalscope perf stress tests across configurable concurrency levels.
4. Collect metrics (`ttft_avg`, `tpot_avg`, `output_token_throughput`, `latency_avg`) at each level.
5. Generate a Markdown/CSV performance report via `scripts/generate_report.py`.

### File layout

| File | Purpose |
| ---- | ------- |
| `SKILL.md` | Skill definition, phases, input parameters, and report format |
| `configs/*.yaml` | Pre-built model configs (DeepSeek-V3, DeepSeek-V4-Flash, Qwen3-32B, Qwen3.5-27B, Qwen3.5-35B-A3B) |
| `scripts/run_benchmark.sh` | Orchestrates server launch + evalscope run + report generation |
| `scripts/validate_configs.py` | Validates a YAML config before launch |
| `scripts/generate_report.py` | Produces the Markdown/CSV performance report |

### Quick start

1. Open a conversation with the AI agent inside the vllm-ascend dev container.
2. Invoke the skill (e.g. `/ascend-vllm-serving-auto-benchmark`).
3. Provide `model_path` and `model_name` (or pass a `config_file` from `configs/`).
4. The agent runs the five-phase workflow and writes the report to the output directory.

### Key guidelines

- Concurrency levels and requests-per-level are fixed for a given run — do not change them mid-run.
- Always launch the server on a free port and clean it up after the run.
- A failed server startup must still produce a report entry marked as failure, not a silent skip.
- Reuse a ready-to-use `configs/*.yaml` when benchmarking a known model instead of hand-tuning flags.

## vLLM Ascend Serving Tune Loop Skill

Autonomous, evidence-driven performance optimization loop for vLLM-Ascend serving on Huawei Ascend 910C NPU. Runs a fixed baseline, then iterates through a layered tuning plan (scheduler → ACLGraph → kernel) and produces an optimization ledger plus a final comparison report.

### What it does

This skill drives a long-running optimization campaign across 7 phases:

1. Establish a fixed baseline benchmark (the contract that never changes between iterations).
2. Iterate over ranked tuning levers — one lever per iteration, change one thing at a time (RLCR principle).
3. Revalidate at c=1/4/8 with the fixed workload after every change; gate WIN/LOSS on `ttft_avg` at c=1.
4. Append every verdict (WIN/LOSS/NEUTRAL/STARTUP_FAIL/BENCHMARK_FAIL/SKIPPED_CONFLICT) to the ledger immediately.
5. Resume cleanly across session interruptions via the ledger; drive the campaign to completion with the auto-resume wrapper.
6. Synthesize a final report (`final_report.md`) with baseline-vs-best, winning copy-pasteable config, and failed levers.

### File layout

| File | Purpose |
| ---- | ------- |
| `SKILL.md` | Skill definition, phases, ledger discipline, resume logic, report format |
| `references/tuning-knobs-reference.md` | Catalog of tunable Ascend knobs and their expected impact |
| `scripts/auto_resume_wrapper.sh` | Relaunches the `claude` session until `final_report.md` exists (cross-session resume) |
| `scripts/generate_tuning_report.py` | Synthesizes `final_report.md` from the ledger |

### Quick start

1. Open a conversation with the AI agent inside the vllm-ascend dev container.
2. Invoke the skill (e.g. `/ascend-vllm-serving-tune-loop`).
3. Provide `model_path`, `model_name`, and `tensor_parallel_size`; optionally `baseline_launch_command` to preserve a known-good config across the campaign.
4. To run the full campaign unattended, drive it with the wrapper:
   ```bash
   bash scripts/auto_resume_wrapper.sh --work-dir <work_dir> -- claude -p "..." --max-turns 0
   ```
5. The agent writes the ledger incrementally and `final_report.md` when the loop exits.

### Key guidelines

- One lever per iteration — never bundle changes; it destroys attribution.
- The benchmark contract (concurrency levels, token sizes, requests-per-level, target metric) is frozen after Phase 1.
- Append ledger entries immediately after each verdict; never defer writes — they are the crash-recovery hand-off.
- c=1 `ttft_avg` is the authoritative WIN/LOSS metric; c=4/8 alone never justifies a WIN.
- Drive the campaign through the auto-resume wrapper; do not manually relaunch `claude` mid-campaign (it races with the wrapper over the ledger).
- Run the mandatory post-iteration cleanup between every iteration to avoid zombie processes and NPU memory leaks.
