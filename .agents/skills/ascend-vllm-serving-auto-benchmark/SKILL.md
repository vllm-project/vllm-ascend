---
name: ascend-vllm-serving-auto-benchmark
description: "Benchmark vLLM-Ascend serving on Ascend 910C, validate YAML configs, launch the server, run evalscope perf, and generate Markdown/CSV reports."
disable-model-invocation: true
---

# ascend-vllm-serving-auto-benchmark

## What this skill is for

Use this skill when the user wants a reproducible serving benchmark for `vllm-ascend` on Ascend 910C:

- LLM serving performance benchmark / pressure test
- compare TP / PP / model config variants
- generate a report that can be attached to a PR, issue, or tuning note

Trigger phrases: `benchmark`, `pressure test`, `性能测试`, `压测`, `ascend benchmark`, `vllm-ascend serving 性能`

## Read order

1. Read this file first.
2. If the user provides a YAML config, validate it with `scripts/validate_configs.py`.
3. Run `scripts/run_benchmark.sh`.
4. If report parsing looks wrong, inspect `scripts/generate_report.py`.

## Required inputs

| Parameter | Required | Notes |
|-----------|----------|-------|
| `model_path` | Yes, unless `config_file` provides it | Local path or model ID |
| `model_name` | Yes, unless `config_file` provides it | Used in serving API and reports |
| `config_file` | No | One of `configs/*.yaml` or a compatible YAML file |

Common optional inputs:

- `tensor_parallel_size`
- `pipeline_parallel_size`
- `max_model_len`
- `dtype`
- `quantization`
- `port`
- `parallel_levels`
- `requests_per_level`
- `input_tokens`
- `output_tokens`
- `output_dir`

## Preconditions

Before running:

1. Confirm the target machine is really an Ascend benchmark environment.
2. Check `vllm-ascend`, `torch-npu`, and `evalscope[perf]` are installed.
3. Confirm the model path exists or the model ID is accessible.
4. Confirm the chosen port is free before server start.

## Workflow

The implementation lives in `scripts/run_benchmark.sh` and should be used as the default execution path.

### Phase 1: validate inputs and environment

- Run `npu-smi info`
- Validate YAML via `scripts/validate_configs.py` when `config_file` is provided
- Detect `vllm-ascend` and `evalscope`

### Phase 2: launch vLLM-Ascend

- Start exactly one serving process for this benchmark
- Apply every `server.env` entry from YAML configs, not only a hand-picked subset
- Preserve explicit CLI overrides such as `--npu-devices`, `--no-aclgraph`, and `--no-nz`

### Phase 3: warm up

- Default warm-up: 20 requests at `c=1`
- Warm-up is a normal benchmark precondition and should happen before the measured run unless the user explicitly skips it

### Phase 4: main evalscope benchmark

- Keep concurrency levels and request counts fixed within a run
- Prefer the current evalscope output directory flag when available
- If the installed evalscope does not support an output flag, run it from the intended artifact directory instead of writing results into an ambiguous cwd

### Phase 5: report generation

Generate:

- `benchmark_report.md`
- `benchmark_results.csv`

`scripts/generate_report.py` must handle both JSONL and JSON evalscope outputs.

## Config rules

YAML configs under `configs/` are the source of model-specific defaults.

- `server.env` is authoritative for environment variables
- `model.*`, `server.*`, `workload.*`, `benchmark.*`, and `sla.*` must stay consistent with `scripts/validate_configs.py`
- CLI arguments may override config values for the current run

## Failure handling

- If server startup fails, surface the failure clearly and keep `server.log`
- Do not silently skip report generation because a JSON file used a different shape than JSONL
- Do not mix results from different NPU types in one report
- Always record the exact server command and evalscope command used for the run

## Output contract

The final output directory should contain at least:

```text
<output_dir>/
├── benchmark_report.md
├── benchmark_results.csv
├── evalscope_results/
├── npu_info.txt
├── server.log
├── server_cmd.txt
└── evalscope_cmd.txt
```

## Quick start

This skill starts a serving process, occupies NPU resources, and may run a long
benchmark campaign, so it must be invoked explicitly. Use the form for your
agent and never let the model invoke it implicitly (`disable-model-invocation`
is already set in the frontmatter above).

- Codex: `$ascend-vllm-serving-auto-benchmark`
- Claude Code: `/ascend-vllm-serving-auto-benchmark`

```text
/ascend-vllm-serving-auto-benchmark
config_file=.agents/skills/ascend-vllm-serving-auto-benchmark/configs/qwen3-32b.yaml
```

```text
/ascend-vllm-serving-auto-benchmark
model_path=/data/models/Qwen3.5-27B
model_name=Qwen3.5-27B
tensor_parallel_size=2
dtype=bfloat16
parallel_levels="1 4 8 16 32 64"
requests_per_level="10 40 80 160 320 640"
```
