---
name: vllm-ascend-model-adapter
description: "Adapt and debug existing or new models for vLLM on Ascend NPU. Implement in /vllm-workspace/vllm and /vllm-workspace/vllm-ascend, validate via direct vllm serve from /workspace, and deliver one signed commit in the current repo."
---

# vLLM Ascend Model Adapter

## Overview

Adapt Hugging Face or local models to run on `vllm-ascend` with minimal changes, deterministic validation, and single-commit delivery. This skill is for both already-supported models and new architectures not yet registered in vLLM.

## Read order

1. Start with `references/workflow-checklist.md`.
2. Read `references/multimodal-ep-aclgraph-lessons.md` (feature-first checklist).
3. If startup/inference fails, read `references/troubleshooting.md`.
4. If checkpoint is fp8-on-NPU, read `references/fp8-on-npu-lessons.md`.
5. Before handoff, read `references/deliverables.md`.

## Hard constraints

- Never upgrade `transformers`.
- Primary implementation roots are fixed by Dockerfile:
    - `/vllm-workspace/vllm`
    - `/vllm-workspace/vllm-ascend`
- Start `vllm serve` from `/workspace` with direct command by default.
- Default API port is `8000` unless user explicitly asks otherwise.
- Feature-first default: try best to validate ACLGraph / EP / flashcomm1 / multimodal out-of-box. MTP is validated only when the checkpoint explicitly supports it (inferred from config + weight keys in Step 2).
- `--enable-expert-parallel` and flashcomm1 checks are MoE-only; for non-MoE models mark as not-applicable with evidence.
- If any feature cannot be enabled, keep evidence and explain reason in final report.
- Do not rely on `PYTHONPATH=<modified-src>:$PYTHONPATH` unless debugging fallback is strictly needed.
- Keep code changes minimal and focused on the target model.
- **Never introduce modeling files or patches into `vllm-ascend`**. All model adaptation code belongs in `/vllm-workspace/vllm`. If a model cannot function on Ascend without adding modeling code to `vllm-ascend`, stop — raise a GitHub issue to analyze the root cause instead.
- Final deliverable commit must be one single signed commit in the current working repo (`git commit -sm ...`).
- Keep final docs in Chinese and compact.
- **Dummy-first is encouraged for speed, but dummy is NOT fully equivalent to real weights.**
- **Never sign off adaptation using dummy-only evidence; real-weight gate is mandatory.**

## Execution playbook

### 1) Collect context

- **Run NPU environment sanity check first** (see `references/workflow-checklist.md` §0.5). Verify NPU devices are visible, `torch_npu` is importable, NPU tensor creation works, and available NPU count ≥ required TP size. If any check fails, stop and resolve before proceeding.
- Confirm model path (default `/models/<model-name>`; if environment differs, confirm with user explicitly).
- Confirm implementation roots (`/vllm-workspace/vllm`, `/vllm-workspace/vllm-ascend`).
- Confirm delivery root (the current git repo where the final commit is expected).
- Confirm runtime import path points to `/vllm-workspace/*` install.
- Use default expected feature set: ACLGraph + EP + flashcomm1 + multimodal (if model has VL capability). MTP is validated only if the checkpoint explicitly supports it (determined in Step 2).
- User requirements extend this baseline, not replace it.

### 2) Analyze model first

- Inspect `config.json`, processor files, modeling files, tokenizer files.
- **Classify model type**:
    - High-level: LLM / VLM (Vision-Language) / Whisper (ASR).
    - For LLM, identify attention sub-type: standard full attention, sliding window attention, Mamba (SSM), multi-latent attention (MLA), or a hybrid of the above.
- Identify architecture class, attention variant, quantization type, and multimodal requirements.
- Check state-dict key prefixes (and safetensors index) to infer mapping needs.
- Decide whether support already exists in `vllm/model_executor/models/registry.py`.

### 3) Analyze new operators (Ascend compatibility gate)

- Identify any new operators introduced in the model or its modeling code.
- Classify each new operator by type and draw the appropriate conclusion:
    - **Torch** (native PyTorch op): Functional on Ascend ✅; performance is uncertain — note in report.
    - **Triton** kernel: Functional correctness uncertain ⚠️; requires explicit verification on Ascend; accuracy also uncertain.
    - **CUDA** kernel: Not supported on Ascend ❌; check whether a fallback implementation exists.
- **CUDA operator early-exit gate**: If any CUDA operator has no fallback (pure CUDA kernel with no Torch/Triton alternative), **stop here** — skip all subsequent validation steps and directly file a GitHub issue that explains:
    - which operator blocks Ascend support,
    - why no fallback exists,
    - recommended path forward (e.g., implement a custom Ascend op in `vllm-ascend`).
- **Triton operator early-exit gate**: If a Triton kernel is verified to be non-functional on Ascend (correctness failure or unacceptable accuracy degradation), **stop here** — file a GitHub issue that explains:
    - which Triton kernel fails and the observed failure mode,
    - recommended path forward (e.g., replace with a Torch-native fallback or implement a custom Ascend op).
- If every CUDA operator has a fallback and every Triton kernel passes verification, document fallback paths and continue.

### 4) Analyze framework-side code

- Identify vLLM framework modules changed to support the new model (e.g., scheduler, attention backend, sampler, weight loader, worker) — anything beyond the model file and operators.
- For each changed module, check whether `vllm-ascend` already overrides or depends on it:
    - If the module is a **common vLLM module already covered by vllm-ascend**, check whether the existing vllm-ascend patch still applies correctly after the upstream change. If the patch needs updating, update it; otherwise no further action is needed.
    - If the module is **not covered by vllm-ascend** and contains Ascend-incompatible logic, add a minimal corresponding override under `/vllm-workspace/vllm-ascend`.
- Keep framework-side patches minimal and scoped to the incompatible code paths only.

### 5) Choose adaptation strategy (new-model capable)

- Reuse existing vLLM architecture if compatible.
- If architecture is missing or incompatible, implement native support:
    - add model adapter under `vllm/model_executor/models/`;
    - add processor under `vllm/transformers_utils/processors/` when needed;
    - register architecture in `vllm/model_executor/models/registry.py`;
    - implement explicit weight loading/remap rules (including fp8 scale pairing, KV/QK norm sharding, rope variants).
- If remote code needs newer transformers symbols, do not upgrade dependency.
- If unavoidable, copy required modeling files from sibling transformers source and keep scope explicit.
- If failure is backend-specific (kernel/op/platform) and would require adding modeling code to `vllm-ascend`, do not proceed — raise a GitHub issue to analyze the root cause instead.

### 6) Implement minimal code changes (in vLLM source only)

- Do not introduce modeling files or patches in `/vllm-workspace/vllm-ascend`.
- Touch only files required for this model adaptation.
- Keep weight mapping explicit and auditable.
- Avoid unrelated refactors.

### 6.5) Intermediate NPU unit-test gate (before full serve)

Run targeted unit tests on NPU for any new operators (from Step 3) and framework changes (from Step 4) **before** launching the full serve pipeline. This catches NPU-specific failures in seconds rather than minutes.

#### What to test

- **New operators**: for each new Torch/Triton operator introduced by the model, write a minimal standalone test that constructs a representative input tensor and asserts output shape and dtype are correct on NPU.
- **Framework changes**: for each vLLM framework module touched (scheduler, attention backend, sampler, weight loader, worker), write a unit test that exercises the changed code path on NPU with a small synthetic input.

#### Test structure

```python
# example skeleton — adapt per operator/module
import torch
import torch_npu  # ensures NPU backend is initialized

def test_<operator_or_module>():
    # construct minimal representative input on NPU
    x = torch.randn(<shape>, dtype=torch.bfloat16, device="npu")
    # invoke the operator or changed module path
    out = <operator_or_function>(x, ...)
    # assert correctness: shape, dtype, no exception, optionally a numeric check
    assert out.shape == <expected_shape>
    assert out.dtype == torch.bfloat16
    assert not torch.isnan(out).any()

if __name__ == "__main__":
    test_<operator_or_module>()
    print("PASS")
```

Place tests under `/tmp/npu_unit_tests/` (ephemeral; not committed).

#### Retry and early-exit policy

- Run each test. If it **passes**: proceed to Step 7.
- If it **fails**: attempt a fix and re-run. This counts as **attempt 1**.
- If it **fails again**: attempt a second fix and re-run. This counts as **attempt 2**.
- If it **still fails after 2 attempts**: **early exit** — do not proceed to serve validation. File a GitHub issue documenting:
    - which operator or module test failed,
    - the observed failure mode (error message + stack trace),
    - both fix attempts and why they did not resolve the issue,
    - recommended path forward.

### 7) Two-stage validation on Ascend (direct run)

#### Stage A: dummy fast gate (recommended first)

- Run from `/workspace` with `--load-format dummy`.
- Goal: fast validate architecture path / operator path / API path.
- Do not treat `Application startup complete` as pass by itself; request smoke is mandatory.
- Require at least:
    - startup readiness (`/v1/models` 200),
    - one text request 200,
    - if VL model, one text+image request 200,
    - ACLGraph evidence where expected.

#### Stage B: real-weight mandatory gate (must pass before sign-off)

- Remove `--load-format dummy` and validate with real checkpoint.
- Goal: validate real-only risks:
    - weight key mapping,
    - fp8/fp4 dequantization path,
    - KV/QK norm sharding with real tensor shapes,
    - load-time/runtime stability.
- Require HTTP 200 and non-empty output before declaring success.
- Do not pass Stage B on startup-only evidence.

### 8) Validate inference and features

- Send `GET /v1/models` first.
- Send at least one OpenAI-compatible text request.
- For multimodal models, require at least one text+image request.
- Validate architecture registration and loader path with logs (no unresolved architecture, no fatal missing-key errors).
- Try feature-first validation: EP + ACLGraph path first; eager path as fallback/isolation.
- If startup succeeds but first request crashes (false-ready), treat as runtime failure and continue root-cause isolation.
- For `torch._dynamo` + `interpolate` + `NPU contiguous` failures on VL paths, try `TORCHDYNAMO_DISABLE=1` as diagnostic/stability fallback.
- For multimodal processor API mismatch (for example `skip_tensor_conversion` signature mismatch), use text-only isolation (`--limit-mm-per-prompt` set image/video/audio to 0) to separate processor issues from core weight loading issues.
- Capacity baseline by default (single machine): `max-model-len=128k` + `max-num-seqs=16`.
- Then expand concurrency (e.g., 32/64) if requested or feasible.

> **Note**: Accuracy evaluation and performance benchmarking are out of scope for this skill. They are handled by a dedicated separate skill. If requested, invoke that skill after completing this step.

### 9) Backport, generate artifacts, and commit in delivery repo

- If implementation happened in `/vllm-workspace/*`, backport minimal final diff to current working repo.
- Generate test config YAML at `tests/e2e/models/configs/<ModelName>.yaml` following the schema of existing configs (must include `model_name`, `hardware`, `tasks` with accuracy metrics, and `num_fewshot`). Use accuracy results from evaluation to populate metric values.
- Generate tutorial markdown at `docs/source/tutorials/models/<ModelName>.md` following the standard template (Introduction, Supported Features, Environment Preparation with docker tabs, Deployment with serve script, Functional Verification with curl example, Accuracy Evaluation, Performance). Fill in model-specific details: HF path, hardware requirements, TP size, max-model-len, served-model-name, sample curl, and accuracy table.
- Update `docs/source/tutorials/models/index.md` to include the new tutorial.
- Confirm test config YAML and tutorial doc are included in the staged files.
- Commit code changes once (single signed commit).

### 10) Prepare handoff artifacts

- Write comprehensive Chinese analysis report.
- Write compact Chinese runbook for server startup and validation commands.
- Include feature status matrix (supported / unsupported / checkpoint-missing / not-applicable).
- Include dummy-vs-real validation matrix and explicit non-equivalence notes.
- Include changed-file list, key logs, and final commit hash.
- Post the SKILL.md content (or a link to it) as a comment on the originating GitHub issue to document the AI-assisted workflow.

## Quality gate before final answer

- Service starts successfully from `/workspace` with direct command.
- OpenAI-compatible inference request succeeds (not startup-only).
- Key feature set is attempted and reported: ACLGraph / EP / flashcomm1 / multimodal. MTP reported if checkpoint supports it.
- Capacity baseline (`128k + bs16`) result is reported, or explicit reason why not feasible.
- **Dummy stage evidence is present (if used), and real-weight stage evidence is present (mandatory).**
- Test config YAML exists at `tests/e2e/models/configs/<ModelName>.yaml` and follows the established schema (`model_name`, `hardware`, `tasks`, `num_fewshot`).
- Tutorial doc exists at `docs/source/tutorials/models/<ModelName>.md` and follows the standard template (Introduction, Supported Features, Environment Preparation, Deployment, Functional Verification, Accuracy Evaluation, Performance).
- Tutorial index at `docs/source/tutorials/models/index.md` includes the new model entry.
- Exactly one signed commit contains all code changes in current working repo.
- Final response includes commit hash, file paths, key commands, known limits, and failure reasons where applicable.
