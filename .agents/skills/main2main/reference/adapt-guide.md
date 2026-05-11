# Adapt Guide

Use this guide during the adapt phase of each main2main step. The goal is not
to copy upstream vLLM changes into vllm-ascend. The goal is to understand which
upstream contracts changed, then update the Ascend implementation that depends
on those contracts.

This file is only about adaptation decisions and code changes. Mechanical
pipeline work, such as updating the pinned vLLM commit reference, is handled by
`SKILL.md` and `scripts/update_commit_reference.py`.

## Inputs

For each step, use these files:

- `/tmp/main2main/steps/<step-id>/changed-files.txt` — file paths changed by the upstream step
- `/tmp/main2main/steps/<step-id>/upstream.patch` — full upstream diff for the step

Read `changed-files.txt` first. It is a cheap routing signal that tells you
which parts of `upstream.patch` deserve attention.

---

## Step 1: Analyze vLLM Changes

1. Read `upstream.patch` and `changed-files.txt`. Cross-reference against the 'vLLM Key Areas to Focus On' table below to identify which subsystems are touched before reading any actual diff.
2. Find the relevant chunks in `upstream.patch` and identify the concrete change: new/removed abstract methods, changed signatures, renamed config fields, moved imports, changed constructor args, dependency bumps, or changed return types.
3. Use the File Mapping Table below to find likely vllm-ascend locations.

The key question: **does vllm-ascend subclass, override, call, import, or read anything this patch changed?**, Internal implementation changes only need adaptation when vllm-ascend directly depends on the behavior.

## Step 2: Adapt vLLM Ascend Project
For each related change in vLLM , evaluate whether adaptation in vLLM Ascend is needed:

- **Internal Architecture Changes**
  Check internal interfaces of vLLM core modules (scheduler, executor, model runner, etc.)
  Update vLLM Ascend's Ascend-specific implementations (e.g., NPU worker/model runner, custom attention、custom ops)
  Preserve vLLM Ascend specific modifications (e.g., code under vllm_ascend/)

- **Dependency Changes**
  Check for dependency version changes in pyproject.toml or setup.py
  Update dependency declarations in vLLM Ascend

- **Version Compatibility**
   Use `vllm_version_is()` guards when the change must coexist with the release version (see Version Compatibility Rules in SKILL.md)

When a feature genuinely can't be supported on Ascend yet, add a stub with a `# TODO` comment referencing the issue.

A no-op adapt (nothing to change) is fine, but it does not skip CI — the updated commit reference still needs verification.

---

### vLLM Key Areas to Focus On

When analyzing vLLM changes, pay special attention to these areas that typically require vLLM Ascend adaptation:

1. **Platform Interface** (`vllm/platforms/`)
   - New abstract methods — implement immediately; missing ones cause `TypeError: Can't instantiate abstract class AscendPlatform` at runtime, not at import time, so they won't surface until a test actually executes
   - Method signature changes
   - New platform capability flags

2. **Worker / Model Runner** (`vllm/v1/worker/`, `vllm/v1/worker/gpu/model_runner.py`)
   - New or removed parameters in `execute_model` or `load_model` — vllm-ascend heavily overrides these; signature mismatches cause `TypeError` during inference
   - New lifecycle methods
   - Changes to model runner initialization

3. **Attention** (`vllm/model_executor/layers/attention/`, `vllm/v1/attention/`)
   - New parameters in `forward()` — vllm-ascend registers its own backend; interface changes require updating both registration and implementation
   - Changes to attention backend interface
   - MLA-specific updates

4. **MoE** (`vllm/model_executor/layers/fused_moe/`)
   - FusedMoE layer signature changes — vllm-ascend has Ascend-specific MoE kernels that call into this interface
   - Router interface changes
   - Activation function changes

5. **Config** (`vllm/config*.py`)
   - Field renames or moves between config classes — vllm-ascend reads config fields directly in many places; a rename causes `AttributeError` everywhere it's accessed
   - New required fields
   - Constructor changes

6. **Distributed** (`vllm/distributed/`)
   - Changes to collective op interfaces
   - KV transfer protocol changes
   - Device communicator updates

7. **Speculative Decoding** (`vllm/v1/worker/gpu/spec_decode/`, `vllm/config/speculative.py`)
   - Import path changes
   - Config field changes
   - New proposer interface methods — vllm-ascend has MTP and Eagle proposer implementations

8. **Compilation** (`vllm/compilation/`)
   - Pass manager interface changes
   - New required passes
   - Changes to how passes register

9. **Quantization** (`vllm/model_executor/layers/quantization/`)
   - Quantization config changes
   - compress-tensor method changes

10. **Models** (`vllm/model_executor/models/`)
    - Changes to model forward signatures — when vllm-ascend overrides a model's forward method, signature changes break inference
    - New model architectures

---

## vllm-ascend Key File Locations

| Project | Path |
|---------|------|
| vLLM Ascend version compatibility | `vllm-ascend/docs/source/conf.py` |
| vLLM Ascend source code | `vllm_ascend/` |
| **Core Modules** | |
| Ascend-specific attention | `vllm_ascend/attention/` |
| Ascend-specific executor | `vllm_ascend/worker/` |
| Ascend-specific ops | `vllm_ascend/ops/` |
| **Specialized Implementations** | |
| Ascend 310P specific | `vllm_ascend/_310p/` |
| EPLB load balancing | `vllm_ascend/eplb/` |
| XLite compiler | `vllm_ascend/xlite/` |
| **Compilation & Fusion** | |
| Graph fusion pass manager | `vllm_ascend/compilation/` |
| Compilation passes | `vllm_ascend/compilation/passes/` |
| **Quantization** | |
| Quantization methods | `vllm_ascend/quantization/` |
| ModelSlim integration | `vllm_ascend/quantization/methods/modelslim/` |
| **Distributed & KV Cache** | |
| KV transfer | `vllm_ascend/distributed/kv_transfer/` |
| Device communicators | `vllm_ascend/distributed/device_communicators/` |
| **Speculative Decoding** | |
| MTP proposer | `vllm_ascend/spec_decode/mtp_proposer.py` |
| Eagle proposer | `vllm_ascend/spec_decode/eagle_proposer.py` |
| **Utility Modules** | |
| Common utilities | `vllm_ascend/utils.py` |
| Ascend config | `vllm_ascend/ascend_config.py` |
| Common utilities | `vllm_ascend/utils.py` |
| Environment variables | `vllm_ascend/envs.py` |
| Attention | `vllm_ascend/attention/` |
| Worker | `vllm_ascend/worker/` |
| Ops | `vllm_ascend/ops/` |
| Distributed | `vllm_ascend/distributed/` |
| Compilation | `vllm_ascend/compilation/` |
| Quantization | `vllm_ascend/quantization/` |
| Speculative decoding | `vllm_ascend/spec_decode/` |
| 310P specific | `vllm_ascend/_310p/` |
| EPLB | `vllm_ascend/eplb/` |
| XLite | `vllm_ascend/xlite/` |

---

## File Mapping Table

Use this table after identifying a changed upstream symbol. It points to likely vllm-ascend locations, not guaranteed locations.

| vLLM upstream path | vllm-ascend path | What to check |
|:---|:---|:---|
| `vllm/platforms/` | `vllm_ascend/platform.py` | Abstract methods, platform capabilities |
| `vllm/v1/worker/` | `vllm_ascend/worker/` | Worker lifecycle, model loading, `execute_model` |
| `vllm/v1/worker/gpu/model_runner.py` | `vllm_ascend/worker/model_runner_v1.py`, `vllm_ascend/worker/v2/model_runner.py` | Runner initialization and execution |
| `vllm/v1/attention/` | `vllm_ascend/attention/` | Backend interface and metadata |
| `vllm/model_executor/layers/attention/` | `vllm_ascend/attention/`, `vllm_ascend/ops/mm_encoder_attention.py` | Attention wrappers and kernels |
| `vllm/model_executor/layers/fused_moe/` | `vllm_ascend/ops/fused_moe/` | MoE kernel interface, router, experts |
| `vllm/distributed/` | `vllm_ascend/distributed/` | Collective ops, TP/PP, KV transfer |
| `vllm/config*.py` | `vllm_ascend/ascend_config.py`, plus call sites under `vllm_ascend/` | Config fields and constructor args |
| `vllm/compilation/` | `vllm_ascend/compilation/` | Passes, fusion rules, registration |
| `vllm/model_executor/models/` | `vllm_ascend/models/` | Model forward signatures and loaders |
| `vllm/model_executor/layers/quantization/` | `vllm_ascend/quantization/` | Quantization methods and kernels |
| `vllm/model_executor/layers/layernorm.py` | `vllm_ascend/ops/layernorm.py` | LayerNorm op interface |
| `vllm/model_executor/custom_op.py` | `vllm_ascend/ops/` | Custom op registration |
| `vllm/v1/worker/gpu/spec_decode/` | `vllm_ascend/spec_decode/` | MTP/Eagle proposer interfaces |
| `requirements*`, `constraints*`, `pyproject.toml`, `setup.py`, `setup.cfg` | Matching dependency files in vllm-ascend | Dependency versions |
