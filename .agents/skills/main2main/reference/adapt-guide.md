# Adapt Guide

This document describes how to analyze an upstream vLLM diff and translate it
into adaptation changes in vllm-ascend. Read this at the start of each step's
adapt phase.

## Inputs Available

Available inputs for this step:
- `/tmp/main2main/steps/<step-id>/upstream.patch` — full diff for this step
- `/tmp/main2main/steps/<step-id>/changed-files.txt` — list of changed file paths

---

## Step 1: Analyze vLLM Changes

Read `upstream.patch` and `changed-files.txt`. Cross-reference against the Key Areas table below to identify which subsystems are touched before reading any actual diff.


---

## Step 2: vLLM Key Areas to Focus On

When analyzing vLLM changes (`upstream.patch`,`changed-files.txt`), pay special attention to these areas that typically require vLLM Ascend adaptation:

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

## Step 3: Adapt vLLM Ascend Project
For each related change in vLLM from the file vllm_changes.md, evaluate whether adaptation in vLLM Ascend is needed:

- **Internal Architecture Changes**
  Check internal interfaces of vLLM core modules (scheduler, executor, model runner, etc.)
  Update vLLM Ascend's Ascend-specific implementations (e.g., NPU worker/model runner, custom attention、custom ops)
  Preserve vLLM Ascend specific modifications (e.g., code under vllm_ascend/)

- **Dependency Changes**
  Check for dependency version changes in pyproject.toml or setup.py
  Update dependency declarations in vLLM Ascend

- **Version Compatibility**
   Use `vllm_version_is()` guards when the change must coexist with the release version (see Version Compatibility Rules in SKILL.md)
   When the right pattern is unclear, check how other version guards are structured: `grep -rn 'vllm_version_is' vllm_ascend/`

When a feature genuinely can't be supported on Ascend yet, add a stub with a `# TODO` comment referencing the issue.

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

| vLLM upstream path | vllm-ascend path | Notes |
|:---|:---|:---|
| `vllm/platforms/` | `vllm_ascend/platform.py` | Abstract methods, platform capabilities |
| `vllm/v1/worker/` | `vllm_ascend/worker/` | Worker lifecycle, model loading, execute_model |
| `vllm/v1/worker/gpu/model_runner.py` | `vllm_ascend/worker/model_runner_v1.py`, `worker/v2/model_runner.py` | Heavily overridden |
| `vllm/v1/attention/` | `vllm_ascend/attention/` | Attention backend interface |
| `vllm/model_executor/layers/attention/` | `vllm_ascend/attention/`, `vllm_ascend/ops/mm_encoder_attention.py` | |
| `vllm/model_executor/layers/fused_moe/` | `vllm_ascend/ops/fused_moe/` | MoE kernel interface, router |
| `vllm/distributed/` | `vllm_ascend/distributed/` | Collective ops, TP/PP, KV transfer |
| `vllm/config*.py` | `vllm_ascend/ascend_config.py` + many files that read config | Config class fields, constructor args |
| `vllm/compilation/` | `vllm_ascend/compilation/` | Compilation passes, fusion rules |
| `vllm/model_executor/models/` | `vllm_ascend/models/` | Model forward signatures |
| `vllm/model_executor/layers/quantization/` | `vllm_ascend/quantization/` | Quantization kernels, compress-tensor |
| `vllm/model_executor/layers/layernorm.py` | `vllm_ascend/ops/layernorm.py` | |
| `vllm/model_executor/custom_op.py` | `vllm_ascend/ops/` (files registering custom ops) | |
| `vllm/v1/worker/gpu/spec_decode/` | `vllm_ascend/spec_decode/` | MTP/Eagle proposer |
| `requirements*.txt` / `pyproject.toml` | `requirements*.txt` / `pyproject.toml` | Dependency versions |

