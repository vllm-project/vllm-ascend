# Adapt Guide

You arrive here with two files:
- `/tmp/main2main/steps/<step-id>/upstream.patch` — the full diff for this step
- `/tmp/main2main/steps/<step-id>/changed-files.txt` — list of changed file paths

The goal is to figure out which upstream changes require a response in vllm-ascend, and make those changes. Here's the process.

---

## Step 1: Triage with changed-files.txt

Read `changed-files.txt` first — it's fast. Cross-reference against the Key Areas table below. This tells you which subsystems are touched before you read any actual diff.

Files outside the Key Areas (tests, docs, benchmarks, `.github/`) almost never require adaptation unless their changes hint at an interface shift — e.g., a new test exercising an API that vllm-ascend overrides.

Flag the subsystems that need attention. You'll go through them one by one in Step 2.

---

## Step 2: Analyze each affected subsystem

For each flagged subsystem:

1. **Find the relevant diff chunks in upstream.patch**
   ```bash
   grep -n "^diff --git" /tmp/main2main/steps/<step-id>/upstream.patch
   # Then read the specific section for the subsystem file
   ```

2. **Check what specifically changed** using the Key Areas section below — each subsystem has a list of what to look for. These are the changes that historically break vllm-ascend.

3. **Find the corresponding vllm-ascend file** using the File Mapping Table below, then grep for the affected symbol:
   ```bash
   grep -rn '<function_or_class_name>' vllm_ascend/
   ```

4. **Decide if adaptation is needed.** The rule from SKILL.md applies: abstract methods, function signatures, config field locations always need follow-up. Internal implementation of methods vllm-ascend doesn't override can usually be skipped.

---

## Step 3: Apply changes

For each change that needs adaptation:
- Update the corresponding vllm-ascend file
- Use `vllm_version_is()` guards when the change must coexist with the release version (see Version Compatibility Rules in SKILL.md)
- If you're unsure of the right pattern, check how other version guards are structured: `grep -rn 'vllm_version_is' vllm_ascend/`

If a needed feature genuinely can't be supported on Ascend yet, add a stub with a `# TODO` comment that references the issue.

---

## Step 4: Update commit reference

After all changes for this step are done:

```bash
grep -Frl "<OLD_COMMIT>" . | xargs sed -i "s/<OLD_COMMIT>/<NEW_COMMIT>/g"
grep -Frn "<OLD_COMMIT>" .   # verify nothing remains
```
---

## Key Areas

When you see changes in these paths in `changed-files.txt`, here's what to specifically look for in the diff.

**Platform Interface** (`vllm/platforms/`)
Look for: new abstract methods, method signature changes, new platform capability flags.
Why it matters: `AscendPlatform` inherits from vLLM's platform base class. Any new abstract method will cause `TypeError: Can't instantiate abstract class AscendPlatform` at runtime — not at import time, so it won't surface until a test actually runs.

**Worker / Model Runner** (`vllm/v1/worker/`, `vllm/v1/worker/gpu/model_runner.py`)
Look for: new or removed method parameters, changes to `execute_model` or `load_model` signatures, new lifecycle methods.
Why it matters: vllm-ascend has heavily overridden model runner implementations. Signature mismatches here cause `TypeError` at the point the method is called during inference.

**Attention** (`vllm/model_executor/layers/attention/`, `vllm/v1/attention/`)
Look for: new parameters in `forward()`, changes to attention backend interface, MLA-specific changes.
Why it matters: vllm-ascend registers its own attention backend. Interface changes here require updating the registration and the implementation.

**MoE** (`vllm/model_executor/layers/fused_moe/`)
Look for: FusedMoE layer signature changes, router interface changes, activation function changes.
Why it matters: vllm-ascend has Ascend-specific MoE kernel implementations that call into vLLM's MoE layer interface.

**Config** (`vllm/config*.py`)
Look for: field renames, field moves between config classes, new required fields, constructor changes.
Why it matters: vllm-ascend reads config fields directly in many places. A renamed field causes `AttributeError` everywhere it's accessed.

**Distributed** (`vllm/distributed/`)
Look for: changes to collective op interfaces, KV transfer protocol changes, device communicator updates.
Why it matters: vllm-ascend has Ascend-specific distributed implementations that must match vLLM's distributed interface.

**Speculative Decoding** (`vllm/v1/worker/gpu/spec_decode/`, `vllm/config/speculative.py`)
Look for: import path changes, config field changes, new proposer interface methods.
Why it matters: vllm-ascend has MTP and Eagle proposer implementations that depend on these interfaces.

**Compilation** (`vllm/compilation/`)
Look for: pass manager interface changes, new required passes, changes to how passes register.
Why it matters: vllm-ascend has custom compilation passes.

**Quantization** (`vllm/model_executor/layers/quantization/`)
Look for: quantization config changes, compress-tensor method changes, new quantization methods.
Why it matters: vllm-ascend has ModelSlim and other quantization integrations.

**Models** (`vllm/model_executor/models/`)
Look for: new model architectures added, changes to model forward signatures.
Why it matters: if vllm-ascend overrides a model's forward method, signature changes break inference.

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
| Platform detection | `vllm_ascend/platform.py` |
| Environment variables | `vllm_ascend/envs.py` |
