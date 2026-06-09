---
name: vllm-ascend-model-adapter
description: "Adapt and debug existing or new models for vLLM on Ascend NPU. Implement in /vllm-workspace/vllm and /vllm-workspace/vllm-ascend, validate via direct vllm serve from /workspace, and deliver one signed commit in the current repo."
---

# vLLM Ascend Model Adapter

## Overview

Adapt Hugging Face or local models to run on `vllm-ascend` with minimal changes, deterministic validation, and single-commit delivery. This skill is for both already-supported models and new architectures not yet registered in vLLM.

## Read order

1. Start with `references/workflow-checklist.md`.
2. Read `references/model-layer-baseline.md` after model classification and build a layer-by-layer compatibility matrix first.
3. Read `references/model-adapter-and-weight-loading-baseline.md` after the layer matrix.
4. If the model is multimodal or processor behavior is suspicious, read `references/processor-and-multimodal-baseline.md`.
5. If the model is a routed-expert MoE LLM, read `references/moe-fused-analysis.md` to establish the current `vllm-ascend` MoE capability baseline before deciding what must be adapted in router, experts, shared experts, or EP paths.
6. If the model uses non-standard attention or attention-related adaptation is likely, read `references/attention-v1-analysis.md` to establish the current `vllm-ascend` attention capability baseline before deciding what must be adapted.
7. Read `references/operator-compatibility-baseline.md` before operator gating or any Ascend-op retry.
8. Read `references/framework-integration-baseline.md` when the likely failure is beyond model files and may involve scheduler/worker/sampler/backend integration.
9. If checkpoint or runtime path is quantized, read `references/quantization-baseline.md`.
10. Read `references/multimodal-ep-aclgraph-lessons.md` (feature-first checklist).
11. If startup/inference fails, read `references/troubleshooting.md`.
12. If checkpoint is fp8-on-NPU, read `references/fp8-on-npu-lessons.md`.
13. Before handoff, read `references/deliverables.md`.

## Scripts

Reusable shell/Python scripts live in `.agents/skills/vllm-ascend-model-adapter/scripts/`. Make them executable once:

```bash
chmod +x .agents/skills/vllm-ascend-model-adapter/scripts/*.sh
```

| Script | Purpose |
|---|---|
| `check_npu_env.sh <TP_SIZE> [CANN_PATH] [ATB_PATH]` | NPU sanity check (CANN, ATB, torch_npu, device count) |
| `check_roots.sh <VLLM_SRC> <VLLM_ASCEND_SRC> [WORK_DIR]` | Verify source roots and runtime import |
| `triage_model.sh <MODEL_PATH>` | Model directory listing + config.json field scan |
| `classify_model.py <MODEL_PATH>` | Classify model type from config.json |
| `session_reset.sh [PORT]` | Kill stale vllm processes and verify port is free |
| `smoke_test.sh <SERVED_NAME> [PORT] [--multimodal]` | Readiness poll + OpenAI-compatible smoke requests |

## Hard constraints

- Never upgrade `transformers`.
- Primary implementation roots are user-specified at entry (see Entry Points above). Defaults are `/vllm-workspace/vllm` and `/vllm-workspace/vllm-ascend` if the user does not specify otherwise.
- Start `vllm serve` from `$WORK_DIR` (user-specified, default `/workspace`) with direct command. Create the directory if it does not exist.
- Default API port is `8000` unless user explicitly asks otherwise.
- Feature-first default: try best to validate ACLGraph / EP / flashcomm1 / multimodal out-of-box. MTP is validated only when the checkpoint explicitly supports it (inferred from config + weight keys in Step 2).
- `--enable-expert-parallel` and flashcomm1 checks are MoE-only; for non-MoE models mark as not-applicable with evidence.
- If any feature cannot be enabled, keep evidence and explain reason in final report.
- If logs and memory telemetry confirm the blocking issue is HBM capacity on NPU, stop exploring `cpu-offload` as a remediation path. Report the model as hardware-blocked and explicitly tell the user to add cards or move to a larger-HBM Ascend machine.
- Do not rely on `PYTHONPATH=<modified-src>:$PYTHONPATH` unless debugging fallback is strictly needed.
- Keep code changes minimal and focused on the target model.
- **Never introduce modeling files or patches into `vllm-ascend`**. All model adaptation code belongs in `/vllm-workspace/vllm`. If a model cannot function on Ascend without adding modeling code to `vllm-ascend`, stop — raise a GitHub issue to analyze the root cause instead.
- Final deliverable commit must be one single signed commit in the current working repo (`git commit -sm ...`).
- Keep final docs in Chinese and compact.
- **Accuracy baseline rule**: when the skill needs to declare, compare, or document model accuracy, use the evaluation score published on ModelScope for the corresponding model with the same parameter scale as the primary reference baseline. Prefer `gsm8k` first; if ModelScope does not provide `gsm8k`, use another available public evaluation dataset score and state explicitly which dataset was used as the fallback baseline.
- **Dummy-first is encouraged for speed, but dummy is NOT fully equivalent to real weights.**
- **Never sign off adaptation using dummy-only evidence; real-weight gate is mandatory.**

## Entry points

When this skill is invoked, **use the `AskUserQuestion` tool to interactively collect the required inputs before doing anything else**. Ask all inputs in a single call so the user sees one consolidated form. Do not proceed until all values are confirmed.

Call `AskUserQuestion` in **two rounds** (tool limit: max 4 questions per call, min 2 options each):

**Round 1** (4 questions):
1. **Model source** — options: `Already on disk (specify path)`, `Download from ModelScope`, `Download from HuggingFace`, `Other (specify)`
2. **Model checkpoint path or model ID** — if already on disk: local path (e.g. `/models/gemma-4-E4B-it`); if downloading: model ID (e.g. `google/gemma-4-E4B-it`). Options: provide path/ID / `Other (specify)`
3. **Served model name** — options: `Same as checkpoint basename` / `Other (specify)`
4. **Tensor parallel size** — options: `8 (A2)`, `16 (A3)`, `Other (specify)`

**Round 2** (4 questions):
5. **Python environment activation command** — options: `conda activate vllm-ascend`, `source <venv>/bin/activate`, `Other (specify)`, `None (already active)`
6. **vLLM source root** — options: `/vllm-workspace/vllm` / `Other (specify)`
7. **vllm-ascend source root** — options: `/vllm-workspace/vllm-ascend` / `Other (specify)`
8. **Working directory for vllm serve** — options: `/workspace` (default, will be created if missing) / `Other (specify)`

**Round 3 (conditional)** — only triggered during the NPU sanity check (§0.5), not upfront:
- **CANN toolkit `set_env.sh` path** — ask via `AskUserQuestion` only if not auto-detected under `/usr/local/Ascend`. Options: `/usr/local/Ascend/ascend-toolkit/set_env.sh` / `Other (specify)`.
- **Additional env file (ATB/NNAL)** — after sourcing CANN, ask: "Is there an additional env file to source (e.g. ATB/NNAL `set_env.sh`)?" Options: `Yes (specify path in notes)` / `No`. Source if provided.

After the user submits, echo the resolved values back in a confirmation block before starting the execution playbook:

```
  Model source           : <on-disk | modelscope | huggingface>
  Model checkpoint path  : <value>
  Served model name      : <value>
  Tensor parallel size   : <value>
  vLLM source root       : <value>
  vllm-ascend source root: <value>
  Python env activate    : <value>
  Working directory      : <value>
```

Then immediately run the activation command (if not `None`) and verify it succeeded before proceeding:

```bash
# activate the user-specified environment
<activation command>

# verify correct python is active
which python
python -c "import sys; print(sys.executable, sys.version)"
```

If activation fails, stop and report the error before continuing.

---

## Execution playbook

### 0.7) Download model (if not already on disk)

Skip this step if model source is "already on disk" and `$MODEL_PATH` exists.

**ModelScope download:**

```bash
# Resolve download directory from $MODELSCOPE_CACHE (or default ~/.cache/modelscope/hub)
MS_CACHE="${MODELSCOPE_CACHE:-$HOME/.cache/modelscope/hub}"
echo "ModelScope cache: $MS_CACHE"

# Download (blocking — wait for completion before proceeding)
modelscope download --model "$MODEL_ID" --local_dir "$MS_CACHE/$(echo $MODEL_ID | tr '/' '___')"
# After download, set MODEL_PATH to the downloaded directory
MODEL_PATH="$MS_CACHE/$(echo $MODEL_ID | tr '/' '___')"
```

**HuggingFace download:**

```bash
huggingface-cli download "$MODEL_ID" --local-dir "$MODEL_PATH"
```

After download, verify the model path exists and contains `config.json` before proceeding:

```bash
if [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "ERROR: $MODEL_PATH/config.json not found — download may have failed or path is wrong"
  exit 1
fi
echo "OK: model path verified at $MODEL_PATH"
```

**Hard gate**: Do not proceed to Step 1 until `$MODEL_PATH/config.json` exists.

### 1) Collect context

- **Run NPU environment sanity check first** using the provided script:
  ```bash
  bash scripts/check_npu_env.sh "$TP_SIZE"
  # If ATB/NNAL set_env.sh is at a non-default path, pass it as the second argument.
  # Default path tried: /usr/local/Ascend/nnal/atb/set_env.sh
  ```
  Verifies: CANN sourced, ATB/NNAL sourced, NPU devices visible, `torch_npu` importable, NPU tensor creation works, available NPU count ≥ TP size. If any check fails, stop and resolve before proceeding. See `references/workflow-checklist.md` §0.5 for details.
- **Confirm implementation roots** using the provided script:
  ```bash
  bash scripts/check_roots.sh "$VLLM_SRC" "$VLLM_ASCEND_SRC" "$WORK_DIR"
  ```
- **Ensure WORK_DIR exists** (create if missing):
  ```bash
  mkdir -p "$WORK_DIR"
  ```
- **Gate on model path**: verify `$MODEL_PATH/config.json` exists before running triage. If missing, stop and re-run Step 0.7.
- Model path, served model name, TP size, and implementation roots are already confirmed via the Entry Points above — use those values throughout.

### 2) Analyze model first

- **Run fast triage** to collect model inventory and key config fields:
  ```bash
  bash scripts/triage_model.sh "$MODEL_PATH"
  ```
- **Classify model type** using the provided script:
  ```bash
  python scripts/classify_model.py "$MODEL_PATH"
  ```
  The script reads `config.json` and outputs: high-level type (LLM / VLM / Whisper), attention sub-type (standard / sliding-window / MLA / Mamba / hybrid), MoE status, MTP status, quantization type, and key numeric parameters. Use the `CLASSIFICATION_SUMMARY:` JSON line for downstream decisions.
- Inspect processor files, modeling files, tokenizer files as needed.
- Check state-dict key prefixes (and safetensors index) to infer mapping needs.
- Decide whether support already exists in `vllm/model_executor/models/registry.py`.
- If the model is MoE, explicitly plan an EP-first validation path and only fall back to TP-only evidence when EP fails with a documented root cause.

### 2.1) Build the model layer compatibility matrix first

This step is mandatory for every model adaptation.

Read:

- `references/model-layer-baseline.md`

Treat “layer” here as **model components** such as embedding, rope/position, attention, MLP, MoE, norm, lm_head, and multimodal projector/encoder if present.

Current requirement: only two layer templates are standardized now:

- `dense llm`
- `moe llm`

Choose the template based on `CLASSIFICATION_SUMMARY`.

Required output before any specialized gap analysis:

```markdown
## Layer-by-Layer Compatibility Matrix

| Layer | Current capability | Model requirement | Gap | Adaptation plan |
| --- | --- | --- | --- | --- |
| ...use the dense-llm or moe-llm template from `references/model-layer-baseline.md`... |
```

Rules:

- Use the `dense llm` template for non-MoE decoder-only dense LLMs.
- Use the `moe llm` template for routed-expert decoder-only LLMs.
- Fill every row in the selected template.
- `Current capability` must point to an existing repo path, known-good implementation, or existing backend assumption.
- `Model requirement` must come from `config.json`, modeling code, checkpoint structure, or runtime evidence.
- `Gap` must be concrete.
- `Adaptation plan` must name the intended fix location or say the row is already covered and only needs validation.

Do not skip from model classification directly to attention or operators. Use this matrix to decide which layer-specific analyses are actually needed.

### 2.2) Analyze model adapter and weight loading first

Run this step for every model adaptation, even if the eventual failure looks backend-related.

Read:

- `references/model-adapter-and-weight-loading-baseline.md`

Required output before changing code:

```markdown
## Model Adapter Gap Analysis

### 1. Current Capability
- Existing registered architecture:
- Reusable adapter path:
- Existing weight loading assumptions:
- Existing shard/remap support:

### 2. Model Requirement
- `architectures` / `model_type`:
- Adapter structure needed:
- Checkpoint key patterns:
- TP / KV / norm / rope / scale loading needs:

### 3. Gap
- Registration gap:
- Adapter gap:
- Weight mapping gap:
- Loader / shard gap:

### 4. Adaptation Plan
- Fix location:
- Minimal files to touch:
- Validation focus:
- Stop / escalate condition:
```

Do not treat an operator/backend failure as primary until this section makes it clear that model registration and weight loading are already aligned.

### 2.3) Analyze processor and multimodal path

Run this step when the model is VLM / Whisper-like, has processor files, or text-only and multimodal behavior diverge.

Read:

- `references/processor-and-multimodal-baseline.md`

Required output when applicable:

```markdown
## Processor And Multimodal Gap Analysis

### 1. Current Capability
- Existing processor path:
- Existing multimodal support assumption:
- Known-good request types:
- Existing transformers compatibility assumptions:

### 2. Model Requirement
- Processor classes from config:
- Remote/local processing behavior:
- Modalities required:
- MM encoder / embedding path requirements:

### 3. Gap
- Processor API mismatch:
- Multimodal dispatch mismatch:
- Input formatting mismatch:
- Unknowns to verify:

### 4. Adaptation Plan
- Fix location:
- Minimal files to touch:
- Validation focus:
- Stop / escalate condition:
```

### 2.4) Analyze MoE adaptation path first

Run this step for every routed-expert decoder-only LLM, and also when the failure symptoms point to router / expert / EP behavior even if model classification is still ambiguous.

Read:

- `references/moe-fused-analysis.md`

Treat this reference as the **current MoE capability baseline** of `vllm-ascend`, not just background reading. The purpose of this step is to compare:

- what the current Ascend MoE backend already supports,
- what the new model's MoE layer actually requires,
- and where the mismatch falls: router, expert structure, shared expert, weight layout, runtime contract, or communication path.

Build that comparison from two sources:

1. **Existing capability baseline** from `references/moe-fused-analysis.md`
2. **New model MoE features** inferred from:
   - `config.json`,
   - modeling code / remote code,
   - checkpoint key patterns,
   - observed runtime failure stage if the model already boots partially

Use the comparison to answer these questions before changing code:

1. Does the model's router map to current `select_experts(...)` capability: top-k, grouped top-k, `softmax`/`sigmoid`/`sqrtsoftplus`, correction bias, hash routing, no custom routing function?
2. Does the model's expert MLP still map to the current `w13/gate_up -> swiglu -> w2/down` contract?
3. Does the model use shared experts or residual/shared-MLP behavior that current `AscendFusedMoE` can already express?
4. Should bring-up first target `ALLGATHER`, or does the model materially depend on `ALLTOALL` / `MC2` / `FUSED_MC2` behavior?
5. Are the checkpoint layout and quant metadata already mappable to current `MoEWeights` / `MoERoutingParams` / `MoEQuantParams` contracts?

Produce an explicit **MoE gap analysis** in your working notes before proceeding to Step 3.

This is a required output, not an optional note. For every `moe llm` adaptation, write the following fixed section before changing code:

```markdown
## MoE Gap Analysis

### 1. Current Capability
- Router capability baseline:
- Expert MLP baseline:
- Shared expert baseline:
- Communication baseline:
- Quantization baseline:

### 2. Model Requirement
- Router/gate behavior:
- Expert structure:
- Shared expert / residual MLP behavior:
- EP/TP/dispatch expectations:
- Quant / weight-layout requirements:

### 3. Gap
- Router gap:
- Expert-structure gap:
- Weight-layout gap:
- Communication/runtime-contract gap:
- Unknowns to verify:

### 4. Adaptation Plan
- Fix location:
- Minimal files to touch:
- First validation path:
- Stop / escalate condition:
```

The `Adaptation Plan` must clearly say whether to:

- wire the model into an already-supported Ascend MoE path,
- change upstream vLLM model/framework code,
- verify an existing `vllm-ascend` MoE path without backend changes,
- or stop and escalate due to a backend capability gap.

Do not start changing code until this comparison is concrete enough to explain why the current MoE backend should already work, or exactly what must be adapted.

### 2.5) Analyze attention adaptation path first

Run this step whenever the classification or model code suggests attention is relevant to the failure or adaptation scope, especially for:

- standard decoder attention with custom masking or KV behavior,
- sliding-window / sink / chunked-prefill / speculative decoding interactions,
- shared-KV or paged-KV assumptions,
- C8 / KV quantization,
- attention metadata construction issues in `model_runner_v1`,
- symptoms such as wrong output only during prefill/decode split, graph replay mismatch, or shape/layout errors in FIA operators.

Read:

- `references/attention-v1-analysis.md`

Treat this reference as the **current attention capability baseline** of `vllm-ascend`, not just background reading. The goal of this step is to compare:

- what the current backend already supports,
- what the new model's attention path requires,
- and what the mismatch implies for adaptation work.

Build that comparison from two sources:

1. **Existing capability baseline** from `references/attention-v1-analysis.md`
2. **New model attention features** inferred from:
   - `config.json`,
   - modeling code / remote code,
   - checkpoint key patterns,
   - observed runtime failure stage if the model already boots partially

Use the comparison to answer these questions before changing code:

1. Which `AscendAttentionState` should this model primarily exercise: `PrefillNoCache`, `PrefillCacheHit`, `DecodeOnly`, `ChunkedPrefill`, or `SpecDecoding`?
2. Is the expected path paged attention, FIA v1, FIA v2, `npu_fusion_attention` fallback, or a C8-specific path?
3. Which attention properties does the new model require: full attention vs sliding window, sink tokens, speculative decode interaction, shared KV, paged KV assumptions, KV quantization, special mask semantics, unusual head dim, MLA/Mamba/hybrid split?
4. Which metadata fields must be correct for this model: `block_table`, `slot_mapping`, `seq_lens*`, `actual_seq_lengths_q`, `query_start_loc`, `attn_mask`?
5. Does the model's attention subtype imply an existing backend should already cover it, or is the failure likely caused by a mismatch between upstream vLLM changes and vllm-ascend's current attention assumptions?

Produce an explicit **attention gap analysis** in your working notes before proceeding to Step 3.

This is a required output, not an optional note. For every model adaptation that touches or may touch attention, write the following fixed section before changing code:

```markdown
## Attention Gap Analysis

### 1. Current Capability
- Backend coverage:
- Supported state(s):
- Expected operator path:
- Supported metadata contract:
- Relevant known-good patterns:

### 2. Model Requirement
- Attention type from `config.json`:
- Attention behavior from modeling code:
- KV/cache behavior:
- Quantization / sink / sliding-window / spec-decode traits:
- Expected runtime stage(s):

### 3. Gap
- Capability mismatch:
- Metadata mismatch:
- Operator/layout mismatch:
- Unknowns to verify:

### 4. Adaptation Plan
- Fix location:
- Minimal code changes expected:
- Validation focus:
- Stop / escalate condition:
```

Fill it with concrete repo-specific content. Do not leave it generic.

Minimum quality bar for this section:

- `Current capability`: what `vllm-ascend` already supports for this attention shape/path
- `Model requirement`: what the new model's attention implementation needs
- `Gap`: the exact mismatch
- `Likely adaptation`: where the fix belongs
  - upstream model adapter / modeling code in `vllm`,
  - upstream framework integration in `vllm`,
  - existing `vllm-ascend` backend path already sufficient and only needs correct wiring,
  - or backend limitation that should stop work and be escalated

The `Adaptation Plan` must explicitly say whether the next step is:

- no backend change, only model-side wiring;
- upstream vLLM framework/model change;
- verification of an already-supported `vllm-ascend` path;
- or escalation because the current backend capability is insufficient.

If the issue is clearly attention-path related, record the suspected execution path, affected metadata fields, and the adaptation plan before moving to Step 3. Do not start implementation until this section is complete.

### 3) Analyze new operators (Ascend compatibility gate)

- Identify any new operators introduced in the model or its modeling code.
- Before changing any operator call path or retrying an Ascend-specific op failure, read:
  - `references/operator-compatibility-baseline.md`
- Classify each new operator by type and draw the appropriate conclusion:
    - **Torch** (native PyTorch op): Functional on Ascend ✅; performance is uncertain — note in report.
    - **Triton** kernel: Functional correctness uncertain ⚠️; requires explicit verification on Ascend; accuracy also uncertain.
    - **CUDA** kernel: Not supported on Ascend ❌; check whether a fallback implementation exists.
- If the failing path involves an Ascend-specific operator such as `torch_npu`, `torch.ops.npu`, or `aclnn*`, do not rely on blind local retries alone. After the first unsuccessful fix attempt, search the **official HiAscend operator documentation** for that operator before the next attempt.
- The HiAscend lookup must capture at least: supported dtype, shape constraints, layout/contiguous requirements, graph-mode limitations, and any fallback or replacement guidance. Record the page title / URL and use that evidence in the next fix attempt.
- **CUDA operator early-exit gate**: If any CUDA operator has no fallback (pure CUDA kernel with no Torch/Triton alternative), **stop here** — skip all subsequent validation steps and directly file a GitHub issue that explains:
    - which operator blocks Ascend support,
    - why no fallback exists,
    - recommended path forward (e.g., implement a custom Ascend op in `vllm-ascend`).
- **Triton operator early-exit gate**: If a Triton kernel is verified to be non-functional on Ascend (correctness failure or unacceptable accuracy degradation), **stop here** — file a GitHub issue that explains:
    - which Triton kernel fails and the observed failure mode,
    - recommended path forward (e.g., replace with a Torch-native fallback or implement a custom Ascend op).
- If every CUDA operator has a fallback and every Triton kernel passes verification, document fallback paths and continue.

Required output for this step:

```markdown
## Operator Compatibility Gap Analysis

### 1. Current Capability
- Existing supported operator class:
- Existing fallback expectations:
- Existing Ascend doc-backed constraints:

### 2. Model Requirement
- New operators introduced:
- Operator type per item:
- Required dtype/layout/shape:
- Expected fallback path:

### 3. Gap
- Unsupported operator:
- Missing fallback:
- Constraint mismatch:
- Unknowns to verify:

### 4. Adaptation Plan
- Fix location:
- Minimal fallback or call-site change:
- Validation focus:
- Stop / escalate condition:
```

### 4) Analyze framework-side code

- Before deciding to patch `vllm-ascend` or a common vLLM runtime module, read:
  - `references/framework-integration-baseline.md`
- Identify vLLM framework modules changed to support the new model (e.g., scheduler, attention backend, sampler, weight loader, worker) — anything beyond the model file and operators.
- For each changed module, check whether `vllm-ascend` already overrides or depends on it:
    - If the module is a **common vLLM module already covered by vllm-ascend**, check whether the existing vllm-ascend patch still applies correctly after the upstream change. If the patch needs updating, update it; otherwise no further action is needed.
    - If the module is **not covered by vllm-ascend** and contains Ascend-incompatible logic, add a minimal corresponding override under `/vllm-workspace/vllm-ascend`.
- Keep framework-side patches minimal and scoped to the incompatible code paths only.

Required output for this step when applicable:

```markdown
## Framework Integration Gap Analysis

### 1. Current Capability
- Existing vllm-ascend coverage:
- Existing patch/override path:
- Existing framework assumptions:

### 2. Model Requirement
- Upstream framework modules touched:
- Runtime path exercised by this model:
- Required framework behavior:

### 3. Gap
- Upstream drift:
- Missing override:
- Metadata / interface mismatch:
- Unknowns to verify:

### 4. Adaptation Plan
- Fix location:
- Existing patch to update vs new override:
- Validation focus:
- Stop / escalate condition:
```

### 4.5) Analyze quantization path

Run this step whenever the checkpoint or runtime path is quantized, including fp8, KV quant, W8A8, compressed-tensors, or any scale-paired load pattern.

Read:

- `references/quantization-baseline.md`
- `references/fp8-on-npu-lessons.md` when fp8 is involved

Required output when applicable:

```markdown
## Quantization Gap Analysis

### 1. Current Capability
- Existing supported quant path:
- Existing safe fallback path:
- Existing KV quant / attention quant support:

### 2. Model Requirement
- Checkpoint quant format:
- Runtime quant expectations:
- KV/cache quant traits:
- Scale / shard / dequant requirements:

### 3. Gap
- Loader quant gap:
- Runtime kernel gap:
- KV quant gap:
- Unknowns to verify:

### 4. Adaptation Plan
- Fix location:
- Minimal quant handling change:
- Validation focus:
- Stop / escalate condition:
```

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

Run the test directly:

```bash
python /tmp/npu_unit_tests/test_<operator_or_module>.py
```

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
- If the failure is on an Ascend-specific operator (`torch_npu`, `torch.ops.npu`, `aclnn*`, or an operator error clearly emitted by Ascend runtime), perform an **official HiAscend operator-doc lookup before attempt 2**. Use the documented dtype / shape / layout / graph constraints to guide the next patch.
- If it **fails again**: attempt a second fix and re-run. This counts as **attempt 2**.
- If it **still fails after 2 attempts**: **early exit** — do not proceed to serve validation. File a GitHub issue documenting:
    - which operator or module test failed,
    - the observed failure mode (error message + stack trace),
    - both fix attempts and why they did not resolve the issue,
    - any HiAscend operator documentation consulted (page title + URL) and what constraint it revealed,
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

- Run readiness poll and smoke requests using the provided script:
  ```bash
  bash scripts/smoke_test.sh <served-name>              # text-only
  bash scripts/smoke_test.sh <served-name> 8000 --multimodal  # VL models
  ```
- Validate architecture registration and loader path with logs (no unresolved architecture, no fatal missing-key errors).
- Try feature-first validation: EP + ACLGraph path first; eager path as fallback/isolation.
- For MoE models, try EP before TP-only runs. If EP fails, preserve the exact command, failure log, and the fallback command used for isolation.
- If startup succeeds but first request crashes (false-ready), treat as runtime failure and continue root-cause isolation.
- For `torch._dynamo` + `interpolate` + `NPU contiguous` failures on VL paths, try `TORCHDYNAMO_DISABLE=1` as diagnostic/stability fallback.
- For multimodal processor API mismatch (for example `skip_tensor_conversion` signature mismatch), use text-only isolation (`--limit-mm-per-prompt` set image/video/audio to 0) to separate processor issues from core weight loading issues.
- Capacity baseline by default (single machine): `max-model-len=128k` + `max-num-seqs=16`.
- Then expand concurrency (e.g., 32/64) if requested or feasible.
- If failure is confirmed as HBM exhaustion during load or first request, do not continue with `cpu-offload` experiments. Record the exact OOM evidence and conclude that more Ascend cards or larger HBM capacity are required.

> **Note**: Accuracy evaluation and performance benchmarking are out of scope for this skill. They are handled by a dedicated separate skill. If requested, invoke that skill after completing this step. However, whenever this skill needs an accuracy reference in docs, YAML, or acceptance criteria, it must use the ModelScope baseline rule above: prefer the matching model-size `gsm8k` score, otherwise fall back to another available dataset score and record the dataset name.

### 9) Backport, generate artifacts, and commit in delivery repo

- If implementation happened in `/vllm-workspace/*`, backport minimal final diff to current working repo.
- Generate test config YAML at `tests/e2e/models/configs/<ModelName>.yaml` following the schema of existing configs (must include `model_name`, `hardware`, `tasks` with accuracy metrics, and `num_fewshot`). Use accuracy results from evaluation to populate metric values.
- Generate tutorial markdown at `docs/source/tutorials/models/<ModelName>.md` following the standard template (Introduction, Supported Features, Environment Preparation with docker tabs, Deployment with serve script, Functional Verification with curl example, Accuracy Evaluation, Performance). Fill in model-specific details: HF path, hardware requirements, TP size, max-model-len, served-model-name, sample curl, and accuracy table.
- Generate adaptation report markdown at `docs/source/tutorials/models/<ModelName>-adaptation-report.md`. The report must summarize architecture, blocking root causes, consulted HiAscend operator docs (if any), code changes, dummy-vs-real differences, false-ready cases, and final validation evidence.
- If the adaptation report lives under `docs/source/tutorials/models/` and is not tutorial-style, add it to `[tool.check_docs_yaml_sync].exclude` in `pyproject.toml`.
- Update `docs/source/tutorials/models/index.md` to include both the new tutorial and the adaptation report entry.
- Confirm test config YAML, tutorial doc, and adaptation report are included in the staged files.
- Commit code changes once (single signed commit).

### 10) Prepare handoff artifacts

- Write comprehensive Chinese analysis report.
- Write compact Chinese runbook for server startup and validation commands.
- Include feature status matrix (supported / unsupported / checkpoint-missing / not-applicable).
- Include dummy-vs-real validation matrix and explicit non-equivalence notes.
- Include changed-file list, key logs, and final commit hash.
- Post the SKILL.md content (or a link to it) as a comment on the originating GitHub issue to document the AI-assisted workflow.

## Quality gate before final answer

- Service starts successfully from `$WORK_DIR` with direct command.
- OpenAI-compatible inference request succeeds (not startup-only).
- Key feature set is attempted and reported: ACLGraph / EP / flashcomm1 / multimodal. MTP reported if checkpoint supports it.
- Capacity baseline (`128k + bs16`) result is reported, or explicit reason why not feasible.
- If the final blocker is HBM capacity, the response explicitly says to add cards or use a larger-HBM Ascend machine, and does not present `cpu-offload` as the recommended next step.
- **Dummy stage evidence is present (if used), and real-weight stage evidence is present (mandatory).**
- Test config YAML exists at `tests/e2e/models/configs/<ModelName>.yaml` and follows the established schema (`model_name`, `hardware`, `tasks`, `num_fewshot`).
- Tutorial doc exists at `docs/source/tutorials/models/<ModelName>.md` and follows the standard template (Introduction, Supported Features, Environment Preparation, Deployment, Functional Verification, Accuracy Evaluation, Performance).
- Adaptation report exists at `docs/source/tutorials/models/<ModelName>-adaptation-report.md` and includes root-cause + validation history.
- Tutorial index at `docs/source/tutorials/models/index.md` includes the new model entry and adaptation report entry.
- Exactly one signed commit contains all code changes in current working repo.
- Final response includes commit hash, file paths, key commands, known limits, and failure reasons where applicable.
