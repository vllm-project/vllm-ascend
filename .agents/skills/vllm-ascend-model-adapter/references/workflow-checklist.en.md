# Workflow Checklist

All scripts referenced below live in `.agents/skills/vllm-ascend-model-adapter/scripts/`.
Make them executable once: `chmod +x .agents/skills/vllm-ascend-model-adapter/scripts/*.sh`

## 0) Environment prerequisites

Set these once per session using the values confirmed at the Entry Points step.

```bash
# --- set to values confirmed with user at entry ---
VLLM_SRC=<user-specified vLLM source root>          # default: /vllm-workspace/vllm
VLLM_ASCEND_SRC=<user-specified vllm-ascend root>   # default: /vllm-workspace/vllm-ascend
WORK_DIR=<user-specified working dir>                # default: /workspace (created if missing)
MODEL_PATH=<local path to model checkpoint>          # set after download or confirmed on-disk

# activate user-specified Python environment (if provided)
# e.g.: conda activate vllm-ascend
#       source /path/to/venv/bin/activate
<activation command>
```

Expected environment:

- Hardware: Ascend A2 or A3 server
- Software: official vllm-ascend Docker image (see `./Dockerfile` for full contents)
- TP=16 typical for A3 (16-NPU), TP=8 typical for A2 (8-NPU)

## 0.5) NPU environment sanity check

Run once at session start before any other work. If any check fails, stop and resolve the environment issue first.

```bash
# Default: auto-detects CANN under /usr/local/Ascend and ~/Ascend.
# Pass explicit CANN path as second arg if auto-detect fails.
# Pass ATB path as third arg if needed.
bash scripts/check_npu_env.sh "$TP_SIZE"
# or with explicit CANN path:
bash scripts/check_npu_env.sh "$TP_SIZE" /path/to/cann/set_env.sh
# or with explicit CANN + ATB paths:
bash scripts/check_npu_env.sh "$TP_SIZE" /path/to/cann/set_env.sh /path/to/atb/set_env.sh
# or to skip CANN sourcing entirely:
bash scripts/check_npu_env.sh "$TP_SIZE" none
```

If any assertion fails: **stop here**, resolve the environment issue, then restart the checklist from Step 0.

## 0.7) Download model (if not already on disk)

Skip if model is already on disk and `$MODEL_PATH/config.json` exists.

**ModelScope:**

```bash
MS_CACHE="${MODELSCOPE_CACHE:-$HOME/.cache/modelscope/hub}"
MODEL_ID=<org/model-name>   # e.g. google/gemma-4-E4B-it
MODEL_PATH="$MS_CACHE/$(echo $MODEL_ID | tr '/' '___')"
modelscope download --model "$MODEL_ID" --local_dir "$MODEL_PATH"
```

**HuggingFace:**

```bash
MODEL_PATH=<target local dir>
huggingface-cli download "$MODEL_ID" --local-dir "$MODEL_PATH"
```

**Gate — must pass before Step 1:**

```bash
if [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "ERROR: $MODEL_PATH/config.json not found — download failed or path is wrong"
  exit 1
fi
echo "OK: model path verified at $MODEL_PATH"
```

## 1) Fast triage commands

```bash
# Ensure WORK_DIR exists
mkdir -p "$WORK_DIR"

bash scripts/triage_model.sh "$MODEL_PATH"
```

## 2) Confirm implementation and delivery roots

```bash
bash scripts/check_roots.sh "$VLLM_SRC" "$VLLM_ASCEND_SRC" "$WORK_DIR"

# delivery root (current repo)
cd <current-repo>
git status -s
```

## 3) Session hygiene (before rerun)

```bash
bash scripts/session_reset.sh
# or with a custom port:
bash scripts/session_reset.sh 8080
```

When user explicitly requests reset (destructive — confirm first):

```bash
cd "$VLLM_SRC" && git reset --hard && git clean -fd
cd "$VLLM_ASCEND_SRC" && git reset --hard && git clean -fd
```

## 4) Model type classification

```bash
python scripts/classify_model.py "$MODEL_PATH"
```

The script reads `config.json` and outputs:

- **High-level type**: LLM / VLM (Vision-Language) / Whisper (ASR)
- **LLM attention sub-type**: standard full attention / sliding-window / MLA / Mamba / hybrid
- **MoE**: yes/no with routed expert count
- **MTP**: enabled/disabled with layer count
- **Quantization**: type or none
- **Key numeric parameters**

The final `CLASSIFICATION_SUMMARY:` line is a JSON object for easy parsing.

## 4.1) Layer-by-layer compatibility matrix

Read:

- `references/model-layer-baseline.md`

This step is mandatory for every model adaptation.

Write:

```markdown
## Layer-by-Layer Compatibility Matrix

| Layer | Current capability | Model requirement | Gap | Adaptation plan |
| --- | --- | --- | --- | --- |
| ...use the dense-llm or moe-llm template from `references/model-layer-baseline.md`... |
```

Requirements:

- Use the `dense llm` template for non-MoE decoder-only dense LLMs.
- Use the `moe llm` template for routed-expert decoder-only LLMs.
- Fill every row in the selected template.
- `Current capability` must reference an existing implementation or backend path.
- `Model requirement` must come from config/modeling/checkpoint/runtime evidence.
- `Gap` must be explicit.
- `Adaptation plan` must name the fix location or say the row only needs verification.

Use this matrix to decide which later specialized analyses are required. Do not jump straight to attention-only reasoning.

## 4.2) Model adapter and weight loading analysis

Read:

- `references/model-adapter-and-weight-loading-baseline.md`

This step is mandatory for every model adaptation.

Write:

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

Do not escalate to backend/operator-first thinking until this section explains why model registration and weight loading are already aligned or exactly what is missing.

## 4.3) Processor and multimodal analysis

Run this step when the model is multimodal, has processor files, or text-only and multimodal behavior differ.

Read:

- `references/processor-and-multimodal-baseline.md`

Write when applicable:

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

## 4.4) MoE adaptation analysis

Run this step for every `moe llm`, and also when runtime symptoms point to router / expert / EP path issues.

Read:

- `references/moe-fused-analysis.md`

Use the reference as the **current `vllm-ascend` MoE capability baseline**. Then compare that baseline with the new model's MoE requirements inferred from:

- `config.json`,
- modeling / remote code,
- checkpoint key patterns,
- and any partially observed runtime behavior.

Pin down:

1. Whether router/gate behavior matches current `select_experts(...)` support.
2. Whether expert structure matches the current `w13/gate_up -> swiglu -> w2/down` contract.
3. Whether shared expert / residual MLP behavior is already expressible.
4. Which communication path should be the first validation target: `ALLGATHER`, `ALLTOALL`, `MC2`, or `FUSED_MC2`.
5. Whether weight layout, routing metadata, and quant metadata can map to current `MoEWeights`, `MoERoutingParams`, and `MoEQuantParams`.

Write down an explicit MoE gap analysis before continuing.

This is a mandatory checkpoint for every `moe llm`. Use the following fixed template and fill it with concrete findings before any code changes:

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

## 4.5) Attention adaptation analysis

Run this step before operator triage when the classification or failure symptoms point to attention-path work.

Read:

- `references/attention-v1-analysis.md`

Mandatory trigger cases:

- attention subtype is `standard`, `sliding-window`, or `hybrid` and the model introduces custom attention code;
- prefill works but decode fails, or decode works but chunked-prefill/spec-decode fails;
- failure involves FIA / paged attention / mask / `block_table` / `slot_mapping` / `query_start_loc` / `seq_lens`;
- the model uses KV quantization, shared KV, sink tokens, or unusual cache behavior.

Use the reference as the **current `vllm-ascend` attention capability baseline**. Then compare that baseline with the new model's attention requirements inferred from:

- `config.json`,
- modeling / remote code,
- checkpoint key patterns,
- and any partially observed runtime behavior.

Pin down:

1. Expected `AscendAttentionState`.
2. Expected operator path: `_npu_paged_attention`, `npu_fused_infer_attention_score`, `npu_fused_infer_attention_score_v2`, `npu_fusion_attention`, or C8 path.
3. New model attention properties: standard/sliding-window/hybrid, sink, chunked-prefill/spec-decode interaction, KV quantization, shared KV, paged KV assumptions, special mask semantics, unusual head dim.
4. Expected metadata contract: `block_table`, `slot_mapping`, `seq_lens*`, `actual_seq_lengths_q`, `query_start_loc`, `attn_mask`.
5. Whether the likely fix belongs in upstream vLLM model code, vLLM framework code, or an Ascend backend assumption.

Write down an explicit attention gap analysis before continuing.

This is a mandatory checkpoint. Use the following fixed template and fill it with concrete findings before any code changes:

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

Minimum required content:

- `Current capability`
- `Model requirement`
- `Gap`
- `Likely adaptation`

The `Adaptation Plan` must clearly say whether to:

- wire the model into an already-supported backend path,
- change upstream vLLM model/framework code,
- verify an existing `vllm-ascend` path without backend changes,
- or stop and escalate due to a backend capability gap.

Do not start changing code until this comparison is concrete enough to explain why the current backend should already work, or exactly what must be adapted.

## 5) Operator compatibility gate

Read:

- `references/operator-compatibility-baseline.md`

Scan the model's new modeling code for custom operators:

```bash
# look for CUDA extensions, triton kernels, custom ops
rg -n "torch\.ops\.|\.cu\b|triton\.jit|@triton\.jit|load_inline|CUDAExtension" \
  "$MODEL_PATH"/*.py "$VLLM_SRC"/vllm/model_executor/models/<new_model>.py 2>/dev/null || true
```

Decision table:

| Operator type | Ascend status | Action |
| --- | --- | --- |
| Torch (native PyTorch) | ✅ functional | Note performance uncertainty in report |
| Triton kernel | ⚠️ uncertain | Verify on Ascend; check accuracy |
| CUDA kernel with fallback | ❌ CUDA blocked | Use fallback; document path |
| CUDA kernel, no fallback | ❌ blocked | **Early exit** — file GitHub issue, skip validation |

If the failing path is an Ascend-specific operator such as `torch_npu`, `torch.ops.npu`, or `aclnn*`, search the **official HiAscend operator documentation** before the next fix attempt. Capture the operator's dtype support, shape constraints, layout/contiguous requirements, graph-mode limitations, and any documented fallback guidance.

If early-exit applies, the GitHub issue must include:

- operator name and file location,
- why no fallback exists,
- recommended path (e.g., custom Ascend op in `vllm-ascend`).

Write:

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

## 6) Framework-side code analysis

Read:

- `references/framework-integration-baseline.md`

Identify vLLM framework modules changed alongside the new model:

```bash
# check what non-model files were touched in the upstream commit
git -C "$VLLM_SRC" diff HEAD~1 --name-only | rg -v "model_executor/models/"
```

For each changed framework module, check vllm-ascend coverage:

```bash
# does vllm-ascend already patch this module?
rg -rn "<module_name>" "$VLLM_ASCEND_SRC"/vllm_ascend/ 2>/dev/null || true
```

- If covered by vllm-ascend → no action needed; change is inherited automatically.
- If not covered and contains Ascend-incompatible logic → add minimal override under `$VLLM_ASCEND_SRC/vllm_ascend/`.

Write when applicable:

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

## 6.2) Quantization analysis

Run this step when the checkpoint or runtime path is quantized.

Read:

- `references/quantization-baseline.md`
- `references/fp8-on-npu-lessons.md` when fp8 is involved

Write when applicable:

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

## 7) New model onboarding checklist

```bash
# architecture mapping check in vLLM
rg -n "<ArchitectureClass>|registry" "$VLLM_SRC"/vllm/model_executor/models/registry.py

# optional: inspect model config and weight index quickly
cat "$MODEL_PATH/config.json"
cat "$MODEL_PATH"/*index*.json 2>/dev/null || true
```

If architecture is missing/incompatible, minimally do:

1. Add model adapter under `$VLLM_SRC/vllm/model_executor/models/<new_model>.py`.
2. Add processor under `$VLLM_SRC/vllm/transformers_utils/processors/<new_model>.py` when needed.
3. Register architecture in `$VLLM_SRC/vllm/model_executor/models/registry.py`.
4. Add explicit loader/remap rules for checkpoint key patterns (qkv/norm/rope/fp8 scales).
5. Touch `$VLLM_ASCEND_SRC` only when backend-specific errors are confirmed.

## 5) Typical implementation touch points

- `$VLLM_SRC/vllm/model_executor/models/<new_model>.py`
- `$VLLM_SRC/vllm/transformers_utils/processors/<new_model>.py`
- `$VLLM_SRC/vllm/model_executor/models/registry.py`
- `$VLLM_ASCEND_SRC/vllm_ascend/...` (only if backend behavior requires it)

## 6.5) Intermediate NPU unit-test gate

For each new operator (Step 5) and changed framework module (Step 6), write and run a minimal NPU unit test **before** launching `vllm serve`.

```bash
mkdir -p /tmp/npu_unit_tests

# run all unit tests written for this adaptation
python /tmp/npu_unit_tests/test_<operator_or_module>.py
```

Retry policy (per test):

| Attempt | Action |
| --- | --- |
| Test passes | Proceed to Stage A serve |
| Fails (attempt 1) | Fix and re-run |
| Fails (attempt 2) | Fix and re-run |
| Fails after 2 attempts | **Early exit** — file GitHub issue; skip serve |

For Ascend-specific operator failures, perform the HiAscend lookup before attempt 2 and include the consulted page title / URL plus the extracted constraints in the issue.

GitHub issue must include: failing test name, error + stack trace, both fix attempts, recommended path forward.

## 7) Two-stage serve templates (direct run, default `:8000`)

### Stage A: dummy fast gate (first try)

```bash
cd "$WORK_DIR"
HCCL_OP_EXPANSION_MODE=AIV \
VLLM_ASCEND_ENABLE_FLASHCOMM1=0 \
vllm serve "$MODEL_PATH" \
  --served-model-name <served-name> \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len <practical-max-len-or-131072> \
  --tensor-parallel-size <TP-size> \
  --max-num-seqs 16 \
  --load-format dummy \
  --port 8000
```

### Stage B: real-weight mandatory gate

```bash
# remove this from Stage A:
--load-format dummy
```

> Note: dummy is not equivalent to real weights. Real gate is mandatory before sign-off.

### EP + ACLGraph (feature-first, MoE only)

```bash
# add to Stage B when model is MoE and validating EP:
--enable-expert-parallel
```

### flashcomm1 check (MoE only)

```bash
# only evaluate flashcomm1 when model is MoE
VLLM_ASCEND_ENABLE_FLASHCOMM1=1
```

### Eager fallback (isolation)

```bash
# add to command for isolation only:
--enforce-eager
```

### TorchDynamo fallback (for VL interpolate-contiguous failures)

```bash
# add env var when logs contain:
# torch._dynamo.exc.TorchRuntimeError + interpolate +
# "NPU contiguous operator only supported contiguous memory format"
TORCHDYNAMO_DISABLE=1
```

## 8) Readiness + smoke checks (must verify true-ready)

```bash
# text smoke (required); add --multimodal for VL models
bash scripts/smoke_test.sh <served-name>
bash scripts/smoke_test.sh <served-name> 8000 --multimodal
```

> `Application startup complete` alone is not success. If first request crashes, treat as runtime failure (false-ready).

## 9) Feature validation checklist (default out-of-box)

1. `GET /v1/models` returns 200.
2. Text request returns 200 and non-empty output.
3. If VL model: text+image request returns 200.
4. ACLGraph evidence exists (`Replaying aclgraph`) where expected.
5. EP path is validated only for MoE models; non-MoE must be marked not-applicable.
6. flashcomm1 is validated only for MoE models; non-MoE must be marked not-applicable.
7. MTP status verified from config + weight index (enabled vs checkpoint-missing).
8. Dummy-vs-real differences are explicitly reported (if any).
9. Any false-ready case is explicitly marked as failure (with log signature).

## 10) Fallback ladder (recommended order)

1. Keep same params and reproduce once to ensure deterministic failure signature.
2. Add `--enforce-eager` to isolate graph-capture influence.
3. For VL + dynamo/interpolate/contiguous failures, add `TORCHDYNAMO_DISABLE=1`.
4. For multimodal-processor suspicion, isolate text-only by:
   - `--limit-mm-per-prompt '{"image":0,"video":0,"audio":0}'`
   - then check whether failure moves from processor layer to model core.
5. If issue persists, map failure signature to known-good implementation and patch minimal code.
6. If the failure signature is confirmed NPU HBM exhaustion, stop the fallback ladder there. Do not continue with `cpu-offload` exploration; report that the user needs more Ascend cards or a larger-HBM machine.

## 11) Capacity baseline + sweep

- Baseline (single machine): **`max-model-len=128k` + `max-num-seqs=16`**.
- If baseline passes, expand to `max-num-seqs=32/64` when requested.
- If baseline cannot pass due hardware/runtime limits, report explicit root cause.
- If the explicit root cause is HBM shortage, the recommendation must be to add cards or use a larger-HBM Ascend machine. Do not recommend `cpu-offload` as the next action.

## 12) Delivery checklist

```bash
# in current working repo (delivery root)
git add <changed-files>
git commit -sm "<message>"
```

Confirm:

- one signed commit only
- Chinese analysis + Chinese runbook present
- feature status matrix included with pass/fail reason
- dummy stage and real stage validation evidence included
- false-ready cases (if any) documented with final fallback status

### Test config generation

- Generate `tests/e2e/models/configs/<ModelName>.yaml` using accuracy results from evaluation.
- Must include: `model_name` (HF path), `hardware` (e.g. "Atlas A2 Series"), `tasks` (list with `name` and `metrics` containing `name` + `value`), `num_fewshot`.
- Follow the schema of existing configs (e.g. `Qwen3-8B.yaml`).

### Tutorial doc generation

- Generate `docs/source/tutorials/models/<ModelName>.md` from the standard template.
- Fill in model-specific details: HF path, hardware requirements, TP size, max-model-len, served-model-name, sample curl request, accuracy table.
- Must include sections: Introduction, Supported Features, Environment Preparation (with docker tabs for A2/A3), Deployment (with serve script), Functional Verification (with curl example), Accuracy Evaluation, Performance.
- Generate `docs/source/tutorials/models/<ModelName>-adaptation-report.md` summarizing the adaptation path, root causes, code changes, consulted HiAscend docs, dummy-vs-real differences, and final validation evidence.
- If the adaptation report is placed under `docs/source/tutorials/models/` and does not follow tutorial sync-block conventions, add it to `[tool.check_docs_yaml_sync].exclude` in `pyproject.toml`.
- Update `docs/source/tutorials/models/index.md` to include the new tutorial entry and the adaptation report entry.

### GitHub issue comment

- Post SKILL.md content or AI-assisted workflow summary as a comment on the originating GitHub issue.

Confirm both test config YAML and tutorial doc are included in the signed commit.
