# Workflow Checklist

All scripts referenced below live in `.agents/skills/vllm-ascend-model-adapter/scripts/`.
Make them executable once: `chmod +x .agents/skills/vllm-ascend-model-adapter/scripts/*.sh`

## 0) Environment prerequisites

Set these once per session using the values confirmed at the Entry Points step.

```bash
# --- set to values confirmed with user at entry ---
VLLM_SRC=<user-specified vLLM source root>          # default: /vllm-workspace/vllm
VLLM_ASCEND_SRC=<user-specified vllm-ascend root>   # default: /vllm-workspace/vllm-ascend
WORK_DIR=/workspace                                  # directory to run vllm serve from
MODEL_ROOT=<parent dir of MODEL_PATH>                # derived from user-specified checkpoint path

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
# Default ATB path is /usr/local/Ascend/nnal/atb/set_env.sh.
# If that path does not exist, the script will warn and continue without it.
# Pass the correct path as the second argument if needed.
bash scripts/check_npu_env.sh "$TP_SIZE"
# or with explicit ATB path:
bash scripts/check_npu_env.sh "$TP_SIZE" /path/to/atb/set_env.sh
# or to skip ATB sourcing entirely:
bash scripts/check_npu_env.sh "$TP_SIZE" none
```

If any assertion fails: **stop here**, resolve the environment issue, then restart the checklist from Step 0.

## 1) Fast triage commands

```bash
MODEL_PATH=${MODEL_ROOT}/<model-name>
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

## 5) Operator compatibility gate

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

If early-exit applies, the GitHub issue must include:

- operator name and file location,
- why no fallback exists,
- recommended path (e.g., custom Ascend op in `vllm-ascend`).

## 6) Framework-side code analysis

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

## 6) Syntax sanity checks

```bash
bash scripts/syntax_check.sh \
  "$VLLM_SRC"/vllm/model_executor/models/<new_model>.py \
  "$VLLM_SRC"/vllm/transformers_utils/processors/<new_model>.py
```

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

GitHub issue must include: failing test name, error + stack trace, both fix attempts, recommended path forward.

## 7) Two-stage serve templates (direct run, default `:8000`)

### Stage A: dummy fast gate (first try)

```bash
cd "$WORK_DIR"
MODEL_PATH=${MODEL_ROOT}/<model-name>

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

## 11) Capacity baseline + sweep

- Baseline (single machine): **`max-model-len=128k` + `max-num-seqs=16`**.
- If baseline passes, expand to `max-num-seqs=32/64` when requested.
- If baseline cannot pass due hardware/runtime limits, report explicit root cause.

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
- Update `docs/source/tutorials/models/index.md` to include the new tutorial entry.

### GitHub issue comment

- Post SKILL.md content or AI-assisted workflow summary as a comment on the originating GitHub issue.

Confirm both test config YAML and tutorial doc are included in the signed commit.
