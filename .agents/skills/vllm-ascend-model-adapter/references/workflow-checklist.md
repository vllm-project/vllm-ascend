# Workflow Checklist

## 0) Environment prerequisites

Set these once per session. Defaults match the official vllm-ascend Docker image.

```bash
# --- configurable paths (adjust if your layout differs) ---
VLLM_SRC=/vllm-workspace/vllm              # vLLM source root
VLLM_ASCEND_SRC=/vllm-workspace/vllm-ascend # vllm-ascend source root
WORK_DIR=/workspace                         # directory to run vllm serve from
MODEL_ROOT=/models                          # parent directory of model checkpoints
```

Expected environment:

- Hardware: Ascend A2 or A3 server
- Software: official vllm-ascend Docker image (see `./Dockerfile` for full contents)
- TP=16 typical for A3 (16-NPU), TP=8 typical for A2 (8-NPU)

## 1) Fast triage commands

```bash
MODEL_PATH=${MODEL_ROOT}/<model-name>
echo "MODEL_PATH=$MODEL_PATH"

# model inventory
ls -la "$MODEL_PATH"

# architecture + quant hints
rg -n "architectures|model_type|quantization_config|torch_dtype|max_position_embeddings|num_nextn_predict_layers|version|num_attention_heads|num_key_value_heads|num_experts" "$MODEL_PATH/config.json"

# state-dict key layout hints (if index exists)
ls -la "$MODEL_PATH"/*index*.json 2>/dev/null || true

# model custom code (if exists)
ls -la "$MODEL_PATH"/*.py 2>/dev/null || true
```

## 2) Confirm implementation and delivery roots

```bash
# implementation roots (fixed by Dockerfile)
cd "$VLLM_SRC" && git status -s
cd "$VLLM_ASCEND_SRC" && git status -s

# runtime import source check (expect vllm-workspace path)
python - <<'PY'
import vllm
print(vllm.__file__)
PY

# direct-run working directory
cd "$WORK_DIR" && pwd

# delivery root (current repo)
cd <current-repo>
git status -s
```

## 3) Session hygiene (before rerun)

```bash
# stop stale servers
pkill -f "vllm serve|api_server|EngineCore" || true

# confirm port 8000 is free
netstat -ltnp 2>/dev/null | rg ':8000' || true
```

When user explicitly requests reset:

```bash
cd "$VLLM_SRC" && git reset --hard && git clean -fd
cd "$VLLM_ASCEND_SRC" && git reset --hard && git clean -fd
```

## 4) Model type classification

Determine the high-level model type from `config.json`:

```bash
# check model_type, architectures, and attention-related fields
rg -n "model_type|architectures|attention_type|sliding_window|mamba|mla|num_nextn_predict" \
  "$MODEL_PATH/config.json"
```

Classification:

- **LLM sub-types**: standard full attention / sliding window attention / Mamba (SSM) / multi-latent attention (MLA) / hybrid.
- **VLM**: any model with vision encoder or multimodal processor.
- **Whisper / ASR**: encoder-decoder speech model.

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
python -m py_compile \
  "$VLLM_SRC"/vllm/model_executor/models/<new_model>.py

python -m py_compile \
  "$VLLM_SRC"/vllm/transformers_utils/processors/<new_model>.py 2>/dev/null || true
```

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
# readiness
for i in $(seq 1 200); do
  curl -sf http://127.0.0.1:8000/v1/models >/tmp/models.json && break
  sleep 3
done

# text smoke (required)
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<served-name>","messages":[{"role":"user","content":"say hi"}],"temperature":0,"max_tokens":16}'

# VL smoke (required for multimodal models)
# send one text+image OpenAI-compatible request and require non-empty choices.
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
