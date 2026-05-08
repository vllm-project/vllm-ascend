# Adapt — Analyzing Upstream Changes and Adapting vllm-ascend

This document describes how to analyze an upstream vLLM diff and translate it
into adaptation changes in vllm-ascend. Read this at the start of each step's
adapt phase (Phase 4 in SKILL.md).

## Inputs Available

Phase 3 has generated two files for the current step:

- `/tmp/main2main/steps/<step-id>/upstream.patch` — the full upstream diff
- `/tmp/main2main/steps/<step-id>/changed-files.txt` — changed file paths

These are shared with the fix-ci phase if CI fails later.

## Analysis Method

1. **Read changed-files.txt** to see the scope.
2. **Check the file mapping table** to identify relevant changes.
3. **Read upstream.patch** for the mapped files. Focus on:
   - Function signature changes (added/removed/renamed parameters)
   - Import path changes (moved modules)
   - New abstract methods in base classes
   - Deleted or renamed classes/functions
   - Config field additions or relocations
4. **Apply changes** to the corresponding vllm-ascend files.

## File Mapping Table

| vLLM upstream path | vllm-ascend path | Typical changes |
|:---|:---|:---|
| `vllm/platforms/` | `vllm_ascend/platform/` | Platform interface, abstract methods |
| `vllm/v1/worker/` | `vllm_ascend/worker/` | Worker lifecycle, model loading |
| `vllm/v1/attention/` | `vllm_ascend/attention/` | Attention backend interface |
| `vllm/model_executor/layers/attention/` | `vllm_ascend/attention/` | Attention layer API |
| `vllm/model_executor/layers/fused_moe/` | `vllm_ascend/ops/fused_moe/` | MoE kernel interface |
| `vllm/distributed/` | `vllm_ascend/distributed/` | Collective ops, TP/PP logic |
| `vllm/config*` | `vllm_ascend/` (multiple) | Config class fields, constructor args |
| `vllm/compilation/` | `vllm_ascend/compilation/` | Compilation passes, fusion rules |
| `vllm/model_executor/models/` | `vllm_ascend/models/` | Model forward signatures |
| `vllm/model_executor/layers/quantization/` | `vllm_ascend/quantization/` | Quantization kernels |
| `vllm/v1/worker/gpu/spec_decode/` | `vllm_ascend/worker/` | Speculative decoding |

Files not in this table (docs/, tests/, benchmarks/) generally don't need adaptation.

## Adaptation Rules

1. **Use `vllm_version_is()` for version compatibility:**
   ```python
   from vllm_ascend.utils import vllm_version_is
   if vllm_version_is(">=0.19.0"):
       # new API
   else:
       # old API
   ```

2. **For signature changes**, find every call site with `grep -r` and update.

3. **For module moves**, update imports. Prefer version-guarded imports for
   backward compatibility.

4. **For deleted interfaces**, stop using them or provide a local equivalent.

5. **For new abstract methods**, implement in `vllm_ascend/platform.py` with
   an Ascend-appropriate implementation matching the base class signature.

## Round Ledger

Append a line to `/tmp/main2main/round-ledger.jsonl`:
```json
{"phase": "adapt", "step": 1, "files_modified": ["vllm_ascend/platform.py"], "summary": "..."}
```
