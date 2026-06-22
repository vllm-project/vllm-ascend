# Model Adapter And Weight Loading Baseline

This document defines the current capability baseline of `vllm-ascend-model-adapter` at the **model registration, model wiring, weight mapping, weight loading** layer, which is used to compare with new model requirements and output gap analysis.

## 1. What problem does this layer solve?

The question answered at this level is not "Can the Ascend operator run?", but:

- Whether vLLM already knows this architecture;
- Whether the model already has a reusable adapter;
- Whether the key structure of checkpoint can be mapped to the current implementation of vLLM;
- TP/KV head/norm/rope/fp8 scale Whether these weight loading rules are already overridden by existing implementations.

If this layer is not connected, it will usually fail before attention/operator is executed.

## 2. Current capability baseline

### 2.1 Already have default capabilities

The current skill assumes that the following capabilities already exist or can be quickly reused:

- Determine the model entry through the `architectures` field of `config.json`;
- Determine whether the architecture has been registered through `vllm/model_executor/models/registry.py`;
- Add or reuse model adapter in `/vllm-workspace/vllm/vllm/model_executor/models/`;
- Add processor when needed;
- Solve the problem of inconsistency between checkpoint key and vLLM layer name through explicit weight remap rules;
- Keep implementation happening mainly in `vllm`, not `vllm-ascend`.

### 2.2 Default adaptation position

By default, this layer should prioritize:

- `vllm/model_executor/models/<new_model>.py`
- `vllm/model_executor/models/registry.py`
- `vllm/transformers_utils/processors/<new_model>.py` (if processor required)

Only if it is confirmed that the incompatibility is at the Ascend backend level, continue to `vllm-ascend` analysis.

### 2.3 Known high frequency capability boundaries

The current skill has by default regarded the following as high-frequency adaptation items for this layer:

- architecture registration is missing;
- remote code is not compatible with native vLLM adapter;
- qkv/o_proj/gate/moe layer naming is inconsistent;
- The weight loading rules related to q_norm / k_norm / kv_norm / rope are inconsistent;
- KV head replication or TP shard mode causes norm / head dimension mismatch;
- fp8 checkpoint requires weight + scale to be loaded in pairs;
- The safetensors index or shard key pattern is inconsistent with existing loader assumptions.

## 3. Typical input evidence for this layer

When doing gap analysis, give priority to:

- `config.json`
- `architectures`
- `model_type`
- `auto_map`
-Module naming for modeling/remote code
- safetensors index
- checkpoint key prefix and layer naming pattern
- Error reporting during load phase (missing keys / unexpected keys / shape mismatch)

## 4. Typical failure signals at this level

The following symptoms usually fall into this tier:

- architecture not recognized
- registry miss
- Model class import succeeded but `load_weights` failed
- missing / unexpected key appears in large numbers
- Some layers have shape mismatch, but attention/operator has not been actually executed yet
- qk norm / kv norm / tp shard dimensions do not match
- fp8 scale/weight pairing is incomplete

## 5. Adaptation judgment principle

### 5.1 Prioritize reuse of existing adapters

If the new model is just a slight variation of an already supported architecture, take precedence:

- Reuse existing adapter;
- Increase minimum remap / conditional path;
- Avoid creating a new set of model implementations.

### 5.2 Conditions for creating a new adapter

Only when the following conditions are true at the same time, it is preferable to create a new adapter:

- There is no suitable schema in the registry;
- The input/layer structure of the existing adapter is quite different from that of the new model;
- Cannot be reused cleanly through remap;
- processor / weight layout / execution semantics are all obviously different.

### 5.3 Don’t erroneously transfer model adaptation issues to Ascend backend

If the problem is essentially:

- architecture registration is missing;
- The weight key mapping is incorrect;
- processor binding error;
- The layer wiring in the vLLM model file does not match;

Then you should fix the `vllm` side first instead of changing `vllm-ascend` directly.

## 6. Fixed output template of this layer

Whenever this layer is involved, first write:

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

## 7. The most common adaptation actions

- Complement architecture mapping in `registry.py`;
- Reuse proximity model adapter;
- Added `load_weights` remap;
- Added shard rules for KV/QK norm, replicated KV heads, and rope variants;
- Add scale pairing / dequant load path for fp8 checkpoint;
- When there is a strong dependency on the processor, continue to expand the processor layer adaptation.

## 8. When to stop and upgrade

If you have confirmed:

- The model adapter is correct;
- Weight mapping is correct;
- The load phase passed;
- Failure only occurs at the Ascend backend/operator layer;

Then enter the attention/operator/framework layer analysis.

If the model must add a new modeling file in `vllm-ascend` to work, according to the current skill constraints, the analysis should be stopped and upgraded, not done directly.
