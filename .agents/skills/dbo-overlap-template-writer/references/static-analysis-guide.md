# DBO Static Analysis Guide

This document describes the methodology for analyzing model code to identify communication operators and hook insertion points without profiling data.

## Overview

Static analysis is the primary method for identifying:
1. Model type classification
2. Hook insertion points
3. Communication operator locations
4. Hook call sequences

Use profiling-based analysis (see `profiling-guide.md`) only when:
- Model has conditional branches (dynamic routing, optional MoE layers)
- Communication time measurement is needed
- A2/A3 mode cannot be determined statically

---

## Step 1: Model File Location

### 1.1 Find from Registry

```bash
grep -n "<ModelName>" <vllm_path>/vllm/model_executor/models/registry.py
```

Example output:
```python
"LlamaForCausalLM":        ("llama",      "LlamaForCausalLM"),
"Qwen3MoeForCausalLM":     ("qwen3_moe",  "Qwen3MoeForCausalLM"),
#                            ↑ filename
```

### 1.2 Locate Model File

```
<vllm_path>/vllm/model_executor/models/<filename>.py
```

---

## Step 2: Model Type Classification

### 2.1 Keyword Search

```bash
MODEL_FILE="<vllm_path>/vllm/model_executor/models/<filename>.py"

# MLA detection (DeepSeek series)
grep -n "MultiHeadLatentAttention\|MLAModules" $MODEL_FILE

# MoE expert layer detection
grep -n "FusedMoE\|SharedFusedMoE" $MODEL_FILE

# Standard Attention Linear detection
grep -n "QKVParallelLinear\|MergedColumnParallelLinear" $MODEL_FILE

# Standard RowLinear detection
grep -n "RowParallelLinear" $MODEL_FILE
```

### 2.2 Classification Decision Tree

```
Has MLA + Has QKVParallelLinear + Has FusedMoE
    → Hybrid(MLA+QKV)+MoE model (e.g., Bailing V2.5)

Has MultiHeadLatentAttention/MLAModules + Has FusedMoE (no QKVParallelLinear)
    → MLA+MoE model (e.g., DeepSeek V2/V3, GLM-MoE-DSA)

No MLA + Has FusedMoE/SharedFusedMoE
    → MoE model (e.g., Qwen3-MoE, Bailing MoE, GLM-4 MoE)

No MLA + No FusedMoE
    → Dense model (e.g., LLaMA, Qwen3-Dense)
```

---

## Step 3: Trace Wrapper Classes

vLLM classes are wrapped by vllm-ascend with hook insertion points in the wrapper's `apply_impl()`.

### 3.1 Linear Operators

| vLLM Class | vllm-ascend Wrapper | Hook | File |
|------------|---------------------|------|------|
| `QKVParallelLinear` | `SequenceColumnParallelOp` | `dbo_linear_column_hook` | `linear_op.py` |
| `MergedColumnParallelLinear` | `MLPColumnParallelOp` | `dbo_linear_column_hook` | `linear_op.py` |
| `RowParallelLinear` (o_proj) | `SequenceRowParallelOp` | `dbo_linear_row_hook` | `linear_op.py` |
| `RowParallelLinear` (down_proj) | `MLPRowParallelOp` | **No hook** | `linear_op.py` |

### 3.2 MLA Operators

| vLLM Class | vllm-ascend Implementation | Hook | File |
|------------|---------------------------|------|------|
| `MultiHeadLatentAttention` / `MLAModules` | MLA v1 implementation | `dbo_mla_preprocess_hook` | `mla_v1.py` |

### 3.3 MoE Operators

| Communication Mode | Dispatcher | Hooks | File |
|-------------------|------------|-------|------|
| A2 (AllGather) | `PrepareAndFinalize` | `dbo_moe_prepare_hook`, `dbo_moe_finalize_hook` | `prepare_finalize.py` |
| A3 (AlltoAll) | `MoEAlltoAllTokenDispatcher` | `dbo_moe_prepare_hook`, `dbo_moe_finalize_hook` | `token_dispatcher.py` |

---

## Step 4: Analyze DecoderLayer.forward()

Read the `forward()` method of the DecoderLayer class to understand:
1. Order of Attention, MLP/MoE execution
2. Position of LayerNorm and residual connections
3. Exact sequence of communication operators

### 4.1 Typical Patterns

**Dense Model Pattern:**
```python
def forward(self, hidden_states, ...):
    # Self Attention
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(hidden_states)  # → QKV AG → Attn → o_proj RS
    hidden_states = residual + hidden_states

    # MLP
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)  # → MLP AG → MLP → down RS
    hidden_states = residual + hidden_states
    return hidden_states
```

**MoE Model Pattern:**
```python
def forward(self, hidden_states, ...):
    # Self Attention
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(hidden_states)  # → QKV AG → Attn → o_proj RS
    hidden_states = residual + hidden_states

    # MoE
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)  # → EP AG → Expert → EP RS
    hidden_states = residual + hidden_states
    return hidden_states
```

**MLA+MoE Model Pattern:**
```python
def forward(self, hidden_states, ...):
    # MLA
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(hidden_states)  # → MLA AG → MLA → o_proj RS
    hidden_states = residual + hidden_states

    # MoE
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)  # → EP AG → Expert → EP RS
    hidden_states = residual + hidden_states
    return hidden_states
```

---

## Step 5: Verify A2 vs A3 Mode

### 5.1 Check MoE Implementation

```bash
VLLM_ASCEND_DIR="<vllm_ascend_path>"

# Check if model uses AlltoAll dispatcher
grep -rn "MoEAlltoAllTokenDispatcher" $VLLM_ASCEND_DIR/models/<model_name>.py

# Check if model uses PrepareAndFinalize
grep -rn "PrepareAndFinalize\|prepare_finalize" $VLLM_ASCEND_DIR/models/<model_name>.py
```

### 5.2 Mode-Specific Hook Behavior

**A2 Mode:**
- `dbo_moe_prepare_hook`: EP AllGather
- `dbo_moe_finalize_hook`: EP ReduceScatter

**A3 Mode:**
- `dbo_moe_prepare_hook`: Dispatch AlltoAll
- `dbo_moe_finalize_hook`: Combine AlltoAll
- Additional `MOE_DISPATCH` event key

---

## Step 6: Generate Hook Sequence

Based on the analysis above, derive the complete hook call sequence for each layer.

See `hook-sequences.md` for detailed sequences for each model type.

---

## Common Pitfalls

### 1. Missing Hook in MLP Down Projection

`MLPRowParallelOp` currently has **no hook**. If down_proj uses this operator, it won't trigger `dbo_linear_row_hook`.

**Workaround**: Check the model file to see which operator down_proj uses. If it's `SequenceRowParallelOp` instead, the hook will trigger.

### 2. Conditional AllGather in QKV

`SequenceColumnParallelOp` only triggers AllGather when:
```python
if get_forward_context().flash_comm_v1_enabled and need_all_gather:
    input_ = tensor_model_parallel_all_gather(input_, 0)
```

If `flash_comm_v1_enabled` is False or `need_all_gather` is False, no communication occurs, but hooks still trigger.

### 3. Hybrid Model Dual Hooks

In Hybrid models, `dbo_mla_preprocess_hook` and `dbo_linear_column_hook` both exist but only one triggers per layer:
- Full Attention layer: `dbo_mla_preprocess_hook` triggers
- Linear Attention layer: `dbo_linear_column_hook` triggers

---

## Checklist

- [ ] Located model file from registry.py
- [ ] Classified model type (Dense/MoE/MLA+MoE/Hybrid)
- [ ] Identified all Attention and MLP/MoE classes
- [ ] Traced wrapper classes in vllm-ascend
- [ ] Determined A2 vs A3 communication mode
- [ ] Derived hook call sequence from DecoderLayer.forward()
- [ ] Identified any conditional execution paths
- [ ] Verified against reference implementations