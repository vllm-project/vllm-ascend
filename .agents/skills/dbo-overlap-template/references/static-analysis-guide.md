# DBO Static Code Analysis Guide

This document explains how to precisely identify various communication operators by directly reading/grepping model Python files, thereby determining the DBO overlap strategy **without relying on vague architectural feature inference**.

---

## I. Analysis Flow Overview

```
Model .py file
    |
    +-- Step 1: Identify model type (Dense / MoE / MLA+MoE)
    |       grep: MultiHeadLatentAttention, FusedMoE, QKVParallelLinear ...
    |
    +-- Step 2: Identify communication mode (A2 AllGather / A3 AlltoAll)
    |       grep: npu_moe_distribute_dispatch, async_all_to_all, token_dispatcher ...
    |
    +-- Step 3: Locate communication operator call order in each layer's forward()
    |       Read DecoderLayer.forward() method
    |
    +-- Step 4: Map to hook instrumentation points
            Confirm the actual code location for each hook
```

---

## II. Step 1: Identify Model Type

### 2.1 Execute grep Commands

```bash
MODEL_FILE="/path/to/vllm/model_executor/models/<model>.py"

echo "=== MLA Detection ==="
grep -n "MultiHeadLatentAttention\|MLAModules\|mla_preprocess\|latent_attention" $MODEL_FILE

echo "=== MoE Detection ==="
grep -n "FusedMoE\|SharedFusedMoE\|fused_moe\|MoELayer" $MODEL_FILE

echo "=== Attention Linear Detection ==="
grep -n "QKVParallelLinear\|MergedColumnParallelLinear" $MODEL_FILE

echo "=== MLP Linear Detection ==="
grep -n "RowParallelLinear\|ColumnParallelLinear" $MODEL_FILE
```

### 2.2 Decision Rules

| grep result | Model type | Corresponding template |
|-------------|-----------|----------------------|
| Has `MultiHeadLatentAttention`/`MLAModules` + Has `FusedMoE`/`SharedFusedMoE` | **MLA+MoE** | `DeepseekAllgatherTemplate` / `DeepseekAlltoallTemplate` |
| No MLA + Has `FusedMoE`/`SharedFusedMoE` | **MoE** | `QwenMoEAllgatherTemplate` / `QwenMoEAlltoallTemplate` |
| No MLA + No FusedMoE | **Dense** | `QwenDenseAllgatherTemplate` |

### 2.3 Typical Code Characteristics for Each Model Type

#### Dense Model (e.g., LLaMA, Qwen3-Dense)

```python
# Typical Attention layer
class LlamaAttention(nn.Module):
    def __init__(self, ...):
        self.qkv_proj = QKVParallelLinear(...)   # <- ColumnLinear
        self.o_proj = RowParallelLinear(...)      # <- RowLinear

# Typical MLP layer
class LlamaMLP(nn.Module):
    def __init__(self, ...):
        self.gate_up_proj = MergedColumnParallelLinear(...)  # <- ColumnLinear
        self.down_proj = RowParallelLinear(...)               # <- RowLinear
```

**Identification features**:
- `QKVParallelLinear` or `MergedColumnParallelLinear` (Attention QKV)
- `MergedColumnParallelLinear` (MLP Gate+Up)
- `RowParallelLinear` (o_proj and down_proj)
- **No** `FusedMoE`, **No** `MultiHeadLatentAttention`

#### MoE Model (e.g., Qwen2-MoE, Qwen3-MoE, Mixtral)

```python
# Typical Attention layer (same as Dense)
class Qwen2MoeAttention(nn.Module):
    def __init__(self, ...):
        self.qkv_proj = QKVParallelLinear(...)
        self.o_proj = RowParallelLinear(...)

# Typical MoE layer
class Qwen2MoeSparseMoeBlock(nn.Module):
    def __init__(self, ...):
        self.experts = FusedMoE(...)             # <- MoE expert layer
        # or
        self.shared_expert = SharedFusedMoE(...) # <- Shared expert (some models)
```

**Identification features**:
- `QKVParallelLinear` (Attention QKV)
- `RowParallelLinear` (o_proj)
- `FusedMoE` or `SharedFusedMoE` (MoE expert layer)
- **No** `MultiHeadLatentAttention`

#### MLA+MoE Model (DeepSeek V2/V3/R1)

```python
# Typical MLA Attention layer
class DeepseekV2Attention(nn.Module):
    def __init__(self, ...):
        # Note: No QKVParallelLinear, uses MLA-specific structure
        self.kv_a_proj_with_mqa = ReplicatedLinear(...)
        self.q_a_proj = ReplicatedLinear(...)
        # MLA preprocess is handled in mla_v1.py

# Typical MoE layer (similar to Qwen MoE)
class DeepseekV2MoE(nn.Module):
    def __init__(self, ...):
        self.experts = FusedMoE(...)
        self.shared_experts = DeepseekV2MLP(...)  # Shared expert
```

**Identification features**:
- `MultiHeadLatentAttention` or `MLAModules` (MLA attention)
- **No** `QKVParallelLinear` (MLA does not use standard QKV projection)
- `FusedMoE` (MoE expert layer)
- `RowParallelLinear` (o_proj)

---

## III. Step 2: Identify Communication Mode (A2 vs A3)

### 3.1 A3 AlltoAll Indicator Detection

```bash
MODEL_FILE="/path/to/model.py"
VLLM_ASCEND_DIR="/path/to/vllm_ascend"

echo "=== A3 AlltoAll indicator detection ==="
# Search in model file
grep -n "npu_moe_distribute_dispatch\|async_all_to_all\|AlltoAll\|alltoall" $MODEL_FILE

# Search in vllm_ascend MoE implementation
grep -rn "npu_moe_distribute_dispatch\|async_all_to_all" $VLLM_ASCEND_DIR/ops/fused_moe/

echo "=== token_dispatcher detection (A3 AlltoAll MoE dispatch implementation) ==="
grep -n "token_dispatcher\|TokenDispatcher\|MoEAlltoAllTokenDispatcher" $MODEL_FILE
grep -rn "class.*TokenDispatcher" $VLLM_ASCEND_DIR/ops/fused_moe/
```

### 3.2 Decision Rules

| Detection result | Communication mode | Hook file used |
|-----------------|-------------------|----------------|
| Found `npu_moe_distribute_dispatch` or `async_all_to_all` | **A3 AlltoAll** | `token_dispatcher.py` |
| Found `MoEAlltoAllTokenDispatcher` | **A3 AlltoAll** | `token_dispatcher.py` |
| None of the above found, MoE uses AllGather | **A2 AllGather** | `prepare_finalize.py` |
| Dense model (no MoE) | **A2/A3 identical** (TP communication is always AllGather/RS) | `linear_op.py` |

### 3.3 Confirm Hook File Paths

```bash
# Confirm A3 AlltoAll hook instrumentation points
grep -n "dbo_moe_prepare_hook\|dbo_moe_finalize_hook" $VLLM_ASCEND_DIR/ops/fused_moe/token_dispatcher.py

# Confirm A2 AllGather hook instrumentation points
grep -n "dbo_moe_prepare_hook\|dbo_moe_finalize_hook" $VLLM_ASCEND_DIR/ops/fused_moe/prepare_finalize.py

# Confirm Linear hook instrumentation points
grep -n "dbo_linear_column_hook\|dbo_linear_row_hook" $VLLM_ASCEND_DIR/ops/linear_op.py

# Confirm MLA hook instrumentation points
grep -n "dbo_mla_preprocess_hook" $VLLM_ASCEND_DIR/attention/mla_v1.py
```

---

## IV. Step 3: Locate Communication Operator Call Order in Each Layer's forward()

### 4.1 Read DecoderLayer.forward()

Find the `forward()` method of `DecoderLayer` (or equivalent class name) in the model file and map out the call order:

```bash
# Find DecoderLayer class definition
grep -n "class.*DecoderLayer\|class.*Block\|class.*Layer" $MODEL_FILE

# Read forward method (usually within 20-50 lines after class definition)
```

**Typical Dense model forward order**:
```python
def forward(self, hidden_states, ...):
    # 1. LayerNorm
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    
    # 2. Attention (internally calls QKVParallelLinear -> Attention -> RowParallelLinear)
    hidden_states, _ = self.self_attn(hidden_states, ...)
    #   +-- qkv_proj: ColumnLinear -> AllGather
    #   +-- attention compute
    #   +-- o_proj: RowLinear -> ReduceScatter
    
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    
    # 3. MLP (internally calls MergedColumnParallelLinear -> activation -> RowParallelLinear)
    hidden_states = self.mlp(hidden_states)
    #   +-- gate_up_proj: ColumnLinear -> AllGather
    #   +-- activation
    #   +-- down_proj: RowLinear -> ReduceScatter
    
    hidden_states = residual + hidden_states
    return hidden_states
```

**Typical MoE model forward order**:
```python
def forward(self, hidden_states, ...):
    # 1. Attention (same as Dense)
    hidden_states, _ = self.self_attn(hidden_states, ...)
    #   +-- qkv_proj: ColumnLinear -> AllGather
    #   +-- attention compute
    #   +-- o_proj: RowLinear -> ReduceScatter
    
    # 2. MoE (internally calls FusedMoE)
    hidden_states = self.mlp(hidden_states)
    #   +-- MoEPrepare: AllGather (A2) or Dispatch AlltoAll (A3)
    #   +-- Expert compute
    #   +-- MoEFinalize: ReduceScatter (A2) or Combine AlltoAll + RS (A3)
```

**Typical MLA+MoE model forward order**:
```python
def forward(self, hidden_states, ...):
    # 1. MLA Attention
    hidden_states, _ = self.self_attn(hidden_states, ...)
    #   +-- MLA preprocess: AllGather (q_c + kv_no_split)
    #   +-- MLA compute
    #   +-- o_proj: RowLinear -> ReduceScatter
    
    # 2. MoE (same as MoE model)
    hidden_states = self.mlp(hidden_states)
```

### 4.2 Special Case Handling

#### MoE Models with Shared Experts

Some MoE models (e.g., Qwen2-MoE, DeepSeek V2) contain shared experts whose communication pattern differs from routed experts:

```bash
# Detect shared experts
grep -n "shared_expert\|SharedFusedMoE\|shared_experts" $MODEL_FILE
```

- `SharedFusedMoE`: Shared experts also go through the MoE dispatch/combine path; hook behavior is the same as `FusedMoE`
- `DeepseekV2MLP` (as shared expert): Goes through the regular MLP path, using `ColumnLinear` + `RowLinear`

#### Hybrid MoE Models with Dense Layers

Some models (e.g., Qwen2-MoE) have some layers as Dense and others as MoE:

```bash
# Detect hybrid layer structure
grep -n "num_experts\|is_moe\|layer_type\|mlp_only_layers" $MODEL_FILE
```

If hybrid layers exist, the template needs to handle cases where `dbo_moe_prepare_hook` and `dbo_moe_finalize_hook` are not called (Dense layers do not trigger MoE hooks).

---

## V. Step 4: Communication Operator Identification Quick Reference Table

### 5.1 Attention Layer Communication Operators

| Operator | Identification keywords | Corresponding hook | Communication type |
|----------|----------------------|-------------------|-------------------|
| QKV projection (standard) | `QKVParallelLinear` | `dbo_linear_column_hook` | AllGather (TP) |
| QKV projection (merged) | `MergedColumnParallelLinear` (in Attention) | `dbo_linear_column_hook` | AllGather (TP) |
| Output projection | `RowParallelLinear` (`o_proj`) | `dbo_linear_row_hook` | ReduceScatter (TP) |
| MLA preprocess | `MultiHeadLatentAttention`, `MLAModules`, `mla_preprocess` | `dbo_mla_preprocess_hook` | AllGather (TP) |

### 5.2 MLP Layer Communication Operators

| Operator | Identification keywords | Corresponding hook | Communication type |
|----------|----------------------|-------------------|-------------------|
| Gate+Up projection | `MergedColumnParallelLinear` (in MLP) | `dbo_linear_column_hook` | AllGather (TP) |
| Down projection | `RowParallelLinear` (`down_proj`) | `dbo_linear_row_hook` | ReduceScatter (TP) |
| MoE Dispatch (A2) | `FusedMoE` (AllGather mode) | `dbo_moe_prepare_hook` | AllGather (EP) |
| MoE Finalize (A2) | `FusedMoE` (ReduceScatter mode) | `dbo_moe_finalize_hook` | ReduceScatter (EP) |
| MoE Dispatch (A3) | `npu_moe_distribute_dispatch`, `async_all_to_all` | `dbo_moe_prepare_hook` | AlltoAll (EP) |
| MoE Combine (A3) | `async_all_to_all` (combine phase) | `dbo_moe_finalize_hook` | AlltoAll (EP) |

### 5.3 Model Type -> Hook Usage Matrix

| Model type | `dbo_mla_preprocess_hook` | `dbo_linear_column_hook` | `dbo_linear_row_hook` | `dbo_moe_prepare_hook` | `dbo_moe_finalize_hook` |
|-----------|--------------------------|------------------------|----------------------|----------------------|------------------------|
| Dense | x | Yes (QKV + Gate/Up) | Yes (o_proj + down) | x | x |
| MoE | x | Yes (QKV only) | Yes (o_proj only) | Yes | Yes |
| MLA+MoE | Yes | x | Yes (o_proj only) | Yes | Yes |

> **Note**: In MoE models, MLP's Gate/Up and Down projections are handled internally by `FusedMoE`, so they no longer trigger `dbo_linear_column_hook` and `dbo_linear_row_hook` (for the MLP part).

---

## VI. Common Model Identification Examples

### 6.1 LLaMA Series (Dense)

```bash
grep -c "FusedMoE\|MultiHeadLatentAttention" llama.py
# Output: 0  -> Dense model

grep -n "QKVParallelLinear\|RowParallelLinear\|MergedColumnParallelLinear" llama.py
# Output:
# 42: self.qkv_proj = QKVParallelLinear(...)
# 48: self.o_proj = RowParallelLinear(...)
# 89: self.gate_up_proj = MergedColumnParallelLinear(...)
# 93: self.down_proj = RowParallelLinear(...)
```

**Conclusion**: Dense model, use `DenseAllgatherTemplate`.

### 6.2 Qwen2-MoE / Qwen3-MoE (MoE)

```bash
grep -c "MultiHeadLatentAttention" qwen2_moe.py
# Output: 0  -> No MLA

grep -n "FusedMoE\|SharedFusedMoE" qwen2_moe.py
# Output:
# 156: self.experts = FusedMoE(...)
# 162: self.shared_expert = SharedFusedMoE(...)  # Some models have this

grep -n "npu_moe_distribute_dispatch\|async_all_to_all" qwen2_moe.py
# Output: 0  -> A2 AllGather mode
```

**Conclusion**: MoE model, A2 mode, use `MoEAllgatherTemplate`.

### 6.3 DeepSeek V2/V3 (MLA+MoE)

```bash
grep -n "MultiHeadLatentAttention\|MLAModules" deepseek_v2.py
# Output:
# 234: class DeepseekV2Attention(MultiHeadLatentAttention):  -> Has MLA

grep -n "FusedMoE" deepseek_v2.py
# Output:
# 312: self.experts = FusedMoE(...)  -> Has MoE

grep -n "npu_moe_distribute_dispatch\|async_all_to_all" deepseek_v2.py
# If found -> A3 AlltoAll mode
# If not found -> A2 AllGather mode
```

**Conclusion**: MLA+MoE model, choose the corresponding template based on A2/A3.

---

## VII. Important Notes

1. **grep result of 0 does not mean it doesn't exist**: Some models use communication operators through inheritance or dynamic imports; check parent class files
2. **`MergedColumnParallelLinear` can appear in both Attention and MLP**: Distinguish based on context (class name, variable name)
3. **A3 AlltoAll indicators may be in the vllm_ascend adaptation layer rather than the original model file**: Check corresponding adaptation files under `vllm_ascend/` as well
4. **Architecture name in registry.py**: Used for `architectures` field matching when registering in `utils.py`; must exactly match the value in `hf_config.architectures`
