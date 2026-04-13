# DBO Hook Call Sequences

This document provides complete hook call sequences for all supported model architectures.

## DBO Fundamentals

DBO splits one batch into two ubatches, executed by two threads on `compute_stream` (ubatch0) and `comm_stream` (ubatch1) respectively. Communication operators are interleaved with computation through `record`/`wait` primitives:

```
ubatch0 (compute_stream):  [Compute] → record(KEY) → [Comm] → ...
ubatch1 (comm_stream):                   wait(KEY)+yield → [Compute] → record(KEY) → [Comm] → ...
```

### Primitive Functions

- `dbo_record_current_stream(event=KEY)`: Records an event on current stream, signaling "my computation reached here"
- `dbo_wait_current_stream_and_yield(event=KEY)`: Waits for counterpart's event, then yields CPU to the other thread
- `dbo_wait_current_stream_and_yield(event=KEY, wait=False)`: Only yields CPU without waiting (for scenarios with no data dependency but need CPU yield)

## Hook Insertion Points

### Hook 1: `dbo_linear_column_hook`
**Location**: `vllm_ascend/ops/linear_op.py` → `MLPColumnParallelOp.apply_impl()` and `SequenceColumnParallelOp.apply_impl()`

**Corresponding to**: `MergedColumnParallelLinear` (MLP gate/up_proj), `QKVParallelLinear` (Attention QKV)

```python
# MLPColumnParallelOp (MLP gate/up_proj)
if forward_context.dbo_enabled:
    forward_context.dbo_template.dbo_linear_column_hook(is_record=True)
    input_parallel = self.comm_group.all_gather(input_, 0)   # ← AllGather
    forward_context.dbo_template.dbo_linear_column_hook(is_record=False)

# SequenceColumnParallelOp (QKV proj)
if forward_context.dbo_enabled:
    forward_context.dbo_template.dbo_linear_column_hook(is_record=True)
    if get_forward_context().flash_comm_v1_enabled and need_all_gather:
        input_ = tensor_model_parallel_all_gather(input_, 0)  # ← AllGather
    forward_context.dbo_template.dbo_linear_column_hook(is_record=False)
```

**Semantics**: `is_record=True` called **before** AllGather starts; `is_record=False` called **after** AllGather completes.

### Hook 2: `dbo_linear_row_hook`
**Location**: `vllm_ascend/ops/linear_op.py` → `SequenceRowParallelOp.apply_impl()` (o_proj) and `MLPRowParallelOp` (down_proj, **note: currently no hook**)

**Corresponding to**: `RowParallelLinear` (o_proj, down_proj)

```python
# SequenceRowParallelOp (o_proj)
if forward_context.dbo_enabled:
    forward_context.dbo_template.dbo_linear_row_hook(is_record=True)
dist.all_to_all_single(...)  # ← Communication (or reduce_scatter)
# ...
if forward_context.dbo_enabled:
    forward_context.dbo_template.dbo_linear_row_hook(is_record=False)

# Another branch (FC2 down_proj)
if forward_context.dbo_enabled:
    forward_context.dbo_template.dbo_linear_row_hook(is_record=True)
output = tensor_model_parallel_reduce_scatter(output_parallel, 0)  # ← ReduceScatter
forward_context.dbo_template.dbo_linear_row_hook(is_record=False)
```

**Semantics**: `is_record=True` called **before** ReduceScatter/AlltoAll starts; `is_record=False` called **after** communication completes.

### Hook 3: `dbo_mla_preprocess_hook`
**Location**: `vllm_ascend/attention/mla_v1.py` → MLA preprocess section

**Corresponding to**: `MultiHeadLatentAttention` / `MLAModules` (only for MLA models)

```python
# mla_v1.py
if forward_context.dbo_enabled:
    forward_context.dbo_template.dbo_mla_preprocess_hook(is_record=True)
    if get_forward_context().flash_comm_v1_enabled:
        q_c = tensor_model_parallel_all_gather(q_c.contiguous(), 0)       # ← AllGather
        kv_no_split = tensor_model_parallel_all_gather(kv_no_split.contiguous(), 0)
    forward_context.dbo_template.dbo_mla_preprocess_hook(is_record=False)
```

**Semantics**: `is_record=True` called **before** MLA AllGather starts; `is_record=False` called **after** AllGather completes.

### Hook 4: `dbo_moe_prepare_hook`
**Location (A2 AllGather)**: `vllm_ascend/ops/fused_moe/prepare_finalize.py` → `PrepareAndFinalize._prepare_with_ep_group()`

**Location (A3 AlltoAll)**: `vllm_ascend/ops/fused_moe/token_dispatcher.py` → `MoEAlltoAllTokenDispatcher.token_dispatch()`

```python
# prepare_finalize.py (A2 AllGather mode)
if forward_context.dbo_enabled:
    forward_context.dbo_template.dbo_moe_prepare_hook(is_record=True)
    hidden_states = get_ep_group().all_gather(hidden_states, 0)  # ← EP AllGather
    forward_context.dbo_template.dbo_moe_prepare_hook(is_record=False)

# token_dispatcher.py (A3 AlltoAll mode)
if forward_context.dbo_enabled:
    forward_context.dbo_template.dbo_moe_prepare_hook(is_record=True)
_, global_input_tokens, handle = async_all_to_all(...)  # ← EP AlltoAll
handle.wait()
if forward_context.dbo_enabled:
    forward_context.dbo_template.dbo_moe_prepare_hook(is_record=False)
```

**Semantics**:
- A2 mode: `is_record=True` **before** AllGather; `is_record=False` **after** AllGather
- A3 mode: `is_record=True` **before** AlltoAll; `is_record=False` **after** AlltoAll (after `handle.wait()`)

### Hook 5: `dbo_moe_finalize_hook`
**Location (A2 AllGather)**: `vllm_ascend/ops/fused_moe/prepare_finalize.py` → `PrepareAndFinalize._finalize_with_ep_group()`

**Location (A3 AlltoAll)**: `vllm_ascend/ops/fused_moe/token_dispatcher.py` → `MoEAlltoAllTokenDispatcher.token_combine()`

```python
# prepare_finalize.py (A2 AllGather mode)
if forward_context.dbo_enabled:
    forward_context.dbo_template.dbo_moe_finalize_hook(is_record=True)
    hidden_states = tensor_model_parallel_reduce_scatter(hidden_states, 0)  # ← ReduceScatter
    forward_context.dbo_template.dbo_moe_finalize_hook(is_record=False)

# token_dispatcher.py (A3 AlltoAll mode)
if forward_context.dbo_enabled:
    forward_context.dbo_template.dbo_moe_finalize_hook(is_record=True)
_, permutated_local_input_tokens, handle = async_all_to_all(...)  # ← EP AlltoAll (combine)
handle.wait()
if forward_context.dbo_enabled:
    forward_context.dbo_template.dbo_moe_finalize_hook(is_record=False)
```

**Semantics**:
- A2 mode: `is_record=True` **before** ReduceScatter; `is_record=False` **after** ReduceScatter
- A3 mode: `is_record=True` **before** Combine AlltoAll; `is_record=False` **after** AlltoAll

---

## Hook Sequences by Model Type

### Dense Model (per layer)

```
dbo_linear_column_hook(is_record=True)   ← Before QKV AllGather
  [QKV AllGather]
dbo_linear_column_hook(is_record=False)  ← After QKV AllGather
  [Attention Compute]
dbo_linear_row_hook(is_record=True)      ← Before o_proj ReduceScatter
  [o_proj ReduceScatter]
dbo_linear_row_hook(is_record=False)     ← After o_proj ReduceScatter
dbo_linear_column_hook(is_record=True)   ← Before MLP gate/up AllGather
  [MLP gate/up AllGather]
dbo_linear_column_hook(is_record=False)  ← After MLP gate/up AllGather
  [MLP Compute]
dbo_linear_row_hook(is_record=True)      ← Before MLP down ReduceScatter
  [MLP down ReduceScatter]
dbo_linear_row_hook(is_record=False)     ← After MLP down ReduceScatter
```

### MoE Model - A2 AllGather (per layer)

```
dbo_linear_column_hook(is_record=True)   ← Before QKV AllGather
  [QKV AllGather]
dbo_linear_column_hook(is_record=False)  ← After QKV AllGather
  [Attention Compute]
dbo_linear_row_hook(is_record=True)      ← Before o_proj ReduceScatter
  [o_proj ReduceScatter]
dbo_linear_row_hook(is_record=False)     ← After o_proj ReduceScatter (pass in A2)
dbo_moe_prepare_hook(is_record=True)     ← Before MoE EP AllGather
  [EP AllGather]
dbo_moe_prepare_hook(is_record=False)    ← After MoE EP AllGather
  [Expert Compute]
dbo_moe_finalize_hook(is_record=True)    ← Before MoE EP ReduceScatter
  [EP ReduceScatter]
dbo_moe_finalize_hook(is_record=False)   ← After MoE EP ReduceScatter
```

### MoE Model - A3 AlltoAll (per layer)

```
dbo_linear_column_hook(is_record=True)   ← Before QKV AllGather
  [QKV AllGather]
dbo_linear_column_hook(is_record=False)  ← After QKV AllGather
  [Attention Compute]
dbo_linear_row_hook(is_record=True)      ← Before o_proj communication
  [o_proj communication]
dbo_linear_row_hook(is_record=False)     ← After o_proj communication
dbo_moe_prepare_hook(is_record=True)     ← Before Dispatch AlltoAll
  [Dispatch AlltoAll + handle.wait()]
dbo_moe_prepare_hook(is_record=False)    ← After Dispatch AlltoAll
  [Expert Compute]
dbo_moe_finalize_hook(is_record=True)    ← Before Combine AlltoAll
  [Combine AlltoAll + handle.wait()]
dbo_moe_finalize_hook(is_record=False)   ← After Combine AlltoAll
```

### MLA+MoE Model - A2 AllGather (per layer)

```
dbo_mla_preprocess_hook(is_record=True)  ← Before MLA AllGather
  [MLA AllGather]
dbo_mla_preprocess_hook(is_record=False) ← After MLA AllGather
  [MLA Compute]
dbo_linear_row_hook(is_record=True)      ← Before o_proj ReduceScatter
  [o_proj ReduceScatter]
dbo_moe_prepare_hook(is_record=True)     ← Before MoE EP AllGather
  [EP AllGather]
dbo_moe_prepare_hook(is_record=False)    ← After MoE EP AllGather
  [Expert Compute]
dbo_moe_finalize_hook(is_record=True)    ← Before MoE EP ReduceScatter
  [EP ReduceScatter]
dbo_moe_finalize_hook(is_record=False)   ← After MoE EP ReduceScatter
```

### MLA+MoE Model - A3 AlltoAll (per layer)

```
dbo_mla_preprocess_hook(is_record=True)  ← Before MLA AllGather
  [MLA AllGather]
dbo_mla_preprocess_hook(is_record=False) ← After MLA AllGather
  [MLA Compute]
dbo_linear_row_hook(is_record=True)      ← Before o_proj communication
  [o_proj communication]
dbo_linear_row_hook(is_record=False)     ← After o_proj communication
dbo_moe_prepare_hook(is_record=True)     ← Before Dispatch AlltoAll
  [Dispatch AlltoAll + handle.wait()]
dbo_moe_prepare_hook(is_record=False)    ← After Dispatch AlltoAll
  [Expert Compute]
dbo_moe_finalize_hook(is_record=True)    ← Before Combine AlltoAll
  [Combine AlltoAll + handle.wait()]
dbo_moe_finalize_hook(is_record=False)   ← After Combine AlltoAll
```

### Hybrid(MLA+QKV)+MoE Model - A2 AllGather (per layer)

Hybrid models contain both Full Attention layers (using MLA) and Linear Attention layers (using standard QKV), sharing the same MoE block.

**Full Attention Layer (uses MLA):**
```
dbo_mla_preprocess_hook(is_record=True)  ← Before MLA AllGather
  [MLA AllGather]
dbo_mla_preprocess_hook(is_record=False) ← After MLA AllGather
  [MLA Compute]
```

**Linear Attention Layer (uses standard QKV):**
```
dbo_linear_column_hook(is_record=True)   ← Before QKV AllGather
  [QKV AllGather]
dbo_linear_column_hook(is_record=False)  ← After QKV AllGather
  [Attention Compute]
```

**Shared MoE Block (both layer types):**
```
dbo_linear_row_hook(is_record=True)      ← Before o_proj ReduceScatter
  [o_proj ReduceScatter]
dbo_moe_prepare_hook(is_record=True)     ← Before MoE EP AllGather
  [EP AllGather]
dbo_moe_prepare_hook(is_record=False)    ← After MoE EP AllGather
  [Expert Compute]
dbo_moe_finalize_hook(is_record=True)    ← Before MoE EP ReduceScatter
  [EP ReduceScatter]
dbo_moe_finalize_hook(is_record=False)   ← After MoE EP ReduceScatter
```

> **Key Point**: `dbo_mla_preprocess_hook` and `dbo_linear_column_hook` play the same role (both record/wait `ATTN_PRE`) but never trigger in the same layer. Reference: `BailingMoEV25AllgatherTemplate`.

### Hybrid(MLA+QKV)+MoE Model - A3 AlltoAll (per layer)

Similar to A2 structure, but with AlltoAll communication for MoE.

**Hook behavior difference from A2:**
- `dbo_linear_row_hook(is_record=False)`: Adds `wait(ATTN_POST, wait=False)` to only yield CPU
- `dbo_moe_prepare_hook(is_record=True)`: Records `MOE_DISPATCH` event before Dispatch AlltoAll
- `dbo_moe_prepare_hook(is_record=False)`: Waits for `MOE_DISPATCH` after Dispatch AlltoAll completes

---

## Communication Mode Detection

### A2 vs A3 Determination

Check the MoE dispatcher type in vllm-ascend:

```bash
VLLM_ASCEND_DIR="<vllm_ascend_path>"

# Check token_dispatcher (A3 AlltoAll indicator)
grep -rn "MoEAlltoAllTokenDispatcher\|async_all_to_all" $VLLM_ASCEND_DIR/ops/fused_moe/token_dispatcher.py | head -5

# Check model adaptation files for dispatcher usage
grep -rn "token_dispatcher\|MoEAlltoAll\|AlltoAllTokenDispatcher" $VLLM_ASCEND_DIR/
```

**Determination Rules**:
- Model uses `MoEAlltoAllTokenDispatcher` (`token_dispatcher.py`) → **A3 AlltoAll mode**
- Model uses `PrepareAndFinalize` (`prepare_finalize.py`) → **A2 AllGather mode**
- If uncertain, default to A2 template and note A3 needs confirmation

---

## Summary Table

| Model Type | A2 Hooks | A3 Hooks | First Layer Sync |
|------------|----------|----------|------------------|
| Dense | `column_hook`, `row_hook` | Same as A2 | Not needed (uses DEFAULT) |
| MoE (no MLA) | `column_hook`, `row_hook`, `moe_prepare_hook`, `moe_finalize_hook` | Same + `MOE_DISPATCH` event | Needed (`ATTN_PRE`) |
| MLA+MoE | `mla_preprocess_hook`, `row_hook`, `moe_prepare_hook`, `moe_finalize_hook` | Same + `MOE_DISPATCH` event | Needed (`ATTN_PRE`) |
| Hybrid+MoE | `mla_preprocess_hook`, `column_hook`, `row_hook`, `moe_prepare_hook`, `moe_finalize_hook` | Same + `MOE_DISPATCH` event | Needed (`ATTN_PRE`) |