---
name: dbo-overlap-template
description: "For vllm-ascend's DBO (Dual-Batch Overlap) feature, automatically analyze a model's forward flow and communication operators based on the user-provided model name or model Python file, generate the corresponding overlap template file, and register it in dbo/utils.py."
---

# DBO Overlap Template Writer

## Overview

This skill generates DBO overlap templates for new models under `vllm_ascend/dbo/overlap_templates/` and registers them in the `select_dbo_templates()` function of `vllm_ascend/dbo/utils.py`, enabling the model to leverage Two-Batch Overlap (DBO) for improved inference performance.

## Trigger Conditions

Triggered when the user wants to adapt a model for DBO (Dual-Batch Overlap) to improve inference performance through dual-batch overlapping.

**The user must provide the following two inputs**:
1. **Model name**: The architecture class name registered in vllm, e.g., `LlamaForCausalLM`, `Qwen3MoeForCausalLM`, `DeepseekV3ForCausalLM`
2. **vllm path**: The root path of the vllm source code or installation directory, e.g., `/path/to/vllm`

> If the user has not provided either of the above, **you must ask the user for it first**. Do not guess or use default values.

## Execution Flow

### Step 1: Locate the Model Implementation File

Based on the user-provided **vllm path** and **model name**, search for the model implementation file in the following fixed directory:

```
<vllm_path>/vllm/model_executor/models/
```

#### 1.1 Look Up the Filename via registry.py (Preferred)

Read `registry.py` in that directory and look up the filename corresponding to the model name in the `_MODELS` dictionary:

```bash
grep -n "<ModelName>" <vllm_path>/vllm/model_executor/models/registry.py
```

The registration format in `registry.py` is:
```python
# _MODELS dictionary example
"LlamaForCausalLM":        ("llama",      "LlamaForCausalLM"),
"Qwen3MoeForCausalLM":     ("qwen3_moe",  "Qwen3MoeForCausalLM"),
"DeepseekV3ForCausalLM":   ("deepseek_v3","DeepseekV3ForCausalLM"),
#                            ^ The first element is the filename (without .py)
```

Extract the filename and locate:
```
<vllm_path>/vllm/model_executor/models/<filename>.py
```

#### 1.2 Handling Not Found

If the corresponding entry is not found in `registry.py`, **clearly inform the user**:
```
Could not find registration info for <ModelName> in <vllm_path>/vllm/model_executor/models/registry.py.
Please verify:
1. Whether the model name is correct (must be the architecture class name registered in vllm, e.g., LlamaForCausalLM)
2. Whether the vllm path is correct (currently using: <vllm_path>)
```

---

### Step 2: Static Code Analysis

Read the model file, analyze the `__init__` and `forward` implementations of the Attention, MLP, and MoE classes, trace back to the corresponding wrapper classes in vllm-ascend, and precisely identify communication operators and hook instrumentation points.

#### 2.1 DBO Principle

DBO splits a batch into two ubatches, which are executed alternately by two threads on `compute_stream` (ubatch0) and `comm_stream` (ubatch1). Stream switching is achieved through `record`/`wait` primitives before and after communication operators, enabling compute-communication overlap:

```
ubatch0 (compute_stream):  [Compute] -> record(KEY) -> [Communication] -> ...
ubatch1 (comm_stream):                    wait(KEY)+yield -> [Compute] -> record(KEY) -> [Communication] -> ...
```

- `dbo_record_current_stream(event=KEY)`: Record an event on the current stream, notifying the other side "my computation has reached this point"
- `dbo_wait_current_stream_and_yield(event=KEY)`: Wait for the other side's event, then yield CPU to the other thread
- `dbo_wait_current_stream_and_yield(event=KEY, wait=False)`: Only yield CPU without waiting for the event (used when there is no data dependency but CPU needs to be yielded)

#### 2.2 Identify Model Type

Read the `__init__` of the Attention and MLP/MoE classes in the model file, searching for the following keywords:

```bash
MODEL_FILE="<vllm_path>/vllm/model_executor/models/<filename>.py"

# Detect MLA (DeepSeek series)
grep -n "MultiHeadLatentAttention\|MLAModules" $MODEL_FILE

# Detect MoE expert layers
grep -n "FusedMoE\|SharedFusedMoE" $MODEL_FILE

# Detect standard Attention Linear
grep -n "QKVParallelLinear\|MergedColumnParallelLinear" $MODEL_FILE

# Detect standard RowLinear
grep -n "RowParallelLinear" $MODEL_FILE
```

**Decision logic**:
```
Contains MultiHeadLatentAttention/MLAModules + Contains FusedMoE  -> MLA+MoE model (e.g., DeepSeek V2/V3)
No MLA + Contains FusedMoE/SharedFusedMoE                        -> MoE model (e.g., Qwen2-MoE, Qwen3-MoE)
No MLA + No FusedMoE                                             -> Dense model (e.g., LLaMA, Qwen3-Dense)
```

#### 2.3 Trace vllm-ascend Wrapper Classes to Locate Hook Instrumentation Points

The `QKVParallelLinear`, `MergedColumnParallelLinear`, `RowParallelLinear`, and other classes in vllm are replaced by corresponding wrapper classes in vllm-ascend. **Hook instrumentation points are in the `apply_impl()` of the wrapper classes**, not in the original vllm classes.

##### Hook 1: `dbo_linear_column_hook`
**Instrumentation location**: `vllm_ascend/ops/linear_op.py` -> `MLPColumnParallelOp.apply_impl()` and `SequenceColumnParallelOp.apply_impl()`

Corresponds to: `MergedColumnParallelLinear` (MLP gate/up_proj), `QKVParallelLinear` (Attention QKV)

```python
# Actual instrumentation code in linear_op.py (MLPColumnParallelOp, for MLP gate/up_proj):
if forward_context.dbo_enabled:
    forward_context.dbo_template.dbo_linear_column_hook(is_record=True)
    input_parallel = self.comm_group.all_gather(input_, 0)   # <- AllGather communication
    forward_context.dbo_template.dbo_linear_column_hook(is_record=False)

# Actual instrumentation code in linear_op.py (SequenceColumnParallelOp, for QKV proj):
if forward_context.dbo_enabled:
    forward_context.dbo_template.dbo_linear_column_hook(is_record=True)
    if get_forward_context().flash_comm_v1_enabled and need_all_gather:
        input_ = tensor_model_parallel_all_gather(input_, 0)  # <- AllGather communication
    forward_context.dbo_template.dbo_linear_column_hook(is_record=False)
```

**Semantics**: `is_record=True` is called **before** AllGather starts; `is_record=False` is called **after** AllGather completes.

##### Hook 2: `dbo_linear_row_hook`
**Instrumentation location**: `vllm_ascend/ops/linear_op.py` -> `SequenceRowParallelOp.apply_impl()` (o_proj) and `MLPRowParallelOp` (down_proj, **note: MLPRowParallelOp currently has no hook**)

Corresponds to: `RowParallelLinear` (o_proj, down_proj)

```python
# Actual instrumentation code in linear_op.py (SequenceRowParallelOp, for o_proj):
if forward_context.dbo_enabled:
    forward_context.dbo_template.dbo_linear_row_hook(is_record=True)
dist.all_to_all_single(...)  # <- Communication (or reduce_scatter)
# ...
if forward_context.dbo_enabled:
    forward_context.dbo_template.dbo_linear_row_hook(is_record=False)

# Another location (FC2 down_proj, in another branch of SequenceRowParallelOp):
if forward_context.dbo_enabled:
    forward_context.dbo_template.dbo_linear_row_hook(is_record=True)
output = tensor_model_parallel_reduce_scatter(output_parallel, 0)  # <- ReduceScatter
forward_context.dbo_template.dbo_linear_row_hook(is_record=False)
```

**Semantics**: `is_record=True` is called **before** ReduceScatter/AlltoAll starts; `is_record=False` is called **after** communication completes.

##### Hook 3: `dbo_mla_preprocess_hook`
**Instrumentation location**: `vllm_ascend/attention/mla_v1.py` -> MLA preprocess section

Corresponds to: `MultiHeadLatentAttention` / `MLAModules` (only for MLA models)

```python
# Actual instrumentation code in mla_v1.py:
if forward_context.dbo_enabled:
    forward_context.dbo_template.dbo_mla_preprocess_hook(is_record=True)
    if get_forward_context().flash_comm_v1_enabled:
        q_c = tensor_model_parallel_all_gather(q_c.contiguous(), 0)       # <- AllGather
        kv_no_split = tensor_model_parallel_all_gather(kv_no_split.contiguous(), 0)
    forward_context.dbo_template.dbo_mla_preprocess_hook(is_record=False)
```

**Semantics**: `is_record=True` is called **before** MLA AllGather starts; `is_record=False` is called **after** AllGather completes.

##### Hook 4: `dbo_moe_prepare_hook`
**Instrumentation location (A2 AllGather)**: `vllm_ascend/ops/fused_moe/prepare_finalize.py` -> `PrepareAndFinalize._prepare_with_ep_group()`

**Instrumentation location (A3 AlltoAll)**: `vllm_ascend/ops/fused_moe/token_dispatcher.py` -> `MoEAlltoAllTokenDispatcher.token_dispatch()`

```python
# prepare_finalize.py (A2 AllGather mode):
if forward_context.dbo_enabled:
    forward_context.dbo_template.dbo_moe_prepare_hook(is_record=True)
    hidden_states = get_ep_group().all_gather(hidden_states, 0)  # <- EP AllGather
    forward_context.dbo_template.dbo_moe_prepare_hook(is_record=False)

# token_dispatcher.py (A3 AlltoAll mode):
if forward_context.dbo_enabled:
    forward_context.dbo_template.dbo_moe_prepare_hook(is_record=True)
_, global_input_tokens, handle = async_all_to_all(...)  # <- EP AlltoAll
handle.wait()
if forward_context.dbo_enabled:
    forward_context.dbo_template.dbo_moe_prepare_hook(is_record=False)
```

**Semantics**:
- A2 mode: `is_record=True` **before** AllGather starts; `is_record=False` **after** AllGather completes
- A3 mode: `is_record=True` **before** AlltoAll starts; `is_record=False` **after** AlltoAll completes (after `handle.wait()`)

##### Hook 5: `dbo_moe_finalize_hook`
**Instrumentation location (A2 AllGather)**: `vllm_ascend/ops/fused_moe/prepare_finalize.py` -> `PrepareAndFinalize._finalize_with_ep_group()`

**Instrumentation location (A3 AlltoAll)**: `vllm_ascend/ops/fused_moe/token_dispatcher.py` -> `MoEAlltoAllTokenDispatcher.token_combine()`

```python
# prepare_finalize.py (A2 AllGather mode):
if forward_context.dbo_enabled:
    forward_context.dbo_template.dbo_moe_finalize_hook(is_record=True)
    hidden_states = tensor_model_parallel_reduce_scatter(hidden_states, 0)  # <- ReduceScatter
    forward_context.dbo_template.dbo_moe_finalize_hook(is_record=False)

# token_dispatcher.py (A3 AlltoAll mode):
if forward_context.dbo_enabled:
    forward_context.dbo_template.dbo_moe_finalize_hook(is_record=True)
_, permutated_local_input_tokens, handle = async_all_to_all(...)  # <- EP AlltoAll (combine)
handle.wait()
if forward_context.dbo_enabled:
    forward_context.dbo_template.dbo_moe_finalize_hook(is_record=False)
```

**Semantics**:
- A2 mode: `is_record=True` **before** ReduceScatter starts; `is_record=False` **after** ReduceScatter completes
- A3 mode: `is_record=True` **before** Combine AlltoAll starts; `is_record=False` **after** AlltoAll completes

#### 2.4 Determine A2 vs A3 Communication Mode

Determine by checking the dispatcher type used by MoE in vllm-ascend:

```bash
VLLM_ASCEND_DIR="<vllm_ascend_path>"

# Check token_dispatcher (A3 AlltoAll indicator)
grep -rn "MoEAlltoAllTokenDispatcher\|async_all_to_all" $VLLM_ASCEND_DIR/ops/fused_moe/token_dispatcher.py | head -5

# Check dispatcher used in model adaptation files
grep -rn "token_dispatcher\|MoEAlltoAll\|AlltoAllTokenDispatcher" $VLLM_ASCEND_DIR/
```

**Decision rules**:
- Model uses `MoEAlltoAllTokenDispatcher` (`token_dispatcher.py`) -> **A3 AlltoAll mode**
- Model uses `PrepareAndFinalize` (`prepare_finalize.py`) -> **A2 AllGather mode**
- If uncertain, default to generating A2 template and note A3 as pending confirmation in comments

#### 2.5 Map Out the Hook Call Sequence for Each Decoder Layer

Read the call order in `DecoderLayer.forward()` in the model file, combined with the hook instrumentation points above, to derive the complete hook call sequence:

**Dense model (per layer)**:
```
dbo_linear_column_hook(is_record=True)   <- Before QKV AllGather starts
dbo_linear_column_hook(is_record=False)  <- After QKV AllGather completes
  [Attention compute]
dbo_linear_row_hook(is_record=True)      <- Before o_proj ReduceScatter starts
dbo_linear_row_hook(is_record=False)     <- After o_proj ReduceScatter completes
dbo_linear_column_hook(is_record=True)   <- Before MLP gate/up AllGather starts
dbo_linear_column_hook(is_record=False)  <- After MLP gate/up AllGather completes
  [MLP compute]
dbo_linear_row_hook(is_record=True)      <- Before MLP down ReduceScatter starts
dbo_linear_row_hook(is_record=False)     <- After MLP down ReduceScatter completes
```

**MoE model (per layer, A2 AllGather)**:
```
dbo_linear_column_hook(is_record=True)   <- Before QKV AllGather starts
dbo_linear_column_hook(is_record=False)  <- After QKV AllGather completes
  [Attention compute]
dbo_linear_row_hook(is_record=True)      <- Before o_proj ReduceScatter starts
  [o_proj ReduceScatter]
dbo_moe_prepare_hook(is_record=True)     <- Before MoE EP AllGather starts
  [EP AllGather]
dbo_moe_prepare_hook(is_record=False)    <- After MoE EP AllGather completes
  [Expert compute]
dbo_moe_finalize_hook(is_record=True)    <- Before MoE EP ReduceScatter starts
  [EP ReduceScatter]
dbo_moe_finalize_hook(is_record=False)   <- After MoE EP ReduceScatter completes
```

**MoE model (per layer, A3 AlltoAll)**:
```
dbo_linear_column_hook(is_record=True)   <- Before QKV AllGather starts
dbo_linear_column_hook(is_record=False)  <- After QKV AllGather completes
  [Attention compute]
dbo_linear_row_hook(is_record=True)      <- Before o_proj communication starts
  [o_proj communication]
dbo_linear_row_hook(is_record=False)     <- After o_proj communication completes
dbo_moe_prepare_hook(is_record=True)     <- Before Dispatch AlltoAll starts
  [Dispatch AlltoAll + handle.wait()]
dbo_moe_prepare_hook(is_record=False)    <- After Dispatch AlltoAll completes
  [Expert compute]
dbo_moe_finalize_hook(is_record=True)    <- Before Combine AlltoAll starts
  [Combine AlltoAll + handle.wait()]
dbo_moe_finalize_hook(is_record=False)   <- After Combine AlltoAll completes
```

**MLA+MoE model (per layer, A2 AllGather)**:
```
dbo_mla_preprocess_hook(is_record=True)  <- Before MLA AllGather starts
  [MLA AllGather]
dbo_mla_preprocess_hook(is_record=False) <- After MLA AllGather completes
  [MLA compute]
dbo_linear_row_hook(is_record=True)      <- Before o_proj ReduceScatter starts
  [o_proj ReduceScatter]
dbo_moe_prepare_hook(is_record=True)     <- Before MoE EP AllGather starts
  [EP AllGather]
dbo_moe_prepare_hook(is_record=False)    <- After MoE EP AllGather completes
  [Expert compute]
dbo_moe_finalize_hook(is_record=True)    <- Before MoE EP ReduceScatter starts
  [EP ReduceScatter]
dbo_moe_finalize_hook(is_record=False)   <- After MoE EP ReduceScatter completes
```

---

### Step 3: Plan the DBO Overlap Strategy

Based on the hook call sequence from Step 2, plan the `record`/`wait` behavior for each hook for A2 and A3 respectively.

**Diagram legend**: In the timing diagrams below, the two rows represent the execution order of ubatch0 (executes first) and ubatch1 (executes second) on their respective streams. `record(KEY)` marks a point on the current stream; `wait(KEY)` waits for the other side's mark before continuing, while yielding CPU to the other thread.

---

#### 3.1 A2 Hardware Overlap Principles

On A2, MoE communication uses EP AllGather (prepare) + EP ReduceScatter (finalize), and TP communication uses AllGather (column) + ReduceScatter (row).

**Three overlap rules**:
1. **ubatch0 post-attention communication** overlaps with **ubatch1 attention compute**
2. **ubatch1 post-attention communication** overlaps with **ubatch0 MLP/MoE compute**
3. **ubatch0 post-MLP/MoE communication** overlaps with **ubatch1 MLP/MoE compute**

##### A2 x Dense Model

```
Timeline ->
ubatch0: [QKV_AG]-[Attn]-record(ATTN_POST)-[o_proj_RS]-[MLP_AG]-[MLP]-record(DEFAULT)-[down_RS]
ubatch1:                  wait(ATTN_POST)-[QKV_AG]-[Attn]-record(ATTN_POST)-[o_proj_RS]-wait(DEFAULT)-[MLP_AG]-...
```

| hook | is_record=True | is_record=False |
|------|---------------|----------------|
| `dbo_linear_column_hook` | `record(DEFAULT)` | `wait(DEFAULT)` |
| `dbo_linear_row_hook` | `record(DEFAULT)` | `wait(DEFAULT)` |

> **Note**: Dense model A2/A3 strategies are identical (TP communication is always AllGather/ReduceScatter), using the default `DEFAULT` event is sufficient, no need to distinguish ATTN_PRE/POST. Reference implementation: `QwenDenseAllgatherTemplate`.

##### A2 x MoE Model (without MLA)

```
Timeline ->
ubatch0: [QKV_AG]-[Attn]-[o_proj_RS]-record(ATTN_POST)-[MoE_Prepare_AG]-[Expert]-record(ATTN_PRE)-[MoE_Finalize_RS]
ubatch1:    ^first layer record(ATTN_PRE)                wait(ATTN_POST)-[QKV_AG]-wait(ATTN_PRE)-[Attn]-[o_proj_RS]-record(ATTN_POST)-...
```

**Rule 1**: ubatch0 `record(ATTN_POST)` before o_proj ReduceScatter -> ubatch1 `wait(ATTN_POST)` before MoE Prepare AllGather (ubatch0 post-attn comm overlaps with ubatch1 attn compute)

**Rule 2**: ubatch0 `record(ATTN_PRE)` after Expert compute (before MoE Finalize) -> ubatch1 `wait(ATTN_PRE)` after QKV AllGather (ubatch1 post-attn comm overlaps with ubatch0 MoE compute)

**Rule 3**: ubatch0 MoE Finalize ReduceScatter overlaps with ubatch1 Expert compute (guaranteed by the wait position from Rule 2)

**First layer special handling**: The first layer has no MoE Finalize from the previous layer. ubatch0 checks `dbo_first_layer_sync` before QKV AllGather (`dbo_linear_column_hook(is_record=True)`), and if True, proactively `record(ATTN_PRE)` and sets it to False, unblocking ubatch1's wait.

| hook | is_record=True | is_record=False |
|------|---------------|----------------|
| `dbo_linear_column_hook` | First layer: `record(ATTN_PRE)` + clear flag; other layers: pass | `wait(ATTN_PRE)` |
| `dbo_linear_row_hook` | `record(ATTN_POST)` | pass |
| `dbo_moe_prepare_hook` | pass | `wait(ATTN_POST)` |
| `dbo_moe_finalize_hook` | `record(ATTN_PRE)` | pass |

> Reference implementation: `QwenMoEAllgatherTemplate`

##### A2 x MLA+MoE Model (DeepSeek Series)

MLA replaces the standard QKV AllGather, and `dbo_mla_preprocess_hook` takes the role of `dbo_linear_column_hook`.

```
Timeline ->
ubatch0: [MLA_AG]-[MLA]-[o_proj_RS]-record(ATTN_POST)-[MoE_Prepare_AG]-[Expert]-record(ATTN_PRE)-[MoE_Finalize_RS]
ubatch1:    ^first layer record(ATTN_PRE)               wait(ATTN_POST)-[MLA_AG]-wait(ATTN_PRE)-[MLA]-[o_proj_RS]-record(ATTN_POST)-...
```

| hook | is_record=True | is_record=False |
|------|---------------|----------------|
| `dbo_mla_preprocess_hook` | First layer: `record(ATTN_PRE)` + clear flag; other layers: pass | `wait(ATTN_PRE)` |
| `dbo_linear_column_hook` | pass (MLA model MLP is shared expert, goes through column hook, but DeepSeek implementation waits ATTN_POST here) | `wait(ATTN_POST)` |
| `dbo_linear_row_hook` | `record(ATTN_POST)` | pass |
| `dbo_moe_prepare_hook` | pass | `wait(ATTN_POST)` |
| `dbo_moe_finalize_hook` | `record(ATTN_PRE)` | pass |

> Reference implementation: `DeepseekAllgatherTemplate`

---

#### 3.2 A3 Hardware Overlap Principles

On A3, MoE communication uses EP AlltoAll (prepare dispatch) + EP AlltoAll (finalize combine), while TP communication is the same as A2.

**Dense model**: A3 strategy is identical to A2 (TP communication type unchanged). `AlltoallTemplate` can be empty (degrades to no overlap) or reuse `AllgatherTemplate`.

**MoE model**: Building on the three A2 rules, AlltoAll's dispatch and combine are two independent communications that can overlap with different computations:

1. **ubatch0 post-attention communication** overlaps with **ubatch1 attention compute** (same as A2)
2. **ubatch0 MoE Dispatch AlltoAll** overlaps with **ubatch1 pre-MoE compute** (between attention and dispatch)
3. **ubatch0 MoE Finalize AlltoAll** overlaps with **ubatch1 Expert compute**
4. **ubatch1 MoE Dispatch AlltoAll** overlaps with **ubatch0 Expert compute**
5. **ubatch1 MoE Finalize AlltoAll** overlaps with **ubatch0 next layer attention compute**

##### A3 x MoE Model (without MLA)

```
Timeline ->
ubatch0: [QKV_AG]-[Attn]-[o_proj]-record(ATTN_POST)-[o_proj_comm]-wait(ATTN_POST,wait=False)
         -record(MOE_DISPATCH)-[Dispatch_A2A]-wait(MOE_DISPATCH,wait=False)-[Expert]
         -record(ATTN_PRE)-[Finalize_A2A]

ubatch1:    ^first layer record(ATTN_PRE)
         wait(ATTN_POST)-[QKV_AG]-[Attn]-[o_proj]-record(ATTN_POST)-[o_proj_comm]
         wait(ATTN_PRE)-wait(MOE_DISPATCH)-[Dispatch_A2A]-[Expert]-record(ATTN_PRE)-[Finalize_A2A]
```

**Key differences (compared to A2)**:
- `dbo_linear_row_hook(is_record=False)`: Added `wait(ATTN_POST, wait=False)`, only yields CPU without waiting for data, letting ubatch1 start QKV as soon as possible while yielding CPU for ubatch0 to execute Dispatch AlltoAll
- `dbo_moe_prepare_hook(is_record=True)`: `record(MOE_DISPATCH)` **before** Dispatch AlltoAll starts, for ubatch1 to wait on
- `dbo_moe_prepare_hook(is_record=False)`: `wait(MOE_DISPATCH)` **after** Dispatch AlltoAll completes, waiting for ubatch0's Dispatch to finish before starting its own
- `dbo_moe_finalize_hook(is_record=True)`: `record(ATTN_PRE)` **before** Finalize AlltoAll starts, for the next layer's ubatch1 QKV to wait on

| hook | is_record=True | is_record=False |
|------|---------------|----------------|
| `dbo_linear_column_hook` | First layer: `record(ATTN_PRE)` + clear flag; other layers: pass | `wait(ATTN_PRE)` |
| `dbo_linear_row_hook` | `record(ATTN_POST)` | `wait(ATTN_POST, wait=False)` |
| `dbo_moe_prepare_hook` | `record(MOE_DISPATCH)` | `wait(MOE_DISPATCH)` |
| `dbo_moe_finalize_hook` | `record(ATTN_PRE)` | pass |

> Note: `QwenMoEAlltoallTemplate` is currently empty (`pass`), pending implementation.

##### A3 x MLA+MoE Model (DeepSeek Series)

```
Timeline ->
ubatch0: [MLA_AG]-[MLA]-[o_proj]-record(ATTN_POST)-[o_proj_comm]-wait(ATTN_POST,wait=False)
         -record(MOE_DISPATCH)-[Dispatch_A2A]-wait(MOE_DISPATCH,wait=False)-[Expert]
         -record(ATTN_PRE)-[Finalize_A2A]

ubatch1:    ^first layer record(ATTN_PRE)
         wait(ATTN_PRE)-[MLA_AG]-[MLA]-[o_proj]-record(ATTN_POST)-[o_proj_comm]
         wait(ATTN_POST,wait=False)-wait(MOE_DISPATCH)-[Dispatch_A2A]-[Expert]
         -record(ATTN_PRE)-[Finalize_A2A]
```

| hook | is_record=True | is_record=False |
|------|---------------|----------------|
| `dbo_mla_preprocess_hook` | First layer: `record(ATTN_PRE)` + clear flag; other layers: pass | `wait(ATTN_PRE)` |
| `dbo_linear_column_hook` | pass | pass |
| `dbo_linear_row_hook` | `record(ATTN_POST)` | `wait(ATTN_POST, wait=False)` |
| `dbo_moe_prepare_hook` | `record(MOE_DISPATCH)` | `wait(MOE_DISPATCH)` |
| `dbo_moe_finalize_hook` | `record(ATTN_PRE)` | pass |

> Reference implementation: `DeepseekAlltoallTemplate`

---

#### 3.3 First Layer Synchronization (dbo_first_layer_sync) Mechanism

The first decoder layer has no MoE Finalize event from the "previous layer" to wait on, requiring special handling:

- **Trigger condition**: The template has a hook with `wait(ATTN_PRE)` (i.e., the `is_record=False` branch of `dbo_linear_column_hook` or `dbo_mla_preprocess_hook`)
- **Handling**: In the corresponding hook's `is_record=True` branch, check `get_forward_context().dbo_first_layer_sync`, and if True, proactively `record(ATTN_PRE)` and set it to False

```python
# Standard pattern (required for MoE/MLA+MoE models):
def dbo_linear_column_hook(self, is_record):   # or dbo_mla_preprocess_hook
    if is_record:
        if get_forward_context().dbo_first_layer_sync:
            dbo_record_current_stream(event=UBatchEventKey.ATTN_PRE)
            get_forward_context().dbo_first_layer_sync = False
    else:
        dbo_wait_current_stream_and_yield(event=UBatchEventKey.ATTN_PRE)
```

**Dense models do not need** this mechanism (they use the DEFAULT event with no cross-layer dependency).

---

### Step 4: Generate Template File

Create a new file under `vllm_ascend/dbo/overlap_templates/`, referencing the code patterns in `references/template-code-patterns.md`.

File naming convention: `<model_family>.py` (e.g., `llama.py`, `qwen2_moe.py`)

---

### Step 5: Register in utils.py

Edit `vllm_ascend/dbo/utils.py`:
1. Add import at the top of the file
2. Add the corresponding `elif` branch in the `select_dbo_templates()` function

---

### Step 6: Verification

Check the generated code:
- `record` and `wait` must appear in pairs (one record and one wait for each EventKey)
- First layer special handling: If the template uses the `ATTN_PRE` event, check whether `dbo_first_layer_sync` protection is needed
- Code style must be consistent with existing templates (`deepseek.py`, `qwen3_moe.py`)

---

## Key File Paths

| File | Description |
|------|-------------|
| `vllm_ascend/dbo/overlap_templates/base.py` | Base class defining 5 hook interfaces |
| `vllm_ascend/dbo/overlap_templates/deepseek.py` | MLA+MoE model reference implementation |
| `vllm_ascend/dbo/overlap_templates/qwen3_moe.py` | MoE model reference implementation |
| `vllm_ascend/dbo/overlap_templates/qwen3_dense.py` | Dense model reference implementation |
| `vllm_ascend/dbo/utils.py` | Template selection entry point, where new templates are registered |
| `vllm_ascend/worker/ubatching.py` | DBO primitive definitions (`UBatchEventKey`, `dbo_record_current_stream`, etc.) |
| `vllm_ascend/ops/linear_op.py` | Hook instrumentation points: `dbo_linear_column_hook`, `dbo_linear_row_hook` |
| `vllm_ascend/attention/mla_v1.py` | Hook instrumentation point: `dbo_mla_preprocess_hook` |
| `vllm_ascend/ops/fused_moe/token_dispatcher.py` | Hook instrumentation points: `dbo_moe_prepare_hook`, `dbo_moe_finalize_hook` (AlltoAll) |
| `vllm_ascend/ops/fused_moe/prepare_finalize.py` | Hook instrumentation points: `dbo_moe_prepare_hook`, `dbo_moe_finalize_hook` (AllGather) |
| `<vllm_path>/vllm/model_executor/models/registry.py` | Architecture name to model filename mapping, looked up in Step 1 |

## Deliverables

1. **New template file**: `vllm_ascend/dbo/overlap_templates/<model_family>.py`
2. **Updated utils.py**: `vllm_ascend/dbo/utils.py` (new import + elif branch)
3. **Analysis report** (inline in the reply): Communication sequence diagram, overlap strategy explanation, purpose of each hook, static analysis result summary
