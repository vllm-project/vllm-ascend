# DBO Communication Flow Analysis Reference

This document describes the complete communication operator execution sequences for each decoder layer in various model types under A2 (AllGather) and A3 (AlltoAll) modes, along with the interleaved execution timeline of two ubatch threads in DBO.

## Background: DBO Dual-Stream Execution Model

DBO splits one batch into two micro-batches (ubatch0 and ubatch1), executing interleaved on two NPU streams:
- **ubatch0** runs on `compute_stream` (stream1)
- **ubatch1** runs on `comm_stream` (stream2)

Two threads synchronize through `record_current_stream` / `wait_current_stream_and_yield` primitives, achieving compute-communication overlap.

---

## 1. Dense Model (e.g., LLaMA, Qwen3-Dense)

### Per-Layer Communication Sequence (A2 AllGather Mode)

```
ColumnLinear(QKV/Gate)
  └─ AllGather (TP dimension aggregates input)
Attention computation
RowLinear(o_proj / down_proj)
  └─ ReduceScatter (TP dimension scatters output)
ColumnLinear(Gate/Up)
  └─ AllGather
MLP computation
RowLinear(down_proj)
  └─ ReduceScatter
```

### DBO Overlap Strategy (A2)

```
Timeline →
ubatch0(stream1): [ColumnAG]─[Attn]─[RowRS]─record─[ColumnAG]─[MLP]─[RowRS]─record─...
ubatch1(stream2):                          wait+yield─[ColumnAG]─[Attn]─[RowRS]─record─...
```

**Strategy**:
- `dbo_linear_row_hook(record=True)`: Record before RowLinear's ReduceScatter **starts** (marking computation complete)
- `dbo_linear_row_hook(record=False)`: Wait+yield after ReduceScatter **completes** (wait for counterpart computation, yield CPU)
- `dbo_linear_column_hook(record=True)`: Record before ColumnLinear's AllGather **starts**
- `dbo_linear_column_hook(record=False)`: Wait+yield after AllGather **completes**

**Reference implementation**: `qwen3_dense.py` → `QwenDenseAllgatherTemplate`

### DBO Overlap Strategy (A3 AlltoAll)

On A3, Dense model's TP communication remains AllGather/ReduceScatter (not AlltoAll), overlap strategy identical to A2.
`AlltoallTemplate` can directly reuse `AllgatherTemplate` logic, or decide whether to enable based on measured performance.

---

## 2. MoE Model (e.g., Qwen3-MoE, no MLA)

### Per-Layer Communication Sequence (A2 AllGather Mode)

```
ColumnLinear(QKV)
  └─ AllGather (TP)
Attention computation
RowLinear(o_proj)
  └─ ReduceScatter (TP)
MoEPrepare
  └─ AllGather (EP, aggregate hidden_states to all EP ranks)
Expert computation (local expert)
MoEFinalize
  └─ ReduceScatter (EP, scatter results back to each rank)
```

### DBO Overlap Strategy (A2)

```
Timeline →
ubatch0(stream1): [QKV_AG]─[Attn]─[RowRS]─record(ATTN_POST)─[MoEPrepare_AG]─[Expert]─[MoEFinalize_RS]─record(ATTN_PRE)─...
ubatch1(stream2):                              wait(ATTN_POST)+yield─[QKV_AG]─record(ATTN_PRE,first_layer)─[Attn]─[RowRS]─...
                                                                               wait(ATTN_PRE)+yield─[MoEPrepare_AG]─...
```

**Strategy**:
- `dbo_linear_row_hook(record=True)`: Record `ATTN_POST` before o_proj RowLinear ReduceScatter
- `dbo_moe_prepare_hook(record=False)`: Wait `ATTN_POST` + yield before MoEPrepare AllGather
- `dbo_linear_column_hook(record=True/False)`: Before/after QKV AllGather, first layer records `ATTN_PRE`, subsequent layers wait `ATTN_PRE` + yield
- `dbo_moe_finalize_hook(record=True)`: Record `ATTN_PRE` before MoEFinalize ReduceScatter (for next layer QKV to wait)

**Reference implementation**: `qwen3_moe.py` → `QwenMoEAllgatherTemplate`

### Per-Layer Communication Sequence (A3 AlltoAll Mode)

```
ColumnLinear(QKV)
  └─ AllGather (TP, TP communication still AllGather on A3)
Attention computation
RowLinear(o_proj)
  └─ ReduceScatter (TP)
MoEDispatch
  └─ AlltoAll (EP, route tokens to corresponding expert ranks)
Expert computation (local expert)
MoECombine
  └─ AlltoAll (EP, route results back to original ranks)
MoEFinalize
  └─ ReduceScatter (EP/TP)
```

### DBO Overlap Strategy (A3 AlltoAll)

Key characteristics of AlltoAll on A3:
1. **AlltoAll traffic is more uniform**, latency more predictable than AllGather
2. **MoEDispatch AlltoAll** can overlap with previous layer's **RowLinear ReduceScatter**
3. **MoECombine AlltoAll** can overlap with next layer's **QKV AllGather**

```
Timeline →
ubatch0(stream1): [QKV_AG]─[Attn]─[RowRS]─record(ATTN_POST)─[Dispatch_A2A]─record(MOE_DISPATCH)─[Expert]─[Combine_A2A]─record(ATTN_PRE)─[Finalize_RS]─...
ubatch1(stream2):                  wait(ATTN_POST)+yield─[QKV_AG]─[Attn]─[RowRS]─wait(MOE_DISPATCH)+yield─[Dispatch_A2A]─...
                                                                                                                          wait(ATTN_PRE)+yield─[QKV_AG]─...
```

**Strategy**:
- `dbo_linear_row_hook(record=True)`: Record `ATTN_POST` before o_proj RowLinear ReduceScatter
- `dbo_linear_row_hook(record=False)`: Wait+yield after ReduceScatter (`wait=False`, only yield no wait, let ubatch1 start QKV earlier)
- `dbo_moe_prepare_hook(record=True)`: Record `MOE_DISPATCH` after MoEDispatch AlltoAll completes
- `dbo_moe_prepare_hook(record=False)`: Wait `MOE_DISPATCH` + yield before MoEDispatch AlltoAll
- `dbo_moe_finalize_hook(record=True)`: Record `ATTN_PRE` after MoECombine AlltoAll completes (for next layer QKV to wait)
- `dbo_linear_column_hook(record=False)`: Wait `ATTN_PRE` + yield before QKV AllGather (wait for upper layer MoECombine completion)

**Note**: In A3 AlltoAll mode, `wait=False` in `dbo_linear_row_hook` is a key optimization—ubatch1 doesn't need to wait for ubatch0's ReduceScatter completion to start QKV AllGather, both can truly parallelize, only need CPU yield to release thread control.

---

## 3. MLA+MoE Model (e.g., DeepSeek V2/V3)

### Per-Layer Communication Sequence (A2 AllGather Mode)

```
MLAPreprocess
  └─ AllGather (TP, aggregate q_c and kv_no_split)
MLA computation
RowLinear(o_proj)
  └─ ReduceScatter (TP)
MoEPrepare
  └─ AllGather (EP)
Expert computation
MoEFinalize
  └─ ReduceScatter (EP)
```

**Reference implementation**: `deepseek.py` → `DeepseekAllgatherTemplate`

### Per-Layer Communication Sequence (A3 AlltoAll Mode)

```
MLAPreprocess
  └─ AllGather (TP, q_c + kv_no_split)
MLA computation
RowLinear(o_proj)
  └─ ReduceScatter (TP)
MoEDispatch
  └─ AlltoAll (EP)
Expert computation
MoECombine
  └─ AlltoAll (EP)
MoEFinalize
  └─ ReduceScatter (EP/TP)
```

### DBO Overlap Strategy (A3 AlltoAll)

```
Timeline →
ubatch0(stream1): [MLA_AG]─[MLA]─[RowRS]─record(ATTN_POST)─[Dispatch_A2A]─record(MOE_DISPATCH)─[Expert]─[Combine_A2A]─record(ATTN_PRE)─[Finalize_RS]─...
ubatch1(stream2):                 wait(ATTN_POST)+yield─[MLA_AG]─[MLA]─[RowRS]─wait(MOE_DISPATCH)+yield─[Dispatch_A2A]─...
                                                                                                            wait(ATTN_PRE)+yield─[MLA_AG]─...
```

**Strategy**:
- `dbo_mla_preprocess_hook(record=True)`: First layer records `ATTN_PRE` (guarded by `dbo_first_layer_sync`)
- `dbo_mla_preprocess_hook(record=False)`: Wait `ATTN_PRE` + yield before MLA AllGather
- `dbo_linear_row_hook(record=True)`: Record `ATTN_POST` before o_proj ReduceScatter
- `dbo_linear_row_hook(record=False)`: Wait+yield after ReduceScatter (`wait=False`)
- `dbo_moe_prepare_hook(record=True)`: Record `MOE_DISPATCH` after MoEDispatch AlltoAll completes
- `dbo_moe_prepare_hook(record=False)`: Wait `MOE_DISPATCH` + yield before MoEDispatch AlltoAll
- `dbo_moe_finalize_hook(record=True)`: Record `ATTN_PRE` after MoECombine AlltoAll completes (for next layer MLA to wait)

**Reference implementation**: `deepseek.py` → `DeepseekAlltoallTemplate`

---

## 4. First Layer Special Handling (dbo_first_layer_sync)

The first decoder layer has no "previous layer" MoEFinalize/MoECombine event to wait for.
Therefore, any hook that needs to wait for `ATTN_PRE` event requires special handling on first layer:

```python
# First layer: ubatch0 actively records, marking "I'm ready"
if get_forward_context().dbo_first_layer_sync:
    dbo_record_current_stream(event=UBatchEventKey.ATTN_PRE)
    get_forward_context().dbo_first_layer_sync = False
```

Scenarios requiring `dbo_first_layer_sync` guard:
- In MoE models: `dbo_linear_column_hook(record=True)` or `dbo_mla_preprocess_hook(record=True)`
- That is: any hook that "waits for previous layer MoEFinalize/MoECombine completion" needs to record itself on first layer to release the wait

---

## 5. Profiling Results to Communication Flow Mapping

This section explains how to map the hook call sequences collected in `references/profiling-guide.md` back to the communication flows described in this document, validating static analysis correctness.

### 5.1 Hook Call Sequence → Communication Flow Mapping

| Hook sequence observed in profiling | Corresponding communication flow node | Document section |
|------------------------------------|--------------------------------------|------------------|
| `dbo_linear_column_hook(record)` | Before ColumnLinear AllGather **starts** | 1. Dense Model |
| `dbo_linear_column_hook(wait)` | After ColumnLinear AllGather **completes** | 1. Dense Model |
| `dbo_linear_row_hook(record)` | Before RowLinear ReduceScatter **starts** | 1. Dense Model |
| `dbo_linear_row_hook(wait)` | After RowLinear ReduceScatter **completes** (A3 wait=False) | 2. MoE Model |
| `dbo_mla_preprocess_hook(record)` | Before MLA AllGather **starts** (first layer record ATTN_PRE) | 3. MLA+MoE Model |
| `dbo_mla_preprocess_hook(wait)` | After MLA AllGather **completes** (wait ATTN_PRE) | 3. MLA+MoE Model |
| `dbo_moe_prepare_hook(wait)` | Before MoEPrepare AllGather **starts** (A2, wait ATTN_POST) | 2. MoE Model A2 |
| `dbo_moe_prepare_hook(record)` | After MoEDispatch AlltoAll **completes** (A3, record MOE_DISPATCH) | 2. MoE Model A3 |
| `dbo_moe_prepare_hook(wait)` | Before MoEDispatch AlltoAll **starts** (A3, wait MOE_DISPATCH) | 2. MoE Model A3 |
| `dbo_moe_finalize_hook(record)` | After MoEFinalize/MoECombine **completes** (record ATTN_PRE) | 2. MoE Model |

### 5.2 Validating Communication Mode via Profiling (A2 vs A3)

**A2 AllGather Mode Profiling Characteristics**:
```
Per-layer hook sequence (MoE model):
  dbo_linear_column_hook(record)   ← Before QKV AllGather
  dbo_linear_column_hook(wait)     ← After QKV AllGather
  dbo_linear_row_hook(record)      ← Before o_proj RS (record ATTN_POST)
  dbo_linear_row_hook(wait)        ← After o_proj RS (A2 mode calls is_record=False but template executes pass)
  dbo_moe_prepare_hook(wait)       ← Before MoEPrepare (wait ATTN_POST)
  ← Note: dbo_moe_prepare_hook(record) doesn't appear (A2 has no AlltoAll, is_record=True executes pass)
  dbo_moe_finalize_hook(record)    ← Before MoEFinalize (record ATTN_PRE)
  ← Note: dbo_moe_finalize_hook(wait) doesn't appear (A2 doesn't have this wait, is_record=False executes pass)
```

**A3 AlltoAll Mode Profiling Characteristics**:
```
Per-layer hook sequence (MoE model):
  dbo_linear_column_hook(record)   ← Before QKV AllGather
  dbo_linear_column_hook(wait)     ← After QKV AllGather
  dbo_linear_row_hook(record)      ← Before o_proj RS (record ATTN_POST)
  dbo_linear_row_hook(wait)        ← After o_proj RS (wait=False, only yield) ← A3 specific
  dbo_moe_prepare_hook(record)     ← After Dispatch AlltoAll completes (record MOE_DISPATCH) ← A3 specific
  dbo_moe_prepare_hook(wait)       ← Before Dispatch AlltoAll (wait MOE_DISPATCH)
  dbo_moe_finalize_hook(record)    ← After Combine AlltoAll completes (record ATTN_PRE)
```

**Key differentiators**:
- `dbo_moe_prepare_hook(record)` **appears** → A3 AlltoAll mode
- `dbo_moe_prepare_hook(record)` **doesn't appear** → A2 AllGather mode
- `dbo_linear_row_hook(wait)` **appears** → A3 AlltoAll mode (wait=False yield)

### 5.3 Profiling Duration and Overlap Benefit Estimation

After measuring communication operator durations via profiling, estimate DBO overlap theoretical benefit:

```
Theoretical speedup = 1 / (1 - overlap_ratio)

Where:
overlap_ratio = min(comm_time, compute_time) / (comm_time + compute_time)
```

#### 5.3.1 Overlap Efficiency Analysis Method

For each overlap interval, calculate efficiency metric:

```
Overlap efficiency = min(T_comm_block, T_compute_segment) / max(T_comm_block, T_compute_segment)

  Efficiency > 80%  → Excellent, record/wait placement reasonable
  Efficiency 50-80% → Good, acceptable
  Efficiency < 50%  → Need to adjust record/wait placement or re-partition communication blocks
```

### 5.4 Handling Profiling Anomalies

| Profiling anomaly | Corresponding communication flow issue | Handling |
|------------------|---------------------------------------|----------|
| Hook call count > expected per layer | Model has multiple sub-layers of same type (e.g., multiple MLPs in Dense layer) | Add call counter in hook, only record/wait at specific counts |
| Hook call order doesn't match this document | Model forward order is special (e.g., post-norm, parallel attention) | Re-read model forward(), update communication flow diagram |
| `dbo_first_layer_sync` not triggered | First layer hook not properly handled | Check template's `dbo_first_layer_sync` condition |
| Gap between two layers' hook calls | Extra computation between layers (e.g., LayerNorm, residual connection) | Normal phenomenon, doesn't affect overlap strategy |

---

## 6. Known Uncovered Communication Paths

This section documents communication paths in the current DBO framework that **are known to exist but not covered by hooks**, along with their priorities and recommendations for adding them. When generating DBO templates for new models, reference this section to determine if additional hook instrumentation is needed.

### 6.1 Communication Path Inventory

| # | Communication Path | File/Class | Comm Type | Comm Group | Affected Model Types | Priority | Status |
|---|-------------------|------------|-----------|------------|---------------------|----------|--------|
| 1 | MLP down_proj ReduceScatter | `vllm_ascend/ops/linear_op.py` → `MLPRowParallelOp.apply_impl()` | ReduceScatter | TP | Dense models, MoE models' shared expert MLP | **P1 (High)** | Uncovered |
| 2 | Shared expert AllReduce | `vllm_ascend/ops/fused_moe/` related files | AllReduce | TP | DeepSeek V2/V3 and other MoE models with shared expert | **P0 (Highest)** | Uncovered |
| 3 | MoE routing Broadcast | MoE routing computation related code | Broadcast | EP/TP | All MoE models | P4 (Ignorable) | Uncovered (very short duration) |

### 6.2 Detailed Analysis

#### 6.2.1 MLPRowParallelOp ReduceScatter (P1)

**Background**: `MLPRowParallelOp` is the wrapper class for MLP layer's down_proj. Its `apply_impl()` contains `tensor_model_parallel_reduce_scatter` communication, but currently **has no DBO hook instrumentation**.

**Comparison**: `SequenceRowParallelOp` (o_proj's wrapper) already has `dbo_linear_row_hook` instrumentation, but `MLPRowParallelOp` does not.

**Impact**:
- **Dense models**: Each layer has 2 RowLinear communications (o_proj + down_proj), but only o_proj is covered by hook. down_proj's ReduceScatter cannot be overlapped.
- **MoE models' shared expert**: shared expert's MLP down_proj also uses `MLPRowParallelOp`, similarly uncovered.

**Typical duration**: 500-1000μs (TP=8, 910B)

**Suggested hook name**: `dbo_mlp_row_hook`

**Instrumentation location**:
```python
# vllm_ascend/ops/linear_op.py → MLPRowParallelOp.apply_impl()
# Add before and after reduce_scatter:
if forward_context.dbo_enabled:
    forward_context.dbo_template.dbo_mlp_row_hook(is_record=True)
output = tensor_model_parallel_reduce_scatter(output_parallel, 0)
if forward_context.dbo_enabled:
    forward_context.dbo_template.dbo_mlp_row_hook(is_record=False)
```

#### 6.2.2 Shared Expert AllReduce (P0)

**Background**: DeepSeek V2/V3 and similar models contain shared expert in MoE layer. Its output requires AllReduce aggregation within TP group. This AllReduce communication currently has no DBO hook.

**Impact**:
- shared expert AllReduce duration typically > 1000μs, is a significant overlap opportunity
- In DBO mode, this communication blocks the stream, cannot overlap with other computation

**Typical duration**: 1000-1800μs (TP=8, 910B)

**Suggested hook name**: `dbo_shared_expert_hook`

**Instrumentation location**: Need to confirm specific file and line number via grep:
```bash
grep -rn "all_reduce" vllm_ascend/ops/fused_moe/
```

```python
# Add before and after shared expert all_reduce:
if forward_context.dbo_enabled:
    forward_context.dbo_template.dbo_shared_expert_hook(is_record=True)
dist.all_reduce(shared_expert_output, group=tp_group)
if forward_context.dbo_enabled:
    forward_context.dbo_template.dbo_shared_expert_hook(is_record=False)
```

#### 6.2.3 MoE Routing Broadcast (P4)

**Background**: After routing computation, MoE models need to broadcast routing results to all EP ranks.

**Impact**: Very short duration (typically < 100μs), limited overlap benefit, overhead may exceed benefit.

**Recommendation**: Don't add hook for now, ignore in template.

### 6.3 Handling New Hooks in Templates

When new hooks are added to `base.py`, handling in templates depends on model type and overlap strategy:

**Dense Model** (new `dbo_mlp_row_hook`):
```
Updated per-layer hook call sequence:
  dbo_linear_column_hook(True)    ← Before QKV AllGather
  dbo_linear_column_hook(False)   ← After QKV AllGather
    [Attention computation]
  dbo_linear_row_hook(True)       ← Before o_proj ReduceScatter
  dbo_linear_row_hook(False)      ← After o_proj ReduceScatter
  dbo_linear_column_hook(True)    ← Before MLP gate/up AllGather
  dbo_linear_column_hook(False)   ← After MLP gate/up AllGather
    [MLP computation]
  dbo_mlp_row_hook(True)          ← Before MLP down_proj ReduceScatter ★New
  dbo_mlp_row_hook(False)         ← After MLP down_proj ReduceScatter ★New
```

**MoE Model (with shared expert)** (new `dbo_shared_expert_hook`):
```
Updated per-layer hook call sequence (A2 AllGather):
  dbo_linear_column_hook(True)    ← Before QKV AllGather
  dbo_linear_column_hook(False)   ← After QKV AllGather
    [Attention computation]
  dbo_linear_row_hook(True)       ← Before o_proj ReduceScatter
    [o_proj ReduceScatter]
  dbo_moe_prepare_hook(True)      ← Before MoE EP AllGather
    [EP AllGather]
  dbo_moe_prepare_hook(False)     ← After MoE EP AllGather
    [Expert computation]
  dbo_moe_finalize_hook(True)     ← Before MoE EP ReduceScatter
    [EP ReduceScatter]
  dbo_moe_finalize_hook(False)    ← After MoE EP ReduceScatter
  dbo_shared_expert_hook(True)    ← Before Shared expert AllReduce ★New
    [Shared expert AllReduce]
  dbo_shared_expert_hook(False)   ← After Shared expert AllReduce ★New
```

### 6.4 How to Discover New Uncovered Communications

When generating DBO templates for new models, check for uncovered communications:

1. **Run profiling**: Get actual communication operator list for the model
2. **Compare hook sequences**: Contrast profiling results with Step 2 static analysis hook sequences
3. **grep locate**: For missing communications, locate specific positions in vllm-ascend code
4. **Assess priority**: Determine if hook needs to be added based on communication duration
5. **Update this document**: Add newly discovered uncovered communication paths to table in 6.1

> **Maintenance note**: When new hooks are added to vllm-ascend code, update the corresponding entry's "Status" column in this section to "Covered", noting the covering PR/commit.