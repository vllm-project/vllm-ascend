# DBO Communication Flow Analysis Reference

This document describes the complete communication operator execution sequence for each decoder layer across different model types under A2 (AllGather) and A3 (AlltoAll) modes, as well as the interleaved execution timing of the two ubatch threads in DBO.

## Background: DBO Dual-Stream Execution Model

DBO splits a batch into two micro-batches (ubatch0 and ubatch1), which are executed alternately on two NPU streams:
- **ubatch0** runs on `compute_stream` (stream1)
- **ubatch1** runs on `comm_stream` (stream2)

The two threads synchronize through `record_current_stream` / `wait_current_stream_and_yield` primitives, enabling compute-communication overlap.

---

## I. Dense Model (e.g., LLaMA, Qwen3-Dense)

### Per-Layer Communication Sequence (A2 AllGather Mode)

```
ColumnLinear(QKV/Gate)
  +-- AllGather (TP dimension input aggregation)
Attention compute
RowLinear(o_proj / down_proj)
  +-- ReduceScatter (TP dimension output distribution)
ColumnLinear(Gate/Up)
  +-- AllGather
MLP compute
RowLinear(down_proj)
  +-- ReduceScatter
```

### DBO Overlap Strategy (A2)

```
Timeline ->
ubatch0(stream1): [ColumnAG]-[Attn]-[RowRS]-record-[ColumnAG]-[MLP]-[RowRS]-record-...
ubatch1(stream2):                          wait+yield-[ColumnAG]-[Attn]-[RowRS]-record-...
```

**Strategy**:
- `dbo_linear_row_hook(record=True)`: record **before** RowLinear ReduceScatter starts (marks compute completion)
- `dbo_linear_row_hook(record=False)`: wait+yield **after** ReduceScatter completes (wait for the other side's compute to finish, yield CPU)
- `dbo_linear_column_hook(record=True)`: record **before** ColumnLinear AllGather starts
- `dbo_linear_column_hook(record=False)`: wait+yield **after** AllGather completes

**Reference implementation**: `qwen3_dense.py` -> `QwenDenseAllgatherTemplate`

### DBO Overlap Strategy (A3 AlltoAll)

On A3, Dense model TP communication is still AllGather/ReduceScatter (not AlltoAll), so the overlap strategy is identical to A2.
`AlltoallTemplate` can directly reuse `AllgatherTemplate` logic, or decide whether to enable based on actual test results.

---

## II. MoE Model (e.g., Qwen3-MoE, without MLA)

### Per-Layer Communication Sequence (A2 AllGather Mode)

```
ColumnLinear(QKV)
  +-- AllGather (TP)
Attention compute
RowLinear(o_proj)
  +-- ReduceScatter (TP)
MoEPrepare
  +-- AllGather (EP, aggregate hidden_states to all EP ranks)
Expert compute (local experts)
MoEFinalize
  +-- ReduceScatter (EP, distribute results back to each rank)
```

### DBO Overlap Strategy (A2)

```
Timeline ->
ubatch0(stream1): [QKV_AG]-[Attn]-[RowRS]-record(ATTN_POST)-[MoEPrepare_AG]-[Expert]-[MoEFinalize_RS]-record(ATTN_PRE)-...
ubatch1(stream2):                                wait(ATTN_POST)+yield-[QKV_AG]-record(ATTN_PRE,first layer)-[Attn]-[RowRS]-...
                                                                                 wait(ATTN_PRE)+yield-[MoEPrepare_AG]-...
```

**Strategy**:
- `dbo_linear_row_hook(record=True)`: record `ATTN_POST` before o_proj RowLinear ReduceScatter
- `dbo_moe_prepare_hook(record=False)`: wait `ATTN_POST` + yield before MoEPrepare AllGather
- `dbo_linear_column_hook(record=True/False)`: Before/after QKV AllGather, first layer records `ATTN_PRE`, subsequent layers wait `ATTN_PRE` + yield
- `dbo_moe_finalize_hook(record=True)`: record `ATTN_PRE` before MoEFinalize ReduceScatter (for next layer's QKV to wait on)

**Reference implementation**: `qwen3_moe.py` -> `QwenMoEAllgatherTemplate`

### Per-Layer Communication Sequence (A3 AlltoAll Mode)

```
ColumnLinear(QKV)
  +-- AllGather (TP, TP communication on A3 is still AllGather)
Attention compute
RowLinear(o_proj)
  +-- ReduceScatter (TP)
MoEDispatch
  +-- AlltoAll (EP, route tokens to corresponding expert ranks)
Expert compute (local experts)
MoECombine
  +-- AlltoAll (EP, route results back to original ranks)
MoEFinalize
  +-- ReduceScatter (EP/TP)
```

### DBO Overlap Strategy (A3 AlltoAll)

Key characteristics of AlltoAll on A3:
1. **AlltoAll communication volume is more uniform**, latency is more predictable than AllGather
2. **MoEDispatch AlltoAll** can overlap with the previous layer's **RowLinear ReduceScatter**
3. **MoECombine AlltoAll** can overlap with the next layer's **QKV AllGather**

```
Timeline ->
ubatch0(stream1): [QKV_AG]-[Attn]-[RowRS]-record(ATTN_POST)-[Dispatch_A2A]-record(MOE_DISPATCH)-[Expert]-[Combine_A2A]-record(ATTN_PRE)-[Finalize_RS]-...
ubatch1(stream2):                    wait(ATTN_POST)+yield-[QKV_AG]-[Attn]-[RowRS]-wait(MOE_DISPATCH)+yield-[Dispatch_A2A]-...
                                                                                                                               wait(ATTN_PRE)+yield-[QKV_AG]-...
```

**Strategy**:
- `dbo_linear_row_hook(record=True)`: record `ATTN_POST` before o_proj RowLinear ReduceScatter
- `dbo_linear_row_hook(record=False)`: wait+yield after ReduceScatter (`wait=False`, only yield without waiting, letting ubatch1 start QKV as soon as possible)
- `dbo_moe_prepare_hook(record=True)`: record `MOE_DISPATCH` after MoEDispatch AlltoAll completes
- `dbo_moe_prepare_hook(record=False)`: wait `MOE_DISPATCH` + yield before MoEDispatch AlltoAll
- `dbo_moe_finalize_hook(record=True)`: record `ATTN_PRE` after MoECombine AlltoAll completes (for next layer's QKV to wait on)
- `dbo_linear_column_hook(record=False)`: wait `ATTN_PRE` + yield before QKV AllGather (wait for previous layer's MoECombine to complete)

**Note**: In A3 AlltoAll mode, `wait=False` in `dbo_linear_row_hook` is a key optimization --
ubatch1 does not need to wait for ubatch0's ReduceScatter to complete before starting QKV AllGather.
The two can truly run in parallel; only a CPU yield to hand over thread control is needed.

---

## III. MLA+MoE Model (e.g., DeepSeek V2/V3)

### Per-Layer Communication Sequence (A2 AllGather Mode)

```
MLAPreprocess
  +-- AllGather (TP, aggregate q_c and kv_no_split)
MLA compute
RowLinear(o_proj)
  +-- ReduceScatter (TP)
MoEPrepare
  +-- AllGather (EP)
Expert compute
MoEFinalize
  +-- ReduceScatter (EP)
```

**Reference implementation**: `deepseek.py` -> `DeepseekAllgatherTemplate`

### Per-Layer Communication Sequence (A3 AlltoAll Mode)

```
MLAPreprocess
  +-- AllGather (TP, q_c + kv_no_split)
MLA compute
RowLinear(o_proj)
  +-- ReduceScatter (TP)
MoEDispatch
  +-- AlltoAll (EP)
Expert compute
MoECombine
  +-- AlltoAll (EP)
MoEFinalize
  +-- ReduceScatter (EP/TP)
```

### DBO Overlap Strategy (A3 AlltoAll)

```
Timeline ->
ubatch0(stream1): [MLA_AG]-[MLA]-[RowRS]-record(ATTN_POST)-[Dispatch_A2A]-record(MOE_DISPATCH)-[Expert]-[Combine_A2A]-record(ATTN_PRE)-[Finalize_RS]-...
ubatch1(stream2):                   wait(ATTN_POST)+yield-[MLA_AG]-[MLA]-[RowRS]-wait(MOE_DISPATCH)+yield-[Dispatch_A2A]-...
                                                                                                              wait(ATTN_PRE)+yield-[MLA_AG]-...
```

**Strategy**:
- `dbo_mla_preprocess_hook(record=True)`: First layer records `ATTN_PRE` (`dbo_first_layer_sync` protected)
- `dbo_mla_preprocess_hook(record=False)`: wait `ATTN_PRE` + yield before MLA AllGather
- `dbo_linear_row_hook(record=True)`: record `ATTN_POST` before o_proj ReduceScatter
- `dbo_linear_row_hook(record=False)`: wait+yield after ReduceScatter (`wait=False`)
- `dbo_moe_prepare_hook(record=True)`: record `MOE_DISPATCH` after MoEDispatch AlltoAll completes
- `dbo_moe_prepare_hook(record=False)`: wait `MOE_DISPATCH` + yield before MoEDispatch AlltoAll
- `dbo_moe_finalize_hook(record=True)`: record `ATTN_PRE` after MoECombine AlltoAll completes (for next layer's MLA to wait on)

**Reference implementation**: `deepseek.py` -> `DeepseekAlltoallTemplate`

---

## IV. First Layer Special Handling (dbo_first_layer_sync)

The first decoder layer has no MoEFinalize/MoECombine event from the "previous layer" to wait on.
Therefore, any hook that needs to wait for the `ATTN_PRE` event requires special handling at the first layer:

```python
# First layer: ubatch0 proactively records, marking "I'm ready"
if get_forward_context().dbo_first_layer_sync:
    dbo_record_current_stream(event=UBatchEventKey.ATTN_PRE)
    get_forward_context().dbo_first_layer_sync = False
```

Scenarios requiring `dbo_first_layer_sync` protection:
- `dbo_linear_column_hook(record=True)` or `dbo_mla_preprocess_hook(record=True)` in MoE models
- In other words: any hook that "waits for the previous layer's MoEFinalize/MoECombine to complete" needs to self-record at the first layer to unblock the wait

---

## V. Mapping Between Profiling Results and Communication Flow

This section explains how to map the hook call sequences collected via `references/profiling-guide.md` back to the communication flow described in this document, thereby verifying the correctness of static analysis.

### 5.1 Hook Call Sequence -> Communication Flow Mapping Table

| Hook sequence observed in profiling | Corresponding communication flow node | Document section |
|-------------------------------------|--------------------------------------|-----------------|
| `dbo_linear_column_hook(record)` | **Before** ColumnLinear AllGather | I. Dense Model |
| `dbo_linear_column_hook(wait)` | **After** ColumnLinear AllGather | I. Dense Model |
| `dbo_linear_row_hook(record)` | **Before** RowLinear ReduceScatter | I. Dense Model |
| `dbo_linear_row_hook(wait)` | **After** RowLinear ReduceScatter (A3 wait=False) | II. MoE Model |
| `dbo_mla_preprocess_hook(record)` | **Before** MLA AllGather (first layer records ATTN_PRE) | III. MLA+MoE Model |
| `dbo_mla_preprocess_hook(wait)` | **After** MLA AllGather (wait ATTN_PRE) | III. MLA+MoE Model |
| `dbo_moe_prepare_hook(wait)` | **Before** MoEPrepare AllGather (A2, wait ATTN_POST) | II. MoE Model A2 |
| `dbo_moe_prepare_hook(record)` | **After** MoEDispatch AlltoAll (A3, record MOE_DISPATCH) | II. MoE Model A3 |
| `dbo_moe_prepare_hook(wait)` | **Before** MoEDispatch AlltoAll (A3, wait MOE_DISPATCH) | II. MoE Model A3 |
| `dbo_moe_finalize_hook(record)` | **After** MoEFinalize/MoECombine (record ATTN_PRE) | II. MoE Model |

### 5.2 Verifying Communication Mode (A2 vs A3) via Profiling

**A2 AllGather mode profiling characteristics**:
```
Per-layer hook sequence (MoE model):
  dbo_linear_column_hook(record)   <- Before QKV AllGather
  dbo_linear_column_hook(wait)     <- After QKV AllGather
  dbo_linear_row_hook(record)      <- Before o_proj RS (record ATTN_POST)
  <- Note: dbo_linear_row_hook(wait) does NOT appear (A2 has no yield here)
  dbo_moe_prepare_hook(wait)       <- Before MoEPrepare (wait ATTN_POST)
  <- Note: dbo_moe_prepare_hook(record) does NOT appear (A2 has no AlltoAll)
  dbo_moe_finalize_hook(record)    <- Before MoEFinalize (record ATTN_PRE)
```

**A3 AlltoAll mode profiling characteristics**:
```
Per-layer hook sequence (MoE model):
  dbo_linear_column_hook(record)   <- Before QKV AllGather
  dbo_linear_column_hook(wait)     <- After QKV AllGather
  dbo_linear_row_hook(record)      <- Before o_proj RS (record ATTN_POST)
  dbo_linear_row_hook(wait)        <- After o_proj RS (wait=False, yield only) <- A3 specific
  dbo_moe_prepare_hook(record)     <- After Dispatch AlltoAll (record MOE_DISPATCH) <- A3 specific
  dbo_moe_prepare_hook(wait)       <- Before Dispatch AlltoAll (wait MOE_DISPATCH)
  dbo_moe_finalize_hook(record)    <- After Combine AlltoAll (record ATTN_PRE)
```

**Key distinguishing points**:
- `dbo_moe_prepare_hook(record)` **appears** -> A3 AlltoAll mode
- `dbo_moe_prepare_hook(record)` **does not appear** -> A2 AllGather mode
- `dbo_linear_row_hook(wait)` **appears** -> A3 AlltoAll mode (wait=False yield)

### 5.3 Profiling Timing and Overlap Benefit Estimation

After measuring communication operator timing via profiling, the theoretical DBO overlap benefit can be estimated:

```
Theoretical speedup = 1 / (1 - overlap_ratio)

Where:
overlap_ratio = min(comm_time, compute_time) / (comm_time + compute_time)
```

**Practical estimation example** (MoE model, A3 AlltoAll):
```
Assuming profiling measurements:
  MoE Dispatch AlltoAll: 1200us
  Expert Compute: 800us
  MoE Combine AlltoAll: 1200us
  QKV AllGather: 400us
  Attention Compute: 600us
  o_proj ReduceScatter: 400us

DBO overlap effect (two ubatches interleaved):
  ubatch0 Dispatch AlltoAll (1200us) parallel with ubatch1 Attention Compute (600us)
  -> Saves ~600us

  ubatch0 Expert Compute (800us) parallel with ubatch1 Dispatch AlltoAll (1200us)
  -> Saves ~800us

  ubatch0 Combine AlltoAll (1200us) parallel with ubatch1 Expert Compute (800us)
  -> Saves ~800us

Total savings: ~2200us / layer (compared to serial execution)
```

### 5.4 Handling Anomalies Discovered During Profiling

| Profiling anomaly | Corresponding communication flow issue | Resolution |
|-------------------|---------------------------------------|------------|
| Hook call count per layer > expected | Model has multiple sub-layers of the same type (e.g., multiple MLPs in a Dense layer) | Add a call counter in the hook, only record/wait at specific counts |
| Hook call order differs from this document | Model forward order is special (e.g., post-norm, parallel attention) | Re-read the model forward(), update the communication flow diagram |
| `dbo_first_layer_sync` not triggered | First layer hook not handled correctly | Check the `dbo_first_layer_sync` condition in the template |
| Gap between hook calls across layers | Extra computation between layers (e.g., LayerNorm, residual connection) | Normal behavior, does not affect overlap strategy |
