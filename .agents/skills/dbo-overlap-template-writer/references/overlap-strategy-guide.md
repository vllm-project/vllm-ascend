# DBO Overlap Strategy Planning Guide

This document guides how to plan DBO overlap strategies for new models in both A2 (AllGather) and A3 (AlltoAll) modes.

## 1. Core Principles

### 1.1 The Essence of Overlap

DBO overlap enables **ubatch0's communication** and **ubatch1's computation** to execute simultaneously (or vice versa). Two ubatches run on different NPU streams, achieving true parallelism through CPU thread interleaving.

### 1.2 Two Core Communication Scheduling Principles

When designing DBO overlap templates, `record`/`wait` placement must follow these two core principles:

#### Principle 1: Consecutive Communication Merging — Schedule consecutive communications or those with only minimal computation between them as a continuous block

**Core idea**: If two communication operators have only minimal computation between them (e.g., LayerNorm, residual addition, activation functions, typically < 100μs), do not insert `record`/`wait` switch points between them. Instead, treat them as a single communication block with `record`/`wait` placed at the block boundaries.

**Rationale**:
- `record`/`wait` + CPU yield has overhead (~10-50μs). If intermediate computation is shorter than this overhead, switching reduces efficiency
- Consecutive communications can pipeline on NPU; inserting synchronization points breaks the pipeline
- Merging consecutive communications into one block enables overlap with longer computation on the other side, improving efficiency

**Judgment criteria**:
```
Computation duration between two communication operators T_compute_between:
  T_compute_between < 200μs  → Recommend merging (no record/wait insertion)
  T_compute_between > 500μs  → Do not merge (sufficient computation for overlap)
```

**Typical merge scenarios**:
```
Scenario 1: o_proj ReduceScatter + MoE Prepare AllGather (A2 mode)
  After o_proj RS → LayerNorm (~30μs) → MoE routing (~50μs) → MoE Prepare AG
  Intermediate computation ~80μs, far less than communication duration → Merge into one block
  ✅ Correct: record(ATTN_POST) → [o_proj RS] → [LayerNorm] → [MoE routing] → [MoE Prepare AG] → wait(ATTN_POST)
  ❌ Wrong: record → [o_proj RS] → wait → record → [MoE Prepare AG] → wait

Scenario 2: MoE Combine AlltoAll + MoE Finalize ReduceScatter (A3 mode)
  After Combine A2A → token unpermute (~40μs) → Finalize RS
  Intermediate computation minimal → Merge
  ✅ Correct: record(ATTN_PRE) → [Combine A2A] → [token unpermute] → [Finalize RS]
  ❌ Wrong: Insert record/wait between Combine and Finalize

Scenario 3: QKV AllGather + Attention computation (do not merge)
  After QKV AG → Attention computation (~2000μs)
  Intermediate computation long → Do not merge, Attention can overlap with counterpart's communication
```

**Manifestation in existing templates**:
- `QwenMoEAllgatherTemplate`: `dbo_linear_row_hook(record=True)` records before o_proj RS starts, `dbo_moe_prepare_hook(wait)` waits after MoE Prepare AG completes. The o_proj RS + MoE Prepare AG between them is merged into a consecutive communication block.
- `DeepseekAlltoallTemplate`: `dbo_moe_finalize_hook(record=True)` records before Combine AlltoAll starts, no separate wait after Finalize RS. Combine + Finalize merged into a consecutive block.

#### Principle 2: Compute-Communication Balance — Make overlapped communication and computation durations as close as possible

**Core idea**: `record`/`wait` switch point placement should make "overlapped communication time" as close as possible to "counterpart ubatch's computation time", maximizing overlap efficiency.

**Rationale**:
- If communication time >> computation time: Large portion of communication un-overlapped, wasting overlap opportunities
- If computation time >> communication time: Large portion of computation waiting for communication completion, NPU utilization drops
- Ideal case: Communication time ≈ computation time, both fully overlapped, total time ≈ max(communication, computation)

**Quantitative metric**:
```
Overlap efficiency = min(T_comm, T_compute) / max(T_comm, T_compute)

  Efficiency > 80%  → Excellent, record/wait placement reasonable
  Efficiency 50-80% → Good, acceptable
  Efficiency < 50%  → Need to adjust record/wait placement
```

**Adjustment strategies**:

```
Case 1: Communication time >> Computation time (communication too long)
  → Consider splitting communication block, overlap with different computation segments
  → Or move more computation into overlap interval (adjust record position to record earlier)

Case 2: Computation time >> Communication time (computation too long)
  → Consider merging more communication into same overlap interval
  → Or delay record position, reducing computation in overlap interval

Case 3: Communication and computation similar (ideal state)
  → Keep current record/wait placement unchanged
```

**Typical analysis example** (MoE model, A3 AlltoAll):

```
Assume profiling measurement results:
  Attention computation:         2000μs
  o_proj ReduceScatter:          400μs
  MoE Dispatch AlltoAll:         1200μs
  Expert computation:            800μs
  MoE Combine AlltoAll:          1200μs
  MoE Finalize RS:               300μs
  QKV AllGather:                 400μs

Analyze balance for each overlap interval:

  Interval 1: ubatch0 [o_proj RS + Dispatch A2A] (1600μs) vs ubatch1 [Attention] (2000μs)
    Efficiency = 1600/2000 = 80% ✅ Good

  Interval 2: ubatch0 [Expert computation] (800μs) vs ubatch1 [Dispatch A2A] (1200μs)
    Efficiency = 800/1200 = 67% ⚠️ Acceptable, but communication long

  Interval 3: ubatch0 [Combine A2A + Finalize RS] (1500μs) vs ubatch1 [Expert computation] (800μs)
    Efficiency = 800/1500 = 53% ⚠️ Communication significantly long, consider adjustment

  Possible optimization: Move Finalize RS out of interval 3, merge with next layer's QKV AG
    Adjusted interval 3: ubatch0 [Combine A2A] (1200μs) vs ubatch1 [Expert computation] (800μs)
    Efficiency = 800/1200 = 67% → Slight improvement
    New interval 4: ubatch0 [Finalize RS + next layer QKV AG] (700μs) vs ubatch1 [Combine A2A] (1200μs)
    → Need comprehensive evaluation of overall effect
```

#### Synergistic Application of Both Principles

In actual template design, both principles must be considered together:

1. **First apply Principle 1**: Identify which communication operators should be merged into consecutive blocks
2. **Then apply Principle 2**: Determine which computation segment each merged communication block overlaps with, adjust `record`/`wait` placement to balance durations
3. **Iterate**: If Principle 2 analysis shows an overlap interval has low efficiency, return to Principle 1 to reconsider communication block partitioning

**Decision flow**:
```
Step 1: List all communication operators and their durations per layer
Step 2: Mark computation durations between adjacent communications
Step 3: Apply Principle 1, merge adjacent communications with < 200μs intermediate computation into blocks
Step 4: Pair communication blocks with counterpart ubatch's computation segments
Step 5: Apply Principle 2, calculate overlap efficiency for each pair
Step 6: If efficiency < 50%, try adjusting communication block partitioning or record/wait placement
Step 7: Determine final record/wait placement scheme
```

### 1.3 record/wait Semantics

```
When ubatch0 reaches a communication operator:
  dbo_record_current_stream(event=KEY)
  → Record event on compute_stream, notify ubatch1: "My computation reached here"

When ubatch1 reaches corresponding position:
  dbo_wait_current_stream_and_yield(event=KEY)
  → Wait for ubatch0's event (ensure data dependency satisfied)
  → yield CPU to ubatch0 thread (let ubatch0 continue executing communication)
```

### 1.4 wait=False Usage Scenario

```python
dbo_wait_current_stream_and_yield(event=KEY, wait=False)
```

Used when ubatch1 doesn't need to wait for an event from ubatch0, only needs to yield CPU.
Typical scenario: In A3 AlltoAll mode, ubatch1's QKV AllGather doesn't depend on ubatch0's ReduceScatter result, only needs to yield so ubatch0's AlltoAll can execute on NPU.

---

## 2. Hook Usage Matrix by Model Type

### 2.1 Dense Model

| Hook | A2 AllGather | A3 AlltoAll |
|------|-------------|-------------|
| `dbo_mla_preprocess_hook` | Not used | Not used |
| `dbo_linear_column_hook` | record+wait (DEFAULT) | record+wait (DEFAULT) |
| `dbo_linear_row_hook` | record+wait (DEFAULT) | record+wait (DEFAULT) |
| `dbo_moe_prepare_hook` | Not used | Not used |
| `dbo_moe_finalize_hook` | Not used | Not used |

**Note**: Dense models have identical A2/A3 strategies. AlltoallTemplate can reuse AllgatherTemplate.

### 2.2 MoE Model (no MLA)

| Hook | A2 AllGather | A3 AlltoAll |
|------|-------------|-------------|
| `dbo_mla_preprocess_hook` | Not used | Not used |
| `dbo_linear_column_hook` | record(ATTN_PRE, first layer) + wait(ATTN_PRE) | record(ATTN_PRE, first layer) + wait(ATTN_PRE) |
| `dbo_linear_row_hook` | record(ATTN_POST) | record(ATTN_POST) + wait(ATTN_POST, wait=False) |
| `dbo_moe_prepare_hook` | wait(ATTN_POST) | record(MOE_DISPATCH) + wait(MOE_DISPATCH) |
| `dbo_moe_finalize_hook` | record(ATTN_PRE) | record(ATTN_PRE) |

**Key A3 differences**:
- `dbo_linear_row_hook` adds `wait=False` yield, enabling ubatch1 to start QKV earlier
- `dbo_moe_prepare_hook` records `MOE_DISPATCH` after AlltoAll completes, ubatch1 waits before AlltoAll starts

### 2.3 MLA+MoE Model (DeepSeek series)

| Hook | A2 AllGather | A3 AlltoAll |
|------|-------------|-------------|
| `dbo_mla_preprocess_hook` | record(ATTN_PRE, first layer) + wait(ATTN_PRE) | record(ATTN_PRE, first layer) + wait(ATTN_PRE) |
| `dbo_linear_column_hook` | Not used | Not used |
| `dbo_linear_row_hook` | record(ATTN_POST) | record(ATTN_POST) + wait(ATTN_POST, wait=False) |
| `dbo_moe_prepare_hook` | wait(ATTN_POST) | record(MOE_DISPATCH) + wait(MOE_DISPATCH) |
| `dbo_moe_finalize_hook` | record(ATTN_PRE) | record(ATTN_PRE) |

### 2.4 Hybrid(MLA+QKV)+MoE Model (e.g., Bailing V2.5)

| Hook | A2 AllGather | A3 AlltoAll |
|------|-------------|-------------|
| `dbo_mla_preprocess_hook` | record(ATTN_PRE, first layer) + wait(ATTN_PRE) | record(ATTN_PRE, first layer) + wait(ATTN_PRE) |
| `dbo_linear_column_hook` | record(ATTN_PRE, first layer) + wait(ATTN_PRE) | record(ATTN_PRE, first layer) + wait(ATTN_PRE) |
| `dbo_linear_row_hook` | record(ATTN_POST) | record(ATTN_POST) + wait(ATTN_POST, wait=False) |
| `dbo_moe_prepare_hook` | wait(ATTN_POST) | record(MOE_DISPATCH) + wait(MOE_DISPATCH) |
| `dbo_moe_finalize_hook` | record(ATTN_PRE) | record(ATTN_PRE) |

**Note**: Hybrid models are the only template type implementing both `dbo_mla_preprocess_hook` and `dbo_linear_column_hook`. Both hooks play the same role (record/wait `ATTN_PRE`) in different layer types but never trigger in the same layer.

---

## 3. A3 AlltoAll Mode Special Planning Logic

### 3.1 Why A3 Needs Different Strategy

On A3 chips, MoE uses `npu_moe_distribute_dispatch` / `async_all_to_all` for AlltoAll communication instead of A2's AllGather. AlltoAll has different communication pattern:
- **AllGather**: Each rank broadcasts its data to everyone, traffic = N × local_size
- **AlltoAll**: Each rank sends different data to each other rank, traffic = N × local_size (but more precise routing)

AlltoAll key characteristic: **dispatch and combine are two independent communication operations** that can overlap with different computations separately.

### 3.2 Optimal Overlap Planning for A3 MoE Models

```
Layer L execution timeline (two ubatches interleaved):

ubatch0 (stream1):
  QKV_AllGather → Attn_Compute → RowLinear_ReduceScatter
  → [record ATTN_POST]
  → Dispatch_AlltoAll
  → [record MOE_DISPATCH]
  → Expert_Compute
  → Combine_AlltoAll
  → [record ATTN_PRE]  ← For next layer ubatch1's QKV to wait
  → Finalize_ReduceScatter

ubatch1 (stream2):
  [wait ATTN_POST + yield]  ← Wait for ubatch0's RowLinear completion, yield CPU
  → QKV_AllGather → Attn_Compute → RowLinear_ReduceScatter
  → [wait MOE_DISPATCH + yield]  ← Wait for ubatch0's Dispatch completion, yield CPU
  → Dispatch_AlltoAll
  → [record MOE_DISPATCH]
  → Expert_Compute
  → Combine_AlltoAll
  → [record ATTN_PRE]
  → Finalize_ReduceScatter
  → [wait ATTN_PRE + yield]  ← Wait for ubatch0 next layer's Combine completion (actually waits at next layer's QKV)
```

**Actual overlap effect**:
- ubatch0's `Dispatch_AlltoAll` parallels with ubatch1's `Attn_Compute`
- ubatch0's `Expert_Compute` parallels with ubatch1's `Dispatch_AlltoAll`
- ubatch0's `Combine_AlltoAll` parallels with ubatch1's `Expert_Compute`

### 3.3 Optimal Overlap Planning for A3 MLA+MoE Models

```
ubatch0 (stream1):
  MLA_AllGather → MLA_Compute → RowLinear_ReduceScatter
  → [record ATTN_POST]
  → Dispatch_AlltoAll
  → [record MOE_DISPATCH]
  → Expert_Compute
  → Combine_AlltoAll
  → [record ATTN_PRE]  ← For next layer ubatch1's MLA_AllGather to wait
  → Finalize_ReduceScatter

ubatch1 (stream2):
  [wait ATTN_PRE + yield]  ← Wait for previous layer ubatch0's Combine completion
  → MLA_AllGather → MLA_Compute → RowLinear_ReduceScatter
  → [wait ATTN_POST + yield, wait=False]  ← Only yield, no wait (MLA_AllGather doesn't depend on ubatch0's RS)
  → Dispatch_AlltoAll
  → [wait MOE_DISPATCH + yield]
  → Expert_Compute
  → Combine_AlltoAll
  → [record ATTN_PRE]
  → Finalize_ReduceScatter
```

---

## 4. New Model Planning Decision Tree

```
Input: New model
    │
    ├─ Has MLA (MultiHeadLatentAttention)?
    │   ├─ Yes → Has QKVParallelLinear simultaneously?
    │   │       ├─ Yes → Hybrid(MLA+QKV)+MoE type
    │   │       │       A2: Reference BailingMoEV25AllgatherTemplate
    │   │       │       A3: Reference BailingMoEV25AlltoallTemplate
    │   │       │
    │   │       └─ No → MLA+MoE type
    │   │               A2: Reference DeepseekAllgatherTemplate
    │   │               A3: Reference DeepseekAlltoallTemplate (Section 3.3 in this doc)
    │   │
    │   └─ No → Has FusedMoE/SharedFusedMoE?
    │           ├─ Yes → MoE type
    │           │       A2: Reference QwenMoEAllgatherTemplate
    │           │       A3: Reference Section 3.2 in this doc
    │           │
    │           └─ No → Dense type
    │                   A2/A3: Reference QwenDenseAllgatherTemplate
    │                   (A3 AlltoallTemplate can be empty or reuse AllgatherTemplate)
```

---

## 5. Validation Checklist

After generating template, check each item:

### 5.1 Basic Correctness
- [ ] Each used `EventKey` has exactly one `record` and one `wait` (within same layer)
- [ ] Hooks using `ATTN_PRE` on first layer have `dbo_first_layer_sync` guard
- [ ] `wait=False` only used for "no data dependency wait needed, only yield CPU" scenarios
- [ ] In AlltoAll mode, `dbo_moe_prepare_hook(is_record=True)` calls `record(MOE_DISPATCH)` after AlltoAll completes (after `handle.wait()`)
- [ ] In AlltoAll mode, `dbo_moe_prepare_hook(is_record=False)` calls `wait(MOE_DISPATCH)` before AlltoAll starts
- [ ] Code style consistent with existing templates (concise, no unnecessary comments)

### 5.2 Communication Scheduling Principle Validation (Principle 1: Consecutive Communication Merging)
- [ ] When computation between adjacent communications < 200μs, no `record`/`wait` inserted between them
- [ ] No unnecessary synchronization points inside merged communication blocks
- [ ] Merged communication blocks as a whole have `record` at block start, `wait` at block end
- [ ] Typical merge scenarios handled correctly: o_proj RS + MoE Prepare AG (A2), Combine A2A + Finalize RS (A3)

### 5.3 Compute-Communication Balance Validation (Principle 2: Overlap Efficiency)
- [ ] Communication/computation duration ratio for each overlap interval is between 0.5~2.0
- [ ] If profiling data available, calculated efficiency (min/max) for each interval, no intervals below 50%
- [ ] If any interval efficiency < 50%, tried adjusting `record`/`wait` placement or re-partitioning communication blocks
- [ ] Listed estimated efficiency for each overlap interval in analysis report