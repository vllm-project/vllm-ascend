# DBO Overlap Strategy Planning Guide

This document guides how to plan DBO overlap strategies for new models under both A2 (AllGather) and A3 (AlltoAll) modes.

## I. Core Principles

### 1.1 The Essence of Overlap

DBO's overlap enables **ubatch0's communication** to run simultaneously with **ubatch1's computation** (or vice versa).
The two ubatches run on different NPU streams, achieving true parallelism through alternating CPU thread execution.

### 1.2 record/wait Semantics

```
ubatch0 reaches a communication operator:
  dbo_record_current_stream(event=KEY)
  -> Records an event on compute_stream, telling ubatch1: "my computation has reached this point"

ubatch1 reaches the corresponding position:
  dbo_wait_current_stream_and_yield(event=KEY)
  -> Waits for ubatch0's event (ensures data dependency is satisfied)
  -> Yields CPU to ubatch0's thread (lets ubatch0 continue executing communication)
```

### 1.3 Usage Scenarios for wait=False

```python
dbo_wait_current_stream_and_yield(event=KEY, wait=False)
```

Used when ubatch1 does not need to wait for a specific event from ubatch0, but only needs to yield CPU.
Typical scenario: In A3 AlltoAll mode, ubatch1's QKV AllGather does not depend on ubatch0's ReduceScatter result.
It only needs to yield to let ubatch0's AlltoAll execute on the NPU.

---

## II. Hook Usage Matrix by Model Type

### 2.1 Dense Model

| Hook | A2 AllGather | A3 AlltoAll |
|------|-------------|-------------|
| `dbo_mla_preprocess_hook` | Not used | Not used |
| `dbo_linear_column_hook` | record+wait (DEFAULT) | record+wait (DEFAULT) |
| `dbo_linear_row_hook` | record+wait (DEFAULT) | record+wait (DEFAULT) |
| `dbo_moe_prepare_hook` | Not used | Not used |
| `dbo_moe_finalize_hook` | Not used | Not used |

**Note**: Dense model A2/A3 strategies are identical. AlltoallTemplate can reuse AllgatherTemplate.

### 2.2 MoE Model (without MLA)

| Hook | A2 AllGather | A3 AlltoAll |
|------|-------------|-------------|
| `dbo_mla_preprocess_hook` | Not used | Not used |
| `dbo_linear_column_hook` | record(ATTN_PRE, first layer) + wait(ATTN_PRE) | record(ATTN_PRE, first layer) + wait(ATTN_PRE) |
| `dbo_linear_row_hook` | record(ATTN_POST) | record(ATTN_POST) + wait(ATTN_POST, wait=False) |
| `dbo_moe_prepare_hook` | wait(ATTN_POST) | record(MOE_DISPATCH) + wait(MOE_DISPATCH) |
| `dbo_moe_finalize_hook` | record(ATTN_PRE) | record(ATTN_PRE) |

**A3 key differences**:
- `dbo_linear_row_hook` adds `wait=False` yield, letting ubatch1 start QKV as soon as possible
- `dbo_moe_prepare_hook` records `MOE_DISPATCH` **after** AlltoAll completes, ubatch1 waits **before** AlltoAll starts

### 2.3 MLA+MoE Model (DeepSeek Series)

| Hook | A2 AllGather | A3 AlltoAll |
|------|-------------|-------------|
| `dbo_mla_preprocess_hook` | record(ATTN_PRE, first layer) + wait(ATTN_PRE) | record(ATTN_PRE, first layer) + wait(ATTN_PRE) |
| `dbo_linear_column_hook` | Not used | Not used |
| `dbo_linear_row_hook` | record(ATTN_POST) | record(ATTN_POST) + wait(ATTN_POST, wait=False) |
| `dbo_moe_prepare_hook` | wait(ATTN_POST) | record(MOE_DISPATCH) + wait(MOE_DISPATCH) |
| `dbo_moe_finalize_hook` | record(ATTN_PRE) | record(ATTN_PRE) |

---

## III. Special Planning Logic for A3 AlltoAll Mode

### 3.1 Why A3 Needs a Different Strategy

On A3 chips, MoE uses `npu_moe_distribute_dispatch` / `async_all_to_all` for AlltoAll communication,
rather than A2's AllGather. The AlltoAll communication pattern differs:
- **AllGather**: Each rank broadcasts its data to everyone, communication volume = N x local_size
- **AlltoAll**: Each rank sends different data to each other rank, communication volume = N x local_size (but routing is more precise)

The key characteristic of AlltoAll: **dispatch and combine are two independent communication operations** that can overlap with different computations separately.

### 3.2 Optimal Overlap Planning for A3 MoE Models

```
Layer L execution timing (two ubatches interleaved):

ubatch0 (stream1):
  QKV_AllGather -> Attn_Compute -> RowLinear_ReduceScatter
  -> [record ATTN_POST]
  -> Dispatch_AlltoAll
  -> [record MOE_DISPATCH]
  -> Expert_Compute
  -> Combine_AlltoAll
  -> [record ATTN_PRE]  <- For next layer's ubatch1 QKV to wait on
  -> Finalize_ReduceScatter

ubatch1 (stream2):
  [wait ATTN_POST + yield]  <- Wait for ubatch0's RowLinear to complete, yield CPU
  -> QKV_AllGather -> Attn_Compute -> RowLinear_ReduceScatter
  -> [wait MOE_DISPATCH + yield]  <- Wait for ubatch0's Dispatch to complete, yield CPU
  -> Dispatch_AlltoAll
  -> [record MOE_DISPATCH]
  -> Expert_Compute
  -> Combine_AlltoAll
  -> [record ATTN_PRE]
  -> Finalize_ReduceScatter
  -> [wait ATTN_PRE + yield]  <- Wait for ubatch0's next layer Combine to complete (actually waits at next layer's QKV)
```

**Actual overlap effect**:
- ubatch0's `Dispatch_AlltoAll` runs in parallel with ubatch1's `Attn_Compute`
- ubatch0's `Expert_Compute` runs in parallel with ubatch1's `Dispatch_AlltoAll`
- ubatch0's `Combine_AlltoAll` runs in parallel with ubatch1's `Expert_Compute`

### 3.3 Optimal Overlap Planning for A3 MLA+MoE Models

```
ubatch0 (stream1):
  MLA_AllGather -> MLA_Compute -> RowLinear_ReduceScatter
  -> [record ATTN_POST]
  -> Dispatch_AlltoAll
  -> [record MOE_DISPATCH]
  -> Expert_Compute
  -> Combine_AlltoAll
  -> [record ATTN_PRE]  <- For next layer's ubatch1 MLA_AllGather to wait on
  -> Finalize_ReduceScatter

ubatch1 (stream2):
  [wait ATTN_PRE + yield]  <- Wait for previous layer's ubatch0 Combine to complete
  -> MLA_AllGather -> MLA_Compute -> RowLinear_ReduceScatter
  -> [wait ATTN_POST + yield, wait=False]  <- Only yield, no wait (MLA_AllGather doesn't depend on ubatch0's RS)
  -> Dispatch_AlltoAll
  -> [wait MOE_DISPATCH + yield]
  -> Expert_Compute
  -> Combine_AlltoAll
  -> [record ATTN_PRE]
  -> Finalize_ReduceScatter
```

---

## IV. Decision Tree for New Model Planning

```
Input: New model
    |
    +-- Does it have MLA (MultiHeadLatentAttention)?
    |   +-- Yes -> MLA+MoE type
    |   |       A2: Reference DeepseekAllgatherTemplate
    |   |       A3: Reference DeepseekAlltoallTemplate (Section 3.3 of this document)
    |   |
    |   +-- No -> Does it have FusedMoE/SharedFusedMoE?
    |           +-- Yes -> MoE type
    |           |       A2: Reference QwenMoEAllgatherTemplate
    |           |       A3: Reference Section 3.2 strategy of this document
    |           |
    |           +-- No -> Dense type
    |                   A2/A3: Reference QwenDenseAllgatherTemplate
    |                   (A3 AlltoallTemplate can be empty or reuse AllgatherTemplate)
```

---

## V. Verification Checklist

After generating the template, check each item:

- [ ] Each used `EventKey` has exactly one `record` and one `wait` (within the same layer)
- [ ] Hooks using `ATTN_PRE` at the first layer have `dbo_first_layer_sync` protection
- [ ] `wait=False` is only used in scenarios where "no data dependency wait is needed, only CPU yield"
- [ ] In AlltoAll mode, `dbo_moe_prepare_hook(record=True)` is **after** AlltoAll completes (after `handle.wait()`)
- [ ] In AlltoAll mode, `dbo_moe_prepare_hook(record=False)` is **before** AlltoAll starts
- [ ] Code style is consistent with existing templates (concise, no unnecessary comments)
