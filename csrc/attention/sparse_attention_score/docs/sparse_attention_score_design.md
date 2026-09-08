# SparseAttentionScore Operator Design

## 1. Operator Overview

`SparseAttentionScore` is a sparse attention operator designed for **Paged KV Cache + Top-K Block Selection**. During the decode stage of LLM inference, as well as small-batch prefill, it reads the relevant blocks from a paged KV cache according to preselected Top-K KV block indices and computes attention.

### Core Computation

```text
O = softmax(Q @ K^T / sqrt(d)) @ V
```

Q attends only to the Top-K KV blocks instead of the complete KV cache, thereby implementing **sparse attention**.

### Comparison with BlockSparseAttention (BSA)

| Dimension | SparseAttentionScore (SASA) | BlockSparseAttention (BSA) |
|---|---|---|
| **Sparse-pattern input** | `select_idx` + `select_num_idx` (a precomputed list of Top-K block IDs) | `block_sparse_mask` (a 2D binary mask converted to indices inside the kernel) |
| **KV storage format** | Paged KV Cache: `[num_physical_blocks, block_size, kv_heads, D]` | Contiguous KV: TND / BNSD / BSND |
| **Q format** | TND: `[total_tokens, num_heads, D]` | TND / BNSD / BSND |
| **Address mapping** | `block_table[batch, logical_id]` to `physical_id` | Direct contiguous access by sequence offset |
| **Task granularity** | 1 token x 1 KV head group after group-head optimization | 1 Q tile x 1 Q head |
| **GQA handling** | A group shares `select_idx`; one KV transfer serves `group_size` heads | Each Q head is an independent task; groups are not merged |
| **`block_size`** | Fixed at 128 for physical paged-cache blocks | Configurable through `blockShapeX` / `blockShapeY` |
| **Use cases** | vLLM decode and long-context sparse inference | General sparse attention for training and inference |
| **Workspace** | No mask-to-index conversion required | Requires workspace for `sparse_idx` and `sparse_count` |

## 2. Input and Output Interface

### Inputs

| Parameter | Shape | Description |
|---|---|---|
| `query` | `[T, N_q, D]` (TND) | Q tensor in BF16, FP16, or FP8 |
| `key` | `[num_blocks, block_size, N_kv, D]` | K tensor in the paged KV cache |
| `value` | `[num_blocks, block_size, N_kv, D]` | V tensor in the paged KV cache |
| `select_idx` | `[N_kv, max_q_seqlen, top_k]` | Top-K logical block IDs for every KV head and Q token |
| `block_table` | `[batch, max_blocks_per_batch]` | Logical-to-physical block mapping |
| `select_num_idx` | `[N_kv, max_q_seqlen]` | Actual number of valid blocks for each token |
| `actual_seq_lengths` | `[batch]` | Q sequence length for each batch item |
| `actual_seq_lengths_kv` | `[batch]` | KV sequence length for each batch item |

### Attributes

| Attribute | Description |
|---|---|
| `num_key_value_heads` | Number of KV heads |
| `scale_value` | Softmax scale; defaults to `1 / sqrt(D)` |
| `block_size` | Paged KV cache block size (128) |
| `top_k` | Maximum number of selected blocks |
| `inner_precise` | Precision mode |

### Output

| Parameter | Shape | Description |
|---|---|---|
| `output` | `[T, N_q, D]` (TND) | Attention output with the same shape as Q |

## 3. Adaptation Logic (Host Tiling)

### Task Decomposition

```text
totalTaskNum = totalQTokens x kvHeads
blockDim = min(totalTaskNum, aicNum)
```

Each task processes one KV head group for one Q token. The group contains `groupSize` Q heads.

### Tiling Data

The host calculates the following tiling data and passes it to the kernel:

- **Base shapes**: `batch`, `numHeads`, `kvHeads`, `embeddingSize`, `blockSize`, and `topK`
- **`groupSize`**: `numHeads / kvHeads`, the GQA group size
- **Task information**: `totalTaskNum` and `firstBatchTaskNum`
- **Tile sizes**: `qBaseTile = 128` and `kvBaseTile = 128`
- **L1 matmul tiles**: M/N/K configurations for MM1 and MM2
- **Buffer counts**: numbers of L1 buffers for Q, K, V, and P

## 4. Kernel Implementation

### 4.1 Overall Pipeline

```text
+-------------------------------------------------------------+
|  Per task: 1 token x groupSize heads x Top-K KV blocks    |
+-------------------------------------------------------------+
|                                                             |
|  +----------+    +----------+    +----------+              |
|  | Load Q   |    | QK MMAD  |    | Softmax  |              |
|  | (once)   |--->| (Cube)   |--->| (Vector) |--+           |
|  +----------+    +----------+    +----------+  |           |
|                                                 v           |
|  +----------+    +----------+    +----------+              |
|  | Load V   |    | PV MMAD  |    | RescaleO |              |
|  | (per blk)|--->| (Cube)   |--->| (Vector) |---> Store O  |
|  +----------+    +----------+    +----------+              |
|                                                             |
|  Pipeline: QK[i] -> SM[i] -> PV[i] -> Rescale[i] (PRE=2)   |
+-------------------------------------------------------------+
```

### 4.2 Task Decomposition and KV Reuse

```cpp
// Each task corresponds to one (token, kvHead) pair.
uint32_t qToken = taskIdx / kvHeads_;
uint32_t kvHeadIdx = taskIdx % kvHeads_;
uint32_t qHeadStart = kvHeadIdx * groupSize;  // First Q head in the group.

// Q offset: read groupSize contiguous heads.
int64_t gmOffsetQ = qToken * strideQO + qHeadStart * embed_;
```

**Group-head KV reuse optimization**: All `groupSize` Q heads in a group share the same `select_idx` because it is indexed by KV head. After optimization, loading a KV block once serves all heads in the group, reducing KV transfers by a factor of `groupSize`.

### 4.3 Matmul Dimensions

```text
QK: M=groupSize, N=kvBlockSize (<=128), K=headDim (128)
    Q[groupSize, D] x K[D, blockSize]^T -> S[groupSize, blockSize]

PV: M=groupSize, N=headDim (128), K=kvBlockSize (<=128)
    P[groupSize, blockSize] x V[blockSize, D] -> OTmp[groupSize, D]
```

### 4.4 Paged KV Cache Address Calculation

```cpp
// KV storage: [physical_block_id, block_size, kv_heads, D]
// Per-row stride = kv_heads * D
// Per-block stride = block_size * kv_heads * D
int64_t gmOffsetK = physicalBlockId * strideKVBlock + kvHeadIdx * embed_;
```

`block_table` translates each `logical_id` from `select_idx` into a `physical_id`, providing address translation for the paged KV cache.

### 4.5 Online Softmax (Iterative Block Updates)

The Top-K KV blocks are processed one at a time using online softmax:

```text
for each KV block:
    S = Q x K^T (BF16 matmul)
    S_scaled = S * scale (BF16)
    nowMax = row_max(S_scaled) (independent per head)
    if not first: nowMax = max(nowMax, cast_bf16(lastMax))
    P = exp(S_scaled - nowMax) (BF16)
    nowSum = reduce_sum(P) (BF16)

    update lastMax/lastSum (FP32):
        correction = exp(lastMax - nowMax)
        lastSum = correction * lastSum + nowSum
        lastMax = nowMax

    PV = P x V (BF16 with FP32 accumulation)
    o_acc = correction * o_acc + PV (FP32)

output = cast_bf16(o_acc / lastSum)
```

### 4.6 Handling a Partial Final Block

The final causal block may contain fewer than `block_size` valid tokens:

```cpp
uint32_t lastLogicalBlockId = (historyLen + qTokenInBatch) / blockSize_;
uint32_t lastBlockTileSize = (historyLen + qTokenInBatch) % blockSize_ + 1;
validTileSize[i] = (logicalId == lastLogicalBlockId) ? lastBlockTileSize : blockSize_;
```

Passing `kvSTileSizeAct = validTileSize[kvBlockIdx]` to matmul ensures that only valid KV rows are included in the computation.

### 4.7 Cube/Vector Core Collaboration

- **Cube Core (AIC)**: Performs QK and PV matmuls and writes results from L0C to UB through FixPipe.
- **Vector Core (AIV)**: Performs softmax operations (scale, max, exp, and sum) and `rescaleO` operations (correction, division, and cast).
- **Cross-core synchronization**: `SetFlag`/`WaitFlag` together with `PipeBarrier` implement the Cube-to-Vector-to-Cube pipeline.

### 4.8 L0 Buffer Pipeline Management

QK and PV matmuls alternate between L0A/L0B buffers. `prefixSumL0AStages` prevents buffer ID conflicts:

```cpp
uint32_t mL0Loop = CeilDiv(groupSize, L0_TILE_M);  // 1 when groupSize <= 16.
mm1L0ATotalStages = mL0Loop * (embed / L0_TILE_K);
mm2L0ATotalStages = mL0Loop * (kvBaseTile / L0_TILE_K);
```

## 5. Key Implementation Differences from BSA

### 5.1 KV Data Loading

| | SASA | BSA |
|---|---|---|
| **K loading** | Loads each physical block independently after translating its address through `block_table` | Gathers contiguous sparse blocks using indices prepared in workspace |
| **Sparse arguments to `blockMmadQK`** | `gatheredKvSTileIdx=0, yBlockNum=1` (processes one block at a time) | `gatheredKvSTileIdx, yBlockNumRsvd` (gathers multiple blocks) |
| **V loading** | Same per-physical-block approach as K | Same sparse-gather approach as BSA K loading |

### 5.2 Q/O Memory Layout

| | SASA (after group-head optimization) | BSA |
|---|---|---|
| **Q GM stride** | `embed_` (heads are contiguous within a group) | `strideQO` (possibly `num_heads * D` or a BNSD stride) |
| **O GM stride** | `embed_` (same as Q) | `strideQO` (same as Q) |
| **`rowNum`** | `groupSize` (for example, 4 or 8) | `qSTileSizeAct` (for example, 128) |

### 5.3 Sparse-Pattern Representation

- **BSA**: The input is `block_sparse_mask[B, N_q, X_blocks, Y_blocks]`, a `uint8` bitmap. The kernel first performs mask-to-index conversion with `EpilogueMask2Idx` on the Vector core, then uses the indices to gather KV data.
- **SASA**: The input directly provides the index list `select_idx[N_kv, Q_seqlen, top_k]` and count `select_num_idx[N_kv, Q_seqlen]`, so no conversion workspace is required.

### 5.4 GQA Strategy

- **BSA**: Each Q head has an independent sparse pattern because `block_sparse_mask` is indexed by `qHeadIdx`. Consequently, GQA groups are not merged: `rowNum = qSTileSizeAct` and no head aggregation is performed.
- **SASA**: `select_idx` is indexed by `kvHeadIdx`, so all Q heads in a group naturally share it. After optimization, `rowNum = groupSize` and the KV data is transferred only once.

## 6. Performance Characteristics

### Transfer-Volume Comparison

The following comparison is for single-token decode with `groupSize=4`, `topK=8`, `D=128`, and `blockSize=128`.

**Before optimization (per-head tasks)**:

- Q transfers: 4 x 128 elements = 512 BF16 values = 1 KB
- KV transfers: 4 x 8 blocks x 128 x 128 x 2 (K and V) = 4 x 256 KB = **1,024 KB**

**After optimization (per-group tasks)**:

- Q transfers: 1 x 4 x 128 elements = 512 BF16 values = 1 KB
- KV transfers: 1 x 8 blocks x 128 x 128 x 2 (K and V) = **256 KB**

KV transfer volume is reduced by **4x**, equal to `groupSize`. This is the main performance bottleneck for long KV caches.

## 7. `inner_precise` Precision Modes

### Modes Supported on A5 (Ascend 950)

SparseAttentionScore on A5 (Ascend 950PR/950DT) supports **only `inner_precise=4`**, the default value. It selects the `LOW_HIGH_MIXED` mixed-precision mode.

| `inner_precise` | Meaning | A5 Support | A2/A3 Support |
|:---:|---|:---:|:---:|
| 0 | `ALL_HIGH`: online softmax and `rescaleO` both use FP32 | No | Yes |
| 1 | `ALL_LOW`: online softmax and `rescaleO` both use FP16 (FP16 input only) | No | Yes |
| 4 | `LOW_HIGH_MIXED`: online softmax uses low precision (BF16/FP16), while `rescaleO` uses FP32 | **Yes** | No |

### Differences Between Modes

**`inner_precise=4` (`LOW_HIGH_MIXED`)** is the only A5 mode and balances performance with accuracy.

Its computation and storage precision are assigned as follows:

```text
+-------------------+---------------------------+------------------------+
| Stage             | Computation precision     | Storage precision      |
+-------------------+---------------------------+------------------------+
| QK matmul         | BF16 x BF16, FP32 accum.  | FixPipe -> UB (BF16)   |
| Online softmax    | BF16                      | P in L1 (BF16/zN)      |
|  - scale/max/exp  | BF16 (low precision)      |                        |
|  - sum            | BF16 (low precision)      |                        |
| PV matmul         | BF16 x BF16, FP32 accum.  | FixPipe -> UB (FP32)   |
| RescaleO          | FP32 (high precision)     | O output (BF16)        |
|  - correction     | FP32                      |                        |
|  - accumulation   | FP32                      |                        |
|  - final division | FP32 -> BF16 cast         |                        |
+-------------------+---------------------------+------------------------+
```

**Difference from `ALL_HIGH` (mode 0)**:

- In mode 0, max, exp, and sum in the softmax stage are also computed in FP32. This provides higher accuracy but requires FP32 intermediate storage, doubling L1 usage, as well as additional cast instructions.
- In mode 4, softmax is computed in BF16 and P is stored in L1 as BF16, reducing L1 usage. The approximation accuracy of exp is limited by BF16's 7-bit mantissa.

**Difference from `ALL_LOW` (mode 1)**:

- In mode 1, `rescaleO` also runs in FP16. Precision degrades substantially after repeated correction multiplications over long sequences.
- Mode 4 accumulates `rescaleO` in FP32, preserving final-output accuracy even when online softmax has many iterations because `top_k` is large.

### Why A5 Uses Mode 4

1. **Hardware adaptation**: The A5 Cube core has a bandwidth advantage when FixPipe writes FP32 intermediates to UB, enabling efficient BF16-to-FP32 PV accumulation.
2. **L1 efficiency**: Storing P in BF16 with the zN layout uses half as much L1 space as FP32 and permits more double-buffering stages.
3. **Accuracy balance**: The BF16 exp approximation introduces about 1-2 ULP of error, while FP32 accumulation in `rescaleO` prevents the final O from suffering catastrophic precision degradation over multiple iterations.

### Accuracy Impact

Typical accuracy under `inner_precise=4` is as follows:

- For QKV values in `[-1, 1]`, `max_diff` is usually below `4e-3` and `mean_diff` below `5e-4`.
- For long sequences with `top_k >= 6`, accumulated BF16 softmax exp error may increase `max_diff` to about `1e-2`.
- Relative error compared with a double-precision golden result using strict BF16 simulation is below 1%.

## 8. Precision Model

The BF16 path under `inner_precise=4` uses the following precision pipeline:

1. QK matmul: `BF16 x BF16 -> FP32 accumulation -> FixPipe cast to BF16`
2. Softmax: `BF16 scale -> BF16 max/subtract -> BF16 exp -> BF16 sum`, independently per row
3. PV matmul: `BF16 x BF16 -> FP32 accumulation`, with OTmp retained in FP32
4. `rescaleO`: `FP32 correction x FP32 o_acc + FP32 pv`
5. Final step: `FP32 / FP32 -> cast to BF16`

Typical accuracy is a relative error below 1%, including the hardware exp approximation and accumulated BF16 truncation.
