# SparseAttentionScore 算子设计文档

## 1. 算子功能概述

`SparseAttentionScore` 是面向 **Paged KV Cache + TopK Block Selection** 场景的稀疏注意力算子。它在 LLM 推理的 decode 阶段（以及小 batch prefill），根据预先选好的 TopK KV block indices，从 paged KV cache 中读取相关 block，完成 attention 计算。

### 核心计算

```text
O = softmax(Q @ K^T / sqrt(d)) @ V
```

其中 Q 只关注 TopK 个 KV block（而非全部 KV），实现 **稀疏注意力**。

### 与 BlockSparseAttention (BSA) 的对比

| 维度 | SparseAttentionScore (SASA) | BlockSparseAttention (BSA) |
|------|---------------------------|---------------------------|
| **稀疏模式输入** | `select_idx` + `select_num_idx`（预计算的 TopK block 编号列表） | `block_sparse_mask`（二维 0/1 mask 矩阵，kernel 内部转为 idx） |
| **KV 存储格式** | Paged KV Cache: `[num_physical_blocks, block_size, kv_heads, D]` | 连续 KV: TND / BNSD / BSND |
| **Q 格式** | TND: `[total_tokens, num_heads, D]` | TND / BNSD / BSND |
| **地址映射** | `block_table[batch, logical_id] → physical_id` | 直接按 seq offset 连续访问 |
| **Task 粒度** | 1 token × 1 KV head group（group-head 优化后） | 1 Q tile × 1 Q head |
| **GQA 处理** | 同 group 共享 selectIdx，一次搬 KV 服务 groupSize 个 head | 每个 Q head 独立 task，不做 group 合并 |
| **block_size** | 固定 128（paged cache 物理块） | 可配置 blockShapeX / blockShapeY |
| **适用场景** | vLLM 推理 decode、长上下文稀疏推理 | 训练/推理通用稀疏 attention |
| **workspace** | 不需要 mask→idx 转换 | 需要 workspace 存 sparse_idx + sparse_count |

## 2. 输入输出接口

### 输入

| 参数 | 形状 | 说明 |
|------|------|------|
| `query` | `[T, N_q, D]` (TND) | Q tensor，bf16/fp16/fp8 |
| `key` | `[num_blocks, block_size, N_kv, D]` | Paged KV cache K |
| `value` | `[num_blocks, block_size, N_kv, D]` | Paged KV cache V |
| `select_idx` | `[N_kv, max_q_seqlen, top_k]` | 每个 kv_head、每个 q_token 的 TopK logical block IDs |
| `block_table` | `[batch, max_blocks_per_batch]` | logical → physical block 映射 |
| `select_num_idx` | `[N_kv, max_q_seqlen]` | 每个 token 实际有效的 block 数 |
| `actual_seq_lengths` | `[batch]` | 每个 batch 的 Q seqlen |
| `actual_seq_lengths_kv` | `[batch]` | 每个 batch 的 KV seqlen |

### Attributes

| 属性 | 说明 |
|------|------|
| `num_key_value_heads` | KV head 数 |
| `scale_value` | softmax scale（默认 1/sqrt(D)）|
| `block_size` | paged KV cache 块大小（128）|
| `top_k` | 最大选取的 block 数 |
| `inner_precise` | 精度模式 |

### 输出

| 参数 | 形状 | 说明 |
|------|------|------|
| `output` | `[T, N_q, D]` (TND) | 注意力输出，与 Q 同 shape |

## 3. 适配逻辑（Host Tiling）

### Task 分解

```text
totalTaskNum = totalQTokens × kvHeads
blockDim = min(totalTaskNum, aicNum)
```

每个 task 处理 1 个 Q token 的 1 个 KV head group（包含 groupSize 个 Q heads）。

### Tiling 数据

Host 侧计算并传递给 kernel 的 tiling 数据包括：

- **基础形状**: batch, numHeads, kvHeads, embeddingSize, blockSize, topK
- **groupSize**: numHeads / kvHeads（GQA group 大小）
- **task 信息**: totalTaskNum, firstBatchTaskNum
- **tile 大小**: qBaseTile=128, kvBaseTile=128
- **L1 matmul tile**: mm1/mm2 的 M/N/K 配置
- **buffer 数量**: Q/K/V/P 的 L1 buffer 个数

## 4. Kernel 实现方案

### 4.1 总体流水线

```text
┌─────────────────────────────────────────────────────────────┐
│  Per Task: 1 token × groupSize heads × topK KV blocks      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Load Q   │    │ QK MMAD  │    │ Softmax  │              │
│  │ (once)   │───▶│ (Cube)   │───▶│ (Vector) │──┐           │
│  └──────────┘    └──────────┘    └──────────┘  │           │
│                                                 ▼           │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Load V   │    │ PV MMAD  │    │RescaleO  │              │
│  │ (per blk)│───▶│ (Cube)   │───▶│ (Vector) │──▶ Store O   │
│  └──────────┘    └──────────┘    └──────────┘              │
│                                                             │
│  Pipeline: QK[i] → SM[i] → PV[i] → Rescale[i] (PRE=2)    │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Task 分解与 KV 复用

```cpp
// 每个 task 对应一个 (token, kvHead) 对
uint32_t qToken = taskIdx / kvHeads_;
uint32_t kvHeadIdx = taskIdx % kvHeads_;
uint32_t qHeadStart = kvHeadIdx * groupSize;  // 组内首个 Q head

// Q offset: 连续读取 groupSize 个 heads
int64_t gmOffsetQ = qToken * strideQO + qHeadStart * embed_;
```

**Group-Head KV Reuse 优化**：同一 group 内的 groupSize 个 Q heads 共享相同的 selectIdx（因为按 kvHead 索引）。优化后一次加载 KV block 即可服务组内所有 heads，减少 groupSize 倍的 KV 搬运。

### 4.3 Matmul 维度

```text
QK: M=groupSize, N=kvBlockSize(≤128), K=headDim(128)
     Q[groupSize, D] × K[D, blockSize]^T → S[groupSize, blockSize]

PV: M=groupSize, N=headDim(128), K=kvBlockSize(≤128)
     P[groupSize, blockSize] × V[blockSize, D] → OTmp[groupSize, D]
```

### 4.4 Paged KV Cache 地址计算

```cpp
// KV 存储: [physical_block_id, block_size, kv_heads, D]
// 每行的 stride = kv_heads * D
// 每 block 的 stride = block_size * kv_heads * D
int64_t gmOffsetK = physicalBlockId * strideKVBlock + kvHeadIdx * embed_;
```

通过 `block_table` 将 `logical_id`（selectIdx 中的值）转为 `physical_id`，实现 paged KV cache 的地址翻译。

### 4.5 Online Softmax（逐 block 迭代更新）

对于 topK 个 KV block 逐一处理，使用 online softmax：

```text
for each KV block:
    S = Q × K^T (bf16 matmul)
    S_scaled = S * scale (bf16)
    nowMax = row_max(S_scaled) (per-head 独立)
    if not first: nowMax = max(nowMax, cast_bf16(lastMax))
    P = exp(S_scaled - nowMax) (bf16)
    nowSum = reduce_sum(P) (bf16)
    
    update lastMax/lastSum (fp32):
        correction = exp(lastMax - nowMax)
        lastSum = correction * lastSum + nowSum
        lastMax = nowMax
    
    PV = P × V (bf16→fp32 accumulate)
    o_acc = correction * o_acc + PV (fp32)

output = cast_bf16(o_acc / lastSum)
```

### 4.6 Partial Last Block 处理

最后一个 causal block 可能不满 block_size：

```cpp
uint32_t lastLogicalBlockId = (historyLen + qTokenInBatch) / blockSize_;
uint32_t lastBlockTileSize = (historyLen + qTokenInBatch) % blockSize_ + 1;
validTileSize[i] = (logicalId == lastLogicalBlockId) ? lastBlockTileSize : blockSize_;
```

通过 `kvSTileSizeAct = validTileSize[kvBlockIdx]` 传给 matmul，确保只计算有效 KV 行。

### 4.7 Cube/Vector 双核协同

- **Cube Core (AIC)**: 执行 QK 和 PV matmul，通过 FixPipe 将结果从 L0C 写到 UB
- **Vector Core (AIV)**: 执行 softmax（scale、max、exp、sum）和 rescaleO（correction、div、cast）
- **Cross-core 同步**: `SetFlag`/`WaitFlag` + `PipeBarrier` 实现 Cube→Vector→Cube 流水

### 4.8 L0 Buffer 流水管理

QK 和 PV matmul 交替使用 L0A/L0B buffer。通过 `prefixSumL0AStages` 计算确保 buffer ID 不冲突：

```cpp
uint32_t mL0Loop = CeilDiv(groupSize, L0_TILE_M);  // = 1 for groupSize≤16
mm1L0ATotalStages = mL0Loop * (embed / L0_TILE_K);
mm2L0ATotalStages = mL0Loop * (kvBaseTile / L0_TILE_K);
```

## 5. 与 BSA 的关键实现差异

### 5.1 KV 数据加载

| | SASA | BSA |
|---|------|-----|
| **K 加载** | 逐 physical block 独立加载（通过 block_table 翻译地址） | gather 连续的 sparse block（workspace 中预排的 idx） |
| **blockMmadQK 的 sparse 参数** | `gatheredKvSTileIdx=0, yBlockNum=1`（每次只处理 1 个 block） | `gatheredKvSTileIdx, yBlockNumRsvd`（gather 多个 block） |
| **V 加载** | 同 K，逐 physical block | 同 BSA 的 sparse gather |

### 5.2 Q/O 内存布局

| | SASA (group-head 优化后) | BSA |
|---|------|-----|
| **Q GM stride** | `embed_`（group 内 heads 连续） | `strideQO`（可能 numHeads*D 或 BNSD stride） |
| **O GM stride** | `embed_`（同 Q） | `strideQO`（同 Q） |
| **rowNum** | `groupSize`（如 4/8） | `qSTileSizeAct`（如 128） |

### 5.3 Sparse 模式表达

- **BSA**: 输入是 `block_sparse_mask[B, N_q, X_blocks, Y_blocks]`（uint8 bitmap）。Kernel 先在 Vector core 上做 mask→idx 转换（`EpilogueMask2Idx`），再用 idx 做 gather KV。
- **SASA**: 输入直接是 idx 列表 `select_idx[N_kv, Q_seqlen, topK]` + count `select_num_idx[N_kv, Q_seqlen]`。无需 workspace 做转换。

### 5.4 GQA 处理策略

- **BSA**: 每个 Q head 有独立的 sparse pattern（因为 `block_sparse_mask` 按 `qHeadIdx` 索引），所以即使是 GQA 也不做 group 合并：`rowNum = qSTileSizeAct`，不做 head 聚合。
- **SASA**: `select_idx` 按 `kvHeadIdx` 索引（同 group 所有 Q heads 共享），天然支持 group 合并。优化后 `rowNum = groupSize`，KV 只搬一次。

## 6. 性能特征

### 搬运量对比（单 token decode, groupSize=4, topK=8, D=128, blockSize=128）

**优化前（per-head task）**：

- Q 搬运: 4 次 × 128 elements = 512 bf16 = 1KB
- KV 搬运: 4 × 8 blocks × 128×128 × 2 dtype = 4 × 256KB = **1024KB**

**优化后（per-group task）**：

- Q 搬运: 1 次 × 4×128 elements = 512 bf16 = 1KB
- KV 搬运: 1 × 8 blocks × 128×128 × 2 dtype = **256KB**

KV 搬运减少 **4×**（= groupSize 倍），这是长 KV cache 场景下的主要性能瓶颈。

## 7. inner_precise 精度模式

### A5 (Ascend 950) 支持的模式

SparseAttentionScore 在 A5 (Ascend 950PR/950DT) 上 **仅支持 `inner_precise=4`**（默认值），对应混合精度模式 `LOW_HIGH_MIXED`。

| inner_precise | 含义 | A5 支持 | A2/A3 支持 |
|:---:|------|:---:|:---:|
| 0 | `ALL_HIGH` — online softmax 和 rescaleO 全部采用 fp32 | 不支持 | 支持 |
| 1 | `ALL_LOW` — online softmax 和 rescaleO 全部采用 fp16（仅 FP16 输入） | 不支持 | 支持 |
| 4 | `LOW_HIGH_MIXED` — online softmax 采用低精度（bf16/fp16），rescaleO 采用 fp32 | **支持** | 不支持 |

### 模式差异

**inner_precise=4 (LOW_HIGH_MIXED)**：A5 的唯一模式，也是性能和精度的折中方案。

具体计算精度分配如下：

```text
┌────────────────────────────────────────────────────────────────┐
│ Stage            │ 计算精度          │ 存储精度               │
├────────────────────────────────────────────────────────────────┤
│ QK matmul        │ bf16×bf16→fp32累加 │ FixPipe→UB (bf16)     │
│ Online Softmax   │ bf16              │ P写入L1 (bf16/zN)      │
│  - scale/max/exp │ bf16 (低精度)     │                        │
│  - sum           │ bf16 (低精度)     │                        │
│ PV matmul        │ bf16×bf16→fp32累加 │ FixPipe→UB (fp32)     │
│ RescaleO         │ fp32 (高精度)     │ O输出 (bf16)           │
│  - correction    │ fp32              │                        │
│  - accumulate    │ fp32              │                        │
│  - final div     │ fp32→bf16 cast    │                        │
└────────────────────────────────────────────────────────────────┘
```

**与 ALL_HIGH (mode 0) 的差异**：

- mode=0 时 softmax 阶段的 max/exp/sum 也在 fp32 下计算，精度更高但需要 fp32 中间存储（L1 占用翻倍）和额外的 cast 指令
- mode=4 时 softmax 在 bf16 下完成，P 以 bf16 格式存入 L1（节省 L1 空间），但 exp 近似精度受限于 bf16 的 7-bit 尾数

**与 ALL_LOW (mode 1) 的差异**：

- mode=1 时 rescaleO 也在 fp16 下执行，长序列多次 correction 累乘后精度退化严重
- mode=4 的 rescaleO 使用 fp32 累积，在 online softmax 迭代次数多（topK 大）时仍能保持最终输出精度

### A5 选择 mode=4 的原因

1. **硬件适配**：A5 Cube 核的 FixPipe 输出到 UB 时对 fp32 intermediate 有带宽优势，可以高效做 bf16→fp32 的 PV 累积
2. **L1 效率**：P 以 bf16 存 L1（zN layout），相比 fp32 节省一半 L1 空间，允许更多的 double-buffering stages
3. **精度平衡**：softmax 的 exp 近似在 bf16 下引入 ~1-2 ULP 误差，但 rescaleO 用 fp32 累积保证最终 O 不会因多次迭代而精度雪崩

### 精度影响

在 `inner_precise=4` 下，典型精度表现：

- QKV 值域 [-1, 1]，`max_diff` 通常 < 4e-3，`mean_diff` < 5e-4
- 长序列（topK≥6）时，softmax 的 bf16 exp 累积误差可能使 `max_diff` 达到 ~1e-2
- 对比双精度 golden（strict bf16 模拟），relative error < 1%

## 8. 精度模型

BF16 路径 (`inner_precise=4`) 的精度链路：

1. QK matmul: `bf16 × bf16 → fp32 累加 → FixPipe cast bf16`
2. Softmax: `bf16 scale → bf16 max/sub → bf16 exp → bf16 sum`（per-row 独立）
3. PV matmul: `bf16 × bf16 → fp32 累加`（OTmp 保持 fp32）
4. RescaleO: `fp32 correction × fp32 o_acc + fp32 pv`
5. 最终: `fp32 / fp32 → cast bf16`

典型精度：relative error < 1%（hardware exp 近似 + bf16 截断累积）。
