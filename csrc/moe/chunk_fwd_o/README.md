# ChunkFwdO 算子说明

`ChunkFwdO` 是一个用于分块门控 delta 规则（Chunk Gated Delta Rule）前向传播过程中的自定义算子。该算子根据 Query、Key、Value、Gate 以及前向隐藏状态，计算并输出前向注意力输出张量 `o`。

---

## 1. 算子功能

在分块序列模型中，计算以下张量：

- **o**：前向注意力输出，结合了块内并行计算（intra-chunk）和块间递推计算（inter-chunk）的结果

具体而言，对于每个分块（chunk），算子执行：

1. **块间贡献**：利用前一分块的隐藏状态 `h`，通过 `q * exp(g) @ h` 计算跨块注意力输出
2. **块内贡献**：在当前分块内，通过 `(q @ k^T * mask) @ v` 计算块内因果注意力输出
3. 将两部分贡献相加，得到最终输出 `o`

---

## 2. 接口定义

### 2.1 ACLNN 接口

每个算子分为两段式调用流程：

1. **获取 workspace 与执行器**
   调用 `aclnnChunkFwdOGetWorkspaceSize` 接口，获取算子执行所需的 workspace 大小，并创建执行器（executor）。

2. **执行算子计算**
   调用 `aclnnChunkFwdO` 接口，在指定的 workspace 和执行器下完成计算。

对应以下 C++ 接口（见 `op_host/op_api/aclnn_chunk_fwd_o.h`）：

```cpp
/* function: aclnnChunkFwdOGetWorkspaceSize
 * parameters :
 * q : required
 * k : required
 * v : required
 * h : required
 * g : required
 * cuSeqlensOptional : optional
 * chunkOffsetsOptional : optional
 * scale : required
 * chunkSize : required
 * oOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnChunkFwdOGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *h,
    const aclTensor *g,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkOffsetsOptional,
    double scale,
    int64_t chunkSize,
    const aclTensor *oOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* function: aclnnChunkFwdO
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnChunkFwdO(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

---

## 3. 参数说明

### 3.1 输入参数（Inputs）

| 参数名 | 输入/输出 | 必选/可选 | 描述 | 数据类型 | 数据格式 | 维度（Shape） | 非连续 Tensor |
|---|---|---|---|---|---|---|---|
| `q` | 输入 | 必选 | Query 输入张量 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, HK, T, K]` | 支持 |
| `k` | 输入 | 必选 | Key 输入张量 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, HK, T, K]` | 支持 |
| `v` | 输入 | 必选 | Value 输入张量 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, HV, T, V]` | 支持 |
| `h` | 输入 | 必选 | 前向保存的隐藏状态张量 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, HV, numChunks, K, V]` | 支持 |
| `g` | 输入 | 必选 | Gate 输入张量 | `FLOAT16`、`BFLOAT16`、`FLOAT` | `ND` | `[B, HV, T]` | 支持 |
| `cuSeqlensOptional` | 输入 | 可选 | 变长序列的累计长度信息 | `INT64` | `ND` | 1 维 | - |
| `chunkOffsetsOptional` | 输入 | 可选 | 分块索引信息，按 `[tokenBatchIdx, batchChunkIdx]` 成对扁平化 | `INT64` | `ND` | 1 维，长度需能被 2 整除 | - |

### 3.2 属性参数（Attributes）

| 参数名 | 输入/输出 | 必选/可选 | 描述 | 数据类型 | 取值约束 |
|---|---|---|---|---|---|
| `scale` | 输入 | 必选 | 缩放系数 | `double` | 建议按 `1 / sqrt(K)` 设置 |
| `chunkSize` | 输入 | 必选 | 分块大小 | `int64_t` | 仅支持 `64` / `128` |

### 3.3 输出参数（Outputs）

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度（Shape） | 非连续 Tensor |
|---|---|---|---|---|---|---|
| `oOut` | 输出 | 前向注意力输出张量 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, HV, T, V]` | 支持 |
| `workspaceSize` | 输出 | Device 侧所需 workspace 大小 | `uint64_t` | - | 标量 | - |
| `executor` | 输出 | 算子执行器，封装了计算流程 | `aclOpExecutor*` | - | - | - |

### 3.4 形状与约束

- `q`、`k` 的形状必须为 `[B, HK, T, K]`，二者完全同形。
- `v`、`oOut` 的形状必须为 `[B, HV, T, V]`。
- `q` 和 `v` 的 `B`、`T` 必须一致，head 数允许不同。
- `g` 的形状必须为 `[B, HV, T]`，head 维与 `v` 对齐。
- `h` 的形状必须为 `[B, HV, numChunks, K, V]`，head 维与 `v` 对齐，`K` 维与 `q/k` 对齐，`V` 维与 `v` 对齐。
- GVA 约束：`HV % HK == 0`，映射关系为 `hk = hv / (HV / HK)`。
- 当前实现要求 `K = 128`。
- 当前实现要求 `V = 128` 或 `256`。
- `chunkSize` 当前仅支持 `64` 或 `128`。
- 当启用变长模式时，`cuSeqlensOptional` 和 `chunkOffsetsOptional` 用于描述变长分块；二者需要同时提供，且当前实现仅支持 `B = 1`。

---

## 4. 调用约束与执行语义

### 4.1 可选参数约束

- `cuSeqlensOptional` 和 `chunkOffsetsOptional`：
    - 二者任意一个出现时进入变长模式，当前实现要求二者同时提供
    - 变长模式仅支持 `B = 1`

### 4.2 形状约束（强约束）

必须满足以下条件：

- `q, k`: `[B, HK, T, K]`
- `v, oOut`: `[B, HV, T, V]`
- `g`: `[B, HV, T]`
- `h`: `[B, HV, numChunks, K, V]`
- `HV % HK == 0`

额外限制：

- `K = 128`
- `V ∈ {128, 256}`
- `chunkSize ∈ {64, 128}`

### 4.3 变长模式（VarLen）

当提供 `cuSeqlensOptional` 时：

- `chunkOffsetsOptional` 必须同时提供
- `chunkOffsetsOptional` 是扁平化的一维 int64 数组，语义为连续的 `[tokenBatchIdx, batchChunkIdx]` pair
- `chunkOffsetsOptional` 的长度必须能被 2 整除
- `numChunks` 由 `chunkOffsetsOptional` 的 pair 数推导
- 当前实现仅支持 `B = 1`

### 4.4 数值语义

- `scale`：
    - 必须显式传入
    - 推荐设置为：`1 / sqrt(K)`

- 当前算子实现配置为：

```text
USE_G = True
USE_DW = True
USE_G_GAMMA = False
```

即：

- 启用 Gate
- 启用 delta rule 的 WY 分解
- 不使用 `gGamma`

---

## 5. Torch 测试调用示例

### 5.1 定长模式调用示例

```python
import torch
import torch_npu
import math

def test_chunk_fwd_o_fixed_len():
    # 参数设置
    B, HK, HV, T, K, V = 1, 2, 4, 256, 128, 128
    chunk_size = 64
    num_chunks = (T + chunk_size - 1) // chunk_size
    scale = 1.0 / math.sqrt(K)
    device = "npu:0"
    dtype = torch.bfloat16

    # 构造输入
    q = torch.randn(B, HK, T, K, device=device, dtype=dtype)
    k = torch.randn(B, HK, T, K, device=device, dtype=dtype)
    v = torch.randn(B, HV, T, V, device=device, dtype=dtype)
    h = torch.randn(B, HV, num_chunks, K, V, device=device, dtype=dtype)
    g = torch.randn(B, HV, T, device=device, dtype=dtype)

    # 调用算子（定长：cu_seqlens / chunk_indices 传 None）
    o = torch.ops.npu.npu_chunk_fwd_o(
        q, k, v, h, scale,
        g=g,
        g_gamma=None,
        cu_seqlens=None,
        chunk_indices=None,
        chunk_size=chunk_size,
        transpose_state_layout=False,
    )

    print("o shape:", o.shape)
    assert o.shape == v.shape
    print("Execution Successful!")

if __name__ == "__main__":
    test_chunk_fwd_o_fixed_len()
```

### 5.2 变长模式调用示例

变长模式下 `B = 1`，多个序列在 `T` 维拼接，`cu_seqlens` 为各序列累计长度（长度 `token_batch + 1`），`chunk_indices` 为 `[token_batch_id, chunk_id]` 二元组扁平化后的 `int` 列表。

```python
import torch
import torch_npu
import math

def test_chunk_fwd_o_varlen():
    # 参数设置：变长模式仅支持 B = 1
    B, HK, HV, K, V = 1, 2, 4, 128, 128
    chunk_size = 64
    scale = 1.0 / math.sqrt(K)
    device = "npu:0"
    dtype = torch.bfloat16

    # 假设拼接后总长度 T 由若干变长子序列组成
    seqlens = [80, 64, 64, 48]                    # 各子序列长度
    T = sum(seqlens)                              # 256
    cu_seqlens = [0]
    for s in seqlens:
        cu_seqlens.append(cu_seqlens[-1] + s)     # [0, 80, 144, 208, 256]

    # 由 cu_seqlens 推导每个子序列的分块，构造 [token_batch_id, chunk_id] 列表
    chunk_indices_pairs = []
    for tb, s in enumerate(seqlens):
        n_chunks = math.ceil(s / chunk_size)
        for c in range(n_chunks):
            chunk_indices_pairs.append([tb, c])
    num_chunks = len(chunk_indices_pairs)         # 注意：要求为偶数
    chunk_indices_flat = [x for pair in chunk_indices_pairs for x in pair]

    # 构造输入
    q = torch.randn(B, HK, T, K, device=device, dtype=dtype)
    k = torch.randn(B, HK, T, K, device=device, dtype=dtype)
    v = torch.randn(B, HV, T, V, device=device, dtype=dtype)
    h = torch.randn(B, HV, num_chunks, K, V, device=device, dtype=dtype)
    g = torch.randn(B, HV, T, device=device, dtype=dtype)

    # 调用算子
    o = torch.ops.npu.npu_chunk_fwd_o(
        q, k, v, h, scale,
        g=g,
        g_gamma=None,
        cu_seqlens=cu_seqlens,                    # list[int]
        chunk_indices=chunk_indices_flat,         # 扁平化的 [tb, c] 列表
        chunk_size=chunk_size,
        transpose_state_layout=False,
    )

    print("o shape:", o.shape)
    assert o.shape == v.shape
    print("Execution Successful!")

if __name__ == "__main__":
    test_chunk_fwd_o_varlen()
```

> 完整可运行示例（含参考实现与精度对比）见 `tests/pta/test_fwd_o.py`，运行脚本见 `tests/pta/run_gdn_fwd_o.sh`。

---

## 6. 目录结构

```text
chunk_fwd_o/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── CMakeLists.txt
│   ├── chunk_fwd_o_def.cpp
│   ├── chunk_fwd_o_tiling.cpp
│   ├── chunk_fwd_o_tiling.h
│   ├── chunk_fwd_o_tiling_processor.h
│   └── op_api/
│       ├── aclnn_chunk_fwd_o.cpp
│       ├── aclnn_chunk_fwd_o.h
│       ├── chunk_fwd_o.cpp
│       └── chunk_fwd_o.h
├── op_kernel/
│   ├── chunk_fwd_o.cpp
│   ├── chunk_fwd_o_struct.h
│   ├── epilogue/
│   └── gemm/
└── tests/
    └── pta/
        ├── data_compare_o.py
        ├── run_gdn_fwd_o.sh
        └── test_fwd_o.py
```
