# ChunkGatedDeltaRuleFwdH 算子说明
`ChunkGatedDeltaRuleFwdH` 是一个用于分块门控 delta 规则（Chunk Gated Delta Rule）前向传播过程中的自定义算子。该算子负责计算各分块（chunk）的递推隐藏状态 `h`，即沿序列方向逐块传播状态矩阵，为后续前向输出计算提供跨块（inter-chunk）状态信息；同时输出经 WY 分解修正后的 `v_new`。

---

## 1. 算子功能

在分块序列模型中，计算以下张量：

- **h**：各分块起始处的隐藏状态张量，形状为 `[B, HV, numChunks, K, V]`
- **v_new**：各分块经 WY 分解修正后的 Value 张量，形状与 `u` 相同（`[B, HV, T, V]`）
- **finalState**：最后一个分块结束后的最终隐藏状态，形状为 `[N, HV, K, V]`（`N` 为变长子序列数，定长时 `N = B`），仅在 `output_final_state = true` 时返回有效值

具体而言，该算子沿分块维度顺序执行递推计算：

$$
S_{[c+1]} = \exp(g_{[c]}) \cdot S_{[c]} + k_{[c]}^\top \cdot u_{[c]}
$$

其中：
- $S_{[c]}$ 为第 $c$ 个分块起始处的隐藏状态矩阵
- $g_{[c]}$ 为第 $c$ 个分块的门控衰减值
- $k_{[c]}$ 为第 $c$ 个分块的 Key 矩阵
- $u_{[c]}$ 为第 $c$ 个分块经 WY 分解修正后的 Value 矩阵

算子将每个分块的起始状态 $S_{[c]}$ 存储至输出张量 `h` 中，供后续输出计算算子使用。

---

## 2. 接口定义

### 2.1 ACLNN 接口

每个算子分为两段式调用流程：

1. **获取 workspace 与执行器**
   调用 `aclnnChunkGatedDeltaRuleFwdHGetWorkspaceSize` 接口，获取算子执行所需的 workspace 大小，并创建执行器（executor）。

2. **执行算子计算**
   调用 `aclnnChunkGatedDeltaRuleFwdH` 接口，在指定的 workspace 和执行器下完成计算。

对应以下 C++ 接口（见 `op_host/op_api/aclnn_chunk_gated_delta_rule_fwd_h.h`）：

```cpp
/* funtion: aclnnChunkGatedDeltaRuleFwdHGetWorkspaceSize
 * parameters (order aligned with chunk_gated_delta_rule_fwd_h Python API):
 * k : required
 * w : required
 * u : required
 * gOptional : optional, only non-null aclTensor is supported
 * gkOptional : optional, reserved (must be nullptr)
 * initalStateOptional : optional
 * outputFinalState : required
 * chunkSize : required
 * saveNewValue : reserved (must be true)
 * cuSeqlensOptional : optional
 * chunkIndicesOptional : optional
 * useExp2 : reserved (must be false)
 * transposeStateLayout : reserved (must be false)
 * hOut : required
 * vNewOut : required
 * finalStateOut : optional
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnChunkGatedDeltaRuleFwdHGetWorkspaceSize(
    const aclTensor *k,
    const aclTensor *w,
    const aclTensor *u,
    const aclTensor *gOptional,
    const aclTensor *gkOptional,
    const aclTensor *initalStateOptional,
    bool outputFinalState,
    int64_t chunkSize,
    bool saveNewValue,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    bool useExp2,
    bool transposeStateLayout,
    const aclTensor *hOut,
    const aclTensor *vNewOut,
    const aclTensor *finalStateOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnChunkGatedDeltaRuleFwdH
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnChunkGatedDeltaRuleFwdH(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

---

## 3. 参数说明

参数表与 `torch_custom/fla_npu/op_plugin/ops/opapi/FLANpuOpApi.cpp` 中 `npu_chunk_gated_delta_rule_fwd_h` 的算子签名保持一致。

### 3.1 输入参数（Inputs）

| 参数名 | 输入/输出 | 必选/可选 | 描述 | 数据类型 | 数据格式 | 维度（Shape） | 非连续 Tensor |
|---|---|---|---|---|---|---|---|
| `k` | 输入 | 必选 | Key 输入张量 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, HK, T, K]` | 支持 |
| `w` | 输入 | 必选 | WY 分解中的 W 矩阵 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, HV, T, K]` | 支持 |
| `u` | 输入 | 必选 | 修正后的 Value 张量 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, HV, T, V]` | 支持 |
| `gOptional` | 输入 | 接口为可选；当前实现要求非空 | Gate 输入张量 | `FLOAT16`、`BFLOAT16`、`FLOAT` | `ND` | `[B, HV, T]` | 支持 |
| `gkOptional` | 输入 | 接口为可选；当前实现要求传空 | 预留的逐通道门控张量 | `FLOAT16`、`BFLOAT16` | `ND` | 未启用 | - |
| `initalStateOptional` | 输入 | 可选 | 初始隐藏状态张量；不传则默认全零 | `FLOAT16`、`BFLOAT16`、`FLOAT` | `ND` | `[B, HV, K, V]` | 支持 |
| `cuSeqlensOptional` | 输入 | 可选 | 变长序列的累计长度信息 | `INT64` | `ND` | 1 维 | - |
| `chunkIndicesOptional` | 输入 | 可选 | 分块索引信息（`[token_batch_id, chunk_id]` 二元组扁平化） | `INT64` | `ND` | 1 维，长度需能被 2 整除 | - |

### 3.2 属性参数（Attributes）

| 参数名 | 输入/输出 | 必选/可选 | 描述 | 数据类型 | 取值约束 |
|---|---|---|---|---|---|
| `outputFinalState` | 输入 | 必选 | 是否输出最终隐藏状态 `finalStateOut` | `bool` | `true` / `false` |
| `chunkSize` | 输入 | 必选 | 分块大小 | `int64_t` | 仅支持 `64` / `128` |
| `saveNewValue` | 输入 | 接口为必传；当前实现要求 `true` | 是否保存修正后的 `v_new` | `bool` | 仅支持 `true` |
| `useExp2` | 输入 | 接口为必传；当前实现要求 `false` | 是否使用 `exp2` 计算门控 | `bool` | 仅支持 `false` |
| `transposeStateLayout` | 输入 | 接口为必传；当前实现要求 `false` | 是否转置状态张量布局 | `bool` | 仅支持 `false` |

### 3.3 输出参数（Outputs）

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度（Shape） | 非连续 Tensor |
|---|---|---|---|---|---|---|
| `hOut` | 输出 | 各分块起始隐藏状态张量 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, HV, numChunks, K, V]` | 支持 |
| `vNewOut` | 输出 | 经 WY 分解修正的 Value 张量 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, HV, T, V]` | 支持 |
| `finalStateOut` | 输出 | 最终隐藏状态张量；`outputFinalState = false` 时无意义 | `FLOAT16`、`BFLOAT16`、`FLOAT` | `ND` | `[N, HV, K, V]` | 支持 |
| `workspaceSize` | 输出 | Device 侧所需 workspace 大小 | `uint64_t` | - | 标量 | - |
| `executor` | 输出 | 算子执行器，封装了计算流程 | `aclOpExecutor*` | - | - | - |

### 3.4 形状与约束

- `k` 的形状必须为 `[B, HK, T, K]`，与 `k` 同形的仅 `k` 自身（无其他 Q/K 张量）。
- `w` 的形状必须为 `[B, HV, T, K]`，head 维与 value 侧对齐。
- `u`、`vNewOut` 的形状必须为 `[B, HV, T, V]`。
- `k` 与 `u` 的 `B`、`T` 必须一致，head 数允许不同（GVA）。
- `gOptional` 的形状必须为 `[B, HV, T]`，head 维与 `u` 对齐。
- `hOut` 的形状必须为 `[B, HV, numChunks, K, V]`；定长时 `numChunks = ceil(T / chunkSize)`，变长时 `numChunks` 由 `chunkIndices` 推导。
- `initalStateOptional`（若提供）的形状必须为 `[B, HV, K, V]`。
- `finalStateOut` 的形状必须为 `[N, HV, K, V]`（`N` 为变长子序列数，定长时 `N = B`）。
- **GVA 约束**：`HV % HK == 0`；value head 索引 `hv` 映射到 key head `hk = hv / (HV / HK)`，`k`/`w` 的 key 侧偏移使用 `hk`，其余张量使用 `hv`。
- 当前实现要求 `K = 128`。
- 当前实现要求 `V = 128` 或 `256`（`V = 256` 时 kernel 使用对应 Catlass tile 路径）。
- `chunkSize` 当前仅支持 `64` 或 `128`。
- 当启用变长模式时，`cuSeqlensOptional` 和 `chunkIndicesOptional` 须同时提供，且当前实现仅支持 `B = 1`。

---

## 4. 调用约束与执行语义

### 4.1 可选参数约束

- `gOptional`：
  - 接口层为可选，但当前实现要求传入有效张量（不可为空）
- `gkOptional`：
  - 接口层为可选；**当前实现要求传空指针，否则执行失败**
- `initalStateOptional`：
  - 若不提供（传空指针），则默认初始状态为全零矩阵
  - 形状必须为 `[B, HV, K, V]`
- `cuSeqlensOptional` 和 `chunkIndicesOptional`：
  - 二者须同时提供或同时省略；同时提供时启用变长模式（varlen）
  - 变长模式仅支持 `B = 1`
- `saveNewValue` / `useExp2` / `transposeStateLayout`：
  - 当前实现下分别只允许 `true` / `false` / `false`

### 4.2 形状约束（强约束）

必须满足以下条件：

- `k`: `[B, HK, T, K]`
- `w`: `[B, HV, T, K]`
- `u, vNewOut`: `[B, HV, T, V]`
- `gOptional`: `[B, HV, T]`
- `hOut`: `[B, HV, numChunks, K, V]`
- `initalStateOptional`: `[B, HV, K, V]`（若提供）
- `finalStateOut`: `[N, HV, K, V]`
- `HV % HK == 0`（GVA）

额外限制：

- `K = 128`
- `V ∈ {128, 256}`
- `chunkSize ∈ {64, 128}`

### 4.3 变长模式（VarLen）

当提供 `cuSeqlensOptional` 时：

- `chunkIndicesOptional` 必须同时提供
- `numChunks` 由 `chunkIndices` 推导
- 要求 `numChunks` 为偶数
- 当前实现仅支持 `B = 1`

### 4.4 数值语义

- 该算子的核心递推公式为：

```text
S_{[c+1]} = exp(g_{[c]}) * S_{[c]} + k_{[c]}^T @ u_{[c]}
```

其中：
- `g_{[c]}` 为第 `c` 个分块的门控衰减值（在 log 空间中累加后取 exp）
- `k_{[c]}^T @ u_{[c]}` 为当前分块的键值外积累积

- 当前算子实现配置为：

```text
USE_G = True
USE_DW = True
USE_G_GAMMA = False
```

即：

- 启用 Gate
- 计算基于 WY 分解的修正值
- 不使用 `gGamma`

---

## 5. Torch 测试调用示例

PyTorch 端通过 `torch.ops.npu.npu_chunk_gated_delta_rule_fwd_h` 调用，签名（见 `FLANpuOpApi.cpp`）：

```python
h, v_new, final_state = torch.ops.npu.npu_chunk_gated_delta_rule_fwd_h(
    k, w, u,
    g=...,                       # 当前不可为 None
    gk=None,                     # 预留，必须为 None
    initial_state=None,          # 可选；不传则默认全零初始状态
    output_final_state=False,    # 是否输出最终状态
    chunk_size=64,               # 64 或 128
    save_new_value=True,         # 预留，必须为 True
    cu_seqlens=None,             # 变长模式下传 list[int]
    chunk_indices=None,          # 变长模式下传扁平化的 [tb, c] list[int]
    use_exp2=False,              # 预留，必须为 False
    transpose_state_layout=False # 预留，必须为 False
)
```

### 5.1 定长模式调用示例

```python
import torch
import torch_npu
import math

def test_chunk_gated_delta_rule_fwd_h_fixed_len():
    # 参数设置（GVA 示例：HK=2, HV=4）
    B, HK, HV, T, K, V = 1, 2, 4, 256, 128, 128
    chunk_size = 64
    num_chunks = (T + chunk_size - 1) // chunk_size
    device = "npu:0"
    dtype = torch.bfloat16

    # 构造输入
    k = torch.randn(B, HK, T, K, device=device, dtype=dtype)
    w = torch.randn(B, HV, T, K, device=device, dtype=dtype)
    u = torch.randn(B, HV, T, V, device=device, dtype=dtype)
    g = torch.randn(B, HV, T, device=device, dtype=torch.float32)
    initial_state = torch.zeros(B, HV, K, V, device=device, dtype=dtype)

    # 调用算子（定长：cu_seqlens / chunk_indices 传 None）
    h, v_new, final_state = torch.ops.npu.npu_chunk_gated_delta_rule_fwd_h(
        k, w, u,
        g=g,
        gk=None,
        initial_state=initial_state,
        output_final_state=True,
        chunk_size=chunk_size,
        save_new_value=True,
        cu_seqlens=None,
        chunk_indices=None,
        use_exp2=False,
        transpose_state_layout=False,
    )

    print("h shape:", h.shape)
    print("v_new shape:", v_new.shape)
    print("final_state shape:", final_state.shape)
    assert h.shape == (B, HV, num_chunks, K, V)
    assert v_new.shape == (B, HV, T, V)
    assert final_state.shape == (B, HV, K, V)
    print("Execution Successful!")

if __name__ == "__main__":
    test_chunk_gated_delta_rule_fwd_h_fixed_len()
```

### 5.2 变长模式调用示例

变长模式下 `B = 1`，多个序列在 `T` 维拼接，`cu_seqlens` 为各序列累计长度（长度 `token_batch + 1`），`chunk_indices` 为 `[token_batch_id, chunk_id]` 二元组扁平化后的 `int` 列表。

```python
import torch
import torch_npu
import math

def test_chunk_gated_delta_rule_fwd_h_varlen():
    # 参数设置：变长模式仅支持 B = 1（GVA：HK=2, HV=4）
    B, HK, HV, K, V = 1, 2, 4, 128, 128
    chunk_size = 64
    device = "npu:0"
    dtype = torch.bfloat16

    # 假设拼接后总长度 T 由若干变长子序列组成
    seqlens = [80, 64, 64, 48]                        # 各子序列长度
    T = sum(seqlens)                                  # 256
    cu_seqlens = [0]
    for s in seqlens:
        cu_seqlens.append(cu_seqlens[-1] + s)         # [0, 80, 144, 208, 256]

    # 由 cu_seqlens 推导每个子序列的分块，构造 [token_batch_id, chunk_id] 列表
    chunk_indices_pairs = []
    for tb, s in enumerate(seqlens):
        n_chunks = math.ceil(s / chunk_size)
        for c in range(n_chunks):
            chunk_indices_pairs.append([tb, c])
    num_chunks = len(chunk_indices_pairs)             # 注意：要求为偶数
    chunk_indices_flat = [x for pair in chunk_indices_pairs for x in pair]

    # 构造输入
    k = torch.randn(B, HK, T, K, device=device, dtype=dtype)
    w = torch.randn(B, HV, T, K, device=device, dtype=dtype)
    u = torch.randn(B, HV, T, V, device=device, dtype=dtype)
    g = torch.randn(B, HV, T, device=device, dtype=torch.float32)

    # 调用算子（变长模式：传入 cu_seqlens 与 chunk_indices；本例不带 initial_state）
    h, v_new, final_state = torch.ops.npu.npu_chunk_gated_delta_rule_fwd_h(
        k, w, u,
        g=g,
        gk=None,
        initial_state=None,
        output_final_state=True,
        chunk_size=chunk_size,
        save_new_value=True,
        cu_seqlens=cu_seqlens,                        # list[int]
        chunk_indices=chunk_indices_flat,             # 扁平化的 [tb, c] 列表
        use_exp2=False,
        transpose_state_layout=False,
    )

    N = len(seqlens)
    print("h shape:", h.shape)
    print("v_new shape:", v_new.shape)
    print("final_state shape:", final_state.shape)
    assert h.shape == (B, HV, num_chunks, K, V)
    assert v_new.shape == (B, HV, T, V)
    assert final_state.shape == (N, HV, K, V)
    print("Execution Successful!")

if __name__ == "__main__":
    test_chunk_gated_delta_rule_fwd_h_varlen()
```

> 完整可运行示例（含参考实现与精度对比）见 `tests/pta/test_fwd_h.py`，运行脚本见 `tests/pta/run_gdn_fwd_h.sh`。

---

## 6. 目录结构

```text
chunk_gated_delta_rule_fwd_h/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── CMakeLists.txt
│   ├── chunk_gated_delta_rule_fwd_h_def.cpp
│   ├── chunk_gated_delta_rule_fwd_h_tiling.cpp
│   ├── chunk_gated_delta_rule_fwd_h_tiling.h
│   └── op_api/
│       ├── aclnn_chunk_gated_delta_rule_fwd_h.cpp
│       ├── aclnn_chunk_gated_delta_rule_fwd_h.h
│       ├── chunk_gated_delta_rule_fwd_h.cpp
│       └── chunk_gated_delta_rule_fwd_h.h
├── op_kernel/
│   └── chunk_gated_delta_rule_fwd_h.cpp
└── tests/
    └── pta/
        ├── data_compare_h.py
        ├── run_gdn_fwd_h.sh
        └── test_fwd_h.py
```
