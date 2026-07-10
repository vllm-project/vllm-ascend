# RecomputeWUFwd 算子说明

`RecomputeWUFwd` 是一个用于分块序列计算中的前向重计算算子。该算子基于 `k`、`v`、`beta`、`A` 和 `g`，在每个 chunk 内重计算并输出中间张量 `w` 和 `u`。

该算子主要用于避免前向阶段保存较大的中间结果，在后续计算或反向计算前按需重建：

- **u**：由 `A` 与 `v * beta` 做 chunk 内矩阵乘得到
- **w**：由 `A` 与 `k * beta * exp(g)` 做 chunk 内矩阵乘得到

---

## 1. 算子功能

在分块序列模型中，对每个 batch、head 和 chunk，执行如下逻辑：

```text
vb      = v * beta[..., None]
kbg_exp = k * beta[..., None] * exp(g)[..., None]

u = A @ vb
w = A @ kbg_exp
```

其中：

- `A` 为 chunk 内局部矩阵，形状为 `[B, HV, T, chunkSize]`
- `vb` 的局部形状为 `[curChunkSize, V]`
- `kbg_exp` 的局部形状为 `[curChunkSize, K]`
- `u` 的局部输出形状为 `[curChunkSize, V]`
- `w` 的局部输出形状为 `[curChunkSize, K]`

当最后一个 chunk 不满 `chunkSize` 时，实际参与计算的行数为 `curChunkSize`。

---

## 2. 接口定义

### 2.1 ACLNN 接口

每个算子分为两段式调用流程：

1. **获取 workspace 与执行器**  
   调用 `aclnnRecomputeWUFwdGetWorkspaceSize` 接口，获取算子执行所需的 workspace 大小，并创建执行器（executor）。

2. **执行算子计算**  
   调用 `aclnnRecomputeWUFwd` 接口，在指定的 workspace 和 executor 下完成计算。

对应 C++ 接口如下：

```cpp
// 获取执行所需的 workspace 大小
aclnnStatus aclnnRecomputeWUFwdGetWorkspaceSize(
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *beta,
    const aclTensor *a,
    const aclTensor *g,
    const aclTensor *gk,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize,
    const aclTensor *wOut,
    const aclTensor *uOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor
);

// 执行算子
aclnnStatus aclnnRecomputeWUFwd(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream
);
```

---

## 3. 参数说明

### 3.1 输入参数（Inputs）

| 参数名                 | 输入/输出 | 必选/可选                    | 描述                         | 使用说明                                                                 | 数据类型                       | 数据格式 | 维度（Shape）          | 非连续 Tensor |
| ---------------------- | --------- | ---------------------------- | ---------------------------- | ------------------------------------------------------------------------ | ------------------------------ | -------- | ---------------------- | ------------- |
| `k`                    | 输入      | 必选                         | Key 输入张量                 | 参与 `w` 的重计算；ACLNN 接口执行前会先转为连续内存                       | `FLOAT16`、`BFLOAT16`          | `ND`     | `[B, HK, T, K]`         | 支持          |
| `v`                    | 输入      | 必选                         | Value 输入张量               | 参与 `u` 的重计算；ACLNN 接口执行前会先转为连续内存                       | `FLOAT16`、`BFLOAT16`          | `ND`     | `[B, HV, T, V]`         | 支持          |
| `beta`                 | 输入      | 必选                         | beta 权重张量                | 用于缩放 `v` 和 `k`；ACLNN 接口执行前会先转为连续内存                     | `FLOAT16`、`BFLOAT16`、`FLOAT` | `ND`     | `[B, HV, T]`            | 支持          |
| `a` / `A`              | 输入      | 必选                         | chunk 内局部矩阵             | 每个 chunk 内参与 `A @ vb` 和 `A @ kbg_exp`；ACLNN 接口执行前会先转为连续内存 | `FLOAT16`、`BFLOAT16`          | `ND`     | `[B, HV, T, chunkSize]` | 支持          |
| `g`                    | 输入      | ACLNN 当前实现要求必传       | Gate / log-decay 输入张量     | 当前实现中用于计算 `exp(g)`；ACLNN 接口执行前会先转为连续内存             | `FLOAT16`、`BFLOAT16`、`FLOAT` | `ND`     | `[B, HV, T]`            | 支持          |
| `gk`                   | 输入      | 接口存在；当前实现要求传空   | 预留输入                     | 当前 ACLNN 封装要求 `gk == nullptr`，否则参数检查失败                    | `FLOAT16`、`BFLOAT16`、`FLOAT` | `ND`     | 未启用                 | -             |
| `cuSeqlensOptional`    | 输入      | 可选                         | 变长序列累计长度信息         | 变长模式输入；表示每条序列的起止位置，形状为 `[N + 1]`                   | `INT64`                        | `ND`     | 1 维                   | -             |
| `chunkIndicesOptional` | 输入      | 可选                         | 变长模式 chunk 索引信息      | 逻辑上表示为 `[numChunks, 2]`，实际按一维数组 `[numChunks * 2]` 传入       | `INT64`                        | `ND`     | 1 维                   | -             |

---

### 3.2 属性参数（Attributes）

| 参数名      | 输入/输出 | 必选/可选 | 描述       | 使用说明                         | 数据类型  | 取值约束     |
| ----------- | --------- | --------- | ---------- | -------------------------------- | --------- | ------------ |
| `chunkSize` | 输入      | 必选      | 分块大小   | 当前 tiling 显式支持 `64` 或 `128` | `int64_t` | `64` / `128` |

---

### 3.3 输出参数（Outputs）

| 参数名          | 输入/输出 | 描述                         | 数据类型              | 数据格式 | 维度（Shape）  | 非连续 Tensor |
| --------------- | --------- | ---------------------------- | --------------------- | -------- | -------------- | ------------- |
| `wOut` / `w`    | 输出      | 重计算得到的 `w` 张量         | `FLOAT16`、`BFLOAT16` | `ND`     | `[B, HV, T, K]` | 支持          |
| `uOut` / `u`    | 输出      | 重计算得到的 `u` 张量         | `FLOAT16`、`BFLOAT16` | `ND`     | `[B, HV, T, V]` | 支持          |
| `workspaceSize` | 输出      | Device 侧所需 workspace 大小 | `uint64_t`            | -        | 标量           | -             |
| `executor`      | 输出      | 算子执行器                   | `aclOpExecutor*`      | -        | -              | -             |

---

### 3.4 形状与约束

当前实现显式检查以下维度要求：

- `k` 必须为 4 维：`[B, HK, T, K]`
- `v` 必须为 4 维：`[B, HV, T, V]`
- `beta` 必须为 3 维：`[B, HV, T]`
- `A` 必须为 4 维：`[B, HV, T, chunkSize]`
- `g` 必须为 3 维：`[B, HV, T]`
- `k` 与 `v` 的 `B`、`T` 必须一致；`K` 维与 `k` 对齐，`V` 维与 `v` 对齐
- `beta`、`g`、`A` 的 head 维须与 `v` 对齐（`HV`）
- **GVA 约束**：`HV % HK == 0`；读 `k` 时使用 `hk = hv / (HV / HK)`，写 `w`/`u` 及 `v`/`beta`/`g`/`A` 使用 value head 索引 `hv`
- `w` 输出形状为 `[B, HV, T, K]`（**非** `empty_like(k)` 的 `[B, HK, T, K]`）

额外限制：

- `chunkSize` 当前仅支持 `64` 或 `128`
- 变长模式下要求 `B = 1`
- `gk` 当前实现未启用，必须传 `nullptr`
- 当前 kernel 主要面向 `K = 128`，`V ∈ {128, 256}`
- **TilingKey**：`V = 128` 时 Key=1，`V = 256` 时 Key=2（Cube Catlass tile 分支不同）

---

## 4. 调用约束与执行语义

### 4.1 可选参数约束

- `g`：
    - 在算子定义中为 optional
    - 但当前 ACLNN 封装中要求 `g != nullptr`
    - 实际调用时应按必传处理

- `gk`：
    - 接口层保留
    - 当前实现未使用
    - **必须传入 `nullptr`，否则参数检查失败**

- `cuSeqlensOptional` 和 `chunkIndicesOptional`：
    - 二者用于变长模式
    - 当 `cuSeqlensOptional` 非空时，`chunkIndicesOptional` 也必须非空
    - 变长模式当前仅支持 `B = 1`

---

### 4.2 定长模式（Fixed Length）

当 `cuSeqlensOptional == nullptr` 时启用定长模式。

chunk 数量计算方式：

```text
chunkNum = B * ceil(T / chunkSize)
```

每个 batch 内按 `T` 维切分 chunk：

```text
chunk 0: [0, chunkSize)
chunk 1: [chunkSize, 2 * chunkSize)
...
last chunk: [n * chunkSize, T)
```

如果最后一个 chunk 不满 `chunkSize`，则使用实际长度 `curChunkSize` 参与计算。

---

### 4.3 变长模式（VarLen）

当提供 `cuSeqlensOptional` 时启用变长模式。

约束：

- `B` 必须为 `1`
- `cuSeqlensOptional` 必须为一维 `INT64` 数组
- `chunkIndicesOptional` 必须为一维 `INT64` 数组
- `cuSeqlensOptional[0]` 必须为 `0`
- `cuSeqlensOptional` 必须严格递增
- `chunkIndicesOptional` 必须与 `cuSeqlensOptional` 和 `chunkSize` 推导出的 chunk 列表完全一致

`chunkIndicesOptional` 的逻辑格式为：

```text
[
  [seq_id, chunk_id],
  [seq_id, chunk_id],
  ...
]
```

实际接口传入 flatten 后的一维数组：

```text
[seq_id_0, chunk_id_0, seq_id_1, chunk_id_1, ...]
```

示例：

```text
cuSeqlensOptional = [0, 4, 10]
chunkSize = 4

序列划分：
- 第 0 条序列长度 = 4  → 需要 1 个 chunk
- 第 1 条序列长度 = 6  → 需要 2 个 chunk

chunkIndicesOptional（二维逻辑表示）：
[
  [0, 0],   # 第 0 条序列，第 0 个 chunk
  [1, 0],   # 第 1 条序列，第 0 个 chunk
  [1, 1]    # 第 1 条序列，第 1 个 chunk
]

flatten 后实际传入：
[0, 0, 1, 0, 1, 1]
```

---

### 4.4 数值语义

对每个 chunk，设当前 chunk 范围为 `[bos, eos)`，当前 chunk 长度为：

```text
curChunkSize = eos - bos
```

在每个 head 上执行：

```text
vb[t, v] = v[t, v] * beta[t]

kbg_exp[t, k] = k[t, k] * beta[t] * exp(g[t])

u_chunk = A_chunk @ vb_chunk

w_chunk = A_chunk @ kbg_exp_chunk
```

其中：

```text
A_chunk       : [curChunkSize, curChunkSize]
vb_chunk      : [curChunkSize, V]
kbg_exp_chunk : [curChunkSize, K]
u_chunk       : [curChunkSize, V]
w_chunk       : [curChunkSize, K]
```

---

### 4.5 Workspace 说明

当前 kernel 中 workspace 用作中间结果缓存：

- AIV 阶段计算 `vb = v * beta`
- AIC 阶段使用 `vb` 计算 `u = A @ vb`
- AIV 阶段随后计算 `kbg_exp = k * beta * exp(g)`
- AIC 阶段使用 `kbg_exp` 计算 `w = A @ kbg_exp`

实现中 `vb` 和 `kbg_exp` 复用同一段 workspace，不同时保留。

---

## 5. Torch 测试调用示例

该算子可通过 PyTorch 接口直接调用，底层的两阶段接口（workspace + executor）已被封装，无需手动处理。

`RecomputeWUFwd` 根据 `k`、`v`、`beta`、`A` 和 `g` 重计算并输出 `w`、`u`：

- `w`: `[B, HV, T, K]`
- `u`: `[B, HV, T, V]`

其中：

- `k`: `[B, HK, T, K]`
- `v`: `[B, HV, T, V]`
- `beta`: `[B, HV, T]`
- `A`: `[B, HV, T, chunk_size]`
- `g`: `[B, HV, T]`
- GVA：`HV % HK == 0`

当前实现中：

- `chunk_size` 仅支持 `64` 或 `128`
- `g` 必须传入，不能为 `None`
- `gk` 当前未启用，必须传 `None`
- 变长场景下 `B` 必须为 `1`
- 变长场景下 `chunk_indices` 为 `[seq_id, chunk_id]` 的 flatten 形式

---

### 5.1 定长场景

```python
import torch
import torch_npu

# 设备
device = "npu:0"

# 基本参数（GVA 示例：HK=2, HV=4；MHA 时令 HK=HV）
B, HK, HV, T, K, V = 1, 2, 4, 1024, 128, 128
chunk_size = 64

# 构造输入
k = torch.randn(B, HK, T, K, device=device, dtype=torch.float16)
v = torch.randn(B, HV, T, V, device=device, dtype=torch.float16)

# beta shape: [B, HV, T]
beta = torch.randn(B, HV, T, device=device, dtype=torch.float32)

# A shape: [B, HV, T, chunk_size]
A = torch.randn(B, HV, T, chunk_size, device=device, dtype=torch.float16)

# g shape: [B, HV, T]
g = torch.randn(B, HV, T, device=device, dtype=torch.float32)

# 定长场景下 cu_seqlens 和 chunk_indices 均为 None。
# gk 当前未启用，必须为 None。
w, u = torch.ops.npu.npu_recompute_wu_fwd(
    k,
    v,
    beta,
    A,
    g,
    chunk_size,
    gk=None,
    cu_seqlens=None,
    chunk_indices=None
)

print(w.shape, u.shape)
# 期望输出:
# torch.Size([B, HV, T, K]) torch.Size([B, HV, T, V])
```

---

### 5.2 变长场景

```python
import torch
import torch_npu

device = "npu:0"

# 变长场景当前仅支持 B = 1
B, HK, HV, K, V = 1, 2, 4, 128, 128
chunk_size = 64

# 3 条变长序列
seq_lens = [70, 130, 60]
cu_seqlens_list = [0]
for seq_len in seq_lens:
    cu_seqlens_list.append(cu_seqlens_list[-1] + seq_len)

cu_seqlens = torch.tensor(cu_seqlens_list, device=device, dtype=torch.int64)
total_len = cu_seqlens_list[-1]

# 构造 chunk_indices。
# 注意：recompute_wu_fwd 的 chunk_indices 不是 [start, end]，
# 而是 [seq_id, chunk_id] 的 flatten 形式。
#
# 示例：
# seq_lens = [70, 130, 60], chunk_size = 64
#
# seq 0 长度 70:
#   [seq_id=0, chunk_id=0]
#   [seq_id=0, chunk_id=1]
#
# seq 1 长度 130:
#   [seq_id=1, chunk_id=0]
#   [seq_id=1, chunk_id=1]
#   [seq_id=1, chunk_id=2]
#
# seq 2 长度 60:
#   [seq_id=2, chunk_id=0]
#
# flatten 后:
# [0, 0, 0, 1, 1, 0, 1, 1, 1, 2, 2, 0]
chunk_indices_list = []
for seq_id, seq_len in enumerate(seq_lens):
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    for chunk_id in range(num_chunks):
        chunk_indices_list.extend([seq_id, chunk_id])

chunk_indices = torch.tensor(chunk_indices_list, device=device, dtype=torch.int64)

# 变长场景下 T 使用 total_len
T = total_len

k = torch.randn(B, HK, T, K, device=device, dtype=torch.float16)
v = torch.randn(B, HV, T, V, device=device, dtype=torch.float16)

# beta shape: [B, HV, T]
beta = torch.randn(B, HV, T, device=device, dtype=torch.float32)

# A shape: [B, HV, T, chunk_size]
A = torch.randn(B, HV, T, chunk_size, device=device, dtype=torch.float16)

# g shape: [B, HV, T]
g = torch.randn(B, HV, T, device=device, dtype=torch.float32)

w, u = torch.ops.npu.npu_recompute_wu_fwd(
    k,
    v,
    beta,
    A,
    g,
    chunk_size,
    gk=None,
    cu_seqlens=cu_seqlens,
    chunk_indices=chunk_indices
)

print(w.shape, u.shape)
# 期望输出:
# torch.Size([1, HV, total_len, K]) torch.Size([1, HV, total_len, V])
```

---

### 5.3 说明

- 定长场景：
    - `cu_seqlens=None`
    - `chunk_indices=None`

- 变长场景：
    - `B` 必须为 `1`
    - `cu_seqlens` 为一维 `int64` Tensor
    - `cu_seqlens[0]` 必须为 `0`
    - `cu_seqlens` 必须非递减，允许 0-length sequence
    - `chunk_indices` 为一维 `int64` Tensor
    - `chunk_indices` 必须是 `[seq_id, chunk_id]` 的 flatten 形式

- 当前实现：
    - `g` 必须传入
    - `gk` 当前未启用，必须传 `None`
    - `chunk_size` 仅支持 `64` 或 `128`

- 输出：
    - `w`: `[B, HV, T, K]`
    - `u`: `[B, HV, T, V]`

---

## 6. 目录结构

```text
recompute_wu_fwd/
├── examples/
│   └── test_aclnn_recompute_wu_fwd.cpp
├── op_host/
│   ├── op_api/
│   │   ├── aclnn_recompute_wu_fwd.cpp
│   │   └── aclnn_recompute_wu_fwd.h
│   ├── op_tiling/
│   │   ├── recompute_wu_fwd_tiling.cpp
│   │   └── recompute_wu_fwd_tiling.h
│   ├── recompute_wu_fwd_def.cpp
│   └── CMakeLists.txt
└── op_kernel/
    ├── recompute_wu_fwd_common.h
    ├── recompute_wu_fwd_cube.h
    ├── recompute_wu_fwd_vector.h
    └── recompute_wu_fwd.cpp
```
