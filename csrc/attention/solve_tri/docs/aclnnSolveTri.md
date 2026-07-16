# aclnnSolveTri

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|Ascend 950PR/Ascend 950DT|      √     |
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|      √     |
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|      √     |

## 功能说明

- 接口功能：计算下三角矩阵求逆 $(I + A)^{-1}$，其中 $A$ 是严格下三角矩阵。

- 计算公式：

  SolveTri 算子用于高效计算块对角下三角矩阵的逆矩阵。对于输入矩阵 $A$（严格下三角），输出为 $(I + A)^{-1}$。

  该算子常用于 Gated Delta Rule 线性注意力机制中的 chunk-wise 矩阵求逆操作。

  $$
  Y = (I + A)^{-1}
  $$

  其中 $A$ 是严格下三角矩阵（对角线为0），$I$ 是单位矩阵。

## 函数原型

每个算子分为两段式接口，必须先调用 `aclnnSolveTriGetWorkspaceSize` 接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用 `aclnnSolveTri` 接口执行计算。

```cpp
// 获取执行所需的 workspace 大小
aclnnStatus aclnnSolveTriGetWorkspaceSize(
    const aclTensor *x,
    const aclIntArray *cuSeqlens,
    const aclIntArray *chunkIndices,
    const char *layout,
    const aclTensor *xOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

// 执行算子计算
aclnnStatus aclnnSolveTri(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## 参数说明

### aclnnSolveTriGetWorkspaceSize

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| x | 输入 | 公式中的输入矩阵 A，Device 侧的 aclTensor，数据类型支持 FLOAT16、BFLOAT16 |
| cuSeqlens | 输入 | 变长序列模式下的累积序列长度，可选参数。Device 侧的 aclIntArray，数据类型为 INT64 |
| chunkIndices | 输入 | 变长序列模式下的 chunk 索引，可选参数。Device 侧的 aclIntArray，数据类型为 INT64 |
| layout | 输入 | 数据布局模式，支持 "bhtd"、"bsnd"、"tnd" |
| xOut | 输出 | 公式中的输出矩阵 Y，Device 侧的 aclTensor，数据类型与 x 一致 |
| workspaceSize | 输出 | 返回执行该算子所需的 workspace 大小 |
| executor | 输出 | 返回算子执行器 |

### aclnnSolveTri

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| workspace | 输入 | 在 Device 侧申请的 workspace 内存地址 |
| workspaceSize | 输入 | 在 Device 侧申请的 workspace 大小 |
| executor | 输入 | 算子执行器 |
| stream | 输入 | 执行流 |

## 输入约束

1. **数据类型**：输入 x 仅支持 FLOAT16 和 BFLOAT16
2. **chunkSize**：最后一维（矩阵大小）支持 16、32、64 或 128
3. **数据布局**：
   - `bhtd`: 输入 shape 为 `[B, H, T, chunkSize]`
   - `bsnd`: 输入 shape 为 `[B, T, H, chunkSize]`
   - `tnd`: 输入 shape 为 `[total_T, H, chunkSize]`，需配合 cu_seqlens 和 chunk_indices 使用
4. **变长模式**：当 layout 为 "tnd" 时，cu_seqlens 和 chunk_indices 必须提供，数据类型为 INT64

## 输出说明

| 输出 | 数据类型 | 描述 |
|------|----------|------|
| xOut | FLOAT16/BFLOAT16 | 输出与输入同 shape，为 $(I + A)^{-1}$ 的计算结果 |

## 返回值

| 返回值 | 描述 |
|--------|------|
| ACLNN_SUCCESS | 成功 |
| 其他 | 失败 |

## PyTorch 接口

```python
torch.ops.npu.npu_solve_tri(
    x: Tensor,
    cu_seqlens: Optional[List[int]] = None,
    chunk_indices: Optional[List[int]] = None,
    layout: str = "bsnd"
) -> Tensor
```

### 参数说明

| 参数 | 类型 | 描述 |
|------|------|------|
| x | Tensor | 输入下三角矩阵，shape 为 `[B, H, T, chunkSize]` (bhtd)、`[B, T, H, chunkSize]` (bsnd) 或 `[total_T, H, chunkSize]` (tnd) |
| cu_seqlens | Optional[List[int]] | 变长模式下的累积序列长度，如 `[0, 100, 200]` 表示两个序列长度分别为 100 和 100 |
| chunk_indices | Optional[List[int]] | 变长模式下的 chunk 索引，格式为 `[seq_id_0, chunk_id_0, seq_id_1, chunk_id_1, ...]` |
| layout | str | 数据布局，可选 "bhtd"、"bsnd"、"tnd"，默认 "bsnd" |

### 返回值

| 返回值 | 类型 | 描述 |
|--------|------|------|
| output | Tensor | 输出矩阵，shape 与输入一致 |

### 使用示例

```python
import torch
import torch_npu
import fla_npu

# 基本用法 (BSND layout)
B, T, H, chunkSize = 2, 128, 4, 64
x = torch.randn(B, T, H, chunkSize, dtype=torch.float16, device="npu")
y = torch.ops.npu.npu_solve_tri(x, layout="bsnd")


# 变长序列模式 (TND layout)
seq_lens = [100, 128, 80]  # 3 个序列
total_T = sum(seq_lens)
cu_seqlens = [0] + list(torch.cumsum(torch.tensor(seq_lens), 0).tolist())
# 计算 chunk_indices
chunk_indices = []
for seq_idx, seq_len in enumerate(seq_lens):
    num_chunks = (seq_len + chunkSize - 1) // chunkSize
    for chunk_id in range(num_chunks):
        chunk_indices.extend([seq_idx, chunk_id])

x_tnd = torch.randn(total_T, H, chunkSize, dtype=torch.float16, device="npu")
y_tnd = torch.ops.npu.npu_solve_tri(
    x_tnd,
    cu_seqlens=cu_seqlens,
    chunk_indices=chunk_indices,
    layout="tnd"
)
```
