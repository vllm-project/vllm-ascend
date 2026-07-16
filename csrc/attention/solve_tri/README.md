# SolveTri 算子说明

`SolveTri` 是用于计算下三角矩阵求逆的自定义算子，主要应用于 Gated Delta Rule 线性注意力机制中的 chunk-wise 矩阵求逆操作。

---

## 1. 算子功能

计算 $(I + A)^{-1}$，其中 $A$ 是严格下三角矩阵（对角线为0）。

$$Y = (I + A)^{-1}$$

该算子支持三种数据布局：

- **BHTD**: `[Batch, Head, T, chunkSize]`
- **BSND**: `[Batch, T, Head, chunkSize]`
- **TND**: `[num_tokens, Head, chunkSize]`（变长序列模式）

---

## 2. 接口定义

### 2.1 ACLNN 接口

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

### 2.2 PyTorch 接口

```python
torch.ops.npu.npu_solve_tri(
    x: Tensor,
    cu_seqlens: Optional[List[int]] = None,
    chunk_indices: Optional[List[int]] = None,
    layout: str = "bsnd"
) -> Tensor
```

---

## 3. 输入参数

| 参数 | 数据类型 | 是否必须 | 描述 |
|------|----------|----------|------|
| x | FLOAT16/BFLOAT16 | 是 | 输入下三角矩阵 |
| cu_seqlens | INT64 | TND 模式必须 | 累积序列长度 |
| chunk_indices | INT64 | TND 模式必须 | chunk 索引数组 |
| layout | string | 否 | 数据布局，默认 "bsnd" |

---

## 4. 输入约束

1. **数据类型**：仅支持 FLOAT16 和 BFLOAT16
2. **chunkSize**：最后一维支持 16、32、64 或 128
3. **输入维度**：
   - BHTD/BSND: 4D tensor
   - TND: 3D tensor
4. **变长模式**：TND layout 必须提供 cu_seqlens 和 chunk_indices

---

## 5. 输出参数

| 输出 | 数据类型 | 描述 |
|------|----------|------|
| xOut | FLOAT16/BFLOAT16 | 输出矩阵，shape 与输入一致 |

---

## 6. 算子实现

### 6.1 算法原理

使用 **MCH (Matrix Chain Halving) + MBH (Matrix Block Halving)** 算法高效计算下三角矩阵的逆：

1. 将矩阵分块为 $2 \times 2$ 块矩阵
2. 利用下三角矩阵的结构特性递归求解
3. 通过 AIC 核执行 CUBE 矩阵乘法，AIV 核生成辅助矩阵

### 6.2 架构适配

- **Ascend 910B/910C (arch20/arch30)**：使用 `solve_tri_cube.h` + `solve_tri_vector.h`，AIC 执行 MCH+MBH，AIV 生成 GM 辅助矩阵，通过 `SyncAll` 全核同步。
- **Ascend 950 (arch35)**：使用 `arch35/solve_tri_ascend950.h`，MCH 与 MBH 合一实现，AIV 通过 UB gather 对角块并直接写入 L1（紧凑 NZ 布局），AIC 使用 UB 常驻中间结果，通过 `CrossCoreSetFlag`/`WaitFlag` 逐 tile 握手，避免 GM 中转。

## 7. 目录结构

```
solve_tri/
├── docs/
│   └── aclnnSolveTri.md
├── op_host/
│   ├── op_api/
│   │   ├── aclnn_solve_tri.cpp
│   │   ├── aclnn_solve_tri.h
│   │   └── solve_tri.cpp
│   ├── solve_tri_def.cpp
│   ├── solve_tri_tiling.cpp
│   ├── solve_tri_tiling.h
│   └── CMakeLists.txt
├── op_kernel/
│   ├── arch35/
│   │   ├── mem.h
│   │   ├── solve_tri_ascend950.h
│   │   └── solve_tri_ascend950_common.h
│   ├── solve_tri.cpp
│   ├── solve_tri_common.h
│   ├── solve_tri_cube.h
│   └── solve_tri_vector.h
├── CMakeLists.txt
├── solve_tri_torch_adpt.h
└── README.md
```
