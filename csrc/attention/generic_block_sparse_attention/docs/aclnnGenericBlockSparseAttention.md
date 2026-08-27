# aclnnGenericBlockSparseAttention

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                        |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>      |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>      |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                       |    ×     |
| <term>Atlas 推理系列产品</term>                                |    ×     |
| <term>Atlas 训练系列产品</term>                                |    ×     |

## 功能说明

- **接口功能**：块稀疏注意力（Generic Block Sparse Attention，GBSA）。根据 `sparseBlockIdx` / `sparseBlockCount` 指定的逻辑 KV 块，经 `blockTable` 映射到 Paged KV Cache 的物理页，执行 Online Softmax 形式的 FlashAttention 计算。

- **计算公式**：

  $$
  attentionOut = Softmax(scaleValue \cdot query \cdot key_{sparse}^{T}) \cdot value_{sparse}
  $$

- **调用前置**：必须先调用 [aclnnGenericBlockSparseAttentionMetadata](../../sparse_attention_score_metadata/docs/aclnnGenericBlockSparseAttentionMetadata.md) 生成 `metadata`（INT32，shape `[1024]`），再传入本接口。`metadata` 为不透明调度表，调用者不应解析或修改。

- **布局符号**：
    - `T`：各 batch Query token 累加长度
    - `N` / `Nq`：Query head 数；`Nkv`：KV head 数
    - `D`：head dim（当前仅支持 128）
    - `PAGED_BBND`：`[numBlocks, blockSize, Nkv, D]`

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用 `aclnnGenericBlockSparseAttentionGetWorkspaceSize` 获取 workspace 与执行器，再调用 `aclnnGenericBlockSparseAttention` 执行计算。

```cpp
aclnnStatus aclnnGenericBlockSparseAttentionGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *sparseBlockIdx,
    const aclTensor *sparseBlockCount,
    const aclTensor *metadataOptional,
    const aclTensor *attenMaskOptional,
    const aclTensor *qDequantScaleOptional,
    const aclTensor *kDequantScaleOptional,
    const aclTensor *vDequantScaleOptional,
    const aclTensor *pQuantScaleOptional,
    const aclTensor *cuSeqLengthsQOptional,
    const aclTensor *cuSeqLengthsKvOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKvOptional,
    const aclTensor *blockTableOptional,
    const aclIntArray *blockShape,
    int64_t isPackedGQA,
    char *layoutQ,
    char *layoutKv,
    double scaleValue,
    int64_t maskType,
    int64_t quantType,
    double dstTypeMax,
    int64_t softmaxPrecision,
    int64_t winLeft,
    int64_t winRight,
    int64_t returnSoftmaxlse,
    aclTensor *attentionOut,
    aclTensor *softmaxLseOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

```cpp
aclnnStatus aclnnGenericBlockSparseAttention(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnGenericBlockSparseAttentionGetWorkspaceSize

- **参数说明**

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
| :--- | :---: | :--- | :--- | :--- | :---: | :--- | :---: |
| query | 输入 | Query | 常规路径 `layoutQ=TND` | FLOAT16、BFLOAT16、FLOAT8_E4M3FN | ND | `[T, Nq, D]` | × |
| key | 输入 | Paged Key cache | 常规路径 `layoutKv=PAGED_BBND`；允许 **仅 dim0** 非连续 | 同 query | ND | `[numBlocks, blockSize, Nkv, D]` | √（仅 dim0） |
| value | 输入 | Paged Value cache | 同 key；可与 key 使用不同 dim0 stride | 同 query | ND | `[numBlocks, blockSize, Nkv, D]` | √（仅 dim0） |
| sparseBlockIdx | 输入 | Q 块选中的 KV 逻辑块索引 | packed GQA 时 3D | INT32 | ND | `[Nkv, totalQBlocks, topK]` | √ |
| sparseBlockCount | 输入 | 每个 Q 块实际 KV 块数 | 元素 ∈ `[0, topK]` | INT32 | ND | `[Nkv, totalQBlocks]` | √ |
| metadataOptional | 输入 | Metadata 调度表 | **必传**，INT32 `[1024]` | INT32 | ND | `[1024]` | √ |
| attenMaskOptional | 输入 | Attention mask | 常规路径可传 `nullptr` | - | ND | - | √ |
| q/k/vDequantScaleOptional、pQuantScaleOptional | 输入 | FP8 量化 scale | `quantType=5` 时按实现要求传入 | FLOAT 等 | ND | 依量化协议 | √ |
| cuSeqLengthsQOptional | 输入 | Q 存储长度前缀和 | TND 必传，首元素为 0 | INT64 | ND | `[B+1]` | √ |
| cuSeqLengthsKvOptional | 输入 | KV 存储长度前缀和 | PAGED_BBND 必传 | INT64 | ND | `[B+1]` | √ |
| sequsedQOptional | 输入 | Q 实际有效长度 | 可选；任务按 actual 打包 | INT32 | ND | `[B]` | √ |
| sequsedKvOptional | 输入 | KV 实际有效长度 | 可选 | INT32 | ND | `[B]` | √ |
| blockTableOptional | 输入 | 逻辑块→物理页 | PAGED_BBND 必传 | INT32 | ND | `[B, maxBlocksPerBatch]` | √ |
| blockShape | 输入 | `[blockShapeX, blockShapeY]` | 当前仅 `[1, 128]` | aclIntArray | - | 长度 2 | - |
| isPackedGQA | 输入 | packed GQA 开关 | **仅支持 1** | INT64 | - | - | - |
| layoutQ | 输入 | Query 布局 | 常规路径 `"TND"` | STRING | - | - | - |
| layoutKv | 输入 | KV 布局 | 常规路径 `"PAGED_BBND"` | STRING | - | - | - |
| scaleValue | 输入 | Softmax 前 scale | `0` 表示使用 `1/sqrt(D)` | DOUBLE | - | - | - |
| maskType | 输入 | Mask 类型 | 常规路径仅 `1`（因果） | INT64 | - | - | - |
| quantType | 输入 | 量化类型 | `0` 非量化；`5` FP8 全量化 | INT64 | - | - | - |
| dstTypeMax | 输入 | 量化预留 | 非量化可填 `0` | DOUBLE | - | - | - |
| softmaxPrecision | 输入 | Softmax 精度 | `0`/`1`；950 仅 `1`；A2/A3 上 bf16 仅 `0` | INT64 | - | - | - |
| winLeft / winRight | 输入 | 滑窗 | `maskType!=2` 时为 `-1` | INT64 | - | - | - |
| returnSoftmaxlse | 输入 | 是否输出 LSE | `0`/`1`；FP8 路径不支持 `1` | INT64 | - | - | - |
| attentionOut | 输出 | 注意力输出 | shape 与 query 一致 | 同 query（FP8 可为 FP16/BF16） | ND | 同 query | × |
| softmaxLseOptional | 输出 | Softmax LSE | `returnSoftmaxlse=1` 时必传 | FLOAT | ND | TND：`[T, Nq, 1]` | × |
| workspaceSize | 输出 | Device workspace 字节数 | 按返回值申请 | - | - | - | - |
| executor | 输出 | 算子执行器 | 不可为空 | - | - | - | - |

- **返回值：** `aclnnStatus`，参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## aclnnGenericBlockSparseAttention

| 参数名 | 输入/输出 | 描述 |
| :--- | :---: | :--- |
| workspace | 输入 | Device workspace；`workspaceSize==0` 时可传 `nullptr` |
| workspaceSize | 输入 | 第一段接口返回的大小 |
| executor | 输入 | 第一段接口返回的执行器 |
| stream | 输入 | ACL Stream |

- **返回值：** `aclnnStatus`，参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

1. **常规路径**：`layoutQ=TND`，`layoutKv=PAGED_BBND`，`maskType=1`，`blockShape=[1,128]`，`D=128`，`isPackedGQA=1`。
2. **GQA**：`Nq % Nkv == 0`。
3. **metadata**：必传，且须与当前输入/属性配套生成，不可跨 case 复用。
4. **Paged**：`blockTable`、`cuSeqLengthsQ`、`cuSeqLengthsKv` 必传。
5. **Softmax 精度**：
   - Ascend 950：仅 `softmaxPrecision=1`
   - Atlas A2/A3：`fp16` 支持 `0/1`；`bf16` 仅 `0`
6. **FP8**：`quantType=5` 当且仅当 Q/K/V 为 `FLOAT8_E4M3FN`；不支持同时 `returnSoftmaxlse=1`。
7. **KV 非连续**：仅允许 dim0（物理页轴）非连续；dim1–3 必须按行主序连续。aclnn 不对 key/value 做 Contiguous；tiling 使用 `kStride0`/`vStride0`。
8. **seqused**：可选。有 seqused 时任务空间按 actual Q 长度计，GM 与稀疏索引仍使用 cu 存储偏移（pad 在段末）。

## 调用示例

Tensor 创建与资源释放参见[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。完整 PyTorch 样例见：

- [test_torch_generic_block_sparse_attention.py](../examples/test_torch_generic_block_sparse_attention.py)
- dim0 非连续对照：[test_torch_gbsa_kv_dim0_strided.py](../examples/test_torch_gbsa_kv_dim0_strided.py)

```cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_generic_block_sparse_attention.h"
#include "aclnnop/aclnn_generic_block_sparse_attention_metadata.h"

// 1) 先跑 Metadata，得到 metadata[1024]
// 2) 再调用主算子：
aclnnStatus RunGbsa(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const aclTensor *sparseBlockIdx, const aclTensor *sparseBlockCount,
    const aclTensor *metadata, const aclTensor *cuQ, const aclTensor *cuKv,
    const aclTensor *blockTable, aclTensor *attentionOut, aclTensor *softmaxLse,
    aclrtStream stream)
{
    const int64_t blockShapeData[] = {1, 128};
    aclIntArray *blockShape = aclCreateIntArray(blockShapeData, 2);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    char layoutQ[] = "TND";
    char layoutKv[] = "PAGED_BBND";
    aclnnStatus ret = aclnnGenericBlockSparseAttentionGetWorkspaceSize(
        query, key, value, sparseBlockIdx, sparseBlockCount, metadata,
        nullptr, nullptr, nullptr, nullptr, nullptr,
        cuQ, cuKv, nullptr, nullptr, blockTable, blockShape,
        /*isPackedGQA*/ 1, layoutQ, layoutKv,
        /*scaleValue*/ 0.0, /*maskType*/ 1, /*quantType*/ 0,
        /*dstTypeMax*/ 0.0, /*softmaxPrecision*/ 1,
        /*winLeft*/ -1, /*winRight*/ -1, /*returnSoftmaxlse*/ 1,
        attentionOut, softmaxLse, &workspaceSize, &executor);
    aclDestroyIntArray(blockShape);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            return ret;
        }
    }
    ret = aclnnGenericBlockSparseAttention(workspace, workspaceSize, executor, stream);
    if (ret == ACLNN_SUCCESS) {
        ret = aclrtSynchronizeStream(stream);
    }
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    return ret;
}
```
