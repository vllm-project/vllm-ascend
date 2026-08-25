# aclnnGenericBlockSparseAttentionMetadata

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3训练系列产品/Atlas A3推理系列产品 | √ |
| Atlas A2训练系列产品/Atlas A2推理系列产品 | √ |
| Atlas 200I/500 A2推理产品 | × |
| Atlas推理系列产品 | × |
| Atlas训练系列产品 | × |

## 功能说明

- 接口功能：生成`aclnnSparseAttentionScore`计算所需的`metadata`。
- 输出`metadata`应直接传给`aclnnSparseAttentionScore`，不建议单独使用。
- `metadata`为不透明数据，调用者不应解析或修改其中的内容。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用`aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize`获取workspace大小，再调用`aclnnGenericBlockSparseAttentionMetadata`执行计算。

```cpp
aclnnStatus aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize(
    const aclTensor *sparseBlockIdx,
    const aclTensor *sparseBlockCount,
    const aclTensor *cuSeqLengthsOptional,
    const aclTensor *cuSeqLengthsKvOptional,
    const aclTensor *seqUsedQOptional,
    const aclTensor *seqUsedKvOptional,
    int64_t maxQSeqLen,
    int64_t maxKvSeqLen,
    int64_t numQHeads,
    int64_t numKvHeads,
    int64_t headDim,
    const aclIntArray *blockShape,
    int64_t isPackedGQA,
    const char *qInputLayout,
    const char *kvInputLayout,
    int64_t maskType,
    int64_t quantType,
    int64_t softmaxPrecision,
    int64_t windowSizeLeft,
    int64_t windowSizeRight,
    const aclTensor *metadata,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

```cpp
aclnnStatus aclnnGenericBlockSparseAttentionMetadata(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize

- **参数说明**

<table style="undefined;table-layout: fixed; width: 1800px"><colgroup>
<col style="width: 190px"><col style="width: 90px"><col style="width: 420px"><col style="width: 430px">
<col style="width: 100px"><col style="width: 90px"><col style="width: 280px"><col style="width: 100px">
</colgroup><thead><tr><th>参数名</th><th>输入/输出</th><th>描述</th><th>使用说明</th><th>数据类型</th><th>数据格式</th><th>维度(shape)</th><th>非连续Tensor</th></tr></thead><tbody>
<tr><td>sparseBlockIdx</td><td>输入</td><td>每个Q块选择的KV块逻辑索引。</td><td>必选。最后一维表示每个Q块可存储的最大KV块数量。</td><td>INT32</td><td>ND</td><td>TND为3维；BSND/BNSD为4维。</td><td>√</td></tr>
<tr><td>sparseBlockCount</td><td>输入</td><td>每个Q块实际选择的KV块数量。</td><td>必选。每个元素范围为[0, sparseBlockIdx最后一维]。</td><td>INT32</td><td>ND</td><td>TND为2维；BSND/BNSD为3维。</td><td>√</td></tr>
<tr><td>cuSeqLengthsOptional</td><td>输入</td><td>各Batch的Query存储长度前缀和，首元素为0。</td><td>TND场景必选；BSND/BNSD场景可选。</td><td>INT64</td><td>ND</td><td>1维，shape为(B+1,)</td><td>√</td></tr>
<tr><td>cuSeqLengthsKvOptional</td><td>输入</td><td>各Batch的Key/Value存储长度前缀和，首元素为0。</td><td>KV为TND、PAGED_BBND或PAGED_BNBD时必选。</td><td>INT64</td><td>ND</td><td>1维，shape为(B+1,)</td><td>√</td></tr>
<tr><td>seqUsedQOptional</td><td>输入</td><td>各Batch实际参与计算的Query长度。</td><td>可选，优先级高于cuSeqLengthsOptional。</td><td>INT32</td><td>ND</td><td>1维，shape为(B,)</td><td>√</td></tr>
<tr><td>seqUsedKvOptional</td><td>输入</td><td>各Batch实际参与计算的Key/Value长度。</td><td>可选。取值应与主算子对应输入保持一致。</td><td>INT32</td><td>ND</td><td>1维，shape为(B,)</td><td>√</td></tr>
<tr><td>maxQSeqLen</td><td>输入</td><td>Query最大Sequence Length。</td><td>必须大于0。</td><td>INT64</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>maxKvSeqLen</td><td>输入</td><td>Key/Value最大Sequence Length。</td><td>必须大于0。</td><td>INT64</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>numQHeads</td><td>输入</td><td>Query的head数。</td><td>必须大于0。</td><td>INT64</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>numKvHeads</td><td>输入</td><td>Key/Value的head数。</td><td>必须大于0。</td><td>INT64</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>headDim</td><td>输入</td><td>每个head的特征维度。</td><td>必须大于0。</td><td>INT64</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>blockShape</td><td>输入</td><td>Q、KV方向的逻辑块大小，格式为[blockShapeX, blockShapeY]。</td><td>长度固定为2；当前blockShapeX仅支持1，blockShapeY必须为16的倍数。</td><td>aclIntArray</td><td>-</td><td>1维，元素个数为2</td><td>-</td></tr>
<tr><td>isPackedGQA</td><td>输入</td><td>是否由同一KV head group共享稀疏块索引。</td><td>当前ACLNN接口仅支持1。</td><td>INT64</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>qInputLayout</td><td>输入</td><td>Query的数据排布。</td><td>支持TND、BSND、BNSD。</td><td>STRING</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>kvInputLayout</td><td>输入</td><td>Key/Value的数据排布。</td><td>支持TND、BSND、BNSD、PAGED_BBND、PAGED_BNBD。</td><td>STRING</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>maskType</td><td>输入</td><td>Attention mask类型。</td><td>支持0、1、2；滑窗场景使用2。</td><td>INT64</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>quantType</td><td>输入</td><td>量化类型。</td><td>取0表示不量化；非0值仅Atlas A5支持。</td><td>INT64</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>softmaxPrecision</td><td>输入</td><td>Softmax精度模式。</td><td>取值应与主算子对应属性保持一致。</td><td>INT64</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>windowSizeLeft</td><td>输入</td><td>滑窗向左包含的token数。</td><td>maskType不为2时必须为-1；maskType为2时必须非负。</td><td>INT64</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>windowSizeRight</td><td>输入</td><td>滑窗向右包含的token数。</td><td>maskType不为2时必须为-1；maskType为2时必须非负。</td><td>INT64</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>metadata</td><td>输出</td><td>`aclnnSparseAttentionScore`计算所需的metadata。</td><td>Device侧Tensor，由调用者申请；其内容为不透明格式，不应解析或修改。</td><td>INT32</td><td>ND</td><td>1维，shape固定为(1024,)</td><td>×</td></tr>
<tr><td>workspaceSize</td><td>输出</td><td>返回Device侧workspace大小。</td><td>调用者应按返回值申请workspace。</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>executor</td><td>输出</td><td>返回包含算子计算流程的执行器。</td><td>不可为空。</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
</tbody></table>

- **返回值：** 返回`aclnnStatus`状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## aclnnGenericBlockSparseAttentionMetadata

- **参数说明**

| 参数名 | 输入/输出 | 描述 |
| :--- | :---: | :--- |
| workspace | 输入 | Device侧workspace地址；workspaceSize为0时可传nullptr。 |
| workspaceSize | 输入 | 第一段接口返回的workspace大小。 |
| executor | 输入 | 第一段接口返回的执行器。 |
| stream | 输入 | 执行任务的ACL Stream。 |

- **返回值：** 返回`aclnnStatus`状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- `blockShape`长度固定为2；当前`blockShape[0]`仅支持1，`blockShape[1]`必须为16的倍数。
- 当前仅支持`isPackedGQA=1`，此时同一KV head group内的Q head共享稀疏块索引。
- Query为TND时：
  - 必须传入`cuSeqLengthsOptional`，Batch为其元素个数减1；
  - `sparseBlockIdx` shape为`[numKvHeads, totalQBlocks, maxSparseBlockCount]`；
  - `sparseBlockCount` shape为`[numKvHeads, totalQBlocks]`；
  - `totalQBlocks`为各Batch的Q块数量之和；
  - `seqUsedQOptional[i]`不能超过对应Batch的Query存储长度。
- Query为BSND/BNSD时：
  - `sparseBlockIdx` shape为`[batch, numKvHeads, maxQBlocks, maxSparseBlockCount]`；
  - `sparseBlockCount` shape为`[batch, numKvHeads, maxQBlocks]`；
  - `maxQBlocks=CeilDiv(maxQSeqLen, blockShape[0])`。
- `sparseBlockCount`中的每个元素必须位于`[0, maxSparseBlockCount]`，`maxSparseBlockCount`为`sparseBlockIdx`的最后一维大小。
- `seqUsedQOptional`、`seqUsedKvOptional`、`cuSeqLengthsOptional`和`cuSeqLengthsKvOptional`的Batch信息需要保持一致；实际序列长度不能超过对应的存储长度和最大序列长度。
- 输出`metadata`固定为长度1024的INT32 Tensor，只能与生成它时使用的输入和属性配套传给`aclnnSparseAttentionScore`，不能跨不同shape或属性复用。

## 调用示例

以下示例展示两段式接口调用顺序。Tensor创建、输入拷贝和资源释放方式请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_generic_block_sparse_attention_metadata.h"

aclnnStatus RunMetadata(const aclTensor *sparseBlockIdx, const aclTensor *sparseBlockCount,
                        aclTensor *metadata, aclrtStream stream)
{
    const int64_t blockShapeData[] = {1, 128};
    aclIntArray *blockShape = aclCreateIntArray(blockShapeData, 2);
    if (blockShape == nullptr) {
        return ACL_ERROR_BAD_ALLOC;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize(
        sparseBlockIdx, sparseBlockCount, nullptr, nullptr, nullptr, nullptr,
        16, 2048, 32, 8, 128, blockShape, 1, "BSND", "PAGED_BBND",
        0, 0, 0, -1, -1, metadata, &workspaceSize, &executor);
    aclDestroyIntArray(blockShape);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            return ret;
        }
    }
    ret = aclnnGenericBlockSparseAttentionMetadata(workspace, workspaceSize, executor, stream);
    if (ret == ACL_SUCCESS) {
        ret = aclrtSynchronizeStream(stream);
    }
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    return ret;
}
```
