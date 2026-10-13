# GenericBlockSparseAttentionMetadata

## 产品支持情况

| 产品 | 是否支持 |
| ------------------------------------------------------------ | :------: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：

  `GenericBlockSparseAttentionMetadata`是`GenericBlockSparseAttention`的前置算子，根据Query序列长度、Q/KV头数和稀疏块信息生成主算子使用的分核结果`metadataOptional`。本算子不执行Attention计算。

  典型调用流程如下：

  1. 准备`sparseBlockIdx`、`sparseBlockCount`、序列长度和属性等输入。
  2. 调用`aclnnGenericBlockSparseAttentionMetadata`生成`metadataOptional`。
  3. 调用`aclnnGenericBlockSparseAttention`，将上一步得到的`metadataOptional`作为主算子输入。

## 参数说明

> **说明：**<br>
> B表示Batch Size，`totalQBlocks`表示按各Batch的Query存储长度划分后的Q块总数，`maxSparseBlockCount`表示`sparseBlockIdx`的最后一维。

<table style="undefined;table-layout: fixed; width: 1200px"><colgroup>
  <col style="width: 220px">
  <col style="width: 130px">
  <col style="width: 550px">
  <col style="width: 180px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>sparseBlockIdx</td>
      <td>输入</td>
      <td>稀疏块索引。当前shape为[numKeyValueHeads, totalQBlocks, maxSparseBlockCount]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sparseBlockCount</td>
      <td>输入</td>
      <td>每个Q块实际选择的KV块数量，当前shape为[numKeyValueHeads, totalQBlocks]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cuSeqLengthsQOptional</td>
      <td>输入</td>
      <td>各Batch中Query序列存储长度的前缀和，当前必须传入，shape为(B+1,)。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cuSeqLengthsKvOptional</td>
      <td>输入</td>
      <td>各Batch中Key/Value序列存储长度的前缀和，layoutKv为"TND"时必传，非TND时不传，shape为(B+1,)。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sequsedQOptional</td>
      <td>可选输入</td>
      <td>各Batch中Query的实际序列长度，shape为(B,)；不传时使用Query存储长度。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sequsedKvOptional</td>
      <td>输入</td>
      <td>各Batch中Key/Value的实际序列长度，layoutKv为"PA_BBND"时必传，shape为(B,)。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>maxQSeqLen</td>
      <td>属性</td>
      <td>主算子Query的最大Sequence Length，必须大于0。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxKvSeqLen</td>
      <td>属性</td>
      <td>主算子Key/Value的最大Sequence Length，必须大于0。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numQHeads</td>
      <td>属性</td>
      <td>主算子Query的head数。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numKvHeads</td>
      <td>属性</td>
      <td>主算子Key/Value的head数。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headDim</td>
      <td>属性</td>
      <td>主算子Query、Key和Value每个head的特征维度，当前仅支持128。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockShape</td>
      <td>属性</td>
      <td>稀疏块形状[blockShapeX, blockShapeY]。Arch35非量化支持[1,Y]，Y为1～256任意整数；其余场景仅支持[1,128]。须与主算子一致。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>isPackedGQA</td>
      <td>属性</td>
      <td>同一个group内的qHead是否共享相同的稀疏pattern，当前仅支持1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQ</td>
      <td>属性</td>
      <td>主算子Query的数据排布格式，当前仅支持"TND"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKv</td>
      <td>属性</td>
      <td>主算子Key/Value的数据排布格式，当前仅支持"PA_BBND"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskType</td>
      <td>属性</td>
      <td>Attention计算的掩码类型，当前仅支持1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantType</td>
      <td>属性</td>
      <td>量化类型，当前支持0；Ascend 950上可选5。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmaxPrecision</td>
      <td>属性</td>
      <td>Softmax计算精度级别，取值为0或1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>winLeft</td>
      <td>属性</td>
      <td>滑窗Attention向前包含的token数，当前仅支持-1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>winRight</td>
      <td>属性</td>
      <td>滑窗Attention向后包含的token数，当前仅支持-1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadataOptional</td>
      <td>输出</td>
      <td>主算子使用的分核结果，shape固定为(1024,)。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 本算子必须与`GenericBlockSparseAttention`配套使用。所有对应输入和属性须与随后调用的主算子一致，每次调用主算子前均须重新生成`metadataOptional`。
- 支持`layoutQ="TND"`、`layoutKv="PA_BBND"`（Arch35非量化另支持`TND`）、`isPackedGQA=1`、`blockShapeX=1`和`headDim=128`。Arch35非量化支持blockShapeY为1～256任意整数，新Y生成普通分核调度；Y=128保留现有FD条件。
- `numQHeads >= numKvHeads`且`numQHeads % numKvHeads == 0`，`numQHeads / numKvHeads`不超过128。
- `sparseBlockIdx`最后一维须不小于`sparseBlockCount`中所有元素的最大值，且当前不超过256。
- 输出`metadataOptional`是固定为INT32、shape为(1024,)的不透明数据，不得解析、修改或跨不同输入和属性复用。
- 完整约束见[aclnnGenericBlockSparseAttentionMetadata](./docs/aclnnGenericBlockSparseAttentionMetadata.md)。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| -------- | -------- | ---- |
| aclnn API | [test_aclnn_generic_block_sparse_attention_metadata](./examples/test_aclnn_generic_block_sparse_attention_metadata.cpp) | 通过[aclnnGenericBlockSparseAttentionMetadata](./docs/aclnnGenericBlockSparseAttentionMetadata.md)两段式接口调用GenericBlockSparseAttentionMetadata算子。 |
