# GenericBlockSparseAttentionMetadata

## 产品支持情况

- <term>Ascend 950PR&950DT系列产品</term>：支持
- <term>Atlas A3系列产品</term>：支持
- <term>Atlas A2系列产品</term>：支持
- <term>Atlas 200I/500 A2推理产品</term>：不支持
- <term>Atlas推理系列产品</term>：不支持
- <term>Atlas训练系列产品</term>：不支持

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
      <td>各Batch中Query序列存储长度的前缀和，TND必须传入；BSND/BNSD不使用，建议传入nullptr，shape为(B+1,)。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cuSeqLengthsKvOptional</td>
      <td>输入</td>
      <td>各Batch中Key/Value序列存储长度的前缀和，layoutKv为"TND"时必传，PA_BBND/PA_BNBD时不传；BSND/BNSD时不使用，建议传入nullptr，shape为(B+1,)。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sequsedQOptional</td>
      <td>条件必选输入</td>
      <td>各Batch中Query的实际序列长度，shape为(B,)；BSND/BNSD必传，TND可选且不传时使用Query存储长度。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sequsedKvOptional</td>
      <td>输入</td>
      <td>各Batch中Key/Value的实际序列长度，shape为(B,)；PA_BBND/PA_BNBD及BSND/BNSD KV必传，TND KV可选。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>maxQSeqLen</td>
      <td>属性</td>
      <td>Query最大序列长度；TND允许-1（未指定）或非负值；BSND/BNSD必须等于shape容量Sq且大于0。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxKvSeqLen</td>
      <td>属性</td>
      <td>Key/Value最大序列长度；允许-1（未指定）或非负值，不用于推导BSND/BNSD的实际KV长度。</td>
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
      <td>主算子Query、Key和Value每个head的特征维度。headDim取值范围见<a href="#产品差异约束">产品差异约束</a>。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockShape</td>
      <td>属性</td>
      <td>稀疏块形状[blockShapeX, blockShapeY]。blockShapeX=1；Atlas A2/A3的Y支持1、2、4、8、16、32、64、128，Ascend 950非量化的Y支持1-128任意整数；量化场景仅支持[1,128]。须与主算子一致。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutSparsePattern</td>
      <td>属性</td>
      <td>稀疏pattern布局，当前仅支持4；同一KV head对应的Q heads共享稀疏pattern。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQ</td>
      <td>属性</td>
      <td>主算子Query的数据排布格式，支持"TND"、"BSND"和"BNSD"，适用范围见<a href="#产品差异约束">产品差异约束</a>。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKv</td>
      <td>属性</td>
      <td>主算子Key/Value的数据排布格式，支持"PA_BBND"、"PA_BNBD"、"TND"、"BSND"或"BNSD"，适用范围见<a href="#产品差异约束">产品差异约束</a>。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskType</td>
      <td>属性</td>
      <td>Attention计算的掩码类型，支持0（非因果）或1（causal），适用范围见<a href="#产品差异约束">产品差异约束</a>。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantType</td>
      <td>属性</td>
      <td>量化类型，支持0或5，适用范围见<a href="#产品差异约束">产品差异约束</a>。</td>
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
      <td>residualBlockMode</td>
      <td>属性</td>
      <td>0：尾部不完整块由显式稀疏列表决定；1：隐式加入尾部不完整块，该块不包含在显式列表中。取值的适用范围见<a href="#产品差异约束">产品差异约束</a>。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>isConsistentTopk</td>
      <td>属性</td>
      <td>同一batch、同一head内，各Q块选择的KV块最大数量是否一致。取值false或true，须与主算子/Metadata保持一致。</td>
      <td>BOOL</td>
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
- 布局组合及适用范围见[产品差异约束](#产品差异约束)。`layoutSparsePattern=4`、`blockShapeX=1`；headDim范围见[产品差异约束](#产品差异约束)，blockShapeY范围见上方参数表。
- `numQHeads >= numKvHeads`且`numQHeads % numKvHeads == 0`，`numQHeads / numKvHeads`不超过128。
- `sparseBlockIdx`最后一维须不小于`sparseBlockCount`中所有元素的最大值，且须大于0。容量上限：Atlas A2/A3为512，Ascend 950非量化为INT32_MAX，Ascend 950 FP8量化为256。
- Atlas A2/A3的每组有效稀疏索引须非负、严格升序且不重复，位于前`sparseBlockCount`个位置；随机升序即可，不要求等步长或物理页连续。`residualBlockMode=1`时显式列表仅包含当前可见范围内的完整KV块，不包含隐式尾块。
- 输出`metadataOptional`是固定为INT32、shape为(1024,)的不透明数据，不得解析、修改或跨不同输入和属性复用。
- 完整约束见[aclnnGenericBlockSparseAttentionMetadata](./docs/aclnnGenericBlockSparseAttentionMetadata.md)。

### 产品差异约束

- Ascend 950非量化FLOAT16/BFLOAT16场景：`headDim`支持1～256的任意值；`layoutQ/layoutKv`支持`TND/PA_BBND`、`TND/PA_BNBD`、`TND/TND`、`BSND/BSND`、`BNSD/BNSD`、`BSND/PA_BBND`、`BNSD/PA_BBND`、`BSND/PA_BNBD`和`BNSD/PA_BNBD`，不支持其他组合；`maskType`支持0（非因果）和1（causal）；`residualBlockMode`支持0或1。
- Atlas A2/A3非量化FLOAT16/BFLOAT16场景：`headDim`支持1～256的任意值；`layoutQ/layoutKv`支持`TND/PA_BBND`和`TND/TND`；`blockShape=[1,Y]`，Y支持1、2、4、8、16、32、64、128；`maskType`支持0（非因果）和1（causal）；`residualBlockMode`支持0或1。TND KV须传入`cuSeqLengthsKvOptional`；causal模式下各batch须满足实际KV长度不小于实际Q长度。
- Ascend 950 FP8量化场景：`headDim`仅支持128，`layoutQ/layoutKv`仅支持`TND/PA_BBND`，`maskType=1`、`residualBlockMode=0`、`blockShape=[1,128]`。
- 稀疏索引最后一维容量：Atlas A2/A3支持1～512，Ascend 950非量化FLOAT16/BFLOAT16支持1～INT32_MAX，Ascend 950 FP8量化支持1～256；容量均须不小于稀疏块有效计数的最大值。
- `quantType`：Ascend 950支持0（非量化）或5（FP8量化）；Atlas A2/A3仅支持0。
- `softmaxPrecision`：Ascend 950非量化FLOAT16/BFLOAT16场景支持0或1，FP8量化场景仅支持1；Atlas A2/A3上FLOAT16支持0或1，BFLOAT16仅支持0。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| -------- | -------- | ---- |
| aclnn API | [test_aclnn_generic_block_sparse_attention_metadata](./examples/test_aclnn_generic_block_sparse_attention_metadata.cpp) | 通过[aclnnGenericBlockSparseAttentionMetadata](./docs/aclnnGenericBlockSparseAttentionMetadata.md)两段式接口调用GenericBlockSparseAttentionMetadata算子。 |
