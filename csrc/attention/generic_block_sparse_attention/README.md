# GenericBlockSparseAttention

## 产品支持情况

- <term>Ascend 950PR&950DT系列产品</term>：支持
- <term>Atlas A3系列产品</term>：支持
- <term>Atlas A2系列产品</term>：支持
- <term>Atlas 200I/500 A2推理产品</term>：不支持
- <term>Atlas推理系列产品</term>：不支持
- <term>Atlas训练系列产品</term>：不支持

## 功能说明

- 算子功能：

  `GenericBlockSparseAttention`是基于CATLASS模板库实现的高性能块稀疏注意力算子，支持沿S轴的块级稀疏模式。通过`sparseBlockIdx`指定每个Q块选择的KV块，通过`sparseBlockCount`指定每个Q块实际保留的KV块数量。调用前须先执行`aclnnGenericBlockSparseAttentionMetadata`生成`metadataOptional`。

  典型调用流程如下：

  1. 准备`query`、`key`、`value`、`sparseBlockIdx`、`sparseBlockCount`等输入。
  2. 调用`aclnnGenericBlockSparseAttentionMetadata`生成`metadataOptional`。
  3. 调用`aclnnGenericBlockSparseAttention`，将上一步得到的`metadataOptional`传入主算子。

- 计算公式：

  稀疏块大小：$blockShapeX \times blockShapeY$

  $$
  attentionOut = Softmax(scaleValue \cdot query \cdot key_{sparse}^{T} + atten\_mask) \cdot value_{sparse}
  $$

  其中$key_{sparse}$、$value_{sparse}$为按`sparseBlockIdx`/`sparseBlockCount`从PA_BBND/PA_BNBD分页缓存或TND/BSND/BNSD连续KV中选取的KV块。

## 参数说明

> **说明：**<br>
> 参数维度含义：B表示Batch Size，T表示Total tokens，N1表示query的head数，N2表示KV的head数，D表示Head Dim，topK表示`sparseBlockIdx`最后一维`maxSparseBlockCount`。<br>
> 各布局均使用上述head数记号。GQA下N1与N2可以不同，约束见下文。

<table style="undefined;table-layout: fixed; width: 1200px"><colgroup>
  <col style="width: 220px">
  <col style="width: 120px">
  <col style="width: 520px">
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
      <td>query</td>
      <td>输入</td>
      <td>公式中的query。layoutQ为"TND"时，shape为[T, N1, D]；BSND为[B,Sq,N1,D]，BNSD为[B,N1,Sq,D]，N1为query的head数。</td>
      <td>FLOAT16、BFLOAT16、FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key</td>
      <td>输入</td>
      <td>公式中的key。PA_BBND时shape为[numBlocks, blockSize, N2, D]；PA_BNBD时shape为[numBlocks, N2, blockSize, D]；TND时为[Tkv,N2,D]；BSND为[B,Sk,N2,D]，BNSD为[B,N2,Sk,D]，N2为KV的head数。</td>
      <td>FLOAT16、BFLOAT16、FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value</td>
      <td>输入</td>
      <td>公式中的value，shape与key一致。</td>
      <td>FLOAT16、BFLOAT16、FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sparseBlockIdx</td>
      <td>输入</td>
      <td>稀疏块索引。layoutSparsePattern=4时，shape为[N2, totalQBlocks, topK]，N2为KV的head数。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sparseBlockCount</td>
      <td>输入</td>
      <td>每个Q块实际选择的KV块数量。layoutSparsePattern=4时，shape为[N2, totalQBlocks]，N2为KV的head数。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>metadataOptional</td>
      <td>输入</td>
      <td>必须传入前置Metadata算子生成的配套调度信息，shape为[1024]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>attenMaskOptional</td>
      <td>可选输入</td>
      <td>保留的掩码输入；当前使用内置因果或非因果模式，传入nullptr。</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>qDequantScaleOptional</td>
      <td>可选输入</td>
      <td>Query反量化系数；当前支持的quantType=0/5场景传入nullptr。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>kDequantScaleOptional</td>
      <td>可选输入</td>
      <td>Key反量化系数；当前支持的quantType=0/5场景传入nullptr。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>vDequantScaleOptional</td>
      <td>可选输入</td>
      <td>Value反量化系数；当前支持的quantType=0/5场景传入nullptr。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pQuantScaleOptional</td>
      <td>可选输入</td>
      <td>P量化系数；当前支持的quantType=0/5场景传入nullptr。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cuSeqLengthsQOptional</td>
      <td>输入</td>
      <td>各batch中query序列长度前缀和，layoutQ为"TND"时必传，shape为[B+1,]。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cuSeqLengthsKvOptional</td>
      <td>输入</td>
      <td>各batch中key/value序列长度前缀和，layoutKv为"TND"时必传，PA_BBND/PA_BNBD时不传；BSND/BNSD时不使用，建议传入nullptr，shape为[B+1,]。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sequsedQOptional</td>
      <td>条件必选输入</td>
      <td>各batch中query实际有效长度，shape为[B,]；BSND/BNSD必传，TND可选且不传时按cu差分得到的存储长度处理。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sequsedKvOptional</td>
      <td>输入</td>
      <td>各batch中kv实际有效长度，shape为[B,]；PA_BBND/PA_BNBD及BSND/BNSD KV必传，TND KV可选。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>blockTableOptional</td>
      <td>输入</td>
      <td>PagedAttention页表，shape为[B, maxNumBlocksPerBatch]；PA_BBND/PA_BNBD必传，TND/BSND/BNSD KV必须为nullptr。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>blockShape</td>
      <td>属性</td>
      <td>稀疏块形状[blockShapeX, blockShapeY]。blockShapeX=1；Atlas A2/A3的Y支持1、2、4、8、16、32、64、128，Ascend 950非量化的Y支持1-128任意整数；量化场景仅支持[1,128]。</td>
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
      <td>query数据排布格式，支持"TND"、"BSND"和"BNSD"，适用范围见<a href="#产品差异约束">产品差异约束</a>。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKv</td>
      <td>属性</td>
      <td>key/value数据排布格式，支持"PA_BBND"、"PA_BNBD"、"TND"、"BSND"或"BNSD"，适用范围见<a href="#产品差异约束">产品差异约束</a>。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>属性</td>
      <td>缩放系数；传0时算子内按$1/\sqrt{D}$处理。</td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskType</td>
      <td>属性</td>
      <td>掩码类型，支持0（非因果）或1（causal），适用范围见<a href="#产品差异约束">产品差异约束</a>。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantType</td>
      <td>属性</td>
      <td>量化类型；支持0或5，适用范围见<a href="#产品差异约束">产品差异约束</a>。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstTypeMax</td>
      <td>属性</td>
      <td>保留参数，当前必须为0。</td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmaxPrecision</td>
      <td>属性</td>
      <td>Softmax计算精度级别，取值0或1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>winLeft</td>
      <td>属性</td>
      <td>滑窗左边界，当前必须为-1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>winRight</td>
      <td>属性</td>
      <td>滑窗右边界，当前必须为-1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>returnSoftmaxlse</td>
      <td>属性</td>
      <td>是否输出softmaxLse。非量化支持0或1；Atlas A2/A3仅在layoutKv为PA_BBND且residualBlockMode=0时支持1；FP8量化仅支持0。</td>
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
      <td>attentionOut</td>
      <td>输出</td>
      <td>shape与query一致；非量化时dtype与query一致，FP8量化输出为FLOAT16或BFLOAT16。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>softmaxLseOptional</td>
      <td>输出</td>
      <td>Softmax log-sum-exp结果；quantType=0时可选输出FLOAT32；TND为[Tq,N1,1]，BSND/BNSD为[B,N1,Sq,1]，关闭时传入nullptr。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 调用前须先执行`aclnnGenericBlockSparseAttentionMetadata`生成`metadataOptional`，再调用本接口；metadata须与当前输入/属性配套，每次调用须重新生成。
- Atlas A2/A3及Ascend 950非量化FLOAT16/BFLOAT16场景支持1～256的任意headDim；Ascend 950 FP8量化场景仅支持128。PA的KV页blockSize固定为128，与稀疏blockShapeY独立。
- TND且layoutSparsePattern=4时：totalQBlocks按cuSeqLengthsQ差分得到的存储长度分块；sparse分块与QKV寻址均按该存储长度，不以seqused重切分；topK须≥`sparseBlockCount`中所有元素的最大值。容量上限：Atlas A2/A3为512，Ascend 950非量化为INT32_MAX，Ascend 950 FP8量化为256。
- Atlas A2/A3的每组有效稀疏索引须非负、严格升序且不重复，位于前`sparseBlockCount`个位置；随机升序即可，不要求等步长或物理页连续。`residualBlockMode=1`时，显式列表只包含当前可见KV范围内的完整块，尾部不完整块由算子隐式加入。
- seqused与对应cu前缀和同时传入时（layoutQ为TND时的Q侧，及layoutKv为TND时的KV侧）：分核/任务空间按各batch实际有效长度（seqused）累加；各batch的seqused元素须≤对应cu存储长度，且须与Metadata侧完全一致。
- layoutKv为"PA_BBND"/"PA_BNBD"时须传`sequsedKvOptional`，不传`cuSeqLengthsKvOptional`。
- 输入query、key、value的数据类型必须一致。
- 输入query的head数为N1，输入key和value的head数为N2，则N1 >= N2且N1 % N2 == 0；groupSize=N1/N2当前须≤128。
- PA_BBND/PA_BNBD下key/value仅dim0（物理页轴）可非连续；页内元素须连续（PA_BBND为blockSize×N2×D，PA_BNBD为N2×blockSize×D）。

### 产品差异约束

- Ascend 950非量化FLOAT16/BFLOAT16场景：`headDim`支持1～256的任意值；`layoutQ/layoutKv`支持`TND/PA_BBND`、`TND/PA_BNBD`、`TND/TND`、`BSND/BSND`、`BNSD/BNSD`、`BSND/PA_BBND`、`BNSD/PA_BBND`、`BSND/PA_BNBD`和`BNSD/PA_BNBD`，不支持其他组合；`maskType`支持0（非因果）和1（causal）；`residualBlockMode`支持0或1。
- Atlas A2/A3非量化FLOAT16/BFLOAT16场景：`headDim`支持1～256的任意值；`layoutQ/layoutKv`支持`TND/PA_BBND`和`TND/TND`；`blockShape=[1,Y]`，Y支持1、2、4、8、16、32、64、128；`maskType`支持0（非因果）和1（causal）；`residualBlockMode`支持0或1。TND KV须传入`cuSeqLengthsKvOptional`，Q/K/V按连续TND存储；causal模式下各batch须满足实际KV长度不小于实际Q长度。
- Ascend 950 FP8量化场景：`headDim`仅支持128，`layoutQ/layoutKv`仅支持`TND/PA_BBND`，`maskType=1`、`residualBlockMode=0`、`blockShape=[1,128]`。
- 稀疏索引最后一维容量：Atlas A2/A3支持1～512，Ascend 950非量化FLOAT16/BFLOAT16支持1～INT32_MAX，Ascend 950 FP8量化支持1～256；容量均须不小于稀疏块有效计数的最大值。
- `quantType`：Ascend 950支持0（非量化）或5（FP8量化）；Atlas A2/A3仅支持0。
- `softmaxPrecision`：Ascend 950非量化FLOAT16/BFLOAT16场景支持0或1，FP8量化场景仅支持1；Atlas A2/A3上FLOAT16支持0或1，BFLOAT16仅支持0。

### A2/A3实现

- 内部KV基块为256 token，独立于用户传入的稀疏块Y和PA页大小128；调用方无需设置基块。
- K按128-token子块生成搬运记录，V复用记录；连续片段可合并，满足地址和步长条件的两个稀疏块可用pair搬运。随机索引或随机页表不满足合并条件时使用对应回退路径。
- 满足条件的宽tile路径支持分段V搬运与PV计算重叠；QK的K搬运使用双缓冲。分核任务使用每核游标和KV256阶段记录。
- 搬运记录、宽tile和流水线属于内部实现，不改变稀疏索引、残块或输出语义。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| -------- | -------- | ---- |
| aclnn API | [test_aclnn_generic_block_sparse_attention](./examples/test_aclnn_generic_block_sparse_attention.cpp) | 通过[aclnnGenericBlockSparseAttention](./docs/aclnnGenericBlockSparseAttention.md)两段式接口调用GenericBlockSparseAttention算子 |
| PyTorch API | - | 通过[generic_block_sparse_attention](../../attention/generic_block_sparse_attention/docs/torchapi_generic_block_sparse_attention.md)接口调用generic_block_sparse_attention算子。 |
