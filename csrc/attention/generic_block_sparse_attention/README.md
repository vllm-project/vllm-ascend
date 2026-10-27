# GenericBlockSparseAttention

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                              |    ×    |
| <term>Atlas 训练系列产品</term>                              |    ×    |

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

  其中$key_{sparse}$、$value_{sparse}$为按`sparseBlockIdx`/`sparseBlockCount`从Paged KV Cache中选取的KV块。

## 参数说明

> **说明：**<br>
> 参数维度含义：B表示Batch Size，T表示Total tokens，N表示Head Num，D表示Head Dim，topK表示`sparseBlockIdx`最后一维`maxSparseBlockCount`。<br>
> TND中的N为query的headNum（记为N1），PA_BBND中的N为key/value的headNum（记为N2）。GQA下N1与N2可以不同，约束见下文。

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
      <td>公式中的query。layoutQ为"TND"时，shape为[T, N, D]，N为query的headNum（N1）。</td>
      <td>FLOAT16、BFLOAT16、FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key</td>
      <td>输入</td>
      <td>公式中的key。layoutKv为"PA_BBND"时，shape为[numBlocks, blockSize, N, D]，N为kv的headNum（N2）。</td>
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
      <td>稀疏块索引。TND + isPackedGQA=1时，shape为[N, totalQBlocks, topK]，N为kv的headNum（N2）。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sparseBlockCount</td>
      <td>输入</td>
      <td>每个Q块实际选择的KV块数量。TND + isPackedGQA=1时，shape为[N, totalQBlocks]，N为kv的headNum（N2）。</td>
      <td>INT32</td>
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
      <td>各batch中key/value序列长度前缀和，layoutKv为"TND"时必传，非TND时不传，shape为[B+1,]。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sequsedQOptional</td>
      <td>可选输入</td>
      <td>各batch中query实际有效长度；不传时按cu前缀和差分得到的存储长度处理。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sequsedKvOptional</td>
      <td>输入</td>
      <td>各batch中kv实际有效长度，layoutKv为"PA_BBND"时必传，shape为[B,]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>blockTableOptional</td>
      <td>输入</td>
      <td>PagedAttention页表，shape为[B, maxNumBlocksPerBatch]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>blockShape</td>
      <td>属性</td>
      <td>稀疏块形状[blockShapeX, blockShapeY]。Arch35非量化FP16/BF16支持[1, Y]，Y为1～256任意整数；其他场景仅支持[1,128]。Attention KV tile和PA page均固定为128。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>isPackedGQA</td>
      <td>属性</td>
      <td>同group内qHead是否共享稀疏pattern，当前仅支持1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQ</td>
      <td>属性</td>
      <td>query数据排布格式，当前仅支持"TND"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKv</td>
      <td>属性</td>
      <td>key/value数据排布格式，当前仅支持"PA_BBND"。</td>
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
      <td>掩码类型，当前仅支持1（内置causal mask）。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantType</td>
      <td>属性</td>
      <td>量化类型；当前支持0，Ascend 950上可选5。</td>
      <td>INT64</td>
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
      <td>returnSoftmaxlse</td>
      <td>属性</td>
      <td>是否输出softmaxLse，当前仅支持0。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attentionOut</td>
      <td>输出</td>
      <td>公式中的attentionOut，数据类型和shape与query保持一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>softmaxLseOptional</td>
      <td>输出</td>
      <td>Softmax log-sum-exp结果；quantType=0时可选输出FLOAT32 [Tq,Nq,1]，关闭时传入nullptr。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- A5新增TND非PA代码路径（尚未运行构建或设备验证）：layoutQ/layoutKv均为TND，Q/K/V为FLOAT16或BFLOAT16，quantType=0；Q/KV各自必传cu前缀，seqused可选，blockTable必须为空。详见[设计与实现范围](./docs/a5_tnd_non_pa_design.md)。

- 调用前须先执行`aclnnGenericBlockSparseAttentionMetadata`生成`metadataOptional`，再调用本接口；metadata须与当前输入/属性配套，每次调用须重新生成。
- query/key/value的headDim(D)当前仅支持128；KV页blockSize当前仅支持128，且须等于blockShapeY。
- TND + isPackedGQA=1时：totalQBlocks按cuSeqLengthsQ差分得到的存储长度分块；sparse分块与QKV寻址均按该存储长度，不以seqused重切分；topK须≥`sparseBlockCount`中所有元素的最大值，当前上限为256。
- seqused与对应cu前缀和同时传入时（Q侧，及layoutKv为TND时的KV侧）：分核/任务空间按各batch实际有效长度（seqused）累加；各batch的seqused元素须≤对应cu存储长度，且须与Metadata侧完全一致。
- layoutKv为"PA_BBND"时须传`sequsedKvOptional`，不传`cuSeqLengthsKvOptional`。
- 输入query、key、value的数据类型必须一致。
- 输入query的headNum为N1，输入key和value的headNum为N2，则N1 >= N2且N1 % N2 == 0；groupSize=N1/N2当前须≤128。
- PA_BBND下key/value仅dim0（物理页轴）可非连续；页内blockSize×N2×D须连续。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| -------- | -------- | ---- |
| aclnn API | [test_aclnn_generic_block_sparse_attention](./examples/test_aclnn_generic_block_sparse_attention.cpp) | 通过[aclnnGenericBlockSparseAttention](./docs/aclnnGenericBlockSparseAttention.md)两段式接口调用GenericBlockSparseAttention算子 |
| PyTorch API | - | 通过[generic_block_sparse_attention](../../attention/generic_block_sparse_attention/docs/torchapi_generic_block_sparse_attention.md)接口调用generic_block_sparse_attention算子。 |
