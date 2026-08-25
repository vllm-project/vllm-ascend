# GenericBlockSparseAttention 改造方案

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                  |    ×     |
| <term>Atlas 推理系列产品</term>                          |    ×     |
| <term>Atlas 训练系列产品</term>                          |    ×     |

## 功能说明

本系统由以下接口组成：

- **attention metadata接口** `aclnnGenericBlockSparseAttentionMetadata`：根据idx、count、seqlen等信息进行稀疏attention的分核与负载均衡

- **attention接口** `aclnnGenericBlockSparseAttention`：核心稀疏attention计算

- **接口功能**：支持沿着S轴任意粒度的稀疏注意力计算，通过sparseBlockIdx指定每个Q块保留的KV块，sparseBlockCount指定每个Q块需要多少KV保留块，实现高效的稀疏注意力计算。

- **计算公式**：稀疏块大小：$blockShapeX \times blockShapeY$，selectIdx指定稀疏模式

  $$
  attentionOut = Softmax(scale \cdot query \cdot key_{sparse}^T + atten\_mask) \cdot value_{sparse}
  $$

  输入query、key、value的数据排布格式支持从多种维度排布解读，可通过layoutQ和layoutKv传入。

    - B：表示输入样本批量大小（Batch）
    - T：B和S合轴紧密排列的长度（Total tokens）
    - S：表示输入样本序列长度（Seq-Length）
    - H：表示隐藏层的大小（Head-Size）
    - N：表示多头数（Head-Num）
    - D：表示隐藏层最小的单元尺寸，需满足D=H/N（Head-Dim）

  当前支持的布局：

    - layoutQ: "TND" "BNSD" "BSND"
    - layoutKv: "TND" "BNSD" "BSND" "PAGED_BBND" "PAGED_BNBD"

## 1. aclnnGenericBlockSparseAttention

### GenericBlockSparseAttention 函数说明

每个算子分为[两段式接口](https://gitcode.com/cann/ops-transformer/blob/master/docs/zh/context/%E4%B8%A4%E6%AE%B5%E5%BC%8F%E6%8E%A5%E5%8F%A3.md)，必须先调用"aclnnGenericBlockSparseAttentionGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnGenericBlockSparseAttention"接口执行计算。

第一段接口：

```cpp
__attribute__((visibility("default"))) aclnnStatus aclnnGenericBlockSparseAttentionGetWorkspaceSize(
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
    int64_t returnSoftmaxLse,
    aclTensor *attentionOut,
    aclTensor *softmaxLseOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

第二段接口：

```cpp
__attribute__((visibility("default"))) aclnnStatus aclnnGenericBlockSparseAttention(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

### aclnnGenericBlockSparseAttentionGetWorkspaceSize 参数说明

- **参数说明**

<table style="table-layout: fixed; width: 2000px">
  <colgroup>
    <col style="width: 150px">
    <col style="width: 100px">
    <col style="width: 200px">
    <col style="width: 246px">
    <col style="width: 275px">
    <col style="width: 101px">
    <col style="width: 190px">
    <col style="width: 146px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>query</td>
      <td>输入，必选</td>
      <td>Device侧的aclTensor，公式中的query。</td>
      <td>支持的shape为：
 <ul>
          <li>TND: [totalQTokens, headNum, headDim]。</li>
          <li>BNSD: [batch, headNum, maxQSeqLength, headDim]。</li>
    <li>BSND: [batch, maxQSeqLength, headNum, headDim]。</li>
        </ul>
      </td>
      <td>FLOAT16/BFLOAT16/FLOAT8_E4M3FN/FLOAT4_E2M1FN/HIFLOAT8</td>
      <td>ND</td>
      <td>3/4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>key</td>
      <td>输入，必选</td>
      <td>Device侧的aclTensor，公式中的key。</td>
      <td>
  作为原始Key输入时，支持的shape为：
 <ul>
          <li>TND: [totalKTokens, numKeyValueHeads, headDim]。</li>
          <li>BNSD: [batch, numKeyValueHeads, maxKvSeqLength, headDim]。</li>
    <li>BSND: [batch, maxKvSeqLength, numKeyValueHeads, headDim]。</li>
        </ul>
  作为Key Cache输入时，支持的shape为：
 <ul>
          <li>PAGED_BBND: [numBlocks, blockSize, numKeyValueHeads, headDim]。</li>
          <li>PAGED_BNBD: [numBlocks, numKeyValueHeads, blockSize, headDim]。</li>
        </ul>
      </td>
      <td>FLOAT16/BFLOAT16/FLOAT8_E4M3FN/FLOAT4_E2M1FN/HIFLOAT8</td>
      <td>ND</td>
      <td>3/4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>value</td>
      <td>输入，必选</td>
      <td>Device侧的aclTensor，公式中的value。</td>
      <td>
  作为原始Value输入时，支持的shape为：
 <ul>
          <li>TND: [totalVTokens, numKeyValueHeads, headDim]。</li>
          <li>BNSD: [batch, numKeyValueHeads, maxKvSeqLength, headDim]。</li>
    <li>BSND: [batch, maxKvSeqLength, numKeyValueHeads, headDim]。</li>
        </ul>
  作为Value Cache输入时，支持的shape为：
 <ul>
          <li>PAGED_BBND: [numBlocks, blockSize, numKeyValueHeads, headDim]。</li>
          <li>PAGED_BNBD: [numBlocks, numKeyValueHeads, blockSize, headDim]。</li>
        </ul>
  其中blockSize为cache的页的大小，支持范围[16, 512]，需要满足16对齐
      </td>
      <td>FLOAT16/BFLOAT16/FLOAT8_E4M3FN/FLOAT4_E2M1FN/HIFLOAT8</td>
      <td>ND</td>
      <td>3/4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sparseBlockIdx</td>
      <td>输入，必选</td>
      <td>Device侧的aclTensor，稀疏块索引数组，指定每个Q块选择的KV块索引。</td>
      <td>
  存储每个Q块选择的KV块索引，支持的shape随query布局变化：
  <ul>
    <li>query为TND布局时：</li>
    <ul>
   <li>每个qHead对应的KV稀疏pattern不一致（isPackedGQA=0）：<br>[headNum, totalQBlocks, maxSparseBlockCount]</li>
   <li>GQA/MQA下，同group每个qHead对应的KV稀疏pattern一致（isPackedGQA=1）：<br>[numKeyValueHeads, totalQBlocks, maxSparseBlockCount]</li>
    </ul>
    <li>query为BNSD/BSND布局时：</li>
    <ul>
   <li>每个qHead对应的KV稀疏pattern不一致（isPackedGQA=0）：<br>[batch, headNum, ceilDiv(maxQSeqLength, blockShapeX), maxSparseBlockCount]</li>
   <li>GQA/MQA下，同group每个qHead对应的KV稀疏pattern一致（isPackedGQA=1）：<br>[batch, numKeyValueHeads, ceilDiv(maxQSeqLength, blockShapeX), maxSparseBlockCount]</li>
    </ul>
  </ul>
  其中totalQBlocks = Σ ceilDiv(qSeqLen_i, blockShapeX)，i为batch索引，qSeqLen_i由cuSeqLengthsQOptional指定。
<br>maxSparseBlockCount为sparseBlockCount tensor中所有元素的最大值，即所有Q块选择的KV块数量的最大值。传入值只需 >= 该最大值即可，不限制上限。
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
 <tr>
      <td>sparseBlockCount</td>
      <td>输入，必选</td>
      <td>Device侧的aclTensor，每个Q块实际选择的KV块数量。</td>
      <td>
  存储每个Q块实际选择的KV块数量，支持的shape随query布局变化：
  <ul>
    <li>query为TND布局时：</li>
    <ul>
   <li>每个qHead对应的KV稀疏pattern不一致（isPackedGQA=0）：<br>[headNum, totalQBlocks]</li>
   <li>GQA/MQA下，同group每个qHead对应的KV稀疏pattern一致（isPackedGQA=1）：<br>[numKeyValueHeads, totalQBlocks]</li>
    </ul>
    <li>query为BNSD/BSND布局时：</li>
    <ul>
   <li>每个qHead对应的KV稀疏pattern不一致（isPackedGQA=0）：<br>[batch, headNum, ceilDiv(maxQSeqLength, blockShapeX)]</li>
   <li>GQA/MQA下，同group每个qHead对应的KV稀疏pattern一致（isPackedGQA=1）：<br>[batch, numKeyValueHeads, ceilDiv(maxQSeqLength, blockShapeX)]</li>
    </ul>
  </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
 <tr>
      <td>metadataOptional</td>
      <td>输入，可选</td>
      <td>Device侧的aclTensor，稀疏attention的分核信息。</td>
      <td>
        由aicpu算子计算得出
      </td>
      <td>INT64</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>attenMaskOptional</td>
      <td>输入，可选</td>
      <td>Device侧的aclTensor，公式中的atten_mask。</td>
      <td>atten_mask会与稀疏pattern叠加产生作用。</td>
      <td>BOOL</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
 <tr>
      <td>qDequantScaleOptional</td>
      <td>输入，可选</td>
      <td>Device侧的aclTensor，query的反量化缩放因子。</td>
      <td>详情见“量化相关说明”
      </td>
      <td>FLOAT32/FLOAT8_E8M0</td>
      <td>ND</td>
      <td>x</td>
      <td>×</td>
    </tr>
    <tr>
      <td>kDequantScaleOptional</td>
      <td>输入，可选</td>
      <td>Device侧的aclTensor，key的反量化缩放因子。</td>
      <td>详情见“量化相关说明”
      </td>
      <td>FLOAT32/FLOAT8_E8M0</td>
      <td>ND</td>
      <td>x</td>
      <td>×</td>
    </tr>
    <tr>
      <td>vDequantScaleOptional</td>
      <td>输入，可选</td>
      <td>Device侧的aclTensor，value的反量化缩放因子。</td>
      <td>详情见“量化相关说明”
      </td>
      <td>FLOAT32/FLOAT8_E8M0</td>
      <td>ND</td>
      <td>x</td>
      <td>×</td>
    </tr>
 <tr>
      <td>pQuantScaleOptional</td>
      <td>输入，可选</td>
      <td>非mx量化模式下，online-softmax的结果P矩阵所需的量化系数。</td>
      <td>详情具体见“量化相关说明”。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>x</td>
      <td>x</td>
    </tr>
    <tr>
      <td>cuSeqLengthsQOptional</td>
      <td>输入，可选</td>
      <td>Device侧的aclTensor，描述每个Batch对应的query序列长度，以前缀和形式存储。</td>
      <td>可选输入，用于变长序列场景：
        <ul>
          <li>当layoutQ为"TND"时：该项输入必须配置</li>
          <li>当layoutQ为"BNSD""BSND"时：如配置该项输入，算子内会按该输入指定的实际序列长度进行处理；<br>如不配置该项输入(传入nullptr)，算子内会按照query的shape中的S进行处理。</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cuSeqLengthsKvOptional</td>
      <td>输入，可选</td>
      <td>Device侧的aclTensor，描述每个Batch对应的key/value序列长度，以前缀和形式存储。</td>
      <td>可选输入，用于变长序列场景：
        <ul>
          <li>当layoutKv为"TND"/"PAGED_BBND"/"PAGED_BNBD"时：该项输入必须配置</li>
          <li>当layoutKv为"BNSD""BSND"时：如配置该项输入，算子内会按该输入指定的实际序列长度进行处理；<br>如不配置该项输入(传入nullptr)，算子内会按照key/value的shape中的S进行处理。</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>1</td>
      <td>-</td>
    </tr>
    </tr>
    <tr>
      <td>sequsedQOptional</td>
      <td>输入，可选</td>
      <td>Device侧的aclTensor，各batch中query的实际序列长度。</td>
      <td>与cuSeqLengthsQOptional互斥。
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedKvOptional</td>
      <td>输入，可选</td>
      <td>Device侧的aclTensor，各batch中kv的实际序列长度。</td>
      <td>与cuSeqLengthsKvOptional互斥。
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>    <tr>
      <td>blockTableOptional</td>
      <td>输入，可选</td>
      <td>Device侧的aclTensor，Block表用于PagedAttention。</td>
      <td>如配置此输入，则表示使用paged attention，layout、KV shape均需对应配置。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
 <tr>
      <td>blockShape</td>
      <td>输入，Attr</td>
      <td>代表稀疏块形状数组。</td>
      <td>含两个元素[blockShapeX, blockShapeY]。<br>blockShapeX支持任意值，不可超过int64表示范围。<br>blockShapeY支持按16对齐的任意值，不可超过int64表示范围。<br>开启量化功能时的约束具体见“量化相关说明”。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
 <tr>
      <td>isPackedGQA</td>
      <td>输入，Attr</td>
      <td>代表进行块状稀疏时，同一个group内的qHead是否共享同样的稀疏pattern<br>（注：不同batch之间不会共享同样的稀疏pattern，该入参仅区分head维度的共享情况）。</td>
      <td>若取值为0，则代表同一个group内的qHead不共享同样的稀疏pattern；<br>若取值为1，则代表同一个group内的qHead共享同样的稀疏pattern。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQ</td>
      <td>输入，Attr</td>
      <td>Host侧的string，代表输入query的数据排布格式。</td>
      <td>当前支持"TND""BNSD""BSND"。</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKv</td>
      <td>输入，Attr</td>
      <td>Host侧的string，代表输入key、value的数据排布格式。</td>
      <td>当前仅支持"TND""BNSD""BSND""PAGED_BBND""PAGED_BNBD"，详情见layout说明。</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
 <tr>
      <td>scaleValue</td>
      <td>输入，Attr</td>
      <td>Host侧的double，公式中的scale，代表缩放系数。</td>
      <td>一般设置为D^-0.5。</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskType</td>
      <td>输入，Attr</td>
      <td>Host侧的int64_t，表示attention计算中的掩码类型。</td>
      <td>
 具体见“掩码相关说明”
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantType</td>
      <td>输入，Attr</td>
      <td>代表采用的量化手段。</td>
      <td>具体见“量化相关说明”。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
 <tr>
      <td>dstTypeMax</td>
      <td>输入，Attr</td>
      <td>MXFP4 CX量化时,传入的自定义量化量程。</td>
      <td>
        当前仅当quantType=3，4时候，支持设置该值0.0，或者[6.0, 12.0]。
        <ul>
          <li>0.0：代表Amax(DType)为量化结果数据类型的最大值。</li>
          <li>[6.0, 12.0]：取值为6.0-12.0代表Amax(DType)为传入值。</li>
        </ul>
      </td>
      <td>double</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmaxPrecision</td>
      <td>输入，Attr</td>
      <td>Softmax计算采取的精度级别。</td>
      <td>
        控制online softmax阶段以及rescale阶段运算使用的数据类型。<br>当前只支持传0或1
        <ul>
          <li>0：表示online softmax和rescale全部采取fp32数据类型，适合追求计算精度的场景使用。</li>
          <li>1：表示混合精度运算，在性能与精度上取得一个折中。<br>online softmax采取fp16/bf16数据类型（与attentionOut数据类型相同），rescale采取fp32数据类型，<br>在online softmax阶段可能发生数值溢出。</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>winLeft</td>
      <td>输入，Attr</td>
      <td>Host侧的int64_t，滑窗attention场景下，滑窗需要向前包含多少个token。</td>
      <td>用于滑窗attention场景，不使能时必须为-1，需要与maskType，mask配合使用，具体见“掩码说明”。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>winRight</td>
      <td>输入，Attr</td>
      <td>Host侧的int64_t，滑窗attention场景下，滑窗需要向后包含多少个token。</td>
      <td>用于滑窗attention场景，不使能时必须为-1，需要与maskType，mask配合使用，具体见“掩码说明”。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>returnSoftmaxLse</td>
      <td>输入，Attr</td>
      <td>Host侧的int64_t，是否使能softmaxLse输出的标志位。</td>
      <td>
 当前只支持传0或1
 <ul>
          <li>0：表示不输出softmaxLse</li>
   <li>1：表示输出softmaxLse，相比不输出softmaxLse可能存在性能损失</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
 <tr>
      <td>attentionOut</td>
      <td>输出</td>
      <td>Device侧的aclTensor，公式中的attentionOut。</td>
      <td>数据类型和shape与query保持一致。</td>
      <td>FLOAT16/BFLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>softmaxLseOptional</td>
      <td>输出</td>
      <td>Device侧的aclTensor，Softmax计算的log-sum-exp中间结果。</td>
      <td>支持的shape随着query的shape改变：
 <ul>
          <li>query为"TND": [totalQTokens, headNum, 1]。</li>
          <li>query为"BNSD": [batch, headNum, maxQSeqLength, 1]。</li>
    <li>query为"BSND": [batch, maxQSeqLength, headNum, 1]。</li>
        </ul>
      </td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回op执行器，包含算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
</tbody>
</table>

- **返回值**
  aclnnStatus：
  返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

### aclnnGenericBlockSparseAttention 参数说明

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnGenericBlockSparseAttentionGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值**
  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

### layout对应关系说明

<table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
  </colgroup>
  <thead>
    <tr>
      <th>layoutQ</th>
      <th>layoutKv</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">TND</td>
      <td>TND</td>
      <td>用于原始KV输入，每个batch的seqlen拼接在一起的场景。</td>
    </tr>
    <tr>
      <td>PAGED_BBND</td>
      <td>用于paged kv cache输入，数据按[numBlocks, blockSize, numKeyValueHeads, headDim]排布。</td>
    </tr>
 <tr>
      <td>PAGED_BNBD</td>
      <td>用于paged kv cache输入，数据按[numBlocks, numKeyValueHeads, blockSize, headDim]排布。</td>
    </tr>
    <tr>
      <td rowspan="3">BSND</td>
      <td>BSND</td>
      <td>用于原始KV输入。</td>
    </tr>
 <tr>
      <td>PAGED_BBND</td>
      <td>用于paged kv cache输入，数据按[numBlocks, blockSize, numKeyValueHeads, headDim]排布。</td>
    </tr>
 <tr>
      <td>PAGED_BNBD</td>
      <td>用于paged kv cache输入，数据按[numBlocks, numKeyValueHeads, blockSize, headDim]排布。</td>
    </tr>
    <tr>
      <td rowspan="3">BNSD</td>
      <td>BNSD</td>
      <td>用于原始KV输入。</td>
    </tr>
 <tr>
      <td>PAGED_BBND</td>
      <td>用于paged kv cache输入，数据按[numBlocks, blockSize, numKeyValueHeads, headDim]排布。</td>
    </tr>
 <tr>
      <td>PAGED_BNBD</td>
      <td>用于paged kv cache输入，数据按[numBlocks, numKeyValueHeads, blockSize, headDim]排布。</td>
    </tr>
  </tbody>
  </table>

### paged attention相关说明

<table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
  </colgroup>
  <thead>
    <tr>
  <th>blockTable</th>
  <th>kvLayout</th>
  <th>Key/Value</th>
 </tr>
</thead>
<tbody>
    <tr>
      <td rowspan="2">非空，shape为[batch, maxNumBlocksPerBatch]，代表使能paged cache</td>
      <td>PAGED_BBND</td>
      <td>[numBlocks, blockSize, numKeyValueHeads, headDim]</td>
    </tr>
    <tr>
      <td>PAGED_BNBD</td>
      <td>[numBlocks, blockSize, numKeyValueHeads, headDim]</td>
    </tr>
    <tr>
      <td rowspan="3">空，代表不使能paged cache，算子接收原始KV输入</td>
      <td>TND</td>
      <td>[totalKTokens, numKeyValueHeads, headDim]</td>
    </tr>
 <tr>
      <td>BSND</td>
      <td>[batch, maxKvSeqLength, numKeyValueHeads, headDim]</td>
    </tr>
 <tr>
      <td>BNSD</td>
      <td>[batch, numKeyValueHeads, maxKvSeqLength, headDim]</td>
    </tr>
  </tbody>
  </table>

### 量化相关说明

对A5代际

<table style="undefined;table-layout: fixed;width: 2000px"><colgroup>
  <col style="width: 100px">
  <col style="width: 30px">
  <col style="width: 60px">
  <col style="width: 200px">
  <col style="width: 200px">
  <col style="width: 100px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
  <th>quantType</th>
  <th>QKV的数据类型</th>
  <th>对称/非对称</th>
  <th>P量化动态/静态</th>
  <th>量化粒度</th>
  <th>量化参数shape</th>
  <th>量化参数dType</th>
 </tr>
</thead>
<tbody>
    <tr>
      <td>0</td>
      <td>非量化，QKV直接作为输入进行计算</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>
  qDequantScaleOptional(不传)<br>
  kDequantScaleOptional(不传)<br>
  vDequantScaleOptional(不传)<br>
  pQuantScaleOptional(不传)
   </td>
   <td>-</td>
    </tr>
    <tr>
      <td>1</td>
      <td rowspan="2">FLOAT8_E4M3</td>
      <td rowspan="4">对称</td>
      <td>静态</td>
   <td>perGroup，QKV均沿S维度分组，group大小和稀疏块尺寸必须相同；<br>特别的，当KV为paged cache时，blockSize需要为blockShapeY的整数倍</td>
      <td>
  qDequantScaleOptional(必选)：
  <ul>
          <li>TND: [batch*ceilDiv(qSeqLength, blockShapeX), headNum, 1]。</li>
    <li>BNSD: [batch, headNum, ceilDiv(maxQSeqLength, blockShapeX), 1]。</li>
    <li>BSND: [batch, ceilDiv(maxQSeqLength, blockShapeX), headNum, 1]。</li>
        </ul>
  kDequantScaleOptional(必选)：
  <ul>
          <li>TND: [batch*ceilDiv(qSeqLength, blockShapeY), kvHeadNum, 1]。</li>
    <li>BNSD: [batch, kvHeadNum, ceilDiv(maxKvSeqLength, blockShapeY), 1]。</li>
    <li>BSND: [batch, ceilDiv(maxKvSeqLength, blockShapeY), kvHeadNum, 1]。</li>
    <li>PAGED_BBND: [batch, ceilDiv(blockSize, blockShapeY), kvHeadNum, 1]。</li>
    <li>PAGED_BNBD: [batch, kvHeadNum, ceilDiv(blockSize, blockShapeY), 1]。</li>
        </ul>
  vDequantScaleOptional(必选)：
  <ul>
          <li>TND: [batch*ceilDiv(qSeqLength, blockShapeY), kvHeadNum, 1]。</li>
    <li>BNSD: [batch, kvHeadNum, ceilDiv(maxKvSeqLength, blockShapeY), 1]。</li>
    <li>BSND: [batch, ceilDiv(maxKvSeqLength, blockShapeY), kvHeadNum, 1]。</li>
    <li>PAGED_BBND: [batch, ceilDiv(blockSize, blockShapeY), kvHeadNum, 1]。</li>
    <li>PAGED_BNBD: [batch, kvHeadNum, ceilDiv(blockSize, blockShapeY), 1]。</li>
        </ul>
  pQuantScaleOptional(可选)：
  <ul>
          <li>输入时，仅包含单一元素，用于用户控制P的静态量化系数: [1]。</li>
    <li>nullptr: 算子默认P的静态量化系数为448.0。</li>
        </ul>
   </td>
   <td>FLOAT32</td>
    </tr>
    <tr>
      <td>2</td>
      <td>动态</td>
   <td rowspan="3">micro scaling，QKV沿着矩阵乘累加轴，按固定大小32进行分组；<br>特别的，当KV为paged cache时，blockSize需要为64的整数倍</td>
      <td rowspan="3">
  qDequantScaleOptional(必选)：
  <ul>
          <li>TND: [totalQTokens, headNum, ceilDiv(headDim, 64), 2]。</li>
    <li>BNSD: [batch, headNum, maxQSeqLength, ceilDiv(headDim, 64), 2]。</li>
    <li>BSND: [batch, maxQSeqLength, headNum, ceilDiv(headDim, 64), 2]。</li>
        </ul>
  kDequantScaleOptional(必选)：
  <ul>
          <li>TND: [totalKTokens, kvHeadNum, ceilDiv(headDim, 64), 2]。</li>
    <li>BNSD: [batch, kvHeadNum, maxKvSeqLength, ceilDiv(headDim, 64), 2]。</li>
    <li>BSND: [batch, maxKvSeqLength, kvHeadNum, ceilDiv(headDim, 64), 2]。</li>
    <li>PAGED_BBND: [batch, blockSize, kvHeadNum, ceilDiv(headDim, 64), 2]。</li>
    <li>PAGED_BNBD: [batch, kvHeadNum, blockSize, ceilDiv(headDim, 64), 2]。</li>
        </ul>
  vDequantScaleOptional(必选)：
  <ul>
          <li>TND: [batch*ceilDiv(kvSeqLength, 64), kvHeadNum, headDim, 2]。</li>
    <li>BNSD: [batch, kvHeadNum, ceilDiv(maxKvSeqLength, 64), headDim, 2]。</li>
    <li>BSND: [batch, ceilDiv(maxKvSeqLength, 64), kvHeadNum, headDim, 2]。</li>
    <li>PAGED_BBND: [batch, ceilDiv(blockSize, 64), kvHeadNum, headDim, 2]。</li>
    <li>PAGED_BNBD: [batch, kvHeadNum, ceilDiv(blockSize, 64), headDim, 2]。</li>
        </ul>
   </td>
   <td rowspan="3">FLOAT8_E4M3</td>
    </tr>
 <tr>
      <td>3</td>
   <td rowspan="2">FLOAT4_E2M1</td>
      <td>动态OCP</td>
    </tr>
 <tr>
      <td>4</td>
      <td>动态CX</td>
    </tr>
 <tr>
      <td>5</td>
      <td>FLOAT8_E4M3</td>
      <td>对称</td>
      <td>静态</td>
   <td>不传入量化系数，而是在算子内直接将P cast成fp8</td>
      <td>
  qDequantScaleOptional(不传)<br>
  kDequantScaleOptional(不传)<br>
  vDequantScaleOptional(不传)<br>
  pQuantScaleOptional(不传)
   </td>
   <td>FLOAT32</td>
    </tr>
  </tbody>
  </table>

### 掩码说明

<table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
<col style="width: 144px">
  <col style="width: 500px">
<col style="width: 144px">
  </colgroup>
  <thead>
    <tr>
  <th>maskType</th>
  <th>含义</th>
  <th>attentionMaskOptional</th>
  <th>winLeft/winRight</th>
 </tr>
</thead>
<tbody>
    <tr>
      <td>0</td>
   <td>不加mask</td>
      <td>不传</td>
      <td>-1/-1</td>
    </tr>
    <tr>
      <td>1</td>
   <td>causal mask</td>
      <td>[2048,2048]的下三角，int8类型，下部为0，上部为1</td>
      <td>-1/-1</td>
    </tr>
    <tr>
      <td>2</td>
   <td>window mask</td>
      <td>[2048,2048]的下三角，int8类型，下部为0，上部为1</td>
      <td>实际window包括的向前/向后看的token数</td>
    </tr>
    <tr>
      <td>3~5</td>
   <td>各类特化mask</td>
      <td>后续补充mask描述</td>
      <td>-1/-1</td>
    </tr>
  </tbody>
  </table>

### GenericBlockSparseAttention 约束说明

- 确定性计算：aclnnGenericBlockSparseAttention默认确定性实现。
- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
- layoutQ当前仅支持"TND"和"BNSD"和"BSND"。
- layoutKv当前仅支持"TND"和"BNSD"和"BSND""PAGED_BBND""PAGED_BNBD"。
- query、key、value的InputLayout必须保持一致（paged场景下layoutKv可为PAGED格式）。
- 输入query、key、value的数据类型必须一致，支持FLOAT16、BFLOAT16、FLOAT8_E4M3FN和FLOAT4_E2M1。
- cuSeqLengthsQOptional在layoutQ为"TND"时必选；cuSeqLengthsKvOptional在layoutKv为"TND"/"PAGED_BBND"/"PAGED_BNBD"时必选。<br>- sequsedQOptional与cuSeqLengthsQOptional互斥，sequsedKvOptional与cuSeqLengthsKvOptional互斥。
- sparseBlockIdx第4维maxSparseBlockCount应 >= sparseBlockCount tensor中所有元素的最大值，不限制上限。
- blockTableOptional为nullptr时表示不开启PagedAttention特性，非nullptr时表示使用PagedAttention。
- returnSoftmaxLse仅支持配置0或1，分别表示不开启/开启softmaxLse输出。
- maskType当前支持0~5，分别表示不加mask、causal mask、windowed mask等模式，详见"掩码说明"。
- winLeft和winRight不使能时必须为-1。
- 量化相关约束详见"量化相关说明"。

### GenericBlockSparseAttention 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```cpp
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_generic_sparse_attention.h"

using namespace std;

#define CHECK_RET(cond, return_expr) \
    do {                               \
        if (!(cond)) {                   \
            return_expr;                   \
        }                                \
    } while (0)

#define LOG_PRINT(message, ...)     \
    do {                              \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    if (shape.empty()) {
        LOG_PRINT("CreateAclTensor: ERROR - shape is empty\n");
        return -1;
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] <= 0) {
            LOG_PRINT("CreateAclTensor: ERROR - shape[%zu]=%ld is invalid\n", i, shape[i]);
            return -1;
        }
    }
    auto size = GetShapeSize(shape) * sizeof(T);
    if (hostData.size() != static_cast<size_t>(GetShapeSize(shape))) {
        LOG_PRINT("CreateAclTensor: ERROR - hostData size mismatch: %zu vs %ld\n",
                  hostData.size(), GetShapeSize(shape));
        return -1;
    }
    *deviceAddr = nullptr;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    if (shape.size() > 1) {
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
    }
    *tensor = nullptr;
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed - returned nullptr\n"); return -1);
    return 0;
}

int main() {
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int32_t batch = 1;
    int32_t qSeqlen = 128;
    int32_t kvSeqlen = 128;
    int32_t numHeads = 1;
    int32_t numKvHeads = 1;
    int32_t headDim = 128;
    int32_t blockShapeX = 64;
    int32_t blockShapeY = 128;
    int32_t qBlockNum = (qSeqlen + blockShapeX - 1) / blockShapeX;
    int32_t kvBlockNum = (kvSeqlen + blockShapeY - 1) / blockShapeY;
    int64_t totalQTokens = batch * qSeqlen;
    int64_t totalKvTokens = batch * kvSeqlen;

    aclTensor *queryTensor = nullptr;
    aclTensor *keyTensor = nullptr;
    aclTensor *valueTensor = nullptr;
    aclTensor *sparseBlockIdxTensor = nullptr;
    aclTensor *sparseBlockCountTensor = nullptr;
    aclTensor *cuSeqLengthsQTensor = nullptr;
    aclTensor *cuSeqLengthsKvTensor = nullptr;
    aclTensor *metadataOptionalTensor = nullptr;
    aclTensor *attentionOutTensor = nullptr;

    void *queryDeviceAddr = nullptr;
    void *keyDeviceAddr = nullptr;
    void *valueDeviceAddr = nullptr;
    void *sparseBlockIdxDeviceAddr = nullptr;
    void *sparseBlockCountDeviceAddr = nullptr;
    void *cuSeqLengthsQDeviceAddr = nullptr;
    void *cuSeqLengthsKvDeviceAddr = nullptr;
    void *metadataOptionalDeviceAddr = nullptr;
    void *attentionOutDeviceAddr = nullptr;
    void *workspaceAddr = nullptr;

    std::vector<int64_t> queryShape = {totalQTokens, numHeads, headDim};
    std::vector<op::fp16_t> queryHostData(totalQTokens * numHeads * headDim, 1.0f);
    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create query tensor\n"); return ret);

    std::vector<int64_t> kvShape = {totalKvTokens, numKvHeads, headDim};
    std::vector<op::fp16_t> keyHostData(totalKvTokens * numKvHeads * headDim, 1.0f);
    std::vector<op::fp16_t> valueHostData(totalKvTokens * numKvHeads * headDim, 1.0f);
    ret = CreateAclTensor(keyHostData, kvShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create key tensor\n"); return ret);
    ret = CreateAclTensor(valueHostData, kvShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create value tensor\n"); return ret);

    int32_t maxSparseBlockCount = kvBlockNum;
    std::vector<int32_t> idxHostData(numHeads * qBlockNum * maxSparseBlockCount, 0);
    for (int i = 0; i < numHeads * qBlockNum * maxSparseBlockCount; i++) {
        idxHostData[i] = i % kvBlockNum;
    }
    std::vector<int64_t> idxShape = {numHeads, qBlockNum, maxSparseBlockCount};
    ret = CreateAclTensor(idxHostData, idxShape, &sparseBlockIdxDeviceAddr, aclDataType::ACL_INT32, &sparseBlockIdxTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create sparseBlockIdx tensor\n"); return ret);

    std::vector<int32_t> countHostData(numHeads * qBlockNum, kvBlockNum);
    std::vector<int64_t> countShape = {numHeads, qBlockNum};
    ret = CreateAclTensor(countHostData, countShape, &sparseBlockCountDeviceAddr, aclDataType::ACL_INT32, &sparseBlockCountTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create sparseBlockCount tensor\n"); return ret);

    std::vector<int64_t> cuSeqHostData = {0, qSeqlen};
    std::vector<int64_t> cuSeqShape = {batch + 1};
    ret = CreateAclTensor(cuSeqHostData, cuSeqShape, &cuSeqLengthsQDeviceAddr, aclDataType::ACL_INT64, &cuSeqLengthsQTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create cuSeqLengthsQ tensor\n"); return ret);
    ret = CreateAclTensor(cuSeqHostData, cuSeqShape, &cuSeqLengthsKvDeviceAddr, aclDataType::ACL_INT64, &cuSeqLengthsKvTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create cuSeqLengthsKv tensor\n"); return ret);

    std::vector<int64_t> attentionOutShape = {totalQTokens, numHeads, headDim};
    std::vector<op::fp16_t> attentionOutHostData(totalQTokens * numHeads * headDim, 0.0f);
    ret = CreateAclTensor(attentionOutHostData, attentionOutShape, &attentionOutDeviceAddr, aclDataType::ACL_FLOAT16, &attentionOutTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create attentionOut tensor\n"); return ret);

    const char* qLayoutStr = "TND";
    const char* kvLayoutStr = "TND";
    char qLayoutBuffer[16] = {0};
    char kvLayoutBuffer[16] = {0};
    strncpy(qLayoutBuffer, qLayoutStr, sizeof(qLayoutBuffer) - 1);
    strncpy(kvLayoutBuffer, kvLayoutStr, sizeof(kvLayoutBuffer) - 1);

    float scaleValue = 1.0f / std::sqrt(static_cast<float>(headDim));

    // ========== metaData ==========
    {
        uint64_t metaWorkspaceSize = 0;
        aclOpExecutor* metaExecutor = nullptr;

        std::vector<int64_t> metadataOptionalShape = {1, 1, 1};
        std::vector<int64_t> metadataOptionalHost(1, 0);
        int64_t blockShapeValues[] = {blockShapeX, blockShapeY};
        aclIntArray *blockShape = aclCreateIntArray(blockShapeValues, 2);
        ret = CreateAclTensor(metadataOptionalHost, metadataOptionalShape, &metadataOptionalDeviceAddr, aclDataType::ACL_INT64, &metadataOptionalTensor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create metaData tensor\n"); return ret);

        ret = aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize(
            sparseBlockIdxTensor, sparseBlockCountTensor,
            cuSeqLengthsQTensor, cuSeqLengthsKvTensor, nullptr, nullptr,
            qSeqlen, kvSeqlen, numHeads, numKvHeads, headDim,
            blockShape, 0,
            qLayoutBuffer, kvLayoutBuffer,
            0, 0, 0, -1, -1,
            metadataOptionalTensor, &metaWorkspaceSize, &metaExecutor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("metaData GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

        if (metaWorkspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, metaWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate metaData workspace failed. ERROR: %d\n", ret); return ret);
        }

        ret = aclnnGenericBlockSparseAttentionMetadata(workspaceAddr, metaWorkspaceSize, metaExecutor, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("metaData execution failed. ERROR: %d\n", ret); return ret);
        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

        if (workspaceAddr) { aclrtFree(workspaceAddr); workspaceAddr = nullptr; }
    }

    // ========== Attention ==========
    {
        uint64_t attnWorkspaceSize = 0;
        aclOpExecutor* attnExecutor = nullptr;

        ret = aclnnGenericBlockSparseAttentionGetWorkspaceSize(
            queryTensor, keyTensor, valueTensor,
            sparseBlockIdxTensor, sparseBlockCountTensor, metadataOptionalTensor,
            nullptr, nullptr, nullptr, nullptr, nullptr,
            cuSeqLengthsQTensor, cuSeqLengthsKvTensor, nullptr, nullptr, nullptr,
            blockShape, 0,
            qLayoutBuffer, kvLayoutBuffer,
            scaleValue, 0, 0, 0.0, 0, -1, -1, 0,
            attentionOutTensor, nullptr,
            &attnWorkspaceSize, &attnExecutor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Attention GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

        if (attnWorkspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, attnWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate attention workspace failed. ERROR: %d\n", ret); return ret);
        }

        ret = aclnnGenericBlockSparseAttention(workspaceAddr, attnWorkspaceSize, attnExecutor, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Attention execution failed. ERROR: %d\n", ret); return ret);
        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    }

    int64_t outSize = GetShapeSize(attentionOutShape);
    std::vector<op::fp16_t> resultData(outSize, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(op::fp16_t), attentionOutDeviceAddr,
                      outSize * sizeof(op::fp16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    uint64_t printNum = 10;
    LOG_PRINT("attentionOut results (first %lu elements):\n", printNum);
    for (uint64_t i = 0; i < printNum && i < resultData.size(); i++) {
        LOG_PRINT("  index %lu: %f\n", i, static_cast<float>(resultData[i]));
    }

    if (queryTensor) aclDestroyTensor(queryTensor);
    if (keyTensor) aclDestroyTensor(keyTensor);
    if (valueTensor) aclDestroyTensor(valueTensor);
    if (sparseBlockIdxTensor) aclDestroyTensor(sparseBlockIdxTensor);
    if (sparseBlockCountTensor) aclDestroyTensor(sparseBlockCountTensor);
    if (cuSeqLengthsQTensor) aclDestroyTensor(cuSeqLengthsQTensor);
    if (cuSeqLengthsKvTensor) aclDestroyTensor(cuSeqLengthsKvTensor);
    if (metadataOptionalTensor) aclDestroyTensor(metadataOptionalTensor);
    if (attentionOutTensor) aclDestroyTensor(attentionOutTensor);
    if (queryDeviceAddr) aclrtFree(queryDeviceAddr);
    if (keyDeviceAddr) aclrtFree(keyDeviceAddr);
    if (valueDeviceAddr) aclrtFree(valueDeviceAddr);
    if (sparseBlockIdxDeviceAddr) aclrtFree(sparseBlockIdxDeviceAddr);
    if (sparseBlockCountDeviceAddr) aclrtFree(sparseBlockCountDeviceAddr);
    if (cuSeqLengthsQDeviceAddr) aclrtFree(cuSeqLengthsQDeviceAddr);
    if (cuSeqLengthsKvDeviceAddr) aclrtFree(cuSeqLengthsKvDeviceAddr);
    if (metadataOptionalDeviceAddr) aclrtFree(metadataOptionalDeviceAddr);
    if (attentionOutDeviceAddr) aclrtFree(attentionOutDeviceAddr);
    if (workspaceAddr) aclrtFree(workspaceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    LOG_PRINT("Test completed successfully!\n");
    return 0;
}
```

## 2. aclnnGenericBlockSparseAttentionMetadata

### GenericBlockSparseAttentionMetadata 函数说明

每个算子分为[两段式接口](https://gitcode.com/cann/ops-transformer/blob/master/docs/zh/context/%E4%B8%A4%E6%AE%B5%E5%BC%8F%E6%8E%A5%E5%8F%A3.md)，必须先调用"aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnGenericBlockSparseAttentionMetadata"接口执行计算。

第一段接口：

```cpp
__attribute__((visibility("default"))) aclnnStatus aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize(
    const aclTensor *sparseBlockIdx,
    const aclTensor *sparseBlockCount,
    const aclTensor *cuSeqLengthsQOptional,
    const aclTensor *cuSeqLengthsKvOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKvOptional,
    int64_t maxQSeqlen,
    int64_t maxKvSeqlen,
    int64_t numQHeads,
    int64_t numKvHeads,
    int64_t headDim,
    const aclIntArray *blockShape,
    int64_t isPackedGQA,
    char *layoutQ,
    char *layoutKv,
    int64_t maskType,
    int64_t quantType,
    int64_t softmaxPrecision,
    int64_t winLeft,
    int64_t winRight,
    aclTensor *metadata,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

第二段接口：

```cpp
__attribute__((visibility("default"))) aclnnStatus aclnnGenericBlockSparseAttentionMetadata(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

### aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize 参数说明

- **参数说明**

<table style="table-layout: fixed; width: 2000px">
  <colgroup>
    <col style="width: 150px">
    <col style="width: 100px">
    <col style="width: 200px">
    <col style="width: 200px">
    <col style="width: 200px">
    <col style="width: 100px">
    <col style="width: 100px">
    <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>sparseBlockIdx</td>
      <td>输入，必选</td>
      <td>Device侧的aclTensor，稀疏块索引数组，指定每个Q块选择的KV块索引。</td>
      <td>
  存储每个Q块选择的KV块索引，支持的shape随query布局变化：
  <ul>
    <li>query为TND布局时：</li>
    <ul>
   <li>每个qHead对应的KV稀疏pattern不一致（isPackedGQA=0）：<br>[headNum, totalQBlocks, maxSparseBlockCount]</li>
   <li>GQA/MQA下，同group每个qHead对应的KV稀疏pattern一致（isPackedGQA=1）：<br>[numKeyValueHeads, totalQBlocks, maxSparseBlockCount]</li>
    </ul>
    <li>query为BNSD/BSND布局时：</li>
    <ul>
   <li>每个qHead对应的KV稀疏pattern不一致（isPackedGQA=0）：<br>[batch, headNum, ceilDiv(maxQSeqLength, blockShapeX), maxSparseBlockCount]</li>
   <li>GQA/MQA下，同group每个qHead对应的KV稀疏pattern一致（isPackedGQA=1）：<br>[batch, numKeyValueHeads, ceilDiv(maxQSeqLength, blockShapeX), maxSparseBlockCount]</li>
    </ul>
  </ul>
  其中totalQBlocks = Σ ceilDiv(qSeqLen_i, blockShapeX)，i为batch索引，qSeqLen_i由cuSeqLengthsQOptional指定。
<br>maxSparseBlockCount为sparseBlockCount tensor中所有元素的最大值，即所有Q块选择的KV块数量的最大值。传入值只需 >= 该最大值即可，不限制上限。
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sparseBlockCount</td>
      <td>输入，必选</td>
      <td>Device侧的aclTensor，每个Q块实际选择的KV块数量。</td>
      <td>
  存储每个Q块实际选择的KV块数量，支持的shape随query布局变化：
  <ul>
    <li>query为TND布局时：</li>
    <ul>
   <li>每个qHead对应的KV稀疏pattern不一致（isPackedGQA=0）：<br>[headNum, totalQBlocks]</li>
   <li>GQA/MQA下，同group每个qHead对应的KV稀疏pattern一致（isPackedGQA=1）：<br>[numKeyValueHeads, totalQBlocks]</li>
    </ul>
    <li>query为BNSD/BSND布局时：</li>
    <ul>
   <li>每个qHead对应的KV稀疏pattern不一致（isPackedGQA=0）：<br>[batch, headNum, ceilDiv(maxQSeqLength, blockShapeX)]</li>
   <li>GQA/MQA下，同group每个qHead对应的KV稀疏pattern一致（isPackedGQA=1）：<br>[batch, numKeyValueHeads, ceilDiv(maxQSeqLength, blockShapeX)]</li>
    </ul>
  </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqLengthsQOptional</td>
      <td>输入，可选</td>
      <td>Device侧的aclTensor，描述每个Batch对应的query序列长度，以前缀和形式存储。</td>
      <td>可选输入，用于变长序列场景：
        <ul>
          <li>当layoutQ为"TND"时：该项输入必须配置</li>
          <li>当layoutQ为"BNSD""BSND"时：如配置该项输入，算子内会按该输入指定的实际序列长度进行处理；<br>如不配置该项输入(传入nullptr)，算子内会按照maxQSeqlen进行处理。</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cuSeqLengthsKvOptional</td>
      <td>输入，可选</td>
      <td>Device侧的aclTensor，描述每个Batch对应的key/value序列长度，以前缀和形式存储。</td>
      <td>可选输入，用于变长序列场景：
        <ul>
          <li>当layoutKv为"TND"/"PAGED_BBND"/"PAGED_BNBD"时：该项输入必须配置</li>
          <li>当layoutKv为"BNSD""BSND"时：如配置该项输入，算子内会按该输入指定的实际序列长度进行处理；<br>如不配置该项输入(传入nullptr)，算子内会按照maxKvSeqlen进行处理。</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>1</td>
      <td>-</td>
    </tr>
    </tr>
    <tr>
      <td>sequsedQOptional</td>
      <td>输入，可选</td>
      <td>Device侧的aclTensor，各batch中query的实际序列长度。</td>
      <td>与cuSeqLengthsQOptional互斥。
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedKvOptional</td>
      <td>输入，可选</td>
      <td>Device侧的aclTensor，各batch中kv的实际序列长度。</td>
      <td>与cuSeqLengthsKvOptional互斥。
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr> <tr>
      <td>maxQSeqlen</td>
      <td>输入，Attr</td>
      <td>所有batch中的qSeqlen的最大值。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
 <tr>
      <td>maxKvSeqlen</td>
      <td>输入，Attr</td>
      <td>所有batch中的kvSeqlen的最大值。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
 <tr>
      <td>numQHeads</td>
      <td>输入，Attr</td>
      <td>query的head数。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
 <tr>
      <td>numKvHeads</td>
      <td>输入，Attr</td>
      <td>key/value的head数。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
 <tr>
      <td>headDim</td>
      <td>输入，Attr</td>
      <td>query/key/value的embed。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
 <tr>
      <td>blockShape</td>
      <td>输入，Attr</td>
      <td>代表稀疏块形状数组。</td>
      <td>含两个元素[blockShapeX, blockShapeY]。<br>blockShapeX支持任意值，不可超过int64表示范围。<br>blockShapeY支持按16对齐的任意值，不可超过int64表示范围。<br>开启量化功能时的约束具体见“量化相关说明”。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
 <tr>
      <td>isPackedGQA</td>
      <td>输入，Attr</td>
      <td>代表进行块状稀疏时，同一个group内的qHead是否共享同样的稀疏pattern<br>（注：不同batch之间不会共享同样的稀疏pattern，该入参仅区分head维度的共享情况）。</td>
      <td>若取值为0，则代表同一个group内的qHead不共享同样的稀疏pattern；<br>若取值为1，则代表同一个group内的qHead共享同样的稀疏pattern。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQ</td>
      <td>输入，Attr</td>
      <td>代表输入query的数据排布格式。</td>
      <td>当前支持"TND""BNSD""BSND"。</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKv</td>
      <td>输入，Attr</td>
      <td>代表输入key、value的数据排布格式。</td>
      <td>当前仅支持"TND""BNSD""BSND""PAGED_BBND""PAGED_BNBD"，详情见layout说明。</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
 <tr>
      <td>maskType</td>
      <td>输入，Attr</td>
      <td>表示attention计算中的掩码类型。</td>
      <td>
 0代表不加mask场景，其余见“掩码相关说明”
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantType</td>
      <td>输入，Attr</td>
      <td>代表采用的量化手段。</td>
      <td>取值为0时代表不量化，其余具体见“量化相关说明”。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmaxPrecision</td>
      <td>输入，Attr</td>
      <td>Softmax计算采取的精度级别。</td>
      <td>
        控制online softmax阶段以及rescale阶段运算使用的数据类型。<br>当前只支持传0或1
        <ul>
          <li>0：表示online softmax和rescale全部采取fp32数据类型，适合追求计算精度的场景使用。</li>
          <li>1：表示混合精度运算，在性能与精度上取得一个折中。<br>online softmax采取fp16/bf16数据类型（与attentionOut数据类型相同），rescale采取fp32数据类型，<br>在online softmax阶段可能发生数值溢出。</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>winLeft</td>
      <td>输入，Attr</td>
      <td>Host侧的int64_t，滑窗attention场景下，滑窗需要向前包含多少个token。</td>
      <td>用于滑窗attention场景，不使能时必须为-1，需要与maskType，mask配合使用，具体见“掩码说明”。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>winRight</td>
      <td>输入，Attr</td>
      <td>Host侧的int64_t，滑窗attention场景下，滑窗需要向后包含多少个token。</td>
      <td>用于滑窗attention场景，不使能时必须为-1，需要与maskType，mask配合使用，具体见“掩码说明”。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadataOptional</td>
      <td>输出</td>
      <td>Device侧的aclTensor，稀疏attention的分核信息。</td>
      <td>
        -
      </td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回op执行器，包含算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

- **返回值**
  aclnnStatus：
  返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

### aclnnGenericBlockSparseAttentionMetadata 参数说明

- **参数说明**

<table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值**
  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

### GenericBlockSparseAttentionMetadata 约束说明

- 确定性计算：aclnnGenericBlockSparseAttentionMetadata默认确定性实现。
- maskType当前支持0~5，详见"掩码说明"。
- winLeft和winRight不使能时必须为-1。

### GenericBlockSparseAttentionMetadata 调用示例

Metadata接口的调用示例请参见[1. aclnnGenericBlockSparseAttention](#1-aclnnGenericBlockSparseAttention)的调用示例，两者配合使用。
