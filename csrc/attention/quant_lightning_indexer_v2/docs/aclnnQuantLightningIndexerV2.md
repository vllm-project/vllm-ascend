# aclnnQuantLightningIndexerV2

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/quant_lightning_indexer_v2)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：`QuantLightningIndexerV2`是推理场景下，稀疏attention前处理的计算，选出关键的稀疏token，并对输入query和key进行量化实现存8算8，获取最大收益。

- 版本演进：在QuantLightningIndexer的基础上，新增压缩key场景、分核计算metadata、稀疏value输出等能力。

- 计算公式：

$$
out = \text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(\left(Scale_Q@Scale_K^T\right)\odot\left(Q_{index}^{Quant}@{\left(K_{index}^{Quant}\right)}^T\right)\right)\right]\right\}
$$

主要计算过程为：

1. 将某个token对应的输入参数`query`（$Q_{index}^{Quant}\in\R^{g\times d}$）乘以给定上下文`key`（$K_{index}^{Quant}\in\R^{S_{k}\times d}$），得到相关性。
2. 相关性结果与`query`和`key`对应的反量化系数`query_dequant_scale`（$Scale_Q$）和`key_dequant_scale`（$Scale_K^T$）相乘，通过激活函数$ReLU$过滤无效负相关信号后，得到当前Token与所有前序Token的相关性分数向量。
3. 将其与权重系数`weights`（$W$）相乘后，沿g的方向，选取前$Top-k$个索引值得到输出$out$，作为Attention的输入。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnQuantLightningIndexerV2GetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnQuantLightningIndexerV2"接口执行计算。

```Cpp
aclnnStatus aclnnQuantLightningIndexerV2GetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *weights,
    const aclTensor *queryDequantScale,
    const aclTensor *keyDequantScale,
    const aclTensor *cuSeqLensQOptional,
    const aclTensor *cuSeqLensKOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKOptional,
    const aclTensor *cmpResidualKOptional,
    const aclTensor *blockTableOptional,
    const aclTensor *outputIdxOffsetOptional,
    const aclTensor *metadataOptional,
    int64_t          topk,
    int64_t          quantMode,
    int64_t          maxSeqlenQOptional,
    char            *layoutQOptional,
    char            *layoutKOptional,
    int64_t          maskModeOptional,
    int64_t          cmpRatioOptional,
    int64_t          returnValueOptional,
    const aclTensor *sparseIndicesOut,
    const aclTensor *sparseValuesOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnQuantLightningIndexerV2(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnQuantLightningIndexerV2GetWorkspaceSize

- **参数说明：**

> [!NOTE]
>
> - query、key、weights参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、N（Head Num）表示多头数、D（Head Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
> - S1表示query shape中的S，S2表示key shape中的S，T1表示query shape中的T，N1表示query shape中的N，N2表示key shape中的N。
> - maxBlockNumPerSeq表示每个Batch中最大sequsedK对应的block数量，S2_MAX表示sequsedK中的最大值

  <table style="undefined;table-layout: fixed; width: 1601px"><colgroup>
  <col style="width: 264px">
  <col style="width: 132px">
  <col style="width: 232px">
  <col style="width: 330px">
  <col style="width: 164px">
  <col style="width: 119px">
  <col style="width: 215px">
  <col style="width: 145px">
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
    </tr></thead>
  <tbody>
    <tr>
      <td>query</td>
      <td>输入</td>
      <td>公式中量化后的 Query。</td>
      <td>不支持空tensor。</td>
      <td>INT8、FLOAT8_e4m3fn、HIFLOAT8、FLOAT4_e2m1</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_query为BSND时，shape为(B,S1,N1,D)。</li>
                <li>layout_query为TND时，shape为(T1,N1,D)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>key</td>
      <td>输入</td>
      <td>公式中量化后的 Key。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>block_num为PageAttention时block总数，block_size为一个block的token数。</li>
                <li>layout_key为PA_BSND时，shape为(block_num, block_size, N2, D)。</li>
                <li>layout_key为BSND时，shape为(B, K_S, N2, D)，layout_key为TND时，shape为(K_T, N2, D)。</li>
          </ul>
      </td>
      <td>INT8、FLOAT8_e4m3fn、HIFLOAT8、FLOAT4_e2m1</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_key为PA_BSND时，shape为(block_num, block_size, N2, D)。</li>
          </ul>
      </td>
      <td>支持0轴非连续</td>
    </tr>
    <tr>
      <td>weights</td>
      <td>输入</td>
      <td>公式中的权重系数 W。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_query为BSND时，shape为(B,S1,N1)。</li>
                <li>layout_query为TND时，shape为(T1,N1)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>queryDequantScale</td>
      <td>输入</td>
      <td>公式中 Query 的反量化系数。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT16、FLOAT32、FLOAT8_e8m0</td>
      <td>ND</td>
      <td>
          <ul>
                <li>quantMode为3/5时，layout_query为BSND时shape为(B,S1,N1,D/64,2)，layout_query为TND时shape为(T1,N1,D/64,2)。</li>
                <li>quantMode为4时，shape为(1,)。</li>
                <li>其他场景shape与weights保持一致。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>keyDequantScale</td>
      <td>输入</td>
      <td>公式中 Key 的反量化系数。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT16、FLOAT32、FLOAT8_e8m0</td>
      <td>ND</td>
      <td>
          <ul>
                <li>quantMode为3/5时，layout_key为PA_BSND、BSND、TND对应的shape分别为(block_num,block_size,N2,D/64,2)、(B,K_S,N2,D/64,2)、(K_T,N2,D/64,2)。</li>
                <li>quantMode为4时，shape为(1,)。</li>
                <li>其他场景下，layout_key为PA_BSND、BSND、TND对应的shape分别为(block_num,block_size,N2)、(B,K_S,N2)、(K_T,N2)。</li>
          </ul>
      </td>
      <td>支持0轴非连续</td>
    </tr>
    <tr>
      <td>cuSeqLensQOptional</td>
      <td>输入</td>
      <td>每个Batch中，Query的有效token数（TND场景使用cu_seqlens格式）。</td>
      <td>
          <ul>
                <li>当layout_query为TND时，该入参必须传入，且以该入参元素的数量作为B值，该入参中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B+1,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>cuSeqLensKOptional</td>
      <td>输入</td>
      <td>每个Batch中，Key的有效token数（TND场景使用cu_seqlens格式）。</td>
      <td>
          <ul>
                <li>当layout_key为TND时，该入参必须传入。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B+1,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>sequsedQOptional</td>
      <td>输入</td>
      <td>每个Batch中，Query的有效token数（BSND场景使用seqused格式）。</td>
      <td>该入参中每个Batch的有效token数不超过query中的维度S大小且不小于0。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>sequsedKOptional</td>
      <td>输入</td>
      <td>每个Batch中，Key的有效token数（BSND场景使用seqused格式）。</td>
      <td>
          <ul>
                <li>该入参中每个Batch的有效token数不超过key中的维度S大小且不小于0。</li>
                <li>当layout_key为PA_BSND时，该入参必须传入。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>cmpResidualKOptional</td>
      <td>输入</td>
      <td>压缩场景下Key的残余长度。</td>
      <td>需满足0 <= cmpResidualKOptional[i] < cmpRatioOptional。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>blockTableOptional</td>
      <td>输入</td>
      <td>表示PageAttention中KV存储使用的block映射表。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>PageAttention场景下，block_table必须为二维，第一维长度需要等于B，第二维长度不能小于maxBlockNumPerSeq。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B, S2_MAX/block_size)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>outputIdxOffsetOptional</td>
      <td>输入</td>
      <td>输出索引的偏移量。</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>layout_query为BSND时shape为(B,S1,N2)，layout_query为TND时shape为(T1,N2)。</td>
      <td>x</td>
    </tr>
    <tr>
      <td>metadataOptional</td>
      <td>输入</td>
      <td>QuantLightningIndexerV2Metadata算子传入的分核信息。</td>
      <td>
          <ul>
                <li>包含使用核数、分块大小以及每个核处理数据的起始点等内容。</li>
                <li>shape大小为[1024]，当前不支持传空。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(1024,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>topk</td>
      <td>输入</td>
      <td>topK阶段需要保留的Key token索引数量。</td>
      <td>支持[1, 8192]。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode</td>
      <td>输入</td>
      <td>量化模式。</td>
      <td>
          <ul>
                <li>支持传入 1（FLOAT8_e4m3fn量化）、2（Per-Token-Head量化）、3（MXFP8量化）、4（HIFLOAT8量化）、5（MXFP4量化）。</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenQOptional</td>
      <td>输入</td>
      <td>Query的最大序列长度。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQOptional</td>
      <td>输入</td>
      <td>用于标识输入Query的数据排布格式。</td>
      <td>
          <ul>
                <li>支持BSND、TND。</li>
          </ul>
      </td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKOptional</td>
      <td>输入</td>
      <td>用于标识输入Key的数据排布格式。</td>
      <td>
          <ul>
                <li>支持 PA_BSND、BSND、TND。</li>
          </ul>
      </td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskModeOptional</td>
      <td>输入</td>
      <td>表示sparse的模式。</td>
      <td>
          <ul>
                <li>0代表defaultMask模式。</li>
                <li>3代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpRatioOptional</td>
      <td>输入</td>
      <td>key的压缩倍数。</td>
      <td>
          <ul>
                <li>支持 (0, 128] 内的正整数。</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>returnValueOptional</td>
      <td>输入</td>
      <td>表示是否输出sparseValuesOut。</td>
      <td>
          <ul>
                <li>1表示输出，0表示不输出。</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseIndicesOut</td>
      <td>输出</td>
      <td>公式中的Indices输出。</td>
      <td>不支持空tensor。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_query为"BSND"时输出shape为[B, S1, N2, topk]。</li>
                <li>layout_query为"TND"时输出shape为[T1, N2, topk]。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>sparseValuesOut</td>
      <td>输出</td>
      <td>公式中的Indices输出对应的value值。</td>
      <td>
          <ul>
                <li>returnValue为1时输出有效值，无效部分填bf16负无穷；returnValue为0时输出shape为(0,)的空tensor。</li>
          </ul>
      </td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>returnValue为1时shape与sparseIndicesOut保持一致；returnValue为0时shape为(0,)。</td>
      <td>x</td>
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
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

<!-- npu="950" id10 -->
- <term>Ascend 950PR/Ascend 950DT</term>：
    - `layout_key` 额外支持 BSND 和 TND；支持 PA_BSND、BSND、TND。
    - `quant_mode` 支持 1（FLOAT8_e4m3fn量化）、2（INT8量化）、3（MXFP8量化）、4（HIFLOAT8量化）和 5（MXFP4量化）。
    - `cmp_ratio` 支持 (0, 128] 内任意正整数。
    - 支持 `return_value`。
    - query 和 key：`quant_mode` 为 1/3 时支持 FLOAT8_e4m3fn，`quant_mode` 为 2 时支持 INT8，`quant_mode` 为 4 时支持 HIFLOAT8，`quant_mode` 为 5 时支持 FLOAT4_e2m1。
    - query_dequant_scale 和 key_dequant_scale：`quant_mode` 为 1/4 时支持 FLOAT32，`quant_mode` 为 2 时支持 FLOAT16，`quant_mode` 为 3/5 时支持 FLOAT8_e8m0。
    - weights：`quant_mode` 为 2 时支持 FLOAT16，`quant_mode` 为 1/3/4/5 时支持 FLOAT32。
    - query Q_N 支持 [1, 64]。
<!-- end id10 -->
<!-- npu="A3,910b" id11 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    - `layout_key` 仅支持 PA_BSND。
    - `quant_mode` 仅支持 2（Per-Token-Head量化）。
    - `cmp_ratio` 仅支持 2 的幂次方且范围为 [1, 128]，即 1/2/4/8/16/32/64/128。
    - 不支持 `outputIdxOffsetOptional`。
    - 不支持 `return_value`。
    - query 和 key：支持 INT8，不支持 FLOAT8_e4m3fn、HIFLOAT8 和 FLOAT4_e2m1。
    - query_dequant_scale 和 key_dequant_scale：支持 FLOAT16，不支持 FLOAT32 和 FLOAT8_e8m0。
    - weights：支持 FLOAT16，不支持 FLOAT32。
    - query Q_N 仅支持 64。
    - topk 仅支持 [1, 2048]。
<!-- end id11 -->

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
        <thead>
            <th>返回值</th>
            <th>错误码</th>
            <th>描述</th>
        </thead>
        <tbody>
            <tr>
                <td>ACLNN_ERR_PARAM_NULLPTR</td>
                <td>161001</td>
                <td>如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_PARAM_INVALID</td>
                <td>161002</td>
                <td>query、key、weights、queryDequantScale、keyDequantScale、cuSeqLensQOptional、cuSeqLensKOptional、sequsedQOptional、sequsedKOptional、cmpResidualKOptional、blockTableOptional、metadataOptional、layoutQOptional、layoutKOptional、topk、quantMode、maskModeOptional、cmpRatioOptional、returnValueOptional、sparseIndicesOut、sparseValuesOut的数据类型和数据格式不在支持的范围内。</td>
            </tr>
        </tbody>
    </table>

## aclnnQuantLightningIndexerV2

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnQuantLightningIndexerV2GetWorkspaceSize获取。</td>
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

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- headdim 支持 128。
- block_size 取值为 16 的倍数，最大支持 1024。
- 当 `layout_key` 不为 PA_BSND 时，`layout_query` 和 `layout_key` 必须一致。
- 当 `quant_mode` 为 3/5 时，`queryDequantScale` 和 `keyDequantScale` 的维数分别比 `query` 和 `key` 多 1，前缀维度保持一致，末两维为(D/64, 2)；D必须为64的倍数，每个scale对应D轴上连续32个逻辑元素。
- 当传入的参数layout_query为TND时，必须传入cuSeqlensQOptional，如果也传入sequsedQOptional，应保证由sequsedQOptional传入的各个batch的query长度不超过根据cuSeqlensQOptional计算出的各个batch的q序列长度。当某个batch由sequsedQOptional传入的q序列长度seqlen1小于由cuSeqlensQOptional计算出的query长度seqlen2时，会启用TND Padding功能，将该batch的seqlen2与seqlen1差值部分的query输出的sparseIndices和sparseValues全部置为无效值。部分长序列场景下，如果需要填充的无效数据过多，由于硬件限制可能会导致aicore执行超时，可以通过(seqlen2 - seqlen1) * topk来计算需要填充的数据量，建议将这个数据量控制在4亿以内。
- **确定性说明：** aclnnQuantLightningIndexerV2 默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_quant_lightning_indexer_v2.cpp
 * \brief
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "securec.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_lightning_indexer_v2.h"
#include "aclnn/opdev/platform.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true :(false))

#define LOG_PRINT(message, ...)     \
  do {                              \
    (void)printf(message, ##__VA_ARGS__); \
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
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
    return ret;
  }
  ret = aclrtSetDevice(deviceId);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
    return ret;
  }
  ret = aclrtCreateStream(stream);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
    return ret;
  }
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
    return ret;
  }

  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
    return ret;
  }

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

struct TensorResources {
    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;
    void* weightsDeviceAddr = nullptr;
    void* qScaleDeviceAddr = nullptr;
    void* kScaleDeviceAddr = nullptr;
    void* metadataDeviceAddr = nullptr;
    void* sparseIndicesDeviceAddr = nullptr;
    void* sparseValuesDeviceAddr = nullptr;

    aclTensor* queryTensor = nullptr;
    aclTensor* keyTensor = nullptr;
    aclTensor* weightsTensor = nullptr;
    aclTensor* qScaleTensor = nullptr;
    aclTensor* kScaleTensor = nullptr;
    aclTensor* metadataTensor = nullptr;
    aclTensor* sparseIndicesTensor = nullptr;
    aclTensor* sparseValuesTensor = nullptr;
};

int InitializeTensors(TensorResources& resources) {
    int64_t B = 2;
    int64_t S1 = 4;
    int64_t S2 = 8;
    int64_t N1 = 64;
    int64_t N2 = 1;
    int64_t D = 128;
    int64_t topk = 512;

    std::vector<int64_t> queryShape = {B, S1, N1, D};
    std::vector<int64_t> keyShape = {B, S2, N2, D};
    std::vector<int64_t> weightsShape = {B, S1, N1};
    std::vector<int64_t> qScaleShape = {B, S1, N1};
    std::vector<int64_t> kScaleShape = {B, S2, N2};
    std::vector<int64_t> metadataShape = {1024};
    std::vector<int64_t> sparseIndicesShape = {B, S1, N2, topk};
    std::vector<int64_t> sparseValuesShape = {B, S1, N2, topk};

    int64_t queryShapeSize = GetShapeSize(queryShape);
    int64_t keyShapeSize = GetShapeSize(keyShape);
    int64_t weightsShapeSize = GetShapeSize(weightsShape);
    int64_t qScaleShapeSize = GetShapeSize(qScaleShape);
    int64_t kScaleShapeSize = GetShapeSize(kScaleShape);
    int64_t metadataShapeSize = GetShapeSize(metadataShape);
    int64_t sparseIndicesShapeSize = GetShapeSize(sparseIndicesShape);
    int64_t sparseValuesShapeSize = GetShapeSize(sparseValuesShape);

    std::vector<uint8_t> queryHostData(queryShapeSize, 0x38);
    std::vector<uint8_t> keyHostData(keyShapeSize, 0x38);
    std::vector<float> weightsHostData(weightsShapeSize, 0.01f);
    std::vector<float> qScaleHostData(qScaleShapeSize, 1.0f);
    std::vector<float> kScaleHostData(kScaleShapeSize, 1.0f);
    std::vector<int32_t> metadataHostData(metadataShapeSize, 0);
    std::vector<int32_t> sparseIndicesHostData(sparseIndicesShapeSize, 0);
    std::vector<uint16_t> sparseValuesHostData(sparseValuesShapeSize, 0);

    int ret = CreateAclTensor(queryHostData, queryShape, &resources.queryDeviceAddr,
                              aclDataType::ACL_FLOAT8_E4M3FN, &resources.queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { return ret; }

    ret = CreateAclTensor(keyHostData, keyShape, &resources.keyDeviceAddr,
                          aclDataType::ACL_FLOAT8_E4M3FN, &resources.keyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { return ret; }

    ret = CreateAclTensor(weightsHostData, weightsShape, &resources.weightsDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.weightsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { return ret; }

    ret = CreateAclTensor(qScaleHostData, qScaleShape, &resources.qScaleDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.qScaleTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { return ret; }

    ret = CreateAclTensor(kScaleHostData, kScaleShape, &resources.kScaleDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.kScaleTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { return ret; }

    ret = CreateAclTensor(metadataHostData, metadataShape, &resources.metadataDeviceAddr,
                          aclDataType::ACL_INT32, &resources.metadataTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { return ret; }

    ret = CreateAclTensor(sparseIndicesHostData, sparseIndicesShape, &resources.sparseIndicesDeviceAddr,
                          aclDataType::ACL_INT32, &resources.sparseIndicesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { return ret; }

    ret = CreateAclTensor(sparseValuesHostData, sparseValuesShape, &resources.sparseValuesDeviceAddr,
                          aclDataType::ACL_BF16, &resources.sparseValuesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { return ret; }

    return ACL_SUCCESS;
}

int ExecuteQuantLightningIndexerV2(TensorResources& resources, aclrtStream stream,
                                   void** workspaceAddr, uint64_t* workspaceSize) {
    int64_t topk = 512;
    int64_t quantMode = 1;
    int64_t maskMode = 0;
    int64_t cmpRatio = 1;
    int64_t returnValue = 1;
    constexpr const char layoutQStr[] = "BSND";
    constexpr const char layoutKStr[] = "BSND";
    constexpr size_t layoutQLen = sizeof(layoutQStr);
    constexpr size_t layoutKLen = sizeof(layoutKStr);
    char layoutQ[layoutQLen];
    char layoutK[layoutKLen];
    errno_t memcpyRet = memcpy_s(layoutQ, sizeof(layoutQ), layoutQStr, layoutQLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutQ failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    memcpyRet = memcpy_s(layoutK, sizeof(layoutK), layoutKStr, layoutKLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutK failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    aclOpExecutor* executor;

    int ret = aclnnQuantLightningIndexerV2GetWorkspaceSize(
        resources.queryTensor, resources.keyTensor, resources.weightsTensor,
        resources.qScaleTensor, resources.kScaleTensor,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        resources.metadataTensor,
        topk, quantMode, -1, layoutQ, layoutK, maskMode, cmpRatio, returnValue,
        resources.sparseIndicesTensor, resources.sparseValuesTensor,
        workspaceSize, &executor);

    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnQuantLightningIndexerV2GetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnQuantLightningIndexerV2(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnQuantLightningIndexerV2 failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

int PrintOutResult(const std::vector<int64_t>& shape, void* deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<int32_t> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
    return ret;
  }
  LOG_PRINT("sparse_indices result (first 10 elements):\n");
  for (int64_t i = 0; i < size && i < 10; i++) {
    LOG_PRINT("  [%ld] = %d\n", i, resultData[i]);
  }
  return ACL_SUCCESS;
}

void CleanupResources(TensorResources& resources, void* workspaceAddr,
                     aclrtStream stream, int32_t deviceId) {
    if (resources.queryTensor) { aclDestroyTensor(resources.queryTensor); }
    if (resources.keyTensor) { aclDestroyTensor(resources.keyTensor); }
    if (resources.weightsTensor) { aclDestroyTensor(resources.weightsTensor); }
    if (resources.qScaleTensor) { aclDestroyTensor(resources.qScaleTensor); }
    if (resources.kScaleTensor) { aclDestroyTensor(resources.kScaleTensor); }
    if (resources.metadataTensor) { aclDestroyTensor(resources.metadataTensor); }
    if (resources.sparseIndicesTensor) { aclDestroyTensor(resources.sparseIndicesTensor); }
    if (resources.sparseValuesTensor) { aclDestroyTensor(resources.sparseValuesTensor); }

    if (resources.queryDeviceAddr) { aclrtFree(resources.queryDeviceAddr); }
    if (resources.keyDeviceAddr) { aclrtFree(resources.keyDeviceAddr); }
    if (resources.weightsDeviceAddr) { aclrtFree(resources.weightsDeviceAddr); }
    if (resources.qScaleDeviceAddr) { aclrtFree(resources.qScaleDeviceAddr); }
    if (resources.kScaleDeviceAddr) { aclrtFree(resources.kScaleDeviceAddr); }
    if (resources.metadataDeviceAddr) { aclrtFree(resources.metadataDeviceAddr); }
    if (resources.sparseIndicesDeviceAddr) { aclrtFree(resources.sparseIndicesDeviceAddr); }
    if (resources.sparseValuesDeviceAddr) { aclrtFree(resources.sparseValuesDeviceAddr); }

    if (workspaceAddr) { aclrtFree(workspaceAddr); }
    if (stream) { aclrtDestroyStream(stream); }
    aclrtResetDevice(deviceId);
    aclFinalize();
}

} // namespace

int main() {
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        return 0;
    }
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    TensorResources resources = {};
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    int64_t B = 2;
    int64_t S1 = 4;
    int64_t N2 = 1;
    int64_t topk = 512;
    std::vector<int64_t> sparseIndicesShape = {B, S1, N2, topk};
    int ret = ACL_SUCCESS;

    ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    ret = InitializeTensors(resources);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("InitializeTensors failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = ExecuteQuantLightningIndexerV2(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("ExecuteQuantLightningIndexerV2 failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    PrintOutResult(sparseIndicesShape, resources.sparseIndicesDeviceAddr);

    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}
```
