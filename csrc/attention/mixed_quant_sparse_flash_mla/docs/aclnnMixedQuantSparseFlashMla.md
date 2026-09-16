# aclnnMixedQuantSparseFlashMla

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
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

- 接口功能：

  `aclnnMixedQuantSparseFlashMla`算子旨在完成量化和稀疏场景下的MLA（Multi-head Latent Attention）注意力计算。该接口支持以下三类计算模式：
  - **SWA（Sliding Window Attention）**：仅传入`oriKvOptional`，对原始KV做滑动窗口注意力。
  - **CSA（Compressed Sparse Attention）**：同时传入`oriKvOptional`、`cmpKvOptional`和`cmpSparseIndicesOptional`，对原始KV窗口和topK选择出的压缩KV共同做注意力。
  - **HCA（Heavily Compressed Attention）**：同时传入`oriKvOptional`和`cmpKvOptional`，对原始KV窗口和连续压缩KV段共同做注意力。

  `aclnnMixedQuantSparseFlashMlaMetadata`是`aclnnMixedQuantSparseFlashMla`的分核信息，在主算子执行前生成。当前版本主算子必须传入该metadata。典型调用流程如下：

  **该算子不建议单独使用，建议与aclnnMixedQuantSparseFlashMlaMetadata算子配合使用，形成完整的工作流。**

  1. 根据调用场景准备对应输入。
  2. 调用`aclnnMixedQuantSparseFlashMlaMetadata`生成`metadata`，作为`aclnnMixedQuantSparseFlashMla`的入参。
  3. 调用`aclnnMixedQuantSparseFlashMla`，生成计算结果。

- 计算公式：

  $$
  O = \text{softmax}(Q \cdot \tilde{K}^T \cdot \text{softmax\_scale}) \cdot \tilde{V}
  $$

  其中$\tilde{K} = \tilde{V}$（共享KV），$\tilde{K}$由滑动窗口内的原始KV和因果边界内的压缩KV拼接而成，具体参与计算的KV范围由模板模式和mask参数决定：

  - 滑动窗口部分（oriKv）：对第$i_{S1}$个Query token，其因果对角线位置为$\text{oriThreshold} = S2_{act} - S1_{act} + i_{S1} + 1$，窗口范围为$[\max(\text{oriThreshold} - \text{oriWinLeft} - 1, 0), \text{oriThreshold} + \text{oriWinRight})$。

  - 压缩KV部分（cmpKv）：因果边界阈值为$\text{cmpThreshold} = \lfloor \frac{\text{oriThreshold}}{\text{cmpRatio}} \rfloor$。HCA场景取$[0, \text{cmpThreshold})$内的连续压缩KV；CSA场景通过TopK索引从压缩KV中按需收集，仅保留$\text{beginIdx} < \text{cmpThreshold}$的块。

  注意力计算采用Online Softmax（Flash Attention V2），S2方向按512分块循环，sinks作为每行softmax的初始最大值：

  $$
  \text{rowMax}^{(0)} = \text{sinks}[g], \quad \text{rowSum}^{(0)} = 1.0, \quad O^{(0)} = 0
  $$

  $$
  S^{(t)} = Q \cdot K_{tile}^{(t)T} \cdot \text{softmaxScale}
  $$

  $$
  \text{rowMax}^{(t+1)} = \max(\text{rowMax}^{(t)}, \max(S^{(t)}, \text{dim}=-1))
  $$

  $$
  \text{rowSum}^{(t+1)} = \exp(\text{rowMax}^{(t)} - \text{rowMax}^{(t+1)}) \cdot \text{rowSum}^{(t)} + \sum \exp(S^{(t)} - \text{rowMax}^{(t+1)})
  $$

  $$
  O^{(t+1)} = \exp(\text{rowMax}^{(t)} - \text{rowMax}^{(t+1)}) \cdot O^{(t)} + \exp(S^{(t)} - \text{rowMax}^{(t+1)}) \cdot V_{tile}^{(t)}
  $$

  $$
  O_{final} = O^{(T_{s2})} / \text{rowSum}^{(T_{s2})}
  $$

- 符号说明

  | 符号                | 含义                                                      |
  | ------------------- | --------------------------------------------------------- |
  | $Q$                 | Query输入，形状为[G, D]（单行）                           |
  | $K_{tile}^{t}$      | 第t个S2分块的KV数据，K=V（共享KV）                         |
  | $S_t$               | 第t个分块的QK缩放注意力分数                                |
  | $P_t$               | 第t个分块的softmax概率                                     |
  | $O_t$               | 第t个分块后的累加输出                                      |
  | $softmaxScale$      | 缩放系数，通常取每个注意力头维度的倒数平方根                |
  | $B$                 | Batch Size                                                |
  | $S1$/$S1_{act}$     | Query序列长度/实际有效长度                                 |
  | $S2$/$S2_{act}$     | 原始KV序列长度/实际有效长度                                |
  | $N1$                | Query头数                                                 |
  | $N2$                | KV头数                                                    |
  | $G$                 | GQA分组比，$G=N1/N2$                                      |
  | $D$                 | 每个注意力头的维度                                        |
  | $sinks$             | 注意力汇点，形状为[N1]                                    |
  | $cmpRatio$          | cmpKv的压缩倍率，用于换算cmp侧mask的压缩前KV长度            |

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用`aclnnMixedQuantSparseFlashMlaGetWorkspaceSize`接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnMixedQuantSparseFlashMla`执行实际计算。

```Cpp
aclnnStatus aclnnMixedQuantSparseFlashMlaGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *oriKvOptional,
    const aclTensor *cmpKvOptional,
    const aclTensor *oriSparseIndicesOptional,
    const aclTensor *cmpSparseIndicesOptional,
    const aclTensor *oriBlockTableOptional,
    const aclTensor *cmpBlockTableOptional,
    const aclTensor *cuSeqlensQOptional,
    const aclTensor *cuSeqlensOriKvOptional,
    const aclTensor *cuSeqlensCmpKvOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedOriKvOptional,
    const aclTensor *sequsedCmpKvOptional,
    const aclTensor *cmpResidualKvOptional,
    const aclTensor *oriTopkLengthOptional,
    const aclTensor *cmpTopkLengthOptional,
    const aclTensor *sinksOptional,
    const aclTensor *metadataOptional,
    int64_t          quantMode,
    int64_t          ropeHeadDim,
    double           softmaxScale,
    int64_t          cmpRatio,
    int64_t          oriMaskMode,
    int64_t          cmpMaskMode,
    int64_t          oriWinLeft,
    int64_t          oriWinRight,
    char            *layoutQOptional,
    char            *layoutKvOptional,
    int64_t          topkValueMode,
    bool             returnSoftmaxLse,
    const aclTensor *attnOutOut,
    const aclTensor *softmaxLseOutOptional,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnMixedQuantSparseFlashMla(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnMixedQuantSparseFlashMlaGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1500px"><colgroup>
    <col style="width: 200px">
    <col style="width: 100px">
    <col style="width: 300px">
    <col style="width: 300px">
    <col style="width: 120px">
    <col style="width: 100px">
    <col style="width: 280px">
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
    </tr></thead>
  <tbody>
    <tr>
      <td>q（aclTensor*）</td>
      <td>输入</td>
      <td>Query输入张量。</td>
      <td>不支持空Tensor。qN支持1-128；D仅支持512。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>
        <ul>
          <li>layoutQ为BSND时：(b, qS, qN, qD)</li>
          <li>layoutQ为TND时：(qT, qN, qD)</li>
        </ul>
      </td>
      <td>√</td>
    </tr>
    <tr>
      <td>oriKvOptional（aclTensor*）</td>
      <td>输入</td>
      <td>原始KV输入张量，Key与Value共享同一份数据。</td>
      <td>SWA/CSA/HCA场景必须传入。量化KV布局由quantMode决定：quant_mode为1时，依次由rope（64，bfloat16）、nope（448，FLOAT8_E4M3FN）、scale（7，bfloat16）、pad（18B）拼接而成；quant_mode为2时，依次由nope（448，FLOAT8_E4M3FN）、rope（64，bfloat16）、scale（7，FLOAT8_E8M0）、pad（1B）拼接而成。当前仅支持1和2，量化模式2仅支持layout_kv为PA_BBND。各量化模式均支持使用UINT8、FLOAT8_E4M3FN作为单字节存储视图，底层字节内容保持不变。</td>
      <td>详见quantMode</td>
      <td>ND</td>
      <td>
        <ul>
          <li>layoutKv为PA_BBND时：(oriKvBlockNums, oriKvBlockSize, kvN, kvD)，oriKvBlockSize支持1到1024</li>
          <li>layoutKv为BSND时：(b, oriKvS, kvN, kvD)</li>
          <li>layoutKv为TND时：(oriKvT, kvN, kvD)</li>
        </ul>
        kvN仅支持1，kvD由quantMode决定。
      </td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpKvOptional（aclTensor*）</td>
      <td>输入</td>
      <td>压缩KV输入张量，Key与Value共享同一份数据。</td>
      <td>CSA/HCA场景必须传入，SWA场景不传入。量化KV布局由quantMode决定，同oriKvOptional。</td>
      <td>详见quantMode</td>
      <td>ND</td>
      <td>
        <ul>
          <li>layoutKv为PA_BBND时：(cmpKvBlockNums, cmpKvBlockSize, kvN, kvD)，cmpKvBlockSize支持1到1024</li>
          <li>layoutKv为BSND时：(b, cmpKvS, kvN, kvD)</li>
          <li>layoutKv为TND时：(cmpKvT, kvN, kvD)</li>
        </ul>
        kvN仅支持1，kvD由quantMode决定。
      </td>
      <td>√</td>
    </tr>
    <tr>
      <td>oriSparseIndicesOptional（aclTensor*）</td>
      <td>输入</td>
      <td>代表离散取oriKvCache的索引。</td>
      <td>可选输入，无效位置填-1，其余为非负整数。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>
        <ul>
          <li>layoutQ为BSND时：(b, qS, kvN, oriKvK)</li>
          <li>layoutQ为TND时：(qT, kvN, oriKvK)</li>
        </ul>
        其中oriKvK为对oriKvOptional的TopK稀疏选择数，范围支持大于0。
      </td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpSparseIndicesOptional（aclTensor*）</td>
      <td>输入</td>
      <td>代表离散取cmpKvCache的TopK索引。</td>
      <td>cmpKv稀疏场景必须传入，其他不传入。无效位置填-1，其余为非负整数。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>
        <ul>
          <li>layoutQ为BSND时：(b, qS, kvN, cmpKvK)</li>
          <li>layoutQ为TND时：(qT, kvN, cmpKvK)</li>
        </ul>
        其中cmpKvK为对cmpKvOptional的TopK稀疏选择数，范围支持大于0。
      </td>
      <td>√</td>
    </tr>
    <tr>
      <td>oriBlockTableOptional（aclTensor*）</td>
      <td>输入</td>
      <td>PageAttention中oriKvCache存储使用的block映射表。</td>
      <td>layoutKv为PA_BBND时必须传入。第二维长度不小于所有batch中最大的S2对应的block数量。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(b, Ceil(oriKvSMax/oriKvBlockSize))</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpBlockTableOptional（aclTensor*）</td>
      <td>输入</td>
      <td>PageAttention中cmpKvCache存储使用的block映射表。</td>
      <td>cmpKv传入且layoutKv为PA_BBND时必须传入。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(b, Ceil(cmpKvSMax/cmpKvBlockSize))</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqlensQOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中q的有效token数（前缀和形式）。</td>
      <td>layoutQOptional为TND时必须传入。每个元素表示当前batch与之前所有batch的token数总和。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(b+1,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqlensOriKvOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中oriKv的有效token数（前缀和形式）。</td>
      <td>layoutKvOptional为TND时必须传入。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(b+1,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqlensCmpKvOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中cmpKv的有效token数（前缀和形式）。</td>
      <td>layoutKvOptional为TND且存在cmpKvOptional时必须传入。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(b+1,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedQOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中q实际参与运算的token数。</td>
      <td>当前暂不支持指定该参数。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(b,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedOriKvOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中oriKv实际参与运算的token数。</td>
      <td>layoutKvOptional为PA_BBND时必须传入；layoutKvOptional为BSND时可选传入，用于指定每个batch的oriKv有效长度；layoutKvOptional为TND时使用cuSeqlensOriKvOptional表达序列边界。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(b,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedCmpKvOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中cmpKv实际参与运算的token数。</td>
      <td>可选输入。传入时shape必须为(B,)，作为每个batch的cmp逻辑有效长度，优先于cmpKvOptional shape、cuSeqlensCmpKvOptional或PA block table推导。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(b,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpResidualKvOptional（aclTensor*）</td>
      <td>输入</td>
      <td>压缩KV余数，用于恢复cmp侧mask使用的压缩前KV长度。</td>
      <td>可选输入。传入时shape必须为(B,)，第b个batch按cmp_len * cmpRatio + cmpResidualKvOptional[b]恢复压缩前KV长度；在cmpRatio不等于1且cmpMaskMode为3场景必传。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(b,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>oriTopkLengthOptional（aclTensor*）</td>
      <td>输入</td>
      <td>用于标识oriSparseIndicesOptional实际参与计算的长度。</td>
      <td>oriMaskMode=0且传入oriSparseIndicesOptional时必传；其他场景不支持传入。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>
        <ul>
          <li>layoutQ为BSND时：(b, qS, kvN)</li>
          <li>layoutQ为TND时：(qT, kvN)</li>
        </ul>
      </td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpTopkLengthOptional（aclTensor*）</td>
      <td>输入</td>
      <td>用于标识cmpSparseIndicesOptional实际参与计算的长度。</td>
      <td>cmpMaskMode=0且传入cmpSparseIndicesOptional时必传；其他场景不支持传入。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>
        <ul>
          <li>layoutQ为BSND时：(b, qS, kvN)</li>
          <li>layoutQ为TND时：(qT, kvN)</li>
        </ul>
      </td>
      <td>√</td>
    </tr>
    <tr>
      <td>sinksOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示各注意力头设置独立可学习虚拟偏移项，用于维持长文本推理时的稳定性。</td>
      <td>必须传入。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(qN,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>metadataOptional（aclTensor*）</td>
      <td>输入</td>
      <td>AICPU算子aclnnMixedQuantSparseFlashMlaMetadata的分核信息。</td>
      <td>必须传入。由aclnnMixedQuantSparseFlashMlaMetadata算子生成。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(1024,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>quantMode（int64_t）</td>
      <td>输入</td>
      <td>表示量化模式。</td>
      <td>表示量化模式。量化模式1表示K、V nope为per-token-group量化，scale类型为bfloat16，量化模式2表示K、V nope为per-token-group量化，scale类型为FLOAT8_E8M0。当前仅支持1和2，量化模式2仅支持layout_kv为PA_BBND。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ropeHeadDim（int64_t）</td>
      <td>输入</td>
      <td>表示rope头的维度。</td>
      <td>仅支持64。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmaxScale（double）</td>
      <td>输入</td>
      <td>缩放系数，对应公式中的softmaxScale。</td>
      <td>建议值为1/√D，其中D为每个注意力头的维度。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpRatio（int64_t）</td>
      <td>输入</td>
      <td>cmpKv相对于压缩前KV长度的压缩倍率，用于恢复cmp侧mask使用的压缩前KV长度。</td>
      <td>cmpRatio支持1到128。在cmpKv未传入时，仅支持默认值1。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriMaskMode（int64_t）</td>
      <td>输入</td>
      <td>q和oriKv计算的mask模式。</td>
      <td>支持：<br/>0: No mask。<br/>3: rightDownCausal模式。<br/>4: sliding window模式。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpMaskMode（int64_t）</td>
      <td>输入</td>
      <td>q和cmpKv计算的mask模式。</td>
      <td>支持：<br/>0: No mask。<br/>3: rightDownCausal模式。cmpKv未传入时仅支持默认值0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriWinLeft（int64_t）</td>
      <td>输入</td>
      <td>表示q和oriKvOptional计算中q对历史token计算的数量。</td>
      <td>支持-1或非负数，-1表示无穷大，即全部参与运算。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriWinRight（int64_t）</td>
      <td>输入</td>
      <td>表示q和oriKvOptional计算中q对未来token计算的数量。</td>
      <td>支持-1或非负数，-1表示无穷大，即全部参与运算。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQOptional（char*）</td>
      <td>输入</td>
      <td>标识输入q的数据排布格式。</td>
      <td>支持"BSND"和"TND"。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKvOptional（char*）</td>
      <td>输入</td>
      <td>标识输入oriKvOptional和cmpKvOptional的数据排布格式。</td>
      <td>支持"PA_BBND"、"BSND"和"TND"。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topkValueMode（int64_t）</td>
      <td>输入</td>
      <td>topk索引取值模式。</td>
      <td>当前支持1。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>returnSoftmaxLse（bool）</td>
      <td>输入</td>
      <td>是否返回softmaxLse。</td>
      <td>支持true或false。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attnOutOut（aclTensor*）</td>
      <td>输出</td>
      <td>注意力计算输出。</td>
      <td>-</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>与q的shape一致</td>
      <td>×</td>
    </tr>
    <tr>
      <td>softmaxLseOutOptional（aclTensor*）</td>
      <td>输出</td>
      <td>softmax的log-sum-exp结果。</td>
      <td>returnSoftmaxLse为false时返回占位Tensor；returnSoftmaxLse为true时返回softmax的log-sum-exp结果。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>
        <ul>
          <li>layoutQ为BSND时：(b, kvN, qS, qN/kvN)</li>
          <li>layoutQ为TND时：(kvN, qT, qN/kvN)</li>
          <li>returnSoftmaxLse为false时：占位Tensor</li>
        </ul>
      </td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
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

- **常见字段释义**

|    命名    |                            含义                            |
| :---------: | :---------------------------------------------------------: |
|      b      |      输入样本batch大小                |
|     qS     |      输入q的序列长度      |
|    oriKvS    |  输入oriKvOptional的序列长度  |
|    cmpKvS    |  输入cmpKvOptional的序列长度  |
|     qN     |        输入q的头数        |
|    kvN    |    输入oriKvOptional/cmpKvOptional的头数    |
|      qD      |          输入q的注意力头的维度         |
|      kvD      |          输入oriKvOptional/cmpKvOptional的注意力头的维度         |
|     qT     |          输入q所有batch序列长度的累加和          |
|     oriKvT    |          输入oriKvOptional所有batch序列长度的累加和          |
|     cmpKvT    |          输入cmpKvOptional所有batch序列长度的累加和          |
|      oriKvK      |           输入oriSparseIndicesOptional中topK选出的token个数         |
|      cmpKvK      |           输入cmpSparseIndicesOptional中topK选出的token个数         |
|      oriKvSMax      |           输入oriKvOptional的最大序列长度         |
|      cmpKvSMax      |           输入cmpKvOptional的最大序列长度         |
|      oriKvBlockSize      |           输入oriKvOptional在PagedAttention场景下的block大小         |
|      cmpKvBlockSize      |           输入cmpKvOptional在PagedAttention场景下的block大小         |
|      oriKvBlockNums      |           输入oriKvOptional在PagedAttention场景下的block数量         |
|      cmpKvBlockNums      |           输入cmpKvOptional在PagedAttention场景下的block数量         |

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1200px"><colgroup>
  <col style="width: 262px">
  <col style="width: 121px">
  <col style="width: 817px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入参数是必选输入、输出或者必选属性，且是空指针。</td>
    </tr>
    <tr>
      <td rowspan="1">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="1">161002</td>
      <td>输入变量的数据类型、数据格式、属性值不在支持的范围内。</td>
    </tr>
  </tbody>
  </table>

## aclnnMixedQuantSparseFlashMla

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1154px"><colgroup>
  <col style="width: 153px">
  <col style="width: 121px">
  <col style="width: 880px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnMixedQuantSparseFlashMlaGetWorkspaceSize获取。</td>
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

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算

  - aclnnMixedQuantSparseFlashMla默认采用确定性实现，相同输入多次调用结果一致。

- 使用约束

  - 参数cuSeqlensQOptional、cuSeqlensOriKvOptional、cuSeqlensCmpKvOptional、sequsedQOptional、sequsedOriKvOptional、sequsedCmpKvOptional、cmpResidualKvOptional、oriBlockTableOptional、cmpBlockTableOptional等输入属于tensor。由于算子在Tiling阶段无法获取tensor的具体数值，tiling侧不对值进行校验，正确性需要用户自行保证。若上述参数传入非法值，会触发未定义行为（精度问题、非法内存访问导致的程序崩溃等）。
  - aclnnMixedQuantSparseFlashMlaMetadata和aclnnMixedQuantSparseFlashMla的入参在调用时应该保持一致。由于算子分为两个接口分段调用，算子无法自行校验，正确性需要由用户自行保证。若接口传入参数不一致，会发生未定义行为（精度问题、非法内存访问导致的程序崩溃等）。
  - oriTopkLengthOptional、cmpTopkLengthOptional表示ori/cmp sparseIndices实际参与计算的长度。其值不能大于sparseIndicesOptional的最后一维大小，且当sequsedQOptional传入时，topkLength对应有效部分的值需要大于等于0。
  - 当oriMaskMode/cmpMaskMode为0时，oriKvK/cmpKvK需要大于等于oriTopkLengthOptional/cmpTopkLengthOptional的最大值。
  - cmpResidualKvOptional配合cmpRatio使用，可恢复压缩前KV长度。且每个batch的值需要小于cmpRatio。
  - attnOutOut：tensor类型，公式中的输出，数据类型支持BFLOAT16。数据格式支持ND。限制：该输出参数的shape与入参q的shape保持一致，dtype与q一致。
  - returnSoftmaxLse=False时返回shape为[1]的值为0的tensor；returnSoftmaxLse=True时返回FLOAT32的log-sum-exp结果。
  - cuSeqlensQOptional、cuSeqlensOriKvOptional、cuSeqlensCmpKvOptional须满足首元素为0，且序列整体呈非递减排列，即任一元素不小于其前一个元素。
  - 当layoutKv为PA_BBND时，oriKvOptional和cmpKvOptional支持0轴非连续。
  - 各参数shape中以相同符号表示的维度，其对应轴的实际数值需保持一致。

### 特性参数组

|      特性参数组      |     参数字段名称     |
| :-------------------: | :-------------------: |
|      公共参数组      | q、quantMode、oriKvOptional、cmpKvOptional、metadataOptional、ropeHeadDim、softmaxScale、layoutQ、layoutKv、attnOutOut |
|      Mask参数组      | oriMaskMode、cmpMaskMode、oriWinLeft、oriWinRight |
|   SeqLens参数组   | cuSeqlensQOptional、cuSeqlensOriKvOptional、cuSeqlensCmpKvOptional、sequsedQOptional、sequsedOriKvOptional、sequsedCmpKvOptional |
|   稀疏压缩参数组    | cmpRatio、cmpResidualKvOptional、oriSparseIndicesOptional、cmpSparseIndicesOptional、oriTopkLengthOptional、cmpTopkLengthOptional、topkValueMode |
| Paged Attention参数组 | oriBlockTableOptional、cmpBlockTableOptional |
|   Sinks参数组   | sinksOptional |
|   SoftmaxLse参数组   | returnSoftmaxLse、softmaxLseOutOptional |

### 计算模式说明

|    命名    |    典型场景需传入参数    |    全稀疏场景需传入参数    |
| :---------: | :--------------------------------: | :---------------------------: |
|      SWA      | oriKvOptional | oriKvOptional、oriSparseIndicesOptional |
|      HCA      | oriKvOptional、cmpKvOptional|-|
|      CSA      | oriKvOptional、cmpKvOptional、cmpSparseIndicesOptional| oriKvOptional、oriSparseIndicesOptional、cmpKvOptional、cmpSparseIndicesOptional|

### 参数组约束

#### 公共参数组

- 入参为空的场景处理：
    - 空tensor指必选输入、某调用场景下必传输入和输出的shape size为0，即有任意轴为0。
    - 触发空tensor的用例将全部拦截报错。

- q、oriKvOptional、cmpKvOptional、attnOutOut校验

<table style="undefined;table-layout: fixed; width:1625px"><colgroup>
<col style="width: 147px">
<col style="width: 232px">
<col style="width: 232px">
<col style="width: 293px">
<col style="width: 185px">
</colgroup>
<thead>
<tr>
    <th>参数</th>
    <th>单参数校验</th>
    <th>存在性校验</th>
    <th>一致性校验</th>
    <th>特性交叉校验</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>q</td>
        <td>
            <ul>
                <li>dtype支持BFLOAT16</li>
                <li>layoutQ为BSND时，q的shape为(b, qS, qN, qD)</li>
                <li>layoutQ为TND时，q的shape为(qT, qN, qD)</li>
            </ul>
        </td>
        <td>
            必须传入
        </td>
        <td rowspan="4">
            <ul>
                <li>q、attnOutOut的dtype、shape需相同</li>
                <li>若cmpKvOptional传入，oriKvOptional与cmpKvOptional的dtype需一致</li>
                <li>layoutKv不为PA_BBND时，layoutQ和layoutKv需保持一致</li>
                <li>layoutKv为PA_BBND时，layoutQ可为BSND或TND</li>
            </ul>
        </td>
        <td rowspan="4">
            轴校验：
            <ul>
                <li>b > 0</li>
                <li>qS > 0</li>
                <li>0 < qN <= 128</li>
                <li>qD = 512</li>
                <li>qT > 0</li>
                <li>oriKvS > 0</li>
                <li>cmpKvS > 0</li>
                <li>kvN = 1</li>
                <li>quantMode=1时kvD=608，quantMode=2时kvD=584</li>
                <li>oriKvT > 0</li>
                <li>cmpKvT > 0</li>
                <li>oriKvBlockNums > 0</li>
                <li>cmpKvBlockNums > 0</li>
                <li>1 <= oriKvBlockSize <= 1024</li>
                <li>1 <= cmpKvBlockSize <= 1024</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>oriKvOptional</td>
        <td>
            <ul>
                <li>dtype支持FLOAT8_E4M3FN</li>
                <li>layoutKv为BSND时，oriKvOptional的shape为(b, oriKvS, kvN, kvD)</li>
                <li>layoutKv为TND时，oriKvOptional的shape为(oriKvT, kvN, kvD)</li>
                <li>layoutKv为PA_BBND时，oriKvOptional的shape为(oriKvBlockNums, oriKvBlockSize, kvN, kvD)</li>
            </ul>
        </td>
        <td>
            当前版本必传
        </td>
    </tr>
    <tr>
        <td>attnOutOut</td>
        <td>
            <ul>
                <li>dtype支持BFLOAT16</li>
                <li>layoutQ为BSND时，attnOutOut的shape为(b, qS, qN, qD)</li>
                <li>layoutQ为TND时，attnOutOut的shape为(qT, qN, qD)</li>
            </ul>
        </td>
        <td>
            必须传入
        </td>
    </tr>
    <tr>
        <td>cmpKvOptional</td>
        <td>
            <ul>
                <li>dtype支持FLOAT8_E4M3FN</li>
                <li>layoutKv为BSND时，cmpKvOptional的shape为(b, cmpKvS, kvN, kvD)</li>
                <li>layoutKv为TND时，cmpKvOptional的shape为(cmpKvT, kvN, kvD)</li>
                <li>layoutKv为PA_BBND时，cmpKvOptional的shape为(cmpKvBlockNums, cmpKvBlockSize, kvN, kvD)</li>
            </ul>
        </td>
        <td>
            可选输入
        </td>
    </tr>
</tbody>
</table>
layout匹配关系表：
<table style="undefined;table-layout: fixed; width:1625px"><colgroup>
<col style="width: 400px">
<col style="width: 400px">
<col style="width: 400px">
<col style="width: 400px">
</colgroup>
<thead>
<tr>
    <th>layoutQ</th>
    <th>layoutKv</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>BSND</td>
        <td>
          <li>BSND</li>
          <li>PA_BBND</li>
        </td>
    </tr>
    <tr>
        <td>TND</td>
        <td>
          <li>TND</li>
          <li>PA_BBND</li>
        </td>
    </tr>
</tbody>
</table>

metadataOptional校验
<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>单参数校验</th>
            <th>存在性校验</th>
            <th>一致性校验</th>
            <th>特性交叉校验</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>metadataOptional</td>
            <td>
                <ul>
                    <li>dtype仅支持INT32</li>
                    <li>shape由aclnnMixedQuantSparseFlashMlaMetadata动态计算</li>
                </ul>
            </td>
            <td>当前版本必传</td>
            <td>无</td>
            <td>传入时需与aclnnMixedQuantSparseFlashMlaMetadata生成的结果一致</td>
        </tr>
    </tbody>
</table>

#### Mask参数组

<ul>
    <li>oriMaskMode/cmpMaskMode=0，全计算模式（默认值）</li>
    <li>oriMaskMode/cmpMaskMode=3，Causal模式</li>
    <li>oriMaskMode=4，SlidingWindow模式</li>
</ul>

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 200px">
        <col style="width: 100px">
        <col style="width: 200px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>单参数校验</th>
            <th>存在性校验</th>
            <th>一致性校验</th>
            <th>特性交叉校验</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>oriMaskMode</td>
            <td>
                <ul>
                    <li>dtype支持INT32</li>
                    <li>支持输入范围仅为0、3、4，默认值为0</li>
                </ul>
            </td>
            <td>
                可选，如果不传该参数，默认值为0
            </td>
            <td>
                <ul>
                    <li>无</li>
                </ul>
            </td>
            <td>
                <ul>
                     <li>oriMaskMode支持0、3、4</li>
                 </ul>
             </td>
         </tr>
         <tr>
             <td>cmpMaskMode</td>
             <td>
                 <ul>
                     <li>dtype支持INT32</li>
                     <li>支持输入范围仅为0、3，默认值为0</li>
                 </ul>
             </td>
             <td>
                 可选，如果不传该参数，默认值为0
             </td>
             <td>
                 <ul>
                     <li>无</li>
                 </ul>
             </td>
             <td>
                    <li>cmpKvOptional未传入时，cmpMaskMode必须为0 </li>
            </td>
        </tr>
        <tr>
            <td>oriWinLeft/oriWinRight</td>
            <td>
                <ul>
                    <li>dtype支持INT32</li>
                    <li>支持输入范围仅为-1，>=0</li>
                </ul>
            </td>
            <td>
                可选，如果不传该参数，默认值为-1
            </td>
            <td>
                <ul>
                    无
                </ul>
            </td>
            <td>
                    <li>只有oriMaskMode为4时，oriWinLeft/oriWinRight可以>=0</li>
            </td>
        </tr>
    </tbody>
</table>

#### SeqLens参数组

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>单参数校验</th>
            <th>存在性校验</th>
            <th>一致性校验</th>
            <th>特性交叉校验</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>sequsedQOptional</td>
            <td >
                <ul>
                    <li>dtype支持INT32</li>
                    <li>shape为(b,)</li>
                    <li>取值仅支持非负整数</li>
                    <li>sequsedQOptional中的值需小于等于qS</li>
                </ul>
            </td>
            <td>无</td>
            <td>无</td>
            <td>无</td>
        </tr>
        <tr>
            <td>sequsedOriKvOptional</td>
            <td >
                <ul>
                    <li>dtype支持INT32</li>
                    <li>shape为(b,)</li>
                    <li>取值仅支持非负整数</li>
                    <li>sequsedOriKvOptional中的值需小于等于kvS</li>
                </ul>
            </td>
            <td >
                <ul>
                    <li>当layoutKv为BSND时，可选传入</li>
                    <li>当layoutKv为PA_BBND时，必须传入</li>
                    <li>当oriTopkLengthOptional传入时，可以不传</li>
                </ul>
            </td>
            <td>无</td>
            <td>无</td>
        </tr>
        <tr>
            <td>sequsedCmpKvOptional</td>
            <td >
                <ul>
                    <li>dtype支持INT32</li>
                    <li>shape为(b,)</li>
                    <li>取值仅支持非负整数</li>
                    <li>sequsedCmpKvOptional中的值需小于等于kvS</li>
                </ul>
            </td>
            <td >
                <ul>
                    <li>当layoutKv为BSND时，可选传入</li>
                    <li>当layoutKv为PA_BBND且cmpKvOptional传入时，必须传入</li>
                    <li>当cmpTopkLengthOptional传入时，可以不传</li>
                </ul>
            </td>
            <td>无</td>
            <td>无</td>
        </tr>
        <tr>
            <td>cuSeqlensQOptional</td>
            <td>
                <ul>
                    <li>dtype支持INT32</li>
                    <li>shape为(b+1,)</li>
                    <li>取值仅支持非负整数</li>
                    <li>其值应非递减（大于等于前一个值）排列，第一个元素为0且最后一个元素等于qT</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>当layoutQ为TND时，必须传入</li>
                    <li>当layoutQ不为TND时，不支持传入</li>
                </ul>
            </td>
            <td>无</td>
            <td>无</td>
        </tr>
        <tr>
            <td>cuSeqlensOriKvOptional</td>
            <td>
                <ul>
                    <li>dtype支持INT32</li>
                    <li>shape为(b+1,)</li>
                    <li>取值仅支持非负整数</li>
                    <li>其值应非递减（大于等于前一个值）排列，第一个元素为0且最后一个元素等于oriKvT</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>当layoutKv为TND时，必须传入</li>
                    <li>当layoutKv不为TND时，不支持传入</li>
                </ul>
            </td>
            <td>无</td>
            <td>无</td>
        </tr>
        <tr>
            <td>cuSeqlensCmpKvOptional</td>
            <td>
                <ul>
                    <li>dtype支持INT32</li>
                    <li>shape为(b+1,)</li>
                    <li>取值仅支持非负整数</li>
                    <li>其值应非递减（大于等于前一个值）排列，第一个元素为0且最后一个元素等于cmpKvT</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>当layoutKv为TND时，必须传入</li>
                    <li>当layoutKv不为TND时，不支持传入</li>
                </ul>
            </td>
            <td>无</td>
            <td>无</td>
        </tr>
    </tbody>
</table>

#### 稀疏压缩参数组

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>单参数校验</th>
            <th>存在性校验</th>
            <th>一致性校验</th>
            <th>特性交叉校验</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>cmpRatio</td>
            <td>
                <ul>
                    <li>data_type支持INT32</li>
                    <li>表示cmpKvOptional相对于压缩前KV长度的压缩倍率，需大于0</li>
                    <li>默认值为1</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>可选，默认值为1</li>
                </ul>
            </td>
            <td>无</td>
            <td>
                <ul>
                    <li>cmpKv未传入时，仅支持默认值1。</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>cmpResidualKvOptional</td>
            <td>
                <ul>
                    <li>dtype支持INT32</li>
                    <li>shape为(b,)</li>
                    <li>取值仅支持非负整数</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>只有cmpKvOptional传入才校验</li>
                    <li>可选</li>
                    <li>当cmpMaskMode=3且cmpRatio!=1时，必传</li>
                </ul>
            </td>
            <td>
                <ul>无</ul>
            </td>
            <td>无</td>
        </tr>
        <tr>
            <td>oriSparseIndicesOptional</td>
            <td>
                <ul>
                    <li>dtype支持INT32</li>
                    <li>shape为(qT, kvN, oriKvK)或(b, qS, kvN, oriKvK)</li>
                    <li>无效位置填-1，其余为非负整数</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>可选</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>当layoutQ为TND时，该shape为(qT, kvN, oriKvK)</li>
                    <li>当layoutQ为BSND时，该shape为(b, qS, kvN, oriKvK)</li>
                </ul>
            </td>
            <td>无</td>
        </tr>
        <tr>
            <td>cmpSparseIndicesOptional</td>
            <td>
                <ul>
                    <li>dtype支持INT32</li>
                    <li>shape为(qT, kvN, cmpKvK)或(b, qS, kvN, cmpKvK)</li>
                    <li>无效位置填-1，其余为非负整数</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>只有cmpKvOptional传入才校验</li>
                    <li>可选</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>当layoutQ为TND时，该shape为(qT, kvN, cmpKvK)</li>
                    <li>当layoutQ为BSND时，该shape为(b, qS, kvN, cmpKvK)</li>
                </ul>
            </td>
            <td>无</td>
        </tr>
        <tr>
            <td>oriTopkLengthOptional</td>
            <td>
                <ul>
                    <li>dtype支持INT32</li>
                    <li>shape为(b, qS, kvN)或(qT, kvN)</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>oriMaskMode=0且oriSparseIndicesOptional不为空时，必须传入</li>
                    <li>其他场景不支持传入</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>当layoutQ为TND时，该shape为(qT, kvN)</li>
                    <li>当layoutQ为BSND时，该shape为(b, qS, kvN)</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>当oriMaskMode不为0时，不支持传入</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>cmpTopkLengthOptional</td>
            <td>
                <ul>
                    <li>dtype支持INT32</li>
                    <li>shape为(b, qS, kvN)或(qT, kvN)</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>只有cmpKvOptional传入才校验</li>
                    <li>cmpMaskMode=0且cmpSparseIndicesOptional不为空时，必须传入</li>
                    <li>其他场景不支持传入</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>当layoutQ为TND时，该shape为(qT, kvN)</li>
                    <li>当layoutQ为BSND时，该shape为(b, qS, kvN)</li>
                </ul>
            </td>
            <td>无</td>
        </tr>
        <tr>
            <td>topkValueMode</td>
            <td>
                <ul>
                    <li>data_type支持INT32</li>
                    <li>topK索引取值模式，默认值为1</li>
                </ul>
            </td>
            <td>可选属性，默认值为1</td>
            <td>无</td>
            <td>无</td>
        </tr>
    </tbody>
</table>

#### Paged Attention参数组

当layoutKv为PA_BBND时，开启Paged Attention

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>单参数校验</th>
            <th>存在性校验</th>
            <th>一致性校验</th>
            <th>特性交叉校验</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>oriBlockTableOptional</td>
            <td>
                <ul>
                    <li>dtype仅支持INT32</li>
                    <li>shape为(b, Ceil(oriKvSMax/oriKvBlockSize))</li>
                    <li>值只能为正整数</li>
                </ul>
            </td>
            <td>可选</td>
            <td>无</td>
            <td>
                <ul>
                    <li>oriBlockTableOptional存在时，必须传入sequsedOriKvOptional</li>
                    <li>PagedAttention开启情况下，blockTable必须不为空</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>cmpBlockTableOptional</td>
            <td>
                <ul>
                    <li>dtype仅支持INT32</li>
                    <li>shape为(b, Ceil(cmpKvSMax/cmpKvBlockSize))</li>
                    <li>值只能为正整数</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>只有cmpKvOptional传入才校验</li>
                    <li>可选</li>
                </ul>
            </td>
            <td>无</td>
            <td>
                <ul>
                    <li>cmpBlockTableOptional存在时，必须传入sequsedCmpKvOptional</li>
                    <li>PagedAttention开启情况下，blockTable必须不为空</li>
                </ul>
            </td>
        </tr>
    </tbody>
</table>

#### Sinks参数组

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>单参数校验</th>
            <th>存在性校验</th>
            <th>一致性校验</th>
            <th>特性交叉校验</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>sinksOptional</td>
            <td>
                <ul>
                    <li>dtype支持FLOAT32</li>
                    <li>shape为(qN, )</li>
                </ul>
            </td>
            <td> 当前版本必传 </td>
            <td> 无 </td>
            <td> 无 </td>
        </tr>
    </tbody>
</table>

#### SoftmaxLse参数组

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>单参数校验</th>
            <th>存在性校验</th>
            <th>一致性校验</th>
            <th>特性交叉校验</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>returnSoftmaxLse</td>
            <td>
                <ul>
                    <li>data_type仅支持bool</li>
                    <li>true代表开启softmaxLse，false代表关闭softmaxLse</li>
                </ul>
            </td>
            <td>可选，默认值为false</td>
            <td rowspan="2">
                <ul>
                     <li>当returnSoftmaxLse为false时，输出shape为[1]的值为0的tensor</li>
                    <li>当returnSoftmaxLse为true时，softmaxLseOutOptional的shape与layoutQ的关系如下：<ul><li>layoutQ为BSND时，softmaxLseOutOptional的shape为(b, kvN, qS, qN/kvN)</li><li>layoutQ为TND时，softmaxLseOutOptional的shape为(kvN, qT, qN/kvN)</li></ul></li>
                </ul>
            </td>
            <td>无</td>
        </tr>
        <tr>
            <td>softmaxLseOutOptional</td>
            <td>
                <ul>
                    <li>data_type仅支持FLOAT32</li>
                </ul>
            </td>
            <td>无</td>
            <td>无</td>
        </tr>
    </tbody>
</table>

## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_mixed_quant_sparse_flash_mla.cpp
 * \brief
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_mixed_quant_sparse_flash_mla.h"
#include "aclnnop/aclnn_mixed_quant_sparse_flash_mla_metadata.h"

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

namespace {

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

uint16_t FloatToBf16(float f)
{
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  uint32_t lsb = (bits >> 16) & 1u;
  uint32_t roundingBias = 0x7fffu + lsb;
  bits += roundingBias;
  return static_cast<uint16_t>(bits >> 16);
}

float Bf16ToFloat(uint16_t h)
{
  uint32_t bits = static_cast<uint32_t>(h) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

void PrintOutResult(const std::vector<int64_t>& shape, void** deviceAddr)
{
  auto size = GetShapeSize(shape);
  std::vector<uint16_t> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size && i < 10; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, Bf16ToFloat(resultData[i]));
  }
}

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream)
{
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateContext(context, deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetCurrentContext(*context);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
  auto size = GetShapeSize(shape) * sizeof(T);
  if (size > 0) {
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
  } else {
    *deviceAddr = nullptr;
  }

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

std::vector<uint16_t> MakeBf16Data(int64_t size, float value)
{
  std::vector<uint16_t> data(static_cast<size_t>(size), FloatToBf16(value));
  return data;
}

std::vector<uint8_t> MakeFp8Data(int64_t size, uint8_t value)
{
  std::vector<uint8_t> data(static_cast<size_t>(size), value);
  return data;
}

}  // namespace

int main()
{
  int32_t deviceId = 0;
  aclrtContext context = nullptr;
  aclrtStream stream = nullptr;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  int64_t B = 1;
  int64_t S1 = 1;
  int64_t S2 = 1024;
  int64_t N1 = 64;
  int64_t N2 = 1;
  int64_t D = 512;
  int64_t K = 512;
  int64_t oriBlockSize = 128;
  int64_t cmpBlockSize = 128;
  int64_t s2Act = 1024;
  int64_t cmpRatio = 4;
  int64_t oriWinLeft = 127;
  int64_t oriWinRight = 0;
  int64_t oriMaskMode = 4;
  int64_t cmpMaskMode = 3;
  int64_t quantMode = 1;
  int64_t ropeHeadDim = 64;
  int64_t tileSize = 64;
  double softmaxScale = 1.0 / sqrt(static_cast<double>(D));

  int64_t nopeHeadDim = D - ropeHeadDim;
  int64_t quantScaleHeadDim = (nopeHeadDim + tileSize - 1) / tileSize;
  int64_t kvD = nopeHeadDim + ropeHeadDim * 2 + quantScaleHeadDim * 2 + 18;

  int64_t T1 = B * S1;
  int64_t cmpKvLen = s2Act / cmpRatio;
  int64_t oriBlockNum = ((s2Act + oriBlockSize - 1) / oriBlockSize) * B;
  int64_t cmpBlockNum = ((cmpKvLen + cmpBlockSize - 1) / cmpBlockSize) * B;

  std::vector<int64_t> qShape = {T1, N1, D};
  std::vector<int64_t> oriKvShape = {oriBlockNum, oriBlockSize, N2, kvD};
  std::vector<int64_t> cmpKvShape = {cmpBlockNum, cmpBlockSize, N2, kvD};
  std::vector<int64_t> cmpSparseIndicesShape = {T1, N2, K};
  std::vector<int64_t> oriBlockTableShape = {B, (s2Act + oriBlockSize - 1) / oriBlockSize};
  std::vector<int64_t> cmpBlockTableShape = {B, (cmpKvLen + cmpBlockSize - 1) / cmpBlockSize};
  std::vector<int64_t> cuSeqLensQShape = {B + 1};
  std::vector<int64_t> seqUsedOriKvShape = {B};
  std::vector<int64_t> seqUsedCmpKvShape = {B};
  std::vector<int64_t> cmpResidualKvShape = {B};
  std::vector<int64_t> sinksShape = {N1};
  std::vector<int64_t> metadataShape = {1024};
  std::vector<int64_t> attnOutShape = {T1, N1, D};
  std::vector<int64_t> softmaxLseShape = {T1, N1, 1};
  std::vector<int64_t> emptyShape = {0};

  void* qDeviceAddr = nullptr;
  void* oriKvDeviceAddr = nullptr;
  void* cmpKvDeviceAddr = nullptr;
  void* cmpSparseIndicesDeviceAddr = nullptr;
  void* oriBlockTableDeviceAddr = nullptr;
  void* cmpBlockTableDeviceAddr = nullptr;
  void* cuSeqLensQDeviceAddr = nullptr;
  void* cuSeqLensOriKvDeviceAddr = nullptr;
  void* cuSeqLensCmpKvDeviceAddr = nullptr;
  void* seqUsedQDeviceAddr = nullptr;
  void* seqUsedOriKvDeviceAddr = nullptr;
  void* seqUsedCmpKvDeviceAddr = nullptr;
  void* cmpResidualKvDeviceAddr = nullptr;
  void* sinksDeviceAddr = nullptr;
  void* metadataDeviceAddr = nullptr;
  void* attnOutDeviceAddr = nullptr;
  void* softmaxLseDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* oriKv = nullptr;
  aclTensor* cmpKv = nullptr;
  aclTensor* cmpSparseIndices = nullptr;
  aclTensor* oriBlockTable = nullptr;
  aclTensor* cmpBlockTable = nullptr;
  aclTensor* cuSeqLensQ = nullptr;
  aclTensor* cuSeqLensOriKv = nullptr;
  aclTensor* cuSeqLensCmpKv = nullptr;
  aclTensor* seqUsedQ = nullptr;
  aclTensor* seqUsedOriKv = nullptr;
  aclTensor* seqUsedCmpKv = nullptr;
  aclTensor* cmpResidualKv = nullptr;
  aclTensor* sinks = nullptr;
  aclTensor* metadata = nullptr;
  aclTensor* attnOut = nullptr;
  aclTensor* softmaxLse = nullptr;

  int64_t qSize = GetShapeSize(qShape);
  int64_t oriKvSize = GetShapeSize(oriKvShape);
  int64_t cmpKvSize = GetShapeSize(cmpKvShape);
  int64_t cmpSparseIndicesSize = GetShapeSize(cmpSparseIndicesShape);
  int64_t oriBlockTableSize = GetShapeSize(oriBlockTableShape);
  int64_t cmpBlockTableSize = GetShapeSize(cmpBlockTableShape);
  int64_t attnOutSize = GetShapeSize(attnOutShape);
  int64_t softmaxLseSize = GetShapeSize(softmaxLseShape);

  std::vector<uint16_t> qHostData = MakeBf16Data(qSize, 1.0f);
  std::vector<uint8_t> oriKvHostData = MakeFp8Data(oriKvSize, 0x38);
  std::vector<uint8_t> cmpKvHostData = MakeFp8Data(cmpKvSize, 0x38);
  std::vector<int32_t> cmpSparseIndicesHostData(cmpSparseIndicesSize);
  std::vector<int32_t> oriBlockTableHostData(oriBlockTableSize);
  std::iota(oriBlockTableHostData.begin(), oriBlockTableHostData.end(), 0);
  std::vector<int32_t> cmpBlockTableHostData(cmpBlockTableSize);
  std::iota(cmpBlockTableHostData.begin(), cmpBlockTableHostData.end(), 0);
  std::vector<int32_t> cuSeqLensQHostData(B + 1);
  for (int64_t i = 0; i <= B; i++) {
    cuSeqLensQHostData[i] = static_cast<int32_t>(i * S1);
  }
  std::vector<int32_t> emptyHostData;
  std::vector<int32_t> seqUsedOriKvHostData(B, static_cast<int32_t>(s2Act));
  std::vector<int32_t> seqUsedCmpKvHostData(B, static_cast<int32_t>(cmpKvLen));
  std::vector<int32_t> cmpResidualKvHostData(B);
  for (int64_t i = 0; i < B; i++) {
    cmpResidualKvHostData[i] = seqUsedOriKvHostData[i] % static_cast<int32_t>(cmpRatio);
  }
  std::vector<float> sinksHostData(N1, 1.0f);
  std::vector<int32_t> metadataHostData(1024, 0);
  std::vector<uint16_t> attnOutHostData = MakeBf16Data(attnOutSize, 0.0f);
  std::vector<float> softmaxLseHostData(softmaxLseSize, 0.0f);

  std::mt19937 gen(42);
  for (int64_t t = 0; t < T1; t++) {
    for (int64_t n = 0; n < N2; n++) {
      for (int64_t k = 0; k < K; k++) {
        cmpSparseIndicesHostData[t * N2 * K + n * K + k] = static_cast<int32_t>(gen() % cmpKvLen);
      }
    }
  }

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_BF16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(oriKvHostData, oriKvShape, &oriKvDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &oriKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cmpKvHostData, cmpKvShape, &cmpKvDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &cmpKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cmpSparseIndicesHostData, cmpSparseIndicesShape, &cmpSparseIndicesDeviceAddr,
                        aclDataType::ACL_INT32, &cmpSparseIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(oriBlockTableHostData, oriBlockTableShape, &oriBlockTableDeviceAddr, aclDataType::ACL_INT32,
                        &oriBlockTable);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cmpBlockTableHostData, cmpBlockTableShape, &cmpBlockTableDeviceAddr, aclDataType::ACL_INT32,
                        &cmpBlockTable);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cuSeqLensQHostData, cuSeqLensQShape, &cuSeqLensQDeviceAddr, aclDataType::ACL_INT32, &cuSeqLensQ);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(emptyHostData, emptyShape, &cuSeqLensOriKvDeviceAddr, aclDataType::ACL_INT32, &cuSeqLensOriKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(emptyHostData, emptyShape, &cuSeqLensCmpKvDeviceAddr, aclDataType::ACL_INT32, &cuSeqLensCmpKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(emptyHostData, emptyShape, &seqUsedQDeviceAddr, aclDataType::ACL_INT32, &seqUsedQ);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(seqUsedOriKvHostData, seqUsedOriKvShape, &seqUsedOriKvDeviceAddr, aclDataType::ACL_INT32, &seqUsedOriKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(seqUsedCmpKvHostData, seqUsedCmpKvShape, &seqUsedCmpKvDeviceAddr, aclDataType::ACL_INT32, &seqUsedCmpKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cmpResidualKvHostData, cmpResidualKvShape, &cmpResidualKvDeviceAddr, aclDataType::ACL_INT32, &cmpResidualKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sinksHostData, sinksShape, &sinksDeviceAddr, aclDataType::ACL_FLOAT, &sinks);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(metadataHostData, metadataShape, &metadataDeviceAddr, aclDataType::ACL_INT32, &metadata);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attnOutHostData, attnOutShape, &attnOutDeviceAddr, aclDataType::ACL_BF16, &attnOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxLseHostData, softmaxLseShape, &softmaxLseDeviceAddr, aclDataType::ACL_FLOAT, &softmaxLse);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  char layoutQ[] = "TND";
  char layoutKv[] = "PA_BBND";

  uint64_t metadataWorkspaceSize = 0;
  aclOpExecutor* metadataExecutor = nullptr;

  ret = aclnnMixedQuantSparseFlashMlaMetadataGetWorkspaceSize(
      cuSeqLensQ, cuSeqLensOriKv, cuSeqLensCmpKv,
      seqUsedQ, seqUsedOriKv, seqUsedCmpKv,
      cmpResidualKv, nullptr, nullptr,
      N1, N2, D, quantMode, B,
      S1, S2, cmpKvLen,
      0, K, ropeHeadDim,
      cmpRatio, oriMaskMode, cmpMaskMode,
      oriWinLeft, oriWinRight,
      layoutQ, layoutKv,
      true, true,
      metadata,
      &metadataWorkspaceSize, &metadataExecutor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMixedQuantSparseFlashMlaMetadataGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* metadataWorkspaceAddr = nullptr;
  if (metadataWorkspaceSize > 0) {
    ret = aclrtMalloc(&metadataWorkspaceAddr, metadataWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate metadata workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnMixedQuantSparseFlashMlaMetadata(metadataWorkspaceAddr, metadataWorkspaceSize, metadataExecutor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMixedQuantSparseFlashMlaMetadata failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream after metadata failed. ERROR: %d\n", ret); return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = aclnnMixedQuantSparseFlashMlaGetWorkspaceSize(
      q, oriKv, cmpKv,
      nullptr, cmpSparseIndices,
      oriBlockTable, cmpBlockTable,
      cuSeqLensQ, nullptr, nullptr,
      nullptr, seqUsedOriKv, seqUsedCmpKv,
      cmpResidualKv, nullptr, nullptr,
      sinks, metadata,
      quantMode, ropeHeadDim,
      softmaxScale, cmpRatio,
      oriMaskMode, cmpMaskMode,
      oriWinLeft, oriWinRight,
      layoutQ, layoutKv,
      1, false,
      attnOut, softmaxLse,
      &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMixedQuantSparseFlashMlaGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnMixedQuantSparseFlashMla(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMixedQuantSparseFlashMla failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  PrintOutResult(attnOutShape, &attnOutDeviceAddr);

  aclDestroyTensor(q);
  aclDestroyTensor(oriKv);
  aclDestroyTensor(cmpKv);
  aclDestroyTensor(cmpSparseIndices);
  aclDestroyTensor(oriBlockTable);
  aclDestroyTensor(cmpBlockTable);
  aclDestroyTensor(cuSeqLensQ);
  aclDestroyTensor(cuSeqLensOriKv);
  aclDestroyTensor(cuSeqLensCmpKv);
  aclDestroyTensor(seqUsedQ);
  aclDestroyTensor(seqUsedOriKv);
  aclDestroyTensor(seqUsedCmpKv);
  aclDestroyTensor(cmpResidualKv);
  aclDestroyTensor(sinks);
  aclDestroyTensor(metadata);
  aclDestroyTensor(attnOut);
  aclDestroyTensor(softmaxLse);

  aclrtFree(qDeviceAddr);
  aclrtFree(oriKvDeviceAddr);
  aclrtFree(cmpKvDeviceAddr);
  aclrtFree(cmpSparseIndicesDeviceAddr);
  aclrtFree(oriBlockTableDeviceAddr);
  aclrtFree(cmpBlockTableDeviceAddr);
  if (cuSeqLensQDeviceAddr != nullptr) {
    aclrtFree(cuSeqLensQDeviceAddr);
  }
  if (seqUsedOriKvDeviceAddr != nullptr) {
    aclrtFree(seqUsedOriKvDeviceAddr);
  }
  if (seqUsedCmpKvDeviceAddr != nullptr) {
    aclrtFree(seqUsedCmpKvDeviceAddr);
  }
  if (cmpResidualKvDeviceAddr != nullptr) {
    aclrtFree(cmpResidualKvDeviceAddr);
  }
  aclrtFree(sinksDeviceAddr);
  aclrtFree(metadataDeviceAddr);
  aclrtFree(attnOutDeviceAddr);
  aclrtFree(softmaxLseDeviceAddr);
  if (metadataWorkspaceSize > 0) {
    aclrtFree(metadataWorkspaceAddr);
  }
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtDestroyContext(context);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
