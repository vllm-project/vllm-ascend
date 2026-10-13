# aclnnGenericBlockSparseAttention

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

- **接口功能**：GenericBlockSparseAttention稀疏注意力计算，支持沿着S轴任意粒度的块稀疏模式，通过sparseBlockIdx指定每个Q块选择的KV块，sparseBlockCount指定每个Q块保留的KV块数量，实现高效的稀疏注意力计算。须先调用aclnnGenericBlockSparseAttentionMetadata生成metadata。

- **计算公式**：稀疏块大小：$blockShapeX \times blockShapeY$

  $$
  attentionOut = Softmax(scaleValue \cdot query \cdot key_{sparse}^{T} + atten\_mask) \cdot value_{sparse}
  $$

  输入排布由layoutQ、layoutKv指定，当前仅支持layoutQ="TND"、layoutKv="PA_BBND"。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnGenericBlockSparseAttentionGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnGenericBlockSparseAttention"接口执行计算。

```c++
aclnnStatus aclnnGenericBlockSparseAttentionGetWorkspaceSize(
  const aclTensor   *query,
  const aclTensor   *key,
  const aclTensor   *value,
  const aclTensor   *sparseBlockIdx,
  const aclTensor   *sparseBlockCount,
  const aclTensor   *metadataOptional,
  const aclTensor   *attenMaskOptional,
  const aclTensor   *qDequantScaleOptional,
  const aclTensor   *kDequantScaleOptional,
  const aclTensor   *vDequantScaleOptional,
  const aclTensor   *pQuantScaleOptional,
  const aclTensor   *cuSeqLengthsQOptional,
  const aclTensor   *cuSeqLengthsKvOptional,
  const aclTensor   *sequsedQOptional,
  const aclTensor   *sequsedKvOptional,
  const aclTensor   *blockTableOptional,
  const aclIntArray *blockShape,
  char              *layoutQ,
  char              *layoutKv,
  int64_t            layoutSparsePattern,
  double             scaleValue,
  int64_t            maskType,
  int64_t            quantType,
  double             dstTypeMax,
  int64_t            softmaxPrecision,
  int64_t            winLeft,
  int64_t            winRight,
  int64_t            returnSoftmaxlse,
  int64_t            residualBlockMode,
  bool               isConsistentTopK,
  aclTensor         *attentionOut,
  aclTensor         *softmaxLseOptional,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnGenericBlockSparseAttention(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  aclrtStream       stream)
```

## aclnnGenericBlockSparseAttentionGetWorkspaceSize

- **参数说明**

  表格中shape变量含义见<a href="#基准说明">基准说明</a>。

  <table style="table-layout: fixed; width: 100%; word-break: break-word; overflow-wrap: anywhere;">
  <colgroup>
  <col style="width: 12%">
  <col style="width: 6%">
  <col style="width: 15%">
  <col style="width: 25%">
  <col style="width: 11%">
  <col style="width: 5%">
  <col style="width: 20%">
  <col style="width: 6%">
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
      <td>输入</td>
      <td>公式中的query。</td>
      <td>当前仅支持layoutQ="TND"。其他layout的shape见<a href="#layout对应关系说明">layout对应关系说明</a>。</td>
      <td>FLOAT16、BFLOAT16、FLOAT8_E4M3FN、FLOAT4_E2M1FN、HIFLOAT8</td>
      <td>ND</td>
      <td>[totalQTokens, headNum, headDim]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>key</td>
      <td>输入</td>
      <td>公式中的key。</td>
      <td>可作为原始Key或Paged KV Cache。当前仅支持layoutKv="PA_BBND"。其余layout及原始KV的shape见<a href="#layout对应关系说明">layout对应关系说明</a>、<a href="#paged-attention相关说明">Paged Attention相关说明</a>；dim0非连续见<a href="#其他约束">其他约束</a>。</td>
      <td>FLOAT16、BFLOAT16、FLOAT8_E4M3FN、FLOAT4_E2M1FN、HIFLOAT8</td>
      <td>ND</td>
      <td>[numBlocks, blockSize, numKeyValueHeads, headDim]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>value</td>
      <td>输入</td>
      <td>公式中的value。</td>
      <td>可作为原始Value输入或Paged KV Cache输入，shape与key一致。当前仅支持layoutKv="PA_BBND"。</td>
      <td>FLOAT16、BFLOAT16、FLOAT8_E4M3FN、FLOAT4_E2M1FN、HIFLOAT8</td>
      <td>ND</td>
      <td>与key一致</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sparseBlockIdx</td>
      <td>输入</td>
      <td>稀疏块索引数组，指定每个Q块选择的KV块索引。</td>
      <td>取值须为合法KV块索引（按cu存储长度分块）；无效位置可用-1填充，有效值须落在前sparseBlockCount个位置。详细参考<a href="#layout对应关系说明">layout对应关系说明</a>。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>[numKeyValueHeads, totalQBlocks, maxKvBlockCount]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sparseBlockCount</td>
      <td>输入</td>
      <td>每个Q块实际选择的KV块数量。</td>
      <td>详细参考<a href="#layout对应关系说明">layout对应关系说明</a>。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>[numKeyValueHeads, totalQBlocks]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>metadataOptional</td>
      <td>输入</td>
      <td>AICPU算子aclnnGenericBlockSparseAttentionMetadata的分核结果。</td>
      <td>
        必须传入。由aclnnGenericBlockSparseAttentionMetadata生成，每次调用须重新生成。
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(1024,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>attenMaskOptional</td>
      <td>输入</td>
      <td>公式中的atten_mask。</td>
      <td>atten_mask会与稀疏pattern叠加，详见<a href="#掩码说明">掩码说明</a>。当前不支持，必须传入nullptr。</td>
      <td>BOOL</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>qDequantScaleOptional</td>
      <td>输入</td>
      <td>query的反量化缩放因子。</td>
      <td>详情见<a href="#量化相关说明">量化相关说明</a>。当前仅quantType=0/5场景可传nullptr；quantType=1~4当前不支持，须传入nullptr。
      </td>
      <td>FLOAT32、FLOAT8_E8M0</td>
      <td>ND</td>
      <td>x</td>
      <td>×</td>
    </tr>
    <tr>
      <td>kDequantScaleOptional</td>
      <td>输入</td>
      <td>key的反量化缩放因子。</td>
      <td>详情见<a href="#量化相关说明">量化相关说明</a>。当前仅quantType=0/5场景可传nullptr；quantType=1~4当前不支持，须传入nullptr。
      </td>
      <td>FLOAT32、FLOAT8_E8M0</td>
      <td>ND</td>
      <td>x</td>
      <td>×</td>
    </tr>
    <tr>
      <td>vDequantScaleOptional</td>
      <td>输入</td>
      <td>value的反量化缩放因子。</td>
      <td>详情见<a href="#量化相关说明">量化相关说明</a>。当前仅quantType=0/5场景可传nullptr；quantType=1~4当前不支持，须传入nullptr。
      </td>
      <td>FLOAT32、FLOAT8_E8M0</td>
      <td>ND</td>
      <td>x</td>
      <td>×</td>
    </tr>
    <tr>
      <td>pQuantScaleOptional</td>
      <td>输入</td>
      <td>非mx量化模式下，online-softmax的结果P矩阵所需的量化系数。</td>
      <td>详情见<a href="#量化相关说明">量化相关说明</a>。当前须传入nullptr。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>x</td>
      <td>x</td>
    </tr>
    <tr>
      <td>cuSeqLengthsQOptional</td>
      <td>输入</td>
      <td>描述每个Batch对应的query序列长度，以前缀和形式存储。</td>
      <td>
        可选输入，用于变长序列场景。
        <ul>
          <li>layoutQ为TND时必须传入。</li>
          <li>layoutQ为BNSD/BSND时：如传入，算子内按该输入指定的实际序列长度处理；如传入nullptr，按query的shape中的S处理（当前不支持）。</li>
          <li>元素为前缀和：第0个元素为0，最后一个元素等于totalQTokens，后一个元素须≥前一个元素；相邻差分得到各batch的<strong>存储长度</strong> qStorageLen_i，且满足 Σ qStorageLen_i = totalQTokens。</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>ND</td>
      <td>(B+1,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>cuSeqLengthsKvOptional</td>
      <td>输入</td>
      <td>描述每个Batch对应的key/value序列长度，以前缀和形式存储。</td>
      <td>
        可选输入，用于变长序列场景。
        <ul>
          <li>layoutKv为TND时必须传入。</li>
          <li>layoutKv非TND时不传（须为nullptr）；PA_BBND不传此参数。</li>
          <li>layoutKv为BNSD/BSND时：如传入，算子内按该输入指定的实际序列长度处理；如传入nullptr，按key/value的shape中的S处理（当前不支持）。</li>
          <li>元素为前缀和：第0个元素为0，最后一个元素等于各batch KV存储长度之和，后一个元素须≥前一个元素。</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>ND</td>
      <td>(B+1,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sequsedQOptional</td>
      <td>输入</td>
      <td>各batch中query的实际序列长度。</td>
      <td>
        <ul>
          <li>不指定实际长度可传入nullptr，表示与cuSeqLengthsQOptional差分得到的存储长度相同。</li>
          <li>传入时shape为(B,)，每个元素须≥0且≤对应batch的cu存储长度。与cu并存时的双长度语义见<a href="#其他约束">其他约束</a>。</li>
        </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sequsedKvOptional</td>
      <td>输入</td>
      <td>各batch中kv的实际序列长度。</td>
      <td>
        <ul>
          <li>layoutKv为PA_BBND时必须传入，shape为(B,)，每个元素须≥0。</li>
          <li>layoutKv为TND时：不指定实际长度可传入nullptr，表示与cuSeqLengthsKvOptional差分得到的存储长度相同；传入时shape为(B,)，每个元素须≥0且≤对应batch的cu存储长度。与cu并存时的双长度语义见<a href="#其他约束">其他约束</a>。</li>
        </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>blockTableOptional</td>
      <td>输入</td>
      <td>Block表用于PagedAttention。</td>
      <td>PA_BBND必须传入此参数；A5 TND非PA必须传入nullptr。详见<a href="#paged-attention相关说明">Paged Attention相关说明</a>。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B, maxNumBlocksPerBatch)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>blockShape</td>
      <td>输入</td>
      <td>代表稀疏块形状数组。</td>
      <td>含两个元素[blockShapeX, blockShapeY]。blockShapeX=1。Arch35非量化FP16/BF16支持1≤blockShapeY≤256的任意整数，不要求16对齐；其他场景仅支持blockShapeY=128。Attention KV tile及PA page size固定为128，与稀疏blockShapeY独立。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQ</td>
      <td>输入</td>
      <td>代表输入query的数据排布格式。</td>
      <td>目标支持"TND""BNSD""BSND"，详见<a href="#layout对应关系说明">layout对应关系说明</a>。当前仅支持"TND"。</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKv</td>
      <td>输入</td>
      <td>代表输入key、value的数据排布格式。</td>
      <td>目标支持"TND""BNSD""BSND""PA_BBND""PA_BNBD"，详见<a href="#layout对应关系说明">layout对应关系说明</a>。支持"PA_BBND"；A5新增"TND"非PA，仅支持FLOAT16/BFLOAT16、quantType=0。</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutSparsePattern</td>
      <td>输入</td>
      <td>代表输入的sparseBlockIdx、sparseBlockCount的数据排布格式。</td>
      <td>当前仅支持取"4"，详细参考<a href="#layout对应关系说明">layout对应关系说明</a>。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>输入</td>
      <td>公式中的scale，代表缩放系数。</td>
      <td>一般设置为D^-0.5；传0时算子内按1/√D处理。</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskType</td>
      <td>输入</td>
      <td>表示attention计算中的掩码类型。</td>
      <td>
        取值0~5，含义见<a href="#掩码说明">掩码说明</a>。<br>
        当前仅支持1（causal mask）。
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantType</td>
      <td>输入</td>
      <td>代表采用的量化手段。</td>
      <td>
        取值0~5，完整配置见<a href="#量化相关说明">量化相关说明</a>：<br>
        当前可用：0；Ascend 950上可选5。取值1~4传入将校验失败。
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstTypeMax</td>
      <td>输入</td>
      <td>MXFP4 CX量化时传入的自定义量化量程。</td>
      <td>
        <ul>
          <li>当前版本不支持自定义量程，必须传入0.0。</li>
        </ul>
      </td>
      <td>double</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmaxPrecision</td>
      <td>输入</td>
      <td>Softmax计算采取的精度级别。</td>
      <td>
        控制online softmax阶段以及rescale阶段运算使用的数据类型。取值0或1：
        <ul>
          <li>0：online softmax和rescale全部采取fp32，适合追求计算精度的场景。</li>
          <li>1：混合精度；online softmax采取fp16/bf16（与attentionOut相同），rescale采取fp32，online softmax阶段可能数值溢出。</li>
        </ul>
        芯片约束：Ascend 950非量化FLOAT16/BFLOAT16场景支持0或1，FP8路径仅支持1；Atlas A2/A3上FLOAT16可配置0或1，BFLOAT16仅支持0。
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>winLeft</td>
      <td>输入</td>
      <td>滑窗attention场景下，滑窗需要向前包含多少个token。</td>
      <td>用于滑窗attention；不使能时必须为-1，需与maskType配合，见<a href="#掩码说明">掩码说明</a>。当前只支持传入-1。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>winRight</td>
      <td>输入</td>
      <td>滑窗attention场景下，滑窗需要向后包含多少个token。</td>
      <td>用于滑窗attention；不使能时必须为-1，需与maskType配合，见<a href="#掩码说明">掩码说明</a>。当前只支持传入-1。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>returnSoftmaxlse</td>
      <td>输入</td>
      <td>是否使能softmaxLse输出的标志位。</td>
      <td>
    quantType=0支持0或1；FP8路径仅支持0。
    <ul>
          <li>0：表示不输出softmaxLse，softmaxLseOptional须传入nullptr</li>
          <li>1：输出FLOAT32 [Tq,Nq,1]，softmaxLseOptional必传</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>residualBlockMode</td>
      <td>输入</td>
      <td>表示KV序列以blockShapeY为单位进行稀疏后，尾部不完整块的状态。</td>
      <td>当前仅支持取0或1。0代表尾部不完整块是否参与运算由sparseBlockIdx传入的值决定，1表示尾部不完整块必定参与运算，但必定不包含在sparseBlockIdx中。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>isConsistentTopK</td>
      <td>输入</td>
      <td>前置算子进行稀疏块选择时，同一batch同一head内每个Q块选择的KV块最大数量是否一致。</td>
      <td>仅支持0或1。0代表不一致，1代表一致。</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attentionOut</td>
      <td>输出</td>
      <td>公式中的attentionOut。</td>
      <td>数据类型和shape与query保持一致。当前仅支持layoutQ="TND"。</td>
      <td>FLOAT16、BFLOAT16（FP8输入时由本tensor指定）</td>
      <td>ND</td>
      <td>与query一致</td>
      <td>√</td>
    </tr>
    <tr>
      <td>softmaxLseOptional</td>
      <td>输出</td>
      <td>Softmax计算的log-sum-exp中间结果。</td>
      <td>quantType=0且returnSoftmaxlse=1时输出FLOAT32 [Tq,Nq,1]；关闭时传入nullptr。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 280px">
  <col style="width: 90px">
  <col style="width: 785px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>query/key/value/sparseBlockIdx/sparseBlockCount/attentionOut 等必选指针为空。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>layout、maskType、blockShape、softmaxPrecision、quantType、returnSoftmaxlse、layoutSparsePattern、residualBlockMode、isConsistentTopK 等与约束不匹配。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_NULLPTR</td>
      <td>561103</td>
      <td>metadata 为空或 Contiguous/InferShape 失败（如 layout 不支持、缺少 blockTable 等）。</td>
    </tr>
  </tbody>
  </table>

## aclnnGenericBlockSparseAttention

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

  返回 aclnnStatus 状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

### 基准说明

资料约束中，常见字段释义如下：

| 命名 | 含义 |
| :---: | :--- |
| B | Batch Size，输入样本批量大小 |
| S | Sequence Length，序列长度 |
| H | Head Size，hidden层大小 |
| N | Head Num，多头数 |
| D | Head Dim，head维度，且满足D=H/N |
| T | 所有Batch序列长度的累加和 |
| totalQTokens | query的T |
| totalKTokens | key的T |
| totalVTokens | value的T |
| headNum | query的N |
| numKeyValueHeads | key/value的N |
| headDim | D |
| batch | B |
| maxQSeqLength | query在BNSD/BSND下的S |
| maxKvSeqLength | key/value在BNSD/BSND下的S |
| numBlocks | Paged KV Cache的物理页数 |
| blockSize | 每一页容纳的token数 |
| maxNumBlocksPerBatch | blockTable第二维，须≥ceilDiv(maxKvSeqLength, blockSize) |
| qStorageLen_i | cuSeqLengthsQ前缀和差分得到的各batch存储长度 |
| totalQBlocks | 按存储长度分块后的Q块总数，即$\sum_i \mathrm{ceilDiv}(qStorageLen_i, blockShapeX)$ |
| totalKBlocks | 按存储长度分块后的KV块总数，即$\sum_i \mathrm{ceilDiv}(kvStorageLen_i, blockShapeY)$ |
| maxKvBlockCount | sparseBlockIdx最后一维，须不小于sparseBlockCount中所有元素的最大值，当前上限为256 |
| blockShapeX / blockShapeY | 稀疏块在Q方向、KV方向的块大小 |

### layout对应关系说明

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 160px">
  <col style="width: 180px">
  <col style="width: 815px">
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
      <td>PA_BBND</td>
      <td>用于paged kv cache输入，数据按[numBlocks, blockSize, numKeyValueHeads, headDim]排布。</td>
    </tr>
    <tr>
      <td>PA_BNBD</td>
      <td>用于paged kv cache输入，数据按[numBlocks, numKeyValueHeads, blockSize, headDim]排布。</td>
    </tr>
    <tr>
      <td rowspan="3">BSND</td>
      <td>BSND</td>
      <td>用于原始KV输入。</td>
    </tr>
    <tr>
      <td>PA_BBND</td>
      <td>用于paged kv cache输入，数据按[numBlocks, blockSize, numKeyValueHeads, headDim]排布。</td>
    </tr>
    <tr>
      <td>PA_BNBD</td>
      <td>用于paged kv cache输入，数据按[numBlocks, numKeyValueHeads, blockSize, headDim]排布。</td>
    </tr>
    <tr>
      <td rowspan="3">BNSD</td>
      <td>BNSD</td>
      <td>用于原始KV输入。</td>
    </tr>
    <tr>
      <td>PA_BBND</td>
      <td>用于paged kv cache输入，数据按[numBlocks, blockSize, numKeyValueHeads, headDim]排布。</td>
    </tr>
    <tr>
      <td>PA_BNBD</td>
      <td>用于paged kv cache输入，数据按[numBlocks, numKeyValueHeads, blockSize, headDim]排布。</td>
    </tr>
  </tbody>
  </table>

当前代码支持组合：layoutQ="TND" + layoutKv="PA_BBND"；A5新增layoutQ="TND" + layoutKv="TND"非PA（FLOAT16/BFLOAT16、quantType=0）。BSND/BNSD等组合尚未开放。新增非PA路径尚未完成构建和设备验证。

query、attentionOut的shape由layoutQ决定：

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 160px">
  <col style="width: 815px">
  <col style="width: 180px">
  </colgroup>
  <thead>
    <tr>
      <th>layoutQ</th>
      <th>shape</th>
      <th>当前支持</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>TND</td>
      <td>[totalQTokens, headNum, headDim]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>BNSD</td>
      <td>[batch, headNum, maxQSeqLength, headDim]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>BSND</td>
      <td>[batch, maxQSeqLength, headNum, headDim]</td>
      <td>×</td>
    </tr>
  </tbody>
  </table>

sparseBlockIdx、sparseBlockCount的shape由layoutSparsePattern决定，当前layoutSparsePattern值仅支持传4。

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 140px">
  <col style="width: 380px">
  <col style="width: 260px">
  <col style="width: 375px">
  </colgroup>
  <thead>
    <tr>
      <th>layoutSparsePattern</th>
      <th>sparseBlockIdx</th>
      <th>sparseBlockCount</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>[batch, numKeyValueHeads, maxQBlockCount, maxKvBlockCount]</td>
      <td>[batch, numKeyValueHeads, maxQBlockCount]</td>
      <td><ul><li>同一个Group中的qHead共享sparsePattern</li><li>表示每个Q块选择了哪些KV块</li></ul></td>
    </tr>
    <tr>
      <td>1</td>
      <td>[batch, numKeyValueHeads, maxKvBlockCount, maxQBlockCount]</td>
      <td>[batch, numKeyValueHeads, maxKvBlockCount]</td>
      <td><ul><li>同一个Group中的qHead共享sparsePattern</li><li>表示每个KV块选择了哪些Q块</li></ul></td>
    </tr>
    <tr>
      <td>2</td>
      <td>[batch, headNum, maxQBlockCount, maxKvBlockCount]</td>
      <td>[batch, headNum, maxQBlockCount]</td>
      <td><ul><li>同一个Group中的qHead有独立的sparsePattern</li><li>表示每个Q块选择了哪些KV块</li></ul></td>
    </tr>
    <tr>
      <td>3</td>
      <td>[batch, headNum, maxKvBlockCount, maxQBlockCount]</td>
      <td>[batch, headNum, maxKvBlockCount]</td>
      <td><ul><li>同一个Group中的qHead有独立的sparsePattern</li><li>表示每个KV块选择了哪些Q块</li></ul></td>
    </tr>
    <tr>
      <td>4</td>
      <td>[numKeyValueHeads, totalQBlocks, maxKvBlockCount]</td>
      <td>[numKeyValueHeads, totalQBlocks]</td>
      <td><ul><li>同一个Group中的qHead共享sparsePattern</li><li>表示每个Q块选择了哪些KV块</li></ul></td>
    </tr>
    <tr>
      <td>5</td>
      <td>[numKeyValueHeads, totalKBlocks, maxQBlockCount]</td>
      <td>[numKeyValueHeads, totalKBlocks]</td>
      <td><ul><li>同一个Group中的qHead共享sparsePattern</li><li>表示每个KV块选择了哪些Q块</li></ul></td>
    </tr>
    <tr>
      <td>6</td>
      <td>[headNum, totalQBlocks, maxKvBlockCount]</td>
      <td>[headNum, totalQBlocks]</td>
      <td><ul><li>同一个Group中的qHead有独立的sparsePattern</li><li>表示每个Q块选择了哪些KV块</li></ul></td>
    </tr>
    <tr>
      <td>7</td>
      <td>[headNum, totalKBlocks, maxQBlockCount]</td>
      <td>[headNum, totalKBlocks]</td>
      <td><ul><li>同一个Group中的qHead有独立的sparsePattern</li><li>表示每个KV块选择了哪些Q块</li></ul></td>
    </tr>
  </tbody>
  </table>

key、value的shape由layoutKv及是否使能Paged Cache决定，见<a href="#paged-attention相关说明">Paged Attention相关说明</a>。

### Paged Attention相关说明

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 420px">
  <col style="width: 160px">
  <col style="width: 575px">
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
      <td>PA_BBND</td>
      <td>[numBlocks, blockSize, numKeyValueHeads, headDim]</td>
    </tr>
    <tr>
      <td>PA_BNBD</td>
      <td>[numBlocks, numKeyValueHeads, blockSize, headDim]</td>
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

PA_BBND必须传入非空blockTable。A5 TND非PA的K/V为[totalKTokens,numKeyValueHeads,headDim]，blockTable必须为nullptr；cuSeqLengthsKvOptional必传，sequsedKvOptional可选。PA_BNBD尚不支持。

### 量化相关说明

下列量化配置按Ascend 950（A5）代际描述；当前可用quantType=0，Ascend 950上可选quantType=5。

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 120px">
  <col style="width: 240px">
  <col style="width: 130px">
  <col style="width: 240px">
  <col style="width: 260px">
  <col style="width: 280px">
  <col style="width: 280px">
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
          <li>PA_BBND: [batch, ceilDiv(blockSize, blockShapeY), kvHeadNum, 1]。</li>
          <li>PA_BNBD: [batch, kvHeadNum, ceilDiv(blockSize, blockShapeY), 1]。</li>
        </ul>
        vDequantScaleOptional(必选)：
        <ul>
          <li>TND: [batch*ceilDiv(qSeqLength, blockShapeY), kvHeadNum, 1]。</li>
          <li>BNSD: [batch, kvHeadNum, ceilDiv(maxKvSeqLength, blockShapeY), 1]。</li>
          <li>BSND: [batch, ceilDiv(maxKvSeqLength, blockShapeY), kvHeadNum, 1]。</li>
          <li>PA_BBND: [batch, ceilDiv(blockSize, blockShapeY), kvHeadNum, 1]。</li>
          <li>PA_BNBD: [batch, kvHeadNum, ceilDiv(blockSize, blockShapeY), 1]。</li>
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
          <li>PA_BBND: [batch, blockSize, kvHeadNum, ceilDiv(headDim, 64), 2]。</li>
          <li>PA_BNBD: [batch, kvHeadNum, blockSize, ceilDiv(headDim, 64), 2]。</li>
        </ul>
        vDequantScaleOptional(必选)：
        <ul>
          <li>TND: [batch*ceilDiv(kvSeqLength, 64), kvHeadNum, headDim, 2]。</li>
          <li>BNSD: [batch, kvHeadNum, ceilDiv(maxKvSeqLength, 64), headDim, 2]。</li>
          <li>BSND: [batch, ceilDiv(maxKvSeqLength, 64), kvHeadNum, headDim, 2]。</li>
          <li>PA_BBND: [batch, ceilDiv(blockSize, 64), kvHeadNum, headDim, 2]。</li>
          <li>PA_BNBD: [batch, kvHeadNum, ceilDiv(blockSize, 64), headDim, 2]。</li>
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

当前可用quantType=0；Ascend 950上可选quantType=5。quantType=1~4当前不支持。

### 掩码说明

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 160px">
  <col style="width: 140px">
  <col style="width: 655px">
  <col style="width: 200px">
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
      <td>不传 attenMaskOptional（内置 causal）；规划 BOOL [2048,2048] 下三角</td>
      <td>-1/-1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>window mask</td>
      <td>规划：BOOL [2048,2048] 下三角，与 winLeft/winRight 配合</td>
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

当前仅支持maskType=1（算子内置causal，attenMaskOptional须为nullptr，winLeft/winRight为-1）。maskType为0/2/3~5当前不支持。

### 其他约束

- A5 TND非PA：Q/KV分别必传INT64 [B+1]存储前缀，首项为0、单调不减、末项等于各自T；seqused为可选INT32 [B]，值在[0,对应存储段长度]内。Q/KV前缀数组必须描述相同batch数。
- TND非PA的ACLNN入口会将Q/K/V归一化为连续张量；直接进入Tiling的非连续Dense描述不支持。稀疏KV块号是batch内逻辑块号，不是物理页号。
- blockShapeX=1、isPackedGQA=1、maskType=1。Arch35非量化FP16/BF16允许blockShapeY为1～256任意整数，paged PA_BBND与non-paged TND均支持softmaxPrecision=0/1；Y=128保留原有FD调度，其余Y使用普通分核。要求sparseBlockCount指定的候选前缀为升序、非负有效索引，-1仅可放在前缀之后；topK与Y由调用方配套设置。有效KvLen<Qlen时沿用空输出语义；空结果O=0，LSE使用-FLT_MAX。

- 确定性计算：aclnnGenericBlockSparseAttention默认确定性实现。
- 调用前须先执行aclnnGenericBlockSparseAttentionMetadata生成metadataOptional，再调用本接口；metadata须与当前输入/属性配套，每次调用须重新生成。
- query/key/value的headDim(D) tiling范围为1～512；KV页blockSize固定为128，与blockShapeY相互独立。
- TND + isPackedGQA=1时：totalQBlocks按cuSeqLengthsQ差分得到的存储长度分块，即$\sum_i \mathrm{ceilDiv}(qStorageLen_i, blockShapeX)$；sparse分块与QKV寻址均按该存储长度，不以seqused重切分。
- layoutKv为PA_BBND时须传sequsedKvOptional，不传cuSeqLengthsKvOptional。
- seqused与对应cu前缀和同时传入时（Q侧，及layoutKv为TND时的KV侧）：分核/任务空间按各batch实际有效长度（seqused）累加；各batch的seqused元素须≤对应cu存储长度，且须与Metadata侧完全一致。
- 输入query、key、value的数据类型必须一致。
- PA_BBND下key/value仅dim0（物理页轴）可非连续；页内blockSize×numKeyValueHeads×headDim须连续；stride0 ≥ blockSize×numKeyValueHeads×headDim且按numKeyValueHeads×headDim对齐。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```c++
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_generic_block_sparse_attention.h"
#include "aclnnop/aclnn_generic_block_sparse_attention_metadata.h"

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

uint16_t FloatToFp16(float f)
{
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  uint32_t sign = (bits >> 31) & 0x1u;
  int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
  uint32_t mant = (bits >> 13) & 0x3ffu;
  if (exp <= 0) {
    return static_cast<uint16_t>(sign << 15);
  }
  if (exp >= 31) {
    return static_cast<uint16_t>((sign << 15) | 0x7c00u);
  }
  return static_cast<uint16_t>((sign << 15) | (static_cast<uint32_t>(exp) << 10) | mant);
}

float Fp16ToFloat(uint16_t h)
{
  uint32_t sign = (h >> 15) & 0x1u;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x3ffu;
  uint32_t f;
  if (exp == 0) {
    f = (sign << 31) | (mant << 13);
  } else if (exp == 31) {
    f = (sign << 31) | 0x7f800000u | (mant << 13);
  } else {
    f = (sign << 31) | ((exp + 127u - 15u) << 23) | (mant << 13);
  }
  float result;
  std::memcpy(&result, &f, sizeof(result));
  return result;
}

void PrintOutResult(const std::vector<int64_t>& shape, void** deviceAddr)
{
  auto size = GetShapeSize(shape);
  std::vector<uint16_t> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                         size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size && i < 10; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, Fp16ToFloat(resultData[i]));
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

std::vector<uint16_t> MakeFp16Data(int64_t size, float value)
{
  std::vector<uint16_t> data(static_cast<size_t>(size), FloatToFp16(value));
  return data;
}

}  // namespace

int main()
{
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtContext context = nullptr;
  aclrtStream stream = nullptr;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // Regular path smoke case: TND query + PA_BBND KV
  int64_t B = 1;
  int64_t S1 = 4;
  int64_t S2 = 256;
  int64_t N1 = 4;
  int64_t N2 = 1;
  int64_t D = 128;
  int64_t topK = 2;
  int64_t blockSize = 128;
  int64_t blockShapeX = 1;
  double scaleValue = 1.0 / sqrt(static_cast<double>(D));

  int64_t T = B * S1;
  int64_t maxBlocks = (S2 + blockSize - 1) / blockSize;
  int64_t totalQBlocks = T;  // blockShapeX == 1

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> qShape = {T, N1, D};
  std::vector<int64_t> kvShape = {maxBlocks, blockSize, N2, D};
  std::vector<int64_t> sparseIdxShape = {N2, totalQBlocks, topK};
  std::vector<int64_t> sparseCountShape = {N2, totalQBlocks};
  std::vector<int64_t> blockTableShape = {B, maxBlocks};
  std::vector<int64_t> cuSeqQShape = {B + 1};
  std::vector<int64_t> sequsedKvShape = {B};
  std::vector<int64_t> metadataShape = {1024};
  std::vector<int64_t> attnOutShape = {T, N1, D};

  void* qDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* sparseIdxDeviceAddr = nullptr;
  void* sparseCountDeviceAddr = nullptr;
  void* metadataDeviceAddr = nullptr;
  void* cuSeqQDeviceAddr = nullptr;
  void* sequsedKvDeviceAddr = nullptr;
  void* blockTableDeviceAddr = nullptr;
  void* attnOutDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* k = nullptr;
  aclTensor* v = nullptr;
  aclTensor* sparseIdx = nullptr;
  aclTensor* sparseCount = nullptr;
  aclTensor* metadata = nullptr;
  aclTensor* cuSeqQ = nullptr;
  aclTensor* sequsedKv = nullptr;
  aclTensor* blockTable = nullptr;
  aclTensor* attnOut = nullptr;

  int64_t qSize = GetShapeSize(qShape);
  int64_t kvSize = GetShapeSize(kvShape);
  int64_t sparseIdxSize = GetShapeSize(sparseIdxShape);
  int64_t sparseCountSize = GetShapeSize(sparseCountShape);
  int64_t blockTableSize = GetShapeSize(blockTableShape);
  int64_t attnOutSize = GetShapeSize(attnOutShape);

  std::vector<uint16_t> qHostData = MakeFp16Data(qSize, 1.0f);
  std::vector<uint16_t> kHostData = MakeFp16Data(kvSize, 1.0f);
  std::vector<uint16_t> vHostData = MakeFp16Data(kvSize, 1.0f);
  std::vector<int32_t> sparseIdxHostData(sparseIdxSize, -1);
  std::vector<int32_t> sparseCountHostData(sparseCountSize, 0);
  std::vector<int32_t> blockTableHostData(blockTableSize);
  std::iota(blockTableHostData.begin(), blockTableHostData.end(), 0);
  std::vector<int64_t> cuSeqQHostData = {0, S1};
  std::vector<int32_t> sequsedKvHostData = {static_cast<int32_t>(S2)};
  std::vector<int32_t> metadataHostData(1024, 0);
  std::vector<uint16_t> attnOutHostData = MakeFp16Data(attnOutSize, 0.0f);

  // Causal-like sparse window: keep up to topK trailing KV blocks per Q token.
  int64_t history = S2 - S1;
  for (int64_t qBlock = 0; qBlock < totalQBlocks; qBlock++) {
    int64_t visible = history + qBlock + 1;
    int64_t lastKvBlock = std::min(maxBlocks - 1, (visible - 1) / blockSize);
    int64_t count = std::min(topK, lastKvBlock + 1);
    int64_t start = lastKvBlock - count + 1;
    for (int64_t i = 0; i < count; i++) {
      sparseIdxHostData[qBlock * topK + i] = static_cast<int32_t>(start + i);
    }
    sparseCountHostData[qBlock] = static_cast<int32_t>(count);
  }

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kvShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, kvShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sparseIdxHostData, sparseIdxShape, &sparseIdxDeviceAddr, aclDataType::ACL_INT32, &sparseIdx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sparseCountHostData, sparseCountShape, &sparseCountDeviceAddr, aclDataType::ACL_INT32,
                        &sparseCount);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(metadataHostData, metadataShape, &metadataDeviceAddr, aclDataType::ACL_INT32, &metadata);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cuSeqQHostData, cuSeqQShape, &cuSeqQDeviceAddr, aclDataType::ACL_INT64, &cuSeqQ);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sequsedKvHostData, sequsedKvShape, &sequsedKvDeviceAddr, aclDataType::ACL_INT32, &sequsedKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(blockTableHostData, blockTableShape, &blockTableDeviceAddr, aclDataType::ACL_INT32,
                        &blockTable);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attnOutHostData, attnOutShape, &attnOutDeviceAddr, aclDataType::ACL_FLOAT16, &attnOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  const int64_t blockShapeData[] = {blockShapeX, blockSize};
  aclIntArray* blockShape = aclCreateIntArray(blockShapeData, 2);
  CHECK_RET(blockShape != nullptr, LOG_PRINT("aclCreateIntArray failed\n"); return -1);

  char layoutQ[] = "TND";
  char layoutKv[] = "PA_BBND";

  // 3. 先调用 Metadata，再调用主算子
  uint64_t metadataWorkspaceSize = 0;
  aclOpExecutor* metadataExecutor = nullptr;
  ret = aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize(
      sparseIdx, sparseCount, cuSeqQ, nullptr, nullptr, sequsedKv, S1, S2, N1, N2, D, blockShape, layoutQ, layoutKv, 4, 1,
      0, 1, -1, -1, 0, 0, metadata, &metadataWorkspaceSize, &metadataExecutor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);

  void* metadataWorkspaceAddr = nullptr;
  if (metadataWorkspaceSize > 0) {
    ret = aclrtMalloc(&metadataWorkspaceAddr, metadataWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate metadata workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnGenericBlockSparseAttentionMetadata(metadataWorkspaceAddr, metadataWorkspaceSize, metadataExecutor,
                                                 stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGenericBlockSparseAttentionMetadata failed. ERROR: %d\n", ret);
            return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream after metadata failed. ERROR: %d\n", ret);
            return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnGenericBlockSparseAttentionGetWorkspaceSize(
      q, k, v, sparseIdx, sparseCount, metadata, nullptr, nullptr, nullptr, nullptr, nullptr, cuSeqQ, nullptr, nullptr,
      sequsedKv, blockTable, blockShape, layoutQ, layoutKv, 4, scaleValue, 1, 0, 0.0, 1, -1, -1, 0, 0, 0, attnOut, nullptr,
      &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnGenericBlockSparseAttentionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnGenericBlockSparseAttention(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGenericBlockSparseAttention failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(attnOutShape, &attnOutDeviceAddr);

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyIntArray(blockShape);
  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(v);
  aclDestroyTensor(sparseIdx);
  aclDestroyTensor(sparseCount);
  aclDestroyTensor(metadata);
  aclDestroyTensor(cuSeqQ);
  aclDestroyTensor(sequsedKv);
  aclDestroyTensor(blockTable);
  aclDestroyTensor(attnOut);

  // 7. 释放device资源
  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(sparseIdxDeviceAddr);
  aclrtFree(sparseCountDeviceAddr);
  aclrtFree(metadataDeviceAddr);
  aclrtFree(cuSeqQDeviceAddr);
  aclrtFree(sequsedKvDeviceAddr);
  aclrtFree(blockTableDeviceAddr);
  aclrtFree(attnOutDeviceAddr);
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
