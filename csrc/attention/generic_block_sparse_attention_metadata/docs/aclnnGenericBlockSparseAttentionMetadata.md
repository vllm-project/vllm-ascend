# aclnnGenericBlockSparseAttentionMetadata

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

- **接口功能**：生成`aclnnGenericBlockSparseAttention`计算所需的`metadataOptional`。
- 输出`metadataOptional`必须作为主算子`aclnnGenericBlockSparseAttention`的`metadataOptional`输入使用，每次调用主算子前均须重新生成。
- `metadataOptional`为不透明数据，调用者不应解析或修改其中的内容。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用`aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize`接口获取workspace大小以及包含算子计算流程的执行器，再调用`aclnnGenericBlockSparseAttentionMetadata`接口执行计算。

```c++
aclnnStatus aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize(
  const aclTensor   *sparseBlockIdx,
  const aclTensor   *sparseBlockCount,
  const aclTensor   *cuSeqLengthsQOptional,
  const aclTensor   *cuSeqLengthsKvOptional,
  const aclTensor   *sequsedQOptional,
  const aclTensor   *sequsedKvOptional,
  int64_t            maxQSeqLen,
  int64_t            maxKvSeqLen,
  int64_t            numQHeads,
  int64_t            numKvHeads,
  int64_t            headDim,
  const aclIntArray *blockShape,
  const char         *layoutQ,
  const char         *layoutKv,
  int64_t            layoutSparsePattern,
  int64_t            maskType,
  int64_t            quantType,
  int64_t            softmaxPrecision,
  int64_t            winLeft,
  int64_t            winRight,
  int64_t            residualBlockMode,
  bool               isConsistentTopK,
  const aclTensor   *metadataOptional,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnGenericBlockSparseAttentionMetadata(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize

- **参数说明**

  表格中shape变量含义见<a href="#基准说明">基准说明</a>。

  <table style="table-layout: fixed; width: 100%; word-break: break-word; overflow-wrap: anywhere;">
  <colgroup>
  <col style="width: 13%">
  <col style="width: 7%">
  <col style="width: 17%">
  <col style="width: 28%">
  <col style="width: 10%">
  <col style="width: 6%">
  <col style="width: 13%">
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
      <td>sparseBlockIdx</td>
      <td>输入</td>
      <td>稀疏块索引数组，指定每个Q块选择的KV块索引。</td>
      <td>取值须为合法KV块索引（按cu存储长度分块）；无效位置可用-1填充，有效值须落在前sparseBlockCount个位置。详细参考<a href="../../generic_block_sparse_attention/docs/aclnnGenericBlockSparseAttention.md#layout对应关系说明">layout对应关系说明</a>。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>[numKeyValueHeads, totalQBlocks, maxSparseBlockCount]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sparseBlockCount</td>
      <td>输入</td>
      <td>每个Q块实际选择的KV块数量。</td>
      <td>详细参考<a href="../../generic_block_sparse_attention/docs/aclnnGenericBlockSparseAttention.md#layout对应关系说明">layout对应关系说明</a>。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>[numKeyValueHeads, totalQBlocks]</td>
      <td>×</td>
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
      <td>maxQSeqLen</td>
      <td>输入</td>
      <td>主算子Query的最大Sequence Length。</td>
      <td>必须大于0，并与主算子及长度类输入保持一致，详见<a href="#其他约束">其他约束</a>。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxKvSeqLen</td>
      <td>输入</td>
      <td>主算子Key/Value的最大Sequence Length。</td>
      <td>必须大于0，并与主算子及长度类输入保持一致，详见<a href="#其他约束">其他约束</a>。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numQHeads</td>
      <td>输入</td>
      <td>主算子Query的head数。</td>
      <td>必须大于0，并与主算子保持一致。Q/KV头数关系见<a href="#其他约束">其他约束</a>。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numKvHeads</td>
      <td>输入</td>
      <td>主算子Key/Value的head数。</td>
      <td>必须大于0，并与主算子保持一致。Q/KV头数关系见<a href="#其他约束">其他约束</a>。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headDim</td>
      <td>输入</td>
      <td>主算子Query、Key和Value每个head的特征维度。</td>
      <td>当前仅支持128，并须与主算子保持一致。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockShape</td>
      <td>输入</td>
      <td>代表稀疏块形状数组。</td>
      <td>含两个元素[blockShapeX, blockShapeY]。blockShapeX=1。Arch35且quantType=0时支持1≤blockShapeY≤256的任意整数，其余场景仅支持128；必须与主算子一致。Y≠128时生成普通分核调度，Y=128保留现有FD条件。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQ</td>
      <td>输入</td>
      <td>代表输入query的数据排布格式。</td>
      <td>目标支持"TND""BNSD""BSND"，详见<a href="../../generic_block_sparse_attention/docs/aclnnGenericBlockSparseAttention.md#layout对应关系说明">layout对应关系说明</a>。当前仅支持"TND"。</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKv</td>
      <td>输入</td>
      <td>代表输入key、value的数据排布格式。</td>
      <td>目标支持"TND""BNSD""BSND""PA_BBND""PA_BNBD"，详见<a href="../../generic_block_sparse_attention/docs/aclnnGenericBlockSparseAttention.md#layout对应关系说明">layout对应关系说明</a>。支持"PA_BBND"；A5 quantType=0新增"TND"非PA。</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutSparsePattern</td>
      <td>输入</td>
      <td>代表输入的sparseBlockIdx、sparseBlockCount的数据排布格式。</td>
      <td>当前仅支持取"4"，详细参考<a href="../../generic_block_sparse_attention/docs/aclnnGenericBlockSparseAttention.md#layout对应关系说明">layout对应关系说明</a>。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskType</td>
      <td>输入</td>
      <td>表示attention计算中的掩码类型。</td>
      <td>
        取值0~5，含义见<a href="../../generic_block_sparse_attention/docs/aclnnGenericBlockSparseAttention.md#掩码说明">掩码说明</a>。<br>
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
        取值0~5，完整配置见<a href="../../generic_block_sparse_attention/docs/aclnnGenericBlockSparseAttention.md#量化相关说明">量化相关说明</a>：<br>
        当前可用：0；Ascend 950上可选5。取值1~4传入将校验失败。
      </td>
      <td>INT64</td>
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
      <td>用于滑窗attention；不使能时必须为-1，需与maskType配合，见<a href="../../generic_block_sparse_attention/docs/aclnnGenericBlockSparseAttention.md#掩码说明">掩码说明</a>。当前只支持传入-1。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>winRight</td>
      <td>输入</td>
      <td>滑窗attention场景下，滑窗需要向后包含多少个token。</td>
      <td>用于滑窗attention；不使能时必须为-1，需与maskType配合，见<a href="../../generic_block_sparse_attention/docs/aclnnGenericBlockSparseAttention.md#掩码说明">掩码说明</a>。当前只支持传入-1。</td>
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
      <td>-</td>
    </tr>
    <tr>
      <td>metadataOptional</td>
      <td>输出</td>
      <td>AICPU算子生成的分核结果。</td>
      <td>必须传入。调用者须预先申请INT32、shape为(1024,)的Device侧Tensor；其内容为不透明格式，不应解析、修改或跨不同输入和属性复用。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(1024,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>当前返回0。</td>
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
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>必选Tensor未传入，或Tensor类型、shape、layout、blockShape、头数关系、maskType、quantType、softmaxPrecision、窗口属性等不满足当前约束。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
      <td>561101</td>
      <td>API内部创建aclOpExecutor失败。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_NULLPTR</td>
      <td>561103</td>
      <td>workspaceSize或executor为空，或API内部执行Contiguous、注册AICPU任务时出现空指针异常。</td>
    </tr>
  </tbody>
  </table>

## aclnnGenericBlockSparseAttentionMetadata

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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。当前workspaceSize为0，可传入nullptr。</td>
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

  返回 aclnnStatus 状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

### 基准说明

<table style="undefined;table-layout: fixed; width: 1155px">
<colgroup>
  <col style="width: 260px">
  <col style="width: 895px">
</colgroup>
<thead>
  <tr>
    <th>命名</th>
    <th>含义</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>B</td>
    <td>Batch Size，输入样本批量大小</td>
  </tr>
  <tr>
    <td>numQHeads</td>
    <td>主算子query的Head Num</td>
  </tr>
  <tr>
    <td>numKvHeads</td>
    <td>主算子key/value的Head Num</td>
  </tr>
  <tr>
    <td>headDim</td>
    <td>主算子query、key和value的Head Dim</td>
  </tr>
  <tr>
    <td>qStorageLen_i</td>
    <td>cuSeqLengthsQOptional相邻元素差分得到的第i个Batch的Query存储长度</td>
  </tr>
  <tr>
    <td>totalQBlocks</td>
    <td>按存储长度分块后的Q块总数，即$\sum_i \mathrm{ceilDiv}(qStorageLen_i, blockShapeX)$</td>
  </tr>
  <tr>
    <td>maxSparseBlockCount</td>
    <td>sparseBlockIdx最后一维，须不小于sparseBlockCount中所有元素的最大值</td>
  </tr>
</tbody>
</table>

### 其他约束

- 本接口必须与`aclnnGenericBlockSparseAttention`配套使用。共同Tensor和属性须与随后调用的主算子完全一致，每次调用主算子前均须重新生成`metadataOptional`。主算子的完整约束见[aclnnGenericBlockSparseAttention](../../generic_block_sparse_attention/docs/aclnnGenericBlockSparseAttention.md#约束说明)。
- 当前支持`layoutQ="TND"`、`layoutKv="PA_BBND"`（A5 quantType=0另支持`layoutKv="TND"`非PA）、`isPackedGQA=1`、`headDim=128`、`blockShapeX=1`、`maskType=1`以及`winLeft=winRight=-1`。Arch35 quantType=0允许blockShapeY为1～256任意整数，其余场景要求128；主算子的Arch35非量化统一路径要求升序有效索引前缀。
- `sparseBlockIdx`的shape为`[numKvHeads, totalQBlocks, maxSparseBlockCount]`，`sparseBlockCount`的shape为`[numKvHeads, totalQBlocks]`。`maxSparseBlockCount`须大于0、不超过256，且须不小于`sparseBlockCount`中所有元素的最大值。
- `cuSeqLengthsQOptional`当前必须传入；`sequsedQOptional`可传入nullptr。传入sequsedQ时，每个元素须位于`[0, 对应Batch存储长度]`范围内。分核按实际长度累加，稀疏块分块仍按cu前缀和描述的存储长度计算。
- layoutKv为PA_BBND时须传`sequsedKvOptional`，不传`cuSeqLengthsKvOptional`。layoutKv为TND时，`sequsedKvOptional`可与`cuSeqLengthsKvOptional`并存，双长度语义同Q侧。
- `numQHeads >= numKvHeads`且`numQHeads % numKvHeads == 0`，`groupSize = numQHeads / numKvHeads`不超过128。
- 输出`metadataOptional`固定为INT32、shape为`(1024,)`，不得解析、修改或跨不同输入和属性复用。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```c++
#include "acl/acl.h"
#include "aclnnop/aclnn_generic_block_sparse_attention_metadata.h"

aclnnStatus RunMetadata(const aclTensor *sparseBlockIdx, const aclTensor *sparseBlockCount,
                        const aclTensor *cuSeqLengthsQOptional, const aclTensor *sequsedKvOptional,
                        aclTensor *metadataOptional, aclrtStream stream)
{
    const int64_t blockShapeData[] = {1, 128};
    aclIntArray *blockShape = aclCreateIntArray(blockShapeData, 2);
    if (blockShape == nullptr) {
        return ACL_ERROR_BAD_ALLOC;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize(
        sparseBlockIdx, sparseBlockCount, cuSeqLengthsQOptional, nullptr, nullptr, sequsedKvOptional, 16, 2048,
        32, 8, 128, blockShape, "TND", "PA_BBND", 4, 1, 0, 0, -1, -1, 0, 0, metadataOptional, &workspaceSize,
        &executor);
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
