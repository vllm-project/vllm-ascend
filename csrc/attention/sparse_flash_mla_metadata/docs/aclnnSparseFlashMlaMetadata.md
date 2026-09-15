# aclnnSparseFlashMlaMetadata

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

- 算子功能：`aclnnSparseFlashMlaMetadata`是`aclnnSparseFlashMla`算子的前置算子，用于后续Attention计算生成负载均衡的任务划分方案。本算子不执行实际的Attention计算，而是根据输入参数在AI CPU计算出每个AI Core应处理的Attention计算起止范围，从而最大化计算资源的利用率，避免各Core间负载不均衡的问题。

  **该算子不建议单独使用，建议与aclnnSparseFlashMla算子配合使用，形成完整的工作流。**
- 场景简称：SWA（Sliding Window Attention）、CSA（Compressed Sparse Attention）、HCA（Heavily Compressed Attention）。
- 计算公式：

  该算子为AICPU调度算子，不涉及数值计算。核心流程为：解析各Batch的Q/KV序列长度 → 根据mask模式计算每个S1G块的有效S2范围 → 基于开销模型进行负载均衡分核 → 输出分核元数据。

  输出metadata tensor的shape为(1024,)，数据类型为INT32，内部结构如下：

    - FA Metadata区域（AIC_CORE_NUM × 9个INT32），每个AICore的FA阶段任务信息：

    | 索引 | 含义 |
    | :--- | :--- |
    | 0 | core_enable，该核是否启用 |
    | 1 | bn2_start，BN2起始索引 |
    | 2 | m_start，M（S1G）起始索引 |
    | 3 | s2_start，S2起始索引 |
    | 4 | bn2_end，BN2结束索引 |
    | 5 | m_end，M结束索引 |
    | 6 | s2_end，S2结束索引 |
    | 7 | first_fd_data_workspace_idx，第一份FD归约数据的workspace偏移 |
    | 8 | max_s2_block_num，单核上分配到的最多的s2 block数 |

    - FD Metadata区域（AIV_CORE_NUM × 8个INT32），每个AIVCore的FD归约任务信息：

    | 索引 | 含义 |
    | :--- | :--- |
    | 0 | core_enable，该核是否启用 |
    | 1 | bn2_idx，归约任务的BN2索引 |
    | 2 | m_idx，归约任务的M索引 |
    | 3 | workspace_idx，归约数据在workspace中的存放位置 |
    | 4 | workspace_num，S2核间切分份数 |
    | 5 | m_start，M轴起点 |
    | 6 | m_num，M轴行数 |

- 符号说明

  | 符号                | 含义                                                      |
  | ------------------- | --------------------------------------------------------- |
  | B                   | Batch Size                                                |
  | N1/N2               | Query/KV头数                                              |
  | D                   | 每个注意力头的维度                                        |
  | G                   | GQA分组比，G=N1/N2                                        |
  | S1/S2               | Query/KV序列长度                                          |
  | S1G                 | S1×G方向的分块索引                                        |
  | mBaseSize           | M轴基本块大小，等于G                                      |
  | s2BaseSize          | S2轴基本块大小，固定为512                                  |

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用`aclnnSparseFlashMlaMetadataGetWorkspaceSize`接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnSparseFlashMlaMetadata`执行实际计算。

```c++
aclnnStatus aclnnSparseFlashMlaMetadataGetWorkspaceSize(
    const aclTensor *cuSeqlensQOptional,
    const aclTensor *cuSeqlensOriKvOptional,
    const aclTensor *cuSeqlensCmpKvOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedOriKvOptional,
    const aclTensor *sequsedCmpKvOptional,
    const aclTensor *cmpResidualKvOptional,
    const aclTensor *oriTopkLengthOptional,
    const aclTensor *cmpTopkLengthOptional,
    int64_t          numHeadsQ,
    int64_t          numHeadsKv,
    int64_t          headDim,
    int64_t          batchSize,
    int64_t          maxSeqlenQ,
    int64_t          maxSeqlenOriKv,
    int64_t          maxSeqlenCmpKv,
    int64_t          oriTopk,
    int64_t          cmpTopk,
    int64_t          cmpRatio,
    int64_t          oriMaskMode,
    int64_t          cmpMaskMode,
    int64_t          oriWinLeft,
    int64_t          oriWinRight,
    const char      *layoutQOptional,
    const char      *layoutKvOptional,
    bool             hasOriKv,
    bool             hasCmpKv,
    const aclTensor *metaData,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```c++
aclnnStatus aclnnSparseFlashMlaMetadata(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnSparseFlashMlaMetadataGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1500px"><colgroup>
    <col style="width: 220px">
    <col style="width: 100px">
    <col style="width: 300px">
    <col style="width: 300px">
    <col style="width: 120px">
    <col style="width: 100px">
    <col style="width: 160px">
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
      <td>cuSeqlensQOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中q的有效token数（前缀和形式）。</td>
      <td><ul><li>支持空Tensor。</li><li>shape固定为(B+1, )。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqlensOriKvOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中oriKv的有效token数（前缀和形式）。</td>
      <td><ul><li>支持空Tensor。</li><li>shape固定为(B+1, )。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqlensCmpKvOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中cmpKv的有效token数（前缀和形式）。</td>
      <td><ul><li>支持空Tensor。</li><li>shape固定为(B+1, )。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedQOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中q实际参与运算的token数。</td>
      <td><ul><li>支持空Tensor。</li><li>shape固定为(B, )。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedOriKvOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中oriKv实际参与运算的token数。</td>
      <td><ul><li>支持空Tensor。</li><li>shape固定为(B, )。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedCmpKvOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中cmpKv实际参与运算的token数。</td>
      <td><ul><li>支持空Tensor。</li><li>shape固定为(B, )。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpResidualKvOptional（aclTensor*）</td>
      <td>输入</td>
      <td>压缩KV余数，用于按cmp_len * cmpRatio + residual恢复cmp侧mask使用的压缩前KV长度。</td>
      <td>在CSA、HCA、cmpRatio不等于1且cmpMaskMode为3场景必传，layoutKvOptional为BSND、TND、PA_BBND时均可使用。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维</td>
      <td>√</td>
    </tr>
    <tr>
      <td>oriTopkLengthOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同q token对应的oriKvOptional部分关键稀疏token的个数。</td>
      <td><ul><li>SWA稀疏ori_kv场景必须传入，其他场景支持空Tensor。</li><li>shape为(B, S1, N2)或(T1, N2)。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>2维、3维</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpTopkLengthOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同q token对应的cmpKvOptional部分关键稀疏token的个数。</td>
      <td><ul><li>支持空Tensor。</li><li>shape为(B, S1, N2)或(T1, N2)。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>2维、3维</td>
      <td>√</td>
    </tr>
    <tr>
      <td>numHeadsQ（int64_t）</td>
      <td>输入</td>
      <td>Query的多头数。</td>
      <td>支持[1, 128]。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeadsKv（int64_t）</td>
      <td>输入</td>
      <td>KV的多头数。</td>
      <td>仅支持1。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headDim（int64_t）</td>
      <td>输入</td>
      <td>每个注意力头的维度。</td>
      <td>仅支持512。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batchSize（int64_t）</td>
      <td>输入</td>
      <td>输入样本批量大小。</td>
      <td>layoutQOptional为TND时无需手动指定，建议值为0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenQ（int64_t）</td>
      <td>输入</td>
      <td>所有Batch中q的最大有效token数。</td>
      <td>传入0时表示由接口推导，建议值为0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenOriKv（int64_t）</td>
      <td>输入</td>
      <td>所有Batch中oriKv的最大有效token数。</td>
      <td>传入0时表示由接口推导，建议值为0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenCmpKv（int64_t）</td>
      <td>输入</td>
      <td>所有Batch中cmpKv的最大有效token数。</td>
      <td>传入0时表示由接口推导，建议值为0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriTopk（int64_t）</td>
      <td>输入</td>
      <td>从oriKv中筛选的稀疏token个数。</td>
      <td>SWA稀疏ori_kv场景为主算子oriSparseIndicesOptional最后一维K，且必须大于0；其他场景建议值为0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpTopk（int64_t）</td>
      <td>输入</td>
      <td>从cmpKv中筛选的稀疏token个数。</td>
      <td>CSA场景下仅支持512或1024，SWA、HCA场景下为0，建议值为0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpRatio（int64_t）</td>
      <td>输入</td>
      <td>cmpKv相对于压缩前KV长度的压缩倍率，用于恢复cmp侧mask使用的压缩前KV长度。</td>
      <td>传入cmpKv时支持[1, 128]；仅传入oriKv时传0；CSA场景传1、2或4，HCA场景传128，建议值为0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriMaskMode（int64_t）</td>
      <td>输入</td>
      <td>q和oriKv计算的mask模式。</td>
      <td>0: No Mask。<br/>3: RightDownCausal模式。<br/>4: Band模式。<br/>建议值为0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpMaskMode（int64_t）</td>
      <td>输入</td>
      <td>q和cmpKv计算的mask模式。</td>
      <td>0: No Mask。<br/>3: RightDownCausal模式。<br/>建议值为0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriWinLeft（int64_t）</td>
      <td>输入</td>
      <td>滑动窗口向左扩展的token数。</td>
      <td>支持-1或非负数，其中-1表示窗口不受限，建议值为-1。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriWinRight（int64_t）</td>
      <td>输入</td>
      <td>滑动窗口向右扩展的token数。</td>
      <td>支持-1或非负数，其中-1表示窗口不受限，建议值为-1。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQOptional（char*）</td>
      <td>输入</td>
      <td>标识输入q的数据排布格式。</td>
      <td>支持"BSND"和"TND"，建议值为"BSND"。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKvOptional（char*）</td>
      <td>输入</td>
      <td>标识输入KV的数据排布格式。</td>
      <td>支持"PA_BBND"、"BSND"和"TND"，建议值为"BSND"。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>hasOriKv（bool）</td>
      <td>输入</td>
      <td>是否传入oriKv。</td>
      <td>根据是否传入oriKv设置，建议值为true。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>hasCmpKv（bool）</td>
      <td>输入</td>
      <td>是否传入cmpKv。</td>
      <td>SWA场景为false，CSA、HCA场景为true。根据是否传入cmpKv设置，建议值为true。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metaData（aclTensor*）</td>
      <td>输出</td>
      <td>分核元数据输出，供SparseFlashMla算子使用。</td>
      <td>shape固定为(1024, )。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维</td>
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

  <ul>
    <!-- npu="A3" id7 -->
    <li><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> ：不支持sequsedQOptional、cmpTopkLengthOptional，numHeadsQ/numHeadsKv仅支持1、2、4、8、16、32、64、128；SWA稀疏ori_kv场景支持oriTopkLengthOptional、oriTopk大于0及oriMaskMode为0，oriWinLeft和oriWinRight支持非负数；其他SWA场景oriTopk为0、oriMaskMode为4、oriWinLeft为127、oriWinRight为0；cmpTopk仅支持0、512、1024，cmpMaskMode仅支持3，cmpRatio在SWA支持0、CSA支持1、2或4、HCA支持128。</li>
    <!-- end id7 -->
    <!-- npu="910b" id8 -->
    <li><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> ：不支持sequsedQOptional、cmpTopkLengthOptional，numHeadsQ/numHeadsKv仅支持1、2、4、8、16、32、64、128；SWA稀疏ori_kv场景支持oriTopkLengthOptional、oriTopk大于0及oriMaskMode为0，oriWinLeft和oriWinRight支持非负数；其他SWA场景oriTopk为0、oriMaskMode为4、oriWinLeft为127、oriWinRight为0；cmpTopk仅支持0、512、1024，cmpMaskMode仅支持3，cmpRatio在SWA支持0、CSA支持1、2或4、HCA支持128。</li>
    <!-- end id8 -->
  </ul>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

    <!-- npu="950" id9 -->
    - <term>Ascend 950PR/Ascend 950DT</term>：

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
        <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
        <td>561101</td>
        <td>创建aclOpExecutor失败。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_NULLPTR</td>
        <td>561103</td>
        <td>workspaceSize或executor为空指针；可选输入做连续化处理后为空指针；或添加SparseFlashMlaMetadata AICPU任务失败。</td>
      </tr>
      <tr>
        <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="8">161002</td>
        <td>batchSize或maxSeqlenQ为负数。</td>
      </tr>
      <tr>
        <td>numHeadsQ不在[1,128]范围内，numHeadsKv不为1，numHeadsQ不能被numHeadsKv整除，或numHeadsQ/numHeadsKv不在[1,128]范围内。</td>
      </tr>
      <tr>
        <td>headDim不为512。</td>
      </tr>
      <tr>
        <td>oriMaskMode不为0、3、4，或cmpMaskMode不为0、3。</td>
      </tr>
      <tr>
        <td>oriWinLeft或oriWinRight小于-1。</td>
      </tr>
      <tr>
        <td>hasCmpKv为true时，cmpRatio不在[1,128]范围内。</td>
      </tr>
      <tr>
        <td>hasCmpKv为true时，cmpTopk为负数。</td>
      </tr>
      <tr>
        <td>layoutQOptional、layoutKvOptional、cuSeqlens、seqused或metaData的shape、数据类型、必选关系不在支持范围内。</td>
      </tr>
    </tbody>
    </table>
  <!-- end id9 -->
  <!-- npu="A3" id10 -->
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：

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
        <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
        <td>561101</td>
        <td>创建aclOpExecutor失败。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_NULLPTR</td>
        <td>561103</td>
        <td>workspaceSize或executor为空指针；可选输入做连续化处理后为空指针；或添加SparseFlashMlaMetadata AICPU任务失败。</td>
      </tr>
      <tr>
        <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="9">161002</td>
        <td>batchSize或maxSeqlenQ为负数。</td>
      </tr>
      <tr>
        <td>numHeadsQ不在[1,128]范围内，numHeadsKv不为1，numHeadsQ不能被numHeadsKv整除，或numHeadsQ/numHeadsKv不是[1,128]范围内的2的幂。</td>
      </tr>
      <tr>
        <td>headDim不为512。</td>
      </tr>
      <tr>
        <td>非SWA稀疏ori_kv场景oriMaskMode不为4，SWA稀疏ori_kv场景oriMaskMode不为0，或cmpMaskMode不为3。</td>
      </tr>
      <tr>
        <td>非SWA稀疏ori_kv场景oriWinLeft不为127，或oriWinRight不为0；SWA稀疏ori_kv场景oriWinLeft或oriWinRight为负数。</td>
      </tr>
      <tr>
        <td>SWA场景cmpRatio不为0，或cmpRatio与CSA、HCA场景不匹配。</td>
      </tr>
      <tr>
        <td>cmpTopk不为0、512或1024。</td>
      </tr>
      <tr>
        <td>SWA稀疏ori_kv场景未传入oriTopkLengthOptional，或oriTopkLengthOptional的shape、数据类型不符合规格；cmpTopkLengthOptional传入非空Tensor。</td>
      </tr>
      <tr>
        <td>layoutQOptional、layoutKvOptional、cuSeqlens、seqused或metaData的shape、数据类型、必选关系不在支持范围内。</td>
      </tr>
    </tbody>
    </table>
  <!-- end id10 -->
  <!-- npu="910b" id11 -->
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：

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
        <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
        <td>561101</td>
        <td>创建aclOpExecutor失败。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_NULLPTR</td>
        <td>561103</td>
        <td>workspaceSize或executor为空指针；可选输入做连续化处理后为空指针；或添加SparseFlashMlaMetadata AICPU任务失败。</td>
      </tr>
      <tr>
        <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="9">161002</td>
        <td>batchSize或maxSeqlenQ为负数。</td>
      </tr>
      <tr>
        <td>numHeadsQ不在[1,128]范围内，numHeadsKv不为1，numHeadsQ不能被numHeadsKv整除，或numHeadsQ/numHeadsKv不是[1,128]范围内的2的幂。</td>
      </tr>
      <tr>
        <td>headDim不为512。</td>
      </tr>
      <tr>
        <td>非SWA稀疏ori_kv场景oriMaskMode不为4，SWA稀疏ori_kv场景oriMaskMode不为0，或cmpMaskMode不为3。</td>
      </tr>
      <tr>
        <td>非SWA稀疏ori_kv场景oriWinLeft不为127，或oriWinRight不为0；SWA稀疏ori_kv场景oriWinLeft或oriWinRight为负数。</td>
      </tr>
      <tr>
        <td>SWA场景cmpRatio不为0，或cmpRatio与CSA、HCA场景不匹配。</td>
      </tr>
      <tr>
        <td>cmpTopk不为0、512或1024。</td>
      </tr>
      <tr>
        <td>SWA稀疏ori_kv场景未传入oriTopkLengthOptional，或oriTopkLengthOptional的shape、数据类型不符合规格；cmpTopkLengthOptional传入非空Tensor。</td>
      </tr>
      <tr>
        <td>layoutQOptional、layoutKvOptional、cuSeqlens、seqused或metaData的shape、数据类型、必选关系不在支持范围内。</td>
      </tr>
    </tbody>
    </table>
  <!-- end id11 -->

## aclnnSparseFlashMlaMetadata

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnSparseFlashMlaMetadataGetWorkspaceSize获取。</td>
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

    - aclnnSparseFlashMlaMetadata默认采用确定性实现，相同输入多次调用结果一致。

- 通用规格约束
    - B（Batch）表示输入样本批量大小，q、oriKvOptional、cmpKvOptional为配套的aclnnSparseFlashMla算子的入参，S1表示layoutQOptional=BSND时，q shape中的S轴的大小，T1表示layoutQOptional=TND时，q shape中的T轴的大小，S2表示layoutKvOptional=BSND时，oriKvOptional shape中的S轴的大小，S3表示layoutKvOptional=BSND时，cmpKvOptional shape中的S轴的大小，N2表示oriKvOptional、cmpKvOptional shape中的N轴的大小。
    - 参数cuSeqlensQOptional、cuSeqlensOriKvOptional、cuSeqlensCmpKvOptional要求其值为当前Batch与前序Batch有效token数的累加值，第一个元素固定为0，后一个元素的值必须大于等于前一个元素的值。
    - 参数sequsedQOptional、sequsedOriKvOptional、sequsedCmpKvOptional要求其值表示每个Batch中的有效token数。
    - layoutQOptional和layoutKvOptional组合仅支持"BSND"/"BSND"、"TND"/"TND"、"BSND"/"PA_BBND"、"TND"/"PA_BBND"；非PA_BBND场景下layoutQOptional和layoutKvOptional必须一致。
    - 参数cmpResidualKvOptional需满足cmpResidualKvOptional[i] < cmpRatio。
<!-- npu="950" id12 -->
- Ascend 950PR/Ascend 950DT约束：
    - hasOriKv为true时，oriTopk大于0认为oriKvOptional部分是稀疏的，oriTopk为0则认为oriKvOptional部分是非稀疏的。
    - hasCmpKv为true时，cmpTopk大于0认为cmpKvOptional部分是稀疏的，cmpTopk为0则认为cmpKvOptional部分是非稀疏的。
    - hasOriKv为true，oriTopk不为0且oriMaskMode为0时，oriTopkLengthOptional必须传入，此时取oriMaskMode规则与oriTopkLengthOptional元素的最小值作为当前q token对应的oriKvOptional的有效seqlen，其他oriKvOptional稀疏场景取oriMaskMode规则与oriTopk的最小值作为当前q token对应的oriKvOptional的有效seqlen。
    - hasCmpKv为true，cmpTopk不为0且cmpMaskMode为0时，cmpTopkLengthOptional必须传入，此时取cmpMaskMode规则与cmpTopkLengthOptional元素的最小值作为当前q token对应的cmpKvOptional的有效seqlen，其他cmpKvOptional稀疏场景取cmpMaskMode规则与cmpTopk的最小值作为当前q token对应的cmpKvOptional的有效seqlen。
    - layoutQOptional=BSND场景
        - maxSeqlenQ必须传入S1的值。
    - layoutKvOptional=BSND场景
        - hasOriKv为true时，maxSeqlenOriKv必须传入S2的值。
        - hasCmpKv为true时，maxSeqlenCmpKv必须传入S3的值。
    - layoutQOptional=TND场景
        - cuSeqlensQOptional必须传入。
    - layoutKvOptional=TND场景
        - hasOriKv为true时，cuSeqlensOriKvOptional必须传入。
        - hasCmpKv为true时，cuSeqlensCmpKvOptional必须传入。
    - layoutKvOptional=PA_BBND场景
        - hasOriKv为true，oriTopk不为0且oriMaskMode为0时（oriTopkLengthOptional必传场景），sequsedOriKvOptional可选传入，其他场景sequsedOriKvOptional必须传入。
        - hasCmpKv为true，cmpTopk不为0且cmpMaskMode为0时（cmpTopkLengthOptional必传场景），sequsedCmpKvOptional可选传入，其他场景sequsedCmpKvOptional必须传入。
    - Batch取值规则
        - layoutQOptional为BSND时，优先通过sequsedQOptional的shape推导batch，sequsedQOptional未传入则通过batch_size获取batch数。
        - layoutQOptional为TND时，优先通过sequsedQOptional的shape推导batch，sequsedQOptional未传入则通过cuSeqlensQOptional的shape推导batch。
    - q Seqlen取值规则
        - layoutQOptional为BSND时，优先通过sequsedQOptional中的元素获取seqlen，sequsedQOptional未传入则通过maxSeqlenQ获取seqlen。
        - layoutQOptional为TND时，优先通过sequsedQOptional中的元素获取seqlen，sequsedQOptional未传入则通过cuSeqlensQOptional中的元素获取seqlen。
    - oriKvOptional Seqlen取值规则
        - layoutKvOptional为BSND时，优先通过sequsedOriKvOptional中的元素获取seqlen，sequsedOriKvOptional未传入则通过maxSeqlenOriKv获取seqlen。
        - layoutKvOptional为TND时，优先通过sequsedOriKvOptional中的元素获取seqlen，sequsedOriKvOptional未传入则通过cuSeqlensOriKvOptional中的元素获取seqlen。
        - layoutKvOptional为PA_BBND时，优先通过sequsedOriKvOptional中的元素获取seqlen，sequsedOriKvOptional未传入则通过oriTopkLengthOptional获取seqlen。
    - cmpKvOptional Seqlen取值规则
        - layoutKvOptional为BSND时，优先通过sequsedCmpKvOptional中的元素获取seqlen，sequsedCmpKvOptional未传入则通过maxSeqlenCmpKv获取seqlen。
        - layoutKvOptional为TND时，优先通过sequsedCmpKvOptional中的元素获取seqlen，sequsedCmpKvOptional未传入则通过cuSeqlensCmpKvOptional中的元素获取seqlen。
        - layoutKvOptional为PA_BBND时，优先通过sequsedCmpKvOptional中的元素获取seqlen，sequsedCmpKvOptional未传入则通过cmpTopkLengthOptional获取seqlen。
<!-- end id12 -->
<!-- npu="A3" id13 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品约束：
    - SWA稀疏ori_kv场景下，仅支持SWA模板，`hasOriKv`为true、`hasCmpKv`为false、`oriTopk`大于0、`oriMaskMode`为0，`oriWinLeft`和`oriWinRight`为非负数，且必须传入`oriTopkLengthOptional`。`oriTopk`应与配套主算子oriSparseIndicesOptional最后一维K保持一致；`oriTopkLengthOptional`表示每个q token和KV head的左对齐有效索引条目数，取值应在[0, K]范围内；Metadata仅使用`oriTopkLengthOptional`生成任务切分。配套主算子在PA_BBND场景仍要求传入`sequsedOriKvOptional`。
    - layoutQOptional为TND时，`cuSeqlensQOptional`必须传入。
    - layoutKvOptional为PA_BBND时，`sequsedOriKvOptional`必须传入。BSND场景可选传入`sequsedOriKvOptional`覆盖每个batch的oriKv有效长度；TND场景使用`cuSeqlensOriKvOptional`表达oriKv序列边界。
    - layoutKvOptional为TND时，`cuSeqlensOriKvOptional`必须传入；若hasCmpKv为true，`cuSeqlensCmpKvOptional`也必须传入。
    - `sequsedCmpKvOptional`为所有layoutKvOptional下的可选输入，显式传入时用于覆盖cmp侧逻辑有效长度。
    - `cmpResidualKvOptional`为`aclnnSparseFlashMlaMetadata`和`aclnnSparseFlashMla`的可选输入，在CSA、HCA、cmpRatio不等于1且cmpMaskMode为3场景必传，用于恢复cmp侧mask使用的压缩前长度。
<!-- end id13 -->
<!-- npu="910b" id14 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品约束：
    - SWA稀疏ori_kv场景下，仅支持SWA模板，`hasOriKv`为true、`hasCmpKv`为false、`oriTopk`大于0、`oriMaskMode`为0，`oriWinLeft`和`oriWinRight`为非负数，且必须传入`oriTopkLengthOptional`。`oriTopk`应与配套主算子oriSparseIndicesOptional最后一维K保持一致；`oriTopkLengthOptional`表示每个q token和KV head的左对齐有效索引条目数，取值应在[0, K]范围内；Metadata仅使用`oriTopkLengthOptional`生成任务切分。配套主算子在PA_BBND场景仍要求传入`sequsedOriKvOptional`。
    - layoutQOptional为TND时，`cuSeqlensQOptional`必须传入。
    - layoutKvOptional为PA_BBND时，`sequsedOriKvOptional`必须传入。BSND场景可选传入`sequsedOriKvOptional`覆盖每个batch的oriKv有效长度；TND场景使用`cuSeqlensOriKvOptional`表达oriKv序列边界。
    - layoutKvOptional为TND时，`cuSeqlensOriKvOptional`必须传入；若hasCmpKv为true，`cuSeqlensCmpKvOptional`也必须传入。
    - `sequsedCmpKvOptional`为所有layoutKvOptional下的可选输入，显式传入时用于覆盖cmp侧逻辑有效长度。
    - `cmpResidualKvOptional`为`aclnnSparseFlashMlaMetadata`和`aclnnSparseFlashMla`的可选输入，在CSA、HCA、cmpRatio不等于1且cmpMaskMode为3场景必传，用于恢复cmp侧mask使用的压缩前长度。
<!-- end id14 -->

## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```c++
/**
 * @file test_aclnn_sparse_flash_mla_metadata.cpp
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <limits>
#include <functional>
#include <utility>
#include "acl/acl.h"
#include "aclnnop/aclnn_sparse_flash_mla_metadata.h"

#define CHECK_LOG_RET(cond, ret_val, fmt, ...)      \
    do {                                            \
        if (!(cond)) {                              \
            printf(fmt "\n", ##__VA_ARGS__);        \
            return (ret_val);                       \
        }                                           \
    } while (0)

// 参考 sparse_flash_mla_metadata.h
constexpr uint32_t AIC_CORE_MAX_NUM = 36;
constexpr uint32_t AIV_CORE_MAX_NUM = 72;
constexpr uint32_t SMLA_METADATA_TOTAL_SIZE = 1024;
constexpr uint32_t FA_METADATA_SIZE = 9;
constexpr uint32_t FD_METADATA_SIZE = 8;

// FA Metadata Index Definitions
constexpr uint32_t FA_CORE_ENABLE_INDEX = 0;
constexpr uint32_t FA_BN2_START_INDEX = 1;
constexpr uint32_t FA_M_START_INDEX = 2;
constexpr uint32_t FA_S2_START_INDEX = 3;
constexpr uint32_t FA_BN2_END_INDEX = 4;
constexpr uint32_t FA_M_END_INDEX = 5;
constexpr uint32_t FA_S2_END_INDEX = 6;
constexpr uint32_t FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 7;
constexpr uint32_t FA_S2_MAX_NUM = 8;

// FD Metadata Index Definitions
constexpr uint32_t FD_CORE_ENABLE_INDEX = 0;
constexpr uint32_t FD_BN2_IDX_INDEX = 1;
constexpr uint32_t FD_M_IDX_INDEX = 2;
constexpr uint32_t FD_WORKSPACE_IDX_INDEX = 3;
constexpr uint32_t FD_WORKSPACE_NUM_INDEX = 4;
constexpr uint32_t FD_M_START_INDEX = 5;
constexpr uint32_t FD_M_NUM_INDEX = 6;

struct SmlaMetadata {
    uint32_t faMetadata[AIC_CORE_MAX_NUM][FA_METADATA_SIZE];
    uint32_t fdMetadata[AIV_CORE_MAX_NUM][FD_METADATA_SIZE];
};

struct ScopeGuard
{
    explicit ScopeGuard(std::function<void()> onExitScope) : m_exitFunc(std::move(onExitScope)),
        m_isDismissed(false) {}
    // 禁止拷贝
    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;

    ~ScopeGuard()
    {
        if (!m_isDismissed) {
            m_exitFunc();
        }
    }

    void Dismiss()
    {
        m_isDismissed = true;
    }

    std::function<void()> m_exitFunc;
    bool m_isDismissed;
};

struct Tensor {
    void *hostAddr { nullptr };
    void *deviceAddr { nullptr };
    aclTensor *data { nullptr };
};

struct ArgScenario {
    bool hasCuSeq { false };
    bool hasSeqused { false };
};

struct ArgContext {
    // required input
    int64_t numHeadsQ { 0 };
    int64_t numHeadsKv { 0 };
    int64_t headDim { 0 };
    // optional input
    Tensor cuSeqlensQOptional {};
    Tensor cuSeqlensOriKvOptional {};
    Tensor cuSeqlensCmpKvOptional {};
    Tensor sequsedQOptional {};
    Tensor sequsedOriKvOptional {};
    Tensor sequsedCmpKvOptional {};
    Tensor cmpResidualKvOptional {};
    Tensor oriTopkLengthOptional {};
    Tensor cmpTopkLengthOptional {};
    int64_t batchSize { 0 };
    int64_t maxSeqlenQ { 0 };
    int64_t maxSeqlenOriKv { 0 };
    int64_t maxSeqlenCmpKv { 0 };
    int64_t oriTopk { 0 };
    int64_t cmpTopk { 0 };
    int64_t cmpRatio { 0 };
    int64_t oriMaskMode { 0 };
    int64_t cmpMaskMode { 0 };
    int64_t oriWinLeft { -1 };
    int64_t oriWinRight { -1 };
    char *layoutQOptional { nullptr };
    char *layoutKvOptional { nullptr };
    bool hasOriKv { true };
    bool hasCmpKv { true };
    // output
    Tensor metadata {};
};

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

aclnnStatus Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，初始化
    auto ret = aclInit(nullptr);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclInit failed. ERROR: %d", ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtSetDevice failed. ERROR: %d", ret);
    ret = aclrtCreateStream(stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtCreateStream failed. ERROR: %d", ret);
    return ACL_SUCCESS;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

aclnnStatus CreateTensor(aclDataType dataType, const std::vector<int64_t> &shape, Tensor &tensor)
{
    auto size = GetShapeSize(shape) * aclDataTypeSize(dataType);
    // 调用aclrtMallocHost申请host侧内存
    auto ret = aclrtMallocHost(&(tensor.hostAddr), size);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMallocHost failed. ERROR: %d", ret);
    memset(tensor.hostAddr, 0, size);
    // 调用aclrtMalloc申请device侧内存
    ret = aclrtMalloc(&(tensor.deviceAddr), size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMalloc failed. ERROR: %d", ret);
    // 调用aclCreateTensor接口创建aclTensor
    tensor.data = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND,
        shape.data(), shape.size(), tensor.deviceAddr);
    CHECK_LOG_RET(tensor.data != nullptr, ACL_ERROR_FAILURE, "aclCreateTensor failed");
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(tensor.deviceAddr, size, tensor.hostAddr, size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy failed. ERROR: %d", ret);
    return ACL_SUCCESS;
}

void DestroyTensor(Tensor &tensor)
{
    if (tensor.data != nullptr) {
        aclDestroyTensor(tensor.data);
        tensor.data = nullptr;
    }
    if (tensor.deviceAddr != nullptr) {
        aclrtFree(tensor.deviceAddr);
        tensor.deviceAddr = nullptr;
    }
    if (tensor.hostAddr != nullptr) {
        aclrtFreeHost(tensor.hostAddr);
        tensor.hostAddr = nullptr;
    }
}

void DestroyArgs(ArgContext &context)
{
    DestroyTensor(context.metadata);
    DestroyTensor(context.cuSeqlensQOptional);
    DestroyTensor(context.cuSeqlensOriKvOptional);
    DestroyTensor(context.cuSeqlensCmpKvOptional);
    DestroyTensor(context.sequsedQOptional);
    DestroyTensor(context.sequsedOriKvOptional);
    DestroyTensor(context.sequsedCmpKvOptional);
    DestroyTensor(context.cmpResidualKvOptional);
    DestroyTensor(context.oriTopkLengthOptional);
    DestroyTensor(context.cmpTopkLengthOptional);

    if (context.layoutQOptional != nullptr) {
        free(context.layoutQOptional);
        context.layoutQOptional = nullptr;
    }
    if (context.layoutKvOptional != nullptr) {
        free(context.layoutKvOptional);
        context.layoutKvOptional = nullptr;
    }
}

aclnnStatus CreateArgs(const ArgScenario &scenario, ArgContext &context)
{
    ScopeGuard argsGuard([&] { DestroyArgs(context); });
    aclnnStatus ret;

    context.numHeadsQ = 64;
    context.numHeadsKv = 1;
    context.headDim = 512;
    ret = CreateTensor(aclDataType::ACL_INT32, { SMLA_METADATA_TOTAL_SIZE }, context.metadata);     // 1024: Fix size
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create metadata failed. Error: %d", ret);
    context.oriTopk = 0;
    context.cmpTopk = 0;
    context.cmpRatio = 128;
    context.oriMaskMode = 4;
    context.cmpMaskMode = 3;
    context.oriWinLeft = 127;
    context.oriWinRight = 0;
    context.layoutQOptional = (char *)malloc(sizeof(char) * 16);
    context.layoutKvOptional = (char *)malloc(sizeof(char) * 16);
    CHECK_LOG_RET(context.layoutQOptional != nullptr, ACL_ERROR_FAILURE, "Create layoutQOptional failed");
    CHECK_LOG_RET(context.layoutKvOptional != nullptr, ACL_ERROR_FAILURE, "Create layoutKvOptional failed");
    strcpy(context.layoutQOptional, "BSND");                // BSND,TND
    strcpy(context.layoutKvOptional, "BSND");               // BSND,TND,PA_BBND
    context.hasOriKv = true;
    context.hasCmpKv = true;

    context.batchSize = 4;
    context.maxSeqlenOriKv = 1024;
    context.maxSeqlenCmpKv = 1024;
    context.maxSeqlenQ = 1024;

    if (scenario.hasCuSeq) {
        // (B+1,), first element is always 0
        ret = CreateTensor(aclDataType::ACL_INT32, { context.batchSize + 1 }, context.cuSeqlensQOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqlensQOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, { context.batchSize + 1 }, context.cuSeqlensOriKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqlensOriKvOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, { context.batchSize + 1 }, context.cuSeqlensCmpKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqlensCmpKvOptional failed. Error: %d", ret);
    }

    if (scenario.hasSeqused) {
        // (B,)
        ret = CreateTensor(aclDataType::ACL_INT32, { context.batchSize }, context.sequsedQOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create sequsedQOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, { context.batchSize }, context.sequsedOriKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create sequsedOriKvOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, { context.batchSize }, context.sequsedCmpKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create sequsedCmpKvOptional failed. Error: %d", ret);
    }

    if (context.hasCmpKv && context.cmpRatio != 1 && context.cmpMaskMode == 3) {
        // (B,)
        ret = CreateTensor(aclDataType::ACL_INT32, { context.batchSize }, context.cmpResidualKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cmpResidualKvOptional failed. Error: %d", ret);
    }

    argsGuard.Dismiss();
    return ACL_SUCCESS;
}

int main() {
    // 1. （固定写法）device/stream初始化，参考对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Init acl failed. ERROR: %d", ret);
    ScopeGuard sysGuard([&] { Finalize(deviceId, stream); });

    // 2. 构造输入与输出，需要根据API的接口定义构造
    ArgScenario scenario {};
    scenario.hasCuSeq = false;
    scenario.hasSeqused = false;
    ArgContext context {};
    ret = CreateArgs(scenario, context);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create input arguments failed. ERROR: %d", ret);
    ScopeGuard argsGuard([&] { DestroyArgs(context); });

    // 3. 调用CANN算子库API，需要修改为具体的API
    // 调用aclnnSparseFlashMlaMetadata第一段接口
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;
    ret = aclnnSparseFlashMlaMetadataGetWorkspaceSize(
        context.cuSeqlensQOptional.data, context.cuSeqlensOriKvOptional.data, context.cuSeqlensCmpKvOptional.data,
        context.sequsedQOptional.data, context.sequsedOriKvOptional.data, context.sequsedCmpKvOptional.data,
        context.cmpResidualKvOptional.data, context.oriTopkLengthOptional.data, context.cmpTopkLengthOptional.data,
        context.numHeadsQ, context.numHeadsKv, context.headDim, context.batchSize, context.maxSeqlenQ,
        context.maxSeqlenOriKv, context.maxSeqlenCmpKv, context.oriTopk, context.cmpTopk, context.cmpRatio,
        context.oriMaskMode, context.cmpMaskMode, context.oriWinLeft, context.oriWinRight, context.layoutQOptional,
        context.layoutKvOptional, context.hasOriKv, context.hasCmpKv, context.metadata.data, &workspaceSize, &executor);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret,
        "aclnnSparseFlashMlaMetadataGetWorkspaceSize failed. ERROR: %d\n", ret);

    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "allocate workspace failed. ERROR: %d\n", ret);
    }
    ScopeGuard workspaceGuard([&] {
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
            workspaceAddr = nullptr;
        }
    });

    // 调用aclnnSparseFlashMlaMetadata第二段接口
    ret = aclnnSparseFlashMlaMetadata(workspaceAddr, workspaceSize, executor, stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclnnSparseFlashMlaMetadata failed. ERROR: %d\n", ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtSynchronizeStream failed. ERROR: %d\n", ret);

    // 5. 打印输出
    SmlaMetadata result {};
    ret = aclrtMemcpy(&result, sizeof(result), context.metadata.deviceAddr, sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy failed. ERROR: %d\n", ret);

    for (uint32_t i = 0; i < AIC_CORE_MAX_NUM; ++i) {
        printf("AIC Core%u\n", i);
        printf("    Core Enable : %u\n", result.faMetadata[i][FA_CORE_ENABLE_INDEX]);
        printf("    Start BN2   : %u\n", result.faMetadata[i][FA_BN2_START_INDEX]);
        printf("    Start M     : %u\n", result.faMetadata[i][FA_M_START_INDEX]);
        printf("    Start S2    : %u\n", result.faMetadata[i][FA_S2_START_INDEX]);
        printf("    End BN2     : %u\n", result.faMetadata[i][FA_BN2_END_INDEX]);
        printf("    End M       : %u\n", result.faMetadata[i][FA_M_END_INDEX]);
        printf("    End S2      : %u\n", result.faMetadata[i][FA_S2_END_INDEX]);
        printf("    First Workspace Index : %u\n", result.faMetadata[i][FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX]);
        printf("    Max S2 Block Num : %u\n", result.faMetadata[i][FA_S2_MAX_NUM]);
    }
    for (uint32_t i = 0; i < AIV_CORE_MAX_NUM; ++i) {
        printf("AIV Core%u\n", i);
        printf("    Core Enable             : %u\n", result.fdMetadata[i][FD_CORE_ENABLE_INDEX]);
        printf("    FD Task BN2 Idx         : %u\n", result.fdMetadata[i][FD_BN2_IDX_INDEX]);
        printf("    FD Task M Idx           : %u\n", result.fdMetadata[i][FD_M_IDX_INDEX]);
        printf("    FD Task Workspace Idx   : %u\n", result.fdMetadata[i][FD_WORKSPACE_IDX_INDEX]);
        printf("    FD Task Workspace Num   : %u\n", result.fdMetadata[i][FD_WORKSPACE_NUM_INDEX]);
        printf("    FD Subtask M Start      : %u\n", result.fdMetadata[i][FD_M_START_INDEX]);
        printf("    FD Subtask M Num        : %u\n", result.fdMetadata[i][FD_M_NUM_INDEX]);
    }

    return 0;
}
```
