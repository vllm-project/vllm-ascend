# aclnnSparseFlashAttention

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/sparse_flash_attention)

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>|     √      |
|<term>Atlas A2 推理系列产品</term>   | √  |
|<term>Atlas A3 推理系列产品</term>   | √  |

## 功能说明

- 接口功能：sparse_flash_attention（SFA）是针对大序列长度推理场景的高效注意力计算模块，该模块通过“只计算关键部分”大幅减少计算量，然而会引入大量的离散访存，造成数据搬运时间增加，进而影响整体性能。

- 计算公式：

$$
\text{softmax}(\frac{Q@\tilde{K}^T}{\sqrt{d_k}})@\tilde{V}
$$

其中$\tilde{K},\tilde{V}$为基于某种选择算法（如`lightning_indexer`）得到的重要性较高的Key和Value，一般具有稀疏或分块稀疏的特征，$d_k$为$Q,\tilde{K}$每一个头的维度。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnSparseFlashAttentionGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSparseFlashAttention”接口执行计算。

```Cpp
aclnnStatus aclnnSparseFlashAttentionGetWorkspaceSize(
    const aclTensor     *query,
    const aclTensor     *key,
    const aclTensor     *value, 
    const aclTensor     *sparseIndices,
    const aclTensor     *blockTable,
    const aclTensor     *actualSeqLengthsQuery,
    const aclTensor     *actualSeqLengthsKv,
    const aclTensor     *queryRope,
    const aclTensor     *keyRope,
    double              scaleValue,
    int64_t             sparseBlockSize,
    char                *layoutQuery,
    char                *layoutKv,
    int64_t             sparseMode,
    int64_t             preTokens,
    int64_t             nextTokens,
    int64_t             attentionMode,
    bool                returnSoftmaxLse,
    const aclTensor     *attentionOutOut,
    const aclTensor     *softmaxMaxOut,
    const aclTensor     *softmaxSumOut,
    uint64_t            *workspaceSize,
    aclOpExecutor       **executor)
```

```Cpp
aclnnStatus aclnnSparseFlashAttention(
    void             *workspace, 
    uint64_t          workspaceSize, 
    aclOpExecutor    *executor, 
    const aclrtStream stream)
```

## aclnnSparseFlashAttentionGetWorkspaceSize

- **参数说明：**

  > [!NOTE]  
  >
  >- query、key、value参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、N（Head Num）表示多头数、D（Head Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
  >- Q\_S和S1表示query shape中的S，KV\_S和S2表示key shape中的S，Q\_N和N1表示num\_query\_heads，KV\_N和N2表示num\_key\_value\_heads，T1表示query shape中的T，T2表示key shape中的输入样本序列长度的累加和。

  <table style="undefined;table-layout: fixed; width: 1494px"><colgroup>
  <col style="width: 146px">
  <col style="width: 110px">
  <col style="width: 301px">
  <col style="width: 500px">
  <col style="width: 328px">
  <col style="width: 101px">
  <col style="width: 400px">
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
    </tr></thead>
  <tbody>
    <tr>
      <td>query（aclTensor）</td>
      <td>输入</td>
      <td>attention结构的Query输入。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT16、BFLOAT16</td>
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
      <td>key（aclTensor）</td>
      <td>输入</td>
      <td>attention结构的Key输入</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>block_num为PageAttention时block总数。</li>
          </ul>
      </td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_kv为PA_BSND时，shape为(block_num, block_size, KV_N, D)。</li>
                <li>layout_kv为BSND时，shape为(B, S2, KV_N, D)。</li>
                <li>layout_kv为TND时，shape为(T2, KV_N, D)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>value（aclTensor）</td>
      <td>输入</td>
      <td>attention结构的Value输入。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>shape与key的shape一致。</td>
      <td>x</td>
    </tr>
    <tr>
      <td>sparseIndices（aclTensor）</td>
      <td>输入</td>
      <td>离散取kvCache的索引。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>sparse_size为一次离散选取的block数，需要保证每行有效值均在前半部分，无效值均在后半部分，且需要满足sparse_size大于0。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_query为BSND时，shape为(B, Q_S, KV_N, sparse_size)。</li>
                <li>layout_query为TND时，shape为(Q_T, KV_N, sparse_size)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>blockTable（aclTensor）</td>
      <td>输入</td>
      <td>表示PageAttention中kvCache存储使用的block映射表。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>第二维长度不小于所有batch中最大的S2对应的block数量，即S2_max / block_size向上取整。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>shape支持(B,S2/block_size)。</td>
      <td>x</td>
    </tr>
    <tr>
      <td>actualSeqLengthsQuery（aclTensor）</td>
      <td>输入</td>
      <td>表示不同Batch中query的有效token数。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>如果不指定seqlen可传入None，表示和query的shape的S长度相同。</li>
                <li>该入参中每个Batch的有效token数不超过query中的维度S大小且不小于0。支持长度为B的一维tensor。</li>
                <li>layout_query为TND时，该入参必须传入，且以该入参元素的数量作为B值，该参数中每个元素的值表示当前batch与之前所有batch的token数总和。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>actualSeqLengthsKv（aclTensor）</td>
      <td>输入</td>
      <td>表示不同Batch中key和value的有效token数。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>如果不指定seqlen可传入None，表示和key的shape的S长度相同。</li>
                <li>该参数中每个Batch的有效token数不超过key/value中的维度S大小且不小于0。支持长度为B的一维tensor。</li>
                <li>当layout_kv为TND或PA_BSND时，该入参必须传入。</li>
                <li>layout_kv为TND，该参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>queryRope（aclTensor）</td>
      <td>输入</td>
      <td>表示MLA结构中的query的rope信息。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_query为TND时，shape为(B,S1,N1,Dr)。</li>
                <li>layout_query为BSND时，shape为(T1,N1,Dr)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>keyRope（aclTensor）</td>
      <td>输入</td>
      <td>表示MLA结构中的key的rope信息。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_kv为TND时，shape为(B,S1,N1,Dr)。</li>
                <li>layout_kv为BSND时，shape为(T1,N1,Dr)。</li>
                <li>layout_kv为PA_BSND时，shape为(block_num,block_size,N2,Dr)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>scaleValue（double）</td>
      <td>输入</td>
      <td>代表缩放系数。</td>
      <td>-</td>
      <td>FLOAT16</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseBlockSize（int64_t）</td>
      <td>输入</td>
      <td>代表sparse阶段的block大小。</td>
      <td>
          <ul>
                <li>sparse_block_size为1时，为Token-wise稀疏化场景，将每个token视为独立单元，在计算重要性分数时，评估每个查询token与每个键值token之间的独立关联程度。</li>
                <li>sparse_block_size为大于1小于等于128时，为Block-wise稀疏化场景，将token序列划分为固定大小的连续块，以块为单位进行重要性评估，块内token共享相同的稀疏化决策。</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQuery（char）</td>
      <td>输入</td>
      <td>标识输入query的数据排布格式。</td>
      <td>
          <ul>
                <li>用户不特意指定时可传入默认值"BSND"。</li>
                <li>支持传入BSND和TND。</li>
          </ul>
      </td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKv（char）</td>
      <td>输入</td>
      <td>标识输入key的数据排布格式。</td>
      <td>
          <ul>
                <li>用户不特意指定时可传入默认值"BSND"。</li>
                <li>支持传入TND、BSND和PA_BSND，其中PA_BSND在使能PageAttention时使用。</li>
          </ul>
      </td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseMode（int64_t）</td>
      <td>输入</td>
      <td>表示sparse的模式。</td>
      <td>
          <ul>
                <li>sparse_mode为0时，代表全部计算。</li>
                <li>sparse_mode为3时，代表rightDownCausal模式的mask，对应以右下顶点往左上为划分线的下三角场景。</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>preTokens（int64_t）</td>
      <td>输入</td>
      <td>用于稀疏计算，表示attention需要和前几个Token计算关联。</td>
      <td>仅支持默认值2^63-1。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nextTokens（int64_t）</td>
      <td>输入</td>
      <td>用于稀疏计算，表示attention需要和后几个Token计算关联。</td>
      <td>仅支持默认值2^63-1。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attentionMode（int64_t）</td>
      <td>输入</td>
      <td>-</td>
      <td>仅支持传入2，表示MLA-absorb模式。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>returnSoftmaxLse（bool）</td>
      <td>输入</td>
      <td>用于表示是否返回softmax_max和softmax_sum。</td>
      <td>
          <ul>
                <li>True表示返回，False表示不返回；默认值为False。</li>
                <li>该参数仅在训练且layout_kv不为PA_BSND场景支持。</li>
          </ul>
      </td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attentionOut（aclTensor）</td>
      <td>输出</td>
      <td>公式中的输出。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_query为BSND时，shape为(B,S1,N1,D)。</li>
                <li>layout_query为TND时shape为(T1,N1,D)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>softmaxMaxOut（aclTensor）</td>
      <td>输出</td>
      <td>Attention算法对query乘key的结果，取max得到softmax_max。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_query为BSND时，shape为(B,N2,S1,N1/N2)。</li>
                <li>layout_query为TND时shape为(N2,T1,N1/N2)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
   <tr>
      <td>softmaxSumOut（aclTensor）</td>
      <td>输出</td>
      <td>Attention算法query乘key的结果减去softmax_max, 再取exp，接着求sum，得到softmax_sum。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_query为BSND时，shape为(B,N2,S1,N1/N2)。</li>
                <li>layout_query为TND时shape为(N2,T1,N1/N2)。</li>
          </ul>
      </td>
      <td>x</td>
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
      <td>executor（aclOpExecutor）</td>
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

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

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
                <td>query、key、value、sparseIndices、blockTable、actualSeqLengthsQuery、actualSeqLengthsKv、queryRope、keyRope、scaleValue、sparseBlockSize、layoutQuery、layoutKv、sparseMode、attentionMode、returnSoftmaxLse、attentionOut、softmaxMaxOut、softmaxSumOut的数据类型和数据格式不在支持的范围内。</td>
            </tr>
        </tbody>
    </table>

## aclnnSparseFlashAttention

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnSparseFlashAttentionGetWorkspaceSize获取。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：aclnnSparseFlashAttention默认确定性实现。
- 该接口支持推理场景下使用。
- N1支持1~64和128。
- block_size为一个block的token数，block_size取值为16的倍数，且最大支持1024。
- 参数query中的D和key、value的D值相等为512，参数query_rope中的Dr和key_rope的Dr值相等为64。
- 参数query、key、value的数据类型必须保持一致。
- 支持sparse_block_size整除block_size。
    - <term>Ascend 950PR/Ascend 950DT</term>：
        - 只支持sparse_block_size为1。
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
        - 支持[1,128]，且要求是2的幂次方，在PageAttention场景下要求sparse_block_size整除block_size

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_incre_flash_attention_v4.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "securec.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_sparse_flash_attention.h"

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
    void* valueDeviceAddr = nullptr;
    void* sparseIndicesDeviceAddr = nullptr;
    void* attentionOutDeviceAddr = nullptr;
    void* softmaxMaxDeviceAddr = nullptr;
    void* softmaxSumDeviceAddr = nullptr;
    void* queryRopeDeviceAddr = nullptr;
    void* keyRopeDeviceAddr = nullptr;

    aclTensor* queryTensor = nullptr;
    aclTensor* keyTensor = nullptr;
    aclTensor* valueTensor = nullptr;
    aclTensor* sparseIndicesTensor = nullptr;
    aclTensor* attentionOutTensor = nullptr;
    aclTensor* softmaxMaxTensor = nullptr;
    aclTensor* softmaxSumTensor = nullptr;
    aclTensor* queryRopeTensor = nullptr;
    aclTensor* keyRopeTensor = nullptr; 
};

int InitializeTensors(TensorResources& resources) {
    std::vector<int64_t> queryShape = {1, 2, 1, 512};
    std::vector<int64_t> keyShape = {1, 2, 1, 512};
    std::vector<int64_t> valueShape = {1, 2, 1, 512};
    std::vector<int64_t> sparseIndicesShape = {1, 2, 1, 2};
    std::vector<int64_t> attentionOutShape = {1, 2, 1, 512};
    std::vector<int64_t> softmaxMaxShape = {1, 2, 1, 16};
    std::vector<int64_t> softmaxSumShape = {1, 2, 1, 16};
    std::vector<int64_t> queryRopeShape = {1, 2, 1, 64};
    std::vector<int64_t> keyRopeShape = {1, 2, 1, 64};

    int64_t queryShapeSize = GetShapeSize(queryShape);
    int64_t keyShapeSize = GetShapeSize(keyShape);
    int64_t valueShapeSize = GetShapeSize(valueShape);
    int64_t sparseIndicesShapeSize =  GetShapeSize(sparseIndicesShape);
    int64_t attentionOutShapeSize = GetShapeSize(attentionOutShape);
    int64_t softmaxMaxShapeSize = GetShapeSize(softmaxMaxShape);
    int64_t softmaxSumShapeSize = GetShapeSize(softmaxSumShape);
    int64_t queryRopeShapeSize = GetShapeSize(queryRopeShape);
    int64_t keyRopeShapeSize = GetShapeSize(keyRopeShape);

    std::vector<float> queryHostData(queryShapeSize, 1);
    std::vector<float> keyHostData(keyShapeSize, 1);
    std::vector<float> valueHostData(valueShapeSize, 1);
    std::vector<int32_t> sparseIndicesHostData(sparseIndicesShapeSize, 1);
    std::vector<float> attentionOutHostData(attentionOutShapeSize, 1);
    std::vector<float> softmaxMaxHostData(softmaxMaxShapeSize, 1);
    std::vector<float> softmaxSumHostData(softmaxSumShapeSize, 1);
    std::vector<float> queryRopeHostData(queryRopeShapeSize, 1);
    std::vector<float> keyRopeHostData(keyRopeShapeSize, 1);

    // Create query aclTensor.
    int ret = CreateAclTensor(queryHostData, queryShape, &resources.queryDeviceAddr, 
                             aclDataType::ACL_FLOAT16, &resources.queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create key aclTensor.
    ret = CreateAclTensor(keyHostData, keyShape, &resources.keyDeviceAddr, 
                         aclDataType::ACL_FLOAT16, &resources.keyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create value aclTensor.
    ret = CreateAclTensor(valueHostData, valueShape, &resources.valueDeviceAddr, 
                         aclDataType::ACL_FLOAT16, &resources.valueTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create sparseIndices aclTensor.
    ret = CreateAclTensor(sparseIndicesHostData, sparseIndicesShape, &resources.sparseIndicesDeviceAddr, 
                         aclDataType::ACL_INT32, &resources.sparseIndicesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create queryRope aclTensor.
    ret = CreateAclTensor(queryRopeHostData, queryRopeShape, &resources.queryRopeDeviceAddr, 
                         aclDataType::ACL_FLOAT16, &resources.queryRopeTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create keyRope aclTensor.
    ret = CreateAclTensor(keyRopeHostData, keyRopeShape, &resources.keyRopeDeviceAddr, 
                         aclDataType::ACL_FLOAT16, &resources.keyRopeTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create attention_out aclTensor.
    ret = CreateAclTensor(attentionOutHostData, attentionOutShape, &resources.attentionOutDeviceAddr, 
                         aclDataType::ACL_FLOAT16, &resources.attentionOutTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create softmax_max aclTensor.
    ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &resources.softmaxMaxDeviceAddr, 
                         aclDataType::ACL_FLOAT, &resources.softmaxMaxTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create softmax_sum aclTensor.
    ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &resources.softmaxSumDeviceAddr, 
                         aclDataType::ACL_FLOAT, &resources.softmaxSumTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    return ACL_SUCCESS;
}

int ExecuteSparseFlashAttention(TensorResources& resources, aclrtStream stream, 
                              void** workspaceAddr, uint64_t* workspaceSize) {
    int64_t d = 2;
    double scaleValue = 1 / sqrt(d);
    int64_t sparseBlockSize = 64;
    constexpr const char layerOutStr[] = "BSND";
    constexpr size_t layerOutLen = sizeof(layerOutStr);
    char layoutQuery[layerOutLen];
    char layoutKv[layerOutLen];
    errno_t memcpyRet = memcpy_s(layoutQuery, sizeof(layoutQuery), layerOutStr, layerOutLen);
    if (memcpyRet != 0) {
        LOG_PRINT("memcpy_s layoutQuery failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    memcpyRet = memcpy_s(layoutKv, sizeof(layoutKv), layerOutStr, layerOutLen);
    if (memcpyRet != 0) {
        LOG_PRINT("memcpy_s layoutKv failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    int64_t sparseMode = 3;
    int64_t preTokens = 9223372036854775807;
    int64_t nextTokens = 9223372036854775807;
    int64_t attentionMode = 2;
    bool returnSoftmaxLse = false;
    aclOpExecutor* executor;

    int ret = aclnnSparseFlashAttentionGetWorkspaceSize(resources.queryTensor, resources.keyTensor, resources.valueTensor, resources.sparseIndicesTensor, nullptr, nullptr, nullptr, resources.queryRopeTensor, resources.keyRopeTensor,
                                                    scaleValue, sparseBlockSize, layoutQuery, layoutKv, sparseMode, preTokens,
                                                    nextTokens, attentionMode, returnSoftmaxLse, resources.attentionOutTensor, resources.softmaxMaxTensor, resources.softmaxSumTensor, workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnSparseFlashAttentionGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnSparseFlashAttention(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnSparseFlashAttention failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

int PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<aclFloat16> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
  }
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, aclFloat16ToFloat(resultData[i]));
  }
  return ACL_SUCCESS;
}

void CleanupResources(TensorResources& resources, void* workspaceAddr, 
                     aclrtStream stream, int32_t deviceId) {
    if (resources.queryTensor) {
      aclDestroyTensor(resources.queryTensor);
    }
    if (resources.keyTensor) {
      aclDestroyTensor(resources.keyTensor);
    }
    if (resources.valueTensor) {
      aclDestroyTensor(resources.valueTensor);
    }
    if (resources.sparseIndicesTensor) {
      aclDestroyTensor(resources.sparseIndicesTensor);
    }
    if (resources.attentionOutTensor) {
      aclDestroyTensor(resources.attentionOutTensor);
    }
    if (resources.softmaxMaxTensor) {
      aclDestroyTensor(resources.softmaxMaxTensor);
    }
    if (resources.softmaxSumTensor) {
      aclDestroyTensor(resources.softmaxSumTensor);
    }
    if (resources.queryRopeTensor) {
      aclDestroyTensor(resources.queryRopeTensor);
    }
    if (resources.keyRopeTensor) {
      aclDestroyTensor(resources.keyRopeTensor);
    }

    if (resources.queryDeviceAddr) {
      aclrtFree(resources.queryDeviceAddr);
    }
    if (resources.keyDeviceAddr) {
      aclrtFree(resources.keyDeviceAddr);
    }
    if (resources.valueDeviceAddr) {
      aclrtFree(resources.valueDeviceAddr);
    }
    if (resources.sparseIndicesDeviceAddr) {
      aclrtFree(resources.sparseIndicesDeviceAddr);
    }
    if (resources.attentionOutDeviceAddr) {
      aclrtFree(resources.attentionOutDeviceAddr);
    }
    if (resources.softmaxMaxDeviceAddr) {
      aclrtFree(resources.softmaxMaxDeviceAddr);
    }
    if (resources.softmaxSumDeviceAddr) {
      aclrtFree(resources.softmaxSumDeviceAddr);
    }
    if (resources.queryRopeDeviceAddr) {
      aclrtFree(resources.queryRopeDeviceAddr);
    }
    
    if (resources.keyRopeDeviceAddr) {
      aclrtFree(resources.keyRopeDeviceAddr);
    }

    if (workspaceAddr) {
      aclrtFree(workspaceAddr);
    }
    if (stream) {
      aclrtDestroyStream(stream);
    }
    
    aclrtResetDevice(deviceId);
    aclFinalize();
}

} // namespace

int main() {

    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    TensorResources resources = {};
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<int64_t> attentionOutShape = {1, 2, 1, 16};
    std::vector<int64_t> softmaxMaxShape = {1, 2, 1, 16};
    std::vector<int64_t> softmaxSumShape = {1, 2, 1, 16}; 
    int ret = ACL_SUCCESS;

    // 1. Initialize device and stream
    ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }


    // 2. Initialize tensors
    ret = InitializeTensors(resources);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 3. Execute the operation
    ret = ExecuteSparseFlashAttention(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 4. Synchronize stream
    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 5. Process results
    printf("-----------attentionOut输出-----------\n");
    PrintOutResult(attentionOutShape, &resources.attentionOutDeviceAddr);
    printf("-----------softmaxMax输出-----------\n");
    PrintOutResult(softmaxMaxShape, &resources.softmaxMaxDeviceAddr);
    printf("-----------softmaxSum输出-----------\n");
    PrintOutResult(softmaxSumShape, &resources.softmaxSumDeviceAddr);
    // 6. Cleanup resources
    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}
```
