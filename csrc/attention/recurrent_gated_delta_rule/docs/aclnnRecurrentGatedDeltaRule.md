# aclnnRecurrentGatedDeltaRule

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/recurrent_gated_delta_rule)

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 接口功能：完成变步长的Recurrent Gated Delta Rule计算。

- 计算公式：

  Recurrent Gated Delta Rule（循环门控Delta规则，RGDR）是一种应用于循环神经网络的算子，也被应用于一种线性注意力机制中。
  在每个时间步 $t$，网络根据当前的输入 $q_t$、$k_t$、$v_t$ 和上一个隐藏状态 $S_{t-1}$，计算当前的注意力输出 $o_t$ 和新的隐藏状态 $S_t$。
  在这个过程中，门控单元会决定有多少新信息存入隐藏状态，以及有多少旧信息需要被遗忘。

  $$
  S_t := S_{t-1}(\alpha_t Diag(\alpha_{kt})(I - \beta_t k_t k_t^T)) + \beta_t v_t k_t^T = \alpha_t Diag(\alpha_{kt})S_{t-1} + \beta_t (v_t - \alpha_t Diag(\alpha_{kt})S_{t-1}k_t)k_t^T
  $$

  $$
  o := \frac{S_t q_t}{\sqrt{d_k}}
  $$

  其中，$S_{t-1},S_t \in R^{d_v \times d_k}$，$q_t, k_t \in R^{d_k}$，$v_t \in R^{d_v}$，$\alpha_t \in R$，$\alpha_k \in R^{d_k}$，$\beta_t \in R$，$o \in R^{d_v}$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnRecurrentGatedDeltaRuleGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnRecurrentGatedDeltaRule”接口执行计算。

```cpp
aclnnStatus aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *beta,
    aclTensor       *stateRef,
    const aclTensor *actualSeqLengths,
    const aclTensor *ssmStateIndices,
    const aclTensor *g,
    const aclTensor *gk,
    const aclTensor *numAcceptedTokens,
    float            scaleValue,
    aclTensor       *out,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnRecurrentGatedDeltaRule(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnRecurrentGatedDeltaRuleGetWorkspaceSize

- **参数说明**

  <table style="undefined; table-layout: fixed; width: 1450px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 350px">
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 165px">
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>query</td>
      <td>输入</td>
      <td>公式中的q。</td>
      <td>不支持空Tensor。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>(T, Nk, Dk)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>key</td>
      <td>输入</td>
      <td>公式中的k。</td>
      <td>不支持空Tensor。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>(T, Nk, Dk)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>value</td>
      <td>输入</td>
      <td>公式中的v。</td>
      <td>不支持空Tensor。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>(T, Nv, Dv)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>输入</td>
      <td>公式中的β。</td>
      <td>不支持空Tensor。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>(T, Nv)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>stateRef</td>
      <td>输入&输出</td>
      <td>状态矩阵，公式中的S。</td>
      <td>不支持空Tensor。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>(BlockNum, Nv, Dv, Dk)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>actualSeqLengths</td>
      <td>输入</td>
      <td>不同batch的有效序列长度。</td>
      <td>不支持空Tensor。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>ssmStateIndices</td>
      <td>输入</td>
      <td>输入序列到状态矩阵的映射索引。</td>
      <td><ul><li>不支持空Tensor。</li><li>state[ssmStateIndices[i]]表示第i个token的状态矩阵。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>(T,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>g</td>
      <td>输入</td>
      <td>衰减系数，公式中的α=e^g。</td>
      <td><ul><li>不支持空Tensor。</li><li>如果传入nullptr，则表示全0的tensor。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(T, Nv)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gk</td>
      <td>输入</td>
      <td>衰减系数，公式中的αk=e^gk</td>
      <td><ul><li>不支持空Tensor。</li><li>如果传入nullptr，则表示全0的tensor。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(T, Nv, Dk)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>numAcceptedTokens</td>
      <td>输入</td>
      <td>每个序列接受的token数量。</td>
      <td>不支持空Tensor。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>输入</td>
      <td>query的缩放因子，对应公式中的 1/sqrt(d_k)。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的o。</td>
      <td>-</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>(T, Nv, Dv)</td>
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
  </tbody>
  </table>
  
  其中 $B$ 表示batch size，令 $L_i$ 表示第i个序列的长度，则 $T=\sum_i^B L_i$ 表示累积序列长度。$N_k$ 表示key的头数，$N_v$ 表示value的头数，$D_k$ 表示key向量的维度，$D_v$ 表示value向量的维度。

- **返回值**

  aclnnStatus： 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed; width: 1050px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 670px">
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
      <td>query, key, value, beta, stateRef, actualSeqLengths, ssmStateIndices, numAcceptedTokens, out存在空指针。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>输入Tensor的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>输入Tensor的数据格式不在支持范围内。</td>
    </tr>
    <tr>
      <td>输入Tensor的shape不在支持范围内。</td>
    </tr>
  </tbody>
  </table>

## aclnnRecurrentGatedDeltaRule

- **参数说明**
  <table style="undefined;table-layout: fixed; width: 1050px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 670px">
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
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnRecurrentGatedDeltaRuleGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值**

  aclnnStatus： 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnRecurrentGatedDeltaRule默认确定性实现。
- 输入shape大小需满足约束：$L_i \le 8$，$N_k \le 256$，$N_v \le 256$，$D_k \le 256$，$D_v \le 256$。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_recurrent_gated_delta_rule.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<aclFloat16> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        if (i >= 5) { // print the first five data
            break;
        }
        LOG_PRINT("mean result[%ld] is: %f\n", i, aclFloat16ToFloat(resultData[i]));
    }
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    // AscendCL初始化
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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

int main()
{
    // 1.device/context/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    void *queryDeviceAddr = nullptr;
    void *keyDeviceAddr = nullptr;
    void *valueDeviceAddr = nullptr;
    void *gamaDeviceAddr = nullptr;
    void *betaDeviceAddr = nullptr;
    void *stateRefDeviceAddr = nullptr;
    void *actSeqLenDeviceAddr = nullptr;
    void *ssmStaIdDeviceAddr = nullptr;
    void *numAccTokDeviceAddr = nullptr;
    void *attnOutDeviceAddr = nullptr;

    aclTensor *query = nullptr;
    aclTensor *key = nullptr;
    aclTensor *value = nullptr;
    aclTensor *gama = nullptr;
    aclTensor *gamak = nullptr;
    aclTensor *beta = nullptr;
    aclTensor *stateRef = nullptr;
    aclTensor *actSeqLen = nullptr;
    aclTensor *ssmStaId = nullptr;
    aclTensor *numAccTok = nullptr;
    aclTensor *attnOut = nullptr;

    // 自定义输入与属性
    int32_t batchSize = 2;
    int32_t mtp = 2;
    int32_t headKNum = 4;
    int32_t headVNum = 8;
    int32_t dimV = 32;
    int32_t dimK = 32;


    std::vector<int64_t> stateShape = {batchSize * mtp, headVNum, dimV, dimK};
    std::vector<int64_t> qkShape = {batchSize * mtp, headKNum, dimK};
    std::vector<int64_t> vShape = {batchSize * mtp, headVNum, dimV};
    std::vector<int64_t> gamaShape = {batchSize * mtp, headVNum};
    std::vector<int64_t> actSeqLenShape = {batchSize};
    std::vector<int64_t> ssmStaIdShape = {batchSize * mtp};
    std::vector<int16_t> stateRefHostData(GetShapeSize(stateShape));
    std::vector<int16_t> queryHostData(GetShapeSize(qkShape));
    std::vector<int16_t> keyHostData(GetShapeSize(qkShape));
    std::vector<int16_t> valueHostData(GetShapeSize(vShape));
    std::vector<float> gamaHostData(GetShapeSize(gamaShape));
    std::vector<int16_t> betaHostData(GetShapeSize(gamaShape));
    std::vector<int32_t> actSeqLenHostData(batchSize, mtp);
    std::vector<int32_t> ssmStaIdHostData(batchSize * mtp);
    std::vector<int32_t> numAccTokHostData(batchSize, 1);
    for (int i = 0; i < stateRefHostData.size(); i++) {
        stateRefHostData[i] = 1;
    }
    for (int i = 0; i < queryHostData.size(); i++) {
        queryHostData[i] = 1;
    }
    for (int i = 0; i < keyHostData.size(); i++) {
        keyHostData[i] = 1;
    }
    for (int i = 0; i < valueHostData.size(); i++) {
        valueHostData[i] = 0;
    }
    for (int i = 0; i < betaHostData.size(); i++) {
        betaHostData[i] = 0;
    }
    for (int i = 0; i < ssmStaIdHostData.size(); i++) {
        ssmStaIdHostData[i] = i;
    }

    std::vector<int16_t> attnOutHostData(valueHostData);

    ret = CreateAclTensor(stateRefHostData, stateShape, &stateRefDeviceAddr, aclDataType::ACL_BF16, &stateRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(queryHostData, qkShape, &queryDeviceAddr, aclDataType::ACL_BF16, &query);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(keyHostData, qkShape, &keyDeviceAddr, aclDataType::ACL_BF16, &key);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(valueHostData, vShape, &valueDeviceAddr, aclDataType::ACL_BF16, &value);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gamaHostData, gamaShape, &gamaDeviceAddr, aclDataType::ACL_FLOAT, &gama);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(betaHostData, gamaShape, &betaDeviceAddr, aclDataType::ACL_BF16, &beta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(actSeqLenHostData, actSeqLenShape, &actSeqLenDeviceAddr, aclDataType::ACL_INT32, &actSeqLen);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(ssmStaIdHostData, ssmStaIdShape, &ssmStaIdDeviceAddr, aclDataType::ACL_INT32, &ssmStaId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(numAccTokHostData, actSeqLenShape, &numAccTokDeviceAddr, aclDataType::ACL_INT32, &numAccTok);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(attnOutHostData, vShape, &attnOutDeviceAddr, aclDataType::ACL_BF16, &attnOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    float scale = 1.0;
    aclOpExecutor *executor;
    // 调用aclnnRecurrentGatedDeltaRuleGetWorkspaceSize第一段接口
    ret = aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(query, key, value, beta, stateRef, actSeqLen, ssmStaId, gama,
                                                       gamak, numAccTok, scale, attnOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRecurrentGatedDeltaRuleGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnRecurrentGatedDeltaRule第二段接口
    ret = aclnnRecurrentGatedDeltaRule(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRecurrentGatedDeltaRule failed. ERROR: %d\n", ret); return ret);

    // 4. 同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    PrintOutResult(stateShape, &stateRefDeviceAddr);
    PrintOutResult(vShape, &attnOutDeviceAddr);

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(query);
    aclDestroyTensor(key);
    aclDestroyTensor(value);
    aclDestroyTensor(gama);
    aclDestroyTensor(beta);
    aclDestroyTensor(stateRef);
    aclDestroyTensor(actSeqLen);
    aclDestroyTensor(ssmStaId);
    aclDestroyTensor(numAccTok);
    aclDestroyTensor(attnOut);

    // 7. 释放device资源
    aclrtFree(query);
    aclrtFree(key);
    aclrtFree(value);
    aclrtFree(gama);
    aclrtFree(beta);
    aclrtFree(stateRef);
    aclrtFree(actSeqLen);
    aclrtFree(ssmStaId);
    aclrtFree(numAccTok);
    aclrtFree(attnOut);
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
