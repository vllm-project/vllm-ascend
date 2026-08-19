# aclnnMsaIndexScore

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/msa_index_score)

## 产品支持情况

| 产品                                                      | 是否支持 |
| --------------------------------------------------------- | :------: |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品</term> |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>  |    √     |
| <term>Ascend 950PR/Ascend 950DT</term>                    |    ×     |

## 功能说明

- **算子功能**：计算 MSA（MiniMax Sparse Attention）模块 Index Branch 中的 block score。对每个 query token 与每个 KV sparse block，取该 block 内所有因果可见 token 的 $Q_{idx}$ 和 $K_{idx}$（可选 int8 反量化）的"matmul+maxpool"运算获得逐block的重要性分数score，用作 Index Branch 中后续的 TopK 输入。Prefill 与 Decode 由同一接口承载。

- **计算公式**：

    - 非量化场景：

    $$
    score = Maxpool[ Q_{idx}@K_{idx}^{T} ]
    $$

    - int8量化场景：

    $$
    score = Maxpool[ scale \cdot Q_{idx}@K_{idx}^{T}  ]
    $$

    完整公式（含因果 mask 与 local_mask）：

    $$
    score = Maxpool[(scale \cdot) Q_{idx}@K_{idx}^{T} + atten\_mask] + local\_mask
    $$

    $local\_mask$ 由 `startLoc`、`initBlocks`、`localBlocks` 生成：逻辑 block $[0, initBlocks)$ 写入 $1\mathrm{e}30$；窗口 $[max(0, startLoc+1-localBlocks), startLoc]$ 写入 $1\mathrm{e}29$（覆盖同位置的 init）。两者均为 0 时关闭 $local\_mask$。

- **参数介绍**：

    > - B（Batch Size）表示输入样本批量大小
    > - S（Sequence Length）表示序列长度，$S1$ 为 query 侧、$S2$ 为 key 侧
    > - T 表示所有 Batch 序列长度累加和，$T1$ 为 query 侧、$T2$ 为 key 侧
    > - N（Head Num）表示头数，$N1$ 为 query 侧、$N2$ 为 key 侧
    > - D（Head Dim）表示单个注意力头维度；
    > - PageAttention 场景下 $block\_num$ 为物理 block 总数、$block\_size$ 为每个 block 的 token 数，$maxBlockNumPerSeq$ 为每个 batch 最大逻辑 block 数（通常 $\ge\lceil S2/block\_size\rceil$），$M_b=\lceil S2/block\_size\rceil$为逻辑 block 总数

## 函数原型

每个算子分为[两段式接口](https://gitcode.com/cann/ops-transformer/blob/master/docs/zh/context/two_phase_api.md)，必须先调用 `aclnnMsaIndexScoreGetWorkspaceSize` 接口获取入参并计算所需 workspace 大小，再调用 `aclnnMsaIndexScore` 接口执行计算。

```cpp
aclnnStatus aclnnMsaIndexScoreGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *blockTableOptional,
    const aclTensor *scaleOptional,
    const aclTensor *attenMaskOptional,
    const aclTensor *actualSeqQlenOptional,
    const aclTensor *actualSeqKlenOptional,
    const aclTensor *startLoc,
    char            *layoutKeyOptional,
    int64_t          sparseMode,
    int64_t          initBlocks,
    int64_t          localBlocks,
    const aclTensor *score,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor);
aclnnStatus aclnnMsaIndexScore(
    void           *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor  *executor,
    aclrtStream     stream);
```

## aclnnMsaIndexScoreGetWorkspaceSize

### 参数说明

| 参数名                        | 输入/输出 | 描述                                              | 使用说明                                                     | 数据类型                     | 数据格式 | 维度                                                         |
| ----------------------------- | --------- | ------------------------------------------------- | ------------------------------------------------------------ | ---------------------------- | -------- | ------------------------------------------------------------ |
| query                         | 输入      | 公式中的$Q_{idx}$                                 | 支持的shape为：TND                                           | BFLOAT16, FLOAT16            | ND       | 3（$[T1, N1, D]$）                                           |
| key                           | 输入      | 公式中的$K_{idx}$                                 | 支持的shape为：TND、BNBD、BBND                               | BFLOAT16, FLOAT16, INT8      | ND, NZ   | 3（$[T2, N2, D]$）或 4（$[block\_num, N2, block\_size, D]$、$[block\_num, block\_size, N2, D]$） |
| blockTableOptional            | 输入      | 表示当前传入的使用PageAttention存储的block映射表  | PageAttention场景下，blockTableOptional需为2维；第二维长度不能小于 $maxBlockNumPerSeq$ | INT32                        | ND       | 2（$[B, S2/block\_size]$）                                   |
| scaleOptional                 | 输入      | 公式中的$scale$，反量化系数                       | 非量化场景传入`nullptr`；量化场景时必选。PA 为 BNB/BBN；TND 为 $[T2, N2]$（N2=1 时可 $[T2]$） | FLOAT                        | ND, NZ   | 3（$[block\_num, N2, block\_size]$、$[block\_num, block\_size, N2]$）或 2（$[T2, N2]$） |
| attenMaskOptional             | 输入      | 控制因果可见的mask掩码。                          | 仅在sparseMode=3时使用，作为base mask控制causal可见；取值为1代表该位不参与计算，为0代表该位参与计算 | INT8                         | ND       | 2（$[2048, 2048]$）                                          |
| actualSeqQlenOptional | 输入      | 每个Batch中，Query的有效token数                   | 当传入TND时，该入参必须传入，单调不减（前缀和）              | INT32                        | ND       | 1（$[B+1]$）                                                 |
| actualSeqKlenOptional   | 输入      | 每个Batch中，Key的有效token数                     | key 为 TND 时必须传入，单调不减（前缀和，$[B+1]$）；PageAttention 场景下为各请求可见 $S2$（$[B]$） | INT32                        | ND       | 1（$[B]$ 或 $[B+1]$）                                        |
| startLoc                      | 输入      | 当前 query 所在逻辑 block 索引（非 token 前缀） | 与 `initBlocks` / `localBlocks` 一起生成 $local\_mask$ | INT32                        | ND       | 1（$[B]$）                                                   |
| layoutKeyOptional             | 输入      | key 的数据排布                                | 取值 `"TND"` / `"BBND"` / `"BNBD"`。不传或空串时默认 `"BBND"`。必须与 `key` 实际 shape 一致，不可仅凭维度推断 | CHAR*                        | -        | -                                                            |
| sparseMode                    | 输入      | 表示sparse的模式                                  | 为0时，代表defaultMask模式；为3时，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。 | INT64                        | -        | -                                                            |
| initBlocks                    | 输入      | $local\_mask$ 强制选中的头部 block 数             | 对逻辑 block $[0, initBlocks)$ 写入高分 $1\mathrm{e}30$。可选，默认 $0$；须 $\ge 0$ 且 $\le maxBlockNumPerSeq$ | INT64                        | -        | -                                                            |
| localBlocks                   | 输入      | $local\_mask$ 强制选中的局部窗口长度              | 窗口为 $[max(0, startLoc+1-localBlocks), startLoc]$，写入高分 $1\mathrm{e}29$（覆盖同位置 init）。可选，默认 $1$（对齐 MiniMax HF）；与 Triton raw score 对齐时置 $0$ | INT64                        | -        | -                                                            |
| score                         | 输出      | 公式中的$score$                                   | 逐block的重要性分数；末维为对齐后的逻辑 block 数             | FLOAT                        | ND       | 3（$[N1, T1, RoundUp(maxBlockNumPerSeq, 16)]$）              |
| workspaceSize                 | 输出      | 所需 workspace 字节数                             | -                                                            | uint64_t                     | -        | -                                                            |
| executor                      | 输出      | 算子执行器                                        | -                                                            | aclOpExecutor**              | -        | -                                                            |

### 返回值

| 返回码                  | 错误码 | 说明                                     |
| ----------------------- | ------ | ---------------------------------------- |
| ACLNN_SUCCESS           | 0      | 执行成功                                 |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 必选入参或出参为空指针                   |
| ACLNN_ERR_PARAM_INVALID | 161002 | 数据类型、数据格式、维度或取值不满足约束 |

## aclnnMsaIndexScore

### 参数说明

| 参数名        | 输入/输出 | 描述                               |
| ------------- | --------- | ---------------------------------- |
| workspace     | 输入      | device 侧 workspace 地址           |
| workspaceSize | 输入      | workspace 字节数，由第一段接口返回 |
| executor      | 输入      | 算子执行器，由第一段接口返回       |
| stream        | 输入      | acl stream                         |

### 返回值

| 返回码                  | 错误码 | 说明       |
| ----------------------- | ------ | ---------- |
| ACLNN_SUCCESS           | 0      | 执行成功   |
| ACLNN_ERR_PARAM_INVALID | 161002 | 参数不合法 |

## 约束说明

- 当前 $block\_size$ 值使用128。
- `layoutKeyOptional` 必须显式指定 key 布局：`"BBND"`（$[block\_num, block\_size, N2, D]$）、`"BNBD"`（$[block\_num, N2, block\_size, D]$）或 `"TND"`（$[T2, N2, D]$）。不传时默认 `"BBND"`。
- PageAttention（`layoutKey` 为 `"BBND"` / `"BNBD"`）场景下，`blockTableOptional` 必须传入；TND key 场景不得传入 `blockTableOptional`，`actualSeqKlenOptional` 为 $[B+1]$ 前缀和。
- 非量化场景下，`key` dtype 与 `query` 相同（当前为 BFLOAT16 / FLOAT16），`scaleOptional` 必须为 `nullptr`；量化场景下仅支持 INT8，`scaleOptional` 必选，dtype 为 FLOAT：PA 为 $[block\_num, N2, block\_size]$ 或 $[block\_num, block\_size, N2]$，TND 为 $[T2, N2]$。当前不支持 FP8 与 <term>Ascend 950PR/Ascend 950DT</term>。
- `sparseMode` 当前仅支持 0、3：
    - 为 0 时，代表 defaultMask 模式，`attenMaskOptional` 传入 `nullptr`；
    - 为 3 时，代表 rightDownCausal 模式，`attenMaskOptional` 必须传入，shape 为 $[2048, 2048]$，取值为 1 代表该位不参与计算，为 0 代表该位参与计算。
- `initBlocks`、`localBlocks` 必须 $\ge 0$ 且不超过逻辑 block 数（PA 为 `blockTableOptional` 第二维；TND 为 score 末维对齐宽度）。两者均为 0 时跳过 $local\_mask$。


## 调用示例

示例代码如下（BBND PageAttention）。TND：`layoutKeyOptional="TND"`，`key` 为 $[T2, N2, D]$，`blockTableOptional` 传 `nullptr`，`actualSeqKlenOptional` 为 $[B+1]$ 前缀和。BNBD：`layoutKeyOptional="BNBD"`，`key` 为 $[block\_num, N2, block\_size, D]$。含 TND/BNBD 的精度自验证见 [test_aclnn_msa_index_score.cpp](../examples/test_aclnn_msa_index_score.cpp)。

```Cpp
#include <iostream>
#include <vector>
#include <cstdint>
#include "acl/acl.h"
#include "aclnnop/aclnn_msa_index_score.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true : (false))

#define LOG_PRINT(message, ...)         \
  do {                                  \
    (void)printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int64_t RoundUp(int64_t value, int64_t align) {
  return (value + align - 1) / align * align;
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
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

struct TensorResources {
  void* queryDeviceAddr = nullptr;
  void* keyDeviceAddr = nullptr;
  void* blockTableDeviceAddr = nullptr;
  void* attenMaskDeviceAddr = nullptr;
  void* actualSeqQlenDeviceAddr = nullptr;
  void* actualSeqKlenDeviceAddr = nullptr;
  void* startLocDeviceAddr = nullptr;
  void* scoreDeviceAddr = nullptr;

  aclTensor* queryTensor = nullptr;
  aclTensor* keyTensor = nullptr;
  aclTensor* blockTableTensor = nullptr;
  aclTensor* attenMaskTensor = nullptr;
  aclTensor* actualSeqQlenTensor = nullptr;
  aclTensor* actualSeqKlenTensor = nullptr;
  aclTensor* startLocTensor = nullptr;
  aclTensor* scoreTensor = nullptr;
};

int InitializeTensors(TensorResources& resources) {
  // TND query + PA_BBND key，sparseMode=3
  constexpr int64_t B = 1;
  constexpr int64_t T1 = 2;
  constexpr int64_t N1 = 2;
  constexpr int64_t N2 = 1;
  constexpr int64_t D = 128;
  constexpr int64_t S2 = 256;
  constexpr int64_t blockSize = 128;
  constexpr int64_t blockNum = 2;
  constexpr int64_t maxBlockNumPerSeq = S2 / blockSize;  // 2
  const int64_t scoreStride = RoundUp(maxBlockNumPerSeq, 16);

  std::vector<int64_t> queryShape = {T1, N1, D};
  std::vector<int64_t> keyShape = {blockNum, blockSize, N2, D};           // BBND
  std::vector<int64_t> blockTableShape = {B, maxBlockNumPerSeq};
  std::vector<int64_t> attenMaskShape = {2048, 2048};
  std::vector<int64_t> actualSeqQlenShape = {B + 1};
  std::vector<int64_t> actualSeqKlenShape = {B};
  std::vector<int64_t> startLocShape = {B};
  std::vector<int64_t> scoreShape = {N1, T1, scoreStride};

  std::vector<uint16_t> queryHostData(GetShapeSize(queryShape), 0x3C00);  // fp16 1.0
  std::vector<uint16_t> keyHostData(GetShapeSize(keyShape), 0x3C00);
  std::vector<int32_t> blockTableHostData = {0, 1};
  std::vector<int8_t> attenMaskHostData(GetShapeSize(attenMaskShape), 0);  // 0: 参与计算
  std::vector<int32_t> actualSeqQlenHostData = {0, static_cast<int32_t>(T1)};
  std::vector<int32_t> actualSeqKlenHostData = {static_cast<int32_t>(S2)};
  std::vector<int32_t> startLocHostData = {1};  // 当前 query 所在逻辑 block 索引
  std::vector<float> scoreHostData(GetShapeSize(scoreShape), 0.0f);

  int ret = CreateAclTensor(queryHostData, queryShape, &resources.queryDeviceAddr,
                            aclDataType::ACL_FLOAT16, &resources.queryTensor);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    return ret;
  }
  ret = CreateAclTensor(keyHostData, keyShape, &resources.keyDeviceAddr,
                        aclDataType::ACL_FLOAT16, &resources.keyTensor);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    return ret;
  }
  ret = CreateAclTensor(blockTableHostData, blockTableShape, &resources.blockTableDeviceAddr,
                        aclDataType::ACL_INT32, &resources.blockTableTensor);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    return ret;
  }
  ret = CreateAclTensor(attenMaskHostData, attenMaskShape, &resources.attenMaskDeviceAddr,
                        aclDataType::ACL_INT8, &resources.attenMaskTensor);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    return ret;
  }
  ret = CreateAclTensor(actualSeqQlenHostData, actualSeqQlenShape,
                        &resources.actualSeqQlenDeviceAddr, aclDataType::ACL_INT32,
                        &resources.actualSeqQlenTensor);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    return ret;
  }
  ret = CreateAclTensor(actualSeqKlenHostData, actualSeqKlenShape,
                        &resources.actualSeqKlenDeviceAddr, aclDataType::ACL_INT32,
                        &resources.actualSeqKlenTensor);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    return ret;
  }
  ret = CreateAclTensor(startLocHostData, startLocShape, &resources.startLocDeviceAddr,
                        aclDataType::ACL_INT32, &resources.startLocTensor);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    return ret;
  }
  ret = CreateAclTensor(scoreHostData, scoreShape, &resources.scoreDeviceAddr,
                        aclDataType::ACL_FLOAT, &resources.scoreTensor);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    return ret;
  }
  return ACL_SUCCESS;
}

int ExecuteMsaIndexScore(TensorResources& resources, aclrtStream stream,
                         void** workspaceAddr, uint64_t* workspaceSize) {
  int64_t sparseMode = 3;
  int64_t initBlocks = 0;
  int64_t localBlocks = 1;
  char layoutKey[] = "BBND";
  aclOpExecutor* executor = nullptr;

  // 非量化：scaleOptional 传 nullptr
  int ret = aclnnMsaIndexScoreGetWorkspaceSize(
      resources.queryTensor,
      resources.keyTensor,
      resources.blockTableTensor,
      nullptr,  // scaleOptional
      resources.attenMaskTensor,
      resources.actualSeqQlenTensor,
      resources.actualSeqKlenTensor,
      resources.startLocTensor,
      layoutKey,
      sparseMode,
      initBlocks,
      localBlocks,
      resources.scoreTensor,
      workspaceSize,
      &executor);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclnnMsaIndexScoreGetWorkspaceSize failed. ERROR: %d\n", ret);
    return ret;
  }

  if (*workspaceSize > 0ULL) {
    ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
      return ret;
    }
  }

  ret = aclnnMsaIndexScore(*workspaceAddr, *workspaceSize, executor, stream);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclnnMsaIndexScore failed. ERROR: %d\n", ret);
    return ret;
  }
  return ACL_SUCCESS;
}

int PrintScoreOutResult(const std::vector<int64_t>& shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0.0f);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
    return ret;
  }
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("score result[%ld] is: %f\n", i, resultData[i]);
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
  if (resources.blockTableTensor) {
    aclDestroyTensor(resources.blockTableTensor);
  }
  if (resources.attenMaskTensor) {
    aclDestroyTensor(resources.attenMaskTensor);
  }
  if (resources.actualSeqQlenTensor) {
    aclDestroyTensor(resources.actualSeqQlenTensor);
  }
  if (resources.actualSeqKlenTensor) {
    aclDestroyTensor(resources.actualSeqKlenTensor);
  }
  if (resources.startLocTensor) {
    aclDestroyTensor(resources.startLocTensor);
  }
  if (resources.scoreTensor) {
    aclDestroyTensor(resources.scoreTensor);
  }

  if (resources.queryDeviceAddr) {
    aclrtFree(resources.queryDeviceAddr);
  }
  if (resources.keyDeviceAddr) {
    aclrtFree(resources.keyDeviceAddr);
  }
  if (resources.blockTableDeviceAddr) {
    aclrtFree(resources.blockTableDeviceAddr);
  }
  if (resources.attenMaskDeviceAddr) {
    aclrtFree(resources.attenMaskDeviceAddr);
  }
  if (resources.actualSeqQlenDeviceAddr) {
    aclrtFree(resources.actualSeqQlenDeviceAddr);
  }
  if (resources.actualSeqKlenDeviceAddr) {
    aclrtFree(resources.actualSeqKlenDeviceAddr);
  }
  if (resources.startLocDeviceAddr) {
    aclrtFree(resources.startLocDeviceAddr);
  }
  if (resources.scoreDeviceAddr) {
    aclrtFree(resources.scoreDeviceAddr);
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

}  // namespace

int main() {
  int32_t deviceId = 0;
  aclrtStream stream = nullptr;
  TensorResources resources = {};
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  constexpr int64_t T1 = 2;
  constexpr int64_t N1 = 2;
  constexpr int64_t maxBlockNumPerSeq = 2;
  std::vector<int64_t> scoreShape = {N1, T1, RoundUp(maxBlockNumPerSeq, 16)};
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

  ret = ExecuteMsaIndexScore(resources, stream, &workspaceAddr, &workspaceSize);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("ExecuteMsaIndexScore failed. ERROR: %d\n", ret);
    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return ret;
  }

  ret = aclrtSynchronizeStream(stream);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return ret;
  }

  PrintScoreOutResult(scoreShape, &resources.scoreDeviceAddr);

  CleanupResources(resources, workspaceAddr, stream, deviceId);
  return 0;
}
```

TND / BNBD 与上面 BBND 示例的差异如下（完整精度用例见 [test_aclnn_msa_index_score.cpp](../examples/test_aclnn_msa_index_score.cpp) 中的 `L1-bnbd*`、`L1-tnd*`、`L0-tnd-tiny`）。

```Cpp
// BNBD PageAttention：layoutKey="BNBD"，key 为 [block_num, N2, block_size, D]
char layoutKeyBnbd[] = "BNBD";
std::vector<int64_t> keyShapeBnbd = {blockNum, N2, blockSize, D};

// TND packed key：layoutKey="TND"，不传 blockTableOptional，actualSeqKlenOptional 为 [B+1] 前缀和
char layoutKeyTnd[] = "TND";
constexpr int64_t T2 = 256;
std::vector<int64_t> keyShapeTnd = {T2, N2, D};
std::vector<int64_t> actualSeqKlenShapeTnd = {B + 1};  // 例如 {0, T2}
aclTensor* blockTableTnd = nullptr;
```
