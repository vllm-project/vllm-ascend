# aclnnMsaIndexScore

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/msa_index_score)

## 产品支持情况

<!-- npu="910b" id1 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id3 -->

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

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1600px"><colgroup>
  <col style="width: 200px">
  <col style="width: 100px">
  <col style="width: 280px">
  <col style="width: 420px">
  <col style="width: 280px">
  <col style="width: 100px">
  <col style="width: 220px">
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>query</td>
      <td>输入</td>
      <td>公式中的$Q_{idx}$。</td>
      <td>支持的shape为：TND。</td>
      <td>BFLOAT16、FLOAT16、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>3（$[T1, N1, D]$）</td>
    </tr>
    <tr>
      <td>key</td>
      <td>输入</td>
      <td>公式中的$K_{idx}$。</td>
      <td>支持的shape为：TND、BNBD、BBND。A2/A3 与 Ascend 950 上 PA key 允许 dim0（物理 page）非连续，page 内其余轴须连续。TND 不允许非连续。</td>
      <td>BFLOAT16、FLOAT16、INT8、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>3（$[T2, N2, D]$）或 4（$[block\_num, N2, block\_size, D]$、$[block\_num, block\_size, N2, D]$）</td>
    </tr>
    <tr>
      <td>blockTableOptional</td>
      <td>输入</td>
      <td>表示当前传入的使用PageAttention存储的block映射表。</td>
      <td>PageAttention场景下，blockTableOptional需为2维；第二维长度不能小于 $maxBlockNumPerSeq$。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>2（$[B, S2/block\_size]$）</td>
    </tr>
    <tr>
      <td>scaleOptional</td>
      <td>输入</td>
      <td>公式中的$scale$，反量化系数。</td>
      <td>非量化场景传入 <code>nullptr</code>；量化场景时必选。PA 为 BNB/BBN；TND 为 $[T2, N2]$（N2=1 时可 $[T2]$）。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>3（$[block\_num, N2, block\_size]$、$[block\_num, block\_size, N2]$）或 2（$[T2, N2]$）</td>
    </tr>
    <tr>
      <td>attenMaskOptional</td>
      <td>输入</td>
      <td>控制因果可见的mask掩码。</td>
      <td>仅在sparseMode=3时使用，作为base mask控制causal可见；取值为1代表该位不参与计算，为0代表该位参与计算。</td>
      <td>INT8</td>
      <td>ND</td>
      <td>2（$[2048, 2048]$）</td>
    </tr>
    <tr>
      <td>actualSeqQlenOptional</td>
      <td>输入</td>
      <td>每个Batch中，Query的有效token数。</td>
      <td>当传入TND时，该入参必须传入，单调不减（前缀和）。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1（$[B+1]$）</td>
    </tr>
    <tr>
      <td>actualSeqKlenOptional</td>
      <td>输入</td>
      <td>每个Batch中，Key的有效token数。</td>
      <td>key 为 TND 时必须传入，单调不减（前缀和，$[B+1]$）；PageAttention 场景下为各请求可见 $S2$（$[B]$）。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1（$[B]$ 或 $[B+1]$）</td>
    </tr>
    <tr>
      <td>startLoc</td>
      <td>输入</td>
      <td>当前 query 所在逻辑 block 索引（非 token 前缀）。</td>
      <td>与 <code>initBlocks</code> / <code>localBlocks</code> 一起生成 $local\_mask$。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1（$[B]$）</td>
    </tr>
    <tr>
      <td>layoutKeyOptional</td>
      <td>输入</td>
      <td>key 的数据排布。</td>
      <td>取值 <code>"TND"</code> / <code>"BBND"</code> / <code>"BNBD"</code>。不传或空串时默认 <code>"BBND"</code>。必须与 <code>key</code> 实际 shape 一致，不可仅凭维度推断。</td>
      <td>CHAR*</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseMode</td>
      <td>输入</td>
      <td>表示sparse的模式。</td>
      <td>为0时，代表defaultMask模式；为3时，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>initBlocks</td>
      <td>输入</td>
      <td>$local\_mask$ 强制选中的头部 block 数。</td>
      <td>对逻辑 block $[0, initBlocks)$ 写入高分 $1\mathrm{e}30$。可选，默认 $0$；须 $\ge 0$ 且 $\le maxBlockNumPerSeq$。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>localBlocks</td>
      <td>输入</td>
      <td>$local\_mask$ 强制选中的局部窗口长度。</td>
      <td>窗口为 $[max(0, startLoc+1-localBlocks), startLoc]$，写入高分 $1\mathrm{e}29$（覆盖同位置 init）。可选，默认 $1$（对齐 MiniMax HF）；与 Triton raw score 对齐时置 $0$。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>score</td>
      <td>输出</td>
      <td>公式中的$score$。</td>
      <td>逐block的重要性分数；末维为对齐后的逻辑 block 数。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>3（$[N1, T1, RoundUp(maxBlockNumPerSeq, 16)]$）</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
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
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
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
      <td>ACLNN_SUCCESS</td>
      <td>0</td>
      <td>执行成功。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>必选入参或出参为空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>数据类型、数据格式、维度或取值不满足约束。</td>
    </tr>
  </tbody>
  </table>

## aclnnMsaIndexScore

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnMsaIndexScoreGetWorkspaceSize获取。</td>
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

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
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
      <td>ACLNN_SUCCESS</td>
      <td>0</td>
      <td>执行成功。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>参数不合法。</td>
    </tr>
  </tbody>
  </table>

## 约束说明

- 当前 $block\_size$ 值使用128。
- `layoutKeyOptional` 必须显式指定 key 布局：`"BBND"`（$[block\_num, block\_size, N2, D]$）、`"BNBD"`（$[block\_num, N2, block\_size, D]$）或 `"TND"`（$[T2, N2, D]$）。不传时默认 `"BBND"`。
- PageAttention（`layoutKey` 为 `"BBND"` / `"BNBD"`）场景下，`blockTableOptional` 必须传入；TND key 场景不得传入 `blockTableOptional`，`actualSeqKlenOptional` 为 $[B+1]$ 前缀和。
- 非量化场景下，`key` dtype 与 `query` 相同（BFLOAT16 / FLOAT16；Ascend 950 另支持 HIFLOAT8 / FLOAT8_E5M2 / FLOAT8_E4M3FN），`scaleOptional` 必须为 `nullptr`。量化场景仅支持 INT8（fp16 query），`scaleOptional` 必选，dtype 为 FLOAT：PA 为 $[block\_num, N2, block\_size]$ 或 $[block\_num, block\_size, N2]$，TND 为 $[T2, N2]$。三种 FP8 仅 950，query 与 key 必须同型。
- `sparseMode` 当前仅支持 0、3：
    - 为 0 时，代表 defaultMask 模式，`attenMaskOptional` 传入 `nullptr`；
    - 为 3 时，代表 rightDownCausal 模式，`attenMaskOptional` 必须传入，shape 为 $[2048, 2048]$，取值为 1 代表该位不参与计算，为 0 代表该位参与计算。
- `initBlocks`、`localBlocks` 必须 $\ge 0$ 且不超过逻辑 block 数（PA 为 `blockTableOptional` 第二维；TND 为 score 末维对齐宽度）。两者均为 0 时跳过 $local\_mask$。
- `q_len` / `kv_len` 允许为 0（含整 batch）。对应请求跳过 QK；空 KV 的 score 填 $-inf$；整 batch $T1=0$ 时 `SetBlockDim(1)`（对齐 FIA PR 9246）。
- PageAttention `key`（BBND/BNBD）允许首轴非连续（`key | gap | key | ...`），tiling 通过 `GetInputStride` 读取 dim0 元素 stride 写入 `strideKvBlock`；非首轴必须连续。TND `key` 不允许非连续。`scale` 仍按逻辑 page 紧凑布局。
- PageAttention `blockTable` 第二维可以大于实际 KV 逻辑 block 数；score 末维为 $\mathrm{RoundUp}(width, 16)$。Ascend 950 C2UB 路径对超过 256 列的末维按 256 列滑窗 flush，并补写后续 $-inf$。
- 精度自验证矩阵：36 条 fp16/bf16/int8（含 PA key dim0 stride 与宽 `blockTable`）+ 4 条 FP8。950 通过标准末行 `[PASS]: 40/40 cases passed`。A2/A3 跳过 FP8，期望 36/36（skipped 4）。容差 fp16/bf16/int8 为 $1\mathrm{e}{-3}$，FP8 为 $2\mathrm{e}{-2}$。
- Ascend 950 核实现位于 `op_kernel/arch35/`：当前与 A2 共用 8-page S workspace（非量化 fp16 / int8 fp32）；Cube 原生三种 FP8。`--run_example` 默认 soc 为 910b，950 必须显式 `--soc=ascend950`。


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
