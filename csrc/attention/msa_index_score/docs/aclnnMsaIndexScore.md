# aclnnMsaIndexScore

[📄 View Source](https://gitcode.com/cann/ops-transformer/tree/master/attention/msa_index_score)

## Product Support

| Product                                                               | Supported |
| --------------------------------------------------------------------- | :-------: |
| <term>Atlas A2 Training Series/Atlas 800I A2 Inference Product</term> |     √     |
| <term>Atlas A3 Training Series/Atlas A3 Inference Series</term>       |     √     |
| <term>Ascend 950PR/Ascend 950DT</term>                                |     ×     |

## Function Description

- **Operator behavior**: Computes block scores for the Index Branch of the MSA (MiniMax Sparse Attention) module. For every query token and KV sparse block, it applies matmul and max-pooling to $Q_{idx}$ and $K_{idx}$ over all causally visible tokens in the block, with optional INT8 dequantization. The resulting per-block importance scores are used as the input to the subsequent TopK stage. Prefill and decode share the same interface.

- **Formulas**:

    - Non-quantized:

  $$
  score = Maxpool[ Q_{idx}@K_{idx}^{T} ]
  $$
    - INT8-quantized:

  $$
  score = Maxpool[ scale \cdot Q_{idx}@K_{idx}^{T}  ]
  $$

  Complete formula, including the causal mask and $local\_mask$:

  $$
  score = Maxpool[(scale \cdot) Q_{idx}@K_{idx}^{T} + atten\_mask] + local\_mask
  $$

  $local\_mask$ is generated from `startLoc`, `initBlocks`, and `localBlocks`. Logical blocks in $[0, initBlocks)$ are assigned $1\mathrm{e}30$. Blocks in the window $[max(0, startLoc+1-localBlocks), startLoc]$ are assigned $1\mathrm{e}29$, overriding an init-block score at the same position. Setting both block counts to 0 disables $local\_mask$.

- **Notation**:

  > - B (Batch Size) is the number of input samples.
  > - S (Sequence Length) is the sequence length. $S1$ is the query length and $S2$ is the key length.
  > - T is the sum of sequence lengths across the batch. $T1$ is for queries and $T2$ is for keys.
  > - N (Head Num) is the number of heads. $N1$ is the number of query heads and $N2$ is the number of key heads.
  > - D (Head Dim) is the dimension of each attention head.
  > - For PageAttention, $block\_num$ is the total number of physical blocks, $block\_size$ is the number of tokens per block, and $maxBlockNumPerSeq$ is the maximum number of logical blocks per batch item, typically $\ge\lceil S2/block\_size\rceil$. $M_b=\lceil S2/block\_size\rceil$ is the total number of logical blocks.

## Function Prototypes

This operator uses a [two-stage interface](https://gitcode.com/cann/ops-transformer/blob/master/docs/zh/context/two_phase_api.md). Call `aclnnMsaIndexScoreGetWorkspaceSize` first to validate the inputs and obtain the required workspace size, then call `aclnnMsaIndexScore` to execute the computation.

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

### Parameters

| Parameter             | Input/Output | Description                                        | Usage                                                                                                                                                                                                                  | Data Type               | Format | Dimensions                                                                                      |
| --------------------- | ------------ | -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- | ------ | ----------------------------------------------------------------------------------------------- |
| query                 | Input        | $Q_{idx}$ in the formula                           | The supported layout is TND.                                                                                                                                                                                           | BFLOAT16, FLOAT16       | ND     | 3 ($[T1, N1, D]$)                                                                               |
| key                   | Input        | $K_{idx}$ in the formula                           | The supported layouts are TND, BNBD, and BBND.                                                                                                                                                                         | BFLOAT16, FLOAT16, INT8 | ND, NZ | 3 ($[T2, N2, D]$) or 4 ($[block\_num, N2, block\_size, D]$, $[block\_num, block\_size, N2, D]$) |
| blockTableOptional    | Input        | PageAttention logical-to-physical block mapping    | Required for PageAttention. It must be 2D, and its second dimension must be at least $maxBlockNumPerSeq$.                                                                                                              | INT32                   | ND     | 2 ($[B, S2/block\_size]$)                                                                       |
| scaleOptional         | Input        | Dequantization coefficient $scale$ in the formula  | Pass `nullptr` for non-quantized input. This parameter is required for quantized input. Its layout is BNB/BBN for PageAttention or $[T2, N2]$ for TND; $[T2]$ is also accepted when N2=1.                              | FLOAT                   | ND, NZ | 3 ($[block\_num, N2, block\_size]$, $[block\_num, block\_size, N2]$) or 2 ($[T2, N2]$)          |
| attenMaskOptional     | Input        | Mask controlling causal visibility                 | Used only when `sparseMode=3` as the base causal mask. A value of 1 excludes the position from computation, while 0 includes it.                                                                                       | INT8                    | ND     | 2 ($[2048, 2048]$)                                                                              |
| actualSeqQlenOptional | Input        | Valid query-token counts for each batch item       | Required for TND input. Values must be non-decreasing prefix sums.                                                                                                                                                     | INT32                   | ND     | 1 ($[B+1]$)                                                                                     |
| actualSeqKlenOptional | Input        | Valid key-token counts for each batch item         | For a TND key, this parameter is required and contains non-decreasing prefix sums with shape $[B+1]$. For PageAttention, it contains the visible $S2$ for each request with shape $[B]$.                               | INT32                   | ND     | 1 ($[B]$ or $[B+1]$)                                                                            |
| startLoc              | Input        | Logical block index containing the current query   | Used with `initBlocks` and `localBlocks` to generate $local\_mask$. It is a block index, not a token prefix.                                                                                                           | INT32                   | ND     | 1 ($[B]$)                                                                                       |
| layoutKeyOptional     | Input        | Key layout                                         | Supported values are `"TND"`, `"BBND"`, and `"BNBD"`. An omitted or empty value defaults to `"BBND"`. It must match the actual `key` shape and must not be inferred from the number of dimensions alone.               | CHAR*                   | -      | -                                                                                               |
| sparseMode            | Input        | Sparse-mask mode                                   | 0 selects `defaultMask`; 3 selects the `rightDownCausal` mask, a lower-triangular region aligned to the upper-right vertex.                                                                                            | INT64                   | -      | -                                                                                               |
| initBlocks            | Input        | Number of leading blocks forced by $local\_mask$   | Assigns $1\mathrm{e}30$ to logical blocks in $[0, initBlocks)$. Optional; defaults to 0. The value must be $\ge 0$ and $\le maxBlockNumPerSeq$.                                                                        | INT64                   | -      | -                                                                                               |
| localBlocks           | Input        | Length of the local window forced by $local\_mask$ | Assigns $1\mathrm{e}29$ to $[max(0, startLoc+1-localBlocks), startLoc]$, overriding init-block scores at the same positions. Optional; defaults to 1 to match MiniMax HF. Set it to 0 when matching Triton raw scores. | INT64                   | -      | -                                                                                               |
| score                 | Output       | $score$ in the formula                             | Per-block importance scores. The final dimension is the aligned number of logical blocks.                                                                                                                              | FLOAT                   | ND     | 3 ($[N1, T1, RoundUp(maxBlockNumPerSeq, 16)]$)                                                  |
| workspaceSize         | Output       | Required workspace size in bytes                   | -                                                                                                                                                                                                                      | uint64_t                | -      | -                                                                                               |
| executor              | Output       | Operator executor                                  | -                                                                                                                                                                                                                      | aclOpExecutor**         | -      | -                                                                                               |

### Return Values

| Return Code             | Error Code | Description                                                     |
| ----------------------- | ---------- | --------------------------------------------------------------- |
| ACLNN_SUCCESS           | 0          | Execution succeeded.                                            |
| ACLNN_ERR_PARAM_NULLPTR | 161001     | A required input or output is a null pointer.                   |
| ACLNN_ERR_PARAM_INVALID | 161002     | A data type, format, dimension, or value violates a constraint. |

## aclnnMsaIndexScore

### Parameters

| Parameter     | Input/Output | Description                                                    |
| ------------- | ------------ | -------------------------------------------------------------- |
| workspace     | Input        | Device-side workspace address.                                 |
| workspaceSize | Input        | Workspace size in bytes returned by the first-stage interface. |
| executor      | Input        | Operator executor returned by the first-stage interface.       |
| stream        | Input        | ACL stream.                                                    |

### Return Values

| Return Code             | Error Code | Description          |
| ----------------------- | ---------- | -------------------- |
| ACLNN_SUCCESS           | 0          | Execution succeeded. |
| ACLNN_ERR_PARAM_INVALID | 161002     | Invalid parameter.   |

## Constraints

- The current $block\_size$ is 128.
- `layoutKeyOptional` must explicitly identify the key layout: `"BBND"` for $[block\_num, block\_size, N2, D]$, `"BNBD"` for $[block\_num, N2, block\_size, D]$, or `"TND"` for $[T2, N2, D]$. If omitted, it defaults to `"BBND"`.
- For PageAttention (`layoutKey` is `"BBND"` or `"BNBD"`), `blockTableOptional` is required. For a TND key, `blockTableOptional` must be omitted and `actualSeqKlenOptional` must contain $[B+1]$ prefix sums.
- For non-quantized input, `key` must have the same data type as `query`, currently BFLOAT16 or FLOAT16, and `scaleOptional` must be `nullptr`. Quantized input supports INT8 only and requires a FLOAT `scaleOptional`: $[block\_num, N2, block\_size]$ or $[block\_num, block\_size, N2]$ for PageAttention, and $[T2, N2]$ for TND. FP8 and <term>Ascend 950PR/Ascend 950DT</term> are not currently supported.
- `sparseMode` currently supports only 0 and 3:
    - 0 selects `defaultMask`, and `attenMaskOptional` must be `nullptr`.
    - 3 selects `rightDownCausal`, and `attenMaskOptional` is required with shape $[2048, 2048]$. A value of 1 excludes the position from computation, while 0 includes it.
- `initBlocks` and `localBlocks` must be $\ge 0$ and must not exceed the number of logical blocks: the second dimension of `blockTableOptional` for PageAttention, or the aligned final score dimension for TND. Setting both to 0 skips $local\_mask$.

## Example

The following example uses BBND PageAttention. For TND, set `layoutKeyOptional="TND"`, use a $[T2, N2, D]$ `key`, pass `nullptr` for `blockTableOptional`, and provide $[B+1]$ prefix sums in `actualSeqKlenOptional`. For BNBD, set `layoutKeyOptional="BNBD"` and use a $[block\_num, N2, block\_size, D]$ `key`. See [test_aclnn_msa_index_score.cpp](../examples/test_aclnn_msa_index_score.cpp) for TND and BNBD accuracy self-checks.

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
  std::vector<int8_t> attenMaskHostData(GetShapeSize(attenMaskShape), 0);  // 0: included in computation
  std::vector<int32_t> actualSeqQlenHostData = {0, static_cast<int32_t>(T1)};
  std::vector<int32_t> actualSeqKlenHostData = {static_cast<int32_t>(S2)};
  std::vector<int32_t> startLocHostData = {1};  // Logical block index containing the current query
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

  // Non-quantized input: pass nullptr for scaleOptional
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

The TND and BNBD cases differ from the BBND example above as shown below. See the `L1-bnbd*`, `L1-tnd*`, and `L0-tnd-tiny` cases in [test_aclnn_msa_index_score.cpp](../examples/test_aclnn_msa_index_score.cpp) for complete accuracy tests.

```Cpp
// BNBD PageAttention: layoutKey="BNBD"; key is [block_num, N2, block_size, D]
char layoutKeyBnbd[] = "BNBD";
std::vector<int64_t> keyShapeBnbd = {blockNum, N2, blockSize, D};

// Packed TND key: layoutKey="TND"; omit blockTableOptional; actualSeqKlenOptional is a [B+1] prefix sum
char layoutKeyTnd[] = "TND";
constexpr int64_t T2 = 256;
std::vector<int64_t> keyShapeTnd = {T2, N2, D};
std::vector<int64_t> actualSeqKlenShapeTnd = {B + 1};  // For example, {0, T2}
aclTensor* blockTableTnd = nullptr;
```
