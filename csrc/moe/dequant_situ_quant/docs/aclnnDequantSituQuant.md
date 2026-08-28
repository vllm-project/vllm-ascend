# aclnnDequantSituQuant

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 接口功能：在Situ激活函数前后添加dequant和quant操作，实现x的DequantSituQuant计算。
- 计算公式：

  $$
  dequantOut = cast\_to\_float(x) \times dequantScale + dequantBias
  $$

  $$
  situOut = Situ(dequantOut) = \beta \times \tanh(gate / \beta) \times sigmoid(gate) \times up
  $$

  $$
  out = Quant(situOut, quantScale, quantOffset)
  $$

  其中，当activateLeft为true时，gate取dequantOut的前半部分，up取后半部分；当activateLeft为false时，gate取dequantOut的后半部分，up取前半部分。当linearBeta > 0时，up会被进一步变换为 $linear\_beta \times \tanh(up / linear\_beta)$。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用"aclnnDequantSituQuantGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnDequantSituQuant"接口执行计算。

```Cpp
aclnnStatus aclnnDequantSituQuantGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *dequantScale,
    const aclTensor *dequantBiasOptional,
    const aclTensor *quantScaleOptional,
    const aclTensor *quantOffsetOptional,
    float             beta,
    float             linearBeta,
    bool              activateLeft,
    char             *quantModeOptional,
    const aclTensor *yOut,
    const aclTensor *scaleOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnDequantSituQuant(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnDequantSituQuantGetWorkspaceSize

- **参数说明：**

  | 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 维度(shape) |
  |--------|-----------|------|----------|----------|-------------|
  | x | 输入 | 输入待处理的数据 | shape为(N...,H)，最后一维需要是2的倍数，且x的维度必须大于1维。不支持空Tensor。 | INT8 | 2-8 |
  | dequantScale | 输入 | 反量化scale | shape为(H,)或(1,)。当shape为(H,)时，取值H和x最后一维保持一致。 | FLOAT32 | 1 |
  | dequantBiasOptional | 输入 | 反量化bias | shape为(H,)或(1,)。可选参数，支持传空指针。 | FLOAT32 | 1 |
  | quantScaleOptional | 输入 | 量化的scale | 当quantModeOptional为static时，shape为(H/2,)或(1,)；当quantModeOptional为dynamic时，shape为(H/2,)，作为smoothScale使用。可选参数，支持传空指针。 | FLOAT32 | 1 |
  | quantOffsetOptional | 输入 | 量化的offset | shape为(H/2,)或(1,)。仅当quantModeOptional为static时有效。可选参数，支持传空指针。 | FLOAT32 | 1 |
  | beta | 输入 | Situ激活的beta参数 | 不能为0。 | Float | - |
  | linearBeta | 输入 | Situ激活的linear_beta参数 | 当值≤0时不启用linear_beta变换。 | Float | - |
  | activateLeft | 输入 | 是否对输入的左半部分做Situ激活 | 当值为false时，对输入的右半部分做激活。 | Bool | - |
  | quantModeOptional | 输入 | 量化模式 | 支持"static"和"dynamic"。 | String | - |
  | yOut | 输出 | 量化后的输出 | shape为(N...,H/2)。 | INT8 | - |
  | scaleOut | 输出 | 动态量化的scale | shape为(N,...)，与yOut去除尾轴后的shape一致。 | FLOAT32 | - |

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- x的最后一维需要是2的倍数，且x的维数必须大于1维。
- beta参数不能为0。
- 当quantModeOptional为static时，quantScaleOptional必须提供。
- 当quantModeOptional为dynamic时，quantScaleOptional可选（作为smoothScale使用）。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```C++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_dequant_situ_quant.h"

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) return ret;
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) return ret;

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                              aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1. Initialize device and stream
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclInit(nullptr);
    ret = aclrtSetDevice(deviceId);
    ret = aclrtCreateStream(&stream);

    // 2. Construct inputs: x=[16, 64], static quant mode
    int64_t rowLen = 16;
    int64_t inDimy = 64;
    int64_t outDimy = 32;
    double beta = 1.0;
    double linearBeta = 0.0;
    bool activateLeft = false;

    std::vector<int64_t> xShape = {rowLen, inDimy};
    std::vector<int64_t> dequantScaleShape = {inDimy};
    std::vector<int64_t> quantScaleShape = {1};
    std::vector<int64_t> quantOffsetShape = {1};
    std::vector<int64_t> yShape = {rowLen, outDimy};
    std::vector<int64_t> scaleOutShape = {rowLen};

    auto xSize = GetShapeSize(xShape);
    std::vector<int8_t> xHostData(xSize);
    for (int64_t i = 0; i < xSize; i++) {
        xHostData[i] = static_cast<int8_t>((i * 7 + 3) % 100);
    }
    std::vector<float> dequantScaleHostData(inDimy, 0.1f);
    std::vector<float> quantScaleHostData(1, 1.0f);
    std::vector<float> quantOffsetHostData(1, 0.0f);
    std::vector<int8_t> yHostData(GetShapeSize(yShape), 0);
    std::vector<float> scaleOutHostData(GetShapeSize(scaleOutShape), 0.0f);

    void* xDeviceAddr = nullptr;
    void* dsDeviceAddr = nullptr;
    void* qsDeviceAddr = nullptr;
    void* qoDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* dequantScale = nullptr;
    aclTensor* quantScale = nullptr;
    aclTensor* quantOffset = nullptr;
    aclTensor* y = nullptr;
    aclTensor* scaleOut = nullptr;

    CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_INT8, &x);
    CreateAclTensor(dequantScaleHostData, dequantScaleShape, &dsDeviceAddr, aclDataType::ACL_FLOAT, &dequantScale);
    CreateAclTensor(quantScaleHostData, quantScaleShape, &qsDeviceAddr, aclDataType::ACL_FLOAT, &quantScale);
    CreateAclTensor(quantOffsetHostData, quantOffsetShape, &qoDeviceAddr, aclDataType::ACL_FLOAT, &quantOffset);
    CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_INT8, &y);
    CreateAclTensor(scaleOutHostData, scaleOutShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scaleOut);

    // 3. Call aclnnDequantSituQuantGetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnDequantSituQuantGetWorkspaceSize(x, dequantScale, nullptr, quantScale, quantOffset,
                                                 beta, linearBeta, activateLeft, const_cast<char*>("static"),
                                                 y, scaleOut, &workspaceSize, &executor);

    // 4. Allocate workspace and call aclnnDequantSituQuant
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    ret = aclnnDequantSituQuant(workspaceAddr, workspaceSize, executor, stream);
    ret = aclrtSynchronizeStream(stream);

    // 5. Copy output and cleanup
    std::vector<int8_t> npuYResult(GetShapeSize(yShape), 0);
    aclrtMemcpy(npuYResult.data(), npuYResult.size() * sizeof(int8_t), yDeviceAddr,
                GetShapeSize(yShape) * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);

    aclDestroyTensor(x);
    aclDestroyTensor(dequantScale);
    aclDestroyTensor(quantScale);
    aclDestroyTensor(quantOffset);
    aclDestroyTensor(y);
    aclDestroyTensor(scaleOut);
    aclrtFree(xDeviceAddr);
    aclrtFree(dsDeviceAddr);
    aclrtFree(qsDeviceAddr);
    aclrtFree(qoDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    if (workspaceSize > 0) aclrtFree(workspaceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
