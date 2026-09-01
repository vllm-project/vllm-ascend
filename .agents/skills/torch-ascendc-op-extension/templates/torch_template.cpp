// 注意事项见 SKILL.md Step 2。
// Tiling 结构体和常量通过 #include 引入（单一数据源），ComputeTiling 为示例占位，
// 替换为源工程的实际计算逻辑。
//
// ⚠️ Stream 同步反模式（详见 SKILL.md「Stream 同步模式」和 references/anti_patterns.md）：
//   ❌ stream(false) + 函数调用 → 乱序：不清 queue，kernel 先于之前操作执行
//   ❌ lambda 内传 NPUStream + OpCommand → 死锁：queue 等 lambda，lambda 等 queue 空
//   ❌ zeros_like 创建输出 → 乱序：zeros_like 入 queue 但 kernel 不入 queue，用 empty_like
//   ✅ stream(true) + 函数调用 → 清 queue，安全

#include <cstdint>
#include "acl/acl.h"                        // 必须在 torch 之前 include
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "../op_kernel/xxx_tiling.h"            // Tiling 结构体 + 常量（单一数据源）

// ---- extern 声明 kernel 入口（不 include .asc！）----
// <<<>>> 直调语法的 kernel 编译后生成 C 符号
// 从 xxx_kernel.asc 中找到核函数签名，去掉 __global__ __vector__，加 extern "C"
// 注意：_custom 后缀是 kernel 入口，_torch 后缀是 PyTorch 接入层函数，避免重名
extern "C" void xxx_custom(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
                            uint8_t *x1, uint8_t *x2, uint8_t *y, uint8_t *tiling);

// ---- Host 侧 Tiling 计算（示例占位，替换为源工程的实际逻辑）----
// 从原 .asc 的 main() 中提取 tiling 计算代码，改写到此处
static XxxTilingData ComputeTiling(int64_t totalElements, int64_t availableCoreNum)
{
    uint32_t usedCoreNum = (totalElements < availableCoreNum)
                           ? static_cast<uint32_t>(totalElements)
                           : static_cast<uint32_t>(availableCoreNum);
    usedCoreNum = (usedCoreNum > 0) ? usedCoreNum : 1;
    uint32_t elementsPerCore = (totalElements + usedCoreNum - 1) / usedCoreNum;

    XxxTilingData tiling;
    tiling.totalElements = static_cast<uint32_t>(totalElements);
    tiling.usedCoreNum = usedCoreNum;
    tiling.elementsPerCore = elementsPerCore;
    return tiling;
}

// ---- 算子实现 ----
// _torch 后缀区分 PyTorch 接入层函数和 _custom 后缀的 kernel 入口
namespace ascend_kernel {

at::Tensor xxx_torch(const at::Tensor& x1, const at::Tensor& x2)
{
    // 1. 参数校验（根据算子语义调整 dtype 和输入数量）
    TORCH_CHECK(x1.scalar_type() == at::kFloat, "only FP32 supported");
    TORCH_CHECK(x1.scalar_type() == x2.scalar_type(), "x1 and x2 must have the same dtype");
    TORCH_CHECK(x1.is_privateuseone(), "x1 must be on NPU");   // C++ API，不是 is_npu()
    TORCH_CHECK(x2.is_privateuseone(), "x2 must be on NPU");

    // 2. 分配输出（用 empty_like，禁止 zeros_like）
    at::Tensor y = at::empty_like(x1);

    int64_t totalElements = x1.numel();
    TORCH_CHECK(totalElements > 0, "input tensors must not be empty");

    // 3. 获取框架 stream
    // stream(true) 在返回 ACL stream 前会清 queue（等待之前任务完成），
    // 确保与之前 NPU 操作的正确同步。
    // 禁止使用 stream(false)：不清 queue，直接启动 kernel 可能导致乱序。
    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);

    // 4. 查询可用核数 & 计算 Tiling
    int32_t deviceId = -1;
    aclrtGetDevice(&deviceId);
    int64_t availableCoreNum = 0;
    auto ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
    TORCH_CHECK(ret == ACL_SUCCESS && availableCoreNum > 0, "failed to get NPU core count");

    XxxTilingData tiling = ComputeTiling(totalElements, availableCoreNum);
    uint32_t blockNum = tiling.usedCoreNum;

    // 5. Tiling 数据搬到 device
    at::Tensor tilingTensor = at::empty(
        {static_cast<int64_t>(sizeof(XxxTilingData))},
        x1.options().dtype(at::kByte));
    aclrtMemcpy(tilingTensor.mutable_data_ptr(), sizeof(XxxTilingData),
        &tiling, sizeof(XxxTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    // 6. 调用 kernel（blockDim = 实际使用的 block 数；l2Ctrl 传 nullptr）
    xxx_custom(blockNum, nullptr, aclStream,
        reinterpret_cast<uint8_t*>(x1.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(x2.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(y.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(tilingTensor.mutable_data_ptr()));

    return y;
}

} // namespace ascend_kernel
