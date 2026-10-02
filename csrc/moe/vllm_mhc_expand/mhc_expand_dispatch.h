// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
#include <limits>
#include <torch_npu/csrc/aten/CustomFunctions.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>

namespace vllm_ascend {

// Snapshot metadata before enqueue: callers may change a TensorImpl's shape or
// strides immediately after this operation returns. Storage ownership also keeps
// temporary inputs and replaced tensor storages alive until submission.
class MhcExpandDescriptor {
public:
    explicit MhcExpandDescriptor(const at::Tensor& tensor)
        : storage_(tensor.storage()), rank_(tensor.dim()), offset_(tensor.storage_offset()),
          storageElements_(storage_.nbytes() / tensor.itemsize()),
          data_(const_cast<void*>(storage_.data())),
          dtype_(kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(tensor.scalar_type())])
    {
        for (int64_t i = 0; i < rank_; ++i) {
            sizes_[i] = tensor.size(i);
            strides_[i] = tensor.stride(i);
        }
    }
    aclTensor* Create() const
    {
        static const auto create = GET_OP_API_FUNC(aclCreateTensor);
        TORCH_CHECK(create != nullptr, "aclCreateTensor is unavailable");
        // Same base-format conversion as the existing project adapter.
        const auto format = rank_ == 3 ? ACL_FORMAT_NCL : ACL_FORMAT_ND;
        return create(sizes_.data(), rank_, dtype_, strides_.data(), offset_, format,
                      &storageElements_, 1, data_);
    }
private:
    c10::Storage storage_;
    std::array<int64_t, 3> sizes_{}, strides_{};
    int64_t rank_, offset_, storageElements_;
    void* data_;
    aclDataType dtype_;
};

// Run ACLNN preparation and submission together on the framework's queue thread.
inline int ExecuteMhcExpand(const MhcExpandDescriptor& x, int64_t mult, const MhcExpandDescriptor& y,
                            c10_npu::NPUStream npuStream, aclrtStream stream)
{
    const c10_npu::NPUStreamGuard streamGuard(npuStream.unwrap());
    using Query = int (*)(const aclTensor*, int64_t, const aclTensor*, uint64_t*, aclOpExecutor**);
    using Execute = int (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);
    static const auto getWorkspace = reinterpret_cast<Query>(
        GetOpApiFuncAddr("aclnnVllmMhcExpandGetWorkspaceSize"));
    static const auto execute = reinterpret_cast<Execute>(GetOpApiFuncAddr("aclnnVllmMhcExpand"));
    static const auto initMemory = reinterpret_cast<InitHugeMemThreadLocal>(
        GetOpApiFuncAddr("InitHugeMemThreadLocal"));
    static const auto uninitMemory = reinterpret_cast<UnInitHugeMemThreadLocal>(
        GetOpApiFuncAddr("UnInitHugeMemThreadLocal"));
    static const auto releaseMemory = reinterpret_cast<ReleaseHugeMem>(GetOpApiFuncAddr("ReleaseHugeMem"));
    TORCH_CHECK(getWorkspace && execute, "mHC Expand ACLNN API is unavailable");
    if (initMemory) {
        initMemory(nullptr, false);
    }
    struct MemoryScope {
        UnInitHugeMemThreadLocal finish;
        ~MemoryScope() { if (finish) finish(nullptr, false); }
    } memoryScope{uninitMemory};
    struct Resources {
        aclTensor* input = nullptr;
        aclTensor* output = nullptr;
        ReleaseHugeMem release = nullptr;
        ~Resources()
        {
            if (input) Release(input);
            if (output) Release(output);
            if (release) release(nullptr, false);
        }
    } resources;
    resources.release = releaseMemory;
    resources.input = x.Create();
    resources.output = y.Create();
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto workspaceStatus = getWorkspace(resources.input, mult, resources.output, &workspaceSize, &executor);
    TORCH_CHECK(workspaceStatus == 0, "aclnnVllmMhcExpandGetWorkspaceSize failed: ", aclGetRecentErrMsg());
    at::Tensor workspace;
    if (workspaceSize != 0) {
        TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                    "mHC Expand workspace size overflows int64");
        workspace = at::empty({static_cast<int64_t>(workspaceSize)}, at::TensorOptions(npuStream.device()).dtype(at::kByte));
    }
    void* data = workspace.defined() ? const_cast<void*>(workspace.storage().data()) : nullptr;
    const auto status = execute(data, workspaceSize, executor, stream);
    TORCH_CHECK(status == 0, "aclnnVllmMhcExpand failed: ", aclGetRecentErrMsg());
    return status;
}

inline void LaunchMhcExpand(const at::Tensor& x, int64_t mult, const at::Tensor& y)
{
    // Logical contiguity does not imply ND storage: NZ inputs need an explicit
    // conversion because the generated custom ACLNN interface accepts only ND.
    if (!IsOpInputBaseFormat(x)) {
        const auto input = at_npu::native::custom_ops::npu_format_cast(x, static_cast<int64_t>(ACL_FORMAT_ND));
        EXEC_NPU_CMD(aclnnVllmMhcExpand, input, mult, y);
        return;
    }
    // Preserve the existing caller-side path for thread-local core controls.
    if (c10_npu::is_core_control_enabled()) {
        EXEC_NPU_CMD(aclnnVllmMhcExpand, x, mult, y);
        return;
    }
    const auto npuStream = c10_npu::getCurrentNPUStream();
    const auto stream = npuStream.stream(false);
    const MhcExpandDescriptor input(x), output(y);
    at_npu::native::OpCommand::RunOpApiV2("aclnnVllmMhcExpand",
        [input, mult, output, npuStream, stream]() -> int {
            return ExecuteMhcExpand(input, mult, output, npuStream, stream);
        });
}

}  // namespace vllm_ascend
