// SPDX-License-Identifier: Apache-2.0
#ifdef VLLM_ASCEND_ENGRAM_AICPU
#include <array>
#include "engram_aicpu/urma_params.h"
#include <cstring>
#include <torch/custom_class.h>
#include <torch/library.h>
#include <acl/acl_rt.h>
#include <runtime/rts/rts_kernel.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

namespace {
constexpr size_t HEADER_BYTES = 20;
constexpr char SO_NAME[] = "libengram_aicpu.so";
constexpr char FUNCTION_NAME[] = "EngramUrmaGather";
class EngramUrmaAicpu : public torch::CustomClassHolder {
public:
    explicit EngramUrmaAicpu(std::string path) {
        aclrtBinaryLoadOption option{};
        option.type = ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE;
        option.value.cpuKernelMode = 1;
        aclrtBinaryLoadOptions options{&option, 1};
        TORCH_CHECK(aclrtBinaryLoadFromFile(path.c_str(), &options, &binary_) == ACL_SUCCESS,
                    "Unable to load Engram AICPU binary: ", path);
        auto rc = aclrtRegisterCpuFunc(binary_, FUNCTION_NAME, FUNCTION_NAME, &function_);
        if (rc != ACL_SUCCESS) {
            TORCH_CHECK(false, "Unable to register Engram AICPU function: ", rc);
        }
    }
    // Captured graphs retain this function. Keep its binary registered until
    // the owning runtime context is destroyed.
    ~EngramUrmaAicpu() override = default;
    int64_t function_handle() const { return reinterpret_cast<int64_t>(function_); }
private:
    aclrtBinHandle binary_ = nullptr;
    aclrtFuncHandle function_ = nullptr;
};
void urma_gather(int64_t function, const at::Tensor& metadata, at::Tensor state,
                 const at::Tensor& ids, at::Tensor codes, int64_t head_start,
                 int64_t local_heads, int64_t vocab_start, int64_t vocab_end,
                 int64_t chip, int64_t die, bool close) {
    TORCH_CHECK(metadata.scalar_type()==at::kByte && metadata.numel()==128 && metadata.is_contiguous(), "Invalid URMA metadata");
    TORCH_CHECK(state.scalar_type()==at::kLong && state.numel()==1 && state.is_contiguous(), "Invalid URMA state slot");
    TORCH_CHECK((ids.scalar_type()==at::kLong || ids.scalar_type()==at::kInt) && ids.dim()==2 && ids.stride(1)==1, "Invalid URMA IDs");
    TORCH_CHECK(codes.scalar_type()==at::kByte && codes.dim()==2 && codes.is_contiguous(), "Invalid URMA staging");
    TORCH_CHECK(local_heads>0 && head_start>=0 && head_start+local_heads<=ids.size(1), "Invalid URMA heads");
    TORCH_CHECK(codes.size(0)==ids.size(0)*local_heads && codes.size(1)>0, "Invalid URMA staging shape");
    TORCH_CHECK(vocab_start<=vocab_end && chip>=0 && chip<2 && die>=0 && die<2, "Invalid URMA vocabulary or endpoint");
    TORCH_CHECK(metadata.device()==codes.device() && state.device()==codes.device() && ids.device()==codes.device(), "URMA tensors must share an NPU");
    EngramUrmaParams params{reinterpret_cast<uint64_t>(metadata.data_ptr()),reinterpret_cast<uint64_t>(state.data_ptr()),
                            reinterpret_cast<uint64_t>(ids.data_ptr()),reinterpret_cast<uint64_t>(codes.data_ptr()),
                            ids.size(0),codes.size(1),ids.stride(0),head_start,local_heads,vocab_start,vocab_end,
                            static_cast<int64_t>(ids.element_size()),static_cast<uint64_t>(chip),static_cast<uint64_t>(die),static_cast<uint64_t>(close)};
    auto stream=c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;cmd.Name("EngramUrmaGather");
    cmd.SetCustomHandler([params,stream,function]() -> int {
        constexpr char name[]="EngramUrmaGather";
        std::array<char,HEADER_BYTES+sizeof(params)+sizeof(SO_NAME)+sizeof(name)> buffer{};
        const uint32_t length=HEADER_BYTES+sizeof(params),io_addresses=4;
        std::memcpy(buffer.data(),&length,4);std::memcpy(buffer.data()+4,&io_addresses,4);
        std::memcpy(buffer.data()+HEADER_BYTES,&params,sizeof(params));
        std::memcpy(buffer.data()+length,SO_NAME,sizeof(SO_NAME));
        std::memcpy(buffer.data()+length+sizeof(SO_NAME),name,sizeof(name));
        rtCpuKernelArgs_t args{};args.baseArgs.args=buffer.data();args.baseArgs.argsSize=buffer.size();
        args.baseArgs.soNameAddrOffset=length;args.baseArgs.kernelNameAddrOffset=length+sizeof(SO_NAME);
        return rtsLaunchCpuKernel(reinterpret_cast<rtFuncHandle>(function),1,stream,nullptr,&args);
    });cmd.Run();
}

} // namespace
TORCH_LIBRARY_FRAGMENT(_C_ascend, m) {
    m.class_<EngramUrmaAicpu>("EngramUrmaAicpu")
        .def(torch::init<std::string>())
        .def("function_handle", &EngramUrmaAicpu::function_handle);
    m.def("engram_urma_gather(int function, Tensor metadata, Tensor(a!) state, Tensor ids, Tensor(b!) codes, int head_start, int local_heads, int vocab_start, int vocab_end, int chip, int die, bool close=False) -> ()");
    m.impl("engram_urma_gather", c10::DispatchKey::PrivateUse1, &urma_gather);
}
#endif
