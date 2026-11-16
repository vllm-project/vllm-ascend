// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
// Portions adapted from UCM, Copyright (c) 2026 Huawei Technologies Co., Ltd.
// MIT License: Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including without
// limitation the rights to use, copy, modify, merge, publish, distribute,
// sublicense, and/or sell copies of the Software, and to permit persons to whom
// the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <torch/custom_class.h>
#include <torch/library.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <acl/acl.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#ifdef CANN_KV_OFFLOAD_FFTS
#if __has_include("runtime/rt_ffts_plus.h")
#include "runtime/rt_ffts_plus.h"
#else
#include "rt_external_ffts.h"
#endif
#endif

namespace vllm_ascend {
namespace {
constexpr int64_t kPinnedCapability = 1;
constexpr int64_t kSdmaCapability = 2;
constexpr uintptr_t kHostPageSize = 4096;
#ifdef CANN_KV_OFFLOAD_FFTS
constexpr size_t kMaxContexts = std::numeric_limits<uint16_t>::max();
constexpr uint16_t kReadyLanes = 8;
constexpr uint16_t kPreloadContexts = 128;
constexpr uint32_t kSdmaMoveHeader = 0x1E70;
constexpr uint8_t kCommunicationTask = 0x5A;
static_assert(sizeof(rtFftsPlusComCtx_t) == 128);
static_assert(sizeof(rtFftsPlusSdmaCtx_t) == 128);
#endif

int64_t dram_capabilities()
{
    int64_t capabilities = 0;
#ifdef CANN_HOST_REGISTER_V2_PINNED
    capabilities |= kPinnedCapability;
#endif
#ifdef CANN_KV_OFFLOAD_FFTS
    capabilities |= kSdmaCapability;
#endif
    return capabilities;
}

// Ownership is per worker/device. No process-global registration cache: separate
// TP workers must each register their own virtual mapping with their device.
class KVOffloadHostMemory : public torch::CustomClassHolder {
public:
    KVOffloadHostMemory(torch::Tensor memory, bool mapped) : memory_(std::move(memory))
    {
        TORCH_CHECK(memory_.device().is_cpu() && memory_.is_contiguous(),
                    "KV offload registration requires contiguous CPU memory");
        host_ = reinterpret_cast<uintptr_t>(memory_.data_ptr());
        size_ = memory_.nbytes();
        TORCH_CHECK(size_ > 0 && host_ % kHostPageSize == 0,
                    "KV offload registration requires nonempty, page-aligned mmap memory");
        [[maybe_unused]] aclError ret;
        if (mapped) {
#ifdef CANN_KV_OFFLOAD_FFTS
#ifdef CANN_HOST_REGISTER_V2_PINNED
            ret = aclrtHostRegisterV2(memory_.data_ptr(), size_,
                                      ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED);
            TORCH_CHECK(ret == ACL_SUCCESS, "KV offload mapped host registration failed: ", ret);
            void* device = nullptr;
            ret = aclrtHostGetDevicePointer(memory_.data_ptr(), &device, 0);
            if (ret != ACL_SUCCESS || device == nullptr) {
                aclrtHostUnregister(memory_.data_ptr());
                TORCH_CHECK(false, "KV offload host device-address lookup failed: ", ret);
            }
#else
            void* device = nullptr;
            ret = aclrtHostRegister(memory_.data_ptr(), size_, ACL_HOST_REGISTER_MAPPED, &device);
            TORCH_CHECK(ret == ACL_SUCCESS, "KV offload mapped host registration failed: ", ret);
            if (device == nullptr) {
                aclrtHostUnregister(memory_.data_ptr());
                TORCH_CHECK(false, "KV offload host registration returned a null device address");
            }
#endif
            mapped_ = reinterpret_cast<uintptr_t>(device);
#else
            TORCH_CHECK(false, "KV offload SDMA is unavailable in this build");
#endif
        } else {
#ifdef CANN_HOST_REGISTER_V2_PINNED
            ret = aclrtHostRegisterV2(memory_.data_ptr(), size_, ACL_HOST_REG_PINNED);
            TORCH_CHECK(ret == ACL_SUCCESS, "KV offload pinned host registration failed: ", ret);
#else
            TORCH_CHECK(false, "KV offload pinned host registration is unavailable in this build");
#endif
        }
        registered_ = true;
    }

    ~KVOffloadHostMemory() override
    {
        if (registered_) {
            const auto ret = aclrtHostUnregister(memory_.data_ptr());
            if (ret != ACL_SUCCESS) {
                TORCH_WARN("KV offload host unregister failed during destruction: ", ret);
            }
        }
    }

    uintptr_t DeviceAddress(uintptr_t host, size_t bytes) const
    {
        TORCH_CHECK(registered_ && mapped_ != 0, "KV offload memory is not device mapped");
        TORCH_CHECK(host >= host_ && host - host_ <= size_ && bytes <= size_ - (host - host_),
                    "KV offload DMA range is outside the registered mmap");
        return mapped_ + (host - host_);
    }

    void AcquireBatch() { ++batches_; }
    void ReleaseBatch() { --batches_; }

    void close()
    {
        TORCH_CHECK(batches_ == 0, "Cannot unregister KV offload memory with outstanding SDMA batches");
        if (!registered_) { return; }
        const auto ret = aclrtHostUnregister(memory_.data_ptr());
        TORCH_CHECK(ret == ACL_SUCCESS, "KV offload host unregister failed: ", ret);
        registered_ = false;
        memory_ = torch::Tensor();
    }

private:
    torch::Tensor memory_;
    uintptr_t host_{0};
    uintptr_t mapped_{0};
    size_t size_{0};
    size_t batches_{0};
    bool registered_{false};
};

class KVOffloadSdmaBatch : public torch::CustomClassHolder {
public:
    KVOffloadSdmaBatch(c10::intrusive_ptr<KVOffloadHostMemory> memory,
                       torch::Tensor src, torch::Tensor dst, torch::Tensor sizes, int64_t direction)
        : memory_(std::move(memory)), src_(std::move(src)), dst_(std::move(dst)), sizes_(std::move(sizes))
    {
#ifdef CANN_KV_OFFLOAD_FFTS
        TORCH_CHECK(direction == 0 || direction == 1, "KV offload SDMA supports H2D and D2H only");
        for (const auto* tensor : {&src_, &dst_, &sizes_}) {
            TORCH_CHECK(tensor->device().is_cpu() && tensor->is_contiguous() &&
                        tensor->dim() == 1 && tensor->scalar_type() == torch::kInt64,
                        "KV offload SDMA descriptors must be contiguous 1D CPU int64 tensors");
        }
        const auto count = src_.numel();
        TORCH_CHECK(dst_.numel() == count && sizes_.numel() == count,
                    "KV offload SDMA descriptor lengths differ");
        contexts_.resize(count);
        const auto* sources = src_.data_ptr<int64_t>();
        const auto* destinations = dst_.data_ptr<int64_t>();
        const auto* lengths = sizes_.data_ptr<int64_t>();
        for (int64_t i = 0; i < count; ++i) {
            TORCH_CHECK(lengths[i] > 0 && static_cast<uint64_t>(lengths[i]) <=
                        std::numeric_limits<uint32_t>::max(), "Invalid KV offload SDMA copy size");
            auto source = static_cast<uintptr_t>(sources[i]);
            auto destination = static_cast<uintptr_t>(destinations[i]);
            if (direction == 0) {
                source = memory_->DeviceAddress(source, lengths[i]);
            } else {
                destination = memory_->DeviceAddress(destination, lengths[i]);
            }
            TORCH_CHECK(source != 0 && destination != 0, "Null KV offload SDMA address");
            auto* context = reinterpret_cast<rtFftsPlusSdmaCtx_t*>(&contexts_[i]);
            context->contextType = RT_CTX_TYPE_SDMA;
            context->threadDim = 1;
            context->sdmaSqeHeader = kSdmaMoveHeader;
            context->sourceAddressBaseL = static_cast<uint32_t>(source);
            context->sourceAddressBaseH = static_cast<uint32_t>(source >> 32);
            context->destinationAddressBaseL = static_cast<uint32_t>(destination);
            context->destinationAddressBaseH = static_cast<uint32_t>(destination >> 32);
            context->nonTailDataLength = static_cast<uint32_t>(lengths[i]);
            context->tailDataLength = static_cast<uint32_t>(lengths[i]);
        }
        // Each launch has local context IDs, with up to eight independent lanes.
        for (size_t begin = 0; begin < contexts_.size(); begin += kMaxContexts) {
            const auto count = std::min(kMaxContexts, contexts_.size() - begin);
            for (size_t i = kReadyLanes; i < count; ++i) {
                auto& predecessor = contexts_[begin + i - kReadyLanes];
                auto& successor = contexts_[begin + i];
                predecessor.successorNum = 1;
                predecessor.successorList[0] = static_cast<uint16_t>(i);
                successor.predCntInit = 1;
                successor.predCnt = 1;
            }
        }
        memory_->AcquireBatch();
        acquired_ = true;
#else
        TORCH_CHECK(false, "KV offload SDMA is unavailable in this build");
#endif
    }

    ~KVOffloadSdmaBatch() override
    {
        // Normal completion calls release() after the Python event is complete.
        // Only exceptional/abandoned transfers need a blocking lifetime fence.
        if (launched_) {
            const auto ret = aclrtSynchronizeStream(stream_);
            if (ret != ACL_SUCCESS) {
                // A failed device fence cannot justify releasing DMA inputs.
                // Retain these resources until process exit after a device fault.
#ifdef CANN_KV_OFFLOAD_FFTS
                (void)new std::vector<rtFftsPlusComCtx_t>(std::move(contexts_));
#endif
                (void)memory_.release();
                acquired_ = false;
                TORCH_WARN("KV offload SDMA cleanup sync failed; retaining mapped memory and descriptors: ", ret);
                return;
            }
        }
        ReleaseMemory();
    }

    void launch()
    {
#ifdef CANN_KV_OFFLOAD_FFTS
        TORCH_CHECK(acquired_ && !launched_, "KV offload SDMA batch cannot be launched twice");
        stream_ = c10_npu::getCurrentNPUStream().stream();
        // Retain all contexts even if a later launch throws after earlier chunks.
        launched_ = true;
        for (size_t begin = 0; begin < contexts_.size(); begin += kMaxContexts) {
            const auto count = static_cast<uint16_t>(std::min(kMaxContexts, contexts_.size() - begin));
            rtFftsPlusSqe_t sqe{};
            sqe.fftsType = RT_FFTS_PLUS_TYPE;
            sqe.totalContextNum = count;
            const auto ready = std::min(count, kReadyLanes);
            sqe.readyContextNum = ready;
            sqe.preloadContextNum = std::min(ready, kPreloadContexts);
            sqe.subType = kCommunicationTask;
            rtFftsPlusTaskInfo_t task{};
            task.fftsPlusSqe = &sqe;
            task.descBuf = contexts_.data() + begin;
            task.descBufLen = sizeof(rtFftsPlusComCtx_t) * count;
            task.descAddrType = RT_FFTS_PLUS_CTX_DESC_ADDR_TYPE_HOST;
            const auto ret = rtFftsPlusTaskLaunchWithFlag(&task, reinterpret_cast<rtStream_t>(stream_), 0);
            TORCH_CHECK(ret == RT_ERROR_NONE, "KV offload FFTS SDMA launch failed: ", ret);
        }
#else
        TORCH_CHECK(false, "KV offload SDMA is unavailable in this build");
#endif
    }

    // Caller must have queried/synchronized the event following this batch.
    void release()
    {
        launched_ = false;
#ifdef CANN_KV_OFFLOAD_FFTS
        contexts_.clear();
#endif
        src_ = dst_ = sizes_ = torch::Tensor();
        ReleaseMemory();
    }

private:
    void ReleaseMemory()
    {
        if (acquired_) {
            memory_->ReleaseBatch();
            acquired_ = false;
        }
        memory_.reset();
    }
    c10::intrusive_ptr<KVOffloadHostMemory> memory_;
    torch::Tensor src_, dst_, sizes_;
#ifdef CANN_KV_OFFLOAD_FFTS
    std::vector<rtFftsPlusComCtx_t> contexts_;
#endif
    aclrtStream stream_{nullptr};
    bool acquired_{false};
    bool launched_{false};
};
}  // namespace

TORCH_LIBRARY_FRAGMENT(_C_ascend, m)
{
    m.def("kv_offload_dram_capabilities() -> int", &dram_capabilities);
    m.class_<KVOffloadHostMemory>("KVOffloadHostMemory")
        .def(torch::init<torch::Tensor, bool>())
        .def("close", &KVOffloadHostMemory::close);
    m.class_<KVOffloadSdmaBatch>("KVOffloadSdmaBatch")
        .def(torch::init<c10::intrusive_ptr<KVOffloadHostMemory>, torch::Tensor,
                         torch::Tensor, torch::Tensor, int64_t>())
        .def("launch", &KVOffloadSdmaBatch::launch)
        .def("release", &KVOffloadSdmaBatch::release);
}
}  // namespace vllm_ascend
