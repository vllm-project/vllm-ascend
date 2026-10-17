// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <acl/acl.h>
#include <cstdio>
#include <limits>
#include <memory>
#include <unordered_set>
#include <vector>

namespace {
constexpr size_t kPage = 2 * 1024 * 1024;
constexpr size_t kAlignment = 512;
constexpr size_t kDevicePadding = 32;

size_t align_up(size_t bytes, size_t alignment) {
    TORCH_CHECK(bytes <= std::numeric_limits<size_t>::max() - alignment,
                "EPLB storage size overflow");
    return (bytes + alignment - 1) / alignment * alignment;
}

void check(aclError status, const char* operation) {
    TORCH_CHECK(status == ACL_SUCCESS, operation, ": ACL status ", status);
}

struct CompactAllocation {
    void* pointer = nullptr;
    aclrtContext context = nullptr;

    explicit CompactAllocation(size_t capacity) {
        check(aclrtGetCurrentContext(&context), "get EPLB context");
        check(aclrtMalloc(&pointer, capacity, ACL_MEM_MALLOC_HUGE_ONLY), "allocate compact EPLB weights");
    }

    ~CompactAllocation() {
        aclrtContext previous = nullptr;
        // External DataPtrs are not tracked by the NPU caching allocator's
        // record_stream. Wait for all streams before the last storage frees
        // the shared allocation; HIXL also pins storages until deregistration.
        aclrtGetCurrentContext(&previous);
        auto status = aclrtSetCurrentContext(context);
        if (status == ACL_SUCCESS) {
            try {
                if (!c10_npu::npuSynchronizeDevice(false)) {
                    if (previous != nullptr) aclrtSetCurrentContext(previous);
                    return;
                }
            } catch (const std::exception& error) {
                std::fprintf(stderr, "compact EPLB synchronization failed: %s\n", error.what());
                if (previous != nullptr) aclrtSetCurrentContext(previous);
                return;
            }
        }
        if (status == ACL_SUCCESS) status = aclrtFree(pointer);
        if (previous != nullptr) aclrtSetCurrentContext(previous);
        if (status != ACL_SUCCESS) {
            std::fprintf(stderr, "compact EPLB allocation cleanup failed: %d\n", status);
        }
    }
};

void delete_storage_owner(void* owner) {
    delete static_cast<std::shared_ptr<CompactAllocation>*>(owner);
}

std::vector<int64_t> compact_eplb_weights(at::TensorList tensors) {
    TORCH_CHECK(!tensors.empty(), "Cannot compact empty EPLB weights");
    const auto device = tensors[0].device();
    TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1, "EPLB compaction requires NPU tensors");
    int current_device = -1;
    check(aclrtGetDevice(&current_device), "get EPLB device");
    TORCH_CHECK(current_device == device.index(), "EPLB compaction device mismatch");
    std::unordered_set<c10::StorageImpl*> seen;
    std::vector<c10::Storage> storages;
    size_t capacity = 0, payload = 0;
    for (const auto& tensor : tensors) {
        TORCH_CHECK(tensor.device() == device, "EPLB storages must share one NPU device");
        const auto& storage = tensor.storage();
        if (!seen.insert(storage.unsafeGetStorageImpl()).second) continue;
        TORCH_CHECK(storage.nbytes() > 0 && storage.data(), "Cannot compact empty EPLB storage");
        const auto stride = align_up(storage.nbytes() + kDevicePadding, kAlignment);
        TORCH_CHECK(capacity <= std::numeric_limits<size_t>::max() - stride,
                    "EPLB allocation size overflow");
        capacity += stride;
        payload += storage.nbytes();
        storages.push_back(storage);
    }
    capacity = align_up(capacity, kPage);
    // Drain PyTorch's host task queues too: ACL synchronization alone can
    // finish before load-time format-cast kernels have even been submitted.
    TORCH_CHECK(c10_npu::npuSynchronizeDevice(), "synchronize before EPLB compaction");
    auto owner = std::make_shared<CompactAllocation>(capacity);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(owner->pointer) % kPage == 0,
                "Compact EPLB allocation must be page aligned");
    size_t offset = 0;
    for (auto& storage : storages) {
        const auto bytes = storage.nbytes();
        auto* target = static_cast<unsigned char*>(owner->pointer) + offset;
        check(aclrtMemcpy(target, bytes, storage.data(), bytes, ACL_MEMCPY_DEVICE_TO_DEVICE),
              "copy EPLB storage");
        const auto stride = align_up(bytes + kDevicePadding, kAlignment);
        check(aclrtMemset(target + bytes, stride - bytes, 0, stride - bytes), "pad EPLB storage");
        c10::DataPtr data(target, new std::shared_ptr<CompactAllocation>(owner),
                          &delete_storage_owner, device);
        auto* impl = storage.unsafeGetStorageImpl();
        // Keep the same NPUStorageImpl, format descriptor, offsets and all
        // aliases. UniqueStorageShareExternalPointer requires unique storage
        // ownership, which real Parameters and their EPLB views do not have.
        auto previous = impl->set_data_ptr(std::move(data));
        impl->set_resizable(false);
        impl->set_allocator(nullptr);
        offset += stride;
    }
    return {static_cast<int64_t>(reinterpret_cast<uintptr_t>(owner->pointer)),
            static_cast<int64_t>(capacity), static_cast<int64_t>(payload),
            static_cast<int64_t>(storages.size())};
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(_C_ascend, ops) {
    ops.def("compact_eplb_weights(Tensor(a!)[] tensors) -> int[]",
            &compact_eplb_weights);
}
