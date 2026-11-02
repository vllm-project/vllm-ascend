// SPDX-License-Identifier: Apache-2.0
// Queue-aware bridge: keep the native launch ordered with TorchNPU producers.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <c10/core/impl/PyObjectSlot.h>
#include <array>
#include <vector>

extern "C" void launch_hisparse_union_resolve60103(void**, uint64_t*, int, void*);

namespace {
void resolve(const std::vector<at::Tensor>& tensors, const std::vector<int64_t>& geometry) {
    TORCH_CHECK(tensors.size() == 19 && geometry.size() == 23, "Invalid HiSparse resolver launch descriptor");
    c10_npu::NPUGuard guard(tensors[0].device());
    // Queue callbacks can be destroyed by HCCL/event cleanup threads while
    // holding a queue mutex. A Python-facing TensorImpl destructor can acquire
    // the GIL, inverting the main-thread event-record lock order. Independent
    // detached TensorImpls retain the same storage without Python ownership.
    std::vector<at::Tensor> owners;
    owners.reserve(tensors.size());
    std::array<void*, 19> pointers{};
    std::array<uint64_t, 23> sizes{};
    for (size_t i = 0; i < tensors.size(); ++i) {
        TORCH_CHECK(tensors[i].device() == tensors[0].device(), "HiSparse resolver device mismatch");
        owners.emplace_back(tensors[i].detach());
        TORCH_CHECK(owners.back().unsafeGetTensorImpl()->pyobj_slot()->load_pyobj() == nullptr,
                    "HiSparse queue owner must not carry a Python object");
        TORCH_CHECK(owners.back().unsafeGetTensorImpl() != tensors[i].unsafeGetTensorImpl(),
                    "HiSparse queue owner must have an independent TensorImpl");
        pointers[i] = tensors[i].numel() ? tensors[i].data_ptr() : nullptr;
    }
    for (size_t i = 0; i < geometry.size(); ++i) sizes[i] = uint64_t(geometry[i]);
    auto stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand command;
    command.Name("HiSparseUnionResolve");
    command.SetCustomHandler([pointers, sizes, stream, owners = std::move(owners)]() mutable -> int {
        launch_hisparse_union_resolve60103(pointers.data(), sizes.data(), int(sizes[0]), stream);
        return 0;
    });
    command.Run();
}

void resolve_meta(const std::vector<at::Tensor>&, const std::vector<int64_t>&) {}
}  // namespace

TORCH_LIBRARY(hisparse_fused60103, m) {
    // Conservative list mutation annotation includes DGI/LRU and all optional
    // outputs. Empty device tensors denote omitted optional ABI arguments.
    m.def("resolve(Tensor(a!)[] tensors, int[] geometry) -> ()");
}
TORCH_LIBRARY_IMPL(hisparse_fused60103, PrivateUse1, m) { m.impl("resolve", TORCH_FN(resolve)); }
TORCH_LIBRARY_IMPL(hisparse_fused60103, Meta, m) { m.impl("resolve", TORCH_FN(resolve_meta)); }
