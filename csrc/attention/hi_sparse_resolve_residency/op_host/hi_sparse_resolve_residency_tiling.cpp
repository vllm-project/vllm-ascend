// SPDX-License-Identifier: Apache-2.0
#include "hi_sparse_resolve_residency_tiling.h"
#include "register/op_impl_registry.h"
#include "../op_kernel/resolver_scratch.h"
#include <cstring>

namespace optiling {
namespace {
ge::graphStatus TileHiSparseResolveResidency(gert::TilingContext* context) {
    const auto* attrs = context->GetAttrs();
    const auto* geometry = attrs ? attrs->GetListInt(0) : nullptr;
    if (!geometry || geometry->GetSize() != 25) return ge::GRAPH_FAILED;
    const auto* data = geometry->GetData();
    if (data[0] <= 0 || data[0] > INT32_MAX || data[1] <= 0 || data[1] > data[2] ||
        data[2] > 32768 || data[21] <= 0 || data[21] > data[2] ||
        data[22] <= data[21] || data[22] > 65536) return ge::GRAPH_FAILED;
    using namespace vllm_ascend::hisparse;
    const uint64_t ubBytes = resolver_scratch_words(data[2], data[22]) * sizeof(int32_t) + kResolverDescriptorBytes;
    if (ubBytes > kResolverUbBytes) return ge::GRAPH_FAILED;

    ResolverTilingData tiling{};
    for (size_t i = 0; i < 25; ++i) tiling.geometry[i] = static_cast<uint64_t>(data[i]);
    tiling.ubBytes = static_cast<uint32_t>(ubBytes);
    auto* raw = context->GetRawTilingData();
    if (!raw || raw->GetCapacity() < sizeof(tiling)) return ge::GRAPH_FAILED;
    std::memcpy(raw->GetData(), &tiling, sizeof(tiling));
    raw->SetDataSize(sizeof(tiling));
    context->SetBlockDim(static_cast<uint32_t>(data[0]));
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PrepareHiSparseResolveResidency(gert::TilingParseContext*) {
    return ge::GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_OPTILING(HiSparseResolveResidency)
    .Tiling(TileHiSparseResolveResidency)
    .TilingParse<HiSparseResolveResidencyCompileInfo>(PrepareHiSparseResolveResidency);
}  // namespace optiling
