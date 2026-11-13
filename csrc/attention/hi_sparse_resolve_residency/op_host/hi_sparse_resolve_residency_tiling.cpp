// SPDX-License-Identifier: Apache-2.0
#include "hi_sparse_resolve_residency_tiling.h"
#include "register/op_impl_registry.h"
#include "../op_kernel/hi_sparse_resolve_residency_tiling_data.h"
#include "../op_kernel/resolver_scratch.h"

namespace optiling {
namespace {
enum ResolverInput : size_t {
    kLogical = 0,
    kStates = 2,
    kSource = 3,
    kResident = 4,
};

constexpr size_t kTagsOutput = 0;

enum ResolverAttr : size_t {
    kHostRows,
    kHotBlockSize,
    kSourceBlockSize,
    kResidentBlockSize,
    kResidentNullBlock,
    kAttentionBlockStride,
    kMaxUnionRows,
    kAttentionCapacity,
    kLogicalStride,
    kSourceStride,
    kResidentStride,
    kHotTableStride,
    kAttentionStride,
    kCountsStride,
    kByteMissMask,
};

ge::graphStatus TileHiSparseResolveResidency(gert::TilingContext* context) {
    using namespace vllm_ascend::hisparse;
    const auto* logical = context->GetInputShape(kLogical);
    const auto* tags = context->GetOutputShape(kTagsOutput);
    const auto* attrs = context->GetAttrs();
    if (!logical || !tags || !attrs) return ge::GRAPH_FAILED;
    const auto& logicalShape = logical->GetOriginShape();
    const auto& tagShape = tags->GetOriginShape();
    if (logicalShape.GetDimNum() != 2 || tagShape.GetDimNum() != 2) return ge::GRAPH_FAILED;

    const int64_t rows = logicalShape.GetDim(0);
    const int64_t topk = logicalShape.GetDim(1);
    const int64_t hot = tagShape.GetDim(1);
    const auto* unionRows = attrs->GetInt(kMaxUnionRows);
    if (rows <= 0 || rows > INT32_MAX || topk <= 0 || topk > hot || hot > 32768 ||
        !unionRows || *unionRows <= 0 || *unionRows > hot) return ge::GRAPH_FAILED;
    if (resolver_scratch_bytes(hot, *unionRows + 1) > kResolverUbBytes) return ge::GRAPH_FAILED;
    const uint64_t hashSize = resolver_hash_capacity(hot, *unionRows);

    const auto* states = context->GetOptionalInputShape(kStates);
    const auto* source = context->GetOptionalInputShape(kSource);
    const auto* resident = context->GetOptionalInputShape(kResident);
    if ((states && states->GetOriginShape().GetDimNum() != 1) ||
        (source && source->GetOriginShape().GetDimNum() != 2) ||
        (resident && resident->GetOriginShape().GetDimNum() != 2)) return ge::GRAPH_FAILED;

    auto* tiling = context->GetTilingData<ResolverTilingData>();
    auto* workspace = context->GetWorkspaceSizes(1);
    if (!tiling || !workspace) return ge::GRAPH_FAILED;
    tiling->rows = rows;
    tiling->topk = topk;
    tiling->hotSlots = hot;
    tiling->tagRows = tagShape.GetDim(0);
    tiling->stateIndexCount = states ? states->GetOriginShape().GetDim(0) : 0;
    tiling->sourceRows = source ? source->GetOriginShape().GetDim(0) : 0;
    tiling->sourceColumns = source ? source->GetOriginShape().GetDim(1) : 0;
    tiling->residentRows = resident ? resident->GetOriginShape().GetDim(0) : 0;
    tiling->residentColumns = resident ? resident->GetOriginShape().GetDim(1) : 0;
    tiling->hashSize = hashSize;
    tiling->ubBytes = static_cast<uint32_t>(resolver_scratch_bytes(hot, hashSize));

    // ACLNN view dimensions describe the logical shape. The adapter passes
    // actual row strides separately for padded input and output views.
    tiling->hostRows = *attrs->GetInt(kHostRows);
    tiling->hotBlockSize = *attrs->GetInt(kHotBlockSize);
    tiling->sourceBlockSize = *attrs->GetInt(kSourceBlockSize);
    tiling->residentBlockSize = *attrs->GetInt(kResidentBlockSize);
    tiling->residentNullBlock = *attrs->GetInt(kResidentNullBlock);
    tiling->attentionBlockStride = *attrs->GetInt(kAttentionBlockStride);
    tiling->maxUnionRows = *attrs->GetInt(kMaxUnionRows);
    tiling->attentionCapacity = *attrs->GetInt(kAttentionCapacity);
    tiling->logicalStride = *attrs->GetInt(kLogicalStride);
    tiling->sourceStride = *attrs->GetInt(kSourceStride);
    tiling->residentStride = *attrs->GetInt(kResidentStride);
    tiling->hotTableStride = *attrs->GetInt(kHotTableStride);
    tiling->attentionStride = *attrs->GetInt(kAttentionStride);
    tiling->countsStride = *attrs->GetInt(kCountsStride);
    tiling->byteMissMask = *attrs->GetInt(kByteMissMask);

    context->SetBlockDim(static_cast<uint32_t>(rows));
    workspace[0] = 0;
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
