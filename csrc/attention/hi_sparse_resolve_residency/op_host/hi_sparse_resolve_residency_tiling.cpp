// SPDX-License-Identifier: Apache-2.0
#include "hi_sparse_resolve_residency_tiling.h"
#include "register/op_impl_registry.h"
#include "../op_kernel/resolver_scratch.h"

namespace optiling {
namespace {
constexpr size_t kLogicalInput = 0;
constexpr size_t kHotTableInput = 1;
constexpr size_t kRequestsInput = 2;
constexpr size_t kStatesInput = 3;
constexpr size_t kSourceInput = 4;
constexpr size_t kResidentInput = 5;
constexpr size_t kValidRowsInput = 6;
constexpr size_t kTagsOutput = 0;
constexpr size_t kLruOutput = 1;
constexpr size_t kAttentionOutput = 5;
constexpr size_t kSwapHostOutput = 7;
constexpr size_t kSwapDeviceOutput = 8;
constexpr size_t kSwapCountsOutput = 9;

constexpr size_t kHostRowsAttr = 0;
constexpr size_t kHotBlockSizeAttr = 1;
constexpr size_t kSourceBlockSizeAttr = 2;
constexpr size_t kResidentBlockSizeAttr = 3;
constexpr size_t kResidentNullBlockAttr = 4;
constexpr size_t kAttentionBlockStrideAttr = 5;
constexpr size_t kMaxUnionRowsAttr = 6;
constexpr size_t kAttentionCapacityAttr = 7;
constexpr size_t kLogicalStrideAttr = 8;
constexpr size_t kSourceStrideAttr = 9;
constexpr size_t kResidentStrideAttr = 10;
constexpr size_t kHotTableStrideAttr = 11;
constexpr size_t kAttentionStrideAttr = 12;
constexpr size_t kCountsStrideAttr = 13;
constexpr size_t kByteMissMaskAttr = 15;

ge::graphStatus TileHiSparseResolveResidency(gert::TilingContext* context) {
    using namespace vllm_ascend::hisparse;
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("HiSparseResolveResidency", "TilingContext is null."),
                return ge::GRAPH_FAILED);
    const auto* logical = context->GetInputShape(kLogicalInput);
    const auto* tags = context->GetOutputShape(kTagsOutput);
    const auto* lru = context->GetOutputShape(kLruOutput);
    const auto* attrs = context->GetAttrs();
    OP_CHECK_IF(logical == nullptr || tags == nullptr || lru == nullptr || attrs == nullptr,
                OP_LOGE(context->GetNodeName(), "failed to query shapes or attributes."),
                return ge::GRAPH_FAILED);

    const auto& logicalShape = logical->GetOriginShape();
    const auto& tagShape = tags->GetOriginShape();
    OP_CHECK_IF(logicalShape.GetDimNum() != 2 || tagShape.GetDimNum() != 2,
                OP_LOGE(context->GetNodeName(),
                        "TopK, hot tags and block table must be two-dimensional."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tagShape.GetDims() != lru->GetOriginShape().GetDims(),
                OP_LOGE(context->GetNodeName(), "LRU output geometry must match hot tags."),
                return ge::GRAPH_FAILED);

    const int64_t rows = logicalShape.GetDim(0);
    const int64_t topk = logicalShape.GetDim(1);
    const int64_t hot = tagShape.GetDim(1);
    OP_CHECK_IF(rows <= 0 || rows > INT32_MAX,
                OP_LOGE(context->GetNodeName(), "HiSparse TopK row count must fit int32."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(topk <= 0 || topk > hot || hot > 32768,
                OP_LOGE(context->GetNodeName(),
                        "HiSparse TopK must be non-empty and fit the hot cache."),
                return ge::GRAPH_FAILED);

    const auto* hostRows = attrs->GetInt(kHostRowsAttr);
    const auto* hotBlockSize = attrs->GetInt(kHotBlockSizeAttr);
    const auto* attentionBlockStride = attrs->GetInt(kAttentionBlockStrideAttr);
    const auto* unionRows = attrs->GetInt(kMaxUnionRowsAttr);
    const auto* attentionCapacity = attrs->GetInt(kAttentionCapacityAttr);
    OP_CHECK_IF(hostRows == nullptr || hotBlockSize == nullptr || attentionBlockStride == nullptr ||
                    unionRows == nullptr || attentionCapacity == nullptr,
                OP_LOGE(context->GetNodeName(), "required attributes are missing."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*unionRows <= 0 || *unionRows > hot,
                OP_LOGE(context->GetNodeName(), "maxUnionRows must fit the hot cache."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*hostRows <= 0 || *hostRows > INT32_MAX,
                OP_LOGE(context->GetNodeName(), "host row capacity must fit int32."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*hotBlockSize <= 0 || *attentionBlockStride < *hotBlockSize,
                OP_LOGE(context->GetNodeName(), "invalid hot/attention block strides."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(resolver_scratch_bytes(hot, *unionRows + 1) > kResolverUbBytes,
                OP_LOGE(context->GetNodeName(),
                        "HiSparse scratch requirement exceeds the compiled UB budget; "
                        "lower device_buffer_size or speculative tokens."),
                return ge::GRAPH_FAILED);

    const auto* states = context->GetOptionalInputShape(kStatesInput);
    const auto* source = context->GetOptionalInputShape(kSourceInput);
    const auto* resident = context->GetOptionalInputShape(kResidentInput);
    OP_CHECK_IF(states != nullptr && states->GetOriginShape().GetDimNum() != 1,
                OP_LOGE(context->GetNodeName(), "states must be one-dimensional."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(source != nullptr && source->GetOriginShape().GetDimNum() != 2,
                OP_LOGE(context->GetNodeName(), "source must be two-dimensional."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(resident != nullptr && resident->GetOriginShape().GetDimNum() != 2,
                OP_LOGE(context->GetNodeName(), "resident must be two-dimensional."),
                return ge::GRAPH_FAILED);

    const auto* sourceBlockSize = attrs->GetInt(kSourceBlockSizeAttr);
    const auto* residentBlockSize = attrs->GetInt(kResidentBlockSizeAttr);
    const auto* sourceStride = attrs->GetInt(kSourceStrideAttr);
    const auto* residentStride = attrs->GetInt(kResidentStrideAttr);
    OP_CHECK_IF(sourceBlockSize == nullptr || residentBlockSize == nullptr ||
                    sourceStride == nullptr || residentStride == nullptr,
                OP_LOGE(context->GetNodeName(), "required attributes are missing."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(source != nullptr && *sourceBlockSize <= 0,
                OP_LOGE(context->GetNodeName(), "invalid source block size."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(resident != nullptr && *residentBlockSize <= 0,
                OP_LOGE(context->GetNodeName(), "invalid resident block size."),
                return ge::GRAPH_FAILED);

    const auto* logicalStride = attrs->GetInt(kLogicalStrideAttr);
    const auto* hotTableStride = attrs->GetInt(kHotTableStrideAttr);
    const auto* attentionStride = attrs->GetInt(kAttentionStrideAttr);
    const auto* countsStride = attrs->GetInt(kCountsStrideAttr);
    const auto* byteMissMask = attrs->GetInt(kByteMissMaskAttr);
    OP_CHECK_IF(logicalStride == nullptr || hotTableStride == nullptr || attentionStride == nullptr ||
                    countsStride == nullptr || byteMissMask == nullptr,
                OP_LOGE(context->GetNodeName(), "required attributes are missing."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*logicalStride <= 0 || *hotTableStride <= 0 ||
                    *attentionStride < topk || *countsStride <= 0,
                OP_LOGE(context->GetNodeName(), "invalid input or output strides."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(source != nullptr && *sourceStride < source->GetOriginShape().GetDim(1),
                OP_LOGE(context->GetNodeName(), "source row stride must cover its columns."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(resident != nullptr && *residentStride < resident->GetOriginShape().GetDim(1),
                OP_LOGE(context->GetNodeName(), "resident row stride must cover its columns."),
                return ge::GRAPH_FAILED);

    const auto* attention = context->GetOutputShape(kAttentionOutput);
    const auto* swapHost = context->GetOutputShape(kSwapHostOutput);
    const auto* swapDevice = context->GetOutputShape(kSwapDeviceOutput);
    const auto* swapCounts = context->GetOutputShape(kSwapCountsOutput);
    OP_CHECK_IF(attention == nullptr || swapHost == nullptr || swapDevice == nullptr || swapCounts == nullptr,
                OP_LOGE(context->GetNodeName(), "failed to query output shapes."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*attentionCapacity < 0 || *attentionCapacity > INT32_MAX,
                OP_LOGE(context->GetNodeName(), "compact Attention capacity must fit int32."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*attentionCapacity > 0 && attention->GetOriginShape().GetDimNum() == 0,
                OP_LOGE(context->GetNodeName(), "compact Attention requires an output tensor."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((swapHost->GetOriginShape().GetDimNum() == 0) !=
                    (swapDevice->GetOriginShape().GetDimNum() == 0) ||
                    (swapHost->GetOriginShape().GetDimNum() == 0) !=
                        (swapCounts->GetOriginShape().GetDimNum() == 0),
                OP_LOGE(context->GetNodeName(), "all compact swap outputs are required together."),
                return ge::GRAPH_FAILED);

    auto* tiling = context->GetTilingData<HiSparseResolveResidencyTilingData>();
    auto* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_IF(tiling == nullptr || workspace == nullptr,
                OP_LOGE(context->GetNodeName(), "failed to query tiling or workspace."),
                return ge::GRAPH_FAILED);

    const uint64_t hashSize = resolver_hash_capacity(hot, *unionRows);
    tiling->set_rows(rows);
    tiling->set_topk(topk);
    tiling->set_hotSlots(hot);
    tiling->set_tagRows(tagShape.GetDim(0));
    tiling->set_stateIndexCount(states ? states->GetOriginShape().GetDim(0) : 0);
    tiling->set_sourceRows(source ? source->GetOriginShape().GetDim(0) : 0);
    tiling->set_sourceColumns(source ? source->GetOriginShape().GetDim(1) : 0);
    tiling->set_residentRows(resident ? resident->GetOriginShape().GetDim(0) : 0);
    tiling->set_residentColumns(resident ? resident->GetOriginShape().GetDim(1) : 0);
    tiling->set_hashSize(hashSize);
    tiling->set_ubBytes(static_cast<uint32_t>(resolver_scratch_bytes(hot, hashSize)));

    tiling->set_hostRows(*hostRows);
    tiling->set_hotBlockSize(*hotBlockSize);
    tiling->set_sourceBlockSize(*sourceBlockSize);
    tiling->set_residentBlockSize(*residentBlockSize);
    tiling->set_residentNullBlock(*attrs->GetInt(kResidentNullBlockAttr));
    tiling->set_attentionBlockStride(*attentionBlockStride);
    tiling->set_maxUnionRows(*unionRows);
    tiling->set_attentionCapacity(*attentionCapacity);
    tiling->set_logicalStride(*logicalStride);
    tiling->set_sourceStride(*sourceStride);
    tiling->set_residentStride(*residentStride);
    tiling->set_hotTableStride(*hotTableStride);
    tiling->set_attentionStride(*attentionStride);
    tiling->set_countsStride(*countsStride);
    tiling->set_byteMissMask(*byteMissMask);

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
