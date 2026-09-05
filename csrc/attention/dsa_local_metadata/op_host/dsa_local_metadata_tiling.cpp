/**
 * @file dsa_local_metadata_tiling.cpp
 * @brief DsaLocalMetadata TilingFunc implementation
 */

#include "dsa_local_metadata_tiling.h"
#include "tiling_base/error_log.h"
#include "register/op_def_registry.h"
#include "log/ops_log.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    OPS_LOG_D(context, "TilingFunc for DsaLocalMetadata running.");

    // ========== 1. Get operator attributes ==========
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    int64_t localStart = *(attrs->GetAttrPointer<int64_t>(0));
    int64_t localEnd = *(attrs->GetAttrPointer<int64_t>(1));
    int64_t numReqs = *(attrs->GetAttrPointer<int64_t>(2));
    bool computeStartPos = *(attrs->GetAttrPointer<bool>(3));

    // ========== 2. Core distribution ==========
    // The kernel computes an inclusive cumsum across requests, so the whole
    // batch must stay on one core (block-granular stores from multiple cores
    // would also overlap on neighbouring segments). A single AIV core is far
    // more than enough: num_reqs is bounded by max_num_seqs.
    uint32_t usedCoreNum = 1;

    // ========== 3. Set tiling_key ==========
    context->SetTilingKey(1);

    // ========== 4. Fill TilingData ==========
    DsaLocalMetadataTilingData tiling;
    tiling.set_numReqs(static_cast<uint32_t>(numReqs));
    tiling.set_localStart(static_cast<int32_t>(localStart));
    tiling.set_localEnd(static_cast<int32_t>(localEnd));
    tiling.set_computeStartPos(computeStartPos ? 1u : 0u);

    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // ========== 5. Set block_dim ==========
    context->SetBlockDim(usedCoreNum);

    OPS_LOG_D(context,
        "numReqs: %u, localStart: %d, localEnd: %d, computeStartPos: %u",
        (uint32_t)numReqs, (int32_t)localStart, (int32_t)localEnd, computeStartPos ? 1u : 0u);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4DsaLocalMetadata([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DsaLocalMetadata)
    .Tiling(TilingFunc)
    .TilingParse<DsaLocalMetadataCompileInfo>(TilingPrepare4DsaLocalMetadata);

}  // namespace optiling
