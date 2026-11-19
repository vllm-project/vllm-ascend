/**
 * 五阶段算子共用 Host tiling 填充。
 */
#ifndef K2Q_CSR_TILING_COMMON_H
#define K2Q_CSR_TILING_COMMON_H

#include <cstdint>
#include "../op_kernel/k2q_csr_tiling.h"
#include "arch35/k2q_csr_tiling_arch35.h"

#include "log/log.h"
// vllm-ascend：头文件在 csrc/common/include/tiling_base/（勿用 ops-transformer 的 op_host/）
#include "tiling_base/tiling_templates_registry.h"
#include "tiling_base/tiling_util.h"
#include "platform/platform_infos_def.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace k2q_csr_tiling {

using namespace Ops::Transformer::OpTiling;

constexpr uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;
constexpr int64_t ROW_MAP_UB_MAX = 32768;
constexpr int64_t DEFAULT_TILE_EDGES = 2048;
constexpr int64_t EMIT_TILE_EDGES = 2048;
constexpr int64_t UB_RESERVE_BYTES = 8 * 1024;
constexpr uint64_t SIMT_DCACHE_BYTES = 32ULL * 1024ULL;

enum class Stage : int32_t {
    Meta = 0,
    Hist = 1,
    RowPrefix = 2,
    TilePrefix = 3,
    Scatter = 4,
};

inline ge::graphStatus GetPlatformInfo(gert::TilingContext *context, uint64_t &ubSize, int64_t &aivNum,
                                       platform_ascendc::SocVersion &socVersion)
{
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    socVersion = ascendcPlatform.GetSocVersion();
    aivNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(aivNum == 0, OP_LOGE(context, "aivNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus ReadRequiredAttr(gert::TilingContext *context, size_t idx, int64_t &out)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *p = attrs->GetAttrPointer<int64_t>(idx);
    OP_CHECK_NULL_WITH_CONTEXT(context, p);
    out = *p;
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckNonNeg(gert::TilingContext *context, const char *name, int64_t v)
{
    OP_CHECK_IF(v < 0, OP_LOGE(context, "%s must be >= 0, got %ld", name, v), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/** Meta: in0=cu_q, in1=cu_b；attrs: order,total_rows,max_kv,H,T,topk */
inline ge::graphStatus GetShapeAttrsMeta(gert::TilingContext *context, int64_t &H, int64_t &T, int64_t &topk,
                                         int64_t &B, int64_t &totalRows, int64_t &maxKv, int64_t &rowMapElems,
                                         int32_t &orderMethod)
{
    auto cuBShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, cuBShape);
    auto cuB = EnsureNotScalar(cuBShape->GetStorageShape());
    OP_CHECK_IF(cuB.GetDimNum() != 1, OP_LOGE(context, "cu_block_lens must be 1-D"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(cuB.GetDim(0) < 1, OP_LOGE(context, "cu_block_lens length must be >= 1"), return ge::GRAPH_FAILED);
    B = cuB.GetDim(0) - 1;

    int64_t order = 0;
    OP_CHECK_IF(ReadRequiredAttr(context, 0, order) != ge::GRAPH_SUCCESS, OP_LOGE(context, "order_method"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ReadRequiredAttr(context, 1, totalRows) != ge::GRAPH_SUCCESS, OP_LOGE(context, "total_rows"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ReadRequiredAttr(context, 2, maxKv) != ge::GRAPH_SUCCESS, OP_LOGE(context, "max_kv"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ReadRequiredAttr(context, 3, H) != ge::GRAPH_SUCCESS, OP_LOGE(context, "num_heads"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ReadRequiredAttr(context, 4, T) != ge::GRAPH_SUCCESS, OP_LOGE(context, "num_tokens"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ReadRequiredAttr(context, 5, topk) != ge::GRAPH_SUCCESS, OP_LOGE(context, "topk"),
                return ge::GRAPH_FAILED);
    if (CheckNonNeg(context, "total_rows", totalRows) != ge::GRAPH_SUCCESS ||
        CheckNonNeg(context, "max_kv", maxKv) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    orderMethod = static_cast<int32_t>(order);
    rowMapElems = B * maxKv;
    return ge::GRAPH_SUCCESS;
}

/** Hist: in0=q2k；attrs: total_rows,max_kv,use_simt,batch */
inline ge::graphStatus GetShapeAttrsHist(gert::TilingContext *context, int64_t &H, int64_t &T, int64_t &topk,
                                         int64_t &B, int64_t &totalRows, int64_t &maxKv, int64_t &rowMapElems,
                                         int32_t &orderMethod)
{
    (void)orderMethod;
    auto q2kShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, q2kShape);
    auto q2k = EnsureNotScalar(q2kShape->GetStorageShape());
    OP_CHECK_IF(q2k.GetDimNum() != 3, OP_LOGE(context, "q2k must be [H,T,topk]"), return ge::GRAPH_FAILED);
    H = q2k.GetDim(0);
    T = q2k.GetDim(1);
    topk = q2k.GetDim(2);

    OP_CHECK_IF(ReadRequiredAttr(context, 0, totalRows) != ge::GRAPH_SUCCESS, OP_LOGE(context, "total_rows"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ReadRequiredAttr(context, 1, maxKv) != ge::GRAPH_SUCCESS, OP_LOGE(context, "max_kv"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ReadRequiredAttr(context, 3, B) != ge::GRAPH_SUCCESS, OP_LOGE(context, "batch"),
                return ge::GRAPH_FAILED);
    if (CheckNonNeg(context, "total_rows", totalRows) != ge::GRAPH_SUCCESS ||
        CheckNonNeg(context, "max_kv", maxKv) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    orderMethod = 0;
    rowMapElems = B * maxKv;
    return ge::GRAPH_SUCCESS;
}

/** RowPrefix/TilePrefix: attrs: total_rows,max_kv,use_simt,H,T,topk,B */
inline ge::graphStatus GetShapeAttrsPrefix(gert::TilingContext *context, int64_t &H, int64_t &T, int64_t &topk,
                                           int64_t &B, int64_t &totalRows, int64_t &maxKv, int64_t &rowMapElems,
                                           int32_t &orderMethod)
{
    OP_CHECK_IF(ReadRequiredAttr(context, 0, totalRows) != ge::GRAPH_SUCCESS, OP_LOGE(context, "total_rows"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ReadRequiredAttr(context, 1, maxKv) != ge::GRAPH_SUCCESS, OP_LOGE(context, "max_kv"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ReadRequiredAttr(context, 3, H) != ge::GRAPH_SUCCESS, OP_LOGE(context, "num_heads"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ReadRequiredAttr(context, 4, T) != ge::GRAPH_SUCCESS, OP_LOGE(context, "num_tokens"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ReadRequiredAttr(context, 5, topk) != ge::GRAPH_SUCCESS, OP_LOGE(context, "topk"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ReadRequiredAttr(context, 6, B) != ge::GRAPH_SUCCESS, OP_LOGE(context, "batch"),
                return ge::GRAPH_FAILED);
    if (CheckNonNeg(context, "total_rows", totalRows) != ge::GRAPH_SUCCESS ||
        CheckNonNeg(context, "max_kv", maxKv) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    orderMethod = 0;
    rowMapElems = B * maxKv;
    return ge::GRAPH_SUCCESS;
}

/** Scatter: in0=q2k, in1=cu_q；attrs: total_rows,max_kv,use_simt,q_global_offset */
inline ge::graphStatus GetShapeAttrsScatter(gert::TilingContext *context, int64_t &H, int64_t &T, int64_t &topk,
                                            int64_t &B, int64_t &totalRows, int64_t &maxKv, int64_t &rowMapElems,
                                            int32_t &orderMethod)
{
    auto q2kShape = context->GetInputShape(0);
    auto cuQShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, q2kShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, cuQShape);
    auto q2k = EnsureNotScalar(q2kShape->GetStorageShape());
    auto cuQ = EnsureNotScalar(cuQShape->GetStorageShape());
    OP_CHECK_IF(q2k.GetDimNum() != 3, OP_LOGE(context, "q2k must be [H,T,topk]"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(cuQ.GetDimNum() != 1, OP_LOGE(context, "cu_seqlens must be 1-D"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(cuQ.GetDim(0) < 1, OP_LOGE(context, "cu_seqlens length must be >= 1"), return ge::GRAPH_FAILED);
    H = q2k.GetDim(0);
    T = q2k.GetDim(1);
    topk = q2k.GetDim(2);
    B = cuQ.GetDim(0) - 1;

    OP_CHECK_IF(ReadRequiredAttr(context, 0, totalRows) != ge::GRAPH_SUCCESS, OP_LOGE(context, "total_rows"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ReadRequiredAttr(context, 1, maxKv) != ge::GRAPH_SUCCESS, OP_LOGE(context, "max_kv"),
                return ge::GRAPH_FAILED);
    if (CheckNonNeg(context, "total_rows", totalRows) != ge::GRAPH_SUCCESS ||
        CheckNonNeg(context, "max_kv", maxKv) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    orderMethod = 0;
    rowMapElems = B * maxKv;
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus GetShapeAttrsInfo(gert::TilingContext *context, Stage stage, int64_t &H, int64_t &T,
                                         int64_t &topk, int64_t &B, int64_t &totalRows, int64_t &maxKv,
                                         int64_t &rowMapElems, int32_t &orderMethod)
{
    switch (stage) {
        case Stage::Meta:
            return GetShapeAttrsMeta(context, H, T, topk, B, totalRows, maxKv, rowMapElems, orderMethod);
        case Stage::Hist:
            return GetShapeAttrsHist(context, H, T, topk, B, totalRows, maxKv, rowMapElems, orderMethod);
        case Stage::RowPrefix:
        case Stage::TilePrefix:
            return GetShapeAttrsPrefix(context, H, T, topk, B, totalRows, maxKv, rowMapElems, orderMethod);
        case Stage::Scatter:
            return GetShapeAttrsScatter(context, H, T, topk, B, totalRows, maxKv, rowMapElems, orderMethod);
        default:
            OP_LOGE(context, "unknown stage");
            return ge::GRAPH_FAILED;
    }
}

inline int64_t CalcTileEdges(uint64_t ubSize, int64_t rowMapElems, int32_t useGather, int64_t T, int64_t totalRows)
{
    int64_t rowMapBytes = useGather ? ((rowMapElems * 4 + 31) / 32 * 32) : 0;
    int64_t tokenBytes = (T * 4 + 31) / 32 * 32;
    int64_t countsBytes = ((totalRows + 1) * 4 + 31) / 32 * 32;
    int64_t avail =
        static_cast<int64_t>(ubSize) - rowMapBytes - tokenBytes - countsBytes - UB_RESERVE_BYTES;
    if (avail <= 0) {
        return 256;
    }
    int64_t tile = avail / 4;
    tile = tile > DEFAULT_TILE_EDGES ? DEFAULT_TILE_EDGES : tile;
    return tile > 0 ? tile : 256;
}

inline ge::graphStatus GetWorkspaceSize(gert::TilingContext *context, int64_t T, int64_t B, int64_t maxKv)
{
    size_t *ws = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    int64_t meta = k2q_csr_arch35::MetaWorkspaceBytes(B, maxKv, T);
    ws[0] = WS_SYS_SIZE + static_cast<size_t>(meta);
    return ge::GRAPH_SUCCESS;
}

/** use_simt：Hist attr2；Prefix attr2；Scatter attr2；Meta 无此 attr → 0 */
inline int32_t ReadUseSimt(gert::TilingContext *context, Stage stage, bool isArch35)
{
    if (stage == Stage::Meta || !isArch35) {
        return 0;
    }
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return 0;
    }
    const int64_t *useSimtPtr = attrs->GetAttrPointer<int64_t>(2);
    return (useSimtPtr != nullptr && *useSimtPtr != 0) ? 1 : 0;
}

/** q_global_offset：仅 Scatter attr3；其余阶段固定 0 */
inline int32_t ReadQGlobalOffset(gert::TilingContext *context, Stage stage)
{
    if (stage != Stage::Scatter) {
        return 0;
    }
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return 0;
    }
    const int64_t *ptr = attrs->GetAttrPointer<int64_t>(3);
    return (ptr != nullptr && *ptr != 0) ? 1 : 0;
}

inline int64_t UsedCoresForStage(Stage stage, int64_t numGroups, int64_t aivNum, int64_t totalRows, int64_t H)
{
    switch (stage) {
        case Stage::Meta:
            return 1;
        case Stage::RowPrefix: {
            int64_t cores = H > 0 ? H : 1;
            if (cores > aivNum) {
                cores = aivNum;
            }
            return cores < 1 ? 1 : cores;
        }
        case Stage::Hist:
        case Stage::Scatter:
            return numGroups;
        case Stage::TilePrefix: {
            int64_t cores = totalRows > 0 ? totalRows : 1;
            if (cores > aivNum) {
                cores = aivNum;
            }
            return cores < 1 ? 1 : cores;
        }
        default:
            return 1;
    }
}

inline ge::graphStatus FillTiling(gert::TilingContext *context, Stage stage)
{
    uint64_t ubSize = 0;
    int64_t aivNum = 0;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, aivNum, socVersion) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    int64_t H = 0, T = 0, topk = 0, B = 0, totalRows = 0, maxKv = 0, rowMapElems = 0;
    int32_t orderMethod = 0;
    OP_CHECK_IF(GetShapeAttrsInfo(context, stage, H, T, topk, B, totalRows, maxKv, rowMapElems, orderMethod) !=
                    ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    int64_t N = T * topk;
    bool isArch35 = k2q_csr_arch35::IsArch35Soc(socVersion);
    int32_t useSimt = ReadUseSimt(context, stage, isArch35);
    int32_t qGlobalOffset = ReadQGlobalOffset(context, stage);

    int64_t numGroups = 1;
    int64_t qPerGroup = T > 0 ? T : 1;
    if (useSimt != 0) {
        k2q_csr_arch35::FillCudaLikeGroups(T, aivNum, numGroups, qPerGroup);
    } else {
        k2q_csr_arch35::FillMultiCoreGroups(T, aivNum, numGroups, qPerGroup);
    }

    // 仅 SIMT Hist/Scatter 需为 DCache 预留 LocalMemory；MC/SIMD 不走 SIMT
    if (useSimt != 0 && (stage == Stage::Hist || stage == Stage::Scatter)) {
        if (ubSize > SIMT_DCACHE_BYTES) {
            (void)context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - SIMT_DCACHE_BYTES));
        }
    }

    int64_t usedCores = UsedCoresForStage(stage, numGroups, aivNum, totalRows, H);

    OP_CHECK_IF(GetWorkspaceSize(context, T, B, maxKv) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    K2qCsrTilingData *tiling = context->GetTilingData<K2qCsrTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(K2qCsrTilingData), 0, sizeof(K2qCsrTilingData)) != EOK,
                OP_LOGE(context, "memset tiling failed"), return ge::GRAPH_FAILED);

    int32_t useGather = (rowMapElems <= ROW_MAP_UB_MAX) ? 1 : 0;
    int64_t tileEdges = CalcTileEdges(ubSize, rowMapElems, useGather, T, totalRows);
    int64_t numTiles = (N + tileEdges - 1) / tileEdges;
    if (numTiles <= 0) {
        numTiles = 1;
    }
    int64_t totalUnits = H * numTiles;

    tiling->H = H;
    tiling->T = T;
    tiling->topk = topk;
    tiling->N = N;
    tiling->B = B;
    tiling->totalRows = totalRows;
    tiling->maxKv = maxKv;
    tiling->rowMapElems = rowMapElems;
    tiling->tileEdges = tileEdges;
    tiling->numTiles = numTiles;
    tiling->totalUnits = totalUnits;
    tiling->workPer = totalUnits;
    tiling->maxIters = totalUnits;
    tiling->usedCores = usedCores;
    tiling->orderMethod = orderMethod;
    tiling->useGather = useGather;
    tiling->emitTileEdges = EMIT_TILE_EDGES;
    tiling->isArch35 = isArch35 ? 1 : 0;
    tiling->numGroups = static_cast<int32_t>(numGroups);
    tiling->qPerGroup = static_cast<int32_t>(qPerGroup);
    tiling->reserved0 = static_cast<int32_t>(stage);
    tiling->useSimt = useSimt;
    tiling->qGlobalOffset = qGlobalOffset;

    // Hist/Scatter 已无核内 SyncAll（-1 由 Host fill_）；保持默认调度以利多核重叠
    context->SetBlockDim(static_cast<uint32_t>(usedCores));
    return ge::GRAPH_SUCCESS;
}

} // namespace k2q_csr_tiling
} // namespace optiling

#endif
