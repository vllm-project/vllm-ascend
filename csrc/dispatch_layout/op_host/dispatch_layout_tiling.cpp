#include <queue>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <string>

#include "log/ops_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "../op_kernel/dispatch_layout_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/hccl/hccl_tiling.h"
#include "platform/platform_infos_def.h"

using namespace ge;
namespace {
constexpr uint32_t INPUT_TOPK_IDX_INDEX = 0;

constexpr uint32_t OUTPUT_NUM_TOKEN_PER_RANK_INDEX = 0;
constexpr uint32_t OUTPUT_NUM_TOKEN_PER_EXPERT_INDEX = 1;
constexpr uint32_t OUTPUT_IS_TOKEN_IN_RANK_INDEX = 2;

constexpr uint32_t ATTR_NUM_TOKENS_INDEX = 0;
constexpr uint32_t ATTR_NUM_RANKS_INDEX = 1;
constexpr uint32_t ATTR_NUM_EXPERTS_INDEX = 2;
constexpr uint32_t ATTR_NUM_TOPK_INDEX = 3;
const int64_t MAX_COMM_WORLD_SIZE = 384;
const int64_t MAX_MOE_EXPERTS_NUM = 384;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
constexpr uint32_t KERNEL_USE_WORKSPACE = 1 * 1024 * 1024;
constexpr uint32_t KERNEL_A2_ARG_SIZE = 1 * 1024 * 1024;

constexpr uint32_t TWO_DIMS = 2;
constexpr uint32_t K_MAX = 16;
}  // namespace

namespace optiling {
static void PrintTilingDataInfo(const char *nodeName, DispatchLayoutTilingData &tilingData)
{
    OPS_LOG_D(nodeName, "numToken is %u.", tilingData.dispatchLayoutInfo.numTokens);
    OPS_LOG_D(nodeName, "numRanks is %u.", tilingData.dispatchLayoutInfo.numRanks);
    OPS_LOG_D(nodeName, "numExperts is %u.", tilingData.dispatchLayoutInfo.numExperts);
    OPS_LOG_D(nodeName, "numTopk is %u.", tilingData.dispatchLayoutInfo.numTopk);
    OPS_LOG_D(nodeName, "totalUbSize is %lu.", tilingData.dispatchLayoutInfo.totalUbSize);
}

static ge::graphStatus GetAttrAndSetTilingData(gert::TilingContext *context, const char *nodeName,
    DispatchLayoutTilingData &tilingData)
{
    auto attrs = context->GetAttrs();
    OPS_CHECK(attrs == nullptr, OPS_LOG_E(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

    auto numTokensPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_NUM_TOKENS_INDEX));
    auto numRanksPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_NUM_RANKS_INDEX));
    auto numExpertsPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_EXPERTS_INDEX);
    auto numTopkPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_NUM_TOPK_INDEX));

    OPS_CHECK(numTokensPtr == nullptr, OPS_LOG_E(nodeName, "numTokensPtr is null."), return ge::GRAPH_FAILED);
    OPS_CHECK(numRanksPtr == nullptr, OPS_LOG_E(nodeName, "numRanksPtr is null."), return ge::GRAPH_FAILED);
    OPS_CHECK(numExpertsPtr == nullptr, OPS_LOG_E(nodeName, "numExpertsPtr is null."), return ge::GRAPH_FAILED);
    OPS_CHECK(numTopkPtr == nullptr, OPS_LOG_E(nodeName, "numTopkPtr is null."), return ge::GRAPH_FAILED);

    OPS_CHECK((*numRanksPtr <= 0) || (*numRanksPtr > MAX_COMM_WORLD_SIZE),
        OPS_LOG_E(nodeName, "rankSize is invalid, only support (0, %ld], but got rankSize=%ld.", MAX_COMM_WORLD_SIZE, *numRanksPtr),
        return ge::GRAPH_FAILED);
    OPS_CHECK((*numExpertsPtr <= 0) || (*numExpertsPtr > MAX_MOE_EXPERTS_NUM),
        OPS_LOG_E(nodeName, "numExperts is invalid, only support (0, %ld], but got numExperts=%ld.", MAX_MOE_EXPERTS_NUM, *numExpertsPtr),
        return ge::GRAPH_FAILED);
    OPS_CHECK((*numTopkPtr <= 0) || (*numTopkPtr > K_MAX),
        OPS_LOG_E(nodeName, "numTopkPtr is invalid, only support (0, %u], but got numTopk=%ld.", K_MAX, *numTopkPtr),
        return ge::GRAPH_FAILED);

    tilingData.dispatchLayoutInfo.numTokens = static_cast<uint32_t>(*numTokensPtr);
    tilingData.dispatchLayoutInfo.numRanks = static_cast<uint32_t>(*numRanksPtr);
    tilingData.dispatchLayoutInfo.numExperts = static_cast<uint32_t>(*numExpertsPtr);
    tilingData.dispatchLayoutInfo.numTopk = static_cast<uint32_t>(*numTopkPtr);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context, const char *nodeName)
{
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OPS_CHECK(workSpaces == nullptr, OPS_LOG_E(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + KERNEL_USE_WORKSPACE + KERNEL_A2_ARG_SIZE;
    return ge::GRAPH_SUCCESS;
}

static bool CheckTensorDataType(gert::TilingContext *context, const char *nodeName)
{
    auto topkIdx = context->GetInputDesc(INPUT_TOPK_IDX_INDEX);
    auto numTokensPerRank = context->GetOutputDesc(OUTPUT_NUM_TOKEN_PER_RANK_INDEX);
    auto numTokensPerExpert = context->GetOutputDesc(OUTPUT_NUM_TOKEN_PER_EXPERT_INDEX);
    auto isTokenInRank = context->GetOutputDesc(OUTPUT_IS_TOKEN_IN_RANK_INDEX);

    OPS_CHECK(topkIdx == nullptr, OPS_LOG_E(nodeName, "topkIdx is null."), return false);
    OPS_CHECK(numTokensPerRank == nullptr, OPS_LOG_E(nodeName, "numTokensPerRank is null."), return false);
    OPS_CHECK(numTokensPerExpert == nullptr, OPS_LOG_E(nodeName, "numTokensPerExpert is null."), return false);
    OPS_CHECK(isTokenInRank == nullptr, OPS_LOG_E(nodeName, "isTokenInRank is null."), return false);

    OPS_CHECK((topkIdx->GetDataType() != ge::DT_INT64),
        OPS_LOG_E(nodeName, "topkIdx datatype is invalid, datatype should be int, but is %d.",
            static_cast<ge::DataType>(topkIdx->GetDataType())), return false);
    OPS_CHECK((numTokensPerRank->GetDataType() != ge::DT_INT32),
        OPS_LOG_E(nodeName, "numTokensPerRank datatype is invalid, datatype should be int, but is %d.",
            static_cast<ge::DataType>(numTokensPerRank->GetDataType())), return false);
    OPS_CHECK((numTokensPerExpert->GetDataType() != ge::DT_INT32),
        OPS_LOG_E(nodeName, "numTokensPerExpert datatype is invalid, datatype should be int, but is %d.",
            static_cast<ge::DataType>(numTokensPerExpert->GetDataType())), return false);
    OPS_CHECK((isTokenInRank->GetDataType() != ge::DT_INT32),
        OPS_LOG_E(nodeName, "isTokenInRank datatype is invalid, datatype should be int, but is %d.",
            static_cast<ge::DataType>(isTokenInRank->GetDataType())), return false);

    return true;
}

static bool CheckTensorShape(gert::TilingContext *context, const char *nodeName)
{
    const gert::StorageShape *topkIdxStorageShape = context->GetInputShape(INPUT_TOPK_IDX_INDEX);
    int64_t topkIdxDim0 = topkIdxStorageShape->GetStorageShape().GetDim(0);
    int64_t topkIdxDim1 = topkIdxStorageShape->GetStorageShape().GetDim(1);
    
    OPS_CHECK((topkIdxStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS),
        OPS_LOG_E(nodeName, "topkIdx must be 2-dimension, but get %lu dim.",
            topkIdxStorageShape->GetStorageShape().GetDimNum()), return false);

    return true;
}

static ge::graphStatus TilingCheckTensor(
    gert::TilingContext *context, const char *nodeName)
{
    OPS_CHECK(!CheckTensorDataType(context, nodeName),
        OPS_LOG_E(nodeName, "params dataType is invalid."),
        return ge::GRAPH_FAILED);

    OPS_CHECK(!CheckTensorShape(context, nodeName),
        OPS_LOG_E(nodeName, "params dataType is invalid."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DispatchLayoutTilingFuncImpl(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    DispatchLayoutTilingData *tilingData = context->GetTilingData<DispatchLayoutTilingData>();
    OPS_CHECK(tilingData == nullptr, OPS_LOG_E(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    OPS_LOG_I(nodeName, "Enter NotifyDispatch tiling check func.");

    OPS_CHECK(GetAttrAndSetTilingData(context, nodeName, *tilingData) != ge::GRAPH_SUCCESS,
        OPS_LOG_E(nodeName, "Get attr and set tiling data failed."),
        return ge::GRAPH_FAILED);

    OPS_CHECK(TilingCheckTensor(context, nodeName) != ge::GRAPH_SUCCESS,
        OPS_LOG_E(nodeName, "Tiling check param failed."),
        return ge::GRAPH_FAILED);

    OPS_CHECK(SetWorkSpace(context, nodeName) != ge::GRAPH_SUCCESS,
        OPS_LOG_E(nodeName, "Tiling set workspace failed."),
        return ge::GRAPH_FAILED);

    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    fe::PlatFormInfos &platformInfo = *platformInfoPtr;

    std::string socVersion;
    (void)platformInfo.GetPlatformResWithLock("version", "Short_SoC_version", socVersion);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t blockDim;
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    blockDim = aivNum;
    context->SetBlockDim(blockDim);
    tilingData->dispatchLayoutInfo.totalUbSize = ubSize;
    OPS_LOG_D(nodeName, "blockDim=%u, aivNum=%u, ubSize=%lu", blockDim, aivNum, ubSize);
    PrintTilingDataInfo(nodeName, *tilingData);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DispatchLayoutTilingFunc(gert::TilingContext *context)
{
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    fe::PlatFormInfos &platformInfo = *platformInfoPtr;

    std::string socVersion;
    ge::graphStatus ret;
    ret = DispatchLayoutTilingFuncImpl(context);
    return ret;
}

struct DispatchLayoutCompileInfo {};
ge::graphStatus TilingParseForDispatchLayout(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DispatchLayout)
    .Tiling(DispatchLayoutTilingFunc)
    .TilingParse<DispatchLayoutCompileInfo>(TilingParseForDispatchLayout);
}  // namespace optiling
