#include <queue>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <string>
#include <type_traits>

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "log/ops_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "../op_kernel/moe_combine_normal_tiling.h"

using namespace AscendC;
using namespace ge;

namespace {
        class Mc2TilingUtils {
    public:
        #define HCCL_BUFFSIZE  "HCCL_BUFFSIZE"
        static uint64_t GetMaxWindowSize()
        {
            uint16_t defaultWindowSize = 200;
            if (getenv(HCCL_BUFFSIZE) == nullptr) {
                OPS_LOG_D("", "Env HCCL_BUFFSIZE don't set");
            } else {
                try {
                    std::string envStr(getenv(HCCL_BUFFSIZE));
                    defaultWindowSize = std::stoi(envStr);
                } catch (...) {
                    OPS_LOG_E("", "Unknown Exception encountered when parser env HCCL_BUFFERSIZE");
                }
            }
            const uint64_t maxWindowSize = static_cast<uint64_t>(defaultWindowSize) * 1024UL * 1024UL;
            OPS_LOG_I("", "Get maxWindowSize is %lu", maxWindowSize);
            return maxWindowSize;
        }
    };
    constexpr uint32_t RECV_X_INDEX = 0;
    constexpr uint32_t TOKEN_SRC_INFO_INDEX = 1;
    constexpr uint32_t EP_RECV_COUNTS_INDEX = 2;
    constexpr uint32_t TOPK_WEIGHTS_INDEX = 3;
    constexpr uint32_t TP_RECV_COUNTS_INDEX = 4;
    constexpr uint32_t OUTPUT_X_INDEX = 0;

    constexpr uint32_t ATTR_GROUP_EP_INDEX = 0;
    constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;
    constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 2;
    constexpr uint32_t ATTR_GROUP_TP_INDEX = 3;
    constexpr uint32_t ATTR_TP_WORLD_SIZE_INDEX = 4;
    constexpr uint32_t ATTR_TP_RANK_ID_INDEX = 5;
    constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 6;
    constexpr uint32_t ATTR_GLOBAL_BS_INDEX = 7;

    constexpr uint32_t TWO_DIMS = 2U;
    constexpr uint32_t ONE_DIM = 1U;
    constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U; // numeric representation of AlltoAll
    constexpr uint32_t OP_TYPE_REDUCE_SCATTER = 7U; // numeric representation of ReduceScatter

    constexpr size_t  MAX_GROUP_NAME_LENGTH = 128UL;
    constexpr int64_t MAX_EP_WORLD_SIZE = 384;
    constexpr int64_t MIN_EP_WORLD_SIZE = 2;
    constexpr int64_t MAX_TP_WORLD_SIZE = 2;
    constexpr int64_t BS_UPPER_BOUND = 8000;

    constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
    constexpr int32_t HCCL_BUFFER_SIZE_DEFAULT = 200 * 1024 * 1024; // Bytes
    constexpr int64_t MOE_EXPERT_MAX_NUM = 512;
    constexpr int64_t K_MAX = 16;
    constexpr int64_t H_MIN = 1024;
    constexpr int64_t H_MAX = 7168;
    constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
    constexpr uint64_t TRIPLE = 3;
    constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
    constexpr uint64_t SCALE_RECV_IDX_BUFFER = 44UL; // scale32B + 3*4 src info
    constexpr uint64_t COMBINE_STATE_WIN_OFFSET = 3U * 1024UL * 1024UL;
    constexpr uint64_t DOUBLE_DATA_BUFFER = 2UL;
    constexpr uint64_t MAX_OUT_DTYPE_SIZE = 2UL;
    constexpr uint64_t UB_ALIGN = 32UL;
    constexpr int64_t DISPATCH_STATUS_MAX_SUPPORT_NUM = 1280UL;

    enum class CommQuantMode : int32_t {
        NON_QUANT = 0,
        INT12_QUANT = 1,
        INT8_QUANT = 2
    };
    using CommQuantModeType = std::underlying_type<CommQuantMode>;
}

namespace optiling {

// Specific to A3
static void PrintTilingDataInfo(const char *nodeName, MoeCombineNormalTilingData& tilingData)
{
    OPS_LOG_D(nodeName, "epWorldSize is %u.", tilingData.moeCombineNormalInfo.epWorldSize);
    OPS_LOG_D(nodeName, "tpWorldSize is %u.", tilingData.moeCombineNormalInfo.tpWorldSize);
    OPS_LOG_D(nodeName, "epRankId is %u.", tilingData.moeCombineNormalInfo.epRankId);
    OPS_LOG_D(nodeName, "tpRankId is %u.", tilingData.moeCombineNormalInfo.tpRankId);
    OPS_LOG_D(nodeName, "expertShardType is %u.", tilingData.moeCombineNormalInfo.expertShardType);
    OPS_LOG_D(nodeName, "moeExpertNum is %u.", tilingData.moeCombineNormalInfo.moeExpertNum);
    OPS_LOG_D(nodeName, "moeExpertPerRankNum is %u.", tilingData.moeCombineNormalInfo.moeExpertPerRankNum);
    OPS_LOG_D(nodeName, "globalBs is %u.", tilingData.moeCombineNormalInfo.globalBs);
    OPS_LOG_D(nodeName, "bs is %u.", tilingData.moeCombineNormalInfo.bs);
    OPS_LOG_D(nodeName, "k is %u.", tilingData.moeCombineNormalInfo.k);
    OPS_LOG_D(nodeName, "h is %u.", tilingData.moeCombineNormalInfo.h);
    OPS_LOG_D(nodeName, "aivNum is %u.", tilingData.moeCombineNormalInfo.aivNum);
    OPS_LOG_D(nodeName, "totalUbSize is %lu.", tilingData.moeCombineNormalInfo.totalUbSize);
    OPS_LOG_D(nodeName, "totalWinSize is %lu.", tilingData.moeCombineNormalInfo.totalWinSize);
}

static ge::graphStatus GetAttrAndSetTilingData(gert::TilingContext *context, MoeCombineNormalTilingData &tilingData,
    const char *nodeName, std::string &groupEp, std::string &groupTp)
{
    auto attrs = context->GetAttrs();
    OPS_CHECK(attrs == nullptr, OPS_LOG_E(nodeName, "attrs is null."), return ge::GRAPH_FAILED);

    auto groupEpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    auto groupTpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_TP_INDEX));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto tpRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_RANK_ID_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);

    // Check for null
    OPS_CHECK((groupEpPtr == nullptr) || (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
        (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH), OPS_LOG_E(nodeName, "groupEp is invalid."),
        return ge::GRAPH_FAILED);
    OPS_CHECK(epWorldSizePtr == nullptr, OPS_LOG_E(nodeName, "epWorldSize is null."), return ge::GRAPH_FAILED);
    OPS_CHECK(tpWorldSizePtr == nullptr, OPS_LOG_E(nodeName, "tpWorldSize is null."), return ge::GRAPH_FAILED);
    OPS_CHECK(epRankIdPtr == nullptr, OPS_LOG_E(nodeName, "epRankId is null."), return ge::GRAPH_FAILED);
    OPS_CHECK(tpRankIdPtr == nullptr, OPS_LOG_E(nodeName, "tpRankId is null."), return ge::GRAPH_FAILED);
    OPS_CHECK(moeExpertNumPtr == nullptr, OPS_LOG_E(nodeName, "moeExpertNum is null."), return ge::GRAPH_FAILED);

    // Check if it meets uint32_t and other constraints
    int64_t moeExpertNum = *moeExpertNumPtr;
    int64_t epWorldSize = *epWorldSizePtr;
    OPS_CHECK((epWorldSize < MIN_EP_WORLD_SIZE) || (epWorldSize > MAX_EP_WORLD_SIZE),
        OPS_LOG_E(nodeName, "epWorldSize is invalid, only support [%ld, %ld], but got epWorldSize=%ld.",
        MIN_EP_WORLD_SIZE, MAX_EP_WORLD_SIZE, epWorldSize), return ge::GRAPH_FAILED);
    OPS_CHECK((*tpWorldSizePtr < 0) || (*tpWorldSizePtr > MAX_TP_WORLD_SIZE),
        OPS_LOG_E(nodeName, "tpWorldSize is invalid, only support [0, %ld], but got tpWorldSize=%ld.",
        MAX_TP_WORLD_SIZE, *tpWorldSizePtr), return ge::GRAPH_FAILED);
    OPS_CHECK((*epRankIdPtr < 0) || (*epRankIdPtr >= epWorldSize),
        OPS_LOG_E(nodeName, "epRankId is invalid, only support [0, %ld), but got epRankId=%ld.",
        epWorldSize, *epRankIdPtr), return ge::GRAPH_FAILED);

    if (*tpWorldSizePtr > 1) {
        OPS_CHECK((*tpRankIdPtr < 0) || (*tpRankIdPtr >= *tpWorldSizePtr),
            OPS_LOG_E(nodeName, "tpRankId is invalid, only support [0, %ld), but got tpRankId=%ld.",
            *tpWorldSizePtr, *tpRankIdPtr), return ge::GRAPH_FAILED);
        OPS_CHECK((groupTpPtr == nullptr) || (strnlen(groupTpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
            (strnlen(groupTpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
            OPS_LOG_E(nodeName, "groupTpPtr is null."), return ge::GRAPH_FAILED);
        groupTp = std::string(groupTpPtr);
    } else {
        OPS_CHECK(*tpRankIdPtr != 0,
            OPS_LOG_E(nodeName, "tpRankId is invalid, NoTp mode only support 0, but got tpRankId=%ld.", *tpRankIdPtr),
            return ge::GRAPH_FAILED);
    }
    OPS_CHECK((moeExpertNum <= 0) || (moeExpertNum > MOE_EXPERT_MAX_NUM),
        OPS_LOG_E(nodeName, "moeExpertNum is invalid, only support (0, %ld], but got moeExpertNum=%ld.",
        MOE_EXPERT_MAX_NUM, moeExpertNum), return ge::GRAPH_FAILED);
    int64_t moePerRankNum = moeExpertNum / epWorldSize;
    int64_t curDispatchStatusNum = moePerRankNum * epWorldSize;
    OPS_CHECK((curDispatchStatusNum > DISPATCH_STATUS_MAX_SUPPORT_NUM),
        OPS_LOG_E(nodeName, "The moe experts num must meet the conditions,"
        " (moeExpertNum / epWorldSize) * epWorldSize <= 1280, but cur is %ld.",
        curDispatchStatusNum), return ge::GRAPH_FAILED);

    groupEp = std::string(groupEpPtr);
    tilingData.moeCombineNormalInfo.epWorldSize = static_cast<uint32_t>(epWorldSize);
    tilingData.moeCombineNormalInfo.tpWorldSize = static_cast<uint32_t>(*tpWorldSizePtr);
    tilingData.moeCombineNormalInfo.epRankId = static_cast<uint32_t>(*epRankIdPtr);
    tilingData.moeCombineNormalInfo.tpRankId = static_cast<uint32_t>(*tpRankIdPtr);
    tilingData.moeCombineNormalInfo.moeExpertNum = static_cast<uint32_t>(moeExpertNum);

    return ge::GRAPH_SUCCESS;
}

static bool CheckInputTensorDim(gert::TilingContext *context, const char *nodeName)
{
    const gert::StorageShape *recvXStorageShape = context->GetInputShape(RECV_X_INDEX);
    OPS_CHECK(recvXStorageShape == nullptr, OPS_LOG_E(nodeName, "recvX is null."), return false);
    OPS_CHECK(recvXStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OPS_LOG_E(nodeName, "recvX must be 2-dimension, but got %lu dim",
        recvXStorageShape->GetStorageShape().GetDimNum()), return false);
    OPS_LOG_D(nodeName, "recvX dim0 = %ld", recvXStorageShape->GetStorageShape().GetDim(0));
    OPS_LOG_D(nodeName, "recvX dim1 = %ld", recvXStorageShape->GetStorageShape().GetDim(1));

    const gert::StorageShape *tokenSrcInfoStorageShape = context->GetInputShape(TOKEN_SRC_INFO_INDEX);
    OPS_CHECK(tokenSrcInfoStorageShape == nullptr, OPS_LOG_E(nodeName, "tokenSrcInfoForCombine is null."), return false);
    OPS_CHECK(tokenSrcInfoStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OPS_LOG_E(nodeName, "tokenSrcInfoForCombine must be 1-dimension, but got %lu dim",
        tokenSrcInfoStorageShape->GetStorageShape().GetDimNum()), return false);
    OPS_LOG_D(nodeName, "tokenSrcInfoForCombine dim0 = %ld", tokenSrcInfoStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *topkWeightsStorageShape = context->GetInputShape(TOPK_WEIGHTS_INDEX);
    OPS_CHECK(topkWeightsStorageShape == nullptr, OPS_LOG_E(nodeName, "topkWeights is null."), return false);
    OPS_CHECK(topkWeightsStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OPS_LOG_E(nodeName, "topkWeights must be 2-dimension, but got %lu dim",
        topkWeightsStorageShape->GetStorageShape().GetDimNum()), return false);
    OPS_LOG_D(nodeName, "topkWeights dim0 = %ld", topkWeightsStorageShape->GetStorageShape().GetDim(0));
    OPS_LOG_D(nodeName, "topkWeights dim1 = %ld", topkWeightsStorageShape->GetStorageShape().GetDim(1));

    return true;
}

static bool CheckOptionalInputTensorDim(gert::TilingContext *context, const char *nodeName)
{
    const gert::StorageShape *tpRecvCountsStorageShape = context->GetOptionalInputShape(TP_RECV_COUNTS_INDEX);
    OPS_CHECK(tpRecvCountsStorageShape == nullptr, OPS_LOG_E(nodeName, "tpRecvCounts is null."), return false);
    OPS_CHECK(tpRecvCountsStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OPS_LOG_E(nodeName, "tpRecvCounts must be 1-dimension, but got %lu dim",
        tpRecvCountsStorageShape->GetStorageShape().GetDimNum()), return false);
    OPS_LOG_D(nodeName, "tpRecvCounts dim0 = %ld", tpRecvCountsStorageShape->GetStorageShape().GetDim(0));

    return true;
}

static bool CheckOutputTensorDim(gert::TilingContext *context, const char *nodeName)
{
    const gert::StorageShape *xStorageShape = context->GetOutputShape(OUTPUT_X_INDEX);
    OPS_CHECK(xStorageShape == nullptr, OPS_LOG_E(nodeName, "x is null."), return false);
    OPS_CHECK(xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OPS_LOG_E(nodeName, "x must be 2-dimension, but got %lu dim", xStorageShape->GetStorageShape().GetDimNum()),
        return false);
    OPS_LOG_D(nodeName, "x dim0 = %ld", xStorageShape->GetStorageShape().GetDim(0));
    OPS_LOG_D(nodeName, "x dim1 = %ld", xStorageShape->GetStorageShape().GetDim(1));

    return true;
}

static bool CheckTensorDim(gert::TilingContext *context, const char *nodeName)
{
    OPS_CHECK(!CheckInputTensorDim(context, nodeName),
        OPS_LOG_E(nodeName, "param shape of input tensor is invalid"), return false);

    OPS_CHECK(!CheckOptionalInputTensorDim(context, nodeName),
        OPS_LOG_E(nodeName, "param shape of optional input tensor is invalid"), return false);

    OPS_CHECK(!CheckOutputTensorDim(context, nodeName),
        OPS_LOG_E(nodeName, "param shape of output tensor is invalid"), return false);

    return true;
}

// Validate data type
static bool CheckTensorDataType(gert::TilingContext *context, const char *nodeName)
{
    auto recvXDesc = context->GetInputDesc(RECV_X_INDEX);
    OPS_CHECK(recvXDesc == nullptr, OPS_LOG_E(nodeName, "recvXDesc is null."), return false);
    OPS_CHECK((recvXDesc->GetDataType() != ge::DT_BF16) && (recvXDesc->GetDataType() != ge::DT_FLOAT16),
        OPS_LOG_E(nodeName, "recvX dataType is invalid, dataType should be bf16 or float16, but is "
        ), return false);
    auto tokenSrcInfoDesc = context->GetInputDesc(TOKEN_SRC_INFO_INDEX);
    OPS_CHECK(tokenSrcInfoDesc == nullptr, OPS_LOG_E(nodeName, "tokenSrcInfoDesc is null."), return false);
    OPS_CHECK((tokenSrcInfoDesc->GetDataType() != ge::DT_INT32), OPS_LOG_E(nodeName, "tokenSrcInfoForCombine dataType is invalid,"
        " dataType should be int32, but is"), return false);
    auto tpRecvCountsDesc = context->GetOptionalInputDesc(TP_RECV_COUNTS_INDEX);
    OPS_CHECK(tpRecvCountsDesc == nullptr, OPS_LOG_E(nodeName, "tpRecvCountsDesc is null."), return false);
    OPS_CHECK((tpRecvCountsDesc->GetDataType() != ge::DT_INT32),
        OPS_LOG_E(nodeName, "tpRecvCounts dataType is invalid, dataType should be int32, but is "), return false);
    auto topkWeightsDesc = context->GetInputDesc(TOPK_WEIGHTS_INDEX);
    OPS_CHECK(topkWeightsDesc == nullptr, OPS_LOG_E(nodeName, "topkWeightsDesc is null."), return false);
    OPS_CHECK((topkWeightsDesc->GetDataType() != ge::DT_FLOAT),
        OPS_LOG_E(nodeName, "topkWeights dataType is invalid, dataType should be float, but is "),
         return false);
    auto xDesc = context->GetOutputDesc(OUTPUT_X_INDEX);
    OPS_CHECK(xDesc == nullptr, OPS_LOG_E(nodeName, "xDesc is null."), return false);
    OPS_CHECK((xDesc->GetDataType() != recvXDesc->GetDataType()), OPS_LOG_E(nodeName,
        "x dataType is invalid, dataType should be equal to recvX dataType , but is "),
        return false);
    return true;
}

static bool CheckTensorFormat(gert::TilingContext *context, const char *nodeName)
{
    auto recvXDesc = context->GetInputDesc(RECV_X_INDEX);
    OPS_CHECK(recvXDesc == nullptr, OPS_LOG_E(nodeName, "recvXDesc is null."), return false);
    OPS_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(recvXDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OPS_LOG_E(nodeName, "recvXFormat is invalid"), return false);

    auto tokenSrcInfoDesc = context->GetInputDesc(TOKEN_SRC_INFO_INDEX);
    OPS_CHECK(tokenSrcInfoDesc == nullptr, OPS_LOG_E(nodeName, "tokenSrcInfoDesc is null."), return false);
    OPS_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(tokenSrcInfoDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OPS_LOG_E(nodeName, "tokenSrcInfoFormat is invalid"), return false);

    auto tpRecvCountsDesc = context->GetOptionalInputDesc(TP_RECV_COUNTS_INDEX);
    OPS_CHECK(tpRecvCountsDesc == nullptr, OPS_LOG_E(nodeName, "tpRecvCountsDesc is null."), return false);
    OPS_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(tpRecvCountsDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OPS_LOG_E(nodeName, "tpRecvCountsFormat is invalid"), return false);

    auto topkWeightsDesc = context->GetInputDesc(TOPK_WEIGHTS_INDEX);
    OPS_CHECK(topkWeightsDesc == nullptr, OPS_LOG_E(nodeName, "topkWeightsDesc is null."), return false);
    OPS_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(topkWeightsDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OPS_LOG_E(nodeName, "topkWeightsFormat is invalid"), return false);

    auto xDesc = context->GetOutputDesc(OUTPUT_X_INDEX);
    OPS_CHECK(xDesc == nullptr, OPS_LOG_E(nodeName, "xDesc is null."), return false);
    OPS_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
                    OPS_LOG_E(nodeName, "xFormat is invalid"), return false);

    return true;
}

static bool CheckTensorShape(gert::TilingContext *context, MoeCombineNormalTilingData &tilingData,
    const char *nodeName, uint32_t localExpertNum)
{
    const gert::StorageShape *topkWeightsStorageShape = context->GetInputShape(TOPK_WEIGHTS_INDEX);
    int64_t topkWeightsDim0 = topkWeightsStorageShape->GetStorageShape().GetDim(0);
    int64_t topkWeightsDim1 = topkWeightsStorageShape->GetStorageShape().GetDim(1);
    int64_t moeExpertNum = static_cast<int64_t>(tilingData.moeCombineNormalInfo.moeExpertNum);
    OPS_CHECK((topkWeightsDim1 <= 0) || (topkWeightsDim1 > K_MAX || (topkWeightsDim1 > moeExpertNum)),
        OPS_LOG_E(nodeName, "topkWeights's dim1(K) should be in (0, min(%ld, moeExpertNum %ld)], "
        "but got topkWeights's dim1=%ld.", K_MAX, moeExpertNum, topkWeightsDim1), return false);
    tilingData.moeCombineNormalInfo.k = static_cast<uint32_t>(topkWeightsDim1);

    // Validate recvX dimensions and set h
    int64_t tpWorldSize = static_cast<int64_t>(tilingData.moeCombineNormalInfo.tpWorldSize);
    const gert::StorageShape *recvXStorageShape = context->GetInputShape(RECV_X_INDEX);
    int64_t recvXDim1 = recvXStorageShape->GetStorageShape().GetDim(1);
    OPS_CHECK((recvXDim1 < H_MIN) || (recvXDim1 > H_MAX),
        OPS_LOG_E(nodeName, "recvX's dim1(H) should be in [%ld, %ld], but got %ld.",
        H_MIN, H_MAX, recvXDim1), return false); // 32-byte aligned
    tilingData.moeCombineNormalInfo.h = static_cast<uint32_t>(recvXDim1);

    // Validate epRecvCount and tpRecvCount dimensions
    int64_t epWorldSize = static_cast<int64_t>(tilingData.moeCombineNormalInfo.epWorldSize);
    int64_t moeExpertPerRankNum = static_cast<int64_t>(tilingData.moeCombineNormalInfo.moeExpertPerRankNum);

    // Validate x dimensions
    const gert::StorageShape *xStorageShape = context->GetOutputShape(OUTPUT_X_INDEX);
    int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OPS_CHECK(xDim0 != topkWeightsDim0, OPS_LOG_E(nodeName,
        "x's dim0 not equal to bs, bs = %ld, x's dim0 = %ld", topkWeightsDim0, xDim0), return false);
    OPS_CHECK(xDim1 != recvXDim1, OPS_LOG_E(nodeName,
        "x's dim1 not equal to h, x's dim1 = %ld, h = %ld", xDim1, recvXDim1), return false);

    return true;
}

static bool CheckAttrs(gert::TilingContext *context, MoeCombineNormalTilingData &tilingData,
    const char *nodeName, uint32_t &localMoeExpertNum)
{
    uint32_t epWorldSize = tilingData.moeCombineNormalInfo.epWorldSize;
    uint32_t tpWorldSize = tilingData.moeCombineNormalInfo.tpWorldSize;
    uint32_t moeExpertNum = tilingData.moeCombineNormalInfo.moeExpertNum;

    // Validate if moe expert number can be evenly distributed across multiple machines
    OPS_CHECK(moeExpertNum % epWorldSize != 0,
        OPS_LOG_E(nodeName, "moeExpertNum should be divisible by epWorldSize, "
        "but got moeExpertNum=%d, epWorldSize=%d.", moeExpertNum, epWorldSize), return false);
    localMoeExpertNum = moeExpertNum / epWorldSize;
    OPS_CHECK(localMoeExpertNum <= 0,
        OPS_LOG_E(nodeName, "localMoeExpertNum is invalid, localMoeExpertNum = %d", localMoeExpertNum), return false);
    // Validate if expert number per card equals 1 when tp=2
    OPS_CHECK((localMoeExpertNum > 1) && (tpWorldSize > 1),
        OPS_LOG_E(nodeName, "Cannot support multi-moeExpert %d in a rank when tpWorldSize = %d > 1",
        localMoeExpertNum, tpWorldSize), return false);
    tilingData.moeCombineNormalInfo.moeExpertPerRankNum = localMoeExpertNum;

    // Validate topkWeights dimension 0 and set bs
    const gert::StorageShape *topkWeightsStorageShape = context->GetInputShape(TOPK_WEIGHTS_INDEX);
    int64_t topkWeightsDim0 = topkWeightsStorageShape->GetStorageShape().GetDim(0);
    OPS_CHECK((topkWeightsDim0 <= 0) || (topkWeightsDim0 > BS_UPPER_BOUND),
        OPS_LOG_E(nodeName, "Invalid topkWeights dims0(BS) %ld. Should be between [1, %ld].",
        topkWeightsDim0, BS_UPPER_BOUND), return false);
    tilingData.moeCombineNormalInfo.bs = static_cast<uint32_t>(topkWeightsDim0);

    // Validate globalBS
    auto attrs = context->GetAttrs();
    OPS_CHECK(attrs == nullptr, OPS_LOG_E(nodeName, "attrs is null."), return false);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_GLOBAL_BS_INDEX);
    OPS_CHECK(globalBsPtr == nullptr, OPS_LOG_E(nodeName, "globalBs is null."), return false);
    OPS_LOG_D(nodeName, "MoeCombineNormal *globalBsPtr = %ld, bs = %ld, epWorldSize = %u\n",
        *globalBsPtr, topkWeightsDim0, epWorldSize);

    OPS_CHECK((*globalBsPtr != 0) && ((*globalBsPtr < static_cast<int64_t>(epWorldSize) * topkWeightsDim0) ||
        ((*globalBsPtr) % (static_cast<int64_t>(epWorldSize)) != 0)), OPS_LOG_E(nodeName, "globalBS is invalid, only "
        "support 0 or maxBs(maxBs is the largest bs on all ranks) * epWorldSize, but got globalBS=%ld, "
        "bs=%ld, epWorldSize=%u.", *globalBsPtr, topkWeightsDim0, epWorldSize),  return false);

    tilingData.moeCombineNormalInfo.globalBs = static_cast<uint32_t>(*globalBsPtr);
    if (*globalBsPtr == 0) {
        tilingData.moeCombineNormalInfo.globalBs = static_cast<uint32_t>(topkWeightsDim0) * epWorldSize;
    }

    return true;
}

static ge::graphStatus TilingCheckMoeCombineNormal(gert::TilingContext *context, const char *nodeName)
{
    // Check parameter shape information
    OPS_CHECK(!CheckTensorDim(context, nodeName),
                    OPS_LOG_E(nodeName, "param shape is invalid"), return ge::GRAPH_FAILED);
    // Check parameter dataType information
    OPS_CHECK(!CheckTensorDataType(context, nodeName),
                    OPS_LOG_E(nodeName, "param dataType is invalid"), return ge::GRAPH_FAILED);
    // Check parameter format information
    OPS_CHECK(!CheckTensorFormat(context, nodeName),
                    OPS_LOG_E(nodeName, "param Format is invalid"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspace(gert::TilingContext *context, const char *nodeName)
{
    size_t *workspace = context->GetWorkspaceSizes(1);
    OPS_CHECK(workspace == nullptr, OPS_LOG_E(nodeName, "get workspace failed"),
        return ge::GRAPH_FAILED);
    workspace[0] = SYSTEM_NEED_WORKSPACE;
    OPS_LOG_D(nodeName, "workspace[0] size is %ld", workspace[0]);
    return ge::GRAPH_SUCCESS;
}


static void SetHCommCfg(gert::TilingContext *context, MoeCombineNormalTilingData *tiling,
    const std::string groupEp, const std::string groupTp)
{
    const char* nodeName = context->GetNodeName();
    OPS_LOG_D(nodeName, "MoeCombineNormal groupEp = %s, groupTp = %s", groupEp.c_str(), groupTp.c_str());
    uint32_t opType1 = OP_TYPE_ALL_TO_ALL;
    uint32_t opType2 = OP_TYPE_REDUCE_SCATTER;
    std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh;level1:pairwise";
    std::string algConfigReduceScatterStr = "ReduceScatter=level0:ring";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp, opType1, algConfigAllToAllStr);
    mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling1);

    mc2CcTilingConfig.SetGroupName(groupTp);
    mc2CcTilingConfig.SetOpType(opType2);
    mc2CcTilingConfig.SetAlgConfig(algConfigReduceScatterStr);
    mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling2);
}

static ge::graphStatus MoeCombineNormalA3TilingFuncImpl(gert::TilingContext* context)
{
    const char *nodeName = context->GetNodeName();
    OPS_LOG_D(nodeName, "Enter MoeCombineNormal Tiling func");
    MoeCombineNormalTilingData *tilingData = context->GetTilingData<MoeCombineNormalTilingData>();
    OPS_CHECK(tilingData == nullptr, OPS_LOG_E(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    std::string groupEp = "";
    std::string groupTp = "";
    uint32_t localMoeExpertNum = 1;

    // Get input parameter attributes
    OPS_CHECK(GetAttrAndSetTilingData(context, *tilingData, nodeName, groupEp, groupTp) == ge::GRAPH_FAILED,
        OPS_LOG_E(nodeName, "Getting attr failed."), return ge::GRAPH_FAILED);

    // Check input/output dim, format, dataType
    OPS_CHECK(TilingCheckMoeCombineNormal(context, nodeName) != ge::GRAPH_SUCCESS,
                    OPS_LOG_E(nodeName, "Tiling check params failed"), return ge::GRAPH_FAILED);

    // Check if attribute values are valid
    OPS_CHECK(!CheckAttrs(context, *tilingData, nodeName, localMoeExpertNum),
        OPS_LOG_E(nodeName, "attr check failed."), return ge::GRAPH_FAILED);

    uint32_t epRankId = tilingData->moeCombineNormalInfo.epRankId;

    // Check shape dimensions and assign h, k
    OPS_CHECK(!CheckTensorShape(context, *tilingData, nodeName, localMoeExpertNum),
        OPS_LOG_E(nodeName, "param dim check failed."), return ge::GRAPH_FAILED);

    // Validate win area size
    uint64_t maxWindowSize = Mc2TilingUtils::GetMaxWindowSize();
    uint64_t h = static_cast<uint64_t>(tilingData->moeCombineNormalInfo.h);
    uint64_t epWorldSize = static_cast<uint64_t>(tilingData->moeCombineNormalInfo.epWorldSize);
    uint64_t k = static_cast<uint64_t>(tilingData->moeCombineNormalInfo.k);
    uint64_t maxBs = static_cast<uint64_t>(tilingData->moeCombineNormalInfo.globalBs)/ epWorldSize;
    // Combine data area: token start address aligned to 512
    uint64_t tokenNeedSizeCombine = ((h * MAX_OUT_DTYPE_SIZE  + WIN_ADDR_ALIGN - 1UL) / WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    // Dispatch data area: token start aligned to 512, valid token length h_align_32b + scale(32b) + triplet(3*4b)
    uint64_t tokenActualLen = ((h * MAX_OUT_DTYPE_SIZE  + UB_ALIGN - 1UL) / UB_ALIGN) * UB_ALIGN + SCALE_RECV_IDX_BUFFER;
    uint64_t tokenNeedSizeDispatch = ((tokenActualLen + WIN_ADDR_ALIGN - 1UL) / WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    uint64_t actualSize = (maxBs * k * (tokenNeedSizeCombine + tokenNeedSizeDispatch) + COMBINE_STATE_WIN_OFFSET) * 
                           DOUBLE_DATA_BUFFER;
    OPS_CHECK((actualSize > maxWindowSize),
        OPS_LOG_E(nodeName, "HCCL_BUFFSIZE is too SMALL, maxBs = %lu, h = %lu, epWorldSize = %lu, localMoeExpertNum = %u,"
            " tokenNeedSizeDispatch = %lu, tokenNeedSizeCombine = %lu, k = %lu, NEEDED_HCCL_BUFFSIZE("
            "((maxBs * tokenNeedSizeDispatch) + (maxBs * tokenNeedSizeCombine * k) + 3MB) * 2) = %luMB, HCCL_BUFFSIZE=%luMB.",
            maxBs, h, epWorldSize, localMoeExpertNum, tokenNeedSizeDispatch, tokenNeedSizeCombine, k,
            actualSize / MB_SIZE + 1UL, maxWindowSize / MB_SIZE),
        return ge::GRAPH_FAILED);
    tilingData->moeCombineNormalInfo.totalWinSize = maxWindowSize;

    OPS_CHECK(SetWorkspace(context, nodeName) != ge::GRAPH_SUCCESS,
                    OPS_LOG_E(context->GetNodeName(), "Tiling set workspace Failed"),
                    return ge::GRAPH_FAILED);

    SetHCommCfg(context, tilingData, groupEp, groupTp);

    uint64_t tpWorldSize = static_cast<uint64_t>(tilingData->moeCombineNormalInfo.tpWorldSize);

    uint32_t blockDim = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(blockDim);
    tilingData->moeCombineNormalInfo.aivNum = aivNum;
    tilingData->moeCombineNormalInfo.totalUbSize = ubSize;
    context->SetScheduleMode(1); // Set to batch mode, all cores start simultaneously
    OPS_LOG_D(nodeName, "blockdim = %u, aivNum = %lu, ubsize = %lu", blockDim, aivNum, ubSize);
    PrintTilingDataInfo(nodeName, *tilingData);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeCombineNormalTilingFunc(gert::TilingContext* context)
{
    // recvX data type int32 is not supported
    auto recvXDesc = context->GetInputDesc(RECV_X_INDEX);
    const char *nodeName = context->GetNodeName();
    OPS_CHECK(recvXDesc == nullptr, OPS_LOG_E(nodeName, "recvXDesc is null."), return ge::GRAPH_FAILED);
    // Check if recvX data type is DT_INT32
    OPS_CHECK((recvXDesc->GetDataType() == ge::DT_INT32),
                    OPS_LOG_E(nodeName, "recvX dataType is invalid, dataType should be bf16 or float16, but is "),
                     return ge::GRAPH_FAILED);

    ge::graphStatus ret = MoeCombineNormalA3TilingFuncImpl(context);
    return ret;
}

struct MoeCombineNormalCompileInfo {};
ge::graphStatus TilingParseForMoeCombineNormal(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeCombineNormal)
    .Tiling(MoeCombineNormalTilingFunc)
    .TilingParse<MoeCombineNormalCompileInfo>(TilingParseForMoeCombineNormal);
} // namespace optiling
