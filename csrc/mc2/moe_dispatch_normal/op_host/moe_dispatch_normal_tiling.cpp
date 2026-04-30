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

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "log/ops_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "../op_kernel/moe_dispatch_normal_tiling.h"

using namespace AscendC;
using namespace ge;
namespace {
class Mc2TilingUtils {
public:
#define HCCL_BUFFSIZE "HCCL_BUFFSIZE"
    static uint64_t GetMaxWindowSize()
    {
        uint16_t defaultWindowSize = 200;
        if (getenv(HCCL_BUFFSIZE) == nullptr) {
            OPS_LOG_D("", "Env HCCL_BUFFSIZE don't set");
        } else {
            try {
                std::string envStr(getenv(HCCL_BUFFSIZE));
                defaultWindowSize = std::stoi(envStr);
            } catch (const std::invalid_argument &ia) {
                OPS_LOG_E("", "Invalid argument when parsing HCCL_BUFFSIZE: %s", ia.what());
            } catch (const std::out_of_range &oor) {
                OPS_LOG_E("", "Out of range when parsing HCCL_BUFFSIZE: %s", oor.what());
            }
        }
        const uint64_t maxWindowSize = static_cast<uint64_t>(defaultWindowSize) * 1024UL * 1024UL;
        OPS_LOG_I("", "Get maxWindowSize is %lu", maxWindowSize);
        return maxWindowSize;
    }
};
constexpr uint32_t X_INDEX = 0U;
constexpr uint32_t EXPERT_IDS_INDEX = 1U;
constexpr uint32_t SEND_OFFSET_INDEX = 2U;
constexpr uint32_t SEND_TOKENIDX_INDEX = 3U;
constexpr uint32_t RECV_OFFSET_INDEX = 4U;
constexpr uint32_t RECV_COUNT_INDEX = 5U;

constexpr uint32_t OUTPUT_EXPAND_X_INDEX = 0U;
constexpr uint32_t OUTPUT_DYNAMIC_SCALES_INDEX = 1U;
constexpr uint32_t OUTPUT_ASSIST_INFO_INDEX = 2U;

constexpr uint32_t ATTR_GROUP_EP_INDEX = 0;
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 2;
constexpr uint32_t ATTR_GROUP_TP_INDEX = 3;
constexpr uint32_t ATTR_TP_WORLD_SIZE_INDEX = 4;
constexpr uint32_t ATTR_TP_RANK_ID_INDEX = 5;
constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 6;
constexpr uint32_t ATTR_QUANT_MODE_INDEX = 7;
constexpr uint32_t ATTR_GLOBAL_BS_INDEX = 8;

constexpr uint32_t TWO_DIMS = 2;
constexpr uint32_t ONE_DIM = 1;
constexpr uint32_t DYNAMIC_SCALE_DIM_NUM = 1;
constexpr uint64_t INIT_TILINGKEY = 10000;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;
constexpr uint32_t NO_SCALES = 0;
constexpr uint32_t DYNAMIC_SCALES = 2;
constexpr uint32_t OP_TYPE_ALL_GATHER = 6;

constexpr size_t MAX_GROUP_NAME_LENGTH = 128UL;
constexpr int64_t MAX_EP_WORLD_SIZE = 384;
constexpr int64_t MIN_EP_WORLD_SIZE = 2;
constexpr int64_t MAX_TP_WORLD_SIZE = 2;
constexpr int64_t BS_UPPER_BOUND = 8000;  // Maximum bs

constexpr uint32_t TILINGKEY_TP_WORLD_SIZE = 100;
constexpr uint32_t TP_WORLD_SIZE_TWO = 2;
constexpr int64_t MOE_EXPERT_MAX_NUM = 512;
constexpr int64_t K_MAX = 16;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
constexpr uint32_t WORKSPACE_ELEMENT_OFFSET = 512;
constexpr int64_t H_MIN = 1024;
constexpr int64_t H_MAX = 7168;
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
constexpr uint64_t TRIPLE = 3;
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint64_t SCALE_EXPAND_IDX_BUFFER = 44UL;  // scale32B + 3*4expandIdx
constexpr uint64_t DOUBLE_DATA_BUFFER = 2UL;
constexpr uint64_t MAX_OUT_DTYPE_SIZE = 2UL;
constexpr uint64_t UB_ALIGN = 32UL;
constexpr int64_t DISPATCH_STATUS_MAX_SUPPORT_NUM = 1280UL;
}  // namespace

namespace optiling {
static void PrintTilingDataInfo(const char *nodeName, MoeDispatchNormalTilingData &tilingData)
{
    OPS_LOG_D(nodeName, "epWorldSize is %u.", tilingData.moeDispatchNormalInfo.epWorldSize);
    OPS_LOG_D(nodeName, "tpWorldSize is %u.", tilingData.moeDispatchNormalInfo.tpWorldSize);
    OPS_LOG_D(nodeName, "epRankId is %u.", tilingData.moeDispatchNormalInfo.epRankId);
    OPS_LOG_D(nodeName, "tpRankId is %u.", tilingData.moeDispatchNormalInfo.tpRankId);
    OPS_LOG_D(nodeName, "moeExpertNum is %u.", tilingData.moeDispatchNormalInfo.moeExpertNum);
    OPS_LOG_D(nodeName, "quantMode is %u.", tilingData.moeDispatchNormalInfo.quantMode);
    OPS_LOG_D(nodeName, "globalBs is %u.", tilingData.moeDispatchNormalInfo.globalBs);
    OPS_LOG_D(nodeName, "bs is %u.", tilingData.moeDispatchNormalInfo.bs);
    OPS_LOG_D(nodeName, "k is %u.", tilingData.moeDispatchNormalInfo.k);
    OPS_LOG_D(nodeName, "h is %u.", tilingData.moeDispatchNormalInfo.h);
    OPS_LOG_D(nodeName, "aivNum is %u.", tilingData.moeDispatchNormalInfo.aivNum);
    OPS_LOG_D(nodeName, "totalUbSize is %lu.", tilingData.moeDispatchNormalInfo.totalUbSize);
    OPS_LOG_D(nodeName, "totalWinSize is %lu.", tilingData.moeDispatchNormalInfo.totalWinSize);
}

static bool CheckTensorDim(gert::TilingContext *context, const char *nodeName, const uint32_t quantMode)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    OPS_CHECK(xStorageShape == nullptr, OPS_LOG_E(nodeName, "xShape is null."), return false);
    OPS_CHECK(xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OPS_LOG_E(nodeName,
            "xShape dims must be 2, but current dim num is %lu.",
            xStorageShape->GetStorageShape().GetDimNum()),
        return false);
    int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OPS_LOG_D(nodeName, "x dim0 = %ld", xDim0);
    OPS_LOG_D(nodeName, "x dim1 = %ld", xDim1);

    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    OPS_CHECK(expertIdStorageShape == nullptr, OPS_LOG_E(nodeName, "expertIdShape is null."), return false);
    OPS_CHECK(expertIdStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OPS_LOG_E(nodeName,
            "expertIdShape dims must be 2, but current dim num is %lu.",
            expertIdStorageShape->GetStorageShape().GetDimNum()),
        return false);
    OPS_LOG_D(nodeName, "expertId dim0 = %ld", expertIdStorageShape->GetStorageShape().GetDim(0));
    OPS_LOG_D(nodeName, "expertId dim1 = %ld", expertIdStorageShape->GetStorageShape().GetDim(1));

    const gert::StorageShape *expandXStorageShape = context->GetOutputShape(OUTPUT_EXPAND_X_INDEX);
    OPS_CHECK(expandXStorageShape == nullptr, OPS_LOG_E(nodeName, "expandXShape is null."), return false);
    OPS_CHECK(expandXStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OPS_LOG_E(nodeName,
            "expandXShape dims must be 2, but current dim num is %lu.",
            expandXStorageShape->GetStorageShape().GetDimNum()),
        return false);
    OPS_LOG_D(nodeName, "expandX dim0 = %ld", expandXStorageShape->GetStorageShape().GetDim(0));
    OPS_LOG_D(nodeName, "expandX dim1 = %ld", expandXStorageShape->GetStorageShape().GetDim(1));

    if (quantMode == DYNAMIC_SCALES) {
        const gert::StorageShape *dynamicScalesStorageShape = context->GetOutputShape(OUTPUT_DYNAMIC_SCALES_INDEX);
        OPS_CHECK(
            dynamicScalesStorageShape == nullptr, OPS_LOG_E(nodeName, "dynamicScalesShape is null."), return false);
        OPS_CHECK(dynamicScalesStorageShape->GetStorageShape().GetDimNum() != DYNAMIC_SCALE_DIM_NUM,
            OPS_LOG_E(nodeName,
                "dynamicScalesShape dims must be %u, but current dim num is %lu.",
                DYNAMIC_SCALE_DIM_NUM,
                dynamicScalesStorageShape->GetStorageShape().GetDimNum()),
            return false);
        OPS_LOG_D(nodeName, "dynamicScales dim0 = %ld", dynamicScalesStorageShape->GetStorageShape().GetDim(0));
    }

    const gert::StorageShape *assistInfoStorageShape = context->GetOutputShape(OUTPUT_ASSIST_INFO_INDEX);
    OPS_CHECK(assistInfoStorageShape == nullptr, OPS_LOG_E(nodeName, "assistInfoShape is null."), return false);
    OPS_CHECK(assistInfoStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OPS_LOG_E(nodeName,
            "assistInfoShape dims must be 1, but current dim num is %lu.",
            assistInfoStorageShape->GetStorageShape().GetDimNum()),
        return false);
    OPS_LOG_D(nodeName, "assistInfoForCombine dim0 = %ld", assistInfoStorageShape->GetStorageShape().GetDim(0));

    return true;
}

static bool CheckTensorDataType(gert::TilingContext *context, const char *nodeName, const uint32_t quantMode)
{
    auto xDesc = context->GetInputDesc(X_INDEX);
    OPS_CHECK(xDesc == nullptr, OPS_LOG_E(nodeName, "xDesc is null."), return false);
    OPS_CHECK((xDesc->GetDataType() != ge::DT_BF16) && (xDesc->GetDataType() != ge::DT_FLOAT16),
        OPS_LOG_E(nodeName, "x dataType is invalid, dataType should be bf16 or float16, but is ."),
        return false);

    auto expertIdDesc = context->GetInputDesc(EXPERT_IDS_INDEX);
    OPS_CHECK(expertIdDesc == nullptr, OPS_LOG_E(nodeName, "expertIdDesc is null."), return false);
    OPS_CHECK(expertIdDesc->GetDataType() != ge::DT_INT32,
        OPS_LOG_E(nodeName, "expertId dataType is invalid, dataType should be int32, but is ."),
        return false);

    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    OPS_CHECK(expandXDesc == nullptr, OPS_LOG_E(nodeName, "expandXDesc is null."), return false);
    if (quantMode != NO_SCALES) {
        OPS_CHECK(expandXDesc->GetDataType() != ge::DT_INT8,
            OPS_LOG_E(nodeName, "expandX dataType is invalid, dataType should be int8, but is."),
            return false);
    } else {
        OPS_CHECK(expandXDesc->GetDataType() != xDesc->GetDataType(),
            OPS_LOG_E(nodeName, "expandX dataType is invalid, dataType should be equal to x dataType , but is."),
            return false);
    }

    if (quantMode == DYNAMIC_SCALES) {
        auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
        OPS_CHECK(dynamicScalesDesc == nullptr, OPS_LOG_E(nodeName, "dynamicScalesDesc is null."), return false);
        OPS_CHECK(dynamicScalesDesc->GetDataType() != ge::DT_FLOAT,
            OPS_LOG_E(nodeName, "dynamicScales dataType is invalid, dataType should be float, but is ."),
            return false);
    }

    auto assistInfoDesc = context->GetOutputDesc(OUTPUT_ASSIST_INFO_INDEX);
    OPS_CHECK(assistInfoDesc == nullptr, OPS_LOG_E(nodeName, "assistInfoDesc is null."), return false);
    OPS_CHECK(assistInfoDesc->GetDataType() != ge::DT_INT32,
        OPS_LOG_E(nodeName, "assistInfoForCombine dataType is invalid, dataType should be int32, but is ."),
        return false);

    return true;
}

static bool CheckTensorFormat(gert::TilingContext *context, const char *nodeName, const uint32_t quantMode)
{
    auto xDesc = context->GetInputDesc(X_INDEX);
    OPS_CHECK(xDesc == nullptr, OPS_LOG_E(nodeName, "xDesc is null."), return false);
    OPS_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OPS_LOG_E(nodeName, "x format is invalid."),
        return false);

    auto expertIdDesc = context->GetInputDesc(EXPERT_IDS_INDEX);
    OPS_CHECK(expertIdDesc == nullptr, OPS_LOG_E(nodeName, "expertIdDesc is null."), return false);
    OPS_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(expertIdDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OPS_LOG_E(nodeName, "expertId format is invalid."),
        return false);

    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    OPS_CHECK(expandXDesc == nullptr, OPS_LOG_E(nodeName, "expandXDesc is null."), return false);
    OPS_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(expandXDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OPS_LOG_E(nodeName, "expandX format is invalid."),
        return false);

    if (quantMode == DYNAMIC_SCALES) {
        auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
        OPS_CHECK(dynamicScalesDesc == nullptr, OPS_LOG_E(nodeName, "dynamicScalesDesc is null."), return false);
        OPS_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(dynamicScalesDesc->GetStorageFormat())) ==
                            ge::FORMAT_FRACTAL_NZ,
            OPS_LOG_E(nodeName, "dynamicScales format is invalid."),
            return false);
    }

    auto assistInfoDesc = context->GetOutputDesc(OUTPUT_ASSIST_INFO_INDEX);
    OPS_CHECK(assistInfoDesc == nullptr, OPS_LOG_E(nodeName, "assistInfoDesc is null."), return false);
    OPS_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(assistInfoDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OPS_LOG_E(nodeName, "assistInfoForCombine format is invalid."),
        return false);

    return true;
}

static ge::graphStatus GetAttrAndSetTilingData(gert::TilingContext *context, const char *nodeName,
    MoeDispatchNormalTilingData &tilingData, std::string &groupEp, std::string &groupTp)
{
    auto attrs = context->GetAttrs();
    OPS_CHECK(attrs == nullptr, OPS_LOG_E(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

    auto groupEpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    auto groupTpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_TP_INDEX));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto tpRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_RANK_ID_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);

    // Check for null
    OPS_CHECK((groupEpPtr == nullptr) || (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
                        (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
        OPS_LOG_E(nodeName, "groupEpPtr is null."),
        return ge::GRAPH_FAILED);
    OPS_CHECK(epWorldSizePtr == nullptr, OPS_LOG_E(nodeName, "epWorldSizePtr is null."), return ge::GRAPH_FAILED);
    OPS_CHECK(tpWorldSizePtr == nullptr, OPS_LOG_E(nodeName, "tpWorldSizePtr is null."), return ge::GRAPH_FAILED);
    OPS_CHECK(epRankIdPtr == nullptr, OPS_LOG_E(nodeName, "epRankIdPtr is null."), return ge::GRAPH_FAILED);
    OPS_CHECK(tpRankIdPtr == nullptr, OPS_LOG_E(nodeName, "tpRankIdPtr is null."), return ge::GRAPH_FAILED);
    OPS_CHECK(moeExpertNumPtr == nullptr, OPS_LOG_E(nodeName, "moeExpertNumPtr is null."), return ge::GRAPH_FAILED);
    OPS_CHECK(quantModePtr == nullptr, OPS_LOG_E(nodeName, "quantModePtr is null."), return ge::GRAPH_FAILED);

    // Check if it meets uint32_t and other constraints
    int64_t moeExpertNum = *moeExpertNumPtr;
    int64_t epWorldSize = *epWorldSizePtr;
    OPS_CHECK((epWorldSize < MIN_EP_WORLD_SIZE) || (epWorldSize > MAX_EP_WORLD_SIZE),
        OPS_LOG_E(nodeName,
            "epWorldSize is invalid, only support [%ld, %ld], but got epWorldSize=%ld.",
            MIN_EP_WORLD_SIZE,
            MAX_EP_WORLD_SIZE,
            epWorldSize),
        return ge::GRAPH_FAILED);
    OPS_CHECK((*tpWorldSizePtr < 0) || (*tpWorldSizePtr > MAX_TP_WORLD_SIZE),
        OPS_LOG_E(nodeName,
            "tpWorldSize is invalid, only support [0, %ld], but got tpWorldSize=%ld.",
            MAX_TP_WORLD_SIZE,
            *tpWorldSizePtr),
        return ge::GRAPH_FAILED);
    OPS_CHECK((*epRankIdPtr < 0) || (*epRankIdPtr >= epWorldSize),
        OPS_LOG_E(
            nodeName, "epRankId is invalid, only support [0, %ld), but got epRankId=%ld.", epWorldSize, *epRankIdPtr),
        return ge::GRAPH_FAILED);
    if (*tpWorldSizePtr > 1) {
        OPS_CHECK((*tpRankIdPtr < 0) || (*tpRankIdPtr >= *tpWorldSizePtr),
            OPS_LOG_E(nodeName,
                "tpRankId is invalid, only support [0, %ld), but got tpRankId=%ld.",
                *tpWorldSizePtr,
                *tpRankIdPtr),
            return ge::GRAPH_FAILED);
        OPS_CHECK((groupTpPtr == nullptr) || (strnlen(groupTpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
                            (strnlen(groupTpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
            OPS_LOG_E(nodeName, "groupTpPtr is null."),
            return ge::GRAPH_FAILED);
        groupTp = std::string(groupTpPtr);
    } else {
        OPS_CHECK(*tpRankIdPtr != 0,
            OPS_LOG_E(nodeName, "tpRankId is invalid, NoTp mode only support 0, but got tpRankId=%ld.", *tpRankIdPtr),
            return ge::GRAPH_FAILED);
    }
    OPS_CHECK((moeExpertNum <= 0) || (moeExpertNum > MOE_EXPERT_MAX_NUM),
        OPS_LOG_E(nodeName,
            "moeExpertNum is invalid, only support (0, %ld], but got moeExpertNum=%ld.",
            MOE_EXPERT_MAX_NUM,
            moeExpertNum),
        return ge::GRAPH_FAILED);
    OPS_CHECK(
        (*quantModePtr < static_cast<int64_t>(NO_SCALES)) || (*quantModePtr > static_cast<int64_t>(DYNAMIC_SCALES)),
        OPS_LOG_E(nodeName,
            "quantMode is invalid, only support [0, %u], but got quantMode=%ld.",
            DYNAMIC_SCALES,
            *quantModePtr),
        return ge::GRAPH_FAILED);

    int64_t moePerRankNum = moeExpertNum / epWorldSize;
    int64_t curDispatchStatusNum = moePerRankNum * epWorldSize;
    OPS_CHECK((curDispatchStatusNum > DISPATCH_STATUS_MAX_SUPPORT_NUM),
        OPS_LOG_E(nodeName,
            "The moe experts num must meet the conditions,"
            " (moeExpertNum / epWorldSize * epWorldSize <= 1280, but cur is %ld.",
            curDispatchStatusNum),
        return ge::GRAPH_FAILED);

    groupEp = std::string(groupEpPtr);
    tilingData.moeDispatchNormalInfo.epWorldSize = static_cast<uint32_t>(epWorldSize);
    tilingData.moeDispatchNormalInfo.tpWorldSize = static_cast<uint32_t>(*tpWorldSizePtr);
    tilingData.moeDispatchNormalInfo.epRankId = static_cast<uint32_t>(*epRankIdPtr);
    tilingData.moeDispatchNormalInfo.tpRankId = static_cast<uint32_t>(*tpRankIdPtr);
    tilingData.moeDispatchNormalInfo.moeExpertNum = static_cast<uint32_t>(moeExpertNum);
    tilingData.moeDispatchNormalInfo.quantMode = static_cast<uint32_t>(*quantModePtr);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrs(
    gert::TilingContext *context, const char *nodeName, MoeDispatchNormalTilingData &tilingData, uint32_t &localMoeExpertNum)
{
    uint32_t epWorldSize = tilingData.moeDispatchNormalInfo.epWorldSize;
    uint32_t tpWorldSize = tilingData.moeDispatchNormalInfo.tpWorldSize;
    uint32_t moeExpertNum = tilingData.moeDispatchNormalInfo.moeExpertNum;

    // Validate if moe expert number can be evenly distributed across multiple machines
    localMoeExpertNum = moeExpertNum / epWorldSize;
    OPS_CHECK(moeExpertNum % epWorldSize != 0,
        OPS_LOG_E(nodeName,
            "moeExpertNum should be divisible by epWorldSize, "
            "but moeExpertNum=%u, epWorldSize=%u.",
            moeExpertNum,
            epWorldSize),
        return ge::GRAPH_FAILED);
    OPS_CHECK(localMoeExpertNum <= 0,
        OPS_LOG_E(nodeName, "localMoeExpertNum is invalid, localMoeExpertNum = %d", localMoeExpertNum),
        return ge::GRAPH_FAILED);

    // Validate input x dimension 0 and set bs
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    OPS_CHECK((xDim0 > BS_UPPER_BOUND) || (xDim0 <= 0),
        OPS_LOG_E(
            nodeName, "xDim0(BS) is invalid. Should be between [1, %ld], but got xDim0=%ld.", BS_UPPER_BOUND, xDim0),
        return ge::GRAPH_FAILED);
    tilingData.moeDispatchNormalInfo.bs = static_cast<uint32_t>(xDim0);

    // Validate globalBS
    auto attrs = context->GetAttrs();
    OPS_CHECK(attrs == nullptr, OPS_LOG_E(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_GLOBAL_BS_INDEX);
    OPS_CHECK(globalBsPtr == nullptr, OPS_LOG_E(nodeName, "globalBsPtr is nullptr."), return ge::GRAPH_FAILED);
    OPS_LOG_D(nodeName, "MoeDispatchNormal *globalBsPtr = %ld, bs = %ld, epWorldSize = %u\n", *globalBsPtr, xDim0, epWorldSize);
    OPS_CHECK(*globalBsPtr <= 0,
        OPS_LOG_E(nodeName,
            "globalBS is invalid, should be positive, but got globalBS=%ld.",
            *globalBsPtr),
        return ge::GRAPH_FAILED);

    tilingData.moeDispatchNormalInfo.globalBs = static_cast<uint32_t>(*globalBsPtr);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorShape(gert::TilingContext *context, const char *nodeName,
    MoeDispatchNormalTilingData &tilingData, const uint32_t quantMode, const int64_t localMoeExpertNum)
{
    uint32_t A = 0U;
    uint32_t globalBs = tilingData.moeDispatchNormalInfo.globalBs;

    // Validate input x dimension 1 and set h, bs already validated
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    const int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OPS_CHECK((xDim1 < H_MIN) || (xDim1 > H_MAX),
        OPS_LOG_E(nodeName, "xShape dims1(H) should be in [%ld, %ld], but got %ld.", H_MIN, H_MAX, xDim1),
        return ge::GRAPH_FAILED);  // 32-byte aligned
    tilingData.moeDispatchNormalInfo.h = static_cast<uint32_t>(xDim1);

    // Validate expert_id dimensions and set k
    int64_t moeExpertNum = static_cast<int64_t>(tilingData.moeDispatchNormalInfo.moeExpertNum);
    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    const int64_t expertIdsDim0 = expertIdStorageShape->GetStorageShape().GetDim(0);
    const int64_t expertIdsDim1 = expertIdStorageShape->GetStorageShape().GetDim(1);
    OPS_CHECK(xDim0 != expertIdsDim0,
        OPS_LOG_E(nodeName,
            "xShape's dim0 not equal to expertIdShape's dim0, "
            "xShape's dim0 is %ld, expertIdShape's dim0 is %ld.",
            xDim0,
            expertIdsDim0),
        return ge::GRAPH_FAILED);
    OPS_CHECK((expertIdsDim1 <= 0) || (expertIdsDim1 > K_MAX) || (expertIdsDim1 > moeExpertNum),
        OPS_LOG_E(nodeName,
            "expertIdShape's dim1(k) should be in (0, min(%ld, moeExpertNum=%ld)], "
            "but got expertIdShape's dim1=%ld.",
            K_MAX,
            moeExpertNum,
            expertIdsDim1),
        return ge::GRAPH_FAILED);
    tilingData.moeDispatchNormalInfo.k = static_cast<uint32_t>(expertIdsDim1);

    A = globalBs;

    // Validate expandX dimensions
    const gert::StorageShape *expandXStorageShape = context->GetOutputShape(OUTPUT_EXPAND_X_INDEX);
    const int64_t expandXDim0 = expandXStorageShape->GetStorageShape().GetDim(0);
    const int64_t expandXDim1 = expandXStorageShape->GetStorageShape().GetDim(1);

    OPS_CHECK(xDim1 != expandXDim1,
        OPS_LOG_E(nodeName,
            "expandX's dim1 not equal to xShape's dim1, "
            "xShape's dim1 is %ld, expandX's dim1 is %ld.",
            xDim1,
            expandXDim1),
        return ge::GRAPH_FAILED);

    // Validate dynamicScales dimensions
    if (quantMode != NO_SCALES) {
        const gert::StorageShape *dynamicScalesStorageShape = context->GetOutputShape(OUTPUT_DYNAMIC_SCALES_INDEX);
        const int64_t dynamicScalesDim0 = dynamicScalesStorageShape->GetStorageShape().GetDim(0);
    }

    // Validate assistInfo dimensions
    const gert::StorageShape *assistInfoStorageShape = context->GetOutputShape(OUTPUT_ASSIST_INFO_INDEX);
    const int64_t assistInfoDim0 = assistInfoStorageShape->GetStorageShape().GetDim(0);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingCheckMoeDispatchNormal(
    gert::TilingContext *context, const char *nodeName, const uint32_t quantMode)
{
    OPS_CHECK(!CheckTensorDim(context, nodeName, quantMode),
        OPS_LOG_E(nodeName, "params shape is invalid."),
        return ge::GRAPH_FAILED);
    OPS_CHECK(!CheckTensorDataType(context, nodeName, quantMode),
        OPS_LOG_E(nodeName, "params dataType is invalid."),
        return ge::GRAPH_FAILED);
    OPS_CHECK(!CheckTensorFormat(context, nodeName, quantMode),
        OPS_LOG_E(nodeName, "params format is invalid."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static void CalTilingKey(uint64_t &tilingKey, const uint32_t quantMode, const uint32_t tpWorldSize)
{
    tilingKey += static_cast<uint64_t>(quantMode);
    if (tpWorldSize == TP_WORLD_SIZE_TWO) {
        tilingKey += static_cast<uint64_t>(TILINGKEY_TP_WORLD_SIZE);
    }

    return;
}

static void SetHcommCfg(const gert::TilingContext *context, MoeDispatchNormalTilingData *tiling, const std::string groupEp,
    const std::string groupTp)
{
    const char *nodeName = context->GetNodeName();
    OPS_LOG_D(nodeName, "MoeDispatchNormal groupEp = %s, groupTp = %s", groupEp.c_str(), groupTp.c_str());
    uint32_t opType1 = OP_TYPE_ALL_TO_ALL;
    uint32_t opType2 = OP_TYPE_ALL_GATHER;
    std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh;level1:pairwise";
    std::string algConfigAllGatherStr = "AllGather=level0:ring";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp, opType1, algConfigAllToAllStr);
    mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling1);

    mc2CcTilingConfig.SetGroupName(groupTp);
    mc2CcTilingConfig.SetOpType(opType2);
    mc2CcTilingConfig.SetAlgConfig(algConfigAllGatherStr);
    mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling2);
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context, const char *nodeName)
{
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OPS_CHECK(workSpaces == nullptr, OPS_LOG_E(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    workSpaces[0] = static_cast<uint64_t>(SYSTEM_NEED_WORKSPACE + WORKSPACE_ELEMENT_OFFSET * aivNum * aivNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDispatchNormalA3TilingFuncImpl(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    MoeDispatchNormalTilingData *tilingData = context->GetTilingData<MoeDispatchNormalTilingData>();
    OPS_CHECK(tilingData == nullptr, OPS_LOG_E(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    std::string groupEp = "";
    std::string groupTp = "";
    uint32_t quantMode = NO_SCALES;
    uint32_t localMoeExpertNum = 1;
    OPS_LOG_I(nodeName, "Enter MoeDispatchNormal tiling check func.");

    // Get input parameter attributes
    OPS_CHECK(GetAttrAndSetTilingData(context, nodeName, *tilingData, groupEp, groupTp) != ge::GRAPH_SUCCESS,
        OPS_LOG_E(nodeName, "Get attr and set tiling data failed."),
        return ge::GRAPH_FAILED);

    quantMode = tilingData->moeDispatchNormalInfo.quantMode;

    // Check input/output dim, format, dataType
    OPS_CHECK(TilingCheckMoeDispatchNormal(context, nodeName, quantMode) != ge::GRAPH_SUCCESS,
        OPS_LOG_E(nodeName, "Tiling check param failed."),
        return ge::GRAPH_FAILED);

    // Check if attribute values are valid
    OPS_CHECK(CheckAttrs(context, nodeName, *tilingData, localMoeExpertNum) != ge::GRAPH_SUCCESS,
        OPS_LOG_E(nodeName, "Check attr failed."),
        return ge::GRAPH_FAILED);

    uint32_t epRankId = tilingData->moeDispatchNormalInfo.epRankId;

    // Check shape dimensions and assign h, k
    OPS_CHECK(
        CheckTensorShape(context, nodeName, *tilingData, quantMode, static_cast<int64_t>(localMoeExpertNum)) !=
            ge::GRAPH_SUCCESS,
        OPS_LOG_E(nodeName, "Check tensor shape failed."),
        return ge::GRAPH_FAILED);

    // Validate win area size
    uint64_t maxWindowSize = Mc2TilingUtils::GetMaxWindowSize();
    uint64_t h = static_cast<uint64_t>(tilingData->moeDispatchNormalInfo.h);
    uint64_t k = static_cast<uint64_t>(tilingData->moeDispatchNormalInfo.k);
    uint64_t epWorldSize = static_cast<uint64_t>(tilingData->moeDispatchNormalInfo.epWorldSize);
    uint64_t maxBs = static_cast<uint64_t>(tilingData->moeDispatchNormalInfo.globalBs) / epWorldSize;

    // Dispatch data area: token start aligned to 512, valid token length h_align_32b + scale(32b) + triplet(3*4b)
    uint64_t tokenActualLen =
        ((h * MAX_OUT_DTYPE_SIZE + UB_ALIGN - 1UL) / UB_ALIGN) * UB_ALIGN + SCALE_EXPAND_IDX_BUFFER;
    uint64_t tokenNeedSizeDispatch = ((tokenActualLen + WIN_ADDR_ALIGN - 1UL) / WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    // Not considering dual stream size
    uint64_t actualSize = maxBs * k * tokenNeedSizeDispatch * DOUBLE_DATA_BUFFER;
    OPS_CHECK((actualSize > maxWindowSize),
        OPS_LOG_E(nodeName,
            "HCCL_BUFFSIZE is too SMALL, maxBs = %lu, h = %lu, epWorldSize = %lu,"
            " localMoeExpertNum = %u, tokenNeedSizeDispatch = %lu,"
            " k = %lu, NEEDED_HCCL_BUFFSIZE(maxBs * k * tokenNeedSizeDispatch) = %luMB,"
            " HCCL_BUFFSIZE=%luMB.",
            maxBs,
            h,
            epWorldSize,
            localMoeExpertNum,
            tokenNeedSizeDispatch,
            k,
            actualSize / MB_SIZE + 1UL,
            maxWindowSize / MB_SIZE),
        return ge::GRAPH_FAILED);
    tilingData->moeDispatchNormalInfo.totalWinSize = maxWindowSize;
    OPS_LOG_D(nodeName, "windowSize = %lu", maxWindowSize);

    OPS_CHECK(SetWorkSpace(context, nodeName) != ge::GRAPH_SUCCESS,
        OPS_LOG_E(nodeName, "Tiling set workspace failed."),
        return ge::GRAPH_FAILED);
    SetHcommCfg(context, tilingData, groupEp, groupTp);
    uint32_t tpWorldSize = tilingData->moeDispatchNormalInfo.tpWorldSize;
    uint64_t tilingKey = INIT_TILINGKEY;
    CalTilingKey(tilingKey, quantMode, tpWorldSize);
    OPS_LOG_D(nodeName, "tilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);
    uint32_t blockDim = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(blockDim);
    context->SetScheduleMode(1);  // Set to batch mode, all cores start simultaneously
    tilingData->moeDispatchNormalInfo.totalUbSize = ubSize;
    tilingData->moeDispatchNormalInfo.aivNum = aivNum;
    OPS_LOG_D(nodeName, "blockDim=%u, aivNum=%u, ubSize=%lu", blockDim, aivNum, ubSize);
    PrintTilingDataInfo(nodeName, *tilingData);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDispatchNormalTilingFunc(gert::TilingContext *context)
{
    ge::graphStatus ret = MoeDispatchNormalA3TilingFuncImpl(context);
    return ret;
}

struct MoeDispatchNormalCompileInfo {};
ge::graphStatus TilingParseForMoeDispatchNormal(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeDispatchNormal)
    .Tiling(MoeDispatchNormalTilingFunc)
    .TilingParse<MoeDispatchNormalCompileInfo>(TilingParseForMoeDispatchNormal);
}  // namespace optiling