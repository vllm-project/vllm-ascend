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
#include "../op_kernel/notify_dispatch_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/hccl/hccl_tiling.h"
#include "platform/platform_infos_def.h"

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
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U;  // numeric representation of AlltoAll

constexpr uint32_t INPUT_SEND_DATA_INDEX = 0;
constexpr uint32_t INPUT_TOKEN_PER_EXPERT_INDEX = 1;

constexpr uint32_t OUTPUT_SEND_DATA_OFFSET_INDEX = 0;
constexpr uint32_t OUTPUT_RECV_DATA_INDEX = 1;

constexpr uint32_t ATTR_SEND_COUNT_INDEX = 0;
constexpr uint32_t ATTR_NUM_TOKENS_INDEX = 1;
constexpr uint32_t ATTR_COMM_GROUP_INDEX = 2;
constexpr uint32_t ATTR_RANK_SIZE_INDEX = 3;
constexpr uint32_t ATTR_RANK_ID_INDEX = 4;
constexpr uint32_t ATTR_LOCAL_RANK_SIZE_INDEX = 5;
constexpr uint32_t ATTR_LOCAL_RANK_ID_INDEX = 6;

const size_t MAX_GROUP_NAME_LENGTH = 128UL;
const int64_t MAX_COMM_WORLD_SIZE = 384;

constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
constexpr uint32_t KERNEL_USE_WORKSPACE = 1 * 1024 * 1024;
constexpr uint32_t KERNEL_A2_ARG_SIZE = 1 * 1024 * 1024;
constexpr int32_t HCCL_BUFFER_SIZE_DEFAULT = 200 * 1024 * 1024;  // Bytes
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;

constexpr static int TILING_KEY_FLOAT16 = 20;
constexpr static int TILING_KEY_BFLOAT16 = 21;
constexpr static int TILING_KEY_FLOAT = 22;
constexpr static int TILING_KEY_INT = 23;
constexpr static int TILING_KEY_A2_TYPE = 100;

constexpr static int ALL_TO_ALL_CORE_NUM = 32;
}  // namespace

namespace optiling {
static void PrintTilingDataInfo(const char *nodeName, NotifyDispatchTilingData &tilingData)
{
    OPS_LOG_D(nodeName, "rankSize is %u.", tilingData.notifyDispatchInfo.rankSize);
    OPS_LOG_D(nodeName, "rankId is %u.", tilingData.notifyDispatchInfo.rankId);
    OPS_LOG_D(nodeName, "localRankSize is %u.", tilingData.notifyDispatchInfo.localRankSize);
    OPS_LOG_D(nodeName, "localRankId is %u.", tilingData.notifyDispatchInfo.localRankId);
    OPS_LOG_D(nodeName, "sendCount is %u.", tilingData.notifyDispatchInfo.sendCount);
    OPS_LOG_D(nodeName, "numTokens is %u.", tilingData.notifyDispatchInfo.numTokens);
    OPS_LOG_D(nodeName, "aivNum is %u.", tilingData.notifyDispatchInfo.aivNum);
    OPS_LOG_D(nodeName, "totalUbSize is %lu.", tilingData.notifyDispatchInfo.totalUbSize);
}

static ge::graphStatus GetAttrAndSetTilingData(gert::TilingContext *context, const char *nodeName,
    NotifyDispatchTilingData &tilingData, std::string &commGroup)
{
    auto attrs = context->GetAttrs();
    OPS_CHECK(attrs == nullptr, OPS_LOG_E(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

    auto sendCountPtr = attrs->GetAttrPointer<int64_t>(ATTR_SEND_COUNT_INDEX);
    auto numTokenPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_TOKENS_INDEX);
    auto commGroupPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_COMM_GROUP_INDEX));
    auto rankSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_RANK_SIZE_INDEX);
    auto rankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_RANK_ID_INDEX);
    auto localRankSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_LOCAL_RANK_SIZE_INDEX);
    auto localRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_LOCAL_RANK_ID_INDEX);

    OPS_CHECK((commGroupPtr == nullptr) || (strnlen(commGroupPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
                        (strnlen(commGroupPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
        OPS_LOG_E(nodeName, "commGroupPtr is null."),
        return ge::GRAPH_FAILED);
    OPS_CHECK(sendCountPtr == nullptr, OPS_LOG_E(nodeName, "sendCountPtr is null."), return ge::GRAPH_FAILED);
    OPS_CHECK(numTokenPtr == nullptr, OPS_LOG_E(nodeName, "numTokenPtr is null."), return ge::GRAPH_FAILED);
    OPS_CHECK(rankSizePtr == nullptr, OPS_LOG_E(nodeName, "rankSizePtr is null."), return ge::GRAPH_FAILED);
    OPS_CHECK(rankIdPtr == nullptr, OPS_LOG_E(nodeName, "rankIdPtr is null."), return ge::GRAPH_FAILED);
    OPS_CHECK(
        localRankSizePtr == nullptr, OPS_LOG_E(nodeName, "localRankSizePtr is null."), return ge::GRAPH_FAILED);
    OPS_CHECK(localRankIdPtr == nullptr, OPS_LOG_E(nodeName, "localRankIdPtr is null."), return ge::GRAPH_FAILED);

    OPS_CHECK((*rankSizePtr <= 0) || (*rankSizePtr > MAX_COMM_WORLD_SIZE),
        OPS_LOG_E(nodeName,
            "rankSize is invalid, only support (0, %ld], but got rankSize=%ld.",
            MAX_COMM_WORLD_SIZE,
            *rankSizePtr),
        return ge::GRAPH_FAILED);
    OPS_CHECK((*rankIdPtr < 0) || (*rankIdPtr >= *rankSizePtr),
        OPS_LOG_E(nodeName, "rankId is invalid, only support [0, %ld), but got rankId=%ld.", *rankSizePtr, *rankIdPtr),
        return ge::GRAPH_FAILED);
    OPS_CHECK((*sendCountPtr <= 0),
        OPS_LOG_E(nodeName, "sendCount is invalid, only support > 0, but got sendCount=%ld.", *sendCountPtr),
        return ge::GRAPH_FAILED);
    OPS_CHECK((*numTokenPtr <= 0),
        OPS_LOG_E(nodeName, "numTokenPtr is invalid, only support > 0, but got numTokenPtr=%ld.", *numTokenPtr),
        return ge::GRAPH_FAILED);

    commGroup = std::string(commGroupPtr);
    tilingData.notifyDispatchInfo.rankSize = static_cast<uint32_t>(*rankSizePtr);
    tilingData.notifyDispatchInfo.rankId = static_cast<uint32_t>(*rankIdPtr);
    tilingData.notifyDispatchInfo.localRankSize = static_cast<uint32_t>(*localRankSizePtr);
    tilingData.notifyDispatchInfo.localRankId = static_cast<uint32_t>(*localRankIdPtr);
    tilingData.notifyDispatchInfo.sendCount = static_cast<uint32_t>(*sendCountPtr);
    tilingData.notifyDispatchInfo.numTokens = static_cast<uint32_t>(*numTokenPtr);

    return ge::GRAPH_SUCCESS;
}

static void SetHcommCfg(const gert::TilingContext *context,
    NotifyDispatchTilingData *tiling, const std::string commGroup)
{
    const char *nodeName = context->GetNodeName();
    OPS_LOG_D(nodeName, "NotifyDispatch commGroup = %s", commGroup.c_str());
    uint32_t opType1 = OP_TYPE_ALL_TO_ALL;
    std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh;level1:pairwise";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(commGroup, opType1, algConfigAllToAllStr);
    mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling1);
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context, const char *nodeName)
{
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OPS_CHECK(workSpaces == nullptr, OPS_LOG_E(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + KERNEL_USE_WORKSPACE + KERNEL_A2_ARG_SIZE;
    return ge::GRAPH_SUCCESS;
}

static bool CheckTensorDataType(
    gert::TilingContext *context, const char *nodeName)
{
    auto sendData = context->GetInputDesc(INPUT_SEND_DATA_INDEX);
    OPS_CHECK(sendData == nullptr, OPS_LOG_E(nodeName, "sendData is null."), return false);
    OPS_CHECK((sendData->GetDataType() != ge::DT_BF16) && (sendData->GetDataType() != ge::DT_FLOAT16) &&
                        (sendData->GetDataType() != ge::DT_FLOAT) && (sendData->GetDataType() != ge::DT_INT32),
        OPS_LOG_E(nodeName,
            "sendData datatype is invalid, datatype should be bf16 or float16 or float or int, but is %d.",
            static_cast<ge::DataType>(sendData->GetDataType())),
        return false);
    uint64_t dataSize;
    if ((sendData->GetDataType() == ge::DT_BF16) || (sendData->GetDataType() == ge::DT_FLOAT16)) {
        dataSize = 2;
    } else {
        dataSize = 4;
    }
    auto tokenPerExpertData = context->GetInputDesc(INPUT_TOKEN_PER_EXPERT_INDEX);
    OPS_CHECK(tokenPerExpertData == nullptr, OPS_LOG_E(nodeName, "tokenPerExpertData is null."), return false);
    OPS_CHECK((tokenPerExpertData->GetDataType() != ge::DT_BF16) && (tokenPerExpertData->GetDataType() != ge::DT_FLOAT16) &&
                        (tokenPerExpertData->GetDataType() != ge::DT_FLOAT) && (tokenPerExpertData->GetDataType() != ge::DT_INT32),
        OPS_LOG_E(nodeName,
            "tokenPerExpertData datatype is invalid, datatype should be bf16 or float16 or float or int, but is %d.",
            static_cast<ge::DataType>(tokenPerExpertData->GetDataType())),
        return false);
    
    auto sendDataOffset = context->GetInputDesc(OUTPUT_SEND_DATA_OFFSET_INDEX);
    OPS_CHECK(sendDataOffset == nullptr, OPS_LOG_E(nodeName, "sendDataOffset is null."), return false);
    OPS_CHECK((sendDataOffset->GetDataType() != ge::DT_BF16) && (sendDataOffset->GetDataType() != ge::DT_FLOAT16) &&
                        (sendDataOffset->GetDataType() != ge::DT_FLOAT) && (sendDataOffset->GetDataType() != ge::DT_INT32),
        OPS_LOG_E(nodeName,
            "sendDataOffset datatype is invalid, datatype should be bf16 or float16 or float or int, but is %d.",
            static_cast<ge::DataType>(sendDataOffset->GetDataType())),
        return false);
    
    auto recvData = context->GetInputDesc(OUTPUT_RECV_DATA_INDEX);
    OPS_CHECK(recvData == nullptr, OPS_LOG_E(nodeName, "recvData is null."), return false);
    OPS_CHECK((recvData->GetDataType() != ge::DT_BF16) && (recvData->GetDataType() != ge::DT_FLOAT16) &&
                        (recvData->GetDataType() != ge::DT_FLOAT) && (recvData->GetDataType() != ge::DT_INT32),
        OPS_LOG_E(nodeName,
            "recvData datatype is invalid, datatype should be bf16 or float16 or float or int, but is %d.",
            static_cast<ge::DataType>(recvData->GetDataType())),
        return false);

    // Verify the size of the win area
    NotifyDispatchTilingData *tilingData = context->GetTilingData<NotifyDispatchTilingData>();
    uint64_t maxWindowSize = Mc2TilingUtils::GetMaxWindowSize();
    uint64_t actualSize = dataSize * tilingData->notifyDispatchInfo.sendCount;
    if (actualSize > maxWindowSize) {
        OPS_LOG_E(nodeName, "HCCL_BUFFSIZE is too SMALL, should larger than %lu", actualSize);
        return false;
    }
    return true;
}

static ge::graphStatus TilingCheckTensor(
    gert::TilingContext *context, const char *nodeName)
{
    OPS_CHECK(!CheckTensorDataType(context, nodeName),
        OPS_LOG_E(nodeName, "params dataType is invalid."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus NotifyDispatchTilingFuncImpl(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    NotifyDispatchTilingData *tilingData = context->GetTilingData<NotifyDispatchTilingData>();
    OPS_CHECK(tilingData == nullptr, OPS_LOG_E(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    std::string commGroup = "";
    OPS_LOG_I(nodeName, "Enter NotifyDispatch tiling check func.");

    OPS_CHECK(GetAttrAndSetTilingData(context, nodeName, *tilingData, commGroup) != ge::GRAPH_SUCCESS,
        OPS_LOG_E(nodeName, "Get attr and set tiling data failed."),
        return ge::GRAPH_FAILED);

    OPS_CHECK(TilingCheckTensor(context, nodeName) != ge::GRAPH_SUCCESS,
        OPS_LOG_E(nodeName, "Tiling check param failed."),
        return ge::GRAPH_FAILED);

    OPS_CHECK(SetWorkSpace(context, nodeName) != ge::GRAPH_SUCCESS,
        OPS_LOG_E(nodeName, "Tiling set workspace failed."),
        return ge::GRAPH_FAILED);
    SetHcommCfg(context, tilingData, commGroup);

    int tilingKey = TILING_KEY_INT;
    auto sendDtype = context->GetInputDesc(0)->GetDataType();
    if (sendDtype == ge::DT_FLOAT16) {
        tilingKey = TILING_KEY_FLOAT16;
    } else if (sendDtype == ge::DT_BF16) {
        tilingKey = TILING_KEY_BFLOAT16;
    } else if (sendDtype == ge::DT_FLOAT) {
        tilingKey = TILING_KEY_FLOAT;
    }

    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    fe::PlatFormInfos &platformInfo = *platformInfoPtr;

    std::string socVersion;
    (void)platformInfo.GetPlatformResWithLock("version", "Short_SoC_version", socVersion);

    if (socVersion == "Ascend910B") {
        tilingKey = tilingKey + TILING_KEY_A2_TYPE;
    }
    context->SetTilingKey(tilingKey);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t blockDim;
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    blockDim = aivNum;
    context->SetBlockDim(blockDim);
    tilingData->notifyDispatchInfo.totalUbSize = ubSize;
    tilingData->notifyDispatchInfo.aivNum = aivNum;
    OPS_LOG_D(nodeName, "blockDim=%u, aivNum=%u, ubSize=%lu", blockDim, aivNum, ubSize);
    PrintTilingDataInfo(nodeName, *tilingData);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus NotifyDispatchTilingFunc(gert::TilingContext *context)
{
    ge::graphStatus ret = NotifyDispatchTilingFuncImpl(context);
    return ret;
}

struct NotifyDispatchCompileInfo {};
ge::graphStatus TilingParseForNotifyDispatch(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(NotifyDispatch)
    .Tiling(NotifyDispatchTilingFunc)
    .TilingParse<NotifyDispatchCompileInfo>(TilingParseForNotifyDispatch);
}  // namespace optiling