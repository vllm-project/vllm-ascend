/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ShmemCommZbMoeDistributeDispatch tiling function implementation file
 * History: 2025-08-20 create ShmemCommZbMoeDistributeDispatch tiling function implementation file
 */

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
#include "tiling/hccl/hccl_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/error_log.h"
#include "register/op_def_registry.h"
#include "../op_kernel/zb_moe_distribute_dispatch_tiling.h"

using namespace AscendC;
using namespace ge;
namespace {
constexpr uint32_t X_INDEX = 0U;
constexpr uint32_t EXPERT_IDS_INDEX = 1U;
constexpr uint32_t SCALES_INDEX = 2U;
constexpr uint32_t X_ACTIVE_MASK_INDEX = 3U;
constexpr uint32_t ELASTIC_INFO_INDEX = 4U;
constexpr uint32_t OUTPUT_EXPAND_X_INDEX = 0U;
constexpr uint32_t OUTPUT_DYNAMIC_SCALES_INDEX = 1U;
constexpr uint32_t OUTPUT_ASSIST_INFO_INDEX = 2U;
constexpr uint32_t OUTPUT_EXPERT_TOKEN_NUMS_INDEX = 3U;
constexpr uint32_t OUTPUT_EP_RECV_COUNTS_INDEX = 4U;
constexpr uint32_t OUTPUT_TP_RECV_COUNTS_INDEX = 5U;

// constexpr uint32_t ATTR_GROUP_EP_INDEX = -1;
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 0;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 1;
constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 2;
// constexpr uint32_t ATTR_GROUP_TP_INDEX = -1;
constexpr uint32_t ATTR_TP_WORLD_SIZE_INDEX = 3;
constexpr uint32_t ATTR_TP_RANK_ID_INDEX = 4;
constexpr uint32_t ATTR_EXPERT_SHARD_TYPE_INDEX = 5;
constexpr uint32_t ATTR_SHARED_EXPERT_NUM_INDEX = 6;
constexpr uint32_t ATTR_SHARED_EXPERT_RANK_NUM_INDEX = 7;
constexpr uint32_t ATTR_QUANT_MODE_INDEX = 8;
constexpr uint32_t ATTR_GLOBAL_BS_INDEX = 9;
constexpr uint32_t ATTR_EXPERT_TOKEN_NUMS_TYPE_INDEX = 10;
constexpr uint32_t ATTR_EXT_INFO_INDEX = 11;
constexpr uint32_t ATTR_COMM_ALG_INDEX = 12;
constexpr uint32_t ATTR_ZERO_EXPERT_NUM_INDEX = 13;
constexpr uint32_t ATTR_COPY_EXPERT_NUM_INDEX = 14;
constexpr uint32_t ATTR_CONST_EXPERT_NUM_INDEX = 15;

constexpr uint32_t TWO_DIMS = 2;
constexpr uint32_t ONE_DIM = 1;
constexpr uint32_t DYN_SCALE_DIMS = 1;
constexpr uint32_t ASSIST_INFO_DIMS = 1;
constexpr uint32_t DYNAMIC_SCALE_DIM_NUM = 1;
constexpr uint64_t INIT_TILINGKEY = 10000;
constexpr uint32_t ARR_LENGTH = 128;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;
constexpr uint32_t NO_SCALES = 0;
constexpr uint32_t STATIC_SCALES = 1;
constexpr uint32_t DYNAMIC_SCALES = 2;
constexpr uint32_t OP_TYPE_ALL_GATHER = 6;

constexpr uint32_t UNQUANT_MODE = 0;
constexpr uint32_t STATIC_QUANT_MODE = 1;
constexpr uint32_t DYNAMIC_QUANT_MODE = 2;
constexpr size_t MAX_GROUP_NAME_LENGTH = 128UL;
constexpr int64_t MAX_SHARED_EXPERT_NUM = 4;
constexpr int64_t MAX_EP_WORLD_SIZE = 768L;  // 384 * 2
constexpr int64_t MIN_EP_WORLD_SIZE = 2;
constexpr int64_t EP_RESTRICT_8 = 8;
constexpr int64_t MAX_TP_WORLD_SIZE = 2;
constexpr int64_t BS_UPPER_BOUND = 512;

constexpr uint64_t NUM_10 = 10ULL;
constexpr uint32_t TILINGKEY_SCALES = 10;
constexpr uint32_t TILINGKEY_TP_WORLD_SIZE = 100;
constexpr uint32_t TILINGKEY_COMM_ALG = 1000;
constexpr uint32_t TP_WORLD_SIZE_TWO = 2;
constexpr uint32_t VERSION_2 = 2;
constexpr uint32_t HCOMMCNT_2 = 2;
constexpr int64_t MOE_EXPERT_MAX_NUM = 1024;
constexpr int64_t K_MAX = 16;
constexpr size_t SYSTEM_NEED_WORKSPACE = 16UL * 1024UL * 1024UL;
constexpr uint32_t WORKSPACE_ELEMENT_OFFSET = 512;
constexpr uint32_t RANK_LIST_NUM = 2;
constexpr int32_t HCCL_BUFFER_SIZE_DEFAULT = 200 * 1024 * 1024;  // Bytes
constexpr int64_t H_MIN = 1024;
constexpr int64_t H_MAX = 8192;
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
constexpr uint64_t TRIPLE = 3;
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint64_t FULL_MESH_DATA_ALIGN = 480UL;
constexpr uint64_t SCALE_EXPAND_IDX_BUFFER = 44UL;  // scale32B + 3*4expandIdx
constexpr uint64_t DOUBLE_DATA_BUFFER = 2UL;
constexpr uint64_t MAX_OUT_DTYPE_SIZE = 2UL;
constexpr uint64_t UB_ALIGN = 32UL;
constexpr int64_t ELASTIC_METAINFO_OFFSET = 4;
}  // namespace

namespace optiling {
static void PrintTilingDataInfo(const char *nodeName, ZbMoeDistributeDispatchTilingData &tilingData)
{
    OP_LOGD(nodeName, "epWorldSize is %u.", tilingData.moeDistributeDispatchInfo.epWorldSize);
    OP_LOGD(nodeName, "tpWorldSize is %u.", tilingData.moeDistributeDispatchInfo.tpWorldSize);
    OP_LOGD(nodeName, "epRankId is %u.", tilingData.moeDistributeDispatchInfo.epRankId);
    OP_LOGD(nodeName, "tpRankId is %u.", tilingData.moeDistributeDispatchInfo.tpRankId);
    OP_LOGD(nodeName, "expertShardType is %u.", tilingData.moeDistributeDispatchInfo.expertShardType);
    OP_LOGD(nodeName, "sharedExpertNum is %u.", tilingData.moeDistributeDispatchInfo.sharedExpertNum);
    OP_LOGD(nodeName, "sharedExpertRankNum is %u.", tilingData.moeDistributeDispatchInfo.sharedExpertRankNum);
    OP_LOGD(nodeName, "moeExpertNum is %u.", tilingData.moeDistributeDispatchInfo.moeExpertNum);
    OP_LOGD(nodeName, "quantMode is %u.", tilingData.moeDistributeDispatchInfo.quantMode);
    OP_LOGD(nodeName, "globalBs is %u.", tilingData.moeDistributeDispatchInfo.globalBs);
    OP_LOGD(nodeName, "bs is %u.", tilingData.moeDistributeDispatchInfo.bs);
    OP_LOGD(nodeName, "k is %u.", tilingData.moeDistributeDispatchInfo.k);
    OP_LOGD(nodeName, "h is %u.", tilingData.moeDistributeDispatchInfo.h);
    OP_LOGD(nodeName, "aivNum is %u.", tilingData.moeDistributeDispatchInfo.aivNum);
    OP_LOGD(nodeName, "totalUbSize is %lu.", tilingData.moeDistributeDispatchInfo.totalUbSize);
    OP_LOGD(nodeName, "totalWinSize is %lu.", tilingData.moeDistributeDispatchInfo.totalWinSize);
    OP_LOGD(nodeName, "hasElastic is %d.", tilingData.moeDistributeDispatchInfo.hasElasticInfo);
    OP_LOGD(nodeName, "zeroComputeExpertNum is %d", tilingData.moeDistributeDispatchInfo.zeroComputeExpertNum);
    OP_LOGD(nodeName, "cumSumUBMinValue is %d", tilingData.moeDistributeDispatchInfo.cumSumUBMinValue);
}

static bool CheckTensorDim(const gert::TilingContext *context, const char *nodeName, const bool isScales,
                           const uint32_t quantMode, const bool isActiveMask, const bool hasElasticInfo)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(nodeName, "xShape is null."), return false);
    OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE(nodeName, "xShape dims must be 2, but current dim num is %lu.",
                            xStorageShape->GetStorageShape().GetDimNum()),
                    return false);
    int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "x dim0 = %ld", xDim0);
    OP_LOGD(nodeName, "x dim1 = %ld", xDim1);

    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdStorageShape == nullptr, OP_LOGE(nodeName, "expertIdShape is null."), return false);
    OP_TILING_CHECK(expertIdStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE(nodeName, "expertIdShape dims must be 2, but current dim num is %lu.",
                            expertIdStorageShape->GetStorageShape().GetDimNum()),
                    return false);
    const int64_t expertIdDim0 = expertIdStorageShape->GetStorageShape().GetDim(0);
    const int64_t expertIdDim1 = expertIdStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "expertId dim0 = %ld", expertIdDim0);
    OP_LOGD(nodeName, "expertId dim1 = %ld", expertIdDim1);

    // 如果scales不为空进行shape维度检查
    if (isScales) {
        const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
        OP_TILING_CHECK(scalesStorageShape == nullptr, OP_LOGE(nodeName, "scalesShape is null."), return false);
        OP_TILING_CHECK(scalesStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                        OP_LOGE(nodeName, "scalesShape dims must be 2, but current dim num is %lu.",
                                scalesStorageShape->GetStorageShape().GetDimNum()),
                        return false);
        OP_LOGD(nodeName, "scales dim0 = %ld", scalesStorageShape->GetStorageShape().GetDim(0));
        OP_LOGD(nodeName, "scales dim1 = %ld", scalesStorageShape->GetStorageShape().GetDim(1));
    }

    if (isActiveMask) {
        const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
        OP_TILING_CHECK(xActiveMaskStorageShape == nullptr, OP_LOGE(nodeName, "xActiveMask shape is null."),
                        return false);
        const int64_t xActiveMaskDimNum = xActiveMaskStorageShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(
            ((xActiveMaskDimNum != ONE_DIM) && (xActiveMaskDimNum != TWO_DIMS)),
            OP_LOGE(nodeName, "xActiveMask shape dim must be 1 or 2, but current dim num is %ld.", xActiveMaskDimNum),
            return false);
        OP_TILING_CHECK((xActiveMaskStorageShape->GetStorageShape().GetDim(0) != xDim0),
                        OP_LOGE(nodeName, "The input of xActiveMask dim0 = %ld is not equal to x dim0 = %ld.",
                                xActiveMaskStorageShape->GetStorageShape().GetDim(0), xDim0),
                        return false);
        OP_TILING_CHECK(
            ((xActiveMaskDimNum == TWO_DIMS) && (xActiveMaskStorageShape->GetStorageShape().GetDim(1) != expertIdDim1)),
            OP_LOGE(nodeName, "The input of xActiveMask dim1 = %ld is not equal to expertId dim1 = %ld.",
                    xActiveMaskStorageShape->GetStorageShape().GetDim(1), expertIdDim1),
            return false);
    }
    if (hasElasticInfo) {
        const gert::StorageShape *elasticInfoStorageShape = context->GetOptionalInputShape(ELASTIC_INFO_INDEX);
        OP_TILING_CHECK(elasticInfoStorageShape == nullptr, OP_LOGE(nodeName, "elasticInfo is null."), return false);
        OP_TILING_CHECK(elasticInfoStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                        OP_LOGE(nodeName, "elasticInfo dim must be 1, but current dim num is %lu.",
                                elasticInfoStorageShape->GetStorageShape().GetDimNum()),
                        return false);
        OP_LOGD(nodeName, "elasticInfo dim0 = %ld", elasticInfoStorageShape->GetStorageShape().GetDim(0));
    }

    const gert::StorageShape *expandXStorageShape = context->GetOutputShape(OUTPUT_EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXStorageShape == nullptr, OP_LOGE(nodeName, "expandXShape is null."), return false);
    OP_TILING_CHECK(expandXStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE(nodeName, "expandXShape dims must be 2, but current dim num is %lu.",
                            expandXStorageShape->GetStorageShape().GetDimNum()),
                    return false);
    OP_LOGD(nodeName, "expandX dim0 = %ld", expandXStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "expandX dim1 = %ld", expandXStorageShape->GetStorageShape().GetDim(1));

    if (quantMode == DYNAMIC_SCALES) {
        const gert::StorageShape *dynamicScalesStorageShape = context->GetOutputShape(OUTPUT_DYNAMIC_SCALES_INDEX);
        OP_TILING_CHECK(dynamicScalesStorageShape == nullptr, OP_LOGE(nodeName, "dynamicScalesShape is null."),
                        return false);
        OP_TILING_CHECK(dynamicScalesStorageShape->GetStorageShape().GetDimNum() != DYNAMIC_SCALE_DIM_NUM,
                        OP_LOGE(nodeName, "dynamicScalesShape dims must be %u, but current dim num is %lu.",
                                DYNAMIC_SCALE_DIM_NUM, dynamicScalesStorageShape->GetStorageShape().GetDimNum()),
                        return false);
        OP_LOGD(nodeName, "dynamicScales dim0 = %ld", dynamicScalesStorageShape->GetStorageShape().GetDim(0));
    }

    const gert::StorageShape *assistInfoStorageShape = context->GetOutputShape(OUTPUT_ASSIST_INFO_INDEX);
    OP_TILING_CHECK(assistInfoStorageShape == nullptr, OP_LOGE(nodeName, "assistInfoShape is null."), return false);
    OP_TILING_CHECK(assistInfoStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE(nodeName, "assistInfoShape dims must be 1, but current dim num is %lu.",
                            assistInfoStorageShape->GetStorageShape().GetDimNum()),
                    return false);
    OP_LOGD(nodeName, "assistInfoForCombine dim0 = %ld", assistInfoStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *expertTokenNumsStorageShape = context->GetOutputShape(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OP_TILING_CHECK(expertTokenNumsStorageShape == nullptr, OP_LOGE(nodeName, "expertTokenNumsShape is null."),
                    return false);
    OP_TILING_CHECK(expertTokenNumsStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE(nodeName, "expertTokenNumsShape dims must be 1, but current dim num is %lu.",
                            expertTokenNumsStorageShape->GetStorageShape().GetDimNum()),
                    return false);
    OP_LOGD(nodeName, "expertTokenNums dim0 = %ld", expertTokenNumsStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *epRecvCountStorageShape = context->GetOutputShape(OUTPUT_EP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(epRecvCountStorageShape == nullptr, OP_LOGE(nodeName, "epRecvCountShape is null."), return false);
    OP_TILING_CHECK(epRecvCountStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE(nodeName, "epRecvCountShape dims must be 1, but current dim num is %lu.",
                            epRecvCountStorageShape->GetStorageShape().GetDimNum()),
                    return false);
    OP_LOGD(nodeName, "epRecvCount dim0 = %ld", epRecvCountStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *tpRecvCountStorageShape = context->GetOutputShape(OUTPUT_TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(tpRecvCountStorageShape == nullptr, OP_LOGE(nodeName, "tpRecvCountShape is null."), return false);
    OP_TILING_CHECK(tpRecvCountStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE(nodeName, "tpRecvCountShape dims must be 1, but current dim num is %lu.",
                            tpRecvCountStorageShape->GetStorageShape().GetDimNum()),
                    return false);
    OP_LOGD(nodeName, "tpRecvCount dim0 = %ld", tpRecvCountStorageShape->GetStorageShape().GetDim(0));

    return true;
}

static bool CheckTensorDataType(const gert::TilingContext *context, const char *nodeName, const bool isScales,
                                const uint32_t quantMode, const bool isActiveMask, const bool hasElasticInfo)
{
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE(nodeName, "xDesc is null."), return false);
    OP_TILING_CHECK((xDesc->GetDataType() != ge::DT_BF16) && (xDesc->GetDataType() != ge::DT_FLOAT16),
                    OP_LOGE(nodeName, "x dataType is invalid, dataType should be bf16 or float16, but is %d.",
                            static_cast<ge::DataType>(xDesc->GetDataType())),
                    return false);

    auto expertIdDesc = context->GetInputDesc(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdDesc == nullptr, OP_LOGE(nodeName, "expertIdDesc is null."), return false);
    OP_TILING_CHECK(expertIdDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(nodeName, "expertId dataType is invalid, dataType should be int32, but is %d.",
                            static_cast<ge::DataType>(expertIdDesc->GetDataType())),
                    return false);

    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(SCALES_INDEX);
        OP_TILING_CHECK(scalesDesc == nullptr, OP_LOGE(nodeName, "scalesDesc is null."), return false);
        OP_TILING_CHECK(scalesDesc->GetDataType() != ge::DT_FLOAT,
                        OP_LOGE(nodeName, "scales dataType is invalid, dataType should be float, but is %d.",
                                static_cast<ge::DataType>(scalesDesc->GetDataType())),
                        return false);
    }

    if (isActiveMask) {
        auto xActiveMaskDesc = context->GetOptionalInputDesc(X_ACTIVE_MASK_INDEX);
        OP_TILING_CHECK(xActiveMaskDesc == nullptr, OP_LOGE(nodeName, "xActiveMaskDesc is null."), return false);
        OP_TILING_CHECK(xActiveMaskDesc->GetDataType() != ge::DT_BOOL,
                        OP_LOGE(nodeName, "xActiveMask dataType is invalid, dataType should be bool, but is %d.",
                                static_cast<ge::DataType>(xActiveMaskDesc->GetDataType())),
                        return false);
    }

    if (hasElasticInfo) {
        auto elasticInfoDesc = context->GetOptionalInputDesc(ELASTIC_INFO_INDEX);
        OP_TILING_CHECK(elasticInfoDesc == nullptr, OP_LOGE(nodeName, "elasticInfoDesc is null."), return false);
        OP_TILING_CHECK(elasticInfoDesc->GetDataType() != ge::DT_INT32,
                        OP_LOGE(nodeName, "elasticInfoDesc dataType is invalid, dataType should be int32, but is %d.",
                                static_cast<ge::DataType>(elasticInfoDesc->GetDataType())),
                        return false);
    }

    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE(nodeName, "expandXDesc is null."), return false);
    if (quantMode != NO_SCALES) {
        OP_TILING_CHECK(expandXDesc->GetDataType() != ge::DT_INT8,
                        OP_LOGE(nodeName, "expandX dataType is invalid, dataType should be int8, but is %d.",
                                static_cast<ge::DataType>(expandXDesc->GetDataType())),
                        return false);
    } else {
        OP_TILING_CHECK(
            expandXDesc->GetDataType() != xDesc->GetDataType(),
            OP_LOGE(nodeName, "expandX dataType is invalid, dataType should be equal to x dataType %d, but is %d.",
                    static_cast<ge::DataType>(xDesc->GetDataType()),
                    static_cast<ge::DataType>(expandXDesc->GetDataType())),
            return false);
    }

    if (quantMode == DYNAMIC_SCALES) {
        auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
        OP_TILING_CHECK(dynamicScalesDesc == nullptr, OP_LOGE(nodeName, "dynamicScalesDesc is null."), return false);
        OP_TILING_CHECK(dynamicScalesDesc->GetDataType() != ge::DT_FLOAT,
                        OP_LOGE(nodeName, "dynamicScales dataType is invalid, dataType should be float, but is %d.",
                                static_cast<ge::DataType>(dynamicScalesDesc->GetDataType())),
                        return false);
    }

    auto assistInfoDesc = context->GetOutputDesc(OUTPUT_ASSIST_INFO_INDEX);
    OP_TILING_CHECK(assistInfoDesc == nullptr, OP_LOGE(nodeName, "assistInfoDesc is null."), return false);
    OP_TILING_CHECK(assistInfoDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(nodeName, "assistInfoForCombine dataType is invalid, dataType should be int32, but is %d.",
                            static_cast<ge::DataType>(assistInfoDesc->GetDataType())),
                    return false);

    auto expertTokenNumsDesc = context->GetOutputDesc(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OP_TILING_CHECK(expertTokenNumsDesc == nullptr, OP_LOGE(nodeName, "expertTokenNumsDesc is null."), return false);
    OP_TILING_CHECK(expertTokenNumsDesc->GetDataType() != ge::DT_INT64,
                    OP_LOGE(nodeName, "expertTokenNums dataType is invalid, dataType should be int64, but is %d.",
                            static_cast<ge::DataType>(expertTokenNumsDesc->GetDataType())),
                    return false);

    auto epRecvCountsDesc = context->GetOutputDesc(OUTPUT_EP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(epRecvCountsDesc == nullptr, OP_LOGE(nodeName, "epRecvCountsDesc is null."), return false);
    OP_TILING_CHECK(epRecvCountsDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(nodeName, "epRecvCounts dataType is invalid, dataType should be int32, but is %d.",
                            static_cast<ge::DataType>(epRecvCountsDesc->GetDataType())),
                    return false);

    auto tpRecvCountsDesc = context->GetOutputDesc(OUTPUT_TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(tpRecvCountsDesc == nullptr, OP_LOGE(nodeName, "tpRecvCountsDesc is null."), return false);
    OP_TILING_CHECK(tpRecvCountsDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(nodeName, "tpRecvCounts dataType is invalid, dataType should be int32, but is %d.",
                            static_cast<ge::DataType>(tpRecvCountsDesc->GetDataType())),
                    return false);

    return true;
}

static bool CheckTensorFormat(const gert::TilingContext *context, const char *nodeName, const bool isScales,
                              const uint32_t quantMode, const bool isActiveMask, const uint32_t hasElasticInfo)
{
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE(nodeName, "xDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(nodeName, "x format is invalid."), return false);

    auto expertIdDesc = context->GetInputDesc(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdDesc == nullptr, OP_LOGE(nodeName, "expertIdDesc is null."), return false);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(expertIdDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "expertId format is invalid."), return false);

    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(SCALES_INDEX);
        OP_TILING_CHECK(scalesDesc == nullptr, OP_LOGE(nodeName, "scalesDesc is null."), return false);
        OP_TILING_CHECK(
            static_cast<ge::Format>(ge::GetPrimaryFormat(scalesDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
            OP_LOGE(nodeName, "scales format is invalid."), return false);
    }

    if (isActiveMask) {
        auto xActiveMaskDesc = context->GetOptionalInputDesc(X_ACTIVE_MASK_INDEX);
        OP_TILING_CHECK(xActiveMaskDesc == nullptr, OP_LOGE(nodeName, "xActiveMaskDesc is null."), return false);
        OP_TILING_CHECK(
            static_cast<ge::Format>(ge::GetPrimaryFormat(xActiveMaskDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
            OP_LOGE(nodeName, "xActiveMask format is invalid."), return false);
    }

    if (static_cast<bool>(hasElasticInfo)) {
        auto elasticInfoDesc = context->GetOptionalInputDesc(ELASTIC_INFO_INDEX);
        OP_TILING_CHECK(elasticInfoDesc == nullptr, OP_LOGE(nodeName, "elasticInfoDesc is null."), return false);
        OP_TILING_CHECK(
            static_cast<ge::Format>(ge::GetPrimaryFormat(elasticInfoDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
            OP_LOGE(nodeName, "elasticInfo format is invalid."), return false);
    }

    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE(nodeName, "expandXDesc is null."), return false);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(expandXDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "expandX format is invalid."), return false);

    if (quantMode == DYNAMIC_SCALES) {
        auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
        OP_TILING_CHECK(dynamicScalesDesc == nullptr, OP_LOGE(nodeName, "dynamicScalesDesc is null."), return false);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(dynamicScalesDesc->GetStorageFormat())) ==
                            ge::FORMAT_FRACTAL_NZ,
                        OP_LOGE(nodeName, "dynamicScales format is invalid."), return false);
    }

    auto assistInfoDesc = context->GetOutputDesc(OUTPUT_ASSIST_INFO_INDEX);
    OP_TILING_CHECK(assistInfoDesc == nullptr, OP_LOGE(nodeName, "assistInfoDesc is null."), return false);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(assistInfoDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "assistInfoForCombine format is invalid."), return false);

    auto expertTokenNumsDesc = context->GetOutputDesc(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OP_TILING_CHECK(expertTokenNumsDesc == nullptr, OP_LOGE(nodeName, "expertTokenNumsDesc is null."), return false);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(expertTokenNumsDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "expertTokenNums format is invalid."), return false);

    auto epRecvCountsDesc = context->GetOutputDesc(OUTPUT_EP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(epRecvCountsDesc == nullptr, OP_LOGE(nodeName, "epRecvCountsDesc is null."), return false);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(epRecvCountsDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "epRecvCounts format is invalid."), return false);

    auto tpRecvCountsDesc = context->GetOutputDesc(OUTPUT_TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(tpRecvCountsDesc == nullptr, OP_LOGE(nodeName, "tpRecvCountsDesc is null."), return false);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(tpRecvCountsDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "tpRecvCounts format is invalid."), return false);

    return true;
}

static ge::graphStatus CheckAndSetGroupInfo(const gert::TilingContext *context, const char *nodeName,
                                            ZbMoeDistributeDispatchTilingData &tilingData)
{
    auto attrs = context->GetAttrs();
    // auto groupEpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    // auto groupTpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_TP_INDEX));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto tpRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_RANK_ID_INDEX);
    int64_t epWorldSize = *epWorldSizePtr;

    // 判空
    // OP_TILING_CHECK((groupEpPtr == nullptr) || (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
    //                     (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
    //                 OP_LOGE(nodeName, "groupEpPtr is null."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE(nodeName, "epWorldSizePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpWorldSizePtr == nullptr, OP_LOGE(nodeName, "tpWorldSizePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epRankIdPtr == nullptr, OP_LOGE(nodeName, "epRankIdPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpRankIdPtr == nullptr, OP_LOGE(nodeName, "tpRankIdPtr is null."), return ge::GRAPH_FAILED);

    // 判断是否有效
    OP_TILING_CHECK((epWorldSize < MIN_EP_WORLD_SIZE) || (epWorldSize > MAX_EP_WORLD_SIZE),
                    OP_LOGE(nodeName, "epWorldSize is invalid, only support [%ld, %ld], but got epWorldSize=%ld.",
                            MIN_EP_WORLD_SIZE, MAX_EP_WORLD_SIZE, epWorldSize),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*tpWorldSizePtr < 0) || (*tpWorldSizePtr > MAX_TP_WORLD_SIZE),
                    OP_LOGE(nodeName, "tpWorldSize is invalid, only support [0, %ld], but got tpWorldSize=%ld.",
                            MAX_TP_WORLD_SIZE, *tpWorldSizePtr),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*epRankIdPtr < 0) || (*epRankIdPtr >= epWorldSize),
                    OP_LOGE(nodeName, "epRankId is invalid, only support [0, %ld), but got epRankId=%ld.", epWorldSize,
                            *epRankIdPtr),
                    return ge::GRAPH_FAILED);
    if (*tpWorldSizePtr > 1) {
        OP_TILING_CHECK((*tpRankIdPtr < 0) || (*tpRankIdPtr >= *tpWorldSizePtr),
                        OP_LOGE(nodeName, "tpRankId is invalid, only support [0, %ld), but got tpRankId=%ld.",
                                *tpWorldSizePtr, *tpRankIdPtr),
                        return ge::GRAPH_FAILED);
        // OP_TILING_CHECK((groupTpPtr == nullptr) || (strnlen(groupTpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
        //                     (strnlen(groupTpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
        //                 OP_LOGE(nodeName, "groupTpPtr is null."), return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(
            *tpRankIdPtr != 0,
            OP_LOGE(nodeName, "tpRankId is invalid, NoTp mode only support 0, but got tpRankId=%ld.", *tpRankIdPtr),
            return ge::GRAPH_FAILED);
    }
    tilingData.moeDistributeDispatchInfo.epWorldSize = static_cast<uint32_t>(epWorldSize);
    tilingData.moeDistributeDispatchInfo.tpWorldSize = static_cast<uint32_t>(*tpWorldSizePtr);
    tilingData.moeDistributeDispatchInfo.epRankId = static_cast<uint32_t>(*epRankIdPtr);
    tilingData.moeDistributeDispatchInfo.tpRankId = static_cast<uint32_t>(*tpRankIdPtr);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAndSetExpertInfo(const gert::TilingContext *context, const char *nodeName,
                                             ZbMoeDistributeDispatchTilingData &tilingData)
{
    auto attrs = context->GetAttrs();
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto expertShardPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_SHARED_EXPERT_NUM_INDEX));
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    auto expertTokenNumsTypePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_EXPERT_TOKEN_NUMS_TYPE_INDEX));
    int64_t moeExpertNum = *moeExpertNumPtr;
    int64_t epWorldSize = *epWorldSizePtr;
    int64_t sharedExpertRankNum = *sharedExpertRankNumPtr;

    OP_TILING_CHECK(expertShardPtr == nullptr, OP_LOGE(nodeName, "expertShardPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertNumPtr == nullptr, OP_LOGE(nodeName, "sharedExpertNumPtr is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertRankNumPtr == nullptr, OP_LOGE(nodeName, "sharedExpertRankNumPtr is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(moeExpertNumPtr == nullptr, OP_LOGE(nodeName, "moeExpertNumPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(quantModePtr == nullptr, OP_LOGE(nodeName, "quantModePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertTokenNumsTypePtr == nullptr, OP_LOGE(nodeName, "expertTokenNumsTypePtr is null."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        *expertShardPtr != 0,
        OP_LOGE(nodeName, "expertShardType is invalid, only support 0, but got expertShardType=%ld.", *expertShardPtr),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*sharedExpertNumPtr < 0) || (*sharedExpertNumPtr > MAX_SHARED_EXPERT_NUM),
                    OP_LOGE(nodeName, "sharedExpertNum is invalid, only support [0, %ld], but got sharedExpertNum=%ld.",
                            MAX_SHARED_EXPERT_NUM, *sharedExpertNumPtr),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (sharedExpertRankNum < 0) || (sharedExpertRankNum >= epWorldSize),
        OP_LOGE(nodeName, "sharedExpertRankNum is invalid, only support [0, %ld), but got sharedExpertRankNum=%ld.",
                epWorldSize, sharedExpertRankNum),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((moeExpertNum <= 0) || (moeExpertNum > MOE_EXPERT_MAX_NUM),
                    OP_LOGE(nodeName, "moeExpertNum is invalid, only support (0, %ld], but got moeExpertNum=%ld.",
                            MOE_EXPERT_MAX_NUM, moeExpertNum),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (*quantModePtr < static_cast<int64_t>(NO_SCALES)) || (*quantModePtr > static_cast<int64_t>(DYNAMIC_SCALES)),
        OP_LOGE(nodeName, "quantMode is invalid, only support [0, %u], but got quantMode=%ld.", DYNAMIC_SCALES,
                *quantModePtr),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*expertTokenNumsTypePtr != 0) && (*expertTokenNumsTypePtr != 1),
                    OP_LOGE(nodeName, "expertTokenNumsType only support 0 or 1, but got expertTokenNumsType=%ld.",
                            *expertTokenNumsTypePtr),
                    return ge::GRAPH_FAILED);

    tilingData.moeDistributeDispatchInfo.expertShardType = static_cast<uint32_t>(*expertShardPtr);
    tilingData.moeDistributeDispatchInfo.sharedExpertNum = static_cast<uint32_t>(*sharedExpertNumPtr);
    tilingData.moeDistributeDispatchInfo.sharedExpertRankNum = static_cast<uint32_t>(sharedExpertRankNum);
    if (tilingData.moeDistributeDispatchInfo.sharedExpertRankNum == 0U) {
        if (tilingData.moeDistributeDispatchInfo.sharedExpertNum == 1U) {
            tilingData.moeDistributeDispatchInfo.sharedExpertNum = 0U;
        }
    }
    tilingData.moeDistributeDispatchInfo.moeExpertNum = static_cast<uint32_t>(moeExpertNum);
    tilingData.moeDistributeDispatchInfo.quantMode = static_cast<uint32_t>(*quantModePtr);
    tilingData.moeDistributeDispatchInfo.expertTokenNumsType = static_cast<uint32_t>(*expertTokenNumsTypePtr);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAndSetSpecialExpertInfo(const gert::TilingContext *context, const char *nodeName,
                                                    ZbMoeDistributeDispatchTilingData &tilingData, bool &isSetCommAlg)
{
    auto attrs = context->GetAttrs();
    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_COMM_ALG_INDEX));
    auto zeroExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_ZERO_EXPERT_NUM_INDEX));
    auto copyExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_COPY_EXPERT_NUM_INDEX));
    auto constExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_CONST_EXPERT_NUM_INDEX));
    int64_t moeExpertNum = *(attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX));
    int64_t zeroExpertNum = *zeroExpertNumPtr;
    int64_t copyExpertNum = *copyExpertNumPtr;
    int64_t constExpertNum = *constExpertNumPtr;
    int64_t zeroComputeExpertNum = zeroExpertNum + copyExpertNum + constExpertNum;

    // 判空
    OP_TILING_CHECK(zeroExpertNumPtr == nullptr, OP_LOGE(nodeName, "zeroExpertNumPtr is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(copyExpertNumPtr == nullptr, OP_LOGE(nodeName, "copyExpertNumPtr is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(constExpertNumPtr == nullptr, OP_LOGE(nodeName, "constExpertNumPtr is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(commAlgPtr == nullptr, OP_LOGE(nodeName, "commAlgPtr is nullptr."), return ge::GRAPH_FAILED);

    // 判断是否有效
    OP_TILING_CHECK((zeroExpertNum < 0),
                    OP_LOGE(nodeName, "zeroExpertNum less than 0, zeroExpertNum is %ld.", zeroExpertNum),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((copyExpertNum < 0),
                    OP_LOGE(nodeName, "copyExpertNum less than 0, copyExpertNum is %ld.", copyExpertNum),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((constExpertNum < 0),
                    OP_LOGE(nodeName, "constExpertNum less than 0, constExpertNum is %ld.", constExpertNum),
                    return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "zeroExpertNum=%ld,copyExpertNum= %ld, constExpertNum=%ld", zeroExpertNum, copyExpertNum,
            constExpertNum);
    OP_TILING_CHECK(
        zeroComputeExpertNum + moeExpertNum > INT32_MAX,
        OP_LOGE(nodeName,
                "zeroExpertNum[%ld] + copyExpertNum[%ld] + constExpertNum[%ld] + moeExpertNum[%ld] exceed INT32_MAX.",
                zeroExpertNum, copyExpertNum, constExpertNum, moeExpertNum),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (strlen(commAlgPtr) != 0) && (strcmp(commAlgPtr, "fullmesh_v1") != 0) &&
            (strcmp(commAlgPtr, "fullmesh_v2") != 0),
        OP_LOGE(nodeName,
                "Attr commAlg is invalid, current only support fullmesh_v1 and fullmesh_v2, but got commAlg = %s.",
                commAlgPtr),
        return ge::GRAPH_FAILED);

    isSetCommAlg = ((strcmp(commAlgPtr, "fullmesh_v2") == 0) ? true : false);
    OP_LOGD(nodeName, "ZbMoeDistributeDispatch isSetCommAlg = %d\n", isSetCommAlg);
    tilingData.moeDistributeDispatchInfo.zeroComputeExpertNum = static_cast<int32_t>(zeroComputeExpertNum);
    OP_LOGD(nodeName, "ZbMoeDistributeDispatch zeroComputeExpertNum = %d\n",
            tilingData.moeDistributeDispatchInfo.zeroComputeExpertNum);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAttrAndSetTilingData(const gert::TilingContext *context, const char *nodeName,
                                               ZbMoeDistributeDispatchTilingData &tilingData, bool &isSetCommAlg)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

    // 获取通信参数
    OP_TILING_CHECK(CheckAndSetGroupInfo(context, nodeName, tilingData) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Get hccl attr and set tiling data failed."), return ge::GRAPH_FAILED);

    // 获取expert
    OP_TILING_CHECK(CheckAndSetExpertInfo(context, nodeName, tilingData) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Get expert attr and set tiling data failed."), return ge::GRAPH_FAILED);

    // 获取特殊专家与commAlg
    OP_TILING_CHECK(CheckAndSetSpecialExpertInfo(context, nodeName, tilingData, isSetCommAlg) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Get special expert, commAlg attr and set tiling data failed."),
                    return ge::GRAPH_FAILED);

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto shmemPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXT_INFO_INDEX);

    int64_t moeExpertNum = *moeExpertNumPtr;
    int64_t epWorldSize = *epWorldSizePtr;
    int64_t sharedExpertRankNum = *sharedExpertRankNumPtr;

    uint32_t localMoeExpertNum = static_cast<uint32_t>(moeExpertNum) /
                                 (static_cast<uint32_t>(epWorldSize) - static_cast<uint32_t>(sharedExpertRankNum));
    uint32_t lastDim = localMoeExpertNum * static_cast<uint32_t>(epWorldSize);
    std::vector<int64_t> srcShapeDim = {1, lastDim};
    auto srcShape = ge::Shape(srcShapeDim);
    uint32_t cumSumUBMaxValue = 0;
    uint32_t cumSumUBMinValue = 0;
    AscendC::GetCumSumMaxMinTmpSize(srcShape, sizeof(float), true, true, cumSumUBMaxValue, cumSumUBMinValue);
    tilingData.moeDistributeDispatchInfo.cumSumUBMinValue = static_cast<uint32_t>(cumSumUBMinValue);
    OP_LOGD(nodeName, "lastDim = %d, ZbMoeDistributeDispatch cumSumUBMinValue = %d\n", lastDim,
            tilingData.moeDistributeDispatchInfo.cumSumUBMinValue);
    
    tilingData.moeDistributeDispatchInfo.shmemptr = static_cast<uint64_t>(*shmemPtr);

    return ge::GRAPH_SUCCESS;
}

static bool CheckSharedAttrs(const char *nodeName, const ZbMoeDistributeDispatchTilingData &tilingData)
{
    uint32_t sharedExpertNum = tilingData.moeDistributeDispatchInfo.sharedExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchInfo.sharedExpertRankNum;

    // 校验共享专家卡数和共享专家数是否只有一个为0
    OP_TILING_CHECK(
        (sharedExpertNum == 0U) && (sharedExpertRankNum > 0U),
        OP_LOGE(nodeName, "sharedExpertRankNum is invalid, only support 0 when sharedExpertNum is 0, but got %u.",
                sharedExpertRankNum),
        return false);
    OP_TILING_CHECK(
        (sharedExpertNum > 0U) && (sharedExpertRankNum == 0U),
        OP_LOGE(nodeName, "sharedExpertNum is invalid, only support 0 when sharedExpertRankNum is 0, but got %u.",
                sharedExpertNum),
        return false);

    if ((sharedExpertNum > 0U) && (sharedExpertRankNum > 0U)) {
        // 校验共享专家卡数能否整除共享专家数
        OP_TILING_CHECK(
            ((sharedExpertRankNum % sharedExpertNum) != 0U),
            OP_LOGE(nodeName,
                    "sharedExpertRankNum should be divisible by sharedExpertNum, but sharedExpertRankNum=%u, "
                    "sharedExpertNum=%u.",
                    sharedExpertRankNum, sharedExpertNum),
            return false);
    }

    return true;
}

static bool CheckCommAlgAttrs(const char *nodeName, const ZbMoeDistributeDispatchTilingData &tilingData,
                              bool isActiveMask, bool isSetCommAlg)
{
    uint32_t tpWorldSize = tilingData.moeDistributeDispatchInfo.tpWorldSize;
    uint32_t hasElasticInfo = tilingData.moeDistributeDispatchInfo.hasElasticInfo;
    int32_t zeroComputeExpertNum = tilingData.moeDistributeDispatchInfo.zeroComputeExpertNum;

    // 校验动态缩容和FullMesh_v2不能同时启用
    OP_TILING_CHECK((isSetCommAlg && hasElasticInfo),
                    OP_LOGE(nodeName, "Cannot support elasticInfo when comm_alg = fullmesh_v2"), return false);
    // 校验特殊专家和FullMesh_v2不能同时启用
    OP_TILING_CHECK((isSetCommAlg && (zeroComputeExpertNum > 0)),
                    OP_LOGE(nodeName, "Cannot support zeroComputeExpert when comm_alg = fullmesh_v2"), return false);
    // 校验ActiveMask和FullMesh_v2不能同时启用
    OP_TILING_CHECK((isSetCommAlg && isActiveMask),
                    OP_LOGE(nodeName, "Cannot support xActiveMask when comm_alg = fullmesh_v2"), return false);
    // 检查comm_alg和tpWorldSize是否冲突
    OP_TILING_CHECK(isSetCommAlg && (tpWorldSize == TP_WORLD_SIZE_TWO),
                    OP_LOGE(nodeName, "When comm_alg is fullmesh_v2, tp_world_size cannot be 2."), return false);

    return true;
}

static ge::graphStatus CheckAttrs(const gert::TilingContext *context, const char *nodeName,
                                  ZbMoeDistributeDispatchTilingData &tilingData, uint32_t &localMoeExpertNum,
                                  bool isActiveMask, bool isSetCommAlg)
{
    uint32_t epWorldSize = tilingData.moeDistributeDispatchInfo.epWorldSize;
    uint32_t tpWorldSize = tilingData.moeDistributeDispatchInfo.tpWorldSize;
    uint32_t moeExpertNum = tilingData.moeDistributeDispatchInfo.moeExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchInfo.sharedExpertRankNum;
    uint64_t shmemptr = tilingData.moeDistributeDispatchInfo.shmemptr;


    // 校验shmem地址不为空
    OP_TILING_CHECK(shmemptr == 0, OP_LOGE(nodeName, "shmemptr is invalid."), return false);
    
    OP_TILING_CHECK(!CheckSharedAttrs(nodeName, tilingData),
                    OP_LOGE(nodeName, "Check shared expert related attributes failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckCommAlgAttrs(nodeName, tilingData, isActiveMask, isSetCommAlg),
                    OP_LOGE(nodeName, "Check comm_alg related attributes failed."), return ge::GRAPH_FAILED);

    // 校验moe专家数量能否均分给多机
    localMoeExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum);
    OP_TILING_CHECK(moeExpertNum % (epWorldSize - sharedExpertRankNum) != 0,
                    OP_LOGE(nodeName,
                            "moeExpertNum should be divisible by (epWorldSize - sharedExpertRankNum), "
                            "but moeExpertNum=%u, epWorldSize=%u, sharedExpertRankNum=%u.",
                            moeExpertNum, epWorldSize, sharedExpertRankNum),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(localMoeExpertNum <= 0,
                    OP_LOGE(nodeName, "localMoeExpertNum is invalid, localMoeExpertNum = %u", localMoeExpertNum),
                    return ge::GRAPH_FAILED);
    // 校验tp=2时单个moe卡上专家数是否等于1
    OP_TILING_CHECK((tpWorldSize > 1) && (localMoeExpertNum > 1),
                    OP_LOGE(nodeName,
                            "Cannot support multi-moeExpert %u "
                            "in a rank when tpWorldSize = %u > 1",
                            localMoeExpertNum, tpWorldSize),
                    return ge::GRAPH_FAILED);
    // 校验tp=2时是否没有动态缩容参数
    OP_TILING_CHECK((tpWorldSize > 1) && (tilingData.moeDistributeDispatchInfo.hasElasticInfo),
                    OP_LOGE(nodeName,
                            "Cannot support elasticInfo"
                            " when tpWorldSize = %u > 1",
                            tpWorldSize),
                    return ge::GRAPH_FAILED);

    // 校验输入x的dim 0并设bs
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK((xDim0 > BS_UPPER_BOUND) || (xDim0 <= 0),
                    OP_LOGE(nodeName, "xDim0(BS) is invalid. Should be between [1, %ld], but got xDim0=%ld.",
                            BS_UPPER_BOUND, xDim0),
                    return ge::GRAPH_FAILED);
    tilingData.moeDistributeDispatchInfo.bs = static_cast<uint32_t>(xDim0);

    // 校验globalBS
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_GLOBAL_BS_INDEX);
    OP_TILING_CHECK(globalBsPtr == nullptr, OP_LOGE(nodeName, "globalBsPtr is nullptr."), return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "ZbMoeDistributeDispatch *globalBsPtr = %ld, bs = %ld, epWorldSize = %u\n", *globalBsPtr, xDim0,
            epWorldSize);
    OP_TILING_CHECK(
        (*globalBsPtr != 0) && ((*globalBsPtr < xDim0 * static_cast<int64_t>(epWorldSize)) ||
                                ((*globalBsPtr) % (static_cast<int64_t>(epWorldSize)) != 0)),
        OP_LOGE(nodeName,
                "globalBS is invalid, only "
                "support 0 or maxBs(maxBs is the largest bs on all ranks) * epWorldSize, but got globalBS=%ld, "
                "bs=%ld, epWorldSize=%u.",
                *globalBsPtr, xDim0, epWorldSize),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(((*globalBsPtr > (xDim0 * static_cast<int64_t>(epWorldSize))) && isSetCommAlg),
                    OP_LOGE(nodeName,
                            "Different bs on different rank cannot work when comm_alg = fullmesh_v2, globalBS=%ld, "
                            "bs=%ld, epWorldSize=%u.",
                            *globalBsPtr, xDim0, epWorldSize),
                    return ge::GRAPH_FAILED);
    if (*globalBsPtr == 0) {
        tilingData.moeDistributeDispatchInfo.globalBs = static_cast<uint32_t>(xDim0) * epWorldSize;
    } else {
        tilingData.moeDistributeDispatchInfo.globalBs = static_cast<uint32_t>(*globalBsPtr);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorShape(const gert::TilingContext *context, const char *nodeName,
                                        ZbMoeDistributeDispatchTilingData &tilingData, const uint32_t quantMode,
                                        const bool isScales, const bool isSharedExpert, const bool hasElasticInfo,
                                        const int64_t localMoeExpertNum)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

    auto zeroExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_ZERO_EXPERT_NUM_INDEX));
    auto copyExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_COPY_EXPERT_NUM_INDEX));
    auto constExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_CONST_EXPERT_NUM_INDEX));

    int64_t zeroExpertNum = *zeroExpertNumPtr;
    int64_t copyExpertNum = *copyExpertNumPtr;
    int64_t constExpertNum = *constExpertNumPtr;

    uint32_t A = 0U;
    uint32_t globalBs = tilingData.moeDistributeDispatchInfo.globalBs;
    uint32_t sharedExpertNum = tilingData.moeDistributeDispatchInfo.sharedExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchInfo.sharedExpertRankNum;

    // 校验输入x的维度1并设h, bs已校验过
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    const int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK((xDim1 < H_MIN) || (xDim1 > H_MAX),
                    OP_LOGE(nodeName, "xShape dims1(H) should be in [%ld, %ld], but got %ld.", H_MIN, H_MAX, xDim1),
                    return ge::GRAPH_FAILED);  // 32字节对齐
    tilingData.moeDistributeDispatchInfo.h = static_cast<uint32_t>(xDim1);

    // 校验expert_id的维度并设k
    int64_t moeExpertNum = static_cast<int64_t>(tilingData.moeDistributeDispatchInfo.moeExpertNum);
    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    const int64_t expertIdsDim0 = expertIdStorageShape->GetStorageShape().GetDim(0);
    const int64_t expertIdsDim1 = expertIdStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(xDim0 != expertIdsDim0,
                    OP_LOGE(nodeName,
                            "xShape's dim0 not equal to expertIdShape's dim0, "
                            "xShape's dim0 is %ld, expertIdShape's dim0 is %ld.",
                            xDim0, expertIdsDim0),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((expertIdsDim1 <= 0) || (expertIdsDim1 > K_MAX) ||
                        (expertIdsDim1 > moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum),
                    OP_LOGE(nodeName,
                            "expertIdShape's dim1(k) should be in (0, min(%ld, moeExpertNum + zeroExpertNum + "
                            "copyExpertNum + constExpertNum = %ld)], "
                            "but got expertIdShape's dim1=%ld.",
                            K_MAX, moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum, expertIdsDim1),
                    return ge::GRAPH_FAILED);
    tilingData.moeDistributeDispatchInfo.k = static_cast<uint32_t>(expertIdsDim1);

    // 校验scales的维度
    if (isScales) {
        const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
        const int64_t scalesDim0 = scalesStorageShape->GetStorageShape().GetDim(0);
        const int64_t scalesDim1 = scalesStorageShape->GetStorageShape().GetDim(1);
        OP_TILING_CHECK(scalesDim0 != (static_cast<int64_t>(sharedExpertNum) + moeExpertNum),
                        OP_LOGE(nodeName,
                                "scales's dim0 not equal to sharedExpertNum + moeExpertNum, "
                                "scales's dim0 is %ld, sharedExpertNum is %ld, moeExpertNum is %ld.",
                                scalesDim0, static_cast<int64_t>(sharedExpertNum), moeExpertNum),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(xDim1 != scalesDim1,
                        OP_LOGE(nodeName,
                                "scales's dim1 not equal to xShape's dim1, "
                                "xShape's dim1 is %ld, scales's dim1 is %ld.",
                                xDim1, scalesDim1),
                        return ge::GRAPH_FAILED);
    }
    uint32_t rankNumPerSharedExpert = 0;
    uint32_t epWorldSizeU32 = tilingData.moeDistributeDispatchInfo.epWorldSize;
    uint32_t maxBs = globalBs / epWorldSizeU32;
    uint32_t maxSharedGroupNum = 0;
    if ((sharedExpertNum != 0U) && (sharedExpertRankNum != 0U)) {  // 除零保护
        rankNumPerSharedExpert = sharedExpertRankNum / sharedExpertNum;
        maxSharedGroupNum = (epWorldSizeU32 + rankNumPerSharedExpert - 1U) / rankNumPerSharedExpert;
    }
    if (isSharedExpert) {  // 本卡为共享专家
        A = maxBs * maxSharedGroupNum;
    } else {  // 本卡为moe专家
        A = globalBs * std::min(localMoeExpertNum, expertIdsDim1);
    }

    // 校验elasticInfo的维度，并更新一下最大输出的值
    if (hasElasticInfo) {
        const gert::StorageShape *elasticInfoStorageShape = context->GetOptionalInputShape(ELASTIC_INFO_INDEX);
        const int64_t elasticInfoDim0 = elasticInfoStorageShape->GetStorageShape().GetDim(0);
        const int64_t epWorldSize = static_cast<int64_t>(tilingData.moeDistributeDispatchInfo.epWorldSize);

        OP_TILING_CHECK(elasticInfoDim0 != (ELASTIC_METAINFO_OFFSET + RANK_LIST_NUM * epWorldSize),
                        OP_LOGE(nodeName,
                                "elasticInfo's dim0 not equal to 4 + 2 * epWorldSize, "
                                "elasticInfo's dim0 is %ld, epWorldSize is %ld.",
                                elasticInfoDim0, epWorldSize),
                        return ge::GRAPH_FAILED);
        A = std::max(static_cast<int64_t>(maxBs * maxSharedGroupNum),
                     globalBs * std::min(localMoeExpertNum, expertIdsDim1));
    }

    // 校验expandX的维度
    int64_t tpWorldSize = static_cast<int64_t>(tilingData.moeDistributeDispatchInfo.tpWorldSize);
    const gert::StorageShape *expandXStorageShape = context->GetOutputShape(OUTPUT_EXPAND_X_INDEX);
    const int64_t expandXDim0 = expandXStorageShape->GetStorageShape().GetDim(0);
    const int64_t expandXDim1 = expandXStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(expandXDim0 < tpWorldSize * static_cast<int64_t>(A),
                    OP_LOGE(nodeName,
                            "expandX's dim0 not greater than or equal to A*tpWorldSize, "
                            "expandX's dim0 is %ld, A*tpWorldSize is %ld.",
                            expandXDim0, tpWorldSize * A),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(xDim1 != expandXDim1,
                    OP_LOGE(nodeName,
                            "expandX's dim1 not equal to xShape's dim1, "
                            "xShape's dim1 is %ld, expandX's dim1 is %ld.",
                            xDim1, expandXDim1),
                    return ge::GRAPH_FAILED);

    // 校验dynamicScales的维度
    if (quantMode != NO_SCALES) {
        const gert::StorageShape *dynamicScalesStorageShape = context->GetOutputShape(OUTPUT_DYNAMIC_SCALES_INDEX);
        const int64_t dynamicScalesDim0 = dynamicScalesStorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(
            dynamicScalesDim0 < static_cast<int64_t>(A) * tpWorldSize,
            OP_LOGE(
                nodeName,
                "dynamicScales's dim0 should be equal to or greater than A*tpWorldSize, dynamicScales's dim0 is %ld, "
                "A*tpWorldSize is %ld.",
                dynamicScalesDim0, A * tpWorldSize),
            return ge::GRAPH_FAILED);
    }

    // 校验assistInfo的维度
    // const gert::StorageShape *assistInfoStorageShape = context->GetOutputShape(OUTPUT_ASSIST_INFO_INDEX);
    // const int64_t assistInfoDim0 = assistInfoStorageShape->GetStorageShape().GetDim(0);
    // OP_TILING_CHECK(assistInfoDim0 < static_cast<int64_t>(A * TRIPLE),
    //                 OP_LOGE(nodeName,
    //                         "assistInfoDim0 < A * 3,"
    //                         " assistInfoDim0 is %ld, A * 3 is %ld.",
    //                         assistInfoDim0, static_cast<int64_t>(A * TRIPLE)),
    //                 return ge::GRAPH_FAILED);

    // 校验expertTokenNums的维度
    const gert::StorageShape *expertTokenNumsStorageShape = context->GetOutputShape(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    const int64_t expertTokenNumsDim0 = expertTokenNumsStorageShape->GetStorageShape().GetDim(0);

    if (hasElasticInfo) {
        OP_TILING_CHECK(expertTokenNumsDim0 != (localMoeExpertNum > 1 ? localMoeExpertNum : 1),
                        OP_LOGE(nodeName,
                                "elastic scaling expertTokenNums's Dim0 not equal to max(localMoeExpertNum,1), "
                                "expertTokenNumsDim0 is %ld, "
                                "localMoeExpertNum is %ld.",
                                expertTokenNumsDim0, localMoeExpertNum),
                        return ge::GRAPH_FAILED);
    } else if (isSharedExpert) {
        OP_TILING_CHECK(expertTokenNumsDim0 != 1,
                        OP_LOGE(nodeName, "shared expertTokenNums's dim0 %ld not equal to 1.", expertTokenNumsDim0),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(
            expertTokenNumsDim0 != localMoeExpertNum,
            OP_LOGE(nodeName,
                    "moe expertTokenNums's Dim0 not equal to localMoeExpertNum, expertTokenNumsDim0 is %ld, "
                    "localMoeExpertNum is %ld.",
                    expertTokenNumsDim0, localMoeExpertNum),
            return ge::GRAPH_FAILED);
    }

    // 校验epRecvCount和tpRecvCount的维度
    int64_t epWorldSize = static_cast<int64_t>(tilingData.moeDistributeDispatchInfo.epWorldSize);
    const gert::StorageShape *epRecvCountStorageShape = context->GetOutputShape(OUTPUT_EP_RECV_COUNTS_INDEX);
    const gert::StorageShape *tpRecvCountStorageShape = context->GetOutputShape(OUTPUT_TP_RECV_COUNTS_INDEX);
    const int64_t epRecvCountDim0 = epRecvCountStorageShape->GetStorageShape().GetDim(0);
    const int64_t tpRecvCountDim0 = tpRecvCountStorageShape->GetStorageShape().GetDim(0);
    int64_t epRecvCount = ((isSharedExpert) ? epWorldSize : epWorldSize * localMoeExpertNum);
    if (hasElasticInfo) {
        epRecvCount = std::max(epWorldSize, epWorldSize * localMoeExpertNum);
    }
    if (tpWorldSize == MAX_TP_WORLD_SIZE) {
        epRecvCount *= tpWorldSize;
    }
    OP_TILING_CHECK(
        epRecvCountDim0 < epRecvCount,
        OP_LOGE(
            nodeName,
            "dimension 0 of epRecvCount should be greater than or equal to epWorldSize * localMoeExpertNum * "
            "tpWorldSize, "
            "but dimension 0 of epRecvCount is %ld, epWorldSize is %ld, localMoeExpertNum is %ld, tpWorldSize is %ld.",
            epRecvCountDim0, epWorldSize, localMoeExpertNum, tpWorldSize),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        tpRecvCountDim0 != tpWorldSize,
        OP_LOGE(nodeName,
                "dimension 0 of tpRecvCount should be equal to tpWorldSize, but dimension 0 of tpRecvCount is %ld, "
                "tpWorldSize is %ld.",
                tpRecvCountDim0, tpWorldSize),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingCheckMoeDistributeDispatch(gert::TilingContext *context, const char *nodeName,
                                                        const bool isActiveMask, const bool isScales,
                                                        const bool hasElasticInfo, const uint32_t quantMode)
{
    OP_TILING_CHECK(!CheckTensorDim(context, nodeName, isScales, quantMode, isActiveMask, hasElasticInfo),
                    OP_LOGE(nodeName, "params shape is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorDataType(context, nodeName, isScales, quantMode, isActiveMask, hasElasticInfo),
                    OP_LOGE(nodeName, "params dataType is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorFormat(context, nodeName, isScales, quantMode, isActiveMask, hasElasticInfo),
                    OP_LOGE(nodeName, "params format is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static void CalTilingKey(uint64_t &tilingKey, const bool isScales, const uint32_t quantMode, const uint32_t tpWorldSize,
                         const bool isSetCommAlg)
{
    tilingKey += static_cast<uint64_t>(quantMode);
    if (isScales) {
        tilingKey += static_cast<uint64_t>(TILINGKEY_SCALES);
    }
    if (tpWorldSize == TP_WORLD_SIZE_TWO) {
        tilingKey += static_cast<uint64_t>(TILINGKEY_TP_WORLD_SIZE);
    }
    if (isSetCommAlg) {
        tilingKey += static_cast<uint64_t>(TILINGKEY_COMM_ALG);
    }

    return;
}

static void SetHcommCfg(const gert::TilingContext *context, ZbMoeDistributeDispatchTilingData *tiling)
{
    auto attrs = context->GetAttrs();
    // auto groupEpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    // auto groupTpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_TP_INDEX));
    std::string groupEp = "";
    std::string groupTp = "";
    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "ZbMoeDistributeDispatch groupEp = %s, groupTp = %s", groupEp.c_str(), groupTp.c_str());
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

static ge::graphStatus CheckWinSize(ZbMoeDistributeDispatchTilingData &tilingData, const char *nodeName,
                                    const bool isSetCommAlg, uint32_t &localMoeExpertNum)
{
    // uint64_t maxWindowSize = Mc2TilingUtils::GetMaxWindowSize();
    uint16_t defaultWindowSize = 4000;
    const uint64_t maxWindowSize = static_cast<uint64_t>(defaultWindowSize) * 1024UL * 1024UL;
    uint32_t sharedExpertNum = tilingData.moeDistributeDispatchInfo.sharedExpertNum;
    uint64_t h = static_cast<uint64_t>(tilingData.moeDistributeDispatchInfo.h);
    uint64_t k = static_cast<uint64_t>(tilingData.moeDistributeDispatchInfo.k);
    uint64_t epWorldSize = static_cast<uint64_t>(tilingData.moeDistributeDispatchInfo.epWorldSize);
    uint64_t maxBs = static_cast<uint64_t>(tilingData.moeDistributeDispatchInfo.globalBs) / epWorldSize;
    // combine数据区 token首地址对齐512
    uint64_t tokenNeedSizeCombine = ((h * MAX_OUT_DTYPE_SIZE + WIN_ADDR_ALIGN - 1UL) / WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    // dispatch数据区 token首对齐512，有效token长度h_align_32b + scale(32b) + 三元组(3*4b)
    uint64_t tokenActualLen =
        ((h * MAX_OUT_DTYPE_SIZE + UB_ALIGN - 1UL) / UB_ALIGN) * UB_ALIGN + SCALE_EXPAND_IDX_BUFFER;
    uint64_t tokenNeedSizeDispatch = 0;
    if (isSetCommAlg) {
        tokenNeedSizeDispatch = ((tokenActualLen + FULL_MESH_DATA_ALIGN - 1UL) / FULL_MESH_DATA_ALIGN) * WIN_ADDR_ALIGN;
    } else {
        tokenNeedSizeDispatch = ((tokenActualLen + WIN_ADDR_ALIGN - 1UL) / WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    }
    uint64_t actualSize = ((maxBs * tokenNeedSizeDispatch * epWorldSize * static_cast<uint64_t>(localMoeExpertNum)) +
                           (maxBs * tokenNeedSizeCombine * (k + static_cast<uint64_t>(sharedExpertNum)))) *
                          DOUBLE_DATA_BUFFER;
    OP_TILING_CHECK(
        (actualSize > maxWindowSize),
        OP_LOGE(
            nodeName,
            "Shmem_Buff is too SMALL, maxBs = %lu, h = %lu, epWorldSize = %lu,"
            " localMoeExpertNum = %u, sharedExpertNum = %u, tokenNeedSizeDispatch = %lu, tokenNeedSizeCombine = %lu,"
            " k = %lu, NEEDED_HCCL_BUFFSIZE(((maxBs * tokenNeedSizeDispatch * ep_worldsize * localMoeExpertNum) +"
            " (maxBs * tokenNeedSizeCombine * (k + sharedExpertNum))) * 2) = %luMB,"
            " Shmem_Buff=%luMB.",
            maxBs, h, epWorldSize, localMoeExpertNum, sharedExpertNum, tokenNeedSizeDispatch, tokenNeedSizeCombine, k,
            actualSize / MB_SIZE + 1UL, maxWindowSize / MB_SIZE),
        return ge::GRAPH_FAILED);
    tilingData.moeDistributeDispatchInfo.totalWinSize = maxWindowSize;
    OP_LOGD(nodeName, "windowSize = %lu", maxWindowSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context, const char *nodeName)
{
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + static_cast<size_t>(WORKSPACE_ELEMENT_OFFSET * aivNum * aivNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetAivInfo(gert::TilingContext *context, ZbMoeDistributeDispatchTilingData *tilingData,
                                  const char *nodeName)
{
    uint32_t blockDim = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(blockDim);
    context->SetScheduleMode(1);  // 设置为batch mode模式, 所有核同时启动
    tilingData->moeDistributeDispatchInfo.totalUbSize = ubSize;
    tilingData->moeDistributeDispatchInfo.aivNum = aivNum;
    OP_LOGD(nodeName, "blockDim=%u, aivNum=%u, ubSize=%lu", blockDim, aivNum, ubSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetParamsAndSetTilingData(const gert::TilingContext *context,
                                                 ZbMoeDistributeDispatchTilingData *tilingData, bool &isActiveMask,
                                                 bool &isScales, bool &hasElasticInfo)
{
    const char *nodeName = context->GetNodeName();
    // 获取scales
    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
    isScales = (scalesStorageShape != nullptr);

    // 获取xActiveMask
    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
    isActiveMask = (xActiveMaskStorageShape != nullptr);
    tilingData->moeDistributeDispatchInfo.isTokenMask =
        ((isActiveMask) && (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == ONE_DIM));
    tilingData->moeDistributeDispatchInfo.isExpertMask =
        ((isActiveMask) && (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS));

    // 获取elasticInfo
    const gert::StorageShape *elasticInfoStorageShape = context->GetOptionalInputShape(ELASTIC_INFO_INDEX);
    hasElasticInfo = (elasticInfoStorageShape != nullptr);
    tilingData->moeDistributeDispatchInfo.hasElasticInfo = hasElasticInfo;
    uint32_t quantMode = tilingData->moeDistributeDispatchInfo.quantMode;

    // 检查quantMode和scales是否匹配
    OP_TILING_CHECK(quantMode == STATIC_SCALES, OP_LOGE(nodeName, "cannot support static quant now."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((isScales && (quantMode == NO_SCALES)) || ((!isScales) && (quantMode == STATIC_SCALES)),
                    OP_LOGE(nodeName, "quant mode and scales not match, isScales is %d, quantMode is %u.",
                            static_cast<int32_t>(isScales), quantMode),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeDispatchA3TilingFuncImpl(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    ZbMoeDistributeDispatchTilingData *tilingData = context->GetTilingData<ZbMoeDistributeDispatchTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    bool isScales = false;
    bool isActiveMask = false;
    bool hasElasticInfo = false;
    bool isSetCommAlg = false;
    uint32_t localMoeExpertNum = 1;
    OP_LOGI(nodeName, "Enter ZbMoeDistributeDispatch tiling check func.");

    // 获取入参属性
    OP_TILING_CHECK(GetAttrAndSetTilingData(context, nodeName, *tilingData, isSetCommAlg) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Get attr and set tiling data failed."), return ge::GRAPH_FAILED);
    uint32_t quantMode = tilingData->moeDistributeDispatchInfo.quantMode;
    OP_TILING_CHECK(
        GetParamsAndSetTilingData(context, tilingData, isActiveMask, isScales, hasElasticInfo) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Get params and set tiling data failed."), return ge::GRAPH_FAILED);

    // 检查输入输出的dim、format、dataType
    OP_TILING_CHECK(TilingCheckMoeDistributeDispatch(context, nodeName, isActiveMask, isScales, hasElasticInfo,
                                                     quantMode) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling check param failed."), return ge::GRAPH_FAILED);

    // 检查属性的取值是否合法
    OP_TILING_CHECK(
        CheckAttrs(context, nodeName, *tilingData, localMoeExpertNum, isActiveMask, isSetCommAlg) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check attr failed."), return ge::GRAPH_FAILED);

    uint32_t epRankId = tilingData->moeDistributeDispatchInfo.epRankId;
    uint32_t sharedExpertNum = tilingData->moeDistributeDispatchInfo.sharedExpertNum;
    uint32_t sharedExpertRankNum = tilingData->moeDistributeDispatchInfo.sharedExpertRankNum;
    bool isSharedExpert = (epRankId < sharedExpertRankNum);

    // 检查shape各维度并赋值h,k
    OP_TILING_CHECK(CheckTensorShape(context, nodeName, *tilingData, quantMode, isScales, isSharedExpert,
                                     hasElasticInfo, static_cast<int64_t>(localMoeExpertNum)) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check tensor shape failed."), return ge::GRAPH_FAILED);

    // 校验win区大小
    OP_TILING_CHECK(CheckWinSize(*tilingData, nodeName, isSetCommAlg, localMoeExpertNum) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling check window size failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetWorkSpace(context, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling set workspace failed."), return ge::GRAPH_FAILED);

    SetHcommCfg(context, tilingData);

    uint64_t tilingKey = INIT_TILINGKEY;
    uint32_t tpWorldSize = tilingData->moeDistributeDispatchInfo.tpWorldSize;
    CalTilingKey(tilingKey, isScales, quantMode, tpWorldSize, isSetCommAlg);
    OP_LOGD(nodeName, "tilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);

    OP_TILING_CHECK(SetAivInfo(context, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling set aiv info failed."), return ge::GRAPH_FAILED);

    PrintTilingDataInfo(nodeName, *tilingData);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ZbMoeDistributeDispatchTilingFunc(gert::TilingContext *context)
{
    ge::graphStatus ret;
    ret = MoeDistributeDispatchA3TilingFuncImpl(context);
    return ret;
}

struct MoeDistributeDispatchCompileInfo {};
static ge::graphStatus TilingParseForZbMoeDistributeDispatch(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ZbMoeDistributeDispatch)
    .Tiling(ZbMoeDistributeDispatchTilingFunc)
    .TilingParse<MoeDistributeDispatchCompileInfo>(TilingParseForZbMoeDistributeDispatch);
}  // namespace optiling
