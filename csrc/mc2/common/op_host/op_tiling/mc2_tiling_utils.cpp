/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mc2_tiling_utils.cpp
 * \brief
 */

#include <cstdlib>

#include "graph/utils/type_utils.h"
#include "mc2_hcom_topo_info.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_strategy.h"
#include "mc2_tiling_utils.h"

namespace mc2tiling {
namespace {
constexpr char DEUBG_MODE_ENV[] = "ASCEND_MC2_DEBUG_MODE";
constexpr char DEUBG_COMM_ALG_ENV[] = "ASCEND_MC2_DEBUG_COMM_ALG";
constexpr char DEUBG_STEP_SIZE_ENV[] = "ASCEND_MC2_DEBUG_STEP_SIZE";
constexpr char HCCL_BUFFSIZE[] = "HCCL_BUFFSIZE";
} // namespace
uint8_t Mc2TilingUtils::GetDebugMode()
{
    auto debugModeEnv = getenv(DEUBG_MODE_ENV);
    uint8_t debugMode = 0;
    if (debugModeEnv != nullptr) {
        debugMode = static_cast<uint8_t>(std::atoi(debugModeEnv));
    }
    OP_LOGI("", "Get ASCEND_MC2_DEBUG_MODE is %u", debugMode);
    return debugMode;
}

uint8_t Mc2TilingUtils::GetDebugCommAlg()
{
    auto debugCommAlgEnv = getenv(DEUBG_COMM_ALG_ENV);
    uint8_t debugCommAlg = 0;
    if (debugCommAlgEnv != nullptr) {
        debugCommAlg = static_cast<uint8_t>(std::atoi(debugCommAlgEnv));
    }
    OP_LOGI("", "Get ASCEND_MC2_DEBUG_COMM_ALG is %u", debugCommAlg);
    return debugCommAlg;
}

uint8_t Mc2TilingUtils::GetDebugStepSize()
{
    auto debugStepSizeEnv = getenv(DEUBG_STEP_SIZE_ENV);
    uint8_t debugStepSize = 0;
    if (debugStepSizeEnv != nullptr) {
        debugStepSize = static_cast<uint8_t>(std::atoi(debugStepSizeEnv));
    }
    OP_LOGI("", "Get ASCEND_MC2_DEBUG_STEP_SIZE is %u", debugStepSize);
    return debugStepSize;
}

matmul_tiling::DataType ConvertGeTypeToMmType(const std::string &opName, ge::DataType type)
{
    static const std::map<ge::DataType, matmul_tiling::DataType> GE_TO_MM_MAP = {
        {ge::DT_BF16, matmul_tiling::DataType::DT_BFLOAT16},
        {ge::DT_FLOAT16, matmul_tiling::DataType::DT_FLOAT16},
        {ge::DT_FLOAT, matmul_tiling::DataType::DT_FLOAT},
        {ge::DT_HIFLOAT8, matmul_tiling::DataType::DT_HIFLOAT8},
        {ge::DT_FLOAT8_E4M3FN, matmul_tiling::DataType::DT_FLOAT8_E4M3FN},
        {ge::DT_FLOAT8_E5M2, matmul_tiling::DataType::DT_FLOAT8_E5M2},
    };

    auto iterator = GE_TO_MM_MAP.find(type);
    if (iterator != GE_TO_MM_MAP.end()) {
        return iterator->second;
    }

    OP_LOGI(opName, "cannot find matmul_tiling datatype according to ge datatype: %d", static_cast<int32_t>(type));
    return matmul_tiling::DataType::DT_MAX;
}

ge::DataType ConvertMmTypeToGeType(const std::string &opName, matmul_tiling::DataType type)
{
    static const std::map<matmul_tiling::DataType, ge::DataType> MM_TO_GE_MAP = {
        {matmul_tiling::DataType::DT_BFLOAT16, ge::DT_BF16},
        {matmul_tiling::DataType::DT_FLOAT16, ge::DT_FLOAT16},
        {matmul_tiling::DataType::DT_FLOAT, ge::DT_FLOAT},
        {matmul_tiling::DataType::DT_HIFLOAT8, ge::DT_HIFLOAT8},
        {matmul_tiling::DataType::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN},
        {matmul_tiling::DataType::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2},
    };

    auto iterator = MM_TO_GE_MAP.find(type);
    if (iterator != MM_TO_GE_MAP.end()) {
        return iterator->second;
    }

    OP_LOGI(opName, "cannot find ge datatype according to  matmul_tiling datatype: %d", static_cast<int32_t>(type));
    return ge::DT_MAX;
}

uint64_t GetDataTypeSize(const std::string &opName, ge::DataType type)
{
    static const std::map<ge::DataType, int64_t> DATA_TYPE_SIZE_MAP = {
        {ge::DT_BF16, 2},     {ge::DT_FLOAT16, 2},       {ge::DT_FLOAT, 4},
        {ge::DT_HIFLOAT8, 1}, {ge::DT_FLOAT8_E4M3FN, 1}, {ge::DT_FLOAT8_E5M2, 1},
    };

    auto iterator = DATA_TYPE_SIZE_MAP.find(type);
    if (iterator != DATA_TYPE_SIZE_MAP.end()) {
        return iterator->second;
    }

    OP_LOGI(opName, "cannot find datatype size according to ge datatype: %d", static_cast<int32_t>(type));
    return 0;
}

HcclDataType ConvertGeTypeToHcclType(const std::string &opName, ge::DataType type)
{
    static const std::map<ge::DataType, HcclDataType> HCCL_DATA_TYPE_MAP = {
        {ge::DataType::DT_INT8, HcclDataType::HCCL_DATA_TYPE_INT8},
        {ge::DataType::DT_UINT8, HcclDataType::HCCL_DATA_TYPE_UINT8},
        {ge::DataType::DT_INT16, HcclDataType::HCCL_DATA_TYPE_INT16},
        {ge::DataType::DT_UINT16, HcclDataType::HCCL_DATA_TYPE_UINT16},
        {ge::DataType::DT_INT32, HcclDataType::HCCL_DATA_TYPE_INT32},
        {ge::DataType::DT_UINT32, HcclDataType::HCCL_DATA_TYPE_UINT32},
        {ge::DataType::DT_FLOAT16, HcclDataType::HCCL_DATA_TYPE_FP16},
        {ge::DataType::DT_FLOAT, HcclDataType::HCCL_DATA_TYPE_FP32},
        {ge::DataType::DT_BF16, HcclDataType::HCCL_DATA_TYPE_BFP16},
        {ge::DataType::DT_HIFLOAT8, HcclDataType::HCCL_DATA_TYPE_HIF8},
        {ge::DataType::DT_FLOAT8_E4M3FN, HcclDataType::HCCL_DATA_TYPE_FP8E4M3},
        {ge::DataType::DT_FLOAT8_E5M2, HcclDataType::HCCL_DATA_TYPE_FP8E5M2},
    };

    auto iterator = HCCL_DATA_TYPE_MAP.find(type);
    if (iterator != HCCL_DATA_TYPE_MAP.end()) {
        return iterator->second;
    }

    OP_LOGI(opName, "cannot find HcclDataType according to ge datatype: %d", static_cast<int32_t>(type));
    return HcclDataType::HCCL_DATA_TYPE_RESERVED;
}

bool CheckSuppportedFormat(ge::Format format)
{
    static const std::set<ge::Format> SUPPORT_FORMAT_SET = {ge::FORMAT_ND};

    return (SUPPORT_FORMAT_SET.count(format) != 0);
}

bool IsDeterministic()
{
    if (getenv(HCCL_DETERMINISTIC) == nullptr) {
        return false;
    }
    std::string envStr(getenv(HCCL_DETERMINISTIC));
    std::transform(envStr.begin(), envStr.end(), envStr.begin(), ::toupper);
    if (envStr == "FALSE") {
        return false;
    }
    OP_LOGI("MatmulReduceScatter", "Set HCCL_DETERMINISTIC is true");
    return true;
}

uint8_t Mc2GetCommAlgo(int64_t rankDim, uint64_t mValue, const char *group, const gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(context->GetNodeName(), "fail to get platform info!");
        return COMM_ALG_DEFAULT;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
        return COMM_ALG_FULL_MESH;
    }

    auto debugCommAlg = mc2tiling::Mc2TilingUtils::GetDebugCommAlg();
    if (rankDim == 2) { // 如果是2p
        if ((debugCommAlg != 0) && (debugCommAlg != COMM_ALG_FULL_MESH)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context->GetNodeName(), "debugCommAlg", std::to_string(debugCommAlg).c_str(),
                "The value of debugCommAlg must be 0 or COMM_ALG_FULL_MESH when rankDim is 2");
            return COMM_ALG_DEFAULT;
        }
        return COMM_ALG_FULL_MESH;
    } else if (rankDim <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "rankDim", std::to_string(rankDim).c_str(),
                                              "The value of rankDim must be positive");
        return COMM_ALG_DEFAULT;
    }

    uint32_t commSets = 0;
    if (Mc2Hcom::MC2HcomTopology::TryGetGroupTopoType(group, &commSets) != HCCL_SUCCESS) {
        OP_LOGW(context->GetNodeName(), " GroupTopoInfo not set.");
        return COMM_ALG_DEFAULT;
    }
    OP_LOGD(context->GetNodeName(), " comm_sets from TopoInfo is %u, COMM_MESH is %u", commSets, COMM_MESH);

    // 如果平台只支持fullmesh
    if (commSets == COMM_MESH) {
        if ((debugCommAlg != 0) && (debugCommAlg != COMM_ALG_FULL_MESH)) { // 环境变量设置非fullmesh
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context->GetNodeName(), "debugCommAlg", std::to_string(debugCommAlg).c_str(),
                "The value of debugCommAlg must be 0 or COMM_ALG_FULL_MESH when commSets is COMM_MESH");
            return COMM_ALG_DEFAULT;
        }
        return COMM_ALG_FULL_MESH;
    }

    // reducescatter支持doublering或switch
    if (debugCommAlg != 0) { // 环境变量设置非doublering/switch
        if ((debugCommAlg != COMM_ALG_DOUBLE_RING) && (debugCommAlg != COMM_ALG_SWITCH_WING)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context->GetNodeName(), "debugCommAlg", std::to_string(debugCommAlg).c_str(),
                "The value of debugCommAlg must be COMM_ALG_DOUBLE_RING or COMM_ALG_SWITCH_WING");
            return COMM_ALG_DEFAULT;
        }
        return debugCommAlg;
    }
    if ((mValue % CHECK_VALUE_ODD != 0) || (mValue % static_cast<uint64_t>(rankDim) != 0)) {
        OP_LOGW(context->GetNodeName(), " m value is odd or cannot be divided by rankDim.");
        return COMM_ALG_DEFAULT;
    }
    return COMM_ALG_DOUBLE_RING;
}

bool CheckRankSize(const NpuArch npuArch, const uint32_t rankSize)
{
    static const std::map<NpuArch, std::set<uint32_t>> SUPPORT_RANK_SIZE_SET = {
        {NpuArch::DAV_2002, {1, 2, 4}},
        {NpuArch::DAV_2201, {1, 2, 4, 8}},
        {NpuArch::DAV_3510, {1, 2, 4, 8, 16, 32, 64}},
    };
    auto it = SUPPORT_RANK_SIZE_SET.find(npuArch);
    if (it != SUPPORT_RANK_SIZE_SET.end()) {
        return it->second.count(rankSize) != 0;
    }

    return false;
}

bool CheckDataTypeVaild(ge::DataType type, std::initializer_list<ge::DataType> supportDtypeList)
{
    return std::find(supportDtypeList.begin(), supportDtypeList.end(), type) != supportDtypeList.end();
}

void UpdateMatmulV3Args(optiling::mc2_matmul_v3_advanced::Mc2MatMulV3Args &mmV3Args, const TilingArgs &args,
                        const char *opName)
{
    mmV3Args.opName = opName;
    mmV3Args.isATrans = args.isATrans;
    mmV3Args.isBTrans = args.isBTrans;
    mmV3Args.isHf32 = false;
    mmV3Args.hasBias = args.isBias;
    mmV3Args.aType = args.geAType;
    mmV3Args.bType = args.geBType;
    mmV3Args.cType = args.geCType;
    mmV3Args.biasType = args.geBiasType;
    mmV3Args.aFormat = ge::FORMAT_ND;
    mmV3Args.bFormat = ge::FORMAT_ND;
    mmV3Args.outFormat = ge::FORMAT_ND;
    mmV3Args.mValue = args.mValue;
    mmV3Args.nValue = args.nValue;
    mmV3Args.kValue = args.kValue;
    mmV3Args.aDtypeSize = GetDataTypeSize(opName, mmV3Args.aType);
    mmV3Args.bDtypeSize = GetDataTypeSize(opName, mmV3Args.bType);
}

ge::graphStatus GetMatmulV3PriorityPolicy(const NpuArch npuArch, std::vector<int32_t> &priorities, const char *opName)
{
    const static std::map<NpuArch, std::vector<int32_t>> MATMUL_V3_PRIOR_MAP = {
        {NpuArch::DAV_3510, {optiling::mc2_matmul_v3_advanced::strategy::BASE}},
    };
    if (MATMUL_V3_PRIOR_MAP.find(npuArch) != MATMUL_V3_PRIOR_MAP.end()) {
        priorities = MATMUL_V3_PRIOR_MAP.at(npuArch);
    }

    if (priorities.empty()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "npuArch", std::to_string(static_cast<uint32_t>(npuArch)).c_str(),
                                              "The value of npuArch must be within the supported range.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

uint32_t Mc2TilingUtils::GetCommSets(const char *group)
{
    uint32_t commSets = 0;
    if (Mc2Hcom::MC2HcomTopology::TryGetGroupTopoType(group, &commSets) != HCCL_SUCCESS) {
        OP_LOGW("", " GroupTopoInfo not set.");
        return COMM_UNDEFINED;
    }
    OP_LOGI("", "Get commSets is %u", commSets);
    return commSets;
}

ge::graphStatus Mc2TilingUtils::CommonParamCheck(const gert::TilingContext *context)
{
    const gert::StorageShape *aShape = context->GetInputShape(0);
    const gert::StorageShape *bShape = context->GetInputShape(1);
    OP_TILING_CHECK(aShape == nullptr || bShape == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "aShape, bShape"), return ge::GRAPH_FAILED);

    uint64_t aShapeDimNum = aShape->GetStorageShape().GetDimNum();
    uint64_t bShapeDimNum = bShape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(aShapeDimNum != 2 || bShapeDimNum != 2,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        context->GetNodeName(), "aShape, bShape",
                        (std::to_string(aShapeDimNum) + "D, " + std::to_string(bShapeDimNum) + "D").c_str(), "2D"),
                    return ge::GRAPH_FAILED);

    auto aTensor = context->GetInputDesc(0);
    auto bTensor = context->GetInputDesc(1);
    auto output = context->GetOutputDesc(0);
    OP_TILING_CHECK(aTensor == nullptr || bTensor == nullptr || output == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "aTensor, bTensor, output"),
                    return ge::GRAPH_FAILED);

    auto aShapeFormat = aTensor->GetStorageFormat();
    auto bShapeFormat = bTensor->GetStorageFormat();
    auto outputFormat = output->GetStorageFormat();
    OP_TILING_CHECK(aShapeFormat != outputFormat,
                    OP_LOGE_FOR_INVALID_FORMAT(context->GetNodeName(), "aShape, output",
                                               ge::TypeUtils::FormatToAscendString(aShapeFormat).GetString(),
                                               ge::TypeUtils::FormatToAscendString(outputFormat).GetString()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (mc2tiling::SUPPORTED_FORMAT.count(aShapeFormat) == 0 || mc2tiling::SUPPORTED_FORMAT.count(bShapeFormat) == 0),
        OP_LOGE_FOR_INVALID_FORMAT(context->GetNodeName(), "aShape, bShape",
                                   ge::TypeUtils::FormatToAscendString(aShapeFormat).GetString(), "ND"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

mc2tiling::HcclDataType Mc2TilingUtils::GetDataType(ge::DataType type)
{
    if (mc2tiling::HCCL_DATA_TYPE.find(type) != mc2tiling::HCCL_DATA_TYPE.end()) {
        return mc2tiling::HCCL_DATA_TYPE.at(type);
    }
    return mc2tiling::HcclDataType::HCCL_DATA_TYPE_RESERVED;
}

uint64_t Mc2TilingUtils::GetMaxWindowSize()
{
    uint16_t defaultWindowSize = 200;
    if (getenv(HCCL_BUFFSIZE) == nullptr) {
        OP_LOGD("", "Env HCCL_BUFFSIZE don't set");
    } else {
        try {
            std::string envStr(getenv(HCCL_BUFFSIZE));
            defaultWindowSize = std::stoi(envStr);
        } catch (...) {
            OP_LOGE("", "Unknown Exception encountered when parser env HCCL_BUFFERSIZE");
        }
    }
    const uint64_t maxWindowSize = static_cast<uint64_t>(defaultWindowSize) * 1024UL * 1024UL;
    OP_LOGI("", "Get maxWindowSize is %lu", maxWindowSize);
    return maxWindowSize;
}

bool GetRankSize(const std::string &opName, const char *group, int64_t &rankSize)
{
    uint32_t rankNum = static_cast<uint32_t>(rankSize);
    if (Mc2Hcom::MC2HcomTopology::CommGetInstSizeByGroup(group, &rankNum) != HCCL_SUCCESS) {
        OP_LOGE(opName, " fail to get group ranksize.");
        return false;
    }
    rankSize = static_cast<int64_t>(rankNum);
    return true;
};

bool Mc2TilingUtils::CheckRankSize(NpuArch npuArch, uint32_t rankSize)
{
    auto it = supportedRankSizeSet.find(npuArch);
    if (it != supportedRankSizeSet.end()) {
        return it->second.count(rankSize) != 0;
    }

    return false;
}

bool Mc2TilingUtils::InferGroupSize(Mc2MatmulShapeInfo &mmInfo, uint64_t &groupSizeM, uint64_t &groupSizeN,
                                    uint64_t &groupSizeK)
{
    // calculate groupSizeM
    if (groupSizeM == 0) {
        // get M of x1
        auto mValue = mmInfo.x1Shape->GetStorageShape().GetDim(mmInfo.x1Shape->GetStorageShape().GetDimNum() - 2);
        int64_t scaleMValue = 0;
        if (mmInfo.isMxfp) {
            // x1Scale is 3 dims in mx scene, 2 dims in perblock scene
            scaleMValue =
                mmInfo.x1ScaleShape->GetStorageShape().GetDim(mmInfo.x1ScaleShape->GetStorageShape().GetDimNum() - 3);
        } else {
            scaleMValue =
                mmInfo.x1ScaleShape->GetStorageShape().GetDim(mmInfo.x1ScaleShape->GetStorageShape().GetDimNum() - 2);
        }
        OP_TILING_CHECK(scaleMValue == 0,
                        OP_LOGE_FOR_INVALID_VALUE(mmInfo.opName, "x1Scale m dimension", "0", "non-zero"), return false);
        OP_TILING_CHECK((mValue % scaleMValue) != 0,
                        OP_LOGE_FOR_INVALID_VALUE(mmInfo.opName, "x1 and x1Scale m dimension",
                                                  (std::to_string(mValue) + ", " + std::to_string(scaleMValue)).c_str(),
                                                  "x1 m dimension should be divisible by x1Scale m dimension"),
                        return false);
        groupSizeM = mValue / scaleMValue;
    }

    if (groupSizeN == 0) {
        uint64_t nIdx = 1;
        if (mmInfo.isBTrans) {
            nIdx = 0; // shape is[n, k] when x2 is transposed
        }
        auto nValue = mmInfo.x2Shape->GetStorageShape().GetDim(nIdx);
        auto scaleNValue = mmInfo.x2ScaleShape->GetStorageShape().GetDim(nIdx);
        OP_TILING_CHECK(scaleNValue == 0,
                        OP_LOGE_FOR_INVALID_VALUE(mmInfo.opName, "x2Scale n dimension", "0", "non-zero"), return false);
        OP_TILING_CHECK((nValue % scaleNValue) != 0,
                        OP_LOGE_FOR_INVALID_VALUE(mmInfo.opName, "x2 and x2Scale n dimension",
                                                  (std::to_string(nValue) + ", " + std::to_string(scaleNValue)).c_str(),
                                                  "x2 n dimension should be divisible by x2Scale n dimension"),
                        return false);
        groupSizeN = nValue / scaleNValue;
    }

    if (groupSizeK == 0) {
        // get k of x1 according to x1shape
        auto kValue = mmInfo.x1Shape->GetStorageShape().GetDim(mmInfo.x1Shape->GetStorageShape().GetDimNum() - 1);
        int64_t scaleKValue = 0;
        if (mmInfo.isMxfp) {
            scaleKValue =
                mmInfo.x1ScaleShape->GetStorageShape().GetDim(mmInfo.x1ScaleShape->GetStorageShape().GetDimNum() - 2) *
                2; // in mxfp scene, scaleshape is [m, k/2, 2]
        } else {
            scaleKValue =
                mmInfo.x1ScaleShape->GetStorageShape().GetDim(mmInfo.x1ScaleShape->GetStorageShape().GetDimNum() - 1);
        }
        OP_TILING_CHECK(scaleKValue == 0,
                        OP_LOGE_FOR_INVALID_VALUE(mmInfo.opName, "x1Scale k dimension", "0", "non-zero"), return false);
        OP_TILING_CHECK((kValue % scaleKValue) != 0,
                        OP_LOGE_FOR_INVALID_VALUE(mmInfo.opName, "x1 and x1Scale k dimension",
                                                  (std::to_string(kValue) + ", " + std::to_string(scaleKValue)).c_str(),
                                                  "x1 k dimension should be divisible by x1Scale k dimension"),
                        return false);
        groupSizeK = kValue / scaleKValue;
    }

    return true;
}

} // namespace mc2tiling
