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
 * \file prompt_flash_attention_tiling_v2.cpp
 * \brief
 */
#include "prompt_flash_attention_tiling_v2.h"
#include <queue>

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/data_copy_transpose_tiling.h"
#include "log/log.h"
#include "err/ops_err.h"
#include "../op_kernel/arch35/prompt_flash_attention_template_tiling_key.h"
#include "../op_kernel/arch35/prompt_flash_attention_tiling_regbase.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ge;
using namespace AscendC;
using namespace matmul_tiling;
namespace optiling {
namespace v2 {
constexpr uint32_t NUM_0 = 0;

constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t VALUE_INDEX = 2;
constexpr uint32_t QUERY_ROPE_INDEX = 3;
constexpr uint32_t KEY_ROPE_INDEX = 4;
constexpr uint32_t ATTENTION_OUT_INDEX = 0;
constexpr uint32_t PSE_SHIFT_INDEX = 3;
constexpr uint32_t ATTEN_MASK_INDEX = 4;
constexpr uint32_t ACTUAL_SEQ_Q_INDEX = 5;
constexpr uint32_t ACTUAL_SEQ_KV_INDEX = 6;
constexpr uint32_t DEQ_SCALE1_INDEX = 7;
constexpr uint32_t QUANT_SCALE1_INDEX = 8;
constexpr uint32_t DEQ_SCALE2_INDEX = 9;
constexpr uint32_t QUANT_SCALE2_INDEX = 10;
constexpr uint32_t QUANT_OFFSET2_INDEX = 11;
constexpr uint32_t ANTIQUANT_SCALE_INDEX = 12;
constexpr uint32_t ANTIQUANT_OFFSET_INDEX = 13;


constexpr uint32_t ATTR_N_INDEX = 0;
constexpr uint32_t ATTR_SCALE_INDEX = 1;
constexpr uint32_t ATTR_PRE_TOKEN_INDEX = 2;
constexpr uint32_t ATTR_NEXT_TOKEN_INDEX = 3;
constexpr uint32_t ATTR_INPUT_LAYOUT_INDEX = 4;
constexpr uint32_t ATTR_NUM_KV_HEADS_INDEX = 5;
constexpr uint32_t INPUT_QKV_SHAPE_MIN_DIMS = 3;
constexpr uint32_t INPUT_QKV_SHAPE_MAX_DIMS = 5;
constexpr uint32_t BYTE_BLOCK = 32; // The block size of datacopy, which moves data at the block granularity.

constexpr uint32_t MASKDIM_SS = 2;
constexpr uint32_t MASKDIM_1SS_BSS = 3;
constexpr uint32_t MASKDIM_11SS_B1SS = 4;
constexpr uint32_t PSESHIFTDIM_4 = 4;

constexpr uint32_t KV_CACHE_DIM_NUMS_5 = 5;
constexpr uint32_t KV_CACHE_DIM_NUMS_4 = 4;
constexpr uint32_t KV_CACHE_DIM_NUMS_3 = 3;
constexpr uint32_t KV_CACHE_DIM_0 = 0;
constexpr uint32_t KV_CACHE_DIM_1 = 1;
constexpr uint32_t KV_CACHE_DIM_2 = 2;
constexpr uint32_t KV_CACHE_DIM_3 = 3;
constexpr uint32_t KV_CACHE_DIM_4 = 4;

constexpr uint64_t EMPTY_KV_TILING_KEY = 20;
constexpr uint32_t LOOP_BEGIN_NUM = 0;
constexpr uint32_t SPARSE_MODE_NO_MASK = 0;
constexpr uint32_t SPARSE_MODE_ALL_MASK = 1;
constexpr uint32_t SPARSE_MODE_LEFT_UP = 2;
constexpr uint32_t SPARSE_MODE_RIGHT_DOWN = 3;
constexpr uint32_t SPARSE_MODE_BAND = 4;
constexpr int64_t SPARSE_MODE_INT_MAX = 2147483647;
constexpr uint32_t SPARSE_OPTIMIZE_ATTENTION_SIZE = 2048;
// The current requirement is a multiple of 128, and to prevent cross block handling, the mm base is also set to 128.
constexpr int32_t BLOCK_SIZE_BASE = 128;
constexpr int32_t BLOCK_SIZE_MAX = 512;
constexpr int32_t BLOCK_SIZE_BASE_FOR_NO_QUANT = 16;
constexpr int32_t BLOCK_SIZE_MAX_FOR_NO_QUANT = 1024;

constexpr uint32_t SOUTER_FACTOR_SUB = 32;
constexpr uint32_t SOUTER_FACTOR_DEFAULT = 64;
constexpr uint32_t SINNER_FACTOR_SUB = 64;
constexpr uint32_t SINNER_FACTOR_DEFAULT = 128;
constexpr uint32_t SINNER_FACTOR_DOUBLE = 256;

constexpr uint32_t FROM_FUSED_FLAG = 71;
constexpr uint32_t CV_RATIO = 2U; // Vector : Cube
constexpr uint32_t MATMUL_NORM_MIN_SEQ = 128;
constexpr uint32_t MATMUL_NORM_MIN_HEADSIZE = 128;
constexpr uint32_t MM2_UB_NUM = 2;

constexpr uint32_t BLIMIT = 65536;
constexpr uint32_t NLIMIT = 256;      // n <= 256
constexpr uint32_t SLIMIT = 20971520; // s, kvs <= 20MB
constexpr uint32_t DLIMIT = 512;      // D <= 512
constexpr uint32_t HLIMIT = 65535;    // warning: H <= 65536
constexpr uint32_t GLIMIT_64 = 64;
constexpr uint32_t GLIMIT_128 = 128;

constexpr uint32_t MLA_QKD_SIZE = 192;
constexpr uint32_t MLA_VD_SIZE = 128; // typical scene for PFA MLA, can be deleted after subsequent generalization.

static const int64_t GM_ALIGN = 512;
static const int64_t SLOPE_N_DIM_NUM = 1L;
uint64_t BENCHMARK_TILING_KEY_ = 1000000000000000000;
constexpr uint32_t ATTR_SPARSE_MODE = 6;
constexpr uint32_t ATTR_INNER_PRECISE = 7;

constexpr int64_t PSE_TYPE_2_TILING_V2 = 2;
constexpr int64_t PSE_TYPE_3_TILING_V2 = 3;
constexpr uint32_t QUERY_SHAPE_DIM_D_128_TILING_V2 = 128;
constexpr int32_t ROPE_DIMENSION_SIZE_TILING_V2 = 64;
constexpr int32_t POS_SHIFT_MAX = 1048576; // 2^20
constexpr int32_t POS_SHIFT_MIN = -1048576; // -2^20

constexpr uint32_t BATCH_MODE_SCHEDULE = 1;
constexpr int32_t D_SIZE_BASE_16 = 16;
constexpr int32_t D_SIZE_BASE_32 = 32;

const std::vector<std::tuple<ge::DataType, ge::DataType, ge::DataType>> inOutDtypeSupported = {
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    {ge::DT_BF16, ge::DT_BF16, ge::DT_BF16},
    {ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16},
    {ge::DT_INT8, ge::DT_INT8, ge::DT_BF16},
    {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT16},
    {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_BF16},
    {ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_FLOAT16},
    {ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_BF16},
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8},
    {ge::DT_BF16, ge::DT_BF16, ge::DT_INT8},
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN},
    {ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN},
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_HIFLOAT8},
    {ge::DT_BF16, ge::DT_BF16, ge::DT_HIFLOAT8},
    {ge::DT_INT8, ge::DT_INT8, ge::DT_INT8}
};

template <typename T>
static auto AlignUp(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    if (num1 < 0) {
        return -(-num1 / num2) * num2;
    }
    return (num1 + num2 - 1) / num2 * num2;
}

template <typename T>
static auto CeilDivision(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

template <typename T>
static auto CalcTailSize(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    T mod = num1 % num2;
    return mod != 0 ? mod : num2;
}

template <typename T>
static auto CeilDiv(const T n1, const T n2) -> T
{
    if (n1 == 0) {
        return 0;
    }
    return (n2 != 0) ? (((n1 - 1) / n2) + 1) : n1;
}

template <typename vecT, typename T>
static bool VecContains(const vecT& vec, const T& value)
{
    return std::find(vec.begin(), vec.end(), value) != vec.end();
}

enum class LayoutType : uint8_t {
    NONE = 0,
    LAYOUT_BSH = 1,
    LAYOUT_BSND = 1,
    LAYOUT_SBH = 2,
    LAYOUT_BNSD = 3,
    LAYOUT_TND = 4,
    LAYOUT_NTD = 5,
};

enum class PfaSparseEnum : uint8_t {
    PFA_ALL = 0,
    PFA_NONE = 1,
    PFA_ANY = 2,
    PFA_CAUSAL = 3,
    PFA_BAND = 4,
    PFA_PREFIX = 5,
    PFA_BAND_COMPRESS = 6,
    PFA_RIGHT_DOWN_CAUSAL = 7,
    PFA_RIGHT_DOWN_CAUSAL_BAND = 8,
    PFA_BAND_LEFT_UP_CAUSAL = 9
};

enum PfaAttenMaskCompressMode : uint8_t {
    PFA_NO_COMPRESS_MODE = 0,
    PFA_LEFT_UP_CAUSAL_MODE,
    PFA_RIGHT_DOWN_CAUSAL_MODE,
    PFA_BAND_MODE,
    PFA_PREFIX_MODE,
    PFA_RIGHT_DOWN_CAUSAL_BAND_MODE,
    PFA_BAND_LEFT_UP_CAUSAL_MODE
};

enum class PfaPseType : uint8_t {
    PSE_OUTER_MUL_ADD_TYPE = 0, // v2 default
    PSE_OUTER_ADD_MUL_TYPE, // v1 current usage
    PSE_INNER_MUL_ADD_TYPE,
    PSE_INNER_MUL_ADD_SQRT_TYPE,
    PSE_INVALID_TYPE,
    PSE_NONE_TYPE = 9
};

enum class PfaPseShapeType : uint8_t {
    PSE_B_N2_G_S1_S2 = 0,
    PSE_B_N2_G_1_S2 = 1,
    PSE_B_N2_G_SLOPE,
    PSE_1_N2_G_SLOPE
};

bool PromptFlashAttentionTilingV2::CheckNonEmptyShapeExceptions(const ContextParamsForPFATiling& contextKeyParams,
    const gert::StorageShape* shape, const std::string &sName) const 
{
    OP_CHECK_IF(shape == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "%s shape is null.", sName.c_str()),
        return false);
    OP_CHECK_IF(shape->GetStorageShape().GetShapeSize() == gert::Shape::kInvalidDimValue,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Get the shape size of %s failed.", sName.c_str()),
        return false);
    return true;
}

void PromptFlashAttentionTilingV2::PromptFlashAttentionInitOutputSplit(int64_t totalSize,
    PromptFlashAttentionTilingDataV2 &tilingData) 
{
    PromptAttentionInitOutputParams *initParams = &tilingData.promptAttentionInitOutputParams;
    // Upward rounding, coreNum has been verified to be non-zero when obtained.
    uint32_t singleCoreSize = (totalSize + coreNum - 1) / (coreNum);

    if (enablePostQuant) {
        // requiring that the number of points allocated to each kernel must be even.
        singleCoreSize = ((singleCoreSize + 1) / 2) * 2; // 2 : fill in 0
    }

    initParams->set_singleCoreSize(singleCoreSize);
    initParams->set_totalOutputSize(totalSize);
}

ge::graphStatus PromptFlashAttentionTilingV2::CheckEmptyTensor(ContextParamsForPFATiling& contextKeyParams) 
{
    OP_CHECK_IF((contextKeyParams.queryInputShape->GetStorageShape().GetShapeSize() == 0 &&
        contextKeyParams.outputShape->GetStorageShape().GetShapeSize() != 0) ||
        (contextKeyParams.queryInputShape->GetStorageShape().GetShapeSize() != 0 &&
        contextKeyParams.outputShape->GetStorageShape().GetShapeSize() == 0),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "query shape size is %llu byte, but attention Out shape size is %llu byte, they cannot be empty while the other is not",
            contextKeyParams.queryInputShape->GetStorageShape().GetShapeSize(), contextKeyParams.outputShape->GetStorageShape().GetShapeSize()),
        return ge::GRAPH_FAILED);
    if (contextKeyParams.queryInputShape->GetStorageShape().GetShapeSize() == 0 &&
        contextKeyParams.outputShape->GetStorageShape().GetShapeSize() == 0) {
            emptyTensor = true;
            return ge::GRAPH_SUCCESS;
    }
    if (enableTensorList) {
        emptyTensor = (contextKeyParams.emptyTensor == 1U);
    } else {
        emptyTensor = ((contextKeyParams.keyInputShape->GetStorageShape().GetShapeSize() == 0) &&
        (contextKeyParams.valueInputShape->GetStorageShape().GetShapeSize() == 0)) ||
        (contextKeyParams.emptyTensor == 1U);
    }
    return ge::GRAPH_SUCCESS;
}

void PromptFlashAttentionTilingV2::SetEmptyTensor(ContextParamsForPFATiling& contextKeyParams,
    uint32_t& numBlocksToBeSet, PromptFlashAttentionTilingDataV2& tilingData) 
{
    faRunFlag_ = true;
    quantMode = NoQuantMode;
    PromptFlashAttentionInitOutputSplit(contextKeyParams.outputShape->GetStorageShape().GetShapeSize(), tilingData);
    tilingData.promptAttentionInitOutputParams.set_totalSoftMaxLseOutputSize(0);
    if (contextKeyParams.isSoftMaxLseEnable) {
        PromptFlashAttentionInitSoftmaxLseOutputSplit(contextKeyParams.lseoutputShape->GetStorageShape().GetShapeSize(),
            tilingData);
    }
    tilingData.promptAttentionInitOutputParams.set_needInit(1);
    auto platformInfoPtr = context_->GetPlatformInfo();
        OP_CHECK_IF(platformInfoPtr == nullptr,
            OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "platformInfoPtr is null!"),
            return);
            auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    numBlocksToBeSet = ascendcPlatform.CalcTschBlockDim(coreNum, aicNum, coreNum);

    size_t* workspace = contextKeyParams.workspaceSize;
    const size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    workspace[0] = sysWorkspaceSize;
}

bool PromptFlashAttentionTilingV2::CheckIODataType(ContextParamsForPFATiling& contextKeyParams) 
{
    outputType = contextKeyParams.outputDataType;
    inputType = contextKeyParams.inputDataType;

    std::vector<ge::DataType> allowedDtypes = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, 
        ge::DT_HIFLOAT8, ge::DT_FLOAT8_E4M3FN};
    std::vector<uint32_t> dataTypeSizeArray = {FLOAT16SIZE, BFLOAT16SIZE, INT8SIZE, FLOAT8SIZE, FLOAT8SIZE, FLOAT8SIZE};

    auto inputTypeCheck = std::find(allowedDtypes.begin(), allowedDtypes.end(), inputType);
    auto outputTypeCheck = std::find(allowedDtypes.begin(), allowedDtypes.end(), outputType);
    if (inputTypeCheck != allowedDtypes.end()) {
        uint32_t inputTypeIndex = std::distance(allowedDtypes.begin(), inputTypeCheck);
        dataTypeSize = dataTypeSizeArray[inputTypeIndex];
    }
    maskElemSize = dataTypeSize;
    if (outputTypeCheck != allowedDtypes.end()) {
        uint32_t outputTypeIndex = std::distance(allowedDtypes.begin(), outputTypeCheck);
        outputDataTypeSize = dataTypeSizeArray[outputTypeIndex];
    }

    if (inputType != ge::DT_FLOAT16) {
        OP_LOGW(contextKeyParams.opName, "innerprecise will not take effect when input type is %s!",
            GetPfaDataTypeStr(inputType).c_str());
    }
    OP_CHECK_IF((inputTypeCheck == allowedDtypes.end() || outputTypeCheck == allowedDtypes.end()),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "inputType(%s) and outputType(%s) should be DT_FLOAT16, DT_BF16, DT_INT8.",
            GetPfaDataTypeStr(inputType).c_str(), GetPfaDataTypeStr(outputType).c_str()),
        return false);
    return true;
}

bool PromptFlashAttentionTilingV2::SetInputLayout(const char* layout) 
{
    if (layout == nullptr) {
        inputLayout = InputLayout::BSH;
        return true;
    }

    std::string layoutStr(layout);
    if (layoutStr == "" || layoutStr == "BSH" || layoutStr == "BSH_NBSD" || layoutStr == "BSH_BNSD") {
        inputLayout = InputLayout::BSH;
    } else if (layoutStr == "TND" || layoutStr == "TND_NTD") {
        inputLayout = InputLayout::TND;
    } else if (layoutStr == "NTD" || layoutStr == "NTD_TND") {
        inputLayout = InputLayout::NTD;
    } else if (layoutStr == "BSND" || layoutStr == "BSND_NBSD" || layoutStr == "BSND_BNSD") {
        inputLayout = InputLayout::BSND;
    } else if (layoutStr == "BNSD" || layoutStr == "BNSD_BSND" || layoutStr == "BNSD_NBSD") {
        inputLayout = InputLayout::BNSD;
    } else {
        return false;
    }

    return true;
}

// 0 不转置; 1 BNSD_BSND; 2 BSND_BNSD; 3 BSH_BNSD; 4 BNSD_NBSD; 5 BSND_NBSD; 6 BSH_NBSD; 7 NTD_TND; 8 TND_NTD
uint32_t GetTransposeLayout(const std::string &layout) 
{
    const std::map<std::string, uint32_t> transposeLayoutMp = {
        {"BNSD_BSND", 1},
        {"BSND_BNSD", 2},
        {"BSH_BNSD", 3},
        {"BNSD_NBSD", 4},
        {"BSND_NBSD", 5},
        {"BSH_NBSD", 6},
        {"NTD_TND", 7},
        {"TND_NTD", 8}
    };
    if (transposeLayoutMp.find(layout) != transposeLayoutMp.end()) {
        return transposeLayoutMp.at(layout);
    }
    return 0;
}

int64_t GetMaxSeq(const gert::Tensor* actualSeqLength) 
{
    int64_t max = actualSeqLength->GetData<int64_t>()[0];
    for (int i = 1; i < actualSeqLength->GetShapeSize(); ++i) {
        max = std::max(max, actualSeqLength->GetData<int64_t>()[i] - actualSeqLength->GetData<int64_t>()[i - 1]);
    }
    return max;
}

bool PromptFlashAttentionTilingV2::SetShape(ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape* shape,
    const std::string inputName, int64_t& b, int64_t& n, int64_t& s, int64_t& d, int64_t& h, int64_t& t) const 
{
    // when enablePA, k_v has two shapes:
    // 1. [>=blockNums, N, blockSize, D]
    // 2. [>=blockNums, blockSize, H]
    if (enablePA && (inputName != "query")) {
        if (shape->GetStorageShape().GetDimNum() == 3) { // 3 for dim num
            b = shape->GetStorageShape().GetDim(0);
            s = shape->GetStorageShape().GetDim(1);
            h = shape->GetStorageShape().GetDim(2); // 2 for H dim
            n = static_cast<int64_t>(*contextKeyParams.numKeyValueHeads);
            n = n > 0 ? n : static_cast<int64_t>(*contextKeyParams.headsNumber);
            d = n > 0 ? h / n : 0;
        } else if (shape->GetStorageShape().GetDimNum() == 4) { // 4 for dim num
            b = shape->GetStorageShape().GetDim(0);
            n = shape->GetStorageShape().GetDim(1);
            s = shape->GetStorageShape().GetDim(2); // 2 for Sequence length
            d = shape->GetStorageShape().GetDim(3); // 3 for D dim
            h = n * d;
        } else {
            b = shape->GetStorageShape().GetDim(0);
            n = shape->GetStorageShape().GetDim(1);
            s = shape->GetStorageShape().GetDim(3); // 3 for Sequence length
            d = shape->GetStorageShape().GetDim(2) * shape->GetStorageShape().GetDim(4); // 2 for D1 dim，4 for D0 dim
            h = n * d;
        }
    } else if ((inputLayout == InputLayout::BNSD)) {
        if (isKVHasPrefix && inputName == "keysharedprefix") {
            b = contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(0); // 0 for Prefix KV batch
            n = contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(1); // 1 for Prefix KV N 
            s = contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(2); // 2 for Prefix KV Sequence length
            d = contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(3); // 3 for Prefix KV D dim
            h = n * d;
            return true;
        }
        b = shape->GetStorageShape().GetDim(0);
        n = shape->GetStorageShape().GetDim(1);
        s = shape->GetStorageShape().GetDim(2); // 2 for Sequence length
        d = shape->GetStorageShape().GetDim(3); // 3 for D dim
        h = n * d;
    } else if ((inputLayout == InputLayout::BSH)) {
        if (isKVHasPrefix && inputName == "keysharedprefix") {
            b = contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(0); // 0 for Prefix KV batch
            s = contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(1); // 1 for Prefix KV Sequence length
            h = contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(2); // 2 for Prefix KV H
            n = n > 0 ? n : static_cast<int64_t>(*contextKeyParams.headsNumber);
            d = n > 0 ? h / n : 0;
            return true;
        }
        b = shape->GetStorageShape().GetDim(0);
        s = shape->GetStorageShape().GetDim(1);
        h = shape->GetStorageShape().GetDim(2); // 2 for H dim
        if (inputName == "query") {
            n = static_cast<int64_t>(*contextKeyParams.headsNumber);
        } else {
            n = static_cast<int64_t>(*contextKeyParams.numKeyValueHeads);
            n = n > 0 ? n : static_cast<int64_t>(*contextKeyParams.headsNumber);
        }
        d = n > 0 ? h / n : 0;
    } else if ((inputLayout == InputLayout::BSND)) {
        if (isKVHasPrefix && inputName == "keysharedprefix") {
            b = contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(0); // 0 for Prefix KV batch
            s = contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(1); // 1 for Prefix KV Sequence length
            n = contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(2); // 2 for Prefix KV N
            d = contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(3); // 3 for Prefix KV D dim
            h = n * d;
            return true;
        }
        b = shape->GetStorageShape().GetDim(0);
        s = shape->GetStorageShape().GetDim(1);
        n = shape->GetStorageShape().GetDim(2); // 2 for head dim
        d = shape->GetStorageShape().GetDim(3); // 3 for D dim
        h = n * d;
    } else if ((inputLayout == InputLayout::TND || inputLayout == InputLayout::NTD)) {
        if (isKVHasPrefix && inputName == "keysharedprefix") {
            return false;
        }
        if (isMaxWorkspace) {
            b = 1;
            s = inputLayout == InputLayout::TND ? shape->GetStorageShape().GetDim(0) : shape->GetStorageShape().GetDim(1);
        } else {
            b = static_cast<int64_t>(contextKeyParams.actualSequenceLengthQ->GetShapeSize());
            s = (inputName == "query") ? GetMaxSeq(contextKeyParams.actualSequenceLengthQ) : GetMaxSeq(contextKeyParams.actualSequenceLengthKV);
        }
        t = inputLayout == InputLayout::TND ? shape->GetStorageShape().GetDim(0) : shape->GetStorageShape().GetDim(1);
        n = inputLayout == InputLayout::TND ? shape->GetStorageShape().GetDim(1) : shape->GetStorageShape().GetDim(0);
        d = shape->GetStorageShape().GetDim(2); // 2 for D dim
        h = n * d;
    } else {
        return false;
    }
    return true;
}

bool PromptFlashAttentionTilingV2::GetAndCheckShape(ContextParamsForPFATiling& contextKeyParams,
    PFAShapeInfo& shapeInfo, const gert::StorageShape* shape, const std::string& sName) const 
{
    std::string layoutStr(contextKeyParams.layout);
    OP_CHECK_IF((shape->GetStorageShape().GetDimNum() != 3) && (inputLayout == InputLayout::BSH || inputLayout == InputLayout::TND || inputLayout == InputLayout::NTD),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "dim num of %s should = 3 when inputLayout is %s,"
            "but dim num is %zu.", sName.c_str(), layoutStr.c_str(), shape->GetStorageShape().GetDimNum()), return false);
    OP_CHECK_IF((shape->GetStorageShape().GetDimNum() != 4) && (inputLayout != InputLayout::BSH && inputLayout != InputLayout::TND && inputLayout != InputLayout::NTD),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "dim num of %s should = 4 when inputLayout is %s,"
            "but dim num is %zu.", sName.c_str(), layoutStr.c_str(), shape->GetStorageShape().GetDimNum()), return false);

    int64_t b = 0;
    int64_t n = 0;
    int64_t s = 0;
    int64_t d = 0;
    int64_t h = 0;
    int64_t t = 0;
    if (!SetShape(contextKeyParams, shape, sName, b, n, s, d, h, t)) {
        return false;
    }
    OP_CHECK_IF((b > BLIMIT || b <= 0), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "batch size of %s should be less than or equal to %u and > 0, but batch size = %ld.", sName.c_str(), BLIMIT, b), return false);
    OP_CHECK_IF((s < 0), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "seq size of %s should >= 0, but sequence size = %ld.", sName.c_str(), s), return false);
    if (s > SLIMIT) {
        OP_LOGW(contextKeyParams.opName, "sequence size of %s should <= 20M, but seq = %ld.", sName.c_str(), s);
    }
    if (inputLayout == InputLayout::BSH) {
        OP_CHECK_IF((d > DLIMIT || d <= 0), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "d size of %s should be less than or equal to %u and > 0, but d = %ld, and layout is BSH, d = h / headsNumber.",
            sName.c_str(), DLIMIT, d), return false);
    } else {
        OP_CHECK_IF((d > DLIMIT || d <= 0), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "d size of %s should be less than or equal to %u and > 0, but d = %ld.", sName.c_str(), DLIMIT, d), return false);
    }
    if (((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::BSND)) && (h > HLIMIT)) {
        OP_LOGW(contextKeyParams.opName, "layout = %s, h(%ld) size of %s should be less than or equal to 65535, "
            "which may cause precision problem! Please use BNSD or BNSD_BSND instead.", contextKeyParams.layout, h, sName.c_str());
    }
    OP_CHECK_IF((inputLayout == InputLayout::TND) && (t <= 0),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "When layout is TND, t should > 0, but t = %ld.", t),
        return false);
    shapeInfo.b = static_cast<uint32_t>(b);
    shapeInfo.n = static_cast<uint64_t>(n);
    shapeInfo.s = static_cast<uint64_t>(s);
    shapeInfo.d = static_cast<uint32_t>(d);
    shapeInfo.h = static_cast<uint64_t>(h);
    shapeInfo.t = static_cast<uint32_t>(t);
    return true;
}

bool PromptFlashAttentionTilingV2::GetAndCheckRopeShape(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& shapeInfo,
    PFAShapeInfo& ropeShapeInfo, const gert::StorageShape* shape, const std::string& sName, const std::string& rName) const 
{
    std::string layoutStr(contextKeyParams.layout);
    OP_CHECK_IF((shape->GetStorageShape().GetDimNum() != 3) && (inputLayout == InputLayout::BSH || inputLayout == InputLayout::TND || inputLayout == InputLayout::NTD),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "dim num of %s should = 3 when inputLayout is %s,"
            "but dim num is %zu.", sName.c_str(), layoutStr.c_str(), shape->GetStorageShape().GetDimNum()), return false);
    OP_CHECK_IF((shape->GetStorageShape().GetDimNum() != 4) && (inputLayout != InputLayout::BSH && inputLayout != InputLayout::TND && inputLayout != InputLayout::NTD),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "dim num of %s should = 4 when inputLayout is %s,"
            "but dim num is %zu.", sName.c_str(), layoutStr.c_str(), shape->GetStorageShape().GetDimNum()), return false);

    OP_CHECK_IF((shapeInfo.d != 128 && shapeInfo.d != 512),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When queryRope and keyRope is not null, d size of %s must be 128 or 512, "
        "but d = %d.", sName.c_str(), shapeInfo.d), return false);

    int64_t b = 0;
    int64_t n = 0;
    int64_t s = 0;
    int64_t d = 0;
    int64_t h = 0;
    int64_t t = 0;
    if (!SetShape(contextKeyParams, shape, sName, b, n, s, d, h, t)) {
        return false;
    }

    if (inputLayout == InputLayout::BSH) {
        OP_CHECK_IF((d != 64),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When queryRope and keyRope is not null, d size of %s must be 64, "
            "but d = %ld, and layout is BSH, d = h / headNumber.", rName.c_str(), d), return false);
    } else {
        OP_CHECK_IF((d != 64),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When queryRope and keyRope is not null, d size of %s must be 64, "
            "but d = %ld.", rName.c_str(), d), return false);
    }
    
    if (!enableTensorList || sName == "query") {
        OP_CHECK_IF((b != shapeInfo.b), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "batch size of %s should be equal to %s batch size, but %s batch size = %ld, %s batch size = %u.",
            rName.c_str(), sName.c_str(), rName.c_str(), b, sName.c_str(), shapeInfo.b), return false);
    }
    OP_CHECK_IF((n != shapeInfo.n), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "head number of %s should be equal to %s head number, but %s head number = %ld, %s head number = %u.",
        rName.c_str(), sName.c_str(), rName.c_str(), n, sName.c_str(), shapeInfo.n), return false);
    OP_CHECK_IF((s != shapeInfo.s), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "seq length of %s should be equal to %s seq length, but %s seq length = %ld, %s seq length = %u.",
        rName.c_str(), sName.c_str(), rName.c_str(), s, sName.c_str(), shapeInfo.s), return false);
    OP_CHECK_IF((t != shapeInfo.t), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "t of %s should be equal to %s t, but %s t = %ld, %s t = %u.",
        rName.c_str(), sName.c_str(), rName.c_str(), t, sName.c_str(), shapeInfo.t), return false);

    if (((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::BSND)) && (static_cast<uint32_t>(h) > HLIMIT)) {
        OP_LOGW(contextKeyParams.opName, "layout = %s, h(%ld) size of %s should be less than or equal to 65535, "
            "which may cause precision problem! Please use BNSD or BNSD_BSND instead.",
            contextKeyParams.layout, h, sName.c_str());
    }
    ropeShapeInfo.b = static_cast<uint32_t>(b);
    ropeShapeInfo.n = static_cast<uint64_t>(n);
    ropeShapeInfo.s = static_cast<uint64_t>(s);
    ropeShapeInfo.d = static_cast<uint32_t>(d);
    ropeShapeInfo.h = static_cast<uint64_t>(h);
    ropeShapeInfo.t = static_cast<uint32_t>(t);
    return true;
}

void PromptFlashAttentionTilingV2::GetQueryDimAndOutDim(const gert::StorageShape* queryShape, const gert::StorageShape* outShape,
    const std::string &layoutStr, int64_t &tmpqueryDim, int64_t &outDim, uint32_t i) const 
{
    if (layoutStr == "BNSD_BSND" || layoutStr == "BSND_BNSD") {
        if (i == 1) { // BNSD_BSND：query:N, output:S; BSND_BNSD：query:S, output:N
            tmpqueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i + 1);
        } else if (i == 2) { // 2: current queryDimNum; BNSD_BSND：query:S, output:N; BSND_BNSD：query:N, output:S
            tmpqueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i - 1);
        }
    } else if (layoutStr == "BSH_BNSD") {
        if (i == 2) { // 2: current queryDimNum; BSH_BNSD：query:H, output:ND
            tmpqueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i + 1) * outShape->GetStorageShape().GetDim(i - 1);
        }
    } else if (layoutStr == "BSND_NBSD") {
        if (i == 1) { // BSND_NBSD：query:S, output:B
            tmpqueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i + 1);
        } else if (i == 2) { // 2: current queryDimNum; BSND_NBSD：query:N, output:S
            tmpqueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i - 2); // 2: 获取outShape的第i - 2个维度
        }
    } else if (layoutStr == "BNSD_NBSD") {
        if (i == 1) { // BNSD_NBSD：query:N, output:B
            tmpqueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i - 1);
        } else if (i == 2) { // 2: current queryDimNum; BNSD_NBSD：query:S, output:S
            tmpqueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i);
        }
    } else if (layoutStr == "BSH_NBSD") {
        if (i == 2) { // 2: current queryDimNum; BSH_NBSD：query:H, output:ND
            tmpqueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i + 1) * outShape->GetStorageShape().GetDim(i - 2); //  2: 获取outShape的第i - 2个维度
        }
    } else if (layoutStr == "NTD_TND" || layoutStr == "TND_NTD") {
        if (i == 0) { // query:N/T, output:T/N
            tmpqueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i + 1);
        } else if (i == 1) { // 2 for current queryDimNum; Q:T/N, Output:N/T
            tmpqueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i - 1);
        }
    } else {
        tmpqueryDim = queryShape->GetStorageShape().GetDim(i);
        outDim = outShape->GetStorageShape().GetDim(i);
    }
}

bool PromptFlashAttentionTilingV2::CheckQueryOutParamsConsistency(const ContextParamsForPFATiling& contextKeyParams,
    const gert::StorageShape* queryShape, const gert::StorageShape* outShape) const 
{
    const size_t queryDimNum = queryShape->GetStorageShape().GetDimNum();
    const size_t outDimNum = outShape->GetStorageShape().GetDimNum();
    int64_t tmpqueryDim = 0;
    int64_t outDim = 0;
    std::string layoutStr(contextKeyParams.layout);
    bool isLayoutShapeSupport = layoutStr == "BSH_BNSD" || layoutStr == "BSH_NBSD";
 	OP_CHECK_IF(queryDimNum != outDimNum && !isLayoutShapeSupport,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "tensor query shape dimNum(%zu) must be consistent with tensor output shape dimNum(%zu)!",
            queryDimNum, outDimNum),
        return false);
    OP_CHECK_IF((queryDimNum < INPUT_QKV_SHAPE_MIN_DIMS) || (queryDimNum > INPUT_QKV_SHAPE_MAX_DIMS),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "tensor query shape dimNum(%zu) is invalid! Only support range [%u, %u].", queryDimNum,
            INPUT_QKV_SHAPE_MIN_DIMS, INPUT_QKV_SHAPE_MAX_DIMS),
        return false);

    for (uint32_t i = 0; i < queryDimNum; ++i) {
        if ((i == queryDimNum - 1) && enablePFAMLA) {
            continue;
        }
        GetQueryDimAndOutDim(queryShape, outShape, layoutStr, tmpqueryDim, outDim, i);
        OP_CHECK_IF(!isQKVDDifferent && (tmpqueryDim != outDim), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "tensor query shape (%ld) do not equal to tensor output shape(%ld) in dim %u for %s.",
            tmpqueryDim, outDim, i, layoutStr.c_str()),
            return false);

        OP_CHECK_IF(isQKVDDifferent && (i != queryDimNum - 1) && (tmpqueryDim != outDim), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "tensor query shape (%ld) do not equal to tensor output shape(%ld) in dim %u for %s.",
            tmpqueryDim, outDim, i, layoutStr.c_str()),
            return false);
    }
    return true;
}

bool PromptFlashAttentionTilingV2::CheckKVDataType(ContextParamsForPFATiling& contextKeyParams) const
{
    ge::DataType keyDataType = contextKeyParams.kDataType;
    ge::DataType valueDataType = contextKeyParams.vDataType;

    std::vector<ge::DataType> allowedDtypes = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8,
        ge::DT_HIFLOAT8, ge::DT_FLOAT8_E4M3FN};

    auto keyTypeCheck = std::find(allowedDtypes.begin(), allowedDtypes.end(), keyDataType);

    OP_CHECK_IF((keyDataType != valueDataType), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "DataType of key(%s) not equal datatype of value(%s).",
        GetPfaDataTypeStr(contextKeyParams.kDataType).c_str(), GetPfaDataTypeStr(contextKeyParams.vDataType).c_str()),
        return false);
    OP_CHECK_IF((keyTypeCheck == allowedDtypes.end()), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
         "DataType of key(%s) should be DT_FLOAT16, DT_BF16, DT_INT8.", GetPfaDataTypeStr(contextKeyParams.kDataType).c_str()),
        return false);
    return true;
}

bool PromptFlashAttentionTilingV2::CheckRopeDataType(ContextParamsForPFATiling& contextKeyParams) const
{
    if (enablePertensorQuant || enablePerblockQuant || enableIFAMLAFullQuant) {
        return true;
    }
    ge::DataType queryDataType = contextKeyParams.inputDataType;
    ge::DataType keyDataType = contextKeyParams.kDataType;
    ge::DataType queryRopeDataType = contextKeyParams.qRopeDataType;
    ge::DataType keyRopeDataType = contextKeyParams.kRopeDataType;
    OP_CHECK_IF((queryDataType != queryRopeDataType), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "DataType of query rope(%s) is not equal to datatype of query(%s).",
        GetPfaDataTypeStr(queryRopeDataType).c_str(), GetPfaDataTypeStr(queryDataType).c_str()),
        return false);
    OP_CHECK_IF((keyDataType != keyRopeDataType), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "DataType of key rope(%s) is not equal to datatype of key(%s).",
        GetPfaDataTypeStr(keyRopeDataType).c_str(), GetPfaDataTypeStr(keyDataType).c_str()),
        return false);
    return true;
}

bool PromptFlashAttentionTilingV2::CheckKeyValueParamsConsistency(ContextParamsForPFATiling& contextKeyParams,
    const gert::StorageShape* keyShape, const gert::StorageShape* valueShape) 
{
    if (enableTensorList) {
        return true;
    }
    const size_t keyDimNum = keyShape->GetStorageShape().GetDimNum();
    const size_t valueDimNum = valueShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(keyDimNum != valueDimNum, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "tensor key shape dimNum(%zu) must be consistent with tensor value shape dimNum(%zu)!", keyDimNum, valueDimNum),
        return false);
    OP_CHECK_IF((keyDimNum < INPUT_QKV_SHAPE_MIN_DIMS) || (keyDimNum > INPUT_QKV_SHAPE_MAX_DIMS),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "tensor key shape dimNum(%zu) is invalid! Only support range [%u, %u].",
            keyDimNum, INPUT_QKV_SHAPE_MIN_DIMS, INPUT_QKV_SHAPE_MAX_DIMS),
        return false);
    for (uint32_t i = 0; i < keyDimNum; ++i) {
        if (((keyDimNum != KV_CACHE_DIM_NUMS_5 && i == keyDimNum - 1) || (keyDimNum == KV_CACHE_DIM_NUMS_5 && i == KV_CACHE_DIM_2)) 
            && enablePFAMLA) { // 使能PFAMLA时，k和v的最后一维，或Nz时k和v的第三维允许不一致
            continue;
        }
        int64_t tmpKeyDim = keyShape->GetStorageShape().GetDim(i);
        int64_t tmpValueDim = valueShape->GetStorageShape().GetDim(i);
        
        OP_CHECK_IF(!isQKVDDifferent && (tmpKeyDim != tmpValueDim), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "tensor key shape(%ld) do not equal to tensor value shape(%ld) in dim %u.", tmpKeyDim, tmpValueDim, i),
            return false);
        OP_CHECK_IF(isQKVDDifferent && (i != keyDimNum - 1) && (tmpKeyDim != tmpValueDim), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "tensor key shape(%ld) do not equal to tensor value shape(%ld) in dim %u.", tmpKeyDim, tmpValueDim, i),
            return false);
    }
    return true;
}

bool PromptFlashAttentionTilingV2::CheckInputDimAndHeadNum(ContextParamsForPFATiling& contextKeyParams,
    const uint64_t nQAttr, const uint64_t nKVAttr) 
{
    uint64_t nQ = nQAttr;
    uint64_t nKV = nKVAttr;
    if (nKVAttr == 0U) { // Detected that nKVAttr is the default value, which means that the customer did not pass in.
        nKV = nQAttr;
    }

    const gert::StorageShape* queryShape = contextKeyParams.queryInputShape;
    const gert::StorageShape* keyShape = contextKeyParams.keyInputShape;
    const gert::StorageShape* valueShape = contextKeyParams.valueInputShape;
    uint64_t queryShapeHeadNum = nQ;
    uint64_t keyShapeHeadNum = nKV;
    uint64_t valueShapeHeadNum = nKV;
    const size_t queryDim = queryShape->GetStorageShape().GetDimNum();
    const size_t keyDim = keyShape->GetStorageShape().GetDimNum();
    const size_t valueDim = valueShape->GetStorageShape().GetDimNum();
    const size_t nIdx = (inputLayout == InputLayout::BNSD || inputLayout == InputLayout::TND) ? 1U : // BNSD/TND: 1
        (inputLayout == InputLayout::BSND) ? 2U : 0U; //  BSND:2; NTD:0

    if (((inputLayout == InputLayout::BNSD) || (inputLayout == InputLayout::BSND)) && (!enablePA)) {
        if ((queryDim == 4) && (keyDim == 4) && (valueDim == 4)) { // dim num: 4
            queryShapeHeadNum = static_cast<uint64_t>(queryShape->GetStorageShape().GetDim(nIdx));
            keyShapeHeadNum = static_cast<uint64_t>(keyShape->GetStorageShape().GetDim(nIdx));
            valueShapeHeadNum = static_cast<uint64_t>(valueShape->GetStorageShape().GetDim(nIdx));
        } else {
            OP_LOGE(contextKeyParams.opName, "input dim of q(%zu), k(%zu), v(%zu) must be 4 for BNSD or BSND format!",
                queryDim, keyDim, valueDim);
            return false;
        }
    } else if ((inputLayout == InputLayout::TND) && (!enablePA)) {
        if ((queryDim == 3) && (keyDim == 3) && (valueDim == 3)) { // dim num: 3
            queryShapeHeadNum = static_cast<uint64_t>(queryShape->GetStorageShape().GetDim(nIdx));
            keyShapeHeadNum = static_cast<uint64_t>(keyShape->GetStorageShape().GetDim(nIdx));
            valueShapeHeadNum = static_cast<uint64_t>(valueShape->GetStorageShape().GetDim(nIdx));
        } else {
            OP_LOGE(contextKeyParams.opName, "input dim of q(%zu), k(%zu), v(%zu) must be 3 for TND format!",
                queryDim, keyDim, valueDim);
            return false;
        }
    } else if ((inputLayout == InputLayout::NTD) && (!enablePA)) {
        if ((queryDim == 3) && (keyDim == 3) && (valueDim == 3)) { // dim num: 3
            queryShapeHeadNum = static_cast<uint64_t>(queryShape->GetStorageShape().GetDim(nIdx));
            keyShapeHeadNum = static_cast<uint64_t>(keyShape->GetStorageShape().GetDim(nIdx));
            valueShapeHeadNum = static_cast<uint64_t>(valueShape->GetStorageShape().GetDim(nIdx));
        } else {
            OP_LOGE(contextKeyParams.opName, "input dim of q(%zu), k(%zu), v(%zu) must be 3 for NTD format!",
                queryDim, keyDim, valueDim);
            return false;
        }
    }

    OP_CHECK_IF(queryShapeHeadNum != nQ,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "numHeads(%u) in query shape must be equal to numHeads(%u) in attr!", queryShapeHeadNum, nQ),
        return false);
    OP_CHECK_IF(keyShapeHeadNum != nKV,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "numHeads(%u) in key shape do not match numKeyValueHeads(%u) in attr!", keyShapeHeadNum, nKV),
        return false);
    OP_CHECK_IF(valueShapeHeadNum != nKV,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "numHeads(%u) in value shape do not match numKeyValueHeads(%u) in attr!", valueShapeHeadNum, nKV),
        return false);
    if (enableIFAMLA || enableIFA) {
        gSize = nQ / nKV;
    }
    return true;
}

bool PromptFlashAttentionTilingV2::SetAndCheckHeadNumRatio(ContextParamsForPFATiling& contextKeyParams, PromptFlashAttentionTilingDataV2& tilingData) 
{
    const int64_t nQ = *contextKeyParams.headsNumber;
    const int64_t nKV = *contextKeyParams.numKeyValueHeads;

    if ((nQ < 0) || (nKV < 0)) {
        OP_LOGE(contextKeyParams.opName, "numHeads(%d) or numKeyValueHeads(%d) is negative!", nQ, nKV);
        return false;
    }

    if (!CheckInputDimAndHeadNum(contextKeyParams, nQ, nKV)) {
        return false;
    }

    if (nKV == 0) { // Detected that nKV is the default value, which means that the customer did not pass in.
        tilingData.promptAttentionBaseParams.set_headNumRatio(1);
        tilingData.promptAttentionBaseParams.set_gOfMla(1);
        return true;
    }

    if (nQ % nKV != 0) {
        OP_LOGE(contextKeyParams.opName, "numHeads(%d) must be divisible by numKeyValueHeads(%d)!", nQ, nKV);
        return false;
    }

    if (enableKVAntiquant || enablePerblockQuant || enablePertensorQuant) {	 
        if (nQ > NLIMIT) {
            OP_LOGE(contextKeyParams.opName, "the numheads of input query cannot be larger than 256, but numheads = %d", nQ);	 
             return false;
        }
        if (nQ / nKV > GLIMIT_64) { // G cannot be greater than 64.	 
            OP_LOGE(contextKeyParams.opName, "In antiquant and fullquant scenario, the G(numHeads / numKeyValueHeads) connot be larger than 64, but G = %d", nQ / nKV);	 
            return false; 
        } 
     } else { 
        if ((enableIFAMLA || enableIFAMLAFullQuant) && (nQ / nKV > GLIMIT_128)) { // G cannot be greater than 128. 
            OP_LOGE(contextKeyParams.opName, "In mla decode (non quant and fullquant) scenario, the G(numHeads / numKeyValueHeads) connot be larger than 128, but G = %d", nQ / nKV); 
            return false; 
        } 
     } 

    if (enableIFAMLA || enableIFA) {
        tilingData.promptAttentionBaseParams.set_headNumRatio(1);
        tilingData.promptAttentionBaseParams.set_gOfMla(gSize);
    } else {
        tilingData.promptAttentionBaseParams.set_headNumRatio(nQ / nKV);
        tilingData.promptAttentionBaseParams.set_gOfMla(1);
    }

    return true;
}

bool PromptFlashAttentionTilingV2::CheckPostQuantShape(const ContextParamsForPFATiling &contextKeyParams,
                                                       const gert::StorageShape *quantScale2Shape,
                                                       const gert::StorageShape *quantOffset2Shape,
                                                       const PFAShapeInfo &queryShapeInfo,
                                                       const PFAShapeInfo &valueShapeInfo) const
{
    if (quantOffset2Shape != nullptr) {
        OP_CHECK_IF(quantScale2Shape->GetStorageShape() != quantOffset2Shape->GetStorageShape(),
                    OP_LOGE(contextKeyParams.opName, "quantScale2 and quantOffset2 should have same shape."),
                    return false);
    }
    std::string layoutString(contextKeyParams.layout);
    size_t quantScale2Dim = quantScale2Shape->GetStorageShape().GetDimNum();
    int64_t quantScale2ShapeSize = quantScale2Shape->GetStorageShape().GetShapeSize();
    uint64_t quantScale2ShapeSizePerChannel = static_cast<uint64_t>(queryShapeInfo.n) * static_cast<uint64_t>(valueShapeInfo.d);
    bool isSupportedLayout = layoutString == "BSH" || layoutString == "BSND" || layoutString == "BNSD" || layoutString == "BNSD_BSND";
    if (quantScale2Dim == 1) {
        if(quantScale2ShapeSize == quantScale2ShapeSizePerChannel && isSupportedLayout){
            //supportLayout + post quant perChannel场景 支持 shape [H]
            return true;
        } else {
            OP_CHECK_IF(quantScale2ShapeSize != 1,
                    OP_LOGE(contextKeyParams.opName,
                            "For post quant per-tensor, quantScale2 and quantOffset2 shape only support [1]."),
                    return false);
        }
    } else {
        if (isSupportedLayout) {
            OP_CHECK_IF((static_cast<uint64_t>(quantScale2ShapeSize) != quantScale2ShapeSizePerChannel),
                        OP_LOGE(contextKeyParams.opName,
                                "For post quant per-channel, when layout is %s, quantScale2/quantOffset2 dim multiply "
                                "result only support qN * vD(%u * %u = %lu), now is (%ld).",
                                layoutString.c_str(), queryShapeInfo.n, valueShapeInfo.d,
                                quantScale2ShapeSizePerChannel, quantScale2ShapeSize),
                        return false);
        } else {
            OP_CHECK_IF(quantScale2Shape->GetStorageShape() != gert::Shape({queryShapeInfo.n, valueShapeInfo.d}),
                        OP_LOGE(contextKeyParams.opName,
                                "For post quant per-channel, when layout is %s, "
                                "quantScale2/quantOffset2 expect shape is [%u, %u].",
                                layoutString.c_str(), queryShapeInfo.n, valueShapeInfo.d),
                        return false);
        }
    }
    return true;
}

bool PromptFlashAttentionTilingV2::CheckPerTensorQuantParams(const ContextParamsForPFATiling& contextKeyParams,
    const PFAShapeInfo& queryShapeInfo) const {
    const gert::StorageShape* deqScale1Shape = contextKeyParams.deqScale1Shape;
    const gert::StorageShape* quantScale1Shape = contextKeyParams.scale1Shape;
    const gert::StorageShape* deqScale2Shape = contextKeyParams.deqScale2Shape;
    const gert::StorageShape* quantScale2Shape = contextKeyParams.scale2Shape;
    OP_CHECK_IF((contextKeyParams.queryQuantMode == nullptr || contextKeyParams.keyAntiquantMode == nullptr ||
        contextKeyParams.valueAntiquantMode == nullptr), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "queryQuantMode, keyAntiquantMode or valueAntiquantMode is nullptr in per-tensor quant scenario."), return false);
    OP_CHECK_IF((*contextKeyParams.queryQuantMode != static_cast<int64_t>(AntiquantTypeEnum::PER_CHANNEL) ||
        *contextKeyParams.keyAntiquantMode != static_cast<int64_t>(AntiquantTypeEnum::PER_CHANNEL)||
        *contextKeyParams.valueAntiquantMode != static_cast<int64_t>(AntiquantTypeEnum::PER_CHANNEL)),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "queryQuantMode(%ld), keyAntiquantMode(%ld) or valueQuantMode(%ld) "
            "is not supported in per-tensor quant scenario.! Only support Per-Tensor(0).", *contextKeyParams.queryQuantMode,
            *contextKeyParams.keyAntiquantMode, *contextKeyParams.valueAntiquantMode), return false);
    OP_CHECK_IF((quantScale2Shape != nullptr), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "Postquant is not supported in per-tensor quant scenario."), return false);
    OP_CHECK_IF((contextKeyParams.inputDataType != ge::DT_INT8), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "InputParamsType must be INT8 in per-tensor quant scenario, now is %s", 
            GetPfaDataTypeStr(contextKeyParams.inputDataType).c_str()), return false);
    std::string layoutStr(contextKeyParams.layout);
    const std::vector<std::string> unsupportedLayoutList = {"BNSD_NBSD", "BSND_NBSD", "BSH_NBSD", "BSH_BNSD",
        "BSND_BNSD", "TND", "NTD", "NTD_TND", "TND_NTD"};
    OP_CHECK_IF(std::find(unsupportedLayoutList.begin(), unsupportedLayoutList.end(), layoutStr) != unsupportedLayoutList.end(),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "%s is not supported in per-tensor quant scenario.", layoutStr.c_str()), return false);
    OP_CHECK_IF((deqScale1Shape == nullptr) || (quantScale1Shape == nullptr) || (deqScale2Shape == nullptr),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "DeqScale1, quantScale1 or deqScale2 is nullptr in per-tensor quant scenario."), return false);
    OP_CHECK_IF((deqScale1Shape != nullptr && deqScale1Shape->GetStorageShape().GetShapeSize() == 0) ||
                (quantScale1Shape != nullptr && quantScale1Shape->GetStorageShape().GetShapeSize() == 0) ||
                (deqScale2Shape != nullptr && deqScale2Shape->GetStorageShape().GetShapeSize() == 0),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "DeqScale1, quantScale1 or deqScale2 is empty tensor in per-tensor quant scenario."), return false);
    OP_CHECK_IF(enablePFARope, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Rope is not supported in per-tensor quant scenario."), return false);
    OP_CHECK_IF(enableAlibiPse, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "AlibiPse is not supported in per-tensor quant scenario."), return false);
    OP_CHECK_IF(queryShapeInfo.s == 1, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "The q_s should not be equal to 1 in per-tensor quant scenario."), return false);
    const size_t dIdx = (inputLayout == InputLayout::TND || inputLayout == InputLayout::BSH) ? 2U : 3U; // TND/BSH:2; BSND/BNSD/BNSD_BSND:3
    uint64_t keyShapeD = contextKeyParams.keyInputShape->GetStorageShape().GetDim(dIdx);
    uint64_t valueShapeD = contextKeyParams.valueInputShape->GetStorageShape().GetDim(dIdx);
    if (inputLayout == InputLayout::BSH) {
        int64_t nKV = *contextKeyParams.numKeyValueHeads;
        if (nKV == 0) {
            nKV = *contextKeyParams.headsNumber;
        }
        keyShapeD = static_cast<uint64_t>(keyShapeD / nKV);
        valueShapeD = static_cast<uint64_t>(valueShapeD / nKV);
    }
    OP_CHECK_IF(keyShapeD != valueShapeD, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "The D size of keyInputShape and valueInputShape must be equal in per-tensor quant scenario, but now keyShapeD is %lu, valueShapeD is %lu.",
            keyShapeD, valueShapeD), return false);
    return true;           
}                                     

bool PromptFlashAttentionTilingV2::CheckPerblockQuantParams(const ContextParamsForPFATiling& contextKeyParams, 
    const PFAShapeInfo& queryShapeInfo, const PFAShapeInfo& keyShapeInfo, const PFAShapeInfo& valueShapeInfo) const 
{
    const ge::DataType dequantScaleQueryType = contextKeyParams.dequantScaleQueryType;
    const ge::DataType KeyAntiquantScaleType = contextKeyParams.KeyAntiquantScaleType;
    const ge::DataType valueAntiquantScaleType = contextKeyParams.valueAntiquantScaleType;
    const gert::StorageShape* dequantScaleQueryShape = contextKeyParams.dequantScaleQueryShape;
    const gert::StorageShape* keyAntiquantScaleShape = contextKeyParams.KeyAntiquantScaleShape;
    const gert::StorageShape* valueAntiquantScaleshape = contextKeyParams.valueAntiquantScaleShape;
    const gert::StorageShape* deqScale1Shape = contextKeyParams.deqScale1Shape;
    const gert::StorageShape* quantScale1Shape = contextKeyParams.scale1Shape;
    const gert::StorageShape* deqScale2Shape = contextKeyParams.deqScale2Shape;
    const gert::StorageShape* antiquantScaleShape = contextKeyParams.antiquantScaleShape;
    OP_CHECK_IF((dequantScaleQueryShape == nullptr) || (keyAntiquantScaleShape == nullptr) || (valueAntiquantScaleshape == nullptr),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "dequantScaleQuery, keyAntiquantScale or valueAntiquantScale is nullptr in per-block quant scenario."),
        return false);
    OP_CHECK_IF((deqScale1Shape != nullptr) || (deqScale2Shape != nullptr),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "deqScale1 or deqScale2 is not supported in per-block quant scenario."),
        return false);
    OP_CHECK_IF((quantScale1Shape != nullptr) && (quantScale1Shape->GetStorageShape().GetDimNum() != 1),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "quantScale1's dim must be must be 1 in per-block quant scenario, "
            "now quantScale1's dim is %zu.", quantScale1Shape->GetStorageShape().GetDimNum()),
        return false);
    OP_CHECK_IF((quantScale1Shape != nullptr) && (quantScale1Shape->GetStorageShape().GetDimNum() == 1) && (quantScale1Shape->GetStorageShape().GetDim(0) != 1),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "quantScale1's shape must be must be [1] in per-block quant scenario, "
            "now quantScale1's shape is [%u].", quantScale1Shape->GetStorageShape().GetDim(0)),
        return false);
    OP_CHECK_IF(antiquantScaleShape != nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "antiquantScale is not supported in per-block quant scenario."),
        return false);
    // 全量化暂不支持 keyAntiquantOffset, valueAntiquantOffset
    OP_CHECK_IF((contextKeyParams.KeyAntiquantOffsetShape != nullptr) ||
        (contextKeyParams.valueAntiquantOffsetShape != nullptr),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "keyAntiquantOffset and valueAntiquantOffset should be null in per-block quant scenario."),
            return false);
    const size_t dequeryDim = dequantScaleQueryShape->GetStorageShape().GetDimNum();
    const size_t dekeyDim = keyAntiquantScaleShape->GetStorageShape().GetDimNum();
    const size_t devalueDim = valueAntiquantScaleshape->GetStorageShape().GetDimNum();
    constexpr uint32_t fp8QBlockSize = 128U; // 128 is SOuterSize
    constexpr uint32_t fp8KVBlockSize = 256U; // 256 is SInnerSize
    std::string layoutStr(contextKeyParams.layout);
    if (enablePseShift) {
        OP_CHECK_IF(*contextKeyParams.pseType == 0, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "Pse is not supported in per-block quant scenario!"), return false);
    }
    OP_CHECK_IF(enableAlibiPse, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "AlibiPse is not supported in per-block quant scenario!"),
        return false);
    // When PA and tensorlist are enable, they may affect the shape parsing of query, key and value,
    OP_CHECK_IF(enablePA, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "PA is not supported in per-block quant scenario!"),
        return false);
    OP_CHECK_IF(enableTensorList, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "tensorlist is not supported in per-block quant scenario!"),
        return false);
    OP_CHECK_IF((contextKeyParams.inputDataType != ge::DT_FLOAT8_E4M3FN) && (contextKeyParams.inputDataType != ge::DT_HIFLOAT8),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "inputType must be FLOAT8_E4M3FN or HIFLOAT8 in per-block quant scenario, now is %s.", 
            GetPfaDataTypeStr(inputType).c_str()),
        return false);
    OP_CHECK_IF((dequantScaleQueryType != ge::DT_FLOAT) || (KeyAntiquantScaleType != ge::DT_FLOAT) || (valueAntiquantScaleType != ge::DT_FLOAT),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "dequantscale type must be DT_FLOAT in per-block quant scenario," 
            "now dequantScaleQuery's type is %s, KeyAntiquantScale's type is %s, valueAntiquantScale's type is %s.",
            GetPfaDataTypeStr(dequantScaleQueryType).c_str(), GetPfaDataTypeStr(KeyAntiquantScaleType).c_str(), GetPfaDataTypeStr(valueAntiquantScaleType).c_str()),
        return false);
    const std::vector<std::string> unsupportedLayoutList = {"BNSD_NBSD", "BSH_NBSD", "BSH_BNSD", "BSND_BNSD", "BSND_NBSD", "NTD"};
    OP_CHECK_IF((std::find(unsupportedLayoutList.begin(), unsupportedLayoutList.end(), layoutStr) != unsupportedLayoutList.end()) || inputLayout == InputLayout::TND,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "In per-block quant scenario, the layout %s is not supported.", layoutStr.c_str()),
        return false);
    OP_CHECK_IF((queryShapeInfo.d > 128) || (keyShapeInfo.d > 128) || (valueShapeInfo.d > 128), // 128 is the limit for d.
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "query d size, key d size and value d size must be less than or equal to 128 in per-block quant scenario," 
            "now query d = %u, key d = %u, value d = %u.",
            queryShapeInfo.d, keyShapeInfo.d, valueShapeInfo.d),
        return false);
    if (inputLayout == InputLayout::NTD) {
        OP_CHECK_IF((dequeryDim != 3) || (dekeyDim != 3) || (devalueDim != 3),   // 3 is the number of dimensions of the dequant scale.
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "When layout is %s, dequantscale's dim must be 3 in per-block quant scenario," 
                "now dequantScaleQuery's dim is %zu, keyAntiquantScale's dim is %zu, valueAntiquantScale's dim is %zu.",
                layoutStr.c_str(), dequeryDim, dekeyDim, devalueDim),
            return false);
        if (isMaxWorkspace) return true;
        OP_CHECK_IF((dequantScaleQueryShape->GetStorageShape().GetDim(0) != queryShapeInfo.n) ||
                    (dequantScaleQueryShape->GetStorageShape().GetDim(1) != queryShapeInfo.t / fp8QBlockSize + queryShapeInfo.b) ||
                    (dequantScaleQueryShape->GetStorageShape().GetDim(2) != CeilDivision(queryShapeInfo.d, fp8KVBlockSize)),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "When layout is %s, dequantScaleQueryShape must be [%u, %u, %u] in per-block quant scenario, now is  [%u, %u, %u].",
                layoutStr.c_str(), queryShapeInfo.n, queryShapeInfo.t / fp8QBlockSize + queryShapeInfo.b, CeilDivision(queryShapeInfo.d, fp8KVBlockSize),
                dequantScaleQueryShape->GetStorageShape().GetDim(0), dequantScaleQueryShape->GetStorageShape().GetDim(1),
                dequantScaleQueryShape->GetStorageShape().GetDim(2)),
            return false); 
        OP_CHECK_IF((keyAntiquantScaleShape->GetStorageShape().GetDim(0) != keyShapeInfo.n) ||
                    (keyAntiquantScaleShape->GetStorageShape().GetDim(1) != keyShapeInfo.t / fp8KVBlockSize + keyShapeInfo.b) ||
                    (keyAntiquantScaleShape->GetStorageShape().GetDim(2) != CeilDivision(keyShapeInfo.d, fp8KVBlockSize)),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "When layout is %s, keyAntiquantScaleShape must be [%u, %u, %u] in per-block quant scenario, now is [%u, %u, %u].",
                layoutStr.c_str(), keyShapeInfo.n, keyShapeInfo.t / fp8KVBlockSize + keyShapeInfo.b, CeilDivision(keyShapeInfo.d, fp8KVBlockSize),
                keyAntiquantScaleShape->GetStorageShape().GetDim(0), keyAntiquantScaleShape->GetStorageShape().GetDim(1),
                keyAntiquantScaleShape->GetStorageShape().GetDim(2)),
            return false);
        OP_CHECK_IF((valueAntiquantScaleshape->GetStorageShape().GetDim(0) != valueShapeInfo.n) ||
                    (valueAntiquantScaleshape->GetStorageShape().GetDim(1) != valueShapeInfo.t / fp8KVBlockSize + valueShapeInfo.b) ||
                    (valueAntiquantScaleshape->GetStorageShape().GetDim(2) != CeilDivision(valueShapeInfo.d, fp8KVBlockSize)),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "When layout is %s, valueAntiquantScaleshape must be [%u, %u, %u] in per-block quant scenario, now is [%u, %u, %u].",
                layoutStr.c_str(), valueShapeInfo.n, valueShapeInfo.t / fp8KVBlockSize + valueShapeInfo.b, CeilDivision(valueShapeInfo.d, fp8KVBlockSize),
                valueAntiquantScaleshape->GetStorageShape().GetDim(0), valueAntiquantScaleshape->GetStorageShape().GetDim(1),
                valueAntiquantScaleshape->GetStorageShape().GetDim(2)),
            return false);
        OP_CHECK_IF((innerPrecise == MSD_HIGH_PERFORMANCE_EXPEND_NUM) || (innerPrecise == MSD_HIGH_PRECISION_EXPEND_NUM),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "innerPrecise [%d] is currently not supported.(Value 2 or 3 is not supported).", innerPrecise), return false);
    } else {
        OP_CHECK_IF((dequeryDim != 4) || (dekeyDim != 4) || (devalueDim != 4),   // 4 is the number of dimensions of the dequant scale.
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "dequantscale's dim must be 4 in per-block quant scenario," 
                "now dequantScaleQuery's dim is %zu, keyAntiquantScale's dim is %zu, valueAntiquantScale's dim is %zu.",
                dequeryDim, dekeyDim, devalueDim),
            return false);
        OP_CHECK_IF((dequantScaleQueryShape->GetStorageShape().GetDim(0) != queryShapeInfo.b) ||
                    (dequantScaleQueryShape->GetStorageShape().GetDim(1) != queryShapeInfo.n) ||
                    (dequantScaleQueryShape->GetStorageShape().GetDim(2) != CeilDivision(queryShapeInfo.s, static_cast<uint64_t>(fp8QBlockSize))) ||   // 2 is the dim of dequantscale along s1.
                    (dequantScaleQueryShape->GetStorageShape().GetDim(3) != 1U),  // 3 is the dim of dequantscale along d.
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "dequantScaleQueryShape must be [%u, %u, %u, %u] in per-block quant scenario, now is  [%u, %u, %u, %u].",
                queryShapeInfo.b, queryShapeInfo.n, CeilDivision(queryShapeInfo.s, static_cast<uint64_t>(fp8QBlockSize)), 1,
                dequantScaleQueryShape->GetStorageShape().GetDim(0), dequantScaleQueryShape->GetStorageShape().GetDim(1),
                dequantScaleQueryShape->GetStorageShape().GetDim(2), dequantScaleQueryShape->GetStorageShape().GetDim(3)),
            return false); 
        OP_CHECK_IF((keyAntiquantScaleShape->GetStorageShape().GetDim(0) != keyShapeInfo.b) ||
                    (keyAntiquantScaleShape->GetStorageShape().GetDim(1) != keyShapeInfo.n) ||
                    (keyAntiquantScaleShape->GetStorageShape().GetDim(2) != CeilDivision(keyShapeInfo.s, static_cast<uint64_t>(fp8KVBlockSize))) || //  2 is the dim of dequantscale along s2.
                    (keyAntiquantScaleShape->GetStorageShape().GetDim(3) != 1U), // 3 is the dim of dequantscale along d.
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "keyAntiquantScaleShape must be [%u, %u, %u, %u] in per-block quant scenario, now is [%u, %u, %u, %u].",
                keyShapeInfo.b, keyShapeInfo.n, CeilDivision(keyShapeInfo.s, static_cast<uint64_t>(fp8KVBlockSize)), 1,
                keyAntiquantScaleShape->GetStorageShape().GetDim(0), keyAntiquantScaleShape->GetStorageShape().GetDim(1),
                keyAntiquantScaleShape->GetStorageShape().GetDim(2), keyAntiquantScaleShape->GetStorageShape().GetDim(3)),
            return false);
        OP_CHECK_IF((valueAntiquantScaleshape->GetStorageShape().GetDim(0) != valueShapeInfo.b) ||
                    (valueAntiquantScaleshape->GetStorageShape().GetDim(1) != valueShapeInfo.n) ||
                    (valueAntiquantScaleshape->GetStorageShape().GetDim(2) != CeilDivision(valueShapeInfo.s, static_cast<uint64_t>(fp8KVBlockSize))) || // 2 is the dim of dequantscale along s2.
                    (valueAntiquantScaleshape->GetStorageShape().GetDim(3) != 1U), // 3 is the dim of dequantscale along d.
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "valueAntiquantScaleshape must be [%u, %u, %u, %u] in per-block quant scenario, now is [%u, %u, %u, %u].",
                valueShapeInfo.b, valueShapeInfo.n, CeilDivision(valueShapeInfo.s, static_cast<uint64_t>(fp8KVBlockSize)), 1,
                valueAntiquantScaleshape->GetStorageShape().GetDim(0), valueAntiquantScaleshape->GetStorageShape().GetDim(1),
                valueAntiquantScaleshape->GetStorageShape().GetDim(2), valueAntiquantScaleshape->GetStorageShape().GetDim(3)),
            return false); 
    }
    return true; 
}

bool PromptFlashAttentionTilingV2::CheckPostQuantParams(const ContextParamsForPFATiling& contextKeyParams, 
    const PFAShapeInfo& queryShapeInfo, const PFAShapeInfo& valueShapeInfo) const 
{
    const gert::StorageShape* quantScale2Shape = contextKeyParams.scale2Shape;
    const gert::StorageShape* quantOffset2Shape = contextKeyParams.offset2Shape;
    const ge::DataType quantScale2Type = contextKeyParams.quantScale2Type;
    int64_t quantScale2ShapeSize = 0;

    OP_CHECK_IF(outputType != ge::DT_INT8 && outputType != ge::DT_FLOAT8_E4M3FN && outputType != ge::DT_HIFLOAT8,
               OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
               "invalid output type [%s], only support int8, fp8_e4m3fn_t, hifloat8_t",
               GetPfaDataTypeStr(outputType).c_str()),
               return false);
    // Basic verification: quantScale2 must be inputted and not an empty tensor
    OP_CHECK_IF(quantScale2Shape == nullptr, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "quant_scale2_shape is nullptr in post quant scenario."),
        return false);
    quantScale2ShapeSize = quantScale2Shape->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(quantScale2ShapeSize == 0, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "quant_scale2 is empty tensor in post quant scenario."),
        return false);
    
    OP_CHECK_IF((quantOffset2Shape != nullptr) && (quantScale2Type != contextKeyParams.quantOffset2Type),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "post quant scale dtype(%s) and offset dtype(%s) must be consistent.",
            GetPfaDataTypeStr(quantScale2Type).c_str(), GetPfaDataTypeStr(contextKeyParams.quantOffset2Type).c_str()),
        return false);
    
    OP_CHECK_IF((quantScale2Type != ge::DT_BF16) && (quantScale2Type != ge::DT_FLOAT),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "post quant scale dtype(%s) only support bf16 and fp32.",
            GetPfaDataTypeStr(quantScale2Type).c_str()),
        return false);
    
    if (inputType == ge::DT_BF16) {
        OP_CHECK_IF(quantScale2Type != ge::DT_FLOAT && quantScale2Type != ge::DT_BF16,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "invalid post quant scale dtype(%s), when q is %s, only support float32 and bf16",
            GetPfaDataTypeStr(quantScale2Type).c_str(), GetPfaDataTypeStr(inputType).c_str()),
            return false);
    } else {
        OP_CHECK_IF(quantScale2Type != ge::DT_FLOAT,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "invalid post quant scale dtype(%s), when q is %s, only support float32",
            GetPfaDataTypeStr(quantScale2Type).c_str(), GetPfaDataTypeStr(inputType).c_str()),
            return false);
    }

    OP_CHECK_IF(!CheckPostQuantShape(contextKeyParams, quantScale2Shape, quantOffset2Shape, queryShapeInfo, valueShapeInfo),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "post quant params check failed!"),
        return false);
    return true;
}

bool PromptFlashAttentionTilingV2::CheckAntiquantParamsShape(ContextParamsForPFATiling& contextKeyParams) const
{
    const gert::StorageShape* antiquantScaleShape = contextKeyParams.antiquantScaleShape;
    // 伪量化收编至ifa tiling，pfa接口kv若传入伪量化dtype均会在此被拦截
    OP_CHECK_IF(contextKeyParams.antiquantScale == nullptr || antiquantScaleShape == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant scale is nullptr"),
        return false);
    return true;
}

bool PromptFlashAttentionTilingV2::GetAndCheckPrefixShape(ContextParamsForPFATiling& contextKeyParams,
    PFAShapeInfo& keyShapeInfo, PFAShapeInfo& prefixShapeInfo) const 
{
    int64_t prefixSeqInnerSize = 0;
    int64_t bPrefix = 0U;
    int64_t nPrefix = 0U;
    int64_t dPrefix = 0U;
    int64_t hPrefix = 0U;
    int64_t tPrefix = 0;

    if (!SetShape(contextKeyParams, contextKeyParams.keyInputShape, "keysharedprefix", bPrefix, nPrefix,
                  prefixSeqInnerSize, dPrefix, hPrefix, tPrefix)) {
        return false;
    }
    OP_CHECK_IF((bPrefix != 1),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "prefix batch num(%ld) only support 1!", bPrefix),
                return false);
    if (inputLayout == InputLayout::BSH) {
        OP_CHECK_IF((hPrefix != keyShapeInfo.h),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "prefix H(%ld) should be same with KV H(%u)!",
                                                hPrefix, keyShapeInfo.h),
                    return false);
    } else {
        OP_CHECK_IF((nPrefix != keyShapeInfo.n) || (dPrefix != keyShapeInfo.d),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                                                "prefix N(%ld) and D(%ld) should be same with KV N(%u) and D(%u)!",
                                                nPrefix, dPrefix, keyShapeInfo.n, keyShapeInfo.d),
                    return false);
    }
    prefixShapeInfo.b = static_cast<uint32_t>(bPrefix);
    prefixShapeInfo.n = static_cast<uint64_t>(nPrefix);
    prefixShapeInfo.s = static_cast<uint64_t>(prefixSeqInnerSize);
    prefixShapeInfo.d = static_cast<uint32_t>(dPrefix);
    prefixShapeInfo.h = static_cast<uint64_t>(hPrefix);
    return true;
}

bool PromptFlashAttentionTilingV2::CheckKeyValuePrefixConsistency(ContextParamsForPFATiling& contextKeyParams,
    const gert::StorageShape* keyShape) 
{
    size_t prefixKeyDim = contextKeyParams.keySharedPrefix->GetStorageShape().GetDimNum();
    size_t prefixValueDim = contextKeyParams.valueSharedPrefix->GetStorageShape().GetDimNum();
    size_t KVDim = keyShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(((prefixKeyDim != KVDim) || (prefixKeyDim != prefixValueDim)),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "dim num of key_shared_prefix and value_shared_prefix should be same with KV, "
            "but key_shared_prefix dim(%zu), value_shared_prefix dim(%zu), KV dim(%zu)!", prefixKeyDim, prefixValueDim, KVDim),
        return false);
    for (uint32_t i = 0; i < prefixKeyDim; i++) {
        int64_t tmpPrefixKeyDim = contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(i);
        int64_t tmpPrefixValueDim = contextKeyParams.valueSharedPrefix->GetStorageShape().GetDim(i);
        OP_CHECK_IF(((tmpPrefixKeyDim == 0) || (tmpPrefixValueDim == 0)), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "key_shared_prefix and value_shared_prefix not support empty tensor,"
            "but key_shared_prefix[%u]:%ld, value_shared_prefix[%u]:%ld!", i, tmpPrefixKeyDim, i, tmpPrefixValueDim),
            return false);
        OP_CHECK_IF(((tmpPrefixKeyDim != tmpPrefixValueDim)),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "shape of key_shared_prefix should be same with value_shared_prefix,"
                "but key_shared_prefix[%u]:%ld, value_shared_prefix[%u]:%ld!", i, tmpPrefixKeyDim, i, tmpPrefixValueDim),
            return false);
    }
    auto keySharedPrefixDataType = contextKeyParams.keySharedPrefixDataType;
    auto valueSharedPrefixDataType = contextKeyParams.valueSharedPrefixDataType;
    OP_CHECK_IF((keySharedPrefixDataType != valueSharedPrefixDataType), 
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "When system prefix is used, dataType of keySharedPrefixDataType(%s) and dataType of valueSharedPrefixDataType(%s) must be same.",
            GetPfaDataTypeStr(keySharedPrefixDataType).c_str(), GetPfaDataTypeStr(valueSharedPrefixDataType).c_str()),
        return false);
    return true;
}

bool PromptFlashAttentionTilingV2::CheckActSharedPrefix(ContextParamsForPFATiling& contextKeyParams,
    const uint32_t sPrefix, const uint32_t sKV) 
{
    size_t prefixDimNum = contextKeyParams.actualSharedPrefixLen->GetStorageShape().GetDimNum();
    OP_CHECK_IF((prefixDimNum != 1), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "actualSharedPrefixLen dim num(%zu) should be 1!", prefixDimNum),
        return false);
    int64_t prefixShapeSize = contextKeyParams.actualSharedPrefixLen->GetStorageShape().GetShapeSize();
    OP_CHECK_IF((prefixShapeSize != 1), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "actualSharedPrefixLen length(%ld) should be 1!", prefixShapeSize),
        return false);
    OP_CHECK_IF((contextKeyParams.actualSharedPrefixLen->GetData<int64_t>() == nullptr),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "input actualSharedPrefixLen GetData is nullptr!"),
        return false);
    actualSharedPrefixLen = contextKeyParams.actualSharedPrefixLen->GetData<int64_t>()[0];
    OP_CHECK_IF((actualSharedPrefixLen > sPrefix) || (actualSharedPrefixLen < 0), 
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "actualSharedPrefixLen(%ld) must be in range[0, %u]!", actualSharedPrefixLen, sPrefix),
        return false);

    if (actualSharedPrefixLen + sKV > SLIMIT) {
        OP_LOGW(contextKeyParams.opName,
            "Seq size of actualSharedPrefixLen(%ld) + kv_s(%u) should <= 20m", actualSharedPrefixLen, sKV);
    }
    return true;
}

bool PromptFlashAttentionTilingV2::CheckPAKeyValueShape(ContextParamsForPFATiling& contextKeyParams, int64_t& keyDim1,
    PFAShapeInfo& queryShapeInfo, const gert::StorageShape* keyShape, const gert::StorageShape* valueShape,
    const size_t keyDim, const int32_t* blockSize, int64_t blockNumValid, int64_t headNumRatio) 
{
    int64_t keyDim2 = keyShape->GetStorageShape().GetDim(KV_CACHE_DIM_1);
    int64_t keyDim3 = keyShape->GetStorageShape().GetDim(KV_CACHE_DIM_2);
    int64_t keyDim4 = 0;
    int64_t keyDim5 = 0;
    int64_t valueDim1 = valueShape->GetStorageShape().GetDim(KV_CACHE_DIM_0);
    int64_t valueDim2 = valueShape->GetStorageShape().GetDim(KV_CACHE_DIM_1);
    int64_t valueDim3 = valueShape->GetStorageShape().GetDim(KV_CACHE_DIM_2);
    int64_t valueDim4 = 0;
    int64_t valueDim5 = 0;
    if (keyDim == KV_CACHE_DIM_NUMS_4) {
        keyDim4 = keyShape->GetStorageShape().GetDim(KV_CACHE_DIM_3);
        valueDim4 = valueShape->GetStorageShape().GetDim(KV_CACHE_DIM_3);
    }
    if (keyDim == KV_CACHE_DIM_NUMS_5) {
        keyDim4 = keyShape->GetStorageShape().GetDim(KV_CACHE_DIM_3);
        valueDim4 = valueShape->GetStorageShape().GetDim(KV_CACHE_DIM_3);
        keyDim5 = keyShape->GetStorageShape().GetDim(KV_CACHE_DIM_4);
        valueDim5 = valueShape->GetStorageShape().GetDim(KV_CACHE_DIM_4);
    }
    OP_CHECK_IF((keyDim == KV_CACHE_DIM_NUMS_5) && ((keyDim1 != valueDim1) || (keyDim2 != valueDim2) || 
        ((keyDim3 != valueDim3) && (!enablePFAMLA)) || (keyDim4 != valueDim4) || ((keyDim5 != valueDim5) && (!enablePFAMLA))), 
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "The dim num of key and value are inconsistent when PA enable. key shape [%ld, %ld, %ld, %ld, %ld],"
            " value shape [%ld, %ld, %ld, %ld, %ld].",
            keyDim1, keyDim2, keyDim3, keyDim4, keyDim5, valueDim1, valueDim2, valueDim3, valueDim4, valueDim5),
        return false);
    OP_CHECK_IF((keyDim == KV_CACHE_DIM_NUMS_4) && ((keyDim1 != valueDim1) || (keyDim2 != valueDim2) || 
        (keyDim3 != valueDim3) || ((keyDim4 != valueDim4) && (!enablePFAMLA))), 
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "the dim num of key and value are inconsistent when PA enable. key shape [%ld,%ld,%ld,%ld],"
            " value shape [%ld,%ld,%ld,%ld].",
            keyDim1, keyDim2, keyDim3, keyDim4, valueDim1, valueDim2, valueDim3, valueDim4),
        return false);
    OP_CHECK_IF((keyDim == KV_CACHE_DIM_NUMS_3) && ((keyDim1 != valueDim1) || (keyDim2 != valueDim2) || 
        ((keyDim3 != valueDim3) && (!enablePFAMLA))), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "the dim num of key and value are inconsistent when PA enable. key shape [%ld,%ld,%ld], value shape [%ld,%ld,%ld].",
            keyDim1, keyDim2, keyDim3, valueDim1, valueDim2, valueDim3),
        return false);
    std::string layoutStr(contextKeyParams.layout);

    uint32_t dataTypeSizeValue = FLOAT16SIZE;
    std::vector<ge::DataType> allowedDtypes = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E4M3FN};
    std::vector<uint32_t> dataTypeSizeArray = {FLOAT16SIZE, BFLOAT16SIZE, INT8SIZE, FLOAT8SIZE, FLOAT8SIZE, FLOAT8SIZE};

    auto inputTypeCheck = std::find(allowedDtypes.begin(), allowedDtypes.end(), inputType);
    if (inputTypeCheck != allowedDtypes.end()){
        uint32_t inputTypeIndex = std::distance(allowedDtypes.begin(), inputTypeCheck);
        dataTypeSizeValue = dataTypeSizeArray[inputTypeIndex];
    }
    if (enableIFAMLAFullQuant) {
        if (inputLayout == InputLayout::BNSD || inputLayout == InputLayout::TND || inputLayout == InputLayout::NTD) {
            OP_CHECK_IF(((keyDim != KV_CACHE_DIM_NUMS_3) && (keyDim != KV_CACHE_DIM_NUMS_4) && (keyDim != KV_CACHE_DIM_NUMS_5)), 
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, // dim num: 3/4
                "the layout of query is %s, key and value layout should be [>=%ld, %d, %u] or [>=%ld, %u, %d, %u] or [>=%ld, %u, %u, %d, %d] when PA enable.",
                    layoutStr.c_str(), blockNumValid, *blockSize, queryShapeInfo.h / headNumRatio, 
                    blockNumValid, queryShapeInfo.n / headNumRatio, *blockSize, (queryShapeInfo.h / queryShapeInfo.n), 
                    blockNumValid, queryShapeInfo.n / headNumRatio, (queryShapeInfo.h / queryShapeInfo.n) * dataTypeSizeValue / BYTE_BLOCK, *blockSize, BYTE_BLOCK / dataTypeSizeValue),
                return false);
        } else if (inputLayout == InputLayout::BSH || inputLayout == InputLayout::BSND) {
            OP_CHECK_IF(((keyDim != KV_CACHE_DIM_NUMS_3) && (keyDim != KV_CACHE_DIM_NUMS_5)), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "the layout of query is %s, key and value layout should be [>=%ld, %d, %u] or [>=%ld, %u, %u, %d, %d] when PA enable."
                " now key and value shape [%ld, %ld, %ld, %ld].",
                    layoutStr.c_str(), blockNumValid, *blockSize, queryShapeInfo.h / headNumRatio, 
                    blockNumValid, queryShapeInfo.n / headNumRatio, (queryShapeInfo.h / queryShapeInfo.n) * dataTypeSizeValue / BYTE_BLOCK, *blockSize, BYTE_BLOCK / dataTypeSizeValue, 
                    keyDim1, keyDim2, keyDim3, keyDim4),
                return false);
        }
    }
    return true;
}

bool PromptFlashAttentionTilingV2::CheckPACacheShape(ContextParamsForPFATiling& contextKeyParams, const size_t keyDim, PFAShapeInfo& shapeInfo,
    const gert::StorageShape* shape, const int32_t* blockSize, int64_t blockNumValid, int64_t headNumRatio, const std::string& sName) 
{
    std::string layoutStr(contextKeyParams.layout);
    int64_t dim1 = shape->GetStorageShape().GetDim(KV_CACHE_DIM_0);
    int64_t dim2 = shape->GetStorageShape().GetDim(KV_CACHE_DIM_1);
    int64_t dim3 = shape->GetStorageShape().GetDim(KV_CACHE_DIM_2);
    int64_t dim4 = 0;
    int64_t dim5 = 0;
    int64_t tempBlockSize = dim2;
    int64_t tempH = dim3;
    int64_t tempN = 0;
    int64_t tempD = 0;
    int64_t tempD0 = 0;
    int64_t tempD1 = 0;
    if (keyDim == 3) {    // dim num: 3
        paLayoutType = 1; // If it is three-dimensional, paLayoutType = 1
        OP_CHECK_IF(
            ((tempBlockSize != *blockSize) || (tempH * headNumRatio != shapeInfo.h)), 
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "the shape of %s [%ld, %ld, %ld] is wrong, which should be [>=%ld, %d, %u] when PA BnBsH enable", sName.c_str(),
                dim1, dim2, dim3, blockNumValid, *blockSize, shapeInfo.h / headNumRatio),
            return false);
        // In the BSH input of the PA scenario, it is required that the h of the KV matrix does not exceed 65535.
        // The dim and dim3 of the K/V have already been verified to be equal, so only the K matrix is verified here.
        OP_CHECK_IF(dim3 > HLIMIT, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "layout of key/value is BSH, the %s h(%ld) should not > 65535 when PA enable", sName.c_str(), dim3),
            return false);
    } else if (keyDim == 4) {    // dim num: 4
        dim4 = shape->GetStorageShape().GetDim(3);  // 3: The third dimension.
        tempN = dim2;
        tempBlockSize = dim3;
        tempD = dim4;
        paLayoutType = 0; // If it is four-dimensional, paLayoutType = 0
        OP_CHECK_IF(((tempN * headNumRatio != shapeInfo.n) || (tempBlockSize != *blockSize) ||
            (tempD != (shapeInfo.h / shapeInfo.n))), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "the shape of %s [%ld, %ld, %ld, %ld] is wrong, which should be [>=%ld, %u, %d, %u] when PA BnNBsD enable!",
                sName.c_str(), dim1, dim2, dim3, dim4, blockNumValid, shapeInfo.n / headNumRatio, *blockSize,
                (shapeInfo.h / shapeInfo.n)),
            return false);
    } else {
        dim4 = shape->GetStorageShape().GetDim(KV_CACHE_DIM_3);
        dim5 = shape->GetStorageShape().GetDim(KV_CACHE_DIM_4);
        tempN = dim2;
        tempD1 = dim3;
        tempBlockSize = dim4;
        tempD0 = dim5;
        paLayoutType = 2; // If it is five-dimensional, paLayoutType = 2

        uint32_t dataTypeSizeValue = FLOAT16SIZE;
        std::vector<ge::DataType> allowedDtypes = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E4M3FN};
        std::vector<uint32_t> dataTypeSizeArray = {FLOAT16SIZE, BFLOAT16SIZE, INT8SIZE, FLOAT8SIZE, FLOAT8SIZE, FLOAT8SIZE};

        auto inputTypeCheck = std::find(allowedDtypes.begin(), allowedDtypes.end(), inputType);
        if (inputTypeCheck != allowedDtypes.end()){
            uint32_t inputTypeIndex = std::distance(allowedDtypes.begin(), inputTypeCheck);
            dataTypeSizeValue = dataTypeSizeArray[inputTypeIndex];
        }

        if (enableIFAMLAFullQuant && sName == "keyRope") {
            dataTypeSizeValue = BFLOAT16SIZE;
        }

        OP_CHECK_IF(((tempN * headNumRatio != shapeInfo.n) || (tempBlockSize != *blockSize) || (tempD1 * tempD0 != shapeInfo.d) || tempD0 != (BYTE_BLOCK / dataTypeSizeValue)), 
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "the shape of %s [%ld, %ld, %ld, %ld, %ld] is wrong, which should be [>=%ld, %u, %u, %d, %d] when PA NZ enable!",
                sName.c_str(), dim1, dim2, dim3, dim4, dim5, blockNumValid, shapeInfo.n / headNumRatio, shapeInfo.d * dataTypeSizeValue / BYTE_BLOCK, *blockSize, BYTE_BLOCK / dataTypeSizeValue),
            return false);
    }
    return true;
}

bool PromptFlashAttentionTilingV2::CheckBlockTableShape(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo,
    PFAShapeInfo& queryRopeShapeInfo, const int32_t* blockSize, const gert::StorageShape* blockTableShape,
    PromptFlashAttentionTilingDataV2& tilingData) 
{
    const gert::Tensor* actSeqLenKV = contextKeyParams.actualSequenceLengthKV;
    int32_t actualSeqKVPerBatch = 0;
    int32_t blockNumPerBatch = 0;
    int64_t blockNumValid = 0;
    int32_t maxBlockNumPerBatch = 0;
    for (uint32_t i = 0; i < queryShapeInfo.b; i++) {
        actualSeqKVPerBatch = (actSeqLenKV->GetShapeSize() > 1) ? static_cast<int32_t>(actSeqLenKV->GetData<int64_t>()[i]) :
            static_cast<int32_t>(actSeqLenKV->GetData<int64_t>()[0]);
        blockNumPerBatch = (actualSeqKVPerBatch + *blockSize - 1) / *blockSize;
        blockNumValid += blockNumPerBatch;
        if (blockNumPerBatch > maxBlockNumPerBatch) {
            maxBlockNumPerBatch = blockNumPerBatch;
        }
    }
    const gert::StorageShape* keyShape = contextKeyParams.keyInputShape;
    const gert::StorageShape* valueShape = contextKeyParams.valueInputShape;
    const size_t keyDim = keyShape->GetStorageShape().GetDimNum();
    int64_t keyDim1 = keyShape->GetStorageShape().GetDim(KV_CACHE_DIM_0);   
    int64_t headNumRatio = (enableIFAMLA || enableIFA) ?  gSize :
        static_cast<int64_t>(tilingData.promptAttentionBaseParams.get_headNumRatio());
    if (!CheckPAKeyValueShape(contextKeyParams, keyDim1, queryShapeInfo, keyShape, valueShape, keyDim, blockSize,
        blockNumValid, headNumRatio)) {
        return false;
    }
    if (!CheckPACacheShape(contextKeyParams, keyDim, queryShapeInfo, keyShape, blockSize, blockNumValid, headNumRatio, "key")) {
        return false;
    }
    if (enableIFAMLA || enablePFARope) {
        const gert::StorageShape* keyRopeShape = contextKeyParams.keyRopeInputShape;
        const size_t keyRopeDim = keyRopeShape->GetStorageShape().GetDimNum();
        OP_CHECK_IF((keyRopeDim != keyDim),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "the dim of keyRope(%zu) should be equal to the dim of key(%zu) when PA enable!",
                keyRopeDim, keyDim),
            return false);
        if (!CheckPACacheShape(contextKeyParams, keyDim, queryRopeShapeInfo, keyRopeShape, blockSize, blockNumValid,
            headNumRatio, "keyRope")) {
            return false;
        }  
    }

    // check blockTable Shape
    size_t blockTableDim = blockTableShape->GetStorageShape().GetDimNum();
    int64_t blockTableDim1 = blockTableShape->GetStorageShape().GetDim(0);
    string blockTableStr = "";
    for (size_t i = 0; i < blockTableDim; i++) {
        blockTableStr += std::to_string(blockTableShape->GetStorageShape().GetDim(i));
        blockTableStr += ",";
    }
    blockTableStr.pop_back();

    OP_CHECK_IF(blockTableDim != 2, // PA BlockTable dim size
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "the dim of block table must be 2 when PA enable! blockTable dims [%zu], Shape [%s].",
            blockTableDim, blockTableStr.c_str()),
        return false);

    OP_CHECK_IF(((blockTableDim1 != queryShapeInfo.b) || (blockTableDim2 < maxBlockNumPerBatch)),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "block table shape should be [%u, >=%d], now is [%ld, %d] when PA enable!",
            queryShapeInfo.b, maxBlockNumPerBatch, blockTableDim1, blockTableDim2),
        return false);

    paBlockNumSum = keyDim1;
    S2 = maxBlockNumPerBatch * (*blockSize);
    if (S2 > SLIMIT) {
        OP_LOGW(contextKeyParams.opName,
            "seq should <= 20m, but kv seq = %u when PA enable", S2);
    }
    return true;
}

bool PromptFlashAttentionTilingV2::CheckMaskShape(ContextParamsForPFATiling& contextKeyParams, const int32_t* sparseMode,
    int64_t& attenMaskBatch, int64_t& attenMaskS1, int64_t& attenMaskS2, bool& checkMask, const uint64_t sQ, const uint64_t sK,
    const uint32_t batchSize, std::string& strMaskShape) 
{
    int64_t attenMaskN = 1U;
    const gert::StorageShape* attenMaskShape = contextKeyParams.attentionMaskShape;
    size_t attenMaskDim = attenMaskShape->GetStorageShape().GetDimNum();
    if (attenMaskDim == MASKDIM_SS) {
        if (isDefaultSparseMode || (sparseMode != nullptr && *sparseMode == SPARSE_MODE_ALL_MASK)) { // sparse 0、1时不支持二维mask
            OP_LOGE(contextKeyParams.opName, "The current dimension of the mask is 2. "
                "When sparseMode is 0 or 1, the mask dimension only supports 3 and 4. "
                "Please use 3D mask \[B,QS,KVS\]\/\[1,QS,KVS\] or 4D mask \[B,1,QS,KVS\]\/\[1,1,QS,KVS\].");
            return false;
        } else {
            attenMaskS1 = attenMaskShape->GetStorageShape().GetDim(0);
            attenMaskS2 = attenMaskShape->GetStorageShape().GetDim(1);
            strMaskShape = std::to_string(attenMaskS1) + ", " + std::to_string(attenMaskS2);
        }
    } else if (attenMaskDim == MASKDIM_1SS_BSS) {
        attenMaskBatch = attenMaskShape->GetStorageShape().GetDim(0);
        attenMaskS1 = attenMaskShape->GetStorageShape().GetDim(1);
        attenMaskS2 = attenMaskShape->GetStorageShape().GetDim(2); // 2: When the dim is 3, the second dimension is S2.
        strMaskShape = std::to_string(attenMaskBatch) + ", " + std::to_string(attenMaskS1) + ", " + 
            std::to_string(attenMaskS2);
    } else if (attenMaskDim == MASKDIM_11SS_B1SS) {
        attenMaskBatch = attenMaskShape->GetStorageShape().GetDim(0);
        attenMaskN = attenMaskShape->GetStorageShape().GetDim(1);
        attenMaskS1 = attenMaskShape->GetStorageShape().GetDim(2); // 2: When the dim is 4, the second dimension is S1.
        attenMaskS2 = attenMaskShape->GetStorageShape().GetDim(3); // 3: When the dim is 4, the third dimension is S2.
        strMaskShape = std::to_string(attenMaskBatch) + ", " + std::to_string(attenMaskN) + ", " + 
            std::to_string(attenMaskS1) + ", " + std::to_string(attenMaskS2);
    } else {
        OP_LOGE(contextKeyParams.opName, "attenMask dim(%zu) must be 2 or 3 or 4!", attenMaskDim);
        return false;
    }

    if (isDefaultSparseMode || (sparseMode != nullptr && *sparseMode == SPARSE_MODE_ALL_MASK)) {
        checkMask = (attenMaskS1 >= sQ) && (attenMaskS2 >= sK) &&
            (attenMaskBatch == 1 || attenMaskBatch == batchSize) && (attenMaskN == 1);
        if (attenMaskN != 1) {
            OP_LOGE(contextKeyParams.opName, "The second dimension of the 4D mask must be 1, "
                "but now it is %lld!", attenMaskN);
        }
    } else if ((sparseMode != nullptr) && ((*sparseMode == SPARSE_MODE_LEFT_UP) ||
        (*sparseMode == SPARSE_MODE_RIGHT_DOWN) || (*sparseMode == SPARSE_MODE_BAND))) {
        checkMask = (attenMaskBatch == 1) && (attenMaskN == 1) &&
            (attenMaskS1 == SPARSE_OPTIMIZE_ATTENTION_SIZE) && (attenMaskS2 == SPARSE_OPTIMIZE_ATTENTION_SIZE);
    }
    return true;
}

void PromptFlashAttentionTilingV2::SetSparseModeData(ContextParamsForPFATiling& contextKeyParams,
    const gert::StorageShape* attenMaskShape, const int32_t* sparseMode, const int64_t* preTokens,
    const int64_t* nextTokens) 
{
    if (*preTokens > SPARSE_MODE_INT_MAX) {
        sparsePreTokens = SPARSE_MODE_INT_MAX;
    } else if (*preTokens < -(SPARSE_MODE_INT_MAX)) {
        sparsePreTokens = -(SPARSE_MODE_INT_MAX);
    } else {
        sparsePreTokens = *preTokens;
    }

    if (*nextTokens > SPARSE_MODE_INT_MAX) {
        sparseNextTokens = SPARSE_MODE_INT_MAX;
    } else if (*nextTokens < -(SPARSE_MODE_INT_MAX)) {
        sparseNextTokens = -(SPARSE_MODE_INT_MAX);
    } else {
        sparseNextTokens = *nextTokens;
    }
    if ((attenMaskShape != nullptr) && (sparseMode != nullptr)) {
        if (*sparseMode == SPARSE_MODE_LEFT_UP) {
            sparsePreTokens = SPARSE_MODE_INT_MAX;
            sparseNextTokens = 0;
        // Right down tokens are calculated on the kernel side.
        } else if (*sparseMode == SPARSE_MODE_RIGHT_DOWN) {
            sparsePreTokens = SPARSE_MODE_INT_MAX;
        } else if (*sparseMode == SPARSE_MODE_ALL_MASK) {
            sparsePreTokens = SPARSE_MODE_INT_MAX;
            sparseNextTokens = SPARSE_MODE_INT_MAX;
        } else if (*sparseMode == SPARSE_MODE_BAND) {
            isBandMode = true;
        }
        sparseModeVal = *sparseMode;
        OP_LOGI(contextKeyParams.opName, "sparseMode is %d.", sparseModeVal);
    }
    if ((sparseMode != nullptr) && (*sparseMode == SPARSE_MODE_NO_MASK)) {
        // sparse mode need process attention mask when empty tensor scenes as same.
        if (((attenMaskShape != nullptr) && (attenMaskShape->GetStorageShape().GetShapeSize() == 0)) || 
            (attenMaskShape == nullptr)) {
            sparsePreTokens = SPARSE_MODE_INT_MAX;
            sparseNextTokens = SPARSE_MODE_INT_MAX;
            sparseModeVal = *sparseMode;
        }
    }
    if (isDefaultSparseMode && enableLeftPadding) {
        // For scenes with sparse mode=0 and left padding, the attention mask part is fully calculated
        sparsePreTokens = SPARSE_MODE_INT_MAX;
        sparseNextTokens = SPARSE_MODE_INT_MAX;
    }
}

bool PromptFlashAttentionTilingV2::CheckMaskShapeCrossSparse(ContextParamsForPFATiling& contextKeyParams,
    PromptFlashAttentionTilingDataV2& tilingData, const int32_t* sparseMode, uint64_t sQ, const uint64_t sK,
    const uint32_t batchSize) 
{
    if (isMaxWorkspace || !enableMask) {
        return true;
    }
    if (enableIFA || enableIFAMLA || enablePFAMerge) {
        sQ /= gSize; // 合轴场景使用原始的seq长度校验
    }
    int64_t attenMaskBatch = 1;
    int64_t attenMaskS1 = 0;
    int64_t attenMaskS2 = 0;
    bool checkMask = 0;
    std::string strMaskShape;
    if (!CheckMaskShape(contextKeyParams, sparseMode, attenMaskBatch, attenMaskS1, attenMaskS2,
        checkMask, sQ, sK, batchSize, strMaskShape)) {
        return false;
    }
    
    if (isDefaultSparseMode || ((sparseMode != nullptr) && (*sparseMode == SPARSE_MODE_ALL_MASK))) {
        OP_CHECK_IF(!checkMask,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "attenMask batch(%ld) must be 1 or %u, attenMask Q_S(%ld) must be larger than or equal to sQ(%u),"
                "attenMask KV_S(%ld) must be larger than or equal to sK(%u), please check",
                attenMaskBatch, batchSize, attenMaskS1, sQ, attenMaskS2, sK),
            return false);
    }
    if ((sparseMode != nullptr) && ((*sparseMode == SPARSE_MODE_LEFT_UP) ||
        (*sparseMode == SPARSE_MODE_RIGHT_DOWN) || (*sparseMode == SPARSE_MODE_BAND)) && !checkMask) {
        OP_LOGE(contextKeyParams.opName,
            "attenMask shape must be (2048, 2048) or (1, 2048, 2048) or (1, 1, 2048, 2048) when sparse mode = %d, but now it's (%s).",
                *sparseMode, strMaskShape.c_str());
        return false;
    }
    tilingData.promptAttentionSingleCoreParams.set_attenMaskBatch(attenMaskBatch);
    attenMaskShapeType = attenMaskBatch > 1 ? 1 : 2; // 1 for multi-batch and 2 for 1 batch, same as fa
    return true;
}

bool PromptFlashAttentionTilingV2::CheckPFAMerge(ContextParamsForPFATiling& contextKeyParams,
    const PFAShapeInfo& queryShapeInfo) const 
{
    const int32_t pfaMergeGSLimit = pfaMergeQsLimit * pfaMergeGLimit;
    if (queryShapeInfo.d > 256U && (queryShapeInfo.d % 64) != 0) { // 256U, 64: d > 256 must be multiple of 64 for memory alignment
        return false;
    }

    const int64_t nQ = *contextKeyParams.headsNumber;
    const int64_t nKV = *contextKeyParams.numKeyValueHeads;
    if ((nKV > 0) && (static_cast<uint64_t>(nQ / nKV) * queryShapeInfo.s > pfaMergeGSLimit)) {
        return false;
    }
    if ((nKV == 0) && (queryShapeInfo.s > pfaMergeGSLimit)) {
        return false;
    }

    // 隔离高阶特性
    std::string layoutStr(contextKeyParams.layout);
    bool isTransposeLayout = layoutStr == "BNSD_BSND" || layoutStr == "BSND_BNSD" || layoutStr == "BSH_BNSD" ||
        layoutStr == "NTD" || layoutStr == "NTD_TND" || layoutStr == "TND_NTD";
    bool hasCrossoverAttr = enablePseShift || enableAlibiPse ||
        enablePerblockQuant || enablePertensorQuant || enablePostQuant || enableLeftPadding || enableTensorList ||
        enableIFAMLAFullQuant || contextKeyParams.isSoftMaxLseEnable || isTransposeLayout || enableLearnSink;

    return !hasCrossoverAttr;
}

bool PromptFlashAttentionTilingV2::CheckIO(ContextParamsForPFATiling& contextKeyParams,
    PFAShapeInfo& queryShapeInfo, PFAShapeInfo& valueShapeInfo) 
{
    const gert::StorageShape* queryShape = contextKeyParams.queryInputShape;
    const gert::StorageShape* outShape = contextKeyParams.outputShape;
    const gert::StorageShape* valueShape = contextKeyParams.valueInputShape;
    // check dtype
    if (!CheckIODataType(contextKeyParams)) {
        return false;
    }

    // check layout
    OP_CHECK_IF((!SetInputLayout(contextKeyParams.layout)),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, 
            "Invalid input layout:%s. Currently only TND/NTD/BSH/BNSD/BSND/BSND_BNSD/BNSD_BSND/BSH_BNSD/BSND_NBSD/BNSD_NBSD/BSH_NBSD layout are supported.",
            contextKeyParams.layout),
        return false);

    // check query shape size and bnsd
    OP_CHECK_IF((!GetAndCheckShape(contextKeyParams, queryShapeInfo, queryShape, "query")),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Get and check query shape failed."),
        return false);
    if (queryShapeInfo.s == 1) {
        if (inputLayout != InputLayout::NTD) {
 	  	    enableIFAMask = true;
 	  	}
        std::string layoutStr(contextKeyParams.layout);
        bool isTransposeLayout = layoutStr == "BNSD_BSND" || layoutStr == "BSND_BNSD" || layoutStr == "BSH_BNSD" ||
            layoutStr == "NTD" || layoutStr == "NTD_TND";
        if (!enableAlibiPse && !enablePerblockQuant && !isTransposeLayout) {
            enableIFA = true;
        }
    }
    enablePFAMerge = CheckPFAMerge(contextKeyParams, queryShapeInfo);
    // check value shape
    if (enablePA) { // only get shape
        int64_t b = 0;
        int64_t n = 0;
        int64_t s = 0;
        int64_t d = 0;
        int64_t h = 0;
        int64_t t = 0;
        if (!SetShape(contextKeyParams, valueShape, "value", b, n, s, d, h, t)) {
            return false;
        }
        valueShapeInfo.d = d;
    } else {
        OP_CHECK_IF((!GetAndCheckShape(contextKeyParams, valueShapeInfo, valueShape, "value")),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Get and check value shape failed."),
        return false);
    }
    if (queryShapeInfo.d == MLA_QKD_SIZE && valueShapeInfo.d == MLA_VD_SIZE) {
        enablePFAMLA = true;
    }

    if(queryShapeInfo.d != valueShapeInfo.d && !enablePFAMLA){
        isQKVDDifferent = true;
    }
    OP_CHECK_IF(isQKVDDifferent && (contextKeyParams.inputDataType != ge::DT_FLOAT16 && contextKeyParams.inputDataType != ge::DT_BF16),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Query data type must be float16 or bf16 when query and key headdim is not equal to value headdim."),
        return false);
    OP_CHECK_IF(isQKVDDifferent && (outputType != ge::DT_FLOAT16 && outputType != ge::DT_BF16),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Output data type must be float16 or bf16 when query and key headdim is not equal to value headdim."),
        return false);
    OP_CHECK_IF(isQKVDDifferent && (queryShapeInfo.d > 128),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Query headdim must smaller than 128 when query and key headdim is not equal to value headdim."),
        return false);
    OP_CHECK_IF(isQKVDDifferent && (valueShapeInfo.d > 128),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Value headdim must smaller than 128 when query and key headdim is not equal to value headdim."),
        return false);
    // check query and output consistency
    OP_CHECK_IF((!CheckQueryOutParamsConsistency(contextKeyParams, queryShape, outShape)),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Query and output consistency check failed."),
        return false);
    return true;
}

bool PromptFlashAttentionTilingV2::CheckKV(ContextParamsForPFATiling& contextKeyParams,
    PFAShapeInfo& keyShapeInfo, PFAShapeInfo& valueShapeInfo) 
{
    const gert::StorageShape* keyShape = contextKeyParams.keyInputShape;
    const gert::StorageShape* valueShape = contextKeyParams.valueInputShape;

    // check dtype
    if (!CheckKVDataType(contextKeyParams)) {
        return false;
    }

    // check key shape size and bnsd
    OP_CHECK_IF((!enablePA && !GetAndCheckShape(contextKeyParams, keyShapeInfo, keyShape, "key")),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Get and check key shape failed."),
        return false);
    
    // check value shape size and bnsd
    OP_CHECK_IF((!enablePA && !GetAndCheckShape(contextKeyParams, valueShapeInfo, valueShape, "value")),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Get and check value shape failed."),
        return false);

    OP_CHECK_IF(isQKVDDifferent && (contextKeyParams.kDataType != ge::DT_FLOAT16 && contextKeyParams.kDataType != ge::DT_BF16),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Key type must must be float16 or bf16 when query and key headdim is not equal to value headdim."),
        return false);
    OP_CHECK_IF(isQKVDDifferent && (contextKeyParams.vDataType != ge::DT_FLOAT16 && contextKeyParams.vDataType != ge::DT_BF16),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Value type must must be float16 or bf16 when query and key headdim is not equal to value headdim."),
        return false);
    OP_CHECK_IF(isQKVDDifferent && (keyShapeInfo.d > 128),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Key headdim must smaller than 128 when query and key headdim is not equal to value headdim."),
        return false);
    OP_CHECK_IF(isQKVDDifferent && (valueShapeInfo.d > 128),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Value headdim must smaller than 128 when query and key headdim is not equal to value headdim."),
        return false);
    // check key and value consistency
    OP_CHECK_IF((!CheckKeyValueParamsConsistency(contextKeyParams, keyShape, valueShape)),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Key and value consistency check failed."),
        return false);

    return true;
}

bool PromptFlashAttentionTilingV2::CheckQueryAndKey(ContextParamsForPFATiling& contextKeyParams,
    PFAShapeInfo& queryShapeInfo, PFAShapeInfo& keyShapeInfo, PromptFlashAttentionTilingDataV2& tilingData) 
{
    // check numhead ratio
    if (!SetAndCheckHeadNumRatio(contextKeyParams, tilingData)) {
        return false;
    }

    // check b and s for not PA
    OP_CHECK_IF(!isMaxWorkspace && (queryShapeInfo.b != keyShapeInfo.b) &&
        (!enableTensorList) && (!enablePA), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "query batch must be equal to key/value batch, query batch = %u , key/value batch = %u.",
        queryShapeInfo.b, keyShapeInfo.b), return false);
    // check d size
    OP_CHECK_IF(!isMaxWorkspace && (queryShapeInfo.d != keyShapeInfo.d) &&
        (!enableTensorList) && (!enablePA), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "query d size must be equal to key d size, query d = %u , key d = %u.",
        queryShapeInfo.d, keyShapeInfo.d), return false);
    return true;
}

bool PromptFlashAttentionTilingV2::CheckIFAMLA(ContextParamsForPFATiling& contextKeyParams, const PFAShapeInfo& queryShapeInfo) const 
{
    if (enableIFAMLAFullQuant) {
        constexpr uint32_t maxQuerySeqLenInIfaMla = 16U; // ifa mla场景qS最大支持16
        OP_CHECK_IF((queryShapeInfo.s > maxQuerySeqLenInIfaMla || queryShapeInfo.s < 1),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "input query's sequence length is %u, it should be "
                "in range of [1, %u] when enable ifa mla fullquant", queryShapeInfo.s, maxQuerySeqLenInIfaMla),
            return false);
        static const std::set<uint32_t> supportNumHeadInIfaMlaInt8 = {1U, 2U, 4U, 8U, 16U, 32U, 64U, 128U}; // ifa mla fullquant int8场景qN支持范围
        OP_CHECK_IF((supportNumHeadInIfaMlaInt8.find(queryShapeInfo.n) == supportNumHeadInIfaMlaInt8.end() && contextKeyParams.inputDataType == ge::DT_INT8),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "input query's heads num is %u, it should be in range of "
                "{1, 2, 4, 8, 16, 32, 64, 128} when enable ifa mla int8 fullquant", queryShapeInfo.n),
            return false);
        static const std::set<uint32_t> supportNumHeadInIfaMlaFp8 = {32U, 64U, 128U}; // ifa mla fp8场景qN支持范围
        OP_CHECK_IF((supportNumHeadInIfaMlaFp8.find(queryShapeInfo.n) == supportNumHeadInIfaMlaFp8.end() && contextKeyParams.inputDataType == ge::DT_FLOAT8_E4M3FN),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "input query's heads num is %u, it should be in range of "
                "{32, 64, 128} when enable ifa mla fp8 fullquant", queryShapeInfo.n),
            return false);
    } else {
        OP_CHECK_IF((queryShapeInfo.s < 1),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "input query's sequence length is %u, it should be "
                "larger than 0 when enable ifa mla", queryShapeInfo.s),
            return false);
        static const std::set<uint32_t> supportNumHeadInIfaMla = {1U, 2U, 4U, 8U, 16U, 32U, 64U, 128U}; // ifa mla场景qN支持范围
        OP_CHECK_IF((supportNumHeadInIfaMla.find(queryShapeInfo.n) == supportNumHeadInIfaMla.end()),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "input query's heads num is %u, it should be in range of "
                "{1, 2, 4, 8, 16, 32, 64, 128} when enable ifa mla", queryShapeInfo.n),
            return false);
    }
    const int64_t nKV = *contextKeyParams.numKeyValueHeads; // ifa mla场景不支持g = 1, 因此在nKV用默认值0, nQ替代也属于异常场景
    OP_CHECK_IF((nKV != 1U),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "input key/value's heads num is %u, it should be 1 when enable "
            "ifa mla", nKV),
        return false);
    return true;
}

bool PromptFlashAttentionTilingV2::CheckRope(ContextParamsForPFATiling& contextKeyParams,
    PFAShapeInfo& queryShapeInfo, PFAShapeInfo& keyShapeInfo, PFAShapeInfo& queryRopeShapeInfo) 
{
    if (contextKeyParams.queryRopeInputShape == nullptr && contextKeyParams.keyRopeInputShape == nullptr) {
        return true;
    }
    OP_CHECK_IF((contextKeyParams.queryRopeInputShape == nullptr && contextKeyParams.keyRopeInputShape != nullptr),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "queryRope is null, but keyRope is not null, "
        "they should be consistent."), return false);
    OP_CHECK_IF((contextKeyParams.queryRopeInputShape != nullptr && contextKeyParams.keyRopeInputShape == nullptr),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "queryRope is not null, but keyRope is null, "
        "they should be consistent."), return false);
    OP_CHECK_IF(!(CheckRopeDataType(contextKeyParams)),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Check rope datatype failed."),
        return false);
    const gert::StorageShape* queryRopeShape = contextKeyParams.queryRopeInputShape;
    const gert::StorageShape* keyRopeShape = contextKeyParams.keyRopeInputShape;

    // check queryRope shape
    OP_CHECK_IF((!GetAndCheckRopeShape(contextKeyParams, queryShapeInfo, queryRopeShapeInfo, queryRopeShape, "query", "queryRope")),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Get and check queryRope shape failed."),
        return false);
    // check keyRope shape
    PFAShapeInfo keyRopeShapeInfo;
    OP_CHECK_IF((!enablePA && !GetAndCheckRopeShape(contextKeyParams, keyShapeInfo, keyRopeShapeInfo, keyRopeShape, "key", "keyRope")),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Get and check queryRope shape failed."),
        return false);
    enableIFA = false;
    enableIFAMask = false;
    enablePFAMerge = false;
    if (queryShapeInfo.d == QUERY_SHAPE_DIM_D_128_TILING_V2) {
        enablePFARope = true;
    } else {
        enableIFAMLA = true;
    }
    OP_LOGI(contextKeyParams.opName, "enableIFAMLA is %d, enablePA is %d, enableMask is %d, faRunFlag_ is %d", 
                enableIFAMLA, enablePA, enableMask, faRunFlag_);
    return true;
}

bool PromptFlashAttentionTilingV2::CheckLayout(ContextParamsForPFATiling& contextKeyParams) const 
{
    if (contextKeyParams.layout == nullptr) {
        return false;
    }
    std::string layoutStr(contextKeyParams.layout);
    const std::vector<std::string> unsupportedLayoutList = {"NTD", "BSND_BNSD", "BSH_BNSD", "BNSD_BSND", "NTD_TND"};
    const std::vector<std::string> unsupportedLayoutList2 = {"BNSD_NBSD", "BSH_NBSD", "BSND_NBSD", "TND_NTD"};
    if (enableIFAMLA) {
        OP_CHECK_IF(std::find(unsupportedLayoutList.begin(), unsupportedLayoutList.end(), layoutStr) != unsupportedLayoutList.end(),
            OP_LOGE(contextKeyParams.opName, "When decode mla scenario is applied, layout does not support NTD, BSND_BNSD, BSH_BNSD, "
            "BNSD_BSND, NTD_TND, but got %s", layoutStr.c_str()),
            return false);
    } else if (!enablePertensorQuant && !enablePerblockQuant) {
        OP_CHECK_IF(std::find(unsupportedLayoutList2.begin(), unsupportedLayoutList2.end(), layoutStr) != unsupportedLayoutList2.end(),
            OP_LOGE(contextKeyParams.opName, "When prefill mla or gqa scenario is applied, layout does not support BNSD_NBSD, BSH_NBSD, "
            "BSND_NBSD, TND_NTD, but got %s", layoutStr.c_str()),
            return false);
    }
    return true;
}

bool PromptFlashAttentionTilingV2::CheckQuant(ContextParamsForPFATiling& contextKeyParams,
    PFAShapeInfo& queryShapeInfo, PFAShapeInfo& keyShapeInfo, const PFAShapeInfo& valueShapeInfo) const
{
    const gert::StorageShape* deqScale1Shape = contextKeyParams.deqScale1Shape;
    const gert::StorageShape* quantScale1Shape = contextKeyParams.scale1Shape;
    const gert::StorageShape* deqScale2Shape = contextKeyParams.deqScale2Shape;
    const gert::StorageShape* keyShape = contextKeyParams.keyInputShape;
    // per-tensor quant check
    if (enablePertensorQuant) {
        OP_CHECK_IF(!CheckPerTensorQuantParams(contextKeyParams, queryShapeInfo),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "per-tensor quant params check failed!"),
            return false);
        const size_t keyDim = keyShape->GetStorageShape().GetDimNum();
        OP_CHECK_IF((keyDim == KV_CACHE_DIM_NUMS_5),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "PA_NZ is not support in per-tensor quant scenario."),
            return false);
    }

    // per-block quant check
    if (enablePerblockQuant) {
        OP_CHECK_IF(!CheckPerblockQuantParams(contextKeyParams, queryShapeInfo, keyShapeInfo, valueShapeInfo),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "per-block quant params check failed!"),
            return false);
    }

    OP_CHECK_IF(((contextKeyParams.inputDataType == ge::DT_INT8) && (contextKeyParams.outputDataType == ge::DT_FLOAT16) &&
        ((contextKeyParams.scale2Shape != nullptr) || (contextKeyParams.offset2Shape != nullptr))),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "When query dtype is int8 and output dtype is fp16, quantScale2 and quantOffset2 should be null."),
        return false);

    // quant/dequant dtype check
    if ((inputType == ge::DT_FLOAT16) && enablePostQuant) {
        OP_CHECK_IF((deqScale1Shape != nullptr) || (quantScale1Shape != nullptr) || (deqScale2Shape != nullptr),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "When input dtype is fp16 and postQuant is enable, "
                "dequantScale1, quantScale1 and dequantScale2 should be null."),
            return false);
    }

    // post-quant check
    if (enablePostQuant) {
        OP_CHECK_IF(!CheckPostQuantParams(contextKeyParams, queryShapeInfo, valueShapeInfo),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "post quant params check failed!"),
            return false);
    }
    OP_CHECK_IF(enableKVAntiquant && !CheckAntiquantParamsShape(contextKeyParams),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "antiquant params check failed!"),
        return false);
    OP_CHECK_IF(enableKVAntiquant && (enableIFAMLA || enablePFARope || enablePFAMLA),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "MLA do not support antiquant."),
        return false);
    return true;
}

bool PromptFlashAttentionTilingV2::CheckQScaleShape4MLAFullQuant(ContextParamsForPFATiling& contextKeyParams) const 
{
    auto queryInputShape = contextKeyParams.queryInputShape->GetStorageShape();
    int32_t qDimNum = static_cast<int32_t>(queryInputShape.GetDimNum());
    auto dequantScaleQueryShape = contextKeyParams.dequantScaleQueryShape->GetStorageShape();
    int32_t qScaleDimNum = static_cast<int32_t>(dequantScaleQueryShape.GetDimNum());
    std::string layoutStr(contextKeyParams.layout);
    if (layoutStr == "BSH") {
        OP_CHECK_IF((qDimNum != qScaleDimNum),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "When MLAFullQuant enables and the layout of query is %s, dim num of dequantScaleQuery(%d) should be equal to queryDimNum(%d).",
                layoutStr.c_str(), qScaleDimNum, qDimNum),
            return false);
        OP_CHECK_IF((queryInputShape.GetDim(0) != dequantScaleQueryShape.GetDim(0)),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "When MLAFullQuant enables and the layout of query is %s, the %drd dim of dequantScaleQuery(%d) should be equal to queryInputShape(%d).",
                layoutStr.c_str(), 0, static_cast<int32_t>(dequantScaleQueryShape.GetDim(0)), static_cast<int32_t>(queryInputShape.GetDim(0))),
            return false);
        OP_CHECK_IF((queryInputShape.GetDim(1) != dequantScaleQueryShape.GetDim(1)),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "When MLAFullQuant enables and the layout of query is %s, the %drd dim of dequantScaleQuery(%d) should be equal to queryInputShape(%d).",
                layoutStr.c_str(), 1, static_cast<int32_t>(dequantScaleQueryShape.GetDim(1)),static_cast<int32_t>(queryInputShape.GetDim(1))),
            return false);
        OP_CHECK_IF((*contextKeyParams.headsNumber != dequantScaleQueryShape.GetDim(2)),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "When MLAFullQuant enables and the layout of query is %s, the %drd dim of dequantScaleQuery(%d) should be equal to numHeads(%d).",
                layoutStr.c_str(), 2, static_cast<int32_t>(dequantScaleQueryShape.GetDim(2)), *contextKeyParams.headsNumber),
            return false);
    } else if (layoutStr == "BSND" || layoutStr == "BNSD" || layoutStr == "TND") {
        OP_CHECK_IF((qScaleDimNum != qDimNum - 1),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "When MLAFullQuant enables and the layout of query is %s, dim num of dequantScaleQuery(%d) should be equal to [queryDimNum - 1(%d)].",
                layoutStr.c_str(), qScaleDimNum, qDimNum - 1),
            return false);
        for (uint32_t i = 0; i < qScaleDimNum; i++) {
            OP_CHECK_IF((queryInputShape.GetDim(i) != dequantScaleQueryShape.GetDim(i)),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "When MLAFullQuant enables and the layout of query is %s, the %urd dim of dequantScaleQuery(%d) should be equal to queryInputShape(%d).",
                    layoutStr.c_str(), i, static_cast<int32_t>(dequantScaleQueryShape.GetDim(i)), static_cast<int32_t>(queryInputShape.GetDim(i))),
                return false);
        }
    }
    return true;
}

bool PromptFlashAttentionTilingV2::CheckKVScaleShape4MLAFullQuant(ContextParamsForPFATiling& contextKeyParams) const 
{
    const gert::StorageShape* keyAntiquantScaleShape = contextKeyParams.KeyAntiquantScaleShape;
    const gert::StorageShape* valueAntiquantScaleShape = contextKeyParams.valueAntiquantScaleShape;

    OP_CHECK_IF((keyAntiquantScaleShape->GetStorageShape().GetDimNum() != 1 || valueAntiquantScaleShape->GetStorageShape().GetDimNum() != 1),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "When MLAFullQuant enables, dim num of keyAntiquantScaleShape(%zu) or valueAntiquantScaleShape(%zu) must be 1.",
            static_cast<int32_t>(keyAntiquantScaleShape->GetStorageShape().GetDimNum()),
            static_cast<int32_t>(valueAntiquantScaleShape->GetStorageShape().GetDimNum())),
        return false);
    OP_CHECK_IF((keyAntiquantScaleShape->GetStorageShape().GetDim(0) != 1 || valueAntiquantScaleShape->GetStorageShape().GetDim(0) != 1),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "When MLAFullQuant enables, the %drd dim of keyAntiquantScaleShape(%d) or valueAntiquantScaleShape(%d) must be 1.",
            0, keyAntiquantScaleShape->GetStorageShape().GetDim(0), valueAntiquantScaleShape->GetStorageShape().GetDim(0)),
        return false);
    return true;
}

bool PromptFlashAttentionTilingV2::CheckMLAFullQuant(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo) const
{
    // check layout
    std::string layoutStr(contextKeyParams.layout);
    const std::vector<std::string> supportedLayoutList = {"BSH", "BSND", "BNSD", "TND"};
    OP_CHECK_IF(std::find(supportedLayoutList.begin(), supportedLayoutList.end(), layoutStr) == supportedLayoutList.end() && contextKeyParams.inputDataType == ge::DT_FLOAT8_E4M3FN,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "When MLAFullQuant enables, the layout of Q(%s) must be BSH/BSND/BNSD/TND.", layoutStr.c_str()), return false);
    // check DSize
    OP_CHECK_IF((queryShapeInfo.d != 512), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "When MLAFullQuant enables, the d(%d) size of query should be 512.", queryShapeInfo.d), return false);
    // 新增对int8的拦截,关于layout的拦截
    const std::vector<std::string> supportedLayoutListInt8 = {"BSH", "BSND", "TND", "BSH_NBSD", "BSND_NBSD", "TND_NTD"};
    OP_CHECK_IF(std::find(supportedLayoutListInt8.begin(), supportedLayoutListInt8.end(), layoutStr) == supportedLayoutListInt8.end() && contextKeyParams.inputDataType == ge::DT_INT8,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "When MLAFullQuant INT8 is enabled, the layout of Q(%s) must be BSH/BSND/TND/BSH_NBSD/BSND_NBSD/TND_NTD.",
            layoutStr.c_str()),
        return false);
    // 新增对int8的拦截,关于输入K/V必须为五维的拦截
    const gert::StorageShape* keyShape = contextKeyParams.keyInputShape;
    OP_CHECK_IF((keyShape->GetStorageShape().GetDimNum()!= KV_CACHE_DIM_NUMS_5 && contextKeyParams.inputDataType == ge::DT_INT8),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "when the dtype of query is int8 in MLA, KV layout must be PANZ."), return false);
    // check QKV dtype for fp8_e4m3/int8, output dtype for bf16, QK Rope Type for bf16
    std::vector<ge::DataType> allowedDtypes = {ge::DT_FLOAT8_E4M3FN, ge::DT_INT8};
    auto inputTypeCheck = std::find(allowedDtypes.begin(), allowedDtypes.end(), contextKeyParams.inputDataType);
    OP_CHECK_IF((inputTypeCheck == allowedDtypes.end() || contextKeyParams.outputDataType != ge::DT_BF16),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "When MLAFullQuant enables, dataType of Q(%s), K(%s) and V(%s) must be fp8_e4m3/int8, datatype of output(%s) must be bf16.",
            GetPfaDataTypeStr(contextKeyParams.inputDataType).c_str(), GetPfaDataTypeStr(contextKeyParams.kDataType).c_str(),
            GetPfaDataTypeStr(contextKeyParams.vDataType).c_str(), GetPfaDataTypeStr(contextKeyParams.outputDataType).c_str()),
        return false);
    OP_CHECK_IF((contextKeyParams.qRopeDataType != ge::DT_BF16 || contextKeyParams.kRopeDataType != ge::DT_BF16),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When MLAFullQuant enables, dataType of queryRope(%s) and keyRope(%s) must be bf16.",
            GetPfaDataTypeStr(contextKeyParams.qRopeDataType).c_str(), GetPfaDataTypeStr(contextKeyParams.kRopeDataType).c_str()), return false);
    // check QKV QuantMode
    OP_CHECK_IF((*contextKeyParams.queryQuantMode != static_cast<int64_t>(AntiquantTypeEnum::PER_TOKEN_HEAD)),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When MLAFullQuant enables, queryQuantMode (%ld) is InValid! "
            "Only support Per-Token-Head(3).", *contextKeyParams.queryQuantMode), return false);
    OP_CHECK_IF((*contextKeyParams.keyAntiquantMode != static_cast<int64_t>(AntiquantTypeEnum::PER_CHANNEL)||
        *contextKeyParams.valueAntiquantMode != static_cast<int64_t>(AntiquantTypeEnum::PER_CHANNEL)),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "When MLAFullQuant enables, keyAntiquantMode (%ld) or valueQuantMode (%ld) is InValid! Only support Per-Tensor(0).",
            *contextKeyParams.keyAntiquantMode, *contextKeyParams.valueAntiquantMode), return false);
    // check QKV scale
    OP_CHECK_IF((contextKeyParams.dequantScaleQuery == nullptr), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "When MLAFullQuant enables, dequantScaleQuery should not be nullptr."), return false);
    OP_CHECK_IF((contextKeyParams.keyAntiquantScale == nullptr || contextKeyParams.valueAntiquantScale == nullptr),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "When MLAFullQuant enables, keyAntiQuantScale or valueAntiquantScale should not be nullptr."), return false);
    OP_CHECK_IF((contextKeyParams.dequantScaleQueryType != ge::DT_FLOAT || contextKeyParams.KeyAntiquantScaleType != ge::DT_FLOAT ||
        contextKeyParams.valueAntiquantScaleType != ge::DT_FLOAT), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "When MLAFullQuant enables, dataType of dequantScaleQuery(%s), KeyAntiquantScale(%s) and valueAntiquantScale(%s) must be float32.",
            GetPfaDataTypeStr(contextKeyParams.dequantScaleQueryType).c_str(), GetPfaDataTypeStr(contextKeyParams.KeyAntiquantScaleType).c_str(),
            GetPfaDataTypeStr(contextKeyParams.valueAntiquantScaleType).c_str()), return false);
    OP_CHECK_IF((!CheckQScaleShape4MLAFullQuant(contextKeyParams) || !CheckKVScaleShape4MLAFullQuant(contextKeyParams)),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "When MLAFullQuant enables, check dequantScaleQuery, keyAntiQuantScale or valueAntiquantScale shape failed."), return false);
    // 全量化暂不支持 keyAntiquantOffset, valueAntiquantOffset, quantScale1, dequantScale1, dequantScale2
    OP_CHECK_IF((contextKeyParams.KeyAntiquantOffsetShape != nullptr) || (contextKeyParams.valueAntiquantOffsetShape != nullptr),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "When MLAFullQuant enables, keyAntiquantOffset and valueAntiquantOffset should be null."), return false);
    OP_CHECK_IF((contextKeyParams.deqScale1Shape != nullptr || contextKeyParams.scale1Shape != nullptr ||
        contextKeyParams.deqScale2Shape != nullptr), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "When MLAFullQuant enables, quantScale1, dequantScale1 and dequantScale2 should be null."), return false);
    // 全量化不支持system prefix
    OP_CHECK_IF((contextKeyParams.keySharedPrefix != nullptr || contextKeyParams.valueSharedPrefix != nullptr),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "When MLAFullQuant enables, keySharedPrefix and valueSharedPrefix should be null."), return false);
    return true;
}

bool PromptFlashAttentionTilingV2::CheckPrefix(ContextParamsForPFATiling& contextKeyParams,
    const PFAShapeInfo& queryShapeInfo, PFAShapeInfo& keyShapeInfo,
    PromptFlashAttentionTilingDataV2& tilingData) 
{
    PFAShapeInfo prefixShapeInfo;
    tilingData.promptAttentionBaseParams.set_prefixSeqInnerSize(0);
    if (!isKVHasPrefix) {
        tilingData.promptAttentionBaseParams.set_isActualSharedPrefixLenNull(1);
        actualSharedPrefixLen = 0;
        return true;
    }
    std::string layoutStr(contextKeyParams.layout);
    OP_CHECK_IF(
        (layoutStr == "BSND_BNSD" || layoutStr == "BSH_BNSD"),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "when %s is used, system prefix is not supported!",
        layoutStr.c_str()), return false);
    // The prefix does not support TND, tensorlist, pfa mla, ifa mla, left padding, PA and alibi
    OP_CHECK_IF(
        (inputLayout == InputLayout::TND || inputLayout == InputLayout::NTD),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "when TND/NTD is used, system prefix is not supported!"),
        return false);
    OP_CHECK_IF(enableTensorList && (queryShapeInfo.s > 1), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "when tensorlist is used and q_s is greater than 1, system prefix is not supported!"),
        return false);
    OP_CHECK_IF(enableIFAMLA || enablePFARope, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "when system prefix is used, rope is not supported!"),
        return false);
    OP_CHECK_IF(enablePFAMLA, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "when system prefix is used, query, key and value D should be the same, input query's D ane key's D = 192, but value's D = 128!"),
        return false);
    OP_CHECK_IF(enableLeftPadding, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "when system prefix is used, leftpadding is not supported!"),
        return false);
    OP_CHECK_IF((contextKeyParams.inputDataType == ge::DT_INT8) && (contextKeyParams.kDataType == ge::DT_INT8),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "when system prefix is used, query and key/value should not both be int8!"),
        return false);
    OP_CHECK_IF(enablePA, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "when system prefix is used, PA is not supported!"),
        return false);
    OP_CHECK_IF(enableAlibiPse, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "When pseType = 2/3, system prefix is not supported!"),
        return false);
    OP_CHECK_IF(enablePostQuant && (outputType == ge::DT_FLOAT8_E4M3FN || outputType == ge::DT_HIFLOAT8),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "when system prefix is used, invalid output type [%s], only support int8.",
            GetPfaDataTypeStr(outputType).c_str()),
        return false);
    // get prefix shape
    const gert::StorageShape* keyShape = contextKeyParams.keyInputShape;

    OP_CHECK_IF(!GetAndCheckPrefixShape(contextKeyParams, keyShapeInfo, prefixShapeInfo),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Get and check prefix shape failed."),
        return false);

    // check prefix kv consistency
    OP_CHECK_IF(!CheckKeyValuePrefixConsistency(contextKeyParams, keyShape),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "key value prefix consistency check failed."),
        return false);

    // check actSharedPrefix
    if (!isMaxWorkspace && (contextKeyParams.actualSharedPrefixLen != nullptr) &&
        (contextKeyParams.actualSharedPrefixLen->GetStorageShape().GetShapeSize() > 0)) {
        if (!CheckActSharedPrefix(contextKeyParams, prefixShapeInfo.s, keyShapeInfo.s)) {
            return false;
        }
        tilingData.promptAttentionBaseParams.set_isActualSharedPrefixLenNull(0);
    } else {
        tilingData.promptAttentionBaseParams.set_isActualSharedPrefixLenNull(1);
        actualSharedPrefixLen = prefixShapeInfo.s;
    }
    tilingData.promptAttentionBaseParams.set_prefixSeqInnerSize(prefixShapeInfo.s);
    return true;
}

bool PromptFlashAttentionTilingV2::CheckActSeq(const ContextParamsForPFATiling& contextKeyParams, const PFAShapeInfo& queryShapeInfo) const 
{
    if (inputLayout != InputLayout::TND && inputLayout != InputLayout::NTD) {
        return true;
    }

    const gert::Tensor* actSeqLen = contextKeyParams.actualSequenceLengthQ;
    const gert::Tensor* actSeqLenKV = contextKeyParams.actualSequenceLengthKV;
    OP_CHECK_IF(actSeqLen == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is TND/NTD, actualSequenceLengthQ can not be nullptr"),
        return false);
    OP_CHECK_IF(actSeqLenKV == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is TND/NTD, actualSequenceLengthKV can not be nullptr"),
        return false);

    auto batchOfQuery = actSeqLen->GetShapeSize();
    auto batchOfKey = actSeqLenKV->GetShapeSize();
    OP_CHECK_IF(!enablePA && (batchOfQuery != batchOfKey),
                OP_LOGE(contextKeyParams.opName,
                        "When layout is TND/NTD and page attention is not enable, "
                        "the batch size of actualSequenceLengthQ and actualSequenceLengthKV must be equal, "
                        "batch size of actualSequenceLengthQ = %ld, batch size of actualSequenceLengthKV = %ld",
                        batchOfQuery, batchOfKey),
                return false);
    // layout为TND, kv PA管理场景 actualSequenceLengthKV size可以为1或者>=B
    OP_CHECK_IF(enablePA && (batchOfKey != 1 && batchOfKey < batchOfQuery),
                OP_LOGE(contextKeyParams.opName,
                        "When layout is TND/NTD and page attention is enable, "
                        "the size of actualSequenceLengthKV (%ld) should be greater than or equal to "
                        "the size of actualSequenceLengthQ(%ld) or equal to 1 ",
                        batchOfKey, batchOfQuery),
                return false);

    int64_t lastActSeq = 0;
    int64_t lastActSeqKV = 0;
    int64_t curActSeqKV = 0;
    uint32_t batchSize = queryShapeInfo.b; // actSeqLengthSize and actSeqLengthKVSize are equal
    for (uint32_t i = LOOP_BEGIN_NUM; i < batchSize; ++i) {
        int64_t curActSeq = actSeqLen->GetData<int64_t>()[i];
        if (enablePA && batchOfKey == 1) {
            curActSeqKV = actSeqLenKV->GetData<int64_t>()[0];
        } else {
            curActSeqKV = actSeqLenKV->GetData<int64_t>()[i];
        }

        OP_CHECK_IF(curActSeq < 0,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "actualSeqLengths[%u] = %ld, should >= 0",
                i, curActSeq),
            return false);
        OP_CHECK_IF(((i >= 1) && (curActSeq < lastActSeq)),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "When layout is TND, actualSeqLengths must not be decreasing, but actSeqLen[%u]=%ld is smaller than actSeqLen[%u]=%ld",
                i, curActSeq, i - 1, lastActSeq),
            return false);
        OP_CHECK_IF(curActSeqKV < 0,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "actualSeqLengthsKv[%u] = %ld, should >= 0",
                i, curActSeqKV),
            return false);
        if (!enablePA) {
            OP_CHECK_IF(((i >= 1) && (curActSeqKV < lastActSeqKV)),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "When layout is TND, actualSeqLengthsKv must not be decreasing, but actSeqLenKV[%u]=%ld is smaller than actSeqLenKV[%u]=%ld",
                    i, curActSeqKV, i - 1, lastActSeqKV),
                return false);
        }    
        lastActSeq = curActSeq;
        lastActSeqKV = curActSeqKV;
    }

    const gert::StorageShape* queryShape = contextKeyParams.queryInputShape;
    const gert::StorageShape* keyShape = contextKeyParams.keyInputShape;
    if (inputLayout == InputLayout::TND) {
        OP_CHECK_IF(actSeqLen->GetData<int64_t>()[batchSize - 1] != queryShape->GetStorageShape().GetDim(0),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "When layout is TND, the last element of Actual_seq_lengths(%ld) must be equal to T(%ld)",
                actSeqLen->GetData<int64_t>()[batchSize - 1], queryShape->GetStorageShape().GetDim(0)),
            return false);
    } else {
        OP_CHECK_IF(actSeqLen->GetData<int64_t>()[batchSize - 1] != queryShape->GetStorageShape().GetDim(1),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "When layout is NTD, the last element of Actual_seq_lengths(%ld) must be equal to T(%ld)",
                actSeqLen->GetData<int64_t>()[batchSize - 1], queryShape->GetStorageShape().GetDim(1)),
            return false);       
    }

    if (!enablePA) {
        if (inputLayout == InputLayout::TND) {
            OP_CHECK_IF(actSeqLenKV->GetData<int64_t>()[batchSize - 1] != keyShape->GetStorageShape().GetDim(0),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "When layout is TND, the last element of Actual_seq_lengths_kv(%ld) must be equal to T(%ld)",
                    actSeqLenKV->GetData<int64_t>()[batchSize - 1], keyShape->GetStorageShape().GetDim(0)),
                return false);
        } else {
            OP_CHECK_IF(actSeqLenKV->GetData<int64_t>()[batchSize - 1] != keyShape->GetStorageShape().GetDim(1),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "When layout is NTD, the last element of Actual_seq_lengths_kv(%ld) must be equal to T(%ld)",
                    actSeqLenKV->GetData<int64_t>()[batchSize - 1], keyShape->GetStorageShape().GetDim(1)),
                return false);
        }
    }

    return true;
}

bool PromptFlashAttentionTilingV2::CheckActSeqLen(ContextParamsForPFATiling& contextKeyParams,
    PFAShapeInfo& queryShapeInfo, PFAShapeInfo& keyShapeInfo) 
{
    const gert::Tensor* actSeqLen = contextKeyParams.actualSequenceLengthQ;
    const gert::Tensor* actSeqLenKV = contextKeyParams.actualSequenceLengthKV;

    constexpr int64_t actSeqLenDimsQMin = 1;  // The min value of the length of actualSeqQ is 1
    constexpr int64_t actSeqLenDimsKVMin = 1; // The min value of the length of actualSeqKV is 1

    std::string layoutStr(contextKeyParams.layout);
    if (enableActSeqLen) {   // check the length of actual_seq_lengthsQ, whether is 1 or batch size
        OP_CHECK_IF(enableIFAMLAFullQuant && (inputLayout != InputLayout::TND),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "The layout is %s, Actual_seq_lengths cannot be configured in MLA Fullquant and non-TND/TND_NTD scenarios, only supported when layout is TND/TND_NTD!", layoutStr.c_str()),
            return false);
        OP_CHECK_IF((actSeqLenDims < queryShapeInfo.b) && (actSeqLenDims > actSeqLenDimsQMin),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "Dim(%ld) of actual_seq_lengths must equal to 1 or greater than or equal to batch size(%u)!",
                actSeqLenDims, queryShapeInfo.b),
            return false);
        uint32_t actSeqLengthSize = std::min(static_cast<uint32_t>(actSeqLenDims), queryShapeInfo.b);
        for (uint32_t i = LOOP_BEGIN_NUM; i < actSeqLengthSize; ++i) {
            int64_t actSeqTmp = actSeqLen->GetData<int64_t>()[i];
            if ((inputLayout == InputLayout::TND || inputLayout == InputLayout::NTD) && i >= 1) {
                actSeqTmp -= actSeqLen->GetData<int64_t>()[i - 1];
            }
            OP_CHECK_IF(actSeqTmp < 0 || actSeqTmp > queryShapeInfo.s, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "Actual_seq_lengths[%u](%ld) must be in range[0, %u]!", i, actSeqTmp, queryShapeInfo.s),
                return false);
            // query act seq len padding情况下不支持合轴
            if (actSeqTmp < queryShapeInfo.s) {
                enablePFAMerge = false;
            }
        }
        t1Size = actSeqLen->GetData<int64_t>()[actSeqLengthSize - 1];
    }

    if (enableActSeqLenKV) { // check the length of actual_seq_lengthsKV,whether is 1 or batch size
        // Batch of key is 1 under tensorlist mode, the number of actual_seq_lengthsKV should be compared
        // with batch of query, batch of key is equal to query under non-tensorlist mode.
        OP_CHECK_IF((actSeqLenKVDims < queryShapeInfo.b) && (actSeqLenKVDims > actSeqLenDimsKVMin),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "Dim(%ld) of actSeqLen kv must equal to 1 or greater than or equal to batch size(%u)!",
                actSeqLenKVDims, queryShapeInfo.b),
            return false);
        uint32_t actSeqLengthKVSize = std::min(static_cast<uint32_t>(actSeqLenKVDims), queryShapeInfo.b);
        for (uint32_t i = LOOP_BEGIN_NUM; i < actSeqLengthKVSize; ++i) {
            int64_t actSeqKVTmp = static_cast<int64_t>(actSeqLenKV->GetData<int64_t>()[i]);
            if ((inputLayout == InputLayout::TND || inputLayout == InputLayout::NTD) && !enablePA && i >= 1) {
                actSeqKVTmp -= static_cast<int64_t>(actSeqLenKV->GetData<int64_t>()[i - 1]);
            }
            if (!enableTensorList && !enablePA) {
                int64_t actSeqKVMax = static_cast<int64_t>(keyShapeInfo.s);
                OP_CHECK_IF((actSeqKVTmp < 0) || (actSeqKVTmp > actSeqKVMax),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                        "Actual_seq_lengths_kv[%u](%ld) must be in range[0, %ld]!", i, actSeqKVTmp, actSeqKVMax),
                    return false);
            } else if (enableTensorList) {
                int64_t actSeqKVMax;
                if ((inputLayout == InputLayout::BSND) || (inputLayout == InputLayout::BSH)) {
                    actSeqKVMax = contextKeyParams.kTensorList[i]->GetStorageShape().GetDim(1);
                } else {
                    actSeqKVMax = contextKeyParams.kTensorList[i]->GetStorageShape().GetDim(2); // 2: Dim S for BNSD
                }
                OP_CHECK_IF((actSeqKVTmp < 0) || (actSeqKVTmp > actSeqKVMax),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                        "Actual_seq_lengths_kv[%u](%ld) must be in range[0, %ld]!", i, actSeqKVTmp, actSeqKVMax),
                    return false);
            }
        }
        t2Size = actSeqLenKV->GetData<int64_t>()[actSeqLengthKVSize - 1];
    }
    return true;
}

bool PromptFlashAttentionTilingV2::CheckPATypeAndShape(ContextParamsForPFATiling& contextKeyParams,
    PFAShapeInfo& queryShapeInfo, PFAShapeInfo& queryRopeShapeInfo, PromptFlashAttentionTilingDataV2& tilingData) 
{
    OP_CHECK_IF(isQKVDDifferent,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Not support PA when query and key headdim is not equal to value headdim."),
        return false);
    // The interception that is mutually exclusive with the left padding has been implemented in FIA.
    OP_CHECK_IF(enableTensorList,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "not support tensorlist when blockTable is not null"),
        return false);
    OP_CHECK_IF((!enableActSeqLenKV) && !isMaxWorkspace,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "actual seq length kv can't be null when blockTable is not null"),
        return false);

    const gert::StorageShape* blockTableShape = contextKeyParams.blockTableShape;
    // check blockTable shape is nullptr
    OP_CHECK_IF(blockTableShape == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "blockTable can't be nullptr when PA enable"),
        return false);
    
    // check blockTable each dim cannot be 0
    OP_CHECK_IF(blockTableShape->GetStorageShape().GetShapeSize() == 0,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "blockTable each dim can't be 0, blockTable Shape [%ld, %ld].",
            blockTableShape->GetStorageShape().GetDim(0), blockTableShape->GetStorageShape().GetDim(1)),
        return false);

    // check Dtype
    ge::DataType blockTableType = contextKeyParams.blockTableType;
    OP_CHECK_IF((blockTableType != ge::DT_INT32),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "blockTable(%s) only support int32 when PA enable", 
        GetPfaDataTypeStr(blockTableType).c_str()),
        return false);

    // check block table shape
    const int32_t* blockSize = contextKeyParams.blockSize;
    // Tiling sinking scene, workspace needs to be calculated, at this time, blockTableDim2 * blockSize is used as S2.
    blockTableDim2 = static_cast<int32_t>(blockTableShape->GetStorageShape().GetDim(1));

    // PFA PA blockSize % 128 == 0
    if (enablePertensorQuant) {
        OP_CHECK_IF((!enableIFAMLA && !enableIFA && !(queryShapeInfo.s == 1 && enableAlibiPse) && (*blockSize % BLOCK_SIZE_BASE != 0 || *blockSize < BLOCK_SIZE_BASE || *blockSize > BLOCK_SIZE_MAX)),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "block size(%d) should be a multiple of %d, and should be in range of [%d, %d] when Full Quant and PA enable",
                *blockSize, BLOCK_SIZE_BASE, BLOCK_SIZE_BASE, BLOCK_SIZE_MAX),
            return false);
    } else if (!enableIFAMLAFullQuant) {            
        OP_CHECK_IF((!enableIFAMLA && !enableIFA && !(queryShapeInfo.s == 1 && enableAlibiPse) && (*blockSize % BLOCK_SIZE_BASE_FOR_NO_QUANT != 0 || *blockSize < BLOCK_SIZE_BASE_FOR_NO_QUANT || *blockSize > BLOCK_SIZE_MAX_FOR_NO_QUANT)),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "block size(%d) should be a multiple of %d, and should be in range of [%d, %d] when PA enable and no quant",
                *blockSize, BLOCK_SIZE_BASE_FOR_NO_QUANT, BLOCK_SIZE_BASE_FOR_NO_QUANT, BLOCK_SIZE_MAX_FOR_NO_QUANT),
            return false);
    }
    // IFA PA blockSize % 16 == 0
    ifaBlockSizeBase /= static_cast<int32_t>(dataTypeSize);
    if (enablePertensorQuant) {
        OP_CHECK_IF(((enableIFAMLA || enableIFA || (queryShapeInfo.s == 1 && enableAlibiPse)) && (*blockSize % ifaBlockSizeBase != 0 || *blockSize < ifaBlockSizeBase || *blockSize > BLOCK_SIZE_MAX)),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "block size(%d) should be a multiple of %d, and should be in range of [%d, %d] when Full Quant and PA enable",
                *blockSize, ifaBlockSizeBase, ifaBlockSizeBase, BLOCK_SIZE_MAX),
            return false);
    } else if (!enableIFAMLAFullQuant) {
        OP_CHECK_IF(((enableIFAMLA || enableIFA || (queryShapeInfo.s == 1 && enableAlibiPse)) && (*blockSize % BLOCK_SIZE_BASE_FOR_NO_QUANT != 0 || *blockSize < BLOCK_SIZE_BASE_FOR_NO_QUANT || *blockSize > BLOCK_SIZE_MAX_FOR_NO_QUANT)),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "block size(%d) should be a multiple of %d, and should be in range of [%d, %d] when PA enable and no quant",
                *blockSize, BLOCK_SIZE_BASE_FOR_NO_QUANT, BLOCK_SIZE_BASE_FOR_NO_QUANT, BLOCK_SIZE_MAX_FOR_NO_QUANT),
            return false);
    }
    // MLA 全量化 blocksize == 128
    OP_CHECK_IF((enableIFAMLAFullQuant && *blockSize != BLOCK_SIZE_BASE),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "block size(%d) should be %d, when MLA Full Quant and PA enable", *blockSize, BLOCK_SIZE_BASE),
        return false);
    //非量化场景
    if(!enableIFAMLAFullQuant && !enablePertensorQuant){
        if (enableIFAMLA || enablePFARope || enablePFAMLA){
            OP_CHECK_IF((*blockSize % BLOCK_SIZE_BASE_FOR_NO_QUANT != 0 || *blockSize < BLOCK_SIZE_BASE_FOR_NO_QUANT || *blockSize > BLOCK_SIZE_MAX_FOR_NO_QUANT),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "block size(%d) should be a multiple of %d, and should be in range of [%d, %d] when PA enable and no quant mla",
                *blockSize, BLOCK_SIZE_BASE_FOR_NO_QUANT, BLOCK_SIZE_BASE_FOR_NO_QUANT, BLOCK_SIZE_MAX_FOR_NO_QUANT),
            return false);
        }else if (queryShapeInfo.d == 64 || queryShapeInfo.d == 128){ // 64, 128: Typical hidden dimensions per head in transformer models
            OP_CHECK_IF((*blockSize % BLOCK_SIZE_BASE_FOR_NO_QUANT != 0 || *blockSize < BLOCK_SIZE_BASE_FOR_NO_QUANT || *blockSize > BLOCK_SIZE_MAX_FOR_NO_QUANT),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "block size(%d) should be a multiple of %d, and should be in range of [%d, %d] when PA enable and no quant with d[%d] is 64/128",
                *blockSize, BLOCK_SIZE_BASE_FOR_NO_QUANT, BLOCK_SIZE_BASE_FOR_NO_QUANT, BLOCK_SIZE_MAX_FOR_NO_QUANT, queryShapeInfo.d),
            return false);
        }else{
            if (queryShapeInfo.s > 1){
                OP_CHECK_IF((*blockSize % BLOCK_SIZE_BASE != 0 || *blockSize < BLOCK_SIZE_BASE || *blockSize > BLOCK_SIZE_MAX),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "block size(%d) should be a multiple of %d, and should be in range of [%d, %d] when PA enable and no quant with qs[%d] is greater than 1",
                    *blockSize, BLOCK_SIZE_BASE, BLOCK_SIZE_BASE, BLOCK_SIZE_MAX, queryShapeInfo.s),
                return false);
            }else{
                OP_CHECK_IF((*blockSize % BLOCK_SIZE_BASE_FOR_NO_QUANT != 0 || *blockSize < BLOCK_SIZE_BASE_FOR_NO_QUANT || *blockSize > BLOCK_SIZE_MAX),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "block size(%d) should be a multiple of %d, and should be in range of [%d, %d] when PA enable and no quant with qs[%d] is 1",
                    *blockSize, BLOCK_SIZE_BASE_FOR_NO_QUANT, BLOCK_SIZE_BASE_FOR_NO_QUANT, BLOCK_SIZE_MAX, queryShapeInfo.s),
                return false);
            }
        }
    }

    if (isMaxWorkspace) {
        S2 = blockTableDim2 * (*blockSize);
        return true;
    }
    if (!CheckBlockTableShape(contextKeyParams, queryShapeInfo, queryRopeShapeInfo, blockSize, blockTableShape, tilingData)) {
        return false;
    }

    tilingData.promptAttentionBaseParams.set_PAlayoutType(paLayoutType);
    return true;
}

bool PromptFlashAttentionTilingV2::CheckPseShiftTypeAndShape(ContextParamsForPFATiling& contextKeyParams,
    uint32_t b, uint32_t n, uint32_t s1, uint32_t s2) 
{
    const gert::StorageShape* pseShiftShape = contextKeyParams.pseShiftShape;
    OP_CHECK_IF(isQKVDDifferent,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Not support pse shift when query and key headdim is not equal to value headdim."),
        return false);   
    OP_CHECK_IF(enableIFAMLA || enablePFAMLA || enablePFARope,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "MLA do not support pseShift."),
        return false);
    if (!CheckNonEmptyShapeExceptions(contextKeyParams, pseShiftShape, "pseShift")) {
        return false;
    }
    std::string layoutStr(contextKeyParams.layout);
    pseShiftElemType = contextKeyParams.pseShiftDataType;

    OP_CHECK_IF((inputType == ge::DT_FLOAT16 && pseShiftElemType != ge::DT_FLOAT16),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "q type is fp16, but pse shift type is not fp16, pse shift type = %s", GetPfaDataTypeStr(pseShiftElemType).c_str()),
        return false);

    OP_CHECK_IF((inputType == ge::DT_BF16 && pseShiftElemType != ge::DT_BF16),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "q type is bf16, but pse shift type is not bf16, pse shift type = %s", GetPfaDataTypeStr(pseShiftElemType).c_str()),
        return false);

    OP_CHECK_IF((inputType == ge::DT_INT8 && pseShiftElemType != ge::DT_FLOAT16),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "q type is int8, but pse shift type is not fp16, pse shift type = %s", GetPfaDataTypeStr(pseShiftElemType).c_str()),
        return false);

    if (pseShiftElemType == ge::DT_FLOAT16) {
        pseShiftElemSize = FLOAT16SIZE;
    } else if (pseShiftElemType == ge::DT_BF16) {
        pseShiftElemSize = BFLOAT16SIZE;
    }
    pseShiftTypeByteNum = BYTE_BLOCK / pseShiftElemSize;

    size_t pseShiftDim = pseShiftShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF((pseShiftDim != PSESHIFTDIM_4),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "pse shift shape must be 4 dimension, rather than %zu dimension", pseShiftDim),
        return false);

    pseShiftBatch = pseShiftShape->GetStorageShape().GetDim(0);
    int64_t pseShiftN = pseShiftShape->GetStorageShape().GetDim(1); // 1: The sirst dimension is N.
    pseShiftS1 = pseShiftShape->GetStorageShape().GetDim(2); // 2: The second dimension is S1.
    pseShiftS2 = pseShiftShape->GetStorageShape().GetDim(3); // 3: The third dimension is S2.
    OP_CHECK_IF((!isMaxWorkspace && ((pseShiftBatch != 1 && pseShiftBatch != b) || (pseShiftN != n) || (pseShiftS1 < s1) || (pseShiftS2 < s2))),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "pse shift shape must be [1 or %u, %u, >=%u, >=%u], but now it is [%ld, %ld, %ld, %ld], the layout is %s",
            b, n ,s1, s2, pseShiftBatch, pseShiftN, pseShiftS1, pseShiftS2, layoutStr.c_str()),
        return false);
    return true;
}

bool PromptFlashAttentionTilingV2::CheckInnerPrecise(ContextParamsForPFATiling& contextKeyParams,
    PromptFlashAttentionTilingDataV2& tilingData) 
{
    // 0: Invalid plural number; 4: Invalid if greater than or equal to 4; 0,1,2,3 are effective values for innerPrecise.
    if (innerPrecise >= 4 || innerPrecise < 0) {
        OP_LOGE(contextKeyParams.opName, "innerPrecise [%d] should be 0,1,2,3, please check.", innerPrecise);
        return false;
    }
    // Determine if the bit1 bit of innerPrecise requires invalid correction.
    tilingData.promptAttentionBaseParams.set_isRowInvalid((innerPrecise >> 1) & 1);
    // Determine the bit0 bit of innerPrecise, high-performance or high-precision mode.
    innerPrecise = HIGH_PRECISION; // High-precision contains high-performance and high-precision mode.

    // FP16 pse is forced to enter high-precision mode.
    if ((contextKeyParams.pseShift != nullptr) && (inputType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PERFORMANCE)) {
        innerPrecise = HIGH_PRECISION;
        OP_LOGW(contextKeyParams.opName, "when the input is fp16 and enable pseshift, the mode is forcibly switched to high-precision!");
    }
    if (inputType != ge::DT_FLOAT16) {
        OP_LOGW(contextKeyParams.opName, "innerPrecise will not take effect when input type is %s!",
            GetPfaDataTypeStr(inputType).c_str());
    }
    if ((inputType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PERFORMANCE)) {
        softmaxDataTypeSize = FLOAT16SIZE; // The default size is fp32.
    }
    return true;
}

bool PromptFlashAttentionTilingV2::CheckMaskTypeAndShape(ContextParamsForPFATiling& contextKeyParams,
    PromptFlashAttentionTilingDataV2& tilingData) 
{
    const gert::StorageShape* attenMaskShape = contextKeyParams.attentionMaskShape;
    OP_CHECK_IF((attenMaskShape->GetStorageShape().GetShapeSize() == gert::Shape::kInvalidDimValue),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Get the shape size of attenMask failed."),
        return false);

    auto maskDataType = contextKeyParams.maskDataType;
    std::vector<ge::DataType> allowedMaskDtypes = {ge::DT_FLOAT16, ge::DT_BOOL, ge::DT_INT8, ge::DT_UINT8};
    std::vector<uint32_t> maskTypeSizeArray = {FLOAT16SIZE, BOOLSIZE, INT8SIZE, UINT8SIZE};

    auto maskTypeCheck = std::find(allowedMaskDtypes.begin(), allowedMaskDtypes.end(), maskDataType);

    OP_CHECK_IF((maskTypeCheck == allowedMaskDtypes.end()), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "maskType(%s) should be DT_FLOAT16, DT_BOOL, DT_INT8 and DT_UINT8", GetPfaDataTypeStr(maskDataType).c_str()),
        return false);
    int32_t maskTypeIndex = std::distance(allowedMaskDtypes.begin(), maskTypeCheck);
    maskElemSize = maskTypeSizeArray[maskTypeIndex];

    // When in fp16 high-precision mode, the mask type only supports bool or int8.
    OP_CHECK_IF(((inputType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PRECISION)) &&
        (maskDataType != ge::DT_BOOL) && (maskDataType != ge::DT_INT8) && (maskDataType != ge::DT_UINT8),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "maskType[%s] should be bool, int8 or uint8 when innerprecise = %d and input type is fp16.",
            GetPfaDataTypeStr(maskDataType).c_str(), innerPrecise),
        return false);
    // When bf16, the mask type only supports bool or int8.
    OP_CHECK_IF((inputType == ge::DT_BF16) && (maskDataType != ge::DT_BOOL) &&
        (maskDataType != ge::DT_INT8) && (maskDataType != ge::DT_UINT8),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "maskType[%s] should be bool, int8 or uint8 when input type is bfloat16", GetPfaDataTypeStr(maskDataType).c_str()),
        return false);

    int64_t maskKVsSize = 2048; // 2048 : default the last frist dim.
    int64_t maskQsSize = 2048; // 2048 : default the last second dim.
    // 1: last frist dim, 2: last second dim
    maskKVsSize = attenMaskShape->GetStorageShape().GetDim(attenMaskShape->GetStorageShape().GetDimNum() - 1);
    maskQsSize = attenMaskShape->GetStorageShape().GetDim(attenMaskShape->GetStorageShape().GetDimNum() - 2); // 2 for Q dim index

    tilingData.promptAttentionBaseParams.set_maskKVsSize(maskKVsSize);
    tilingData.promptAttentionBaseParams.set_maskQsSize(maskQsSize);
    return true;
}

void PromptFlashAttentionTilingV2::SetSparseType(uint32_t qS)
{
    if (!faRunFlag_) {
        return;
    }
    if (sparseModeVal == SPARSE_MODE_NO_MASK) {
        if (sparsePreTokens >= qS && sparseNextTokens == 0) {
            sparseType = static_cast<uint8_t>(PfaSparseEnum::PFA_CAUSAL);
        } else if (sparsePreTokens >= qS && !enablePA && sparseNextTokens >= S2) {
            sparseType = static_cast<uint8_t>(PfaSparseEnum::PFA_ALL);
        } else {
            sparseType = static_cast<uint8_t>(PfaSparseEnum::PFA_BAND);
        }
    } else if (sparseModeVal == SPARSE_MODE_ALL_MASK) {
        sparseType = static_cast<uint8_t>(PfaSparseEnum::PFA_ALL);
    } else if (sparseModeVal == SPARSE_MODE_LEFT_UP) {
        sparseType = static_cast<uint8_t>(PfaSparseEnum::PFA_CAUSAL);
    } else if (sparseModeVal == SPARSE_MODE_RIGHT_DOWN) {
        if (!enablePA && qS == S2) {
            sparseType = static_cast<uint8_t>(PfaSparseEnum::PFA_CAUSAL);
        } else {
            sparseType = static_cast<uint8_t>(PfaSparseEnum::PFA_BAND);
        }
    } else if (sparseModeVal == SPARSE_MODE_BAND) {
        sparseType = static_cast<uint8_t>(PfaSparseEnum::PFA_BAND);
    }
}

bool PromptFlashAttentionTilingV2::CheckSparseMode(ContextParamsForPFATiling& contextKeyParams, uint32_t qS) 
{
    const int32_t* sparseMode = contextKeyParams.sparseMode;
    const int64_t* nextTokens = contextKeyParams.nextToken;
    const int64_t* preTokens = contextKeyParams.preToken;

    bool sparseCheck = false;
    if (sparseMode != nullptr) {
        sparseCheck = ((*sparseMode != SPARSE_MODE_NO_MASK) && (*sparseMode != SPARSE_MODE_LEFT_UP) &&
            (*sparseMode != SPARSE_MODE_RIGHT_DOWN) && (*sparseMode != SPARSE_MODE_ALL_MASK) &&
            (*sparseMode != SPARSE_MODE_BAND));
        OP_CHECK_IF(sparseCheck, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "sparse_mode = %d is out of range. Currently only 0,1,2,3,4 are supported.", *sparseMode),
            return false);
    }
    const gert::StorageShape* attenMaskShape = contextKeyParams.attentionMaskShape;
    SetSparseModeData(contextKeyParams, attenMaskShape, sparseMode, preTokens, nextTokens);

    OP_CHECK_IF((sparseNextTokens * (-1)) > sparsePreTokens, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "nexttoken line should be higher than pretoken line, preTokens = %ld, nextTokens = %ld.",
        sparsePreTokens, sparseNextTokens),
        return false);

    SetSparseType(qS);
    OP_CHECK_IF((isQKVDDifferent && sparseModeVal != 0 && sparseModeVal != 2 && sparseModeVal != 3) ,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "Not support sparse mode %d when query and key headdim is not equal to value headdim.",
        sparseModeVal),
        return false);
    return true;
}

bool PromptFlashAttentionTilingV2::CheckPACrossover(ContextParamsForPFATiling& contextKeyParams,
    PFAShapeInfo& queryShapeInfo) 
{
    if (enablePA) {
        OP_CHECK_IF((isKVHasPrefix),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "when system prefix is used, page attention is not supported!"),
            return false);
        OP_CHECK_IF((enableKVAntiquant),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "not support antiquant when blockTable is not null"),
            return false);
    }
    if (enableActSeqLenKV && !enableTensorList) {
        const gert::Tensor* actSeqLenKV = contextKeyParams.actualSequenceLengthKV;
        uint32_t actSeqLenKVSize = std::min(static_cast<uint32_t>(actSeqLenKVDims), queryShapeInfo.b);
        for (uint32_t i = LOOP_BEGIN_NUM; i < actSeqLenKVSize; ++i) {
            OP_CHECK_IF(enablePA && (actSeqLenKV->GetData<int64_t>()[i] < 0), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "Actual_seq_lengths_kv[%u](%ld) must >= 0", i, actSeqLenKV->GetData<int64_t>()[i]),
                return false);
        }
    }
    return true;
}

bool PromptFlashAttentionTilingV2::CheckMaskCrossIFAMLA(ContextParamsForPFATiling& contextKeyParams,
    const int32_t *sparseMode, uint64_t queryS) 
{
    if (sparseMode == nullptr) {
        return true;
    }
    if (enableIFAMLAFullQuant) {
        if (queryS == 1U) {
            OP_CHECK_IF(!((*sparseMode == SPARSE_MODE_NO_MASK) && (!enableMask)),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "Only support sparse 0 without mask when ifa mla fullquant and query's sequence length is 1, "
                    "input sparse mode is %d and there has%smask",
                    *sparseMode, enableMask ? " " : " no "),
                return false);
        } else {
            if (contextKeyParams.inputDataType == ge::DT_FLOAT8_E4M3FN) {
                OP_CHECK_IF(!(((*sparseMode == SPARSE_MODE_RIGHT_DOWN) && (enableMask)) || ((*sparseMode == SPARSE_MODE_NO_MASK) && (!enableMask))),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                        "Only support sparse 3 with mask, or sparse 0 without mask when ifa mla fullquant fp8 and query's sequence length is > 1, "
                        "input sparse mode is %d and there has%smask",
                        *sparseMode, enableMask ? " " : " no "),
                    return false);
            } else if (contextKeyParams.inputDataType == ge::DT_INT8) {
                OP_CHECK_IF(!((*sparseMode == SPARSE_MODE_RIGHT_DOWN) && (enableMask)),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "Only support sparse 3 with mask, when ifa mla fullquant int8 and query's sequence length is > 1, "
                    "input sparse mode is %d and there has%smask",
                    *sparseMode, enableMask ? " " : " no "),
                return false);
            }
        }
    } else {
        OP_CHECK_IF(!(((*sparseMode == SPARSE_MODE_RIGHT_DOWN) && (enableMask)) || (*sparseMode == SPARSE_MODE_NO_MASK) || ((*sparseMode == SPARSE_MODE_BAND) && (enableMask))),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "Only support sparse 3 with mask, sparse 4 with mask, or sparse 0 when ifa mla, "
                "input sparse mode is %d and there has%smask",
                *sparseMode, enableMask ? " " : " no "),
            return false);
    }
    
    return true;
}

bool PromptFlashAttentionTilingV2::CheckMaskCrossover(ContextParamsForPFATiling& contextKeyParams,
    PFAShapeInfo& queryShapeInfo, PromptFlashAttentionTilingDataV2& tilingData) 
{
    auto maskDataType = contextKeyParams.maskDataType;
    const int32_t* sparseMode = contextKeyParams.sparseMode;
    const gert::StorageShape* attenMaskShape = contextKeyParams.attentionMaskShape;
    if ((sparseMode != nullptr) && (*sparseMode == SPARSE_MODE_LEFT_UP || *sparseMode == SPARSE_MODE_RIGHT_DOWN ||
        *sparseMode == SPARSE_MODE_ALL_MASK || *sparseMode == SPARSE_MODE_BAND)) {
        OP_CHECK_IF(((attenMaskShape != nullptr) && (attenMaskShape->GetStorageShape().GetShapeSize() == 0)) || 
            (attenMaskShape == nullptr), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, 
                "attenMask should not be null when sparse_mode is %d.", *sparseMode),
            return false);

        // When sparse=2, 3, 4, the mask type only supports bool, int8, uint8
        OP_CHECK_IF((*sparseMode != SPARSE_MODE_ALL_MASK) && (maskDataType != ge::DT_BOOL) &&
            (maskDataType != ge::DT_INT8) && (maskDataType != ge::DT_UINT8),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "maskType[%s] should be bool, int8 or uint8 when sparse mode is %d.",
                GetPfaDataTypeStr(maskDataType).c_str(), *sparseMode),
            return false);
    }
    // FP16 mask type does not support invalid line correction.
    OP_CHECK_IF((contextKeyParams.attentionMask != nullptr && maskDataType == ge::DT_FLOAT16 && (tilingData.promptAttentionBaseParams.get_isRowInvalid() != 0)),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "maskType[%s] should not be float16 when innerPrecise = 2 or 3",
            GetPfaDataTypeStr(maskDataType).c_str()),
        return false);
    if (!CheckMaskShapeCrossSparse(contextKeyParams, tilingData, sparseMode, queryShapeInfo.s, S2 + actualSharedPrefixLen,
        queryShapeInfo.b)) {
        return false;
    }
    if (enableIFAMLA && (!CheckMaskCrossIFAMLA(contextKeyParams, sparseMode, queryShapeInfo.s / gSize))) {
        return false;
    }
    return true;
}

bool PromptFlashAttentionTilingV2::CheckTNDLayoutCrossover(ContextParamsForPFATiling& contextKeyParams) 
{
    if (inputLayout != InputLayout::TND) {
        return true;
    }

    std::string layoutStr(contextKeyParams.layout);
    if (enableIFAMLA && layoutStr == "TND_NTD") { // Decode MLA
        OP_CHECK_IF(enablePostQuant,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "In Decode MLA scenario, when layout is TND_NTD, post quant is not supported!"),
            return false);
    }

    OP_CHECK_IF(enableLeftPadding,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is TND, left padding is not supported!"),
        return false);
    
    OP_CHECK_IF(enableTensorList,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is TND, tensorlist is not supported!"),
        return false);

    OP_CHECK_IF(enablePseShift,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is TND, pse is not supported!"),
        return false);

    return true;
}

bool PromptFlashAttentionTilingV2::CheckNTDLayoutCrossover(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo) 
{
    if (inputLayout != InputLayout::NTD) {
        return true;
    }

    std::string layoutStr(contextKeyParams.layout);
    if (enablePFAMLA || enablePFARope) { // Prefill MLA
 	    OP_CHECK_IF(enablePerblockQuant || enablePertensorQuant,
 	        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "In prefill MLA scenario, when layout is %s, full quant is not supported!", layoutStr.c_str()),
 	        return false);
 	}

    if (!enablePFAMLA && !enablePFARope && !enableIFAMLA && !(enablePerblockQuant && layoutStr == "NTD_TND")) { // GQA
        OP_CHECK_IF((CHECK_D_LIMITED_SCENARIO(queryShapeInfo.d)),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "In GQA scenario, when layout is %s, d size of query must be 64 or 128, but got d = %d.",
            layoutStr.c_str(), queryShapeInfo.d), return false);
    }
    if (!enablePFAMLA && !enablePFARope && !enableIFAMLA && !enablePertensorQuant && !enablePerblockQuant && !enableIFAMLAFullQuant) { // GQA
        OP_CHECK_IF(isQKVDDifferent,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "In GQA scenario, not support layout %s when query and key headdim is not equal to value headdim.",
            layoutStr.c_str()), return false);
    }

    OP_CHECK_IF(enableLeftPadding,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is %s, left padding is not supported!", layoutStr.c_str()),
        return false);
    
    OP_CHECK_IF(enableTensorList,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is %s, tensorlist is not supported!", layoutStr.c_str()),
        return false);

    OP_CHECK_IF(enablePseShift,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is %s, pse is not supported!", layoutStr.c_str()),
        return false);

    return true;
}

bool PromptFlashAttentionTilingV2::CheckTransposeLayoutCrossover(ContextParamsForPFATiling& contextKeyParams,
    PFAShapeInfo& queryShapeInfo) 
{
    std::string layoutStr(contextKeyParams.layout);
    if (layoutStr != "BSH_BNSD" && layoutStr != "BSND_BNSD" && layoutStr != "BNSD_BSND") {
        return true;
    }
    if (enablePFAMLA || enablePFARope) { // Prefill MLA
        OP_CHECK_IF(enablePerblockQuant || enablePertensorQuant,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "In prefill MLA scenario, when layout is %s, full quant is not supported!",
            layoutStr.c_str()), return false);
    }
    bool isGqa = !enablePFAMLA && !enablePFARope && !enableIFAMLA && !enablePertensorQuant && !enablePerblockQuant && !enableIFAMLAFullQuant;
    if (isGqa) { // GQA
        OP_CHECK_IF(isQKVDDifferent,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "In GQA scenario, not support layout %s when query and key headdim is not equal to value headdim.",
            layoutStr.c_str()), return false);
    }
    if (layoutStr == "BSH_BNSD" || layoutStr == "BSND_BNSD") {
        if (isGqa) { // GQA
            OP_CHECK_IF((CHECK_D_LIMITED_SCENARIO(queryShapeInfo.d)),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "In GQA scenario, when layout is %s, d size of query must be 64 or 128, but got d = %d.",
                layoutStr.c_str(), queryShapeInfo.d), return false);
        }
        OP_CHECK_IF(enableLeftPadding,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is %s, left padding is not supported!",
            layoutStr.c_str()), return false);
        
        OP_CHECK_IF(enableTensorList,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is %s, tensorlist is not supported!",
            layoutStr.c_str()), return false);

        OP_CHECK_IF(enablePseShift,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is %s, pse is not supported!",
            layoutStr.c_str()), return false);
    } else if (layoutStr == "BNSD_BSND") {
        if (isGqa) { // GQA
            OP_CHECK_IF((contextKeyParams.outputDataType == ge::DT_INT8 && queryShapeInfo.d % D_SIZE_BASE_32 != 0),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "In GQA scenario, when layout is %s and output dtype is int8, d size should be a multiple of %d, but got d = %d.",
                layoutStr.c_str(), D_SIZE_BASE_32, queryShapeInfo.d), return false);
            OP_CHECK_IF((queryShapeInfo.d % D_SIZE_BASE_16 != 0),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "In GQA scenario, when layout is %s, d size should be a multiple of %d, but got d = %d.",
                layoutStr.c_str(), D_SIZE_BASE_16, queryShapeInfo.d), return false);
        }
    }
    return true;
}

bool PromptFlashAttentionTilingV2::CheckLearnSink(ContextParamsForPFATiling &contextKeyParams, PFAShapeInfo &queryShapeInfo)
{
    if (!enableLearnSink) {
        return true;
    }

    OP_CHECK_IF(contextKeyParams.learnableSink->GetStorageShape().GetDim(0) != queryShapeInfo.n, 
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When learnable sink is used, shape of learnable sink(%ld) must be same with query's N(%ld).", 
            contextKeyParams.learnableSink->GetStorageShape().GetDim(0), queryShapeInfo.n),
        return false);
    OP_CHECK_IF(contextKeyParams.learnableSinkDataType != contextKeyParams.inputDataType, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, 
            "DataType of learnable sink(%s) is not equal to datatype of query(%s).", GetPfaDataTypeStr(contextKeyParams.learnableSinkDataType).c_str(), 
            GetPfaDataTypeStr(contextKeyParams.inputDataType).c_str()),
        return false);
    OP_CHECK_IF(contextKeyParams.learnableSinkDataType != ge::DT_BF16 && contextKeyParams.learnableSinkDataType != ge::DT_FLOAT16, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, 
            "When learnable sink is used, dataType of learnable sink(%s) must be bf16 or fp16.", GetPfaDataTypeStr(contextKeyParams.learnableSinkDataType).c_str()),
        return false);
    OP_CHECK_IF(queryShapeInfo.d != 192 && queryShapeInfo.d != 128 && queryShapeInfo.d != 64, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "When learnable sink is used, query headdim must be one of {192, 128, 64}."),
        return false);
    OP_CHECK_IF(enablePseShift, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, 
            "When learnable sink is used, pse is not supported!"),
        return false);
    OP_CHECK_IF(enableAlibiPse, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, 
            "When learnable sink is used, AlibiPse is not supported!"),
        return false);
    OP_CHECK_IF(enableLeftPadding, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, 
            "when learnable sink is used, leftpadding is not supported!"),
        return false);
    OP_CHECK_IF(isKVHasPrefix, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, 
            "when learnable sink is used, system prefix is not supported!"),
        return false);
    OP_CHECK_IF(enablePostQuant, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, 
            "when learnable sink is used, post quant is not supported!"),
        return false);
    OP_CHECK_IF(innerPrecise != HIGH_PRECISION, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "innerPrecise must be high-precision in learnable sink, now is %ld", innerPrecise),
        return false);
    OP_CHECK_IF(enableIFAMLAFullQuant || enableIFAMLA || enablePerblockQuant || enablePertensorQuant,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Learnable sink only supports no-quantized GQA mode"),
        return false);
    return true;
}

bool PromptFlashAttentionTilingV2::ParseActualSeqLengths(ContextParamsForPFATiling& contextKeyParams,
    PFAShapeInfo& queryShapeInfo, std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV) 
{
    uint32_t lenDims = queryShapeInfo.b; // The current length of the actSeqLen array is equal to batch size b.
    const gert::Tensor* actSeqLenData = contextKeyParams.actualSequenceLengthQ;
    const gert::Tensor* actSeqLenDataKV = contextKeyParams.actualSequenceLengthKV;
    actSeqLenDims = (actSeqLenData != nullptr) ? actSeqLenData->GetShapeSize() : 0;
    actSeqLenKVDims = (actSeqLenDataKV != nullptr) ? actSeqLenDataKV->GetShapeSize() : 0;

    if (inputLayout == InputLayout::TND || inputLayout == InputLayout::NTD) {
        if ((actSeqLenData == nullptr) || (actSeqLenDataKV == nullptr)) {
            return false;
        }
        middleActualSeqLengths = static_cast<uint32_t>(actSeqLenData->GetData<int64_t>()[lenDims-1]);
    }
    for (uint32_t i = LOOP_BEGIN_NUM; i < lenDims; i++) {
        if (inputLayout == InputLayout::TND || inputLayout == InputLayout::NTD) {
            actualSeqLengths[i] = (i == 0) ? static_cast<uint32_t>(actSeqLenData->GetData<int64_t>()[0]) :
                static_cast<uint32_t>(actSeqLenData->GetData<int64_t>()[i]) - static_cast<uint32_t>(actSeqLenData->GetData<int64_t>()[i - 1]);
            actualSeqLengthsKV[i] = static_cast<uint32_t>(actSeqLenDataKV->GetData<int64_t>()[i]);
            if (enableIFAMLA) {
                actualSeqLengths[i] *= gSize;
            }
            if (!enablePA && i >= 1) {
                actualSeqLengthsKV[i] -= static_cast<uint32_t>(actSeqLenDataKV->GetData<int64_t>()[i - 1]);
            }
            if (actualSeqLengthsKV[i] != S2) {
                needInit = 1;
            }
            continue;
        }

        if (!enableActSeqLen) {
            actualSeqLengths[i] = queryShapeInfo.s;
        } else if (enableIFAMLA || enableIFA) {
            actualSeqLengths[i] = (actSeqLenDims > 1) ? static_cast<uint32_t>(actSeqLenData->GetData<int64_t>()[i]) :
                static_cast<uint32_t>(actSeqLenData->GetData<int64_t>()[0]);
            if (actualSeqLengths[i] != queryShapeInfo.s / gSize) {
                needInit = 1;
            }
            if (inputLayout == InputLayout::BSND || inputLayout == InputLayout::BSH) {
                actualSeqLengths[i] *= gSize;
            } else {
                actualSeqLengths[i] = queryShapeInfo.s;
            }
        } else {
            actualSeqLengths[i] = (actSeqLenDims > 1) ? static_cast<uint32_t>(actSeqLenData->GetData<int64_t>()[i]) :
                static_cast<uint32_t>(actSeqLenData->GetData<int64_t>()[0]);
            if (actualSeqLengths[i] != queryShapeInfo.s) {
                needInit = 1;
            }
        }
        middleActualSeqLengths += actualSeqLengths[i];
        if (!enableActSeqLenKV) {       // The user did not input act_seq_kv
            if (!enableTensorList) {
                actualSeqLengthsKV[i] = S2;
            } else {
                if ((inputLayout == InputLayout::BSND) || (inputLayout == InputLayout::BSH)) {
                    actualSeqLengthsKV[i] = contextKeyParams.kTensorList[i]->GetStorageShape().GetDim(1);
                } else { // 2: Obtain the second dimension
                    actualSeqLengthsKV[i] = contextKeyParams.kTensorList[i]->GetStorageShape().GetDim(2);
                }
            }
        } else {
            actualSeqLengthsKV[i] = (actSeqLenKVDims > 1) ? static_cast<uint32_t>(actSeqLenDataKV->GetData<int64_t>()[i]) :
                static_cast<uint32_t>(actSeqLenDataKV->GetData<int64_t>()[0]);
            if (actualSeqLengthsKV[i] != S2) {
                needInit = 1;
            }
        }
        maxActualseqKV = std::max(maxActualseqKV, actualSeqLengthsKV[i]);
    }

    return true;
}

bool PromptFlashAttentionTilingV2::CheckMultiFeatureCrossover(ContextParamsForPFATiling& contextKeyParams,
    PFAShapeInfo& queryShapeInfo, std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV,
    PromptFlashAttentionTilingDataV2& tilingData) 
{
    if (!ParseActualSeqLengths(contextKeyParams, queryShapeInfo, actualSeqLengths, actualSeqLengthsKV)) {
        return false;
    }

    uint32_t lenDims = queryShapeInfo.b; // The current length of the actSeqLen array is equal to batch size b.
    int64_t preTokensPerbatch = 0;
    int64_t nextTokensPerbatch = 0;
    for (uint32_t i = LOOP_BEGIN_NUM; i < lenDims; i++) {
        if (sparseModeVal == SPARSE_MODE_RIGHT_DOWN) {
            preTokensPerbatch = SPARSE_MODE_INT_MAX;
            if (enableIFAMLA) {
                nextTokensPerbatch = actualSeqLengthsKV[i] + actualSharedPrefixLen - actualSeqLengths[i] / gSize;
            } else {
                nextTokensPerbatch = actualSeqLengthsKV[i] + actualSharedPrefixLen - actualSeqLengths[i];
            }
        } else if (sparseModeVal == SPARSE_MODE_BAND) {
            preTokensPerbatch = sparsePreTokens - actualSeqLengthsKV[i] - actualSharedPrefixLen + actualSeqLengths[i];
            nextTokensPerbatch = sparseNextTokens + actualSeqLengthsKV[i] + actualSharedPrefixLen - actualSeqLengths[i];
        } else {
            preTokensPerbatch = sparsePreTokens;
            nextTokensPerbatch = sparseNextTokens;
        }
        if ((nextTokensPerbatch < 0) ||
            (actualSeqLengths[i] > (actualSeqLengthsKV[i] + actualSharedPrefixLen + preTokensPerbatch))) {
            needInit = 1;
        }

        OP_LOGI(contextKeyParams.opName, "preTokensPerbatch[%u] is %ld, nextTokensPerbatch[%u] is %ld",
                i, preTokensPerbatch, i, nextTokensPerbatch);

        OP_LOGI(contextKeyParams.opName,
            "actualSeqLengths[%u] is %ld, actualSeqLengthsKV[%u] is %ld, actualSharedPrefixLen is %ld, needInit is %u",
            i, actualSeqLengths[i], i, actualSeqLengthsKV[i], actualSharedPrefixLen, needInit);

        tilingData.promptAttentionInitOutputParams.set_needInit(needInit);

        if (enableAlibiPse) {
            OP_CHECK_IF((actualSeqLengths[i] != actualSeqLengthsKV[i]),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "When pseType = 2/3, actualSeqLengths[%u](seq size of query)=%ld must be equal to actualSeqLengthsKv[%u](seq size of key)=%ld",
                    i, actualSeqLengths[i], i, actualSeqLengthsKV[i]),
                return false);
        }
    }
    return true;
}

bool PromptFlashAttentionTilingV2::CheckPerblockCrossover(ContextParamsForPFATiling& contextKeyParams,
    PromptFlashAttentionTilingDataV2& tilingData)
{
    if (!enablePerblockQuant) {
        return true;
    }
    OP_CHECK_IF(enableActSeqLen && (inputLayout == InputLayout::TND), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "ActSeqLen is not supported in per-block quant scenario!"),
            return false);
    OP_CHECK_IF(enableActSeqLenKV && (inputLayout == InputLayout::TND), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "ActSeqLenKV is not supported in per-block quant scenario!"),
            return false);
    // per-block does not support invalid line correction. (innerPrecise = 2/3)
    OP_CHECK_IF(tilingData.promptAttentionBaseParams.get_isRowInvalid() != 0,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "innerPrecise must be 0 or 1 in per-block quant scenario!"),
            return false);
    OP_CHECK_IF(contextKeyParams.isSoftMaxLseEnable, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "LSE is not supported in per-block quant scenario!"),
        return false);
    OP_CHECK_IF(enableLeftPadding, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "leftpadding is not supported in per-block quant scenario!"),
        return false);
    OP_CHECK_IF(enablePFAMLA, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "PFAMLA is not supported in per-block quant scenario!"),
        return false);
    OP_CHECK_IF(enablePFARope, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "Rope is not supported in per-block quant scenario!"),
        return false);
    OP_CHECK_IF(enableMask, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "mask is not supported in per-block quant scenario!"),
        return false);
    OP_CHECK_IF(enablePseShift, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "PseShift is not supported in per-block quant scenario!"),
        return false);
    OP_CHECK_IF(enableAlibiPse, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "AlibiPse is not supported in per-block quant scenario!"),
        return false);
    // 全量化不支持system prefix
    OP_CHECK_IF((contextKeyParams.keySharedPrefix != nullptr || contextKeyParams.valueSharedPrefix != nullptr),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "keySharedPrefix and valueSharedPrefix should be null in per-block quant scenario!"),
        return false);    
    return true;
}

void PromptFlashAttentionTilingV2::SetTilingDataAttribute(ContextParamsForPFATiling& contextKeyParams,
    PromptFlashAttentionTilingDataV2& tilingData) 
{
    tilingData.promptAttentionBaseParams.set_preTokens(sparsePreTokens);
    tilingData.promptAttentionBaseParams.set_nextTokens(sparseNextTokens);
    tilingData.promptAttentionBaseParams.set_sparseMode(static_cast<uint32_t>(sparseModeVal));

    bool isActualSeqLengthsNull = isMaxWorkspace ? true : !enableActSeqLen;
    bool isActualSeqLengthsKVNull = isMaxWorkspace ? true : !enableActSeqLenKV;
    tilingData.promptAttentionBaseParams.set_isActualSeqLengthsNull(static_cast<uint32_t>(isActualSeqLengthsNull));
    tilingData.promptAttentionBaseParams.set_isActualSeqLengthsKVNull(static_cast<uint32_t>(isActualSeqLengthsKVNull));
    tilingData.promptAttentionBaseParams.set_actualSeqLengthsSize(actSeqLenDims);
    tilingData.promptAttentionBaseParams.set_actualSeqLengthsKVSize(actSeqLenKVDims);
    tilingData.promptAttentionBaseParams.set_usePseShift(usePseShift);
    tilingData.promptAttentionBaseParams.set_pseShiftTypeByteNum(pseShiftTypeByteNum);
    tilingData.promptAttentionBaseParams.set_pseMaskMaxSize(pseMaskMaxSize);
    tilingData.promptAttentionSingleCoreParams.set_pseShiftBatch(pseShiftBatch);
    tilingData.promptAttentionBaseParams.set_pseShiftS1Size(pseShiftS1);
    tilingData.promptAttentionBaseParams.set_pseShiftS2Size(pseShiftS2);

    tilingData.promptAttentionBaseParams.set_isKvContinuous(contextKeyParams.isKvContinuous);
    tilingData.promptAttentionBaseParams.set_isQHasLeftPadding(contextKeyParams.queryPaddingSize != nullptr ? 1 : 0);
    tilingData.promptAttentionBaseParams.set_isKVHasLeftPadding(contextKeyParams.kvPaddingSize != nullptr ? 1 : 0);

    tilingData.promptAttentionBaseParams.set_fromFused((contextKeyParams.fromFused == FROM_FUSED_FLAG) ? 1 : 0);
    tilingData.promptAttentionBaseParams.set_isBSNDOut(contextKeyParams.isBSNDOut);
    tilingData.promptAttentionBaseParams.set_transposeLayout(contextKeyParams.transposeLayout);
    tilingData.promptAttentionBaseParams.set_isSoftMaxLseEnable(contextKeyParams.isSoftMaxLseEnable);

    uint32_t originHeadSize = enableIFAMLA ? tilingData.promptAttentionBaseParams.get_headSize() :
                                        tilingData.promptAttentionBaseParams.get_qkHeadSize();
    uint32_t blockElementCnt = BYTE_BLOCK / dataTypeSize;
    if (originHeadSize % blockElementCnt != 0) { // Determine if D is aligned with 32B, using fp16 type with 16 elements.
        tilingData.promptAttentionBaseParams.set_alignedHeadSize((( originHeadSize + blockElementCnt - 1) / 
            blockElementCnt) * blockElementCnt);
        isDNoTail = false;
    } else {
        tilingData.promptAttentionBaseParams.set_alignedHeadSize(originHeadSize);
    }
}

void PromptFlashAttentionTilingV2::GetEnableDN(ContextParamsForPFATiling& contextKeyParams,
    PromptFlashAttentionTilingDataV2& tilingData, PFAShapeInfo& queryShapeInfo, PFAShapeInfo& valueShapeInfo,
    std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV) 
{
    // 使能DN条件：1.sOuter >= 128; 2.d等长且不大于128; 3.输入类型为fp16/bf16; 4.不带mask、pse、MLA等高阶特性; 5.FP8 per-block全量化
    bool isQKVActualSeqLengthsRight = true;
    constexpr uint32_t dLimitDN = 128;
    constexpr uint32_t vecCoreNum = 2;
    constexpr uint32_t sOuterLimitDN = 64;
    enableDN = (!enableMask && !enablePseShift && !enableAlibiPse && !enablePA && !enablePFAMLA && !enablePFARope && 
        (queryShapeInfo.d <= dLimitDN) && (valueShapeInfo.d <= dLimitDN) &&
        !isKVHasPrefix && (contextKeyParams.inputDataType == ge::DT_FLOAT16 || contextKeyParams.inputDataType == ge::DT_BF16 || enablePerblockQuant) &&
        (tilingData.promptAttentionSingleCoreParams.get_singleProcessSOuterSize() * vecCoreNum > sOuterLimitDN));
    for (uint32_t i = LOOP_BEGIN_NUM; i < queryShapeInfo.b; i++) {
        if ((actualSeqLengths[i] % 32 > 0) || (actualSeqLengthsKV[i] <= 128)) { // 32: 只针对对齐场景修改基本快大小; 128: 扩大sInner的KV_S限制
            isQKVActualSeqLengthsRight = false;
            break;
        }
    }

    // 64：扩大sInner的dsize限制
    if (enableDN && isQKVActualSeqLengthsRight && (queryShapeInfo.d == valueShapeInfo.d) && (queryShapeInfo.d == 64)) {   
        tilingData.promptAttentionSingleCoreParams.set_singleProcessSInnerSize(256U);    // 256U: 设置sInner的dsize
    }   

    // 128： perblock全量化，扩大sInner切块大小 d<=128
    if (enableDN && (queryShapeInfo.d == valueShapeInfo.d) && (queryShapeInfo.d <= 128) && enablePerblockQuant) {
        tilingData.promptAttentionSingleCoreParams.set_singleProcessSInnerSize(256U);   // 256U: 设置sInner的dsize
    }
}

void PromptFlashAttentionTilingV2::SetTilingData(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo,
    PFAShapeInfo& queryRopeShapeInfo, PFAShapeInfo& valueShapeInfo, PromptFlashAttentionTilingDataV2& tilingData) 
{
    typeByteNum = BYTE_BLOCK / dataTypeSize;
    outputTypeByteNum = BYTE_BLOCK / outputDataTypeSize;
    softmaxTypeByteNum = BYTE_BLOCK / softmaxDataTypeSize;
    maskTypeByteNum = BYTE_BLOCK / maskElemSize;
    tilingData.promptAttentionBaseParams.set_typeByteNum(typeByteNum);
    tilingData.promptAttentionBaseParams.set_maskTypeByteNum(maskTypeByteNum);
    tilingData.promptAttentionBaseParams.set_softmaxTypeByteNum(softmaxTypeByteNum);
    tilingData.promptAttentionBaseParams.set_outputTypeByteNum(outputTypeByteNum);

    uint32_t deqScaleTypeFlag = (contextKeyParams.deqScaleType == DT_UINT64) ? 0U : 1U;
    uint32_t deqScale2TypeFlag = (contextKeyParams.deqScale2Type == DT_UINT64) ? 0U : 1U;
    tilingData.promptAttentionBaseParams.set_deqScaleFlag(deqScaleTypeFlag);
    tilingData.promptAttentionBaseParams.set_deqScale2Flag(deqScale2TypeFlag);
    tilingData.promptAttentionBaseParams.set_scaleValue(*contextKeyParams.scaleValue);
    // Perchannel judgment to be adapted, maintain the existing logic firstly.
    tilingData.promptAttentionBaseParams.set_isQuant2Perchannel(0);
    tilingData.promptAttentionBaseParams.set_isQuant2BF16(0);
    tilingData.promptAttentionBaseParams.set_isQuant2FP16(0);
    // IFA flag
    tilingData.promptAttentionBaseParams.set_isIFA(enableIFA);
    if (enablePostQuant) {
        if (contextKeyParams.scale2Shape->GetStorageShape().GetShapeSize() > 1) {
            tilingData.promptAttentionBaseParams.set_isQuant2Perchannel(1);
        }
        if (contextKeyParams.quantScale2Type == ge::DT_BF16) {
            tilingData.promptAttentionBaseParams.set_isQuant2BF16(1);
        }
        if (contextKeyParams.quantScale2Type == ge::DT_FLOAT16 &&
            contextKeyParams.hasKeyAntiquantScale && contextKeyParams.hasValueAntiquantScale) {
            tilingData.promptAttentionBaseParams.set_isQuant2FP16(1);
        }
    }

    if (enablePA) {
        tilingData.promptAttentionBaseParams.set_blockSize(*contextKeyParams.blockSize);
    } else {
        tilingData.promptAttentionBaseParams.set_blockSize(BLOCK_SIZE_BASE);
    }
    tilingData.promptAttentionBaseParams.set_blockTableDim2(blockTableDim2);
    tilingData.promptAttentionBaseParams.set_PABlockNumSum(paBlockNumSum);
    uint32_t isLayoutSH = (inputLayout == InputLayout::SH) ? 1U : 0U;
    tilingData.promptAttentionBaseParams.set_isLayoutSH(isLayoutSH);
    tilingData.promptAttentionBaseParams.set_dimNumOfseq(queryShapeInfo.b);
    tilingData.promptAttentionBaseParams.set_headSize(queryShapeInfo.d);
    tilingData.promptAttentionBaseParams.set_ropeHeadSize(queryRopeShapeInfo.d);
    if (enableIFAMLA) {
        tilingData.promptAttentionBaseParams.set_qkHeadSize(queryShapeInfo.d + queryRopeShapeInfo.d);
    } else {
        tilingData.promptAttentionBaseParams.set_qkHeadSize(queryShapeInfo.d);
    }
    tilingData.promptAttentionBaseParams.set_vHeadSize(valueShapeInfo.d);
    tilingData.promptAttentionBaseParams.set_seqInnerSize(S2);
    tilingData.promptAttentionBaseParams.set_seqSize(queryShapeInfo.s);
    tilingData.promptAttentionBaseParams.set_headNumSize(queryShapeInfo.n);
    tilingData.promptAttentionBaseParams.set_batchSize(queryShapeInfo.b);
    tilingData.promptAttentionBaseParams.set_t1Size(t1Size);
    tilingData.promptAttentionBaseParams.set_t2Size(t2Size);
    SetTilingDataAttribute(contextKeyParams, tilingData);
}

void PromptFlashAttentionTilingV2::InferTilingMod(const ContextParamsForPFATiling& contextKeyParams,
    std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV, uint32_t actualSeqArrayLen,
    uint32_t d) 
{
    // Determine whether to use the norm template
    int64_t minActualSeqLengths = INT64_MAX;
    int64_t minActualSeqLengthsKV = INT64_MAX;
    for (uint32_t i = 0; i < actualSeqArrayLen; ++i) {
        minActualSeqLengths = std::min(minActualSeqLengths, actualSeqLengths[i]);
        minActualSeqLengthsKV = std::min(minActualSeqLengthsKV, actualSeqLengthsKV[i]);
    }
    if ((minActualSeqLengths >= MATMUL_NORM_MIN_SEQ) && (minActualSeqLengthsKV >= MATMUL_NORM_MIN_SEQ) &&
        (d == MATMUL_NORM_MIN_HEADSIZE) && (inputType == ge::DT_FLOAT16) &&
        (contextKeyParams.kDataType == ge::DT_FLOAT16) && (contextKeyParams.maskDataType == ge::DT_BOOL) &&
        (outputType == ge::DT_FLOAT16) && (usePseShift == 0) && (inputLayout == InputLayout::BNSD) &&
        (sparseModeVal == SPARSE_MODE_BAND) && (!enablePA)) {
        // Currently, only the matmul norm template is open for the X1 scenario
        enableMatmulNorm = true;
    }
}

void PromptFlashAttentionTilingV2::InferSplitCoreMode() 
{
    splitCoreMode = SplitCoreMode::SPLIT_NBS_CUBE;
}

void PromptFlashAttentionTilingV2::InferConstantization() 
{
    isConstantization = true;
}

void PromptFlashAttentionTilingV2::GetMatMulType(matmul_tiling::DataType &mmInputType,
    matmul_tiling::DataType &mmOutputType) 
{
    if (inputType == ge::DT_FLOAT16 && innerPrecise == HIGH_PRECISION) {
        mmInputType = matmul_tiling::DataType::DT_FLOAT16;
        mmOutputType = matmul_tiling::DataType::DT_FLOAT;
    } else if (inputType == ge::DT_BF16) {
        mmInputType = matmul_tiling::DataType::DT_BF16;
        mmOutputType = matmul_tiling::DataType::DT_FLOAT;
    } else if (inputType == ge::DT_INT8) {
        mmInputType = matmul_tiling::DataType::DT_INT8;
        mmOutputType = matmul_tiling::DataType::DT_INT32;
    }
}

bool PromptFlashAttentionTilingV2::EnableMTE2BmmPipe(PromptFlashAttentionTilingDataV2& tilingData,
    matmul_tiling::MatmulApiTiling& bmm, TCubeTiling& bmmTilingData, uint32_t sOuterFactor, uint32_t sInnerFactor) 
{
    // When the size is greater than 16, use xiaoe speculative inference.
    if (tilingData.promptAttentionBaseParams.get_seqSize() > 16) {
        return true;
    }
    uint32_t baseK = 32U;
    uint32_t headSize = tilingData.promptAttentionBaseParams.get_qkHeadSize();
    if (headSize % baseK != 0) {
        return true;
    }

    uint32_t baseM = std::min(uint32_t(128), sOuterFactor);
    uint32_t baseN = std::min(uint32_t(512), sInnerFactor);
    if (enablePA) {
        baseN = BLOCK_SIZE_BASE;
    }
    int32_t ret = 0;
    ret = bmm.SetFixSplit(baseM, baseN, baseK);
    OP_CHECK_IF(ret != 0, OPS_REPORT_VECTOR_INNER_ERR("PromptFlashAttention", "bmm SetFixSplit failed, ret = %d!", ret),
        return false);
    bool res = bmm.GetTiling(bmmTilingData) != -1;
    return res;
}

void PromptFlashAttentionTilingV2::EnableBmmDoubleBuffer(TCubeTiling& bmmTilingData) const
{
    if ((bmmTilingData.get_depthA1() == 1) && (bmmTilingData.get_depthB1() == 1)) {
        bmmTilingData.set_depthA1(2); // 2 : depthA1
        bmmTilingData.set_depthB1(2); // 2 : depthB1
    }
}

bool PromptFlashAttentionTilingV2::PromptFlashAttentionCheckBmm1(PromptFlashAttentionTilingDataV2& tilingData,
    TCubeTiling& bmm1TilingData, int64_t l1SizeRemain, int64_t l0CSize, uint32_t sOuterFactor, uint32_t sInnerFactor, bool autoBaseMNK) 
{
    if (splitCoreMode == SplitCoreMode::SPLIT_NBS_CUBE) {
        sOuterFactor = sOuterFactor * CV_RATIO;
    }
    matmul_tiling::MatmulApiTiling bmm1(ascendPlatformInfo);

    matmul_tiling::DataType bmm1InputType = matmul_tiling::DataType::DT_FLOAT16;
    matmul_tiling::DataType bmm1OutputType = matmul_tiling::DataType::DT_FLOAT16;
    GetMatMulType(bmm1InputType, bmm1OutputType);
    bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1InputType, false);
    bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1InputType, true);
    bmm1.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND_ALIGN, bmm1OutputType);

    int32_t ret = bmm1.SetShape(sOuterFactor, sInnerFactor, tilingData.promptAttentionBaseParams.get_qkHeadSize());
    OP_CHECK_IF(ret != 0, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "bmm1 SetShape failed, ret = %d!", ret),
        return false);
    int32_t ratio = tilingData.promptAttentionBaseParams.get_headNumRatio();
    int32_t strideQ = tilingData.promptAttentionBaseParams.get_qkHeadSize() *
        tilingData.promptAttentionBaseParams.get_headNumSize();
    if (ratio == 0) {
        return false;
    }
    int32_t strideK = strideQ / ratio;
    if ((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::BSND) || (inputLayout == InputLayout::TND)) {
        if (enableKVAntiquant || (inputLayout == InputLayout::TND && enablePA && paLayoutType == 0)) {
            bmm1.SetOrgShape(tilingData.promptAttentionBaseParams.get_seqSize(),
                tilingData.promptAttentionBaseParams.get_seqInnerSize(),
                strideQ, tilingData.promptAttentionBaseParams.get_qkHeadSize());
        } else if (enableIFAMLA || enableIFA) {
            bmm1.SetOrgShape(tilingData.promptAttentionBaseParams.get_seqSize(),
            tilingData.promptAttentionBaseParams.get_seqInnerSize(),
            tilingData.promptAttentionBaseParams.get_qkHeadSize(), strideK);
        } else {
            bmm1.SetOrgShape(tilingData.promptAttentionBaseParams.get_seqSize(),
            tilingData.promptAttentionBaseParams.get_seqInnerSize(), strideQ, strideK);
        }
    } else if (inputLayout == InputLayout::BNSD) {
        if (enablePA && paLayoutType == 1) { // The left matrix of PA is BNSD, and the right matrix is BSH.
            bmm1.SetOrgShape(tilingData.promptAttentionBaseParams.get_seqSize(),
                tilingData.promptAttentionBaseParams.get_seqInnerSize(),
                tilingData.promptAttentionBaseParams.get_qkHeadSize(), strideK);
        } else {
            bmm1.SetOrgShape(tilingData.promptAttentionBaseParams.get_seqSize(),
                tilingData.promptAttentionBaseParams.get_seqInnerSize(),
                tilingData.promptAttentionBaseParams.get_qkHeadSize());
        }
    }

    bmm1.SetBias(false);
    ret = bmm1.SetBufferSpace(l1SizeRemain, l0CSize);
    OP_CHECK_IF(ret != 0, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName,
        "bmm1 SetBufferSpace failed, l1SizeRemain = %ld, l0CSize = %ld, ret = %d!", l1SizeRemain, l0CSize, ret),
        return false);
    if (!enableIFAMLA && enablePA) {
        ret = bmm1.SetFixSplit(sOuterFactor, BLOCK_SIZE_BASE);
    } else {
        ret = bmm1.SetFixSplit(sOuterFactor, sInnerFactor);
    }
    OP_CHECK_IF(ret != 0, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName,
        "bmm1 SetFixSplit failed, l1SizeRemain = %ld, l0CSize = %ld, sOuterFactor = %u, sInnerFactor = %u, ret = %d!",
        l1SizeRemain, l0CSize, sOuterFactor, sInnerFactor, ret),
        return false);

    ret = bmm1.GetTiling(bmm1TilingData);
    if (autoBaseMNK) {
        uint32_t baseM = std::min(uint32_t(128), sOuterFactor);
        uint32_t baseN = std::min(uint32_t(128), sInnerFactor);
        if (enableMatmulNorm) {
            uint32_t baseK = 128U;
            ret = bmm1.SetFixSplit(baseM, baseN, baseK);
            OP_CHECK_IF(ret != 0, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName,
                "bmm1 SetFixSplit failed, ret = %d!", ret),
                return false);
            ret = bmm1.GetTiling(bmm1TilingData);
        } else {
            uint32_t baseK = 64U;
            if (enablePA) {
                baseN = BLOCK_SIZE_BASE;
            }
            if (ret != 0) {
                ret = bmm1.SetFixSplit(baseM, baseN, baseK);
                OP_CHECK_IF(ret != 0, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName,
                    "bmm1 SetFixSplit failed, ret = %d!", ret),
                    return false);
                ret = bmm1.GetTiling(bmm1TilingData);
            }
        }
    }
    // Get tiling fail for bmm1.
    OP_CHECK_IF(ret != 0, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName,
        "bmm1 GetTiling failed, l1SizeRemain = %ld, l0CSize = %ld, sOuterFactor = %u, sInnerFactor = %u, autoBaseMNK = %d,"
        "ret = %d!", l1SizeRemain, l0CSize, sOuterFactor, sInnerFactor, autoBaseMNK, ret),
        return false);

    bmm1TilingData.set_shareMode(0);
    bmm1TilingData.set_shareL1Size(l1SizeRemain);
    bmm1TilingData.set_shareL0CSize(l0CSize);

    bmm1TilingData.set_shareUbSize(0);
    EnableBmmDoubleBuffer(bmm1TilingData); // Open the double buffer for BMM1 calculation, and BMM1's MTE2 can be bound.
    // Open MTE2 Matmul pipeline.
    bool res = EnableMTE2BmmPipe(tilingData, bmm1, bmm1TilingData, sOuterFactor, sInnerFactor);
    // EnableMTE2BmmPipe fail.
    OP_CHECK_IF(res == false, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "EnableMTE2BmmPipe failed!"),
        return false);

    return true;
}

bool PromptFlashAttentionTilingV2::AdjustCVTilingCVDiff(const ContextParamsForPFATiling& contextKeyParams,
    uint32_t& sOuterFactor, uint32_t& sInnerFactor, uint32_t& softmaxSOuterFactor,
    PromptFlashAttentionTilingDataV2& tilingData, const PFAShapeInfo& queryShapeInfo) 
{
    uint32_t minFactor = SOUTER_FACTOR_DEFAULT;
    uint32_t rectangleFactor = SINNER_FACTOR_DEFAULT;
    softmaxSOuterFactor = SOUTER_FACTOR_DEFAULT;

    auto compileInfoPtr = contextKeyParams.compileInfoPtr;
    bool res = PromptFlashAttentionComputeCVDiffParams(tilingData, compileInfoPtr->l1Size, compileInfoPtr->l0CSize,
        minFactor, rectangleFactor);
    OP_CHECK_IF(res == false, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName,
        "PromptFlashAttentionComputeCVDiffParams failed!"),
        return false);

    if (tilingData.promptAttentionBaseParams.get_vHeadSize() <= 128 && !enablePFAMLA) { // 128 for D size
        bool checkDtype = contextKeyParams.inputDataType == ge::DT_FLOAT16 || contextKeyParams.inputDataType == ge::DT_BF16;
        bool checkQueryAndValueS = queryShapeInfo.s <= SOUTER_FACTOR_DEFAULT && S2 > SINNER_FACTOR_DEFAULT;
        uint32_t sparseMode = tilingData.promptAttentionBaseParams.get_sparseMode();
        int32_t preTokens = tilingData.promptAttentionBaseParams.get_preTokens();
        int32_t nextTokens = tilingData.promptAttentionBaseParams.get_nextTokens();
        if (sparseMode == SPARSE_MODE_NO_MASK) {
            preTokens = (preTokens > 0) ? 0 : preTokens;
        } else if (sparseMode == SPARSE_MODE_BAND) {
            nextTokens = (nextTokens > 0) ? 0 : nextTokens;
        }
        // actual calculation area is smaller than SINNER_FACTOR_DEFAULT in SPARSE_MODE_LEFT_UP mode
        bool checkSparseMode = ((sparseMode == SPARSE_MODE_ALL_MASK) || (sparseMode == SPARSE_MODE_RIGHT_DOWN) || \
            (((sparseMode == SPARSE_MODE_NO_MASK) || (sparseMode == SPARSE_MODE_BAND)) && \
            (preTokens + nextTokens > static_cast<int32_t>(SINNER_FACTOR_DEFAULT))));
        if (checkDtype && checkQueryAndValueS && checkSparseMode && !enablePFARope && !enablePerblockQuant) {
            minFactor = SOUTER_FACTOR_SUB;
            rectangleFactor = SINNER_FACTOR_DOUBLE;
            softmaxSOuterFactor = SOUTER_FACTOR_SUB;
        } else if (((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::BSND) || (inputLayout == InputLayout::TND)) && enablePFAMerge) {
            minFactor = SOUTER_FACTOR_SUB;
            rectangleFactor = SINNER_FACTOR_DOUBLE;
        }
    } else if (tilingData.promptAttentionBaseParams.get_vHeadSize() > 128 && !enableIFAMLA && !enableIFA) { // 128 : D size
        if (!faRunFlag_) {
            minFactor = SOUTER_FACTOR_SUB;
            rectangleFactor = SINNER_FACTOR_SUB;
        } else if (((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::BSND) || (inputLayout == InputLayout::TND)) && enablePFAMerge && tilingData.promptAttentionBaseParams.get_vHeadSize() <= 256) { // 256 : D size
            minFactor = SOUTER_FACTOR_SUB;
            rectangleFactor = SINNER_FACTOR_DOUBLE;
        } else {
            minFactor = SOUTER_FACTOR_DEFAULT;
            rectangleFactor = SINNER_FACTOR_DEFAULT;
        }
        softmaxSOuterFactor = SOUTER_FACTOR_SUB;
    } else if (enableIFAMLA || (enableIFA && tilingData.promptAttentionBaseParams.get_vHeadSize() > 128)) { // IFA VD > 128
        if (!faRunFlag_ || enableIFAMLA) {
            minFactor = SOUTER_FACTOR_SUB;
        } else {
            minFactor = SOUTER_FACTOR_DEFAULT;
        }
        rectangleFactor = !faRunFlag_ && !enableIFAMLA && enablePseShift ? SINNER_FACTOR_SUB : SINNER_FACTOR_DEFAULT;
        softmaxSOuterFactor = SOUTER_FACTOR_SUB;
    }
    sOuterFactor = minFactor;
    sInnerFactor = rectangleFactor;

    return true;
}

bool PromptFlashAttentionTilingV2::PromptFlashAttentionCheckBmm2(PromptFlashAttentionTilingDataV2& tilingData,
    TCubeTiling& bmm2TilingData, int64_t l1SizeRemain, int64_t l0CSize, uint32_t sOuterFactor, uint32_t sInnerFactor,
    uint32_t dSplitFactor, bool autoBaseMNK) 
{
    int32_t ret = 0;
    matmul_tiling::MatmulApiTiling bmm2(ascendPlatformInfo);
    matmul_tiling::DataType bmm2InputType = matmul_tiling::DataType::DT_FLOAT16;
    matmul_tiling::DataType bmm2OutputType = matmul_tiling::DataType::DT_FLOAT16;
    GetMatMulType(bmm2InputType, bmm2OutputType);
    bmm2.SetAType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, bmm2InputType, false);
    bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2InputType, false);
    bmm2.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND_ALIGN, bmm2OutputType);
    ret = bmm2.SetShape(sOuterFactor, tilingData.promptAttentionBaseParams.get_vHeadSize(), sInnerFactor);

    OP_CHECK_IF(ret != 0, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName,
        "bmm2 set SetShape failed, sOuterFactor = %u, sInnerFactor = %u, ret = %d!", sOuterFactor, sInnerFactor, ret),
        return false);
    int32_t ratio = tilingData.promptAttentionBaseParams.get_headNumRatio();
    if (ratio == 0) {
        return false;
    }
    int32_t strideV = tilingData.promptAttentionBaseParams.get_vHeadSize() *
        tilingData.promptAttentionBaseParams.get_headNumSize() / ratio;
    if ((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::BSND) || (inputLayout == InputLayout::TND)) {
        bmm2.SetOrgShape(tilingData.promptAttentionBaseParams.get_seqSize(), strideV,
            tilingData.promptAttentionBaseParams.get_seqInnerSize());
        if (enableKVAntiquant || (inputLayout == InputLayout::TND && enablePA && paLayoutType == 0)) {
            bmm2.SetOrgShape(tilingData.promptAttentionBaseParams.get_seqSize(),
                tilingData.promptAttentionBaseParams.get_vHeadSize(),
                tilingData.promptAttentionBaseParams.get_seqInnerSize());
        }
    } else if (inputLayout == InputLayout::BNSD) {
        if (enablePA && paLayoutType == 1) { // The left matrix of PA is BNSD, and the right matrix is of PA is BSH.
            bmm2.SetOrgShape(tilingData.promptAttentionBaseParams.get_seqSize(), strideV,
                tilingData.promptAttentionBaseParams.get_seqInnerSize());
        } else {
            bmm2.SetOrgShape(tilingData.promptAttentionBaseParams.get_seqSize(),
                tilingData.promptAttentionBaseParams.get_vHeadSize(),
                tilingData.promptAttentionBaseParams.get_seqInnerSize());
        }
    }

    bmm2.SetBias(false);
    ret = bmm2.SetBufferSpace(l1SizeRemain, l0CSize);
    OP_CHECK_IF(ret != 0, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName,
        "bmm2 set SetBufferSpace failed, l1SizeRemain = %ld, l0CSize = %ld, sOuterFactor = %u, sInnerFactor = %u, ret = %d!",
        l1SizeRemain, l0CSize, sOuterFactor, sInnerFactor, ret),
        return false);

    if (autoBaseMNK) {
        if (enableMatmulNorm) {
            uint32_t baseM = std::min(uint32_t(128), sOuterFactor);
            uint32_t baseN = std::min(uint32_t(128), tilingData.promptAttentionBaseParams.get_vHeadSize());
            uint32_t baseK = 128U;
            ret = bmm2.SetFixSplit(baseM, baseN, baseK);
            OP_CHECK_IF(ret != 0, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, 
                "bmm2 SetFixSplit failed!"),
                return false);
        }
        ret = bmm2.GetTiling(bmm2TilingData);
    } else {
        if ((isDNoTail) || (splitS2 == 0)) {
            ret = bmm2.SetFixSplit(sOuterFactor, dSplitFactor);
        } else {
            ret = bmm2.SetFixSplit(sOuterFactor, tilingData.promptAttentionBaseParams.get_alignedHeadSize());
        }
        OP_CHECK_IF(ret != 0, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, 
            "bmm2 SetFixSplit failed!"),
            return false);
        ret = bmm2.GetTiling(bmm2TilingData);
    }
    OP_CHECK_IF(ret != 0, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName,
        "bmm2 set GetTiling failed, l1SizeRemain = %ld, l0CSize = %ld, sOuterFactor = %u, sInnerFactor = %u, "
        "autoBaseMNK = %d, ret = %d!", l1SizeRemain, l0CSize, sOuterFactor, sInnerFactor, autoBaseMNK, ret),
        return false);
    bmm2TilingData.set_shareMode(0);
    bmm2TilingData.set_shareL1Size(l1SizeRemain);
    bmm2TilingData.set_shareL0CSize(l0CSize);
    OP_CHECK_IF(ret != 0, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName,
        "bmm2 set shareL0CSize failed, l1SizeRemain = %ld, l0CSize = %ld, sOuterFactor = %u, sInnerFactor = %u,"
        "autoBaseMNK = %d, ret = %d!", l1SizeRemain, l0CSize, sOuterFactor, sInnerFactor, autoBaseMNK, ret),
        return false);
    bmm2TilingData.set_shareUbSize(0);
    return true;
}

bool PromptFlashAttentionTilingV2::PromptFlashAttentionComputeCVDiffParams(PromptFlashAttentionTilingDataV2& tilingData,
    int64_t l1Size, int64_t l0CSize, uint32_t& sOuterFactor, uint32_t &sInnerFactor) 
{
    constexpr uint32_t dSplitFactorBmm2 = 128U;
    int32_t l1SizeRemain = l1Size;
    bool res = PromptFlashAttentionCheckBmm1(tilingData, tilingData.bmm1TilingDataRect, l1SizeRemain, l0CSize, sOuterFactor,
        sInnerFactor, true);
    OP_CHECK_IF(!res, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "PromptFlashAttentionCheckBmm1 failed!"),
        return false);

    res = PromptFlashAttentionCheckBmm2(tilingData, tilingData.bmm2TilingDataRect, l1SizeRemain, l0CSize, sOuterFactor,
        sInnerFactor, dSplitFactorBmm2, true);
    OP_CHECK_IF(!res, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "PromptFlashAttentionCheckBmm2 failed!"),
        return false);

    return true;
}

void PromptFlashAttentionTilingV2::GetPreNextTokensLeftUp(PromptFlashAttentionTilingDataV2& tilingData,
    int64_t actualSeqLength, int64_t actualSeqLengthKV, int64_t& preTokensLeftUp, int64_t& nextTokensLeftUp) 
{
    PromptAttentionBaseParams* baseParams = &tilingData.promptAttentionBaseParams;
    if (baseParams->get_sparseMode() == SPARSE_MODE_RIGHT_DOWN) {
        preTokensLeftUp = SPARSE_MODE_INT_MAX;
        if (enableIFAMLA) {
            if (inputLayout == InputLayout::BSND || inputLayout == InputLayout::BSH || inputLayout == InputLayout::TND) {
                nextTokensLeftUp = actualSeqLengthKV * gSize - actualSeqLength;
            } else { // BNSD场景下分核不做优化
                nextTokensLeftUp = SPARSE_MODE_INT_MAX;
            }
        } else if (enableIFA){
            nextTokensLeftUp = actualSeqLengthKV * gSize - actualSeqLength;
        }else {
            nextTokensLeftUp = actualSeqLengthKV - actualSeqLength;
        }
    } else if (baseParams->get_sparseMode() == SPARSE_MODE_BAND) {
        if (enableIFAMLA) {
            if (inputLayout == InputLayout::BSND || inputLayout == InputLayout::BSH || inputLayout == InputLayout::TND) {
                preTokensLeftUp = baseParams->get_preTokens() * gSize - actualSeqLengthKV * gSize + actualSeqLength;
                nextTokensLeftUp = baseParams->get_nextTokens() * gSize + actualSeqLengthKV * gSize - actualSeqLength;
            } else { // BNSD场景下分核不做优化
                preTokensLeftUp = SPARSE_MODE_INT_MAX;
                nextTokensLeftUp = SPARSE_MODE_INT_MAX;
            }
        } else if (enableIFA){
            preTokensLeftUp = baseParams->get_preTokens() * gSize - actualSeqLengthKV * gSize + actualSeqLength;
            nextTokensLeftUp = baseParams->get_nextTokens() * gSize + actualSeqLengthKV * gSize - actualSeqLength;
        }else {
            preTokensLeftUp = baseParams->get_preTokens() - actualSeqLengthKV + actualSeqLength;
            nextTokensLeftUp = baseParams->get_nextTokens() + actualSeqLengthKV - actualSeqLength;
        }
    } else {
        if (enableIFAMLA) {
            if (inputLayout == InputLayout::BSND || inputLayout == InputLayout::BSH || inputLayout == InputLayout::TND) {
                preTokensLeftUp = baseParams->get_preTokens() * gSize;
                nextTokensLeftUp = baseParams->get_nextTokens() * gSize;
            } else { // BNSD场景下分核不做优化
                preTokensLeftUp = SPARSE_MODE_INT_MAX;
                nextTokensLeftUp = SPARSE_MODE_INT_MAX;
            }
        } else if(enableIFA){
            preTokensLeftUp = baseParams->get_preTokens() * gSize;
            nextTokensLeftUp = baseParams->get_nextTokens() * gSize;
        }else {
            preTokensLeftUp = baseParams->get_preTokens();
            nextTokensLeftUp = baseParams->get_nextTokens();
        }
    }
}

int64_t PromptFlashAttentionTilingV2::GetActualInnerBlockNums(int64_t sInnerIndexStart, int64_t sInnerIndexEnd,
    int64_t innerBlockNums) const 
{
    int64_t sInnerBlockNums = 0;

    if (sInnerIndexEnd < 0) {
        sInnerBlockNums = 0;
    } else if (sInnerIndexEnd < innerBlockNums) {
        sInnerBlockNums = (sInnerIndexStart < 0) ? (sInnerIndexEnd + 1) : (sInnerIndexEnd - sInnerIndexStart + 1);
    } else {
        sInnerBlockNums = (sInnerIndexStart < 0) ? innerBlockNums :
            (sInnerIndexStart < innerBlockNums ? innerBlockNums - sInnerIndexStart : 0);
    }

    return sInnerBlockNums;
}

int64_t PromptFlashAttentionTilingV2::SumOfArithmeticSeries(int64_t an, int64_t d) const 
{
    // 等差数列求和，an：等差数列第n项，d：等差数列公差
    if (d == 0) {
        return 0;
    }
    return (an > 0) ? (an % d + an) * (an / d + 1) / 2 : 0; // 2: 等差数列求和公式分母
}

int64_t PromptFlashAttentionTilingV2::GetCutBlockNums(int64_t blockSeqLengthKV, int64_t blockSeqLength,
        int64_t sInner, int64_t sOuter, int64_t token) const
{
    // 以nextToken视角计算完全被nextToken掩盖的基本块数
    int64_t blockNums = 0;
    int64_t blockToken = token > 0 ? ((token + sInner - 1) / sInner * sInner) : (token / sInner * sInner);
    int64_t outDivIn = sOuter > sInner ? sOuter / sInner : 1;
    int64_t InDivOut = sInner > sOuter ? sInner / sOuter : 1;
    int64_t tolerance = 0;
    int64_t smallSize = 0;
    if (outDivIn >= 1) {
        tolerance = outDivIn;
        smallSize = sInner;
    } else {
        tolerance = InDivOut;
        smallSize = sOuter;
    }

    // nextToken与上边右边构成的大三角形
    int64_t innerCutBlockNums = (blockSeqLengthKV - blockToken) / smallSize - tolerance;
    blockNums += SumOfArithmeticSeries(innerCutBlockNums, tolerance);

    // nextToken与上边左边构成的左侧三角形，需要减去
    int64_t innerCutBlockLeftNums = -blockToken / smallSize - tolerance;
    blockNums -= SumOfArithmeticSeries(innerCutBlockLeftNums, tolerance);

    // nextToken与下边右边构成的下侧三角形，需要减去
    int64_t innerCutBlockDownNums = (blockSeqLengthKV - blockSeqLength - blockToken) / smallSize - tolerance;
    blockNums -= SumOfArithmeticSeries(innerCutBlockDownNums, tolerance);

    // nextToken与下边左边构成的小三角形，是前两个三角形的重叠部分，需要加上
    int64_t innerCutBlockLeftDownNums = (-blockToken - blockSeqLength) / smallSize - tolerance;
    blockNums += SumOfArithmeticSeries(innerCutBlockLeftDownNums, tolerance);
    return blockNums;
}

// 函数内部不处理prefix逻辑，prefix场景下入参需自行传入actualSeqLengthKV + prefix
void PromptFlashAttentionTilingV2::FixParamWithRowInvalid(int64_t& actualSeqLength, int64_t actualSeqLengthKV,
    int64_t& preTokensLeftUp, int64_t& nextTokensLeftUp) const 
{
    // 若出现行无效，需要重新计算nexttokens，pretokens，actualseqlen，以便正确计算分核核数
    int64_t nextTokensError = (nextTokensLeftUp < 0) ? -nextTokensLeftUp : 0;
    nextTokensError = nextTokensError > actualSeqLength ? actualSeqLength : nextTokensError;
    int64_t preTokensError = 0;
    if (enableIFAMLA) {
        preTokensError = (actualSeqLength > actualSeqLengthKV * gSize + preTokensLeftUp) ?
            (actualSeqLength - actualSeqLengthKV * gSize - preTokensLeftUp) : 0;
    } else {
        preTokensError = (actualSeqLength > actualSeqLengthKV + preTokensLeftUp) ?
            (actualSeqLength - actualSeqLengthKV - preTokensLeftUp) : 0;
    }
    preTokensError = preTokensError > actualSeqLength ? actualSeqLength : preTokensError;

    // 若出现上方行无效，需要重新计算nexttokens，pretokens，actualseqlen
    nextTokensLeftUp += nextTokensError;
    preTokensLeftUp -= nextTokensError;
    actualSeqLength -= nextTokensError;

    // 若出现下方行无效，需要重新计算actualseqlen
    actualSeqLength -= preTokensError;
}

int64_t PromptFlashAttentionTilingV2::GetCalcBlockNumsOneHead(int64_t actualSeqLength, int64_t actualSeqLengthKV,
    uint32_t sOuterSize, uint32_t sInnerSize, int64_t preTokensLeftUp, int64_t nextTokensLeftUp, bool isAttenMaskUsed) const
{
    if (!isAttenMaskUsed) {
        int64_t outerBlockNums = (actualSeqLength + sOuterSize - 1) / sOuterSize;
        int64_t innerBlockNums = (actualSeqLengthKV + sInnerSize - 1) / sInnerSize + (actualSharedPrefixLen + sInnerSize - 1) / sInnerSize; 
        int64_t toCalcBlockNums = innerBlockNums * outerBlockNums;
        return toCalcBlockNums;
    } else {
        int64_t innerBlockNums = (actualSeqLengthKV + static_cast<int64_t>(sInnerSize) - 1) /
            static_cast<int64_t>(sInnerSize);
        int64_t blockSeqLengthKV = innerBlockNums * static_cast<int64_t>(sInnerSize);
        int64_t outerBlockNums = (actualSeqLength + static_cast<int64_t>(sOuterSize) - 1) /
            static_cast<int64_t>(sOuterSize);
        int64_t blockSeqLength = outerBlockNums * static_cast<int64_t>(sOuterSize);
        int64_t toCalcBlockNums = innerBlockNums * outerBlockNums;
        // 必须满足pretoken + nexttoken > 0，否则会减出小于0的块数，这里需要去除prefix影响
        toCalcBlockNums -= GetCutBlockNums(blockSeqLengthKV, blockSeqLength, static_cast<int64_t>(sInnerSize),
            static_cast<int64_t>(sOuterSize), nextTokensLeftUp - actualSharedPrefixLen);
        toCalcBlockNums -= GetCutBlockNums(blockSeqLengthKV, blockSeqLength, static_cast<int64_t>(sInnerSize),
            static_cast<int64_t>(sOuterSize), blockSeqLengthKV - blockSeqLength + preTokensLeftUp + actualSharedPrefixLen);

        // prefix部分单独计算
        int64_t innerBlockNumsPrefix = (actualSharedPrefixLen + static_cast<int64_t>(sInnerSize) - 1) /
            static_cast<int64_t>(sInnerSize);
        int64_t blockSharedPrefix = innerBlockNumsPrefix * static_cast<int64_t>(sInnerSize);
        toCalcBlockNums += innerBlockNumsPrefix * outerBlockNums;
        toCalcBlockNums -= GetCutBlockNums(blockSharedPrefix, blockSeqLength, static_cast<int64_t>(sInnerSize),
            static_cast<int64_t>(sOuterSize), nextTokensLeftUp);
        toCalcBlockNums -= GetCutBlockNums(blockSharedPrefix, blockSeqLength, static_cast<int64_t>(sInnerSize),
            static_cast<int64_t>(sOuterSize), blockSharedPrefix - blockSeqLength + preTokensLeftUp);
 
        return toCalcBlockNums;
    }
}

void PromptFlashAttentionTilingV2::ComputeSplitNBSeq(PromptFlashAttentionTilingDataV2& tilingData, uint32_t batchSize,
    const size_t tilingElementArrayLen, std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV,
    uint32_t sOuterSize, uint32_t sInnerSize, double coreWightTarget, uint32_t& curCore) 
{
    PromptAttentionBaseParams* baseParams = &tilingData.promptAttentionBaseParams;
    std::vector<uint32_t> coreSposEnd(tilingElementArrayLen, 0U);
    std::vector<uint32_t> coreSposStart(tilingElementArrayLen, 0U);
    std::vector<uint32_t> coreSidEnd(tilingElementArrayLen, 0U);
    std::vector<uint32_t> coreSidStart(tilingElementArrayLen, 0U);
    std::vector<uint32_t> coreNidEnd(tilingElementArrayLen, 0U);
    std::vector<uint32_t> coreNidStart(tilingElementArrayLen, 0U);
    std::vector<int64_t> sparseStartIdx(tilingElementArrayLen, 0L);
    std::vector<uint32_t> bnStartIdx(tilingElementArrayLen, 0U);
    std::vector<int64_t> gS1StartIdx(tilingElementArrayLen, 0L);
    // Temporary algorithm to be optimized
    int64_t curWight = 0;
    curCore = 0;
    uint32_t tmpCoreNidEnd = 0; // actual seq为0时不分配核
    uint32_t tmpCoreSidEnd = 0;
    uint32_t tmpCoreSposEnd = 0;
    int64_t innerBlockNumsPrefix = (actualSharedPrefixLen + sInnerSize - 1) / sInnerSize;
    for (uint32_t sIdx = 0; sIdx < batchSize; sIdx++) {
        for (uint32_t headNum = 0; headNum < baseParams->get_headNumSize(); headNum++) {
            // 针对行无效情况修正actualseqlen
            int64_t preTokensLeftUp = 0;
            int64_t nextTokensLeftUp = 0;
            GetPreNextTokensLeftUp(tilingData, actualSeqLengths[sIdx], actualSeqLengthsKV[sIdx] + actualSharedPrefixLen,
                preTokensLeftUp, nextTokensLeftUp);
            int64_t actualSeqLength = actualSeqLengths[sIdx];
            int64_t actualSeqLengthKV = actualSeqLengthsKV[sIdx];
            FixParamWithRowInvalid(actualSeqLength, actualSeqLengthKV + actualSharedPrefixLen,
                preTokensLeftUp, nextTokensLeftUp);

            int64_t outerBlockNums = (actualSeqLength + sOuterSize - 1) / sOuterSize;
            int64_t innerBlockNums = (actualSeqLengthKV + sInnerSize - 1) / sInnerSize;
            for (uint32_t sOuterIndex = 0; sOuterIndex < outerBlockNums; sOuterIndex++) {
                int64_t dif = static_cast<int64_t>(coreWightTarget * double(curCore + 1)) - curWight;
                // 非prefix部分计算，去除prefix影响
                int64_t preTokensNoPrefix = preTokensLeftUp + actualSharedPrefixLen;
                int64_t nextTokensNoPrefix = nextTokensLeftUp - actualSharedPrefixLen;
                int64_t sInnerIndexStart = -(preTokensNoPrefix > 0 ? (preTokensNoPrefix + static_cast<int64_t>(sInnerSize) - 1) /
                    static_cast<int64_t>(sInnerSize) : preTokensNoPrefix / static_cast<int64_t>(sInnerSize));
                int64_t sInnerIndexEnd = nextTokensNoPrefix > 0 ? (nextTokensNoPrefix + static_cast<int64_t>(sInnerSize) - 1) /
                    static_cast<int64_t>(sInnerSize) : nextTokensNoPrefix / static_cast<int64_t>(sInnerSize);

                // prefix部分单独计算
                int64_t sInnerIndexStartPrefix = -(preTokensLeftUp > 0 ? (preTokensLeftUp + static_cast<int64_t>(sInnerSize) - 1) /
                    static_cast<int64_t>(sInnerSize) : preTokensLeftUp / static_cast<int64_t>(sInnerSize));
                int64_t sInnerIndexEndPrefix = nextTokensLeftUp > 0 ? (nextTokensLeftUp + static_cast<int64_t>(sInnerSize) - 1) /
                    static_cast<int64_t>(sInnerSize) : nextTokensLeftUp / static_cast<int64_t>(sInnerSize);

                // 当前这一行有多少基本块需要计算
                int64_t actualInnerBlockNums = GetActualInnerBlockNums(sInnerIndexStart, sInnerIndexEnd, innerBlockNums) +
                    GetActualInnerBlockNums(sInnerIndexStartPrefix, sInnerIndexEndPrefix, innerBlockNumsPrefix);
                if (actualInnerBlockNums - dif > dif && !(tmpCoreNidEnd == 0 && tmpCoreSidEnd == 0 && tmpCoreSposEnd == 0)) {
                    coreNidEnd[curCore] = tmpCoreNidEnd;
                    coreSidEnd[curCore] = tmpCoreSidEnd;
                    coreSposEnd[curCore] = tmpCoreSposEnd;
                    curCore += 1;
                    coreNidStart[curCore] = headNum;
                    coreSidStart[curCore] = sIdx;
                    coreSposStart[curCore] = sOuterIndex;
                    bnStartIdx[curCore] = sIdx * baseParams->get_headNumSize() + headNum;
                    gS1StartIdx[curCore] = sOuterIndex;
                }
                tmpCoreNidEnd = headNum + 1;
                tmpCoreSidEnd = sIdx + 1;
                tmpCoreSposEnd = sOuterIndex + 1;

                curWight += actualInnerBlockNums;
                preTokensLeftUp -= sOuterSize;
                nextTokensLeftUp += sOuterSize;
            }
        }
    }
    coreNidEnd[curCore] = tmpCoreNidEnd;
    coreSidEnd[curCore] = tmpCoreSidEnd;
    coreSposEnd[curCore] = tmpCoreSposEnd;
    bnStartIdx[curCore + 1] = batchSize * baseParams->get_headNumSize();
    gS1StartIdx[curCore + 1] = tmpCoreSposEnd;

    PromptAttentionSeqParams* seqParams = &tilingData.promptAttentionSeqParams;
    seqParams->set_CoreHeadNumTail(coreNidStart.data());
    seqParams->set_actualS1(coreNidEnd.data());
    seqParams->set_actualCoreNums(coreSidStart.data());
    seqParams->set_singleCoreHeadNumSize(coreSidEnd.data());
    seqParams->set_coreSeqPosStart(coreSposStart.data());
    seqParams->set_coreSeqPosEnd(coreSposEnd.data());
    faTilingAdapter.multiCoreParamsRegbase.set_bnStartIdx(bnStartIdx.data());
    faTilingAdapter.multiCoreParamsRegbase.set_sparseStartIdx(gS1StartIdx.data());
}

void PromptFlashAttentionTilingV2::SetMultiCoreParamsRegbase(int64_t totalSize, int64_t actualUsedCoreNum)
{
    faTilingAdapter.multiCoreParamsRegbase.set_coreNum(static_cast<int32_t>(actualUsedCoreNum));
    faTilingAdapter.multiCoreParamsRegbase.set_totalSize(totalSize);
    faTilingAdapter.multiCoreParamsRegbase.set_splitFactorSize(CeilDivision(totalSize, actualUsedCoreNum));
    faTilingAdapter.multiCoreParamsRegbase.set_splitFactorTailSize(CalcTailSize(totalSize, faTilingAdapter.multiCoreParamsRegbase.get_splitFactorSize()));
}

void PromptFlashAttentionTilingV2::PromptFlashAttentionSplitNBSeq(PromptFlashAttentionTilingDataV2& tilingData,
    std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV, bool isAttenMaskUsed) 
{
    PromptAttentionBaseParams* baseParams = &tilingData.promptAttentionBaseParams;
    PromptAttentionSingleCoreParams* singleCoreParams = &tilingData.promptAttentionSingleCoreParams;
    uint32_t curCoreNum = coreNum;
    uint32_t batchSize = baseParams->get_dimNumOfseq();
    uint32_t sOuterSize = singleCoreParams->get_singleProcessSOuterSize();
    uint32_t sInnerSize = singleCoreParams->get_singleProcessSInnerSize();

    if (splitCoreMode == SplitCoreMode::SPLIT_NBS_CUBE) { // From the perspective of cube
        sOuterSize = sOuterSize * CV_RATIO;
        curCoreNum = curCoreNum / CV_RATIO;
    }

    int64_t totalBlockNumsOneHead = 0; // The calculation amount of all sequences for a single head

    uint32_t multiSmaxsInnerLoopTimes = 0U;
    for (uint32_t sIdx = 0; sIdx < batchSize; sIdx++) {
        int64_t actualSeqLengthsTmp = actualSeqLengths[sIdx]; // 用于存放减去行无效后，真实的actseqlen
        int64_t preTokensLeftUp = 0;
        int64_t nextTokensLeftUp = 0;
        GetPreNextTokensLeftUp(tilingData, actualSeqLengths[sIdx], actualSeqLengthsKV[sIdx] + actualSharedPrefixLen,
            preTokensLeftUp, nextTokensLeftUp);

        // 计算各sparse mode情况下，减去行无效后真实的actseqlen
        FixParamWithRowInvalid(actualSeqLengthsTmp, actualSeqLengthsKV[sIdx] + actualSharedPrefixLen, preTokensLeftUp, nextTokensLeftUp);

        // sinner方向块数，prefix和origin是分开切的。
        uint32_t sInnerLoopTimes = (actualSeqLengthsKV[sIdx] + sInnerSize - 1) / sInnerSize +
            (actualSharedPrefixLen + sInnerSize - 1) / sInnerSize;
        multiSmaxsInnerLoopTimes = std::max(multiSmaxsInnerLoopTimes, sInnerLoopTimes);

        totalBlockNumsOneHead += GetCalcBlockNumsOneHead(actualSeqLengthsTmp, actualSeqLengthsKV[sIdx], sOuterSize,
            sInnerSize, preTokensLeftUp, nextTokensLeftUp, isAttenMaskUsed);
    }
    singleCoreParams->set_multiSmaxsInnerLoopTimes(multiSmaxsInnerLoopTimes);

    // Amount of computation per core
    double coreWightTarget = (double(totalBlockNumsOneHead * baseParams->get_headNumSize()) / double(curCoreNum));

    int64_t s1OuterSize = (baseParams->get_seqSize() + sOuterSize - 1) / sOuterSize;
    faTilingAdapter.multiCoreParamsRegbase.set_s1OuterSize(s1OuterSize);

    // The tiling structure element needs to have a length greater than or equal to the length specified
    // by TILING_DATA_FIELD_DEF_ARR. If the tiling structure definition specifies a length of 64,
    // the vector definition needs to compare its size with coreNum and take the larger value
    const size_t tilingElementArrayLen = (static_cast<size_t>(curCoreNum) > 64UL) ? static_cast<size_t>(curCoreNum) : 64UL;
    uint32_t curIndx = 0;
    ComputeSplitNBSeq(tilingData, batchSize, tilingElementArrayLen, actualSeqLengths, actualSeqLengthsKV, sOuterSize,
        sInnerSize, coreWightTarget, curIndx);

    uint32_t actualCoreNums = (splitCoreMode == SplitCoreMode::SPLIT_NBS_CUBE) ? (curIndx + 1) * CV_RATIO : curIndx + 1;
    singleCoreParams->set_actualCoreNums(actualCoreNums);
    int64_t sinnerBlocknum = (baseParams->get_seqInnerSize() + sInnerSize - 1) / sInnerSize;
    SetMultiCoreParamsRegbase((totalBlockNumsOneHead / sinnerBlocknum) * baseParams->get_headNumSize(), static_cast<int64_t>((curIndx + 1)));
}

void PromptFlashAttentionTilingV2::PromptFlashAttentionInitSoftmaxLseOutputSplit(int64_t totalSize,
    PromptFlashAttentionTilingDataV2 &tilingData) 
{
    PromptAttentionInitOutputParams *initParams = &tilingData.promptAttentionInitOutputParams;
    initParams->set_totalSoftMaxLseOutputSize(totalSize);
}

void PromptFlashAttentionTilingV2::UpdateTilingKeyLayoutType() 
{
    if (inputLayout == InputLayout::BNSD) {
        inOutLayoutType = InOutLayoutType_BNSD_BNSD;
    } else if (inputLayout == InputLayout::TND) {
        inOutLayoutType = InOutLayoutType_TND_TND;
    } else if (inputLayout == InputLayout::NTD) {
        inOutLayoutType = InOutLayoutType_NTD_NTD;
    } else if (inputLayout == InputLayout::BSH || inputLayout == InputLayout::BSND) {
        inOutLayoutType = InOutLayoutType_BSH_BSH;
    }
}

void PromptFlashAttentionTilingV2::UpdateTilingKeyConfig(ContextParamsForPFATiling& contextKeyParams, PromptFlashAttentionTilingDataV2 &tilingData) 
{
    auto sInner = tilingData.promptAttentionSingleCoreParams.get_singleProcessSInnerSize();
    auto sOuter = tilingData.promptAttentionSingleCoreParams.get_singleProcessSOuterSize() * 2;    
    auto dSize = tilingData.promptAttentionBaseParams.get_qkHeadSize();
    auto dVsize = tilingData.promptAttentionBaseParams.get_vHeadSize();
    if (dSize <= 64) dSize = 64; // 64: adjust qk headsize
    else if (dSize <= 128) dSize = 128; // 128: adjust qk headsize
    else if (dSize <= 256) dSize = 256; // 256: adjust qk headsize
    else if (dSize <= 512) dSize = 512; // 512: adjust qk headsize
    else if (dSize <= 576) dSize = 576; // 576: adjust qk headsize

    if (dVsize <= 64) dVsize = 64; // 64: adjust v headsize
    else if (dVsize <= 128) dVsize = 128; // 128: adjust v headsize
    else if (dVsize <= 256) dVsize = 256; // 256: adjust v headsize
    else if (dVsize <= 512) dVsize = 512; // 512: adjust v headsize

    if (sOuter == 64 && sInner == 64 && dSize == 256 && dVsize == 256) { // 64, 256: head size
        config = Config_S1Aligned64_S2Aligned64_DAligned256_DVAligned256;
    } else if (sOuter == 64 && sInner == 64 && dSize == 512 && dVsize == 512) { // 64, 512: head size
        config = Config_S1Aligned64_S2Aligned64_DAligned512_DVAligned512;
    } else if (sOuter == 64 && sInner == 256 && dSize == 64 && dVsize == 64) { // 64, 256: head size
        config = Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64;
    } else if (sOuter == 64 && sInner == 256 && dSize == 128 && dVsize == 128) { // 64, 256, 128: head size
        config = Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128;
    } else if (sOuter == 128 && sInner == 128 && dSize == 64 && dVsize == 64) { // 64, 128: head size
        config = Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64;
    } else if (sOuter == 128 && sInner == 128 && dSize == 128 && dVsize == 128) { // 128: head size
        if (enablePFARope) {
            config = Config_S1Aligned128_S2Aligned128_DAligned192_DVAligned128;
        } else {
            config = Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128;
        }        
    } else if (sOuter == 128 && sInner == 128 && dSize == 192 && dVsize == 128) { // 192, 128: head size
        config = Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128;
    } else if (sOuter == 128 && sInner == 128 && dSize == 256 && dVsize == 128) { // 128, 256: head size
        config = Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128;
    } else if (sOuter == 128 && sInner == 128 && dSize == 256 && dVsize == 256) { // 256, 128: head size
        config = Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256;
    } else if (sOuter == 128 && sInner == 128 && dSize == 512 && dVsize == 512) { // 512, 128: head size
        config = Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512;
    } else if (sOuter == 128 && sInner == 256 && dSize == 64 && dVsize == 64) { // 64, 256, 128: head size
        config = Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64;
    } else if (sOuter == 64 && sInner == 128 && dSize == 576 && dVsize == 512) { // 64, 128, 576, 512: head size
        config = Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512;
    } else if (sOuter == 64 && sInner == 256 && dSize == 256 && dVsize == 256) { // 64, 256: head size
        config = Config_S1Aligned64_S2Aligned256_DAligned256_DVAligned256;
    } else if (sOuter == 128 && sInner == 256 && dSize == 128 && dVsize == 128) { // 256, 128: head size
        config = Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128;
    } else if (sOuter == 128 && sInner == 128 && dSize == 128 && dVsize == 64) { // 64, 128: head size
        config = Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned64; //qkvd不等长
    } else if (sOuter == 128 && sInner == 128 && dSize == 64 && dVsize == 128) { // 64, 128: head size
        config = Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned128;//qkvd不等长
    } else if (sOuter == 64 && sInner == 256 && dSize == 128 && dVsize == 64) { // 64, 128, 256: head size
        config = Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned64;//qkvd不等长
    } else if (sOuter == 64 && sInner == 256 && dSize == 64 && dVsize == 128) { // 64, 128, 256: head size
        config = Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned128;//qkvd不等长
    } else {
        OP_LOGE(contextKeyParams.opName, "The combination of parameters S1, S2, D, DV is not supported!");
    }
    OP_LOGI(contextKeyParams.opName, "sInner is %d, sOuter is %d, dSize is %d, dVsize is %d, config is %d.", sInner, sOuter, dSize, dVsize, config);
}

void PromptFlashAttentionTilingV2::UpdateTilingKeyPseMode() 
{
    if (usePseShift == 0U) {
        pseMode = PSE_MODE_PSE_NONE_TYPE;
    } else if (usePseShift == 1U) {
        pseMode = PSE_MODE_PSE_OUTER_MUL_ADD_TYPE;
    } else if (usePseShift == 2U) {
        pseMode = PSE_MODE_PSE_INNER_MUL_ADD_TYPE;
    } else if (usePseShift == 3U) {
        pseMode = PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE;
    }
}

void PromptFlashAttentionTilingV2::UpdateTilingKeyQuantMode(ge::DataType inputDataType) 
{
    if (inputDataType == ge::DT_FLOAT16 || inputDataType == ge::DT_BF16) {
        quantMode = NoQuantMode;
    } else if (enablePerblockQuant) {
        quantMode = FULLQUANT_MODE_QKV_PERBLOCK;
    } else if (enableIFAMLAFullQuant) {
        quantMode = FULLQUANT_MODE_Q_PER_TOKEN_HEAD_KV_PER_TENSOR;
    } else {
        quantMode = FullQuantMode;
    }
}

void PromptFlashAttentionTilingV2::UpdateTilingKeyAttenMask(ge::DataType inputDataType) 
{
    // perblock采用新模板
    if (enablePertensorQuant && (inputDataType == ge::DT_INT8)) {
        hasAttenMask = 0;
        return;
    }
    if (enableMask) {
        hasAttenMask = true;
    } else {
        hasAttenMask = false;
    }
}

void PromptFlashAttentionTilingV2::UpdateTilingKeyHasRope(ge::DataType inputDataType) 
{
    if (enablePertensorQuant && (inputDataType == ge::DT_INT8)) {
        hasRope = 0;
        return;
    }
    if (enablePFARope || enableIFAMLAFullQuant) {
        hasRope = true;
    } else {
        hasRope = false;
    }
}

void PromptFlashAttentionTilingV2::UpdateTilingKeyIsPa(ge::DataType inputDataType) 
{
    if (enablePertensorQuant && (inputDataType == ge::DT_INT8)) {
        isPa = 0;
        return;
    }
    if (enablePA) {
        isPa = true;
    } else {
        isPa = false;
    }
}

void PromptFlashAttentionTilingV2::UpdateTilingKeyIsFd(ge::DataType inputDataType) 
{
    if (isKVHasPrefix) {
        isFd = false;
        return;
    }
    if (enablePertensorQuant && (inputDataType == ge::DT_INT8)) {
        isFd = 0;
        return;
    }
    if (enableFlashDecode) {
        isFd = true;
    } else {
        isFd = false;
    }
}

void PromptFlashAttentionTilingV2::UpdateTilingKeyEmptyTensor() 
{
    emptyTensor = 0;
}

void PromptFlashAttentionTilingV2::UpdateTilingKeyEnableKVPrefix() 
{
    enableKVPrefix = isKVHasPrefix;
}

void PromptFlashAttentionTilingV2::UpdateTilingKeySplitCoreMode() 
{
    enableS1OutSplit = false;
}

bool PromptFlashAttentionTilingV2::TilingGetTilingKeyAttentionAscendC(ContextParamsForPFATiling& contextKeyParams, PromptFlashAttentionTilingDataV2 &tilingData) 
{
    auto inputDataType = contextKeyParams.inputDataType; // input q
    auto attenMaskElemType = contextKeyParams.maskDataType;
    auto outputDataType = contextKeyParams.outputDataType; // output tensor
    tilingData.promptAttentionBaseParams.set_attenMaskElemType(attenMaskElemType);
    UpdateTilingKeyLayoutType();
    UpdateTilingKeyConfig(contextKeyParams, tilingData);
    UpdateTilingKeyPseMode();
    UpdateTilingKeyQuantMode(inputDataType);
    UpdateTilingKeyAttenMask(inputDataType);
    UpdateTilingKeyHasRope(inputDataType);
    UpdateTilingKeyIsPa(inputDataType);
    UpdateTilingKeyIsFd(inputDataType);
    UpdateTilingKeyEmptyTensor();
    UpdateTilingKeyEnableKVPrefix();
    UpdateTilingKeySplitCoreMode();
    return true;
}

size_t PromptFlashAttentionTilingV2::GetPFAWorkSpaceSize(PromptFlashAttentionTilingDataV2& tilingData) 
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "platformInfoPtr is null!"),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t curWorkspaceSize = 0;
    uint64_t maxSpmSize = tilingData.promptAttentionTensorSizeRect.get_spmTmpSize();
    int64_t mm1ResSize = tilingData.promptAttentionSingleCoreParams.get_singleProcessSOuterSize() *
        tilingData.promptAttentionSingleCoreParams.get_singleProcessSInnerSize();
    int64_t mm2ResSize = tilingData.promptAttentionSingleCoreParams.get_singleProcessSOuterSize() *
        tilingData.promptAttentionBaseParams.get_headSize();

    // 2:use 2mm ub
    if (!faRunFlag_) {
        curWorkspaceSize = sysWorkspaceSize + coreNum * softmaxDataTypeSize * (maxSpmSize + mm1ResSize * MM2_UB_NUM + mm2ResSize * MM2_UB_NUM);
    } else {
        size_t accumOutSize = 0;
        size_t logSumExpSize = 0;
        if (isMaxWorkspace) { // 计算maxWorkSpaceSize时默认开启FD且使用最大核数进行归约
            uint64_t headDimAlign = AlignUp(tilingData.promptAttentionBaseParams.get_vHeadSize(), BYTE_BLOCK);
            accumOutSize = aicNum * headDimAlign * sizeof(float);
            logSumExpSize = aicNum * BYTE_BLOCK * 2; // 2: Each AI core needs 2 byte blocks for storing log-sum-exp intermediate results (fixed multiplier factor)
        } else if (enableFlashDecode) {
            auto batchSize = tilingData.promptAttentionBaseParams.get_batchSize();
            auto headNumSize = tilingData.promptAttentionBaseParams.get_headNumSize();
            uint64_t headDimAlign = AlignUp(tilingData.promptAttentionBaseParams.get_vHeadSize(), BYTE_BLOCK);
            uint32_t kvSplitPart = faTilingAdapter.inputParamsRegbase.get_kvSplitPart();
            accumOutSize = batchSize * gSize * headNumSize * kvSplitPart * headDimAlign * sizeof(float);
            logSumExpSize = batchSize * gSize * headNumSize * kvSplitPart * BYTE_BLOCK * 2; // 2: fixed multiplier factor
        }

        int64_t bmm2Bytes = 0;
        int64_t vec2Bytes = 0;
        uint32_t dSize = tilingData.promptAttentionBaseParams.get_vHeadSize();
        uint32_t dVBasicBlock = 0;
        if (dSize <= 64) { // 64: headsize is 64
            dVBasicBlock = 64;
        } else if (dSize <= 128) { // 128: headsize is 128
            dVBasicBlock = 128;
        } else if (dSize <= 256) { // 256: headsize is 256
            dVBasicBlock = 256;
        } else if (dSize <= 512) { // 512: headsize is 512
            dVBasicBlock = 512;
        }
        int64_t bmm2ResBlockSize = dVBasicBlock;
        if (dVBasicBlock > 256) { // 256: headsize is 256
            bmm2ResBlockSize = 512L; // 512: headsize is 512
        }
        uint32_t s1BasicBlock = tilingData.promptAttentionSingleCoreParams.get_singleProcessSOuterSize() * 2;
        uint32_t calcTypeSize = softmaxDataTypeSize;
        if ((!enableDN && dSize > 128) || // 128: headsize is 128
            (enableDN && dSize > 192)) {  // 192: headsize is 192
            bmm2Bytes = s1BasicBlock * bmm2ResBlockSize * calcTypeSize;
            if (dVBasicBlock > 256) { // 256: headsize is 256
                vec2Bytes = s1BasicBlock * dVBasicBlock * calcTypeSize;
            }
        }
        bmm2Bytes = AlignUp(bmm2Bytes, GM_ALIGN);
        vec2Bytes = AlignUp(vec2Bytes, GM_ALIGN);
        // 3 for WorkspaceSize calculation
        curWorkspaceSize = static_cast<size_t>((bmm2Bytes + vec2Bytes) * 3 *
            coreNum) + sysWorkspaceSize + accumOutSize + logSumExpSize;
    }
    if (enablePA) {
        // 2 bmm, db, ensure alignment of each structure 64B, dcci cacheline needs
        curWorkspaceSize += static_cast<uint64_t>(coreNum) * 2 * 2 * 64;
    }
    return curWorkspaceSize;
}

ge::graphStatus PromptFlashAttentionTilingV2::SetPlatMemoryInfo(ContextParamsForPFATiling& contextKeyParams) 
{
    // In subsequent version, contextKeyParams will be written as a member variable of the class.
    auto compileInfoPtr = contextKeyParams.compileInfoPtr;
    contextKeyParamsPtr = &contextKeyParams;

    OP_CHECK_IF(compileInfoPtr == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "compileInfoPtr is null"),
        return ge::GRAPH_FAILED);

    coreNum = compileInfoPtr->aivNum;
    OP_CHECK_IF(coreNum == 0,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "coreNum is 0"),
        return ge::GRAPH_FAILED);
    aivNum = compileInfoPtr->aivNum;
    OP_CHECK_IF(aivNum == 0,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "aivNum is 0"),
        return ge::GRAPH_FAILED);
    aicNum = compileInfoPtr->aicNum;
    OP_CHECK_IF(aicNum == 0,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "aicNum is 0"),
        return ge::GRAPH_FAILED);
    ascendPlatformInfo.socVersion = compileInfoPtr->socShortName;
    ascendPlatformInfo.l1Size = compileInfoPtr->l1Size;
    ascendPlatformInfo.l0CSize = compileInfoPtr->l0CSize;
    ascendPlatformInfo.l0ASize = compileInfoPtr->l0ASize;
    ascendPlatformInfo.l0BSize = compileInfoPtr->l0BSize;
    ascendPlatformInfo.ubSize = compileInfoPtr->ubSize;
    OP_LOGI(contextKeyParams.opName, "ascendPlatformInfo:aivNum = %u, aicNum = %u, l1Size = %lu, l0CSize = %lu,"
        "l0ASize = %lu, l0BSize = %lu, ubSize = %lu!", aivNum, aicNum, compileInfoPtr->l1Size, compileInfoPtr->l0CSize,
        compileInfoPtr->l0ASize, compileInfoPtr->l0BSize, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTilingV2::SetAttributeInfo(ContextParamsForPFATiling& contextKeyParams) 
{
    // antiquant check, temporary solution
    if (contextKeyParams.hasKeyAntiquantScale || contextKeyParams.hasValueAntiquantScale) {
        enableKVAntiquant = false;
    } else {
        if (contextKeyParams.inputDataType == ge::DT_FLOAT16 && contextKeyParams.kDataType == ge::DT_INT8) {
            enableKVAntiquant = true;
        }
        OP_CHECK_IF(contextKeyParams.inputDataType == ge::DT_BF16 && contextKeyParams.kDataType == ge::DT_INT8,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "keyAntiquantScale and valueAntiquantScale should not be null, "
            "when data type of query is bf16 and data type of key/value is int8"),
            return ge::GRAPH_FAILED);
    }
    std::tuple<ge::DataType, ge::DataType, ge::DataType> inOutDtypeTuple = {
        contextKeyParams.inputDataType, contextKeyParams.kDataType, contextKeyParams.outputDataType};
    OP_CHECK_IF(!VecContains(inOutDtypeSupported, inOutDtypeTuple),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "Query dtype(%s), key/value dtype(%s), attentionOut dype(%s) is not currently supported.",
                    GetPfaDataTypeStr(contextKeyParams.inputDataType).c_str(),
                    GetPfaDataTypeStr(contextKeyParams.kDataType).c_str(),
                    GetPfaDataTypeStr(contextKeyParams.outputDataType).c_str()),
                return ge::GRAPH_FAILED);

    // prefix check
    isKVHasPrefix = (contextKeyParams.keySharedPrefix != nullptr) &&
        (contextKeyParams.valueSharedPrefix != nullptr) ? true : false;
    OP_CHECK_IF((!isKVHasPrefix && ((contextKeyParams.keySharedPrefix != nullptr) ||
        contextKeyParams.valueSharedPrefix != nullptr)), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "when system prefix is used, key_shared_prefix and value_shared_prefix are required!"),
        return ge::GRAPH_FAILED);

    // actSeqLen check
    const gert::Tensor* actSeqLenData = contextKeyParams.actualSequenceLengthQ;
    const gert::Tensor* actSeqLenDataKV = contextKeyParams.actualSequenceLengthKV;
    actSeqLenDims = (actSeqLenData != nullptr) ? actSeqLenData->GetShapeSize() : 0;
    actSeqLenKVDims = (actSeqLenDataKV != nullptr) ? actSeqLenDataKV->GetShapeSize() : 0;
    enableActSeqLen = !((actSeqLenDims == 0) || (actSeqLenData == nullptr) ||
        (actSeqLenData->GetData<int64_t>() == nullptr));
    enableActSeqLenKV = !((actSeqLenKVDims == 0) || (actSeqLenDataKV == nullptr) ||
        (actSeqLenDataKV->GetData<int64_t>() == nullptr));
    
    //per-tensor or per-block check
    if (contextKeyParams.inputDataType == ge::DT_INT8 && !(contextKeyParams.queryRopeInputShape != nullptr && contextKeyParams.keyRopeInputShape != nullptr)) {
        enablePertensorQuant = true;
    }

    // PA check
    if (contextKeyParams.blockTable != nullptr) {
        OP_CHECK_IF(contextKeyParams.blockSize == nullptr, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "blockSize can't be null when PA enable"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(enablePertensorQuant == true, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "Query dataType can't be INT8 when PA enable, except when mlafullquant int8 is used."),
            return ge::GRAPH_FAILED);
        enablePA = true;
    }

    // Pse
    const gert::StorageShape* pseShiftShape = contextKeyParams.pseShiftShape;
    enablePseShift = (contextKeyParams.pseShift != nullptr) && (pseShiftShape != nullptr);

    if (contextKeyParams.pseType != nullptr) {
        pseType = *contextKeyParams.pseType;
        OP_CHECK_IF((pseType != 0) && (pseType != 2) && (pseType != 3), OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "PseType(%ld) is not support, pseType must be 0/2/3.", pseType),
            return ge::GRAPH_FAILED);
        faTilingAdapter.inputParamsRegbase.set_pseType(pseType);
    }
    if (pseType == PSE_TYPE_2_TILING_V2 || pseType == PSE_TYPE_3_TILING_V2) {
        enableAlibiPse = true;
        enablePseShift = false;
    }

    // innerprecise, 910B defaults to high-performance.
    const int64_t* innerPrecisePtr = contextKeyParams.innerPrecisePtr;
    innerPrecise = innerPrecisePtr != nullptr ? static_cast<int32_t>(*innerPrecisePtr) : HIGH_PERFORMANCE;

    // mask check
    const gert::StorageShape* attenMaskShape = contextKeyParams.attentionMaskShape;
    enableMask = (contextKeyParams.attentionMask != nullptr) && (attenMaskShape != nullptr);

    // sparsemode check
    const int32_t* sparseMode = contextKeyParams.sparseMode;
    isDefaultSparseMode = (sparseMode == nullptr) || ((sparseMode != nullptr) && (*sparseMode == SPARSE_MODE_NO_MASK));

    // tensorlist check
    enableTensorList = contextKeyParams.isKvContinuous == 1 ? false : true;

    // LeftPadding check
    enableLeftPadding = ((contextKeyParams.queryPaddingSize != nullptr) || (contextKeyParams.kvPaddingSize != nullptr));
    if (enableLeftPadding) {
        needInit = 1;
    }

    // postQuant check
    if (contextKeyParams.outputDataType != ge::DT_BF16 && contextKeyParams.outputDataType != ge::DT_FLOAT16) {
        enablePostQuant = true;
    }

    // mla fullquant check
    if((contextKeyParams.queryRopeInputShape != nullptr && contextKeyParams.keyRopeInputShape != nullptr) &&
        ((contextKeyParams.inputDataType == ge::DT_FLOAT8_E4M3FN && contextKeyParams.kDataType == ge::DT_FLOAT8_E4M3FN &&
        contextKeyParams.vDataType == ge::DT_FLOAT8_E4M3FN) || (contextKeyParams.inputDataType == ge::DT_INT8 && 
        contextKeyParams.kDataType == ge::DT_INT8 && contextKeyParams.vDataType == ge::DT_INT8))) {
        enableIFAMLAFullQuant = true;
    }

    //attention sink
    enableLearnSink = contextKeyParams.hasLearnableSink;

    const int64_t *keyAntiquantMode = contextKeyParams.keyAntiquantMode;
    const int64_t *queryQuantMode = contextKeyParams.queryQuantMode;
    const int64_t *valueAntiquantMode = contextKeyParams.valueAntiquantMode;
    if (contextKeyParams.inputDataType == ge::DT_HIFLOAT8 || contextKeyParams.inputDataType == ge::DT_FLOAT8_E4M3FN) {
        if (*keyAntiquantMode == static_cast<int64_t>(AntiquantTypeEnum::PER_BLOCK) && *queryQuantMode == static_cast<int64_t>(AntiquantTypeEnum::PER_BLOCK)
            && *valueAntiquantMode == static_cast<int64_t>(AntiquantTypeEnum::PER_BLOCK)) { // 7: FP8 perblock quant
            enablePerblockQuant = true;
        } else {
            if (!enableIFAMLAFullQuant) {
                enablePertensorQuant = true;
            }
        }
    }
    if (enablePertensorQuant) {
        faRunFlag_ = false;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTilingV2::CheckTensorInvalid(const ContextParamsForPFATiling& contextKeyParams) const 
{
    if (!CheckNonEmptyShapeExceptions(contextKeyParams, contextKeyParams.queryInputShape, "query")) {
        return ge::GRAPH_FAILED;
    }
    if (!CheckNonEmptyShapeExceptions(contextKeyParams, contextKeyParams.keyInputShape, "key")) {
        return ge::GRAPH_FAILED;
    }
    if (!CheckNonEmptyShapeExceptions(contextKeyParams, contextKeyParams.valueInputShape, "value")) {
        return ge::GRAPH_FAILED;
    }
    if (!CheckNonEmptyShapeExceptions(contextKeyParams, contextKeyParams.outputShape, "output")) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool PromptFlashAttentionTilingV2::CheckAlibiPseShiftTypeAndShape(ContextParamsForPFATiling& contextKeyParams, uint32_t n)
{
    const gert::StorageShape* pseShape = contextKeyParams.pseShiftShape;
    OP_CHECK_IF(isQKVDDifferent,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Not support alibi pse when query and key headdim is not equal to value headdim."),
        return false);
    OP_CHECK_IF(enableIFAMLA || enablePFAMLA,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "MLA do not support pseShift."),
        return false);
    if (!CheckNonEmptyShapeExceptions(contextKeyParams, pseShape, "pseShift")) {
        return false;
    }
    auto &inputParams = faTilingAdapter.inputParamsRegbase;

    pseShiftElemType = contextKeyParams.pseShiftDataType;

    OP_CHECK_IF((pseShiftElemType != ge::DT_FLOAT),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "When pseType = 2/3, pse shift type must be float, but pse shift type = %s", GetPfaDataTypeStr(pseShiftElemType).c_str()),
        return false);

    // 0: (B,N2,G,S1,S2), 1: (B,N2,G,1,S2)
    PfaPseShapeType pseShapeType = PfaPseShapeType::PSE_B_N2_G_1_S2; 
    if (pseShape != nullptr && pseShape->GetStorageShape().GetDimNum() != 0) {
        auto &pseShapeDims = pseShape->GetStorageShape();
        int64_t pseDimNum = pseShapeDims.GetDimNum();
        int64_t pseBSize = 0;
        if (pseType == static_cast<int64_t>(PfaPseType::PSE_INNER_MUL_ADD_TYPE) ||
            pseType == static_cast<int64_t>(PfaPseType::PSE_INNER_MUL_ADD_SQRT_TYPE)) {
            OP_CHECK_IF(pseDimNum != SLOPE_N_DIM_NUM, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "The dim of pseShift(%ld) must be 1, when pseType = 2/3.", pseDimNum),
                return false);
            int64_t pseShiftN = pseShape->GetStorageShape().GetDim(0); // 0: The first dimension is N.
            OP_CHECK_IF(pseShiftN != n, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "The length of pseShift(%ld) must be equal to query head number, when pseType = 2/3.", pseShiftN),
                return false);
            if (pseDimNum == 1) {
                pseShapeType = PfaPseShapeType::PSE_1_N2_G_SLOPE;
                pseBSize = 1;
            }
        }
        inputParams.set_pseBSize(static_cast<uint32_t>(pseBSize));
    }

    inputParams.set_pseShapeType(static_cast<uint8_t>(pseShapeType));
    return true;
}

ge::graphStatus PromptFlashAttentionTilingV2::CheckSingleAttribute(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo,
    PFAShapeInfo& keyShapeInfo, PFAShapeInfo& valueShapeInfo, PFAShapeInfo& queryRopeShapeInfo, PromptFlashAttentionTilingDataV2& tilingData) 
{
    if (!CheckIO(contextKeyParams, queryShapeInfo, valueShapeInfo)) {
        OP_LOGE(contextKeyParams.opName, "Check query/ouput failed!");
        return ge::GRAPH_FAILED;
    }
    if (!CheckKV(contextKeyParams, keyShapeInfo, valueShapeInfo)) {
        OP_LOGE(contextKeyParams.opName, "Check key/value failed!");
        return ge::GRAPH_FAILED;
    }
    if (!CheckRope(contextKeyParams, queryShapeInfo, keyShapeInfo, queryRopeShapeInfo)) {
        OP_LOGE(contextKeyParams.opName, "Check queryRope/keyRope failed!");
        return ge::GRAPH_FAILED;
    }
    if (!CheckLayout(contextKeyParams)) {
        OP_LOGE(contextKeyParams.opName, "Check layout failed!");
        return ge::GRAPH_FAILED;
    }
    if (!CheckQueryAndKey(contextKeyParams, queryShapeInfo, keyShapeInfo, tilingData)) {
        OP_LOGE(contextKeyParams.opName, "Check query and key consistency failed!");
        return ge::GRAPH_FAILED;
    }
    if (!isMaxWorkspace && enableIFAMLA && (!CheckIFAMLA(contextKeyParams, queryShapeInfo))) {
        return ge::GRAPH_FAILED;
    }
    if (enableIFAMLAFullQuant && (!CheckMLAFullQuant(contextKeyParams, queryShapeInfo))) {
        return ge::GRAPH_FAILED;
    }
    // print shape info
    OP_LOGI(contextKeyParams.opName,
        "Tiling Info: Q_B is %u, KV_B is %u, Q_N is %u, KV_N is %u, Q_S is %u, KV_S is %u, H is %u, D is %u, "
        "headNumRatio = %u", queryShapeInfo.b, keyShapeInfo.b, queryShapeInfo.n, keyShapeInfo.n, queryShapeInfo.s,
        S2, queryShapeInfo.h, queryShapeInfo.d, tilingData.promptAttentionBaseParams.get_headNumRatio());

    // quant/dequant/antiquant
    if (!CheckQuant(contextKeyParams, queryShapeInfo, keyShapeInfo, valueShapeInfo)) {
        OP_LOGE(contextKeyParams.opName, "Check quant failed!");
        return ge::GRAPH_FAILED;
    }

    // prefix check
    if (!CheckPrefix(contextKeyParams, queryShapeInfo, keyShapeInfo, tilingData)) {
        OP_LOGE(contextKeyParams.opName, "Check prefix failed!");
        return ge::GRAPH_FAILED;
    }

    // actSeq check
    if (!isMaxWorkspace && !CheckActSeq(contextKeyParams, queryShapeInfo)) {
        OP_LOGE(contextKeyParams.opName, "Check actual sequence failed!");
        return ge::GRAPH_FAILED;
    }

    // actSeqLen check
    if (!isMaxWorkspace &&
        !CheckActSeqLen(contextKeyParams, queryShapeInfo, keyShapeInfo)){
        OP_LOGE(contextKeyParams.opName, "Check actual sequence length failed!");
        return ge::GRAPH_FAILED;
    }

    // PA check
    if (!enablePA) {
        if (enableTensorList) {
            S2 = contextKeyParams.maxKVs;
        } else {
            S2 = keyShapeInfo.s;
        }
    } else if (!CheckPATypeAndShape(contextKeyParams, queryShapeInfo, queryRopeShapeInfo, tilingData)) {
        return ge::GRAPH_FAILED;
    }

    // pse check
    if (enablePseShift) {
        if (!CheckPseShiftTypeAndShape(contextKeyParams, queryShapeInfo.b, queryShapeInfo.n, queryShapeInfo.s, S2 + actualSharedPrefixLen)) {
            return ge::GRAPH_FAILED;
        }
        usePseShift = 1;
    } else if (enableAlibiPse) {
        if (!CheckAlibiPseShiftTypeAndShape(contextKeyParams, queryShapeInfo.n)) {
            return ge::GRAPH_FAILED;
        }
        if (pseType == PSE_TYPE_2_TILING_V2) {
            usePseShift = static_cast<uint32_t>(PSE_TYPE_2_TILING_V2);
        } else if (pseType == PSE_TYPE_3_TILING_V2) {
            usePseShift = static_cast<uint32_t>(PSE_TYPE_3_TILING_V2);
        }
    }
    // innerPrecise check
    if (!CheckInnerPrecise(contextKeyParams, tilingData)) {
        return ge::GRAPH_FAILED;
    }

    // mask check
    const int32_t* sparseMode = contextKeyParams.sparseMode;
    if (enableIFA && (sparseMode != nullptr) && (*sparseMode == SPARSE_MODE_RIGHT_DOWN)) {
        enableMask = false; // qs等于1时，sparse3相当于全部有效没被mask
    }
    if (enableMask) {
        if (!CheckMaskTypeAndShape(contextKeyParams, tilingData)) {
            return ge::GRAPH_FAILED;
        }
        tilingData.promptAttentionBaseParams.set_useMask(1);
    } else {
        tilingData.promptAttentionBaseParams.set_useMask(0);
    }

    // sparseMode check
    if (!CheckSparseMode(contextKeyParams, queryShapeInfo.s)) {
        return ge::GRAPH_FAILED;
    }

    // attention sink check
    if (!CheckLearnSink(contextKeyParams, queryShapeInfo)) {
        OP_LOGE(contextKeyParams.opName, "Check sink failed!");
        return ge::GRAPH_FAILED;
    }
    
    return ge::GRAPH_SUCCESS;
}

bool PromptFlashAttentionTilingV2::CheckAlibiPseCrossover(ContextParamsForPFATiling& contextKeyParams) 
{
    if (!enableAlibiPse) {
        return true;
    }

    OP_CHECK_IF(enableLeftPadding,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When pseType = 2/3, left padding is not supported!"),
        return false);
    
    OP_CHECK_IF((enableIFAMLA || enablePFARope),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When pseType = 2/3, rope is not supported, pseType = %ld", pseType),
        return false);

    OP_CHECK_IF((enablePFAMLA),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, 
        "When pseType = 2/3, query, key and value D should be the same, input query's D ane key's D = 192, but value's D = 128."),
        return false);

    return true;
}

ge::graphStatus PromptFlashAttentionTilingV2::CheckCrossoverAttribute(ContextParamsForPFATiling& contextKeyParams,
    PFAShapeInfo& queryShapeInfo, PromptFlashAttentionTilingDataV2& tilingData) 
{
    // PA and prefix,antiquant,actseqlenKV features crossover
    if (!CheckPACrossover(contextKeyParams, queryShapeInfo)) {
        return ge::GRAPH_FAILED;
    }

    // Mask and innerprecise,sparseMode feature crossover
    if (!CheckMaskCrossover(contextKeyParams, queryShapeInfo, tilingData)) {
        return ge::GRAPH_FAILED;
    }

    // TNDLayout and left padding, tensorlist, pse feature crossover
    if (!CheckTNDLayoutCrossover(contextKeyParams)) {
        return ge::GRAPH_FAILED;
    }

    if (!CheckNTDLayoutCrossover(contextKeyParams, queryShapeInfo)) {
        return ge::GRAPH_FAILED;
    }

    if (!CheckTransposeLayoutCrossover(contextKeyParams, queryShapeInfo)) {
        return ge::GRAPH_FAILED;
    }

    if (!CheckAlibiPseCrossover(contextKeyParams)) {
        return ge::GRAPH_FAILED;
    }

    if (!CheckPerblockCrossover(contextKeyParams, tilingData)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTilingV2::AdjustTilingData(ContextParamsForPFATiling& contextKeyParams,
    PromptFlashAttentionTilingDataV2& tilingData, const PFAShapeInfo& queryShapeInfo) 
{
    uint32_t sOuterFactor = 0;
    uint32_t sInnerFactor = 0;
    uint32_t softmaxSInnerFactor = 0;
    uint32_t softmaxSOuterFactor = 0;
    // Currently, there will be no D splitting scenario, and split D = 0 is default when splitting.
    auto ret = AdjustCVTilingCVDiff(contextKeyParams, sOuterFactor, sInnerFactor, softmaxSOuterFactor,
        tilingData, queryShapeInfo);
    OP_CHECK_IF(!ret, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "adjust tiling cv diff fail"),
        return ge::GRAPH_FAILED);
    softmaxSInnerFactor = sInnerFactor;

    tilingData.promptAttentionBaseParams.set_softmaxOuterSize(softmaxSOuterFactor);
    tilingData.promptAttentionSingleCoreParams.set_singleProcessSOuterSize(sOuterFactor);
    tilingData.promptAttentionSingleCoreParams.set_singleProcessSInnerSize(sInnerFactor);
    tilingData.promptAttentionBaseParams.set_splitS2(splitS2);

    sOuterFactorTiling = sOuterFactor;
    softmaxSInnerFactorTiling = softmaxSInnerFactor;
    softmaxSOuterFactorTiling = softmaxSOuterFactor;
    return ge::GRAPH_SUCCESS;
}

bool PromptFlashAttentionTilingV2::IsFlashDecode(ContextParamsForPFATiling& contextKeyParams, uint64_t bng) const
{
    float flashDecodeBNRatio = 0.4F; // 0.4, 经验值
    if (maxActualseqKV < SINNER_FACTOR_DOUBLE) {
        return false;
    }
    if ((bng < flashDecodeBNRatio * aicNum) && (gSize == 1)) {
        OP_LOGD(contextKeyParams.opName, "Flash decode dplit key/value.");
        return true;
    }
    if (gSize > NLIMIT / 2) { // 2: gsize threshold
        return false;
    }

    if ((bng < flashDecodeBNRatio * aicNum) && (maxActualseqKV >= 2048)) { // 2048, 在flash decode + gqa时的经验值
        OP_LOGD(contextKeyParams.opName, "Flash decode And GQA split key/value.");
        return true;
    }
    return false;
}

ge::graphStatus PromptFlashAttentionTilingV2::SplitBNS(PromptFlashAttentionTilingDataV2& tilingData, uint64_t bng) 
{
    uint64_t batchSize = tilingData.promptAttentionBaseParams.get_batchSize();
    uint64_t headNumSize = tilingData.promptAttentionBaseParams.get_headNumSize() * gSize;
    uint64_t headDimAlign = AlignUp(tilingData.promptAttentionBaseParams.get_vHeadSize(), BYTE_BLOCK);
    uint32_t kvSplitLimit = 256U; // 256: 经验值
    int64_t kvSplitPart = aicNum / bng;
    while(((maxActualseqKV / kvSplitPart) < kvSplitLimit) && (kvSplitPart > 1)) {
        kvSplitPart--;
    }

    faTilingAdapter.inputParamsRegbase.set_kvSplitPart(kvSplitPart);
    faTilingAdapter.inputParamsRegbase.set_accumOutSize(batchSize * headNumSize * kvSplitPart * headDimAlign);
    faTilingAdapter.inputParamsRegbase.set_logSumExpSize(batchSize * headNumSize * kvSplitPart * (BYTE_BLOCK / sizeof(float)));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTilingV2::ComputeTilingData(ContextParamsForPFATiling& contextKeyParams,
    std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV,
    PromptFlashAttentionTilingDataV2& tilingData) 
{
    if (splitCoreMode == SplitCoreMode::SPLIT_NBS_CUBE) {
        bool isAttenMaskUsed = (contextKeyParams.attentionMaskShape != nullptr);
        PromptFlashAttentionSplitNBSeq(tilingData, actualSeqLengths, actualSeqLengthsKV, isAttenMaskUsed);
    }

    if (needInit == 1) {
        PromptFlashAttentionInitOutputSplit(contextKeyParams.outputShape->GetStorageShape().GetShapeSize(), tilingData);
    }

    if (contextKeyParams.isSoftMaxLseEnable) {
        PromptFlashAttentionInitSoftmaxLseOutputSplit(contextKeyParams.lseoutputShape->GetStorageShape().GetShapeSize(),
            tilingData);
    }

    if (enableIFA && !enablePFAMerge) {
        PromptAttentionSingleCoreParams* singleCoreParams = &tilingData.promptAttentionSingleCoreParams;
        uint32_t sOuterSize = singleCoreParams->get_singleProcessSOuterSize();
        uint64_t batchSize = tilingData.promptAttentionBaseParams.get_batchSize();
        uint64_t headNumKVSize = tilingData.promptAttentionBaseParams.get_headNumSize(); // IFA kv N
        uint64_t bng = batchSize * headNumKVSize * (gSize + sOuterSize - 1) / sOuterSize;
        if (IsFlashDecode(contextKeyParams, bng)) {
            enableFlashDecode = true;
            return SplitBNS(tilingData, bng);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTilingV2::ComputeTilingKey(ContextParamsForPFATiling& contextKeyParams,
    uint32_t& numBlocksToBeSet, PromptFlashAttentionTilingDataV2& tilingData) 
{
    bool tilingRet = TilingGetTilingKeyAttentionAscendC(contextKeyParams, tilingData);
    OP_CHECK_IF(!tilingRet, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Get tilingKey fail"),
        return ge::GRAPH_FAILED);
    auto platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "platformInfoPtr is null"),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    numBlocksToBeSet = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);

    size_t* workspaces = contextKeyParams.workspaceSize;
    workspaces[0] = GetPFAWorkSpaceSize(tilingData);
    return ge::GRAPH_SUCCESS;
}

void PromptFlashAttentionTilingV2::SetAttenMaskCompressMode()
{
    static std::map<uint32_t, uint8_t> sparseToCompressModeMap = {
        {SPARSE_MODE_NO_MASK, PfaAttenMaskCompressMode::PFA_NO_COMPRESS_MODE},
        {SPARSE_MODE_ALL_MASK, PfaAttenMaskCompressMode::PFA_NO_COMPRESS_MODE},
        {SPARSE_MODE_LEFT_UP, PfaAttenMaskCompressMode::PFA_LEFT_UP_CAUSAL_MODE},
        {SPARSE_MODE_RIGHT_DOWN, PfaAttenMaskCompressMode::PFA_RIGHT_DOWN_CAUSAL_MODE},
        {SPARSE_MODE_BAND, PfaAttenMaskCompressMode::PFA_BAND_MODE}
    };
    auto itr = sparseToCompressModeMap.find(sparseModeVal);
    if (itr == sparseToCompressModeMap.end()) {
        faTilingAdapter.inputParamsRegbase.set_attenMaskCompressMode(0);
    } else {
        faTilingAdapter.inputParamsRegbase.set_attenMaskCompressMode(itr->second);
    }
}

void PromptFlashAttentionTilingV2::SetLayoutType()
{
    static std::map<InputLayout, LayoutType> layoutStrToLayoutTypeMap = {
        {InputLayout::BSH, LayoutType::LAYOUT_BSH},
        {InputLayout::TND, LayoutType::LAYOUT_TND},
        {InputLayout::BSND, LayoutType::LAYOUT_BSND},
        {InputLayout::BNSD, LayoutType::LAYOUT_BNSD},
        {InputLayout::NTD, LayoutType::LAYOUT_NTD},
    };
    auto itr = layoutStrToLayoutTypeMap.find(inputLayout);
    if (itr == layoutStrToLayoutTypeMap.end()) {
        faTilingAdapter.inputParamsRegbase.set_layoutType(static_cast<uint8_t>(0));
    } else {
        faTilingAdapter.inputParamsRegbase.set_layoutType(static_cast<uint8_t>(itr->second));
    }
}

ge::graphStatus PromptFlashAttentionTilingV2::SetQKVStartIdx(ContextParamsForPFATiling& contextKeyParams) 
{
    auto &inputParams = faTilingAdapter.inputParamsRegbase;
    inputParams.set_qStartIdx(0);
    inputParams.set_kvStartIdx(0);
    if (!enableAlibiPse) {
        return ge::GRAPH_SUCCESS;
    }
    int64_t qStartIdx = 0;
    int64_t kvStartIdx = 0;
    auto qStartIdxTensor = contextKeyParams.qStartIdx;
    if (qStartIdxTensor != nullptr) {
        if (qStartIdxTensor->GetShapeSize() >= 1) {
            const int64_t *value = qStartIdxTensor->GetData<int64_t>();
            if (value != nullptr) {
                qStartIdx = value[0];
                OP_CHECK_IF(qStartIdx > INT32_MAX || qStartIdx < INT32_MIN, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "qStartIdx should >= %d and <= %d, but qStartIdx = %ld.", INT32_MIN, INT32_MAX, qStartIdx),
                    return ge::GRAPH_FAILED);
                inputParams.set_qStartIdx(qStartIdx);
            }
        }
    }

    auto kvStartIdxTensor = contextKeyParams.kvStartIdx;
    if (kvStartIdxTensor != nullptr) {
        if (kvStartIdxTensor->GetShapeSize() >= 1) {
            const int64_t *kvValue = kvStartIdxTensor->GetData<int64_t>();
            if (kvValue != nullptr) {
                kvStartIdx = kvValue[0];
                OP_CHECK_IF(kvStartIdx > INT32_MAX || kvStartIdx < INT32_MIN, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "kvStartIdx should >= %d and <= %d, but kvStartIdx = %ld.", INT32_MIN, INT32_MAX, kvStartIdx),
                    return ge::GRAPH_FAILED);
                inputParams.set_kvStartIdx(kvStartIdx);
            }
        }
    }
    // 当kvStartIdx - qStartIdx超出范围后，kernel侧转为float会造成丢失精度。
    OP_CHECK_IF(kvStartIdx - qStartIdx > POS_SHIFT_MAX || kvStartIdx - qStartIdx < POS_SHIFT_MIN, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "kvStartIdx - qStartIdx should >= %d and <= %d, but qStartIdx = %ld, kvStartIdx = %ld.", POS_SHIFT_MIN, POS_SHIFT_MAX, qStartIdx, kvStartIdx),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTilingV2::ConvertContextToPFAParams(ContextParamsForPFATiling& contextKeyParams) const
{
    contextKeyParams.opName = context_->GetNodeName();
    bool inputOutputIsNullPtr = (context_->GetInputDesc(QUERY_INDEX) == nullptr) || (context_->GetInputDesc(KEY_INDEX) == nullptr) ||
        (context_->GetInputDesc(VALUE_INDEX) == nullptr) || (context_->GetOutputDesc(ATTENTION_OUT_INDEX) == nullptr) ||
        (context_->GetInputShape(QUERY_INDEX) == nullptr) || (context_->GetInputShape(KEY_INDEX) == nullptr) ||
        (context_->GetInputShape(VALUE_INDEX) == nullptr) || (context_->GetOutputShape(ATTENTION_OUT_INDEX) == nullptr);
    OP_CHECK_IF(inputOutputIsNullPtr,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "q, k, v or attenOut is nullptr!"),
        return ge::GRAPH_FAILED);

    contextKeyParams.isKvContinuous = 1U;
    contextKeyParams.emptyTensor = 0U;
    contextKeyParams.fromTilingSink = 0U;
    contextKeyParams.pseShift = context_->GetOptionalInputTensor(PSE_SHIFT_INDEX);
    contextKeyParams.attentionMask = context_->GetOptionalInputTensor(ATTEN_MASK_INDEX);
    contextKeyParams.actualSequenceLengthQ = context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    contextKeyParams.actualSequenceLengthKV = context_->GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
    contextKeyParams.antiquantScale = context_->GetOptionalInputTensor(ANTIQUANT_SCALE_INDEX);
    contextKeyParams.antiquantOffset = context_->GetOptionalInputTensor(ANTIQUANT_OFFSET_INDEX);
    contextKeyParams.inputDataType = context_->GetInputDesc(QUERY_INDEX)->GetDataType();
    contextKeyParams.kDataType = context_->GetInputDesc(KEY_INDEX)->GetDataType();
    contextKeyParams.vDataType = context_->GetInputDesc(VALUE_INDEX)->GetDataType();
    contextKeyParams.blockTable = nullptr;
    contextKeyParams.keySharedPrefix = (nullptr);
    contextKeyParams.valueSharedPrefix = (nullptr);
    contextKeyParams.actualSharedPrefixLen = (nullptr);
    contextKeyParams.pseShiftDataType = (contextKeyParams.pseShift != nullptr) ?
    context_->GetOptionalInputDesc(PSE_SHIFT_INDEX)->GetDataType() : contextKeyParams.inputDataType;
    contextKeyParams.maskDataType = (contextKeyParams.attentionMask != nullptr) ?
    context_->GetOptionalInputDesc(ATTEN_MASK_INDEX)->GetDataType() : contextKeyParams.inputDataType;
    contextKeyParams.outputDataType = context_->GetOutputDesc(ATTENTION_OUT_INDEX)->GetDataType();
    contextKeyParams.queryInputShape = context_->GetInputShape(QUERY_INDEX);
    contextKeyParams.keyInputShape = context_->GetInputShape(KEY_INDEX);
    contextKeyParams.valueInputShape = context_->GetInputShape(VALUE_INDEX);
    contextKeyParams.pseShiftShape = context_->GetOptionalInputShape(PSE_SHIFT_INDEX);
    contextKeyParams.attentionMaskShape = context_->GetOptionalInputShape(ATTEN_MASK_INDEX);
    contextKeyParams.deqScale1Shape = context_->GetOptionalInputShape(DEQ_SCALE1_INDEX);
    contextKeyParams.scale1Shape = context_->GetOptionalInputShape(QUANT_SCALE1_INDEX);
    contextKeyParams.deqScale2Shape = context_->GetOptionalInputShape(DEQ_SCALE2_INDEX);
    contextKeyParams.scale2Shape = context_->GetOptionalInputShape(QUANT_SCALE2_INDEX);
    contextKeyParams.offset2Shape = context_->GetOptionalInputShape(QUANT_OFFSET2_INDEX);
    contextKeyParams.antiquantScaleShape = context_->GetOptionalInputShape(ANTIQUANT_SCALE_INDEX);
    contextKeyParams.antiquantOffsetShape = context_->GetOptionalInputShape(ANTIQUANT_OFFSET_INDEX);
    contextKeyParams.outputShape = context_->GetOutputShape(0);
    auto attrs = context_->GetAttrs();
    contextKeyParams.innerPrecisePtr = attrs->GetAttrPointer<int64_t>(ATTR_INNER_PRECISE);
    contextKeyParams.headsNumber = attrs->GetAttrPointer<int64_t>(ATTR_N_INDEX);
    contextKeyParams.sparseMode = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_MODE);
    contextKeyParams.preToken = attrs->GetAttrPointer<int64_t>(ATTR_PRE_TOKEN_INDEX);
    contextKeyParams.nextToken = attrs->GetAttrPointer<int64_t>(ATTR_NEXT_TOKEN_INDEX);
    contextKeyParams.scaleValue = attrs->GetAttrPointer<float>(ATTR_SCALE_INDEX);
    contextKeyParams.layout = attrs->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX);
    contextKeyParams.numKeyValueHeads = attrs->GetAttrPointer<int64_t>(ATTR_NUM_KV_HEADS_INDEX);
    contextKeyParams.workspaceSize = context_->GetWorkspaceSizes(1);
    contextKeyParams.compileInfoPtr = reinterpret_cast<const PromptFlashAttentionCompileInfo *>(context_->GetCompileInfo());
    contextKeyParams.isBSNDOut = (string(contextKeyParams.layout) == "BNSD_BSND") ? 1U : 0U;
    contextKeyParams.transposeLayout = GetTransposeLayout(string(contextKeyParams.layout));
    contextKeyParams.fromFused = NUM_0;

    contextKeyParams.deqScaleType = (context_->GetOptionalInputDesc(DEQ_SCALE1_INDEX) != nullptr) ?
    context_->GetOptionalInputDesc(DEQ_SCALE1_INDEX)->GetDataType() : contextKeyParams.inputDataType;
    contextKeyParams.deqScale2Type = (context_->GetOptionalInputDesc(DEQ_SCALE2_INDEX) != nullptr) ?
    context_->GetOptionalInputDesc(DEQ_SCALE2_INDEX)->GetDataType() : contextKeyParams.inputDataType;

    contextKeyParams.quantScale2Type = (context_->GetOptionalInputDesc(QUANT_SCALE2_INDEX) != nullptr) ?
        context_->GetOptionalInputDesc(QUANT_SCALE2_INDEX)->GetDataType() : ge::DT_FLOAT;
    contextKeyParams.quantOffset2Type = (context_->GetOptionalInputDesc(QUANT_OFFSET2_INDEX) != nullptr) ?
        context_->GetOptionalInputDesc(QUANT_OFFSET2_INDEX)->GetDataType() : ge::DT_FLOAT;

    OP_CHECK_IF(contextKeyParams.workspaceSize == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "workSpaceSize got from ge is nullptr"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void PromptFlashAttentionTilingV2::PFATilingDataconvert(PromptFlashAttentionTilingDataV2& tilingData) 
{
    if (emptyTensor) {
        auto &initOutputParams = faTilingAdapter.initOutputParams;
        initOutputParams.set_singleCoreSize(tilingData.promptAttentionInitOutputParams.get_singleCoreSize());
        initOutputParams.set_totalOutputSize(tilingData.promptAttentionInitOutputParams.get_totalOutputSize());
        initOutputParams.set_totalSoftMaxLseOutputSize(tilingData.promptAttentionInitOutputParams.get_totalSoftMaxLseOutputSize());
        return;
    }
    SetLayoutType();
    auto &inputParams = faTilingAdapter.inputParamsRegbase;
    inputParams.set_bSize(tilingData.promptAttentionBaseParams.get_batchSize());
    inputParams.set_t1Size(tilingData.promptAttentionBaseParams.get_t1Size());
    inputParams.set_t2Size(tilingData.promptAttentionBaseParams.get_t2Size());
    // 将GS1合轴与不合轴场景下，有不同含义的n2Size、gSize与s1Size参数，转化为各自实际的值
    if (enableIFAMLA || enableIFA || enablePFAMerge) {
        inputParams.set_n2Size(tilingData.promptAttentionBaseParams.get_headNumSize());
        inputParams.set_gSize(tilingData.promptAttentionBaseParams.get_gOfMla());
        inputParams.set_s1Size(tilingData.promptAttentionBaseParams.get_seqSize() /
            tilingData.promptAttentionBaseParams.get_gOfMla());
    } else {
        inputParams.set_n2Size(tilingData.promptAttentionBaseParams.get_headNumSize() /
            tilingData.promptAttentionBaseParams.get_headNumRatio());
        inputParams.set_gSize(tilingData.promptAttentionBaseParams.get_headNumRatio());
        inputParams.set_s1Size(tilingData.promptAttentionBaseParams.get_seqSize());
    }
    inputParams.set_s2Size(tilingData.promptAttentionBaseParams.get_seqInnerSize());
    inputParams.set_alignedS2(0); // 默认值
    inputParams.set_dSize(tilingData.promptAttentionBaseParams.get_headSize());
    inputParams.set_dSizeV(tilingData.promptAttentionBaseParams.get_vHeadSize());
    inputParams.set_dSizeRope(ROPE_DIMENSION_SIZE_TILING_V2);
    inputParams.set_scaleValue(tilingData.promptAttentionBaseParams.get_scaleValue());
    inputParams.set_preTokens(tilingData.promptAttentionBaseParams.get_preTokens());
    inputParams.set_nextTokens(tilingData.promptAttentionBaseParams.get_nextTokens());
    inputParams.set_pseS1Size(tilingData.promptAttentionBaseParams.get_pseShiftS1Size());
    inputParams.set_pseS2Size(tilingData.promptAttentionBaseParams.get_pseShiftS2Size());
    if (!enableAlibiPse) {
        inputParams.set_pseBSize(tilingData.promptAttentionSingleCoreParams.get_pseShiftBatch());
        inputParams.set_pseShapeType(0); // 对应训练 PSE_B_N2_G_S1_S2
        inputParams.set_pseType(0); // 对应训练 PSE_OUTER_MUL_ADD_TYPE
    }
    inputParams.set_bandIndex(0); // 训练代码中在TND场景生效，用于计算s2方向循环的起始位置
    inputParams.set_pseEncodeType(0); // 默认值
    inputParams.set_pseAlibiBaseS1(0); // 默认值
    inputParams.set_pseAlibiBaseS2(0); // 默认值
    inputParams.set_attenMaskShapeType(attenMaskShapeType);
    inputParams.set_attenMaskDataType(1); // 默认值
    SetAttenMaskCompressMode();
    inputParams.set_implMode(static_cast<uint8_t>(HIGH_PRECISION));
    inputParams.set_sparseType(sparseType);
    inputParams.set_needDropMaskOp(0); // 默认值
    inputParams.set_keepProb(0); // 默认值
    inputParams.set_keepProbUint8(0); // 默认值
    inputParams.set_dropMaskOuter(0); // 默认值
    inputParams.set_remain(0); // 默认值
    inputParams.set_attenMaskS2Size(tilingData.promptAttentionBaseParams.get_maskKVsSize());
    inputParams.set_rsv1(0); // 默认值
    inputParams.set_s1SparseValidSize(0); // 临时默认值
    inputParams.set_s2SparseValidSize(0); // 临时默认值
    inputParams.set_seed(0); // 默认值
    inputParams.set_offset(0); // 默认值

    // PFA
    inputParams.set_blockSize(tilingData.promptAttentionBaseParams.get_blockSize());
    inputParams.set_blockTableDim2(tilingData.promptAttentionBaseParams.get_blockTableDim2());
    inputParams.set_paBlockNumSum(tilingData.promptAttentionBaseParams.get_PABlockNumSum());
    inputParams.set_prefixSeqInnerSize(tilingData.promptAttentionBaseParams.get_prefixSeqInnerSize());
    inputParams.set_paLayoutType(tilingData.promptAttentionBaseParams.get_PAlayoutType());
    inputParams.set_attenMaskS1Size(tilingData.promptAttentionBaseParams.get_maskQsSize());
    inputParams.set_isActualSeqLengthsNull(tilingData.promptAttentionBaseParams.get_isActualSeqLengthsNull());
    inputParams.set_isActualSeqLengthsKVNull(tilingData.promptAttentionBaseParams.get_isActualSeqLengthsKVNull());
    inputParams.set_actualSeqLengthsSize(tilingData.promptAttentionBaseParams.get_actualSeqLengthsSize());
    inputParams.set_actualSeqLengthsKVSize(tilingData.promptAttentionBaseParams.get_actualSeqLengthsKVSize());
    inputParams.set_deqScaleFlag(tilingData.promptAttentionBaseParams.get_deqScaleFlag());
    inputParams.set_deqScale2Flag(tilingData.promptAttentionBaseParams.get_deqScale2Flag());
    inputParams.set_isKvContinuous(tilingData.promptAttentionBaseParams.get_isKvContinuous());
    inputParams.set_fromFused(tilingData.promptAttentionBaseParams.get_fromFused());
    inputParams.set_isBSNDOut(tilingData.promptAttentionBaseParams.get_isBSNDOut());
    inputParams.set_transposeLayout(tilingData.promptAttentionBaseParams.get_transposeLayout());
    inputParams.set_isGqa(tilingData.promptAttentionBaseParams.get_isIFA() || enablePFAMerge);
    inputParams.set_isSoftMaxLseEnable(tilingData.promptAttentionBaseParams.get_isSoftMaxLseEnable());
    inputParams.set_isActualSharedPrefixLenNull(tilingData.promptAttentionBaseParams.get_isActualSharedPrefixLenNull());
    inputParams.set_isQHasLeftPadding(tilingData.promptAttentionBaseParams.get_isQHasLeftPadding());
    inputParams.set_isKVHasLeftPadding(tilingData.promptAttentionBaseParams.get_isKVHasLeftPadding());
    inputParams.set_ropeHeadSize(tilingData.promptAttentionBaseParams.get_ropeHeadSize());
    inputParams.set_isRowInvalid(static_cast<uint8_t>(tilingData.promptAttentionBaseParams.get_isRowInvalid()));
    inputParams.set_headNumRatio(tilingData.promptAttentionBaseParams.get_headNumRatio());

    auto &initOutputParams = faTilingAdapter.initOutputParams;
    initOutputParams.set_singleCoreSize(tilingData.promptAttentionInitOutputParams.get_singleCoreSize());
    initOutputParams.set_totalOutputSize(tilingData.promptAttentionInitOutputParams.get_totalOutputSize());
    initOutputParams.set_totalSoftMaxLseOutputSize(tilingData.promptAttentionInitOutputParams.get_totalSoftMaxLseOutputSize());
    initOutputParams.set_needInit(static_cast<uint8_t>(tilingData.promptAttentionInitOutputParams.get_needInit()));
    initOutputParams.set_isOneN(static_cast<uint8_t>(tilingData.promptAttentionInitOutputParams.get_isOneN()));

    inputParams.set_isPostQuantPerChnl(tilingData.promptAttentionBaseParams.get_isQuant2Perchannel());
    inputParams.set_isPostQuantBF16(tilingData.promptAttentionBaseParams.get_isQuant2BF16());
    inputParams.set_antiquantPerTensorFlag(0);
    inputParams.set_antiquantPerHeadFlag(0);
    inputParams.set_antiquantParaSeqSize(1);
}

ge::graphStatus PromptFlashAttentionTilingV2::PromptFlashAttentionSetTilingData() const
{
    return ge::GRAPH_SUCCESS;
}

void PromptFlashAttentionTilingV2::GetMaxWorkspaceFlag(ContextParamsForPFATiling& contextKeyParams) 
{
    if ((contextKeyParams.actualSequenceLengthQ != nullptr && contextKeyParams.actualSequenceLengthQ->GetData<int64_t>() == nullptr) || 
        (contextKeyParams.actualSequenceLengthKV != nullptr && contextKeyParams.actualSequenceLengthKV->GetData<int64_t>() == nullptr) || 
        (contextKeyParams.actualSharedPrefixLen != nullptr && contextKeyParams.actualSharedPrefixLen->GetData<int64_t>() == nullptr)) {
        isMaxWorkspace = true;
    } else {
        isMaxWorkspace = false;
    }
}

void PromptFlashAttentionTilingV2::InitializeMaxWorkspace(PFAShapeInfo& queryShapeInfo,
    PFAShapeInfo& keyShapeInfo, std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV) 
{
    uint32_t lenDims = queryShapeInfo.b;
    for (uint32_t i = LOOP_BEGIN_NUM; i < lenDims; i++) {
        actualSeqLengths[i] = queryShapeInfo.s;
        actualSeqLengthsKV[i] = keyShapeInfo.s;
    }
    maxActualseqKV = queryShapeInfo.s;
}

ge::graphStatus PromptFlashAttentionTilingV2::RunBigKernelTilingWithParams(ContextParamsForPFATiling& contextKeyParams,
    uint32_t& numBlocksToBeSet, PromptFlashAttentionTilingDataV2& tilingData) 
{
    GetMaxWorkspaceFlag(contextKeyParams);

    // set memory parameters
    if (SetPlatMemoryInfo(contextKeyParams) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // set sttributes switch
    if (SetAttributeInfo(contextKeyParams) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // tensor ptr is empty or tensor is invalid
    if (CheckTensorInvalid(contextKeyParams) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // empty tensor check
    if (CheckEmptyTensor(contextKeyParams) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (emptyTensor) {
        SetEmptyTensor(contextKeyParams, numBlocksToBeSet, tilingData);
        return ge::GRAPH_SUCCESS;
    }

    // Input, output check
    PFAShapeInfo queryShapeInfo;
    PFAShapeInfo keyShapeInfo;
    PFAShapeInfo valueShapeInfo;
    PFAShapeInfo queryRopeShapeInfo;

    // Check kinds of single attribute
    if (CheckSingleAttribute(contextKeyParams, queryShapeInfo, keyShapeInfo, valueShapeInfo, queryRopeShapeInfo, tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // 同IFA合轴实现
    if (enablePFAMerge) {
        if (*contextKeyParams.numKeyValueHeads > 0) {
            gSize = (*contextKeyParams.headsNumber) / (*contextKeyParams.numKeyValueHeads);
        } else {
            gSize = 1U;
        }
        tilingData.promptAttentionBaseParams.set_headNumRatio(1);
        tilingData.promptAttentionBaseParams.set_gOfMla(gSize);
    }
    OP_CHECK_IF(gSize == 0, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "calculate gSize = 0"), return ge::GRAPH_FAILED);

    if (enableIFAMLA || enableIFA || enablePFAMerge) {
        OP_LOGI(contextKeyParams.opName,
            "enter gs1 merge! enableIFAMLA is %d, enableIFA is %d, enablePFAMerge is %d", enableIFAMLA, enableIFA, enablePFAMerge);
        queryShapeInfo.n = queryShapeInfo.n / gSize;
        queryShapeInfo.s = queryShapeInfo.s * gSize;
    }

    // multi feature crossover check
    std::vector<int64_t> actualSeqLengths(queryShapeInfo.b);
    std::vector<int64_t> actualSeqLengthsKV(queryShapeInfo.b);

    // Check crossover attribute
    if (CheckCrossoverAttribute(contextKeyParams, queryShapeInfo, tilingData) != ge::GRAPH_SUCCESS){
        return ge::GRAPH_FAILED;
    }

    if (isMaxWorkspace) {
        InitializeMaxWorkspace(queryShapeInfo, keyShapeInfo, actualSeqLengths, actualSeqLengthsKV);
    } else {
        if (!CheckMultiFeatureCrossover(contextKeyParams, queryShapeInfo,
            actualSeqLengths, actualSeqLengthsKV, tilingData)) {
            return ge::GRAPH_FAILED;
        }
    }

    // print shape info
    OP_LOGI(contextKeyParams.opName,
        "inputLayout is %s, innerPrecise is %d, scaleValue is %f, preTokens is %ld, nextTokens is %ld",
        string(contextKeyParams.layout).c_str(), innerPrecise, *contextKeyParams.scaleValue,
        *contextKeyParams.preToken, *contextKeyParams.nextToken);

    // set tilingData
    SetTilingData(contextKeyParams, queryShapeInfo, queryRopeShapeInfo, valueShapeInfo, tilingData);

    // Infering whether the tiling mode is S2 full load, CV diff, and whether to use the matmul norm template.
    InferTilingMod(contextKeyParams, actualSeqLengths, actualSeqLengthsKV, queryShapeInfo.b, queryShapeInfo.d);
    InferSplitCoreMode();
    // Whether to enable constant templates
    InferConstantization();

    if (AdjustTilingData(contextKeyParams, tilingData, queryShapeInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // DN check
    GetEnableDN(contextKeyParams, tilingData, queryShapeInfo, valueShapeInfo, actualSeqLengths, actualSeqLengthsKV);

    if (ComputeTilingData(contextKeyParams, actualSeqLengths, actualSeqLengthsKV,
        tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // Compute tiling key.
    if (ComputeTilingKey(contextKeyParams, numBlocksToBeSet, tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (SetQKVStartIdx(contextKeyParams) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

void PromptFlashAttentionTilingV2::SetTilingKey(ContextParamsForPFATiling& contextKeyParams) const 
{
    uint64_t gen_tilingkey = GET_TPL_TILING_KEY(static_cast<uint64_t>(inOutLayoutType), static_cast<uint64_t>(config),
                                                static_cast<uint64_t>(pseMode), static_cast<uint64_t>(quantMode), hasAttenMask,
                                                hasRope, isPa, isFd, emptyTensor,
                                                static_cast<uint64_t>(enableKVPrefix), static_cast<uint64_t>(enableS1OutSplit));
    context_->SetTilingKey(gen_tilingkey);
    OP_LOGI(contextKeyParams.opName, "The new template tilingkey is %llu.", gen_tilingkey);
    OP_LOGI(contextKeyParams.opName, "The new template tilingkey param is inOutLayoutType: %llu, config: %llu, pseMode: %llu, quantMode: %llu, "
            "hasAttenMask: %llu, hasRope: %llu, isPa: %llu, isFd: %llu, emptyTensor: %llu, enableKVPrefix: %llu, enableS1OutSplit:%llu.",
            static_cast<uint64_t>(inOutLayoutType), static_cast<uint64_t>(config), static_cast<uint64_t>(pseMode),
            static_cast<uint64_t>(quantMode), hasAttenMask, hasRope, isPa, isFd, emptyTensor,
            static_cast<uint64_t>(enableKVPrefix), static_cast<uint64_t>(enableS1OutSplit));
}

ge::graphStatus PromptFlashAttentionTilingV2::DoSubOpTiling(PromptFlashAttentionTilingDataV2& tilingData, ContextParamsForPFATiling& contextParamsForPFATiling) {
    uint32_t numBlocksToBeSet;
    auto ret = RunBigKernelTilingWithParams(contextParamsForPFATiling, numBlocksToBeSet, tilingData);
    OP_CHECK_IF(ret == ge::GRAPH_FAILED, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "fail to parse tiling params!"), return ge::GRAPH_FAILED);
    context_->SetBlockDim(numBlocksToBeSet);
    OP_CHECK_IF(memset_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
        0, context_->GetRawTilingData()->GetCapacity()) != EOK,
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "fail to memset tiling data"),
        return ge::GRAPH_FAILED);
    if (faRunFlag_) {
        PFATilingDataconvert(tilingData);
        uint64_t cap = context_->GetRawTilingData()->GetCapacity();
        OP_LOGI(contextParamsForPFATiling.opName, "Tiling Data context GetCapacity: %lu.", cap);
        FlashAttentionScoreSimplifiedTilingData* tiling = context_->GetTilingData<FlashAttentionScoreSimplifiedTilingData>();
        if (tiling == nullptr) {
            OP_LOGE(contextParamsForPFATiling.opName, "tiling get is nullptr");
            return ge::GRAPH_FAILED;
        }
        *tiling = faTilingAdapter;
    } else {
        uint64_t cap = context_->GetRawTilingData()->GetCapacity();
        OP_LOGI(contextParamsForPFATiling.opName, "TilingData context GetCapacity: %lu, faRunFlag_ is %d", cap, faRunFlag_);
        PromptFlashAttentionTilingDataV2* tiling = context_->GetTilingData<PromptFlashAttentionTilingDataV2>();
        *tiling = tilingData;
    }
    // 使用SyncAll，需要设置为batchmode模式，所有核同时启动，否则多流方式下执行可能会卡死
    context_->SetScheduleMode(BATCH_MODE_SCHEDULE);
    return ret;
}

ge::graphStatus PromptFlashAttentionTilingV2::DoOpTiling()
{  
    PromptFlashAttentionTilingDataV2 tilingData;
    ContextParamsForPFATiling contextParamsForPFATiling;
    auto ret = ConvertContextToPFAParams(contextParamsForPFATiling);
    OP_CHECK_IF(ret == ge::GRAPH_FAILED, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "fail to convert to PFAParams"),return ge::GRAPH_FAILED);
    ret = DoSubOpTiling(tilingData, contextParamsForPFATiling);
    SetTilingKey(contextParamsForPFATiling);
    OP_LOGI(contextParamsForPFATiling.opName, "All the PFATiling work is done.");
    return ret;
}

REGISTER_TILING_TEMPLATE_FIA(PromptFlashAttention, PromptFlashAttentionTilingV2, std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_3510)}), 90);
} // namespace v2
} // namespace optiling
