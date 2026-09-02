/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ffn_tiling.cpp
 * \brief
 */
#include <alog_pub.h>
#include "ffn_tiling.h"
#include "ffn_arch35_tiling.h"

#include <algorithm>
#include <climits>
#include <register/op_impl_registry.h>

#include "op_host/tiling_templates_registry.h"
#include "err/ops_err.h"
#include "log/log.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "platform/platform_info.h"

#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_basic_aswt_tiling.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_basic_streamk_tiling.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_data.h"

constexpr int64_t SIMT_UB_SIZE_BYTE = 40960;

using namespace ge;
using namespace AscendC;


namespace optiling {

constexpr uint64_t ONE_MATMUL = 2 * 1000;
constexpr uint64_t HIGH_PERFORMANCE_KEY = 0;
constexpr uint64_t QUANT_KEY = 1;
constexpr uint64_t HIGH_PRECISION_KEY = 3;
constexpr uint64_t HIGH_PRECISION_BF16_KEY = 7;
constexpr uint64_t ANTI_QUANT_KEY = 6;
constexpr uint64_t ANTI_QUANT_PERGROUP_KEY = 12;
constexpr uint32_t HALF_DATA_SIZE = 2;
constexpr uint32_t QUANT_BF16_KEY = 11;
constexpr uint32_t ANTI_QUANT_MSD_KEY = 15;
constexpr uint64_t QUANT_DEQ_FLOAT32_KEY = 13;
constexpr uint32_t QUANT_SMOOTH_KEY = 14;
constexpr uint32_t QUANT_STEPN_KEY = 2002;

class ActiveMap {
public:
    const char *activeName;
    ActiveType activeType;
};

constexpr class ActiveMap ACTIVE_MAP[] = {
    {"fastgelu", ActiveType::FASTGELU}, {"relu", ActiveType::RELU},   {"silu", ActiveType::SILU},
    {"gelu", ActiveType::GELU},         {"geglu", ActiveType::GEGLU}, {"swiglu", ActiveType::SWIGLU},
    {"reglu", ActiveType::REGLU},
};

constexpr int64_t DIMS_2 = 2;
constexpr int64_t DIMS_3 = 3;

constexpr uint32_t X_INDEX = 0;
constexpr uint32_t WEIGHT1_INDEX = 1;
constexpr uint32_t WEIGHT2_INDEX = 2;
constexpr uint32_t TOKENS_ARR_INDEX = 3;
constexpr uint32_t DEQSCALE1_INDEX = 8;
constexpr uint32_t SCALE_INDEX = 6;
constexpr uint32_t ANTIQUANT_SCALE1_INDEX = 10;
constexpr uint32_t ANTIQUANT_SCALE2_INDEX = 11;
constexpr uint32_t TYPICAL1_N1 = 2560;

constexpr uint32_t MAX_EXPERT_NUM = 256;
constexpr uint32_t FP32_DATATYPE_SIZE = 4;
constexpr uint32_t MAX_BASE_BLOCK = 16 * 1024;
constexpr uint32_t MAX_UB_BLOCK = 16 * 256;
constexpr uint32_t MAX_BASEM = 256;
constexpr uint32_t BLOCK_SIZE_FFN = 32;
constexpr uint32_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr uint32_t MATMUL_MIN_SHAPE = 16;
constexpr uint32_t MATMUL_MIN_SHAPE_INT8 = 32;
constexpr uint32_t FIFTEEN = 15;
constexpr uint32_t INVERSE_FIFTEEN = ~15;
constexpr uint32_t SIXTEEN_ALIGN_CONSTANT = 16;
constexpr uint32_t ALIGN32 = 31;
constexpr uint32_t ALIGN64 = 64;
constexpr uint32_t UB_DIVIDE_NUM = 3;
constexpr uint32_t UB_DIVIDE_NUM_N1_ZERO = 12;
constexpr uint32_t UB_DIVIDE_NUM_HIGH_PRECISION = 9;
constexpr uint32_t UB_DIVIDE_NUM_QUANT = 7;
constexpr uint32_t UB_DIVIDE_NUM_QUANT_DEQ_FLOAT32 = 26;
constexpr uint32_t UB_DIVIDE_NUM_QUANT_BF16 = 20;
constexpr uint32_t GLU_UB_DIVIDE_NUM_FP16 = 7; // 2 input, 1 output, 4 tempbuffer
constexpr uint32_t CONSTANT_TWO = 2;
constexpr uint32_t MAX_UINT32 = 4294967295;
constexpr int64_t BEST_L1_PART1 = 160 * 1024;
constexpr int64_t BEST_L1_PART2 = 128 * 1024;
constexpr int64_t BEST_L1_PART_310P = 512 * 1024;
constexpr int64_t BEST_BASEN = 256;
constexpr int64_t GLU_BASEN = 128;
constexpr int64_t BEST_BASEN_MSD = 512;
constexpr uint32_t MSD_N_THRESHOLD = 1024;
constexpr uint32_t MSD_K_THRESHOLD = 2048;
constexpr uint32_t SMALL_TOKEN_BOUND = 64;
constexpr uint32_t TINY_TOKEN_BOUND = 16;
constexpr uint32_t UB_BLOCK_UNIT_SIZE = 32; // initbuffer need 32 bytes align
constexpr uint32_t UB_PER_BLOCK_ALIGN_310P = 1024;
constexpr uint32_t UB_PER_BLOCK_ALIGN = 4 * 1024;
constexpr uint32_t UB_ANTIQUANT_PER_BLOCK_ALIGN_FP16 = 8 * 1024;
constexpr uint32_t CALC_MM2_SINGLE_CORE_NUM = 40;

// High precision fp16: act io used 8 blks, tmp used 4 blks, act cal function used 8 blks
constexpr uint32_t UB_PRECISION_IO_USED_BLOCK_FP16 = 8;
constexpr uint32_t UB_PRECISION_BLOCK_NUM_FP16 = 18;
// High performance fp16: act io used 4 blks,  act cal function used 4 blks
constexpr uint32_t UB_PEFORMENCE_IO_USED_BLOCK_FP16 = 4;
constexpr uint32_t UB_PEFORMENCE_BLOCK_NUM_FP16 = 8;
// antiquant fp16: act io used 8 blks, scale/offset io used 1 blks, act cal function used 4 blks
constexpr uint32_t UB_ANTIQUANT_BLOCK_NUM_FP16 = 13;
// act + scale/offset io = 8 + 1. one block size is ubCalSize(8k align),castweight baseN is 512, 512 * 8 blks = 4096 <
// 8K
constexpr uint32_t UB_ANTIQUANT_IO_USED_BLOCK_FP16 = 9;
// antiquant bp16: act io used 12 blks, scale/offset io used 1 blks, tmp used 4 blks, act cal function used 6 blks
constexpr uint32_t UB_ANTIQUANT_BLOCK_NUM_BP16 = 23;
// act + scale/offset io = 12 + 1. one block size is ubCalSize(4k align)
constexpr uint32_t UB_ANTIQUANT_IO_USED_BLOCK_BP16 = 13;
// Quant fp16 out: inque + outque + api + tmp is 2 + 1
constexpr uint32_t UB_QUANT_IO_BLOCK_NUM_FP16_OUT = 3;
// Quant fp16 out: inque + outque + api + tmp is 2 + 1 + 2 + 4
constexpr uint32_t UB_QUANT_BLOCK_NUM_FP16_OUT = 9;
// Quant fp16 out: inque + outque + api + tmp is 4 + 2
constexpr uint32_t UB_QUANT_IO_BLOCK_NUM_BF16_OUT = 6;
// Quant fp16 out: inque + outque + api + tmp is 4 + 2 + 8 + 8
constexpr uint32_t UB_QUANT_BLOCK_NUM_BF16_OUT = 22;

constexpr int32_t HIGH_PRECISION = 0;
constexpr int32_t HIGH_PERFORMANCE = 1;
constexpr uint32_t MAX_EXPERT_PARALLELISM = 10;

inline static uint32_t SixteenAlign(uint32_t a)
{
    // 16 align down
    return a & INVERSE_FIFTEEN;
}

inline static uint32_t SixteenAlignUp(uint32_t a)
{
    // 16 align up
    return (a + FIFTEEN) & INVERSE_FIFTEEN;
}

inline static uint32_t Ceil(uint32_t a, uint32_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

static bool IsPrivateFormat(ge::Format format)
{
    if (format == ge::FORMAT_NC1HWC0 || format == ge::FORMAT_FRACTAL_Z || format == ge::FORMAT_NDC1HWC0 ||
        format == ge::FORMAT_FRACTAL_Z_3D || format == ge::FORMAT_FRACTAL_NZ || format == ge::FORMAT_NC1HWC0_C04) {
        return true;
    }
    return false;
}

class FFNTiling {
public:
    FFNTilingData tilingData;
    ge::graphStatus RunFusionKernelTiling(gert::TilingContext *context);

protected:
    void Init();
    ge::graphStatus FFNParamsCheck(gert::TilingContext *context);
    ge::graphStatus DataTypeCheck(gert::TilingContext *context);
    ge::graphStatus FormatCheck(const gert::TilingContext *context);
    ge::graphStatus FFNSingleCoreTiling(const gert::TilingContext *context, uint64_t ubSize);
    matmul_tiling::PlatformInfo MmGetPlatInfo(const FFNCompileInfo *compileInfoPtr) const;
    ge::graphStatus FFNApiMM1Tiling(const gert::TilingContext *context, const matmul_tiling::PlatformInfo &platInfo,
                                    matmul_tiling::DataType matmulDtype);
    void FFNApiMM2CalBaseMN(uint32_t &mm2BaseN, uint32_t &baseM, const matmul_tiling::PlatformInfo &platInfo);
    ge::graphStatus FFNApiMM2Tiling(const gert::TilingContext *context, const matmul_tiling::PlatformInfo &platInfo,
                                    matmul_tiling::DataType matmulDtype);
    ge::graphStatus FFNApiTiling(const gert::TilingContext *context, const matmul_tiling::PlatformInfo &platInfo,
                                 matmul_tiling::DataType matmulDtype);
    ge::graphStatus FFNSetTilingKey(gert::TilingContext *context, uint64_t &key);
    ge::graphStatus FFNSetTilingData(gert::TilingContext *context);
    ge::graphStatus CalMM1ValidUbBytes(int64_t &mm1VaildUbBytes) const;
    ge::graphStatus CalMM1BaseM(const gert::TilingContext *context, const uint32_t baseN, const uint64_t l0CSize,
                                const int64_t mm1VaildUbBytes, uint32_t &baseM);
    ge::graphStatus CalMMTilingBaseMNBasicBlock(const uint64_t basicBlkOperTimes, const uint32_t n, uint32_t &baseM,
                                                uint32_t &baseN) const;
    ge::graphStatus CalMM1TilingBaseMNBasicBlock(const gert::TilingContext *context, const uint64_t l0CSize,
                                                 const int64_t mm1VaildUbBytes, uint64_t basicBlkOperTimes,
                                                 uint32_t &baseN);
    ge::graphStatus CalMM1TilingBaseMNKBasicBlock(const gert::TilingContext *context,
                                                  const matmul_tiling::PlatformInfo &platInfo);
    ge::graphStatus CalMM2TilingBaseMNKBasicBlock(const gert::TilingContext *context,
                                                  const matmul_tiling::PlatformInfo &platInfo);
    ge::graphStatus CalMM1TilingBaseMNK(const gert::TilingContext *context,
                                        const matmul_tiling::PlatformInfo &platInfo);
    ge::graphStatus CalMM2TilingBaseMNK(const matmul_tiling::PlatformInfo &platInfo);

    void SetBiasInfo(const gert::TilingContext *context, matmul_tiling::MatmulApiTiling &mm,
                     const uint32_t &irIndex) const;
    void FFNCalMMStep(const uint32_t baseM, const uint32_t baseN, const uint32_t baseK, TCubeTiling &mmTilingData);
    ge::graphStatus FFNSetMM1Tiling(const gert::TilingContext *context, const matmul_tiling::PlatformInfo &platInfo,
                                    matmul_tiling::DataType matmulDtype);
    ge::graphStatus FFNSetMM2Tiling(const gert::TilingContext *context, const matmul_tiling::PlatformInfo &platInfo,
                                    matmul_tiling::DataType matmulDtype);
    ge::graphStatus FFNSetUbDivideBlk();
    ge::graphStatus FFNCalUbSize(uint32_t baseN, uint32_t divideBlkNum, uint32_t ioBlkNum, uint32_t &baseM);
    inline ge::graphStatus N1EqualZeroWithBias2(uint64_t ubSize);
    ge::graphStatus TilingCalcAndSetting(const gert::TilingContext *context,
                                         const matmul_tiling::PlatformInfo &platInfo,
                                         matmul_tiling::DataType matmulDtype);
    ge::graphStatus FFNGlu(gert::TilingContext *context, uint64_t ubSize, uint64_t l1Size, uint64_t l0CSize,
                           uint32_t aivNum);
    ge::graphStatus FFNGluParamsCheck(const gert::TilingContext *context) const;
    ge::graphStatus SetMMTilingType(const gert::TilingContext *context, bool isMM1, matmul_tiling::DataType matmulDtype,
                                    matmul_tiling::MatmulApiTiling &mm) const;
    ge::graphStatus FFNGluCalMM1Tiling(uint64_t ubSize, uint64_t l0CSize);
    ge::graphStatus FFNGluSetMM1Tiling(gert::TilingContext *context, uint64_t l1Size, uint64_t l0CSize,
                                       matmul_tiling::DataType matmulDtype, uint32_t aivNum);
    ge::graphStatus FFNGluCalMM2Tiling(uint64_t l0CSize);
    ge::graphStatus FFNGluSetMM2Tiling(gert::TilingContext *context, uint64_t l1Size, uint64_t l0CSize,
                                       matmul_tiling::DataType matmulDtype);
    ge::graphStatus FFNGetScaleGroupNum(const gert::TilingContext *context, const gert::Tensor *tokensArrTensor);
    void CheckMSD();
    void UpdateMaxTokens();
    ge::graphStatus GetInputShape(const gert::TilingContext *context);

    uint64_t SelectQuantTilingKey() const;

    ge::graphStatus FFNGetQuantScale(const gert::TilingContext *context, const gert::Tensor *tokensArrTensor);

    uint64_t SelectTilingKey() const;

    ge::graphStatus CheckAndGetBasicInfo(gert::TilingContext *context, const FFNCompileInfo *compileInfoPtr);
    ge::graphStatus TilingWithDifferentKN(gert::TilingContext *context, const FFNCompileInfo *compileInfoPtr,
                                          const uint32_t aicNum);
    void SetTilingBaseParams(gert::TilingContext *context, const FFNCompileInfo *compileInfoPtr, const uint32_t aicNum);

    void UpdateMM2BaseNByCoreTiling(uint32_t baseM, uint32_t &mm2BaseN);

    inline uint32_t GetSmallElementsNum(int32_t baseM) const
    {
        if (static_cast<int32_t>(bs) <= baseM) {
            // If bs it not larger than baseM, it is sure that the number of tokens of each expert is not larger than
            // baseM
            return expertNum;
        }

        return 0;
    }

    bool GetTokensIndexFlag(const gert::TilingContext *context) const;

private:
    uint32_t bs;
    uint32_t k1;
    uint32_t n1;
    uint32_t n2;
    uint32_t singleM2;
    uint32_t singleN2;
    uint32_t baseM2_;
    uint32_t baseN2_;
    uint32_t baseM1_;
    uint32_t baseN1_;
    uint32_t baseK1_;
    uint32_t baseK2_;
    uint64_t ubSize_;
    uint32_t mm2BaseM;
    uint32_t expertNum; // When number of expert > 1, the number of tokens for each expert is unknown
    uint32_t xDataTypeSize;
    uint32_t weightDataTypeSize;
    uint32_t ubCalSize;
    uint32_t ubRestBytes;
    uint32_t ubDivideBlkNum = 0;
    uint32_t ubIoBlkNum = 0;
    uint32_t ubBlockAlign = 0;
    ge::DataType xDataType;
    ge::DataType weight1Dtype;
    ge::DataType biasDataType;
    ge::DataType outputDtype;
    ge::DataType deqscaleDtype;
    uint32_t maxTokens;         // When expert tokens are unknown, it equals bs
    uint32_t maxTokensCheckOpt; // max tokens for checking optimization branch
    int32_t innerPrecise;
    matmul_tiling::CubeFormat wFormat = matmul_tiling::CubeFormat::ND;
    matmul_tiling::CubeFormat xFormat = matmul_tiling::CubeFormat::ND;
    matmul_tiling::CubeFormat yFormat = matmul_tiling::CubeFormat::ND;
    bool isPerGroup = false;
    bool isSmooth = false;
    bool isQuantBf16 = false;
    bool isMsdCase = false;
    bool is310P = false;
    bool isAscend950 = false;
    bool isFfnTransB = false;
    bool isRegbase = false;
    uint32_t minBaseNShape = MATMUL_MIN_SHAPE;
    bool isTilingDataValid = false;

    ActiveType GetActiveType(const gert::TilingContext *context, const char *activeType) const;
    ge::graphStatus GetBs(const gert::TilingContext *context, const gert::StorageShape *xShape);
    void CalMM2Single(uint32_t baseM2, uint32_t baseN2);
    void AdjustMM2MNLoops(const uint32_t align, uint32_t &m2Loops, uint32_t &n2Loops);
    void PrintFFNTiling(const gert::TilingContext *context, bool debugLevel);
    void PrintCriticalInfo(gert::TilingContext *context);
    void PrintMatMulTiling(const char *opName, TCubeTiling &matmulTiling) const;
};

void FFNTiling::Init()
{
    isPerGroup = false;
    isSmooth = false;
    isQuantBf16 = false;
    isMsdCase = false;
    is310P = false;
    isAscend950 = false;
    isFfnTransB = false;
    isRegbase = false;
    minBaseNShape = MATMUL_MIN_SHAPE;
    isTilingDataValid = false;
}

// Update baseN of mm2 according to kernel tiling logic, in order to boost performance
void FFNTiling::UpdateMM2BaseNByCoreTiling(uint32_t baseM, uint32_t &mm2BaseN)
{
    // Step1: determine how many AI cubes are available for each shape
    uint32_t aicNum = tilingData.ffnBaseParams.get_coreNum();
    uint32_t count = GetSmallElementsNum(tilingData.mm1TilingData.get_baseM());
    if (count > 1) {
        uint32_t maxExpertParallelism1 = aicNum / Ceil(n1, tilingData.mm1TilingData.get_baseN());
        maxExpertParallelism1 = std::min(MAX_EXPERT_PARALLELISM, std::max<uint32_t>(1, maxExpertParallelism1));
        uint32_t maxExpertParallelism2 = aicNum / Ceil(n2, mm2BaseN);
        maxExpertParallelism2 = std::min(MAX_EXPERT_PARALLELISM, std::max<uint32_t>(1, maxExpertParallelism2));
        maxExpertParallelism2 = std::min(count, std::min(maxExpertParallelism2, maxExpertParallelism1));
        aicNum = aicNum / std::max<uint32_t>(1, maxExpertParallelism2);
    }
    // Step2: determine the number of cubes available for N-dim of each shape, and compute the best baseN
    if (n2 % SIXTEEN_ALIGN_CONSTANT != 0 && aicNum > 1) {
        uint32_t nLoops = aicNum / Ceil(maxTokens, baseM);
        uint32_t newmm2BaseN = std::max(minBaseNShape, SixteenAlignUp(n2 / nLoops));
        if (nLoops > 1 && newmm2BaseN < mm2BaseN) {
            mm2BaseN = newmm2BaseN;
        }
    }
}

ActiveType FFNTiling::GetActiveType(const gert::TilingContext *context, const char *activeType) const
{
    for (const ActiveMap &item : ACTIVE_MAP) {
        size_t len = strlen(item.activeName);
        bool isValidActiveType = strlen(activeType) == len;
        // use for loop instead of strncasecmp to avoid possible out-of-bounds problems
        if (!isValidActiveType) {
            continue;
        }
        for (size_t i = 0; i < len; i++) {
            if (tolower(activeType[i]) != item.activeName[i]) {
                isValidActiveType = false;
                break;
            }
        }
        if (isValidActiveType) {
            OP_LOGI(context, "activeType is %s.", activeType);
            return item.activeType;
        }
    }
    return ActiveType::INVALID_TYPE;
}

ge::graphStatus FFNTiling::GetBs(const gert::TilingContext *context, const gert::StorageShape *xShape)
{
    int64_t tempBs;
    if (is310P) {
        tempBs = xShape->GetStorageShape().GetDim(1) * xShape->GetStorageShape().GetDim(DIMS_2);
    } else {
        tempBs = xShape->GetStorageShape().GetDim(0);
        size_t dimNum = xShape->GetStorageShape().GetDimNum() >= 1 ? xShape->GetStorageShape().GetDimNum() - 1 : 0;
        for (size_t i = 1; i < dimNum; i++) {
            tempBs *= xShape->GetStorageShape().GetDim(i);
        }
    }
    OP_CHECK_IF(xDataTypeSize == 0, OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "get x dtype size is 0"),
               return ge::GRAPH_FAILED);
    int32_t numInOneBlk = BLOCK_SIZE_FFN / xDataTypeSize;
    int32_t maxBs = INT_MAX / numInOneBlk * numInOneBlk;
    OP_CHECK_IF(tempBs > maxBs,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "32Byte-aligned M dim cannot be greater than INT32_MAX"),
               return ge::GRAPH_FAILED);
    bs = static_cast<uint64_t>(tempBs);
    return ge::GRAPH_SUCCESS;
}

void FFNTiling::UpdateMaxTokens()
{
    // When expert tokens are unknown,
    // maxTokens is set as 'bs', which is the maximum possible number of tokens for a single expert,
    // maxTokensCheckOpt is set as the mean value of number of tokens
    maxTokens = bs;
    maxTokensCheckOpt = expertNum <= 1 ? bs : (bs + expertNum - 1) / expertNum;
}

void FFNTiling::CheckMSD()
{
    uint64_t bestMaxTokenMsd =
        static_cast<uint64_t>(static_cast<float>(n1) / static_cast<float>(TYPICAL1_N1) * TINY_TOKEN_BOUND);
    bool isMsdN1K1 =
        (n1 % BEST_BASEN_MSD == 0) && n1 > MSD_N_THRESHOLD && (k1 % BEST_BASEN_MSD == 0) && k1 > MSD_K_THRESHOLD;
    isMsdCase = isMsdN1K1 && (xDataType == ge::DT_BF16 || xDataType == ge::DT_FLOAT16) && weight1Dtype == ge::DT_INT8 &&
                maxTokensCheckOpt <= bestMaxTokenMsd && bs <= 512; // 512: max M value in msd-scenario
}

ge::graphStatus FFNTiling::TilingCalcAndSetting(const gert::TilingContext *context,
                                                const matmul_tiling::PlatformInfo &platInfo,
                                                matmul_tiling::DataType matmulDtype)
{
    if (is310P) {
        OP_CHECK_IF(CalMM1TilingBaseMNKBasicBlock(context, platInfo) != ge::GRAPH_SUCCESS,
                   OP_LOGW(context, "Calculate mm1 baseMNK failed!"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(CalMM2TilingBaseMNKBasicBlock(context, platInfo) != ge::GRAPH_SUCCESS,
                   OP_LOGW(context, "Calculate mm2 baseMNK failed!"), return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(CalMM1TilingBaseMNK(context, platInfo) != ge::GRAPH_SUCCESS,
                   OP_LOGW(context, "Calculate mm1 baseMNK failed!"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(CalMM2TilingBaseMNK(platInfo) != ge::GRAPH_SUCCESS,
                   OP_LOGW(context, "Calculate mm2 baseMNK failed!"), return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(FFNSetMM1Tiling(context, platInfo, matmulDtype) != ge::GRAPH_SUCCESS,
               OP_LOGW(context, "Set mm1 tiling failed!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(FFNSetMM2Tiling(context, platInfo, matmulDtype) != ge::GRAPH_SUCCESS,
               OP_LOGW(context, "Set mm2 tiling failed!"), return ge::GRAPH_FAILED);

    OP_LOGI(context, "Calc tiling success!");
    isTilingDataValid = true;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FFNTiling::FFNGetScaleGroupNum(const gert::TilingContext *context, const gert::Tensor *tokensArrTensor)
{
    uint32_t scale1GroupNum = 1;
    uint32_t scale2GroupNum = 1;
    isPerGroup = false;
    if (context->GetOptionalInputTensor(ANTIQUANT_SCALE1_INDEX)) {
        const gert::StorageShape *antiScale1Shape = context->GetOptionalInputShape(ANTIQUANT_SCALE1_INDEX);
        uint32_t antiScale1DimNum = antiScale1Shape->GetStorageShape().GetDimNum();
        const gert::StorageShape *antiScale2Shape = context->GetOptionalInputShape(ANTIQUANT_SCALE2_INDEX);
        if (tokensArrTensor == nullptr && antiScale1DimNum == DIMS_2) { // scale shape is (G,N)
            scale1GroupNum = antiScale1Shape->GetStorageShape().GetDim(0);
            scale2GroupNum = antiScale2Shape->GetStorageShape().GetDim(0);
            isPerGroup = true;
        } else if (tokensArrTensor != nullptr && antiScale1DimNum == DIMS_3) { // scale shape is (E,G,N)
            scale1GroupNum = antiScale1Shape->GetStorageShape().GetDim(1);
            scale2GroupNum = antiScale2Shape->GetStorageShape().GetDim(1);
            isPerGroup = true;
        }
    }
    tilingData.ffnBaseParams.set_scale1GroupNum(scale1GroupNum);
    tilingData.ffnBaseParams.set_scale2GroupNum(scale2GroupNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FFNTiling::FFNGetQuantScale(const gert::TilingContext *context, const gert::Tensor *tokensArrTensor)
{
    if (context->GetOptionalInputTensor(SCALE_INDEX)) {
        const gert::StorageShape *quantScaleShape = context->GetOptionalInputShape(SCALE_INDEX);
        if (quantScaleShape != nullptr) {
            uint32_t quantScaleDimNum = quantScaleShape->GetStorageShape().GetDimNum();
            uint32_t scaleShape0 = quantScaleShape->GetStorageShape().GetDim(0);
            if (tokensArrTensor == nullptr && scaleShape0 == n1) { // quant scale shape is (N)
                isSmooth = true;
            } else if (tokensArrTensor != nullptr && quantScaleDimNum == DIMS_2) { // quant scale shape is(E,N)
                isSmooth = true;
            }
        } else {
            OP_LOGI(context, "quantScaleShape is nullptr.");
        }
    }
    return ge::GRAPH_SUCCESS;
}

bool FFNTiling::GetTokensIndexFlag(const gert::TilingContext *context) const
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return false;
    }

    const bool *tokensIndexFlagPtr = attrs->GetAttrPointer<bool>(FFN_ATTR_INDEX_TOKENS_INDEX_FLAG);
    if (tokensIndexFlagPtr == nullptr) {
        return false;
    }

    return *tokensIndexFlagPtr;
}

ge::graphStatus FFNTiling::GetInputShape(const gert::TilingContext *context)
{
    const gert::StorageShape *xShape = context->GetInputShape(X_INDEX);
    OP_CHECK_IF(xShape == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "xShape is nullptr"), return ge::GRAPH_FAILED);
    const gert::StorageShape *weight1Shape = context->GetInputShape(WEIGHT1_INDEX);
    OP_CHECK_IF(weight1Shape == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "weight1Shape is nullptr"),
               return ge::GRAPH_FAILED);
    const gert::StorageShape *weight2Shape = context->GetInputShape(WEIGHT2_INDEX);
    OP_CHECK_IF(weight2Shape == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "weight2Shape is nullptr"),
               return ge::GRAPH_FAILED);

    bool isTokensIndex = GetTokensIndexFlag(context);
    uint32_t tokensIndexFlag = static_cast<uint32_t>(isTokensIndex);
    tilingData.ffnBaseParams.set_tokensIndexFlag(tokensIndexFlag);

    // high-dimension input fuses m-axis
    OP_CHECK_IF(GetBs(context, xShape) != ge::GRAPH_SUCCESS,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Get M dim value failed"), return ge::GRAPH_FAILED);
    k1 = xShape->GetStorageShape().GetDim(xShape->GetStorageShape().GetDimNum() - 1);
    auto tokensArrTensor = context->GetOptionalInputTensor(TOKENS_ARR_INDEX);
    if (tokensArrTensor) {
        // origin MOEFFN
        n1 = weight1Shape->GetStorageShape().GetDim(2); // 2: the index of N in weight(E, K, N)
        n2 = weight2Shape->GetStorageShape().GetDim(2); // 2: the index of N in weight(E, K, N)
        expertNum = weight1Shape->GetStorageShape().GetDim(0);
        int64_t tokenNum = tokensArrTensor->GetShapeSize();
        OP_CHECK_IF(
            expertNum != tokenNum || tokenNum > MAX_EXPERT_NUM || expertNum == 0,
            OPS_REPORT_VECTOR_INNER_ERR(
                context->GetNodeName(),
                "Invalid input expert_tokens. ExpertNum in expert_tokens %ld not equal to expertNum %u in weight, "
                "or expertNum %u is larger than max value: %u or equal to 0",
                tokenNum, expertNum, expertNum, MAX_EXPERT_NUM),
            return ge::GRAPH_FAILED);
    } else { // origin FFN
        if (is310P) {
            k1 = xShape->GetStorageShape().GetDim(0) * xShape->GetStorageShape().GetDim(DIMS_3);
            n1 = weight1Shape->GetStorageShape().GetDim(0) * weight1Shape->GetStorageShape().GetDim(DIMS_3);
            n2 = weight2Shape->GetStorageShape().GetDim(0) * weight2Shape->GetStorageShape().GetDim(DIMS_3);
        } else if (isAscend950) {
            // arch35 fused：布局识别统一在 ffn_arch35_tiling.cpp（与 infershape/aclnn 同规则）
            const int64_t w1Dim0 = weight1Shape->GetStorageShape().GetDim(0);
            const int64_t w1Dim1 = weight1Shape->GetStorageShape().GetDim(1);
            const int64_t w2Dim0 = weight2Shape->GetStorageShape().GetDim(0);
            const int64_t w2Dim1 = weight2Shape->GetStorageShape().GetDim(1);
            bool isLinearW = false;
            const bool isSwigluAct =
                tilingData.ffnBaseParams.get_activeType() == static_cast<uint32_t>(ActiveType::SWIGLU);
            if (!FFNArch35Tiling::DetectLayout(context, weight1Shape, weight2Shape,
                                               static_cast<int64_t>(k1), isSwigluAct, isLinearW)) {
                return ge::GRAPH_FAILED;
            }
            if (isLinearW) {
                // linear [N, K]（out, in）
                isFfnTransB = true;
                n1 = static_cast<uint32_t>(w1Dim0);
                n2 = static_cast<uint32_t>(w2Dim0);
            } else {
                // canonical [K, N]
                isFfnTransB = false;
                n1 = static_cast<uint32_t>(w1Dim1);
                n2 = static_cast<uint32_t>(w2Dim1);
            }
        } else {
            n1 = weight1Shape->GetStorageShape().GetDim(1);
            n2 = weight2Shape->GetStorageShape().GetDim(1);
        }
        expertNum = 1;
    }
    return ge::GRAPH_SUCCESS;
}

matmul_tiling::PlatformInfo FFNTiling::MmGetPlatInfo(const FFNCompileInfo *compileInfoPtr) const
{
    matmul_tiling::PlatformInfo platInfo;
    platInfo.socVersion = compileInfoPtr->socVersion;
    platInfo.ubSize = compileInfoPtr->ubSize;
    platInfo.l1Size = compileInfoPtr->l1Size;
    platInfo.l0ASize = compileInfoPtr->l0ASize;
    platInfo.l0BSize = compileInfoPtr->l0BSize;
    platInfo.l0CSize = compileInfoPtr->l0CSize;

    return platInfo;
}

ge::graphStatus FFNTiling::CheckAndGetBasicInfo(gert::TilingContext *context, const FFNCompileInfo *compileInfoPtr)
{
    const uint32_t coreNum = compileInfoPtr->coreNum;
    is310P = compileInfoPtr->socVersion == platform_ascendc::SocVersion::ASCEND310P;
    isAscend950 = compileInfoPtr->socVersion == platform_ascendc::SocVersion::ASCEND950;
    isRegbase = compileInfoPtr->isRegbase;

    OP_CHECK_IF(
        (coreNum == 0 || compileInfoPtr->ubSize == 0 || compileInfoPtr->l1Size == 0 || compileInfoPtr->l0CSize == 0 ||
         compileInfoPtr->l0ASize == 0 || compileInfoPtr->l0BSize == 0),
        OPS_REPORT_VECTOR_INNER_ERR(
            context->GetNodeName(),
            "platform info is invalid, coreNum=%u, ubSize=%lu, l1Size=%lu, l0CSize=%lu, l0ASize=%lu, l0BSize=%lu",
            coreNum, compileInfoPtr->ubSize, compileInfoPtr->l1Size, compileInfoPtr->l0CSize, compileInfoPtr->l0ASize,
            compileInfoPtr->l0BSize),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(FFNParamsCheck(context) != ge::GRAPH_SUCCESS, OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "params is invalid"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetInputShape(context) != ge::GRAPH_SUCCESS,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "get input shape failed"), return ge::GRAPH_FAILED);

    ubSize_ = compileInfoPtr->ubSize - ((expertNum * sizeof(int64_t) + ALIGN32) & ~ALIGN32);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FFNTiling::TilingWithDifferentKN(gert::TilingContext *context, const FFNCompileInfo *compileInfoPtr,
                                                 const uint32_t aicNum)
{
    matmul_tiling::PlatformInfo mmPlatInfo = MmGetPlatInfo(compileInfoPtr);
    ge::graphStatus tilingStatus = ge::GRAPH_FAILED;

    // n1 can be divided to 20 cores at least, BEST_BASEN for each, and the second matrix cannot load to L1 fully
    // In this scenario, tiling might be computed in performance branch
    bool whetherN1K1Satisfy = k1 > BEST_L1_PART2 / ((BEST_BASEN / CONSTANT_TWO) * xDataTypeSize) &&
                              n1 > BEST_BASEN * aicNum / (CONSTANT_TWO * CONSTANT_TWO);
    // cases in 310P will  also be in performance branch
    if ((n1 != 0 && k1 == n2 && n1 == SixteenAlign(n1) && whetherN1K1Satisfy) || is310P) {
        tilingStatus = TilingCalcAndSetting(context, mmPlatInfo, static_cast<matmul_tiling::DataType>(xDataType));
    }

    if (tilingStatus != ge::GRAPH_SUCCESS) {
        FFNSingleCoreTiling(context, ubSize_);
        if (n1 != 0) {
            OP_CHECK_IF(FFNApiTiling(context, mmPlatInfo, static_cast<matmul_tiling::DataType>(xDataType)) !=
                           ge::GRAPH_SUCCESS,
                       OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "run matmul tiling failed"), return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

void FFNTiling::SetTilingBaseParams(gert::TilingContext *context, const FFNCompileInfo *compileInfoPtr,
                                    const uint32_t aicNum)
{
    context->SetBlockDim(compileInfoPtr->blockDim);
    context->SetScheduleMode(1);  // 1: batchmode
    tilingData.ffnBaseParams.set_totalTokens(bs);
    tilingData.ffnBaseParams.set_k1(k1);
    tilingData.ffnBaseParams.set_n1(n1);
    tilingData.ffnBaseParams.set_n2(n2);
    tilingData.ffnBaseParams.set_expertNum(expertNum);
    tilingData.ffnBaseParams.set_coreNum(aicNum);
    tilingData.ffnBaseParams.set_dataTypeSize(xDataTypeSize);
}

ge::graphStatus FFNTiling::RunFusionKernelTiling(gert::TilingContext *context)
{
    Init();
    const FFNCompileInfo *compileInfoPtr = reinterpret_cast<const FFNCompileInfo *>(context->GetCompileInfo());
    OP_CHECK_IF(compileInfoPtr == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "compileInfoPtr is null"),
               return ge::GRAPH_FAILED);

    const uint32_t aicNum = compileInfoPtr->aicCoreNum;
    const uint32_t aivNum = compileInfoPtr->aivCoreNum;

    OP_CHECK_IF(CheckAndGetBasicInfo(context, compileInfoPtr) != ge::GRAPH_SUCCESS,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "run CheckAndGetBasicInfo failed"), return ge::GRAPH_FAILED);
    UpdateMaxTokens();
    CheckMSD();

    if (bs == 0 || k1 == 0 || n2 == 0) {
        context->SetTilingKey(0);
        context->SetBlockDim(0);
        tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
        return ge::GRAPH_SUCCESS;
    }

    SetTilingBaseParams(context, compileInfoPtr, aicNum);

    if (compileInfoPtr->socVersion == platform_ascendc::SocVersion::ASCEND950) {
        FFNArch35TilingParams params{};
        params.activeType = tilingData.ffnBaseParams.get_activeType();
        params.expertNum = expertNum;
        params.bs = bs;
        params.k1 = k1;
        params.n1 = n1;
        params.n2 = n2;
        params.isFfnTransB = isFfnTransB;
        params.xDataType = xDataType;
        params.xDataTypeSize = xDataTypeSize;
        ge::graphStatus fusedRet = FFNArch35Tiling::Tiling(context, compileInfoPtr, tilingData, params);
        if (fusedRet == ge::GRAPH_SUCCESS) {
            return ge::GRAPH_SUCCESS;
        }
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                    "FFN on ascend950 only supports the arch35 fused path "
                                    "(bf16/fp16, gelu/silu/swiglu, single expert, "
                                    "bias1/bias2 both present or both absent)");
        return ge::GRAPH_FAILED;
    }

    if (n1 != 0 && tilingData.ffnBaseParams.get_activeType() >= static_cast<uint32_t>(ActiveType::GEGLU)) {
        OP_CHECK_IF(FFNGlu(context, compileInfoPtr->ubSize, compileInfoPtr->l1Size, compileInfoPtr->l0CSize, aivNum) !=
                       ge::GRAPH_SUCCESS,
                   OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "run FFNGlu failed"), return ge::GRAPH_FAILED);
        PrintFFNTiling(context, true);
        return ge::GRAPH_SUCCESS;
    }
    auto tokensArrTensor = context->GetOptionalInputTensor(TOKENS_ARR_INDEX);
    FFNGetQuantScale(context, tokensArrTensor); // get quant isSmooth info

    OP_CHECK_IF(TilingWithDifferentKN(context, compileInfoPtr, aicNum) != ge::GRAPH_SUCCESS,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "run TilingWithDifferentKN failed"), return ge::GRAPH_FAILED);

    tilingData.ffnBaseParams.set_maxTokens(maxTokens);
    tilingData.mm1TilingData.set_usedCoreNum(aicNum);
    tilingData.mm2TilingData.set_usedCoreNum(aicNum);

    FFNGetScaleGroupNum(context, tokensArrTensor);
    OP_CHECK_IF(FFNSetTilingData(context) != ge::GRAPH_SUCCESS,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "FFNSetTilingData failed"), return ge::GRAPH_FAILED);
    OP_LOGI(context, "RunFusionKernelTiling end");
    return ge::GRAPH_SUCCESS;
}

void FFNTiling::PrintMatMulTiling(const char *opName, TCubeTiling &matmulTiling) const
{
    std::stringstream ss;
    ss << "usedCoreNum " << matmulTiling.get_usedCoreNum() << " M " << matmulTiling.get_M() << " N "
       << matmulTiling.get_N() << " Ka " << matmulTiling.get_Ka() << " Kb " << matmulTiling.get_Kb() << " singleCoreM "
       << matmulTiling.get_singleCoreM() << " singleCoreN " << matmulTiling.get_singleCoreN() << " singleCoreK "
       << matmulTiling.get_singleCoreK() << " baseM " << matmulTiling.get_baseM() << " baseN "
       << matmulTiling.get_baseN() << " baseK " << matmulTiling.get_baseK() << " depthA1 " << matmulTiling.get_depthA1()
       << " depthB1 " << matmulTiling.get_depthB1() << " stepM " << matmulTiling.get_stepM() << " stepN "
       << matmulTiling.get_stepN() << " isBias " << matmulTiling.get_isBias() << " transLength "
       << matmulTiling.get_transLength() << " iterateOrder " << matmulTiling.get_iterateOrder() << " shareMode "
       << matmulTiling.get_shareMode() << " shareL1Size " << matmulTiling.get_shareL1Size() << " shareL0CSize "
       << matmulTiling.get_shareL0CSize() << " shareUbSize " << matmulTiling.get_shareUbSize() << " batchM "
       << matmulTiling.get_batchM() << " batchN " << matmulTiling.get_batchN() << " stepKa "
       << matmulTiling.get_stepKa() << " stepKb " << matmulTiling.get_stepKb() << " dbL0A " << matmulTiling.get_dbL0A()
       << " dbL0B " << matmulTiling.get_dbL0B() << " dbL0C " << matmulTiling.get_dbL0C();
    OP_LOGI(opName, "matmul tiling: %s", ss.str().c_str());
}

void FFNTiling::PrintFFNTiling(const gert::TilingContext *context, bool debugLevel)
{
    if (debugLevel) {
        return;
    }

    PrintMatMulTiling(context->GetNodeName(), tilingData.mm1TilingData);
    PrintMatMulTiling(context->GetNodeName(), tilingData.mm2TilingData);
    std::stringstream ss;
    auto &baseParams = tilingData.ffnBaseParams;
    ss << "totalTokens " << baseParams.get_totalTokens() << " k1 " << baseParams.get_k1() << " n2 "
       << baseParams.get_n2() << " expertNum " << baseParams.get_expertNum() << " maxTokens "
       << baseParams.get_maxTokens() << " coreNum " << baseParams.get_coreNum() << " activeType "
       << baseParams.get_activeType() << " workspace1Size " << baseParams.get_workspace1Size() << " workspace2Size "
       << baseParams.get_workspace2Size() << " syncWorkspaceSize " << baseParams.get_syncWorkspaceSize()
       << " dataTypeSize " << baseParams.get_dataTypeSize();
    OP_LOGD(context->GetNodeName(), "ffnBaseParams: %s", ss.str().c_str());

    auto &singleCoreParams = tilingData.ffnSingleCoreParams;
    std::stringstream ss1;
    ss1 << "baseM1 " << singleCoreParams.get_baseM1() << " baseN1 " << singleCoreParams.get_baseN1() << " baseN2 "
        << singleCoreParams.get_baseN2() << " ubCalSize " << singleCoreParams.get_ubCalSize() << " ubRestBytes "
        << singleCoreParams.get_ubRestBytes() << " mm1ResUbSize " << singleCoreParams.get_mm1ResUbSize() << " tiling "
        << context->GetTilingKey();
    OP_LOGD(context->GetNodeName(), "ffnSingleCoreParams: %s", ss1.str().c_str());
}

void FFNTiling::PrintCriticalInfo(gert::TilingContext *context)
{
    std::stringstream ss;
    ss << "maxTokens  " << maxTokens << "; expertTokens [";
    if (expertNum == 1) {
        ss << bs;
    } else {
        ss << "unknown";
    }
    size_t *workspaces = context->GetWorkspaceSizes(1); // workspaceSize is at index 1
    ss << "]; workspace = " << workspaces[0];
    OP_LOGD(context->GetNodeName(), "Critical Info for kernel: %s.", ss.str().c_str());
}

/*
 * @brief FFN params check.
 * @param context gert::TilingContext
 * The op struct need to adapt is shown as follows:
 * 1) has experts
 *    x(M,K1)  weight1(E,K1,N1)  bias1(E,N1)
 *          \           /            /
 *             MM1(M,N1)
 *                 |
 *            active_layer  weight1(E,K2,N2)  bias2(E,N2)
 *                 |          /                /
 *           MM2(M,N2), N2=K1
 *
 * 2) no experts
 *    x(M,K1)  weight1(K1,N1)  bias1(N1)
 *          \           /            /
 *             MM1(M,N1)
 *                 |
 *            active_layer  weight1(K2,N2)  bias2(N2)
 *                 |          /                /
 *           MM2(M,N2), N2=K1
 *  Notice: E/bias may or may not be. The output shape is equal to x's.
 * @return ge::STATUS.
 */
ge::graphStatus FFNTiling::FFNParamsCheck(gert::TilingContext *context)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char *activeType = attrs->GetAttrPointer<char>(FFN_ATTR_INDEX_ACTIVATION);
    OP_CHECK_NULL_WITH_CONTEXT(context, activeType);
    ActiveType activationType = GetActiveType(context, activeType);
    OP_CHECK_IF(activationType == ActiveType::INVALID_TYPE,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "activeType does not match any of the preset types"),
               return ge::GRAPH_FAILED);
    tilingData.ffnBaseParams.set_activeType(static_cast<uint32_t>(activationType));

    OP_CHECK_IF(DataTypeCheck(context) != ge::GRAPH_SUCCESS || FormatCheck(context) != ge::GRAPH_SUCCESS,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "the data type and format is invalid."), return ge::GRAPH_FAILED);
    weightDataTypeSize = GetSizeByDataType(weight1Dtype);
    innerPrecise = HIGH_PRECISION;
    const int64_t *innerPrecisePtr = attrs->GetAttrPointer<int64_t>(FFN_ATTR_INDEX_INNER_PRECISE);
    if (innerPrecisePtr != nullptr) {
        innerPrecise = static_cast<int32_t>(*innerPrecisePtr);
    }
    OP_LOGI(context, "Inner_precise is %d.", innerPrecise);

    OP_CHECK_IF(innerPrecise != HIGH_PRECISION && innerPrecise != HIGH_PERFORMANCE,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Invalid innerPrecise. Attr inner_precise only support 0/1."),
               return ge::GRAPH_FAILED);

    OP_CHECK_IF(innerPrecise != HIGH_PERFORMANCE && activationType == ActiveType::GEGLU,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Invalid innerPrecise. GEGLU only support high performance."),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FFNTiling::FFNSetUbDivideBlk()
{
    ubBlockAlign = UB_PER_BLOCK_ALIGN;
    if (xDataType == ge::DT_FLOAT16 && (weight1Dtype == ge::DT_INT8 || weight1Dtype == ge::DT_INT4)) {
        ubDivideBlkNum = UB_ANTIQUANT_BLOCK_NUM_FP16;
        ubIoBlkNum = UB_ANTIQUANT_IO_USED_BLOCK_FP16;
        ubBlockAlign = UB_ANTIQUANT_PER_BLOCK_ALIGN_FP16;
        return ge::GRAPH_SUCCESS;
    } else if (xDataType == ge::DT_BF16 && (weight1Dtype == ge::DT_INT8 || weight1Dtype == ge::DT_INT4)) {
        ubDivideBlkNum = UB_ANTIQUANT_BLOCK_NUM_BP16;
        ubIoBlkNum = UB_ANTIQUANT_IO_USED_BLOCK_BP16;
        return ge::GRAPH_SUCCESS;
    } else if ((xDataType == ge::DT_FLOAT16 || xDataType == ge::DT_BF16) && innerPrecise == HIGH_PRECISION) {
        ubDivideBlkNum = UB_PRECISION_BLOCK_NUM_FP16;
        ubIoBlkNum = UB_PRECISION_IO_USED_BLOCK_FP16;
        return ge::GRAPH_SUCCESS;
    } else if (xDataType == ge::DT_FLOAT16 && weight1Dtype == ge::DT_FLOAT16) {
        ubDivideBlkNum = UB_PEFORMENCE_BLOCK_NUM_FP16;
        ubIoBlkNum = UB_PEFORMENCE_IO_USED_BLOCK_FP16;
        if (is310P) {
            ubBlockAlign = UB_PER_BLOCK_ALIGN_310P;
        }
        return ge::GRAPH_SUCCESS;
    } else if (xDataType == ge::DT_INT8) {
        if (outputDtype == ge::DT_BF16) {
            isQuantBf16 = true;
            ubDivideBlkNum = UB_QUANT_BLOCK_NUM_BF16_OUT;
            ubIoBlkNum = UB_QUANT_IO_BLOCK_NUM_BF16_OUT;
            return ge::GRAPH_SUCCESS;
        } else if (outputDtype == ge::DT_FLOAT16) {
            ubDivideBlkNum = UB_QUANT_BLOCK_NUM_FP16_OUT;
            ubIoBlkNum = UB_QUANT_IO_BLOCK_NUM_FP16_OUT;
            return ge::GRAPH_SUCCESS;
        } else {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus FFNTiling::FFNCalUbSize(uint32_t baseN, uint32_t divideBlkNum, uint32_t ioBlkNum, uint32_t &baseM)
{
    if (divideBlkNum == 0 || baseN == 0 || ubBlockAlign == 0) {
        return ge::GRAPH_FAILED;
    }
    ubCalSize = ubSize_ / divideBlkNum;
    ubCalSize = ubCalSize / ubBlockAlign * ubBlockAlign; // 16k/8k/4k align
    ubRestBytes = ubSize_ - ubCalSize * ioBlkNum;
    ubRestBytes = ubRestBytes / UB_BLOCK_UNIT_SIZE * UB_BLOCK_UNIT_SIZE; // 32B align
    if (ubCalSize == 0 || ubRestBytes == 0) {
        return ge::GRAPH_FAILED;
    }
    baseM = ubCalSize / baseN;   // activate function baseM
    baseM = SixteenAlign(baseM);
    tilingData.ffnSingleCoreParams.set_ubCalSize(ubCalSize);
    if (isMsdCase) {
        tilingData.ffnSingleCoreParams.set_ubRestBytes(ubSize_ + ((expertNum * sizeof(int64_t) + ALIGN32) & ~ALIGN32));
    } else {
        tilingData.ffnSingleCoreParams.set_ubRestBytes(ubRestBytes);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FFNTiling::CalMM1ValidUbBytes(int64_t &mm1VaildUbBytes) const
{
    if (innerPrecise == HIGH_PRECISION) {
        mm1VaildUbBytes = ubSize_ / UB_DIVIDE_NUM_HIGH_PRECISION;
    }
    if (xDataType == ge::DT_INT8) {
        if (outputDtype == ge::DT_BF16) {
            mm1VaildUbBytes = ubSize_ / UB_DIVIDE_NUM_QUANT_BF16;
        } else if (deqscaleDtype == ge::DT_FLOAT) {
            mm1VaildUbBytes = ubSize_ / UB_DIVIDE_NUM_QUANT_DEQ_FLOAT32;
        } else {
            mm1VaildUbBytes = ubSize_ / UB_DIVIDE_NUM_QUANT;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FFNTiling::CalMM1BaseM(const gert::TilingContext *context, const uint32_t baseN, const uint64_t l0CSize,
                                       const int64_t mm1VaildUbBytes, uint32_t &baseM)
{
    if (baseN == 0) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(FFNCalUbSize(baseN, ubDivideBlkNum, ubIoBlkNum, baseM) != ge::GRAPH_SUCCESS,
               OP_LOGW(context, "calculate ub failed."), return ge::GRAPH_FAILED);

    uint32_t maxBaseM = std::min<uint32_t>(SixteenAlign(maxTokens), MAX_BASEM);
    baseM = std::min<uint32_t>(mm1VaildUbBytes / baseN, baseM);

    // calculate cube baseM by l0c size
    baseM = std::min<uint32_t>(l0CSize / (baseN * FP32_DATATYPE_SIZE), baseM);
    baseM = std::min<uint32_t>(maxBaseM, baseM);
    baseM = SixteenAlign(baseM); // align down
    if (baseM == 0) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FFNTiling::CalMM1TilingBaseMNBasicBlock(const gert::TilingContext *context, const uint64_t l0CSize,
                                                        const int64_t mm1VaildUbBytes, uint64_t basicBlkOperTimes,
                                                        uint32_t &baseN)
{
    uint32_t coreNum = tilingData.ffnBaseParams.get_coreNum();
    uint32_t nCal = n1 - 1;
    uint32_t mCal = maxTokens - 1;
    uint32_t baseM = MATMUL_MIN_SHAPE;
    float lastRatio = 0.f;
    for (uint32_t tmpBaseN = baseN; tmpBaseN >= SMALL_TOKEN_BOUND; tmpBaseN -= TINY_TOKEN_BOUND) {
        OP_CHECK_IF(CalMM1BaseM(context, tmpBaseN, l0CSize, mm1VaildUbBytes, baseM) != ge::GRAPH_SUCCESS,
                   OP_LOGW(context, "calculate mm1 baseM failed."), return ge::GRAPH_FAILED);
        uint32_t blockDimN = Ceil(n1, tmpBaseN);
        uint64_t curBasicBlkOperTimes = Ceil(Ceil(maxTokens, baseM) * blockDimN, coreNum) * coreNum;
        if (curBasicBlkOperTimes < basicBlkOperTimes) {
            basicBlkOperTimes = curBasicBlkOperTimes;
            lastRatio = 0.0;
            baseM1_ = baseM;
            baseN = tmpBaseN;
        }
        float nRatio = nCal % tmpBaseN * 1.0f / tmpBaseN;
        for (uint32_t tmpBaseM = baseM; tmpBaseM >= SMALL_TOKEN_BOUND; tmpBaseM -= TINY_TOKEN_BOUND) {
            uint64_t blockDim = Ceil(maxTokens, tmpBaseM) * blockDimN;
            if (blockDim > basicBlkOperTimes) {
                break;
            }
            float curRatio = mCal % tmpBaseM * nRatio / tmpBaseM;
            if (blockDim == basicBlkOperTimes && curRatio > lastRatio) {
                lastRatio = curRatio;
                baseM1_ = tmpBaseM;
                baseN = tmpBaseN;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FFNTiling::CalMMTilingBaseMNBasicBlock(const uint64_t basicBlkOperTimes, const uint32_t n,
                                                       uint32_t &baseM, uint32_t &baseN) const
{
    bool isRecomputed = false; // variable indicates whether baseM/baseN is recomputed
    uint32_t blockDimM = Ceil(maxTokens, baseM);
    uint32_t blockDimN = Ceil(n, baseN);
    uint64_t blockDim = blockDimM * blockDimN;
    if (blockDimM == 1) {
        baseN = Ceil(n, basicBlkOperTimes);
        baseN = SixteenAlignUp(baseN);
        isRecomputed = true;
    } else if (blockDimN == 1) {
        baseM = Ceil(maxTokens, basicBlkOperTimes);
        baseM = SixteenAlignUp(baseM);
        isRecomputed = true;
    }
    blockDim = Ceil(maxTokens, baseM) * Ceil(n, baseN);
    if (blockDim >= basicBlkOperTimes && isRecomputed) {
        return ge::GRAPH_SUCCESS;
    }

    uint32_t initialBaseN = baseN;
    uint32_t nCal = n - 1;
    uint32_t mCal = maxTokens - 1;
    float lastRatio = 0.f;
    for (uint32_t tmpBaseM = baseM; tmpBaseM >= TINY_TOKEN_BOUND; tmpBaseM -= TINY_TOKEN_BOUND) {
        float mRatio = mCal % tmpBaseM * 1.0f / tmpBaseM;
        blockDimM = Ceil(maxTokens, tmpBaseM);
        for (uint32_t tmpBaseN = initialBaseN; tmpBaseN >= TINY_TOKEN_BOUND; tmpBaseN -= TINY_TOKEN_BOUND) {
            blockDim = blockDimM * Ceil(n, tmpBaseN);
            if (blockDim > basicBlkOperTimes) {
                break;
            }
            float curRatio = nCal % tmpBaseN * mRatio / tmpBaseN;
            if (blockDim == basicBlkOperTimes && curRatio > lastRatio) {
                lastRatio = curRatio;
                baseM = tmpBaseM;
                baseN = tmpBaseN;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FFNTiling::CalMM1TilingBaseMNKBasicBlock(const gert::TilingContext *context,
                                                         const matmul_tiling::PlatformInfo &platInfo)
{
    int64_t mm1VaildUbBytes = ubSize_ / (UB_DIVIDE_NUM * xDataTypeSize);
    uint32_t coreNum = tilingData.ffnBaseParams.get_coreNum();
    uint32_t baseN = std::min<uint32_t>(SixteenAlign(n1), BEST_BASEN);

    OP_CHECK_IF(FFNSetUbDivideBlk() != ge::GRAPH_SUCCESS,
               OP_LOGW(context, "set ub device block failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CalMM1BaseM(context, baseN, platInfo.l0CSize, mm1VaildUbBytes, baseM1_) != ge::GRAPH_SUCCESS,
               OP_LOGW(context, "calculate mm1 baseM failed."), return ge::GRAPH_FAILED);
    uint32_t blockDimM = Ceil(maxTokens, baseM1_);
    uint64_t blockDim = blockDimM * Ceil(n1, baseN);
    uint64_t basicBlkOperTimes = Ceil(blockDim, coreNum) * coreNum;
    if (blockDim > coreNum) {
        if (blockDimM == 1) {
            baseN = Ceil(n1, basicBlkOperTimes);
            baseN = SixteenAlignUp(baseN);
        } else if (baseN < baseM1_) {
            baseM1_ = Ceil(maxTokens, basicBlkOperTimes);
            baseM1_ = SixteenAlignUp(baseM1_);
        } else {
            OP_CHECK_IF(CalMM1TilingBaseMNBasicBlock(context, platInfo.l0CSize, mm1VaildUbBytes, basicBlkOperTimes,
                                                    baseN) != ge::GRAPH_SUCCESS,
                       OP_LOGW(context, "mm1 calculate baseMN failed."), return ge::GRAPH_FAILED);
        }
    } else {
        OP_CHECK_IF(CalMMTilingBaseMNBasicBlock(basicBlkOperTimes, n1, baseM1_, baseN) != ge::GRAPH_SUCCESS,
                   OP_LOGW(context, "mm1 calculate baseMN failed."), return ge::GRAPH_FAILED);
    }

    // calculate baseK in considering l0B double buffer, so divide 2
    baseK1_ = (platInfo.l0BSize / CONSTANT_TWO) / (baseN * xDataTypeSize);
    baseK1_ = std::min<uint32_t>((platInfo.l0ASize / CONSTANT_TWO) / (baseM1_ * xDataTypeSize), baseK1_);
    baseK1_ = SixteenAlign(baseK1_);
    if (baseK1_ == 0) {
        return ge::GRAPH_FAILED;
    }

    mm1VaildUbBytes = baseM1_ * baseN;
    tilingData.ffnSingleCoreParams.set_baseM1(baseM1_);
    tilingData.ffnSingleCoreParams.set_baseN1(baseN);
    tilingData.ffnSingleCoreParams.set_mm1ResUbSize(mm1VaildUbBytes);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FFNTiling::CalMM1TilingBaseMNK(const gert::TilingContext *context,
                                               const matmul_tiling::PlatformInfo &platInfo)
{
    int64_t mm1VaildUbBytes = ubSize_ / UB_DIVIDE_NUM;
    CalMM1ValidUbBytes(mm1VaildUbBytes);
    uint32_t baseN = MATMUL_MIN_SHAPE;
    if (isMsdCase) {
        baseN = BEST_BASEN_MSD;
        xDataTypeSize = 1;
    } else {
        baseN = BEST_BASEN;
    }
    // calculate baseK in considering l0B double buffer, so divide 2
    baseK1_ = (platInfo.l0BSize / CONSTANT_TWO) / (baseN * xDataTypeSize);
    baseK1_ = SixteenAlign(baseK1_); // align down
    if (baseK1_ == 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t maxBaseM = platInfo.l0CSize / (baseN * FP32_DATATYPE_SIZE);
    // calculate baseM in considering l0A double buffer, so divide 2
    baseM1_ = std::min<uint32_t>((platInfo.l0ASize / CONSTANT_TWO) / (baseK1_ * xDataTypeSize), maxBaseM);
    if (maxTokens <= TINY_TOKEN_BOUND) {
        baseM1_ = TINY_TOKEN_BOUND;
    } else if (maxTokens <= SMALL_TOKEN_BOUND) {
        baseM1_ = SMALL_TOKEN_BOUND;
    }
    baseM1_ = SixteenAlign(baseM1_);
    if (baseM1_ == 0) {
        return ge::GRAPH_FAILED;
    }
    // calculate vector baseM by ub size
    uint32_t baseM = mm1VaildUbBytes / (baseN * xDataTypeSize);
    if (FFNSetUbDivideBlk() == ge::GRAPH_SUCCESS) {
        OP_CHECK_IF(FFNCalUbSize(baseN, ubDivideBlkNum, ubIoBlkNum, baseM) != ge::GRAPH_SUCCESS,
                   OP_LOGW(context, "calculate ub failed."), return ge::GRAPH_FAILED);
    }
    baseM = SixteenAlign(baseM);
    if (baseM == 0) {
        return ge::GRAPH_FAILED;
    }
    // baseM in cube can be divided by baseM in ub
    uint32_t cubeMFactor = baseM1_ / SIXTEEN_ALIGN_CONSTANT;
    uint32_t vectorMFactor = baseM / SIXTEEN_ALIGN_CONSTANT;
    for (uint32_t i = vectorMFactor; i >= 1; --i) {
        if (cubeMFactor % i == 0) {
            baseM = i * SIXTEEN_ALIGN_CONSTANT;
            break;
        }
    }
    mm1VaildUbBytes = baseN * baseM; // numbers in UB, not bytes
    tilingData.ffnSingleCoreParams.set_baseM1(baseM);
    tilingData.ffnSingleCoreParams.set_baseN1(baseN);
    tilingData.ffnSingleCoreParams.set_mm1ResUbSize(mm1VaildUbBytes);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FFNTiling::CalMM2TilingBaseMNKBasicBlock(const gert::TilingContext *context,
                                                         const matmul_tiling::PlatformInfo &platInfo)
{
    uint32_t coreNum = tilingData.ffnBaseParams.get_coreNum();
    uint32_t baseN = std::min<uint32_t>(SixteenAlign(n2), BEST_BASEN);

    uint32_t baseM = std::min<uint32_t>(maxTokens, platInfo.l0CSize / (baseN * FP32_DATATYPE_SIZE));
    baseM = std::min<uint32_t>(SixteenAlign(baseM), MAX_BASEM);
    uint64_t blockDim = Ceil(maxTokens, baseM) * Ceil(n2, baseN);
    uint64_t basicBlkOperTimes = Ceil(blockDim, coreNum) * coreNum;
    OP_CHECK_IF(CalMMTilingBaseMNBasicBlock(basicBlkOperTimes, n2, baseM, baseN) != ge::GRAPH_SUCCESS,
               OP_LOGW(context, "mm2 calculate baseMN failed."), return ge::GRAPH_FAILED);

    // calculate baseK in considering l0B double buffer, so divide 2
    baseK2_ = (platInfo.l0BSize / CONSTANT_TWO) / (baseN * xDataTypeSize);
    baseK2_ = std::min<uint32_t>((platInfo.l0ASize / CONSTANT_TWO) / (baseM * xDataTypeSize), baseK2_);
    baseK2_ = SixteenAlign(baseK2_);
    if (baseK2_ == 0) {
        return ge::GRAPH_FAILED;
    }

    mm2BaseM = baseM;
    tilingData.ffnSingleCoreParams.set_baseN2(baseN);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FFNTiling::CalMM2TilingBaseMNK(const matmul_tiling::PlatformInfo &platInfo)
{
    uint32_t baseN = MATMUL_MIN_SHAPE;
    if (isMsdCase) {
        baseN = BEST_BASEN_MSD;
        xDataTypeSize = 1;
    } else {
        baseN = BEST_BASEN;
    }
    if (tilingData.ffnBaseParams.get_n2() < baseN) {
        return ge::GRAPH_FAILED;
    }
    // calculate baseK in considering l0B double buffer, so divide 2
    baseK2_ = (platInfo.l0BSize / CONSTANT_TWO) / (baseN * xDataTypeSize);
    baseK2_ = SixteenAlign(baseK2_); // align down
    if (baseK2_ == 0) {
        return ge::GRAPH_FAILED;
    }
    // calculate baseM in considering l0A double buffer, so l0ASize divide 2
    uint32_t maxBaseM = platInfo.l0CSize / (baseN * FP32_DATATYPE_SIZE);
    uint32_t baseM = std::min<uint32_t>((platInfo.l0ASize / CONSTANT_TWO) / (baseK2_ * xDataTypeSize), maxBaseM);
    if (maxTokens <= TINY_TOKEN_BOUND) {
        baseM = TINY_TOKEN_BOUND;
    } else if (maxTokens <= SMALL_TOKEN_BOUND) {
        baseM = SMALL_TOKEN_BOUND;
    }
    baseM = SixteenAlign(baseM);
    if (baseM == 0) {
        return ge::GRAPH_FAILED;
    }
    mm2BaseM = baseM;
    tilingData.ffnSingleCoreParams.set_baseN2(baseN);
    return ge::GRAPH_SUCCESS;
}

void FFNTiling::SetBiasInfo(const gert::TilingContext *context, matmul_tiling::MatmulApiTiling &mm,
                            const uint32_t &irIndex) const
{
    auto bias = context->GetOptionalInputDesc(irIndex);
    if (bias == nullptr) {
        mm.SetBias(false);
    } else {
        mm.SetBias(true);
        mm.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                       static_cast<matmul_tiling::DataType>(bias->GetDataType()));
    }
}

void FFNTiling::FFNCalMMStep(const uint32_t baseM, const uint32_t baseN, const uint32_t baseK,
                             TCubeTiling &mmTilingData)
{
    // whether enable double buffer in L1B
    bool divTwo = (maxTokens <= SMALL_TOKEN_BOUND && expertNum > 1);
    uint32_t bestL1Part1 = BEST_L1_PART1;
    uint32_t bestL1Part2 = BEST_L1_PART2;

    if (is310P) {
        divTwo = 1;
        bestL1Part1 = BEST_L1_PART_310P;
        bestL1Part2 = BEST_L1_PART_310P;
    }
    uint32_t mmStepM = 1;
    uint32_t mmStepN = 1;
    uint32_t mmStepKa = (bestL1Part1 >> 1) / (baseM * baseK * xDataTypeSize);
    uint32_t mmStepKb = (bestL1Part2 >> static_cast<uint32_t>(divTwo)) / (baseN * baseK * xDataTypeSize);
    if (mmStepKa > mmStepKb) {
        mmStepKa = mmStepKa / mmStepKb * mmStepKb;
    } else if (mmStepKa < mmStepKb) {
        mmStepKb = mmStepKb / mmStepKa * mmStepKa;
    }
    if (xDataType == ge::DT_INT8 && maxTokensCheckOpt > SMALL_TOKEN_BOUND) {
        mmStepKb = mmStepKb / CONSTANT_TWO;
    }
    uint32_t mmDepthA1 = mmStepKa << 1;
    uint32_t mmDepthB1 = mmStepKb << 1;
    if (xDataType == ge::DT_INT8 && (!isSmooth) && outputDtype == ge::DT_FLOAT16 && deqscaleDtype == ge::DT_UINT64 &&
        maxTokens <= SMALL_TOKEN_BOUND && expertNum > 1 && tilingData.ffnBaseParams.get_n1() % BEST_BASEN == 0 &&
        tilingData.ffnBaseParams.get_n2() % BEST_BASEN == 0) {
        mmStepKb = mmStepKb / CONSTANT_TWO;
        mmStepN = CONSTANT_TWO;
        mmTilingData.set_iterateOrder(1);
    }

    mmTilingData.set_stepKa(mmStepKa);
    mmTilingData.set_depthA1(mmDepthA1);
    mmTilingData.set_stepKb(mmStepKb);
    mmTilingData.set_depthB1(mmDepthB1);
    mmTilingData.set_stepN(mmStepN);
    mmTilingData.set_stepM(mmStepM);

    // xDataTypeSize is modified in msd-branch, so get the right value again
    xDataTypeSize = GetSizeByDataType(xDataType);
}

ge::graphStatus FFNTiling::FFNSetMM1Tiling(const gert::TilingContext *context,
                                           const matmul_tiling::PlatformInfo &platInfo,
                                           matmul_tiling::DataType matmulDtype)
{
    matmul_tiling::MatmulApiTiling mm1(platInfo);
    uint32_t baseN1 = tilingData.ffnSingleCoreParams.get_baseN1();

    mm1.SetBType(matmul_tiling::TPosition::GM, wFormat, matmulDtype, false);
    if (is310P) {
        mm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ, matmulDtype, false);
        mm1.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::NZ,
                     matmul_tiling::DataType::DT_FLOAT16);
    } else {
        mm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmulDtype, false);
        mm1.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND_ALIGN,
                     matmul_tiling::DataType::DT_FLOAT16);
    }
    SetBiasInfo(context, mm1, BIAS1_INDEX);

    mm1.SetOrgShape(maxTokens, tilingData.ffnBaseParams.get_n1(), tilingData.ffnBaseParams.get_k1());
    mm1.SetShape(maxTokens, baseN1, tilingData.ffnBaseParams.get_k1());
    if (!isMsdCase) {
        mm1.SetFixSplit(std::min(baseM1_, maxTokens), baseN1);
    }
    mm1.SetBufferSpace(platInfo.l1Size, platInfo.l0CSize, tilingData.ffnSingleCoreParams.get_mm1ResUbSize());
    if (mm1.GetTiling(tilingData.mm1TilingData) == -1) {
        return ge::GRAPH_FAILED;
    }
    tilingData.mm1TilingData.set_shareMode(0);
    tilingData.mm1TilingData.set_shareL1Size(platInfo.l1Size);
    tilingData.mm1TilingData.set_shareL0CSize(platInfo.l0CSize);
    tilingData.mm1TilingData.set_dbL0C(1); // disable l0c double buffer
    tilingData.mm1TilingData.set_baseM(baseM1_);
    tilingData.mm1TilingData.set_baseN(baseN1);
    tilingData.mm1TilingData.set_baseK(baseK1_);
    FFNCalMMStep(baseM1_, baseN1, baseK1_, tilingData.mm1TilingData);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FFNTiling::FFNSetMM2Tiling(const gert::TilingContext *context,
                                           const matmul_tiling::PlatformInfo &platInfo,
                                           matmul_tiling::DataType matmulDtype)
{
    matmul_tiling::MatmulApiTiling mm2(platInfo);
    uint32_t baseM2 = mm2BaseM;
    uint32_t baseN2 = tilingData.ffnSingleCoreParams.get_baseN2();
    mm2.SetAType(matmul_tiling::TPosition::GM, xFormat, matmulDtype, false);
    mm2.SetBType(matmul_tiling::TPosition::GM, wFormat, matmulDtype, false);
    mm2.SetCType(matmul_tiling::TPosition::GM, yFormat, matmul_tiling::DataType::DT_FLOAT16);
    SetBiasInfo(context, mm2, BIAS2_INDEX);

    mm2.SetOrgShape(maxTokens, tilingData.ffnBaseParams.get_n2(), tilingData.ffnBaseParams.get_n1());
    mm2.SetShape(maxTokens, baseN2, tilingData.ffnBaseParams.get_n1());
    if (!isMsdCase) {
        mm2.SetFixSplit(std::min(baseM2, maxTokens), baseN2);
    }
    mm2.SetBufferSpace(platInfo.l1Size, platInfo.l0CSize);
    if (mm2.GetTiling(tilingData.mm2TilingData) == -1) {
        return ge::GRAPH_FAILED;
    }
    tilingData.mm2TilingData.set_shareMode(0);
    tilingData.mm2TilingData.set_shareL1Size(platInfo.l1Size);
    tilingData.mm2TilingData.set_shareL0CSize(platInfo.l0CSize);
    tilingData.mm2TilingData.set_dbL0C(1); // disable l0c double buffer
    tilingData.mm2TilingData.set_baseM(baseM2);
    tilingData.mm2TilingData.set_baseN(baseN2);
    tilingData.mm2TilingData.set_baseK(baseK2_);
    FFNCalMMStep(baseM2, baseN2, baseK2_, tilingData.mm2TilingData);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FFNTiling::FFNGlu(gert::TilingContext *context, uint64_t ubSize, uint64_t l1Size, uint64_t l0CSize,
                                  uint32_t aivNum)
{
    OP_CHECK_IF(FFNGluParamsCheck(context) != ge::GRAPH_SUCCESS,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "FFN Glu param is invalid"), return ge::GRAPH_FAILED);

    FFNGluCalMM1Tiling(ubSize, l0CSize);
    OP_CHECK_IF(FFNGluSetMM1Tiling(context, l1Size, l0CSize, static_cast<matmul_tiling::DataType>(xDataType), aivNum) !=
                   ge::GRAPH_SUCCESS,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "FFN Glu set mm1 tiling failed"), return ge::GRAPH_FAILED);
    FFNGluCalMM2Tiling(l0CSize);
    OP_CHECK_IF(FFNGluSetMM2Tiling(context, l1Size, l0CSize, static_cast<matmul_tiling::DataType>(xDataType)) !=
                   ge::GRAPH_SUCCESS,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "FFN Glu set mm2 tiling failed"), return ge::GRAPH_FAILED);

    context->SetTilingKey(2); // 2: for glu template
    size_t *workspaces = context->GetWorkspaceSizes(1);
    auto workspace1Size = baseM1_ * baseN1_ * xDataTypeSize * 4 * aivNum; // 4: pingpong buffer and left/right part
    auto workspace2Size = maxTokens * n1 / 2 * xDataTypeSize; // 2: dim in n1 should be divided by 2
    tilingData.ffnBaseParams.set_workspace1Size(workspace1Size);
    tilingData.ffnBaseParams.set_workspace2Size(workspace2Size);
    workspaces[0] = workspace1Size + workspace2Size + SYS_WORKSPACE_SIZE;
    // Check tiling data size not greater than capacity
    OP_CHECK_IF(tilingData.GetDataSize() > context->GetRawTilingData()->GetCapacity(),
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "actual tiling data size %zu > context tiling data size %zu",
                                           tilingData.GetDataSize(), context->GetRawTilingData()->GetCapacity()),
               return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FFNTiling::FFNGluParamsCheck(const gert::TilingContext *context) const
{
    // n1 should be a even number
    OP_CHECK_IF((n1 % 2 != 0),
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "the glu activation function only supports n1 is even"),
               return ge::GRAPH_FAILED);

    OP_CHECK_IF((xDataType != ge::DT_FLOAT16),
               OPS_REPORT_VECTOR_INNER_ERR(
                   context->GetNodeName(), "the glu activation function only supports the data type is float16, the dtype is %s",
                   TypeUtils::DataTypeToSerialString(xDataType).c_str()),
               return ge::GRAPH_FAILED);
    // only supported in no-expert scenario
    bool hasExperts = context->GetOptionalInputTensor(TOKENS_ARR_INDEX) != nullptr;
    OP_CHECK_IF(
        hasExperts,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "the glu activation function only supports the scene without experts"),
        return ge::GRAPH_FAILED);
    // k2 should be equal to n1/2
    const gert::StorageShape *weight2Shape = context->GetInputShape(WEIGHT2_INDEX);
    size_t kIndex = hasExperts ? 1 : 0;
    auto k2 = weight2Shape->GetStorageShape().GetDim(kIndex);
    OP_CHECK_IF((n1 / 2 != k2),
               OPS_REPORT_VECTOR_INNER_ERR(
                   context->GetNodeName(), "the glu activation function only supports k2(%ld) is equal to n1(%u) / 2", k2, n1),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FFNTiling::SetMMTilingType(const gert::TilingContext *context, bool isMM1,
                                           matmul_tiling::DataType matmulDtype,
                                           matmul_tiling::MatmulApiTiling &mm) const
{
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmulDtype, false);
    mm.SetBType(matmul_tiling::TPosition::GM, wFormat, matmulDtype, false);
    uint32_t biasIndex;
    if (isMM1) {
        mm.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_FLOAT16);
        biasIndex = BIAS1_INDEX;
    } else {
        mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
        biasIndex = BIAS2_INDEX;
    }
    auto bias = context->GetOptionalInputDesc(biasIndex);
    if (bias == nullptr) {
        mm.SetBias(false);
    } else {
        mm.SetBias(true);
        mm.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                       static_cast<matmul_tiling::DataType>(bias->GetDataType()));
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FFNTiling::FFNGluCalMM1Tiling(uint64_t ubSize, uint64_t l0CSize)
{
    int64_t mm1VaildUbBytes = ubSize / GLU_UB_DIVIDE_NUM_FP16;
    uint32_t tempN1 = n1 / 2; // 2: n1 dim should divide 2

    uint32_t baseN = BEST_BASEN;
    uint32_t baseM;
    while (tempN1 < baseN) {
        baseN = baseN >> 1;
    }
    if (baseN < MATMUL_MIN_SHAPE) {
        baseN = MATMUL_MIN_SHAPE;
    }
    // calculate vector baseM by ub size
    uint32_t maxBaseM = mm1VaildUbBytes / (baseN * xDataTypeSize);
    baseM = std::min(MAX_UB_BLOCK / baseN, maxBaseM);
    while (maxTokens < baseM) {
        baseM = baseM >> 1;
    }
    if (baseM < MATMUL_MIN_SHAPE) {
        baseM = MATMUL_MIN_SHAPE;
    }

    // cube basic block size should be a multiple of vector basic block
    uint32_t aicAIVFactor = 4;
    if (baseM * aicAIVFactor <= maxTokens) {
        baseM1_ = baseM * aicAIVFactor;
        baseN1_ = baseN;
        // basic block size cannot exceed l0c size
        maxBaseM = l0CSize / (baseN1_ * FP32_DATATYPE_SIZE);
        if (baseN1_ == BEST_BASEN && maxBaseM < maxTokens) {
            baseM1_ = maxBaseM;
        }
    } else {
        baseM1_ = baseM;
        baseN1_ = baseN;
    }

    mm1VaildUbBytes = baseM * baseN;
    tilingData.ffnSingleCoreParams.set_baseM1(baseM);
    tilingData.ffnSingleCoreParams.set_baseN1(baseN);
    tilingData.ffnSingleCoreParams.set_mm1ResUbSize(mm1VaildUbBytes);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FFNTiling::FFNGluSetMM1Tiling(gert::TilingContext *context, uint64_t l1Size, uint64_t l0CSize,
                                              matmul_tiling::DataType matmulDtype, uint32_t aivNum)
{
    matmul_tiling::MatmulApiTiling mm1;
    SetMMTilingType(context, true, matmulDtype, mm1);

    uint32_t tilingCoreNum = aivNum;
    uint32_t tempN1 = n1 / 2; // 2: n1 dim should divide 2

    uint32_t n1Loops = (tempN1 + baseN1_ - 1) / baseN1_;
    if (n1Loops > tilingCoreNum) {
        n1Loops = tilingCoreNum;
    }
    uint32_t singleCoreN1 = (tempN1 + n1Loops - 1) / n1Loops;
    singleCoreN1 = (singleCoreN1 + FIFTEEN) & ~FIFTEEN;
    singleCoreN1 = singleCoreN1 > baseN1_ ? singleCoreN1 : baseN1_;
    n1Loops = (tempN1 + singleCoreN1 - 1) / singleCoreN1;

    uint32_t m1Loops = tilingCoreNum / n1Loops;
    uint32_t singleCoreM1 = (maxTokens + m1Loops - 1) / m1Loops;
    singleCoreM1 = (singleCoreM1 + FIFTEEN) & ~FIFTEEN;
    while (baseM1_ > MATMUL_MIN_SHAPE && baseM1_ > singleCoreM1) {
        baseM1_ = baseM1_ >> 1;
    }

    mm1.SetOrgShape(maxTokens, tilingData.ffnBaseParams.get_n1(), tilingData.ffnBaseParams.get_k1());
    mm1.SetShape(singleCoreM1, singleCoreN1, tilingData.ffnBaseParams.get_k1());
    mm1.SetFixSplit(baseM1_, baseN1_);
    mm1.SetBufferSpace(l1Size, l0CSize, tilingData.ffnSingleCoreParams.get_mm1ResUbSize());

    if (mm1.GetTiling(tilingData.mm1TilingData) == -1) {
        return ge::GRAPH_FAILED;
    }
    tilingData.mm1TilingData.set_shareMode(0);
    tilingData.mm1TilingData.set_shareL1Size(l1Size);
    tilingData.mm1TilingData.set_shareL0CSize(l0CSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FFNTiling::FFNGluCalMM2Tiling(uint64_t l0CSize)
{
    baseN2_ = GLU_BASEN;
    while (n2 < baseN2_) {
        baseN2_ = baseN2_ >> 1;
    }
    if (baseN2_ < MATMUL_MIN_SHAPE) {
        baseN2_ = MATMUL_MIN_SHAPE;
    }

    uint32_t maxBaseM = l0CSize / (baseN2_ * FP32_DATATYPE_SIZE);
    baseM2_ = MAX_BASE_BLOCK / baseN2_;
    while (baseM2_ > maxBaseM || maxTokens < baseM2_) {
        baseM2_ = baseM2_ >> 1;
    }
    if (baseM2_ < MATMUL_MIN_SHAPE) {
        baseM2_ = MATMUL_MIN_SHAPE;
    }

    uint32_t tilingCoreNum = tilingData.ffnBaseParams.get_coreNum(); // number of cubes in mm2

    uint32_t n2Loops = (n2 + baseN2_ - 1) / baseN2_;
    if (n2Loops > tilingCoreNum) {
        n2Loops = tilingCoreNum;
    }
    singleN2 = (n2 + n2Loops - 1) / n2Loops;
    singleN2 = (singleN2 + FIFTEEN) & ~FIFTEEN;
    singleN2 = singleN2 > baseN2_ ? singleN2 : baseN2_;
    n2Loops = (n2 + singleN2 - 1) / singleN2;

    uint32_t m2Loops = tilingCoreNum / n2Loops;
    singleM2 = (maxTokens + m2Loops - 1) / m2Loops;
    singleM2 = (singleM2 + FIFTEEN) & ~FIFTEEN;
    while (baseM2_ > MATMUL_MIN_SHAPE && baseM2_ > singleM2) {
        baseM2_ = baseM2_ >> 1;
    }
    tilingData.ffnSingleCoreParams.set_baseN2(baseN2_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FFNTiling::FFNGluSetMM2Tiling(gert::TilingContext *context, uint64_t l1Size, uint64_t l0CSize,
                                              matmul_tiling::DataType matmulDtype)
{
    matmul_tiling::MatmulApiTiling mm2;
    SetMMTilingType(context, false, matmulDtype, mm2);
    auto k2 = tilingData.ffnBaseParams.get_n1() / 2; // k2 = n1 / 2
    mm2.SetOrgShape(maxTokens, tilingData.ffnBaseParams.get_n2(), k2);
    mm2.SetShape(singleM2, singleN2, k2);
    mm2.SetFixSplit(baseM2_, baseN2_);
    mm2.SetBufferSpace(l1Size, l0CSize);
    if (mm2.GetTiling(tilingData.mm2TilingData) == -1) {
        return ge::GRAPH_FAILED;
    }
    tilingData.mm2TilingData.set_shareMode(0);
    tilingData.mm2TilingData.set_shareL1Size(l1Size);
    tilingData.mm2TilingData.set_shareL0CSize(l0CSize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FFNTiling::DataTypeCheck(gert::TilingContext *context)
{
    auto x = context->GetInputDesc(X_INDEX);
    auto weight1 = context->GetInputDesc(WEIGHT1_INDEX);
    auto weight2 = context->GetInputDesc(WEIGHT2_INDEX);
    auto output = context->GetOutputDesc(0);
    const gert::Tensor *weight2Tensor = context->GetInputTensor(WEIGHT2_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, weight2Tensor);
    OP_CHECK_NULL_WITH_CONTEXT(context, x);
    OP_CHECK_NULL_WITH_CONTEXT(context, weight1);
    OP_CHECK_NULL_WITH_CONTEXT(context, weight2);
    OP_CHECK_NULL_WITH_CONTEXT(context, output);
    xDataType = x->GetDataType();
    weight1Dtype = weight1->GetDataType();
    auto weight2Dtype = weight2->GetDataType();
    outputDtype = output->GetDataType();
    OP_CHECK_IF((weight1Dtype != weight2Dtype ||
                (xDataType != weight1Dtype && weight1Dtype != ge::DT_INT8 && weight1Dtype != ge::DT_INT4)),
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "weight1 and weight2 data type are not same."),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF((xDataType != ge::DT_FLOAT16 && xDataType != ge::DT_INT8 && xDataType != ge::DT_BF16),
               OPS_REPORT_VECTOR_INNER_ERR(
                   context->GetNodeName(), "x, weight1 and weight2 data type only support float16/int8/bfloat16/int4, the dtype is %s",
                   TypeUtils::DataTypeToSerialString(xDataType).c_str()),
               return ge::GRAPH_FAILED);
    xDataTypeSize = GetSizeByDataType(xDataType);
    OP_CHECK_IF((outputDtype != ge::DT_FLOAT16 && outputDtype != ge::DT_BF16),
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "output data type only support float16/bfloat16, the dtype is %s",
                                           TypeUtils::DataTypeToSerialString(outputDtype).c_str()),
               return ge::GRAPH_FAILED);

    minBaseNShape = MATMUL_MIN_SHAPE;
    if (xDataType == ge::DT_INT8) {
        minBaseNShape = MATMUL_MIN_SHAPE_INT8;
        if (context->GetOptionalInputTensor(DEQSCALE1_INDEX)) {
            auto deqscale1 = context->GetOptionalInputDesc(DEQSCALE1_INDEX);
            deqscaleDtype = deqscale1->GetDataType();
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FFNTiling::FormatCheck(const gert::TilingContext *context)
{
    auto x = context->GetInputDesc(X_INDEX);
    auto weight1 = context->GetInputDesc(WEIGHT1_INDEX);
    auto weight2 = context->GetInputDesc(WEIGHT2_INDEX);
    auto output = context->GetOutputDesc(0);
    ge::Format inputFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(x->GetStorageFormat()));
    ge::Format weight1Format = static_cast<ge::Format>(ge::GetPrimaryFormat(weight1->GetStorageFormat()));
    ge::Format weight2Format = static_cast<ge::Format>(ge::GetPrimaryFormat(weight2->GetStorageFormat()));
    ge::Format outputFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(output->GetStorageFormat()));
    OP_CHECK_IF(
        (weight1Format != weight2Format || inputFormat != outputFormat),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "InputFormat, weight1Format, weight2Format and outputFormat are not same"),
        return ge::GRAPH_FAILED);
    if (is310P) {
        OP_CHECK_IF(inputFormat != ge::FORMAT_FRACTAL_NZ,
                   OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                               "inputFormat and outputFormat only support NZ, the format is %s",
                                               TypeUtils::FormatToSerialString(inputFormat).c_str()),
                   return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(IsPrivateFormat(inputFormat),
                   OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Current inputFormat is %s, which is not supported",
                                               TypeUtils::FormatToSerialString(inputFormat).c_str()),
                   return ge::GRAPH_FAILED);
    }
    wFormat = weight1Format == ge::FORMAT_FRACTAL_NZ ? matmul_tiling::CubeFormat::NZ : matmul_tiling::CubeFormat::ND;
    xFormat = inputFormat == ge::FORMAT_FRACTAL_NZ ? matmul_tiling::CubeFormat::NZ : matmul_tiling::CubeFormat::ND;
    yFormat = outputFormat == ge::FORMAT_FRACTAL_NZ ? matmul_tiling::CubeFormat::NZ : matmul_tiling::CubeFormat::ND;
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus FFNTiling::N1EqualZeroWithBias2(uint64_t ubSize)
{
    int64_t mm1VaildUbBytes = ubSize / UB_DIVIDE_NUM_N1_ZERO;
    uint32_t baseN = mm1VaildUbBytes / 2;
    if (n2 < baseN) {
        baseN = n2;
    }

    baseN = (baseN + SIXTEEN_ALIGN_CONSTANT - 1) / SIXTEEN_ALIGN_CONSTANT * SIXTEEN_ALIGN_CONSTANT;
    tilingData.ffnSingleCoreParams.set_baseN2(baseN);
    tilingData.ffnSingleCoreParams.set_ubCalSize(baseN);
    tilingData.ffnSingleCoreParams.set_ubRestBytes(0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FFNTiling::FFNSingleCoreTiling(const gert::TilingContext *context, uint64_t ubSize)
{
    if (n1 == 0) {
        auto bias2 = context->GetOptionalInputDesc(BIAS2_INDEX);
        if (bias2 != NULL) {
            return N1EqualZeroWithBias2(ubSize);
        }
    }

    int64_t mm1VaildUbBytes = 0;
    if (innerPrecise == HIGH_PRECISION) {
        mm1VaildUbBytes = ubSize_ / UB_DIVIDE_NUM_HIGH_PRECISION;
    } else if (xDataType == ge::DT_FLOAT16) {
        mm1VaildUbBytes = ubSize / UB_DIVIDE_NUM;
    } else if (xDataType == ge::DT_INT8) {
        if (outputDtype == ge::DT_BF16) {
            mm1VaildUbBytes = ubSize / UB_DIVIDE_NUM_QUANT_BF16;
        } else {
            mm1VaildUbBytes = ubSize / UB_DIVIDE_NUM_QUANT;
        }
    }
    uint32_t baseN = BEST_BASEN;
    while (n1 < baseN) {
        baseN = baseN >> 1;
    }
    if (baseN < n1) {
        baseN = baseN << 1;
    }
    baseN = std::min<uint32_t>(BEST_BASEN, baseN);
    if (baseN < minBaseNShape) {
        baseN = minBaseNShape;
    }

    uint32_t maxBaseM = mm1VaildUbBytes / (baseN * xDataTypeSize);
    uint32_t baseM = MAX_BASE_BLOCK / baseN;
    if (FFNSetUbDivideBlk() == ge::GRAPH_SUCCESS) {
        OP_CHECK_IF(FFNCalUbSize(baseN, ubDivideBlkNum, ubIoBlkNum, baseM) != ge::GRAPH_SUCCESS,
                   OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "antiquant calculate ub failed."), return ge::GRAPH_FAILED);
    }
    while (baseM > maxBaseM || maxTokens < baseM) {
        baseM = baseM >> 1;
    }
    if (baseM < MATMUL_MIN_SHAPE) {
        baseM = MATMUL_MIN_SHAPE;
    }
    mm1VaildUbBytes = baseM * baseN;
    tilingData.ffnSingleCoreParams.set_baseM1(baseM);
    tilingData.ffnSingleCoreParams.set_baseN1(baseN);
    tilingData.ffnSingleCoreParams.set_mm1ResUbSize(mm1VaildUbBytes);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FFNTiling::FFNApiMM1Tiling(const gert::TilingContext *context,
                                           const matmul_tiling::PlatformInfo &platInfo,
                                           matmul_tiling::DataType matmulDtype)
{
    matmul_tiling::MatmulApiTiling mm1(platInfo);
    mm1.SetAType(matmul_tiling::TPosition::GM, xFormat, matmulDtype, false);
    mm1.SetBType(matmul_tiling::TPosition::GM, wFormat, matmulDtype, false);
    if (is310P) {
        mm1.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::NZ,
                     matmul_tiling::DataType::DT_FLOAT16);
    } else {
        mm1.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND_ALIGN,
                     matmul_tiling::DataType::DT_FLOAT16);
    }
    auto bias1 = context->GetOptionalInputDesc(BIAS1_INDEX);
    if (bias1 == nullptr) {
        mm1.SetBias(false);
    } else {
        mm1.SetBias(true);
        mm1.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        static_cast<matmul_tiling::DataType>(bias1->GetDataType()));
    }
    uint32_t mm1BaseM = std::min<uint32_t>(MAX_BASEM, tilingData.ffnSingleCoreParams.get_baseM1());
    mm1.SetOrgShape(maxTokens, n1, k1);
    mm1.SetShape(mm1BaseM, tilingData.ffnSingleCoreParams.get_baseN1(), k1);
    mm1.SetFixSplit(mm1BaseM, tilingData.ffnSingleCoreParams.get_baseN1());
    mm1.SetBufferSpace(platInfo.l1Size, platInfo.l0CSize, tilingData.ffnSingleCoreParams.get_ubCalSize());

    if (xDataType == ge::DT_INT8) {
        mm1.SetDequantType(matmul_tiling::DequantType::TENSOR);
    }

    if (mm1.GetTiling(tilingData.mm1TilingData) == -1) {
        return ge::GRAPH_FAILED;
    }
    tilingData.mm1TilingData.set_shareMode(0);
    tilingData.mm1TilingData.set_shareL1Size(platInfo.l1Size);
    tilingData.mm1TilingData.set_shareL0CSize(platInfo.l0CSize);

    return ge::GRAPH_SUCCESS;
}

void FFNTiling::FFNApiMM2CalBaseMN(uint32_t &mm2BaseN, uint32_t &baseM, const matmul_tiling::PlatformInfo &platInfo)
{
    while (n2 < mm2BaseN) {
        mm2BaseN = mm2BaseN >> 1;
    }
    if (mm2BaseN < n2) {
        mm2BaseN = mm2BaseN << 1;
    }
    mm2BaseN = std::min<uint32_t>(mm2BaseN, BEST_BASEN);
    if (mm2BaseN < minBaseNShape) {
        mm2BaseN = minBaseNShape;
    }
    uint32_t maxBaseM = platInfo.l0CSize / (mm2BaseN * FP32_DATATYPE_SIZE);
    baseM = MAX_BASE_BLOCK / std::max(MATMUL_MIN_SHAPE, mm2BaseN);
    while (baseM > maxBaseM || maxTokens < baseM) {
        baseM = baseM >> 1;
    }
    if (baseM < MATMUL_MIN_SHAPE) {
        baseM = MATMUL_MIN_SHAPE;
    }
    CalMM2Single(baseM, mm2BaseN);
    while (baseM > MATMUL_MIN_SHAPE && baseM > ((singleM2 + MATMUL_MIN_SHAPE - 1) & ~(MATMUL_MIN_SHAPE - 1))) {
        baseM = baseM >> 1;
    }
    UpdateMM2BaseNByCoreTiling(baseM, mm2BaseN);
}

ge::graphStatus FFNTiling::FFNApiMM2Tiling(const gert::TilingContext *context,
                                           const matmul_tiling::PlatformInfo &platInfo,
                                           matmul_tiling::DataType matmulDtype)
{
    matmul_tiling::MatmulApiTiling mm2(platInfo);
    mm2.SetAType(matmul_tiling::TPosition::GM, xFormat, matmulDtype, false);
    mm2.SetBType(matmul_tiling::TPosition::GM, wFormat, matmulDtype, false);
    if (xDataType == ge::DT_INT8 && outputDtype == ge::DT_BF16) {
        mm2.SetCType(matmul_tiling::TPosition::GM, yFormat, matmul_tiling::DataType::DT_INT32);
    } else {
        mm2.SetCType(matmul_tiling::TPosition::GM, yFormat, matmul_tiling::DataType::DT_FLOAT16);
    }
    auto bias2 = context->GetOptionalInputDesc(BIAS2_INDEX);
    if (bias2 == nullptr) {
        mm2.SetBias(false);
    } else {
        mm2.SetBias(true);
        mm2.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        static_cast<matmul_tiling::DataType>(bias2->GetDataType()));
    }
    uint32_t mm2BaseN = BEST_BASEN;
    uint32_t baseM = MATMUL_MIN_SHAPE;
    FFNApiMM2CalBaseMN(mm2BaseN, baseM, platInfo);

    mm2.SetOrgShape(maxTokens, n2, n1);
    mm2.SetShape(singleM2, singleN2, n1);
    mm2.SetFixSplit(baseM, mm2BaseN);
    mm2.SetBufferSpace(platInfo.l1Size, platInfo.l0CSize, ubSize_);
    if (xDataType == ge::DT_INT8) {
        mm2.SetDequantType(matmul_tiling::DequantType::TENSOR);
    }
    if (mm2.GetTiling(tilingData.mm2TilingData) == -1) {
        return ge::GRAPH_FAILED;
    }
    tilingData.mm2TilingData.set_shareMode(0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FFNTiling::FFNApiTiling(const gert::TilingContext *context, const matmul_tiling::PlatformInfo &platInfo,
                                        matmul_tiling::DataType matmulDtype)
{
    if (FFNApiMM1Tiling(context, platInfo, matmulDtype) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (FFNApiMM2Tiling(context, platInfo, matmulDtype) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    isTilingDataValid = true;
    return ge::GRAPH_SUCCESS;
}

void FFNTiling::CalMM2Single(uint32_t baseM2, uint32_t baseN2)
{
    uint32_t align = std::max(BLOCK_SIZE_FFN / xDataTypeSize, baseN2) - 1;
    singleM2 = (maxTokens + CALC_MM2_SINGLE_CORE_NUM - 1) / CALC_MM2_SINGLE_CORE_NUM;
    if (singleM2 < baseM2) {
        singleM2 = baseM2;
    }
    if (singleM2 > maxTokens) {
        singleM2 = maxTokens;
    }
    singleM2 = (singleM2 + FIFTEEN) & ~FIFTEEN;
    uint32_t m2Loops = (maxTokens + singleM2 - 1) / singleM2;
    uint32_t n2Loops = 1;
    singleN2 = n2;
    AdjustMM2MNLoops(align, m2Loops, n2Loops);
    if (m2Loops != 0) {
        singleM2 = (maxTokens + m2Loops - 1) / m2Loops;
    }
    auto singleM2Tmp = singleM2;
    singleM2 = (singleM2 + FIFTEEN) & ~FIFTEEN;
    if (singleM2 * m2Loops >= singleM2Tmp * (m2Loops + 1)) {
        singleM2 = singleM2Tmp;
    }
    if (n2Loops != 0) {
        singleN2 = (n2 + n2Loops - 1) / n2Loops;
    }
    singleN2 = (singleN2 + align) & ~align;
    if (singleM2 > maxTokens) {
        singleM2 = maxTokens;
    }
    return;
}

void FFNTiling::AdjustMM2MNLoops(const uint32_t align, uint32_t &m2Loops, uint32_t &n2Loops)
{
    if (m2Loops < CALC_MM2_SINGLE_CORE_NUM) {
        uint32_t maxMLoops = 0;
        uint32_t maxNLoops = 0;
        uint32_t maxUsedCore = 0;
        if (m2Loops > CALC_MM2_SINGLE_CORE_NUM / CONSTANT_TWO) {
            singleM2 *= CONSTANT_TWO;
        }
        while (singleM2 > 0) {
            m2Loops = (maxTokens + singleM2 - 1) / singleM2;
            if (m2Loops == 0) {
                break;
            }
            n2Loops = CALC_MM2_SINGLE_CORE_NUM / m2Loops;
            if (n2Loops == 0) {
                break;
            }
            singleN2 = (n2 + n2Loops - 1) / n2Loops;
            singleN2 = (singleN2 + align) & ~align;
            n2Loops = (n2 + singleN2 - 1) / singleN2;
            if (m2Loops * n2Loops > maxUsedCore) {
                maxUsedCore = m2Loops * n2Loops;
                maxMLoops = m2Loops;
                maxNLoops = n2Loops;
            }
            if (maxUsedCore == CALC_MM2_SINGLE_CORE_NUM) {
                break;
            }
            if (singleM2 > SIXTEEN_ALIGN_CONSTANT) {
                singleM2 -= SIXTEEN_ALIGN_CONSTANT;
            } else {
                break;
            }
        }
        m2Loops = maxMLoops;
        n2Loops = maxNLoops;
        if (maxUsedCore < CALC_MM2_SINGLE_CORE_NUM) {
            singleM2 = (maxTokens + CALC_MM2_SINGLE_CORE_NUM - 1) / CALC_MM2_SINGLE_CORE_NUM;
            if (maxTokens > (CALC_MM2_SINGLE_CORE_NUM - 1) * singleM2) {
                m2Loops = CALC_MM2_SINGLE_CORE_NUM;
                n2Loops = 1;
            }
        }
    }
}

uint64_t FFNTiling::SelectQuantTilingKey() const
{
    if (isSmooth) {
        return QUANT_SMOOTH_KEY;
    }
    if (outputDtype == ge::DT_BF16) {
        return QUANT_BF16_KEY;
    }
    if (deqscaleDtype == ge::DT_FLOAT) {
        return QUANT_DEQ_FLOAT32_KEY;
    }
    return QUANT_KEY;
}

uint64_t FFNTiling::SelectTilingKey() const
{
    if (xDataType == ge::DT_INT8 && weight1Dtype == ge::DT_INT8) {
        return SelectQuantTilingKey();
    }

    if (xDataType == ge::DT_BF16 && (weight1Dtype == ge::DT_BF16 || n1 == 0)) {
        return HIGH_PRECISION_BF16_KEY;
    }

    if (n1 == 0) {
        return HIGH_PERFORMANCE_KEY;
    }
    if (isMsdCase) {
        return ANTI_QUANT_MSD_KEY;
    }
    if ((xDataType == ge::DT_BF16 || xDataType == ge::DT_FLOAT16) &&
        (weight1Dtype == ge::DT_INT8 || weight1Dtype == ge::DT_INT4)) {
        if (isPerGroup == false) {
            return ANTI_QUANT_KEY;
        }
        return ANTI_QUANT_PERGROUP_KEY;
    }

    if (innerPrecise == HIGH_PRECISION) {
        return HIGH_PRECISION_KEY;
    }
    return HIGH_PERFORMANCE_KEY;
}

ge::graphStatus FFNTiling::FFNSetTilingKey(gert::TilingContext *context, uint64_t &key)
{
    if (is310P) {
        return ge::GRAPH_SUCCESS;
    }
    // add featurekey offset
    if (key == HIGH_PERFORMANCE_KEY || key == QUANT_KEY) {
        uint64_t featureKey = 0;
        if (isTilingDataValid && tilingData.mm1TilingData.get_baseM() == tilingData.mm2TilingData.get_baseM() &&
            tilingData.mm1TilingData.get_baseN() == tilingData.mm2TilingData.get_baseN() &&
            tilingData.mm1TilingData.get_baseK() == tilingData.mm2TilingData.get_baseK()) {
            featureKey += ONE_MATMUL; // Enable 1 mm
            if (maxTokensCheckOpt <= SMALL_TOKEN_BOUND && key == QUANT_KEY &&
                tilingData.mm1TilingData.get_stepN() == CONSTANT_TWO &&
                tilingData.mm2TilingData.get_stepN() == CONSTANT_TWO) {
                featureKey += 1; // enable stepN=2
            }
        }
        key = featureKey + key;
    }
    context->SetTilingKey(key);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FFNTiling::FFNSetTilingData(gert::TilingContext *context)
{
    uint64_t key = SelectTilingKey();
    FFNSetTilingKey(context, key);
    size_t *workspaces = context->GetWorkspaceSizes(1);
    uint32_t aicNum = tilingData.ffnBaseParams.get_coreNum();
    uint32_t mm1TilingBaseN = tilingData.mm1TilingData.get_baseN();
    uint32_t mm1BaseN = key == QUANT_STEPN_KEY ? mm1TilingBaseN * tilingData.mm1TilingData.get_stepN() : mm1TilingBaseN;
    mm1BaseN = std::max<uint32_t>(mm1BaseN, minBaseNShape);
    uint32_t nLoops = (n1 + mm1BaseN - 1) / mm1BaseN;
    uint32_t maxExpertParallelism1 = std::max<uint32_t>((aicNum) / std::max<uint32_t>(nLoops, 1), 1);
    maxExpertParallelism1 = std::min(maxExpertParallelism1, MAX_EXPERT_PARALLELISM);
    maxExpertParallelism1 = std::max(maxExpertParallelism1, CONSTANT_TWO);
    uint64_t workspace1Tokens = static_cast<uint64_t>(maxTokens);
    uint64_t workspace1Size = 0;
    if (is310P) {
        workspace1Size = 0;
    } else if (key == HIGH_PRECISION_KEY || key == HIGH_PRECISION_BF16_KEY || isQuantBf16 ||
               (xDataType == ge::DT_BF16 &&
                (key == ANTI_QUANT_KEY || key == ANTI_QUANT_PERGROUP_KEY || key == ANTI_QUANT_MSD_KEY))) {
        workspace1Size = uint64_t(workspace1Tokens) * n1 * sizeof(float);
    } else {
        workspace1Size = uint64_t(workspace1Tokens) * n1 * HALF_DATA_SIZE;
    }
    uint64_t workspace2Size = uint64_t(workspace1Tokens) * n1 * xDataTypeSize;
    if (key == QUANT_DEQ_FLOAT32_KEY) {
        workspace1Size = (workspace1Size + ALIGN64 - 1) / ALIGN64 * ALIGN64;
        workspace2Size = (workspace2Size + ALIGN64 - 1) / ALIGN64 * ALIGN64;
    }
    auto syncWorkspaceSize = (aicNum << 1) * 32;
    tilingData.ffnBaseParams.set_workspace1Size(workspace1Size);
    tilingData.ffnBaseParams.set_workspace2Size(workspace2Size);
    tilingData.ffnBaseParams.set_syncWorkspaceSize(syncWorkspaceSize);
    workspaces[0] = workspace1Size + workspace2Size + syncWorkspaceSize + SYS_WORKSPACE_SIZE;
    if (isQuantBf16) {
        workspaces[0] = workspaces[0] + uint64_t(bs) * n2 * sizeof(int32_t);
    }
    if (key == QUANT_DEQ_FLOAT32_KEY) {
        workspaces[0] = workspaces[0] + expertNum * n1 * sizeof(uint64_t) + expertNum * n2 * sizeof(uint64_t);
    }

    if (key == ANTI_QUANT_KEY || key == ANTI_QUANT_PERGROUP_KEY || key == ANTI_QUANT_MSD_KEY) {
        uint64_t workspaceAntiW1Size = static_cast<uint64_t>(k1) * static_cast<uint64_t>(n1) * xDataTypeSize * 2;
        uint64_t workspaceAntiW2Size = static_cast<uint64_t>(n1) * static_cast<uint64_t>(n2) * xDataTypeSize * 2;
        workspaces[0] += (workspaceAntiW1Size + workspaceAntiW2Size);
    }

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    PrintCriticalInfo(context);
    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingFFN(gert::TilingContext *context)
{
    FFNTiling FFN_tiling;
    auto ret = FFN_tiling.RunFusionKernelTiling(context);
    return ret;
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForFFN(gert::TilingParseContext *context)
{
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto compileInfoPtr = context->GetCompiledInfo<FFNCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->aivCoreNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->aicCoreNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
    compileInfoPtr->blockDim = ascendcPlatform.CalcTschBlockDim(compileInfoPtr->coreNum, compileInfoPtr->aicCoreNum,
                                                                compileInfoPtr->aivCoreNum);
    compileInfoPtr->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    if (Ops::Transformer::OpTiling::IsRegbaseSocVersion(context)) {
        compileInfoPtr->isRegbase = true;
        compileInfoPtr->ubSize -= SIMT_UB_SIZE_BYTE;
    }
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(FFN).Tiling(TilingFFN).TilingParse<FFNCompileInfo>(TilingPrepareForFFN);
} // namespace optiling
