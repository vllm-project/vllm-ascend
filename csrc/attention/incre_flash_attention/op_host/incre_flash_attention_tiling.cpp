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
 * \file incre_flash_attention_tiling.cpp
 * \brief
 */

#include <numeric>
#include <graph/utils/type_utils.h>
#include "incre_flash_attention_tiling_base.h"
#include "incre_flash_attention_tiling_impl.h"
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "register/op_def_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "../../common/op_host/fia_tiling_templates_registry.h"

using namespace ge;
using namespace AscendC;
using std::pair;
namespace optiling {

const int64_t tokenDefault = 2147483647; // 2147483647 for token default value
const int32_t sparseDefault = 0;
constexpr int64_t mFdBaseSizeMla = 8;
constexpr uint32_t MAX_SPLIT_RATIO = 2;
constexpr uint32_t BATCH_MODE_SCHEDULE = 1;
constexpr uint32_t SEQ_LEN_MIN = 4096;
constexpr uint32_t SEQ_LEN_MAX = 5120;
constexpr uint32_t SEQ_LEN_MIN_V2 = 37000;
constexpr uint32_t SEQ_LEN_MAX_V2 = 65536;
constexpr uint32_t HEAD_DIM = 512;
constexpr uint32_t HEAD_DIM_V = 512;
constexpr uint32_t SPARSE_MODE = 3;

uint64_t PFA_BENCHMARK_TILING_KEY = 1000000000000000000;

template <typename T> 
inline auto Align(T num, T rnd) -> T
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
}

static int64_t CeilDivision(int64_t num1, int64_t num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

// 获取公约数
static uint32_t increGcd(uint32_t a, uint32_t b)
{
    if (a % b == 0U) {
        return b;
    }
    return increGcd(b, a % b);
}

constexpr uint64_t RecursiveSum()
{
    return 0;
}

template <typename T, typename... Args> 
constexpr uint64_t RecursiveSum(T templateId, Args... templateIds)
{
    return static_cast<uint64_t>(templateId) + 10U * RecursiveSum(templateIds...);
}
constexpr uint64_t IFA_TILINGKEYOFFSET = uint64_t(10000000000000000UL);          // 10^16
constexpr uint64_t IFA_PERF_MODE_TILINGKEYOFFSET = uint64_t(1000000000000000UL); // 10^15
template <typename... Args> 
constexpr uint64_t IFA_GET_TILINGKEY(Args... templateIds)
{
    return RecursiveSum(templateIds...);
}

constexpr uint32_t PRE_LOAD_NUM = 4;
constexpr uint32_t BUF_NUM = 6;
constexpr uint32_t PRE_LOAD_NUM_MLA = 2;
constexpr uint32_t ATB_MODE_VAL = 3;
constexpr uint32_t ATB_INNER_PRECISE = 2;
constexpr uint32_t DIM_NUM_TWO = 2;
constexpr uint32_t DIM_NUM_THREE = 3;
constexpr uint32_t DIM_NUM_FOUR = 4;

void IFATiling::SetCoreNum()
{
    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        coreNum_ = aicNum_; // use aic num in 310p
    } else {
        if (balanceModeFlag_) {
            coreNum_ = aicNum_;
            return;
        }
        if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM || perfMode_ == IfaPerfMode::CUBE_VIEW_MM_FULL_LOAD
            || perfMode_ == IfaPerfMode::CUBE_VIEW_MM_DD) {
            coreNum_ = aicNum_; // use aiv num when compute mm by vec
        } else if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM_MLA) {
            coreNum_ = aicNum_;
        } else {
            coreNum_ = aivNum_; // split core by vec core
        }
    }
}

ge::graphStatus IFATiling::GetNpuInfo()
{
    OP_CHECK_IF(ifaContext_->platformInfo == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(ifaContext_->platformInfo);
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bSize_);

    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();

    OP_CHECK_IF(aicNum_ == 0 || aivNum_ == 0,
        OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName, "num of core obtained is 0."), return GRAPH_FAILED);

    if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P) {
        socVersion_ = IfaSocVersion::SOC_ASCEND_310P;
    } else {
        socVersion_ = IfaSocVersion::SOC_ASCEND_910B;

        cvRatio_ = aivNum_ / aicNum_;
        OP_CHECK_IF((cvRatio_ != 1U) && (cvRatio_ != 2U),
            OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName, "aicNum(%u):aivNum(%u) only support 1:1 or 1:2.", aicNum_, aivNum_), return GRAPH_FAILED);
    }

    OP_LOGI(ifaContext_->opName, "FIA aicNum: %u, aivNum:%u, cvRatio:%u.", aicNum_, aivNum_, cvRatio_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::PreProcess()
{
    if (ProcessBaseInputs() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    bool ret = CheckIfRollBack();
    if (ret) {
        passToOldTiling_ = true;
        return ge::GRAPH_FAILED;
    }

    if (ProcessOptionalTensors() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    SetupPerfMode();
    IsFdBalanceCase();
    balanceModeFlag_ = IsBalanceSplitCore();
    SetCoreNum();
    UpdateL2CacheOffFlag();

    return ge::GRAPH_SUCCESS;
}

bool IFATiling::CheckCommonConditions(const ValidityConfigFD& config) const {
    return CheckBatchAndQSeqSize(config.validBatchSizes, config.validQSeqSizes) &&
           CheckHeadDimensions(config.numHeads, config.numKvHeads,
                               config.headDim, config.headDimV) &&
           CheckQuantizationFlags(config.sparseMode) &&
           CheckActualSeqLengths(config.expectedActualSeqLength);
}

bool IFATiling::CheckBatchAndQSeqSize(const std::vector<int32_t>& validBatchSizes, const std::vector<int32_t>& validQSeqSizes) const {
    if (std::find(validBatchSizes.begin(), validBatchSizes.end(), batchSize_) == validBatchSizes.end())
        return false;

    if (std::find(validQSeqSizes.begin(), validQSeqSizes.end(), qSeqSize_) == validQSeqSizes.end())
        return false;

    return true;
}

bool IFATiling::CheckHeadDimensions(int32_t numHeads, int32_t numKvHeads, int32_t headDim, int32_t headDimV) const {
    return (numHeads_ == numHeads &&
            numKvHeads_ == numKvHeads &&
            headDim_ == headDim &&
            headDimV_ == headDimV);
}

bool IFATiling::CheckQuantizationFlags(int32_t sparseMode) const {
    if (!quantFlag_)
        return false;

    if (sparseMode_ != sparseMode)
        return false;

    if (inputKvLayout_ != IfaLayout::NZ || !pageAttentionFlag_)
        return false;

    int64_t queryQuantMode = ifaContext_->queryQuantMode ? *ifaContext_->queryQuantMode : 0;
    int64_t keyAntiQuantMode = ifaContext_->keyAntiquantMode ? *ifaContext_->keyAntiquantMode : 0;
    int64_t valueAntiQuantMode = ifaContext_->valueAntiquantMode ? *ifaContext_->valueAntiquantMode : 0;

    if (queryQuantMode != DEQUANT_PER_TOKEN_HEAD_MODE ||
        keyAntiQuantMode != DEQUANT_PER_CHANNEL_MODE ||
        valueAntiQuantMode != DEQUANT_PER_CHANNEL_MODE)
        return false;

    return true;
}

bool IFATiling::CheckActualSeqLengths(int64_t expectedActualSeqLength) const {
    const int64_t* actualSeqKv = ifaContext_->actualSeqLengths.tensor->GetData<int64_t>();
    if (actualSeqKv == nullptr)
        return false;

    for (uint32_t bIdx = 0U; bIdx < batchSize_; ++bIdx) {
        int64_t s2 = (actualLenDims_ == 1U) ? actualSeqKv[0] : actualSeqKv[bIdx];

        if (expectedActualSeqLength >= 0) {
            if (s2 < SEQ_LEN_MIN_V2 || s2 > SEQ_LEN_MAX_V2)
                return false;
        } else {
            if (s2 < SEQ_LEN_MIN || s2 > SEQ_LEN_MAX)
                return false;
        }
    }

    return true;
}

bool IFATiling::IsValidFlag3B() {
    ValidityConfigFD cfg{
        .validBatchSizes = {1, 2, 4}, // 表示有效的BatchSize大小
        .validQSeqSizes = {4}, // 表示有效的Qseq大小
        .numHeads = 8,
        .numKvHeads = 1,
        .headDim = HEAD_DIM,
        .headDimV = HEAD_DIM_V,
        .sparseMode = SPARSE_MODE,
        .expectedActualSeqLength = -1 // 表示 [4096, 5120]
    };
    return CheckCommonConditions(cfg);
}

bool IFATiling::IsValidFlag560B() {
    ValidityConfigFD cfg{
        .validBatchSizes = {4},
        .validQSeqSizes = {2, 3, 4},
        .numHeads = 64,
        .numKvHeads = 1,
        .headDim = HEAD_DIM,
        .headDimV = HEAD_DIM_V,
        .sparseMode = SPARSE_MODE,
        .expectedActualSeqLength = -1
    };
    return CheckCommonConditions(cfg);
}

bool IFATiling::IsValidFlag() {
    ValidityConfigFD cfg{
        .validBatchSizes = {1},
        .validQSeqSizes = {2},
        .numHeads = 128,
        .numKvHeads = 1,
        .headDim = HEAD_DIM,
        .headDimV = HEAD_DIM_V,
        .sparseMode = SPARSE_MODE,
        .expectedActualSeqLength = 1 // 1表示整网场景下s2范围：[37000,65536]
    };
    return CheckCommonConditions(cfg);
}

void IFATiling::IsFdBalanceCase() {
    if(IsValidFlag() || IsValidFlag3B() || IsValidFlag560B()) {
        tilingDataMla_.tndSplitCoreParams.set_FdBalanceFlag(1);
    } else {
        tilingDataMla_.tndSplitCoreParams.set_FdBalanceFlag(0);
    }
}

bool IFATiling::IsBalanceSplitCore() {
    if (tilingDataMla_.tndSplitCoreParams.get_FdBalanceFlag() == 1) {
        return true;
    }

    if (perfMode_ != IfaPerfMode::CUBE_VIEW_MM_MLA) {
        return false;
    }

    if (inputLayout_ == IfaLayout::TND) {
        return true;
    }

    if (antiQuantFlag_) {
        return false;
    }

    if (inputLayout_ == IfaLayout::BNSD) {
        return false;
    }

    if (isWorkspace_) { // tiling下沉不走balance
        return false;
    }

    return !DealSameSeqEachBatch();
}

uint32_t IFATiling::GetTypeSize(ge::DataType dtype) const
{   
    static const std::unordered_map<ge::DataType, uint32_t> typeSizeMap = {
        {ge::DT_FLOAT, NUM_BYTES_FLOAT},
        {ge::DT_FLOAT16, NUM_BYTES_FLOAT16},
        {ge::DT_BF16, NUM_BYTES_BF16},
        {ge::DT_BOOL, NUM_BYTES_BOOL},
        {ge::DT_INT8, NUM_BYTES_INT8},
        {ge::DT_UINT8, NUM_BYTES_INT8},
        {ge::DT_INT4, NUM_BYTES_INT8}
    };

    auto it = typeSizeMap.find(dtype);
    if (it != typeSizeMap.end()) {
        return it->second;
    } else {
        return NUM_BYTES_UNDEF;
    }
}

ge::graphStatus IFATiling::SetL2CacheFlag()
{   
    auto kDType = ifaContext_->key.desc->GetDataType();
    uint32_t kvTypeSize = GetTypeSize(kDType);
    if (kvTypeSize == NUM_BYTES_UNDEF) {
        OP_LOGE(ifaContext_->opName, "Data type %s is not currently supported.",
            DataTypeToSerialString(kDType).c_str());
        return ge::GRAPH_FAILED;
    }

    if (ropeFlag_) {
        return ge::GRAPH_SUCCESS;
    }

    uint64_t kvSize = 0;
    auto batchOfQuery = ifaContext_->query.shape->GetStorageShape().GetDim(0);
    auto batchOfKey = ifaContext_->key.shape->GetStorageShape().GetDim(0);
    if (ifaContext_->blockTable.tensor != nullptr) {
        kvSize = ifaContext_->key.shape->GetStorageShape().GetShapeSize();
    } else if (batchOfQuery != batchOfKey) { /* kv noncontinuous */
        for (int64_t size = 0; size < batchOfQuery; ++size) {
            auto keyTensorInList = ifaContext_->kCache[size];
            kvSize += keyTensorInList->GetStorageShape().GetShapeSize();
        }
    } else {
        kvSize = ifaContext_->key.shape->GetStorageShape().GetShapeSize();
    }

    uint64_t l2CacheSize = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(ifaContext_->platformInfo);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2CacheSize);
    // 考虑K、V，1.2为关闭L2Cache的系数
    if (static_cast<double>(kvSize) * kvTypeSize * 2.0f >= l2CacheSize * 1.2) {
        OP_LOGD(ifaContext_->opName, "L2 cache off");
        l2CacheOffFlag_ = 1U ;
    }

    OP_LOGD(ifaContext_->opName, "l2CacheOffFlag_: %u, kvSize: %lu, kvTypeSize: %u, l2CacheSize: %lu", l2CacheOffFlag_,
              kvSize, kvTypeSize, l2CacheSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::QKVPreProcess()
{
    OP_CHECK_IF(ifaContext_->key.desc->GetDataType() != ifaContext_->value.desc->GetDataType(),
        OP_LOGE(ifaContext_->opName, "datatype of k tensor and value tensor is different"), return ge::GRAPH_FAILED);
    batchSizeQ_ = batchSize_ = ifaContext_->query.shape->GetStorageShape().GetDim(0);
    inputQType_ = ifaContext_->query.desc->GetDataType();
    inputKvType_ = ifaContext_->key.desc->GetDataType();
    outputType_ = ifaContext_->attenOut.desc->GetDataType();

    numHeads_ = *ifaContext_->numHeads;
    numKvHeads_ = *ifaContext_->kvHeadNums;
    scaleValue_ = *ifaContext_->scaleValue;
    blockSize_ = *ifaContext_->blockSize;

    OP_LOGI(ifaContext_->opName, "scaleValue_ is %f.", scaleValue_);
    OP_CHECK_IF(numHeads_ == 0U, OP_LOGE(ifaContext_->opName, "the query's heads num is zero"), return ge::GRAPH_FAILED);
    if (numKvHeads_ == 0U) {
        numKvHeads_ = numHeads_;
    }
    OP_CHECK_IF(((numKvHeads_ > numHeads_) || (numHeads_ % numKvHeads_ != 0U)),
        OP_LOGE(ifaContext_->opName, "Attr num_key_value_heads is invalid, n: %u, the key/value's heads num: %u", numHeads_,
            numKvHeads_),
        return ge::GRAPH_FAILED);
    nNumOfQInOneGroup_ = numHeads_ / numKvHeads_;
    groupSplitSize_ = nNumOfQInOneGroup_;
    gOuter_ = 1U;
    s1Outer_ = 1U;

    std::string layout(ifaContext_->layOut);
    uint32_t sOfQuery = 0U;
    uint32_t sOfHeadnum = 0U;
    uint32_t kDimNum = ifaContext_->key.shape->GetStorageShape().GetDimNum();
    if (GetInOutLayoutAndProcessQInfo(layout, sOfQuery, sOfHeadnum, kDimNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    qSeqSize_ = sOfQuery;
    if (layout == "TND" || layout == "TND_NTD") {
        if (QKVPreProcess4TND(layout) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        sOfQuery = qSeqSize_;
        if (isWorkspace_ && ifaContext_->blockTable.tensor != nullptr) {
            uint32_t tndBatchSize = ifaContext_->blockTable.tensor->GetStorageShape().GetDim(0);
            if (tndBatchSize != 0) {
                sOfQuery = (tSeqSize_ + tndBatchSize - 1) / tndBatchSize;
            }
        }
    }
    s1SplitSize_ = qSeqSize_;

    if (GetRopeAndGqaFlag(sOfQuery, kDimNum, sOfHeadnum, layout) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::GetInOutLayoutAndProcessQInfo(const std::string layout, uint32_t& sOfQuery, uint32_t& sOfHeadnum, const uint32_t kDimNum)
{
    bool prefixFlag = !(ifaContext_->keySharedPrefix.tensor == nullptr && ifaContext_->valueSharedPrefix.tensor == nullptr);
    if (layout == "BSH" || layout == "BSH_NBSD") {
        if (GetInOutLayout4BSH(layout, sOfQuery, sOfHeadnum, kDimNum, prefixFlag) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
       }
    } else if (layout == "BSND" || layout == "BSND_NBSD") {
        inputLayout_ = IfaLayout::BSH_BSND;
        outputLayout_ = (layout == "BSND") ? inputLayout_ : IfaLayout::NBSD;
        sOfQuery = ifaContext_->query.shape->GetStorageShape().GetDim(1); // 1, dim of S
        headDim_ = ifaContext_->query.shape->GetStorageShape().GetDim(3); // 3, dim of D
        sOfHeadnum = ifaContext_->query.shape->GetStorageShape().GetDim(2); // 2, dim of N
    } else if (layout == "BNSD" || layout == "BNSD_NBSD") {
        inputLayout_ = IfaLayout::BNSD;
        outputLayout_ = (layout == "BNSD") ? inputLayout_ : IfaLayout::NBSD;
        sOfQuery = ifaContext_->query.shape->GetStorageShape().GetDim(2); // 2, dim of S
        headDim_ = ifaContext_->query.shape->GetStorageShape().GetDim(3); // 3, dim of D
        sOfHeadnum = ifaContext_->query.shape->GetStorageShape().GetDim(1); // 1, dim of N
    } else if (layout == "TND" || layout == "TND_NTD") {
        inputLayout_ = IfaLayout::TND;
        outputLayout_ = (layout == "TND") ? inputLayout_ : IfaLayout::NTD;
        sOfHeadnum = ifaContext_->query.shape->GetStorageShape().GetDim(1); // 2, dim of N
        headDim_ = ifaContext_->query.shape->GetStorageShape().GetDim(2); // 3, dim of D
        sOfQuery = ifaContext_->query.shape->GetStorageShape().GetDim(0); // 1, dim of T, S == 1, D = T / S = T
    } else {
        OP_LOGE(ifaContext_->opName, "Only support input_layout(BSH, BNSD, BSND, TND), actually is %s", layout.c_str());
        return ge::GRAPH_FAILED;
    }
  
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::GetInOutLayout4BSH(const std::string layout, uint32_t& sOfQuery, uint32_t& sOfHeadnum, const uint32_t kDimNum, bool prefixFlag)
{
    inputLayout_ = IfaLayout::BSH_BSND;
    outputLayout_ = (layout == "BSH") ?  inputLayout_ : IfaLayout::NBSD;
    OP_CHECK_IF(ifaContext_->query.shape->GetStorageShape().GetDim(2) % numHeads_ != 0U,
            OP_LOGE(ifaContext_->opName, "H should be an interger multiple of numHeads"),
            return ge::GRAPH_FAILED);
    sOfQuery = ifaContext_->query.shape->GetStorageShape().GetDim(1);
    headDim_ = ifaContext_->query.shape->GetStorageShape().GetDim(2) / numHeads_; // 2, QK dim of H
    int32_t tmpWindowSize = -1;
        if (ifaContext_->windowSize != nullptr) {
            tmpWindowSize = static_cast<int32_t>(*ifaContext_->windowSize);
        }
    slidingFlag_ = (layout == "BSH") && (sOfQuery == 1) && (tmpWindowSize > 0)
                        && (ifaContext_->blockTable.tensor != nullptr) && (!prefixFlag)
                        && (ifaContext_->value.shape->GetStorageShape().GetDimNum() == DIM_BSH) 
                        && (ifaContext_->queryRope.tensor == nullptr) && (inputQType_ == inputKvType_)
                        && ((inputQType_ == ge::DT_BF16) || (inputQType_ == ge::DT_FLOAT16))
                        && (socVersion_ == IfaSocVersion::SOC_ASCEND_910B);
    if (slidingFlag_) {
            headDimV_ = ifaContext_->value.shape->GetStorageShape().GetDim(2) / numKvHeads_; // 2, V dim of H
        } 
        sOfHeadnum = numHeads_;
        if ((*ifaContext_->kvHeadNums != 0U) && (kDimNum == 3U)) { // 3, dim of kv when the layout of kv is BSH
            sOfHeadnum = headDim_ * numHeads_ * numKvHeads_ / ifaContext_->key.shape->GetStorageShape().GetDim(2); // 2, dim of H
        }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::GetRopeAndGqaFlag(const uint32_t sOfQuery, const uint32_t kDimNum, const uint32_t sOfHeadnum,const std::string layout)
{   
    bool prefixFlag = !(ifaContext_->keySharedPrefix.tensor == nullptr && ifaContext_->valueSharedPrefix.tensor == nullptr);
    if (numHeads_ != sOfHeadnum && !prefixFlag) {
        OP_LOGE(ifaContext_->opName, "the query's heads num should be equal to qOfHeadnum");
        return ge::GRAPH_FAILED;
    }
    if (inputKvType_ == ge::DT_INT4 && headDim_ % KVINT4_BYTE_BLOCK != 0U) {
        OP_LOGE(ifaContext_->opName, "Number of heads must be a multiple of %u, current dim of D is %u.", KVINT4_BYTE_BLOCK, headDim_);
        return ge::GRAPH_FAILED;
    }
    if (!slidingFlag_) {
        headDimV_ = headDim_;
    }
    // 元素个数按照基本块大小对齐
    headDimAlign_ = (inputKvType_ == ge::DT_INT4) ? Align(headDim_, KVINT4_BYTE_BLOCK) : Align(headDim_, BYTE_BLOCK);
    headDimVAlign_ = (inputKvType_ == ge::DT_INT4) ? Align(headDimV_, KVINT4_BYTE_BLOCK) : Align(headDimV_, BYTE_BLOCK);

    OP_CHECK_IF((ifaContext_->queryRope.tensor != nullptr && ifaContext_->keyRope.tensor == nullptr),
        OP_LOGE(ifaContext_->opName, "KeyRope is null, but queryRope exists, they should be both null or exist. "),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((ifaContext_->queryRope.tensor == nullptr && ifaContext_->keyRope.tensor != nullptr),
        OP_LOGE(ifaContext_->opName, "QueryRope is null, but keyRope exists, they should be both null or exist. "),
        return ge::GRAPH_FAILED);

     if (ifaContext_->keyRope.tensor != nullptr && ifaContext_->queryRope.tensor != nullptr) {
        ropeFlag_ = true;
    }

    if (sOfQuery > 1U && sOfQuery <= 16U && !ropeFlag_) {  // 投机推理场景，QS在1到16之间
        gqaMtpFlag_ = true;
    }
    if (kDimNum == 5U && !ropeFlag_) {
        gqaKvNZFlag_ = true;
    }
    if (!ropeFlag_ && !gqaMtpFlag_) {
        OP_CHECK_IF(sOfQuery != 1U,
            OP_LOGE(ifaContext_->opName, "In case where MLA is not applied, S of Query:%u is invalid. It should be in range [1, 16]", sOfQuery),
                   return ge::GRAPH_FAILED);
    } else if (layout != "TND" && layout != "TND_NTD") {
        OP_CHECK_IF(sOfQuery > 32, OP_LOGE(ifaContext_->opName, "QueryS(%u) should not be bigger than 32 in MLA.", sOfQuery),
                   return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(layout == "TND" && headDim_ == 512 && !ropeFlag_, OP_LOGE(ifaContext_->opName,
        "When D is 512, inputlayout %s q_rope and k_rope should not be null!", layout.c_str()), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::QKVPreProcess4TND(const std::string layout)
{
    auto qType = ifaContext_->query.desc->GetDataType();
    uint32_t qTypeSize = GetTypeSize(qType);
    if (qTypeSize == NUM_BYTES_UNDEF) {
        OP_LOGE(ifaContext_->opName, "Data type %s is not currently supported.", DataTypeToSerialString(qType).c_str());
        return ge::GRAPH_FAILED;
    }

    if (isWorkspace_) { // TND+tiling下沉场景，不做校验
        tSeqSize_ = qSeqSize_ = ifaContext_->query.shape->GetStorageShape().GetDim(0);
        OP_CHECK_IF((tSeqSize_ > 1024U * 1024U / qTypeSize), // T不大于1M
                   OP_LOGE(ifaContext_->opName, "%s query T should <= 1M", layout.c_str()), return ge::GRAPH_FAILED);
    } else {
        // actualSeqLengths非空校验
        OP_CHECK_IF((ifaContext_->actualSeqLengthsQ.tensor == nullptr) || (ifaContext_->actualSeqLengthsQ.tensor->GetData<int64_t>() == nullptr),
            OP_LOGE(ifaContext_->opName, "%s the query's actual sequence lengths should not be null!", layout.c_str()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((ifaContext_->actualSeqLengths.tensor == nullptr) || (ifaContext_->actualSeqLengths.tensor->GetData<int64_t>() == nullptr),
            OP_LOGE(ifaContext_->opName, "%s the key/value's actual sequence lengths should not be null!", layout.c_str()),
            return ge::GRAPH_FAILED);

        actualLenQDims_ = ifaContext_->actualSeqLengthsQ.tensor->GetShapeSize();
        actualLenDims_ = ifaContext_->actualSeqLengths.tensor->GetShapeSize();

        OP_CHECK_IF((actualLenQDims_ <= 0U),
                   OP_LOGE(ifaContext_->opName, "%s the query's actual sequence lengths shape size(%u) should > 0", layout.c_str(), actualLenQDims_),
                   return ge::GRAPH_FAILED);
        OP_CHECK_IF((actualLenQDims_ != actualLenDims_),
                   OP_LOGE(ifaContext_->opName, "%s the query's actual sequence lengths shape size(%u) should equal the key/value's actual sequence lengths shape size(%u)", layout.c_str(), actualLenQDims_, actualLenDims_),
                   return ge::GRAPH_FAILED);

        batchSizeQ_ = batchSize_ = ifaContext_->actualSeqLengthsQ.tensor->GetShapeSize();
        tSeqSize_ = ifaContext_->query.shape->GetStorageShape().GetDim(0);
        uint32_t tKVSeqSize = ifaContext_->key.shape->GetStorageShape().GetDim(0); 
        OP_CHECK_IF((tSeqSize_ > 1024U * 1024U / qTypeSize), // T不大于1M
                   OP_LOGE(ifaContext_->opName, "%s query T should <= 1M", layout.c_str()), return ge::GRAPH_FAILED);

        const int64_t *actualSeqQTnd = ifaContext_->actualSeqLengthsQ.tensor->GetData<int64_t>();
        const int64_t *actualSeqKVTnd = ifaContext_->actualSeqLengths.tensor->GetData<int64_t>();
        std::vector<int64_t> actualSeqQ(actualLenQDims_);
        int64_t tmpQSeqSize = 0;

        for (int32_t b = 0; b < static_cast<int>(actualLenQDims_); b++) {
            actualSeqQ[b] = (b <= 0) ? actualSeqQTnd[0] : (actualSeqQTnd[b] - actualSeqQTnd[b - 1]);
            OP_CHECK_IF((actualSeqQ[b] < 0) || (actualSeqQ[b] > 32), // 32 MTP最大QS
                       OP_LOGE(ifaContext_->opName, "%s QS(%ld) of batch(%d) computed by the query's actual sequence lengths should be in range [0, 32].", layout.c_str(), actualSeqQ[b], b),
                       return ge::GRAPH_FAILED);
            tmpQSeqSize = std::max(tmpQSeqSize, actualSeqQ[b]);
            qSeqSquareSum_ += actualSeqQ[b] * actualSeqQ[b];
        }

        OP_CHECK_IF((tSeqSize_ != actualSeqQTnd[actualLenQDims_ - 1]),
            OP_LOGE(ifaContext_->opName, "%s T(%u) should be equal to the last element of the query's actual sequence lengths(%ld).", layout.c_str(), tSeqSize_, actualSeqQTnd[actualLenQDims_ - 1]),
            return ge::GRAPH_FAILED);

        if (!pageAttentionFlag_) {
            OP_CHECK_IF((tKVSeqSize != actualSeqKVTnd[actualLenDims_ - 1]),
                OP_LOGE(ifaContext_->opName, "%s T(%u) should be equal to the last element of the key/value's actual sequence lengths(%ld).", layout.c_str(), tKVSeqSize, actualSeqKVTnd[actualLenDims_ - 1]),
                return ge::GRAPH_FAILED);
        }

        qSeqSize_ = static_cast<uint32_t>(tmpQSeqSize);
        OP_LOGI(ifaContext_->opName, "TND MAX actualSeqQ:%u", qSeqSize_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::InputAttrsPreProcess()
{
    const uint32_t *innerPrecisePtr = ifaContext_->innerPrecise;
    innerPrecise_ = innerPrecisePtr ? *innerPrecisePtr : IFA_HIGH_PERFORMANCE; // 默认高性能
    OP_CHECK_IF(((innerPrecise_ != IFA_HIGH_PERFORMANCE) && (innerPrecise_ != IFA_HIGH_PRECISION)),
        OP_LOGE(ifaContext_->opName, "precision mode[%u] should be 0 or 1", innerPrecise_),
        return ge::GRAPH_FAILED); // 当前只支持高精度0和高性能1
    OP_LOGD(ifaContext_->opName, "innerPrecise is %u", innerPrecise_);

    blockTypeSize_ = sizeof(float); // 默认按照float计算
    pageAttentionFlag_ = ifaContext_->blockTable.tensor != nullptr;

    if (!pageAttentionFlag_) {
        uint32_t kvBatch = ifaContext_->key.shape->GetStorageShape().GetDim(0);
        batchContinuousFlag_ = (batchSize_ == kvBatch);
    } else {
        uint32_t blockTableDim0 = ifaContext_->blockTable.tensor->GetStorageShape().GetDim(0);
        uint32_t blockTableDim1 = ifaContext_->blockTable.tensor->GetStorageShape().GetDim(1);
        OP_LOGI(ifaContext_->opName, "pageAttentionFlag_ is true. The shape of blockTable is [%u, %u].", blockTableDim0, blockTableDim1);
        OP_CHECK_IF(
            inputKvType_ == ge::DT_INT4 && inputLayout_ != IfaLayout::BSH_BSND,
            OP_LOGE(ifaContext_->opName,
                "IFA don't support PageAttenion if the KV Inputtype is INT4 or INT32 and inputlayout isn't BSH_BSND currently."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            ifaContext_->blockTable.tensor->GetStorageShape().GetShapeSize() == 0,
            OP_LOGE(ifaContext_->opName, "check blockTable shape failed, blockTable shapeSize is zero."),
            return ge::GRAPH_FAILED);
        if (inputLayout_ == IfaLayout::TND && !isWorkspace_) {
            OP_CHECK_IF(blockTableDim0 != actualLenQDims_,
                OP_LOGE(ifaContext_->opName, "The actual sequence length dimension for Q[%u] in TND must match the B-axis of block table[%u].", actualLenQDims_, blockTableDim0),
                return ge::GRAPH_FAILED);
        }
    }

    // achieve windowSize when sliding
    if (ifaContext_->windowSize != nullptr) {
        windowSize_ = static_cast<int32_t>(*ifaContext_->windowSize);
    }

    if (ifaContext_->softmaxLseFlag != nullptr) {
        softmaxLseFlag_ = *ifaContext_->softmaxLseFlag;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::SetQuantFlag()
{
    antiQuantFlag_ = (inputQType_ != inputKvType_) && (inputKvType_ == ge::DT_INT8 || inputKvType_ == ge::DT_INT4);
    if (antiQuantFlag_) {
        if (innerPrecise_ == IFA_HIGH_PRECISION) {
            msdIterNum_ = inputKvType_ == ge::DT_INT4 ? KVINT4_ITER_NUM : HIGH_PRECISION_ITER_NUM;
        } else {
            msdIterNum_ = inputKvType_ == ge::DT_INT4 ? KVINT4_ITER_NUM : ITER_NUM;
        }
    }

    // 全量化基本校验
    if (inputQType_ == ge::DT_INT8) {
        OP_CHECK_IF(!pageAttentionFlag_ || inputKvLayout_ != IfaLayout::NZ,
                OP_LOGE(ifaContext_->opName, "when the dtype of query is int8 in MLA, PageAttetion should be enabled and KV layout must be NZ"),
                return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ProcessBaseInputs()
{
    if ((CheckBaseInputsNull() != ge::GRAPH_SUCCESS) ||
        (QKVPreProcess() != ge::GRAPH_SUCCESS) ||
        (InputAttrsPreProcess() != ge::GRAPH_SUCCESS) ||
        (KvShapePostProcess() != ge::GRAPH_SUCCESS) ||
        (CheckQKOutShape() != ge::GRAPH_SUCCESS) ||
        (CheckInputFormatAndLimits() != ge::GRAPH_SUCCESS) ||
        (SetL2CacheFlag() != ge::GRAPH_SUCCESS) ||
        (SetQuantFlag() != ge::GRAPH_SUCCESS) ||
        (InitInOutMode() != ge::GRAPH_SUCCESS)) {
            return ge::GRAPH_FAILED;
        }
    return ge::GRAPH_SUCCESS;
}

void IFATiling::SetupPerfMode()
{
    // 310P
    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        if (EnableGQA310P()) {
            perfMode_ = IfaPerfMode::CUBE_VIEW_MM;
        } else {
            perfMode_ = IfaPerfMode::BMM_ALL_BY_VEC;
        }
    } else {
        if (gqaKvNZFlag_) {
            perfMode_ = IfaPerfMode::CUBE_VIEW_MM_DD;
        } else if (ropeFlag_) {
            perfMode_ = IfaPerfMode::CUBE_VIEW_MM_MLA;
        } else {
            if (EnableAllVec()) {
                perfMode_ = IfaPerfMode::BMM_ALL_BY_VEC;
            } else if (EnableCubeViewMM()) {
                perfMode_ = IfaPerfMode::CUBE_VIEW_MM;
            }
        }
    }
}

void IFATiling::UpdatePerfMode()
{
    if (EnableC1V1()) {
        perfMode_ = IfaPerfMode::C1_V1;
    }
}

void IFATiling::UpdateL2CacheOffFlag()
{
    if (perfMode_ != IfaPerfMode::CUBE_VIEW_MM_DD) {
        return;
    }
    if (inputKvLayout_ == IfaLayout::NZ || inputKvLayout_ == IfaLayout::BNSD) {
        l2CacheOffFlag_ = 1U;
        OP_LOGD(ifaContext_->opName, "KV layout is NZ or BNSD, set l2CacheOffFlag_ = 1.");
    } else if (inputKvLayout_ == IfaLayout::BSH_BSND) {
        if (numKvHeads_ == 1U) {
            l2CacheOffFlag_ = 1U;
            OP_LOGD(ifaContext_->opName, "KV layout is BSH and the key/value's heads num is 1, set l2CacheOffFlag_ = 1.");
        }
    }
}

ge::graphStatus IFATiling::ProcessPageAttentionFlag()
{
    maxBlockNumPerBatch_ = ifaContext_->blockTable.tensor->GetStorageShape().GetDim(1);
    sMax_ = maxBlockNumPerBatch_ * blockSize_;
    seqSize_ = sMax_;
    uint32_t kDimNum = ifaContext_->key.shape->GetStorageShape().GetDimNum();
    if (inputLayout_ == IfaLayout::TND) {
        if (kDimNum == 3U) { // BSH
            inputKvLayout_ = IfaLayout::BSH_BSND;
        } else if (kDimNum == 4U) { // BNSD
            inputKvLayout_ = IfaLayout::BNSD;
        } else if (kDimNum == 5U) { // NZ: block_num, N, D/12, block_size, 16
            inputKvLayout_ = IfaLayout::NZ;
        }
    } else {
        if (ropeFlag_ || gqaKvNZFlag_) {
            if (kDimNum == 3U) { // BSH
                inputKvLayout_ = IfaLayout::BSH_BSND;
            } else if (kDimNum == 4U) { // BNSD
                inputKvLayout_ = IfaLayout::BNSD;
            } else if (kDimNum == 5U) { // NZ: block_num, N, D/(32/sizeof(KV_T)), block_size, (32/sizeof(KV_T))
                inputKvLayout_ = IfaLayout::NZ;
            } else {
                OP_LOGE(ifaContext_->opName, "The dim of keyShape[%u] should be one of 3,4,5.", kDimNum);
                    return ge::GRAPH_FAILED;
            }
        } else {
            if (kDimNum == 3U) { // BSH
                inputLayout_ = IfaLayout::BSH_BSND;
            } else { //BNSD
                inputLayout_ = IfaLayout::BNSD;
            }
        }
    }

    const std::string inputLayoutStr = ifaContext_->layOut;
    bool isPAinputLayoutStr = inputLayoutStr != "BNSD" && inputLayoutStr != "TND" && inputLayoutStr != "TND_NTD" &&
        inputLayoutStr != "BNSD_NBSD" && *ifaContext_->innerPrecise != ATB_INNER_PRECISE;
    if (inputQType_ == ge::DT_INT8 && inputKvType_ == ge::DT_INT8 && ifaContext_->keyRope.tensor != nullptr && ifaContext_->queryRope.tensor != nullptr && headDim_ == 512) { // 512 : for MLA
        OP_CHECK_IF((kDimNum == DIM_BNSD),
                    OP_LOGE(ifaContext_->opName, "when the dtype of query is int8 in MLA, KV layout must be NZ"),
                    return ge::GRAPH_FAILED);        
    } else {
        OP_CHECK_IF((kDimNum == DIM_BNSD && isPAinputLayoutStr),
                    OP_LOGE(ifaContext_->opName, "when Page Attention scene, kvcache is BNBD, query layout must be BNSD, BNSD_NBSD, TND, TND_NTD or BNSD_BSND"),
                    return ge::GRAPH_FAILED);   
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::KvShapePostProcess()
{
    if (pageAttentionFlag_) {
        return ProcessPageAttentionFlag();
    }

    auto batchOfQuery = ifaContext_->query.shape->GetStorageShape().GetDim(0);
    auto batchOfKey = ifaContext_->key.shape->GetStorageShape().GetDim(0);

    uint32_t tmpSeqSize = 0U;
    for (size_t i = 0U; i < ifaContext_->kCache.size(); i++) {
        auto keyShape = ifaContext_->kCache[i];
        auto valueShape = ifaContext_->vCache[i];

        if ((keyShape == nullptr) || (valueShape == nullptr)) {
            OP_LOGE(ifaContext_->opName,
                "kv tensor list length should be greater than or equal to q batch, "
                "kv tensor list index[%lu] is null, q batch: %ld",i, batchOfQuery);
            return ge::GRAPH_FAILED;
        }

        if (!ShapeEqual(keyShape->GetStorageShape(), valueShape->GetStorageShape())) {
            OP_LOGE(ifaContext_->opName, "k v shape shoud be same ");
            return ge::GRAPH_FAILED;
        }

        if ((!(pageAttentionFlag_ || batchOfQuery == batchOfKey) && CheckKVShape(i, keyShape, valueShape) != ge::GRAPH_SUCCESS) ||
            CheckKeyShapeTensor(keyShape->GetStorageShape()) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        uint32_t seqSize;
        if (inputLayout_ == IfaLayout::BSH_BSND) {
            seqSize = keyShape->GetStorageShape().GetDim(1);
        } else {
            seqSize = keyShape->GetStorageShape().GetDim(2); // 2, dim of S
        }

        /* 原则上空tensor为S=0，兼容ShapeSize=0场景 */
        if (seqSize != 0U && keyShape->GetStorageShape().GetShapeSize() == 0) {
            seqSize = 0U;
        }

        if (seqSize == 0U) {
            hasZeroSeqKVTensor_ = true;
        }
        if (i == 0U) {
            tmpSeqSize = seqSize;
        } else {
            if (seqSize != tmpSeqSize) {
                isSameSeqAllKVTensor_ = false;
            }
        }

        sMax_ = std::max(seqSize, sMax_);
        kvListSeqLens_.push_back(seqSize);
    }
    seqSize_ = sMax_;
    inputKvLayout_ = inputLayout_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ZeroTensorProcess()
{
    if (sMax_ == 0U) {
        /*
         * 1024，空tensor场景下，作为默认值完成后续计算
         * 避免matmal tiling  softmax tiling异常
         * kernel计算使用真实的seqSize=0, 与actuseq_len流程归一
         */
        seqSize_ = 1024U;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::InitInOutMode()
{
    if (inputQType_ == ge::DT_INT8 && outputType_ == ge::DT_INT8) {
        inOutMode_ = TilingInOutMode::INT8_INT8;
    } else if (inputQType_ == ge::DT_INT8 && outputType_ == ge::DT_FLOAT16) {
        inOutMode_ = TilingInOutMode::INT8_FP16;
    } else if (inputQType_ == ge::DT_FLOAT16 && outputType_ == ge::DT_INT8) {
        inOutMode_ = TilingInOutMode::FP16_INT8;
    } else if (inputQType_ == ge::DT_FLOAT16 && outputType_ == ge::DT_FLOAT16) {
        inOutMode_ = TilingInOutMode::FP16_FP16;
    } else if (inputQType_ == ge::DT_BF16 && outputType_ == ge::DT_BF16) {
        inOutMode_ = TilingInOutMode::BF16_BF16;
    } else if (inputQType_ == ge::DT_BF16 && outputType_ == ge::DT_INT8) {
        inOutMode_ = TilingInOutMode::BF16_INT8;
    } else if (inputQType_ == ge::DT_FLOAT && outputType_ == ge::DT_FLOAT) {
        inOutMode_ = TilingInOutMode::FP32_FP32;
    } else if (inputQType_ == ge::DT_INT8 && outputType_ == ge::DT_BF16) {
        inOutMode_ = TilingInOutMode::INT8_BF16;
    } else {
        OP_LOGE(ifaContext_->opName, "input dtype %d with output dtype %d is not currently supported.", inputQType_,
                  outputType_);
        return ge::GRAPH_FAILED;
    }
    if ((socVersion_ == IfaSocVersion::SOC_ASCEND_310P) && (inOutMode_ != TilingInOutMode::FP16_FP16)) {
        OP_LOGE(ifaContext_->opName,
            "input dtype float16 with output dtype float16 is currently supported when 310P, but "
            "current input dtype is %d and output dtype is %d",
            inputQType_, outputType_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ProcessOptionalTensors()
{
    if ((ProcessActualSeqLen() != ge::GRAPH_SUCCESS) ||
        (ProcessPseShift() != ge::GRAPH_SUCCESS) ||
        (ProcessQuant1() != ge::GRAPH_SUCCESS) ||
        (ProcessQuant2() != ge::GRAPH_SUCCESS) ||
        (ProcessDequant1() != ge::GRAPH_SUCCESS) ||
        (ProcessDequant2() != ge::GRAPH_SUCCESS) ||
        (ProcessQuant() != ge::GRAPH_SUCCESS) ||
        (ProcessAntiQuant() != ge::GRAPH_SUCCESS) ||
        (ProcessAttenMask() != ge::GRAPH_SUCCESS) ||
        (ProcessBlockTable() != ge::GRAPH_SUCCESS) ||
        (ProcessKVPaddingSize() != ge::GRAPH_SUCCESS) ||
        (ProcessMlaRope() != ge::GRAPH_SUCCESS) ||
        (ProcessCvRatio() != ge::GRAPH_SUCCESS) ||
        (ProcessGqaKvNz() != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }

    // for kv shared prefix
    if ((ProcessSharedPrefix() != ge::GRAPH_SUCCESS) || (ProcessSharedPrefixLen() != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ProcessActualSeqLen()
{
    if (inputLayout_ == IfaLayout::TND) {
        if (isWorkspace_) {
            actualSeqLenFlag_ = true;
            maxActualseq_ = sMax_;
            return ge::GRAPH_SUCCESS;
        }

        if (CheckActualSeqLens() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        actualSeqLenFlag_ = true;
        maxActualseq_ = sMax_;
    } else {
        if (ifaContext_->actualSeqLengths.tensor == nullptr) {
            OP_LOGD(ifaContext_->opName, "the key/value's actual sequence lengths is nullptr");
            maxActualseq_ = sMax_;

            // pa场景必须带actual_seq_lens；第1次tiling调用时(isWorkspace为true)
            // actualSeqLengths会被强制置None，需要跳过校验
            OP_LOGD(ifaContext_->opName, "isWorkspace: %d", isWorkspace_);
            if (pageAttentionFlag_ && (!isWorkspace_)) {
                OP_LOGE(ifaContext_->opName,
                    "the key/value's actual sequence lengths is null, it must exist in pageAttention scene");
                return ge::GRAPH_FAILED;
            }
        return ge::GRAPH_SUCCESS;
        }
        OP_LOGD(ifaContext_->opName, "the key/value's actual sequence lengths is not nullptr");

        actualSeqLenFlag_ = true;
        actualLenDims_ = static_cast<uint32_t>(ifaContext_->actualSeqLengths.tensor->GetShapeSize());
        OP_LOGD(ifaContext_->opName, "number of elements in the key/value's actual sequence lengths is %u", actualLenDims_);
        if (actualLenDims_ == 0U) {
            // pa场景必须带actual_seq_lens
            if (pageAttentionFlag_) {
                OP_LOGW(ifaContext_->opName, "the key/value's actual sequence lengths size[%u] can not be zero in pageAttention scene",
                        actualLenDims_);
            }
            maxActualseq_ = sMax_;
            return ge::GRAPH_SUCCESS;
        }

        OP_CHECK_IF(actualLenDims_ != 1U && actualLenDims_ < batchSize_,
            OP_LOGE(ifaContext_->opName,
                "the key/value's actual sequence lengths size[%u] should be greater than q batch[%u] or equal to 1.",
                actualLenDims_, batchSize_),
            return ge::GRAPH_FAILED);

        actualLenDims_ = std::min(actualLenDims_, batchSize_);
    }

    if (ParseActualSeqLens() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckActualSeqLens()
{
    // Q的actual_seq要求非空
    if (ifaContext_->actualSeqLengthsQ.tensor == nullptr) {
        OP_LOGE(ifaContext_->opName, "TND the query's actual sequence lengths should not be null!");
        return ge::GRAPH_FAILED;
    }
    actualLenQDims_ = static_cast<uint32_t>(ifaContext_->actualSeqLengthsQ.tensor->GetShapeSize());
    if (actualLenQDims_ == 0U) {
        OP_LOGE(ifaContext_->opName, "TND actualLenQDims_ is 0!");
        return ge::GRAPH_FAILED;
    }

    // KV的actual_seq要求非空
    if (ifaContext_->actualSeqLengths.tensor == nullptr) {
        OP_LOGE(ifaContext_->opName, "TND the key/value's actual sequence lengths should not be null!");
        return ge::GRAPH_FAILED;
    }
    actualLenDims_ = ifaContext_->actualSeqLengths.tensor->GetShapeSize();
    if (actualLenDims_ == 0U) {
        OP_LOGE(ifaContext_->opName, "TND actualLenDims_ is 0!");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ParseActualSeqLens()
{
    const int64_t *actualLenData = ifaContext_->actualSeqLengths.tensor->GetData<int64_t>();
    if (actualLenData != nullptr) {
        OP_LOGD(ifaContext_->opName, "the data for the actual sequence lengths of key/value is not nullptr");
        uint32_t loop = ((actualLenDims_ == 1U) && (kvListSeqLens_.size() == 1U)) ? 1U : batchSize_;
        for (uint32_t i = 0U; i < loop; i++) {
            int64_t actLen = (actualLenDims_ == 1U) ? actualLenData[0] : actualLenData[i];
            OP_CHECK_IF(
                actLen < 0, // actualSeqLengths必须大于0
                OP_LOGE(ifaContext_->opName,
                          "the value of the key/value's actual sequence lengths[%u] must be greater than or equal to 0, but it is %ld", i,
                          actLen),
                return ge::GRAPH_FAILED);
            OP_LOGI(ifaContext_->opName, "The vlaue of the key/value's actual sequence lengths[%u] is %ld.", i, actLen);
            if (!pageAttentionFlag_) {
                uint32_t seqSize = (kvListSeqLens_.size() == 1) ? kvListSeqLens_[0] : kvListSeqLens_[i];
                OP_CHECK_IF(static_cast<uint32_t>(actLen) > seqSize,
                    OP_LOGE(ifaContext_->opName,
                        "the key/value's actual sequence lengths[%u](%ld) cannot be greater than seq_length(%u) in input key.",
                        i, actLen, seqSize),
                    return ge::GRAPH_FAILED);
            }
            maxActualseq_ =
                maxActualseq_ < static_cast<uint32_t>(actLen) ? static_cast<uint32_t>(actLen) : maxActualseq_;
            if (actLen == 0) {
                hasZeroActualseq_ = true;
            }
            if (actualLenData[i] != actualLenData[0]) {
                isSameActualseq_ = false;
            }
        }
    } else {
        OP_LOGD(ifaContext_->opName, "data of the key/value's actual sequence lengths is nullptr");
        // pa场景必须带actual_seq_lens
        if (pageAttentionFlag_ && (!isWorkspace_)) {
            OP_LOGW(ifaContext_->opName, "data of the key/value's actual sequence lengths can not be nullptr in pageAttention scene");
        }
        maxActualseq_ = sMax_;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ProcessQuant1() const
{
    auto dqtScale1 = ifaContext_->deqScale1.tensor;
    auto qtScale1 = ifaContext_->quantScale1.tensor;
    auto dqtScale2 = ifaContext_->deqScale2.tensor;

    if (inputQType_ != ge::DT_INT8) {
        OP_CHECK_IF(dqtScale1 != nullptr,
                   OP_LOGE(ifaContext_->opName, "when input type is not int8, dqtScale1 should be null"),
                   return ge::GRAPH_FAILED);
        OP_CHECK_IF(qtScale1 != nullptr,
                   OP_LOGE(ifaContext_->opName, "when input type is not int8, qtScale1 should be null"),
                   return ge::GRAPH_FAILED);
        OP_CHECK_IF(dqtScale2 != nullptr,
                   OP_LOGE(ifaContext_->opName, "when input type is not int8, dqtScale2 should be null"),
                   return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckQueryQuantParam4FullQuant(const gert::Shape dequantScaleQueryShape) const
{
    auto queryShape = ifaContext_->query.shape->GetStorageShape();
    std::string layout(ifaContext_->layOut);
    if (layout == "BSH" || layout == "BSH_NBSD") {
        OP_CHECK_IF(queryShape.GetDimNum() != (dequantScaleQueryShape.GetDimNum()),
            OP_LOGE(ifaContext_->opName,
                "when the dtype of query is int8, the dim of the query's dequant scale should be %d, but it is %d",
                static_cast<int32_t>(queryShape.GetDimNum()), static_cast<int32_t>(dequantScaleQueryShape.GetDimNum())),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(queryShape.GetDim(0) != dequantScaleQueryShape.GetDim(0),
                OP_LOGE(ifaContext_->opName,
                    "the %drd dim of the query's dequant scale is %d, the %drd dim of query is %d, "
                    "they should be same when the dtype of query is int8.",
                    0, static_cast<int32_t>(dequantScaleQueryShape.GetDim(0)), 0, static_cast<int32_t>(queryShape.GetDim(0))),
                return ge::GRAPH_FAILED);
        OP_CHECK_IF(queryShape.GetDim(1) != dequantScaleQueryShape.GetDim(1),
                OP_LOGE(ifaContext_->opName,
                    "the %drd dim of the query's dequant scale is %d, the %drd dim of query is %d, "
                    "they should be same when the dtype of query is int8.",
                    1, static_cast<int32_t>(dequantScaleQueryShape.GetDim(1)), 1, static_cast<int32_t>(queryShape.GetDim(1))),
                return ge::GRAPH_FAILED);
        OP_CHECK_IF(numHeads_ != dequantScaleQueryShape.GetDim(2), // 2: dim index
                OP_LOGE(ifaContext_->opName,
                    "the %drd dim of the query's dequant scale is %d, the query's heads num of query is %d, "
                    "they should be same when the dtype of query is int8.",
                    2, static_cast<int32_t>(dequantScaleQueryShape.GetDim(2)), static_cast<int32_t>(numHeads_)), // 2: dim index
                return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(queryShape.GetDimNum() != (dequantScaleQueryShape.GetDimNum() + 1),
            OP_LOGE(ifaContext_->opName,
                "when the dtype of query is int8, the dim of the query's dequant scale should be %d, but it is %d",
                static_cast<int32_t>((queryShape.GetDimNum() - 1)), static_cast<int32_t>(dequantScaleQueryShape.GetDimNum())),
            return ge::GRAPH_FAILED);
        for (uint32_t i = 0U; i < dequantScaleQueryShape.GetDimNum(); i++) {
            OP_CHECK_IF(queryShape.GetDim(i) != dequantScaleQueryShape.GetDim(i),
                OP_LOGE(ifaContext_->opName,
                    "the %urd dim of the query's dequant scale is %d, the %urd dim of query is %d, "
                    "they should be same when the dtype of query is int8.",
                    i, static_cast<int32_t>(dequantScaleQueryShape.GetDim(i)), i, static_cast<int32_t>(queryShape.GetDim(i))),
                return ge::GRAPH_FAILED);
        }
    }

    return ge::GRAPH_SUCCESS;
}
ge::graphStatus IFATiling::CheckKVQuantParam4FullQuant(const gert::Shape dequantScaleKVShape) const
{
    OP_CHECK_IF(dequantScaleKVShape.GetDimNum() != 1U,
        OP_LOGE(ifaContext_->opName,
            "when the dtype of query is int8 in MLA, the dim of the key/value's dequant scale should be 1, but it is %d",
            static_cast<int32_t>(dequantScaleKVShape.GetDimNum())),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(dequantScaleKVShape.GetDim(0) != 1U,
        OP_LOGE(ifaContext_->opName,
            "when the dtype of query is int8 in MLA, the %drd dim of the key/value's dequant scale should be 1, but it is %d",
            0, static_cast<int32_t>(dequantScaleKVShape.GetDim(0))),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ProcessQuant()
{
    if (inputQType_ != ge::DT_INT8) {
        OP_CHECK_IF(ifaContext_->dequantScaleQuery.tensor != nullptr,
        OP_LOGE(ifaContext_->opName, "when the dtype of query is not int8, the query's dequant scale should be null"),
        return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(!ropeFlag_,
        OP_LOGE(ifaContext_->opName, "when the dtype of query is int8, query_rope and key rope should not be null"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((inputLayout_ == IfaLayout::BNSD),
        OP_LOGE(ifaContext_->opName, "when the dtype of query is int8 in MLA, layout BNSD/BNSD_NBSD is not support"),
        return ge::GRAPH_FAILED);

    // 全量化暂不支持quantScale1/deqScale1/deqScale2
    OP_CHECK_IF((ifaContext_->quantScale1.tensor != nullptr || ifaContext_->deqScale1.tensor != nullptr || ifaContext_->deqScale2.tensor != nullptr),
        OP_LOGE(ifaContext_->opName, "when the dtype of query is int8, quantScale1/deqScale1/deqScale2 should be null"), return ge::GRAPH_FAILED);

    // 全量化暂不支持atiquantScale/antiquantOffset
    OP_CHECK_IF((ifaContext_->antiquantScale.tensor != nullptr || ifaContext_->antiquantOffset.tensor != nullptr),
        OP_LOGE(ifaContext_->opName,"when the dtype of query is int8 in MLA, antiquantScale/antiquantOffset should be null"), return ge::GRAPH_FAILED);

    // 全量化暂不支持keyAntiquantOffset/valueAntiquantOffset
    OP_CHECK_IF((ifaContext_->keyAntiquantOffset.tensor != nullptr || ifaContext_->valueAntiquantOffset.tensor != nullptr),
        OP_LOGE(ifaContext_->opName,"when the dtype of query is int8 in MLA, the key's/value's dequant offset should be null"), return ge::GRAPH_FAILED);

    if (CheckQkvQuantParams4FullQuant() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF((ifaContext_->queryRope.desc == nullptr || ifaContext_->keyRope.desc == nullptr),
        OP_LOGE(ifaContext_->opName, "when the dtype of query is int8, query_rope and key rope desc should not be null"), return ge::GRAPH_FAILED);
    OP_CHECK_IF((ifaContext_->queryRope.desc->GetDataType() != ge::DT_BF16 ||
        ifaContext_->keyRope.desc->GetDataType() != ge::DT_BF16),
        OP_LOGE(ifaContext_->opName, "when the dtype of query is int8, query_rope and key rope dtype should be bf16"), return ge::GRAPH_FAILED);

    quantFlag_ = true;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckQkvQuantParams4FullQuant() const
{
    auto dequantScaleQuery = ifaContext_->dequantScaleQuery.tensor;
    auto dequantScaleKey = ifaContext_->keyAntiquantScale.tensor;
    auto dequantScaleValue = ifaContext_->valueAntiquantScale.tensor;
    if (dequantScaleQuery != nullptr && dequantScaleKey != nullptr && dequantScaleValue != nullptr) {
        OP_CHECK_IF(CheckQueryQuantParam4FullQuant(dequantScaleQuery->GetStorageShape()) != ge::GRAPH_SUCCESS,
            OP_LOGE(ifaContext_->opName, "The query's dequant scale shape is illegal"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(CheckKVQuantParam4FullQuant(dequantScaleKey->GetStorageShape()) != ge::GRAPH_SUCCESS,
            OP_LOGE(ifaContext_->opName, "dequant_scale_key shape is illegal"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(CheckKVQuantParam4FullQuant(dequantScaleValue->GetStorageShape()) != ge::GRAPH_SUCCESS,
            OP_LOGE(ifaContext_->opName, "dequant_scale_value shape is illegal"), return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(ifaContext_->opName,
            "when the dtype of query is int8, the query's dequant scale, the key's dequant scale, and the value's dequant scale should not be null");
        return ge::GRAPH_FAILED;
    }

    int64_t queryQuantMode = ifaContext_->queryQuantMode != nullptr ? *ifaContext_->queryQuantMode : 0;
    OP_CHECK_IF((queryQuantMode != DEQUANT_PER_TOKEN_HEAD_MODE),
        OP_LOGE(ifaContext_->opName, "when the dtype of query is int8, the query's quant mode should be 3"), return ge::GRAPH_FAILED);

    int64_t keyQuantMode = ifaContext_->keyAntiquantMode != nullptr ? *ifaContext_->keyAntiquantMode : 0;
    int64_t valueQuantMode = ifaContext_->valueAntiquantMode != nullptr ? *ifaContext_->valueAntiquantMode : 0;
    OP_CHECK_IF((keyQuantMode != DEQUANT_PER_CHANNEL_MODE || valueQuantMode != DEQUANT_PER_CHANNEL_MODE),
        OP_LOGE(ifaContext_->opName, "when the dtype of query is int8, the key's quant mode and the value's quant mode should be 0"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((ifaContext_->dequantScaleQuery.desc == nullptr || ifaContext_->keyAntiquantScale.desc == nullptr ||
        ifaContext_->valueAntiquantScale.desc == nullptr),
        OP_LOGE(ifaContext_->opName, "when the dtype of query is int8, the query/key/value's dequant scale desc should not be null"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((ifaContext_->dequantScaleQuery.desc->GetDataType() != ge::DT_FLOAT ||
        ifaContext_->keyAntiquantScale.desc->GetDataType() != ge::DT_FLOAT ||
        ifaContext_->valueAntiquantScale.desc->GetDataType() != ge::DT_FLOAT),
        OP_LOGE(ifaContext_->opName, "when the dtype of query is int8, the query/key/value's dequant scale dtype should be fp32"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ProcessQuant2Dtype()
{
    if (outputType_ == ge::DT_INT8) {
        OP_CHECK_IF(ifaContext_->quantScale2.tensor == nullptr,
            OP_LOGE(ifaContext_->opName, "output data type is int8, but input tensor of the output's dequant scale is null"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(ifaContext_->quantScale2.desc == nullptr,
            OP_LOGE(ifaContext_->opName, "Desc of the output's dequant scale input tensor is null."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(ifaContext_->quantScale2.desc->GetDataType() != ge::DT_BF16 &&
            ifaContext_->quantScale2.desc->GetDataType() != ge::DT_FLOAT,
            OP_LOGE(ifaContext_->opName, "the output's dequant scale type(%d) should be bf16 or fp32",
                ifaContext_->quantScale2.desc->GetDataType()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(ifaContext_->quantOffset2.desc != nullptr &&
            ifaContext_->quantScale2.desc->GetDataType() != ifaContext_->quantOffset2.desc->GetDataType(),
            OP_LOGE(ifaContext_->opName, "the output's dequant scale dtype(%d) and offset dtype(%d) are not the same",
                ifaContext_->quantScale2.desc->GetDataType(), ifaContext_->quantOffset2.desc->GetDataType()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(inputQType_ != ge::DT_BF16 && ifaContext_->quantScale2.desc->GetDataType() == ge::DT_BF16,
            OP_LOGE(ifaContext_->opName, "the output's dequant scale and offset support bf16 when inputQ type is bf16"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            inputKvType_ == ge::DT_INT4 && ifaContext_->quantScale2.tensor != nullptr,
            OP_LOGE(ifaContext_->opName, "PostQuant is not supported if Input Kv Dtype is INT4 or INT32 currently."),
            return ge::GRAPH_FAILED);
        if (ifaContext_->quantScale2.desc->GetDataType() == ge::DT_BF16) {
            isOutQuantTypeBf16_ = true;
        }
    } else {
        OP_CHECK_IF(ifaContext_->quantScale2.tensor != nullptr,
                   OP_LOGE(ifaContext_->opName, "the output's dequant scale exist, output data type should be INT8, but now it's %s",
                   DataTypeToSerialString(outputType_).c_str()), return ge::GRAPH_FAILED);
        OP_CHECK_IF(ifaContext_->quantOffset2.tensor != nullptr,
                   OP_LOGE(ifaContext_->opName, "the output's dequant offset exist, output data type should be INT8, but now it's %s",
                   DataTypeToSerialString(outputType_).c_str()), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ProcessQuant2()
{
    auto qtScale2 = ifaContext_->quantScale2.tensor;
    auto qtOffset2 = ifaContext_->quantOffset2.tensor;
    auto qtScale2Desc = ifaContext_->quantScale2.desc;
    auto qtOffset2Desc = ifaContext_->quantOffset2.desc;

    if (ProcessQuant2Dtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (outputType_ == ge::DT_INT8) {
        if (qtScale2->GetShapeSize() == 1) {
            OP_LOGD(ifaContext_->opName, "the output's dequant scale is a const value.");
        } else {
            OP_LOGD(ifaContext_->opName, "the output's dequant scale is a tensor.");
            if (CheckQuant2Shape(qtScale2->GetStorageShape()) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            isOutQuantPerChnOut_ = true;
        }

        // for offset optional
        if (qtOffset2 != nullptr && qtOffset2Desc != nullptr && qtScale2Desc != nullptr) {
            if (qtScale2Desc->GetDataType() != qtOffset2Desc->GetDataType()) {
                OP_LOGE(ifaContext_->opName, "the output's dequant scale and offset should have the same data type.");
                return ge::GRAPH_FAILED;
            }
            if (qtOffset2->GetShapeSize() == 1) {
                OP_LOGD(ifaContext_->opName, "the output's dequant offset is a const value.");
            } else {
                OP_LOGD(ifaContext_->opName, "the output's dequant offset is a tensor.");
                OP_CHECK_IF(CheckQuant2Shape(qtOffset2->GetStorageShape()) != ge::GRAPH_SUCCESS,
                    OP_LOGE(ifaContext_->opName, "check the output's dequant offset shape failed"), return ge::GRAPH_FAILED);
                isOutQuantPerChnOut_ = true;
            }
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ProcessDequant1() const
{
    if (ifaContext_->deqScale1.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ProcessDequant2() const
{
    if (ifaContext_->deqScale2.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckKVAntiQuantParamsShapeInPagedAttention(const gert::Shape &inputParaShape) const
{
    if (antiquantPerHeadFlag_ != 0U) { // per-token-head, [block_num, kv_head_num, block_size]
        OP_CHECK_IF((inputParaShape.GetDim(0) != totalBlockNum_),
            OP_LOGE(ifaContext_->opName,
                "The 1st dim of antiquant parameter should be %u instead of the current %ld",
                totalBlockNum_, inputParaShape.GetDim(0)),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (inputParaShape.GetDim(1) != numKvHeads_),
            OP_LOGE(ifaContext_->opName,
                "The 2nd dim of antiquant parameter should be %u instead of the current %ld",
                numKvHeads_, inputParaShape.GetDim(1)),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (inputParaShape.GetDim(2) != blockSize_),
            OP_LOGE(ifaContext_->opName,
                "The 3rd dim of antiquant parameter should be %u instead of the current %ld",
                blockSize_, inputParaShape.GetDim(2)),
            return ge::GRAPH_FAILED);
    } else { // per-token, [block_num, block_size]
        OP_CHECK_IF((inputParaShape.GetDim(0) != totalBlockNum_),
            OP_LOGE(ifaContext_->opName,
                "The 1st dim of antiquant parameter should be %u instead of the current %ld",
                totalBlockNum_, inputParaShape.GetDim(0)),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (inputParaShape.GetDim(1) != blockSize_),
            OP_LOGE(ifaContext_->opName,
                "The 2nd dim of antiquant parameter should be %u instead of the current %ld",
                blockSize_, inputParaShape.GetDim(1)),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckKVAntiQuantParamsInPagedAttention() const {
    auto keyAntiquantScaleTensor = ifaContext_->keyAntiquantScale.tensor;
    auto KeyAntiquantScaleShape = keyAntiquantScaleTensor->GetStorageShape();
    if (CheckKVAntiQuantParamsShapeInPagedAttention(KeyAntiquantScaleShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto keyAntiquantOffsetTensor = ifaContext_->keyAntiquantOffset.tensor;
    if (keyAntiquantOffsetTensor != nullptr) {
        auto KeyAntiquantOffsetShape = keyAntiquantOffsetTensor->GetStorageShape();
        if (CheckKVAntiQuantParamsShapeInPagedAttention(KeyAntiquantOffsetShape) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckKVAntiQuantMode()
{
    if (gqaKvNZFlag_ && (antiquantMode_ != DEQUANT_PER_CHANNEL_MODE) && (antiquantMode_ != DEQUANT_PER_TOKEN_MODE)) {
        OP_LOGE(ifaContext_->opName, "antiquantMode value[%u] should be 0 or 1 in GQA KV NZ", antiquantMode_);
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(gqaKvNZFlag_ && inputLayout_ == IfaLayout::TND && antiquantMode_ == DEQUANT_PER_TOKEN_MODE,
        OP_LOGE(ifaContext_->opName, "Per token antiquant mode is not supported when layout is TND in GQA KV NZ."),
        return ge::GRAPH_FAILED);
    if ((antiquantMode_ != DEQUANT_PER_CHANNEL_MODE) &&
            (antiquantMode_ != DEQUANT_PER_TOKEN_MODE) &&
            (antiquantMode_ != DEQUANT_PER_TENSOR_HEAD_MODE) &&
            (antiquantMode_ != DEQUANT_PER_TOKEN_HEAD_MODE) &&
            (antiquantMode_ != DEQUANT_PER_TOKEN_PA_MODE) &&
            (antiquantMode_ != DEQUANT_PER_TOKEN_HEAD_PA_MODE)) {
        OP_LOGE(ifaContext_->opName,
            "antiquantMode value:%u is invalid, it should be 0、1、2、3、4 or 5", antiquantMode_);
        return ge::GRAPH_FAILED;
    }

    if (antiquantMode_ == DEQUANT_PER_TENSOR_HEAD_MODE) { // 2:per tensor head
        antiquantMode_ = PER_CHANNEL_MODE;
        antiquantPerHeadFlag_ = 1U;
    }
    if (antiquantMode_ == DEQUANT_PER_TOKEN_HEAD_MODE) { // 3:per token head
        antiquantMode_ = PER_TOKEN_MODE;
        antiquantPerHeadFlag_ = 1U;
    }
    if (antiquantMode_ == DEQUANT_PER_TOKEN_PA_MODE) { // 4:per token + pageAttention scale/offset
        antiquantMode_ = PER_TOKEN_MODE;
        antiquantParamsInPagedAttentionFlag_ = 1U;
    }
    if (antiquantMode_ == DEQUANT_PER_TOKEN_HEAD_PA_MODE) { // 5:per token head + pageAttention scale/offset
        antiquantMode_ = PER_TOKEN_MODE;
        antiquantPerHeadFlag_ = 1U;
        antiquantParamsInPagedAttentionFlag_ = 1U;
    }
    OP_CHECK_IF((antiquantParamsInPagedAttentionFlag_ != 0U) && !pageAttentionFlag_,
        OP_LOGE(ifaContext_->opName,
            "the key/value's quant mode 4 and 5 use page attention to manage scale/offset, must be used in page attention scene"),
        return ge::GRAPH_FAILED); 
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckKVAntiQuantPerToken(const gert::Shape &inputParaShape) const
{
    if (gqaKvNZFlag_) {
        OP_CHECK_IF(inputParaShape.GetDimNum() != DIM_PER_TOKEN_KvSplit, 
            OP_LOGE(ifaContext_->opName,
            "The dim of antiquant[%lu] should be %u when per_token mode in GQA KV NZ.",
            inputParaShape.GetDimNum(), DIM_PER_TOKEN_KvSplit), return ge::GRAPH_FAILED);
        OP_CHECK_IF((inputParaShape.GetDim(PER_TOKEN_Split_B) != batchSize_),
            OP_LOGE(ifaContext_->opName,
            "The 1st dim of antiquant should be %u instead of the current %ld when per_token mode in GQA KV NZ.",
            batchSize_, inputParaShape.GetDim(PER_TOKEN_Split_B)), return ge::GRAPH_FAILED);
        OP_CHECK_IF((inputParaShape.GetDim(PER_TOKEN_Split_S) < seqSize_),
            OP_LOGE(ifaContext_->opName,
            "The 2nd dim of antiquant should be greater than or equal to %u instead of the current %ld when per_token mode in GQA KV NZ.",
            seqSize_, inputParaShape.GetDim(PER_TOKEN_Split_S)), return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    if (inputParaShape.GetDimNum() == DIM_PER_TOKEN) {
        OP_CHECK_IF((inputParaShape.GetDim(PER_TOKEN_N) != antiquantNum_),
            OP_LOGE(ifaContext_->opName, "The 1st dim of antiquant should be %u instead of the current %ld",
                antiquantNum_, inputParaShape.GetDim(PER_TOKEN_N)),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((inputParaShape.GetDim(PER_TOKEN_B) != batchSize_),
            OP_LOGE(ifaContext_->opName, "The 2nd dim of antiquant should be %u instead of the current %ld",
                batchSize_, inputParaShape.GetDim(PER_TOKEN_B)),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (inputParaShape.GetDim(PER_TOKEN_S) < seqSize_),
            OP_LOGE(ifaContext_->opName,
                "The 3rd dim of antiquant should be greater than or equal to %u instead of the current %ld",
                seqSize_, inputParaShape.GetDim(PER_TOKEN_S)),
            return ge::GRAPH_FAILED);
    } else if (inputParaShape.GetDimNum() == DIM_PER_TOKEN_KvSplit && kvAntiParamSplitFlag_) {
        OP_CHECK_IF((inputParaShape.GetDim(PER_TOKEN_Split_B) != batchSize_),
            OP_LOGE(ifaContext_->opName,
                "The 1st dim of antiquant should be %u instead of the current %ld",
                batchSize_, inputParaShape.GetDim(PER_TOKEN_Split_B)),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (inputParaShape.GetDim(PER_TOKEN_Split_S) < seqSize_),
            OP_LOGE(ifaContext_->opName,
                "The 2nd dim of antiquant should be greater than or equal to %u instead of the current %ld",
                seqSize_, inputParaShape.GetDim(PER_TOKEN_Split_S)),
            return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(ifaContext_->opName, "The dim of antiquant is illegal, When per_token mode.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckKVAntiQuantParaShapeLegal(const gert::Shape &inputParaShape)
{
    if (kvAntiParamSplitFlag_) {
        antiquantNum_ = 1U;
    }
    if ((antiquantMode_ == PER_CHANNEL_MODE) && gqaKvNZFlag_) {
        if (inputParaShape.GetDimNum() == DIM_PER_CHANNEL_KVNZ_BNSD ||
            inputParaShape.GetDimNum() == DIM_PER_CHANNEL_KVNZ_BSND ||
            inputParaShape.GetDimNum() == DIM_PER_CHANNEL_KVNZ_BSH)
            return CheckKVAntiQuantPerChannel(inputParaShape);
    }
    gert::Shape expectParamShapePerTensor = gert::Shape({antiquantNum_});
    if ((antiquantPerHeadFlag_ != 0U) && (antiquantParamsInPagedAttentionFlag_ == 0U)) {
        // 使用pa管理scale offset后，屏蔽原有形状校验
        return CheckKVAntiQuantPerHead(inputParaShape);
    }
    if (antiquantMode_ == PER_TOKEN_MODE) { // per-token
        // 使用pa管理scale offset后，屏蔽原有形状校验
        if (antiquantParamsInPagedAttentionFlag_ != 0U) {
            return ge::GRAPH_SUCCESS;
        }
        return CheckKVAntiQuantPerToken(inputParaShape);
    } else if (inputParaShape.GetDimNum() == DIM_PER_TENSOR) { // per-tensor
        antiquantMode_ = PER_CHANNEL_MODE;
        antiquantPerTensorFlag_ = 1U;
        OP_CHECK_IF((inputParaShape != expectParamShapePerTensor),
            OP_LOGE(ifaContext_->opName,
                "The shape of antiquant parameter[%ld] is not expected. Expect[%u] When per_tensor mode.",
                inputParaShape.GetDim(BH_B_IDX), antiquantNum_),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    } else if (inputParaShape.GetDimNum() == DIM_PER_CHANNEL_BNSD ||
               inputParaShape.GetDimNum() == DIM_PER_CHANNEL_BSND ||
               inputParaShape.GetDimNum() == DIM_BH) { // per-channel
        return CheckKVAntiQuantPerChannel(inputParaShape);
    } else {
        OP_LOGE(ifaContext_->opName, "The layout[%lu] does not match the dim of antiquant, When per_channel mode.",
                  inputParaShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckAntiQuantParamKeyType(const gert::Tensor *antiquantOffsetTensor,
                                                      const gert::CompileTimeTensorDesc *antiquantScaleDesc,
                                                      const gert::CompileTimeTensorDesc *antiquantOffsetDesc) const
{
    ge::DataType antiquantScaleType = antiquantScaleDesc->GetDataType();
    if (antiquantScaleType != inputQType_) {
        OP_LOGE(ifaContext_->opName, "illegal datatype of antiquant scale, it should be same with input qtype");
        return ge::GRAPH_FAILED;
    }

    if (CheckAntiquantOffsetType(antiquantOffsetTensor, antiquantOffsetDesc, antiquantScaleType) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckAntiQuantParamValueType(const gert::Tensor *antiquantOffsetTensor,
                                                        const gert::CompileTimeTensorDesc *antiquantScaleDesc,
                                                        const gert::CompileTimeTensorDesc *antiquantOffsetDesc) const
{
    ge::DataType valueAntiquantScaleType = antiquantScaleDesc->GetDataType();
    if (valueAntiquantScaleType != ge::DT_FLOAT) {
        OP_LOGE(ifaContext_->opName, "per-token mode is enabled, datatype of antiquant scale should be float32 ");
        return ge::GRAPH_FAILED;
    }

    if (CheckAntiquantOffsetType(antiquantOffsetTensor, antiquantOffsetDesc, valueAntiquantScaleType) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ProcessAntiQuant()
{
    auto antiquantScaleTensor = ifaContext_->antiquantScale.tensor;
    auto antiquantOffsetTensor = ifaContext_->antiquantOffset.tensor;
    auto antiquantScaleDesc = ifaContext_->antiquantScale.desc;
    auto antiquantOffsetDesc = ifaContext_->antiquantOffset.desc;
    auto keyAntiquantOffsetTensor = ifaContext_->keyAntiquantOffset.tensor;
    auto keyAntiquantScaleTensor = ifaContext_->keyAntiquantScale.tensor;
    auto valueAntiquantScaleTensor = ifaContext_->valueAntiquantScale.tensor;
    auto valueAntiquantOffsetTensor = ifaContext_->valueAntiquantOffset.tensor;
    auto keyRopeAntiquantScaleTensor = ifaContext_->keyRopeAntiquantScale.tensor;

    if ((!antiQuantFlag_ && (antiquantScaleTensor != nullptr || antiquantOffsetTensor != nullptr ||
                            keyAntiquantScaleTensor != nullptr || keyAntiquantOffsetTensor != nullptr ||
                            valueAntiquantScaleTensor != nullptr || valueAntiquantOffsetTensor != nullptr ||
                            keyRopeAntiquantScaleTensor != nullptr)) && !quantFlag_) {
        OP_LOGE(ifaContext_->opName, "KV antiquant is unenabled, but antiquant antiquantScale/antiquantOffset exist");
        return ge::GRAPH_FAILED;
    }

    if (!antiQuantFlag_) {
        return ge::GRAPH_SUCCESS;
    }

    if (CheckKeyAndValueAntiquantScaleOffset() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint32_t keyAntiquantMode_kvSep = ifaContext_->keyAntiquantMode != nullptr ? static_cast<uint32_t>(*ifaContext_->keyAntiquantMode) : 0U;
    uint32_t valueAntiquantMode_kvSep = ifaContext_->valueAntiquantMode != nullptr ? static_cast<uint32_t>(*ifaContext_->valueAntiquantMode) : 0U;
    if (CheckKeyAndValueAntiquantOffset(keyAntiquantMode_kvSep, valueAntiquantMode_kvSep) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    
    if (kvAntiParamSplitFlag_) {
        if (CheckKvAntiquant4SplitMode() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        if (ProcessAntiQuantMode() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_LOGD(ifaContext_->opName, "kv antiquant is not split mode");
        if (ifaContext_->antiquantMode != nullptr) {
            antiquantMode_ = static_cast<uint32_t>(*ifaContext_->antiquantMode);
        }
        if (CheckAntiQuantParam(antiquantScaleTensor, antiquantOffsetTensor, antiquantScaleDesc, antiquantOffsetDesc) ==
            ge::GRAPH_FAILED) {
            return ge::GRAPH_FAILED;
        }
    }
    antiqSeqSize_ = GetAntiquantSeqLength();
    OP_LOGD(ifaContext_->opName, "antiquant info, iter num:%u, antiquant mode:%u", msdIterNum_, antiquantMode_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckKeyAndValueAntiquantScaleOffset()
{
    auto keyAntiquantScaleTensor = ifaContext_->keyAntiquantScale.tensor;
    auto keyAntiquantOffsetTensor = ifaContext_->keyAntiquantOffset.tensor;
    auto valueAntiquantScaleTensor = ifaContext_->valueAntiquantScale.tensor;
    auto valueAntiquantOffsetTensor = ifaContext_->valueAntiquantOffset.tensor;
    auto keyRopeAntiquantScaleTensor = ifaContext_->keyRopeAntiquantScale.tensor;
  
    kvAntiParamSplitFlag_ = false;
    if (keyAntiquantScaleTensor != nullptr && valueAntiquantScaleTensor == nullptr) {
        OP_LOGE(ifaContext_->opName, "the value's dequant scale is null, but the key's dequant scale exist");
        return ge::GRAPH_FAILED;
    }
    if (valueAntiquantScaleTensor != nullptr && keyAntiquantScaleTensor == nullptr) {
        OP_LOGE(ifaContext_->opName, "the key's dequant scale is null, but the value's dequant scale exist");
        return ge::GRAPH_FAILED;
    }
    if (keyAntiquantOffsetTensor != nullptr && valueAntiquantOffsetTensor == nullptr) {
        OP_LOGE(ifaContext_->opName, "value's dequant offset is null, but the key's dequant offset exist");
        return ge::GRAPH_FAILED;
    }
    if (valueAntiquantOffsetTensor != nullptr && keyAntiquantOffsetTensor == nullptr) {
        OP_LOGE(ifaContext_->opName, "the key's dequant offset is null, but the value's dequant offset exist");
        return ge::GRAPH_FAILED;
    }
    if (keyAntiquantScaleTensor == nullptr && keyAntiquantOffsetTensor != nullptr) {
        OP_LOGE(ifaContext_->opName, "the key's dequant scale is null, but the key's dequant offset exist");
        return ge::GRAPH_FAILED;
    }

    if (ropeFlag_) {
        if (keyAntiquantScaleTensor != nullptr && keyRopeAntiquantScaleTensor == nullptr) {
            OP_LOGE(ifaContext_->opName, "Mla mode: the tensor of the key_rope's dequant scale is null, but the tensor of the key's dequant scale exist");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckKeyAndValueAntiquantOffset(const uint32_t keyAntiquantModeKvSep,const uint32_t valueAntiquantModeKvSep)
{   
    auto keyAntiquantOffsetTensor = ifaContext_->keyAntiquantOffset.tensor;
    auto valueAntiquantOffsetTensor = ifaContext_->valueAntiquantOffset.tensor;
    auto keyAntiquantOffsetDesc = ifaContext_->keyAntiquantOffset.desc;
    auto valueAntiquantOffsetDesc = ifaContext_->valueAntiquantOffset.desc;
    auto keyAntiquantScaleTensor = ifaContext_->keyAntiquantScale.tensor;
    auto valueAntiquantScaleTensor = ifaContext_->valueAntiquantScale.tensor;
   
    if (keyAntiquantOffsetTensor != nullptr && valueAntiquantOffsetTensor != nullptr) {
        OP_CHECK_IF(
            (keyAntiquantOffsetDesc == nullptr),
            OP_LOGE(ifaContext_->opName, "The tensor of the key's dequant offset isn't nullptr, the description of the key's dequant offset is null"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (valueAntiquantOffsetDesc == nullptr),
            OP_LOGE(ifaContext_->opName, "the description of the value's dequant offset isn't nullptr, the description of the value's dequant offset is null"),
            return ge::GRAPH_FAILED);
        if (keyAntiquantModeKvSep != 0U || valueAntiquantModeKvSep != 1U) {
            OP_CHECK_IF(
                (keyAntiquantOffsetDesc->GetDataType() != valueAntiquantOffsetDesc->GetDataType()),
                OP_LOGE(ifaContext_->opName,
                    "the description of the key's and the value's dequant offset should have the same data type"),
                return ge::GRAPH_FAILED);
            if (!ShapeEqual(keyAntiquantOffsetTensor->GetStorageShape(), valueAntiquantOffsetTensor->GetStorageShape())) {
                OP_LOGE(ifaContext_->opName,
                    "the tensor of the key's and the value's dequant offset should have the same shape");
                return ge::GRAPH_FAILED;
            }
        }
    }

    if (keyAntiquantScaleTensor != nullptr && valueAntiquantScaleTensor != nullptr) {
        if (keyAntiquantModeKvSep != 0U || valueAntiquantModeKvSep != 1U) {
            if (!ShapeEqual(keyAntiquantScaleTensor->GetStorageShape(), valueAntiquantScaleTensor->GetStorageShape())) {
                OP_LOGE(ifaContext_->opName,
                    "The tensor of the key's and the value's dequant scale should have the same shape");
                return ge::GRAPH_FAILED;
            }
        }
        kvAntiParamSplitFlag_ = true;
    }
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckKvAntiquant4SplitMode() const
{   
    OP_LOGD(ifaContext_->opName, "kv antiquant is split mode");
    uint32_t keyAntiquantMode = ifaContext_->keyAntiquantMode != nullptr ? static_cast<uint32_t>(*ifaContext_->keyAntiquantMode) : 0U;
    uint32_t valueAntiquantMode = ifaContext_->valueAntiquantMode != nullptr ? static_cast<uint32_t>(*ifaContext_->valueAntiquantMode) : 0U;
    if (keyAntiquantMode != valueAntiquantMode) {
        if (keyAntiquantMode != 0U || valueAntiquantMode != 1U) {
            OP_LOGE(ifaContext_->opName, "the key's quant mode and the value's quant mode should be the same");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ProcessAntiQuantMode()
{   
    auto keyAntiquantOffsetTensor = ifaContext_->keyAntiquantOffset.tensor;
    auto keyAntiquantScaleDesc = ifaContext_->keyAntiquantScale.desc;
    auto keyAntiquantOffsetDesc = ifaContext_->keyAntiquantOffset.desc;
    auto keyAntiquantScaleTensor = ifaContext_->keyAntiquantScale.tensor;
    auto valueAntiquantOffsetTensor = ifaContext_->valueAntiquantOffset.tensor;
    auto valueAntiquantScaleDesc = ifaContext_->valueAntiquantScale.desc;
    auto valueAntiquantOffsetDesc = ifaContext_->valueAntiquantOffset.desc;

    uint32_t keyAntiquantMode = ifaContext_->keyAntiquantMode != nullptr ? static_cast<uint32_t>(*ifaContext_->keyAntiquantMode) : 0U;
    uint32_t valueAntiquantMode = ifaContext_->valueAntiquantMode != nullptr ? static_cast<uint32_t>(*ifaContext_->valueAntiquantMode) : 0U;

    if (keyAntiquantMode == 0U && valueAntiquantMode == 1U) {
        antiquantMode_ = PER_CHANNEL_TOKEN_MODE;
        if (CheckAntiQuantParamKeyType(keyAntiquantOffsetTensor, keyAntiquantScaleDesc,
                                        keyAntiquantOffsetDesc) == ge::GRAPH_FAILED) {
            return ge::GRAPH_FAILED;
        }
        if (CheckAntiQuantParamValueType(valueAntiquantOffsetTensor, valueAntiquantScaleDesc,
                                            valueAntiquantOffsetDesc) == ge::GRAPH_FAILED) {
            return ge::GRAPH_FAILED;
        }
    } else {
        antiquantMode_ = keyAntiquantMode;
        OP_LOGD(ifaContext_->opName, "org antiquantMode value:%u", antiquantMode_);
        if (CheckKVAntiQuantMode() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    if (CheckAntiQuantParam(keyAntiquantScaleTensor, keyAntiquantOffsetTensor, keyAntiquantScaleDesc,
                            keyAntiquantOffsetDesc) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ProcessBlockTable()
{
    if (!pageAttentionFlag_) {
        return ge::GRAPH_SUCCESS;
    }

    // gm到l1，copynd2nz的srcDValue最大支持65535
    if (inputLayout_ == IfaLayout::BSH_BSND) { // 0: BSH
        if ((numKvHeads_ * headDim_ > COPYND2NZ_SRC_STRIDE_LIMITATION)) {
            OP_LOGE(ifaContext_->opName,
                "When input kvcache layout is BSH, the N * D of kvcache is %u, "
                "exceeds the maximum limit (%u) of the datacopy instruction.",
                numKvHeads_ * headDim_, COPYND2NZ_SRC_STRIDE_LIMITATION);
            return ge::GRAPH_FAILED;
        }

        if (slidingFlag_ && (numKvHeads_ * headDimV_ > COPYND2NZ_SRC_STRIDE_LIMITATION)) {
            OP_LOGE(ifaContext_->opName,
                "When input kvcache layout is BSH, the N * D of vcache is %u, "
                "exceeds the maximum limit (%u) of the datacopy instruction.",
                numKvHeads_ * headDimV_, COPYND2NZ_SRC_STRIDE_LIMITATION);
            return ge::GRAPH_FAILED;
        }
    }

    if (CheckPABlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    totalBlockNum_ = ifaContext_->kCache[0]->GetStorageShape().GetDim(0);
    OP_CHECK_IF(
        maxActualseq_ > blockSize_ * maxBlockNumPerBatch_,
        OP_LOGE(ifaContext_->opName,
            "Invalid actual seq length for PA, max actual seq length(%u) "
            "is larger than blocksize(%u) * max block num per batch(%u)",
            maxActualseq_, blockSize_, maxBlockNumPerBatch_),
        return ge::GRAPH_FAILED);

    if (antiquantParamsInPagedAttentionFlag_ != 0U) {
        // 在处理pa相关信息时，才能获取到totalBlockNum_用于scale/offset校验
        if (CheckKVAntiQuantParamsInPagedAttention() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ProcessKVPaddingSize()
{
    auto kvPaddingSize = ifaContext_->kvPaddingSize.tensor;
    if (kvPaddingSize == nullptr) {
        OP_LOGD(ifaContext_->opName, "KVLeftPadding illegal condition: kvPaddingSize.tensor is nullptr: %d",
                  ifaContext_->kvPaddingSize.tensor == nullptr);
        return ge::GRAPH_SUCCESS;
    }

    if (kvPaddingSize->GetStorageShape().GetShapeSize() == 0) {
        OP_LOGD(ifaContext_->opName, "KVLeftPadding illegal condition: kvPaddingSize.tensor shape is empty: %d",
                  kvPaddingSize->GetStorageShape().GetShapeSize() == 0);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus ret = CheckSupportKVLeftPadding();

    return ret;
}

ge::graphStatus IFATiling::ProcessSharedPrefix()
{
    if (ifaContext_->keySharedPrefix.tensor == nullptr && ifaContext_->valueSharedPrefix.tensor == nullptr) {
        sysPrefixFlag_ = false;
        return ge::GRAPH_SUCCESS;
    }

    if (SharedPrefixCheckBasic() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto keyShape = ifaContext_->keySharedPrefix.tensor->GetStorageShape();
    auto valueShape = ifaContext_->valueSharedPrefix.tensor->GetStorageShape();
    if (SharedPrefixCheckShapes(keyShape, valueShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (inputLayout_ == IfaLayout::BSH_BSND) {
        sMaxPrefix_ = keyShape.GetDim(1); // 1:BSH的S维
    } else {
        sMaxPrefix_ = keyShape.GetDim(2); // 2:BNSD的S维
    }

    if (keyShape.GetShapeSize() == 0) { // 兼容空tensor场景
        sMaxPrefix_ = 0U;
    }

    sysPrefixFlag_ = true;

    return ge::GRAPH_SUCCESS;
}

uint32_t IFATiling::GetAntiquantSeqLength() const
{
    if (antiquantParamsInPagedAttentionFlag_ != 0U) {
        return seqSize_;
    }
    if (kvAntiParamSplitFlag_ && ifaContext_->valueAntiquantScale.tensor->GetStorageShape().GetDimNum() != DIM_NUM_THREE) {
        OP_LOGW(ifaContext_->opName, "the pertoken antiquant's dimensions is [%u], which neeb to be (1, Batch, AntiquantSeqlen)",
            ifaContext_->valueAntiquantScale.tensor->GetStorageShape().GetDimNum());
    }
    const size_t antiquantSIdx = (gqaKvNZFlag_) || 
        (kvAntiParamSplitFlag_ && ifaContext_->valueAntiquantScale.tensor->GetStorageShape().GetDimNum() == DIM_NUM_TWO) ? 1U : 2U;
    return kvAntiParamSplitFlag_ ? ifaContext_->valueAntiquantScale.tensor->GetStorageShape().GetDim(antiquantSIdx) :
                                   ifaContext_->antiquantScale.tensor->GetStorageShape().GetDim(antiquantSIdx);
}

ge::graphStatus IFATiling::ProcessSharedPrefixLen()
{
    auto tensor = ifaContext_->actualSharedPrefixLen.tensor;
    if (tensor == nullptr || tensor->GetStorageShape().GetShapeSize() == 0 || !sysPrefixFlag_) {
        maxActualPrefixLen_ = sMaxPrefix_;
        return ge::GRAPH_SUCCESS;
    }

    maxActualPrefixLen_ = sMaxPrefix_;
    auto actulLenShape = ifaContext_->actualSharedPrefixLen.tensor->GetStorageShape();

    OP_CHECK_IF(
        (actulLenShape.GetDimNum() != 1U || actulLenShape.GetDim(0) != 1U),
        OP_LOGE(ifaContext_->opName, "actual shared prefix shape[%lu] must be 1", actulLenShape.GetDimNum()),
        return ge::GRAPH_FAILED);

    actualLenDimsPrefix_ = 1U;
    const int64_t *actualLenData = ifaContext_->actualSharedPrefixLen.tensor->GetData<int64_t>();
    if (actualLenData != nullptr) {
        OP_CHECK_IF(actualLenData[0] < 0,
                   OP_LOGE(ifaContext_->opName, "actual prefix len[%ld] should be >= 0.", actualLenData[0]),
                   return ge::GRAPH_FAILED);
        maxActualPrefixLen_ = static_cast<uint32_t>(actualLenData[0]);
        OP_CHECK_IF(maxActualPrefixLen_ > sMaxPrefix_,
                   OP_LOGE(ifaContext_->opName, "actual prefix len[%u] should not be larger than S[%u] of prefix tensor",
                             maxActualPrefixLen_, sMaxPrefix_),
                   return ge::GRAPH_FAILED);
    }
    
    uint32_t totalS = maxActualPrefixLen_ + maxActualseq_;
    if (pseShiftFlag_) { // 存在pse时才校验
        OP_CHECK_IF((!(sysPrefixFlag_ && actualLenData == nullptr) && totalS > pseShiftS1_),
                   OP_LOGE(ifaContext_->opName, "total kv S Size (with shared prefix)[%u] bigger than pseShift size[%u]",
                             totalS, pseShiftS1_),
                   return ge::GRAPH_FAILED);
    }

    if (attenMaskFlag_) { // 存在attenMask时才校验
        OP_CHECK_IF((!(sysPrefixFlag_ && actualLenData == nullptr) && totalS > attenMaskSize_),
                   OP_LOGE(ifaContext_->opName,
                             "total kv S Size (with shared prefix)[%u] bigger than attenMask size[%u]", totalS,
                             attenMaskSize_),
                   return ge::GRAPH_FAILED);
    }

    if (antiquantMode_ == PER_TOKEN_MODE) {
        uint32_t perTokenSize = GetAntiquantSeqLength();
        OP_CHECK_IF((!(sysPrefixFlag_ && actualLenData == nullptr) && totalS > perTokenSize),
                   OP_LOGE(ifaContext_->opName,
                             "total kv S Size (with shared prefix)[%u] bigger than antiquant perToken size[%u]", totalS,
                             perTokenSize),
                   return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ProcessCvRatio(){
    // CV1:1 只支持MLA 全量化和非量化 
    if ((cvRatio_ == 1) && (!quantFlag_ || !ropeFlag_)) {
        OP_LOGE(ifaContext_->opName, 
            "when CV 1:1, only support MLA non-quantization(QKV type both are FP16 or BF16) "
            "and MLA fully quantization(QKV type both are int8)");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ProcessMlaRope()
{
    if (!ropeFlag_) {
        return ge::GRAPH_SUCCESS;
    }
    if (CheckMlaQueryRope() != ge::GRAPH_SUCCESS || CheckMlaAttrs() != ge::GRAPH_SUCCESS ||
        CheckMlaKeyRope() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return CheckMlaMisc();
}

ge::graphStatus IFATiling::Split()
{
    amlaMode_ = GetAmlaMode();
    if (balanceModeFlag_) {
        if (isWorkspace_) {
            CalcInnerSize(seqSize_);
            OP_LOGI(ifaContext_->opName, "aicNum_(use for usedCoreNum_ and kvSplitPart_):%u\n", aicNum_);
            usedCoreNum_ = aicNum_;
            splitKVFlag_ = true;
            kvSplit_++;
            kvSplitPart_ = aicNum_;
            return ge::GRAPH_SUCCESS;
        } else if (tilingDataMla_.tndSplitCoreParams.get_FdBalanceFlag() == 1) {
            return SplitBalancedFd();
        } else {
            return SplitBalanced();
        }
    } else {
        return SplitUnbalanced();
    }
}

ge::graphStatus IFATiling::ProcessGqaKvNz() const
{
    if (!gqaKvNZFlag_) {
        return ge::GRAPH_SUCCESS;
    }
    if (CheckGqaTensor() != ge::GRAPH_SUCCESS || CheckGqaAttribute() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void IFATiling::GetActualSeqInfo(const int64_t *actualSeqKv, ActualSeqInfo &actualSeqInfo) const
{
    uint32_t bSize = batchSize_;
    if (inputLayout_ == IfaLayout::TND) {
        // TND格式，actual_seq_q定义为累积长度，这里做转化再分核
        const int64_t *actualSeqQTnd = ifaContext_->actualSeqLengthsQ.tensor->GetData<int64_t>();
        actualSeqInfo.actualSeqQ[0] = actualSeqQTnd[0];
        for (int32_t b = 1; b < static_cast<int>(bSize); b++) {
            actualSeqInfo.actualSeqQ[b] = actualSeqQTnd[b] - actualSeqQTnd[b - 1];
            if (actualLenDims_ != 1U) {
                actualSeqInfo.maxActualseqkv = std::max(actualSeqInfo.maxActualseqkv, actualSeqKv[b]);
            }
        }
    } else {
        for (int32_t b = 0; b < static_cast<int>(bSize); b++) {
            actualSeqInfo.actualSeqQ[b] = qSeqSize_; // 需要检查
            if (actualLenDims_ != 1U) {
                actualSeqInfo.maxActualseqkv = std::max(actualSeqInfo.maxActualseqkv, actualSeqKv[b]);
            }
        }
    }
}

void IFATiling::GetSeqTilingInfo(const int64_t *actualSeqKv,
                                 const ActualSeqInfo &actualSeqInfo, SeqTilingInfo &seqTilingInfo) const
{
    uint32_t bSize = batchSize_;
    uint32_t souter = s1SplitSize_;
    uint32_t s2BasicSize = sInnerSize_;
    uint32_t n2Size = numKvHeads_;
    uint64_t totalS2Length = 0U;
    for (uint32_t bIdx = 0U; bIdx < bSize; bIdx++) {
        uint32_t s1 = actualSeqInfo.actualSeqQ[bIdx];
        uint64_t actualSeqLen = static_cast<uint64_t>(actualSeqKv[0]);
        if (actualLenDims_ != 1U) {
            actualSeqLen = static_cast<uint64_t>(actualSeqKv[bIdx]);
        }
        if (actualSeqInfo.maxActualseqkv <= 0) { //kv actseq全为0
            seqTilingInfo.s1OuterNum[bIdx] = 1U; //kv actseq全为0时，s1不切
            seqTilingInfo.s2OuterNum[bIdx] = 1U; //kv actseq全为0时，每个batch s2份数强制为1
        } else {
            seqTilingInfo.s1OuterNum[bIdx] = (s1 + (souter - 1U)) / souter; // s1总共切多少分，即线段个数
            seqTilingInfo.s2OuterNum[bIdx] = static_cast<uint32_t>((actualSeqLen + s2BasicSize - 1U) / s2BasicSize); // 线段长度
        }

        uint32_t bTotalS2Length = seqTilingInfo.s1OuterNum[bIdx] * seqTilingInfo.s2OuterNum[bIdx] * n2Size; // 线段个数*线段长度*N2=当前batch线段总长度

        totalS2Length += bTotalS2Length; // 更新一次总长度
        OP_LOGI(ifaContext_->opName, "s1OuterNum[%u]:%u, s2OuterNum[%u]:%u\n",
            bIdx, seqTilingInfo.s1OuterNum[bIdx], bIdx, seqTilingInfo.s2OuterNum[bIdx]);
        if ((seqTilingInfo.s1OuterNum[bIdx] > 0U) && (seqTilingInfo.s2OuterNum[bIdx] > 0U)) {
            seqTilingInfo.lastValidBIdx = bIdx;
        }
    }
    if (totalS2Length > coreNum_) {
        seqTilingInfo.avgS2Length = (totalS2Length + coreNum_ - 1U) / coreNum_; // 平均长度向上取整
    }
    OP_LOGI(ifaContext_->opName, "totalS2Length:%lu, avgS2Length:%lu, coreNum_:%u\n",
        totalS2Length, seqTilingInfo.avgS2Length, coreNum_);
}

void IFATiling::FillBalancedSplitCoreInfo(const TilingIndexes &tilingIdx, BalancedSplitTilingInfo &tilingInfo)
{
    uint32_t *coreBEnd = tilingDataMla_.increFlashAttentionCoreParams.get_coreBEnd();
    uint32_t *coreS1OuterEnd = tilingDataMla_.increFlashAttentionCoreParams.get_coreS1OuterEnd();
    uint32_t *coreS2End = tilingDataMla_.increFlashAttentionCoreParams.get_coreS2End();

    coreBEnd[tilingInfo.currCoreIdx] = tilingIdx.bIdx;
    coreS1OuterEnd[tilingInfo.currCoreIdx] = tilingIdx.s1Idx;
    coreS2End[tilingInfo.currCoreIdx] = tilingIdx.s2Idx;
    tilingInfo.coreLoad[tilingInfo.currCoreIdx] = tilingInfo.accumS2Length;
    tilingInfo.currCoreIdx += 1U;
}

void IFATiling::EndSplitForCurrentCore(const TilingIndexes &tilingIdx,
    const SeqTilingInfo &seqTilingInfo, uint32_t &currKvSplitPart, BalancedSplitTilingInfo &tilingInfo)
{
    uint32_t *balanceFDCoreStartKVSplitNum = tilingDataMla_.tndSplitCoreParams.get_balanceFDCoreStartKVSplitNum();
    tilingInfo.accumS2Length += 1U;
    // 更新当前核的End分核信息
    FillBalancedSplitCoreInfo(tilingIdx, tilingInfo);
    if (tilingIdx.s2Idx < seqTilingInfo.s2OuterNum[tilingIdx.bIdx] - 1U) { // 只有切到S2的中间位置，才涉及规约，将currKvSplitPart加1
        currKvSplitPart += 1U;
        balanceFDCoreStartKVSplitNum[tilingInfo.currCoreIdx] = currKvSplitPart - 1U;
    } else {
        balanceFDCoreStartKVSplitNum[tilingInfo.currCoreIdx] = 0U;
    }
    tilingInfo.needUpdate = false;
}

void IFATiling::SplitBalancedForEachHeadFd(
    uint32_t bIdx, const SeqTilingInfo &seqTilingInfo, BalancedSplitTilingInfo &tilingInfo, std::vector<int64_t> &gS1SplitNumOfFdHead, uint32_t s1)
{
    uint32_t *balanceFDCoreBArr = tilingDataMla_.tndSplitCoreParams.get_balanceFDCoreBArr();
    uint32_t *balanceFDCoreS1Arr = tilingDataMla_.tndSplitCoreParams.get_balanceFDCoreS1Arr();
    uint32_t *balanceFDCoreKVSplitArr = tilingDataMla_.tndSplitCoreParams.get_balanceFDCoreKVSplitArr();
    uint32_t *balanceFDCoreStartKVSplitNum = tilingDataMla_.tndSplitCoreParams.get_balanceFDCoreStartKVSplitNum();
    balanceFDCoreStartKVSplitNum[0] = 0U;
    uint32_t *gS1LastPartSizeOfFdHead = tilingDataMla_.tndSplitCoreParams.get_gS1LastPartSizeOfFdHead();

    for (uint32_t s1OuterIdx = 0U; s1OuterIdx < seqTilingInfo.s1OuterNum[bIdx]; s1OuterIdx++) {
        uint32_t currKvSplitPart = 1U; // [B,N2,S1]确定后，S2被切了几份
        for (uint32_t s2Idx = 0U; s2Idx < seqTilingInfo.s2OuterNum[bIdx]; s2Idx++) {
            // 计算当前的目标权重
            uint64_t targetS2Length = static_cast<uint64_t>(tilingInfo.currCoreIdx + 1U) * seqTilingInfo.avgS2Length;
            if (tilingInfo.accumS2Length + 1U >= targetS2Length && tilingInfo.currCoreIdx + 2U <= aicNum_) {
                EndSplitForCurrentCore(TilingIndexes(bIdx, s1OuterIdx, s2Idx), seqTilingInfo, currKvSplitPart, tilingInfo);
            } else {
                tilingInfo.accumS2Length += 1U;
                tilingInfo.needUpdate = true;
            }
        }
        tilingInfo.maxKvSplitPart = std::max(tilingInfo.maxKvSplitPart, currKvSplitPart);
        if (currKvSplitPart > 1U) {
            // S2被切过了，需要规约，记录[B,N,S1]三根轴的idx和切分份数，用于规约
            balanceFDCoreBArr[tilingInfo.tndFDCoreArrLen] = bIdx;
            balanceFDCoreS1Arr[tilingInfo.tndFDCoreArrLen] = s1OuterIdx;
            balanceFDCoreKVSplitArr[tilingInfo.tndFDCoreArrLen] = currKvSplitPart;
            int64_t gSize = nNumOfQInOneGroup_;
            int64_t currFdS1gSize = (s1OuterIdx == seqTilingInfo.s1OuterNum[bIdx] - 1) ? 
                                     s1 * gSize  - (seqTilingInfo.s1OuterNum[bIdx] - 1) * s1SplitSize_ * gSize : s1SplitSize_ * gSize; // 处理尾块
            int64_t currFdS1gSplitSize = (currFdS1gSize + mFdBaseSizeMla - 1) / mFdBaseSizeMla;
            int64_t currFdS1gLastPartSize = currFdS1gSize % mFdBaseSizeMla;
            if (currFdS1gLastPartSize == 0U) {
                currFdS1gLastPartSize = mFdBaseSizeMla;
            }
            gS1SplitNumOfFdHead[tilingInfo.tndFDCoreArrLen] = currFdS1gSplitSize;
            gS1LastPartSizeOfFdHead[tilingInfo.tndFDCoreArrLen] = currFdS1gLastPartSize;
            tilingInfo.tndFDCoreArrLen += 1U;
        }
    }
}

void IFATiling::SplitBalancedForEachHead(
    uint32_t bIdx, const SeqTilingInfo &seqTilingInfo, BalancedSplitTilingInfo &tilingInfo)
{
    uint32_t *balanceFDCoreBArr = tilingDataMla_.tndSplitCoreParams.get_balanceFDCoreBArr();
    uint32_t *balanceFDCoreS1Arr = tilingDataMla_.tndSplitCoreParams.get_balanceFDCoreS1Arr();
    uint32_t *balanceFDCoreKVSplitArr = tilingDataMla_.tndSplitCoreParams.get_balanceFDCoreKVSplitArr();
    uint32_t *balanceFDCoreStartKVSplitNum = tilingDataMla_.tndSplitCoreParams.get_balanceFDCoreStartKVSplitNum();
    balanceFDCoreStartKVSplitNum[0] = 0U;
  
    for (uint32_t s1OuterIdx = 0U; s1OuterIdx < seqTilingInfo.s1OuterNum[bIdx]; s1OuterIdx++) {
        uint32_t currKvSplitPart = 1U; // [B,N2,S1]确定后，S2被切了几份
        for (uint32_t s2Idx = 0U; s2Idx < seqTilingInfo.s2OuterNum[bIdx]; s2Idx++) {
            // 计算当前的目标权重
            uint64_t targetS2Length = static_cast<uint64_t>(tilingInfo.currCoreIdx + 1U) * seqTilingInfo.avgS2Length;
            if (tilingInfo.accumS2Length + 1U >= targetS2Length && tilingInfo.currCoreIdx + 2U <= aicNum_) {
                EndSplitForCurrentCore(TilingIndexes(bIdx, s1OuterIdx, s2Idx), seqTilingInfo, currKvSplitPart, tilingInfo);
            } else {
                tilingInfo.accumS2Length += 1U;
                tilingInfo.needUpdate = true;
            }
        }
        tilingInfo.maxKvSplitPart = std::max(tilingInfo.maxKvSplitPart, currKvSplitPart);
        if (currKvSplitPart > 1U) {
            // S2被切过了，需要规约，记录[B,N,S1]三根轴的idx和切分份数，用于规约
            balanceFDCoreBArr[tilingInfo.tndFDCoreArrLen] = bIdx;
            balanceFDCoreS1Arr[tilingInfo.tndFDCoreArrLen] = s1OuterIdx;
            balanceFDCoreKVSplitArr[tilingInfo.tndFDCoreArrLen] = currKvSplitPart;
            tilingInfo.tndFDCoreArrLen += 1U;
        }
    }
}

void IFATiling::SplitFDMLa(uint32_t tndFDCoreArrLen, std::vector<int64_t> &gS1SplitNumOfFdHead, uint32_t *s2SplitNumOfFdHead, uint32_t aivCoreNum, SeqTilingInfo &seqTilingInfo)
{ 
    uint32_t *gS1IdxEndOfFdHead = tilingDataMla_.tndSplitCoreParams.get_gS1IdxEndOfFdHead();
    uint32_t *gS1IdxEndOfFdHeadSplit = tilingDataMla_.tndSplitCoreParams.get_gS1IdxEndOfFdHeadSplit();
    
    uint32_t totalFDLoad = 0U;
    uint32_t totalFDHeadSplit = 0U;
    
    for (uint32_t i = 0U; i <  tndFDCoreArrLen; i++) { 
        totalFDLoad += s2SplitNumOfFdHead[i] * gS1SplitNumOfFdHead[i];
        totalFDHeadSplit += gS1SplitNumOfFdHead[i];
    }
    // 基于FA开核数量，计算每个Vector需要计算的FD数据量
    uint32_t maxVectorNum = std::min(totalFDHeadSplit, aivCoreNum);  // FD均衡的最小单位为一个归约任务的一个split，所以最多占用totalFDHeadSplit个vector
    double loadThrOfVector = static_cast<double>(totalFDLoad) / static_cast<double>(maxVectorNum);  // 初始化vector的负载上限
    int64_t loadOfCurVector = 0;
    uint32_t curCoreIndex = 0U;
    uint32_t preTmpFDIndexEndOfFdHead = 0U;
    uint32_t preTmpFDIndexEndOfFdHeadSplit = 0U;

    for (uint32_t i = 0U; i <  tndFDCoreArrLen; i++) {
        uint32_t fDKVSplitNum = s2SplitNumOfFdHead[i];
        for (uint32_t gS1SplitIdx = 0U; gS1SplitIdx < gS1SplitNumOfFdHead[i]; gS1SplitIdx++) {
            double remainSpace = loadThrOfVector - loadOfCurVector;  // 计算当前vector剩余负载空间
            // 判断是否放在当前vector的标准是剩余空间是否能容纳一半当前归约块
            if (fDKVSplitNum > remainSpace * MAX_SPLIT_RATIO) {
                gS1IdxEndOfFdHead[curCoreIndex] = preTmpFDIndexEndOfFdHead;
                gS1IdxEndOfFdHeadSplit[curCoreIndex] = preTmpFDIndexEndOfFdHeadSplit;
                curCoreIndex += 1U;
                totalFDLoad -= static_cast<uint32_t>(loadOfCurVector);  // 当前未分配的总负载
                loadThrOfVector = static_cast<double>(totalFDLoad) / static_cast<double>(maxVectorNum - curCoreIndex);  // 根据剩余负载和剩余可用vector更新负载上限，保证最后一个vector能分配所有负载
                loadOfCurVector = 0;
            }
            loadOfCurVector += fDKVSplitNum;
            preTmpFDIndexEndOfFdHead = i;
            preTmpFDIndexEndOfFdHeadSplit = gS1SplitIdx;
        }
    }
    gS1IdxEndOfFdHead[curCoreIndex] = preTmpFDIndexEndOfFdHead;
    gS1IdxEndOfFdHeadSplit[curCoreIndex] = preTmpFDIndexEndOfFdHeadSplit;
    tilingDataMla_.tndSplitCoreParams.set_usedVecNumOfFd(curCoreIndex + 1U);
}

ge::graphStatus IFATiling::SplitBalancedFd()
{
    CalcInnerSize(seqSize_);

    uint32_t s1gBasicSize = FIA_BALANCE_SG_BASIC_SIZE;
    uint32_t bSize = batchSize_;
    uint32_t gSize = nNumOfQInOneGroup_;
    uint32_t n2Size = numKvHeads_;
    s1SplitSize_ = s1gBasicSize / gSize; // QS的切分
    uint32_t souter = s1SplitSize_;

    // 负载均衡场景G轴不切
    groupSplitSize_ = nNumOfQInOneGroup_;
    gOuter_ = (nNumOfQInOneGroup_ + groupSplitSize_ - 1U) / groupSplitSize_;

    const int64_t *actualSeqKv = ifaContext_->actualSeqLengths.tensor->GetData<int64_t>();
    ActualSeqInfo actualSeqInfo(bSize, actualSeqKv[0]);
    GetActualSeqInfo(actualSeqKv, actualSeqInfo);

    OP_LOGI(ifaContext_->opName, "bSize:%u, gSize:%u, n2Size:%u, souter:%u\n", bSize, gSize, n2Size, souter);
    // 分核主流程
    // 计算线段总长度和平均长度
    SeqTilingInfo seqTilingInfo(bSize);
    GetSeqTilingInfo(actualSeqKv, actualSeqInfo, seqTilingInfo);
    std::vector<int64_t> gS1SplitNumOfFdHead(MAX_AIC_CORE_NUM * 2);

    // 启动分核
    BalancedSplitTilingInfo tilingInfo(coreNum_);
    for (uint32_t bIdx = 0U; bIdx < bSize; bIdx++) {
        uint32_t s1 = actualSeqInfo.actualSeqQ[bIdx];
        int64_t s2 = actualLenDims_== 1U? actualSeqKv[0] : actualSeqKv[bIdx]; // 线段长度
        OP_LOGI(ifaContext_->opName, "bIdx:%u, s1:%u, s2:%ld\n", bIdx, s1, s2);
        for (uint32_t nIdx = 0U; nIdx < n2Size; nIdx++) {
            SplitBalancedForEachHeadFd(bIdx, seqTilingInfo, tilingInfo, gS1SplitNumOfFdHead, s1);
        }
    }
    uint32_t *gS1SplitNumOfFdHeadMla = tilingDataMla_.tndSplitCoreParams.get_gS1SplitNumOfFdHeadMla();
    for(uint32_t i = 0U; i < tilingInfo.tndFDCoreArrLen; ++i)
    {
        gS1SplitNumOfFdHeadMla[i] = gS1SplitNumOfFdHead[i];
    }
    if (tilingInfo.needUpdate) {
        // 更新最后一个核的End分核信息
        FillBalancedSplitCoreInfo(TilingIndexes(seqTilingInfo.lastValidBIdx,
                                                seqTilingInfo.s1OuterNum[seqTilingInfo.lastValidBIdx] - 1U,
                                                seqTilingInfo.s2OuterNum[seqTilingInfo.lastValidBIdx] - 1U),
                                  tilingInfo);
    }
    uint32_t lastValidBMoreIdx = seqTilingInfo.lastValidBIdx + 1U;
    if (IsKvZeroBatchSplit(tilingInfo.needUpdate, lastValidBMoreIdx, bSize, seqTilingInfo.s1OuterNum, seqTilingInfo.s2OuterNum)) {
        FillBalancedSplitCoreInfo(TilingIndexes(lastValidBMoreIdx, 0U, 0U), tilingInfo);
    }

    usedCoreNum_ = tilingInfo.currCoreIdx;
    tilingDataMla_.tndSplitCoreParams.set_tndFDCoreArrLen(tilingInfo.tndFDCoreArrLen);
    OP_LOGI(ifaContext_->opName, "usedCoreNum_:%u", usedCoreNum_);
    OP_LOGI(ifaContext_->opName, "tnd FD Core Array Length:%u", tilingInfo.tndFDCoreArrLen);
    OP_LOGI(ifaContext_->opName, "max Kv Split Part:%u", tilingInfo.maxKvSplitPart);
    OP_LOGI(ifaContext_->opName, "avgerage S2 Length:%lu", seqTilingInfo.avgS2Length);
    OP_LOGI(ifaContext_->opName, "sInnerSize_:%u", sInnerSize_);

    if (IsFlashDecode(coreNum_, perfMode_)) {
        splitKVFlag_ = true;
        kvSplit_++;
        kvSplitPart_ = tilingInfo.maxKvSplitPart;
        // 使能fd负载均衡
        uint32_t *balanceFDCoreS1Arr = tilingDataMla_.tndSplitCoreParams.get_balanceFDCoreS1Arr();
        uint32_t *s2SplitNumOfFdHead = tilingDataMla_.tndSplitCoreParams.get_balanceFDCoreKVSplitArr();
        uint32_t aivCoreNumOfFd = (aivNum_ / aicNum_) * usedCoreNum_;        
        SplitFDMLa(tilingInfo.tndFDCoreArrLen, gS1SplitNumOfFdHead, s2SplitNumOfFdHead, aivCoreNumOfFd, seqTilingInfo); //待补充
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::SplitBalanced()
{
    CalcInnerSize(seqSize_);

    uint32_t s1gBasicSize = FIA_BALANCE_SG_BASIC_SIZE;
    uint32_t bSize = batchSize_;
    uint32_t gSize = nNumOfQInOneGroup_;
    uint32_t n2Size = numKvHeads_;
    s1SplitSize_ = s1gBasicSize / gSize; // QS的切分
    uint32_t souter = s1SplitSize_;

    // 负载均衡场景G轴不切
    groupSplitSize_ = nNumOfQInOneGroup_;
    gOuter_ = (nNumOfQInOneGroup_ + groupSplitSize_ - 1U) / groupSplitSize_;

    const int64_t *actualSeqKv = ifaContext_->actualSeqLengths.tensor->GetData<int64_t>();
    ActualSeqInfo actualSeqInfo(bSize, actualSeqKv[0]);
    GetActualSeqInfo(actualSeqKv, actualSeqInfo);

    OP_LOGI(ifaContext_->opName, "bSize:%u, gSize:%u, n2Size:%u, souter:%u\n", bSize, gSize, n2Size, souter);
    // 分核主流程
    // 计算线段总长度和平均长度
    SeqTilingInfo seqTilingInfo(bSize);
    GetSeqTilingInfo(actualSeqKv, actualSeqInfo, seqTilingInfo);

    // 启动分核
    BalancedSplitTilingInfo tilingInfo(coreNum_);
    for (uint32_t bIdx = 0U; bIdx < bSize; bIdx++) {
        uint32_t s1 = actualSeqInfo.actualSeqQ[bIdx];
        int64_t s2 = actualLenDims_== 1U? actualSeqKv[0] : actualSeqKv[bIdx]; // 线段长度
        OP_LOGI(ifaContext_->opName, "bIdx:%u, s1:%u, s2:%ld\n", bIdx, s1, s2);
        for (uint32_t nIdx = 0U; nIdx < n2Size; nIdx++) {
            SplitBalancedForEachHead(bIdx, seqTilingInfo, tilingInfo);
        }
    }
    if (tilingInfo.needUpdate) {
        // 更新最后一个核的End分核信息
        FillBalancedSplitCoreInfo(TilingIndexes(seqTilingInfo.lastValidBIdx,
                                                seqTilingInfo.s1OuterNum[seqTilingInfo.lastValidBIdx] - 1U,
                                                seqTilingInfo.s2OuterNum[seqTilingInfo.lastValidBIdx] - 1U),
                                  tilingInfo);
    }
    uint32_t lastValidBMoreIdx = seqTilingInfo.lastValidBIdx + 1U;
    if (IsKvZeroBatchSplit(tilingInfo.needUpdate, lastValidBMoreIdx, bSize, seqTilingInfo.s1OuterNum, seqTilingInfo.s2OuterNum)) {
        FillBalancedSplitCoreInfo(TilingIndexes(lastValidBMoreIdx, 0U, 0U), tilingInfo);
    }

    usedCoreNum_ = tilingInfo.currCoreIdx;
    tilingDataMla_.tndSplitCoreParams.set_tndFDCoreArrLen(tilingInfo.tndFDCoreArrLen);
    OP_LOGI(ifaContext_->opName, "usedCoreNum_:%u", usedCoreNum_);
    OP_LOGI(ifaContext_->opName, "tnd FD Core Array Length:%u", tilingInfo.tndFDCoreArrLen);
    OP_LOGI(ifaContext_->opName, "max Kv Split Part:%u", tilingInfo.maxKvSplitPart);
    OP_LOGI(ifaContext_->opName, "avgerage S2 Length:%lu", seqTilingInfo.avgS2Length);
    OP_LOGI(ifaContext_->opName, "sInnerSize_:%u", sInnerSize_);

    if (IsFlashDecode(coreNum_, perfMode_)) {
        splitKVFlag_ = true;
        kvSplit_++;
        kvSplitPart_ = tilingInfo.maxKvSplitPart;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::SplitUnbalanced() {
    if (amlaMode_ == IfaAmlaMode::AMLA_3BUF) {
        gMax_ = 512U; // 3buf场景下切512
    }
    if (gqaMtpFlag_) {
        gMax_ = antiQuantFlag_ ? 128U : 256U; // 伪量化场景下，g切32，非伪量化场景切64
    }

    if (ropeFlag_ || gqaMtpFlag_) {
        if (inputLayout_ == IfaLayout::BSH_BSND || (gqaKvNZFlag_ && inputLayout_ == IfaLayout::TND)) {
            s1SplitSize_ = gMax_ / nNumOfQInOneGroup_;
            if (s1SplitSize_ > qSeqSize_) {
                s1SplitSize_ = qSeqSize_;
            }
            s1Outer_ = (qSeqSize_ + s1SplitSize_ - 1U) / s1SplitSize_;
        } else {
            // mla 模式支持G切分
            groupSplitSize_ = gMax_ / qSeqSize_;
            groupSplitSize_ -= (groupSplitSize_ & 1U);
            if (groupSplitSize_ > nNumOfQInOneGroup_) {
                groupSplitSize_ = nNumOfQInOneGroup_;
            }
            gOuter_ = (nNumOfQInOneGroup_ + groupSplitSize_ - 1U) / groupSplitSize_;
        }
    }

    if (IsFlashDecode(coreNum_, perfMode_)) {
        OP_LOGI(ifaContext_->opName, "Enable flashdecode.");
        splitKVFlag_ = true;
        kvSplit_++;
        return SplitBNS();
    }

    CalcInnerSize(seqSize_);
    return SplitBN();
}

ge::graphStatus IFATiling::SplitBN()
{
    uint32_t bn;
    if (inputLayout_ == IfaLayout::BSH_BSND || (gqaKvNZFlag_ && inputLayout_ == IfaLayout::TND)) {
        bn = batchSize_ * numKvHeads_ * s1Outer_;
    } else {
        bn = batchSize_ * numKvHeads_ * gOuter_;
    }

    for (auto& elem : startIdxEachCore_) {
        elem = bn;
    }

    if (isSysPrefixTiling_) {
        return SplitBN_V0();
    }

    if (actualLenDims_ == 1U || bn <= coreNum_ || (actualLenDims_ == 0U && kvListSeqLens_.size() == 1U) ||
        DealSameSeqEachBatch()) {
        return SplitBN_V0();
    }

    std::vector<int64_t> validArray;
    if (actualLenDims_ > 0U) {
        const int64_t *actualLenData = ifaContext_->actualSeqLengths.tensor->GetData<int64_t>();
        validArray = InitSparseValidArray(actualLenData);
    } else {
        validArray = InitSparseValidArray(&kvListSeqLens_[0]);
    }

    SetSparseStartIdx(validArray, bn, coreNum_, startIdxEachCore_, CeilDivision(bn, coreNum_));

    usedCoreNum_ = coreNum_;
    return ge::GRAPH_SUCCESS;
}

void IFATiling::GetEstimatedLoad(int64_t &estimatedLoad) const
{
    if (antiQuantFlag_ && estimatedLoad < MSD_VEC_LOAD) {
        estimatedLoad = MSD_VEC_LOAD;
    } else if (estimatedLoad == 0) {
        // 空tensor输出也计入负载，否则当最后一个batch为空tensor时，分核算法会将该batch优化掉
        estimatedLoad = 1;
    }
}

std::vector<int64_t> IFATiling::InitSparseValidArray(const int64_t *actualLens) const
{
    uint32_t outer;
    if (inputLayout_ == IfaLayout::BSH_BSND || (gqaKvNZFlag_ && inputLayout_ == IfaLayout::TND)) {
        outer = s1Outer_;
    } else {
        outer = gOuter_;
    }
    std::vector<int64_t> res((batchSize_ * numKvHeads_ * outer));
    for (uint32_t b = 0U; b < batchSize_; b++) {
        for (uint32_t n = 0U; n < numKvHeads_ * outer; n++) {
            int64_t estimatedLoad = static_cast<int64_t>(seqSize_);
            if (actualLens != nullptr) {
                estimatedLoad = actualLens[b];
                GetEstimatedLoad(estimatedLoad);
            }
            res[b * numKvHeads_ * outer + n] = estimatedLoad;
        }
    }
    return res;
}

bool IFATiling::BalanceLoad(const std::vector<int64_t> &sparseValidArray, int64_t totalSize, int64_t validAivNum,
                            std::vector<int64_t> &localValue, std::vector<int64_t> &sparseStartIdx) const
{
    // to avoid buffer overflow, or maybe sometimes we want to only verify single core
    int64_t maxVal = *std::max_element(localValue.begin(), localValue.end());
    int64_t tmpMaxVal = maxVal;

    // 从前往后遍历
    for (int64_t idx = 1; idx < validAivNum; ++idx) {
        int64_t start = sparseStartIdx[idx];
        if (start < totalSize && start > 0 && ((localValue[idx - 1] + sparseValidArray[start]) < maxVal)) {
            localValue[idx - 1] += sparseValidArray[start];
            localValue[idx] -= sparseValidArray[start];
            sparseStartIdx[idx] += 1;
        } else if (start == totalSize) {
            break;
        }
    }
    tmpMaxVal = *std::max_element(localValue.begin(), localValue.end());

    // 从后往前遍历
    for (int64_t idx = validAivNum - 1; idx > 0; --idx) {
        int64_t start = sparseStartIdx[idx];
        if (start == totalSize) {
            if (sparseStartIdx[idx - 1] == totalSize) {
                continue;
            }
            localValue[idx - 1] -= sparseValidArray[start - 1];
            localValue[idx] = sparseValidArray[start - 1];
            sparseStartIdx[idx] -= 1;
        } else if (start > 0) {
            if ((localValue[idx] + sparseValidArray[start - 1]) >= tmpMaxVal) {
                continue;
            }
            localValue[idx - 1] -= sparseValidArray[start - 1];
            localValue[idx] += sparseValidArray[start - 1];
            sparseStartIdx[idx] -= 1;
        } else {
            break;
        }
    }
    tmpMaxVal = *std::max_element(localValue.begin(), localValue.end());

    return (tmpMaxVal >= maxVal) ? false : true;
}

void IFATiling::InitLoadValue(const std::vector<int64_t> &sparseValidArray, int64_t totalSize, int64_t validAivNum,
                              const std::vector<int64_t> &sparseStartIdx, std::vector<int64_t> &localValue) const
{
    for (int64_t idx = 0; idx < validAivNum; ++idx) {
        int64_t start = sparseStartIdx[idx];
        int64_t end = ((idx + 1) < validAivNum) ? sparseStartIdx[idx + 1] : totalSize;
        if (start < totalSize) {
            localValue[idx] = std::accumulate(sparseValidArray.begin() + start, sparseValidArray.begin() + end, 0);
        } else {
            break;
        }
    }
}

void IFATiling::SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray, int64_t totalSize, int64_t validAivNum,
                                  uint32_t *sparseStartIdx, int64_t splitFactorSize) const
{
    // initLoad: 使用均分策略, 保证后续不会比均分差
    std::vector<int64_t> localSparseStartIdx(MAX_CORE_NUM, totalSize);
    for (uint32_t idx = 0; idx < MAX_CORE_NUM; ++idx) {
        localSparseStartIdx[idx] = std::min((idx * splitFactorSize), totalSize);
    }
    std::vector<int64_t> localValue(validAivNum, 0);
    InitLoadValue(sparseValidArray, totalSize, validAivNum, localSparseStartIdx, localValue);

    // 负载均衡粗调
    std::vector<int64_t> tmpLocalValue(validAivNum, 0);
    std::vector<int64_t> tmpsparseStartIdx(MAX_CORE_NUM, totalSize);
    int64_t sparseArraySum = std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0);
    int64_t avgVal = CeilDivision(sparseArraySum, validAivNum);

    tmpsparseStartIdx[0] = 0;
    for (uint32_t idx = 1; idx < MAX_CORE_NUM; ++idx) {
        int64_t start = tmpsparseStartIdx[idx - 1];
        int64_t singleLoadValue = 0;
        tmpsparseStartIdx[idx] = start;
        while (singleLoadValue < avgVal && tmpsparseStartIdx[idx] < totalSize) {
            singleLoadValue += sparseValidArray[tmpsparseStartIdx[idx]];
            tmpsparseStartIdx[idx] += 1;
        }

        if ((start + 1) < tmpsparseStartIdx[idx]) {
            int64_t redoSingleLoadValue = singleLoadValue - sparseValidArray[tmpsparseStartIdx[idx] - 1];
            if ((singleLoadValue - avgVal) > (avgVal - redoSingleLoadValue)) {
                tmpsparseStartIdx[idx] = tmpsparseStartIdx[idx] - 1;
                singleLoadValue = redoSingleLoadValue;
            }
            sparseArraySum -= singleLoadValue;
            avgVal = CeilDivision(sparseArraySum, (validAivNum - static_cast<int64_t>(idx)));
        }
    }

    InitLoadValue(sparseValidArray, totalSize, validAivNum, tmpsparseStartIdx, tmpLocalValue);

    // 负载均衡精调
    while (BalanceLoad(sparseValidArray, totalSize, validAivNum, tmpLocalValue, tmpsparseStartIdx)) {
        // 根据负载均衡是否能得到更好预测结果决定是否结束循环
    }

    // exchange initLoad and 负载均衡
    if ((*std::max_element(localValue.begin(), localValue.end())) >
        (*std::max_element(tmpLocalValue.begin(), tmpLocalValue.end()))) {
        localSparseStartIdx.swap(tmpsparseStartIdx);
        localValue.swap(tmpLocalValue);
    }
    for (uint32_t idx = 0; idx < MAX_CORE_NUM; ++idx) {
        sparseStartIdx[idx] = localSparseStartIdx[idx];
    }
}

ge::graphStatus IFATiling::SplitBN_V0()
{
    uint32_t bn;
    if (inputLayout_ == IfaLayout::BSH_BSND || (gqaKvNZFlag_ && inputLayout_ == IfaLayout::TND)) {
        bn = batchSize_ * numKvHeads_ * s1Outer_;
    } else {
        bn = batchSize_ * numKvHeads_ * gOuter_;
    }
    // MLA场景，在单核的最大负载不变的情况下，尽可能少的启动核
    if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM_MLA) {
        coreNum_ = static_cast<uint32_t>(CeilDivision(static_cast<int64_t>(bn),
            CeilDivision(static_cast<int64_t>(bn), static_cast<int64_t>(aicNum_))));
    }
    formerCoreNum_ = bn % coreNum_;
    if (formerCoreNum_ == 0U) {
        blockSplitBn2Range_ = bn / coreNum_;
        tailSplitedBatchRange_ = blockSplitBn2Range_;
    } else {
        blockSplitBn2Range_ = bn / coreNum_ + 1U;
        tailSplitedBatchRange_ = blockSplitBn2Range_ - 1U;
    }

    usedCoreNum_ = bn > coreNum_ ? coreNum_ : bn;

    for (uint32_t i = 0U; i < formerCoreNum_; i++) {
        startIdxEachCore_[i] = blockSplitBn2Range_ * i;
    }

    uint32_t formerBase = formerCoreNum_ * blockSplitBn2Range_;
    for (uint32_t i = formerCoreNum_; i < usedCoreNum_; i++) {
        startIdxEachCore_[i] = formerBase + tailSplitedBatchRange_ * (i - formerCoreNum_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::SplitBNS()
{
    formerCoreNum_ = 0U;
    blockSplitBn2Range_ = 1U;
    tailSplitedBatchRange_ = 1U;

    uint32_t bn = batchSize_ * numKvHeads_;
    kvSplitPart_ = coreNum_ / bn;
    while (((maxActualseq_ / kvSplitPart_) < 512U) && (kvSplitPart_ > 1U)) { // 512, 经验值
        kvSplitPart_--;
    }

    usedCoreNum_ = bn * kvSplitPart_;
    uint32_t computeSeqSize = (seqSize_ + (kvSplitPart_ - 1U)) / kvSplitPart_;
    if (pageAttentionFlag_) {
        computeSeqSize = Align(computeSeqSize, blockSize_);
    }
    if (inputKvType_ == ge::DT_INT4) {
        computeSeqSize = Align(computeSeqSize, 2U);
    }
    CalcInnerSize(computeSeqSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CalcInnerSize(uint32_t seqSize)
{
    /**
     * sInnerSize：s2的切分大小，直接决定了MM的singleN/K和vector的切块大小，但当前切分也并非适用所有case。
     * 1、非GQA场景：按照vector的最大基本块8*1024进行切分，sInnerSize=8192
     * 2、GQA场景：(1) 非伪量化：vector计算比较少，cube MTE2bound，
     *                          因此，cube发射大块，减少通信次数。sInnerSize=8192
     *            (2) 伪量化：vector比较重，尽量较少vector的循环次数,
     *                          因此，cube发小块，期望vector尽量被cube的mte2掩盖。sInnerSize=1024
     */
    sInnerSize_ = MAX_SPLIT_SIZE; // 8192
    if (antiQuantFlag_) {
        if (nNumOfQInOneGroup_ > 1U) {
            if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM || perfMode_ == IfaPerfMode::CUBE_VIEW_MM_FULL_LOAD) {	
                sInnerSize_ = 2048U;	
            } else {	
                sInnerSize_ = 1024U;	
            }
        } else if (nNumOfQInOneGroup_ == 1) {
            if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM_DD) {
                sInnerSize_ = 1024U;
            }
        }
    } else {
        /** 当前版本限制workspace大小不超过32MB，否则会影响网络中前后算子性能，
         *  GQA场景下 nNumOfQInOneGroup_和sInnerSize_切分大小直接影响workspace大小,
         *  具体计算参考CalcWorkSpace函数，这里根据nNumOfQInOneGroup_将sInnerSize_
         *  分为8192，4096，2048三档，nNumOfQInOneGroup_增大时减小sInnerSize_，
         *  保证最终workspace大小不超过32MB。
         */
        uint32_t sInnerSize[3U] = {8192U, 4096U, 2048U};
        uint32_t idx = std::min(nNumOfQInOneGroup_ / 5U, 2U);
        sInnerSize_ = sInnerSize[idx];
    }
    // for ifa_cube_310P
    if ((socVersion_ == IfaSocVersion::SOC_ASCEND_310P) && (nNumOfQInOneGroup_ * qSeqSize_ > 1U)) {
        auto bmm1BufferSize =  40U * 1024U;
        sInnerSize_ = std::min(bmm1BufferSize / nNumOfQInOneGroup_ / qSeqSize_ / static_cast<uint32_t>(sizeof(float)), MAX_SPLIT_SIZE);
        sInnerSize_ = (sInnerSize_ / 128U) *  128U;
    }

    if (ropeFlag_) {
        sInnerSize_ = 512U;
        // FlashDecode时，如果S2的计算量>=256(确保切分后不小于128)但又不足以分2次计算时，则修改sInnerSize_，均分为2份进行计算，确保Nbuffer=2
        if (!isWorkspace_ && splitKVFlag_ && inputLayout_ != IfaLayout::TND) {
            if (seqSize == 256U) {
                sInnerSize_ = 128U;
            } else if (seqSize > 256U && seqSize <= sInnerSize_) {
                sInnerSize_ = (sInnerSize_ + 1U) / 2U;
            }
        }
        if (antiQuantFlag_) {
            sInnerSize_ = 1024U; // 1024: 伪量化场景下，sInnerSize_设置为1024
        }
    }

    // PA特性泛化场景，blockSize_可能为112等值，无法被sInnerSize_整除，当step*base跨block时，搬运处理复杂，通过向下对齐避免
    if (pageAttentionFlag_ && blockSize_ != 0U) {
        if (sInnerSize_ % blockSize_ != 0U) {
            sInnerSize_ = (sInnerSize_ / blockSize_) * blockSize_;
        }
    }

    sInnerLoopTimes_ = (seqSize + sInnerSize_ - 1U) / sInnerSize_;
    sInnerSizeTail_ = seqSize - (sInnerLoopTimes_ - 1U) * sInnerSize_;
    // tiling下沉 && flash decoder场景时，sInnerSize_基块大小不按照真实值修改
    // 否则会导致 tiling下沉 && flash decoder 场景时开辟workspace空间大小小于真实运行时所需的workspace大小
    if (sInnerSize_ > seqSize && (!(isWorkspace_ && splitKVFlag_))) {
        sInnerSize_ = seqSize;
    }
    if (inputKvType_ == ge::DT_INT4) {
        sInnerSize_ = Align(sInnerSize_, 2U);
        sInnerSizeTail_ = seqSize - (sInnerLoopTimes_ - 1U) * sInnerSize_;
        sInnerSizeAlign_ = Align(sInnerSize_, 64U); // 元素个数按照基本块大小对齐
    } else {
        sInnerSizeAlign_ = Align(sInnerSize_, BYTE_BLOCK); // 元素个数按照基本块大小对齐
    }

    CheckUbSpace();
    return ge::GRAPH_SUCCESS;
}

void IFATiling::SetDealBN2Num()
{
    uint32_t elemNum = Align(maxActualseq_, 32U) * ((nNumOfQInOneGroup_ + 1U) / 2U);
    dealBN2Num_ = (8192U + elemNum - 1U) / elemNum;
    if (dealBN2Num_ < DEAL_BN2_NUM) {
        dealBN2Num_ = DEAL_BN2_NUM;
    }
}

bool IFATiling::CalcUbBmm()
{
    uint32_t cubeMSize = nNumOfQInOneGroup_ * qSeqSize_;
    if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM_MLA) {
        uint32_t maxMSize = (inputLayout_ == IfaLayout::TND) ? FIA_BALANCE_SG_BASIC_SIZE : gMax_;
        if (cubeMSize > maxMSize) {
            cubeMSize = maxMSize;
        }
        if (amlaMode_ != IfaAmlaMode::DISABLE_AMLA) {
            cubeMSize = Align(cubeMSize, 16U);
        }
    }
    if (ropeFlag_) {
        mmResUbSize_ = static_cast<size_t>(sInnerSizeAlign_) * static_cast<size_t>(cubeMSize);
        bmm2ResUbSize_ = static_cast<size_t>(headDimAlign_) * static_cast<size_t>(cubeMSize);
    } else {
        mmResUbSize_ = static_cast<size_t>(msdIterNum_) * static_cast<size_t>(sInnerSizeAlign_) * static_cast<size_t>(cubeMSize);
        if (slidingFlag_) {
            bmm2ResUbSize_ = static_cast<size_t>(msdIterNum_) * static_cast<size_t>(headDimVAlign_) * static_cast<size_t>(cubeMSize);
        } else {
            bmm2ResUbSize_ = static_cast<size_t>(msdIterNum_) * static_cast<size_t>(headDimAlign_) * static_cast<size_t>(cubeMSize);
        }
    }

    qPreSizeMla_ = static_cast<size_t>(msdIterNum_) * static_cast<size_t>(nNumOfQInOneGroup_ * (headDimAlign_ + 64U)) * static_cast<size_t>(qSeqSize_);
    if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM_FULL_LOAD) {
        SetDealBN2Num();
        mmResUbSize_ *= dealBN2Num_;
        bmm2ResUbSize_ *= dealBN2Num_;
    }

    if (isSysPrefixTiling_) {
        mmResUbSize_ *= batchSizeQ_;
        bmm2ResUbSize_ *= batchSizeQ_;
    }
    return true;
}

bool IFATiling::CalcUbSoftMax()
{
    auto tmpShape = Shape({nNumOfQInOneGroup_, Align(sInnerSize_, BYTE_BLOCK / blockTypeSize_)});
    softmaxFlashTmpSize_ = GetSoftMaxFlashV2MinTmpSize(tmpShape, blockTypeSize_, blockTypeSize_, true, false);
    return true;
}

bool IFATiling::CalcUbAttenMask()
{
    if (!attenMaskFlag_) {
        selectWithByteMaskTmpMinSize_ = 0U;
        return true;
    }
    // bool/int8/uint8类型的mask，每个占1字节
    attenMaskTypeSize_ = 1U; // 1:sizeof(bool)
    if (ropeFlag_) {
        auto selectWithByteMaskTmpShape =
            Shape({nNumOfQInOneGroup_, Align(sInnerSize_, BYTE_BLOCK / attenMaskTypeSize_)});
        selectWithByteMaskTmpMinSize_ = GetSelectWithBytesMaskMinTmpSize(
            selectWithByteMaskTmpShape, Shape({1, 1}), FP32_BYTES, selectWithByteMaskTmpShape, FP32_BYTES, false);
    } else {
        auto selectWithByteMaskTmpShape =
            Shape({msdIterNum_ * nNumOfQInOneGroup_, Align(sInnerSize_, BYTE_BLOCK / attenMaskTypeSize_)});
        selectWithByteMaskTmpMinSize_ = GetSelectWithBytesMaskMinTmpSize(
            selectWithByteMaskTmpShape, Shape({1, 1}), FP32_BYTES, selectWithByteMaskTmpShape, FP32_BYTES, false);
    }

    return true;
}

std::pair<uint32_t, uint32_t> IFATiling::GetPreLoadNumAndActCoreNum() const
{
    uint32_t preLoadNum = 1U;
    uint32_t actCoreNum = coreNum_;
    if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM || perfMode_ == IfaPerfMode::CUBE_VIEW_MM_FULL_LOAD) {
        preLoadNum = PRE_LOAD_NUM;
    } else if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM_DD) {
        preLoadNum = BUF_NUM;
    } else if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM_MLA) {
        preLoadNum = PRE_LOAD_NUM_MLA;
        if (isWorkspace_) {
            actCoreNum = aicNum_;
        }
    }
    return std::make_pair(preLoadNum, actCoreNum);
}

void IFATiling::CalcWorkSpaceForBmmAll(const IfaWorkSpaceSizeParams& params, uint32_t preLoadNum, uint32_t actCoreNum)
{
    workspaceSize_ += preLoadNum * (mmResUbSize_ * actCoreNum * params.mmResElemSize);
    if (ropeFlag_) {
        workspaceSize_ += preLoadNum * static_cast<size_t>(
            static_cast<float>(mmResUbSize_ * msdIterNum_ * actCoreNum * params.vec1ResElemSize) * params.kvDtypeRatio);
    } else {
        workspaceSize_ += preLoadNum * static_cast<size_t>(
            static_cast<float>(mmResUbSize_ * actCoreNum * params.vec1ResElemSize) * params.kvDtypeRatio);
    }
    workspaceSize_ += preLoadNum * bmm2ResUbSize_ * actCoreNum * params.bmm2ResElemSize;
    workspaceSize_ += preLoadNum * bmm2ResUbSize_ * actCoreNum * params.vec2ResElemSize;
    if (ropeFlag_) {
        workspaceSize_ += preLoadNum * static_cast<size_t>(
            static_cast<float>(qPreSizeMla_ * actCoreNum * params.qPreProcResElemSize) * params.kvDtypeRatio);
    } else {
        workspaceSize_ += preLoadNum * static_cast<size_t>(
            static_cast<float>(bmm2ResUbSize_ * actCoreNum * params.qPreProcResElemSize) * params.kvDtypeRatio);
    }
    workspaceSize_ += static_cast<size_t>(preLoadNum) * gMax_ * actCoreNum * params.nUpdateElemSize; // aMla nUpdate, Gmax=128
    workspaceSize_ += static_cast<size_t>(preLoadNum) * gMax_ * actCoreNum * params.softmaxSumElemSize; // aMla softmaxSum, Gmax=128
    if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM_FULL_LOAD) {
        uint32_t tmpWsNum = 1U; // 1块workspace: aMaxBmm1Gm
        if (antiquantMode_ == PER_TOKEN_MODE) {
            tmpWsNum = 4U; // 4块workspace: aMaxBmm1Gm/qRowSumGm/softmaxResAmaxGm/softmaxResRowSumGm
        }
        workspaceSize_ += preLoadNum *
            static_cast<size_t>(static_cast<float>(actCoreNum * dealBN2Num_ * nNumOfQInOneGroup_ * 32U)) * tmpWsNum;
        workspaceSize_ += preLoadNum *
            static_cast<size_t>(static_cast<float>(actCoreNum * dealBN2Num_ * nNumOfQInOneGroup_ * 32U)); // SoftmaxSumGm
    }
}

ge::graphStatus IFATiling::CalcWorkSpace()
{
    IfaWorkSpaceSizeParams params{};
    if (antiQuantFlag_) {
        if (ropeFlag_) {
            params.mmResElemSize = 2U;       // 2: S已经被转换成FP16存在workspace上
            params.bmm2ResElemSize = 2U;     // 2: O已经被转换成FP16在workspace上
            params.vec2ResElemSize = 2U;
        } else {
            params.mmResElemSize = 4U;       // 4:int32
            params.bmm2ResElemSize = 4U;     // 4:int32
            params.vec2ResElemSize = 4U;     // 4:float
        }
        params.vec1ResElemSize = 1U;     // 1:int8
        params.qPreProcResElemSize = 1U; // int8
        params.kvDtypeRatio = inputKvType_ == ge::DT_INT4 ? 0.5 : 1.0; // 0.5:int4 1.0:elseType
    }

    workspaceSize_ = libapiSize_;
    std::pair<uint32_t, uint32_t> preLoadInfo = GetPreLoadNumAndActCoreNum();
    uint32_t preLoadNum = preLoadInfo.first;
    uint32_t actCoreNum = preLoadInfo.second;
    if (perfMode_ != IfaPerfMode::BMM_ALL_BY_VEC) {
        CalcWorkSpaceForBmmAll(params, preLoadNum, actCoreNum);
    }
    CalcFDWorkSpace(actCoreNum);

    uint32_t mmPACallBackDataSize = 64; // 64: matmul回调信息需要7个uint32值，dcci cacheline需要64B对齐
    if (pageAttentionFlag_) {
        workspaceSize_ += static_cast<size_t>(actCoreNum) * static_cast<size_t>(mmPACallBackDataSize) * static_cast<size_t>(2); // bmm1 bmm2 2份
    }

    if (isSysPrefixTiling_) {
        if (antiQuantFlag_) {
            size_t blockSize = static_cast<size_t>(nNumOfQInOneGroup_) * static_cast<size_t>(BYTE_BLOCK) * static_cast<size_t>(batchSizeQ_);
            workspaceSize_ += actCoreNum * blockSize * 4U; // 4, rowMax1 rowMax2 rowSum1 rowSum2
        }

        size_t blockSize = static_cast<size_t>(nNumOfQInOneGroup_) * static_cast<size_t>(BYTE_BLOCK) * static_cast<size_t>(batchSizeQ_);
        workspaceSize_ += actCoreNum * blockSize * 3U; // 3, sum log exp

        if (!antiQuantFlag_) {
            workspaceSize_ += static_cast<size_t>(batchSizeQ_) * static_cast<size_t>(nNumOfQInOneGroup_) 
            * static_cast<size_t>(headDimAlign_) * static_cast<size_t>(2) * static_cast<size_t>(actCoreNum); // 2:fp16/bf16
        }
    }

    if (ifaContext_->workSpaces) {
        ifaContext_->workSpaces[0] = workspaceSize_;
    }
    return ge::GRAPH_SUCCESS;
}

void IFATiling::CalcFDWorkSpace(const uint32_t actCoreNum) {
    if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM_MLA) {
        MLACalcFDWorkSpace(actCoreNum);
    } else {
        NormalCalcFDWorkSpace(actCoreNum);
    }
}

void IFATiling::NormalCalcFDWorkSpace(const uint32_t actCoreNum) {
    if (splitKVFlag_) {
        uint32_t accumOutSize = 0U;
        uint32_t logSumExpSize = 0U;
        uint32_t FDParamNums = balanceModeFlag_ ? CalcBalanceFDParamNums(actCoreNum) : CalcUnbalanceFDParamNums();
        if (slidingFlag_) {
           accumOutSize = FDParamNums * headDimVAlign_;
        } else {
           accumOutSize = FDParamNums * headDimAlign_;
        }
        logSumExpSize = NUM2 * FDParamNums * (BYTE_BLOCK / blockTypeSize_);  // log和sum的存储空间一致，共需要2份内存
        workspaceSize_ += static_cast<size_t>((accumOutSize + logSumExpSize)) * static_cast<size_t>(blockTypeSize_);
        if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
            workspaceSize_ += static_cast<size_t>(actCoreNum) * 32U; // 每个核SyncAll软同步需要32Byte记录状态
        }
    }
}

void IFATiling::MLACalcFDWorkSpace(const uint32_t actCoreNum) {
    uint32_t accumOutSize = 0U;
    uint32_t logSumExpSize = 0U;

    // BALANCE
    uint32_t FDParamNumsBalance = CalcBalanceFDParamNums(actCoreNum);
    uint32_t accumOutSizeBalance = FDParamNumsBalance * headDimAlign_;
    uint32_t logSumExpSizeBalance = 2U * FDParamNumsBalance * (BYTE_BLOCK / blockTypeSize_);

    // UNBALANCE
    uint32_t FDParamNumsUnbalance = CalcUnbalanceFDParamNums();
    uint32_t accumOutSizeUnbalance = FDParamNumsUnbalance * headDimAlign_;
    uint32_t logSumExpSizeUnbalance = 2U * FDParamNumsUnbalance * (BYTE_BLOCK / blockTypeSize_);

    if (isWorkspace_) {  // tiling下沉场景，无法通过actual_seq_kv判断走哪种分核，针对不同layout采取不同workspace分配策略
        if (inputLayout_ == IfaLayout::TND) {
            accumOutSize = accumOutSizeBalance;
            logSumExpSize = logSumExpSizeBalance;
        } else if (inputLayout_ == IfaLayout::BSH_BSND) {
            accumOutSize = std::max(accumOutSizeBalance, accumOutSizeUnbalance);
            logSumExpSize = std::max(logSumExpSizeBalance, logSumExpSizeUnbalance);
        } else {
            accumOutSize = accumOutSizeUnbalance;
            logSumExpSize = logSumExpSizeUnbalance;
        }
        workspaceSize_ += static_cast<size_t>((accumOutSize + logSumExpSize)) * static_cast<size_t>(blockTypeSize_);
    } else {
        NormalCalcFDWorkSpace(actCoreNum);
    }
}

uint32_t IFATiling::CalcBalanceFDParamNums(const uint32_t actCoreNum) const
{ 
    return actCoreNum * 2U * numKvHeads_ * FIA_BALANCE_SG_BASIC_SIZE;  // 每个核可能有头规约和尾规约，一共两份规约信息
}

uint32_t IFATiling::CalcUnbalanceFDParamNums() const
{
    return batchSizeQ_ * numHeads_ * kvSplitPart_;
}

ge::graphStatus IFATiling::FillTiling()
{
    if (ropeFlag_) {
        return FillTilingMla();
    }
    FillTilingBaseParams();
    FillTilingSplitKV();
    FillTilingCoreParams();
    FillTilingSingleCoreParams();
    FillTilingSingleCoreTensorSize();
    FillTilingSoftmax();
    FillTilingOutputParams();
    return FillTilingBmm() ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

void IFATiling::FillTilingBaseParams()
{
    tilingData_->baseParams.set_batchSize(batchSize_);
    tilingData_->baseParams.set_seqSize(sMax_);
    tilingData_->baseParams.set_qSeqSize(qSeqSize_);
    tilingData_->baseParams.set_headSize(headDim_);
    tilingData_->baseParams.set_headSizeV(headDimV_);
    tilingData_->baseParams.set_blockSize(blockSize_);
    tilingData_->baseParams.set_maxBlockNumPerBatch(maxBlockNumPerBatch_);
    tilingData_->baseParams.set_scaleValue(scaleValue_);
    tilingData_->baseParams.set_kvHeadNum(numKvHeads_);
    tilingData_->baseParams.set_qHeadNum(numHeads_);
    tilingData_->baseParams.set_nNumOfQInOneGroup(numHeads_ / numKvHeads_);
    tilingData_->baseParams.set_batchContinuousFlag(batchContinuousFlag_);
    tilingData_->baseParams.set_pseShiftFlag((pseShiftFlag_) ? 1 : 0);
    tilingData_->baseParams.set_pseShiftB(pseShiftBatch_);
    tilingData_->baseParams.set_pseShiftS(pseShiftS1_);
    tilingData_->baseParams.set_selectWithByteMaskTmpMinSize(selectWithByteMaskTmpMinSize_); // mask

    tilingData_->baseParams.set_actualLenQDims(actualLenQDims_);
    tilingData_->baseParams.set_actualLenDims(isSysPrefixTiling_ ? actualLenDimsPrefix_ : actualLenDims_);
    tilingData_->baseParams.set_msdIterNum(msdIterNum_);
    tilingData_->baseParams.set_kvPaddingFlag(kvPaddingSizeFlag_ ? 1 : 0);
    tilingData_->baseParams.set_antiquantPerTensorFlag(antiquantPerTensorFlag_);
    tilingData_->baseParams.set_antiquantPerHeadFlag(antiquantPerHeadFlag_);
    tilingData_->baseParams.set_antiquantParamsInPagedAttentionFlag(antiquantParamsInPagedAttentionFlag_);
    tilingData_->baseParams.set_attenMaskFlag(attenMaskFlag_ ? 1 : 0);
    tilingData_->baseParams.set_attenMaskSize(attenMaskSize_);
    tilingData_->baseParams.set_l2CacheOffFlag(l2CacheOffFlag_);
    tilingData_->baseParams.set_softmaxLseFlag(softmaxLseFlag_); // whether return lse
    tilingData_->baseParams.set_totalBlockNum(totalBlockNum_);
    tilingData_->baseParams.set_antiqSeqSize(antiqSeqSize_);
    tilingData_->baseParams.set_sparseMode(sparseMode_);
    tilingData_->baseParams.set_slidingFlag(slidingFlag_ ? 1 : 0);
    tilingData_->baseParams.set_windowSize(windowSize_);
}

// for flash decode
void IFATiling::FillTilingSplitKV() const
{
    tilingData_->splitKVParams.set_s2(kvSplitPart_);
    uint32_t sInnerLoopSize_ = (maxActualseq_ + (kvSplitPart_ - 1U)) / kvSplitPart_;
    if (pageAttentionFlag_) {
        sInnerLoopSize_ = Align(sInnerLoopSize_, blockSize_);
        OP_LOGD(ifaContext_->opName, "PA FlashDecode is enabled, sInnerLoopSize is %u, blockSize is %u",
                  sInnerLoopSize_, blockSize_);
    }
    if (inputKvType_ == ge::DT_INT4) {
        sInnerLoopSize_ = Align(sInnerLoopSize_, 2U);
    }
    tilingData_->splitKVParams.set_sInnerLoopSize(sInnerLoopSize_);
    if (inputLayout_ == IfaLayout::TND) {
        tilingData_->splitKVParams.set_accumOutSize(tSeqSize_ * numHeads_ * kvSplitPart_ * headDimAlign_);
        tilingData_->splitKVParams.set_logSumExpSize(NUM2 * batchSizeQ_ * numHeads_ * kvSplitPart_ * qSeqSize_ *   // 2份
                                                    (BYTE_BLOCK / blockTypeSize_)); // 2: sum + max
    } else {
        if (slidingFlag_) {
           tilingData_->splitKVParams.set_accumOutSize(batchSizeQ_ * numHeads_ * kvSplitPart_ * headDimVAlign_);
        } else {
           tilingData_->splitKVParams.set_accumOutSize(batchSizeQ_ * numHeads_ * kvSplitPart_ * headDimAlign_);
        }
        tilingData_->splitKVParams.set_logSumExpSize(NUM2 * batchSizeQ_ * numHeads_ * kvSplitPart_ *
                                                    (BYTE_BLOCK / blockTypeSize_)); // 2: sum + max
    }
    if (!splitKVFlag_) {
        tilingData_->splitKVParams.set_s2(0);
    }
}

void IFATiling::FillTilingCoreParams() const
{
    uint32_t *coreStartIdx = tilingData_->increFlashAttentionCoreParams.get_coreSidxEnd();
    memcpy_s(coreStartIdx, MAX_CORE_NUM * sizeof(uint32_t), startIdxEachCore_, MAX_CORE_NUM * sizeof(uint32_t));
}

void IFATiling::FillTilingSingleCoreParams() const
{
    tilingData_->increFlashAttentionSingleCoreParams.set_sInnerLoopTimes(sInnerLoopTimes_);
    tilingData_->increFlashAttentionSingleCoreParams.set_singleProcessSInnerSize(sInnerSize_);
    tilingData_->increFlashAttentionSingleCoreParams.set_singleProcessSInnerSizeTail(sInnerSizeTail_);
    tilingData_->increFlashAttentionSingleCoreParams.set_usedCoreNum(usedCoreNum_);
    tilingData_->increFlashAttentionSingleCoreParams.set_formerCoreNum(formerCoreNum_);
    tilingData_->increFlashAttentionSingleCoreParams.set_blockSplitBn2Range(blockSplitBn2Range_);
    tilingData_->increFlashAttentionSingleCoreParams.set_tailSplitedBatchRange(tailSplitedBatchRange_);
    tilingData_->increFlashAttentionSingleCoreParams.set_groupSplitSize(groupSplitSize_);
    tilingData_->increFlashAttentionSingleCoreParams.set_s1SplitSize(s1SplitSize_);
}

void IFATiling::FillTilingSingleCoreTensorSize() const
{
    tilingData_->increFlashAttentionSingleCoreTensorSize.set_mmResUbSize(mmResUbSize_);
    tilingData_->increFlashAttentionSingleCoreTensorSize.set_bmm2ResUbSize(bmm2ResUbSize_);
}

void IFATiling::FillTilingSoftmax() const
{
    auto softmaxShape = Shape({1, Align(sInnerSize_, BYTE_BLOCK / blockTypeSize_)});
    SoftMaxFlashV2TilingFunc(softmaxShape, blockTypeSize_, blockTypeSize_, softmaxFlashTmpSize_,
                             tilingData_->softmaxFlashTilingData, true, false);
}

// for zero output
void IFATiling::FillTilingOutputParams() const
{
    tilingData_->outputParams.set_isOutQuantTypeBf16(isOutQuantTypeBf16_);
    tilingData_->outputParams.set_isPerChnOut(isOutQuantPerChnOut_);
}

void IFATiling::AdjustPABmm1Tiling(uint32_t &bmm1BaseN) const
{
    if (bmm1BaseN < blockSize_) {
        while (blockSize_ % bmm1BaseN != 0U) {
            bmm1BaseN /=
                2U; // 2:不断减半，确保1个base块不会跨block拷贝。已校验过blockSize 16/32对齐，因此bmm1BaseN最小值为16/32
        }
    } else if (bmm1BaseN > blockSize_) {
        // nd2nz拷贝时ndnum>1场景性能较差，通过设置baseN <= blocksize避免
        uint32_t tmpBaseN = increGcd(bmm1BaseN, blockSize_);
        bmm1BaseN = tmpBaseN;
    }
    OP_LOGD(ifaContext_->opName, "PA is enabled, blockSize is %u, bmm1 baseN is adjusted to %u", blockSize_, bmm1BaseN);
}

void IFATiling::AdjustPABmm2Tiling() const
{
    uint32_t targetBaseK = 128U;
    if (targetBaseK < blockSize_) {
        while ((blockSize_ % targetBaseK != 0U) ||
               (targetBaseK * tilingData_->bmm2TilingData.get_baseN() * sizeof(float) > L0B_SIZE)) {
            targetBaseK /=
                2U; // 2:不断减半，确保1个base块不会跨block拷贝，已校验过blockSize_16/32对齐，因此targetBaseK最小值为16/32
        }
    } else {
        uint32_t tmpBaseK = increGcd(targetBaseK, blockSize_);
        while (tmpBaseK * tilingData_->bmm2TilingData.get_baseN() * sizeof(float) > L0B_SIZE) {
            tmpBaseK /= 2U; // 2: 不断减半，确保base块大小在LOB有效范围内
        }
        targetBaseK = tmpBaseK;
    }
    // mm api不支持通过 SetFixSplit 设置baseK，需要直接配置tiling结构体
    tilingData_->bmm2TilingData.set_baseK(targetBaseK);
    OP_LOGD(ifaContext_->opName, "PA is enabled, blockSize is %u, bmm2 baseK is adjusted to %u", blockSize_,
              targetBaseK);
}

bool IFATiling::GetBmm1Tiling(const matmul_tiling::DataType &kvType, const uint32_t M) const
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(ifaContext_->platformInfo);
    matmul_tiling::MatmulApiTiling bmm1(ascendcPlatform);
    uint32_t baseN;
    uint32_t bmm1OrgKa;
    bmm1.SetShape(M, sInnerSize_, headDim_);
    bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, kvType, false);
    bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, kvType, true);
    if (antiQuantFlag_) {
        bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND_ALIGN,
                      matmul_tiling::DataType::DT_INT32);
        bmm1OrgKa = headDimAlign_;
        baseN = MAX_MATMUL_BASE; // antiquant to split K
    } else {
        bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
        bmm1OrgKa = headDim_;
        baseN = MATMUL_BASE_N;
    }
    // 存在输入query是BNSD格式，但使能PA，需要按BSH SetOrgShape
    if (inputLayout_ == IfaLayout::BSH_BSND) {
        bmm1.SetOrgShape(M, seqSize_, bmm1OrgKa, headDim_ * numKvHeads_);
    } else {
        bmm1.SetOrgShape(M, seqSize_, bmm1OrgKa, headDim_);
    }
    bmm1.SetBias(false);

    uint32_t bmm1BaseN = std::min(Align(sInnerSize_, 16U), baseN);
    if (pageAttentionFlag_) {
        AdjustPABmm1Tiling(bmm1BaseN);
    }

    if (!isSysPrefixTiling_) {
        // 向下对齐保证M*N不超过L0C，且由于bmm1BaseN有最大限制，L0C_SIZE / sizeof(float) / bmm1BaseN不会小于16
        uint32_t bmm1MaxBaseM = Align(static_cast<uint32_t>(L0C_SIZE / sizeof(float) / bmm1BaseN) - 16U, 16U);
        OP_CHECK_IF((bmm1.SetFixSplit(std::min(Align(M, 16U), bmm1MaxBaseM), bmm1BaseN) == -1),
                   OP_LOGE(ifaContext_->opName, "bmm1 SetFixSplit fail"), return false);
    } else {
        // prefix 模式下A矩阵较大，可能超过L0A，使用默认值-1，由matmul计算baseM
        OP_CHECK_IF((bmm1.SetFixSplit(-1, bmm1BaseN) == -1), OP_LOGE(ifaContext_->opName, "bmm1 SetFixSplit fail"),
                   return false);
    }

    OP_CHECK_IF((bmm1.SetTraverse(matmul_tiling::MatrixTraverse::FIRSTN) == -1),
               OP_LOGE(ifaContext_->opName, "bmm1 SetTraverse fail"), return false);

    if (bmm1.GetTiling(tilingData_->bmm1TilingData) == -1) {
        OP_LOGE(ifaContext_->opName, "bmm1 get tiling fail");
        return false;
    }
    return true;
}

bool IFATiling::GetBmm2Tiling(const matmul_tiling::DataType &kvType, const uint32_t M) const
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(ifaContext_->platformInfo);
    matmul_tiling::MatmulApiTiling bmm2(ascendcPlatform);
    bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, kvType, false);
    bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, kvType, false);
    if (antiQuantFlag_) {
        bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND_ALIGN,
                      matmul_tiling::DataType::DT_INT32);
    } else {
        bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND_ALIGN,
                      matmul_tiling::DataType::DT_FLOAT);
    }
    if (slidingFlag_) {
        // (m, n, k) (so, d, si)
        bmm2.SetShape(M, headDimV_, sInnerSize_);
        // 存在输入query是BNSD格式，但使能PA，需要按BSH SetOrgShape
        if (inputLayout_ == IfaLayout::BSH_BSND) {
            bmm2.SetOrgShape(M, headDimV_ * numKvHeads_, sInnerSizeAlign_, seqSize_);
        } else {
            bmm2.SetOrgShape(M, headDimV_, sInnerSizeAlign_, seqSize_);
        }
    } else {
        // (m, n, k) (so, d, si)
        bmm2.SetShape(M, headDim_, sInnerSize_);
        // 存在输入query是BNSD格式，但使能PA，需要按BSH SetOrgShape
        if (inputLayout_ == IfaLayout::BSH_BSND) {
            bmm2.SetOrgShape(M, headDim_ * numKvHeads_, sInnerSizeAlign_, seqSize_);
        } else {
            bmm2.SetOrgShape(M, headDim_, sInnerSizeAlign_, seqSize_);
        } 
    }
    bmm2.SetBias(false);
    OP_CHECK_IF((bmm2.SetFixSplit(std::min(Align(M, 16U), MAX_MATMUL_BASE_M)) == -1),
               OP_LOGE(ifaContext_->opName, "bmm2 SetFixSplit fail"), return false);

    if (bmm2.GetTiling(tilingData_->bmm2TilingData) == -1) {
        OP_LOGE(ifaContext_->opName, "bmm2 get tiling fail");
        return false;
    }
    if (pageAttentionFlag_) {
        AdjustPABmm2Tiling();
    }
    return true;
}

bool IFATiling::FillTilingBmm() const
{
    matmul_tiling::DataType qType;
    matmul_tiling::DataType kvType;

    if (!GetMatmulType(inputQType_, &qType) || !GetMatmulType(inputKvType_, &kvType)) {
        OP_LOGE(ifaContext_->opName, "get matmul type error");
        return false;
    }
    uint32_t M = msdIterNum_ * nNumOfQInOneGroup_;
    if (isSysPrefixTiling_) {
        M *= batchSizeQ_;
    }
    return GetBmm1Tiling(kvType, M) && GetBmm2Tiling(kvType, M);
}

ge::graphStatus IFATiling::FillTilingMla()
{
    FillTilingBaseParamsMla();
    FillTilingSplitKVMla();
    FillTilingCoreParamsMla();
    FillTilingSingleCoreParamsMla();
    FillTilingSingleCoreTensorSizeMla();
    return ge::GRAPH_SUCCESS;
}

void IFATiling::FillTilingBaseParamsMla()
{
    tilingDataMla_.baseParams.set_batchSize(batchSize_);
    tilingDataMla_.baseParams.set_seqSize(sMax_);
    tilingDataMla_.baseParams.set_qSeqSize(qSeqSize_);
    tilingDataMla_.baseParams.set_blockSize(blockSize_);
    tilingDataMla_.baseParams.set_maxBlockNumPerBatch(maxBlockNumPerBatch_);
    tilingDataMla_.baseParams.set_scaleValue(scaleValue_);
    tilingDataMla_.baseParams.set_nNumOfQInOneGroup(numHeads_ / numKvHeads_);
    tilingDataMla_.baseParams.set_actualLenQDims(actualLenQDims_);
    tilingDataMla_.baseParams.set_actualLenDims(actualLenDims_);
    tilingDataMla_.baseParams.set_attenMaskFlag(attenMaskFlag_ ? 1 : 0);
    tilingDataMla_.baseParams.set_attenMaskSize(attenMaskSize_);
    tilingDataMla_.baseParams.set_sparseMode(sparseMode_);
    tilingDataMla_.baseParams.set_outputLayout(static_cast<uint32_t>(outputLayout_));
    tilingDataMla_.baseParams.set_softmaxLseFlag(softmaxLseFlag_ ? 1 : 0);
    tilingDataMla_.baseParams.set_tSeqSize(tSeqSize_);
}

// for flash decode
void IFATiling::FillTilingSplitKVMla()
{
    tilingDataMla_.splitKVParams.set_s2(kvSplitPart_);
    uint32_t sInnerLoopSize_ = (maxActualseq_ + (kvSplitPart_ - 1U)) / kvSplitPart_;
    if (pageAttentionFlag_) {
        sInnerLoopSize_ = Align(sInnerLoopSize_, blockSize_);
        OP_LOGD(ifaContext_->opName, "PA FlashDecode is enabled, sInnerLoopSize is %u, blockSize is %u",
                  sInnerLoopSize_, blockSize_);
    }
    if (inputKvType_ == ge::DT_INT4) {
        sInnerLoopSize_ = Align(sInnerLoopSize_, 2U);
    }
    tilingDataMla_.splitKVParams.set_sInnerLoopSize(sInnerLoopSize_);
    if (balanceModeFlag_) {
        tilingDataMla_.splitKVParams.set_accumOutSize(aicNum_ * 2U * numKvHeads_ * FIA_BALANCE_SG_BASIC_SIZE * headDimAlign_);   // 每个核可能有头规约和尾规约，一共两份规约信息
        tilingDataMla_.splitKVParams.set_logSumExpSize(2U * aicNum_ * 2U * numKvHeads_ * FIA_BALANCE_SG_BASIC_SIZE *  // 每个核可能有头规约和尾规约，一共两份规约信息;sum + max
                                                    (BYTE_BLOCK / blockTypeSize_));
    } else {
        tilingDataMla_.splitKVParams.set_accumOutSize(batchSizeQ_ * numHeads_ * kvSplitPart_ * headDimAlign_);
        tilingDataMla_.splitKVParams.set_logSumExpSize(2U * batchSizeQ_ * numHeads_ * kvSplitPart_ *
                                                    (BYTE_BLOCK / blockTypeSize_)); // 2: sum + max
    }
    if (!splitKVFlag_) {
        tilingDataMla_.splitKVParams.set_s2(0);
    }
}

void IFATiling::FillTilingCoreParamsMla()
{
    uint32_t *coreStartIdx = tilingDataMla_.increFlashAttentionCoreParams.get_coreSidxEnd();
    memcpy_s(coreStartIdx, MAX_AIC_CORE_NUM * sizeof(uint32_t), startIdxEachCore_, MAX_AIC_CORE_NUM * sizeof(uint32_t));
}

void IFATiling::FillTilingSingleCoreParamsMla()
{
    tilingDataMla_.increFlashAttentionSingleCoreParams.set_singleProcessSInnerSize(sInnerSize_);
    tilingDataMla_.increFlashAttentionSingleCoreParams.set_usedCoreNum(usedCoreNum_);
    tilingDataMla_.increFlashAttentionSingleCoreParams.set_groupSplitSize(groupSplitSize_);
    tilingDataMla_.increFlashAttentionSingleCoreParams.set_s1SplitSize(s1SplitSize_);
}

void IFATiling::FillTilingSingleCoreTensorSizeMla()
{
    tilingDataMla_.increFlashAttentionSingleCoreTensorSize.set_mmResUbSize(mmResUbSize_);
    tilingDataMla_.increFlashAttentionSingleCoreTensorSize.set_bmm2ResUbSize(bmm2ResUbSize_);
}

bool IFATiling::GetMatmulType(ge::DataType getype, matmul_tiling::DataType *mmType) const
{
    static struct {
        ge::DataType a;
        matmul_tiling::DataType b;
    } typeTrans[] = {{ge::DT_FLOAT16, matmul_tiling::DataType::DT_FLOAT16},
                     {ge::DT_BF16, matmul_tiling::DataType::DT_BF16},
                     {ge::DT_INT8, matmul_tiling::DataType::DT_INT8},
                     {ge::DT_INT4, matmul_tiling::DataType::DT_INT4},
                     {ge::DT_FLOAT, matmul_tiling::DataType::DT_FLOAT}};

    for (uint32_t i = 0; i < sizeof(typeTrans) / sizeof(typeTrans[0]); i++) {
        if (typeTrans[i].a == getype) {
            *mmType = typeTrans[i].b;
            return true;
        }
    }
    return false;
}
 // S1 * G > 128
IfaAmlaMode IFATiling::GetAmlaMode() const
{
    // 1.只有NZ才应该走进这个函数，提前判断, 2.伪量化和全量化不走amla
    if ((!ropeFlag_) || inputKvLayout_ != IfaLayout::NZ || antiQuantFlag_ || quantFlag_) {
        return IfaAmlaMode::DISABLE_AMLA;
    }
    OP_LOGI(ifaContext_->opName, "input Key and Value format is NZ and blockSize_[%u], nNumOfQInOneGroup_[%u]",
        blockSize_, nNumOfQInOneGroup_);
    // 当前暂时只支持 大于24batch G128 MTP1,3 512对齐 走3buf模板
    if (blockSize_ == 128U) {
        return IfaAmlaMode::AMLA; // s1 * G <= 128 nz amla
    }
    return IfaAmlaMode::DISABLE_AMLA;
}

ge::graphStatus IFATiling::GetKvLayoutInfo(KvLayoutInfo &kvLayoutInfo) const
{
    switch (inputKvLayout_) {
        case IfaLayout::NZ:
            kvLayoutInfo.kvLayoutVal = 2U;
            kvLayoutInfo.amlaMode = static_cast<uint8_t>(amlaMode_);
            break;
        case IfaLayout::BSH_BSND:
            kvLayoutInfo.kvLayoutVal = 1U;
            break;
        case IfaLayout::BNSD:
            kvLayoutInfo.kvLayoutVal = 0U;
            break;
        default:
            OP_LOGE(ifaContext_->opName, "not support inputKvLayout%u", kvLayoutInfo.kvLayoutVal);
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::GetInputLayoutVal(uint8_t &layoutVal) const
{
    switch (inputLayout_) {
        case IfaLayout::TND:
            layoutVal = 2U;      // 2:TND
            break;
        case IfaLayout::BSH_BSND:
            layoutVal = 1U;
            break;
        case IfaLayout::BNSD:
            layoutVal = 0U;
            break;
        default:
            OP_LOGE(ifaContext_->opName, "not support inputLayout%u", static_cast<uint32_t>(inputLayout_));
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::GetInputQueryVal(uint8_t &inputQVal) const
{
    switch (inputQType_) {
        case ge::DT_FLOAT16:
            inputQVal = 0U;
            break;
        case ge::DT_BF16:
            inputQVal = 2U;
            break;
        case ge::DT_INT8:
            inputQVal = 3U;
            break;
        default:
            OP_LOGE(ifaContext_->opName, "not support inputQType%d", inputQType_);
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::GetInputKvVal(uint8_t &inputKvVal) const
{
    switch (inputKvType_) {
        case ge::DT_FLOAT16:
            inputKvVal = 0U;
            break;
        case ge::DT_BF16:
            inputKvVal = 2U;
            break;
        case ge::DT_INT8:
            inputKvVal = 3U;
            break;
        case ge::DT_INT4:
            inputKvVal = 4U;
            break;
        default:
            OP_LOGE(ifaContext_->opName, "not support inputKvType%d", inputKvType_);
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::GetOutputVal(uint8_t &outputVal) const
{
    switch (outputType_) {
        case ge::DT_FLOAT16:
            outputVal = 0U;
            break;
        case ge::DT_BF16:
            outputVal = 2U;
            break;
        case ge::DT_INT8:
            outputVal = 3U;
            break;
        default:
            OP_LOGE(ifaContext_->opName, "not support outputType %d", outputType_);
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::GenTilingKey() const
{
    uint8_t layoutVal = 0U;
    uint8_t inputQVal = 0U;
    uint8_t inputKvVal = 0U;
    uint8_t outputVal = 0U;
    uint8_t originVal = 0U;
    uint8_t cvRatioVal = 0U;
    uint8_t splitKvVal = kvSplit_ > 0U ? 1U : 0U;
    uint8_t paVal = pageAttentionFlag_ == true ? 1U * 2U : 0U;
    uint8_t antiquantModeVal = antiquantMode_ == PER_TOKEN_MODE ? 1U * 4U : 0U;
    uint64_t modeVal = sysPrefixFlag_ ? 2U : 1U;
    if (atbRunFlag_) {
        modeVal = static_cast<uint64_t>(ATB_MODE_VAL);
    }
    KvLayoutInfo kvLayoutInfo{};
    uint8_t balanceMode = static_cast<uint8_t>(balanceModeFlag_);

    if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM_MLA || perfMode_ == IfaPerfMode::CUBE_VIEW_MM_DD) {
        if (GetKvLayoutInfo(kvLayoutInfo) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    // page attention 新模板上线后删除这里的特殊处理
    if (pageAttentionFlag_ && sMax_ == 0) {
        paVal = 0;
    }

    if (GetInputLayoutVal(layoutVal) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (GetInputQueryVal(inputQVal) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetInputKvVal(inputKvVal) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetOutputVal(outputVal) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    originVal = inputQVal;
    if (ropeFlag_ && quantFlag_) {
        originVal = outputVal; // 此处应该获取ROPE的类型，需要修改
        cvRatioVal = (cvRatio_ == 1) ? 1 : 0; // CV1:1场景为1，其他场景为0
    }

    // sparseMode == 9（tree mask）场景通过IFA_GET_TILINGKEY追加第11位做模板隔离
    uint8_t enableTreeVal = (sparseMode_ == 9U) ? 1U : 0U;

    uint64_t baseOffset =
        modeVal * IFA_TILINGKEYOFFSET + (static_cast<uint64_t>(perfMode_)) * IFA_PERF_MODE_TILINGKEYOFFSET;
    if (antiquantMode_ == PER_TOKEN_MODE || antiquantMode_ == PER_CHANNEL_MODE){
        ifaContext_->tilingKey = baseOffset + IFA_GET_TILINGKEY(layoutVal, inputQVal, inputKvVal, outputVal, originVal,
            (paVal + splitKvVal + antiquantModeVal), 0, kvLayoutInfo.kvLayoutVal, kvLayoutInfo.amlaMode, balanceMode, cvRatioVal, enableTreeVal);
    } else {
        ifaContext_->tilingKey = baseOffset + IFA_GET_TILINGKEY(layoutVal, inputQVal, inputKvVal, outputVal, originVal,
            (paVal + splitKvVal), antiquantMode_, kvLayoutInfo.kvLayoutVal, kvLayoutInfo.amlaMode, balanceMode, cvRatioVal, enableTreeVal);
    }

    OP_LOGI(ifaContext_->opName, "IFA tilingKey: %lu.", ifaContext_->tilingKey);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CalcNumBlocks()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(ifaContext_->platformInfo);
    auto aicNum = aicNum_;
    auto aivNum = aivNum_;
    UpdatePerfMode();
    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        aivNum = aicNum;
    } else {
        if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM_MLA || !splitKVFlag_) {
            if (perfMode_ == IfaPerfMode::C1_V1) { // 2:bn数不超过vector core一半时，CV开启CV 1:1
                aivNum = usedCoreNum_;             // CV 1:1时,GetTaskRation()的结果为1,所以aivNum与aicNum相等
                aicNum = aivNum;
            } else if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM || perfMode_ == IfaPerfMode::CUBE_VIEW_MM_FULL_LOAD ||
                perfMode_ == IfaPerfMode::CUBE_VIEW_MM_MLA || perfMode_ == IfaPerfMode::CUBE_VIEW_MM_DD) {
                aivNum = cvRatio_ * usedCoreNum_;
                aicNum = usedCoreNum_;
            } else {
                aivNum = Align(usedCoreNum_, 2U); // aivNum必须为偶数达成CV 1:2
                aicNum = (aivNum + 1U) / 2U;        // cube核的数量为vector核的数量按2向上对齐
            }
        }
    }
    ifaContext_->numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum); // 暂时与当前代码一致
    OP_LOGI(ifaContext_->opName, "IFA block dim: %u aiv Num: %u aic Num: %u.", ifaContext_->numBlocks, aivNum, aicNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::SharedPrefixTiling()
{
    // 重新配置长度
    isSysPrefixTiling_ = true;
    splitKVFlag_ = false;
    batchSizeQ_ = batchSize_;
    batchSize_ = 1U;
    maxActualseq_ = maxActualPrefixLen_;
    sMax_ = sMaxPrefix_;
    seqSize_ = sMax_;
    batchContinuousFlag_ = true;

    (void)ZeroTensorProcess();
    (void)Split();
    (void)SplitForLseCombine();
    (void)CalcSysPrefixWorkSpace();
    (void)FillSysPrefixTiling();
    (void)CalcSysPrefixNumBlocks();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::FillSysPrefixTiling()
{
    tilingDataPrefix_->set_prefixAttenOutOffset(prefixAttenOutOffset_);
    tilingDataPrefix_->set_userPromptAttenOutOffset(userPromptAttenOutOffset_);
    tilingDataPrefix_->set_tmpLseOffset(tmpLseOffset_);
    tilingDataPrefix_->set_prefixLen(maxActualPrefixLen_);
    tilingDataPrefix_->set_formerCoreNum(formerCoreNumSp_);
    tilingDataPrefix_->set_blockSplitBn2Range(blockSplitBn2RangeSp_);
    tilingDataPrefix_->set_tailSplitedBatchRange(tailSplitedBatchRangeSp_);
    tilingDataPrefix_->set_usedCoreNum(combinUsedCore_);
    tilingDataPrefix_->set_batchSizeQ(batchSizeQ_);
    tilingData_ = &tilingDataPrefix_->base;
    return FillTiling();
}

ge::graphStatus IFATiling::CalcSysPrefixWorkSpace()
{
    size_t size0 = workspaceSize_;
    size_t outSize = static_cast<size_t>(batchSizeQ_) * static_cast<size_t>(numHeads_) * static_cast<size_t>(headDimAlign_) * static_cast<size_t>(blockTypeSize_);
    size_t lseSize = static_cast<size_t>(4) * static_cast<size_t>(batchSizeQ_) * static_cast<size_t>(numHeads_) * static_cast<size_t>(BYTE_BLOCK);

    CalcWorkSpace();

    workspaceSize_ = std::max(workspaceSize_, size0);
    workspaceSize_ = Align(workspaceSize_, 512UL);
    prefixAttenOutOffset_ = workspaceSize_ - libapiSize_;
    workspaceSize_ += outSize;
    userPromptAttenOutOffset_ = workspaceSize_ - libapiSize_;
    workspaceSize_ += outSize;

    tmpLseOffset_ = workspaceSize_ - libapiSize_;
    workspaceSize_ += lseSize;

    if (ifaContext_->workSpaces) {
        ifaContext_->workSpaces[0] = workspaceSize_;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CalcSysPrefixNumBlocks()
{
    uint32_t numBlocks0 = ifaContext_->numBlocks;
    CalcNumBlocks();

    ifaContext_->numBlocks = std::max(numBlocks0, ifaContext_->numBlocks);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::SplitForLseCombine()
{
    uint32_t coreNum = usedCoreNum_;

    uint32_t bn = batchSizeQ_ * numKvHeads_;
    formerCoreNumSp_ = bn % coreNum;
    if (formerCoreNumSp_ == 0U) {
        blockSplitBn2RangeSp_ = bn / coreNum;
        tailSplitedBatchRangeSp_ = blockSplitBn2RangeSp_;
    } else {
        blockSplitBn2RangeSp_ = bn / coreNum + 1U;
        tailSplitedBatchRangeSp_ = blockSplitBn2RangeSp_ - 1U;
    }
    combinUsedCore_ = bn > coreNum ? coreNum : bn;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ConvertQKVandCacheContext(const gert::TilingContext &context, IncreFlashAttentionContext &ifaContext)
{
    ifaContext.query.desc = context.GetInputDesc(QUERY_INPUT_INDEX);
    ifaContext.query.shape = context.GetInputShape(QUERY_INPUT_INDEX);
    ifaContext.key.desc = context.GetInputDesc(KEY_INPUT_INDEX);
    ifaContext.key.shape = context.GetInputShape(KEY_INPUT_INDEX);
    OP_CHECK_IF((ifaContext.query.shape == nullptr) || (ifaContext.key.shape == nullptr),
               OP_LOGE(context.GetNodeName(), "shape of query or shape of key is null."), return ge::GRAPH_FAILED);
    auto batchOfQuery = ifaContext.query.shape->GetStorageShape().GetDim(0);
    auto batchOfKey = ifaContext.key.shape->GetStorageShape().GetDim(0);
    if (batchOfQuery != batchOfKey) {
        ifaContext.kCache.resize(batchOfQuery);
        ifaContext.vCache.resize(batchOfQuery);
        for (int64_t size = 0; size < batchOfQuery; ++size) {
            ifaContext.kCache[size] =
                const_cast<gert::StorageShape *>(context.GetDynamicInputShape(KEY_INPUT_INDEX, size));
            ifaContext.vCache[size] =
                const_cast<gert::StorageShape *>(context.GetDynamicInputShape(VALUE_INPUT_INDEX, size));
        }
    } else {
        ifaContext.kCache.resize(1);
        ifaContext.vCache.resize(1);
        ifaContext.kCache[0] = const_cast<gert::StorageShape *>(context.GetDynamicInputShape(KEY_INPUT_INDEX, 0));
        ifaContext.vCache[0] = const_cast<gert::StorageShape *>(context.GetDynamicInputShape(VALUE_INPUT_INDEX, 0));
    }
    ifaContext.value.desc = context.GetInputDesc(VALUE_INPUT_INDEX);
    ifaContext.value.shape = context.GetInputShape(VALUE_INPUT_INDEX);
    return ge::GRAPH_SUCCESS;
}

static void ConvertOptionalInputsContext(const gert::TilingContext &context, IncreFlashAttentionContext &ifaContext)
{
    ifaContext.pseShift.desc = context.GetOptionalInputDesc(PSE_SHIFT_INPUT_INDEX);
    ifaContext.pseShift.tensor = context.GetOptionalInputTensor(PSE_SHIFT_INPUT_INDEX);
    ifaContext.attenMask.desc = context.GetOptionalInputDesc(ATTEN_MASK_INPUT_INDEX);
    ifaContext.attenMask.tensor = context.GetOptionalInputTensor(ATTEN_MASK_INPUT_INDEX);

    ifaContext.actualSeqLengths.tensor = context.GetOptionalInputTensor(ACT_SEQ_LEN_INPUT_INDEX);
    ifaContext.deqScale1.tensor = context.GetOptionalInputTensor(DEQUANT_SCALE_1_INPUT_INDEX);
    ifaContext.quantScale1.tensor = context.GetOptionalInputTensor(QUANT_SCALE_1_INPUT_INDEX);
    ifaContext.deqScale2.tensor = context.GetOptionalInputTensor(DEQUANT_SCALE_2_INPUT_INDEX);
    ifaContext.quantScale2.tensor = context.GetOptionalInputTensor(QUANT_SCALE_2_INPUT_INDEX);
    ifaContext.quantOffset2.tensor = context.GetOptionalInputTensor(QUANT_OFFSET_2_INPUT_INDEX);
    ifaContext.deqScale1.desc = context.GetOptionalInputDesc(DEQUANT_SCALE_1_INPUT_INDEX);
    ifaContext.quantScale1.desc = context.GetOptionalInputDesc(QUANT_SCALE_1_INPUT_INDEX);
    ifaContext.deqScale2.desc = context.GetOptionalInputDesc(DEQUANT_SCALE_2_INPUT_INDEX);
    ifaContext.quantScale2.desc = context.GetOptionalInputDesc(QUANT_SCALE_2_INPUT_INDEX);
    ifaContext.quantOffset2.desc = context.GetOptionalInputDesc(QUANT_OFFSET_2_INPUT_INDEX);
    ifaContext.antiquantScale.tensor = context.GetOptionalInputTensor(ANTIQUANT_SCALE_INPUT_INDEX);
    ifaContext.antiquantOffset.tensor = context.GetOptionalInputTensor(ANTIQUANT_OFFSET_INPUT_INDEX);
    ifaContext.antiquantScale.desc = context.GetOptionalInputDesc(ANTIQUANT_SCALE_INPUT_INDEX);
    ifaContext.antiquantOffset.desc = context.GetOptionalInputDesc(ANTIQUANT_OFFSET_INPUT_INDEX);
    ifaContext.blockTable.tensor = context.GetOptionalInputTensor(BLOCK_TABLE_INPUT_INDEX);
    ifaContext.blockTable.desc = context.GetOptionalInputDesc(BLOCK_TABLE_INPUT_INDEX);
    ifaContext.kvPaddingSize.tensor = context.GetOptionalInputTensor(KV_PADDING_SIZE_INPUT_INDEX);
    ifaContext.kvPaddingSize.desc = context.GetOptionalInputDesc(KV_PADDING_SIZE_INPUT_INDEX);
}

ge::graphStatus IFATiling::ConvertContext(gert::TilingContext &context, IncreFlashAttentionContext &ifaContext)
{
    if (context.GetNodeName() == nullptr) {
        OP_LOGE("IncreFlashAttention", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    ifaContext.opName = context.GetNodeName();
    ifaContext.platformInfo = context.GetPlatformInfo();
    if (ConvertQKVandCacheContext(context, ifaContext) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    ConvertOptionalInputsContext(context, ifaContext);
    ifaContext.attenOut.desc = context.GetOutputDesc(OUTPUT_INDEX);
    ifaContext.attenOut.shape = context.GetOutputShape(OUTPUT_INDEX);

    auto attrs = context.GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context.GetNodeName(), "attrs got from GE is nullptr"),
               return ge::GRAPH_FAILED);

    ifaContext.numHeads = attrs->GetAttrPointer<uint32_t>(NUM_HEADS_ATTR_INDEX);
    ifaContext.scaleValue = attrs->GetAttrPointer<float>(SCALE_VALUE_ATTR_INDEX);
    ifaContext.layOut = attrs->GetStr(LAYOUT_ATTR_INDEX);
    ifaContext.kvHeadNums = attrs->GetAttrPointer<uint32_t>(KV_NUM_HEADS_ATTR_INDEX);
    ifaContext.blockSize = attrs->GetAttrPointer<uint32_t>(BLOCK_SIZE_ATTR_INDEX);
    ifaContext.innerPrecise = attrs->GetAttrPointer<uint32_t>(INNER_PRECISE_ATTR_INDEX);

    OP_CHECK_IF(context.GetWorkspaceSizes(1) == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "workSpaceSize got from GE is nullptr"),
               return ge::GRAPH_FAILED);
    ifaContext.workSpaces = context.GetWorkspaceSizes(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::RunBigKernelTiling(IncreFlashAttentionContext &context,
                                              IncreFlashAttentionTilingDataV2 &tilingData, bool isWorkspace)
{
    this->ifaContext_ = &context;
    this->tilingData_ = &tilingData.tilingBase;
    this->tilingDataPrefix_ = &tilingData.tilingPrefix;
    this->isWorkspace_ = isWorkspace;

    if ((this->ifaContext_->actualSeqLengths.tensor && !this->ifaContext_->actualSeqLengths.tensor->GetData<int64_t>()) ||
       (this->ifaContext_->actualSeqLengthsQ.tensor && !this->ifaContext_->actualSeqLengthsQ.tensor->GetData<int64_t>())) {
        this->isWorkspace_ = true;
        OP_LOGI(ifaContext_->opName, "IFA tiling sink.");
    }

    if (ProcessCheckAtbFormat() == ge::GRAPH_SUCCESS && atbRunFlag_) {
        return AtbTilingProcess();
    }

    if ((GetNpuInfo() != ge::GRAPH_SUCCESS) || (PreProcess() != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }

    // user prompt tiling
    if ((ZeroTensorProcess() != ge::GRAPH_SUCCESS) ||
        (Split() != ge::GRAPH_SUCCESS) ||
        (FillTiling() != ge::GRAPH_SUCCESS) ||
        (CalcWorkSpace() != ge::GRAPH_SUCCESS) ||
        (CalcNumBlocks() != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }

    if (sysPrefixFlag_ && SharedPrefixTiling() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return GenTilingKey();
}

ge::graphStatus IFATiling::IncreFlashAttentionSetTilingData(gert::TilingContext &context,
                                                            IncreFlashAttentionTilingDataV2 &tilingData)
{
    OP_CHECK_IF(context.GetRawTilingData() == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "RawTilingData got from GE context is nullptr."),
               return GRAPH_FAILED);

    if (ropeFlag_) {
        tilingDataMla_.SaveToBuffer(context.GetRawTilingData()->GetData(), context.GetRawTilingData()->GetCapacity());
        context.GetRawTilingData()->SetDataSize(tilingDataMla_.GetDataSize());
        return ge::GRAPH_SUCCESS;
    }

    if (atbRunFlag_ && pageAttentionFlag_) {
        ifaTilingAtbData.SaveToBuffer(context.GetRawTilingData()->GetData(), context.GetRawTilingData()->GetCapacity());
        context.GetRawTilingData()->SetDataSize(ifaTilingAtbData.GetDataSize());
    } else {
        tilingData.SaveToBuffer(context.GetRawTilingData()->GetData(), context.GetRawTilingData()->GetCapacity());
        context.GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    }

    return ge::GRAPH_SUCCESS;
}

std::string DataTypeToSerialString(ge::DataType type)
{
    const auto it = DATATYPE_TO_STRING_MAP.find(type);
    if (it != DATATYPE_TO_STRING_MAP.end()) {
        return it->second;
    } else {
        OP_LOGE("IncreFlashAttention", "datatype %d not support", type);
        return "UNDEFINED";
    }
}

static void PFAConvertOptionalInputTensor(ContextParamsForPFATiling &contextKeyParams, const gert::TilingContext *context)
{
    contextKeyParams.pseShift = context->GetOptionalInputTensor(PSE_SHIFT_INPUT_INDEX);
    contextKeyParams.attentionMask = context->GetOptionalInputTensor(ATTEN_MASK_INPUT_INDEX);
    contextKeyParams.actualSequenceLengthKV = context->GetOptionalInputTensor(ACT_SEQ_LEN_INPUT_INDEX);
    contextKeyParams.antiquantScale = context->GetOptionalInputTensor(ANTIQUANT_SCALE_INPUT_INDEX);
    contextKeyParams.antiquantOffset = context->GetOptionalInputTensor(ANTIQUANT_OFFSET_INPUT_INDEX);
    contextKeyParams.blockTable = context->GetOptionalInputTensor(BLOCK_TABLE_INPUT_INDEX);
    contextKeyParams.kvPaddingSize = context->GetOptionalInputTensor(KV_PADDING_SIZE_INPUT_INDEX);
}

static void PFAConvertDataTypes(ContextParamsForPFATiling &contextKeyParams, const gert::TilingContext *context)
{
    contextKeyParams.inputDataType = context->GetInputDesc(QUERY_INPUT_INDEX)->GetDataType();
    contextKeyParams.kDataType = context->GetInputDesc(KEY_INPUT_INDEX)->GetDataType();
    contextKeyParams.vDataType = context->GetInputDesc(VALUE_INPUT_INDEX)->GetDataType();
    contextKeyParams.pseShiftDataType = (contextKeyParams.pseShift != nullptr) ?
        context->GetOptionalInputDesc(PSE_SHIFT_INPUT_INDEX)->GetDataType() : contextKeyParams.inputDataType;
    contextKeyParams.maskDataType = (contextKeyParams.attentionMask != nullptr) ?
        context->GetOptionalInputDesc(ATTEN_MASK_INPUT_INDEX)->GetDataType() : contextKeyParams.inputDataType;
    contextKeyParams.blockTableType = (context->GetOptionalInputDesc(BLOCK_TABLE_INPUT_INDEX) != nullptr) ?
        context->GetOptionalInputDesc(BLOCK_TABLE_INPUT_INDEX)->GetDataType() : ge::DT_INT32;
    contextKeyParams.outputDataType = context->GetOutputDesc(0)->GetDataType();
    
    contextKeyParams.deqScaleType = (context->GetOptionalInputDesc(DEQUANT_SCALE_1_INPUT_INDEX) != nullptr) ?
        context->GetOptionalInputDesc(DEQUANT_SCALE_1_INPUT_INDEX)->GetDataType() : contextKeyParams.inputDataType;
    contextKeyParams.deqScale2Type = (context->GetOptionalInputDesc(DEQUANT_SCALE_2_INPUT_INDEX) != nullptr) ?
        context->GetOptionalInputDesc(DEQUANT_SCALE_2_INPUT_INDEX)->GetDataType() : contextKeyParams.inputDataType;

    contextKeyParams.quantScale2Type = (context->GetOptionalInputDesc(QUANT_SCALE_2_INPUT_INDEX) != nullptr) ?
        context->GetOptionalInputDesc(QUANT_SCALE_2_INPUT_INDEX)->GetDataType() : ge::DT_FLOAT;
    contextKeyParams.quantOffset2Type = (context->GetOptionalInputDesc(QUANT_OFFSET_2_INPUT_INDEX) != nullptr) ?
        context->GetOptionalInputDesc(QUANT_OFFSET_2_INPUT_INDEX)->GetDataType() : ge::DT_FLOAT;
}

static void PFAConvertInputShapes(ContextParamsForPFATiling &contextKeyParams, const gert::TilingContext *context)
{
    contextKeyParams.queryInputShape = context->GetInputShape(QUERY_INPUT_INDEX);
    contextKeyParams.keyInputShape = context->GetInputShape(KEY_INPUT_INDEX);
    contextKeyParams.valueInputShape = context->GetInputShape(VALUE_INPUT_INDEX);
    contextKeyParams.pseShiftShape = context->GetOptionalInputShape(PSE_SHIFT_INPUT_INDEX);
    contextKeyParams.attentionMaskShape = context->GetOptionalInputShape(ATTEN_MASK_INPUT_INDEX);
    contextKeyParams.deqScale1Shape = context->GetOptionalInputShape(DEQUANT_SCALE_1_INPUT_INDEX);
    contextKeyParams.scale1Shape = context->GetOptionalInputShape(QUANT_SCALE_1_INPUT_INDEX);
    contextKeyParams.deqScale2Shape = context->GetOptionalInputShape(DEQUANT_SCALE_2_INPUT_INDEX);
    contextKeyParams.scale2Shape = context->GetOptionalInputShape(QUANT_SCALE_2_INPUT_INDEX);
    contextKeyParams.offset2Shape = context->GetOptionalInputShape(QUANT_OFFSET_2_INPUT_INDEX);
    contextKeyParams.antiquantScaleShape = context->GetOptionalInputShape(ANTIQUANT_SCALE_INPUT_INDEX);
    contextKeyParams.antiquantOffsetShape = context->GetOptionalInputShape(ANTIQUANT_OFFSET_INPUT_INDEX);
    contextKeyParams.blockTableShape = context->GetOptionalInputShape(BLOCK_TABLE_INPUT_INDEX);
}

static void PFAConvertAttrs(ContextParamsForPFATiling &contextKeyParams, const gert::RuntimeAttrs *attrs)
{
    contextKeyParams.innerPrecisePtr = attrs->GetAttrPointer<int64_t>(INNER_PRECISE_ATTR_INDEX);
    contextKeyParams.headsNumber = attrs->GetAttrPointer<int64_t>(NUM_HEADS_ATTR_INDEX);

    contextKeyParams.blockSize = attrs->GetAttrPointer<int32_t>(BLOCK_SIZE_ATTR_INDEX);
    contextKeyParams.scaleValue = attrs->GetAttrPointer<float>(SCALE_VALUE_ATTR_INDEX);
    contextKeyParams.layout = attrs->GetAttrPointer<char>(LAYOUT_ATTR_INDEX);
    contextKeyParams.numKeyValueHeads = attrs->GetAttrPointer<int64_t>(KV_NUM_HEADS_ATTR_INDEX);
}

template <typename T>
ge::graphStatus IfaStartSimpleTiling(T& tilingType, IncreFlashAttentionContext &ifaContext,
                                     IncreFlashAttentionTilingDataV2 &ifaTilingData, gert::TilingContext *context)
{
    if (tilingType.RunBigKernelTiling(ifaContext, ifaTilingData) == ge::SUCCESS) {
        context->SetTilingKey(ifaContext.tilingKey);
        context->SetBlockDim(ifaContext.numBlocks);
        tilingType.IncreFlashAttentionSetTilingData(*context, ifaTilingData);
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus TilingIncreFlashAttentionAdapter(gert::TilingContext *context)
{    
    OP_CHECK_IF(context == nullptr, 
        OPS_REPORT_VECTOR_INNER_ERR("IncreflashAttention", "Tiling context is null."), 
        return ge::GRAPH_FAILED);
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "platformInfoPtr is null!"),
        return ge::GRAPH_FAILED);
    auto resultCode = FiaTilingRegistry::GetInstance().DoTilingImpl(context, nullptr);
    return resultCode;
}

ge::graphStatus IFATiling::ProcessCheckAtbFormat()
{
    OP_CHECK_IF(ifaContext_->platformInfo == nullptr,
            OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(ifaContext_->platformInfo);
    pageAttentionFlag_ = ifaContext_->blockTable.tensor != nullptr;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bSize_);
    aicNum_ = ascendcPlatform.GetCoreNumAic();
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    ifaContext_->numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum_, aicNum_, aivNum_);
    if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P) {
        socVersion_ = IfaSocVersion::SOC_ASCEND_310P;
        coreNum_ = aicNum_;
    } else {
        socVersion_ = IfaSocVersion::SOC_ASCEND_910B;
        coreNum_ = aivNum_;
    }

    atbRunFlag_ = AtbCheckFlag();
    return ge::GRAPH_SUCCESS;
}

bool IFATiling::AtbCheckFlag() {
    const uint32_t *innerPrecisePtr = ifaContext_->innerPrecise;
    innerPrecise_ = innerPrecisePtr ? *innerPrecisePtr : IFA_HIGH_PERFORMANCE;
    if (innerPrecise_ != ATB_INNER_PRECISE || !pageAttentionFlag_) {
        return false;
    }

    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        return AtbCheckFlag310();
    } else if (socVersion_ == IfaSocVersion::SOC_ASCEND_910B) {
        return AtbCheckFlag910();
    } else {
        OP_LOGD(ifaContext_->opName, "soc version not supported in ATB.");
        return false;
    }
}

bool IFATiling::AtbCheckFlag310() const
{
    if (ifaContext_->query.shape->GetStorageShape().GetDimNum() == DIM_NUM_TWO) {
        return ifaContext_->query.shape->GetStorageShape().GetDim(1) == *ifaContext_->numHeads * BLOCK_SIZE;
    } else if (ifaContext_->query.shape->GetStorageShape().GetDimNum() == DIM_NUM_THREE) {
        return ifaContext_->query.shape->GetStorageShape().GetDim(DIM_NUM_TWO) == *ifaContext_->numHeads * BLOCK_SIZE;
    } else if (ifaContext_->query.shape->GetStorageShape().GetDimNum() == DIM_NUM_FOUR) {
        return ifaContext_->query.shape->GetStorageShape().GetDim(DIM_NUM_THREE) == BLOCK_SIZE;
    } else {
        return false;
    }
}

ge::graphStatus IFATiling::AtbCheckMask()
{
    if (!attenMaskFlag_) {
        return ge::GRAPH_SUCCESS;
    }

    OP_LOGD(ifaContext_->opName, "attenMaskFlag_:%d", attenMaskFlag_);
    auto maskShape = ifaContext_->pseShift.tensor;
    auto maxPromptLen = maskShape->GetStorageShape().GetDim(1) * BLOCK_SIZE;
    auto maskDimZero = maskShape->GetStorageShape().GetDim(0);
    switch (sparseMode_) {
        case 0U: // NO_MASK
            maskHeadStride_ = 0U;
            maskBatchStride_ = (maskDimZero == static_cast<int64_t>(batchSize_)) ?
                                maskShape->GetStorageShape().GetDim(DIM_NUM_TWO) * maxPromptLen : 0;
            break;
        default:
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus IFATiling::AtbTilingCheck()
{
    numHeads_ = *ifaContext_->numHeads;
    numKvHeads_ = *ifaContext_->kvHeadNums == 0U ? numHeads_ : *ifaContext_->kvHeadNums;
    blockSize_ = *ifaContext_->blockSize;

    inputQType_ = ifaContext_->query.desc->GetDataType();
    inputKvType_ = ifaContext_->key.desc->GetDataType();
    outputType_ = ifaContext_->attenOut.desc->GetDataType();
    batchSize_ = pageAttentionFlag_ ? ifaContext_->blockTable.tensor->GetStorageShape().GetDim(0):
                                      ifaContext_->actualSeqLengths.tensor->GetShapeSize();
    headDim_ = GetHeadSize();
    
    std::string layout(ifaContext_->layOut);
    if (layout == "BSH" || layout == "BSND") {
        inputLayout_ = IfaLayout::BSH_BSND;
    } else if (layout == "BNSD") {
        inputLayout_ = IfaLayout::BNSD;
    } else if (layout == "TND") {
        inputLayout_ = IfaLayout::TND;
    } else {
        return ge::GRAPH_FAILED;
    }

    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        return AtbTilingCheck310();
    } else {
        return AtbTilingCheck910();
    }
}

ge::graphStatus IFATiling::AtbTilingCheck310() {
    OP_CHECK_IF(ifaContext_->attenMask.tensor != nullptr,
            OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName, "increFlashAttention gqa mode not support mask tensor, please use pseShift tensor for input"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->deqScale1.tensor != nullptr,
            OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName, "increFlashAttention gqa mode not support deqScale1"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->quantScale1.tensor != nullptr,
            OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName, "increFlashAttention gqa mode not support quantScale1"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->deqScale2.tensor != nullptr,
            OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName, "increFlashAttention gqa mode not support deqScale2"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->quantScale2.tensor != nullptr,
            OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName, "increFlashAttention gqa mode not support the output's dequant scale"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->quantOffset2.tensor != nullptr,
            OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName, "increFlashAttention gqa mode not support the output's dequant offset"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->antiquantScale.tensor != nullptr,
            OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName, "increFlashAttention gqa mode not support antiquantScale"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->kvPaddingSize.tensor != nullptr,
            OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName, "increFlashAttention gqa mode not support kvPaddingSize"), return ge::GRAPH_FAILED);
    if (CheckInputFormatAndLimits() != ge::SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::AtbParamGet()
{
    scaleValue_ = *ifaContext_->scaleValue;
    qTokens_ = ifaContext_->query.shape->GetStorageShape().GetDim(DIM_NUM_TWO);
    if (pageAttentionFlag_) {
        maxBlockNumPerBatch_ = ifaContext_->blockTable.tensor->GetStorageShape().GetDim(1);
        totalBlockNum_ = ifaContext_->kCache[0]->GetStorageShape().GetDim(0);
        sMax_ = maxBlockNumPerBatch_ * blockSize_;
        const uint32_t headUBSize = 128U;
        headSplit_ = std::max(1U, headUBSize / ((blockSize_ + BLOCK_SIZE * 4U - 1U) / BLOCK_SIZE * 4U));
        seqStepQ_ = CalcSeqStepQ();
        seqStepKv_ = CalcSeqStepKv();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::AtbParamSet()
{
    tilingDataBase_->set_batchSize(batchSize_);
    tilingDataBase_->set_headSize(headDim_);
    tilingDataBase_->set_qHeadNum(numHeads_);
    tilingDataBase_->set_kvHeadNum(numKvHeads_);
    tilingDataBase_->set_totalBlockNum(totalBlockNum_);
    tilingDataBase_->set_scaleValue(scaleValue_);
    tilingDataCore_->set_qTokens(qTokens_);
    tilingDataCore_->set_isTriu(0);
    if (attenMaskFlag_) {
        sparseMode_ = sparseMode_ == 0U ? 1U : sparseMode_;
        attenMaskFlag_ = sparseMode_;
    }
    tilingDataBase_->set_attenMaskFlag(attenMaskFlag_);
    tilingDataCore_->set_maskHeadStride(maskHeadStride_);
    tilingDataCore_->set_maskBatchStride(maskBatchStride_);
    if (pageAttentionFlag_) {
        tilingDataBase_->set_blockSize(blockSize_);
        tilingDataBase_->set_maxBlockNumPerBatch(maxBlockNumPerBatch_);
        tilingDataCore_->set_headSplit(headSplit_);
        tilingDataCore_->set_seqStepQ(seqStepQ_);
        tilingDataCore_->set_seqStepKv(seqStepKv_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::AtbSplitBlock() const
{
    const uint32_t taskNum = GetTotalQBlockNum();
    
    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        // A2A3切分BS1N1, 在kernel中判断
        ifaContext_->numBlocks = taskNum < ifaContext_->numBlocks ? taskNum : ifaContext_->numBlocks;
    }
    const uint32_t taskNumPerCore = taskNum / ifaContext_->numBlocks;
    const uint32_t tailTaskNum = taskNum % ifaContext_->numBlocks;
    uint32_t taskStart = 0U;
    uint32_t taskEnd = 0U;
    std::vector<uint32_t> startBlk(MAX_CORE_NUM, 0U);
    std::vector<uint32_t> endBlk(MAX_CORE_NUM, 0U);
    std::vector<uint32_t> startBatch(MAX_CORE_NUM, 0U);
    std::vector<uint32_t> endBatch(MAX_CORE_NUM, 0U);
    for (uint32_t blockIdx = 0U; blockIdx < ifaContext_->numBlocks; blockIdx++) {
        taskStart = taskEnd;
        taskEnd = blockIdx < tailTaskNum ? taskEnd + taskNumPerCore + 1U : taskEnd + taskNumPerCore;
        startBlk[blockIdx] = taskStart;
        endBlk[blockIdx] = taskEnd;
        startBatch[blockIdx] = static_cast<uint32_t>(taskStart / numHeads_);
        endBatch[blockIdx] = static_cast<uint32_t>((taskEnd - 1U) / numHeads_);
    }
    tilingDataCore_->set_startBlk(startBlk.data());
    tilingDataCore_->set_endBlk(endBlk.data());
    tilingDataCore_->set_startBatch(startBatch.data());
    tilingDataCore_->set_endBatch(endBatch.data());
    tilingDataCore_->set_totalQBlockNum(taskNum);
    return ge::GRAPH_SUCCESS;
}

bool IFATiling::AtbCheckFlag910() {
    if ((ifaContext_->actualSeqLengthsQ.tensor == nullptr) || (ifaContext_->actualSeqLengthsQ.tensor->GetData<int64_t>() == nullptr)) {
        return false;
    }
    actualLenQDims_ = ifaContext_->actualSeqLengthsQ.tensor->GetShapeSize();
    bool multiSeqQ = actualLenQDims_ > 0;
    // 910进入ATB的条件：PA + C8 + qSeqLen>1
    return ifaContext_->blockTable.tensor != nullptr && ifaContext_->key.desc->GetDataType() == ge::DT_INT8 && multiSeqQ;
}

ge::graphStatus IFATiling::AtbTilingCheck910() const {
    return ge::GRAPH_SUCCESS;
}

uint32_t IFATiling::GetTotalWorkspaceSize() const {
    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        return static_cast<uint32_t>(libapiSize_);
    }

    // 根据实际的numBlocks减少下
    uint32_t usrWorkspaceSize = static_cast<uint32_t>(WS_REPEAT_NUM * aivNum_ * BLOCKSIZE_CALC_256 * static_cast<uint32_t>(headDim_) * NUM_BYTES_FLOAT16) + 
                                static_cast<uint32_t>(WS_REPEAT_NUM * aivNum_ * WS_TMP_SIZE_PER_CORE * NUM_BYTES_FLOAT16);
    return usrWorkspaceSize + static_cast<uint32_t>(libapiSize_);
}

uint32_t IFATiling::GetHeadSize() const {
    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        return ifaContext_->query.shape->GetStorageShape().GetDim(1) * BLOCK_SIZE / numHeads_;
    }

    if (ifaContext_->query.shape->GetStorageShape().GetDimNum() == DIM_NUM_TWO) {
        return ifaContext_->query.shape->GetStorageShape().GetDim(DIM_IDX_1) / numHeads_;
    } else if (ifaContext_->query.shape->GetStorageShape().GetDimNum() == DIM_NUM_THREE) {
        return ifaContext_->query.shape->GetStorageShape().GetDim(DIM_IDX_2);
    } else {
        return ifaContext_->query.shape->GetStorageShape().GetDim(DIM_IDX_3);
    }
}

uint32_t IFATiling::CalcSeqStepQ() const
{
    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        return MAX_MATMUL_BASE_M;
    } else if (ifaContext_->actualSeqLengthsQ.tensor != nullptr) {
        uint32_t mmADataNum = static_cast<uint32_t>(L0B_SIZE / DOUBLE_BUFFER_NUM / NUM_BYTES_FLOAT16);
        uint32_t maxSeqStepQ = std::min(MAX_MATMUL_BASE_M, mmADataNum / headDim_ / BLOCK_SIZE * BLOCK_SIZE);
        return maxSeqStepQ;
    }

    return MAX_MATMUL_BASE_M;
}

uint32_t IFATiling::GetTotalQBlockNum() const
{
    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        return batchSize_ * numHeads_;
    } else if (ifaContext_->actualSeqLengthsQ.tensor != nullptr) {
        const int64_t *actualLenDataQ = ifaContext_->actualSeqLengthsQ.tensor->GetData<int64_t>();

        uint32_t totalQblockSum = 0;
        uint32_t curSeqLenQ = 0;
        uint32_t preSeqLenQ = 0;
        for (int32_t bIdx = 0; bIdx < static_cast<int>(actualLenQDims_); bIdx++) {
            // actualLenDataQ里的值单调递增
            curSeqLenQ = static_cast<uint32_t>(actualLenDataQ[bIdx]);
            uint32_t tmpBlkNum = static_cast<uint32_t>(curSeqLenQ - preSeqLenQ + seqStepQ_ - 1) / seqStepQ_;
            totalQblockSum += static_cast<uint32_t>(tmpBlkNum);
            preSeqLenQ = curSeqLenQ;
        }

        return totalQblockSum;
    }

    return batchSize_ * numHeads_;
}

uint32_t IFATiling::CalcSeqStepKv() const
{
    uint32_t tmpSeqStepKv = blockSize_;
    while (tmpSeqStepKv > seqStepQ_) {
        // blockSize不断折半，直到小于cube的seqStep
        tmpSeqStepKv = static_cast<uint32_t>(tmpSeqStepKv / HALF_REDUCE_RATE);
    }

    return tmpSeqStepKv;
}

ge::graphStatus IFATiling::AtbTilingProcess()
{
    pageAttentionFlag_ = ifaContext_->blockTable.tensor != nullptr;
    if (pageAttentionFlag_) {
        this->tilingDataBase_ = &ifaTilingAtbData.tilingBase;
        this->tilingDataCore_ = &ifaTilingAtbData.tilingPerCore;
    }

    if (CheckBaseInputsNull() != ge::SUCCESS || AtbTilingCheck() != ge::SUCCESS || AtbParamGet() != ge::SUCCESS ||
        AtbCheckMask() != ge::SUCCESS || AtbParamSet() != ge::SUCCESS || AtbSplitBlock() != ge::SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    workspaceSize_ = GetTotalWorkspaceSize();
    if (ifaContext_->workSpaces) {
        ifaContext_->workSpaces[0] = workspaceSize_;
    }
    OP_LOGD(ifaContext_->opName, "IFA block dim:%u aivNum:%u aicNum:%u", ifaContext_->numBlocks, aivNum_, aicNum_);
    OP_LOGD(ifaContext_->opName, "batch Size is: %u", batchSize_);

    OP_LOGD(ifaContext_->opName, "headDim_:%d", headDim_);
    OP_LOGD(ifaContext_->opName, "the query's heads num:%u", numHeads_);
    OP_LOGD(ifaContext_->opName, "the key/value's heads num:%u", numKvHeads_);
    OP_LOGD(ifaContext_->opName, "max Block Number Per Batch is: %u", maxBlockNumPerBatch_);
    OP_LOGD(ifaContext_->opName, "total Block Number is:%u ", totalBlockNum_);
    OP_LOGD(ifaContext_->opName, "scaleValue_:%lf", scaleValue_);

    return GenTilingKey();
}

ge::graphStatus IFATiling::DoOpTiling()
{
    OP_CHECK_IF(context_ == nullptr, OPS_REPORT_VECTOR_INNER_ERR("IncreFlashAttention", "Context is nullptr."),
               return ge::GRAPH_FAILED);
    IncreFlashAttentionContext ifaContext;
    if (ConvertContext(*context_, ifaContext) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Error occurred while converting tilingContext to ifa context");
        return ge::GRAPH_FAILED;
    }
    return DoSubOpTiling(ifaContext);
}

ge::graphStatus IFATiling::DoSubOpTiling(IncreFlashAttentionContext& ifaContext) {
    IncreFlashAttentionTilingDataV2 ifaTilingData;
    if (RunBigKernelTiling(ifaContext, ifaTilingData) == ge::SUCCESS) {
        context_->SetTilingKey(ifaContext.tilingKey);
        context_->SetBlockDim(ifaContext.numBlocks);
        context_->SetScheduleMode(BATCH_MODE_SCHEDULE);
        IncreFlashAttentionSetTilingData(*context_, ifaTilingData);
        return ge::GRAPH_SUCCESS;
    }
    // 使用SyncAll，需要设置为batchmode模式，所有核同时启动，否则多流方式下执行可能会卡死
    context_->SetScheduleMode(BATCH_MODE_SCHEDULE);
    return ge::GRAPH_FAILED;
}

IFA_EXTERN_C ge::graphStatus TilingIncreFlashAttention(gert::TilingContext *context)
{
    return TilingIncreFlashAttentionAdapter(context);
}
REGISTER_TILING_TEMPLATE_FIA(IncreFlashAttention, IFATiling, std::vector<int32_t>({(int32_t)NpuArch::DAV_2201, (int32_t)NpuArch::DAV_2002}), 90);
} // namespace optiling