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
 * \file quant_batch_matmul_v3_basic_tiling.cc
 * \brief
 */

#include "quant_batch_matmul_v3_basic_tiling.h"

#include <map>
#include <numeric>
#include <set>
#include <tuple>
#include "common/op_host/op_tiling/hash.h"
#include "graph/utils/type_utils.h"
#include "mc2_log_compat.h"
#include "quant_batch_matmul_v3_tiling_cache.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_type.h"
#include "mc2_log.h"
#include "../../op_kernel/quant_batch_matmul_v3_tiling_key.h"

using AscendC::BLOCK_CUBE;   // uint32_t 16
using AscendC::ONE_BLK_SIZE; // uint32_t 32

namespace {

constexpr uint64_t BASIC_BLOCK_SIZE_64 = 64;
constexpr uint64_t BASIC_BLOCK_SIZE_128 = 128;
constexpr uint64_t BASIC_BLOCK_SIZE_256 = 256;
constexpr uint64_t BASIC_BLOCK_SIZE_512 = 512;
constexpr uint64_t BASIC_BLOCK_SIZE = 256UL * 128UL;
constexpr uint64_t BASIC_BLOCK_K_128_BYTE = 128;
constexpr uint64_t L0C_SIZE_256_KB = 262144;
constexpr uint64_t HALF_FACTOR = 2;
constexpr uint64_t BASIC_BLOCK_LIMIT_L2_SIZE = 128;
constexpr uint64_t BASIC_BLOCK_L2_TILE_MAX = 45;
constexpr uint64_t BASIC_BLOCK_L2_TILE_MIN = 15;
constexpr uint64_t INNER_LEN_L1_MAX = 1024;
constexpr uint64_t INNER_LEN_L1_MEDIUM = 512;
constexpr uint64_t INNER_LEN_L1_MIN = 256;
constexpr double LIMIT_RATIO = 0.9;
constexpr double MN_CLOSE_RATIO = 0.1;
constexpr uint64_t IDX_L2_LOAD = 2;
constexpr uint64_t INNER_MIN = 1024;
constexpr uint64_t ROUND_BIG_SHAPE = 5; // 较大shape定义
constexpr uint64_t SELECT_COL_PARAM = 5;

constexpr double L2_SPLIT_RATIO = 100.0 / 192;         // 经验值，与非量化使能l2cache切分条件相同
constexpr double L2_SPLIT_RATIO_FOR_MIX = 110.0 / 192; // mix场景使能l2cache切分条件
constexpr double L2_TILE_TAIL_RATIO = 0.8; // L2cahe切分，尾块是主块的最小比例，目的是让切分后分区更加均匀
constexpr uint32_t L2_TILE_NUM = 4;        // L2切分的四种切分块
constexpr uint32_t L2_TILE_INDEX = 0;      // 四种切分块之一：无尾块的切分块
constexpr uint32_t L2_TILE_TAIL_INDEX = 1; // 四种切分块之一：列是尾块组成的切分块
constexpr uint32_t L2_TAIL_TILE_INDEX = 2; // 四种切分块之一：行是尾块组成的切分块
constexpr uint32_t L2_TAIL_INDEX = 3;      // 四种切分块之一：行列都是尾块组成的切分块
constexpr uint32_t OUT_TAIL_INDEX = 0;
constexpr uint32_t INNER_TAIL_INDEX = 1;
constexpr uint32_t OUT_L2_SPLIT_INDEX = 2;
constexpr uint32_t INNER_L2_SPLIT_INDEX = 3;

constexpr uint64_t SELECT_COL_ROW_FIRST_MULTI = 5;
constexpr uint64_t MAX_CLASH_NUM = 9;

constexpr uint64_t BENEFITS_MAX_M = 16;
constexpr uint64_t BENEFITS_MIN_K = 13000;
constexpr uint64_t BENEFITS_MAX_K = 18944;
constexpr uint64_t BENEFITS_MIN_N = 1500;
constexpr uint32_t INDEX_0 = 0;
constexpr uint32_t INDEX_1 = 1;
constexpr uint32_t INDEX_2 = 2;

const std::vector<uint64_t> ALL_BASE = {64, 80, 96, 128, 192, 256, 320, 384, 512};
const std::vector<uint64_t> INNER_AXIS_ND_BASE = {128, 256, 512, 1024};
const std::vector<uint64_t> INNER_AXIS_ALL_ND_BASE = {64, 128, 256, 512};
const std::vector<uint64_t> INNER_AXIS_ALIGN_BASE = {64, 96, 128, 192, 256, 320, 384, 512};
const std::vector<uint64_t> INNER_AXIS_ALIGN_NZ_BASE = {64, 96, 128, 160, 192, 256, 320, 384, 512};
// baseM/N from small to large, so baseK from large to small
const std::vector<uint64_t> K_BASE = {1024, 512, 256, 128, 64, 32};

} // namespace

namespace optiling {

bool Mc2QuantBatchMatmulV3BasicTiling::IsCapable()
{
    return true;
}

ge::graphStatus Mc2QuantBatchMatmulV3BasicTiling::GetShapeAttrsInfo()
{
    inputParams_.Reset();
    return Mc2QuantBatchMatmulV3Tiling::GetShapeAttrsInfo();
}

ge::graphStatus Mc2QuantBatchMatmulV3BasicTiling::DoOpTiling()
{
    isUbQuant_ = inputParams_.cDtype == ge::DT_BF16 || inputParams_.isPertoken;
    SetTransAttr(trans_); // mc2流程中不对trans_赋值，这里要补一下
    // 需要给aicoreParams_ 和libApiWorkSpaceSize赋值
    OP_LOGE_IF(!SetPlatformInfoForTiling(), ge::GRAPH_FAILED, inputParams_.opName, "SetPlatformInfoForTiling fail");
    // basic tiling fail -> do tbe tiling
    matmul_tiling::MultiCoreMatmulTiling mm;
    mm.SetDim(aicoreParams_.aicNum);
    GetSocVersion();
    if (CheckUseBasicTiling() && InitTilingData(mm) == ge::GRAPH_SUCCESS) {
        OP_LOGE_IF(!DoBasicTiling(), ge::GRAPH_FAILED, inputParams_.opName, "DoBasicTiling failed.");
    } else {
        OP_LOGD(inputParams_.opName, "basic tiling not support this case.");
        return ge::GRAPH_PARAM_INVALID;
    }

    tilingData_.params.isPerTensor = static_cast<uint32_t>(inputParams_.isPerTensor);
    tilingData_.params.isPertoken = static_cast<uint32_t>(inputParams_.isPertoken);
    tilingData_.params.biasDtype = static_cast<uint32_t>(inputParams_.biasDtype);
    if (isUbQuant_) {
        return CalcUbTiling();
    }
    return ge::GRAPH_SUCCESS;
}

// tbe tiling只有ND进算法，基本块算法在增量场景下相较老模板无收益点，但可以专项求解B NZ的tiling
bool Mc2QuantBatchMatmulV3BasicTiling::IsNetBNZTrans() const
{
    return !inputParams_.transA && (!inputParams_.transB && inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ);
}

bool Mc2QuantBatchMatmulV3BasicTiling::IsNetBNZDecode() const
{
    // 增量场景:在允许不多的重复加载AL1全载矩阵下，走进基本块模板，适合网络shape，可调tiling
    bool isNetDecode = inputParams_.mSize <= BASIC_BLOCK_SIZE_64; // 64: 大部分增量m在64以下
    // trans属性
    return inputParams_.mSizePerNpu == 0UL && isNetDecode && IsNetBNZTrans() &&
           (inputParams_.kSize % BASIC_BLOCK_SIZE_128 == 0);
}

// 当前只处理部分增量B NZ场景，通过tiling获取收益，而不是通过模板收益
// NZ无meta问题，可以不受固定的NZ的base块约束
bool Mc2QuantBatchMatmulV3BasicTiling::CanProcessNetDecode() const
{
    // 当前仅处理部分weight nz的纯cube增量场景
    if (!IsNetBNZDecode() || inputParams_.bFormat != ge::FORMAT_FRACTAL_NZ) {
        return false;
    }
    uint64_t coreNum = aicoreParams_.aicNum;
    uint64_t maxBaseN = GetMaxBaseN();
    uint64_t preBase = ops::CeilAlign(ops::CeilDiv(inputParams_.nSize, coreNum), static_cast<uint64_t>(ONE_BLK_SIZE));
    // 1轮时全载矩阵不会复用多个iterate
    if (preBase >= BASIC_BLOCK_SIZE_64 && preBase <= maxBaseN) {
        return true;
    }
    // 多轮时考虑重复加载量，大m有性能下降风险，暂不考虑
    if (inputParams_.mSize > BLOCK_CUBE) {
        return false;
    }
    uint64_t aFullLoad = inputParams_.mSize * inputParams_.kSize;
    uint64_t fullLoadSize = 96 * KB_SIZE; // 96: 经验值，重复加载量上限
    // 多轮，控制AL1全载矩阵的大小以控制相对老模板的重复加载量，当前m过大时重复加载量过多有性能风险
    uint64_t nCnt = ops::CeilDiv(inputParams_.nSize, BASIC_BLOCK_SIZE_256);
    if (nCnt > coreNum && aFullLoad < fullLoadSize) {
        uint64_t round = ops::CeilDiv(nCnt, coreNum);
        // 8: 经验值，控制A矩阵极小，能容忍极小重复加载下的多轮，尽可能约束确保进该基本块tiling是有收益的。
        bool isBNZInBasic = (inputParams_.nSize >= 8UL * inputParams_.kSize) && (round * aFullLoad <= fullLoadSize);
        return ((nCnt % coreNum == 0U) || (nCnt % coreNum >= coreNum / HALF_FACTOR) || isBNZInBasic);
    }
    return false;
}

bool Mc2QuantBatchMatmulV3BasicTiling::IsPertokenBasicSwitchCondition() const
{
    uint32_t M_LOWER_1_2 = 12544;
    uint32_t N_LOWER_1_2 = 1280;
    uint32_t K_LOWER_1_2 = 1152;
    uint32_t K_UPPER_1_2 = 1536;
    uint32_t BASE_BLOCK_1_2 = 682;
    uint32_t baseM = tilingData_.matmulTiling.baseM;
    uint32_t baseN = tilingData_.matmulTiling.baseN;
    uint32_t baseBlock = 0;

    if (baseM > 0 && baseN > 0) {
        baseBlock = ((inputParams_.mSize + baseM - 1) / baseM) * ((inputParams_.nSize + baseN - 1) / baseN);
    }

    bool pertokenBasicSwitch = (((inputParams_.kSize <= K_LOWER_1_2 && baseBlock >= BASE_BLOCK_1_2) ||
                                 (inputParams_.kSize > K_LOWER_1_2 && inputParams_.kSize <= K_UPPER_1_2 &&
                                  inputParams_.mSize >= M_LOWER_1_2 && inputParams_.nSize >= N_LOWER_1_2))) &&
                               (inputParams_.aFormat == ge::FORMAT_ND) &&
                               (inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ) && inputParams_.isPertoken &&
                               (!inputParams_.transA) && (!inputParams_.transB);
    isAicAiv1_2 = pertokenBasicSwitch;
    return pertokenBasicSwitch;
}

bool Mc2QuantBatchMatmulV3BasicTiling::CheckNotFullLoadForMutliIterate(uint64_t m, uint64_t n, uint64_t k) const
{
    // 增量场景:在允许不多的重复加载AL1全载矩阵下，走进基本块模板，适合网络shape B NZ，可调tiling
    // 在增量场景下，基本块模板相较老模板无任何优势反而每次iterate都需重新加载，因此只能处理B NZ，从tiling上获取收益
    if (CanProcessNetDecode()) {
        return true;
    }

    // 0.8: 经验值，使能基本块模板的阈值，老模板无L2cache切分而MTE2劣化
    bool l2Hit = GetTotalSize(inputParams_.mSize, inputParams_.kSize, inputParams_.nSize) <= compileInfo_.l2Size * 0.8;
    if (k < INNER_MIN && l2Hit) { // k太小会在老模板中全载获得收益
        return false;
    }
    uint64_t minMN = std::min(m, n);
    if (minMN <= BASIC_BLOCK_SIZE_256) {
        return false;
    }
    uint64_t fullLoad = ops::CeilAlign(minMN, static_cast<uint64_t>(BLOCK_CUBE)) *
                        ops::CeilAlign(k, static_cast<uint64_t>(ONE_BLK_SIZE));
    const uint64_t maxMultiLoad = 160 * KB_SIZE; // 160: 单核重复加载的数据量
    uint64_t base = BASIC_BLOCK_SIZE_64;
    if ((m <= n && !inputParams_.transB) || (m > n && inputParams_.transA)) {
        base = BASIC_BLOCK_SIZE_128;
    }
    if (minMN < BASIC_BLOCK_SIZE_512) { // 切分后可能在单核上全载计算多个base块
        fullLoad = std::min(fullLoad, BASIC_BLOCK_SIZE_128 * ops::CeilAlign(k, static_cast<uint64_t>(ONE_BLK_SIZE)));
    }

    if (fullLoad <= 448U * INNER_MIN) { // 448: A/B L1全载大小；
        if ((ops::CeilDiv(std::max(m, n), aicoreParams_.aicNum * base) - 1) * fullLoad > maxMultiLoad) {
            return false;
        }
        uint64_t inner = 1;
        if (m <= n) {
            inner = inputParams_.transA ? m : k;
        } else {
            inner = inputParams_.transB ? k : n;
        }
        // 重复加载不多但是低轴非对齐也会因为重复加载导致劣化
        return inner % BASIC_BLOCK_SIZE_128 == 0;
    }
    return true;
}

bool Mc2QuantBatchMatmulV3BasicTiling::CheckIfUseBasicInMix(uint64_t m, uint64_t /* n */, uint64_t /* k */) const
{
    // mix增量优化模板未接入basic tiling；mix没有使能L2cache切分，不支持过大shape，否则会因为L2cache命中率降低而劣化
    if (isUbQuant_) {
        if (m <= BASIC_BLOCK_SIZE_256) {
            return false;
        }
    }
    return true;
}

void Mc2QuantBatchMatmulV3BasicTiling::GetSocVersion()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    socVersion = ascendcPlatform.GetSocVersion();
}

bool Mc2QuantBatchMatmulV3BasicTiling::CheckInBasicBenefitsRange(uint64_t m, uint64_t n, uint64_t k) const
{
    // 基本块方案实测有收益的场景
    bool ret = false;
    if ((m <= BENEFITS_MAX_M) && (k >= BENEFITS_MIN_K && k <= BENEFITS_MAX_K) && (n >= BENEFITS_MIN_N)) {
        ret = true;
    }
    if ((inputParams_.aDtype != ge::DT_INT8) || (inputParams_.bDtype != ge::DT_INT8)) {
        ret = false;
    }
    if ((inputParams_.cDtype != ge::DT_FLOAT16) && (inputParams_.cDtype != ge::DT_BF16)) {
        ret = false;
    }
    if (socVersion != platform_ascendc::SocVersion::ASCEND910B) {
        ret = false;
    }
    return ret;
}

// 小shape进basic模板无收益点，暂不进
bool Mc2QuantBatchMatmulV3BasicTiling::CheckMNSmallShape(uint64_t m, uint64_t n) const
{
    return std::min(m, n) <= BASIC_BLOCK_SIZE_512 && std::max(m, n) <= KB_SIZE;
}

bool Mc2QuantBatchMatmulV3BasicTiling::CheckUseBasicTiling()
{
    QuantBatchMatmulRunParas runParams_;
    SetQuantBatchMatmulRunParas(runParams_, inputParams_);
    if (CheckSupportConditionQbmm(QbmmType::Pertoken, runParams_, aicoreParams_.aicNum, compileInfo_.supportL0c2Out)) {
        return false;
    }
    if (inputParams_.aFormat == ge::FORMAT_FRACTAL_NZ) {
        return false;
    }
    if (inputParams_.mSizePerNpu > 0UL) {
        OP_LOGD(inputParams_.opName, "get mSizePerNpu: %lu", inputParams_.mSizePerNpu);
        OP_TILING_CHECK(inputParams_.mSizePerNpu > inputParams_.mSize,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "mSizePerNpu",
                                                              std::to_string(inputParams_.mSizePerNpu).c_str(),
                                                              "M in each Npu must not be bigger than total M."),
                        return false);

        OP_TILING_CHECK(
            inputParams_.transA,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "x1 and x2", "true",
                                                  "Cannot support non-continuous M with transpose_x1 true."),
            return false);

        OP_TILING_CHECK(inputParams_.batchA > 1 || inputParams_.batchB > 1 || inputParams_.batchC > 1,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            inputParams_.opName, "x1 and x2",
                            (std::to_string(inputParams_.batchA) + ", " + std::to_string(inputParams_.batchB) + ", " +
                             std::to_string(inputParams_.batchC))
                                .c_str(),
                            "The value of M must be continuous when batch axis is used."),
                        return false);
        return true;
    }

    // 非milan版本或多batch暂不走基本块
    if (!compileInfo_.supportL0c2Out || inputParams_.batchC != 1) {
        return false;
    }

    if (inputParams_.aDtype == ge::DT_INT4) {
        return true;
    }

    // 不进基本块模板
    if (CheckSupportConditionQbmm(QbmmType::BasicLimit, runParams_, aicoreParams_.aicNum,
                                  compileInfo_.supportL0c2Out)) {
        return false;
    }
    // 若shape不在已知基本块收益范围内
    if (!CheckInBasicBenefitsRange(inputParams_.mSize, inputParams_.nSize, inputParams_.kSize) &&
        !CheckSupportConditionQbmm(QbmmType::Basic, runParams_, aicoreParams_.aicNum, compileInfo_.supportL0c2Out)) {
        // mix场景增量应走增量优化模板（tbe
        // tiling），mix暂不支持L2cache切分，因此当前mix增量和超大shape都不能走基本块模板
        if (!CheckIfUseBasicInMix(inputParams_.mSize, inputParams_.nSize, inputParams_.kSize)) {
            return false;
        }
        // 基本块模板暂无法支持L1全载矩阵驻留给多个iterate使用，因此某一轴较小（无法通过行列优先获得收益）并且能全载场景下应走老模板
        if (!CheckNotFullLoadForMutliIterate(inputParams_.mSize, inputParams_.nSize, inputParams_.kSize)) {
            return false;
        }
    }
    // 基本块模板收益点来源于大shape
    return !CheckMNSmallShape(inputParams_.mSize, inputParams_.nSize);
}

uint64_t Mc2QuantBatchMatmulV3BasicTiling::GetTotalCnt(uint64_t baseM, uint64_t baseN) const
{
    uint64_t totalCnt = 1; // 1 最少核数即最少计算一个base块
    OP_TILING_CHECK(
        baseM < BLOCK_CUBE || baseN < BLOCK_CUBE,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "baseM and baseN",
                                              (std::to_string(baseM) + " and " + std::to_string(baseN)).c_str(),
                                              "baseM and baseN must be at least 16."),
        return 1UL);
    uint64_t mCnt = inputParams_.GetTotalBaseMCnt(baseM);    // m方向需要的轮数
    uint64_t nCnt = ops::CeilDiv(inputParams_.nSize, baseN); // n方向需要的轮数
    // 前面保证了shapeSize不超int64
    totalCnt = mCnt * nCnt;
    return totalCnt;
}

void Mc2QuantBatchMatmulV3BasicTiling::DivisibleCoreLayout(uint64_t mCnt, uint64_t nCnt, uint64_t &calcOrder,
                                                           uint64_t round) const
{
    bool rowFirstDivisible = false;
    bool colFirstDivisible = false;
    if (std::max(nCnt, round) % std::min(nCnt, round) == 0) {
        rowFirstDivisible = true;
    }
    if (std::max(mCnt, round) % std::min(mCnt, round) == 0) {
        colFirstDivisible = true;
    }
    if (rowFirstDivisible && !colFirstDivisible) {
        calcOrder = COL_FIRST;
    } else if (!rowFirstDivisible && colFirstDivisible) {
        calcOrder = ROW_FIRST;
    } else if (rowFirstDivisible && colFirstDivisible) {
        // 行列都冲突优先选冲突量少的
        calcOrder = basicTiling_.baseM < basicTiling_.baseN ? COL_FIRST : ROW_FIRST;
    }
    return;
}

std::tuple<uint64_t, uint64_t, uint64_t, uint64_t>
Mc2QuantBatchMatmulV3BasicTiling::CalcCoreDistribution(uint64_t mCnt, uint64_t nCnt, uint64_t calcOrder, uint64_t round,
                                                       uint64_t usedCoreNum) const
{
    uint64_t allCnt = mCnt * nCnt;
    std::vector<uint64_t> mCoreDist(mCnt, 0);
    std::vector<uint64_t> nCoreDist(nCnt, 0);
    uint64_t preCoreNum = allCnt % usedCoreNum;
    if (preCoreNum == 0U) {
        preCoreNum = usedCoreNum;
    }

    uint64_t preTotalBlock = 0;
    if (calcOrder == ROW_FIRST) {
        for (uint64_t i = 0; i < usedCoreNum; ++i) {
            mCoreDist[preTotalBlock / nCnt] += 1;
            nCoreDist[preTotalBlock % nCnt] += 1;
            uint64_t increment = (i >= preCoreNum ? round - 1UL : round);
            preTotalBlock += increment;
        }
    } else {
        for (uint64_t i = 0; i < usedCoreNum; ++i) {
            mCoreDist[preTotalBlock % mCnt] += 1;
            nCoreDist[preTotalBlock / mCnt] += 1;
            uint64_t increment = (i >= preCoreNum ? round - 1UL : round);
            preTotalBlock += increment;
        }
    }

    uint64_t maxMCoreClash = *std::max_element(mCoreDist.begin(), mCoreDist.end());     // 读A时最大行冲突
    uint64_t maxNCoreClash = *std::max_element(nCoreDist.begin(), nCoreDist.end());     // 读B时最大列冲突
    uint64_t numL2CacheMCnt = mCnt - std::count(mCoreDist.begin(), mCoreDist.end(), 0); // L2缓存A矩阵的数据量
    uint64_t numL2CacheNCnt = nCnt - std::count(nCoreDist.begin(), nCoreDist.end(), 0); // L2缓存B矩阵的数据量

    return std::make_tuple(maxMCoreClash, maxNCoreClash, numL2CacheMCnt, numL2CacheNCnt);
}

/*
从L1总加载量和单核计算量指标判断优劣适合m,n较大场景，一般都能用满核
loadSize in L1 when no split k:
for mCnt(ceil(M / baseM))
    for nCnt(ceil(N / baseN))
        for k
            A: m * k * nCnt -> L1
            B: n * k * MCnt -> L1
loadSize = m * k * nCnt + n * k * MCnt = (m * ceil(N / baseN) + n * ceil(M / baseM)) * k
k is same for the same case, and the formula is simplified as
loadSize = m * ceil(N / baseN) + n * ceil(M / baseM)
calcSingleCoreMN = (baseM * baseN) * round
*/
int8_t Mc2QuantBatchMatmulV3BasicTiling::CheckLoadAndCalcSize(uint64_t baseM, uint64_t baseN, uint64_t bestRound,
                                                              uint64_t round, uint64_t &bestLoadSize) const
{
    uint64_t curLoadSize = inputParams_.mSize * ops::CeilDiv(inputParams_.nSize, baseN) +
                           inputParams_.nSize * inputParams_.GetTotalBaseMCnt(baseM);
    // 过大的L1加载量直接不考虑
    if (curLoadSize * LIMIT_RATIO > bestLoadSize) {
        return static_cast<int8_t>(-1);
    }
    bool isUpdate = bestLoadSize * LIMIT_RATIO > curLoadSize;
    if (round >= ROUND_BIG_SHAPE) { // 5: 较大shape的定义
        uint64_t basicLoadSize = inputParams_.mSize * ops::CeilDiv(inputParams_.nSize, basicTiling_.baseN) +
                                 inputParams_.nSize * inputParams_.GetTotalBaseMCnt(basicTiling_.baseM);
        // 0.95: 经验值，较大shape时，缩小MTE2的阈值，选到MTE2（L1加载量）小的base块
        isUpdate = isUpdate || (std::min(baseM, baseN) >= BASIC_BLOCK_SIZE_128 && basicLoadSize * 0.95 > curLoadSize);
    }
    if (isUpdate) {
        return 1;
    }
    uint64_t oriBestLoadSize = bestLoadSize;
    bestLoadSize = std::min(bestLoadSize, curLoadSize);
    // m/n都在低轴时，直接返回加载量小的
    isUpdate = CheckTrans(trans_ == Mc2QuantBatchMatmulV3Trans::A_TRANS, curLoadSize < oriBestLoadSize);
    // m/n只有一个在低轴时，优先选择256对齐的
    isUpdate = isUpdate ||
               CheckTrans(trans_ == Mc2QuantBatchMatmulV3Trans::AB_TRANS, curLoadSize < oriBestLoadSize, baseM) ||
               CheckTrans(trans_ == Mc2QuantBatchMatmulV3Trans::NO_TRANS && inputParams_.bFormat == ge::FORMAT_ND,
                          curLoadSize < oriBestLoadSize, baseN);
    if (isUpdate) {
        return 1;
    }
    // 比较相对轮数，主要是看拖尾轮，如增量场景 16*256*5轮优于16*512*3轮
    uint64_t oriBestsingleCoreMN = (basicTiling_.baseM * basicTiling_.baseN) * bestRound;
    uint64_t singleCoreMN = (baseM * baseN) * round;
    if (oriBestsingleCoreMN * LIMIT_RATIO > singleCoreMN) {
        return 1;
    }
    if (singleCoreMN <= oriBestsingleCoreMN) {
        return 0; // 还需进一步筛选
    }
    return static_cast<int8_t>(-1);
}

bool Mc2QuantBatchMatmulV3BasicTiling::CheckTrans(bool isCheckTrans, bool isSmallerLoadSize, uint64_t base) const
{
    if (!isCheckTrans) {
        return false;
    }
    if (base == BASIC_BLOCK_SIZE_256) {
        return true;
    }
    if (base == BASIC_BLOCK_SIZE_512 && isSmallerLoadSize) {
        return true;
    }
    return false;
}

void Mc2QuantBatchMatmulV3BasicTiling::Int4LowerAxisAlign(uint64_t &baseM, uint64_t &baseN) const
{
    if (inputParams_.aDtype != ge::DT_INT4) {
        return;
    }
    if (!inputParams_.transB) {
        baseN = ops::CeilAlign(baseN, BASIC_BLOCK_SIZE_64);
    }
    if (inputParams_.transA) {
        baseM = ops::CeilAlign(baseM, BASIC_BLOCK_SIZE_64);
    }
    return;
}

void Mc2QuantBatchMatmulV3BasicTiling::ModifyBase(uint64_t &baseM, uint64_t &baseN) const
{
    if (baseM > inputParams_.GetMatmulApiMSize()) {
        uint64_t m0 = inputParams_.transA ? ONE_BLK_SIZE : BLOCK_CUBE;
        baseM = ops::CeilAlign(inputParams_.GetMatmulApiMSize(), m0);
    }
    if (baseN > inputParams_.nSize) {
        uint64_t n0 = inputParams_.transB ? BLOCK_CUBE : ONE_BLK_SIZE;
        baseN = ops::CeilAlign(inputParams_.nSize, n0);
    }
    Int4LowerAxisAlign(baseM, baseN);
}

// 1.选择核数多，轮数，
// 2.在计算访存比相同情况下，同地址访问冲突可接受的情况下，L2缓存数据量少更新
// 3.m,n差不多大时，选择baseM大的，减少MTE1
// basicMetrics: round数，coreClash, firstL2Load, minL1LoadSize
void Mc2QuantBatchMatmulV3BasicTiling::CompareBase(std::vector<uint64_t> &basicMetrics, uint64_t baseM, uint64_t baseN)
{
    // 遍历base候选解有可能相同，剪枝
    if (baseM == basicTiling_.baseM && baseN == basicTiling_.baseN) {
        return;
    }
    bool isUpdate = false;
    uint64_t mSize = inputParams_.GetTotalMatmulApiMSize(baseM);
    uint64_t mCnt = ops::CeilDiv(mSize, baseM);
    uint64_t nCnt = ops::CeilDiv(inputParams_.nSize, baseN);
    uint64_t totalCnt = mCnt * nCnt;
    uint64_t usedCoreNum = std::min(totalCnt, aicoreParams_.aicNum);
    uint64_t round = ops::CeilDiv(totalCnt, usedCoreNum);
    // 如果L1加载量过多或者轮数拖尾，都不更新
    // 3: idx of dataSize of L1 in basicMetrics
    int8_t res = CheckLoadAndCalcSize(baseM, baseN, basicMetrics[0], round, basicMetrics[3]);
    if (res == -1) {
        return;
    } else if (res == 1) {
        isUpdate = true;
    }

    // 在L1加载数和无过度拖尾轮数后，优先轮数。如果核数和轮数近似相等，优先保障计算访存比
    isUpdate = isUpdate || (round >= ROUND_BIG_SHAPE && CheckCalcAndMemRatio(baseM, baseN));

    uint64_t coreClash = 0;
    uint64_t firstL2Load = 0;
    // 在计算访存比相同时，看第一次L2加载量
    if (!isUpdate && (baseM + baseN == (basicTiling_.baseM + basicTiling_.baseN))) {
        uint64_t calcOrder = GetCalcOrder(mCnt, nCnt, mSize, inputParams_.nSize, usedCoreNum);
        auto coreDist = CalcCoreDistribution(mCnt, nCnt, calcOrder, round, usedCoreNum);
        coreClash = std::max(std::get<0>(coreDist), std::get<1>(coreDist));
        // 2: idx of numL2CacheMCnt; 3: idx of numL2CacheNCnt
        firstL2Load = std::get<2>(coreDist) * baseM + std::get<3>(coreDist) * baseN;
        // baseM > baseN时，需要check最大行列冲突，第一轮加载到L2的数据量
        isUpdate = CheckL2Load(basicMetrics, coreClash, firstL2Load);
        // m,n接近的相对比值，考虑MTE1，milan L1->L0A更快
        isUpdate = isUpdate || CheckMTE1(baseM, baseN);
    }

    if (isUpdate) {
        basicTiling_.baseM = baseM;
        basicTiling_.baseN = baseN;
        basicTiling_.usedCoreNum = usedCoreNum;
        basicMetrics[0] = round;
        if (firstL2Load == 0) { // 说明没计算，要计算
            // 2: idx of firstL2Load in basicMetrics
            CalcClashAndFirstL2Load(basicMetrics[1], basicMetrics[IDX_L2_LOAD], mCnt, nCnt, basicMetrics[0]);
        } else {
            basicMetrics[1] = coreClash;
            basicMetrics[IDX_L2_LOAD] = firstL2Load; // 2: idx of firstL2Load in basicMetrics
        }
    }
}

// 计算访存比：越大越好
// 原有公式= (baseM * baseN * baseK) / (16 * 16 * 32) /  (baseK * (baseM + baseN)) -> baseM * baseN / (baseM + baseN)
bool Mc2QuantBatchMatmulV3BasicTiling::CheckCalcAndMemRatio(uint64_t baseM, uint64_t baseN) const
{
    double basicRatio = (basicTiling_.baseM * basicTiling_.baseN * 1.0) / (basicTiling_.baseM + basicTiling_.baseN);
    double curRatio = (baseM * baseN * 1.0) / (baseM + baseN);
    return basicRatio < curRatio;
}

// MTE2 bound场景下，是否需要减少第一轮L2加载量而交换baseM/N
bool Mc2QuantBatchMatmulV3BasicTiling::CheckL2Load(std::vector<uint64_t> &basicMetrics, uint64_t coreClash,
                                                   uint64_t firstL2Load) const
{
    // base从小到大遍历，因此走进该函数时，当前baseM > baseN
    // 相对第一轮L2加载量，纯cube场景下scale随路处理更重要; 另外k小多半不属于cube bound
    if (!isUbQuant_ && inputParams_.nSize >= inputParams_.kSize) {
        return false;
    }
    // 如果baseM > baseN且第一轮L2加载量还增大的情况下，应不swap
    if (firstL2Load >= basicMetrics[IDX_L2_LOAD]) {
        return false;
    }
    // 在纯cube场景下，由于scale fixp随路做，倾向于baseN更大。因此纯cube场景让baseM > baseN的条件应更苛刻
    uint64_t maxCoreClash = isUbQuant_ ? 8 : 6; // 8: mix最大行列冲突，6：cube允许的最大行列冲突
    // 超过最大行列冲突，或者新冲突相较basictiling超1倍，应舍弃
    uint64_t basicCoreClash = basicMetrics[1];
    if ((coreClash > maxCoreClash || coreClash / HALF_FACTOR > basicCoreClash)) {
        return false;
    }
    // basicCoreClash 3: 7168, coreClash 6: 4096, 7168 / (6/3) / betterRatio = 4778 > 4096, 可以交换
    // 0.75:
    // 经验值，mix场景每份冲突第一次加载到L2的数据量的有收益比例，防止该劣化场景：增大了行列冲突，但是减少的L2加载量并不多
    double betterRatio = isUbQuant_ ? 0.75 : 0.85;                         // 0.85: 经验值，纯cube的阈值
    betterRatio = std::max(betterRatio * coreClash / basicCoreClash, 1.3); // 1.3: 最小L2加载量优势倍数
    return (basicMetrics[IDX_L2_LOAD] / betterRatio) >= firstL2Load;
}

// L0A的写速度是L0B的2倍，L1->L0读写并行，但是两个L0A/L0B写不并行，当M,N相近时，让baseM更大收益会更好
// 待完善：若cube bound场景下，baseM > baseN能加快MTE1，提高流水并行度
bool Mc2QuantBatchMatmulV3BasicTiling::CheckMTE1(uint64_t baseM, uint64_t baseN) const
{
    bool isMNClose = std::abs(static_cast<int64_t>(inputParams_.GetMatmulApiMSize() - inputParams_.nSize)) <
                     MN_CLOSE_RATIO * inputParams_.nSize;
    return (baseM < baseN && isMNClose);
}

bool Mc2QuantBatchMatmulV3BasicTiling::CheckBiasAndScale(uint64_t baseN, uint64_t dbL0c) const
{
    // bias int32(BT 1024B)对baseN的影响，不超过256; 开DB不超过128
    // scale uint64(FB 2048B)目前对baseN无影响，api会对超256的scale再做tiling
    uint64_t maxBiasBaseN = dbL0c == 1 ? BASIC_BLOCK_SIZE_256 : BASIC_BLOCK_SIZE_128;
    // FB 2KB, api再切分，tbe tiling亦如此
    uint64_t maxScaleBaseN = dbL0c == 1 ? BASIC_BLOCK_SIZE_512 : BASIC_BLOCK_SIZE_128;
    bool isBiasInvalid = inputParams_.hasBias && (inputParams_.biasDtype == ge::DT_INT32) && baseN > maxBiasBaseN;
    bool isScaleInvalid = !isUbQuant_ && baseN > maxScaleBaseN;
    return !(isBiasInvalid || isScaleInvalid);
}

uint64_t Mc2QuantBatchMatmulV3BasicTiling::GetMaxBaseN() const
{
    // bias int32(BT 1024B)对baseN的影响，不超过256; 开DB不超过128
    // scale uint64(FB 2048B)目前对baseN无影响，api会对超256的scale再做tiling
    if (inputParams_.hasBias && (inputParams_.biasDtype == ge::DT_INT32)) {
        return BASIC_BLOCK_SIZE_256;
    }
    return BASIC_BLOCK_SIZE_512;
}

bool Mc2QuantBatchMatmulV3BasicTiling::CheckDbL0c() const
{
    // dataDtype of l0c is int32_t
    uint64_t dbBaseMN = compileInfo_.l0cSize / NUM_DB / sizeof(int32_t);
    // 不超过L0C大小也不超过BT/FB大小
    return (basicTiling_.baseM * basicTiling_.baseN <= dbBaseMN) && CheckBiasAndScale(basicTiling_.baseN, NUM_DB);
}

bool Mc2QuantBatchMatmulV3BasicTiling::GetBaseK(uint64_t baseM, uint64_t baseN)
{
    // baseN最大为512, baseK至少为64，满足S8/S4
    uint64_t baseKa = GetShapeWithDataType(compileInfo_.l0aSize / NUM_DB / baseM, inputParams_.aDtype);
    uint64_t baseKb = GetShapeWithDataType(compileInfo_.l0aSize / NUM_DB / baseN, inputParams_.bDtype);
    uint64_t baseK = std::min(baseKa, baseKb);
    // K从大到小遍历，尽可能用满L0, 减少指令数，减少scalar
    for (size_t i = 0; i < K_BASE.size(); ++i) {
        if (baseK >= K_BASE[i]) {
            basicTiling_.baseK = K_BASE[i];
            return true;
        }
    }
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "baseK", std::to_string(baseK).c_str(),
                                          "The value of baseK must be valid.");
    return false;
}

void Mc2QuantBatchMatmulV3BasicTiling::CalcClashAndFirstL2Load(uint64_t &coreClash, uint64_t &firstL2Load,
                                                               uint64_t mCnt, uint64_t nCnt, uint64_t round) const
{
    uint64_t calcOrder = GetCalcOrder(mCnt, nCnt, inputParams_.GetTotalMatmulApiMSize(basicTiling_.baseM),
                                      inputParams_.nSize, basicTiling_.usedCoreNum);
    auto coreDist = CalcCoreDistribution(mCnt, nCnt, calcOrder, round, basicTiling_.usedCoreNum);
    coreClash = std::max(std::get<0>(coreDist), std::get<1>(coreDist));
    // 2: idx of firstL2Load in basicMetrics; 2: idx of numL2CacheMCnt; 3: idx of numL2CacheNCnt
    firstL2Load = std::get<2>(coreDist) * basicTiling_.baseM + std::get<3>(coreDist) * basicTiling_.baseN;
}

void Mc2QuantBatchMatmulV3BasicTiling::InitBasicMetrics(std::vector<uint64_t> &basicMetrics)
{
    uint64_t mCnt = inputParams_.GetTotalBaseMCnt(basicTiling_.baseM);
    uint64_t nCnt = ops::CeilDiv(inputParams_.nSize, basicTiling_.baseN);
    // 2: idx of firstL2Load in basicMetrics
    CalcClashAndFirstL2Load(basicMetrics[1], basicMetrics[IDX_L2_LOAD], mCnt, nCnt, basicMetrics[0]);
    // 3: idx of dataSize of L1 in basicMetrics
    basicMetrics[3] = inputParams_.mSize * nCnt + inputParams_.nSize * mCnt;
}

// m,n都为外轴时，核是否用满，适合m,n都不大但超过256的场景
bool Mc2QuantBatchMatmulV3BasicTiling::IsMNSmallForMultiCores(uint64_t coreNum) const
{
    if (inputParams_.transA || (!inputParams_.transB && inputParams_.bFormat == ge::FORMAT_ND)) {
        return false;
    }
    // 512: 很小的轴，存在每128B同地址访问冲突，倾向于不切分；较小的低轴也不适合外轴多切分，数据量不够带宽下降
    if (inputParams_.kSize < INNER_LEN_L1_MEDIUM) {
        return false;
    }
    uint64_t preCoreNum =
        inputParams_.GetTotalBaseMCnt(BASIC_BLOCK_SIZE_128) * ops::CeilDiv(inputParams_.nSize, BASIC_BLOCK_SIZE_256);
    uint64_t preCoreNumBig =
        inputParams_.GetTotalBaseMCnt(BASIC_BLOCK_SIZE_256) * ops::CeilDiv(inputParams_.nSize, BASIC_BLOCK_SIZE_128);
    preCoreNum = std::min(preCoreNum, preCoreNumBig);
    if (preCoreNum > coreNum / HALF_FACTOR) {
        return false;
    }
    return true;
}

void Mc2QuantBatchMatmulV3BasicTiling::ModifyNZBase(uint64_t &baseN, uint64_t coreNum) const
{
    // 小m时尽可能用满核，缩短fixp
    if (inputParams_.mSize <= BLOCK_CUBE) {
        return;
    }
    // 在满足核数能用满带宽的情况下，较大的m，要减少重复加载，一轮时同地址访问冲突也减少。
    const std::vector<uint64_t> bestBaseN = {BASIC_BLOCK_SIZE_256, BASIC_BLOCK_SIZE_512};
    for (size_t i = 0; i < bestBaseN.size(); ++i) {
        uint64_t newNCnt = ops::CeilDiv(inputParams_.nSize, bestBaseN[i]);
        // 0.8:核数能用满带宽的比例，从带宽测试数据得来
        if (newNCnt <= coreNum && newNCnt >= 0.8 * coreNum) {
            baseN = bestBaseN[i];
            return;
        }
    }
}

// 处理网络中A非转置B非转置NZ的增量场景，NZ无meta问题，可以不设置固定的base
bool Mc2QuantBatchMatmulV3BasicTiling::ProcessBNZDecode()
{
    basicTiling_.baseM = ops::CeilAlign(inputParams_.mSize, static_cast<uint64_t>(BLOCK_CUBE));
    uint64_t coreNum = aicoreParams_.aicNum;
    // 手动调参白名单
    QuantBatchMatmulRunParas runParams_;
    SetQuantBatchMatmulRunParas(runParams_, inputParams_);
    if (CheckSupportConditionQbmm(QbmmType::Basic, runParams_, aicoreParams_.aicNum, compileInfo_.supportL0c2Out)) {
        basicTiling_.baseK = BASIC_BLOCK_SIZE_128;
        basicTiling_.baseN = BASIC_BLOCK_SIZE_256;
        basicTiling_.usedCoreNum = std::min(coreNum, ops::CeilDiv(inputParams_.nSize, basicTiling_.baseN));
        return true;
    }
    uint64_t maxBaseN = std::min(GetMaxBaseN(), BASIC_BLOCK_SIZE / basicTiling_.baseM); // 256或512
    uint64_t preBase = ops::CeilAlign(ops::CeilDiv(inputParams_.nSize, coreNum), static_cast<uint64_t>(ONE_BLK_SIZE));
    if (preBase >= BASIC_BLOCK_SIZE_64 && preBase <= maxBaseN) { // 1轮
        ModifyNZBase(preBase, coreNum);
        basicTiling_.baseN = preBase;
    } else if (preBase > maxBaseN) {
        // 多轮，A矩阵可以从L2中获取
        maxBaseN = std::min(maxBaseN, BASIC_BLOCK_SIZE_256);
        basicTiling_.baseN = maxBaseN;
    } else {
        basicTiling_.baseN = BASIC_BLOCK_SIZE_64; // baseN太小，scale随路劣化严重
    }
    Int4LowerAxisAlign(basicTiling_.baseM, basicTiling_.baseN);
    basicTiling_.usedCoreNum = std::min(coreNum, ops::CeilDiv(inputParams_.nSize, basicTiling_.baseN));
    OP_TILING_CHECK(basicTiling_.usedCoreNum <= 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "usedCoreNum",
                                                          std::to_string(basicTiling_.usedCoreNum).c_str(),
                                                          "usedCoreNum must be greater than 0."),
                    return false);
    OP_TILING_CHECK(!GetBaseK(basicTiling_.baseM, basicTiling_.baseN), OP_LOGE(inputParams_.opName, "GetBaseK failed"),
                    return false);
    return true;
}

void Mc2QuantBatchMatmulV3BasicTiling::ProcessMNSmallShape(uint64_t baseM, uint64_t baseN, uint64_t coreNum)
{
    // 1轮，尽可能用多的核
    uint64_t totalCnt = GetTotalCnt(baseM, baseN);
    bool isUpdate = totalCnt > basicTiling_.usedCoreNum && totalCnt <= coreNum;
    isUpdate = isUpdate || (totalCnt == basicTiling_.usedCoreNum && baseN > basicTiling_.baseN);
    if (isUpdate) {
        basicTiling_.baseM = baseM;
        basicTiling_.baseN = baseN;
        basicTiling_.usedCoreNum = totalCnt;
    }
}

bool Mc2QuantBatchMatmulV3BasicTiling::SetBase(const std::vector<uint64_t> &mBases, const std::vector<uint64_t> &nBases)
{
    // default base in milan, 也用于大shape下剪枝小base块
    basicTiling_.baseM = BASIC_BLOCK_SIZE_128;
    basicTiling_.baseN = BASIC_BLOCK_SIZE_256;
    basicTiling_.baseK = GetShapeWithDataType(BASIC_BLOCK_K_128_BYTE, inputParams_.aDtype);
    uint64_t totalCnt = GetTotalCnt(basicTiling_.baseM, basicTiling_.baseN);
    uint64_t coreNum = aicoreParams_.aicNum;
    basicTiling_.usedCoreNum = std::min(totalCnt, coreNum);
    ModifyBase(basicTiling_.baseM, basicTiling_.baseN);
    // 存储vec: round数，coreClash, firstL2Load, 最少的L1加载量
    std::vector<uint64_t> basicMetrics = {ops::CeilDiv(totalCnt, basicTiling_.usedCoreNum), 0, 0, 0};
    InitBasicMetrics(basicMetrics);
    bool isMNSmallForMultiCores = IsMNSmallForMultiCores(coreNum);

    uint64_t baseM = 0;
    uint64_t baseN = 0;
    uint64_t kAlignedLimit = 2048;

    for (size_t i = 0; i < mBases.size(); ++i) {
        // 剪枝：需要在大shape情况下剪掉很小的base[默认base块可解决该问题], 小shape下剪枝很大的base
        // 如果小m已经不大于上一个baseM（取该解已经合适），那么肯定小于当前这个更大的baseM, 则不需要更大的baseM
        if ((inputParams_.GetMatmulApiMSize() < mBases[i] && i > 0 &&
             inputParams_.GetMatmulApiMSize() <= mBases[i - 1])) {
            break;
        }
        baseM = mBases[i];
        for (size_t j = 0; j < nBases.size(); ++j) {
            baseN = nBases[j];
            // k小于2048时，baseN是160,192,320时强制128对齐
            if (inputParams_.kSize <= kAlignedLimit && (baseN == 160 || baseN == 192 || baseN == 320)) {
                continue;
            }
            // 为小shape场景更新base组合, 以及int4场景保证低轴64对齐
            ModifyBase(baseM, baseN);
            // 剪枝：L0C放不下、bias/scale放不下BT/FB，L0C开DB时需考虑bias/scale
            if (BASIC_BLOCK_SIZE / baseM < baseN || !CheckBiasAndScale(baseN)) {
                break;
            }
            if (isMNSmallForMultiCores) {
                ProcessMNSmallShape(baseM, baseN, coreNum);
            } else {
                CompareBase(basicMetrics, baseM, baseN);
            }
        }
    }
    OP_TILING_CHECK(!GetBaseK(basicTiling_.baseM, basicTiling_.baseN), OP_LOGE(inputParams_.opName, "GetBaseK failed"),
                    return false);
    return true;
}

void Mc2QuantBatchMatmulV3BasicTiling::SetCalcOrderinMNClashCase(uint64_t mTotalCnt, uint64_t nTotalCnt)
{
    if (basicTiling_.usedCoreNum >= HALF_FACTOR) { // 除0保护
        basicTiling_.isMclash = mTotalCnt % (basicTiling_.usedCoreNum / HALF_FACTOR) == 0;
        basicTiling_.isNclash = nTotalCnt % (basicTiling_.usedCoreNum / HALF_FACTOR) == 0;
    }
    if (basicTiling_.isMclash) {
        basicTiling_.calOrder = COL_FIRST;
    } else if (basicTiling_.isNclash) {
        basicTiling_.calOrder = ROW_FIRST;
    }

    if (basicTiling_.calOrder == 0) {
        basicTiling_.calOrder = ROW_FIRST; // 默认行优先
    }
}

void Mc2QuantBatchMatmulV3BasicTiling::DetermineCalcOrder()
{
    uint64_t mTotalCnt = inputParams_.GetTotalBaseMCnt(basicTiling_.baseM);
    uint64_t nTotalCnt = ops::CeilDiv(inputParams_.nSize, basicTiling_.baseN);
    uint64_t round = ops::CeilDiv(mTotalCnt * nTotalCnt, basicTiling_.usedCoreNum);
    // 如果n大于5倍的m，进入以下判断分支
    if (inputParams_.nSize > SELECT_COL_PARAM * inputParams_.mSize) {
        basicTiling_.calOrder = COL_FIRST;
        SetCalcOrderinMNClashCase(mTotalCnt, nTotalCnt);
        return;
    }
    // 默认使能row first，如果使能col first会造成冲突数过多，则直接返回来使用默认的row first
    auto coreDistColFirst = CalcCoreDistribution(mTotalCnt, nTotalCnt, COL_FIRST, round, basicTiling_.usedCoreNum);
    auto coreDistRowFirst = CalcCoreDistribution(mTotalCnt, nTotalCnt, ROW_FIRST, round, basicTiling_.usedCoreNum);
    uint64_t mClashColFirstCase = std::get<0>(coreDistColFirst);
    uint64_t nClashColFirstCase = std::get<1>(coreDistColFirst);
    uint64_t coreClashColFirstCase = std::max(mClashColFirstCase, nClashColFirstCase);

    uint64_t mClashRowFirstCase = std::get<0>(coreDistRowFirst);
    uint64_t nClashRowFirstCase = std::get<1>(coreDistRowFirst);
    uint64_t coreClashRowFirstCase = std::max(mClashRowFirstCase, nClashRowFirstCase);
    if (coreClashColFirstCase >= MAX_CLASH_NUM && coreClashColFirstCase > coreClashRowFirstCase) {
        return;
    }

    DivisibleCoreLayout(mTotalCnt, nTotalCnt, basicTiling_.calOrder, round);
    // 兜底calc order设置
    if (basicTiling_.calOrder == 0) {
        SetCalcOrderinMNClashCase(mTotalCnt, nTotalCnt);
    }
}

bool Mc2QuantBatchMatmulV3BasicTiling::CalcL0Tiling()
{
    bool ret = false;
    switch (trans_) {
        case Mc2QuantBatchMatmulV3Trans::B_TRANS:
            ret = SetBase(ALL_BASE, ALL_BASE);
            break;
        case Mc2QuantBatchMatmulV3Trans::NO_TRANS:
            if (IsNetBNZDecode()) {
                ret = ProcessBNZDecode();
            } else {
                ret = SetBase(ALL_BASE,
                              inputParams_.bFormat == ge::FORMAT_ND ? INNER_AXIS_ND_BASE : INNER_AXIS_ALIGN_NZ_BASE);
            }
            break;
        case Mc2QuantBatchMatmulV3Trans::A_TRANS:
            if (inputParams_.mSize >= inputParams_.nSize || inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ) {
                ret = SetBase(INNER_AXIS_ND_BASE, INNER_AXIS_ALIGN_NZ_BASE);
            } else {
                ret = SetBase(INNER_AXIS_ALL_ND_BASE, INNER_AXIS_ALL_ND_BASE);
            }
            break;
        case Mc2QuantBatchMatmulV3Trans::AB_TRANS:
            ret = SetBase(INNER_AXIS_ND_BASE, ALL_BASE);
        default:
            break;
    }

    OP_TILING_CHECK(!ret, OP_LOGE(inputParams_.opName, "set L0 base failed"), return false);
    // calc db for l0c
    if (CheckDbL0c()) {
        basicTiling_.dbL0c = NUM_DB;
    }
    DetermineCalcOrder();
    return true;
}

uint64_t Mc2QuantBatchMatmulV3BasicTiling::CalcL1SizeForBiasAndScale()
{
    uint64_t reservedL1Size = 0;
    if (inputParams_.hasBias && (inputParams_.biasDtype == ge::DT_INT32)) {
        reservedL1Size += basicTiling_.baseN * basicTiling_.dbL0c * sizeof(int32_t);
    }
    if (!isUbQuant_) {
        reservedL1Size += basicTiling_.baseN * basicTiling_.dbL0c * sizeof(uint64_t);
    }
    return reservedL1Size;
}

bool Mc2QuantBatchMatmulV3BasicTiling::CalcL1Tiling()
{
    // 不切K
    basicTiling_.singleCoreK = inputParams_.kSize;
    uint64_t l1Size = compileInfo_.l1Size;
    uint64_t reserveL1Size = CalcL1SizeForBiasAndScale();
    basicTiling_.depthA1 =
        GetShapeWithDataType(l1Size / HALF_FACTOR / basicTiling_.baseM / basicTiling_.baseK, inputParams_.aDtype);
    basicTiling_.depthB1 =
        GetShapeWithDataType(l1Size / HALF_FACTOR / basicTiling_.baseN / basicTiling_.baseK, inputParams_.bDtype);
    uint64_t depthASize =
        GetSizeWithDataType(basicTiling_.depthA1 * basicTiling_.baseM * basicTiling_.baseK, inputParams_.aDtype);
    uint64_t depthBSize =
        GetSizeWithDataType(basicTiling_.depthB1 * basicTiling_.baseN * basicTiling_.baseK, inputParams_.bDtype);
    if (depthASize + depthBSize >= l1Size - reserveL1Size) {
        if (basicTiling_.depthA1 >= basicTiling_.depthB1) {
            basicTiling_.depthA1 = basicTiling_.depthA1 / HALF_FACTOR;
        } else {
            basicTiling_.depthB1 = basicTiling_.depthB1 / HALF_FACTOR;
        }
    }
    basicTiling_.stepKa = basicTiling_.depthA1 / NUM_DB;
    basicTiling_.stepKb = basicTiling_.depthB1 / NUM_DB;
    OP_TILING_CHECK(!GetStepK(basicTiling_.stepKa, basicTiling_.stepKb),
                    OP_LOGE(inputParams_.opName, "GetStepK failed"), return false);
    basicTiling_.depthA1 = basicTiling_.stepKa * NUM_DB;
    basicTiling_.depthB1 = basicTiling_.stepKb * NUM_DB;
    return true;
}

bool Mc2QuantBatchMatmulV3BasicTiling::GetStepK(uint64_t &stepKa, uint64_t &stepKb) const
{
    OP_TILING_CHECK(
        stepKa == 0 || stepKb == 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "stepKa and stepKb",
                                              (std::to_string(stepKa) + " and " + std::to_string(stepKb)).c_str(),
                                              "stepKa and stepKb must not be 0."),
        return false);
    uint64_t kL1 = GetSizeWithDataType(std::min(stepKa, stepKb) * basicTiling_.baseK, inputParams_.aDtype);
    // 小k极其容易全载，导致MTE2与（MTE1/MMAD）串行，考虑拆分DB加载
    if (inputParams_.kSize <= INNER_LEN_L1_MEDIUM) {
        kL1 = std::min(kL1, INNER_LEN_L1_MIN);
    }
    uint64_t stepK =
        GetShapeWithDataType(kL1 / INNER_LEN_L1_MIN * INNER_LEN_L1_MIN / basicTiling_.baseK, inputParams_.aDtype);
    // kL1 aligned to 256B
    if (stepK > 0) {
        if (stepKa >= stepKb) {
            CorrectStepK(stepKa, stepKb, stepK);
        } else {
            CorrectStepK(stepKb, stepKa, stepK);
        }
        // 为K轴在高轴做调整
        ModifyStepKForKOuter(stepKa, stepKb);
        return true;
    }
    if (stepKa >= stepKb) {
        stepKa = stepKa / stepKb * stepKb;
    } else {
        stepKb = stepKb / stepKa * stepKa;
    }
    return true;
}

void Mc2QuantBatchMatmulV3BasicTiling::ModifyStepKForKOuter(uint64_t &stepKa, uint64_t &stepKb) const
{
    if (std::min(stepKa, stepKb) % HALF_FACTOR != 0) { // 保证调整完还是倍数关系
        return;
    }
    uint64_t newStepK = stepKa;
    if (inputParams_.transA) {                    // kA在高轴
        newStepK = std::min(stepKa, HALF_FACTOR); // 高轴K可以适当减少，减少搬运头开销
        // 4: 经验值，一个降，另一个低轴stepK适当增加，降低整体MTE2
        if (inputParams_.transB && stepKb <= 4) {
            if (((stepKa - newStepK) * basicTiling_.baseM) >= stepKb * basicTiling_.baseN) {
                stepKb *= HALF_FACTOR;
            }
        }
        stepKa = newStepK;
    }
    if (!inputParams_.transB && inputParams_.bFormat == ge::FORMAT_ND) {
        newStepK = std::min(stepKb, HALF_FACTOR);
        // 4: 经验值，一个降，另一个低轴stepK适当增加，降低整体MTE2
        if (!inputParams_.transA && stepKa <= 4) {
            if (((stepKb - newStepK) * basicTiling_.baseN) >= stepKa * basicTiling_.baseM) {
                stepKa *= HALF_FACTOR;
            }
        }
        stepKb = newStepK;
    }
}

void Mc2QuantBatchMatmulV3BasicTiling::CorrectStepK(uint64_t &bigStepK, uint64_t &smallStepK, uint64_t minStepK) const
{
    smallStepK = minStepK;
    uint64_t times = bigStepK / smallStepK;
    // 考虑MTE2头开销无法并行导致小m,n流水差
    if (basicTiling_.baseM < BASIC_BLOCK_SIZE_64 || basicTiling_.baseN < BASIC_BLOCK_SIZE_64) {
        times = 1;
    } else if (basicTiling_.baseM < BASIC_BLOCK_SIZE_128 || basicTiling_.baseN < BASIC_BLOCK_SIZE_128) {
        times = std::min(times, HALF_FACTOR);
    }
    bigStepK = times * smallStepK;
}

uint64_t Mc2QuantBatchMatmulV3BasicTiling::GetTotalSize(uint64_t m, uint64_t k, uint64_t n) const
{
    uint64_t sizeA = GetSizeWithDataType(m * k, inputParams_.aDtype);
    uint64_t sizeB = GetSizeWithDataType(k * n, inputParams_.bDtype);
    uint64_t sizeC = GetSizeWithDataType(m * n, inputParams_.cDtype);
    return sizeA + sizeB + sizeC;
}

bool Mc2QuantBatchMatmulV3BasicTiling::IsTileClash(uint64_t outSplit, uint64_t innerSplit,
                                                   std::tuple<uint64_t, uint64_t> &tileClash,
                                                   const std::tuple<uint64_t, uint64_t, uint64_t> &params) const
{
    uint64_t outBase = std::get<0>(params);
    uint64_t innerBase = std::get<1>(params);
    uint64_t calcOrder = std::get<2>(params);
    uint64_t mTileClash = std::get<0>(tileClash);
    uint64_t nTileClash = std::get<1>(tileClash);
    // 如果有轴大小为0，说明没有这个轴对应的tile，直接返回没有冲突
    if (outSplit == 0 || innerSplit == 0) {
        return false;
    }
    uint64_t mCnt = ops::CeilDiv(outSplit, outBase);
    uint64_t nCnt = ops::CeilDiv(innerSplit, innerBase);
    uint64_t totalCnt = mCnt * nCnt;
    uint64_t usedCoreNum = std::min(totalCnt, aicoreParams_.aicNum);
    uint64_t round = ops::CeilDiv(totalCnt, usedCoreNum);
    auto coreDist = CalcCoreDistribution(mCnt, nCnt, calcOrder, round, usedCoreNum);
    uint64_t mClash = std::get<0>(coreDist);
    uint64_t nClash = std::get<1>(coreDist);
    uint64_t coreClash = std::max(mClash, nClash);
    if (coreClash > 6 || mClash > mTileClash || nClash > nTileClash) { // 6：最大行列冲突，超过6认为冲突不可接受
        return true;
    }
    tileClash = std::tie(mClash, nClash);
    return false;
}

uint64_t Mc2QuantBatchMatmulV3BasicTiling::GetCalcOrder(uint64_t mCnt, uint64_t nCnt, uint64_t mSize, uint64_t nSize,
                                                        uint64_t usedCoreNum) const
{
    uint64_t calcOrder = nSize / SELECT_COL_ROW_FIRST_MULTI > mSize ? COL_FIRST : ROW_FIRST;
    bool isMClash = false;
    bool isNClash = false;
    if (usedCoreNum >= HALF_FACTOR) {
        isMClash = mCnt % (usedCoreNum / HALF_FACTOR) == 0;
        isNClash = nCnt % (usedCoreNum / HALF_FACTOR) == 0;
    }
    calcOrder = isMClash ? COL_FIRST : (isNClash ? ROW_FIRST : calcOrder);
    return calcOrder;
}

void Mc2QuantBatchMatmulV3BasicTiling::CalcTileCnt(uint64_t outOriShape, uint64_t innerOriShape, uint64_t outBase,
                                                   uint64_t innerBase,
                                                   std::vector<std::tuple<uint64_t, uint64_t>> &tileCnt) const
{
    uint64_t maxTileBlock = 625; // 625 is 25x25, m n方向base块个数的乘积
    uint64_t outTile;
    uint64_t innerTileRecord = -1;
    uint64_t innerTile;
    for (uint64_t outerTileBlock = maxTileBlock; outerTileBlock > 0; outerTileBlock--) {
        outTile = ops::CeilDiv(outOriShape, outBase * outerTileBlock);
        // 如果当前block下的tile个数相同，为了避免重复计算，直接跳过
        if (outerTileBlock > 1 && outTile == ops::CeilDiv(outOriShape, outBase * (outerTileBlock - 1))) {
            continue;
        }
        for (uint64_t innerTileBlock = maxTileBlock / outerTileBlock; innerTileBlock > 0; innerTileBlock--) {
            innerTile = ops::CeilDiv(innerOriShape, innerBase * innerTileBlock);
            // 如果当前block下的tile个数相同，为了避免重复计算，直接跳过
            if (innerTile == innerTileRecord) {
                continue;
            }
            innerTileRecord = innerTile;
            tileCnt.push_back(std::tie(outTile, innerTile));
        }
    }
    return;
}

bool Mc2QuantBatchMatmulV3BasicTiling::CheckTileTail(uint64_t outTail, uint64_t innerTail, uint64_t outL2SplitTmp,
                                                     uint64_t innerL2SplitTmp) const
{
    if ((outTail != 0 && outTail < outL2SplitTmp * L2_TILE_TAIL_RATIO) ||
        (innerTail != 0 && innerTail < innerL2SplitTmp * L2_TILE_TAIL_RATIO)) {
        return true;
    }
    return false;
}

bool Mc2QuantBatchMatmulV3BasicTiling::CheckTileClash(
    const std::tuple<uint64_t, uint64_t, uint64_t, uint64_t> &tileInfo,
    const std::tuple<uint64_t, uint64_t, uint64_t> &params,
    std::vector<std::tuple<uint64_t, uint64_t>> &tileClash) const
{
    uint64_t outTail = std::get<OUT_TAIL_INDEX>(tileInfo);
    uint64_t innerTail = std::get<INNER_TAIL_INDEX>(tileInfo);
    uint64_t outL2Split = std::get<OUT_L2_SPLIT_INDEX>(tileInfo);
    uint64_t innerL2Split = std::get<INNER_L2_SPLIT_INDEX>(tileInfo);

    return IsTileClash(outL2Split, innerL2Split, tileClash[L2_TILE_INDEX], params) ||
           IsTileClash(outL2Split, innerTail, tileClash[L2_TILE_TAIL_INDEX], params) ||
           IsTileClash(outTail, innerL2Split, tileClash[L2_TAIL_TILE_INDEX], params) ||
           IsTileClash(outTail, innerTail, tileClash[L2_TAIL_INDEX], params);
}

uint64_t Mc2QuantBatchMatmulV3BasicTiling::CalcTile(uint64_t &outTile, uint64_t &innerTile, uint64_t &outL2Split,
                                                    uint64_t &innerL2Split,
                                                    const std::tuple<uint64_t, uint64_t, double> &params) const
{
    uint64_t outOriShape = outL2Split;
    uint64_t innerOriShape = innerL2Split;
    uint64_t outBase = std::get<0>(params);
    uint64_t innerBase = std::get<1>(params);
    uint64_t l2ThreSize = static_cast<uint64_t>(std::get<2>(params)); // 2: idx of l2ThreSize
    uint64_t maxUsedCoreNum = 0;
    uint64_t realCalcOrder = 0;
    bool initFlg = false;
    std::vector<std::tuple<uint64_t, uint64_t>> tileClash(L2_TILE_NUM, {-1, -1});

    std::vector<std::tuple<uint64_t, uint64_t>> tileCnt;
    CalcTileCnt(outOriShape, innerOriShape, outBase, innerBase, tileCnt);

    // 大到小搜索L2切块，先选切块大的，如果更小切块的冲突更少，尾块更均衡，才选择小切块
    for (size_t i = 0; i < tileCnt.size(); ++i) {
        uint64_t outTileTmp = std::get<0>(tileCnt[i]);
        uint64_t innerTileTmp = std::get<1>(tileCnt[i]);
        uint64_t outL2SplitTmp = ops::CeilAlign(ops::CeilDiv(outOriShape, outTileTmp), outBase);
        uint64_t innerL2SplitTmp = ops::CeilAlign(ops::CeilDiv(innerOriShape, innerTileTmp), innerBase);
        uint64_t mCnt = ops::CeilDiv(outL2SplitTmp, outBase);
        uint64_t nCnt = ops::CeilDiv(innerL2SplitTmp, innerBase);
        uint64_t usedCoreNum = std::min(mCnt * nCnt, aicoreParams_.aicNum);
        // 不能减少计算并行度，核数不变少
        if (usedCoreNum < maxUsedCoreNum) {
            continue;
        }

        uint64_t totalSize = GetTotalSize(outL2SplitTmp, inputParams_.kSize, innerL2SplitTmp);
        // 确保L2cache加载量不导致cache置换
        if (totalSize >= l2ThreSize) {
            continue;
        }

        uint64_t outTail = outOriShape % outL2SplitTmp;
        uint64_t innerTail = innerOriShape % innerL2SplitTmp;
        // 尾块大小尽量均衡，initFlg先选择第一组参数，确保不会因为后续条件太苛刻，导致得不到切分参数
        if (CheckTileTail(outTail, innerTail, outL2SplitTmp, innerL2SplitTmp) && initFlg) {
            continue;
        }

        // 切分的tile冲突不能太大，initFlg先选择第一组参数，确保不会因为后续条件太苛刻，导致得不到切分参数
        uint64_t calcOrder = GetCalcOrder(mCnt, nCnt, outL2SplitTmp, innerL2SplitTmp, usedCoreNum);
        std::vector<std::tuple<uint64_t, uint64_t>> tileClashTmp = tileClash;
        std::tuple<uint64_t, uint64_t, uint64_t, uint64_t> tileInfo =
            std::tie(outTail, innerTail, outL2SplitTmp, innerL2SplitTmp);
        if (CheckTileClash(tileInfo, std::tie(outBase, innerBase, calcOrder), tileClashTmp) && initFlg) {
            continue;
        }

        // 找到了更好的参数，更新数据
        outTile = outTileTmp;
        innerTile = innerTileTmp;
        outL2Split = outL2SplitTmp;
        innerL2Split = innerL2SplitTmp;
        maxUsedCoreNum = usedCoreNum;
        realCalcOrder = calcOrder;
        initFlg = true;
        tileClash = tileClashTmp;
        OP_LOGI(inputParams_.opName,
                "Update Params! OutTile: %lu, InnerTile: %lu, OutL2Split: %lu, InnerL2Split: %lu, calcOrder: %lu",
                outTile, innerTile, outL2Split, innerL2Split, calcOrder);
    }
    return realCalcOrder;
}

void Mc2QuantBatchMatmulV3BasicTiling::DoL2CacheTiling()
{
    uint64_t mSize = inputParams_.GetTotalMatmulApiMSize(basicTiling_.baseM);
    uint64_t sizeA = GetSizeWithDataType(mSize * inputParams_.kSize, inputParams_.aDtype);
    uint64_t sizeB = GetSizeWithDataType(inputParams_.kSize * inputParams_.nSize, inputParams_.bDtype);
    uint64_t sizeC = GetSizeWithDataType(mSize * inputParams_.nSize, inputParams_.cDtype);

    uint64_t totalSize = sizeA + sizeB + sizeC;
    uint64_t limitSize = BASIC_BLOCK_LIMIT_L2_SIZE * MB_SIZE;
    uint64_t tileLimit = BASIC_BLOCK_L2_TILE_MIN;
    double l2ThreSize = compileInfo_.l2Size * L2_SPLIT_RATIO;
    if (isUbQuant_) {
        l2ThreSize = compileInfo_.l2Size * L2_SPLIT_RATIO_FOR_MIX;
    }
    if (totalSize < l2ThreSize && sizeA < limitSize && sizeB < limitSize && sizeC < limitSize) {
        OP_LOGD(inputParams_.opName, "data size is small needn't split L2.");
        return;
    }
    uint64_t mL2Split = GetShapeWithDataType(mSize, inputParams_.aDtype);
    uint64_t nL2Split = GetShapeWithDataType(inputParams_.nSize, inputParams_.bDtype);
    uint64_t mTile = 1;
    uint64_t nTile = 1;

    basicTiling_.calOrder =
        CalcTile(mTile, nTile, mL2Split, nL2Split, std::tie(basicTiling_.baseM, basicTiling_.baseN, l2ThreSize));

    uint64_t mTileBlock = ops::CeilDiv(mL2Split, basicTiling_.baseM);
    uint64_t nTileBlock = ops::CeilDiv(nL2Split, basicTiling_.baseN);
    mTile = ops::CeilDiv(mSize, (mTileBlock * basicTiling_.baseM));
    nTile = ops::CeilDiv(inputParams_.nSize, (nTileBlock * basicTiling_.baseN));

    if (mTileBlock >= mSize / basicTiling_.baseM || sizeA <= tileLimit * MB_SIZE) {
        mTile = 1;
        mTileBlock = ops::CeilDiv(mSize, basicTiling_.baseM);
    }
    if (nTileBlock >= inputParams_.nSize / basicTiling_.baseN || sizeB <= tileLimit * MB_SIZE) {
        nTile = 1;
        nTileBlock = ops::CeilDiv(inputParams_.nSize, basicTiling_.baseN);
    }

    basicTiling_.mTileCntl2 = mTile;
    basicTiling_.nTileCntl2 = nTile;
    basicTiling_.mTileBlock = mTileBlock;
    basicTiling_.nTileBlock = nTileBlock;
    OP_LOGD(inputParams_.opName, "nTile or mTile bigger than 1, enable split L2cache.");
    return;
}

// A矩阵全载可能超过一半L1，考虑scale和bias占用的L1空间
// 纯cube和mix cv冰雪在增量场景下baseN不同。
// 小shape场景可能L0C可以开DB，需判断并设置
// L0的计算访存比，防止大shape下选到差距大的base块
bool Mc2QuantBatchMatmulV3BasicTiling::DoBasicTiling()
{
    Mc2QuantBatchMatmulV3HashItem hashValue(inputParams_, aicoreParams_);
    uint32_t tilingKey = Ops::Transformer::MurmurHash(&(hashValue.input()), sizeof(hashValue.input()));
    static MMBasicTilingHash tilingHashCache;
    if (tilingHashCache.Get(tilingKey, hashValue.input(), hashValue)) {
        OP_LOGD(inputParams_.opName, "tiling is in cache, input m_size is %lu, n_size is %lu, k_size is %lu",
                inputParams_.mSize, inputParams_.nSize, inputParams_.kSize);
        basicTiling_ = hashValue.GetTiling();
        PrintBasicTiling();
        return true;
    }
    OP_LOGD(inputParams_.opName, "Do basic tiling.");
    ResetBase(compileInfo_.l0cSize);
    OP_TILING_CHECK(!CalcL0Tiling(), OP_LOGE(inputParams_.opName, "CalcL0Tiling failed"), return false);
    OP_TILING_CHECK(!CalcL1Tiling(), OP_LOGE(inputParams_.opName, "CalcL1Tiling failed"), return false);

    // 基本块与L2切分融合
    basicTiling_.mTileCntl2 = 1;
    basicTiling_.nTileCntl2 = 1;
    basicTiling_.mTileBlock = inputParams_.GetTotalBaseMCnt(basicTiling_.baseM);
    basicTiling_.nTileBlock = ops::CeilDiv(inputParams_.nSize, basicTiling_.baseN);
    DoL2CacheTiling();

    // add to cache
    hashValue.SetTiling(basicTiling_);
    tilingHashCache.Add(tilingKey, hashValue.input(), hashValue);
    PrintBasicTiling();
    return true;
}

void Mc2QuantBatchMatmulV3BasicTiling::ResetBase(const uint64_t l0CSize)
{
    basicTiling_.baseM = (l0CSize == L0C_SIZE_256_KB) ? BASIC_BLOCK_SIZE_256 : BASIC_BLOCK_SIZE_128;
    basicTiling_.baseN = BASIC_BLOCK_SIZE_256;
    basicTiling_.baseK = BASIC_BLOCK_SIZE_64;
}

bool Mc2QuantBatchMatmulV3BasicTiling::IsTilingDataInvalid() const
{
    return (CheckNumberIsValid(basicTiling_.usedCoreNum, inputParams_.opName, "usedCoreNum") ||
            CheckNumberIsValid(basicTiling_.singleCoreK, inputParams_.opName, "singleCoreK") ||
            CheckNumberIsValid2(basicTiling_.baseM, inputParams_.opName, "baseM") ||
            CheckNumberIsValid2(basicTiling_.baseN, inputParams_.opName, "baseN") ||
            CheckNumberIsValid2(basicTiling_.baseK, inputParams_.opName, "baseK") ||
            CheckNumberIsValid(basicTiling_.depthA1, inputParams_.opName, "depthA1") ||
            CheckNumberIsValid(basicTiling_.depthB1, inputParams_.opName, "depthB1") ||
            CheckNumberIsValid(basicTiling_.stepM, inputParams_.opName, "baseK") ||
            CheckNumberIsValid(basicTiling_.stepN, inputParams_.opName, "stepN") ||
            CheckNumberIsValid(basicTiling_.stepKa, inputParams_.opName, "baseK") ||
            CheckNumberIsValid(basicTiling_.stepKb, inputParams_.opName, "stepKb") ||
            CheckNumberIsValid(basicTiling_.iterateOrder, inputParams_.opName, "iterateOrder") ||
            CheckNumberIsValid(basicTiling_.dbL0c, inputParams_.opName, "dbL0c") ||
            CheckNumberIsValid(basicTiling_.mTileCntl2, inputParams_.opName, "mTileCntl2") ||
            CheckNumberIsValid(basicTiling_.nTileCntl2, inputParams_.opName, "nTileCntl2") ||
            CheckNumberIsValid(basicTiling_.mTileBlock, inputParams_.opName, "mTileBlock") ||
            CheckNumberIsValid(basicTiling_.nTileBlock, inputParams_.opName, "nTileBlock"));
}

void Mc2QuantBatchMatmulV3BasicTiling::SetMatmulTilingFromBasicTiling()
{
    tilingData_.matmulTiling.M = inputParams_.GetTotalMatmulApiMSize(basicTiling_.baseM);
    tilingData_.matmulTiling.N = inputParams_.nSize;
    tilingData_.matmulTiling.Ka = inputParams_.kSize;
    tilingData_.matmulTiling.Kb = inputParams_.kSize;
    tilingData_.matmulTiling.usedCoreNum = basicTiling_.usedCoreNum;
    tilingData_.matmulTiling.singleCoreM = basicTiling_.baseM;
    tilingData_.matmulTiling.singleCoreN = basicTiling_.baseN;
    tilingData_.matmulTiling.singleCoreK = basicTiling_.singleCoreK;
    tilingData_.matmulTiling.baseM = basicTiling_.baseM;
    tilingData_.matmulTiling.baseN = basicTiling_.baseN;
    tilingData_.matmulTiling.baseK = basicTiling_.baseK;
    tilingData_.matmulTiling.depthA1 = basicTiling_.depthA1;
    tilingData_.matmulTiling.depthB1 = basicTiling_.depthB1;
    tilingData_.matmulTiling.stepM = basicTiling_.stepM;
    tilingData_.matmulTiling.stepN = basicTiling_.stepN;
    tilingData_.matmulTiling.stepKa = basicTiling_.stepKa;
    tilingData_.matmulTiling.stepKb = basicTiling_.stepKb;
    tilingData_.matmulTiling.iterateOrder = basicTiling_.iterateOrder;
    tilingData_.matmulTiling.dbL0C = basicTiling_.dbL0c; // 1: off, 2:on
    tilingData_.tileL2cacheTiling.mTileCntL2 = basicTiling_.mTileCntl2;
    tilingData_.tileL2cacheTiling.nTileCntL2 = basicTiling_.nTileCntl2;
    tilingData_.tileL2cacheTiling.mTileBlock = basicTiling_.mTileBlock;
    tilingData_.tileL2cacheTiling.nTileBlock = basicTiling_.nTileBlock;
    tilingData_.tileL2cacheTiling.calOrder = basicTiling_.calOrder;
    tilingData_.tileL2cacheTiling.isBasicTiling = 1U;
    tilingData_.params.isMClash = basicTiling_.isMclash; // 判断是不是冲突的标志位
    tilingData_.params.isNClash = basicTiling_.isNclash;
    tilingData_.params.batchA = inputParams_.batchA;
    tilingData_.params.batchB = inputParams_.batchB;
    tilingData_.params.batchC = inputParams_.batchC;
    tilingData_.params.biasThreeDim = static_cast<uint32_t>(inputParams_.batchBias > 1);
}

ge::graphStatus Mc2QuantBatchMatmulV3BasicTiling::DoLibApiTiling()
{
    OP_TILING_CHECK(IsTilingDataInvalid(), OP_LOGE(inputParams_.opName, "check tilingData invalid failed"),
                    return ge::GRAPH_FAILED);
    SetMatmulTilingFromBasicTiling();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

uint64_t Mc2QuantBatchMatmulV3BasicTiling::GetTilingKey() const
{
    if (inputParams_.cDtype == ge::DT_BF16 && IsPertokenBasicSwitchCondition()) {
        uint64_t trans = (static_cast<uint64_t>(inputParams_.transA) << 1) | static_cast<uint64_t>(inputParams_.transB);
        bool isBasicTiling = true;
        uint64_t kernelTemplateType = (static_cast<uint64_t>(isBf16Opt_) << 1) | static_cast<uint64_t>(isBasicTiling);
        uint64_t optionAttrs = static_cast<uint64_t>(isAicAiv1_2) << 1;
        return GET_TPL_TILING_KEY(trans, kernelTemplateType, static_cast<uint64_t>(inputParams_.isPertoken),
                                  optionAttrs);
    }
    return Mc2QuantBatchMatmulV3Tiling::GetTilingKey(true);
}

void Mc2QuantBatchMatmulV3BasicTiling::PrintBasicTiling() const
{
    const BasicTiling &tiling = basicTiling_;
    std::stringstream ss;
    ss << "m_size_per_npu: " << inputParams_.mSizePerNpu << "m_size: " << inputParams_.mSize
       << " n_size: " << inputParams_.nSize << " k_size: " << inputParams_.kSize
       << " used_core_num: " << tiling.usedCoreNum << " base_m: " << tiling.baseM << " base_n: " << tiling.baseN
       << " base_k: " << tiling.baseK << " single_core_k: " << tiling.singleCoreK << " depth_a1: " << tiling.depthA1
       << " depth_b1: " << tiling.depthB1 << " step_m: " << tiling.stepM << "step_n: " << tiling.stepN
       << " step_ka: " << tiling.stepKa << " step_kb: " << tiling.stepKb << " iterate_order: " << tiling.iterateOrder
       << " db_l0c: " << tiling.dbL0c << " m_tile_cnt_l2: " << tiling.mTileCntl2
       << " n_tile_cnt_l2: " << tiling.nTileCntl2 << " m_tile_block: " << tiling.mTileBlock
       << " n_tile_block: " << tiling.nTileBlock << " cal_order: " << tiling.calOrder
       << " is_mclash: " << tiling.isMclash << " is_nclash: " << tiling.isNclash;
    OPS_LOG_D(inputParams_.opName, "basic tiling %s", ss.str().c_str());
}

ge::graphStatus Mc2QuantBatchMatmulV3BasicTiling::CalcUbTiling()
{
    return Mc2QuantBatchMatmulV3Tiling::CalcUbTiling(basicTiling_.baseN, basicTiling_.baseM);
}

bool Mc2QuantBatchMatmulV3BasicTiling::GetUbDequantExtreSpace()
{
    uint64_t usedWorkSpaceSize = sizeof(int32_t) * static_cast<uint64_t>(tilingData_.matmulTiling.baseM) *
                                 tilingData_.matmulTiling.baseN * tilingData_.matmulTiling.usedCoreNum * NUM_DB;
    inputParams_.bf16ExtreWorkSpaceSize = usedWorkSpaceSize;
    OP_LOGD(inputParams_.opName,
            "Do calculating workspace for basic tiling, current workspacesize is 2*usedCoreNum*baseM*baseN.");
    return true;
}

REGISTER_OPS_TILING_TEMPLATE(Mc2QuantBatchMatmulV3, Mc2QuantBatchMatmulV3BasicTiling, 0);
} // namespace optiling
