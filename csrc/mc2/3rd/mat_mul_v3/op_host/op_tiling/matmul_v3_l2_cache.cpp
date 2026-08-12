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
 * \file matmul_v3_l2_cache.cc
 * \brief
 */
#include "matmul_v3_l2_cache.h"
#include "op_host/tiling_key.h"
#include "common/op_host/math_util.h"

using namespace optiling::mc2_matmul_v3;
using Ops::Transformer::MathUtil;

namespace optiling {
namespace mc2_matmul_v3 {
constexpr uint32_t ALL_L2_ENABLE_BIT = 0;
constexpr uint32_t A_L2_DISABLE_BIT = 1;
constexpr uint32_t B_L2_DISABLE_BIT = 2;
constexpr uint32_t BIAS_L2_DISABLE_BIT = 3;
constexpr uint32_t C_L2_DISABLE_BIT = 4;

void Mc2L2Cache::SetL2CacheFlagBase(bool &aEnableL2Cache, bool &bEnableL2Cache) const
{
    auto &tileL2cache = tilingData_.tileL2cacheTiling;
    if (tileL2cache.mTileCntL2 > 1 || tileL2cache.nTileCntL2 > 1) {
        bEnableL2Cache = tileL2cache.mTileCntL2 > 1;
        aEnableL2Cache = tileL2cache.nTileCntL2 > 1;
        return;
    }

    auto &matmulTiling = tilingData_.matmulTiling;
    // m切多核
    if (static_cast<uint64_t>(matmulTiling.singleCoreM) < args_.mValue) {
        bEnableL2Cache = true;
    }
    // n切多核
    if (static_cast<uint64_t>(matmulTiling.singleCoreN) < args_.nValue) {
        aEnableL2Cache = true;
    }
    // 判断单核是否可以全载,此场景K可以全载
    uint64_t mCnt = MathUtil::CeilDivision(static_cast<uint64_t>(args_.mValue), matmulTiling.singleCoreM);
    uint64_t nCnt = MathUtil::CeilDivision(static_cast<uint64_t>(args_.nValue), matmulTiling.singleCoreN);
    uint64_t totalCnt = mCnt * nCnt;
    uint64_t round = MathUtil::CeilDivision(totalCnt, matmulTiling.usedCoreNum);
    bool mFullLoad = (round <= 1UL);
    bool nFullLoad = (round <= 1UL);
    if (!mFullLoad && !nFullLoad) {
        bEnableL2Cache = true;
    }
}

void Mc2L2Cache::SetL2CacheFlagSingleCoreSplitK(bool &aEnableL2Cache, bool &bEnableL2Cache) const
{
    auto &matmulTiling = tilingData_.matmulTiling;
    // m切多核
    bEnableL2Cache = static_cast<uint64_t>(matmulTiling.singleCoreM) < args_.mValue;
    // n切多核
    aEnableL2Cache = static_cast<uint64_t>(matmulTiling.singleCoreN) < args_.nValue;
    // 判断单核是否可以全载
    bool mFullLoad = (matmulTiling.singleCoreM <= matmulTiling.baseM * matmulTiling.stepM);
    bool nFullLoad = (matmulTiling.singleCoreN <= matmulTiling.baseN * matmulTiling.stepN);
    bool kFullLoad = (static_cast<uint64_t>(matmulTiling.singleCoreK) <= args_.kValue);
    // M是外循环，与算子计算逻辑强相关
    if (!mFullLoad && !nFullLoad) {
        bEnableL2Cache = true;
        aEnableL2Cache = !kFullLoad;
    }
}

void Mc2L2Cache::SetL2CacheFlagMultiCoreSplitK(bool &aEnableL2Cache, bool &bEnableL2Cache) const
{
    auto &matmulTiling = tilingData_.matmulTiling;
    // m切多核
    bEnableL2Cache = static_cast<uint64_t>(matmulTiling.singleCoreM) < args_.mValue;
    // 判断单核是否可以全载
    uint64_t mCnt = MathUtil::CeilDivision(static_cast<uint64_t>(args_.mValue), matmulTiling.singleCoreM);
    uint64_t kCnt = MathUtil::CeilDivision(static_cast<uint64_t>(args_.kValue), matmulTiling.singleCoreK);
    uint64_t totalCnt = mCnt * kCnt;
    uint64_t round = MathUtil::CeilDivision(totalCnt, matmulTiling.usedCoreNum);
    bool mFullLoad = (round <= 1UL);
    bool nFullLoad = (static_cast<uint64_t>(matmulTiling.singleCoreN) == args_.nValue);
    bool kFullLoad = (round <= 1UL);
    // N是外循环，与算子计算逻辑强相关
    if (!mFullLoad && !nFullLoad) {
        aEnableL2Cache = true;
        bEnableL2Cache = !kFullLoad;
    }
}

void Mc2L2Cache::SetL2CacheFlag(bool aEnableL2Cache, bool bEnableL2Cache, bool cEnableL2Cache, bool biasEnableL2Cache,
                                uint32_t &l2CacheFlag)
{
    if (aEnableL2Cache && bEnableL2Cache && cEnableL2Cache && biasEnableL2Cache) {
        l2CacheFlag |= (1U << ALL_L2_ENABLE_BIT);
        OP_LOGD(args_.opName, "l2CacheFlag: %u", l2CacheFlag);
        return;
    }

    if (!aEnableL2Cache) {
        l2CacheFlag |= (1U << A_L2_DISABLE_BIT);
    }

    if (!bEnableL2Cache) {
        l2CacheFlag |= (1U << B_L2_DISABLE_BIT);
    }

    if (!cEnableL2Cache) {
        l2CacheFlag |= (1U << C_L2_DISABLE_BIT);
    }

    if (!biasEnableL2Cache) {
        l2CacheFlag |= (1U << BIAS_L2_DISABLE_BIT);
    }

    OP_LOGI(args_.opName, "l2CacheFlag: %u", l2CacheFlag);
}

void Mc2L2Cache::SetL2CacheFlag(Mc2TilingEnable tilingEnable, uint64_t l2Size, uint32_t &l2CacheFlag)
{
    bool aEnableL2Cache = false;
    bool bEnableL2Cache = false;
    bool cEnableL2Cache = false;
    bool biasEnableL2Cache = true;
    auto &matmulTiling = tilingData_.matmulTiling;

    OP_LOGI(args_.opName,
            "mValue: %lu nValue: %lu kValue: %lu "
            "singleCoreM: %d singleCoreN: %d singleCoreK: %d "
            "baseM: %d baseN: %d baseK: %d "
            "stepM: %d stepN: %d stepKa: %d stepKb: %d "
            "depthA1: %d depthB1: %d tilingEnableSplitCore %d tilingEnableFullLoad %d tilingEnableFixOpti %d",
            args_.mValue, args_.nValue, args_.kValue, matmulTiling.singleCoreM, matmulTiling.singleCoreN,
            matmulTiling.singleCoreK, matmulTiling.baseM, matmulTiling.baseN, matmulTiling.baseK, matmulTiling.stepM,
            matmulTiling.stepN, matmulTiling.stepKa, matmulTiling.stepKb, matmulTiling.depthA1, matmulTiling.depthB1,
            static_cast<int32_t>(tilingEnable.tilingEnableSplitCore),
            static_cast<int32_t>(tilingEnable.tilingEnableFullLoad),
            static_cast<int32_t>(tilingEnable.tilingEnableFixOpti));

    uint64_t sizeC = args_.mValue * args_.nValue * GetSizeByDataType(args_.cType);
    cEnableL2Cache = sizeC <= l2Size;

    switch (tilingEnable.tilingEnableSplitCore) {
        case Mc2TilingEnableSplitCore::BASE:
            SetL2CacheFlagBase(aEnableL2Cache, bEnableL2Cache);
            break;
        case Mc2TilingEnableSplitCore::SINGLE_CORE_SPLIT_K:
            cEnableL2Cache = true;
            SetL2CacheFlagSingleCoreSplitK(aEnableL2Cache, bEnableL2Cache);
            break;
        case Mc2TilingEnableSplitCore::DETERMINISTIC_SPLIT_K:
            cEnableL2Cache = true;
            SetL2CacheFlagMultiCoreSplitK(aEnableL2Cache, bEnableL2Cache);
            break;
        default:
            break;
    }

    SetL2CacheFlag(aEnableL2Cache, bEnableL2Cache, cEnableL2Cache, biasEnableL2Cache, l2CacheFlag);
}

} // namespace mc2_matmul_v3
} // namespace optiling
