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
 * \file quant_batch_matmul_v3_basic_tiling.h
 * \brief
 */
#ifndef MC2_QUANT_BATCH_MATMUL_V3_BASIC_TILING_H
#define MC2_QUANT_BATCH_MATMUL_V3_BASIC_TILING_H
#include "quant_batch_matmul_v3_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

struct BasicTiling {
    uint64_t usedCoreNum = 1;
    uint64_t singleCoreK = 1;
    uint64_t baseM = 1;
    uint64_t baseN = 1;
    uint64_t baseK = 1;
    uint64_t stepKa = 1;
    uint64_t stepKb = 1;
    uint64_t depthA1 = 1;
    uint64_t depthB1 = 1;
    uint64_t stepM = 1;
    uint64_t stepN = 1;
    uint64_t iterateOrder = 0;
    uint64_t dbL0c = 1;
    uint64_t isMclash = 0;
    uint64_t isNclash = 0;
    uint64_t mTileCntl2 = 0;
    uint64_t nTileCntl2 = 0;
    uint64_t mTileBlock = 0;
    uint64_t nTileBlock = 0;
    uint64_t calOrder = 0;
};

enum class BasicTilingMode : uint32_t {
    BASIC_TILING_MODE = 0,
    L2_CACHE_TILING_MODE = 1
};

class Mc2QuantBatchMatmulV3BasicTiling : public Mc2QuantBatchMatmulV3Tiling {
public:
    explicit Mc2QuantBatchMatmulV3BasicTiling(gert::TilingContext *contextIn)
        : Mc2QuantBatchMatmulV3Tiling(contextIn), context(contextIn)
    {
    }
    Mc2QuantBatchMatmulV3BasicTiling(gert::TilingContext *contextIn, Mc2QuantBatchMatmulV3TilingData *out)
        : Mc2QuantBatchMatmulV3Tiling(contextIn, out), context(contextIn)
    {
    }
    ~Mc2QuantBatchMatmulV3BasicTiling() override = default;

    BasicTiling basicTiling_;
    Mc2QuantBatchMatmulV3Trans trans_ = Mc2QuantBatchMatmulV3Trans::NO_TRANS;

protected:
    bool IsCapable() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData，mc2使用的直接接口
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;

    bool GetUbDequantExtreSpace() override;
    ge::graphStatus CalcUbTiling() override;

    void PrintBasicTiling() const;
    // // basic tiling
    bool CheckIfUseBasicInMix(uint64_t m, uint64_t n, uint64_t k) const;
    bool CheckNotFullLoadForMutliIterate(uint64_t m, uint64_t n, uint64_t k) const;
    void GetSocVersion();
    bool IsPertokenBasicSwitchCondition() const;
    bool CheckInBasicBenefitsRange(uint64_t m, uint64_t n, uint64_t k) const;
    bool CheckMNSmallShape(uint64_t m, uint64_t n) const;
    bool CheckUseBasicTiling();
    bool DoBasicTiling();
    bool CalcL0Tiling();
    bool CalcL1Tiling();
    void DoL2CacheTiling();
    void SetMatmulTilingFromBasicTiling();
    void CalcUbBasicTiling();
    void CalcBasicTilingWorkspaceSize();
    std::tuple<uint64_t, uint64_t, uint64_t, uint64_t> CalcCoreDistribution(uint64_t mCnt, uint64_t nCnt,
                                                                            uint64_t calcOrder, uint64_t round,
                                                                            uint64_t usedCoreNum) const; // 计算核的排布
    bool SetBase(const std::vector<uint64_t> &mBases, const std::vector<uint64_t> &nBases);
    void CompareBase(std::vector<uint64_t> &basicMetrics, uint64_t baseM, uint64_t baseN);
    bool CheckCalcAndMemRatio(uint64_t baseM, uint64_t baseN) const;
    bool CheckL2Load(std::vector<uint64_t> &basicMetrics, uint64_t coreClash, uint64_t firstL2Load) const;
    bool CheckMTE1(uint64_t baseM, uint64_t baseN) const;
    bool CheckBiasAndScale(uint64_t baseN, uint64_t dbL0c = 1) const;
    uint64_t GetMaxBaseN() const;
    bool CheckDbL0c() const;
    bool GetBaseK(uint64_t baseM, uint64_t baseN);
    void CalcClashAndFirstL2Load(uint64_t &coreClash, uint64_t &firstL2Load, uint64_t mCnt, uint64_t nCnt,
                                 uint64_t round) const;
    void InitBasicMetrics(std::vector<uint64_t> &basicMetrics);
    bool IsMNSmallForMultiCores(uint64_t coreNum) const;
    void ProcessMNSmallShape(uint64_t baseM, uint64_t baseN, uint64_t coreNum);
    bool IsNetBNZTrans() const;
    bool IsNetBNZDecode() const;
    bool CanProcessNetDecode() const;
    void ModifyNZBase(uint64_t &baseN, uint64_t coreNum) const;
    bool ProcessBNZDecode();
    int8_t CheckLoadAndCalcSize(uint64_t baseM, uint64_t baseN, uint64_t bestRound, uint64_t round,
                                uint64_t &bestLoadSize) const;
    bool CheckTrans(bool isCheckTrans, bool isSmallerLoadSize, uint64_t base = 256) const; // 256: ND2NZ aligned
    void Int4LowerAxisAlign(uint64_t &baseM, uint64_t &baseN) const;
    void ModifyBase(uint64_t &baseM, uint64_t &baseN) const;
    bool GetStepK(uint64_t &stepKa, uint64_t &stepKb) const;
    void ModifyStepKForKOuter(uint64_t &stepKa, uint64_t &stepKb) const;
    void CorrectStepK(uint64_t &bigStepK, uint64_t &smallStepK, uint64_t minStepK) const;
    uint64_t CalcL1SizeForBiasAndScale();
    uint64_t GetTotalCnt(uint64_t baseM, uint64_t baseN) const;
    bool IsTilingDataInvalid() const;
    void ResetBase(const uint64_t l0CSize);
    uint64_t GetTotalSize(uint64_t m, uint64_t k, uint64_t n) const;
    void CalcTileCnt(uint64_t outOriShape, uint64_t innerOriShape, uint64_t outBase, uint64_t innerBase,
                     std::vector<std::tuple<uint64_t, uint64_t>> &tileCnt) const;
    bool IsTileClash(uint64_t outSplit, uint64_t innerSplit, std::tuple<uint64_t, uint64_t> &tileClash,
                     const std::tuple<uint64_t, uint64_t, uint64_t> &params) const;
    uint64_t GetCalcOrder(uint64_t mCnt, uint64_t nCnt, uint64_t mSize, uint64_t nSize, uint64_t usedCoreNum) const;
    bool CheckTileTail(uint64_t outTail, uint64_t innerTail, uint64_t outL2SplitTmp, uint64_t innerL2SplitTmp) const;
    bool CheckTileClash(const std::tuple<uint64_t, uint64_t, uint64_t, uint64_t> &tileInfo,
                        const std::tuple<uint64_t, uint64_t, uint64_t> &params,
                        std::vector<std::tuple<uint64_t, uint64_t>> &tileClash) const;
    uint64_t CalcTile(uint64_t &outTile, uint64_t &innerTile, uint64_t &outL2Split, uint64_t &innerL2Split,
                      const std::tuple<uint64_t, uint64_t, double> &params) const;
    void DivisibleCoreLayout(uint64_t mCnt, uint64_t nCnt, uint64_t &calcOrder, uint64_t round) const;
    void DetermineCalcOrder();
    void SetCalcOrderinMNClashCase(uint64_t mTotalCnt, uint64_t nTotalCnt);

private:
    gert::TilingContext *context;
    platform_ascendc::SocVersion socVersion;
    mutable bool isAicAiv1_2 = false;
};

} // namespace optiling
#endif // QUANT_BATCH_MATMUL_V3_BASIC_TILING_H
