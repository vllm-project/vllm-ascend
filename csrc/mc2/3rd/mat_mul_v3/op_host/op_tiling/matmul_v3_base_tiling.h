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
 * \file matmul_v3_base_tiling.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_BASE_TILING_H__
#define __OP_HOST_MATMUL_V3_BASE_TILING_H__

#include "matmul_v3_tiling.h"
#include "matmul_v3_common.h"
#include "matmul_v3_compile_info.h"
#include "op_host/tiling_base.h"
#include "matmul_v3_tuning.h"
#include "ops_legacy/op_tiling/op_cache_tiling.h"
namespace optiling {
namespace mc2_matmul_v3 {
class Mc2MatmulV3BaseTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
public:
    explicit Mc2MatmulV3BaseTiling(gert::TilingContext *context)
        : TilingBaseClass(context), tilingData_(tilingDataSelf_)
    {
    }
    Mc2MatmulV3BaseTiling(gert::TilingContext *context, Mc2MatmulV3TilingData *tilingData,
                          Mc2TilingCalcSelect tilingSelect = Mc2TilingCalcSelect::BASE)
        : TilingBaseClass(context), tilingData_(*tilingData)
    {
        InitCompileInfo();
        tilingSelect_ = tilingSelect;
    }
    ~Mc2MatmulV3BaseTiling() override
    {
    }

protected:
    bool IsCapable() override
    {
        return true;
    };
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

    void InitCompileInfo();

protected:
    virtual ge::graphStatus CheckArgs();
    virtual ge::graphStatus GetArgs();
    virtual ge::graphStatus GetMoreArgs();
    ge::graphStatus CheckDimsAligned310P();
    ge::graphStatus InitTilingData();
    void SetRunInfo();
    void SetNd2NzInfo();
    void SetParamsV310();
    bool GetTilingFromRepo();
    bool GetTilingInputArgs(std::shared_ptr<void> &inputArgs, size_t &size);
    void DebugLog(const std::shared_ptr<tuningtiling::Mc2MatMulV3InputArgs> &inputArgs);
    bool TranslateAoeTiling(tuningtiling::TuningTilingDefPtr &tuningTiling);
    ge::graphStatus SelectNZTiling();
    void DoTilingKey();
    void DoBasicTiling();
    void FormulaicBaseBlockTiling();
    void FormulaicTilingNoTrans();
    void CalL1TilingV200();
    void CalL1Tiling();
    void UpdateL1TilingStepK(uint64_t &stepK);
    bool IsPowerOfTwo(uint64_t x) const;
    void OptimizeBasicKernelStepK();
    void OptimizeLoadBalanceBasicKernel();
    bool DoL2CacheTiling();
    bool DoL2CacheTiling310P();
    virtual void UpdateNd2nzFlag() {};
    void DoNd2NzVectorTiling();
    bool DoBL1FullloadWithFixpipeTiling();
    void CalcNd2NzTiling(uint64_t dtypeSize, uint64_t nValue, uint64_t dValue, uint64_t &baseN, uint64_t &baseD);
    bool CheckUbOverFlow(uint64_t nAligned16, uint64_t nValue, const uint64_t &baseN, const uint64_t &baseD,
                         uint64_t dtypeSize);
    bool CheckBTSize(uint64_t baseN);
    template <Mc2CalcType T>
    void CalcBase(const std::vector<std::vector<uint64_t>> &baseMNK);
    void BalanceBaseBlockTiling();
    void CalBaseMBaseN(uint64_t &baseM, uint64_t &baseN, uint64_t maxBaseM = BASIC_BLOCK_SIZE_128,
                       uint64_t maxBaseN = BASIC_BLOCK_SIZE_256);
    void SetBaseBlockTiling();
    void DoSmallShapeTiling();
    void DoSelectTiling();
    bool IsSupportSingleCoreSplitSmallK(uint64_t xDim, uint64_t yDim) const;
    bool IsSupportSingleCoreSplitK() const;
    bool DoSingleCoreSplitKTiling();
    bool DoAL1FullLoadTiling();
    bool ShouldUseDeterministicMultiCoreSplitKwithSmallMN() const;
    bool DoBL1FullLoadTiling();
    bool DoBL1FullLoadTilingBase();
    bool DoBL1CoreSplitFullLoadTiling();
    bool SupportMultiSplitK() const;
    void GetMoreMultiCoreSplitKArgs();
    bool DoDeterministicMultiCoreSplitKTiling();
    void OptCoreNumsDeterministicMultiCoreSplitK();
    bool IsNkOrder();
    void CalTileFactor(uint64_t &nTile);
    void SetBasicBlockOfMK33(Mc2MatmulV3RunInfo &runInfo);
    void SetBasicBlockOfNK33(Mc2MatmulV3RunInfo &runInfo);
    bool NeedSolveFixBound();
    bool CheckAoeTilingEnable(uint32_t aoeTilingEnable, const std::string &opName);
    void SetBasicBlockOf24(Mc2MatmulV3RunInfo &runInfo, const uint64_t &mTile, const uint64_t &nTile) const;
    Mc2MixNd2NzType GetMixNd2nzType();
    void IsGmToL1ByShape();
    void InitL2SplitParams(Mc2MatmulV3L2SplitParams &l2SplitParams) const;
    bool IsTailSmall(Mc2MatmulV3L2SplitParams &l2SplitParams, uint64_t outL2Split, uint64_t innerL2Split,
                     uint64_t innerMaxConflict) const;
    bool CalcTile(uint64_t &outTile, uint64_t &innerTile, uint64_t &outL2Split, uint64_t &innerL2Split,
                  const bool isInnerBad) const;
    uint64_t GetTotalSize(uint64_t m, uint64_t k, uint64_t n, uint64_t aDtype, uint64_t bDtype) const;
    void DoIncreTiling();
    void GetV2Tiling();
    bool CheckMMTilingDataIsVaild();
    void FormulateBasicBlockDavid();
    void CalcTailBasicBlock();
    bool CheckSingleTilingOk(Mc2MatmulV3RunInfo &tmpRunInfo);
    bool CheckSingleCoreSplitKEdgeCases(const Mc2MatmulV3RunInfo &tmpRunInfo, bool isNKM, bool isMKNsmallK,
                                        bool isNKMsmallK);
    bool IsOnTheWay(ge::Format matFormat, uint64_t innerSize, uint64_t dtypeSize,
                    const std::vector<uint64_t> supportNd2nzList) const;
    bool NeedNd2NzVnchw(uint64_t outerSize, uint64_t innerSize, bool supportNd2NzOnTheWay, uint64_t dtypeSize,
                        ge::Format matFormat) const;

private:
    Mc2MatmulV3TilingData tilingDataSelf_{};

protected:
    Mc2MatmulV3CompileInfo compileInfo_;
    Mc2MatmulV3Args args_;
    matmul_tiling::MultiCoreMatmulTiling mm_;
    Mc2MatmulV3TilingData &tilingData_;
    Mc2MatmulV3RunInfo runInfo_;
    uint64_t aDtypeSize_{1};
    uint64_t bDtypeSize_{1};
    uint64_t cDtypeSize_{1};
    Mc2MatmulV3Trans trans_ = Mc2MatmulV3Trans::NO_TRANS;
    Mc2TilingEnable tilingEnable_;
    bool m256Align_{false};
    bool kA256Align_{false};
    bool kB256Align_{false};
    bool n256Align_{false};
    bool enableCache_{true};
    std::vector<std::vector<uint64_t>> calcMBasic_;
    std::vector<std::vector<uint64_t>> calcMNBasic_;
    uint64_t l2TileLength_{0};
    uint64_t basicBlockBaseM_{BASIC_BLOCK_SIZE_256};
    uint32_t l2CacheFlag_{0};
    bool compileInfoInit_{false};
    Mc2TilingCalcSelect tilingSelect_ = Mc2TilingCalcSelect::ALL;
};
} // namespace mc2_matmul_v3
} // namespace optiling
#endif // __OP_HOST_MATMUL_V3_BASE_TILING_H__
