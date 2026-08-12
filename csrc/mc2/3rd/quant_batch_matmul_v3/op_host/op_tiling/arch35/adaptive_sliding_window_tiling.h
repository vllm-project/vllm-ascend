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
 * \file adaptive_sliding_window_tiling.h
 * \brief
 */
#ifndef MC2_ADAPTIVE_SLIDING_WINDOW_TILING_H
#define MC2_ADAPTIVE_SLIDING_WINDOW_TILING_H
#include "util/math_util.h"
#include "../quant_batch_matmul_v3_tiling_base.h"
#include "../../../op_kernel/arch35/quant_batch_matmul_v3_tiling_data.h"

namespace optiling {

struct Mc2AdaptiveSlidingWinow {
    uint64_t baseM = 0; // 主窗口基本块大小
    uint64_t baseN = 0; // 主窗口基本块大小
    uint64_t baseK = 0;
    uint64_t mBlockCnt = 0;       // m方向基本块数量
    uint64_t nBlockCnt = 0;       // n方向基本块数量
    uint64_t tatolBlockCnt = 0;   // 基本块总数
    uint64_t mTail = 0;           // m方向尾块的有效行数
    uint64_t nTail = 0;           // n方向尾块的有效列数
    uint64_t singleWinM = 0;      // 主窗口的m边长
    uint64_t singleWinN = 0;      // 主窗口的n边长
    uint64_t totalWinCnt = 0;     // 窗口总数，及核执行最大轮数
    uint64_t mainRow = 0;         // 主窗口行数（以窗口为单位，一个窗口为一行）
    uint64_t tailWinBlockCnt = 0; // 尾窗口包含的基本快数量
    uint64_t mTailTile = 1;       // 尾部窗口基本块m方向重切粒度
    uint64_t nTailTile = 1;       // 尾部窗口基本块n方向重切粒度
    bool useTailWinLogic = true;  // 是否使用尾窗口处理逻辑
};

struct Mc2BasicRunInfoTiling {
    uint32_t usedCoreNum = 1;
    uint32_t singleCoreM = 1;
    uint32_t singleCoreN = 1;
    uint32_t singleCoreK = 1;
    uint32_t baseM = 1;
    uint32_t baseN = 1;
    uint32_t baseK = 1;
    uint32_t stepKa = 1;
    uint32_t stepKb = 1;
    uint32_t depthA1 = 1;
    uint32_t depthB1 = 1;
    uint32_t stepM = 1;
    uint32_t stepN = 1;
    uint32_t iterateOrder = 0;
    uint32_t dbL0c = 1;
    uint32_t ubCalcN = 0;
    uint32_t ubCalcM = 0;
    uint32_t scaleFactorA = 0;
    uint32_t scaleFactorB = 0;
};

enum class Mc2BiasMode : uint32_t {
    EXCLUEDE_FROM_TEMPLATE = 0,
    CUBE_BIAS_BF16_TEMPLATE = 1,
    CUBE_BIAS_FP16_TEMPLATE = 2
};

enum class Mc2QMMKernelType : uint32_t {
    NO_VEC_EPILOGUE_WITH_MMAPI = 0,
    NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI = 1,
    VEC_EPILOGUE_WITH_MMAPI = 2,
    VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI = 3,
    VEC_EPILOGUE_WITH_CUSTOM_MM = 4
};

class Mc2AdaptiveSlidingWindowTiling : public Mc2QuantBatchMatmulV3TilingBase {
public:
    explicit Mc2AdaptiveSlidingWindowTiling(gert::TilingContext *context);
    Mc2AdaptiveSlidingWindowTiling(gert::TilingContext *context,
                                   DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams *out);
    ~Mc2AdaptiveSlidingWindowTiling() override = default;

    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData，mc2使用的直接接口
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

    uint64_t GetKernelType() const;

protected:
    ge::graphStatus CalcUbTiling() override;
    bool CheckDtype() const override;
    bool CheckShape(const std::vector<gert::Shape *> &mandatoryShape, const gert::StorageShape *biasShape,
                    const gert::StorageShape *pertokenShape, const std::vector<int64_t> &dimValueOfMKN) const override;
    void CalL1Tiling();
    bool CheckBiasAndScale(uint64_t baseN, uint64_t dbL0c) const;
    bool AnalyseSlidingWinInfo();
    void AdjustBasicBlock();
    void SetBf16Compat();
    void SetTilingData();
    uint32_t CalUsedCoreNum();
    bool CalcBasicBlock();
    void CalcTailBasicBlock();
    void CalcTailBasicBlockAfullLoad();
    uint32_t CalUsedCoreNum(uint32_t mTile, uint32_t nTile);
    uint64_t GetDepthA1B1(uint64_t leftSize, uint64_t perDepthSize, uint64_t depthInit);
    uint64_t GetDepthB1AfullLoad(uint64_t leftSize);
    uint64_t GetScaleFactorBAfullLoad(uint64_t leftSize);
    void CalScaleFactors(uint64_t baseASize, uint64_t baseBSize, uint64_t baseScaleASize, uint64_t baseScaleBSize,
                         uint64_t leftL1Size);
    void CalStepKs();
    bool IsMxKOdd() const;
    bool IsMxBackwardTrans() const;
    void IsAFullLoad();
    void CalL1TilingDepthAfullload(uint64_t leftL1Size);
    void CalL1TilingDepthANotfullload(uint64_t leftL1Size);
    uint64_t GetBiasMode() const;
    bool IsInValidPerblockTailSplit(uint64_t splitCnt) const;
    bool IsInValidWeighNzTailSplit(uint64_t splitCnt, bool isPreSplit) const;
    void Reset();

    DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams tilingDataSelf_;
    DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &tilingData_;

private:
    Mc2AdaptiveSlidingWinow adaptiveWin_;
    Mc2BasicRunInfoTiling basicTiling_;
    bool isAFullLoad_ = false;
    bool isBf16Mix_ = false;
};
} // namespace optiling
#endif // ADAPTIVE_SLIDING_WINDOW_TILING_H
