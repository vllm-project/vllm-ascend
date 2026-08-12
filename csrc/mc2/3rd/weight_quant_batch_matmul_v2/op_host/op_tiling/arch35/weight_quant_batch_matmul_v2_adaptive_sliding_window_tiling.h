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
 * \file weight_quant_batch_matmul_v2_adaptive_sliding_window_tiling.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_ADAPTIVE_SLIDING_WINDOW_TILING_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_ADAPTIVE_SLIDING_WINDOW_TILING_H

#include "../weight_quant_batch_matmul_v2_tiling.h"
#include "../../../op_kernel/weight_quant_batch_matmul_v2_tiling_data.h"

namespace optiling {
namespace Mc2weight_quant_batch_matmul_v2 {

struct AdaptiveSlidingWindow {
    uint64_t baseM = 0; // 主窗口基本块大小
    uint64_t baseN = 0; // 主窗口基本块大小
    uint64_t baseK = 0;
    uint64_t mBlockCnt = 0;       // m方向基本块数量
    uint64_t nBlockCnt = 0;       // n方向基本块数量
    uint64_t totalBlockCnt = 0;   // 基本块总数
    uint64_t mTail = 0;           // m方向尾块的有效行数
    uint64_t nTail = 0;           // n方向尾块的有效列数
    uint64_t totalWinCnt = 0;     // 窗口总数，及核执行最大轮数
    uint64_t tailWinBlockCnt = 0; // 尾窗口包含的基本快数量
    uint64_t mTailTile = 1;       // 尾部窗口基本块m方向重切粒度
    uint64_t nTailTile = 1;       // 尾部窗口基本块n方向重切粒度
    bool useTailWinLogic = true;  // 是否使用尾窗口处理逻辑
};

struct BasicTiling {
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
};

class Mc2WeightQuantBatchMatmulV2TilingASW : public Mc2WeightQuantBatchMatmulV2Tiling {
public:
    explicit Mc2WeightQuantBatchMatmulV2TilingASW(gert::TilingContext *context)
        : Mc2WeightQuantBatchMatmulV2Tiling(context)
    {
        if (context->GetCompileInfo() == nullptr) {
            InitCompileInfo();
        }
    }
    ~Mc2WeightQuantBatchMatmulV2TilingASW() override = default;

protected:
    Mc2OptimizationAlgorithmSubCategory algorithmSubCategory_ = Mc2OptimizationAlgorithmSubCategory::ASW;
    Mc2Mte2Configuration mte2Config_ = Mc2Mte2Configuration::MTE2_INNER_SIZE_512_BUF_NUM_2;

    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    std::unique_ptr<WeightQuantBatchMatmulV2ASWTilingData> tilingData_;

    ge::graphStatus InstantiateTilingData();
    void AnalyseSlidingWinInfo();
    void CalcBasicBlock();
    uint64_t GetShapeWithDataType(uint64_t shapeSize, ge::DataType dtype) const;
    uint64_t GetSizeWithDataType(uint64_t shapeSize, ge::DataType dtype) const;
    void CalcTailBasicBlock();
    bool IsValidWeighNzTailSplit(uint64_t splitCnt, bool isPreSplit) const;
    uint32_t CalUsedCoreNum() const;
    uint32_t CalUsedCoreNum(uint32_t mTile, uint32_t nTile) const;
    void CalL1Tiling();
    void CalL1TilingDepth(uint64_t leftL1Size);
    void CalStepKs();
    bool CheckAntiQuantScale(uint64_t baseN, uint64_t dbL0c = 1) const;
    void SetTilingData();

private:
    // 滑窗相关信息
    AdaptiveSlidingWindow adaptiveWin_;
    BasicTiling basicTiling_;
};
} // namespace Mc2weight_quant_batch_matmul_v2
} // namespace optiling

#endif
