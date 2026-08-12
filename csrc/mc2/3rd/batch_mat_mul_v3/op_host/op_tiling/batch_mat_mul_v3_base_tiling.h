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
 * \file batch_mat_mul_v3_base_tiling.h
 * \brief
 */
#ifndef __OP_HOST_BATCH_MAT_MUL_V3_BASE_TILING_H__
#define __OP_HOST_BATCH_MAT_MUL_V3_BASE_TILING_H__

#include "batch_mat_mul_v3_tiling.h"
#include "op_host/tiling_base.h"
#include "mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.h"
namespace optiling {
namespace Mc2batch_mat_mul_v3 {
struct Mc2BatchShapeInfo {
    uint64_t batchA = 1;
    uint64_t batchA0 = 1;
    uint64_t batchA1 = 1;
    uint64_t batchA2 = 1;
    uint64_t batchA3 = 1;
    uint64_t batchB = 1;
    uint64_t batchB0 = 1;
    uint64_t batchB1 = 1;
    uint64_t batchB2 = 1;
    uint64_t batchB3 = 1;
    uint64_t batchC = 1;
    uint64_t batchC0 = 1;
    uint64_t batchC1 = 1;
    uint64_t batchC2 = 1;
    uint64_t batchC3 = 1;
    bool biasWithBatch = false;
};

enum class Mc2TilingCalcSelect // 选择不同的计算Tiling的方法
{
    ALL = 0,
    COMMON = 1,
    MULTIBATCH = 2,
};

enum class Mc2TilingEnableMultiBatchL1FullLoad : int32_t // 互斥flag, 对应不同全载模板选择
{
    IS_FALSE = 0,
    IS_TRUE = 1,
    MAX = 10 // 模板类别不能超过10个
};

enum class Mc2TilingEnableMultiBatch : int32_t // 互斥flag, 对应不同全载模板选择
{
    IS_FALSE = 0,
    IS_TRUE = 1,
    MAX = 10 // 模板类别不能超过10个
};

enum class Mc2TilingEnableLoadMode : int32_t // 互斥flag, 对应不同全载模板选择
{
    BASE = 0,
    AL1_FULL_LOAD = 1,
    BL1_FULL_LOAD = 2,
    MAX = 10 // 模板类别不能超过10个
};

enum class Mc2TilingEnableMultiBatchOut : int32_t // 互斥flag, 对应不同全载模板选择
{
    IS_FALSE = 0,
    IS_TRUE = 1,
    MAX = 10 // 模板类别不能超过10个
};

enum class Mc2TilingEnableMixNd2Nz : int32_t // 互斥flag, 对应不同全载模板选择
{
    IS_TRUE = 0,
    IS_FALSE = 1,
    MAX = 10 // 模板类别不能超过10个
};

struct Mc2TilingEnable {
    Mc2TilingEnableMultiBatchL1FullLoad tilingEnableMultiBatchL1FullLoad =
        Mc2TilingEnableMultiBatchL1FullLoad::IS_FALSE;
    Mc2TilingEnableMultiBatch tilingEnableMultiBatch = Mc2TilingEnableMultiBatch::IS_FALSE;
    Mc2TilingEnableLoadMode tilingEnableLoadMode = Mc2TilingEnableLoadMode::BASE;
    Mc2TilingEnableMultiBatchOut tilingEnableMultiBatchOut = Mc2TilingEnableMultiBatchOut::IS_FALSE;
    Mc2TilingEnableMixNd2Nz tilingEnableMixNd2Nz = Mc2TilingEnableMixNd2Nz::IS_FALSE;
};

class Mc2BatchMatmulV3BaseTiling : public mc2_matmul_v3::Mc2MatmulV3BaseTiling {
public:
public:
    explicit Mc2BatchMatmulV3BaseTiling(gert::TilingContext *context)
        : Mc2MatmulV3BaseTiling(context, &bmmTilingDataSelf_.matmulTiling), bmmTilingData_(bmmTilingDataSelf_)
    {
    }

    Mc2BatchMatmulV3BaseTiling(gert::TilingContext *context, Mc2BatchMatmulTilingData &bmmTilingData,
                               Mc2TilingCalcSelect tilingSelect = Mc2TilingCalcSelect::COMMON)
        : Mc2MatmulV3BaseTiling(context, &bmmTilingData.matmulTiling), bmmTilingData_(bmmTilingData)
    {
        tilingSelect_ = tilingSelect;
    }

    ~Mc2BatchMatmulV3BaseTiling() override
    {
    }

protected:
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override; // 4
    // 6、保存Tiling数据
    ge::graphStatus PostTiling() override; // 6
    // 7、计算TilingKey
    uint64_t GetTilingKey() const override; // 7

    bool GetBatchInfo();
    bool GetBiasWithBatchInfo();
    void MergeBatchAndMAxis();
    bool CheckBMMTilingDataIsVaild() const;
    void DoTilingKeyCustom();
    void DoUnAlignCommonTiling();
    void DoMultiBatchTiling();
    bool DoMultiBatchOutTiling();
    void DoMultiBatchTilingImpl();
    void DoMultiBatchL1FullLoadTilingImpl();
    void DoCommonTiling();
    void DoL1FullLoadTiling();
    bool IsMultiBatchAL1FullLoad();
    void DoMultiBatchL1FullLoadTiling();
    ge::graphStatus CheckDimsAligned();
    void UpdateMultiBatchNd2nz();
    void CalculateNd2nzWorkspaceSize();
    void CheckandSetDiagonalConflict(uint64_t mCnt, uint64_t nCnt, uint64_t batch, uint64_t usedCoreNum,
                                     uint64_t transConflict, uint64_t newMcnt);
    void DoL2CacheAndCalOrderTiling();

protected:
    Mc2BatchShapeInfo batchInfo_;
    Mc2BatchMatmulTilingData &bmmTilingData_;
    Mc2TilingCalcSelect tilingSelect_ = Mc2TilingCalcSelect::ALL;

private:
    Mc2BatchMatmulTilingData bmmTilingDataSelf_{};
    uint64_t aBatchDimAll_{1};
    uint64_t bBatchDimAll_{1};
    uint64_t cBatchDimAll_{1};

protected:
    Mc2TilingEnable tilingEnable_;
};
} // namespace Mc2batch_mat_mul_v3
} // namespace optiling
#endif // __OP_HOST_BATCH_MAT_MUL_V3_BASE_TILING_H__
