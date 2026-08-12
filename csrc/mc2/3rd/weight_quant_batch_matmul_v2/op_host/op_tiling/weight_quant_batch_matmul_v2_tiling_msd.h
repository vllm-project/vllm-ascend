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
 * \file weight_quant_batch_matmul_v2_tiling_msd.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_MSD_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_MSD_H

#include "weight_quant_batch_matmul_v2_tiling.h"
#include "../../op_kernel/weight_quant_batch_matmul_v2_tiling_data.h"

namespace optiling {

class Mc2WeightQuantBatchMatmulV2Msd : public Mc2WeightQuantBatchMatmulV2Tiling {
public:
    explicit Mc2WeightQuantBatchMatmulV2Msd(gert::TilingContext *context) : Mc2WeightQuantBatchMatmulV2Tiling(context)
    {
        Reset();
    }
    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }
    ~Mc2WeightQuantBatchMatmulV2Msd() override = default;

protected:
    uint32_t order_ = 2; // 展开的阶数
    uint32_t blkDim_ = 0;
    bool splitKFlag_;
    bool highPrecision_;
    uint64_t cubeBaseN_;
    std::unique_ptr<Mc2WeightQuantBatchMatmulV2MsdTilingData> tilingData_;

    void Reset();
    ge::graphStatus PostTiling() override;
    bool IsCapable() override;
    ge::graphStatus InstantiateTilingData();
    ge::graphStatus DoMSDGeneralOpTiling();
    ge::graphStatus DoOpTiling() override;
    uint64_t SplitKByKBlock(uint64_t kBlockNum) const;
    ge::graphStatus DoMSDGroupSplitKOpTiling();
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    bool CheckCacheTiling();
    bool CheckInt4MatmulTiling() const;
    bool CheckInt8MatmulTiling(uint64_t singleCoreNCalc) const;
    bool InvokeCacheTiling();
    bool GetMatMulTiling();
    void ReviseMMTiling() const;
    bool GetTilingFromCache();
    uint64_t GetInnerPreciseTilingKey() const;
};
} // namespace optiling
#endif // WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_MSD_H
