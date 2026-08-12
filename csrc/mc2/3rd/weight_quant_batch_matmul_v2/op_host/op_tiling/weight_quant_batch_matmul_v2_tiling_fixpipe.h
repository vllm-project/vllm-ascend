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
 * \file weight_quant_batch_matmul_v2_tiling_fixpipe.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_FIXPIPE_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_FIXPIPE_H

#include "weight_quant_batch_matmul_v2_tiling.h"
#include "../../op_kernel/weight_quant_batch_matmul_v2_tiling_data.h"

namespace optiling {

class Mc2WeightQuantBatchMatmulV2TilingFixpipe : public Mc2WeightQuantBatchMatmulV2Tiling {
public:
    explicit Mc2WeightQuantBatchMatmulV2TilingFixpipe(gert::TilingContext *context)
        : Mc2WeightQuantBatchMatmulV2Tiling(context)
    {
        Reset();
    };
    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }
    ~Mc2WeightQuantBatchMatmulV2TilingFixpipe() override = default;

protected:
    uint64_t aFullLoad_;
    std::unique_ptr<Mc2WeightQuantBatchMatmulV2FixpipeTilingData> tilingData_;

    void Reset()
    {
        TilingBaseClass::Reset(context_);
        aFullLoad_ = 0;

        OP_TILING_CHECK(memset_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                                 0, context_->GetRawTilingData()->GetCapacity()) != EOK,
                        OP_LOGE(opName_, "fail to memset tiling data"), return;);
    }

    ge::graphStatus PostTiling() override;

    bool IsCapable() override;

    bool CheckDtypeIsCapable() const;

    bool CheckShapeIsCapable() const;

    ge::graphStatus InstantiateTilingData();

    ge::graphStatus DoOpTiling() override;

    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;

    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
};
} // namespace optiling

#endif
