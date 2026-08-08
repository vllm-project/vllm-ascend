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
 * \file fia_tiling_empty_tensor.h
 * \brief
 */
#ifndef FIA_TILING_EMPTY_TENSOR_H
#define FIA_TILING_EMPTY_TENSOR_H

#include "register/tilingdata_base.h"
#include "exe_graph/runtime/tiling_context.h"
#include "../fia_tiling_base.h"
#include "../fia_tiling_info.h"
#include "../../../fused_infer_attention_score/op_host/fused_infer_attention_score_tiling.h"

namespace optiling {

class FiaTilingEmptyTensor : public FiaTilingBase {
public:
    explicit FiaTilingEmptyTensor(gert::TilingContext *context) : FiaTilingBase(context) {}
    ~FiaTilingEmptyTensor() override = default;

protected:
    void InitTilingInfo(TilingInfo *tilingInfo) override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;

private:
    ge::graphStatus GetPlatformInfo();
    void GenTilingKey();

    void InitParams();

    void FillTiling();

    void CalcWorkspaceSize();
    void CalcNumBlocks(uint32_t coreNum);

    uint32_t usedCoreNum_ = 0;

    // platform info
    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;

    // set info to context
    FusedInferAttentionScoreEmptyTensorTilingData tilingData_;
    uint32_t numBlocks_{0};
    uint64_t workspaceSize_{0};
    uint64_t tilingKey_{0};

    // Tiling Info
    FiaTilingInfo *fiaInfo_ = nullptr;
};

} // namespace optiling
#endif // FIA_TILING_EMPTY_TENSOR_H