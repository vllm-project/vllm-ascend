/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file fused_infer_attention_score_tiling_arch38.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FUSEDINFERATTENTIONSCORE_ARCH38_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FUSEDINFERATTENTIONSCORE_ARCH38_H_
#include "register/tilingdata_base.h"
#include "../../../common/op_host/fia_tiling_base.h"

namespace optiling {
ge::graphStatus TilingFusedInferAttentionScoreArch38(gert::TilingContext *context);
class FusedInferAttentionScoreTilingArch38 : public FiaTilingBase {
public:
    explicit FusedInferAttentionScoreTilingArch38(gert::TilingContext *context) : FiaTilingBase(context) {}
    ~FusedInferAttentionScoreTilingArch38() override = default;

protected:
    void InitTilingInfo(TilingInfo *tilingInfo) override {}
    bool IsCapable() override
    {
        return true;
    }
    ge::graphStatus DoOpTiling() override;
};
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_FUSEDINFERATTENTIONSCORE_ARCH38_H_
