/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LIGHTNING_ATTENTION_DECODE_TILING_H
#define LIGHTNING_ATTENTION_DECODE_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/tiling_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(LightningAttentionDecodeBaseParams)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, kvCacheBatchSize);
    TILING_DATA_FIELD_DEF(uint32_t, headNum);
    TILING_DATA_FIELD_DEF(uint32_t, headDim);
    TILING_DATA_FIELD_DEF(uint32_t, actualUsedAivNum);
    TILING_DATA_FIELD_DEF(uint32_t, taskNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LightningAttentionDecodeBaseParamsOp, LightningAttentionDecodeBaseParams)

BEGIN_TILING_DATA_DEF(LightningAttentionDecodeTilingData)
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm1TilingData);
    TILING_DATA_FIELD_DEF_STRUCT(LightningAttentionDecodeBaseParams, laBaseParams);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LightningAttentionDecode, LightningAttentionDecodeTilingData)

struct LightningAttentionDecodeCompileInfo {};

class LightningAttentionDecodeTiling : public TilingBaseClass {
public:
    explicit LightningAttentionDecodeTiling(gert::TilingContext *context)
        : TilingBaseClass(context)
    {
        ascendcPlatform_.reset(new platform_ascendc::PlatformAscendC(context->GetPlatformInfo()));
    }
protected:
    bool IsCapable() override;

    ge::graphStatus GetPlatformInfo() override;

    ge::graphStatus GetShapeAttrsInfo() override;

    ge::graphStatus DoOpTiling() override;

    ge::graphStatus DoLibApiTiling() override;

    uint64_t GetTilingKey() const override;

    ge::graphStatus GetWorkspaceSize() override;

    ge::graphStatus PostTiling() override;

private:
    bool UseMatmul() const;
    bool AnalyzeDType();
    bool SetMatmulTiling();

private:
    LightningAttentionDecodeTilingData tilingData_;

    ge::DataType inputDType_;
    uint32_t aicNum_;
    uint32_t aivNum_;
    uint32_t actualUsedAivNum_;
    uint32_t taskNum_;
    uint32_t headDimBlock_;

    matmul_tiling::DataType mm1InDType_ = matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::DataType mm1OutDType_ = matmul_tiling::DataType::DT_FLOAT;
};

}

#endif // LIGHTNING_ATTENTION_DECODE_TILING_H
