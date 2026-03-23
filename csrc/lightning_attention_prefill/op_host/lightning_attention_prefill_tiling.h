/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LIGHTNING_ATTENTION_TILING_H
#define LIGHTNING_ATTENTION_TILING_H

#include <memory>

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/tiling_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(LightningAttentionPrefillBaseParams)
   TILING_DATA_FIELD_DEF(uint32_t, batchSize);
   TILING_DATA_FIELD_DEF(uint32_t, headNum);
   TILING_DATA_FIELD_DEF(uint32_t, maxSeqLen);
   TILING_DATA_FIELD_DEF(uint32_t, headDim);
   TILING_DATA_FIELD_DEF(uint32_t, blockSize);
   TILING_DATA_FIELD_DEF(uint32_t, actualUsedAivNum);
   TILING_DATA_FIELD_DEF(uint32_t, eleCountPerHead);
   TILING_DATA_FIELD_DEF(uint32_t, eleCountPerBlock);
   TILING_DATA_FIELD_DEF_ARR(uint16_t, 256, blockCountPerBatch); // max batch size 256
   TILING_DATA_FIELD_DEF_ARR(uint16_t, 256, tailBlockSize);      // max batch size 256
   TILING_DATA_FIELD_DEF_ARR(uint16_t, 50, headStart);           // max aiv num: 50
   TILING_DATA_FIELD_DEF_ARR(uint16_t, 50, headEnd);;
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(LightningAttentionPrefillBaseParamsOp, LightningAttentionPrefillBaseParams)

BEGIN_TILING_DATA_DEF(LightningAttentionPrefillTilingData)
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm1TilingData);
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm2TilingData);
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm3TilingData);
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm4TilingData);
  TILING_DATA_FIELD_DEF_STRUCT(LightningAttentionPrefillBaseParams, laBaseParams);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LightningAttentionPrefill, LightningAttentionPrefillTilingData)

struct LightningAttentionPrefillCompileInfo {};

class LightningAttentionPrefillTiling : public TilingBaseClass {
public:
    explicit LightningAttentionPrefillTiling(gert::TilingContext *context)
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
    bool AnalyzeDType();
    void SetHeadStartEnd();
    bool SetMatmulTiling();
    bool SetMatmulTilingForQXK();
    bool SetMatmulTilingForPXV();
    bool SetMatmulTilingForQXKV();
    bool SetMatmulTilingForKXV();
private:
    LightningAttentionPrefillTilingData tilingData_;
    ge::DataType inputDType_;
    uint32_t blockSize_;
    uint32_t calcTypeSize_;
    uint32_t aicNum_;
    uint32_t aivNum_;
    uint32_t actualUsedAivNum_;
    uint32_t taskNum_;
    uint32_t totalBlockCount_ = 0;
    matmul_tiling::DataType mm1InDType_ = matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::DataType mm1OutDType_ = matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::DataType mm2InDType_ = matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::DataType mm2OutDType_ = matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::DataType mm3InDType_ = matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::DataType mm3OutDType_ = matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::DataType mm4InDType_ = matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::DataType mm4OutDType_ = matmul_tiling::DataType::DT_FLOAT;

    uint32_t qSBlockSize_;
    uint32_t kvSBlockSize_;
    uint32_t headDimBlock_;
};

}

#endif