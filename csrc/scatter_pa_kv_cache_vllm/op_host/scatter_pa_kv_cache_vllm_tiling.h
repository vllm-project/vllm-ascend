/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file scatter_pa_kv_cache_vllm_tiling.h
 * \brief
 */

#ifndef SCATTER_PA_KV_CACHE_VLLM_TILING_H_
#define SCATTER_PA_KV_CACHE_VLLM_TILING_H_
#include <cstdint>
#include <cstddef>
#include <vector>
#include <sys/types.h>
#include <string>
#include <utility>
#include <exe_graph/runtime/tiling_context.h>
#include <graph/utils/type_utils.h>
#include <register/tilingdata_base.h>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../moe/causal_conv1d/tiling_base/tiling_base.h"

using namespace Ops::Transformer::OpTiling;
namespace optiling {

BEGIN_TILING_DATA_DEF(ScatterPaKvCacheVllmTilingData)
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);
TILING_DATA_FIELD_DEF(int64_t, tailBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, kHandleNumPerCore);
TILING_DATA_FIELD_DEF(int64_t, vHandleNumPerCore);
TILING_DATA_FIELD_DEF(int64_t, kTailHandleNum);
TILING_DATA_FIELD_DEF(int64_t, vTailHandleNum);
TILING_DATA_FIELD_DEF(int64_t, kLoopNum);
TILING_DATA_FIELD_DEF(int64_t, vLoopNum);
TILING_DATA_FIELD_DEF(int64_t, kHandleNumPerLoop);
TILING_DATA_FIELD_DEF(int64_t, vHandleNumPerLoop);
TILING_DATA_FIELD_DEF(int64_t, keyStride0);
TILING_DATA_FIELD_DEF(int64_t, keyStride1);
TILING_DATA_FIELD_DEF(int64_t, keyStride2);
TILING_DATA_FIELD_DEF(int64_t, valueStride0);
TILING_DATA_FIELD_DEF(int64_t, valueStride1);
TILING_DATA_FIELD_DEF(int64_t, valueStride2);
TILING_DATA_FIELD_DEF(int64_t, kHeadSize);
TILING_DATA_FIELD_DEF(int64_t, vHeadSize);
TILING_DATA_FIELD_DEF(int64_t, batch);
TILING_DATA_FIELD_DEF(int64_t, numBlocks);
TILING_DATA_FIELD_DEF(int64_t, blockSize);
TILING_DATA_FIELD_DEF(int64_t, seqLen);
TILING_DATA_FIELD_DEF(int64_t, numHead);
TILING_DATA_FIELD_DEF(int64_t, numTokens);
TILING_DATA_FIELD_DEF(int64_t, ubSize);
TILING_DATA_FIELD_DEF(int64_t, kStride);
TILING_DATA_FIELD_DEF(int64_t, vStride);
TILING_DATA_FIELD_DEF(int64_t, kOffset);
TILING_DATA_FIELD_DEF(int64_t, vOffset);
TILING_DATA_FIELD_DEF(int64_t, kCacheBlockStride);
TILING_DATA_FIELD_DEF(int64_t, vCacheBlockStride);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterPaKvCacheVllm, ScatterPaKvCacheVllmTilingData)


struct ScatterPaKvCacheVllmCompileInfo {
    int64_t totalCoreNum{0};
    int64_t ubSize{0};
};

class ScatterPaKvCacheVllmTiling : public TilingBaseClass {
public:
    explicit ScatterPaKvCacheVllmTiling(gert::TilingContext *context, int64_t inOutMode)
        : TilingBaseClass(context), inOutMode_(inOutMode)
    {
    }

protected:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus TemplateNormal();
    ge::graphStatus TemplateRope();
    ge::graphStatus TemplateAlibi();
    ge::graphStatus GetIndexDtype();
    ge::graphStatus GetInputDtype();
    ge::graphStatus CheckDimValid();
    ge::graphStatus CheckNormal();
    ge::graphStatus CheckRope();
    ge::graphStatus CheckAlibi();
    void SetInputPos();
    void GetCommonTilingInfo();
    ge::graphStatus CheckSlotMappingShape(int64_t requiredDimNum);
    int64_t RoundUp(int64_t x, int64_t dtypeSize);
    void GenTilingKey();

private:
    platform_ascendc::SocVersion socVersion_;
    int64_t ubSize_ = 0;
    int64_t totalCoreNum_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t blockFactor_;
    int64_t tailBlockFactor_ = 0;
    int64_t kHandleNumPerCore_ = 0;
    int64_t vHandleNumPerCore_ = 0;
    int64_t kLoopNum_ = 0;
    int64_t vLoopNum_ = 0;
    int64_t kHandleNumPerLoop_ = 0;
    int64_t vHandleNumPerLoop_ = 0;
    int64_t kTailHandleNum_ = 0;
    int64_t vTailHandleNum_ = 0;
    int64_t numTokens_ = 0;
    int64_t dtypeByteSize_ = 0;
    int64_t keyStride0_;
    int64_t keyStride1_;
    int64_t keyStride2_;
    int64_t valueStride0_;
    int64_t valueStride1_;
    int64_t valueStride2_;
    int64_t kHeadSize_;
    int64_t vHeadSize_;
    int64_t batch_;
    int64_t numHead_;
    int64_t numBlocks_;
    int64_t blockSize_;
    int64_t seqLen_;
    uint64_t tilingKey_;
    int64_t templateType_ = 0;   // 1: normal, 2: rope, 3: alibi
    int64_t indexDtypeSize_ = 0; // int32 -> 4, int64 -> 8
    int64_t isFullyLoad_ = 0;    // 0: not fully load, 1: fully load
    int64_t inOutMode_;          // 1: single input and output，2: dual input and output
    int64_t inputKey_;
    int64_t inputKeyCacheIn_;
    int64_t inputSlotMapping_;
    int64_t inputValue_;
    int64_t inputValueCacheIn_;
    int64_t inputCompressLens_;
    int64_t inputCompressSeqOffset_;
    int64_t inputSeqLens_;

    gert::Shape inputKeyShape_;
    gert::Shape inputKeyCacheInShape_;
    gert::Shape slotMappingShape_;
    gert::Shape inputValueShape_;
    gert::Shape inputValueCacheInShape_;
    gert::Shape compressLensShape_;
    gert::Shape compressSeqOffsetShape_;
    gert::Shape seqLensShape_;

    ge::DataType inputDtype_;
    ScatterPaKvCacheVllmTilingData tilingData_;
};

struct ScatterPaKvCacheVllmMembaseParams {
    platform_ascendc::SocVersion socVersion;
    int64_t numTokens{0};
    int64_t numHead{0};
    int64_t kHeadSize{0};
    int64_t vHeadSize{0};
    int64_t blockSize{0};
    int64_t typeByteK{0};
    int64_t typeByteV{0};
    int64_t typeByteSlot{0};
    int64_t numBatchs{0};
    int64_t blockFactor{0};
    int64_t tailBlockFactor{0};
    int64_t kStride{0};
    int64_t vStride{0};
    int64_t kOffset{0};
    int64_t vOffset{0};
    int64_t kCacheBlockStride{0};
    int64_t vCacheBlockStride{0};

    int64_t tilingKey{0};
    uint64_t workspaceSize{0};

    uint64_t sysWorkspaceSize{0};
    int64_t usedCoreNum{0};
    uint64_t ubSize{0};

    int64_t templateType{0}; // 1: normal, 2: nz, 3: alibi, 4: rope, 5:siso, 6:omni, 7:normal_nct, 8:siso_nct
};

class ScatterPaKvCacheVllmMembaseTiling : public TilingBaseClass {
public:
    explicit ScatterPaKvCacheVllmMembaseTiling(gert::TilingContext *context) : TilingBaseClass(context)
    {
    }

protected:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;

private:
    ge::graphStatus GetInputDtypeSiso();
    ge::graphStatus GetInputDtypeKv();
    ge::graphStatus GetInputDtype();
    ge::graphStatus CheckInputDimNumSiso();
    ge::graphStatus CheckInputDimNumNorm();
    ge::graphStatus CheckInputDimNumNz();
    ge::graphStatus CheckInputDimNumCompress();
    ge::graphStatus CheckInputDimNumNHSD();
    ge::graphStatus CheckInputDimNum();
    ge::graphStatus CheckInputShapeSiso();
    ge::graphStatus CheckInputShapeNorm();
    ge::graphStatus CheckInputShapeNz();
    ge::graphStatus CheckInputShapeCompress();
    ge::graphStatus CheckInputShape();
    ge::graphStatus GetTemplateType();
    ge::graphStatus DoNoCompressOpTiling();
    ge::graphStatus DoCompressAlibiOpTiling();
    ge::graphStatus DoCompressRopeAndOmniOpTiling();
    ge::graphStatus CheckInputShapeNHSD();
    ge::graphStatus DoNHSDOpTiling();
private:
    ScatterPaKvCacheVllmMembaseParams params_;
    ScatterPaKvCacheVllmTilingData tilingData_;
    gert::Shape inputKeyShape_;
    gert::Shape inputKeyCacheInShape_;
    gert::Shape slotMappingShape_;
    gert::Shape inputValueShape_;
    gert::Shape inputValueCacheInShape_;
    gert::Shape compressLensShape_;
    gert::Shape compressSeqOffsetShape_;
    gert::Shape seqLensShape_;
    ge::DataType inputKeyDtype_;
    ge::DataType inputValueDtype_;
};

} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_SCATTER_PA_KV_CACHE_H_