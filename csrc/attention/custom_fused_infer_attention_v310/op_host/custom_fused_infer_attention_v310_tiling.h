/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file custom_fused_infer_attention_v310_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_NEW_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_NEW_H_

#include <cstdint>
#include <vector>
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "custom_fused_infer_attention_v310_tiling_base.h"
#include "custom_fused_infer_attention_v310_tiling_data.h"

#ifdef ASCENDC_OP_TEST
#define IFA_EXTERN_C extern "C"
#else
#define IFA_EXTERN_C
#endif
namespace optiling {

struct IncreFlashAttentionCompileInfo {};

struct RequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct OptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

struct IncreFlashAttentionContext {
    const char *opName;
    fe::PlatFormInfos *platformInfo;
    RequiredParaInfo query;
    RequiredParaInfo key;
    RequiredParaInfo value;
    OptionalParaInfo attnMask;
    OptionalParaInfo actualSeqLengthsQ;
    OptionalParaInfo actualSeqLengths;
    OptionalParaInfo blockTable;

    RequiredParaInfo attenOut;
    const uint32_t *numHeads;
    const float *scaleValue;
    const uint32_t *kvHeadNums;
    const char *layOut;
    const uint32_t *blockSize;
    const uint32_t *innerPrecise;

    size_t *workSpaces;
    std::vector<gert::StorageShape *> kCache;
    std::vector<gert::StorageShape *> vCache;
    uint64_t tilingKey;
    uint32_t blockDim;
};

enum IfaLayout : uint32_t {
    BSH_BSND = 0,
    BSND = 1,
    TND = 3,
};

enum IfaMaskType : uint32_t {
    NO_MASK = 0,
    MASK_NORM = 1,
    MASK_COMPRESS = 2
};


class CustomFIATiling {
public:
    CustomFIATiling() = default;
    ~CustomFIATiling() = default;

    ge::graphStatus RunCustomFIATiling(IncreFlashAttentionContext &context);
    ge::graphStatus IncreFlashAttentionSetTilingData(gert::TilingContext &context);
    static ge::graphStatus ConvertContext(gert::TilingContext &context, IncreFlashAttentionContext &ifaContext);
private:
    ge::graphStatus CustomFIATilingProcess();
    ge::graphStatus CustomFIAParamSet();
    ge::graphStatus CustomFIAParamGet();
    ge::graphStatus CustomFIASplitBlock();
    void ParseMask();
    uint32_t GetTotalQTaskNum();

    ge::graphStatus InitPlatformInfo();
    ge::graphStatus ParseTilingAttributes();
    bool CheckIfShouldRunCustomFIA();

    ge::graphStatus ParseTndVarlenParams(const gert::Shape& qShape);
    ge::graphStatus ParsePagedAttentionParams();

    ge::graphStatus CheckBaseInputsNull();

    ge::graphStatus CheckInputFormatAndLimits();
    ge::graphStatus CheckInputParameterFormat();
    ge::graphStatus ProcessCheckCustomFIAInput();
    ge::graphStatus CheckCustomFIABaseParams();
    ge::graphStatus CheckCustomFIAInputDtype();
    ge::graphStatus CheckCustomFIAPageAttention();
    ge::graphStatus CheckCustomFIAQueryShape(const gert::StorageShape *queryShape);
    ge::graphStatus CheckCustomFIAKvShapeAndToken(const gert::StorageShape *queryShape,
                                            const gert::StorageShape *keyShape,
                                            const gert::StorageShape *valueShape);

    bool IsSupportFormat(const ge::Format format);

    ge::graphStatus GenTilingKey();

private:
    uint32_t numHeads_ = 0;
    float scaleValue_ = 0;
    uint32_t numKvHeads_ = 0;
    uint32_t qTokens_ = 0;
    uint32_t maskKvLen_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t maskBatchStride_ = 0;

    uint32_t headDim_ = 0;
    uint32_t batchSize_ = 0;
    IfaLayout inputLayout_ = IfaLayout::BSH_BSND;
    uint32_t tSeqSize_ = 1; // Length of the T axis in TND layout.

    ge::DataType inputQType_ = ge::DT_FLOAT16;
    ge::DataType inputKvType_ = ge::DT_FLOAT16;

    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    size_t libapiSize_ = 0;

    uint32_t attenMaskFlag_ = IfaMaskType::NO_MASK;

    uint32_t maxBlockNumPerBatch_ = 0;
    uint32_t totalBlockNum_ = 0;

    uint32_t seqStepQ_ = 0;

    IncreFlashAttentionContext *context_ = nullptr;
    IncreFlashAttentionBaseParams *tilingDataBase_ = nullptr;
    IncreFlashAttentionSplitCoreParams *tilingDataCore_ = nullptr;

    IncreFlashAttentionTilingAtbDataV2 ifaTilingAtbData;

    std::vector<uint32_t> tndQSeqLens_;
};

ge::graphStatus TilingPrepareForIncreFlashAttention(gert::TilingParseContext *context);
ge::graphStatus TilingCustomFIAAdapter(gert::TilingContext *context, IncreFlashAttentionContext &ifaContext);

IFA_EXTERN_C ge::graphStatus TilingIncreFlashAttention(gert::TilingContext *context);

} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_H_
