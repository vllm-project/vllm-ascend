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
 * \file compressor_tiling.h
 * \brief
 */

#ifndef COMPRESSOR_TILING_H
#define COMPRESSOR_TILING_H

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <sstream>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "exe_graph/runtime/tiling_context.h"
#include "register/op_def_registry.h"
#include "../op_kernel/compressor_template_tiling_key.h"
#include "../op_kernel/compressor_tiling_data.h"
#include "platform/platform_info.h"

#ifdef ASCENDC_OP_TEST
#define CMP_EXTERN_C extern "C"
#else
#define CMP_EXTERN_C
#endif
// #define DAY0_SCOPE

namespace optiling {

    // INPUT
    constexpr uint32_t TOKEN_X_INPUT_INDEX = 0;
    constexpr uint32_t WEIGHT_KV_INPUT_INDEX = 1;
    constexpr uint32_t WEIGHT_WGATE_INPUT_INDEX = 2;
    constexpr uint32_t KV_STATE_INPUT_INDEX = 3;
    constexpr uint32_t SCORE_STATE_INPUT_INDEX = 4;
    constexpr uint32_t APE_INPUT_INDEX = 5;
    constexpr uint32_t NORM_WEIGHT_INPUT_INDEX = 6;
    constexpr uint32_t ROPE_SIN_INPUT_INDEX = 7;
    constexpr uint32_t ROPE_COS_INPUT_INDEX = 8;

    // INPUT(OPTION)
    constexpr uint32_t KV_BLOCK_TABLE_INPUT_INDEX = 9;
    constexpr uint32_t SCORE_BLOCK_TABLE_INPUT_INDEX = 10;
    constexpr uint32_t CU_SEQ_LEN_INPUT_INDEX = 11;
    constexpr uint32_t SEQ_USED_INPUT_INDEX = 12;
    constexpr uint32_t START_POS_INPUT_INDEX = 13;

    // ATTR
    constexpr uint32_t ROPE_HEAD_DIM_ATTR_INDEX = 0;
    constexpr uint32_t CMP_RATIO_ATTR_INDEX = 1;
    constexpr uint32_t COFF_ATTR_INDEX = 2;
    constexpr uint32_t NORM_EPS_ATTR_INDEX = 3;
    constexpr uint32_t ROTARY_MODE_ATTR_INDEX = 4;

    // OUTPUT
    constexpr uint32_t CMP_KV_OUTPUT_INDEX = 0;

    constexpr uint32_t COMPRESSOR_DIM_NUM_1 = 1;
    constexpr uint32_t COMPRESSOR_DIM_NUM_2 = 2;
    constexpr uint32_t COMPRESSOR_DIM_NUM_3 = 3;
    constexpr uint32_t COMPRESSOR_DIM_NUM_4 = 4;
    constexpr uint32_t COMPRESSOR_DIM_INDEX_0 = 0;
    constexpr uint32_t COMPRESSOR_DIM_INDEX_1 = 1;
    constexpr uint32_t COMPRESSOR_DIM_INDEX_2 = 2;
    constexpr uint32_t COMPRESSOR_DIM_INDEX_3 = 3;

    // CONSTRAINTS
    constexpr uint32_t MAX_HIDDEN_SIZE = 10240;
    constexpr uint32_t MIN_HIDDEN_SIZE = 1024;
    constexpr uint32_t ALIGN_FACTOR_HIDDEN_SIZE = 512;
    constexpr uint32_t MAX_BLOCK_SIZE = 1024;
    constexpr uint32_t MIN_BLOCK_SIZE = 16;
    constexpr uint32_t ALIGN_FACTOR_BLOCK_SIZE = 16;

    constexpr uint32_t BATCH_MODE_SCHEDULE = 1;
    
struct CompressorCompileInfo {
    int64_t core_num;
};

struct RequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct OptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
    const gert::Tensor *tensor;
};

enum class LayoutType {
    LAYOUT_BSH,
    LAYOUT_TH
};

enum class TemplateId:uint8_t {
    NORMAL = 0,
    EMPTY_X = 1,
    PERF = 2
};

CMP_EXTERN_C ge::graphStatus TilingCompressor(gert::TilingContext *context);
struct CompressorBaseShapeInfo {
    uint32_t bSize = 0; // B
    uint32_t sSize = 0; // S
    uint32_t hSize = 0; // Hidden size
    uint32_t tSize = 0; // T
    uint32_t nSize = 0; // N
    uint32_t dSize = 0; // D
    uint32_t coffSize = 0; // Coff: 1 or 2
    uint32_t csSize = 0; // Compress sequence len
    uint32_t rSize = 0; // Compress ratio
    uint32_t cgSize = 0; // Compress group size
    uint32_t drSize = 0; // Dr
};

const std::vector<int> ROPE_HEAD_DIM {64};
const std::vector<int> COFF {1, 2};
#ifdef DAY0_SCOPE
const std::vector<int> CMP_RATIO {4, 128};
const std::vector<int> ROTARY_MODE {2};
#else
const std::vector<int> CMP_RATIO {2, 4, 8, 16, 32, 64, 128};
const std::vector<int> ROTARY_MODE {1, 2};
#endif
const std::vector<uint32_t> HEAD_DIM {128, 512};

enum class ROTARY_MODE:uint8_t {
    HALF = 1,
    INTERLEAVE = 2
};

struct CompressorContext {
    const char *opName;
    const char *opType;
    fe::PlatFormInfos *platformInfo;
    
    RequiredParaInfo x;
    RequiredParaInfo wkv;
    RequiredParaInfo wgate;
    RequiredParaInfo kvState;
    RequiredParaInfo scoreState;
    RequiredParaInfo ape;
    RequiredParaInfo normWeight;
    RequiredParaInfo ropeSin;
    RequiredParaInfo ropeCos;
    OptionalParaInfo kvBlockTable;
    OptionalParaInfo scoreBlockTable;
    OptionalParaInfo cuSeqlens;
    OptionalParaInfo seqUsed;
    OptionalParaInfo startPos;
    RequiredParaInfo cmpKv;

    const int *ropeHeadDim;
    const int *coff;
    const int *cmpRatio;
    const float *normEps;
    const int *rotaryMode;
    TemplateId templateId;

    ge::DataType dtype = ge::DT_BF16; 
    LayoutType layout = LayoutType::LAYOUT_BSH; 

    size_t *workSpaces;
    uint64_t tilingKey;
    uint32_t blockDim;
};

class CompressorTiling {
public:
    explicit CompressorTiling(CompressorContext *context) : context_(context) {}
    ~CompressorTiling() = default;

    static ge::graphStatus ConvertContext(gert::TilingContext &context, CompressorContext &compressorContext);
    ge::graphStatus RunBigKernelTiling(CompressorTilingData* tilingData);

private:
    static void ConvertRequiredParams(gert::TilingContext &context, CompressorContext &compressorContext);

    static void ConvertOptionalParams(gert::TilingContext &context, CompressorContext &compressorContext);
    ge::graphStatus GetNpuInfo();
    ge::graphStatus SetBaseInfo();
    ge::graphStatus SetPageAttentionInfo();
    ge::graphStatus SetWorkSpaceInfo();
    ge::graphStatus SetScenarioInfo();
    ge::graphStatus SetTemplateId();
    ge::graphStatus SetInnerSplitInfo();
    ge::graphStatus CalcWorkSpace();
    ge::graphStatus CheckSinglePara() const;
    ge::graphStatus GenTilingKey() const;
    template <typename T>
    ge::graphStatus CheckFeatureValueSupport(const T *featureValue, const std::vector<T> &expectFeatureValList,
                                             const std::string &name) const;
    template <typename T>
    ge::graphStatus CheckAttrValueSupport(const T *attrValue, const std::vector<T> &expectAttrValList,
                                          const std::string &name) const;
    template <typename T>
    void LogErrorNumberSupport(const std::vector<T> &expectNumberList, const T &actualValue, const std::string &name,
                               const std::string subName) const;
    ge::graphStatus CheckDimNumInLayoutSupport(const std::string &layout, const gert::StorageShape *shape,
                                               const std::string &name) const;
    ge::graphStatus CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc, const std::string &name) const;
    void LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList, const ge::DataType &actualDtype,
                              const std::string &name) const;
    ge::graphStatus CheckDimNumSupport(const gert::StorageShape *shape, const std::string &name) const;
    ge::graphStatus LogErrorShapeConsistency(const std::string &name, const gert::StorageShape *shape,
                                             const uint32_t &dimNum, const std::string &subName,
                                             const uint32_t &expectNum) const;
    ge::graphStatus CheckSingleParaX() const;
    ge::graphStatus CheckSingleParaWkv() const;
    ge::graphStatus CheckSingleParaWgate() const;
    ge::graphStatus CheckSingleParaKvState() const;
    ge::graphStatus CheckSingleParaScoreState() const;
    ge::graphStatus CheckSingleParaApe() const;
    ge::graphStatus CheckSingleParaNormWeight() const;
    ge::graphStatus CheckSingleParaRopeSin() const;
    ge::graphStatus CheckSingleParaRopeCos() const;
    ge::graphStatus CheckSingleParaKvBlockTable() const;
    ge::graphStatus CheckSingleParaScoreBlockTable() const;
    ge::graphStatus CheckSingleParaCuSeqlens() const;
    ge::graphStatus CheckSingleParaSeqused() const;
    ge::graphStatus CheckSingleParaStartPos() const;
    ge::graphStatus CheckSingleParaCmpKv() const;
    ge::graphStatus CheckSingleParaRopeHeadDim() const;
    ge::graphStatus CheckSingleParaCmpRatio() const;
    ge::graphStatus CheckSingleParaCoff() const;
    ge::graphStatus CheckSingleParaNormEps() const;
    ge::graphStatus CheckSingleParaRotaryMode() const;
    ge::graphStatus CheckRequiredParaExistence() const;
    ge::graphStatus CheckRequiredInOutExistence() const;
    ge::graphStatus CheckRequiredAttrExistence() const;
    ge::graphStatus CheckFeature() const;
    ge::graphStatus CheckShapeConsistency() const;
    ge::graphStatus CheckShapeConsistencyRope() const;
    ge::graphStatus CheckDtypeConsistencyX(const gert::CompileTimeTensorDesc *desc, const std::string &name) const;
    ge::graphStatus CheckDtypeConsistency() const;
    ge::graphStatus CheckMultiParaConsistency() const;
    ge::graphStatus CheckDimNumConsistency() const;
    ge::graphStatus CheckEmptyTensor() const;
    ge::graphStatus CheckScenarioConsistency() const;

    size_t ubSize_ = 0;
    size_t l1Size_ = 0;
    size_t l0cSize_ = 0;
    size_t l0bSize_ = 0;
    uint32_t coreNum_ = 0;
    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    size_t libapiSize_ = 0;
    // platform_ascendc::SocVersion socVersion_;
    size_t workspaceSize_ = 0;
    uint8_t coff = 1;

    uint32_t mBaseSize = 0;
    uint32_t dbaseSize = 0;

    CompressorBaseShapeInfo baseShapeInfo_;
    CompressorContext *context_ = nullptr;
    CompressorBaseParams *baseParams_ = nullptr;
    CompressorPageAttentionParams *pageAttentionParams_ = nullptr;
    CompressorInnerSplitParams *innerSplitParams_ = nullptr;
    CompressorWorkspaceParams *workspaceParams_ = nullptr;
};
} // optiling

#endif
