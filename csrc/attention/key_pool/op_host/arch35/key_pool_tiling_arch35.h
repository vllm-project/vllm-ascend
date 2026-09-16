/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file key_pool_tiling_arch35.h
 * \brief
 */

#ifndef KEY_POOL_TILING_ARCH35_H
#define KEY_POOL_TILING_ARCH35_H

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "exe_graph/runtime/tiling_context.h"
#include "register/op_def_registry.h"
#include "../../op_kernel/arch35/key_pool_template_tiling_key.h"
#include "../../op_kernel/arch35/key_pool_tiling_data.h"
#include "platform/platform_info.h"

#ifdef ASCENDC_OP_TEST
#define CMP_EXTERN_C extern "C"
#else
#define CMP_EXTERN_C
#endif

namespace optiling {

// INPUT
constexpr uint32_t HIDDEN_STATES_INPUT_INDEX = 0;
constexpr uint32_t WK_INPUT_INDEX = 1;
constexpr uint32_t GATE_WEIGHT_INPUT_INDEX = 2;
constexpr uint32_t APE_INPUT_INDEX = 3;
constexpr uint32_t STATE_CACHE_INPUT_INDEX = 4;
constexpr uint32_t CACHE_BLOCK_TABLE_INPUT_INDEX = 5;
constexpr uint32_t START_POS_INPUT_INDEX = 6;
constexpr uint32_t NORM_WEIGHT_INPUT_INDEX = 7;
constexpr uint32_t NORM_BIAS_INPUT_INDEX = 8;
constexpr uint32_t COS_INPUT_INDEX = 9;
constexpr uint32_t SIN_INPUT_INDEX = 10;
constexpr uint32_t CU_SEQLENS_INPUT_INDEX = 11;
constexpr uint32_t SEQUSED_INPUT_INDEX = 12;

// ATTR
constexpr uint32_t CMP_RATIO_ATTR_INDEX = 0;
constexpr uint32_t NORM_EPS_ATTR_INDEX = 1;
constexpr uint32_t ROTARY_MODE_ATTR_INDEX = 2;
constexpr uint32_t STATE_CACHE_STRIDE_DIM0_ATTR_INDEX = 3;

// OUTPUT
constexpr uint32_t POOLED_KEY_OUTPUT_INDEX = 0;

constexpr uint32_t KEY_POOL_DIM_NUM_1 = 1;
constexpr uint32_t KEY_POOL_DIM_NUM_2 = 2;
constexpr uint32_t KEY_POOL_DIM_NUM_3 = 3;
constexpr uint32_t KEY_POOL_DIM_INDEX_0 = 0;
constexpr uint32_t KEY_POOL_DIM_INDEX_1 = 1;
constexpr uint32_t KEY_POOL_DIM_INDEX_2 = 2;

// CONSTRAINTS
constexpr uint32_t MAX_HIDDEN_SIZE = 10240;
constexpr uint32_t MIN_HIDDEN_SIZE = 1024;
constexpr uint32_t ALIGN_FACTOR_HIDDEN_SIZE = 512;
constexpr uint32_t MIN_BLOCK_SIZE = 1;
constexpr uint32_t MAX_BLOCK_SIZE = 1024;
constexpr uint32_t MAX_CMPRATIO_SIZE = 128;
constexpr uint32_t MIN_CMPRATIO_SIZE = 2;

constexpr uint32_t BATCH_MODE_SCHEDULE = 1;

static const std::string HIDDEN_STATES_NAME = "hidden_states";
static const std::string WK_NAME = "wk";
static const std::string GATE_WEIGHT_NAME = "gate_weight";
static const std::string STATE_CACHE_NAME = "state_cache";
static const std::string APE_NAME = "ape";
static const std::string CACHE_BLOCK_TABLE_NAME = "cache_block_table";
static const std::string CU_SEQLENS_NAME = "cu_seqlens";
static const std::string SEQUSED_NAME = "seqused";
static const std::string START_POS_NAME = "start_pos";
static const std::string NORM_WEIGHT_NAME = "norm_weight";
static const std::string NORM_BIAS_NAME = "norm_bias";
static const std::string COS_NAME = "cos";
static const std::string SIN_NAME = "sin";
static const std::string CMP_RATIO_NAME = "cmp_ratio";
static const std::string POOLED_KEY_NAME = "pooled_key";

static std::string DataTypeToSerialString(ge::DataType type);

const std::map<std::string, std::vector<ge::DataType>> DTYPE_SUPPORT_MAP = {
    {HIDDEN_STATES_NAME, {ge::DT_BF16, ge::DT_FLOAT16}},
    {WK_NAME, {ge::DT_BF16, ge::DT_FLOAT16}},
    {GATE_WEIGHT_NAME, {ge::DT_BF16, ge::DT_FLOAT16}},
    {STATE_CACHE_NAME, {ge::DT_FLOAT}},
    {APE_NAME, {ge::DT_FLOAT}},
    {CACHE_BLOCK_TABLE_NAME, {ge::DT_INT32}},
    {CU_SEQLENS_NAME, {ge::DT_INT32}},
    {SEQUSED_NAME, {ge::DT_INT32}},
    {START_POS_NAME, {ge::DT_INT32}},
    {NORM_WEIGHT_NAME, {ge::DT_FLOAT}},
    {NORM_BIAS_NAME, {ge::DT_FLOAT}},
    {COS_NAME, {ge::DT_BF16}},
    {SIN_NAME, {ge::DT_BF16}},
    {POOLED_KEY_NAME, {ge::DT_BF16, ge::DT_FLOAT16}}};

const std::map<std::string, std::vector<uint32_t>> DIM_NUM_MAP = {
    {HIDDEN_STATES_NAME, {KEY_POOL_DIM_NUM_2, KEY_POOL_DIM_NUM_3}},
    {WK_NAME, {KEY_POOL_DIM_NUM_2}},
    {GATE_WEIGHT_NAME, {KEY_POOL_DIM_NUM_2}},
    {STATE_CACHE_NAME, {KEY_POOL_DIM_NUM_3}},
    {APE_NAME, {KEY_POOL_DIM_NUM_2}},
    {CACHE_BLOCK_TABLE_NAME, {KEY_POOL_DIM_NUM_2, KEY_POOL_DIM_NUM_1}},
    {CU_SEQLENS_NAME, {KEY_POOL_DIM_NUM_1}},
    {SEQUSED_NAME, {KEY_POOL_DIM_NUM_1}},
    {START_POS_NAME, {KEY_POOL_DIM_NUM_1}},
    {NORM_WEIGHT_NAME, {KEY_POOL_DIM_NUM_1}},
    {NORM_BIAS_NAME, {KEY_POOL_DIM_NUM_1}},
    {COS_NAME, {KEY_POOL_DIM_NUM_2, KEY_POOL_DIM_NUM_3}},
    {SIN_NAME, {KEY_POOL_DIM_NUM_2, KEY_POOL_DIM_NUM_3}},
    {POOLED_KEY_NAME, {KEY_POOL_DIM_NUM_2, KEY_POOL_DIM_NUM_3}}};

static const std::map<std::string, uint32_t> LAYOUT_DIM_MAP = {
    {"BSH", KEY_POOL_DIM_NUM_3},
    {"TH", KEY_POOL_DIM_NUM_2},
};

const std::map<ge::DataType, std::string> DATATYPE_TO_STRING_MAP = {
    {ge::DT_UNDEFINED, "DT_UNDEFINED"},           // Used to indicate a DataType field has not been set.
    {ge::DT_FLOAT, "DT_FLOAT"},                   // float type
    {ge::DT_FLOAT16, "DT_FLOAT16"},               // fp16 type
    {ge::DT_INT8, "DT_INT8"},                     // int8 type
    {ge::DT_INT16, "DT_INT16"},                   // int16 type
    {ge::DT_UINT16, "DT_UINT16"},                 // uint16 type
    {ge::DT_UINT8, "DT_UINT8"},                   // uint8 type
    {ge::DT_INT32, "DT_INT32"},                   // uint32 type
    {ge::DT_INT64, "DT_INT64"},                   // int64 type
    {ge::DT_UINT32, "DT_UINT32"},                 // unsigned int32
    {ge::DT_UINT64, "DT_UINT64"},                 // unsigned int64
    {ge::DT_BOOL, "DT_BOOL"},                     // bool type
    {ge::DT_DOUBLE, "DT_DOUBLE"},                 // double type
    {ge::DT_DUAL, "DT_DUAL"},                     // dual output type
    {ge::DT_DUAL_SUB_INT8, "DT_DUAL_SUB_INT8"},   // dual output int8 type
    {ge::DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8"}, // dual output uint8 type
    {ge::DT_COMPLEX32, "DT_COMPLEX32"},           // complex32 type
    {ge::DT_COMPLEX64, "DT_COMPLEX64"},           // complex64 type
    {ge::DT_COMPLEX128, "DT_COMPLEX128"},         // complex128 type
    {ge::DT_QINT8, "DT_QINT8"},                   // qint8 type
    {ge::DT_QINT16, "DT_QINT16"},                 // qint16 type
    {ge::DT_QINT32, "DT_QINT32"},                 // qint32 type
    {ge::DT_QUINT8, "DT_QUINT8"},                 // quint8 type
    {ge::DT_QUINT16, "DT_QUINT16"},               // quint16 type
    {ge::DT_RESOURCE, "DT_RESOURCE"},             // resource type
    {ge::DT_STRING_REF, "DT_STRING_REF"},         // string ref type
    {ge::DT_STRING, "DT_STRING"},                 // string type
    {ge::DT_VARIANT, "DT_VARIANT"},               // dt_variant type
    {ge::DT_BF16, "DT_BFLOAT16"},                 // dt_bfloat16 type
    {ge::DT_INT4, "DT_INT4"},                     // dt_variant type
    {ge::DT_UINT1, "DT_UINT1"},                   // dt_variant type
    {ge::DT_INT2, "DT_INT2"},                     // dt_variant type
    {ge::DT_UINT2, "DT_UINT2"}                    // dt_variant type
};

struct KeyPoolCompileInfo {
    int64_t core_num;
};

struct RequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct OptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

enum class LayoutType {
    LAYOUT_BSH,
    LAYOUT_TH
};

enum class TemplateId : uint8_t {
    NORMAL = 0,
    EMPTY_HIDDEN_STATES = 1,
    FULL_LOAD = 2
};

CMP_EXTERN_C ge::graphStatus TilingKeyPool(gert::TilingContext *context);

const std::vector<uint32_t> HEAD_DIM{128, 512};

struct KeyPoolContext {
    const char *opName;
    const char *opType;
    fe::PlatFormInfos *platformInfo;
    int batchConsistency;

    RequiredParaInfo hidden_states;
    RequiredParaInfo wk;
    RequiredParaInfo gate_weight;
    RequiredParaInfo stateCache;
    RequiredParaInfo ape;
    OptionalParaInfo cacheBlockTable;
    OptionalParaInfo seqLens;
    OptionalParaInfo seqUsed;
    OptionalParaInfo startPos;
    OptionalParaInfo normWeight;
    OptionalParaInfo normBias;
    OptionalParaInfo cos;
    OptionalParaInfo sin;
    RequiredParaInfo pooledKey;

    const int *cmpRatio;
    const float *normEps;
    const int *rotaryMode;
    const int *stateCacheStrideDim0;
    TemplateId templateId;

    ge::DataType dtype = ge::DT_BF16;
    LayoutType layout = LayoutType::LAYOUT_BSH;

    size_t *workSpaces;
    uint64_t tilingKey;
    uint32_t blockDim;
};

class KeyPoolTiling {
public:
    explicit KeyPoolTiling(KeyPoolContext *context)
        : context_(context)
    {}
    ~KeyPoolTiling() = default;

    static ge::graphStatus ConvertContext(gert::TilingContext &context, KeyPoolContext &key_poolContext);
    ge::graphStatus RunBigKernelTiling(KeyPoolTilingData *tilingData);

private:
    static void ConvertRequiredParams(gert::TilingContext &context, KeyPoolContext &key_poolContext);

    static void ConvertOptionalParams(gert::TilingContext &context, KeyPoolContext &key_poolContext);
    ge::graphStatus GetNpuInfo();
    ge::graphStatus SetBaseInfo();
    ge::graphStatus SetPageAttentionInfo();
    ge::graphStatus SetWorkSpaceInfo();
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
    ge::graphStatus CheckSingleParaHiddenStates() const;
    ge::graphStatus CheckSingleParaWk() const;
    ge::graphStatus CheckSingleParaGateWeight() const;
    ge::graphStatus CheckSingleParaStateCache() const;
    ge::graphStatus CheckSingleParaApe() const;
    ge::graphStatus CheckSingleParaCacheBlockTable() const;
    ge::graphStatus CheckSingleParaSeqLens() const;
    ge::graphStatus CheckSingleParaSeqUsed() const;
    ge::graphStatus CheckSingleParaStartPos() const;
    ge::graphStatus CheckSingleParaNormWeight() const;
    ge::graphStatus CheckSingleParaNormBias() const;
    ge::graphStatus CheckSingleParaPooledKey() const;
    ge::graphStatus CheckSingleParaCmpRatio() const;
    ge::graphStatus CheckRequiredParaExistence() const;
    ge::graphStatus CheckRequiredInOutExistence() const;
    ge::graphStatus CheckRequiredAttrExistence() const;
    ge::graphStatus CheckFeature() const;
    ge::graphStatus CheckShapeConsistency() const;
    ge::graphStatus CheckDtypeConsistencyHiddenStates(const gert::CompileTimeTensorDesc *desc,
                                                      const std::string &name) const;
    ge::graphStatus CheckDtypeConsistency() const;
    ge::graphStatus CheckMultiParaConsistency() const;
    ge::graphStatus CheckDimNumConsistency() const;
    ge::graphStatus CheckEmptyTensor() const;
    ge::graphStatus CheckBlockDimConstrain() const;

    size_t ubSize_ = 0;
    size_t l1Size_ = 0;
    size_t l0cSize_ = 0;
    size_t l0bSize_ = 0;
    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    size_t libapiSize_ = 0;
    size_t workspaceSize_ = 0;
    KeyPoolContext *context_ = nullptr;
    KeyPoolBaseParams *baseParams_ = nullptr;
    KeyPoolPageAttentionParams *pageAttentionParams_ = nullptr;
    KeyPoolInnerSplitParams *innerSplitParams_ = nullptr;
    KeyPoolWorkspaceParams *workspaceParams_ = nullptr;
};

} // namespace optiling

#endif
