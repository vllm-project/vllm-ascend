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
 * \file mla_prolog_tiling.h
 * \brief
 */

#ifndef MLA_PROLOG_TILING_H
#define MLA_PROLOG_TILING_H

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <sstream>
#include <algorithm>
#include <unordered_set>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling_base/data_copy_transpose_tiling.h"
#include "exe_graph/runtime/tiling_context.h"
#include "register/op_def_registry.h"
#include "../op_kernel/mla_prolog_template_tiling_key.h"
#include "../op_kernel/mla_prolog_tiling_data.h"
#include "platform/soc_spec.h"

#ifdef ASCENDC_OP_TEST
#define MLA_EXTERN_C extern "C"
#else
#define MLA_EXTERN_C
#endif

namespace optiling {

// INPUT
constexpr uint32_t TOKEN_X_INPUT_INDEX = 0;
constexpr uint32_t WEIGHT_DQ_INPUT_INDEX = 1;
constexpr uint32_t WEIGHT_UQ_QR_INPUT_INDEX = 2;
constexpr uint32_t WEIGHT_UK_INPUT_INDEX = 3;
constexpr uint32_t WEIGHT_DKV_KR_INPUT_INDEX = 4;
constexpr uint32_t RMSNORM_GAMMA_CQ_INPUT_INDEX = 5;
constexpr uint32_t RMS_NORM_GAMMA_CKV_INPUT_INDEX = 6;
constexpr uint32_t ROPE_SIN_INPUT_INDEX = 7;
constexpr uint32_t ROPE_COS_INPUT_INDEX = 8;
constexpr uint32_t CACHE_INDEX_INPUT_INDEX = 9;
constexpr uint32_t KV_CACHE_INPUT_INDEX = 10;
constexpr uint32_t KR_CACHE_INPUT_INDEX = 11;

constexpr uint32_t KV_CACHE_INPUT_INDEX_V3 = 9;
constexpr uint32_t KR_CACHE_INPUT_INDEX_V3 = 10;
constexpr uint32_t CACHE_INDEX_INPUT_INDEX_V3 = 11;

// INPUT(OPTION)
constexpr uint32_t DEQUANT_SCALE_X_INDEX = 12;
constexpr uint32_t DEQUANT_SCALE_W_DQ_INDEX = 13;
constexpr uint32_t DEQUANT_SCALE_W_UQ_QR_INDEX = 14;
constexpr uint32_t DEQUANT_SCALE_W_DKV_KR_INDEX = 15;
constexpr uint32_t QUANT_SCALE_CKV_INDEX = 16;
constexpr uint32_t QUANT_SCALE_CKR_INDEX = 17;
constexpr uint32_t SMOOTH_SCALES_CQ_INDEX = 18;
constexpr uint32_t ACTUAL_SEQ_LEN_INDEX = 19;
constexpr uint32_t K_NOPE_CLIP_ALPHA_INDEX = 20;

// OUTPUT
constexpr uint32_t QUERY_OUTPUT_INDEX = 0;
constexpr uint32_t QUERY_ROPE_OUTPUT_INDEX = 1;
constexpr uint32_t KV_CACHE_OUT_OUTPUT_INDEX = 2;
constexpr uint32_t KR_CACHE_OUT_OUTPUT_INDEX = 3;
constexpr uint32_t DEQUANT_SCALE_Q_NOPE_OUTPUT_INDEX = 4;
constexpr uint32_t QUERY_NORM_OUTPUT_INDEX = 5;
constexpr uint32_t DEQUANT_SCALE_Q_NORM_OUTPUT_INDEX = 6;

// ATTR
constexpr uint32_t RMS_NORM_EPSILON_CQ_ATTR_INDEX = 0;
constexpr uint32_t RMS_NORM_EPSILON_CKV_ATTR_INDEX = 1;
constexpr uint32_t CACHE_MODE_ATTR_INDEX = 2;
constexpr uint32_t QUERY_NORM_FLAG_ATTR_INDEX = 3;
constexpr uint32_t WEIGHT_QUANT_MODE_ATTR_INDEX = 4;
constexpr uint32_t KV_CACHE_QUANT_MODE_ATTR_INDEX = 5;
constexpr uint32_t QUERY_QUANT_MODE_ATTR_INDEX = 6;
constexpr uint32_t CKVKR_REPO_MODE_ATTR_INDEX = 7;
constexpr uint32_t QUANT_SCALE_REPO_MODE_ATTR_INDEX = 8;
constexpr uint32_t TILE_SIZE_ATTR_INDEX = 9;
constexpr uint32_t QC_QR_SCALE_ATTR_INDEX = 10;
constexpr uint32_t KC_SCALE_ATTR_INDEX = 11;

constexpr uint32_t MLA_PROLOG_DIM_INDEX_0 = 0;
constexpr uint32_t MLA_PROLOG_DIM_INDEX_1 = 1;
constexpr uint32_t MLA_PROLOG_DIM_INDEX_2 = 2;
constexpr uint32_t MLA_PROLOG_DIM_INDEX_3 = 3;

constexpr uint32_t MLA_PROLOG_DIM_NUM_0 = 0;
constexpr uint32_t MLA_PROLOG_DIM_NUM_1 = 1;
constexpr uint32_t MLA_PROLOG_DIM_NUM_2 = 2;
constexpr uint32_t MLA_PROLOG_DIM_NUM_3 = 3;
constexpr uint32_t MLA_PROLOG_DIM_NUM_4 = 4;

constexpr uint32_t HEAD_SIZE1 = 7168;
constexpr uint32_t HEAD_SIZE2 = 7680;

constexpr char CACHE_MODE_BSND[]{"BSND"};
constexpr char CACHE_MODE_TND[]{"TND"};
constexpr char CACHE_MODE_PA_BSND[]{"PA_BSND"};
constexpr char CACHE_MODE_PA_NZ[]{"PA_NZ"};
constexpr char CACHE_MODE_PA_BLK_BSND[]{"PA_BLK_BSND"};
constexpr char CACHE_MODE_PA_BLK_NZ[]{"PA_BLK_NZ"};

constexpr char V1_OP_NAME[]{"MlaProlog"};
constexpr char V2_OP_NAME[]{"MlaPrologV2"};
constexpr char V3_OP_NAME[]{"MlaPrologV3"};


constexpr uint32_t CACHE_MODE_LEN =
    std::max({sizeof(CACHE_MODE_BSND), sizeof(CACHE_MODE_TND), sizeof(CACHE_MODE_PA_BSND), sizeof(CACHE_MODE_PA_NZ),
              sizeof(CACHE_MODE_PA_BLK_BSND), sizeof(CACHE_MODE_PA_BLK_NZ)});

constexpr uint32_t OP_NAME_LEN = std::max({sizeof(V1_OP_NAME), sizeof(V2_OP_NAME), sizeof(V3_OP_NAME)});

struct MlaPrologBaseShapeInfo {
    uint32_t bSize = 0;        // B
    uint32_t s1Size = 0;       // S1
    uint32_t tSize = 0;        // T
    uint32_t s2Size = 0;       // S2
    uint32_t heSize = 0;       // He
    uint32_t hcqSize = 0;      // Hcq
    uint32_t hckvSize = 0;     // Hckv
    uint32_t headSizeQc = 0;   // N * D
    uint32_t headSizeQr = 0;   // N * Dr
    uint32_t headSizeUqQr = 0; // N * (D + Dr)
    uint32_t nSize = 0;        // N
    uint32_t nkvSize = 0;      // Nkv
    uint32_t dSize = 0;        // D
    uint32_t blockNum = 0;
    uint32_t blockSize = 0;
    uint32_t drSize = 0;    // Dr
    uint32_t dtileSize = 0; // Dtile
};

enum class CACHE_MODE : uint8_t {
    BSND = 0,
    PA_BSND = 1,
    PA_NZ = 2,
    PA_BLK_BSND = 3,
    PA_BLK_NZ = 4,
    TND = 5
};

// cacheMode 字符串 → CACHE_MODE 枚举 hash 查找表
// CheckCacheMode 已保证执行到此处的 cacheMode 必为 6 个合法串之一
inline const std::unordered_map<std::string, CACHE_MODE> CACHE_MODE_HASH_TABLE = {
    {CACHE_MODE_BSND, CACHE_MODE::BSND},
    {CACHE_MODE_TND, CACHE_MODE::TND},
    {CACHE_MODE_PA_BSND, CACHE_MODE::PA_BSND},
    {CACHE_MODE_PA_NZ, CACHE_MODE::PA_NZ},
    {CACHE_MODE_PA_BLK_BSND, CACHE_MODE::PA_BLK_BSND},
    {CACHE_MODE_PA_BLK_NZ, CACHE_MODE::PA_BLK_NZ}};

enum class EMPTY_TENSOR_MODE : uint8_t {
    NON_EMPTY = 0,
    EMPTY_CACHE = 1,
    EMPTY_QUERY = 2
};

enum class ACTUAL_SEQ_MODE : uint8_t {
    DISABLED = 0,
    EN_Q_LEN = 1,
};

enum class QUANT_MODE : int8_t {
    ERROR_MODE = -1,
    NO_QUANT = 0,
    PARTIAL_QUANT_KV_NO_QUANT = 1,
    PARTIAL_QUANT_KV_QUANT_PER_CHANNEL = 2,
    FULL_QUANT_KV_NO_QUANT = 3,
    FULL_QUANT_KV_QUANT_PER_TENSOR = 4,
    PARTIAL_QUANT_KV_QUANT_PER_TILE = 5,
    FULL_QUANT_KV_QUANT_PER_TILE = 6,
    MXFP8_FULL_QUANT_KV_NO_QUANT = 7,
    MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR = 8,
    MXFP8_FULL_QUANT_KV_QUANT_PER_TILE = 9,
    FP8_FULL_QUANT_KV_NO_QUANT = 10,
    FP8_FULL_QUANT_KV_QUANT_PER_TENSOR = 11,
    HIF8_FULL_QUANT_KV_NO_QUANT = 12,
    HIF8_FULL_QUANT_KV_QUANT_PER_TENSOR = 13,
    FP8_FULL_QUANT_KV_QUANT_PER_TILE = 14,
    HIF8_FULL_QUANT_KV_QUANT_PER_TILE = 15
};

enum class WEIGHT_QUANT_MODE : uint8_t {
    NO_QUANT = 0,
    PARTIAL_QUANT = 1,
    FULL_QUANT = 2,
    MXFP8_FULL_QUANT = 3,
    FP8_FULL_QUANT = 4,
    HIF8_FULL_QUANT = 5
};

enum class KV_QUANT_MODE : uint8_t {
    NO_QUANT = 0,
    PER_TENSOR = 1,
    PER_CHANNEL = 2,
    PER_TILE = 3
};

// weightQuantMode(0-5) × kvQuantMode(0-3) → QUANT_MODE 双重哈希查找表
// 外层 Key 是 weightQuantMode，内层 Key 是 kvQuantMode，Value 是 QUANT_MODE
// 只存放合法组合，查不到即非法（含越界与组合非法）
inline const std::unordered_map<int, std::unordered_map<int, QUANT_MODE>> QUANT_MODE_HASH_TABLE = {
    // wq=0 NO_QUANT: 合法 kv 仅 {0}
    {0, {{0, QUANT_MODE::NO_QUANT}}},
    // wq=1 PARTIAL: 合法 kv 为 {0, 2, 3}
    {1,
     {{0, QUANT_MODE::PARTIAL_QUANT_KV_NO_QUANT},
      {2, QUANT_MODE::PARTIAL_QUANT_KV_QUANT_PER_CHANNEL},
      {3, QUANT_MODE::PARTIAL_QUANT_KV_QUANT_PER_TILE}}},
    // wq=2 FULL: 合法 kv 为 {0, 1, 3}
    {2,
     {{0, QUANT_MODE::FULL_QUANT_KV_NO_QUANT},
      {1, QUANT_MODE::FULL_QUANT_KV_QUANT_PER_TENSOR},
      {3, QUANT_MODE::FULL_QUANT_KV_QUANT_PER_TILE}}},
    // wq=3 MXFP8: 合法 kv 为 {0, 1, 3}
    {3,
     {{0, QUANT_MODE::MXFP8_FULL_QUANT_KV_NO_QUANT},
      {1, QUANT_MODE::MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR},
      {3, QUANT_MODE::MXFP8_FULL_QUANT_KV_QUANT_PER_TILE}}},
    // wq=4 FP8: 合法 kv 为 {0, 1, 3}
    {4,
     {{0, QUANT_MODE::FP8_FULL_QUANT_KV_NO_QUANT},
      {1, QUANT_MODE::FP8_FULL_QUANT_KV_QUANT_PER_TENSOR},
      {3, QUANT_MODE::FP8_FULL_QUANT_KV_QUANT_PER_TILE}}},
    // wq=5 HIF8: 合法 kv 为 {0, 1, 3}
    {5,
     {{0, QUANT_MODE::HIF8_FULL_QUANT_KV_NO_QUANT},
      {1, QUANT_MODE::HIF8_FULL_QUANT_KV_QUANT_PER_TENSOR},
      {3, QUANT_MODE::HIF8_FULL_QUANT_KV_QUANT_PER_TILE}}}};

// 各 weightQuantMode 下合法 kvQuantMode 集合说明（用于错误日志 reason 文本，与原实现逐字一致）
inline const std::unordered_map<int, const char *> VALID_KV_REASON_TABLE = {
    {0, "When weightQuantMode==0, must be {0}"},       {1, "When weightQuantMode==1, must be {0, 2, 3}"},
    {2, "When weightQuantMode==2, must be {0, 1, 3}"}, {3, "When weightQuantMode==3, must be {0, 1, 3}"},
    {4, "When weightQuantMode==4, must be {0, 1, 3}"}, {5, "When weightQuantMode==5, must be {0, 1, 3}"}};

// QUANT_MODE → (weightQuantMode, kvQuantMode) 反向映射表
// 用于从 quantMode_ 反推 wq/kvq，V1/V2（按 dtype 推断 quantMode_）与 V3（正向查表得 quantMode_）通用
// 每个 QUANT_MODE 唯一对应一组 (wq, kvq)，与正向表 QUANT_MODE_HASH_TABLE 互逆
inline const std::unordered_map<QUANT_MODE, std::pair<WEIGHT_QUANT_MODE, KV_QUANT_MODE>> QUANT_MODE_REVERSE_TABLE = {
    {QUANT_MODE::NO_QUANT, {WEIGHT_QUANT_MODE::NO_QUANT, KV_QUANT_MODE::NO_QUANT}},
    {QUANT_MODE::PARTIAL_QUANT_KV_NO_QUANT, {WEIGHT_QUANT_MODE::PARTIAL_QUANT, KV_QUANT_MODE::NO_QUANT}},
    {QUANT_MODE::PARTIAL_QUANT_KV_QUANT_PER_CHANNEL, {WEIGHT_QUANT_MODE::PARTIAL_QUANT, KV_QUANT_MODE::PER_CHANNEL}},
    {QUANT_MODE::PARTIAL_QUANT_KV_QUANT_PER_TILE, {WEIGHT_QUANT_MODE::PARTIAL_QUANT, KV_QUANT_MODE::PER_TILE}},
    {QUANT_MODE::FULL_QUANT_KV_NO_QUANT, {WEIGHT_QUANT_MODE::FULL_QUANT, KV_QUANT_MODE::NO_QUANT}},
    {QUANT_MODE::FULL_QUANT_KV_QUANT_PER_TENSOR, {WEIGHT_QUANT_MODE::FULL_QUANT, KV_QUANT_MODE::PER_TENSOR}},
    {QUANT_MODE::FULL_QUANT_KV_QUANT_PER_TILE, {WEIGHT_QUANT_MODE::FULL_QUANT, KV_QUANT_MODE::PER_TILE}},
    {QUANT_MODE::MXFP8_FULL_QUANT_KV_NO_QUANT, {WEIGHT_QUANT_MODE::MXFP8_FULL_QUANT, KV_QUANT_MODE::NO_QUANT}},
    {QUANT_MODE::MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR,
     {WEIGHT_QUANT_MODE::MXFP8_FULL_QUANT, KV_QUANT_MODE::PER_TENSOR}},
    {QUANT_MODE::MXFP8_FULL_QUANT_KV_QUANT_PER_TILE, {WEIGHT_QUANT_MODE::MXFP8_FULL_QUANT, KV_QUANT_MODE::PER_TILE}},
    {QUANT_MODE::FP8_FULL_QUANT_KV_NO_QUANT, {WEIGHT_QUANT_MODE::FP8_FULL_QUANT, KV_QUANT_MODE::NO_QUANT}},
    {QUANT_MODE::FP8_FULL_QUANT_KV_QUANT_PER_TENSOR, {WEIGHT_QUANT_MODE::FP8_FULL_QUANT, KV_QUANT_MODE::PER_TENSOR}},
    {QUANT_MODE::FP8_FULL_QUANT_KV_QUANT_PER_TILE, {WEIGHT_QUANT_MODE::FP8_FULL_QUANT, KV_QUANT_MODE::PER_TILE}},
    {QUANT_MODE::HIF8_FULL_QUANT_KV_NO_QUANT, {WEIGHT_QUANT_MODE::HIF8_FULL_QUANT, KV_QUANT_MODE::NO_QUANT}},
    {QUANT_MODE::HIF8_FULL_QUANT_KV_QUANT_PER_TENSOR, {WEIGHT_QUANT_MODE::HIF8_FULL_QUANT, KV_QUANT_MODE::PER_TENSOR}},
    {QUANT_MODE::HIF8_FULL_QUANT_KV_QUANT_PER_TILE, {WEIGHT_QUANT_MODE::HIF8_FULL_QUANT, KV_QUANT_MODE::PER_TILE}}};

enum class QUERY_QUANT_MODE : uint8_t {
    NO_QUANT = 0,
    PER_TOKEN_HEAD = 1
};

enum class CKVKR_REPO_MODE : uint8_t {
    DIVIDE = 0,
    COMBINE = 1
};

enum class QUANT_SCALE_REPO_MODE : uint8_t {
    DIVIDE = 0,
    COMBINE = 1
};

struct MlaPrologScenarioInfo {
    bool isV1Flag_;
    bool batchSeqFusedFlag_;
    uint8_t splitMFlag_;
    QUANT_MODE quantMode_;
    CACHE_MODE cacheMode_;
    EMPTY_TENSOR_MODE emptyTensorMode_;
    ACTUAL_SEQ_MODE actualSeqMode_;
    WEIGHT_QUANT_MODE weightQuantMode_;
    KV_QUANT_MODE kvQuantMode_;
};

struct MlaPrologCompileInfo {
    int64_t core_num;
};

struct BaseParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct RequiredParaInfo : BaseParaInfo {};

struct OptionalParaInfo : BaseParaInfo {};

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t NUM_BYTES_BF16 = 2;
constexpr uint32_t NUM_BYTES_INT8 = 1;
constexpr uint32_t NUM_BYTES_INT32 = 4;
constexpr uint32_t NUM_BYTES_FP32 = 4;

constexpr uint32_t GROUP_COMPUTE_CUBE_NUM_PER_GROUP = 8U;
constexpr uint32_t HIGH_THROUGHPUT__D_SIZE = 128U;
constexpr uint32_t GROUP_COMPUTE_T_SIZE = 1U;
constexpr uint32_t GROUP_COMPUTE_NKV_SIZE = 8U;
constexpr uint32_t GROUP_COMPUTE_MIN_AIC_NUM = 16U;
constexpr uint32_t GROUP_COMPUTE_MIN_AIV_NUM = 32U;
constexpr uint32_t GROUP_COMPUTE_N_SIZE = 8U;

struct MlaPrologContext {
    const char *opName;
    const char *opType;
    fe::PlatFormInfos *platformInfo;
    RequiredParaInfo tokenX;
    RequiredParaInfo weightDq;
    RequiredParaInfo weightUqQr;
    RequiredParaInfo weightUk;
    RequiredParaInfo weightDkvKr;
    RequiredParaInfo rmsnormGammaCq;
    RequiredParaInfo rmsnormGammaCkv;
    RequiredParaInfo ropeSin;
    RequiredParaInfo ropeCos;
    RequiredParaInfo kvCache;
    RequiredParaInfo krCache;
    OptionalParaInfo cacheIndex;
    OptionalParaInfo dequantScaleX;
    OptionalParaInfo dequantScaleWDq;
    OptionalParaInfo dequantScaleWUqQr;
    OptionalParaInfo dequantScaleWDkvKr;
    OptionalParaInfo quantScaleCkv;
    OptionalParaInfo quantScaleCkr;
    OptionalParaInfo smoothScalesCq;
    OptionalParaInfo actualSeqLen;
    OptionalParaInfo kNopeClipAlpha;
    RequiredParaInfo query;
    RequiredParaInfo queryRope;
    RequiredParaInfo kvCacheOut;
    RequiredParaInfo krCacheOut;
    OptionalParaInfo dequantScaleQNope;
    OptionalParaInfo queryNorm;
    OptionalParaInfo dequantScaleQNorm;

    const float *rmsNormEspilonCq;
    const float *rmsNormEspilonCkv;
    const char *cacheMode;
    const bool *queryNormFlag;

    const int64_t *weightQuantMode;
    const int64_t *kvQuantMode;
    const int64_t *queryQuantMode;
    const int64_t *ckvkrRepoMode;
    const int64_t *quantScaleRepoMode;
    const int64_t *tileSize;

    const float *qcQrScale;
    const float *kcScale;
    // Inferred in ConvertContext from ropeSin/ropeCos nullity (no OpDef Attr).
    bool doRopeValue = true;
    const bool *doRope = nullptr;

    size_t *workSpaces;
    uint64_t tilingKey;
    uint32_t blockDim;
};

class MlaPrologTiling {
public:
    MlaPrologTiling() = default;
    ~MlaPrologTiling() = default;

    ge::graphStatus RunBigKernelTiling(MlaPrologContext &context, MlaPrologTilingData *tilingData);
    static ge::graphStatus ConvertContext(gert::TilingContext &context, MlaPrologContext &mlaPrologContext);

private:
    static void ConvertRequiredParams(gert::TilingContext &context, MlaPrologContext &mlaPrologContext);
    static void ConvertOptionalParams(gert::TilingContext &context, MlaPrologContext &mlaPrologContext);
    ge::graphStatus GetNpuInfo();
    ge::graphStatus SetScenarioInfo();
    ge::graphStatus SetAttrInfo();
    QUANT_MODE GetQuantizationMode() const;
    QUANT_MODE GetQuantizationModeV3() const;
    QUANT_MODE GetQuantizationModeV3Dav() const;
    ge::graphStatus SetShapeInfo();
    ge::graphStatus ProcessBaseInputs();
    ge::graphStatus FillTiling();
    void FillTilingCoreParams();
    ge::graphStatus FillMatmul1Tiling();
    ge::graphStatus FillMatmul2Tiling();
    ge::graphStatus FillMatmul3Tiling();
    ge::graphStatus FillMatmul4Tiling();
    uint32_t CalcSingleCoreN(uint32_t n, uint32_t coreNum, uint32_t alignNum = 16) const;
    bool GetMatmulType(ge::DataType getype, matmul_tiling::DataType *mmType);
    ge::graphStatus CalcWorkSpace();
    ge::graphStatus GenTilingKey() const;

    NpuArch GetCurNpuArch() const;

    MlaPrologBaseShapeInfo baseShapeInfo_;
    MlaPrologScenarioInfo scenarioInfo_;

    uint32_t stepBatchSize_ = 0;
    uint32_t stepNumHeadDequant_ = 0;
    uint32_t mSubSize_ = 0;
    uint32_t mSubCoreNum_ = 0;

    uint32_t mm1BlockNum_ = 0;
    uint32_t mm2BlockNum_ = 0;
    uint32_t mm3BlockNum_ = 0;
    uint32_t mm4BlockNum_ = 0;
    uint32_t vectorBlockNum_ = 0;

    uint32_t singlecoreHeadSizeCq_ = 0;
    uint32_t singlecoreHeadSizeQcQr_ = 0;
    uint32_t singlecoreHeadSizeCkvKr_ = 0;
    uint32_t singlecoreNumHeadSize_ = 0;

    float reciprocalCq_ = 0.00001f;
    float epsilonCq_ = 1.0f;
    float reciprocalCkv_ = 0.00001f;
    float epsilonCkv_ = 1.0f;
    bool queryNormFlag_ = false;

    WEIGHT_QUANT_MODE weightQuantMode_ = WEIGHT_QUANT_MODE::NO_QUANT;
    KV_QUANT_MODE kvQuantMode_ = KV_QUANT_MODE::NO_QUANT;
    QUERY_QUANT_MODE queryQuantMode_ = QUERY_QUANT_MODE::NO_QUANT;
    CKVKR_REPO_MODE ckvkrRepoMode_ = CKVKR_REPO_MODE::DIVIDE;
    QUANT_SCALE_REPO_MODE quantSacleRepoMode_ = QUANT_SCALE_REPO_MODE::DIVIDE;
    uint32_t tileSize_ = 128;
    float qcQrScale_ = 1.0f;
    float kcScale_ = 1.0f;

    ge::DataType mmDateType_ = ge::DT_BF16;
    bool enableDequantOpt_ = false;
    bool enableGroupComputeOpt_ = false; // 低延时场景算例分组标记
    bool enableRope_ = true;             // rope开关，仅 MlaPrologV3 在 DAV_3510 生效

    size_t ubSize_ = 0;
    size_t l1Size_ = 0;
    size_t l0cSize_ = 0;
    size_t l0bSize_ = 0;
    uint32_t coreNum_ = 0;
    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    size_t libapiSize_ = 0;
    size_t workspaceSize_ = 0;

    MlaPrologContext *context_ = nullptr;
    MlaPrologBaseParams *baseParams_ = nullptr;
};

ge::graphStatus TilingPrepareForMlaProlog(gert::TilingParseContext *context);
MLA_EXTERN_C ge::graphStatus TilingMlaProlog(gert::TilingContext *context);
} // namespace optiling

#endif // MLA_PROLOG_TILING_H
