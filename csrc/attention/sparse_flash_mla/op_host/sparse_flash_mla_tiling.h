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
 * \file sparse_flash_mla_tiling.h
 * \brief
 */
#ifndef SPARSE_FLASH_MLA_TILING_H
#define SPARSE_FLASH_MLA_TILING_H

#include <graph/utils/type_utils.h>
#include <exe_graph/runtime/tiling_context.h>
#include <tiling/platform/platform_ascendc.h>
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "platform/platform_info.h"
#include "platform/soc_spec.h"

namespace optiling {
// ------------------公共定义--------------------------
struct SMLATilingRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct SMLATilingOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

enum class SMLALayout : uint32_t {
    BSND = 0,
    TND = 1,
    PA_BBND = 2
};

enum class SMLAAxis : uint32_t {
    B = 0,
    S = 1,
    N = 2,
    D = 3,
    K = 3, // sparse_indices的K和key的D枚举值相同，表达相同位置, 最后一维
    T = 5,
    Bn = 6, // block number
    Bs = 7  // block size
};

enum class SMLATemplateMode : uint32_t {
    SWA_TEMPLATE_MODE = 0,
    HCA_TEMPLATE_MODE = 1,
    CSA_TEMPLATE_MODE = 2,
    ORI_SPARSE_TEMPLATE_MODE = 3,
    ORI_CMP_SPARSE_TEMPLATE_MODE = 4
};

enum class KvStorageMode : uint32_t {
    BATCH_CONTINUOUS = 0,
    TENSOR_LIST = 1,
    PAGE_ATTENTION = 2
};

// ------------------算子原型索引常量定义----------------
// Inputs Index (0-10, common)
constexpr uint32_t Q_INDEX = 0;
constexpr uint32_t ORI_KV_INDEX = 1;
constexpr uint32_t CMP_KV_INDEX = 2;
constexpr uint32_t ORI_SPARSE_INDICES_INDEX = 3;
constexpr uint32_t CMP_SPARSE_INDICES_INDEX = 4;
constexpr uint32_t ORI_BLOCK_TABLE_INDEX = 5;
constexpr uint32_t CMP_BLOCK_TABLE_INDEX = 6;
constexpr uint32_t CU_SEQLENS_Q_INDEX = 7;
constexpr uint32_t CU_SEQLENS_ORI_KV_INDEX = 8;
constexpr uint32_t CU_SEQLENS_CMP_KV_INDEX = 9;
constexpr uint32_t SEQUSED_Q_INDEX = 10;
constexpr uint32_t SEQUSED_ORI_KV_INDEX = 11;
constexpr uint32_t SEQUSED_CMP_KV_INDEX = 12;
constexpr uint32_t CMP_RESIDUAL_KV_INDEX = 13;
constexpr uint32_t ORI_TOPK_LENGTH_INDEX = 14;
constexpr uint32_t CMP_TOPK_LENGTH_INDEX = 15;
constexpr uint32_t SINKS_INDEX = 16;
constexpr uint32_t METADATA_INDEX = 17;
// Outputs Index
constexpr uint32_t ATTN_OUT_INDEX = 0;

// Attributes Index
constexpr uint32_t ATTR_SOFTMAX_SCALE_INDEX = 0;
constexpr uint32_t ATTR_CMP_RATIO_INDEX = 1;
constexpr uint32_t ATTR_ORI_MASK_MODE_INDEX = 2;
constexpr uint32_t ATTR_CMP_MASK_MODE_INDEX = 3;
constexpr uint32_t ATTR_ORI_WIN_LEFT_INDEX = 4;
constexpr uint32_t ATTR_ORI_WIN_RIGHT_INDEX = 5;
constexpr uint32_t ATTR_LAYOUT_Q_INDEX = 6;
constexpr uint32_t ATTR_LAYOUT_KV_INDEX = 7;
constexpr uint32_t ATTR_TOPK_VALUE_MODE_INDEX = 8; // A2/A3
constexpr uint32_t ATTR_RETURN_SOFTMAX_LSE_INDEX = 9;

// Dim Index
constexpr uint32_t DIM_IDX_ONE = 1;
constexpr uint32_t DIM_IDX_TWO = 2;
constexpr uint32_t DIM_IDX_THREE = 3;
constexpr uint32_t DIM_IDX_FOUR = 4;

// Dim Num
constexpr uint32_t DIM_NUM_ONE = 1;
constexpr uint32_t DIM_NUM_TWO = 2;
constexpr uint32_t DIM_NUM_THREE = 3;
constexpr uint32_t DIM_NUM_FOUR = 4;

// 常量
constexpr uint32_t MAX_BLOCK_SIZE = 1024;
constexpr uint32_t COPYND2NZ_SRC_STRIDE_LIMITATION = 65535;
constexpr uint32_t NUM_BYTES_FLOAT = 4;
constexpr uint32_t NUM_BYTES_FLOAT16 = 2;
constexpr uint32_t NUM_BYTES_BF16 = 2;
constexpr uint32_t BYTE_BLOCK = 32;

// 入参限制常量
constexpr uint32_t HEAD_DIM_LIMIT = 128;
constexpr uint32_t SPARSE_LIMIT = 2048;
constexpr uint32_t SPARSE_MODE_LOWER = 3;
constexpr uint32_t METADATA_LIMIT = 1024;
constexpr uint32_t DIM_LIMIT = 512;
constexpr uint32_t TOPK_LIMIT = 1024;
constexpr uint32_t BLOCK_SIZE_LIMIT = 1024;

// -----------算子TilingData定义（A2/A3字段顺序 + A5追加字段）---------------
BEGIN_TILING_DATA_DEF(SparseFlashMlaSwaParams)
TILING_DATA_FIELD_DEF(uint32_t, batchSize)
TILING_DATA_FIELD_DEF(uint32_t, qSeqSize)
TILING_DATA_FIELD_DEF(uint32_t, kvSeqSize)
TILING_DATA_FIELD_DEF(int64_t, paBlockSize)
TILING_DATA_FIELD_DEF(int64_t, oriBlockSize)
TILING_DATA_FIELD_DEF(int64_t, cmpBlockSize)
TILING_DATA_FIELD_DEF(uint32_t, oriMaxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, nNumOfQInOneGroup)
TILING_DATA_FIELD_DEF(uint32_t, actualLenDimsQ)
TILING_DATA_FIELD_DEF(uint32_t, actualLenDimsKV)
TILING_DATA_FIELD_DEF(float, softmaxScale) // 即 scaleValue
TILING_DATA_FIELD_DEF(uint32_t, outputLayout)
TILING_DATA_FIELD_DEF(uint64_t, oriMaskMode)
TILING_DATA_FIELD_DEF(int64_t, oriKvStride0) // A2/A3
TILING_DATA_FIELD_DEF(int64_t, oriWinLeft)
TILING_DATA_FIELD_DEF(int64_t, oriWinRight)
TILING_DATA_FIELD_DEF(int64_t, sparseBlockSize)
TILING_DATA_FIELD_DEF(uint32_t, oriSparseBlockCount) // A5

TILING_DATA_FIELD_DEF(int64_t, topkValueMode)

TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);

TILING_DATA_FIELD_DEF(uint32_t, mmResUbSize);
TILING_DATA_FIELD_DEF(uint32_t, bmm2ResUbSize);
TILING_DATA_FIELD_DEF(uint32_t, returnSoftmaxLse)
TILING_DATA_FIELD_DEF(uint32_t, mBaseSize)
TILING_DATA_FIELD_DEF(uint32_t, s2BaseSize)

TILING_DATA_FIELD_DEF(uint32_t, actualLenDimsOriKV)
TILING_DATA_FIELD_DEF(uint32_t, actualLenDimsCmpKV)
TILING_DATA_FIELD_DEF(uint32_t, cmpResidualKVSize)
TILING_DATA_FIELD_DEF(uint32_t, kvHeadNum)
TILING_DATA_FIELD_DEF(uint32_t, oriKeyStride0) // A5
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(SparseFlashMlaSwaParamsOp, SparseFlashMlaSwaParams)

BEGIN_TILING_DATA_DEF(SparseFlashMlaCmpParams)
TILING_DATA_FIELD_DEF(uint32_t, cmpMaxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, sparseBlockCount) // A2/A3
TILING_DATA_FIELD_DEF(int64_t, cmpRatio)
TILING_DATA_FIELD_DEF(uint64_t, cmpMaskMode)
TILING_DATA_FIELD_DEF(int64_t, cmpKvStride0)         // A2/A3
TILING_DATA_FIELD_DEF(uint32_t, cmpSparseBlockCount) // A5
TILING_DATA_FIELD_DEF(uint32_t, cmpKvSeqSize)        // A5
TILING_DATA_FIELD_DEF(uint32_t, cmpKeyStride0)       // A5
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(SparseFlashMlaCmpParamsOp, SparseFlashMlaCmpParams)

BEGIN_TILING_DATA_DEF(SparseFlashMlaTilingData)
TILING_DATA_FIELD_DEF_STRUCT(SparseFlashMlaSwaParams, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(SparseFlashMlaCmpParams, cmpParams);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(SparseFlashMla, SparseFlashMlaTilingData)

struct SMLAParaInfo {
    SMLATilingRequiredParaInfo q = {nullptr, nullptr};
    SMLATilingOptionalParaInfo oriKv = {nullptr, nullptr};
    SMLATilingOptionalParaInfo cmpKv = {nullptr, nullptr};
    SMLATilingOptionalParaInfo oriSparseIndices = {nullptr, nullptr};
    SMLATilingOptionalParaInfo cmpSparseIndices = {nullptr, nullptr};
    SMLATilingOptionalParaInfo oriBlockTable = {nullptr, nullptr};
    SMLATilingOptionalParaInfo cmpBlockTable = {nullptr, nullptr};
    SMLATilingOptionalParaInfo cuSeqLensQ = {nullptr, nullptr};
    SMLATilingOptionalParaInfo cuSeqLensOriKv = {nullptr, nullptr}; // A5
    SMLATilingOptionalParaInfo cuSeqLensCmpKv = {nullptr, nullptr}; // A5
    SMLATilingOptionalParaInfo cuSeqLensKv = {nullptr, nullptr};    // A2/A3
    SMLATilingOptionalParaInfo seqUsedQ = {nullptr, nullptr};
    SMLATilingOptionalParaInfo sequsedOriKv = {nullptr, nullptr};
    SMLATilingOptionalParaInfo sequsedCmpKv = {nullptr, nullptr};
    SMLATilingOptionalParaInfo cmpResidualKv = {nullptr, nullptr};
    SMLATilingOptionalParaInfo oriTopkLength = {nullptr, nullptr};
    SMLATilingOptionalParaInfo cmpTopkLength = {nullptr, nullptr};
    SMLATilingOptionalParaInfo sinks = {nullptr, nullptr};
    SMLATilingOptionalParaInfo metadata = {nullptr, nullptr};
    SMLATilingRequiredParaInfo attnOut = {nullptr, nullptr};

    const float *softmaxScale = nullptr;
    const uint32_t *cmpRatio = nullptr;
    const uint32_t *oriMaskMode = nullptr;
    const uint32_t *cmpMaskMode = nullptr;
    const int32_t *oriWinLeft = nullptr;
    const int32_t *oriWinRight = nullptr;
    const char *layoutQ = nullptr;
    const char *layoutKv = nullptr;
    const uint32_t *topkValueMode = nullptr;
    const bool *returnSoftmaxLse = nullptr;
};

static std::string SMLADataTypeToSerialString(ge::DataType type);
std::string SMLALayoutToSerialString(SMLALayout layout);

// -----------算子Tiling入参信息类---------------
class SMLATilingInfo {
public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    SMLAParaInfo opParamInfo;

    // Base Param
    NpuArch npuArch = NpuArch::DAV_2201;
    uint32_t bSize = 0;
    uint32_t n1Size = 0;
    uint32_t n2Size = 0;
    uint32_t s1Size = 0;
    int64_t s2Size = 0;
    int64_t cmpS2Size = 0; // A5
    uint32_t gSize = 0;
    uint32_t qHeadDim = 0;
    uint32_t oriKvHeadDim = 0;
    uint32_t cmpKvHeadDim = 0;
    uint32_t qTSize = 0; // 仅TND时生效

    uint32_t actualLenDimsQ = 0;
    uint32_t maxActualseq = 0;
    bool actualSeqLenFlag = false;
    bool isSameSeqAllKVTensor = true;
    bool isSameActualseq = true;
    uint32_t actualLenDimsKV = 0;

    uint32_t actualLenDimsOriKV = 0;
    uint32_t actualLenDimsCmpKV = 0;
    uint32_t cmpResidualKVSize = 0;

    uint32_t oriKeyStride0 = 0; // A5
    uint32_t cmpKeyStride0 = 0; // A5

    float softmaxScale = 0;
    int64_t cmpRatio = 1;
    uint64_t oriMaskMode = 0;
    uint64_t cmpMaskMode = 0;
    uint64_t oriKvStride0 = 0; // A2/A3
    uint64_t cmpKvStride0 = 0; // A2/A3
    int64_t oriWinLeft = 0;
    int64_t oriWinRight = 0;
    int64_t sparseBlockSize = 0;
    int64_t oriSparseBlockCount = 0;
    int64_t cmpSparseBlockCount = 0;
    int64_t sparseBlockCount = 0; // A2/A3

    int64_t topkValueMode = 0;
    // Mask
    int32_t sparseMode = 0;
    // Others Flag
    uint32_t sparseCount = 0;
    bool returnSoftmaxLse = false;

    // PageAttention
    uint32_t blockTypeSize = 0;
    uint32_t oriMaxBlockNumPerBatch = 0;
    int32_t blockSize = 0;
    int32_t oriBlockSize = 0;
    int32_t cmpBlockSize = 0;
    uint32_t cmpMaxBlockNumPerBatch = 0;
    uint32_t totalBlockNum = 0;

    // DType
    ge::DataType qType = ge::DT_FLOAT16;
    ge::DataType oriKvType = ge::DT_FLOAT16;
    ge::DataType cmpKvType = ge::DT_FLOAT16;
    ge::DataType outputType = ge::DT_FLOAT16;

    // Layout
    SMLALayout qLayout = SMLALayout::TND;
    SMLALayout cmpSparseIndicesLayout = SMLALayout::TND;
    SMLALayout oriSparseIndicesLayout = SMLALayout::TND;
    SMLALayout kvLayout = SMLALayout::PA_BBND;
    SMLALayout outLayout = SMLALayout::BSND;

    // template mode
    SMLATemplateMode perfMode = SMLATemplateMode::SWA_TEMPLATE_MODE;
};

// -----------算子Tiling入参信息解析及Check类---------------
class SMLATilingCheck {
public:
    explicit SMLATilingCheck(const SMLATilingInfo &smlaInfo) : smlaInfo_(smlaInfo) {};
    ~SMLATilingCheck() = default;
    virtual ge::graphStatus Process();

private:
    void Init();

    void LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList, const ge::DataType &actualDtype,
                              const std::string &name) const;
    ge::graphStatus CheckLayoutSupport(const SMLALayout &actualLayout, const std::string &name) const;
    template <typename T>
    void LogErrorDimNumSupport(const std::vector<T> &expectNumberList, const T &actualValue,
                               const std::string &name) const;
    template <typename T>
    void LogErrorNumberSupport(const std::vector<T> &expectNumberList, const T &actualValue, const std::string &name,
                               const std::string subName) const;
    ge::graphStatus CheckDimNumSupport(const gert::StorageShape *shape, const std::vector<size_t> &expectDimNumList,
                                       const std::string &name) const;
    void LogErrorLayoutSupport(const std::vector<SMLALayout> &expectLayoutList, const SMLALayout &actualLayout,
                               const std::string &name) const;
    ge::graphStatus CheckDimNumInLayoutSupport(const SMLALayout &layout, const gert::StorageShape *shape,
                                               const std::string &name) const;
    ge::graphStatus CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc, const std::string &name) const;
    ge::graphStatus CheckSinglePara() const;
    ge::graphStatus CheckSingleParaQuery() const;
    ge::graphStatus CheckSingleParaOriKv() const;
    ge::graphStatus CheckSingleParaCmpKv() const;
    ge::graphStatus CheckSingleParaCuSeqLensQ() const;
    ge::graphStatus CheckSingleParaCuSeqLensOriKv() const;
    ge::graphStatus CheckSingleParaCuSeqLensCmpKv() const;
    ge::graphStatus CheckSingleParaCmpResidualKv() const;
    ge::graphStatus CheckSingleParaTopkLength() const;
    ge::graphStatus CheckSingleParaNumHeads() const;
    ge::graphStatus CheckSingleParaKvHeadNums() const;
    ge::graphStatus CheckSingleParaOriSparseIndices() const;
    ge::graphStatus CheckSingleParaCmpSparseIndices() const;
    ge::graphStatus CheckSingleParaSinks() const;
    ge::graphStatus CheckSingleParaMetadata() const;
    ge::graphStatus CheckSingleParaCmpRatio() const;
    ge::graphStatus CheckSingleParaOriMaskMode() const;
    ge::graphStatus CheckSingleParaCmpMaskMode() const;
    ge::graphStatus CheckSingleParaOriKvStride0() const; // A2/A3
    ge::graphStatus CheckSingleParaCmpKvStride0() const; // A2/A3
    ge::graphStatus CheckSingleParaOriWinLeft() const;
    ge::graphStatus CheckSingleParaOriWinRight() const;
    ge::graphStatus CheckSingleParaOriBlockTable() const;
    ge::graphStatus CheckSingleParaCmpBlockTable() const;

    ge::graphStatus CheckParaExistence() const;
    ge::graphStatus CheckExists(const void *pointer, const std::string &name) const;
    ge::graphStatus CheckNotExists(const void *pointer, const std::string &name) const;
    ge::graphStatus CheckExistsByMap(const std::map<std::string, const void *> &paramMap) const;
    ge::graphStatus CheckNotExistsByMap(const std::map<std::string, const void *> &paramMap) const;
    ge::graphStatus CheckExistenceByMap(std::map<std::string, const void *> &existMap,
                                        std::map<std::string, const void *> &notExistMap) const;

    ge::graphStatus CheckFeature() const;
    ge::graphStatus CheckFeatureShape() const;
    ge::graphStatus CheckFeatureLayout() const;
    ge::graphStatus CheckFeatureDtype() const;
    ge::graphStatus CheckFeaturePa() const;

    ge::graphStatus CheckMultiParaConsistency();
    void SetSMLAShapeCompare();
    ge::graphStatus CheckDTypeConsistency(const ge::DataType &actualDtype, const ge::DataType &expectDtype,
                                          const std::string &name) const;
    ge::graphStatus CheckOriAndCmpKv() const;
    ge::graphStatus CheckAttenOut() const;
    ge::graphStatus CheckActualSeqLensQ() const;
    ge::graphStatus CheckActualSeqLens() const;
    ge::graphStatus CheckBlockTable() const;

    gert::Shape queryShapeCmp_{};
    gert::Shape oriKvShapeCmp_{};
    gert::Shape cmpKvShapeCmp_{};
    gert::Shape oriKvSparseIndicesCmp_{};
    gert::Shape cmpKvSparseIndicesCmp_{};
    gert::Shape attenOutShapeCmp_{};

    const char *opName_;
    fe::PlatFormInfos *platformInfo_;
    SMLAParaInfo opParamInfo_;
    const SMLATilingInfo &smlaInfo_;

    uint32_t bSize_ = 0;
    uint32_t n1Size_ = 0;
    uint32_t n2Size_ = 0;
    uint32_t gSize_ = 0;
    uint32_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    int64_t cmpS2Size_ = 0; // A5
    uint32_t qHeadDim_ = 0;
    uint32_t oriKvHeadDim_ = 0;
    uint32_t cmpKvHeadDim_ = 0;

    uint32_t qTSize_ = 0;  // 仅TND时生效
    uint32_t kvTSize_ = 0; // 仅TND时生效
    int64_t cmpRatio_ = 1;
    KvStorageMode kvStorageMode_ = KvStorageMode::BATCH_CONTINUOUS;
    uint32_t oriSparseBlockCount_ = 0; // A5
    uint32_t cmpSparseBlockCount_ = 0; // A5
    uint32_t sparseBlockCount_ = 0;    // A2/A3
    int64_t oriWinLeft_ = 0;
    int64_t oriWinRight_ = 0;

    int64_t topkValueMode_;

    SMLALayout qLayout_ = SMLALayout::TND;
    SMLALayout cmpSparseIndicesLayout_ = SMLALayout::TND;
    SMLALayout oriSparseIndicesLayout_ = SMLALayout::TND;
    SMLALayout outLayout_ = SMLALayout::TND;
    SMLALayout kvLayout_ = SMLALayout::PA_BBND;

    uint32_t oriMaxBlockNumPerBatch_ = 0;
    uint32_t cmpMaxBlockNumPerBatch_ = 0;
    int64_t blockSize_ = 0;
    int32_t oriBlockSize_ = 0;
    int32_t cmpBlockSize_ = 0;

    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    NpuArch npuArch_ = NpuArch::DAV_2201;
    uint64_t l2CacheSize_ = 0;

    bool isSameSeqAllKVTensor_ = true;
    bool isSameActualseq_ = true;
    uint32_t maxActualseq_ = 0;

    ge::DataType qType_ = ge::DT_FLOAT16;
    ge::DataType oriKvType_ = ge::DT_FLOAT16;
    ge::DataType cmpKvType_ = ge::DT_FLOAT16;
    ge::DataType outputType_ = ge::DT_FLOAT16;
};

template <typename T>
inline T Align(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

class SMLAInfoParser {
public:
    explicit SMLAInfoParser(gert::TilingContext *context) : context_(context)
    {
    }
    ~SMLAInfoParser() = default;

    ge::graphStatus CheckRequiredInOutExistence() const;
    ge::graphStatus CheckRequiredAttrExistence() const;
    ge::graphStatus CheckRequiredParaExistence() const;
    ge::graphStatus CheckUnrequiredParaExistence() const;

    ge::graphStatus GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor, SMLALayout &layout,
                                        const std::string &name) const;
    ge::graphStatus GetActualSeqLenQSize(uint32_t &size);
    ge::graphStatus GetOpName();
    ge::graphStatus GetNpuInfo();
    void GetOptionalInputParaInfo();
    void GetInputParaInfo();
    void GetOutputParaInfo();
    ge::graphStatus GetAttrParaInfo();
    ge::graphStatus GetKvCache();
    ge::graphStatus GetOpParaInfo();

    ge::graphStatus GetInOutDataType();
    ge::graphStatus GetQueryAndOutLayout();
    ge::graphStatus GetKvLayout();
    ge::graphStatus GetSMLATemplateMode(SMLATilingInfo &smlaInfo);
    void SetSMLAShape();
    ge::graphStatus GetN1Size();
    ge::graphStatus GetN2Size();
    ge::graphStatus GetGSize();
    ge::graphStatus GetBatchSize();
    ge::graphStatus GetQTSize();
    ge::graphStatus GetKVTSize(); // A2/A3
    ge::graphStatus GetS1Size();
    ge::graphStatus GetS2SizeForPageAttention();
    ge::graphStatus GetS2SizeForTND(); // A2/A3
    ge::graphStatus GetS2Size();
    ge::graphStatus GetMaxBlockNumPerBatch();
    ge::graphStatus GetBlockSize();
    ge::graphStatus GetQHeadDim();
    ge::graphStatus GetValueHeadDim();
    ge::graphStatus GetSparseBlockCount();
    ge::graphStatus GetActualseqInfo();
    ge::graphStatus GetSinks();
    uint64_t GetOptionalInputStride0(uint32_t inputIndex) const;
    void GenerateInfo(SMLATilingInfo &smlaInfo);
    ge::graphStatus Parse(SMLATilingInfo &smlaInfo);
    ge::graphStatus CheckContiguous() const; // A5

    gert::TilingContext *context_ = nullptr;
    const char *opName_;
    fe::PlatFormInfos *platformInfo_;
    SMLAParaInfo opParamInfo_;

    bool HasAxis(const SMLAAxis &axis, const SMLALayout &layout, const gert::Shape &shape) const;
    size_t GetAxisIdx(const SMLAAxis &axis, const SMLALayout &layout) const;
    uint32_t GetAxisNum(const gert::Shape &shape, const SMLAAxis &axis, const SMLALayout &layout) const;
    static constexpr int64_t invalidDimValue_ = std::numeric_limits<int64_t>::min();

    // BaseParams
    uint32_t bSize_ = 0;
    uint32_t n1Size_ = 0;
    uint32_t n2Size_ = 0;
    uint32_t gSize_ = 0;
    uint32_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    int64_t cmpS2Size_ = 0; // A5
    uint32_t headDim_ = 0;
    uint32_t qTSize_ = 0;
    uint32_t orikvTSize_ = 0; // A2/A3
    uint32_t cmpkvTSize_ = 0; // A2/A3
    uint32_t qHeadDim_ = 0;
    uint32_t oriKvHeadDim_ = 0;
    uint32_t cmpKvHeadDim_ = 0;
    int64_t sparseBlockSize_ = 0;
    int64_t oriSparseBlockCount_ = 0; // A5
    int64_t cmpSparseBlockCount_ = 0; // A5
    int64_t sparseBlockCount_ = 0;    // A2/A3
    int64_t oriWinLeft_ = 0;
    int64_t oriWinRight_ = 0;
    int64_t topkValueMode_ = 0;
    uint32_t maxActualseq_ = 0;
    bool isSameSeqAllKVTensor_ = true;
    uint32_t actualLenDimsKV_ = 0;
    uint32_t actualLenDimsQ_ = 0;

    uint32_t actualLenDimsOriKV_ = 0;
    uint32_t actualLenDimsCmpKV_ = 0;
    uint32_t cmpResidualKVSize_ = 0;

    std::vector<uint32_t> oriKeyStridesVec_; // A5
    std::vector<uint32_t> cmpKeyStridesVec_; // A5

    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    // Layout
    SMLALayout qLayout_ = SMLALayout::TND;
    SMLALayout cmpSparseIndicesLayout_ = SMLALayout::TND;
    SMLALayout oriSparseIndicesLayout_ = SMLALayout::TND;
    SMLALayout outLayout_ = SMLALayout::BSND;
    SMLALayout kvLayout_ = SMLALayout::PA_BBND;
    // PageAttention
    uint32_t oriMaxBlockNumPerBatch_ = 0;
    uint32_t cmpMaxBlockNumPerBatch_ = 0;
    int32_t oriBlockSize_ = 0;
    int32_t cmpBlockSize_ = 0;

    // template mode
    SMLATemplateMode perfMode_ = SMLATemplateMode::SWA_TEMPLATE_MODE;

    NpuArch npuArch_ = NpuArch::DAV_2201;
    ge::DataType qType_ = ge::DT_FLOAT16;
    ge::DataType oriKvType_ = ge::DT_FLOAT16;
    ge::DataType cmpKvType_ = ge::DT_FLOAT16;
    ge::DataType cmpSparseIndicesType_ = ge::DT_INT32;
    ge::DataType oriBlockTableType_ = ge::DT_INT32;
    ge::DataType cmpBlockTableType_ = ge::DT_INT32;
    ge::DataType cuSeqLensQType_ = ge::DT_INT32;
    ge::DataType seqsedKvType_ = ge::DT_INT32;
    ge::DataType sinksType_ = ge::DT_INT32;
    ge::DataType metadataType_ = ge::DT_INT32;
    ge::DataType outputType_ = ge::DT_FLOAT16;

    gert::Shape qShape_{};
    gert::Shape oriKvShape_{};
    gert::Shape cmpKvShape_{};
    gert::Shape oriSparseIndicesShape_{};
    gert::Shape cmpSparseIndicesShape_{};
};

// ---------------算子Tiling类---------------
class SparseFlashMlaTiling {
public:
    explicit SparseFlashMlaTiling(gert::TilingContext *context) : context_(context) {};
    ge::graphStatus DoOpTiling(SMLATilingInfo *tilingInfo);

private:
    void SplitBalanced(SMLATilingInfo *tilingInfo);
    void CalcUbBmm(SMLATilingInfo *tilingInfo);
    uint32_t CalcFdLogicalSlotCount(const SMLATilingInfo *tilingInfo, uint32_t aicNum) const;
    uint64_t CalcFdStagingWorkspaceSize(const SMLATilingInfo *tilingInfo, uint32_t aicNum) const;
    gert::TilingContext *context_ = nullptr;
    SMLATemplateMode perfMode_ = SMLATemplateMode::SWA_TEMPLATE_MODE;
    SparseFlashMlaTilingData tilingData_;
    uint32_t blockDim_{0};
    uint64_t workspaceSize_{0};
    uint64_t tilingKey_{0};

    SMLATilingInfo *smlaInfo_ = nullptr;

    size_t mmResUbSize_ = 0;
    size_t bmm2ResUbSize_ = 0;
    uint32_t sInnerLoopTimes_ = 0;
    uint32_t sInnerSize_ = 512; // s2固定切分512
    uint32_t sInnerSizeAlign_ = 0;
    uint32_t usedCoreNum_ = 0;

    uint32_t headDimAlign_ = 0;
    uint32_t mBaseSize_ = 64;
};

} // namespace optiling
#endif
