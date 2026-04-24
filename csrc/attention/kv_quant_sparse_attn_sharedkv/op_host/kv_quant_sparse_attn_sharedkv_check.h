/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kv_quant_sparse_attn_sharedkv_check.h
 * \brief
 */
#ifndef KV_QUANT_SPARSE_ATTN_SHAREDKV_CHECK_H
#define KV_QUANT_SPARSE_ATTN_SHAREDKV_CHECK_H

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

namespace optiling {

const std::string ORI_BLOCK_TABLE_NAME = "ori_block_table";
const std::string CMP_BLOCK_TABLE_NAME = "cmp_block_table";
const std::string SINKS_NAME = "sinks";

const std::string QUERY_NAME = "query";
const std::string KEY_NAME = "key";
const std::string VALUE_NAME = "value";

const std::string ORI_KV_NAME = "ori_kv";
const std::string CMP_KV_NAME = "cmp_kv";
const std::string ORI_SPARSE_INDICES_NAME = "ori_sparse_indices";
const std::string CMP_SPARSE_INDICES_NAME = "cmp_sparse_indices";
const std::string ATTEN_OUT_NAME = "attention_out";

const std::string CU_SEQLENS_Q_NAME = "cu_seqlens_q";
const std::string SEQUSED_KV_NAME = "seqused_kv";

// // ------------------公共定义--------------------------
struct SASTilingRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct SASTilingOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

enum class SASLayout : uint32_t {
    BSND = 0,
    TND = 1,
    PA_ND = 2
};

enum class SASAxis : uint32_t {
    B = 0,
    S = 1,
    N = 2,
    D = 3,
    K = 3,  // sparse_indices的K和key的D枚举值相同，表达相同位置, 最后一维
    T = 5,
    Bn = 6, // block number
    Bs = 7 // block size
};

enum class SASTemplateMode : uint32_t {
    SWA_TEMPLATE_MODE = 0,
    CFA_TEMPLATE_MODE = 1,
    SCFA_TEMPLATE_MODE = 2
};

enum class KvStorageMode : uint32_t {
    BATCH_CONTINUOUS = 0,
    TENSOR_LIST = 1,
    PAGE_ATTENTION = 2
};

struct KvQuantSASTilingShapeCompareParam {
    int64_t B = 1;
    int64_t S = 1;
    int64_t N = 1;
    int64_t D = 1;
    int64_t T = 1;
    // PA
    int64_t Bs = 1;
    int64_t Bn = 1;
};

// ------------------算子原型索引常量定义----------------
// Inputs Index
constexpr uint32_t Q_INDEX = 0;
constexpr uint32_t ORI_KV_INDEX = 1;
constexpr uint32_t CMP_KV_INDEX = 2;
constexpr uint32_t CMP_SPARSE_INDICES_INDEX = 4;
constexpr uint32_t ORI_BLOCK_TABLE_INDEX = 5;
constexpr uint32_t CMP_BLOCK_TABLE_INDEX = 6;
constexpr uint32_t CU_SEQLENS_Q_INDEX = 7;
constexpr uint32_t SEQUSED_KV_INDEX = 11;
constexpr uint32_t SINKS_INDEX = 12;
constexpr uint32_t METADATA_INDEX = 13;
// Outputs Index
constexpr uint32_t ATTN_OUT_INDEX = 0;

// Attributes Index
constexpr uint32_t ATTR_KV_QUANT_SCALE_INDEX = 0;
constexpr uint32_t ATTR_TILE_SIZE_INDEX = 1;
constexpr uint32_t ATTR_ROPE_HEAD_DIM_INDEX = 2;
constexpr uint32_t ATTR_SOTFMAX_SCALE_INDEX = 3;
constexpr uint32_t ATTR_CMP_RATIO_INDEX = 4;
constexpr uint32_t ATTR_ORI_MASK_MODE_INDEX = 5;
constexpr uint32_t ATTR_CMP_MASK_MODE_INDEX = 6;
constexpr uint32_t ATTR_ORI_WIN_LEFT_INDEX = 7;
constexpr uint32_t ATTR_ORI_WIN_RIGHT_INDEX = 8;
constexpr uint32_t ATTR_LAYOUT_Q_INDEX = 9;
constexpr uint32_t ATTR_LAYOUT_KV_INDEX = 10;

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

// 入参限制常量
constexpr uint32_t HEAD_DIM_LIMIT = 128;
constexpr uint32_t SPARSE_LIMIT = 2048;
constexpr uint32_t SPARSE_MODE_LOWER = 3;
constexpr uint32_t MAX_BLOCK_SIZE = 1024;
constexpr uint32_t COPYND2NZ_SRC_STRIDE_LIMITATION = 65535;
constexpr uint32_t NUM_BYTES_FLOAT = 4;
constexpr uint32_t NUM_BYTES_FLOAT16 = 2;
constexpr uint32_t NUM_BYTES_BF16 = 2;
constexpr uint32_t BYTE_BLOCK = 32;
const uint32_t QSFA_MAX_AIC_CORE_NUM = 26; // 25 + 1 保证数组8字节对齐

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

const std::map<SASLayout, size_t> SAS_LAYOUT_DIM_MAP = {
    {SASLayout::BSND, DIM_NUM_FOUR},
    {SASLayout::TND, DIM_NUM_THREE},
    {SASLayout::PA_ND, DIM_NUM_FOUR},
};

std::string SASDataTypeToSerialString(ge::DataType type);
std::string KvQuantSASLayoutToSerialString(SASLayout layout);
std::string GetShapeStr(gert::Shape shape);

// -----------算子Tiling入参信息解析及Check类---------------

class KvQuantSASTilingInfo;

struct KvQuantSASParaInfo {
    SASTilingRequiredParaInfo q = {nullptr, nullptr};
    SASTilingOptionalParaInfo oriKv = {nullptr, nullptr};
    SASTilingOptionalParaInfo cmpKv = {nullptr, nullptr};
    SASTilingOptionalParaInfo cmpSparseIndices = {nullptr, nullptr};
    SASTilingOptionalParaInfo oriBlockTable = {nullptr, nullptr};
    SASTilingOptionalParaInfo cmpBlockTable = {nullptr, nullptr};
    SASTilingOptionalParaInfo cuSeqLensQ = {nullptr, nullptr};
    SASTilingOptionalParaInfo sequsedKv = {nullptr, nullptr};
    SASTilingOptionalParaInfo sinks = {nullptr, nullptr};
    SASTilingOptionalParaInfo metadata = {nullptr, nullptr};
    SASTilingRequiredParaInfo attnOut = {nullptr, nullptr};

    const int64_t *kvQuantMode = nullptr;
    const int64_t *tileSize = nullptr;
    const int64_t *ropeHeadDim = nullptr;
    const float *softmaxScale = nullptr;
    const uint32_t *cmpRatio = nullptr;
    const uint32_t *oriMaskMode = nullptr;
    const uint32_t *cmpMaskMode = nullptr;
    const uint32_t *oriWinLeft = nullptr;
    const uint32_t *oriWinRight = nullptr;
    const char *layoutQ = nullptr;
    const char *layoutKv = nullptr;
};

// -----------算子Tiling入参信息类---------------
class KvQuantSASTilingInfo {
public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    KvQuantSASParaInfo opParamInfo;

    // Base Param
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
    uint32_t bSize = 0;
    uint32_t n1Size = 0;
    uint32_t n2Size = 0;
    uint32_t s1Size = 0;
    int64_t s2Size = 0;
    uint32_t gSize = 0;
    uint32_t qkHeadDim = 0;
    uint32_t qTSize = 0; // 仅TND时生效

    uint32_t actualLenDimsQ = 0;
    uint32_t maxActualseq = 0;
    bool actualSeqLenFlag = false;
    bool isSameSeqAllKVTensor = true;
    bool isSameActualseq = true;
    uint32_t actualLenDimsKV = 0;

    int64_t kvQuantMode = 0;
    int64_t tileSize = 0;
    int64_t ropeHeadDim = 0;
    uint32_t dSize = 0;
    uint32_t dSizeV = 0;
    uint32_t dSizeVInput = 0;
    float softmaxScale = 0;
    int64_t cmpRatio = 0;
    uint64_t oriMaskMode = 0;
    uint64_t cmpMaskMode = 0;
    int64_t oriWinLeft = 0;
    int64_t oriWinRight = 0;
    int64_t sparseBlockSize = 0;
    int64_t sparseBlockCount = 0;
    // Mask
    int32_t sparseMode = 0;
    // Others Flag
    uint32_t sparseCount = 0;
    
    // PageAttention
    uint32_t blockTypeSize = 0;
    uint32_t oriMaxBlockNumPerBatch = 0;
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
    SASLayout qLayout = SASLayout::BSND;
    SASLayout kvLayout = SASLayout::PA_ND;
    SASLayout outLayout = SASLayout::BSND;

};

class KvQuantSASInfoParser {
public:
    explicit KvQuantSASInfoParser(gert::TilingContext *context) : context_(context) {}
    ~KvQuantSASInfoParser() = default;

    ge::graphStatus CheckRequiredInOutExistence() const;
    ge::graphStatus CheckRequiredAttrExistence() const;
    ge::graphStatus CheckRequiredParaExistence() const;

    ge::graphStatus GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
        SASLayout &layout, const std::string &name) const;
    ge::graphStatus GetActualSeqLenQSize(uint32_t &size);
    ge::graphStatus GetOpName();
    ge::graphStatus GetNpuInfo();
    void GetOptionalInputParaInfo();
    void GetInputParaInfo();
    void GetOutputParaInfo();
    ge::graphStatus GetAttrParaInfo();

    ge::graphStatus GetOpParaInfo();

    ge::graphStatus GetInOutDataType();
    ge::graphStatus GetQueryAndOutLayout();
    ge::graphStatus GetKvLayout();
    void SetSASShape();
    ge::graphStatus GetN1Size();
    ge::graphStatus GetN2Size();
    ge::graphStatus GetGSize();
    ge::graphStatus GetBatchSize();
    ge::graphStatus GetQTSize();
    ge::graphStatus GetS1Size();
    ge::graphStatus GetS2SizeForPageAttention();
    ge::graphStatus GetS2Size();
    ge::graphStatus GetMaxBlockNumPerBatch();
    ge::graphStatus GetBlockSize();
    ge::graphStatus GetQkHeadDim();
    ge::graphStatus GetSparseBlockCount();
    ge::graphStatus GetActualseqInfo();
    ge::graphStatus GetDSizeQ();
    ge::graphStatus GetDSizeKV();
    ge::graphStatus GetSinks();
    void GenerateInfo(KvQuantSASTilingInfo &sasInfo);
    ge::graphStatus Parse(KvQuantSASTilingInfo &sasInfo);

public:
    gert::TilingContext *context_ = nullptr;
    const char *opName_;
    fe::PlatFormInfos *platformInfo_;
    KvQuantSASParaInfo opParamInfo_;

    bool HasAxis(const SASAxis &axis, const SASLayout &layout, const gert::Shape &shape) const;
    size_t GetAxisIdx(const SASAxis &axis, const SASLayout &layout) const;
    uint32_t GetAxisNum(const gert::Shape &shape, const SASAxis &axis,const SASLayout &layout) const;
    static constexpr int64_t invalidDimValue_ = std::numeric_limits<int64_t>::min();

    // BaseParams
    uint32_t bSize_ = 0;
    uint32_t n1Size_ = 0;
    uint32_t n2Size_ = 0;
    uint32_t gSize_ = 0;
    uint32_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    uint32_t headDim_ = 0;
    uint32_t qTSize_ = 0;
    uint32_t qkHeadDim_ = 0;
    int64_t sparseBlockSize_ = 0;
    int64_t sparseBlockCount_ = 0;
    uint32_t maxActualseq_ = 0;
    bool isSameSeqAllKVTensor_ = true;
    uint32_t actualLenDimsKV_ = 0;
    uint32_t actualLenDimsQ_ = 0;
    uint32_t dSizeQ_ = 0;
    uint32_t dSizeKV_ = 0;
    // Layout
    SASLayout qLayout_ = SASLayout::BSND;
    SASLayout outLayout_ = SASLayout::BSND;
    SASLayout kvLayout_ = SASLayout::PA_ND;
    // PageAttention
    uint32_t oriMaxBlockNumPerBatch_ = 0;
    uint32_t cmpMaxBlockNumPerBatch_ = 0;
    int32_t oriBlockSize_ = 0;
    int32_t cmpBlockSize_ = 0;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
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
    gert::Shape cmpSparseIndicesShape_{};
};

class KvQuantSASTilingCheck {
public:
    explicit KvQuantSASTilingCheck(const KvQuantSASTilingInfo &sasInfo) : sasInfo_(sasInfo) {};
    ~KvQuantSASTilingCheck() = default;
    virtual ge::graphStatus Process();

private:
    void Init();
    void LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList,
        const ge::DataType &actualDtype, const std::string &name) const;
    ge::graphStatus CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc,
        const std::string &name) const;
    template <typename T> void LogErrorNumberSupport(const std::vector<T> &expectNumberList,
        const T &actualValue, const std::string &name, const std::string subName) const;
    template <typename T> void LogErrorDimNumSupport(const std::vector<T> &expectNumberList,
        const T &actualValue, const std::string &name) const;
    ge::graphStatus CheckDimNumSupport(const gert::StorageShape *shape,
        const std::vector<size_t> &expectDimNumList, const std::string &name) const;
    ge::graphStatus CheckShapeNumSupport(const gert::StorageShape *shape,
        const std::vector<int64_t> &expectShapeNumList, const std::string &name) const;
    ge::graphStatus CheckDimNumInLayoutSupport(const SASLayout &layout,
        const gert::StorageShape *shape, const std::string &name) const;
    void LogErrorLayoutSupport(const std::vector<SASLayout> &expectLayoutList,
        const SASLayout &actualLayout, const std::string &name) const;
    ge::graphStatus GetExpectedShape(gert::Shape &shapeExpected,
    const KvQuantSASTilingShapeCompareParam &param, const SASLayout &layout) const;
    ge::graphStatus CompareShape(KvQuantSASTilingShapeCompareParam &param,
        const gert::Shape &shape, const SASLayout &layout, const std::string &name) const;
    ge::graphStatus CheckLayoutSupport(const SASLayout &actualLayout, const std::string &name) const;
    ge::graphStatus CheckSingleParaQuery() const;
    ge::graphStatus CheckSingleParaKey() const;
    ge::graphStatus CheckSingleParaNumHeads() const;
    ge::graphStatus CheckSingleParaKvHeadNums() const;
    ge::graphStatus CheckSingleParaSparseMode() const;
    ge::graphStatus CheckSingleParaSparseBlockSize() const;
    ge::graphStatus CheckSingleParaCmpSparseIndices() const;
    ge::graphStatus CheckSingleParaBlockTable() const;
    ge::graphStatus CheckSingleParaCuSeqLensQ() const;
    ge::graphStatus CheckSingleParaSequsedKv() const;
    ge::graphStatus CheckSingleParaSinks() const;

    ge::graphStatus CheckSinglePara() const;
    ge::graphStatus CheckDequantScaleNotExistence();
    ge::graphStatus CheckParaExistenceAntiquant() const;
    ge::graphStatus CheckParaExistence();
    ge::graphStatus GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
        const SASLayout &layout, const std::string &name) const;

    ge::graphStatus CheckKVShapeForBatchContinuous();
    uint32_t GetTypeSize(ge::DataType dtype) const;
    ge::graphStatus CheckKVShapeForPageAttention();
    ge::graphStatus CheckKVShape();
    ge::graphStatus CheckKV();
    ge::graphStatus CheckTopK();
    ge::graphStatus CheckTopkShape();
    ge::graphStatus CheckBlockTable() const;
    ge::graphStatus CheckDTypeConsistency(const ge::DataType &actualDtype,
    const ge::DataType &expectDtype, const std::string &name) const;

    ge::graphStatus CheckAttenOut();
    ge::graphStatus CheckAttenOutShape();
    ge::graphStatus CheckActualSeqLensQ();
    ge::graphStatus CheckActualSeqLensQShape();
    ge::graphStatus CheckActualSeqLensQDType();
    ge::graphStatus CheckActualSeqLens();
    ge::graphStatus CheckActualSeqLensDType();
    ge::graphStatus CheckActualSeqLensShape();
    ge::graphStatus CheckMultiParaConsistency();

    ge::graphStatus CheckFeatureAntiquantShape() const;
    ge::graphStatus CheckFeatureAntiquantLayout() const;
    ge::graphStatus CheckFeatureAntiquantDtype() const;
    ge::graphStatus CheckFeatureAntiquantAttr() const;
    ge::graphStatus CheckFeatureAntiquantPa() const;
    ge::graphStatus CheckFeatureAntiquant() const;
    ge::graphStatus CheckFeature() const;

    void SetSASShapeCompare();

private:
    const char *opName_;
    fe::PlatFormInfos *platformInfo_;
    KvQuantSASParaInfo opParamInfo_;
    const KvQuantSASTilingInfo &sasInfo_;

    uint32_t bSize_ = 0;
    uint32_t n1Size_ = 0;
    uint32_t n2Size_ = 0;
    uint32_t gSize_ = 0;
    uint32_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    uint32_t qkHeadDim_ = 0;
    uint32_t vHeadDim_ = 0;
    uint32_t ropeHeadDim_ = 0;
    uint32_t qTSize_ = 0; // 仅TND时生效
    uint32_t kvTSize_ = 0; // 仅TND时生效
    KvStorageMode kvStorageMode_ = KvStorageMode::BATCH_CONTINUOUS;
    uint32_t sparseBlockCount_ = 0;
    uint32_t sparseBlockSize_ = 0;

    uint32_t oriBlockNum_ = 0;
    uint32_t cmpBlockNum_ = 0;

    uint32_t actualLenDimsQ_ = 0;

    uint32_t oriBlockSize_ = 0;
    uint32_t cmpBlockSize_ = 0;
    uint32_t oriBlockTable_ = 0;
    uint32_t cmpBlockTable_ = 0;
    uint32_t kv_quant_mode_ = 0;

    uint32_t oriMaskMode_ = 0;
    uint32_t cmpMaskMode_ = 0;

    SASLayout qLayout_ = SASLayout::BSND;
    SASLayout outLayout_ = SASLayout::BSND;
    SASLayout kvLayout_ = SASLayout::PA_ND;

    uint32_t oriMaxBlockNumPerBatch_ = 0;
    uint32_t cmpMaxBlockNumPerBatch_ = 0;
    //int64_t blockSize_ = 0;

    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    uint64_t l2CacheSize_ = 0;

    uint32_t tileSize_ = 0;

    bool isSameSeqAllKVTensor_ = true;
    bool isSameActualseq_ = true;
    uint32_t maxActualseq_ = 0;

    ge::DataType qType_ = ge::DT_FLOAT16;
    ge::DataType oriKvType_ = ge::DT_FLOAT16;
    ge::DataType cmpKvType_ = ge::DT_FLOAT16;
    ge::DataType outputType_ = ge::DT_FLOAT16;

    gert::Shape queryShapeCmp_{};
    gert::Shape keyShapeCmp_{};
    gert::Shape valueShapeCmp_{};
    gert::Shape topkShapeCmp_{};
    gert::Shape attenOutShapeCmp_{};
};
} // namespace optiling
#endif // KVQUANT_SPARSE_ATTN_SHAREDKV_TILING_H