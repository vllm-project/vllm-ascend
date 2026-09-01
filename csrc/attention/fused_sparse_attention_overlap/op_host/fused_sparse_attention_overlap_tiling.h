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
 * \file fused_sparse_attention_overlap_tiling.h
 * \brief
 */
#ifndef FUSED_SPARSE_ATTENTION_OVERLAP_TILING_H
#define FUSED_SPARSE_ATTENTION_OVERLAP_TILING_H

#include <limits>
#include <map>
#include <string>
#include <vector>
#include <graph/utils/type_utils.h>
#include <exe_graph/runtime/tiling_context.h>
#include <tiling/platform/platform_ascendc.h>
#include "register/tilingdata_base.h"
#include "platform/soc_spec.h"

namespace optiling {
// ------------------Operator prototype index constants----------------
// Inputs Index
constexpr uint32_t QUERY_INPUT_INDEX = 0;
constexpr uint32_t KEY_INPUT_INDEX = 1;
constexpr uint32_t VALUE_INPUT_INDEX = 2;
constexpr uint32_t SPARSE_INDICES_INPUT_INDEX = 3;
constexpr uint32_t BLOCK_TABLE_INPUT_INDEX = 4;
constexpr uint32_t ACT_SEQ_LEN_Q_INPUT_INDEX = 5;
constexpr uint32_t ACT_SEQ_LEN_KV_INPUT_INDEX = 6;
constexpr uint32_t QUERY_ROPE_INPUT_INDEX = 7;
constexpr uint32_t KEY_ROPE_INPUT_INDEX = 8;
constexpr uint32_t SELECTION_KV_BLOCK_TABLE_INPUT_INDEX = 11;
constexpr uint32_t SELECTION_KV_BLOCK_STATUS_INPUT_INDEX = 12;
constexpr uint32_t SELECTION_MEMBERSHIP_MAP_INPUT_INDEX = 13;
// Outputs Index
constexpr uint32_t OUTPUT_INDEX = 0;

// Attributes Index
constexpr uint32_t SCALE_VALUE_ATTR_INDEX = 0;
constexpr uint32_t SPARSE_BLOCK_SIZE_ATTR_INDEX = 1;
constexpr uint32_t LAYOUT_QUERY_ATTR_INDEX = 2;
constexpr uint32_t LAYOUT_KV_ATTR_INDEX = 3;
constexpr uint32_t SPARSE_MODE_ATTR_INDEX = 4;
// Dim Num
constexpr size_t DIM_NUM_TWO = 2;
constexpr size_t DIM_NUM_THREE = 3;
constexpr size_t DIM_NUM_FOUR = 4;
// Constants
constexpr uint32_t MAX_BLOCK_SIZE = 1024;
constexpr uint32_t NUM_BYTES_FLOAT16 = 2;
constexpr uint32_t NUM_BYTES_BF16 = 2;
constexpr uint32_t BYTE_BLOCK = 32;

// ------------------Common definitions--------------------------
enum class FusedSparseAttentionOverlapLayout : uint32_t {
    BSND = 0,
    TND = 1,
    PA_BSND = 2
};

struct FusedSparseAttentionOverlapTilingShapeCompareParam {
    int64_t B = 1;
    int64_t S = 1;
    int64_t N = 1;
    int64_t D = 1;
    int64_t T = 1;
    // PA
    int64_t Bs = 1;
    int64_t Bn = 1;
};

enum class FusedOverlapKvStorageMode : uint32_t {
    BATCH_CONTINUOUS = 0,
    PAGE_ATTENTION = 1
};

enum class FusedSparseAttentionOverlapPerfMode : uint32_t {
    C_TEMPLATE_MODE = 0,
    V_TEMPLATE_MODE
};

enum class FusedSparseAttentionOverlapAxis : uint32_t {
    B = 0,
    S = 1,
    N = 2,
    D = 3,
    K = 3,  // K in sparse_indices and D in key share the same enum value because both are the last dimension
    T = 5,
    Bn = 6, // block number
    Bs = 7, // block size
};

struct FusedSparseAttentionOverlapRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct FusedSparseAttentionOverlapOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

// -----------Operator tiling input structure definitions---------------
struct FusedSparseAttentionOverlapParaInfo {
    FusedSparseAttentionOverlapRequiredParaInfo query = {nullptr, nullptr};
    FusedSparseAttentionOverlapRequiredParaInfo key = {nullptr, nullptr};
    FusedSparseAttentionOverlapRequiredParaInfo value = {nullptr, nullptr};
    FusedSparseAttentionOverlapRequiredParaInfo sparseIndices = {nullptr, nullptr};
    FusedSparseAttentionOverlapOptionalParaInfo blockTable = {nullptr, nullptr};
    FusedSparseAttentionOverlapOptionalParaInfo actualSeqLengthsQ = {nullptr, nullptr};
    FusedSparseAttentionOverlapOptionalParaInfo actualSeqLengths = {nullptr, nullptr};
    FusedSparseAttentionOverlapOptionalParaInfo queryRope = {nullptr, nullptr};
    FusedSparseAttentionOverlapOptionalParaInfo keyRope = {nullptr, nullptr};
    FusedSparseAttentionOverlapRequiredParaInfo attenOut = {nullptr, nullptr};

    const char *layoutQuery = nullptr;
    const char *layoutKV = nullptr;
    const int64_t *sparseBlockSize = nullptr;
    const float *scaleValue = nullptr;
    const int64_t *sparseMode = nullptr;
};

// -----------Operator TilingData definitions---------------
BEGIN_TILING_DATA_DEF(FusedSparseAttentionOverlapBaseParamsMla)
TILING_DATA_FIELD_DEF(uint32_t, batchSize)
TILING_DATA_FIELD_DEF(uint32_t, seqSize)
TILING_DATA_FIELD_DEF(uint32_t, qSeqSize)
TILING_DATA_FIELD_DEF(int64_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(float, scaleValue)
TILING_DATA_FIELD_DEF(uint32_t, nNumOfQInOneGroup)
TILING_DATA_FIELD_DEF(uint32_t, actualLenDimsQ)
TILING_DATA_FIELD_DEF(uint32_t, actualLenDimsKV)
TILING_DATA_FIELD_DEF(uint32_t, outputLayout)
TILING_DATA_FIELD_DEF(uint32_t, sparseMode)
TILING_DATA_FIELD_DEF(int64_t, preTokens)
TILING_DATA_FIELD_DEF(int64_t, nextTokens)
TILING_DATA_FIELD_DEF(uint32_t, attentionMode)
TILING_DATA_FIELD_DEF(uint32_t, returnSoftmaxLse)
TILING_DATA_FIELD_DEF(int64_t, sparseBlockSize)
TILING_DATA_FIELD_DEF(uint32_t, sparseBlockCount)
TILING_DATA_FIELD_DEF(uint32_t, selectionBlockTableStride)
TILING_DATA_FIELD_DEF(uint32_t, selectionStatusStride)
TILING_DATA_FIELD_DEF(uint32_t, selectionMembershipStride)
TILING_DATA_FIELD_DEF(uint32_t, isActualLenDimsNull)
TILING_DATA_FIELD_DEF(uint32_t, isActualLenDimsKVNull)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedSparseAttentionOverlapBaseParamsMlaOp, FusedSparseAttentionOverlapBaseParamsMla)

BEGIN_TILING_DATA_DEF(FusedSparseAttentionOverlapSingleCoreParamsMla)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedSparseAttentionOverlapSingleCoreParamsMlaOp, FusedSparseAttentionOverlapSingleCoreParamsMla)

BEGIN_TILING_DATA_DEF(FusedSparseAttentionOverlapSingleCoreTensorSizeMla)
TILING_DATA_FIELD_DEF(uint32_t, mmResUbSize);
TILING_DATA_FIELD_DEF(uint32_t, bmm2ResUbSize);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedSparseAttentionOverlapSingleCoreTensorSizeMlaOp, FusedSparseAttentionOverlapSingleCoreTensorSizeMla)

BEGIN_TILING_DATA_DEF(FusedSparseAttentionOverlapSplitKVParamsMla)
TILING_DATA_FIELD_DEF(uint32_t, s2)             // Number of S2 partitions
TILING_DATA_FIELD_DEF(uint32_t, accumOutSize)   // FD workspace
TILING_DATA_FIELD_DEF(uint32_t, logSumExpSize)  // FD workspace
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedSparseAttentionOverlapSplitKVParamsMlaOp, FusedSparseAttentionOverlapSplitKVParamsMla)

// Inner base-block parameters
BEGIN_TILING_DATA_DEF(FusedSparseAttentionOverlapInnerSplitParams)
TILING_DATA_FIELD_DEF(uint32_t, mBaseSize)
TILING_DATA_FIELD_DEF(uint32_t, s2BaseSize)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedSparseAttentionOverlapInnerSplitParamsOp, FusedSparseAttentionOverlapInnerSplitParams)

BEGIN_TILING_DATA_DEF(FusedSparseAttentionOverlapTilingDataMla)
TILING_DATA_FIELD_DEF_STRUCT(FusedSparseAttentionOverlapBaseParamsMla, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedSparseAttentionOverlapSplitKVParamsMla, splitKVParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedSparseAttentionOverlapSingleCoreParamsMla, singleCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedSparseAttentionOverlapSingleCoreTensorSizeMla, singleCoreTensorSize);
TILING_DATA_FIELD_DEF_STRUCT(FusedSparseAttentionOverlapInnerSplitParams, innerSplitParams);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedSparseAttentionOverlap, FusedSparseAttentionOverlapTilingDataMla)

template <typename T> inline T Align(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
}

// -----------Operator tiling input information class---------------
struct FusedSparseAttentionOverlapTilingInfo {
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    FusedSparseAttentionOverlapParaInfo opParamInfo;

    // Base Param
    NpuArch npuArch = NpuArch::DAV_2201;
    bool isA5 = false;
    uint32_t bSize = 0;
    uint32_t n1Size = 0;
    uint32_t n2Size = 0;
    uint32_t s1Size = 0;
    int64_t s2Size = 0;
    uint32_t qkHeadDim = 0;
    uint32_t vHeadDim = 0;
    uint32_t gSize = 0;
    uint32_t ropeHeadDim = 0;
    uint32_t qTSize = 0; // Effective only for TND
    uint32_t kvTSize = 0; // Effective only for TND
    float scaleValue = 0;
    int64_t sparseBlockSize = 0;
    int64_t sparseBlockCount = 0;

    int64_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;

    uint32_t actualLenDimsQ = 0;

    bool actualQSeqLenFlag = false;
    bool actualSeqLenFlag = false;
    uint32_t actualLenDimsKV = 0;

    uint32_t sparseMode = 0;
    int64_t preTokens = INT64_MAX;
    int64_t nextTokens = INT64_MAX;
    uint32_t attentionMode = 2;
    bool returnSoftmaxLse = false;

    ge::DataType inputQType = ge::DT_FLOAT16;
    ge::DataType inputKvType = ge::DT_FLOAT16;
    ge::DataType outputType = ge::DT_FLOAT16;

    FusedOverlapKvStorageMode kvStorageMode = FusedOverlapKvStorageMode::BATCH_CONTINUOUS;

    FusedSparseAttentionOverlapLayout qLayout = FusedSparseAttentionOverlapLayout::BSND;
    FusedSparseAttentionOverlapLayout topkLayout = FusedSparseAttentionOverlapLayout::BSND;
    FusedSparseAttentionOverlapLayout outLayout = FusedSparseAttentionOverlapLayout::BSND;
    FusedSparseAttentionOverlapLayout kvLayout = FusedSparseAttentionOverlapLayout::BSND;

    ge::DataType inputQRopeType = ge::DT_FLOAT16;
    ge::DataType inputKRopeType = ge::DT_FLOAT16;

};

// ---------------Operator tiling class---------------
class FusedSparseAttentionOverlapMlaTiling {
public:
    explicit FusedSparseAttentionOverlapMlaTiling(gert::TilingContext *context) : context_(context) {}
    ge::graphStatus DoOpTiling(FusedSparseAttentionOverlapTilingInfo *sfaInfo);

private:
    ge::graphStatus SetBlockDim(uint32_t blockDim) const;
    ge::graphStatus SetTilingKey(uint64_t tilingKey) const;
    ge::graphStatus SetWorkspaceSize(uint64_t workspaceSize) const;
    ge::graphStatus SetTilingData(TilingDef &tilingData) const;
    gert::TilingContext *context_ = nullptr;
    ge::graphStatus GetPlatformInfo();
    void GenTilingKey();

    void ZeroTensorProcess() const;
    void InitParams();

    void SplitBalanced();
    void CalcInnerSize(uint32_t s2Size);

    void FillTilingBaseParamsMla();
    void FillTilingSplitKVMla();

    void FillTilingSingleCoreParamsMla();
    void FillTilingSingleCoreTensorSizeMla();
    void FillTiling();

    void CalcUbBmm();
    void GetWorkspaceSize();

    void CalcBlockDim();

    uint32_t GetTypeSize(ge::DataType dtype) const;

    uint32_t coreNum_ = 0;
    FusedSparseAttentionOverlapPerfMode perfMode_ = FusedSparseAttentionOverlapPerfMode::V_TEMPLATE_MODE;
    size_t mmResUbSize_ = 0;
    size_t bmm2ResUbSize_ = 0;
    uint32_t sInnerSize_ = 0;
    uint32_t sInnerSizeAlign_ = 0;
    uint32_t usedCoreNum_ = 0;

    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    size_t libapiSize_ = 0;

    FusedSparseAttentionOverlapTilingDataMla tilingData_;
    uint32_t blockDim_{0};
    uint64_t workspaceSize_{0};
    uint64_t tilingKey_{0};

    uint32_t headDimAlign_ = 0;
    uint32_t mBaseSize_ = 128;

    FusedSparseAttentionOverlapTilingInfo *sfaInfo_ = nullptr;
};

// -----------Operator tiling input parsing and validation class---------------
class FusedSparseAttentionOverlapTilingCheck {
public:
    explicit FusedSparseAttentionOverlapTilingCheck(const FusedSparseAttentionOverlapTilingInfo &sfaInfo) : sfaInfo_(sfaInfo) {};
    ~FusedSparseAttentionOverlapTilingCheck() = default;
    ge::graphStatus Process();
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
    ge::graphStatus CheckDimNumInLayoutSupport(const FusedSparseAttentionOverlapLayout &layout,
        const gert::StorageShape *shape, const std::string &name) const;
    void LogErrorLayoutSupport(const std::vector<FusedSparseAttentionOverlapLayout> &expectLayoutList,
        const FusedSparseAttentionOverlapLayout &actualLayout, const std::string &name) const;
    ge::graphStatus GetExpectedShape(gert::Shape &shapeExpected,
    const FusedSparseAttentionOverlapTilingShapeCompareParam &param, const FusedSparseAttentionOverlapLayout &layout) const;
    ge::graphStatus CompareShape(FusedSparseAttentionOverlapTilingShapeCompareParam &param,
        const gert::Shape &shape, const FusedSparseAttentionOverlapLayout &layout, const std::string &name) const;
    ge::graphStatus CheckLayoutSupport(const FusedSparseAttentionOverlapLayout &actualLayout, const std::string &name) const;
    ge::graphStatus CheckSingleParaQuery() const;
    ge::graphStatus CheckSingleParaKey() const;
    ge::graphStatus CheckSingleParaSparseMode() const;
    ge::graphStatus CheckSingleParaSparseBlockSize() const;
    ge::graphStatus CheckSingleParaSparseIndices() const;
    ge::graphStatus CheckSinglePara() const;
    ge::graphStatus CheckRopeExistence();
    ge::graphStatus CheckExists(const void *pointer, const std::string &name) const;
    ge::graphStatus CheckParaExistenceMlaNoquant() const;
    ge::graphStatus CheckParaExistence();
    ge::graphStatus GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
        const FusedSparseAttentionOverlapLayout &layout, const std::string &name) const;
    void SetSFAShapeCompare();
    ge::graphStatus CheckQRope();
    ge::graphStatus CheckQRopeShape();
    ge::graphStatus CheckVAndKRopeShapeForBatchContinuous();
    ge::graphStatus CheckVAndKRopeShapeForPageAttention();
    ge::graphStatus CheckVAndKRopeShape();
    ge::graphStatus CheckVAndKRope();
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

    ge::graphStatus CheckFeatureMlaNoQuantShape() const;
    ge::graphStatus CheckFeatureMlaNoQuantLayout() const;
    ge::graphStatus CheckFeatureMlaNoQuantDtype() const;
    ge::graphStatus CheckFeatureMlaNoquantPa() const;
    ge::graphStatus CheckFeatureMlaNoquant() const;

private:
    const char *opName_;
    FusedSparseAttentionOverlapParaInfo opParamInfo_;
    const FusedSparseAttentionOverlapTilingInfo &sfaInfo_;

    uint32_t bSize_ = 0;
    uint32_t n1Size_ = 0;
    uint32_t n2Size_ = 0;
    uint32_t gSize_ = 0;
    uint32_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    uint32_t qkHeadDim_ = 0;
    uint32_t vHeadDim_ = 0;
    uint32_t ropeHeadDim_ = 0;
    uint32_t qTSize_ = 0; // Effective only for TND
    uint32_t kvTSize_ = 0; // Effective only for TND
    FusedOverlapKvStorageMode kvStorageMode_ = FusedOverlapKvStorageMode::BATCH_CONTINUOUS;
    uint32_t sparseBlockCount_ = 0;
    int64_t sparseBlockSize_ = 0;

    FusedSparseAttentionOverlapLayout qLayout_ = FusedSparseAttentionOverlapLayout::BSND;
    FusedSparseAttentionOverlapLayout topkLayout_ = FusedSparseAttentionOverlapLayout::BSND;
    FusedSparseAttentionOverlapLayout outLayout_ = FusedSparseAttentionOverlapLayout::BSND;
    FusedSparseAttentionOverlapLayout kvLayout_ = FusedSparseAttentionOverlapLayout::BSND;

    uint32_t maxBlockNumPerBatch_ = 0;
    int64_t blockSize_ = 0;

    NpuArch npuArch_ = NpuArch::DAV_2201;

    ge::DataType inputQType_ = ge::DT_FLOAT16;
    ge::DataType inputKvType_ = ge::DT_FLOAT16;
    ge::DataType outputType_ = ge::DT_FLOAT16;
    ge::DataType inputQRopeType_ = ge::DT_FLOAT16;
    ge::DataType inputKRopeType_ = ge::DT_FLOAT16;

    gert::Shape keyShapeCmp_{};
    gert::Shape valueShapeCmp_{};
    gert::Shape topkShapeCmp_{};
    gert::Shape queryRopeShapeCmp_{};
    gert::Shape keyRopeShapeCmp_{};
    gert::Shape attenOutShapeCmp_{};
};

class FusedSparseAttentionOverlapInfoParser {
public:
    explicit FusedSparseAttentionOverlapInfoParser(const gert::TilingContext *context) : context_(context) {}
    ~FusedSparseAttentionOverlapInfoParser() = default;

    ge::graphStatus CheckRequiredInOutExistence() const;
    ge::graphStatus CheckRequiredAttrExistence() const;
    ge::graphStatus CheckRequiredParaExistence() const;

    ge::graphStatus GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
        FusedSparseAttentionOverlapLayout &layout, const std::string &name) const;
    ge::graphStatus GetActualSeqLenQSize(uint32_t &size);
    ge::graphStatus GetOpName();
    ge::graphStatus GetNpuInfo();
    void GetOptionalInputParaInfo();
    void GetInputParaInfo();
    void GetOutputParaInfo();
    ge::graphStatus GetAttrParaInfo();
    ge::graphStatus GetOpParaInfo();

    ge::graphStatus GetInOutDataType();
    ge::graphStatus GetBatchSize();
    ge::graphStatus GetQTSize();
    ge::graphStatus GetKVTSize();
    ge::graphStatus GetQkHeadDim();
    ge::graphStatus GetS1Size();
    ge::graphStatus GetKvStorageMode();
    ge::graphStatus GetKvLayout();
    void SetSFAShape();
    ge::graphStatus GetS2SizeForBatchContinuous();
    ge::graphStatus GetMaxBlockNumPerBatch();
    ge::graphStatus GetBlockSize();
    ge::graphStatus GetS2SizeForPageAttention();
    ge::graphStatus GetS2Size();
    ge::graphStatus GetValueHeadDim();
    ge::graphStatus GetRopeHeadDim();
    ge::graphStatus GetQueryAndOutLayout();
    ge::graphStatus GetTopkLayout();
    ge::graphStatus GetN1Size();
    ge::graphStatus GetN2Size();
    ge::graphStatus GetGSize();
    ge::graphStatus GetSparseBlockCount();
    ge::graphStatus GetActualseqInfo();
    void GenerateInfo(FusedSparseAttentionOverlapTilingInfo &sfaInfo);
    ge::graphStatus Parse(FusedSparseAttentionOverlapTilingInfo &sfaInfo);

public:
    bool HasAxis(const FusedSparseAttentionOverlapAxis &axis, const FusedSparseAttentionOverlapLayout &layout, const gert::Shape &shape) const;
    size_t GetAxisIdx(const FusedSparseAttentionOverlapAxis &axis, const FusedSparseAttentionOverlapLayout &layout) const;
    uint32_t GetAxisNum(const gert::Shape &shape, const FusedSparseAttentionOverlapAxis &axis,const FusedSparseAttentionOverlapLayout &layout) const;

    const gert::TilingContext *context_ = nullptr;

    const char *opName_;
    fe::PlatFormInfos *platformInfo_;
    FusedSparseAttentionOverlapParaInfo opParamInfo_;
    static constexpr int64_t invalidDimValue_ = std::numeric_limits<int64_t>::min();

    uint32_t bSize_ = 0;
    uint32_t n1Size_ = 0;
    uint32_t n2Size_ = 0;
    uint32_t gSize_ = 0;
    uint32_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    uint32_t qkHeadDim_ = 0;
    uint32_t vHeadDim_ = 0;
    uint32_t ropeHeadDim_ = 0;
    uint32_t qTSize_ = 0; // Effective only for TND
    uint32_t kvTSize_ = 0; // Effective only for TND
    FusedOverlapKvStorageMode kvStorageMode_ = FusedOverlapKvStorageMode::BATCH_CONTINUOUS;
    uint32_t sparseBlockCount_ = 0;

    FusedSparseAttentionOverlapLayout qLayout_ = FusedSparseAttentionOverlapLayout::BSND;
    FusedSparseAttentionOverlapLayout topkLayout_ = FusedSparseAttentionOverlapLayout::BSND;
    FusedSparseAttentionOverlapLayout outLayout_ = FusedSparseAttentionOverlapLayout::BSND;
    FusedSparseAttentionOverlapLayout kvLayout_ = FusedSparseAttentionOverlapLayout::BSND;
    uint32_t maxBlockNumPerBatch_ = 0;
    uint32_t blockSize_ = 0;

    NpuArch npuArch_ = NpuArch::DAV_2201;
    bool isA5_ = false;

    ge::DataType inputQType_ = ge::DT_FLOAT16;
    ge::DataType inputKvType_ = ge::DT_FLOAT16;
    ge::DataType outputType_ = ge::DT_FLOAT16;
    ge::DataType inputQRopeType_ = ge::DT_FLOAT16;
    ge::DataType inputKRopeType_ = ge::DT_FLOAT16;

    uint32_t actualLenDimsQ_ = 0;
    uint32_t actualLenDimsKV_ = 0;

    gert::Shape queryShape_{};
    gert::Shape keyShape_{};
    gert::Shape valueShape_{};
    gert::Shape sparseIndicesShape_{};
    gert::Shape queryRopeShape_{};
};
} // namespace optiling
#endif // FUSED_SPARSE_ATTENTION_OVERLAP_TILING_H
