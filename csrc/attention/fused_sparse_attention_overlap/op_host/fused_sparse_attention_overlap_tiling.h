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

#include <sstream>
#include <graph/utils/type_utils.h>
#include <exe_graph/runtime/tiling_context.h>
#include <tiling/platform/platform_ascendc.h>
#include "register/tilingdata_base.h"
#include "exe_graph/runtime/tiling_context.h"
#include "platform/soc_spec.h"

namespace optiling {
// ------------------算子原型索引常量定义----------------
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
constexpr uint32_t SOFTMAXMAX_INDEX = 1;
constexpr uint32_t SOFTMAXSUM_INDEX = 2;

// Attributes Index
constexpr uint32_t SCALE_VALUE_ATTR_INDEX = 0;
constexpr uint32_t SPARSE_BLOCK_SIZE_ATTR_INDEX = 1;
constexpr uint32_t LAYOUT_QUERY_ATTR_INDEX = 2;
constexpr uint32_t LAYOUT_KV_ATTR_INDEX = 3;
constexpr uint32_t SPARSE_MODE_ATTR_INDEX = 4;
constexpr uint32_t PRE_TOKENS_ATTR_INDEX = 5;
constexpr uint32_t NEXT_TOKENS_ATTR_INDEX = 6;
constexpr uint32_t ATTENTION_MODE_ATTR_INDEX = 7;
constexpr uint32_t RETURN_SOFTMAX_LSE_ATTR_INDEX = 8;
// Dim Num
constexpr size_t DIM_NUM_TWO = 2;
constexpr size_t DIM_NUM_THREE = 3;
constexpr size_t DIM_NUM_FOUR = 4;
// 常量
constexpr uint32_t MAX_BLOCK_SIZE = 1024;
constexpr uint32_t COPYND2NZ_SRC_STRIDE_LIMITATION = 65535;
constexpr uint32_t NUM_BYTES_FLOAT = 4;
constexpr uint32_t NUM_BYTES_FLOAT16 = 2;
constexpr uint32_t NUM_BYTES_BF16 = 2;
constexpr uint32_t BYTE_BLOCK = 32;
const uint32_t SFA_MAX_AIC_CORE_NUM = 26; // 25 + 1 保证数组8字节对齐

// ------------------公共定义--------------------------
enum class FusedSparseAttentionOverlapLayout : uint32_t {
    BSND = 0,
    TND = 1,
    PA_BSND = 2,
    BNSG = 3,
    NTG = 4
};

struct FusedSparseAttentionOverlapTilingShapeCompareParam {
    int64_t B = 1;
    int64_t S = 1;
    int64_t N = 1;
    int64_t D = 1;
    int64_t T = 1;
    int64_t G = 1;
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
    K = 3,  // sparse_indices的K和key的D枚举值相同，表达相同位置, 最后一维
    T = 5,
    Bn = 6, // block number
    Bs = 7, // block size
    G = 8,
};

struct FusedSparseAttentionOverlapRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct FusedSparseAttentionOverlapOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

// -----------算子Tiling入参结构体定义---------------
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
    FusedSparseAttentionOverlapRequiredParaInfo softmaxMax = {nullptr, nullptr};
    FusedSparseAttentionOverlapRequiredParaInfo softmaxSum = {nullptr, nullptr};

    const char *layoutQuery = nullptr;
    const char *layoutKV = nullptr;
    const int64_t *sparseBlockSize = nullptr;
    const float *scaleValue = nullptr;
    const int64_t *sparseMode = nullptr;
    const int64_t *preTokens = nullptr;
    const int64_t *nextTokens = nullptr;
    const int64_t *attentionMode = nullptr;
    const bool *returnSoftmaxLse = nullptr;
};

struct FusedOverlapInnerSplitParams {
    uint32_t s1GBaseSize = 1;
    uint32_t s2BaseSize = 1;
};

// -----------算子TilingData定义---------------
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
TILING_DATA_FIELD_DEF(uint32_t, s2)             // S2切分份数
TILING_DATA_FIELD_DEF(uint32_t, accumOutSize)   // FD workspace
TILING_DATA_FIELD_DEF(uint32_t, logSumExpSize)  // FD workspace
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedSparseAttentionOverlapSplitKVParamsMlaOp, FusedSparseAttentionOverlapSplitKVParamsMla)

// 内切基本块参数
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

template <typename T>
std::string FusedSparseAttentionOverlapShape2String(const T &shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

static std::string GetShapeStr(gert::Shape shape);
static std::string FusedSparseAttentionOverlapDataTypeToSerialString(ge::DataType type);
std::string FusedSparseAttentionOverlapTensorDesc2String(const gert::StorageShape *shape, const gert::CompileTimeTensorDesc *tensor);
std::string FusedSparseAttentionOverlapDebugTilingContext(const gert::TilingContext *context);
std::string FusedSparseAttentionOverlapLayoutToSerialString(FusedSparseAttentionOverlapLayout layout);

// -----------算子Tiling入参信息类---------------
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
    uint32_t qTSize = 0; // 仅TND时生效
    uint32_t kvTSize = 0; // 仅TND时生效
    float scaleValue = 0;
    uint32_t innerPrecise = 0;
    uint32_t l2CacheOffFlag = 0;
    int64_t sparseBlockSize = 0;
    int64_t sparseBlockCount = 0;

    bool pageAttentionFlag = false;
    int64_t blockSize = 0;
    uint32_t blockTypeSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint32_t totalBlockNum = 0;

    uint32_t actualLenDimsQ = 0;
    uint32_t maxActualseq = 0;

    bool actualQSeqLenFlag = false;
    bool actualSeqLenFlag = false;
    bool isSameSeqAllKVTensor = true;
    bool isSameActualseq = true;
    uint32_t actualLenDimsKV = 0;
    std::vector<int64_t> kvListSeqLens {};

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
    FusedSparseAttentionOverlapLayout softmaxMaxLayout = FusedSparseAttentionOverlapLayout::BNSG;
    FusedSparseAttentionOverlapLayout softmaxSumLayout = FusedSparseAttentionOverlapLayout::BNSG;

    ge::DataType inputQRopeType = ge::DT_FLOAT16;
    ge::DataType inputKRopeType = ge::DT_FLOAT16;

    uint64_t l2CacheSize = 0;
};

// ---------------算子Tiling类---------------
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
    bool DealSameSeqEachBatch();

    void ZeroTensorProcess() const;
    void InitParams();

    void Split();
    bool IsBalanceSplitCore();

    void SplitBalanced();
    void CalcInnerSize(uint32_t s2Size);

    bool IsFlashDecode(uint32_t coreNum);

    void FillTilingBaseParamsMla();
    void FillTilingSplitKVMla();

    void FillTilingSingleCoreParamsMla();
    void FillTilingSingleCoreTensorSizeMla();
    void FillTiling();

    void CalcUbBmm();
    void CheckUbSpace();
    void NormalCalcFDWorkSpace(const uint32_t actCoreNum);
    void CalcFDWorkSpace(const uint32_t actCoreNum);
    void GetWorkspaceSize();

    uint32_t CalcBalanceFDParamNums(const uint32_t actCoreNum) const;

    void CalcBlockDim();

    uint32_t GetTypeSize(ge::DataType dtype) const;

    bool balanceModeFlag_ = false;
    bool splitKVFlag_ = false;

    uint32_t coreNum_ = 0;
    FusedSparseAttentionOverlapPerfMode perfMode_ = FusedSparseAttentionOverlapPerfMode::V_TEMPLATE_MODE;
    uint32_t kvSplitPart_ = 1;
    size_t mmResUbSize_ = 0;
    size_t bmm2ResUbSize_ = 0;
    size_t qPreSizeMla_= 0;
    uint32_t sInnerLoopTimes_ = 0;
    uint32_t sInnerSize_ = 0;
    uint32_t sInnerSizeTail_ = 0;
    uint32_t sInnerSizeAlign_ = 0;
    uint32_t kvSplit_ = 0;
    uint32_t usedCoreNum_ = 0;
    uint32_t formerCoreNum_ = 0;
    uint32_t blockSplitBn2Range_ = 0;
    uint32_t tailSplitedBatchRange_ = 0;

    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    size_t libapiSize_ = 0;

    FusedSparseAttentionOverlapTilingDataMla tilingData_;
    uint32_t blockDim_{0};
    uint64_t workspaceSize_{0};
    uint64_t tilingKey_{0};

    uint32_t headDimAlign_ = 0;
    uint32_t mBaseSize_ = 128;
    uint32_t mFdBaseSize_ = 8;

    FusedSparseAttentionOverlapTilingInfo *sfaInfo_ = nullptr;
};

// -----------算子Tiling入参信息解析及Check类---------------
class FusedSparseAttentionOverlapTilingCheck {
public:
    explicit FusedSparseAttentionOverlapTilingCheck(const FusedSparseAttentionOverlapTilingInfo &sfaInfo) : sfaInfo_(sfaInfo) {};
    ~FusedSparseAttentionOverlapTilingCheck() = default;
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
    ge::graphStatus CheckSingleParaValue() const;
    ge::graphStatus CheckSingleParaQueryRope() const;
    ge::graphStatus CheckSingleParaKeyRope() const;
    ge::graphStatus CheckSingleParaAttenOut() const;
    ge::graphStatus CheckSingleParaNumHeads() const;
    ge::graphStatus CheckSingleParaKvHeadNums() const;
    ge::graphStatus CheckSingleParaLayout() const;
    ge::graphStatus CheckSingleParaSparseMode() const;
    ge::graphStatus CheckSingleParaSparseBlockSize() const;
    ge::graphStatus CheckSingleParaSparseIndices() const;
    ge::graphStatus CheckSinglePara() const;
    ge::graphStatus CheckMultiParaConsistency() const;
    ge::graphStatus CheckRopeExistence();
    ge::graphStatus CheckExists(const void *pointer, const std::string &name) const;
    ge::graphStatus CheckNotExists(const void *pointer, const std::string &name) const;
    ge::graphStatus CheckExistsByMap(const std::map<std::string, const void *> &paramMap) const;
    ge::graphStatus CheckNotExistsByMap(const std::map<std::string, const void *> &paramMap) const;
    ge::graphStatus CheckExistenceByMap(std::map<std::string, const void *> &existMap,
        std::map<std::string, const void *> &notExistMap) const;
    template <typename T> ge::graphStatus CheckAttrValueByMap(
        std::map<std::string, std::pair<const T *, T>> &attrMap) const;
    ge::graphStatus CheckParaExistenceMlaNoquant() const;
    ge::graphStatus CheckParaExistenceGqaNoquant() const;
    ge::graphStatus CheckParaExistenceMla() const;
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
    ge::graphStatus CheckSoftmaxMax();
    ge::graphStatus CheckSoftmaxMaxShape();
    ge::graphStatus CheckSoftmaxSum();
    ge::graphStatus CheckSoftmaxSumShape();
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
    ge::graphStatus CheckFeatureMla() const;
    ge::graphStatus CheckFeature() const;

    ge::graphStatus CheckSingleParaPreTokens() const;
    ge::graphStatus CheckSingleParaNextTokens() const;

private:
    const char *opName_;
    fe::PlatFormInfos *platformInfo_;
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
    uint32_t qTSize_ = 0; // 仅TND时生效
    uint32_t kvTSize_ = 0; // 仅TND时生效
    FusedOverlapKvStorageMode kvStorageMode_ = FusedOverlapKvStorageMode::BATCH_CONTINUOUS;
    uint32_t sparseBlockCount_ = 0;
    int64_t sparseBlockSize_ = 0;

    FusedSparseAttentionOverlapLayout qLayout_ = FusedSparseAttentionOverlapLayout::BSND;
    FusedSparseAttentionOverlapLayout topkLayout_ = FusedSparseAttentionOverlapLayout::BSND;
    FusedSparseAttentionOverlapLayout outLayout_ = FusedSparseAttentionOverlapLayout::BSND;
    FusedSparseAttentionOverlapLayout kvLayout_ = FusedSparseAttentionOverlapLayout::BSND;
    FusedSparseAttentionOverlapLayout softmaxMaxLayout_ = FusedSparseAttentionOverlapLayout::BNSG;
    FusedSparseAttentionOverlapLayout softmaxSumLayout_ = FusedSparseAttentionOverlapLayout::BNSG;

    uint32_t maxBlockNumPerBatch_ = 0;
    int64_t blockSize_ = 0;

    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    NpuArch npuArch_ = NpuArch::DAV_2201;
    bool isA5_ = false;
    uint64_t l2CacheSize_ = 0;

    ge::DataType inputQType_ = ge::DT_FLOAT16;
    ge::DataType inputKvType_ = ge::DT_FLOAT16;
    ge::DataType outputType_ = ge::DT_FLOAT16;
    ge::DataType inputQRopeType_ = ge::DT_FLOAT16;
    ge::DataType inputKRopeType_ = ge::DT_FLOAT16;

    gert::Shape queryShapeCmp_{};
    gert::Shape keyShapeCmp_{};
    gert::Shape valueShapeCmp_{};
    gert::Shape topkShapeCmp_{};
    gert::Shape queryRopeShapeCmp_{};
    gert::Shape keyRopeShapeCmp_{};
    gert::Shape attenOutShapeCmp_{};
    gert::Shape softmaxMaxShapeCmp_{};
    gert::Shape softmaxSumShapeCmp_{};
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
    ge::graphStatus GetKvCache();
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
    ge::graphStatus GetSoftmaxMaxAndSumLayout();
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
    uint32_t qTSize_ = 0; // 仅TND时生效
    uint32_t kvTSize_ = 0; // 仅TND时生效
    FusedOverlapKvStorageMode kvStorageMode_ = FusedOverlapKvStorageMode::BATCH_CONTINUOUS;
    uint32_t sparseBlockCount_ = 0;

    FusedSparseAttentionOverlapLayout qLayout_ = FusedSparseAttentionOverlapLayout::BSND;
    FusedSparseAttentionOverlapLayout topkLayout_ = FusedSparseAttentionOverlapLayout::BSND;
    FusedSparseAttentionOverlapLayout outLayout_ = FusedSparseAttentionOverlapLayout::BSND;
    FusedSparseAttentionOverlapLayout kvLayout_ = FusedSparseAttentionOverlapLayout::BSND;
    FusedSparseAttentionOverlapLayout softmaxMaxLayout_ = FusedSparseAttentionOverlapLayout::BNSG;
    FusedSparseAttentionOverlapLayout softmaxSumLayout_ = FusedSparseAttentionOverlapLayout::BNSG;
    uint32_t maxBlockNumPerBatch_ = 0;
    uint32_t blockSize_ = 0;

    NpuArch npuArch_ = NpuArch::DAV_2201;
    bool isA5_ = false;

    ge::DataType inputQType_ = ge::DT_FLOAT16;
    ge::DataType inputKvType_ = ge::DT_FLOAT16;
    ge::DataType outputType_ = ge::DT_FLOAT16;
    ge::DataType inputQRopeType_ = ge::DT_FLOAT16;
    ge::DataType inputKRopeType_ = ge::DT_FLOAT16;

    uint64_t l2CacheSize_ = 0;

    bool isSameSeqAllKVTensor_ = true;
    bool isSameActualseq_ = true;
    uint32_t maxActualseq_ = 0;

    uint32_t actualLenDimsQ_ = 0;
    uint32_t actualLenDimsKV_ = 0;

    gert::Shape queryShape_{};
    gert::Shape keyShape_{};
    gert::Shape valueShape_{};
    gert::Shape sparseIndicesShape_{};
    gert::Shape queryRopeShape_{};
    gert::Shape keyRopeShape_{};
};
} // namespace optiling
#endif // FUSED_SPARSE_ATTENTION_OVERLAP_TILING_H
