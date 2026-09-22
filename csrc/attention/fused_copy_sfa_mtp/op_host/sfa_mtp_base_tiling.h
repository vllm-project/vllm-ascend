/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_tail_attention_tiling.h
 * \brief
 */
#ifndef SPARSE_TAIL_ATTENTION_TILING_H
#define SPARSE_TAIL_ATTENTION_TILING_H

#include <sstream>
#include <graph/utils/type_utils.h>
#include <exe_graph/runtime/tiling_context.h>
#include <tiling/platform/platform_ascendc.h>
#include "register/tilingdata_base.h"
#include "exe_graph/runtime/tiling_context.h"

namespace optiling {
// Inputs Index
constexpr uint32_t QUERY_INPUT_INDEX = 0;
constexpr uint32_t KEY_INPUT_INDEX = 1;
constexpr uint32_t VALUE_INPUT_INDEX = 2;
constexpr uint32_t SPARSE_INDICES_INPUT_INDEX = 3;
constexpr uint32_t CACHE_TOKENS_INPUT_INDEX = 4;
constexpr uint32_t BLOCK_TABLE_INPUT_INDEX = 5;
constexpr uint32_t ACT_SEQ_LEN_Q_INPUT_INDEX = 6;
constexpr uint32_t ACT_SEQ_LEN_KV_INPUT_INDEX = 7;
constexpr uint32_t QUERY_ROPE_INPUT_INDEX = 8;
constexpr uint32_t KEY_ROPE_INPUT_INDEX = 9;
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
// Constant
constexpr uint32_t MAX_BLOCK_SIZE = 1024;
constexpr uint32_t COPYND2NZ_SRC_STRIDE_LIMITATION = 65535;
constexpr uint32_t NUM_BYTES_FLOAT = 4;
constexpr uint32_t NUM_BYTES_FLOAT16 = 2;
constexpr uint32_t NUM_BYTES_BF16 = 2;
constexpr uint32_t BYTE_BLOCK = 32;
const uint32_t SFA_MAX_AIC_CORE_NUM = 26;
constexpr uint32_t OFFLOAD_SPARSE_INDICES_CAPACITY = 2048;
constexpr uint32_t OFFLOAD_SPARSE_COMPUTE_COUNT = 2048;
constexpr uint32_t OFFLOAD_MTP3_QUERY_COUNT = 4;

enum class CopySfaMtpSFALayout : uint32_t {
    BSND = 0,
    TND = 1,
    PA_BSND = 2
};

struct CopySfaMtpSFATilingShapeCompareParam {
    int64_t B = 1;
    int64_t S = 1;
    int64_t N = 1;
    int64_t D = 1;
    int64_t T = 1;
    // PA
    int64_t Bs = 1;
    int64_t Bn = 1;
};

enum class CopySfaMtpKvStorageMode : uint32_t {
    BATCH_CONTINUOUS = 0,
    PAGE_ATTENTION = 1
};

enum class CopySfaMtpSFAPerfMode : uint32_t {
    C_TEMPLATE_MODE = 0,
    V_TEMPLATE_MODE
};

enum class CopySfaMtpSFAAxis : uint32_t {
    B = 0,
    S = 1,
    N = 2,
    D = 3,
    K = 3,
    T = 5,
    Bn = 6, // block number
    Bs = 7, // block size
};

struct CopySfaMtpSFARequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct CopySfaMtpSFAOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

struct CopySfaMtpSFAParaInfo {
    CopySfaMtpSFARequiredParaInfo query = {nullptr, nullptr};
    CopySfaMtpSFARequiredParaInfo key = {nullptr, nullptr};
    CopySfaMtpSFARequiredParaInfo value = {nullptr, nullptr};
    CopySfaMtpSFARequiredParaInfo sparseIndices = {nullptr, nullptr};
    CopySfaMtpSFARequiredParaInfo cacheTokens = {nullptr, nullptr};
    CopySfaMtpSFAOptionalParaInfo blockTable = {nullptr, nullptr};
    CopySfaMtpSFAOptionalParaInfo actualSeqLengthsQ = {nullptr, nullptr};
    CopySfaMtpSFAOptionalParaInfo actualSeqLengths = {nullptr, nullptr};
    CopySfaMtpSFAOptionalParaInfo queryRope = {nullptr, nullptr};
    CopySfaMtpSFAOptionalParaInfo keyRope = {nullptr, nullptr};
    CopySfaMtpSFARequiredParaInfo attenOut = {nullptr, nullptr};

    const char *layoutQuery = nullptr;
    const char *layoutKV = nullptr;
    const int64_t *sparseBlockSize = nullptr;
    const float *scaleValue = nullptr;
    const int64_t *sparseMode = nullptr;
};

struct CopySfaMtpInnerSplitParams {
    uint32_t s1GBaseSize = 1;
    uint32_t s2BaseSize = 1;
};

BEGIN_TILING_DATA_DEF(CopySfaMtpAttentionBaseParamsMla)
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
TILING_DATA_FIELD_DEF(int64_t, sparseBlockSize)
TILING_DATA_FIELD_DEF(uint32_t, sparseBlockCount)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(CopySfaMtpAttentionBaseParamsMlaOp, CopySfaMtpAttentionBaseParamsMla)

BEGIN_TILING_DATA_DEF(CopySfaMtpAttentionSingleCoreParamsMla)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(CopySfaMtpAttentionSingleCoreParamsMlaOp, CopySfaMtpAttentionSingleCoreParamsMla)

BEGIN_TILING_DATA_DEF(CopySfaMtpAttentionSingleCoreTensorSizeMla)
TILING_DATA_FIELD_DEF(uint32_t, mmResUbSize);
TILING_DATA_FIELD_DEF(uint32_t, bmm2ResUbSize);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(CopySfaMtpAttentionSingleCoreTensorSizeMlaOp, CopySfaMtpAttentionSingleCoreTensorSizeMla)

BEGIN_TILING_DATA_DEF(CopySfaMtpAttentionSplitKVParamsMla)
TILING_DATA_FIELD_DEF(uint32_t, s2)
TILING_DATA_FIELD_DEF(uint32_t, accumOutSize)   // FD workspace
TILING_DATA_FIELD_DEF(uint32_t, logSumExpSize)  // FD workspace
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(CopySfaMtpAttentionSplitKVParamsMlaOp, CopySfaMtpAttentionSplitKVParamsMla)

BEGIN_TILING_DATA_DEF(CopySfaMtpAttentionInnerSplitParams)
TILING_DATA_FIELD_DEF(uint32_t, mBaseSize)
TILING_DATA_FIELD_DEF(uint32_t, s2BaseSize)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(CopySfaMtpAttentionInnerSplitParamsOp, CopySfaMtpAttentionInnerSplitParams)

BEGIN_TILING_DATA_DEF(CopySfaMtpAttentionTilingDataMla)
TILING_DATA_FIELD_DEF_STRUCT(CopySfaMtpAttentionBaseParamsMla, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(CopySfaMtpAttentionSplitKVParamsMla, splitKVParams);
TILING_DATA_FIELD_DEF_STRUCT(CopySfaMtpAttentionSingleCoreParamsMla, singleCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(CopySfaMtpAttentionSingleCoreTensorSizeMla, singleCoreTensorSize);
TILING_DATA_FIELD_DEF_STRUCT(CopySfaMtpAttentionInnerSplitParams, innerSplitParams);
END_TILING_DATA_DEF

template <typename T> inline T Align(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
}

template <typename T>
std::string SFAShape2String(const T &shape)
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
static std::string SFADataTypeToSerialString(ge::DataType type);
std::string CopySfaMtpTensorDesc2String(const gert::StorageShape *shape, const gert::CompileTimeTensorDesc *tensor);
std::string CopySfaMtpDebugTilingContext(const gert::TilingContext *context);
std::string SFALayoutToSerialString(CopySfaMtpSFALayout layout);

struct CopySfaMtpSFATilingInfo {
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    CopySfaMtpSFAParaInfo opParamInfo;

    // Base Param
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
    uint32_t bSize = 0;
    uint32_t n1Size = 0;
    uint32_t n2Size = 0;
    uint32_t s1Size = 0;
    int64_t s2Size = 0;
    uint32_t qkHeadDim = 0;
    uint32_t vHeadDim = 0;
    uint32_t gSize = 0;
    uint32_t ropeHeadDim = 0;
    uint32_t qTSize = 0;
    uint32_t kvTSize = 0;
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

    bool actualSeqLenFlag = false;
    bool isSameSeqAllKVTensor = true;
    bool isSameActualseq = true;
    uint32_t actualLenDimsKV = 0;
    std::vector<int64_t> kvListSeqLens {};

    uint32_t sparseMode = 0;

    ge::DataType inputQType = ge::DT_FLOAT16;
    ge::DataType inputKvType = ge::DT_FLOAT16;
    ge::DataType outputType = ge::DT_FLOAT16;

    CopySfaMtpKvStorageMode kvStorageMode = CopySfaMtpKvStorageMode::BATCH_CONTINUOUS;

    CopySfaMtpSFALayout qLayout = CopySfaMtpSFALayout::BSND;
    CopySfaMtpSFALayout topkLayout = CopySfaMtpSFALayout::BSND;
    CopySfaMtpSFALayout outLayout = CopySfaMtpSFALayout::BSND;
    CopySfaMtpSFALayout kvLayout = CopySfaMtpSFALayout::BSND;

    ge::DataType inputQRopeType = ge::DT_FLOAT16;
    ge::DataType inputKRopeType = ge::DT_FLOAT16;

    uint64_t l2CacheSize = 0;
};

class CopySfaMtpSFAMlaTiling {
public:
    explicit CopySfaMtpSFAMlaTiling(gert::TilingContext *context) : context_(context) {}
    ge::graphStatus DoOpTiling(CopySfaMtpSFATilingInfo *sfaInfo);

private:
    ge::graphStatus SetBlockDim(uint32_t blockDim);
    ge::graphStatus SetTilingKey(uint64_t tilingKey);
    ge::graphStatus SetWorkspaceSize(uint64_t workspaceSize);
    ge::graphStatus SetTilingData(TilingDef &tilingData);
    gert::TilingContext *context_ = nullptr;
    ge::graphStatus GetPlatformInfo();
    void GenTilingKey();
    bool DealSameSeqEachBatch();

    void ZeroTensorProcess();
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

    uint32_t CalcBalanceFDParamNums(const uint32_t actCoreNum);

    void CalcBlockDim();

    bool balanceModeFlag_ = false;
    bool splitKVFlag_ = false;

    uint32_t coreNum_ = 0;
    CopySfaMtpSFAPerfMode perfMode_ = CopySfaMtpSFAPerfMode::V_TEMPLATE_MODE;
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

    CopySfaMtpAttentionTilingDataMla tilingData_;
    uint32_t blockDim_{0};
    uint64_t workspaceSize_{0};
    uint64_t tilingKey_{0};

    uint32_t headDimAlign_ = 0;
    uint32_t mBaseSize_ = 128;
    uint32_t mFdBaseSize_ = 8;

    CopySfaMtpSFATilingInfo *sfaInfo_ = nullptr;
};

class CopySfaMtpSFATilingCheck {
public:
    explicit CopySfaMtpSFATilingCheck(const CopySfaMtpSFATilingInfo &sfaInfo) : sfaInfo_(sfaInfo) {};
    ~CopySfaMtpSFATilingCheck() = default;
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
    ge::graphStatus CheckDimNumInLayoutSupport(const CopySfaMtpSFALayout &layout,
        const gert::StorageShape *shape, const std::string &name) const;
    void LogErrorLayoutSupport(const std::vector<CopySfaMtpSFALayout> &expectLayoutList,
        const CopySfaMtpSFALayout &actualLayout, const std::string &name) const;
    ge::graphStatus GetExpectedShape(gert::Shape &shapeExpected,
    const CopySfaMtpSFATilingShapeCompareParam &param, const CopySfaMtpSFALayout &layout) const;
    ge::graphStatus CompareShape(CopySfaMtpSFATilingShapeCompareParam &param,
        const gert::Shape &shape, const CopySfaMtpSFALayout &layout, const std::string &name) const;
    ge::graphStatus CheckLayoutSupport(const CopySfaMtpSFALayout &actualLayout, const std::string &name) const;
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
        const CopySfaMtpSFALayout &layout, const std::string &name);
    void SetSFAShapeCompare();
    ge::graphStatus CheckQRope();
    ge::graphStatus CheckQRopeShape();
    ge::graphStatus CheckVAndKRopeShapeForBatchContinuous();
    uint32_t GetTypeSize(ge::DataType dtype) const;
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
    ge::graphStatus CheckFeatureMla() const;
    ge::graphStatus CheckFeature() const;
    ge::graphStatus CheckDecodeOnly() const;

private:
    const char *opName_;
    fe::PlatFormInfos *platformInfo_;
    CopySfaMtpSFAParaInfo opParamInfo_;
    const CopySfaMtpSFATilingInfo &sfaInfo_;

    uint32_t bSize_ = 0;
    uint32_t n1Size_ = 0;
    uint32_t n2Size_ = 0;
    uint32_t gSize_ = 0;
    uint32_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    uint32_t qkHeadDim_ = 0;
    uint32_t vHeadDim_ = 0;
    uint32_t ropeHeadDim_ = 0;
    uint32_t qTSize_ = 0;
    uint32_t kvTSize_ = 0;
    CopySfaMtpKvStorageMode kvStorageMode_ = CopySfaMtpKvStorageMode::BATCH_CONTINUOUS;
    uint32_t sparseBlockCount_ = 0;
    int64_t sparseBlockSize_ = 0;

    CopySfaMtpSFALayout qLayout_ = CopySfaMtpSFALayout::BSND;
    CopySfaMtpSFALayout topkLayout_ = CopySfaMtpSFALayout::BSND;
    CopySfaMtpSFALayout outLayout_ = CopySfaMtpSFALayout::BSND;
    CopySfaMtpSFALayout kvLayout_ = CopySfaMtpSFALayout::BSND;

    uint32_t maxBlockNumPerBatch_ = 0;
    int64_t blockSize_ = 0;

    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
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
};

class CopySfaMtpSFAInfoParser {
public:
    explicit CopySfaMtpSFAInfoParser(const gert::TilingContext *context) : context_(context) {}
    ~CopySfaMtpSFAInfoParser() = default;

    ge::graphStatus CheckRequiredInOutExistence() const;
    ge::graphStatus CheckRequiredAttrExistence() const;
    ge::graphStatus CheckRequiredParaExistence() const;

    ge::graphStatus GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
        CopySfaMtpSFALayout &layout, const std::string &name);
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
    ge::graphStatus GetN1Size();
    ge::graphStatus GetN2Size();
    ge::graphStatus GetGSize();
    ge::graphStatus GetSparseBlockCount();
    ge::graphStatus GetActualseqInfo();
    void GenerateInfo(CopySfaMtpSFATilingInfo &sfaInfo);
    ge::graphStatus Parse(CopySfaMtpSFATilingInfo &sfaInfo);

public:
    bool HasAxis(const CopySfaMtpSFAAxis &axis, const CopySfaMtpSFALayout &layout, const gert::Shape &shape) const;
    size_t GetAxisIdx(const CopySfaMtpSFAAxis &axis, const CopySfaMtpSFALayout &layout) const;
    uint32_t GetAxisNum(const gert::Shape &shape, const CopySfaMtpSFAAxis &axis,const CopySfaMtpSFALayout &layout) const;

    const gert::TilingContext *context_ = nullptr;

    const char *opName_;
    fe::PlatFormInfos *platformInfo_;
    CopySfaMtpSFAParaInfo opParamInfo_;
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
    uint32_t qTSize_ = 0;
    uint32_t kvTSize_ = 0;
    CopySfaMtpKvStorageMode kvStorageMode_ = CopySfaMtpKvStorageMode::BATCH_CONTINUOUS;
    uint32_t sparseBlockCount_ = 0;

    CopySfaMtpSFALayout qLayout_ = CopySfaMtpSFALayout::BSND;
    CopySfaMtpSFALayout topkLayout_ = CopySfaMtpSFALayout::BSND;
    CopySfaMtpSFALayout outLayout_ = CopySfaMtpSFALayout::BSND;
    CopySfaMtpSFALayout kvLayout_ = CopySfaMtpSFALayout::BSND;

    uint32_t maxBlockNumPerBatch_ = 0;
    uint32_t blockSize_ = 0;

    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;

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
#endif // SPARSE_TAIL_ATTENTION_TILING_H
