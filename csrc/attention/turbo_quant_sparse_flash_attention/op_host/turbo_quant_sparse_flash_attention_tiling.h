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
 * \file turbo_quant_sparse_flash_attention_tiling.h
 * \brief
 */
#ifndef TURBOQUANT_SPARSE_FLASH_ATTENTION_TILING_H
#define TURBOQUANT_SPARSE_FLASH_ATTENTION_TILING_H

#include <sstream>
#include <graph/utils/type_utils.h>
#include <tiling/platform/platform_ascendc.h>
#include <exe_graph/runtime/tiling_context.h>
#include "register/tilingdata_base.h"
namespace optiling {
// ------------------算子原型索引常量定义----------------
// Inputs Index
// 当前 kernel 的 MM1 K 宽度、vector scratch 容量均按下列值写死，
// host 侧须按同口径校验，泛化维度需同步改造 cube / vector / workspace / tiling。
constexpr uint32_t TQ4_Q_HEAD_DIM = 576U; // 512 latent + 64 rope
constexpr uint32_t TQ4_NOPE_HEAD_DIM = 512U;
constexpr uint32_t TQ4_ROPE_HEAD_DIM = 64U;
// KV slot 布局：256B 打包 nibble + 128B bfloat16 rope + 2B float16 归一化系数。
// 由头维派生而非写死 386，避免布局协议在多处重复维护。
constexpr uint32_t TQ4_KV_SLOT_BYTES = TQ4_NOPE_HEAD_DIM / 2U + TQ4_ROPE_HEAD_DIM * 2U + 2U; // = 386
// kernel 未消费 tile_size 属性，slot 布局按 128 写死
constexpr int64_t TQ4_TILE_SIZE = 128;

constexpr uint32_t QUERY_INPUT_INDEX = 0;
constexpr uint32_t KEY_INPUT_INDEX = 1;
constexpr uint32_t VALUE_INPUT_INDEX = 2;
constexpr uint32_t SPARSE_INDICES_INPUT_INDEX = 3;
constexpr uint32_t KEY_DEQUANT_SCALE_INPUT_INDEX = 4;
constexpr uint32_t VALUE_DEQUANT_SCALE_INPUT_INDEX = 5;
constexpr uint32_t BLOCK_TABLE_INPUT_INDEX = 6;
constexpr uint32_t ACT_SEQ_LEN_Q_INPUT_INDEX = 7;
constexpr uint32_t ACT_SEQ_LEN_KV_INPUT_INDEX = 8;
// Outputs Index
constexpr uint32_t OUTPUT_INDEX = 0;
// Attributes Index
constexpr uint32_t SCALE_VALUE_ATTR_INDEX = 0;
constexpr uint32_t KEY_QUANT_MODE_ATTR_INDEX = 1;
constexpr uint32_t VALUE_QUANT_MODE_ATTR_INDEX = 2;
constexpr uint32_t SPARSE_BLOCK_SIZE_ATTR_INDEX = 3;
constexpr uint32_t LAYOUT_QUERY_ATTR_INDEX = 4;
constexpr uint32_t LAYOUT_KV_ATTR_INDEX = 5;
constexpr uint32_t SPARSE_MODE_ATTR_INDEX = 6;
constexpr uint32_t PRE_TOKENS_ATTR_INDEX = 7;
constexpr uint32_t NEXT_TOKENS_ATTR_INDEX = 8;
constexpr uint32_t ATTENTION_MODE_ATTR_INDEX = 9;
constexpr uint32_t QUANT_SCALE_REPO_MODE_ATTR_INDEX = 10;
constexpr uint32_t TILE_SIZE_ATTR_INDEX = 11;
constexpr uint32_t ROPE_HEAD_DIM_ATTR_INDEX = 12;
constexpr uint32_t RETURN_SOFTMAX_LSE_ATTR_INDEX = 13;
// Dim Num
constexpr size_t DIM_NUM_TWO = 2;
constexpr size_t DIM_NUM_THREE = 3;
constexpr size_t DIM_NUM_FOUR = 4;
// 常量
constexpr uint32_t MAX_BLOCK_SIZE = 1024;
constexpr uint32_t NUM_BYTES_FLOAT16 = 2;
constexpr uint32_t NUM_BYTES_BF16 = 2;
constexpr uint32_t BYTE_BLOCK = 32;

// ------------------公共定义--------------------------
enum class TQSFALayout : uint32_t {
    BSND = 0,
    TND = 1,
    PA_BSND = 2
};

struct TQSFATilingShapeCompareParam {
    int64_t B = 1;
    int64_t S = 1;
    int64_t N = 1;
    int64_t D = 1;
    int64_t T = 1;
    // PA
    int64_t Bs = 1;
    int64_t Bn = 1;
};

enum class TQSFAPerfMode : uint32_t {
    C_TEMPLATE_MODE = 0,
    V_TEMPLATE_MODE
};

enum class TQSFAAxis : uint32_t {
    B = 0,
    S = 1,
    N = 2,
    D = 3,
    K = 3, // sparse_indices的K和key的D枚举值相同，表达相同位置，最后一维
    T = 5,
    Bn = 6, // block number
    Bs = 7, // block size
};

// 两者的区别是「携带 shape」还是「携带 tensor」，与输入的必选 / 可选无关：
// 需要按维度校验的走 ShapeParaInfo，需要取 gert::Tensor（如 ShapeSize）的走
// TensorParaInfo。原名 Required / Optional 易被误解为输入的必选性。
struct TQSFAShapeParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct TQSFATensorParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

// -----------算子Tiling入参结构体定义--------------
struct TQSFAParaInfo {
    TQSFAShapeParaInfo query = {nullptr, nullptr};
    TQSFAShapeParaInfo key = {nullptr, nullptr};
    TQSFAShapeParaInfo value = {nullptr, nullptr};
    TQSFAShapeParaInfo sparseIndices = {nullptr, nullptr};
    // 下面三者已在 OpDef 中声明为 REQUIRED，取值走必选输入接口（见
    // GetOptionalInputParaInfo）；此处沿用 TensorParaInfo 是因为后续校验需要
    // gert::Tensor（取 ShapeSize 等），与其必选性无关。
    TQSFATensorParaInfo blockTable = {nullptr, nullptr};
    TQSFATensorParaInfo actualSeqLengthsQ = {nullptr, nullptr};
    TQSFATensorParaInfo actualSeqLengths = {nullptr, nullptr};
    TQSFATensorParaInfo queryRope = {nullptr, nullptr};
    TQSFATensorParaInfo keyRope = {nullptr, nullptr};
    TQSFATensorParaInfo keyDequantScale = {nullptr, nullptr};
    TQSFATensorParaInfo valueDequantScale = {nullptr, nullptr};
    TQSFAShapeParaInfo attenOut = {nullptr, nullptr};

    const char *layoutQuery = nullptr;
    const char *layoutKV = nullptr;
    const int64_t *sparseBlockSize = nullptr;
    const uint32_t *sparseBlockCount = nullptr;
    const uint32_t *blockSize = nullptr;
    const float *scaleValue = nullptr;
    const int64_t *sparseMode = nullptr;
    const int64_t *attentionMode = nullptr;
    const int64_t *keyQuantMode = nullptr;
    const int64_t *valueQuantMode = nullptr;
    const int64_t *quantScaleRepoMode = nullptr;
    const int64_t *tileSize = nullptr;
    const int64_t *ropeHeadDim = nullptr;
    const int64_t *preTokens = nullptr;
    const int64_t *nextTokens = nullptr;
    const bool *returnSoftmaxLse = nullptr;
};

struct InnerSplitParams {
    uint32_t s1GBaseSize = 1;
    uint32_t s2BaseSize = 1;
};

// -----------算子TilingData定义---------------
BEGIN_TILING_DATA_DEF(TurboQuantSparseFlashAttentionBaseParamsMla)
TILING_DATA_FIELD_DEF(uint32_t, batchSize)
TILING_DATA_FIELD_DEF(uint32_t, seqSize)
TILING_DATA_FIELD_DEF(uint32_t, qSeqSize)
TILING_DATA_FIELD_DEF(int64_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, actualLenDimsQ)
TILING_DATA_FIELD_DEF(uint32_t, actualLenDimsKV)
TILING_DATA_FIELD_DEF(float, scaleValue)
TILING_DATA_FIELD_DEF(uint32_t, nNumOfQInOneGroup)
TILING_DATA_FIELD_DEF(uint32_t, outputLayout)
TILING_DATA_FIELD_DEF(uint32_t, sparseMode)
TILING_DATA_FIELD_DEF(int64_t, sparseBlockSize)
TILING_DATA_FIELD_DEF(uint32_t, sparseBlockCount)
TILING_DATA_FIELD_DEF(int64_t, dSizeVInput)
TILING_DATA_FIELD_DEF(uint32_t, headDim)
TILING_DATA_FIELD_DEF(uint32_t, ropeHeadDim)
TILING_DATA_FIELD_DEF(int64_t, keyQuantMode)
TILING_DATA_FIELD_DEF(int64_t, valueQuantMode)
TILING_DATA_FIELD_DEF(int64_t, tileSize)
TILING_DATA_FIELD_DEF(uint32_t, isActualLenDimsNull)
TILING_DATA_FIELD_DEF(uint32_t, isActualLenDimsKVNull)
TILING_DATA_FIELD_DEF(uint32_t, returnSoftmaxLse)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(TurboQuantSparseFlashAttentionBaseParamsMlaOp, TurboQuantSparseFlashAttentionBaseParamsMla)

BEGIN_TILING_DATA_DEF(TurboQuantSparseFlashAttentionSingleCoreParamsMla)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(TurboQuantSparseFlashAttentionSingleCoreParamsMlaOp,
                           TurboQuantSparseFlashAttentionSingleCoreParamsMla)

BEGIN_TILING_DATA_DEF(TurboQuantSparseFlashAttentionSingleCoreTensorSizeMla)
TILING_DATA_FIELD_DEF(uint32_t, mmResUbSize);
TILING_DATA_FIELD_DEF(uint32_t, bmm2ResUbSize);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(TurboQuantSparseFlashAttentionSingleCoreTensorSizeMlaOp,
                           TurboQuantSparseFlashAttentionSingleCoreTensorSizeMla)

BEGIN_TILING_DATA_DEF(TurboQuantSparseFlashAttentionSplitKVParamsMla)
TILING_DATA_FIELD_DEF(uint32_t, s2)            // S2切分份数
TILING_DATA_FIELD_DEF(uint32_t, accumOutSize)  // FD workspace
TILING_DATA_FIELD_DEF(uint32_t, logSumExpSize) // FD workspace
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(TurboQuantSparseFlashAttentionSplitKVParamsMlaOp,
                           TurboQuantSparseFlashAttentionSplitKVParamsMla)

// 内切基本块参数
BEGIN_TILING_DATA_DEF(TurboQuantSparseFlashAttentionInnerSplitParams)
TILING_DATA_FIELD_DEF(uint32_t, mBaseSize)
TILING_DATA_FIELD_DEF(uint32_t, s2BaseSize)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(TurboQuantSparseFlashAttentionInnerSplitParamsOp,
                           TurboQuantSparseFlashAttentionInnerSplitParams)

BEGIN_TILING_DATA_DEF(TurboQuantSparseFlashAttentionTilingDataMla)
TILING_DATA_FIELD_DEF_STRUCT(TurboQuantSparseFlashAttentionBaseParamsMla, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(TurboQuantSparseFlashAttentionSplitKVParamsMla, splitKVParams);
TILING_DATA_FIELD_DEF_STRUCT(TurboQuantSparseFlashAttentionSingleCoreParamsMla, singleCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(TurboQuantSparseFlashAttentionSingleCoreTensorSizeMla, singleCoreTensorSize);
TILING_DATA_FIELD_DEF_STRUCT(TurboQuantSparseFlashAttentionInnerSplitParams, innerSplitParams);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(TurboQuantSparseFlashAttention, TurboQuantSparseFlashAttentionTilingDataMla)

template <typename T>
inline T Align(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

static std::string TQSFADataTypeToSerialString(ge::DataType type);
std::string TQSFALayoutToSerialString(TQSFALayout layout);

// -----------算子Tiling入参信息定义--------------
struct TQSFATilingInfo {
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    TQSFAParaInfo opParamInfo;

    // Base Param
    uint32_t bSize = 0;
    uint32_t n1Size = 0;
    uint32_t n2Size = 0;
    uint32_t s1Size = 0;
    int64_t s2Size = 0;
    uint32_t qHeadDim = 0;
    uint32_t kHeadDim = 0;
    uint32_t vHeadDim = 0;
    uint32_t gSize = 0;
    uint32_t ropeHeadDim = 0;
    uint32_t qTSize = 0;  // 仅TND时生效
    uint32_t kvTSize = 0; // 仅TND时生效
    float scaleValue = 0;
    uint32_t innerPrecise = 0;
    uint32_t l2CacheOffFlag = 0;
    int64_t sparseBlockSize = 0;
    int64_t sparseBlockCount = 0;

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
    std::vector<int64_t> kvListSeqLens{};

    uint32_t sparseMode = 0;

    int64_t attentionMode = 0;
    int64_t keyQuantMode = 0;
    int64_t valueQuantMode = 0;
    int64_t quantScaleRepoMode = 0;
    int64_t tileSize = 0;
    int64_t preTokens = 0;
    int64_t nextTokens = 0;
    bool returnSoftmaxLse = false;

    ge::DataType inputQType = ge::DT_FLOAT16;
    ge::DataType inputKvType = ge::DT_FLOAT16;
    ge::DataType outputType = ge::DT_FLOAT16;

    TQSFALayout qLayout = TQSFALayout::BSND;
    TQSFALayout topkLayout = TQSFALayout::BSND;
    TQSFALayout outLayout = TQSFALayout::BSND;
    TQSFALayout kvLayout = TQSFALayout::BSND;

    ge::DataType inputQRopeType = ge::DT_FLOAT16;
    ge::DataType inputKRopeType = ge::DT_FLOAT16;

    uint64_t l2CacheSize = 0;
    int64_t dSizeVInput = 0;
};

// ---------------算子Tiling类--------------
class TQSFAMlaTiling {
public:
    explicit TQSFAMlaTiling(gert::TilingContext *context)
        : context_(context)
    {}
    ge::graphStatus DoOpTiling(TQSFATilingInfo *qsfaInfo);

private:
    ge::graphStatus SetBlockDim(uint32_t blockDim) const;
    ge::graphStatus SetTilingKey(uint64_t tilingKey) const;
    ge::graphStatus SetWorkspaceSize(uint64_t workspaceSize) const;
    ge::graphStatus SetTilingData(TilingDef &tilingData) const;
    gert::TilingContext *context_ = nullptr;
    ge::graphStatus GetPlatformInfo();
    void GenTilingKey();

    void InitParams();

    void Split();

    void SplitBalanced();
    void CalcInnerSize(uint32_t qsfaS2Size);

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

    bool balanceModeFlag_ = false;
    bool splitKVFlag_ = false;

    uint32_t coreNum_ = 0;
    TQSFAPerfMode perfMode_ = TQSFAPerfMode::V_TEMPLATE_MODE;
    uint32_t kvSplitPart_ = 1;
    size_t mmResUbSize_ = 0;
    size_t bmm2ResUbSize_ = 0;
    size_t qPreSizeMla_ = 0;
    uint32_t sInnerLoopTimes_ = 0;
    uint32_t sInnerSize_ = 0;
    uint32_t sInnerSizeAlign_ = 0;
    uint32_t kvSplit_ = 0;
    uint32_t usedCoreNum_ = 0;
    uint32_t formerCoreNum_ = 0;
    uint32_t blockSplitBn2Range_ = 0;
    uint32_t tailSplitedBatchRange_ = 0;

    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    size_t libapiSize_ = 0;

    TurboQuantSparseFlashAttentionTilingDataMla tilingData_;
    uint32_t blockDim_{0};
    uint64_t workspaceSize_{0};
    uint64_t tilingKey_{0};

    uint32_t headDimAlign_ = 0;
    uint32_t mBaseSize_ = 128;
    uint32_t mFdBaseSize_ = 8;

    TQSFATilingInfo *qsfaInfo_ = nullptr;
};

// -----------算子Tiling入参信息解析及Check类--------------
class TQSFATilingCheck {
public:
    explicit TQSFATilingCheck(const TQSFATilingInfo &qsfaInfo)
        : qsfaInfo_(qsfaInfo) {};
    ~TQSFATilingCheck() = default;
    ge::graphStatus Process();

private:
    void Init();
    void LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList, const ge::DataType &actualDtype,
                              const std::string &name) const;
    ge::graphStatus CheckDtypeSupport(const gert::CompileTimeTensorDesc *qsfaDesc, const std::string &name) const;
    template <typename T>
    void LogErrorNumberSupport(const std::vector<T> &expectNumberList, const T &actualValue, const std::string &name,
                               const std::string subName) const;
    template <typename T>
    void LogErrorDimNumSupport(const std::vector<T> &expectNumberList, const T &actualValue,
                               const std::string &name) const;
    ge::graphStatus CheckDimNumSupport(const gert::StorageShape *shape, const std::vector<size_t> &qsfaExpectDimNumList,
                                       const std::string &name) const;
    ge::graphStatus CheckDimNumInLayoutSupport(const TQSFALayout &layout, const gert::StorageShape *shape,
                                               const std::string &name) const;
    void LogErrorLayoutSupport(const std::vector<TQSFALayout> &expectLayoutList, const TQSFALayout &actualLayout,
                               const std::string &name) const;
    ge::graphStatus GetExpectedShape(gert::Shape &shapeExpected, const TQSFATilingShapeCompareParam &param,
                                     const TQSFALayout &layout) const;
    ge::graphStatus CompareShape(TQSFATilingShapeCompareParam &param, const gert::Shape &shape,
                                 const TQSFALayout &layout, const std::string &name) const;
    ge::graphStatus CheckLayoutSupport(const TQSFALayout &actualLayout, const std::string &name) const;
    ge::graphStatus CheckSingleParaQuery() const;
    ge::graphStatus CheckSingleParaKey() const;
    ge::graphStatus CheckSingleParaSparseMode() const;
    ge::graphStatus CheckSingleParaSparseBlockSize() const;
    ge::graphStatus CheckSingleParaSparseIndices() const;
    ge::graphStatus CheckSinglePara() const;
    ge::graphStatus CheckMultiParaConsistency() const;
    ge::graphStatus CheckParaExistenceMlaAntiquant() const;
    ge::graphStatus CheckParaExistenceMla() const;
    ge::graphStatus CheckParaExistence();
    void SetTQSFAShapeCompare();
    ge::graphStatus CheckKV();
    ge::graphStatus CheckTopK();
    ge::graphStatus CheckTopkShape();
    ge::graphStatus CheckBlockTable() const;
    ge::graphStatus CheckDTypeConsistency(const ge::DataType &actualDtype, const ge::DataType &expectDtype,
                                          const std::string &name) const;

    ge::graphStatus CheckAttenOut();
    ge::graphStatus CheckAttenOutShape();
    ge::graphStatus CheckActualSeqLensQ();
    ge::graphStatus CheckActualSeqLensQShape();
    ge::graphStatus CheckActualSeqLensQDType();
    ge::graphStatus CheckActualSeqLens();
    ge::graphStatus CheckActualSeqLensDType();
    ge::graphStatus CheckActualSeqLensShape();
    ge::graphStatus CheckMultiParaConsistency();

    ge::graphStatus CheckFeatureMlaAntiquantShape() const;
    ge::graphStatus CheckFeatureMlaAntiquantShapeSizes() const;
    ge::graphStatus CheckFeatureMlaAntiquantShapeSparseAndHeadDim() const;
    ge::graphStatus CheckFeatureMlaAntiquantLayout() const;
    ge::graphStatus CheckFeatureMlaAntiquantDtype() const;
    ge::graphStatus CheckFeatureMlaAntiquantAttr() const;
    ge::graphStatus CheckFeatureMlaAntiquantPa() const;
    ge::graphStatus CheckFeatureMlaAntiquant() const;
    ge::graphStatus CheckFeatureMla() const;
    ge::graphStatus CheckFeature() const;

private:
    const char *opName_;
    fe::PlatFormInfos *platformInfo_;
    TQSFAParaInfo opParamInfo_;
    const TQSFATilingInfo &qsfaInfo_;

    uint32_t bSize_ = 0;
    uint32_t n1Size_ = 0;
    uint32_t n2Size_ = 0;
    uint32_t gSize_ = 0;
    uint32_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    uint32_t qHeadDim_ = 0;
    uint32_t kHeadDim_ = 0;
    uint32_t vHeadDim_ = 0;
    uint32_t qTSize_ = 0;  // 仅TND时生效
    uint32_t kvTSize_ = 0; // 仅TND时生效
    uint32_t sparseBlockCount_ = 0;
    int64_t sparseBlockSize_ = 0;
    int32_t attentionMode_ = 0;
    int32_t keyQuantMode_ = 0;
    int32_t valueQuantMode_ = 0;
    int32_t quantScaleRepoMode_ = 0;
    int64_t tileSize_ = 0;
    int64_t preTokens_ = 0;
    int64_t nextTokens_ = 0;
    int32_t ropeHeadDim_ = 0;

    TQSFALayout qLayout_ = TQSFALayout::BSND;
    TQSFALayout topkLayout_ = TQSFALayout::BSND;
    TQSFALayout outLayout_ = TQSFALayout::BSND;
    TQSFALayout kvLayout_ = TQSFALayout::BSND;

    uint32_t maxBlockNumPerBatch_ = 0;
    int64_t blockSize_ = 0;

    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    uint64_t l2CacheSize_ = 0;

    ge::DataType inputQType_ = ge::DT_FLOAT16;
    ge::DataType inputKvType_ = ge::DT_FLOAT16;
    ge::DataType outputType_ = ge::DT_FLOAT16;

    gert::Shape queryShapeCmp_{};
    gert::Shape keyShapeCmp_{};
    gert::Shape valueShapeCmp_{};
    gert::Shape topkShapeCmp_{};
    gert::Shape attenOutShapeCmp_{};
};

class TQSFAInfoParser {
public:
    explicit TQSFAInfoParser(const gert::TilingContext *context)
        : context_(context)
    {}
    ~TQSFAInfoParser() = default;

    ge::graphStatus CheckRequiredInOutExistence() const;
    ge::graphStatus CheckRequiredAttrExistence() const;
    ge::graphStatus CheckRequiredParaExistence() const;

    ge::graphStatus GetActualSeqLenQSize(uint32_t &size);
    ge::graphStatus GetNpuInfo();
    ge::graphStatus GetOpName();
    void GetOptionalInputParaInfo();
    void GetInputParaInfo();
    void GetOutputParaInfo();
    ge::graphStatus GetAttrParaInfo();
    ge::graphStatus GetOpParaInfo();

    ge::graphStatus GetInOutDataType();
    ge::graphStatus GetQTSize();
    ge::graphStatus GetBatchSize();
    ge::graphStatus GetKVTSize();
    ge::graphStatus GetQHeadDim();
    ge::graphStatus GetKHeadDim();
    ge::graphStatus GetS1Size();
    ge::graphStatus GetKvLayout();
    void SetTQSFAShape();
    ge::graphStatus GetMaxBlockNumPerBatch();
    ge::graphStatus GetBlockSize();
    ge::graphStatus GetS2SizeForPageAttention();
    ge::graphStatus GetS2Size();
    ge::graphStatus GetValueHeadDim();
    ge::graphStatus GetDSizeKV();
    ge::graphStatus GetQueryAndOutLayout();
    ge::graphStatus GetTopkLayout();
    ge::graphStatus GetN1Size();
    ge::graphStatus GetN2Size();
    ge::graphStatus GetGSize();
    ge::graphStatus GetSparseBlockCount();
    ge::graphStatus GetActualseqInfo();
    ge::graphStatus GetShapeAndSizeInfo();
    void GenerateInfo(TQSFATilingInfo &qsfaInfo);
    void FillTilingInfoAttrsAndLayouts(TQSFATilingInfo &qsfaInfo);
    ge::graphStatus Parse(TQSFATilingInfo &qsfaInfo);

    const gert::TilingContext *context_ = nullptr;

    const char *opName_;
    fe::PlatFormInfos *platformInfo_;
    TQSFAParaInfo opParamInfo_;

    uint32_t bSize_ = 0;
    uint32_t n1Size_ = 0;
    uint32_t n2Size_ = 0;
    uint32_t gSize_ = 0;
    uint32_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    uint32_t qHeadDim_ = 0;
    uint32_t kHeadDim_ = 0;
    uint32_t vHeadDim_ = 0;
    int32_t ropeHeadDim_ = 0;
    int64_t dSizeKV_ = 0;
    uint32_t qTSize_ = 0;  // 仅TND时生效
    uint32_t kvTSize_ = 0; // 仅TND时生效
    uint32_t sparseBlockCount_ = 0;

    TQSFALayout qLayout_ = TQSFALayout::BSND;
    TQSFALayout topkLayout_ = TQSFALayout::BSND;
    TQSFALayout outLayout_ = TQSFALayout::BSND;
    TQSFALayout kvLayout_ = TQSFALayout::BSND;

    uint32_t maxBlockNumPerBatch_ = 0;
    uint32_t blockSize_ = 0;

    ge::DataType inputQType_ = ge::DT_FLOAT16;
    ge::DataType inputKvType_ = ge::DT_FLOAT16;
    ge::DataType outputType_ = ge::DT_FLOAT16;

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
};
} // namespace optiling
#endif // TURBOQUANT_SPARSE_FLASH_ATTENTION_TILING_H
