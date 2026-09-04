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
 * \file fused_infer_attention_score_tiling_check.h
 * \brief
 */
#ifndef FUSED_INFER_ATTENTION_SCORE_TILING_CHECK_H
#define FUSED_INFER_ATTENTION_SCORE_TILING_CHECK_H

#include "../../../common/op_host/fia_tiling_info.h"
#include "../../../common/op_host/fia_tiling_shape.h"

namespace optiling {
constexpr int32_t INNER_PRECISE_HIGH_PRECISION = 0;
constexpr int32_t INNER_PRECISE_HIGH_PERFORMANCE = 1;
constexpr int32_t INNER_PRECISE_HIGH_PRECISION_ROW_INVALID = 2;
constexpr int32_t INNER_PRECISE_HIGH_PERFORMANCE_ROW_INVALID = 3;

constexpr int64_t ANTIQUANT_PER_CHANNEL_MODE = 0;
constexpr int64_t ANTIQUANT_PER_TOKEN_MODE = 1;
constexpr int64_t ANTIQUANT_PER_TENSOR_HEAD_MODE = 2;
constexpr int64_t ANTIQUANT_PER_TOKEN_HEAD_MODE = 3;
constexpr int64_t ANTIQUANT_PER_TOKEN_PA_MODE = 4;
constexpr int64_t ANTIQUANT_PER_TOKEN_HEAD_PA_MODE = 5;
constexpr int64_t ANTIQUANT_PER_CHANNEL_TOKEN_MODE = 2;

constexpr uint32_t PRETOKEN_LIMIT_2K = 2048;
constexpr uint32_t KVS_LIMIT = 131088;

constexpr size_t DIM_NUM_ONE = 1;
constexpr size_t DIM_NUM_TWO = 2;
constexpr size_t DIM_NUM_THREE = 3;
constexpr size_t DIM_NUM_FOUR = 4;
constexpr size_t DIM_NUM_FIVE = 5;

constexpr size_t HEAD_DIM_512 = 512;
constexpr size_t SHAPE_NUM_ONE = 1;

std::string RopeModeToSerialString(const RopeMode &ropeMode);
std::string FusedDataTypeToSerialString(ge::DataType type);

template <typename vecT, typename T>
static bool VecContains(const vecT& vec, const T& value)
{
    return std::find(vec.begin(), vec.end(), value) != vec.end();
}

class TilingCheck {
public:
    static ge::graphStatus Check(const FiaTilingInfo &fiaInfo);

protected:
    explicit TilingCheck(const FiaTilingInfo &fiaInfo) : fiaInfo_(fiaInfo) {};
    virtual ~TilingCheck() = default;

    virtual ge::graphStatus Process() = 0;

protected:
    const FiaTilingInfo &fiaInfo_;
};

class FiaTilingCheck : public TilingCheck {
public:
    explicit FiaTilingCheck(const FiaTilingInfo &fiaInfo) : TilingCheck(fiaInfo) {};
    virtual ~FiaTilingCheck() override = default;

    virtual ge::graphStatus Process() override;

private:
    void Init();

    // 单参数校验
    void LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList,
        const ge::DataType &actualDtype, const std::string &name) const;
    ge::graphStatus CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc,
        const std::string &name) const;
    ge::graphStatus CheckFormatSupport(const gert::CompileTimeTensorDesc *desc,
        const std::string &name) const;
    template <typename T>
    void LogErrorNumberSupport(const std::vector<T> &expectNumberList,
        const T &actualValue, const std::string &name, const std::string subName) const;
    template <typename T>
    void LogErrorDimNumSupport(const std::vector<T> &expectNumberList,
        const T &actualValue, const std::string &name) const;
    template <typename T>
    void LogErrorShapeNumSupport(const std::vector<T> &expectNumberList,
        const T &actualValue, const std::string &name) const;
    template <typename T>
    void LogErrorAttrValueSupport(const std::vector<T> &expectNumberList,
        const T &actualValue, const std::string &name) const;
    ge::graphStatus CheckDimNumSupport(const gert::StorageShape *shape,
        const std::vector<size_t> &expectDimNumList, const std::string &name) const;
    ge::graphStatus CheckDimNumSupport(const gert::Tensor *tensor,
        const std::vector<size_t> &expectDimNumList, const std::string &name) const;
    ge::graphStatus CheckShapeSupport(const gert::Tensor *tensor,
        const std::vector<int64_t> &expectShapeList, const std::string &name) const;
    template <typename T>
    ge::graphStatus CheckAttrValueSupport(const T *attrValue,
        const std::vector<T> &expectAttrValList, const std::string &name) const;
    void LogErrorLayoutSupport(const std::vector<FiaLayout> &expectLayoutList,
        const FiaLayout &actualLayout, const std::string &name) const;
    ge::graphStatus CheckLayoutSupport(const FiaLayout &actualLayout, const std::string &name) const;

    ge::graphStatus CheckSingleParaQuery() const;
    ge::graphStatus CheckSingleParaKey() const;
    ge::graphStatus CheckSingleParaValue() const;
    ge::graphStatus CheckSingleParaPseShift() const;
    ge::graphStatus CheckSingleParaAttenMask() const;
    ge::graphStatus CheckSingleParaActualSeqLengthsQ() const;
    ge::graphStatus CheckSingleParaActualSeqLengths() const;
    ge::graphStatus CheckSingleParaDeqScale1() const;
    ge::graphStatus CheckSingleParaQuantScale1() const;
    ge::graphStatus CheckSingleParaDeqScale2() const;
    ge::graphStatus CheckSingleParaQuantScale2() const;
    ge::graphStatus CheckSingleParaQuantOffset2() const;
    ge::graphStatus CheckSingleParaAntiquantScale() const;
    ge::graphStatus CheckSingleParaAntiquantOffset() const;
    ge::graphStatus CheckSingleParaBlockTable() const;
    ge::graphStatus CheckSingleParaQueryPaddingSize() const;
    ge::graphStatus CheckSingleParaKvPaddingSize() const;
    ge::graphStatus CheckSingleParaKeyAntiquantScale() const;
    ge::graphStatus CheckSingleParaKeyAntiquantOffset() const;
    ge::graphStatus CheckSingleParaValueAntiquantScale() const;
    ge::graphStatus CheckSingleParaValueAntiquantOffset() const;
    ge::graphStatus CheckSingleParaKeySharedPrefix() const;
    ge::graphStatus CheckSingleParaValueSharedPrefix() const;
    ge::graphStatus CheckSingleParaQueryRope() const;
    ge::graphStatus CheckSingleParaKeyRope() const;
    ge::graphStatus CheckSingleParaKeyRopeAntiquantScale() const;
    ge::graphStatus CheckSingleParaDequantScaleQuery() const;
    ge::graphStatus CheckSingleParaAttenOut() const;
    ge::graphStatus CheckSingleParaLseOut() const;
    ge::graphStatus CheckSingleParaNumHeads() const;
    ge::graphStatus CheckSingleParaPreToken() const;
    ge::graphStatus CheckSingleParaNextToken() const;
    ge::graphStatus CheckSingleParaScaleValue() const;
    ge::graphStatus CheckSingleParaKvHeadNums() const;
    ge::graphStatus CheckSingleParaLayout() const;
    ge::graphStatus CheckSingleParaBlockSize() const;
    ge::graphStatus CheckSingleParaInnerPrecise() const;
    ge::graphStatus CheckSingleParaAntiquantMode() const;
    ge::graphStatus CheckSingleParaSoftmaxLseFlag() const;
    ge::graphStatus CheckSingleParaKeyAntiquantMode() const;
    ge::graphStatus CheckSingleParaValueAntiquantMode() const;
    ge::graphStatus CheckSingleParaSparseMode() const;
    ge::graphStatus CheckSingleParaQueryQuantMode() const;
    ge::graphStatus CheckSinglePara() const;

    // 存在性校验
    ge::graphStatus CheckRopeExistence() const;
    ge::graphStatus CheckDtypeAndSetQuantFlagMla();
    ge::graphStatus CheckDtypeAndSetQuantFlagGqa();
    ge::graphStatus CheckDtypeAndSetQuantFlag();

    ge::graphStatus CheckExists(const void *pointer, const std::string &name) const;
    ge::graphStatus CheckNotExists(const void *pointer, const std::string &name) const;
    ge::graphStatus CheckExistsByMap(const std::map<std::string, const void *> &paramMap) const;
    ge::graphStatus CheckNotExistsByMap(const std::map<std::string, const void *> &paramMap) const;
    ge::graphStatus CheckExistenceByMap(std::map<std::string, const void *> &existMap,
        std::map<std::string, const void *> &notExistMap) const;
    void LogErrorExistenceEqual(std::map<std::string, const void *> &paramMap) const;
    ge::graphStatus CheckParaExistenceEqual(std::map<std::string, const void *> &paramMap) const;
    template <typename T>
    ge::graphStatus CheckAttrValueByMap(std::map<std::string, std::pair<const T *, T>> &attrMap) const;
    ge::graphStatus CheckParaExistenceMlaNoquant() const;
    ge::graphStatus CheckParaExistenceMlaAntiquant() const;
    ge::graphStatus CheckParaExistenceMlaFullquant() const;
    ge::graphStatus CheckParaExistenceGqaNoquant() const;
    ge::graphStatus CheckParaExistenceGqaNoquantForFullquant() const;
    ge::graphStatus CheckParaExistenceGqaAntiquantInt8Inner() const;
    ge::graphStatus CheckParaExistenceGqaAntiquantInt8() const;
    ge::graphStatus CheckParaExistenceGqaAntiquantInt4() const;
    ge::graphStatus CheckParaExistenceGqaAntiquant() const;
    ge::graphStatus CheckParaExistenceGqaFullquant() const;
    ge::graphStatus CheckParaExistenceMla() const;
    ge::graphStatus CheckParaExistenceGqa() const;
    ge::graphStatus CheckParaExistence();

    ge::graphStatus CheckActualSeqLensQ() const;
    ge::graphStatus CheckActualSeqLensKv() const;
    ge::graphStatus CheckBlockTable() const;
    ge::graphStatus CheckExistenceSystemPrefix() const;
    // 特性交叉校验
    ge::graphStatus CheckFeatureInOutDtype() const;
    ge::graphStatus CheckFeatureActualSeqLensExistence() const;
    ge::graphStatus GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
        const FiaLayout &layout, const std::string &actualSeqLenName, const std::string &attrName);
    ge::graphStatus CheckFeatureActualSeqLensQData();
    ge::graphStatus CheckFeatureActualSeqLensKvData();
    ge::graphStatus CheckFeatureActualSeqLens();
    ge::graphStatus CheckFeatureNoQuantDtype() const;
    ge::graphStatus CheckFeatureLse() const;
    ge::graphStatus CheckFeatureBlockSize() const;
    ge::graphStatus CheckFeatureAxisInfo() const;
    ge::graphStatus CheckFeatureMlaNoquant();
    ge::graphStatus CheckFeatureMlaAntiquant() const;
    ge::graphStatus CheckFeatureMlaFullquant() const;
    ge::graphStatus CheckFeatureTensorList() const;
    ge::graphStatus CheckFeatureSparseMode() const;
    ge::graphStatus CheckFeatureMask() const;
    ge::graphStatus CheckFeatureLearnableSink() const;
    ge::graphStatus CheckFeatureLayout() const;
    ge::graphStatus CheckFeatureGqaNoQuantShape() const;
    ge::graphStatus CheckFeatureGqaNoquant();
    ge::graphStatus CheckFeatureGqaAntiquant() const;
    ge::graphStatus CheckFeatureGqaFullquant() const;
    ge::graphStatus CheckFeatureGqaPrefix() const;
    ge::graphStatus CheckFeaturePostQuant() const;
    int64_t GetQActualSeqLenForPostQuant(uint32_t batchIdx, const gert::Tensor *tempData,
        int64_t actualLenDims) const;
    int64_t GetKVActualSeqLenForPostQuant(uint32_t batchIdx, const gert::Tensor *tempDataKV,
        int64_t actualLenDimsKV) const;
    void CalcPostQuantTokensPerBatch(int64_t &preTokensPerbatch, int64_t &nextTokensPerbatch,
        int64_t actualSeqLenKV, int64_t actualSeqLenQ) const;
    ge::graphStatus CheckFeaturePostQuantLayout() const;
    ge::graphStatus CheckFeaturePostQuantOffsetPerBatch(bool checkPostQuantOffset,
        int64_t preTokensPerbatch, int64_t nextTokensPerbatch, int64_t actualSeqLenKV,
        int64_t actualSeqLenQ, int64_t preTokens, int64_t nextTokens) const;
    ge::graphStatus CheckFeatureLeftPadding() const;
    ge::graphStatus CheckFeaturePSE() const;
    ge::graphStatus CheckFeatureHeadDim() const;
    ge::graphStatus CheckFeatureMla();
    ge::graphStatus CheckFeatureGqa();
    ge::graphStatus CheckFeature();

    // 多参数一致性校验
    void SetFiaShapeCompare();
    ge::graphStatus CheckQAndQRope() const;
    ge::graphStatus CheckQAndQRopeShape() const;
    ge::graphStatus CheckQAndQRopeDType() const;
    ge::graphStatus CheckQShape() const;
    ge::graphStatus CheckQRopeShape() const;
    ge::graphStatus CheckKVDType() const;
    ge::graphStatus CheckKVShapeForBatchContinuous() const;
    ge::graphStatus CheckKVShapeForTensorList() const;
    uint32_t GetTypeSize(ge::DataType dtype) const;
    ge::graphStatus CheckKVShapeForPageAttention() const;
    ge::graphStatus CheckKVShape() const;
    ge::graphStatus CheckKV() const;

    ge::graphStatus CheckAttenOut() const;
    ge::graphStatus CheckPseShiftDType();
    ge::graphStatus CheckPseShiftShape();
    ge::graphStatus CheckPseShift();
    ge::graphStatus SetAttenMaskCompare();
    ge::graphStatus CheckAttentionMask();
    ge::graphStatus CheckTokens();
    ge::graphStatus CheckMask();
    ge::graphStatus CheckSoftmaxLseShape();
    ge::graphStatus CheckSoftmaxLseDType();
    ge::graphStatus CheckSoftmaxLse();
    ge::graphStatus CheckPostQuant();
    ge::graphStatus CheckMultiParaConsistency();
    ge::graphStatus CheckSystemPrefix();
    ge::graphStatus CheckSystemPrefixDtype();
    ge::graphStatus CheckSystemPrefixShape();

private:
    const char *opName_ = nullptr;
    fe::PlatFormInfos *platformInfo_ = nullptr;
    FIAParaInfo opParamInfo_;

    std::vector<gert::StorageShape *> kCache_ = {};
    std::vector<gert::StorageShape *> vCache_ = {};
    std::vector<uint32_t> qSize = {};
    std::vector<uint32_t> kvSize = {};

    // BaseParams
    uint32_t bSize_ = 0;
    uint32_t n1Size_ = 0;
    uint32_t n2Size_ = 0;
    uint32_t gSize_ = 0;
    uint32_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    uint32_t qkHeadDim_ = 0;
    uint32_t vHeadDim_ = 0;
    uint32_t ropeHeadDim_ = 0;
    uint32_t qTSize_ = 0; // 仅TND/NTD时生效
    uint32_t kTSize_ = 0;
    KvStorageMode kvStorageMode_ = KvStorageMode::BATCH_CONTINUOUS;
    RopeMode ropeMode_ = RopeMode::NO_ROPE;

    FiaLayout qLayout_ = FiaLayout::BSND;
    FiaLayout outLayout_ = FiaLayout::BSND;
    FiaLayout kvLayout_ = FiaLayout::BSND;
    FiaLayout pseShiftLayout_ = FiaLayout::BNS1S2;
    FiaLayout softmaxLseLayout_ = FiaLayout::BNS11;
    FiaLayout quantScale2Layout_ = FiaLayout::BNSD;

    // PageAttention
    uint32_t maxBlockNumPerBatch_ = 0;
    int32_t blockSize_ = 0;

    // 局部参数, 暂存
    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    int64_t preTokens_ = 0;
    int64_t nextTokens_ = 0;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    uint64_t l2CacheSize_ = 0;
    uint32_t actualSeqLengthsQSize_ = 0;
    uint32_t actualSeqLengthsKvSize_ = 0;

    ge::DataType inputQType_ = ge::DT_FLOAT16;
    ge::DataType inputKvType_ = ge::DT_FLOAT16;
    ge::DataType outputType_ = ge::DT_FLOAT16;
    ge::DataType inputQRopeType_ = ge::DT_FLOAT16;
    ge::DataType inputKRopeType_ = ge::DT_FLOAT16;

    bool attenMaskFlag_ = false;
    FiaQuantMode quantMode_ = FiaQuantMode::NO_QUANT;

    std::shared_ptr<FiaTilingShapeCompare> queryShapeCmp_ = nullptr;
    std::shared_ptr<FiaTilingShapeCompare> keyShapeCmp_ = nullptr;
    std::shared_ptr<FiaTilingShapeCompare> valueShapeCmp_ = nullptr;
    std::shared_ptr<FiaTilingShapeCompare> queryRopeShapeCmp_ = nullptr;
    std::shared_ptr<FiaTilingShapeCompare> keyRopeShapeCmp_ = nullptr;
    std::shared_ptr<FiaTilingShapeCompare> attenOutShapeCmp_ = nullptr;
    std::shared_ptr<FiaTilingShapeCompare> attenMaskShapeCmp_ = nullptr;
    std::shared_ptr<FiaTilingShapeCompare> pseShiftShapeCmp_ = nullptr;
    std::shared_ptr<FiaTilingShapeCompare> softmaxLseShapeCmp_ = nullptr;
    std::shared_ptr<FiaTilingShapeCompare> quantScale2ShapeCmp_ = nullptr;
    std::shared_ptr<FiaTilingShapeCompare> quantOffset2ShapeCmp_ = nullptr;
    std::shared_ptr<FiaTilingShapeCompare> prefixKeyShapeCmp_ = nullptr;
};
} // namespace optiling
#endif // FUSED_INFER_ATTENTION_SCORE_TILING_CHECK_H
