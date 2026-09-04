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
 * \file fused_infer_attention_score_tiling_info_parser.h
 * \brief
 */

#pragma once

#include "../../common/op_host/fia_tiling_info.h"
#include "../../common/op_host/fia_tiling_shape.h"

namespace optiling {
constexpr int64_t SPARSE_MODE_INT_MAX = 2147483647;
class FiaInfoParser {
public:
    explicit FiaInfoParser(const gert::TilingContext *context) : context_(context)
    {
    }
    ~FiaInfoParser() = default;

    ge::graphStatus CheckRequiredInOutExistence() const;
    ge::graphStatus CheckRequiredAttrExistence() const;
    ge::graphStatus CheckRequiredParaExistence() const;
    ge::graphStatus GetActualSeqLenQSize(uint32_t &size);
    ge::graphStatus GetOpName();
    ge::graphStatus GetNpuInfo();
    ge::graphStatus GetStrides();
    void GetOptionalInputParaMaskInfo();
    void GetOptionalInputParaActualSeqLengthInfo();
    void GetOptionalInputParaPageAttentionInfo();
    void GetOptionalInputParaPostQuantInfo();
    void GetOptionalInputParaRopeInfo();
    void GetOptionalInputParaPseInfo();
    void GetOptionalInputParaLeftPaddingInfo();
    void GetOptionalInputParaPrefixInfo();
    void GetOptionalInputParaLearnableSinkInfo();
    void GetOptionalInputParaQuantInfo();
    void GetOptionalInputParaInfo();
    void GetInputParaInfo();
    void GetOutputParaInfo();
    ge::graphStatus GetAttrParaInfo();
    ge::graphStatus GetKvCache();
    ge::graphStatus GetTensorListCache(uint32_t index, const std::string &name,
                                       std::vector<gert::StorageShape *> &cache);
    ge::graphStatus GetOpParaInfo();

    ge::graphStatus GetEmptyTensorFlag();

    ge::graphStatus GetMaxWorkspaceFlag();
    ge::graphStatus GetLegacyIfaFlag();

    void GetInOutDataType();
    ge::graphStatus GetBatchSize();
    void GetQueryTSize();
    void GetKeyTSize();
    ge::graphStatus GetQkHeadDim();
    ge::graphStatus GetS1Size();
    void GetKvStorageMode();
    void GetQuantMode();
    ge::graphStatus GetKvLayout();
    void SetFiaShape();
    ge::graphStatus GetMaxActualSeq(const gert::Tensor *actualSeqLensTensor, FiaLayout layout,
                                    int64_t &maxActualSeqLen);
    ge::graphStatus GetS2SizeFromActualSeqLens();
    ge::graphStatus GetS2SizeForBatchContinuous();
    ge::graphStatus GetS2SizeForTensorList();
    ge::graphStatus GetMaxBlockNumPerBatch();
    ge::graphStatus GetBlockSize();
    ge::graphStatus GetBlockTableDim2();
    ge::graphStatus GetS2SizeForPageAttention();
    ge::graphStatus GetS2Size();
    ge::graphStatus GetValueHeadDim();
    ge::graphStatus GetRopeMode();
    ge::graphStatus GetMlaMode();
    ge::graphStatus GetRopeHeadDim();
    ge::graphStatus GetQueryAndOutLayout();
    ge::graphStatus GetN1Size();
    ge::graphStatus GetN2Size();
    ge::graphStatus GetGSize();
    void GetUpdateInfo();
    void GetMaskFlag();
    ge::graphStatus GetAttenMaskSparse9Info();
    ge::graphStatus GetAttenMaskInfo();
    void GetPaddingSizeFlag();
    ge::graphStatus GetActualSeqInfo();
    void ProcessKVActualSeqLoop(const int64_t *actualLenData, uint32_t loop);
    void ProcessQActualSeqLoop(const int64_t *actualLenQData, uint32_t loop);
    void GetPreNextToken();
    void UpdatePreNextTokenBySparseMode();
    void ClampPreNextToken();
    ge::graphStatus GetSystemPrefix();
    ge::graphStatus GetPseShiftFlag();
    ge::graphStatus GetPostQuantInfo();
    ge::graphStatus GetFullQuantMode();
    ge::graphStatus GetOldIfaGqaFlag();
    ge::graphStatus GetAntiQuantInfo();
    TilingKeyLayout MapStringToLayout(FiaLayout &layoutString) const;
    void GenerateAxisInfo(FiaTilingInfo &fiaInfo);
    void GenerateDtypeInfo(FiaTilingInfo &fiaInfo);
    void GenerateFeatureInfo(FiaTilingInfo &fiaInfo);
    void GenerateLayoutInfo(FiaTilingInfo &fiaInfo);
    void GenerateInfo(FiaTilingInfo &fiaInfo);
    ge::graphStatus ParseAxisInfo();
    ge::graphStatus ParseFeatureInfo();
    ge::graphStatus Parse(FiaTilingInfo &fiaInfo);

public:
    const gert::TilingContext *context_ = nullptr;

    const char *opName_ = nullptr;
    fe::PlatFormInfos *platformInfo_ = nullptr;
    FIAParaInfo opParamInfo_;
    enum class IfaPseType : int64_t {
        PSE_OUTER_MUL_ADD_TYPE = 0,
        PSE_INNER_MUL_ADD_TYPE = 2,
        PSE_INNER_MUL_ADD_SQRT_TYPE = 3,
    };

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
    uint32_t queryTSize_ = 0; // 仅TND/NTD时生效
    uint32_t keyTSize_ = 0;
    // 增加strides参数，预计5个，k/v/krope/kscale/vscale
    const gert::Stride *keyStrides_ = nullptr;
    const gert::Stride *valueStrides_ = nullptr;
    const gert::Stride *kRopeStrides_ = nullptr;
    const gert::Stride *kScaleStrides_ = nullptr;
    const gert::Stride *vScaleStrides_ = nullptr;
    bool isTensorV1_ = false;
    KvStorageMode kvStorageMode_ = KvStorageMode::BATCH_CONTINUOUS;
    RopeMode ropeMode_ = RopeMode::NO_ROPE;
    MlaMode mlaMode_ = MlaMode::NO_MLA;
    FiaQuantMode quantMode_ = FiaQuantMode::NO_QUANT;
    FiaFullQuantMode fullQuantMode_ = FiaFullQuantMode::NO_FULL_QUANT;

    // Layout
    FiaLayout qLayout_ = FiaLayout::BSND;
    FiaLayout outLayout_ = FiaLayout::BSND;
    FiaLayout kvLayout_ = FiaLayout::BSND;

    // PageAttention
    uint32_t maxBlockNumPerBatch_ = 0;
    int32_t blockSize_ = 0;
    uint32_t blockTableDim2_ = 0;

    // 局部参数, 暂存
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    NpuArch npuArch_ = NpuArch::DAV_3510;

    ge::DataType inputQType_ = ge::DT_FLOAT16;
    ge::DataType inputKvType_ = ge::DT_FLOAT16;
    ge::DataType outputType_ = ge::DT_FLOAT16;
    ge::DataType inputQRopeType_ = ge::DT_FLOAT16;
    ge::DataType inputKRopeType_ = ge::DT_FLOAT16;

    uint64_t l2CacheSize_ = 0;
    std::vector<gert::StorageShape *> kCache_ = {};
    std::vector<gert::StorageShape *> vCache_ = {};
    std::vector<int32_t> qSize_ = {};
    std::vector<int32_t> kvSize_ = {};

    bool emptyTensorFlag_ = false;
    bool isSameSeqAllKVTensor_ = true;
    bool isSameActualseq_ = true;
    bool attenMaskFlag_ = false;
    bool needInit_ = false;
    bool antiQuantFlag_ = false;
    uint32_t antiquantParaSeqSize_ = 0;
    int64_t preToken_ = 0;
    int64_t nextToken_ = 0;
    int32_t sparseMode_ = 0;
    uint32_t attenMaskBatchStride_ = 0;
    uint32_t attenMaskStride_ = 0;
    bool kvPaddingSizeFlag_ = false;
    bool qPaddingSizeFlag_ = false;
    bool pseShiftFlag_ = false;
    uint32_t pseShiftByBatch_ = 0;
    uint32_t pseShiftS1_ = 0;
    uint32_t pseShiftS2_ = 0;
    int64_t pseType_ = 0;
    bool enableAlibiPse_ = false;
    int64_t maxActualseq_ = 0;
    bool isMaxWorkspace_ = false;
    bool isLegacyIfa_ = false;
    bool systemPrefixFlag_ = false;
    int64_t systemPrefixLen_ = 0;
    int64_t systemPrefixMaxLen_ = 0;

    bool isOutQuantPerChnOut_ = false;
    bool isOutQuantTypeBf16_ = false;
    bool isPostQuantEnable_ = false;

    bool isAccumQSeq_ = false;
    bool isAccumKVSeq_ = false;
    uint32_t actualLenQDims_ = 0;
    uint32_t actualLenDims_ = 0;
    std::vector<int64_t> kvListSeqLens_{};

    std::shared_ptr<FiaTilingShape> queryShape_ = nullptr;
    std::shared_ptr<FiaTilingShape> keyShape_ = nullptr;
    std::shared_ptr<FiaTilingShape> valueShape_ = nullptr;
    std::shared_ptr<FiaTilingShape> queryRopeShape_ = nullptr;
    std::shared_ptr<FiaTilingShape> keyRopeShape_ = nullptr;
    std::shared_ptr<FiaTilingShape> keyPrefixShape_ = nullptr;
};
} // namespace optiling
