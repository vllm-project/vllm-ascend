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
 * \file flash_mla_with_kvcache_tiling_info_parser.h
 * \brief
 */

#pragma once

#include "flash_mla_with_kvcache_tiling_info.h"

namespace optiling {
namespace flash_mla_with_kvcache {
class FlashMlaWithKvcacheInfoParser {
public:
    explicit FlashMlaWithKvcacheInfoParser(const gert::TilingContext *context)
        : context_(context)
    {
    }
    ~FlashMlaWithKvcacheInfoParser() = default;

    ge::graphStatus CheckRequiredInOutExistence() const;
    ge::graphStatus CheckRequiredAttrExistence() const;
    ge::graphStatus CheckRequiredParaExistence() const;
    ge::graphStatus GetEmptyTensorFlag();
    ge::graphStatus GetCuSeqLenQSize(int64_t &size);
    ge::graphStatus GetOpName();
    ge::graphStatus GetNpuInfo();
    void GetOptionalInputParaMaskInfo();
    void GetOptionalInputParaSeqLengthInfo();
    void GetOptionalInputParaPageAttentionInfo();
    void GetOptionalInputParaMetadataInfo();

    void GetOptionalInputParaInfo();
    void GetInputParaInfo();
    void GetOutputParaInfo();
    ge::graphStatus GetAttrParaInfo();
    ge::graphStatus GetOpParaInfo();

    void GetInOutDataType();
    ge::graphStatus GetBatchSize();
    void GetQueryTSize();
    void GetKeyTSize();
    ge::graphStatus GetQkHeadDim();
    ge::graphStatus GetS1Size();
    void GetKvStorageMode();
    void GetKvIsContiguous();
    ge::graphStatus GetStrides();

    void SetFaShape();
    ge::graphStatus GetMaxActualSeq(const gert::Tensor *actualSeqLensTensor, FlashMlaWithKvcacheLayout layout, int64_t &maxActualSeqLen);
    ge::graphStatus GetS2SizeFromCuSeqLens();
    ge::graphStatus GetS2SizeForBatchContinuous();
    ge::graphStatus GetMaxBlockNumPerBatch();
    ge::graphStatus GetBlockSize();
    ge::graphStatus GetBlockNum();

    ge::graphStatus GetS2SizeForPageAttention();
    ge::graphStatus GetS2Size();
    ge::graphStatus GetValueHeadDim();
    ge::graphStatus GetInAndOutLayout();
    ge::graphStatus GetN1Size();
    ge::graphStatus GetN2Size();
    ge::graphStatus GetGSize();
    void GetMaskParams();

    ge::graphStatus GetActualSeqInfo();

    void GenerateAxisInfo(FlashMlaWithKvcacheTilingInfo &faInfo);
    void GenerateDtypeInfo(FlashMlaWithKvcacheTilingInfo &faInfo);
    void GenerateFeatureInfo(FlashMlaWithKvcacheTilingInfo &faInfo);
    void GenerateLayoutInfo(FlashMlaWithKvcacheTilingInfo &faInfo);
    void GenerateInfo(FlashMlaWithKvcacheTilingInfo &faInfo);
    ge::graphStatus ParseAxisInfo();
    ge::graphStatus ParseFeatureInfo();
    ge::graphStatus Parse(FlashMlaWithKvcacheTilingInfo &faInfo);

public:
    const gert::TilingContext *context_ = nullptr;

    const char *opName_ = nullptr;
    fe::PlatFormInfos *platformInfo_ = nullptr;
    FlashMlaWithKvcacheParaInfo opParamInfo_;

    // BaseParams
    int64_t bSize_ = 0;
    int64_t n1Size_ = 0;
    int64_t n2Size_ = 0;
    int64_t gSize_ = 0;
    int64_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    int64_t qkHeadDim_ = 0;
    int64_t vHeadDim_ = 0;
    int64_t ropeHeadDim_ = 0;
    int64_t queryTSize_ = 0;
    int64_t keyTSize_ = 0;
    // INT32 seq-lens buffer 元素数（cu_seqlens_q / cache_seqlens），kernel parser 的 actualLenDims
    int64_t actualLenQDims_ = 0;
    int64_t actualLenKvDims_ = 0;
    KvStorageMode kvStorageMode_ = KvStorageMode::BATCH_CONTINUOUS;

    // kvcache strides (tensor v2 view inputs)
    const gert::Stride *keyStrides_ = nullptr;
    const gert::Stride *valueStrides_ = nullptr;
    uint64_t keyBnStride_ = 0;
    uint64_t keyN2Stride_ = 0;
    uint64_t valueBnStride_ = 0;
    uint64_t valueN2Stride_ = 0;

    bool hasViewStride_ = true;
    int32_t keyNonContigDim_ = -1;
    int32_t valueNonContigDim_ = -1;

    // Layout
    FlashMlaWithKvcacheLayout layoutQ_ = FlashMlaWithKvcacheLayout::BSND;
    FlashMlaWithKvcacheLayout layoutKV_ = FlashMlaWithKvcacheLayout::BSND;
    FlashMlaWithKvcacheLayout layoutOut_ = FlashMlaWithKvcacheLayout::BSND;

    // PageAttention
    int64_t maxBlockNumPerBatch_ = 0;
    int32_t blockSize_ = 0;
    int32_t blockNum_ = 0;

    // NPU信息
    NpuArch npuArch_ = NpuArch::DAV_3510;

    // Dtype信息
    ge::DataType inputQType_ = ge::DT_FLOAT16;
    ge::DataType inputKvType_ = ge::DT_FLOAT16;
    ge::DataType outputType_ = ge::DT_FLOAT16;

    // mask 信息
    int64_t winLeft_ = 0;
    int64_t winRight_ = 0;
    int64_t maskMode_ = 0;

    // SoftmaxLSE 信息
    bool softmaxLseFlag_ = false;
    int64_t returnSoftmaxLse_ = 0;

    // 其他属性信息
    float softmaxScale_ = 1.0;

    // seqLen信息
    int64_t seqLenQDims_ = 0;
    int64_t seqLenKvDims_ = 0;
    int64_t cuseqLenQDims_ = 0;
    int64_t cuseqLenKvDims_ = 0;
    int64_t maxSeqQ_ = -1;
    int64_t maxSeqKv_ = -1;

    // Empty Tensor
    bool emptyTensorFlag_ = false;

    // shape信息
    std::shared_ptr<FlashMlaWithKvcacheTilingShape> queryShape_ = nullptr;
    std::shared_ptr<FlashMlaWithKvcacheTilingShape> keyShape_ = nullptr;
    std::shared_ptr<FlashMlaWithKvcacheTilingShape> valueShape_ = nullptr;
};
} // namespace flash_mla_with_kvcache
} // namespace optiling
