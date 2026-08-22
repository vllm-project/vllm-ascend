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
 * \file fia_tiling_info.h
 * \brief
 */

#ifndef FIA_TILING_INFO_H
#define FIA_TILING_INFO_H

#include <vector>
#include "fia_tiling_base.h"
#include "fia_tiling_shape.h"

namespace optiling {
const std::string ACTUAL_SEQ_KV_LEN_NAME = "the key/value's actual sequence lengths";
const std::string ACTUAL_SEQ_Q_LEN_NAME = "the query's actual sequence lengths";
const std::string ATTEN_MASK_NAME = "atten_mask";
const std::string ATTEN_OUT_NAME = "attention_out";
const std::string BLOCK_SIZE_NAME = "block_size";
const std::string BLOCK_TABLE_NAME = "block_table";
const std::string DEQUANT_SCALE_QUERY_NAME = "the query's dequant scale";
const std::string INNER_PRECISE_NAME = "inner_precise";
const std::string KEY_NAME = "key";
const std::string KEY_ANTIQUANT_MODE_NAME = "the key's quant mode";
const std::string KEY_ANTIQUANT_OFFSET_NAME = "the key's quant offset";
const std::string KEY_ANTIQUANT_SCALE_NAME = "the key's quant scale";
const std::string KEY_ROPE_NAME = "key_rope";
const std::string KEY_ROPE_ANTIQUANT_SCALE_NAME = "the key_rope's dequant scale";
const std::string KV_HEADS_NUM_NAME = "the key/value's heads num";
const std::string NEXT_TOKENS_NAME = "next_tokens";
const std::string PRE_TOKENS_NAME = "pre_tokens";
const std::string PSE_SHIFT_NAME = "pse_shift";
const std::string QUANT_OFFSET2_NAME = "the output's dequant offset";
const std::string QUANT_SCALE2_NAME = "the output's dequant scale";
const std::string QUERY_NAME = "query";
const std::string QUERY_HEADS_NUM_NAME = "the query's heads num";
const std::string QUERY_QUANT_MODE_NAME = "the query's quant mode";
const std::string QUERY_ROPE_NAME = "query_rope";
const std::string SOFTMAX_SCALE_NAME = "the softmax's scale";
const std::string SPARSE_MODE_NAME = "sparse_mode";
const std::string VALUE_NAME = "value";
const std::string VALUE_ANTIQUANT_MODE_NAME = "the value's quant mode";
const std::string VALUE_ANTIQUANT_OFFSET_NAME = "the value's dequant offset";
const std::string VALUE_ANTIQUANT_SCALE_NAME = "the value's dequant scale";

const std::string ANTIQUANT_MODE_NAME = "antiquant_mode";
const std::string ANTIQUANT_SCALE_NAME = "antiquant_scale";
const std::string ANTIQUANT_OFFSET_NAME = "antiquant_offset";
const std::string DEQUANT_SCALE1_NAME = "dequant_scale1";
const std::string DEQUANT_SCALE2_NAME = "dequant_scale2";
const std::string KEY_SHARED_PREFIX_NAME = "key_shared_prefix";
const std::string KV_PADDING_SIZE_NAME = "kv_padding_size";
const std::string QUANT_SCALE1_NAME = "quant_scale1";
const std::string QUERY_PADDING_SIZE_NAME = "query_padding_size";
const std::string SOFTMAX_LSE_NAME = "softmax_lse";
const std::string VALUE_SHARED_PREFIX_NAME = "value_shared_prefix";
const std::string ACTUAL_SHARED_PREFIX_LEN_NAME = "actual_shared_prefix_len";
const std::string LEARNABLE_SINK_NAME = "learnable_sink";

constexpr int32_t SPARSE_MODE_NO_MASK = 0;
constexpr int32_t SPARSE_MODE_ALL_MASK = 1;
constexpr int32_t SPARSE_MODE_LEFT_UP = 2;
constexpr int32_t SPARSE_MODE_RIGHT_DOWN = 3;
constexpr int32_t SPARSE_MODE_BAND = 4;
constexpr int32_t SPARSE_MODE_TREE = 9;


enum class FiaQuantMode : uint32_t {
    NO_QUANT = 0,
    ANTI_QUANT,
    FULL_QUANT
};

enum class KvStorageMode : uint32_t {
    BATCH_CONTINUOUS = 0,
    TENSOR_LIST = 1,
    PAGE_ATTENTION = 2
};

enum class RopeMode : uint32_t {
    NO_ROPE = 0,
    ROPE_SPLIT = 1,
    ROPE_COMBINE = 2
};

enum class MlaMode : uint32_t {
    NO_MLA = 0,
    ROPE_SPLIT_D512 = 1,
    ROPE_SPLIT_D128 = 2,
    ROPE_COMBINE_D128 = 3,
};

enum class FiaTilingInOutMode : uint32_t {
    IO_INVALID = 0,
    INT8_INT8 = 1,
    FP16_INT8 = 2,
    INT8_FP16 = 3,
    FP16_FP16 = 4,
    BF16_BF16 = 5,
    FP32_FP32 = 6,
    FP16_FP16_SPLITKV = 7,
    BF16_INT8 = 8,
    INT8_BF16 = 9,
};

enum class TilingKeyLayout : uint32_t {
    BSH_BSND = 0,
    BNSD = 1,
    NZ = 2,
    TND = 3,
    NBSD = 4,
    NTD = 5
};

enum class FiaTemplateId : uint32_t {
    EMPTY_TENSOR = 0,
    HIGH_PERFORMANCE_GQA = 3,
    GENERAL_GQA = 4,
    HIGH_PERFORMANCE_MLA = 5
};

enum class FiaFullQuantMode : uint32_t {
    NO_FULL_QUANT = 0,
    QKV_PER_TENSOR_FULL_QUANT = 1,
    QKV_PER_BLOCK_FULL_QUANT = 2,
    QKV_MXFP8_FULL_QUANT = 3,
    Q_PER_TOKEN_HEAD_KV_PER_TENSOR_FULL_QUANT = 4,
    QK_PER_TOKEN_HEAD_V_PER_HEAD = 5,
};

std::string QuantModeToSerialString(FiaQuantMode fiaQuantMode);
std::string SituationToSerialString(RopeMode ropeMode);

struct FIARequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct FIAOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

struct FIAParaInfo {
    FIARequiredParaInfo query = {nullptr, nullptr};
    FIARequiredParaInfo key = {nullptr, nullptr};
    FIARequiredParaInfo value = {nullptr, nullptr};

    FIAOptionalParaInfo pseShift = {nullptr, nullptr};
    FIAOptionalParaInfo attenMask = {nullptr, nullptr};
    FIAOptionalParaInfo actualSeqLengthsQ = {nullptr, nullptr};
    FIAOptionalParaInfo actualSeqLengths = {nullptr, nullptr};
    FIAOptionalParaInfo deqScale1 = {nullptr, nullptr};
    FIAOptionalParaInfo quantScale1 = {nullptr, nullptr};
    FIAOptionalParaInfo deqScale2 = {nullptr, nullptr};
    FIAOptionalParaInfo quantScale2 = {nullptr, nullptr};
    FIAOptionalParaInfo quantOffset2 = {nullptr, nullptr};
    FIAOptionalParaInfo antiquantScale = {nullptr, nullptr};
    FIAOptionalParaInfo antiquantOffset = {nullptr, nullptr};
    FIAOptionalParaInfo blockTable = {nullptr, nullptr};
    FIAOptionalParaInfo queryPaddingSize = {nullptr, nullptr};
    FIAOptionalParaInfo kvPaddingSize = {nullptr, nullptr};
    FIAOptionalParaInfo keyAntiquantScale = {nullptr, nullptr};
    FIAOptionalParaInfo keyAntiquantOffset = {nullptr, nullptr};
    FIAOptionalParaInfo valueAntiquantScale = {nullptr, nullptr};
    FIAOptionalParaInfo valueAntiquantOffset = {nullptr, nullptr};
    FIAOptionalParaInfo keySharedPrefix = {nullptr, nullptr};
    FIAOptionalParaInfo valueSharedPrefix = {nullptr, nullptr};
    FIAOptionalParaInfo actualSharedPrefixLen = {nullptr, nullptr};
    FIAOptionalParaInfo queryRope = {nullptr, nullptr};
    FIAOptionalParaInfo keyRope = {nullptr, nullptr};
    FIAOptionalParaInfo keyRopeAntiquantScale = {nullptr, nullptr};
    FIAOptionalParaInfo dequantScaleQuery = {nullptr, nullptr};
    FIAOptionalParaInfo learnableSink = {nullptr, nullptr};
    FIAOptionalParaInfo qStartIdx = {nullptr, nullptr};
    FIAOptionalParaInfo kvStartIdx = {nullptr, nullptr};

    FIARequiredParaInfo attenOut = {nullptr, nullptr};
    FIARequiredParaInfo lseOut = {nullptr, nullptr};

    const int32_t *numHeads = nullptr;
    const int64_t *preToken = nullptr;
    const int64_t *nextToken = nullptr;
    const float *scaleValue = nullptr;
    const int32_t *kvHeadNums = nullptr;
    const char *layOut = nullptr;
    const int32_t *blockSize = nullptr;
    const int32_t *innerPrecise = nullptr;
    const int64_t *antiquantMode = nullptr;
    const bool *softmaxLseFlag = nullptr;
    const int64_t *keyAntiquantMode = nullptr;
    const int64_t *valueAntiquantMode = nullptr;
    const int32_t *sparseMode = nullptr;
    const int64_t *queryQuantMode = nullptr;
    const int64_t *pseType = nullptr;
};

class FiaTilingInfo : public TilingInfo {
public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    FIAParaInfo opParamInfo;

    // Base Param
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
    NpuArch npuArch = NpuArch::DAV_2201;
    uint32_t bSize = 0;
    uint32_t n1Size = 0;
    uint32_t n2Size = 0;
    uint32_t s1Size = 0;
    int64_t s2Size = 0;
    uint32_t qkHeadDim = 0;
    uint32_t vHeadDim = 0;
    uint32_t gSize = 0;
    uint32_t ropeHeadDim = 0;
    uint32_t qTSize = 0; // 仅TND/NTD时生效
    uint32_t kTSize = 0;
    float scaleValue = 0;
    int32_t innerPrecise = 0;
    uint32_t l2CacheOffFlag = 0;
    uint64_t l2CacheSize = 0;
    std::vector<gert::StorageShape *> kCache = {};
    std::vector<gert::StorageShape *> vCache = {};
    std::vector<int32_t> qSize = {};
    std::vector<int32_t> kvSize = {};
    // 增加strides参数和tensorv1的标志位
    const gert::Stride *keyStrides = nullptr;
    const gert::Stride *valueStrides = nullptr;
    const gert::Stride *kRopeStrides = nullptr;
    const gert::Stride *kScaleStrides = nullptr;
    const gert::Stride *vScaleStrides = nullptr;
    bool isTensorV1 = false;
    // empty Tensor
    bool emptyTensorFlag = false;
    uint64_t totalOutputSize = 0;
    uint64_t totalLseSize = 0;

    // PageAttention
    bool pageAttentionFlag = false;
    int32_t blockSize = 0;
    uint32_t blockTypeSize = 0; // 计算中间量大小
    uint32_t maxBlockNumPerBatch = 0;
    uint32_t totalBlockNum = 0;
    uint32_t kvLayoutType = 0;

    // antiquant
    bool antiQuantFlag = false;
    uint32_t msdIterNum = 1;
    uint32_t antiqSeqSize = 0;
    uint32_t antiquantMode = 0;
    uint32_t keyAntiquantMode = 0;
    uint32_t valueAntiquantMode = 0;
    bool antiquantSplit = false;

    // SysTem Prefix
    bool sysPrefixFlag = false;
    uint32_t systemPrefixMaxLen = 0;
    uint32_t systemPrefixLen = 0;

    // Q actual_seq_lens
    uint32_t actualLenQDims = 0;
    int64_t maxActualseq = 0;
    bool isAccumQSeq = false;

    // KV actual_seq_lens
    bool actualSeqLenFlag = false;
    bool isSameSeqAllKVTensor = true;
    bool isSameActualseq = true;
    uint32_t actualLenKvDims = 0;
    std::vector<int64_t> kvListSeqLens{};
    bool isAccumKVSeq = false;

    // PSE
    bool pseShiftFlag = false;
    uint32_t pseShiftByBatch = 0U;
    uint32_t pseShiftS1 = 0U;
    uint32_t pseShiftS2 = 0U;
    bool enableAlibiPse = false;

    // Mask
    bool attenMaskFlag = false;
    bool isExistRowInvalid = false;
    uint32_t attenMaskBatchStride = 0;
    uint32_t attenMaskStride = 0;
    int32_t sparseMode = 0;
    int64_t preToken = 0;
    int64_t nextToken = 0;

    // PostQuant
    bool isOutQuantPerChnOut = false;
    bool isOutQuantTypeBf16 = false;
    bool isOutQuantEnable = false;

    // Others Flag
    bool batchContinuousFlag = true;
    bool kvPaddingSizeFlag = false;
    bool qPaddingSizeFlag = false;
    bool softmaxLseFlag = false;
    bool quantFlag = false;
    bool isMaxWorkspace = false;
    bool isLegacyIfa = false;
    bool needInit = false;
    bool slidingFlag = false;
    bool learnableSinkFlag = false;
    bool isQKVDDifferent = false;
    // DType
    FiaTilingInOutMode inOutMode = FiaTilingInOutMode::FP16_FP16;
    ge::DataType inputQType = ge::DT_FLOAT16;
    ge::DataType inputKvType = ge::DT_FLOAT16;
    ge::DataType outputType = ge::DT_FLOAT16;
    ge::DataType inputQRopeType = ge::DT_FLOAT16;
    ge::DataType inputKRopeType = ge::DT_FLOAT16;

    // Layout
    TilingKeyLayout inputKvLayout = TilingKeyLayout::BSH_BSND;
    TilingKeyLayout inputLayout = TilingKeyLayout::BSH_BSND;
    TilingKeyLayout outputLayout = TilingKeyLayout::BSH_BSND;

    // BaseParams
    KvStorageMode kvStorageMode = KvStorageMode::BATCH_CONTINUOUS;
    RopeMode ropeMode = RopeMode::NO_ROPE;
    MlaMode mlaMode = MlaMode::NO_MLA;
    FiaQuantMode quantMode = FiaQuantMode::NO_QUANT;
    FiaFullQuantMode fullQuantMode = FiaFullQuantMode::NO_FULL_QUANT;

    // Layout
    FiaLayout qLayout = FiaLayout::BSND;
    FiaLayout outLayout = FiaLayout::BSND;
    FiaLayout kvLayout = FiaLayout::BSND;
};
} // namespace optiling
#endif // FIA_TILING_INFO_H