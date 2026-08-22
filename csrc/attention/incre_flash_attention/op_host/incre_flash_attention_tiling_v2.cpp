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
 * \file incre_flash_attention_tiling_v2.cpp
 * \brief
 */

#include "incre_flash_attention_tiling_v2.h"
#include "incre_flash_attention_tiling_base.h"
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/data_copy_transpose_tiling.h"
#include "log/log.h"
#include "err/ops_err.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/arch35/incre_flash_attention_template_tiling_key.h"
#include "../op_kernel/arch35/incre_flash_attention_tiling_regbase.h"
#include "../../common/op_host/fia_tiling_templates_registry.h"
#include "../../prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.h"

using namespace ge;
using namespace AscendC;
using namespace optiling::v2;
namespace optiling {
void TilingGetTempCompileInfo(platform_ascendc::PlatformAscendC& ascendcPlatform, PromptFlashAttentionCompileInfo& compileInfo);
const int64_t tokenDefault = 2147483647;  // for token default value
const int32_t sparseDefault = 0;
constexpr int32_t POS_SHIFT_MAX = 1048576; // 2^20
constexpr int32_t POS_SHIFT_MIN = -1048576; // -2^20
constexpr uint32_t BATCH_MODE_SCHEDULE = 1;

ge::graphStatus PFAConvertContext(ContextParamsForPFATiling &contextKeyParams, gert::TilingContext *context)
{
    contextKeyParams.opName = context->GetNodeName();
    contextKeyParams.isKvContinuous = 1U;
    contextKeyParams.emptyTensor = 0U;
    contextKeyParams.fromTilingSink = 0U;
    contextKeyParams.fromFused = 71U; // 71 for default value
    contextKeyParams.pseShift = context->GetOptionalInputTensor(PSE_SHIFT_INPUT_INDEX);
    contextKeyParams.attentionMask = context->GetOptionalInputTensor(ATTEN_MASK_INPUT_INDEX);
    contextKeyParams.actualSequenceLengthKV = context->GetOptionalInputTensor(ACT_SEQ_LEN_INPUT_INDEX);
    contextKeyParams.antiquantScale = context->GetOptionalInputTensor(ANTIQUANT_SCALE_INPUT_INDEX);
    contextKeyParams.antiquantOffset = context->GetOptionalInputTensor(ANTIQUANT_OFFSET_INPUT_INDEX);
    contextKeyParams.blockTable = context->GetOptionalInputTensor(BLOCK_TABLE_INPUT_INDEX);
    contextKeyParams.kvPaddingSize = context->GetOptionalInputTensor(KV_PADDING_SIZE_INPUT_INDEX);
    contextKeyParams.actualSequenceLengthQ = nullptr;
    contextKeyParams.keySharedPrefix = (nullptr);
    contextKeyParams.valueSharedPrefix = (nullptr);
    contextKeyParams.actualSharedPrefixLen = (nullptr);
    contextKeyParams.inputDataType = context->GetInputDesc(QUERY_INPUT_INDEX)->GetDataType();
    contextKeyParams.kDataType = context->GetInputDesc(KEY_INPUT_INDEX)->GetDataType();
    contextKeyParams.vDataType = context->GetInputDesc(VALUE_INPUT_INDEX)->GetDataType();
    contextKeyParams.pseShiftDataType = (contextKeyParams.pseShift != nullptr) ?
        context->GetOptionalInputDesc(PSE_SHIFT_INPUT_INDEX)->GetDataType() : contextKeyParams.inputDataType;
    contextKeyParams.maskDataType = (contextKeyParams.attentionMask != nullptr) ?
        context->GetOptionalInputDesc(ATTEN_MASK_INPUT_INDEX)->GetDataType() : contextKeyParams.inputDataType;
    contextKeyParams.blockTableType = (context->GetOptionalInputDesc(BLOCK_TABLE_INPUT_INDEX) != nullptr) ?
        context->GetOptionalInputDesc(BLOCK_TABLE_INPUT_INDEX)->GetDataType() : ge::DT_INT32;
    contextKeyParams.outputDataType = context->GetOutputDesc(0)->GetDataType();    
    contextKeyParams.deqScaleType = (context->GetOptionalInputDesc(DEQUANT_SCALE_1_INPUT_INDEX) != nullptr) ?
        context->GetOptionalInputDesc(DEQUANT_SCALE_1_INPUT_INDEX)->GetDataType() : contextKeyParams.inputDataType;
    contextKeyParams.deqScale2Type = (context->GetOptionalInputDesc(DEQUANT_SCALE_2_INPUT_INDEX) != nullptr) ?
        context->GetOptionalInputDesc(DEQUANT_SCALE_2_INPUT_INDEX)->GetDataType() : contextKeyParams.inputDataType;
    contextKeyParams.quantScale2Type = (context->GetOptionalInputDesc(QUANT_SCALE_2_INPUT_INDEX) != nullptr) ?
        context->GetOptionalInputDesc(QUANT_SCALE_2_INPUT_INDEX)->GetDataType() : ge::DT_FLOAT;
    contextKeyParams.quantOffset2Type = (context->GetOptionalInputDesc(QUANT_OFFSET_2_INPUT_INDEX) != nullptr) ?
        context->GetOptionalInputDesc(QUANT_OFFSET_2_INPUT_INDEX)->GetDataType() : ge::DT_FLOAT;
    contextKeyParams.queryInputShape = context->GetInputShape(QUERY_INPUT_INDEX);
    contextKeyParams.keyInputShape = context->GetInputShape(KEY_INPUT_INDEX);
    contextKeyParams.valueInputShape = context->GetInputShape(VALUE_INPUT_INDEX);
    contextKeyParams.pseShiftShape = context->GetOptionalInputShape(PSE_SHIFT_INPUT_INDEX);
    contextKeyParams.attentionMaskShape = context->GetOptionalInputShape(ATTEN_MASK_INPUT_INDEX);
    contextKeyParams.deqScale1Shape = context->GetOptionalInputShape(DEQUANT_SCALE_1_INPUT_INDEX);
    contextKeyParams.scale1Shape = context->GetOptionalInputShape(QUANT_SCALE_1_INPUT_INDEX);
    contextKeyParams.deqScale2Shape = context->GetOptionalInputShape(DEQUANT_SCALE_2_INPUT_INDEX);
    contextKeyParams.scale2Shape = context->GetOptionalInputShape(QUANT_SCALE_2_INPUT_INDEX);
    contextKeyParams.offset2Shape = context->GetOptionalInputShape(QUANT_OFFSET_2_INPUT_INDEX);
    contextKeyParams.antiquantScaleShape = context->GetOptionalInputShape(ANTIQUANT_SCALE_INPUT_INDEX);
    contextKeyParams.antiquantOffsetShape = context->GetOptionalInputShape(ANTIQUANT_OFFSET_INPUT_INDEX);
    contextKeyParams.blockTableShape = context->GetOptionalInputShape(BLOCK_TABLE_INPUT_INDEX);
    contextKeyParams.outputShape = context->GetOutputShape(0);
    auto attrs = context->GetAttrs();
    contextKeyParams.innerPrecisePtr = attrs->GetAttrPointer<int64_t>(INNER_PRECISE_ATTR_INDEX);
    contextKeyParams.headsNumber = attrs->GetAttrPointer<int64_t>(NUM_HEADS_ATTR_INDEX);
    contextKeyParams.blockSize = attrs->GetAttrPointer<int32_t>(BLOCK_SIZE_ATTR_INDEX);
    contextKeyParams.scaleValue = attrs->GetAttrPointer<float>(SCALE_VALUE_ATTR_INDEX);
    contextKeyParams.layout = attrs->GetAttrPointer<char>(LAYOUT_ATTR_INDEX);
    contextKeyParams.numKeyValueHeads = attrs->GetAttrPointer<int64_t>(KV_NUM_HEADS_ATTR_INDEX);    
    contextKeyParams.sparseMode = &sparseDefault;
    contextKeyParams.preToken = &tokenDefault;
    contextKeyParams.nextToken = &tokenDefault;
    contextKeyParams.workspaceSize = context->GetWorkspaceSizes(1);
    contextKeyParams.compileInfoPtr = reinterpret_cast<const PromptFlashAttentionCompileInfo *>(context->GetCompileInfo());
    contextKeyParams.isBSNDOut = (string(contextKeyParams.layout) == "BNSD_BSND") ? 1U : 0U;

    return ge::GRAPH_SUCCESS;
}

template <typename T>
inline auto AlignUp(T num, T rnd) -> T {
  return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

static int64_t CeilDivision(int64_t num1, int64_t num2) {
  if (num2 == 0) {
    return 0;
  }
  return (num1 + num2 - 1) / num2;
}

// 获取公约数
static uint32_t increGcd(uint32_t a, uint32_t b)
{
  if (a % b == 0) {
    return b;
  }
  return increGcd(b, a % b);
}

template <typename T>
static auto CalcTailSizefaRun(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    T mod = num1 % num2;
    return mod != 0 ? mod : num2;
}

template <typename vecT, typename T>
static bool VecContains(const vecT& vec, const T& value)
{
    return std::find(vec.begin(), vec.end(), value) != vec.end();
}

std::string DataTypeToString(ge::DataType type)
{
    const auto it = DATATYPE_TO_STRING_MAP.find(type);
    if (it != DATATYPE_TO_STRING_MAP.end()) {
        return it->second;
    } else {
        OP_LOGE("IncreFlashAttention", "datatype %d not support", type);
        return "UNDEFINED";
    }
}

enum class LayoutTypefaRun : uint8_t {
    NONE = 0,
    LAYOUT_BSH = 1,
    LAYOUT_BSND = 1,
    LAYOUT_SBH = 2,
    LAYOUT_BNSD = 3,
    LAYOUT_TND = 4,
};

enum PfaAttenMaskCompressModefaRun : uint8_t {
    PFA_NO_COMPRESS_MODE = 0,
    PFA_LEFT_UP_CAUSAL_MODE,
    PFA_RIGHT_DOWN_CAUSAL_MODE,
    PFA_BAND_MODE,
    PFA_PREFIX_MODE,
    PFA_RIGHT_DOWN_CAUSAL_BAND_MODE,
    PFA_BAND_LEFT_UP_CAUSAL_MODE
};

constexpr uint64_t RecursiveSum() {
  return 0;
}

template <typename T, typename... Args>
constexpr uint64_t RecursiveSum(T templateId, Args... templateIds) {
  return static_cast<uint64_t>(templateId) + 10U * RecursiveSum(templateIds...);
}
constexpr uint64_t IFA_TILINGKEYOFFSET = uint64_t(10000000000000000UL);  // 10^16
constexpr uint64_t IFA_PERF_MODE_TILINGKEYOFFSET = uint64_t(1000000000000000UL);  // 10^15
template <typename... Args>
constexpr uint64_t IFA_GET_TILINGKEY(Args... templateIds) {
  return RecursiveSum(templateIds...);
}

enum class IfaSparseEnum : uint8_t {
    IFA_ALL = 0,
    IFA_NONE = 1,
    IFA_ANY = 2,
    IFA_CAUSAL = 3,
    IFA_BAND = 4,
    IFA_PREFIX = 5,
    IFA_BAND_COMPRESS = 6,
    IFA_RIGHT_DOWN_CAUSAL = 7,
    IFA_RIGHT_DOWN_CAUSAL_BAND = 8,
    IFA_BAND_LEFT_UP_CAUSAL = 9
};

ge::graphStatus IFATilingV2::GetNpuInfo() {
  OP_CHECK_IF(ifaContext_->platformInfo == nullptr,
             OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName, "GetPlatformInfo is null."), return ge::GRAPH_FAILED);

  auto ascendcPlatform = platform_ascendc::PlatformAscendC(ifaContext_->platformInfo);
  libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();

  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize_);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bSize_);

  aicNum_ = ascendcPlatform.GetCoreNumAic();
  aivNum_ = ascendcPlatform.GetCoreNumAiv();
  
  socVersion_ = IfaSocVersion::SOC_ASCEND_950;
  coreNum_ = ascendcPlatform.GetCoreNumAiv();  // default aiv num
  if (NUM2 * aicNum_ <= aivNum_) {
    aivNum_ = NUM2 * aicNum_;
  } else {
    aivNum_ = aivNum_ / NUM2 * NUM2;
    aicNum_ = aivNum_ / NUM2;
  }
  OP_LOGD(ifaContext_->opName, "Platform is Ascend 950. Num of core obtained is %d.", coreNum_);
  
  OP_CHECK_IF(aicNum_ == 0 || aivNum_ == 0, OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName, "Num of core obtained is 0."),
             return GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATilingV2::PreProcess() {
  if (ProcessBaseTensors() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  if (emptyTensor_) {
    return ge::GRAPH_SUCCESS;
  }
  return ProcessOptionalTensors();
}

ge::graphStatus IFATilingV2::CheckPABlockSize() const {
  if (pageAttentionFlag_) {
    OP_CHECK_IF(blockSize_ == 0,
               OP_LOGE(ifaContext_->opName,
               "When page attention is enabled, input attribute blocksize can not be 0."),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<int32_t>(blockSize_) < 0,
               OP_LOGE(ifaContext_->opName,
               "When page attention is enabled, input attribute blocksize[%d] can not be less than 0.",
               static_cast<int32_t>(blockSize_)),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(blockSize_ > MAX_BLOCK_SIZE,
               OP_LOGE(ifaContext_->opName,
               "When page attention is enabled, input attribute blocksize[%u] can not be greater than %u.",
               blockSize_, MAX_BLOCK_SIZE),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(((inputKvType_ == ge::DT_INT4) || (inputKvType_ == ge::DT_FLOAT4_E2M1)) && (blockSize_ % NUM64 != 0),
               OP_LOGE(ifaContext_->opName,
               "When page attention is enabled, if kvCache datatype is int4(int32)/fp4_e2m1, input attribute blocksize[%u] should be 64 aligned.", blockSize_),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(((inputKvType_ == ge::DT_INT8) || (inputKvType_ == ge::DT_HIFLOAT8) ||
               (inputKvType_ == ge::DT_FLOAT8_E4M3FN)) && 
               (blockSize_ % NUM32 != 0),
               OP_LOGE(ifaContext_->opName,
               "When page attention is enabled, if kvCache datatype is int8/hif8/fp8_e4m3fn, input attribute blocksize[%u] should be 32 aligned.", blockSize_),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(((inputKvType_ == ge::DT_FLOAT16) || (inputKvType_ == ge::DT_BF16)) && 
               (blockSize_ % NUM16 != 0),
               OP_LOGE(ifaContext_->opName,
               "When page attention is enabled, "
               "if kvCache datatype is float16/bfloat16, input attribute blocksize[%u] should be 16 aligned.", blockSize_),
               return ge::GRAPH_FAILED);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATilingV2::CheckFormat(ge::Format format, const std::string &sName) const
{
    OP_CHECK_IF(format != FORMAT_ND && format != ge::FORMAT_NCHW && format != ge::FORMAT_NHWC && format != ge::FORMAT_NCDHW,
               OP_LOGE(ifaContext_->opName, "%s format should be ND/NCHW/NHWC/NCDHW.", sName.c_str()),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATilingV2::CheckInputAntiquantFormat() const
{
    auto antiquantScaleDesc = ifaContext_->antiquantScale.desc;
    auto antiquantOffsetDesc = ifaContext_->antiquantOffset.desc;
    auto keyAntiquantScaleDesc = ifaContext_->keyAntiquantScale.desc;
    auto keyAntiquantOffsetDesc = ifaContext_->keyAntiquantOffset.desc;
    auto valueAntiquantScaleDesc = ifaContext_->valueAntiquantScale.desc;
    auto valueAntiquantOffsetDesc = ifaContext_->valueAntiquantOffset.desc;
    OP_CHECK_IF(antiquantScaleDesc != nullptr &&
               CheckFormat(antiquantScaleDesc->GetOriginFormat(), "AntiquantScale") != ge::GRAPH_SUCCESS,
               OP_LOGE(ifaContext_->opName, "AntiquantScale check format failed"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(antiquantOffsetDesc != nullptr &&
               CheckFormat(antiquantOffsetDesc->GetOriginFormat(), "AntiquantOffset") != ge::GRAPH_SUCCESS,
               OP_LOGE(ifaContext_->opName, "AntiquantOffset check format failed"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyAntiquantScaleDesc != nullptr &&
               CheckFormat(keyAntiquantScaleDesc->GetOriginFormat(), "KeyAntiquantScale") != ge::GRAPH_SUCCESS,
               OP_LOGE(ifaContext_->opName, "KeyAntiquantScale check format failed"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyAntiquantOffsetDesc != nullptr &&
               CheckFormat(keyAntiquantOffsetDesc->GetOriginFormat(), "KeyAntiquantOffset") != ge::GRAPH_SUCCESS,
               OP_LOGE(ifaContext_->opName, "KeyAntiquantOffset check format failed"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(valueAntiquantScaleDesc != nullptr &&
               CheckFormat(valueAntiquantScaleDesc->GetOriginFormat(), "ValueAntiquantScale") != ge::GRAPH_SUCCESS,
               OP_LOGE(ifaContext_->opName, "ValueAntiquantScale check format failed"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(valueAntiquantOffsetDesc != nullptr &&
               CheckFormat(valueAntiquantOffsetDesc->GetOriginFormat(),"ValueAntiquantOffset") != ge::GRAPH_SUCCESS,
               OP_LOGE(ifaContext_->opName, "ValueAntiquantOffset check format failed"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

int64_t IFATilingV2::GetMaxSeqLength(const gert::Tensor* actualSeqLength) {
  int64_t max = actualSeqLength->GetData<int64_t>()[0];
  for (int i = 1; i < actualSeqLength->GetShapeSize(); ++i) {
      max = std::max(max, actualSeqLength->GetData<int64_t>()[i] - actualSeqLength->GetData<int64_t>()[i - 1]);
  }
  return max;
}

ge::graphStatus IFATilingV2::ProcessBaseTensors() {
  OP_CHECK_IF(ifaContext_->query.shape == nullptr,
             OP_LOGE(ifaContext_->opName, "Shape of tensor query is null."), return ge::GRAPH_FAILED);
  OP_CHECK_IF(ifaContext_->query.desc == nullptr,
             OP_LOGE(ifaContext_->opName, "Desc of tensor query is null."), return ge::GRAPH_FAILED);
  OP_CHECK_IF(ifaContext_->key.shape == nullptr,
             OP_LOGE(ifaContext_->opName, "Shape of tensor key is null."), return ge::GRAPH_FAILED);
  OP_CHECK_IF(ifaContext_->key.desc == nullptr,
             OP_LOGE(ifaContext_->opName, "Desc of tensor key is null."), return ge::GRAPH_FAILED);
  OP_CHECK_IF(ifaContext_->value.shape == nullptr,
             OP_LOGE(ifaContext_->opName, "Shape of tensor value is null."), return ge::GRAPH_FAILED);
  OP_CHECK_IF(ifaContext_->value.desc == nullptr,
             OP_LOGE(ifaContext_->opName, "Desc of tensor value is null."), return ge::GRAPH_FAILED);
  OP_CHECK_IF(ifaContext_->attenOut.desc == nullptr,
             OP_LOGE(ifaContext_->opName, "Desc of tensor attentionOut is null."), return ge::GRAPH_FAILED);
  OP_CHECK_IF(ifaContext_->attenOut.shape == nullptr,
             OP_LOGE(ifaContext_->opName, "Shape of tensor attentionOut is null."), return ge::GRAPH_FAILED);

  if (ifaContext_->softmaxLseFlag != nullptr) {
    softmaxLseFlag_ = *ifaContext_->softmaxLseFlag;
  }

  OP_CHECK_IF((ifaContext_->query.shape->GetStorageShape().GetShapeSize() == 0 &&
              ifaContext_->attenOut.shape->GetStorageShape().GetShapeSize() != 0) ||
              (ifaContext_->query.shape->GetStorageShape().GetShapeSize() != 0 &&
              ifaContext_->attenOut.shape->GetStorageShape().GetShapeSize() == 0),
              OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName, "query shape size is %llu byte, but attention Out shape size is %llu byte, they cannot be empty while the other is not",
                                          ifaContext_->query.shape->GetStorageShape().GetShapeSize(), ifaContext_->attenOut.shape->GetStorageShape().GetShapeSize()),
              return ge::GRAPH_FAILED);
  if (ifaContext_->query.shape->GetStorageShape().GetShapeSize() == 0 &&
      ifaContext_->attenOut.shape->GetStorageShape().GetShapeSize() == 0) {
    emptyTensor_ = true;
    return ge::GRAPH_SUCCESS;
  }

  OP_CHECK_IF(ifaContext_->key.desc->GetDataType() != ifaContext_->value.desc->GetDataType(),
             OP_LOGE(ifaContext_->opName, "Datatype of key and datatype of value are different."), return ge::GRAPH_FAILED);

  OP_CHECK_IF(ifaContext_->numHeads == nullptr,
             OP_LOGE(ifaContext_->opName, "Attribute numHeads is null."), return ge::GRAPH_FAILED);
  OP_CHECK_IF(ifaContext_->kvHeadNums == nullptr,
             OP_LOGE(ifaContext_->opName, "Attribute numKvHeads is null."), return ge::GRAPH_FAILED);
  OP_CHECK_IF(ifaContext_->scaleValue == nullptr,
             OP_LOGE(ifaContext_->opName, "Attribute scaleValue is null."), return ge::GRAPH_FAILED);
  OP_CHECK_IF(ifaContext_->layOut == nullptr,
             OP_LOGE(ifaContext_->opName, "Attribute inputLayout is null."), return ge::GRAPH_FAILED);
  OP_CHECK_IF(ifaContext_->blockSize == nullptr,
             OP_LOGE(ifaContext_->opName, "Attribute blockSize is null."), return ge::GRAPH_FAILED);

  numHeads_ = *ifaContext_->numHeads;
  numKvHeads_ = *ifaContext_->kvHeadNums;
  blockSize_ = *ifaContext_->blockSize;
  scaleValue_ = *ifaContext_->scaleValue;
  batchSize_ = ifaContext_->query.shape->GetStorageShape().GetDim(NUM0);

  sparseMode_ = ifaContext_->sparseMode == nullptr ? 0 : *ifaContext_->sparseMode;
  preToken_ = ifaContext_->preToken == nullptr ? 0 : *ifaContext_->preToken;
  nextToken_ = ifaContext_->nextToken == nullptr ? 0 : *ifaContext_->nextToken;

  OP_CHECK_IF(numHeads_ == 0, OP_LOGE(ifaContext_->opName, "NumHeads is zero."), return ge::GRAPH_FAILED);
  numKvHeads_ = (numKvHeads_ == 0) ? numHeads_ : numKvHeads_;

  OP_CHECK_IF(((numKvHeads_ > numHeads_) || (numHeads_ % numKvHeads_ != 0)),
              OP_LOGE(ifaContext_->opName, "Attribute numKvHeads is invalid, numHeads:%u, numKvHeads:%u.", numHeads_, numKvHeads_),
              return ge::GRAPH_FAILED);

  inputQType_ = ifaContext_->query.desc->GetDataType();
  inputKvType_ = ifaContext_->key.desc->GetDataType();
  outputType_ = ifaContext_->attenOut.desc->GetDataType();

  std::tuple<ge::DataType, ge::DataType, ge::DataType> inOutDtypeTuple = {inputQType_, inputKvType_, outputType_};
  OP_CHECK_IF(!VecContains(inOutDtypeSupported, inOutDtypeTuple),
      OP_LOGE(ifaContext_->opName,
              "Query dtype(%s), key/value dtype(%s), attentionOut dype(%s) is not currently supported.",
              DataTypeToString(inputQType_).c_str(), DataTypeToString(inputKvType_).c_str(),
              DataTypeToString(outputType_).c_str()),
      return ge::GRAPH_FAILED);

  blockTypeSize_ = sizeof(float);  // 默认按照float计算
  nNumOfQInOneGroup_ = numHeads_ / numKvHeads_;

  inputKvType_ = ifaContext_->key.desc->GetDataType();
  pageAttentionFlag_ = ifaContext_->blockTable.tensor != nullptr;

  if (!pageAttentionFlag_) {
    uint32_t kvBatch = ifaContext_->key.shape->GetStorageShape().GetDim(NUM0);
    batchContinuousFlag_ = (batchSize_ == kvBatch);
  } else if (CheckPABlockSize() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }

  if (ifaContext_->keySharedPrefix.tensor != nullptr && ifaContext_->valueSharedPrefix.tensor != nullptr) {
    enableKVPrefix_ = true;
  }

  std::string layout(ifaContext_->layOut);
  uint32_t nOfQuery = 0;
  if (layout == "BSH") {
    inputLayout_ = IfaLayout::BSH_BSND;
    uint32_t hOfQuery = ifaContext_->query.shape->GetStorageShape().GetDim(NUM2);
    OP_CHECK_IF(hOfQuery % numHeads_ != 0,
               OP_LOGE(ifaContext_->opName, "H[%u] should be an interger multiple of numHeads[%u].", hOfQuery, numHeads_),
               return ge::GRAPH_FAILED);
    sOfQuery_ = ifaContext_->query.shape->GetStorageShape().GetDim(NUM1);
    headDim_ = ifaContext_->query.shape->GetStorageShape().GetDim(NUM2) / numHeads_;
    nOfQuery = numHeads_;
    headDimK_ = ifaContext_->key.shape->GetStorageShape().GetDim(NUM2) / numKvHeads_;
    headDimV_ = ifaContext_->value.shape->GetStorageShape().GetDim(NUM2) / numKvHeads_;
    headDimOut_ = ifaContext_->attenOut.shape->GetStorageShape().GetDim(NUM2) / numHeads_;
  } else if (layout == "BSND") {
    inputLayout_ = IfaLayout::BSH_BSND;
    sOfQuery_ = ifaContext_->query.shape->GetStorageShape().GetDim(NUM1);
    nOfQuery = ifaContext_->query.shape->GetStorageShape().GetDim(NUM2);
    headDim_ = ifaContext_->query.shape->GetStorageShape().GetDim(NUM3);
    headDimK_ = ifaContext_->key.shape->GetStorageShape().GetDim(NUM3);
    headDimV_ = ifaContext_->value.shape->GetStorageShape().GetDim(NUM3);
    headDimOut_ = ifaContext_->attenOut.shape->GetStorageShape().GetDim(NUM3);
  } else if (layout == "BNSD" || layout == "BNSD_BSND") {
    inputLayout_ = IfaLayout::BNSD;
    nOfQuery = ifaContext_->query.shape->GetStorageShape().GetDim(NUM1);
    sOfQuery_ = ifaContext_->query.shape->GetStorageShape().GetDim(NUM2); // 2 : Q_S
    headDim_ = ifaContext_->query.shape->GetStorageShape().GetDim(NUM3);
    headDimK_ = ifaContext_->key.shape->GetStorageShape().GetDim(NUM3);
    headDimV_ = ifaContext_->value.shape->GetStorageShape().GetDim(NUM3);
    headDimOut_ = ifaContext_->attenOut.shape->GetStorageShape().GetDim(NUM3);
  } else if (layout == "TND") {
    inputLayout_ = IfaLayout::TND;
    nOfQuery = ifaContext_->query.shape->GetStorageShape().GetDim(NUM1);
    headDim_ = ifaContext_->query.shape->GetStorageShape().GetDim(NUM2);
    headDimK_ = ifaContext_->key.shape->GetStorageShape().GetDim(NUM2);
    headDimV_ = ifaContext_->value.shape->GetStorageShape().GetDim(NUM2);
    headDimOut_ = ifaContext_->attenOut.shape->GetStorageShape().GetDim(NUM2);
    if (!pageAttentionFlag_) {
      batchContinuousFlag_ = true;
    }
    if (isMaxWorkspace_) {
      // sOfQuery_后续会影响基本块大小，这里设置为T维度值保证基本块大小为最大值
      sOfQuery_ = ifaContext_->query.shape->GetStorageShape().GetDim(NUM0);
      actualLenQDims_ = 1;
      batchSize_ = actualLenQDims_;
    } else {
      if (CheckActualSeqLens() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
      }
      sOfQuery_ = GetMaxSeqLength(ifaContext_->actualSeqLengthsQ.tensor);
      batchSize_ = actualLenQDims_;
      int64_t tOfQuery = static_cast<int64_t>(ifaContext_->query.shape->GetStorageShape().GetDim(NUM0));
      int64_t tOfkv = static_cast<int64_t>(ifaContext_->key.shape->GetStorageShape().GetDim(NUM0));
      int64_t actualSeqLastSizeOfQuery = ifaContext_->actualSeqLengthsQ.tensor->GetData<int64_t>()[batchSize_ - 1U];
      int64_t actualSeqLastSize = ifaContext_->actualSeqLengths.tensor->GetData<int64_t>()[batchSize_ - 1U];
      if (tOfQuery != actualSeqLastSizeOfQuery) {  
        OP_LOGE(ifaContext_->opName,
            "When layout is TND, T of query[%ld] should be equal to the query's actual sequence lengths[%ld].", tOfQuery, actualSeqLastSizeOfQuery);
        return ge::GRAPH_FAILED;
      }
      if (!pageAttentionFlag_ && tOfkv != 0 && tOfkv != actualSeqLastSize) { // 在kv不是空tensor时，T值应与actualseqkv中最后一个batch的值相等
        OP_LOGE(ifaContext_->opName,
            "When layout is TND, T of kv[%ld] should be equal to the kv's actual sequence lengths[%ld].", tOfkv, actualSeqLastSize);
        return ge::GRAPH_FAILED;
      }
    }
  } else {
    OP_LOGE(ifaContext_->opName, "Only support inputLayout(BSH, BNSD, BSND, BNSD_BSND, TND), actually is %s.", layout.c_str());
    return ge::GRAPH_FAILED;
  }
  if (static_cast<uint64_t>(headDim_) != headDimOut_) {
    OP_LOGE(ifaContext_->opName,
      "Dim of Out[%lu] should be equal to Dim of Query[%u]", headDimOut_, headDim_);
    return ge::GRAPH_FAILED;
  }
  if ((!pageAttentionFlag_) && !(headDimK_ == 0 && headDimV_ == 0) &&
    (static_cast<uint64_t>(headDimK_) != headDimOut_ || static_cast<uint64_t>(headDimV_) != headDimOut_)) { // 在kv不是空tensor时，D值应与out的D相等
    OP_LOGE(ifaContext_->opName,
      "When not in pageAttention scenario, Dim of Out[%lu] should be equal to Dim of Key[%u] and Dim of Value[%u]", headDimOut_, headDimK_, headDimV_);
    return ge::GRAPH_FAILED;
  }
  if (((inputKvType_ == ge::DT_INT4) || (inputKvType_ == ge::DT_FLOAT4_E2M1)) &&
      headDim_ % KVINT4_BYTE_BLOCK != 0) {
      OP_LOGE(ifaContext_->opName,
                "When the KV_dtype is int4(int32)/fp4_e2m1, headDims must be a multiple of %u, current dim of D is %u.",
                KVINT4_BYTE_BLOCK, headDim_);
      return ge::GRAPH_FAILED;
  }

  headDimAlign_ = AlignUp(headDim_, BYTE_BLOCK); // 元素个数按照基本块大小对齐

  OP_CHECK_IF(nOfQuery != numHeads_, OP_LOGE(ifaContext_->opName,"N of query[%u] should be equal to numHeads[%u]",
             nOfQuery, numHeads_),
             return ge::GRAPH_FAILED);
  if (CheckLse() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  if (KvShapePostProcess() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  if (emptyTensor_) {
    return ge::GRAPH_SUCCESS;
  }
  antiQuantFlag_ = (inputQType_ != inputKvType_) && ((inputKvType_ == ge::DT_INT8) ||
                   (inputKvType_ == ge::DT_INT4) || (inputKvType_ == ge::DT_HIFLOAT8) ||
                   (inputKvType_ == ge::DT_FLOAT8_E4M3FN) ||
                   (inputKvType_ == ge::DT_FLOAT4_E2M1));
  if (socVersion_ == IfaSocVersion::SOC_ASCEND_950 && antiQuantFlag_ && sOfQuery_ > 1) {
    isPFAFlag_ = true;
  }
  OP_CHECK_IF(sOfQuery_ != NUM1 && !isPFAFlag_,
    OP_LOGE(ifaContext_->opName, "S of query[%u] is invalid, it should be 1.", sOfQuery_),
    return ge::GRAPH_FAILED);

  const uint32_t *innerPrecisePtr = ifaContext_->innerPrecise;
  innerPrecise_ = innerPrecisePtr ? *innerPrecisePtr : IFA_HIGH_PERFORMANCE; // 910B默认高性能

  OP_CHECK_IF(((innerPrecise_ != IFA_HIGH_PERFORMANCE) && (innerPrecise_ != IFA_HIGH_PRECISION) &&
              (innerPrecise_ != HIGH_PERFORMANCE_ROW_INVALID) && (innerPrecise_ != HIGH_PRECISION_ROW_INVALID)),
              OP_LOGE(ifaContext_->opName, "Precision mode[%u] should be 0 or 1 or 2 or 3.", innerPrecise_),
              return ge::GRAPH_FAILED); // pfa支持0-3，第0位：0-高精度，1-高性能；第1位：0-不开启行无效修正，1-开启行无效修正
  isRowInvalid_ = (innerPrecise_ >> 1) & 1;
  innerPrecise_ &= NUM1;

  OP_LOGD(ifaContext_->opName, "InnerPrecise is %u", innerPrecise_);

  if (!pageAttentionFlag_ && CheckQKOutShape() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED; // page_attention don't check this place
  }

  quantFlag_ = (inputQType_ == ge::DT_INT8) && (inputKvType_ == ge::DT_INT8);
  if (CheckInputFormatAndLimits() != ge::GRAPH_SUCCESS ||
      CheckInputAntiquantFormat() != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
  }

  return InitInOutMode();
}

bool IFATilingV2::EnableC1V1() const {
  if (splitKVFlagLocal_) {
    return false;
  }
  // 2:IFA中核数不超过vector总核数一半，可以按1:1启动cube和vector
  return (perfMode_ == IfaPerfMode::NORMAL) && (batchSize_ * numKvHeads_ * NUM2 <= aivNum_) && sOfQuery_ == NUM1;
}

void IFATilingV2::UpdatePerfMode() {
  if (EnableC1V1()) {
    perfMode_ = IfaPerfMode::C1_V1;
  }
}

ge::graphStatus IFATilingV2::CheckInputFormatAndLimits() const {
  auto qFormat = ifaContext_->query.desc->GetOriginFormat();
  auto kFormat = ifaContext_->key.desc->GetOriginFormat();
  auto vFormat = ifaContext_->value.desc->GetOriginFormat();

  OP_CHECK_IF(CheckFormat(qFormat, "Query") != ge::GRAPH_SUCCESS,
             OP_LOGE(ifaContext_->opName, "Query check format failed"),
             return ge::GRAPH_FAILED);
  OP_CHECK_IF(CheckFormat(kFormat, "Key") != ge::GRAPH_SUCCESS,
             OP_LOGE(ifaContext_->opName, "Key check format failed"),
             return ge::GRAPH_FAILED);
  OP_CHECK_IF(CheckFormat(vFormat, "Value") != ge::GRAPH_SUCCESS,
             OP_LOGE(ifaContext_->opName, "Value check format failed"),
             return ge::GRAPH_FAILED);

  OP_CHECK_IF((ifaContext_->attenMask.desc != nullptr &&
              CheckFormat(ifaContext_->attenMask.desc->GetOriginFormat(), "AttenMask") != ge::GRAPH_SUCCESS),
              OP_LOGE(ifaContext_->opName, "AttenMask check format failed"),
              return ge::GRAPH_FAILED);
  OP_CHECK_IF((ifaContext_->pseShift.desc != nullptr &&
              CheckFormat(ifaContext_->pseShift.desc->GetOriginFormat(), "PseShift") != ge::GRAPH_SUCCESS),
              OP_LOGE(ifaContext_->opName, "PseShift check format failed"),
              return ge::GRAPH_FAILED);

  OP_CHECK_IF((nNumOfQInOneGroup_ > NUM64),
             OP_LOGE(ifaContext_->opName, "numHeads / numKvHeads = %u, cannot be greater than 64.", nNumOfQInOneGroup_),
             return ge::GRAPH_FAILED);

  OP_CHECK_IF((inputQType_ == ge::DT_INT8 && inputKvType_ == ge::DT_INT8),
             OP_LOGE(ifaContext_->opName, "IFA not support query/key/value datatype all int8."),
             return ge::GRAPH_FAILED);

  OP_CHECK_IF((batchSize_ > 65536),
             OP_LOGE(ifaContext_->opName, "Batch size:%u cannot be greater than 65536.", batchSize_),
             return ge::GRAPH_FAILED);

  OP_CHECK_IF((headDim_ > NUM512),
             OP_LOGE(ifaContext_->opName, "HeadDims:%u cannot be greater than NUM512.", headDim_),
             return ge::GRAPH_FAILED);
  OP_CHECK_IF((numKvHeads_ > NUM256),
             OP_LOGE(ifaContext_->opName, "NumHeads of key and value:%u cannot be greater than 256.", numKvHeads_),
             return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATilingV2::CheckKVHeadNum(const gert::StorageShape *inputShape) const
{
    uint32_t tmpNumHeads = 0;
    std::string layOutStr = ifaContext_->layOut;
    if (layOutStr == "BSH") {
        auto H = inputShape->GetStorageShape().GetDim(NUM2);
        tmpNumHeads = H / headDim_;
    } else if (layOutStr == "BNSD" || layOutStr == "BNSD_BSND" || layOutStr == "TND") {
        tmpNumHeads = inputShape->GetStorageShape().GetDim(NUM1);
    } else if (layOutStr == "BSND") {
        tmpNumHeads = inputShape->GetStorageShape().GetDim(NUM2);
    }
    OP_CHECK_IF(
        tmpNumHeads != numKvHeads_,
        OP_LOGE(ifaContext_->opName, "IFA check input param failed, tensor in list numHeads[%u] should be %u.",
            tmpNumHeads, numKvHeads_),
            return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool IFATilingV2::CheckEmptyTensor() {
  for (int64_t size = 0; size < ifaContext_->kCache.size(); ++size) {
    auto keyTensorInList = ifaContext_->kCache[size];
    if (keyTensorInList != nullptr && keyTensorInList->GetStorageShape().GetShapeSize() != 0) {
      return false;
    }
  }
  for (int64_t size = 0; size < ifaContext_->vCache.size(); ++size) {
    auto valueTensorInList = ifaContext_->vCache[size];
    if (valueTensorInList != nullptr && valueTensorInList->GetStorageShape().GetShapeSize() != 0) {
      return false;
    }
  }
  return true;
}

ge::graphStatus IFATilingV2::CheckKVShape(int64_t batchOfQuery) {
  for (int64_t size = 0; size < batchOfQuery; ++size) {
    auto keyTensorInList = ifaContext_->kCache[size];
    auto valueTensorInList = ifaContext_->vCache[size];

    OP_CHECK_IF((keyTensorInList == nullptr) || (valueTensorInList == nullptr),
      OP_LOGE(ifaContext_->opName, "IFA check input param failed, key/value tensor list length should be greater than or equal to query batch."),
      return ge::GRAPH_FAILED);
    std::string layOutStr = ifaContext_->layOut;
    if (layOutStr == "BSH") {
      OP_CHECK_IF((keyTensorInList->GetStorageShape().GetDimNum() != DIM_BSH) || (valueTensorInList->GetStorageShape().GetDimNum() != DIM_BSH),
        OP_LOGE(ifaContext_->opName, "IFA check input param failed, the dimension of tensor in tensorList should be 3, key:%lu, value:%lu.",
        keyTensorInList->GetStorageShape().GetDimNum(), valueTensorInList->GetStorageShape().GetDimNum()),
        return ge::GRAPH_FAILED);
    }
    if ((layOutStr == "BNSD") || (layOutStr == "BSND") || (layOutStr == "BNSD_BSND")) {
      OP_CHECK_IF((keyTensorInList->GetStorageShape().GetDimNum() != DIM_BNSD_OR_BSND) || (valueTensorInList->GetStorageShape().GetDimNum() != DIM_BNSD_OR_BSND),
        OP_LOGE(ifaContext_->opName, "IFA check input param failed, the dimension of tensor in tensorList should be 4, key:%lu, value:%lu.",
        keyTensorInList->GetStorageShape().GetDimNum(), valueTensorInList->GetStorageShape().GetDimNum()),
        return ge::GRAPH_FAILED);
    }
    if (layOutStr == "TND") {
      OP_LOGE(ifaContext_->opName, "TND not support TensorInList!");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(keyTensorInList->GetStorageShape().GetDim(NUM0) != NUM1,
      OP_LOGE(ifaContext_->opName, "IFA check input param failed, the batch of tensor in tensorList should be 1, now batch is:%ld, list index:%ld.",
      keyTensorInList->GetStorageShape().GetDim(NUM0), size), return ge::GRAPH_FAILED);
    if (CheckKVHeadNum(keyTensorInList) != ge::GRAPH_SUCCESS ||
        CheckKVHeadNum(valueTensorInList) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATilingV2::CheckKVShapePre() {
  auto batchOfQuery = ifaContext_->query.shape->GetStorageShape().GetDim(NUM0);
  auto batchOfKey = ifaContext_->key.shape->GetStorageShape().GetDim(NUM0);
  if (isMaxWorkspace_) {
    return ge::GRAPH_SUCCESS;
  }
  if (inputLayout_ == IfaLayout::TND) {
    if (CheckEmptyTensor()) {
      emptyTensor_ = true;
      return ge::GRAPH_SUCCESS;
    }
    batchOfQuery = ifaContext_->actualSeqLengthsQ.tensor->GetShapeSize();
    batchOfKey = ifaContext_->actualSeqLengths.tensor->GetShapeSize();
  }
  /* kv continuous */
  if (batchOfQuery == batchOfKey) {
    emptyTensor_ = CheckEmptyTensor();
    return ge::GRAPH_SUCCESS;
  }
  /* kv not continuous */
  if (CheckEmptyTensor()) {
    emptyTensor_ = true;
    return ge::GRAPH_SUCCESS;
  }
  return CheckKVShape(batchOfQuery);
}

ge::graphStatus IFATilingV2::CheckQKOutShape() const
{
    // queryShape (b, 1, h)
  const gert::StorageShape *queryShape = ifaContext_->query.shape;
  const gert::StorageShape *keyShape = ifaContext_->kCache[0];
  const std::string inputLayoutStr = ifaContext_->layOut;

  auto dimOfQ = queryShape->GetStorageShape().GetDimNum();
  auto dimOfK = keyShape->GetStorageShape().GetDimNum();
  auto dimOfOut = ifaContext_->attenOut.shape->GetStorageShape().GetDimNum();
  if (inputLayoutStr == "BSH") {
    OP_CHECK_IF((dimOfQ != DIM_BSH) || (dimOfK != DIM_BSH) || (dimOfOut != DIM_BSH),
      OP_LOGE(ifaContext_->opName, "When inputLayout is BSH, the dimension should be 3, dimOfQ:%lu, dimOfK:%lu, dimOfOut:%lu.", dimOfQ, dimOfK, dimOfOut),
      return ge::GRAPH_FAILED);
    OP_CHECK_IF(queryShape->GetStorageShape().GetDim(NUM1) != NUM1 && !isPFAFlag_,
      OP_LOGE(ifaContext_->opName, "When inputLayout is BSH, the 2nd dimOfQ should be 1,the 2nd dimOfQ:%ld.",
      queryShape->GetStorageShape().GetDim(NUM1)), return ge::GRAPH_FAILED);
    OP_CHECK_IF(queryShape->GetStorageShape().GetDim(NUM2) / numHeads_ != keyShape->GetStorageShape().GetDim(NUM2) / numKvHeads_,
      OP_LOGE(ifaContext_->opName, "When inputLayout is BSH, the 3rd dimOfQ/numHeads not equal the 3rd dimOfK/numKvHeads, dimOfQ/numHeads:%ld, dimOfK/numKvHeads:%ld.",
      queryShape->GetStorageShape().GetDim(NUM2) / numHeads_, keyShape->GetStorageShape().GetDim(NUM2) / numHeads_),
      return ge::GRAPH_FAILED);
  } else if (inputLayoutStr == "TND") {
    OP_CHECK_IF((dimOfQ != DIM_TND) || (dimOfK != DIM_TND) || (dimOfOut != DIM_TND),
      OP_LOGE(ifaContext_->opName, "When inputLayout is TND, the dimension should be 3, dimOfQ:%lu, dimOfK:%lu, dimOfOut:%lu.", dimOfQ, dimOfK, dimOfOut),
      return ge::GRAPH_FAILED);
    OP_CHECK_IF(queryShape->GetStorageShape().GetDim(NUM1) != numHeads_ || keyShape->GetStorageShape().GetDim(NUM1) != numKvHeads_,
      OP_LOGE(ifaContext_->opName, "When inputLayout is TND, the headDims in queryShape[%ld] is not equal to numHeads[%ld] or the headDims in kvShape[%ld] is not equal to numKvHeads[%ld].",
      queryShape->GetStorageShape().GetDim(NUM1), numHeads_, keyShape->GetStorageShape().GetDim(NUM1), numKvHeads_),return ge::GRAPH_FAILED);
  } else {
    OP_CHECK_IF((dimOfQ != DIM_BNSD_OR_BSND) || (dimOfK != DIM_BNSD_OR_BSND) || (dimOfOut != DIM_BNSD_OR_BSND),
      OP_LOGE(ifaContext_->opName, "When inputLayout is BNSD/BSND, the dimension should be 4, dimOfQ:%lu, dimOfK:%lu, dimOfOut:%lu.", dimOfQ, dimOfK, dimOfOut),
      return ge::GRAPH_FAILED);
    OP_CHECK_IF(queryShape->GetStorageShape().GetDim(NUM3) != keyShape->GetStorageShape().GetDim(NUM3),
      OP_LOGE(ifaContext_->opName, "When inputLayout is BNSD/BSND, the headDims in queryShape[%ld] is not equal to the headDims in keyShape[%ld].",
      queryShape->GetStorageShape().GetDim(NUM3), keyShape->GetStorageShape().GetDim(NUM3)),return ge::GRAPH_FAILED);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATilingV2::CheckLse() const
{
  if (!softmaxLseFlag_) {
    return ge::GRAPH_SUCCESS;
  }
  const gert::StorageShape *lseShape = ifaContext_->lseOut.shape;
  const gert::StorageShape *queryShape = ifaContext_->query.shape;
  OP_CHECK_IF(lseShape == nullptr,
             OP_LOGE(ifaContext_->opName, "SoftmaxLse shape is nullptr but softmaxLseFlag is true!"),
             return ge::GRAPH_FAILED);
  OP_CHECK_IF(ifaContext_->lseOut.desc == nullptr, OP_LOGE(ifaContext_->opName, "Desc of lseOut tensor is null."),
             return ge::GRAPH_FAILED);
  auto lseOutType_ = ifaContext_->lseOut.desc->GetDataType();
  OP_CHECK_IF(lseOutType_ != ge::DT_FLOAT,
             OP_LOGE(ifaContext_->opName, "Datatype of lseOut is %s, which is not supported.",
                       DataTypeToString(lseOutType_).c_str()),
             return ge::GRAPH_FAILED);

  if (emptyTensor_) { // q、out为空时，lse为空则不输出，不为空则输出inf，不做拦截
    return ge::GRAPH_SUCCESS;
  }

  if (inputLayout_ == IfaLayout::TND) {
    OP_CHECK_IF(lseShape->GetStorageShape().GetDimNum() != 3,
      OP_LOGE(ifaContext_->opName, "TND SoftmaxLse shape dim(%zu) should be 3!", lseShape->GetStorageShape().GetDimNum()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(((lseShape->GetStorageShape().GetDim(NUM0) != queryShape->GetStorageShape().GetDim(NUM0)) ||
              (lseShape->GetStorageShape().GetDim(NUM1) != numHeads_) || (lseShape->GetStorageShape().GetDim(NUM2) != 1)),
              OP_LOGE(ifaContext_->opName, "SoftmaxLse shape[%ld, %ld, %ld] does not match TN1[%u, %u, 1]!",
                        lseShape->GetStorageShape().GetDim(NUM0), lseShape->GetStorageShape().GetDim(NUM1),
                        lseShape->GetStorageShape().GetDim(NUM2), queryShape->GetStorageShape().GetDim(NUM0), numHeads_),
                        return ge::GRAPH_FAILED);
  } else {
    OP_CHECK_IF(lseShape->GetStorageShape().GetDimNum() != 4,
      OP_LOGE(ifaContext_->opName, "SoftmaxLse shape dim(%zu) should be 4!", lseShape->GetStorageShape().GetDimNum()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(((lseShape->GetStorageShape().GetDim(NUM0) != queryShape->GetStorageShape().GetDim(NUM0)) ||
              (lseShape->GetStorageShape().GetDim(NUM1) != numHeads_) || (lseShape->GetStorageShape().GetDim(NUM2) != sOfQuery_) ||
              (lseShape->GetStorageShape().GetDim(NUM3) != NUM1)), OP_LOGE(ifaContext_->opName, "SoftmaxLse shape[%ld, %ld, %ld, %ld] does not match [B, N, Q_S, 1]([%ld, %u, %u, 1])!",
                        lseShape->GetStorageShape().GetDim(NUM0), lseShape->GetStorageShape().GetDim(NUM1),
                        lseShape->GetStorageShape().GetDim(NUM2), lseShape->GetStorageShape().GetDim(NUM3),
                        queryShape->GetStorageShape().GetDim(NUM0), numHeads_, sOfQuery_),
                        return ge::GRAPH_FAILED);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATilingV2::CheckKvCache() {
  auto blockTable = ifaContext_->blockTable.tensor;
  OP_CHECK_IF(blockTable->GetStorageShape().GetDimNum() != 2U,
             OP_LOGE(ifaContext_->opName, "When Page Attention is enabled, input blockTable dimension[%lu] should be 2",
             blockTable->GetStorageShape().GetDimNum()), 
             return ge::GRAPH_FAILED);
  maxBlockNumPerSeq_ = ifaContext_->blockTable.tensor->GetStorageShape().GetDim(NUM1);
  OP_CHECK_IF(maxBlockNumPerSeq_ <= 0,
            OP_LOGE(ifaContext_->opName,
                      "When Page Attention is enabled, maxBlockNumPerSeq should be larger than 0, actual is %u.", maxBlockNumPerSeq_),
            return ge::GRAPH_FAILED);
  sMax_ = maxBlockNumPerSeq_ * blockSize_;
  seqSize_ = sMax_;
  uint32_t kDimNum = ifaContext_->key.shape->GetStorageShape().GetDimNum();
  OP_CHECK_IF((kDimNum != NUM3 && kDimNum != 4 && kDimNum != 5),
             OP_LOGE(ifaContext_->opName, "When Page Attention is enabled, kvCache dimensions[%u] should be 3 or 4 or 5", kDimNum),
             return ge::GRAPH_FAILED);
  if (kDimNum == 3U) { // BSH
    pageAttentionKvLayoutType_ = KvCacheLayout::KV_CACHE_BSH;
    pageAttentionKvLayoutTypefaRun_ = static_cast<uint8_t>(KvCacheLayout::KV_CACHE_BSH);
  } else if (kDimNum == 4U) { // BNSD
    pageAttentionKvLayoutType_ = KvCacheLayout::KV_CACHE_BNSD;
    pageAttentionKvLayoutTypefaRun_ = static_cast<uint8_t>(KvCacheLayout::KV_CACHE_BNSD);
  } else { // NZ
    pageAttentionKvLayoutType_ = KvCacheLayout::KV_CACHE_NZ;
    pageAttentionKvLayoutTypefaRun_ = static_cast<uint8_t>(KvCacheLayout::KV_CACHE_NZ);
  }
  paBlockNumSumfaRun_ = static_cast<int32_t>(kDimNum);
  const std::string inputLayoutStr = ifaContext_->layOut;
  OP_CHECK_IF((kDimNum == DIM_BNSD && inputLayout_ != IfaLayout::BNSD && inputLayout_ != IfaLayout::TND),
             OP_LOGE(ifaContext_->opName, "When Page Attention is enabled, and KV cache dimensions are 4-dimensional, "
                                         "inputlayout must be BNSD or BNSD_BSND or TND"),
             return ge::GRAPH_FAILED);
  if (CheckKvCacheValue(kDimNum) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

std::string IFATilingV2::GetShapeStr(const gert::Shape &aShape) const {
  std::string shapeStr = "[";
  for (size_t i = 0; i < aShape.GetDimNum(); ++i) {
      shapeStr += std::to_string(aShape.GetDim(i)) + (i + 1 < aShape.GetDimNum() ? ", " : "");
  }
  return shapeStr + "]";
}

ge::graphStatus IFATilingV2::CheckKvCacheValue(uint32_t kDimNum) const {
  OP_CHECK_IF(!ShapeEqual(ifaContext_->key.shape->GetStorageShape(), ifaContext_->value.shape->GetStorageShape()),
            OP_LOGE(ifaContext_->opName, "Key shape%s and value shape%s should be same.",
                      GetShapeStr(ifaContext_->key.shape->GetStorageShape()).c_str(),
                      GetShapeStr(ifaContext_->value.shape->GetStorageShape()).c_str()),
            return ge::GRAPH_FAILED);
  if (kDimNum == 3U) { // BSH
    uint32_t blockSize = ifaContext_->key.shape->GetStorageShape().GetDim(NUM1);
    uint32_t hOfKeyCache = ifaContext_->key.shape->GetStorageShape().GetDim(NUM2);
    uint32_t hOfKey = numKvHeads_ * headDim_;
    OP_CHECK_IF((blockSize != blockSize_),
               OP_LOGE(ifaContext_->opName, "When Page Attention is enabled, blockSize of kvCache[%u] should be %u", blockSize, blockSize_),
               return ge::GRAPH_FAILED);
    if (inputKvType_ == ge::DT_INT4) {
      OP_CHECK_IF((hOfKeyCache != hOfKey),
                 OP_LOGE(ifaContext_->opName,
                 "When Page Attention is enabled, if input kv dataType is INT32, H of kvCache[%u] should be %u; "
                 "if input kv dataType is INT4, H of kvCache[%u] should be %u",
                 hOfKeyCache / NUM8, hOfKey / NUM8, hOfKeyCache, hOfKey),
                 return ge::GRAPH_FAILED);
    } else {
      OP_CHECK_IF((hOfKeyCache != hOfKey),
                 OP_LOGE(ifaContext_->opName, "When Page Attention is enabled, H of kvCache[%u] should be %u", hOfKeyCache, hOfKey),
                 return ge::GRAPH_FAILED);
    }
  } else if (kDimNum == 4U) { // BNSD
    uint32_t nOfKey = ifaContext_->key.shape->GetStorageShape().GetDim(NUM1);
    uint32_t blockSize = ifaContext_->key.shape->GetStorageShape().GetDim(NUM2);
    uint32_t dimOfKey = ifaContext_->key.shape->GetStorageShape().GetDim(NUM3);
    OP_CHECK_IF((blockSize != blockSize_),
               OP_LOGE(ifaContext_->opName, "When Page Attention is enabled, blockSize of kvCache[%u] should be %u", blockSize, blockSize_),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF((nOfKey != numKvHeads_),
               OP_LOGE(ifaContext_->opName, "When Page Attention is enabled, numHeads of kvCache[%u] should be %u", nOfKey, numKvHeads_),
               return ge::GRAPH_FAILED);
    if (inputKvType_ == ge::DT_INT4) {
      OP_CHECK_IF((dimOfKey != headDim_),
                 OP_LOGE(ifaContext_->opName,
                 "When Page Attention is enabled, if input kv dataType is INT32, headDim of kvCache[%u] should be %u; "
                 "if input kv dataType is INT4, headDim of kvCache[%u] should be %u",
                 dimOfKey / NUM8, headDim_ / NUM8, dimOfKey, headDim_),
                 return ge::GRAPH_FAILED);
    } else {
      OP_CHECK_IF((dimOfKey != headDim_),
                 OP_LOGE(ifaContext_->opName, "When Page Attention is enabled, headDim of kvCache[%u] should be %u", dimOfKey, headDim_),
                 return ge::GRAPH_FAILED);
    }
  } else {
      uint32_t nOfKey = ifaContext_->key.shape->GetStorageShape().GetDim(NUM1);
      uint32_t d1OfKey = ifaContext_->key.shape->GetStorageShape().GetDim(NUM2);
      uint32_t blockSize = ifaContext_->key.shape->GetStorageShape().GetDim(NUM3);
      uint32_t d0OfKey = ifaContext_->key.shape->GetStorageShape().GetDim(NUM4);
      OP_CHECK_IF((nOfKey != numKvHeads_),
              OP_LOGE(ifaContext_->opName, "When Page Attention is enabled, numHeads of kvCache[%u] should be %u", nOfKey, numKvHeads_),
              return ge::GRAPH_FAILED);
      OP_CHECK_IF((blockSize != blockSize_),
              OP_LOGE(ifaContext_->opName, "When Page Attention is enabled, blockSize of kvCache[%u] should be %u", blockSize, blockSize_),
              return ge::GRAPH_FAILED);
      if (inputKvType_ == ge::DT_INT4) {
        OP_CHECK_IF((d0OfKey != BLOCK_SIZE),
                    OP_LOGE(ifaContext_->opName,
                    "When PA_NZ is enabled, if input kv dataType is INT32, the last dim (D0) of kvCache[%u] should be %u; "
                    "if input kv dataType is INT4, the last dim (D0) of kvCache[%u] should be %u",
                    d0OfKey / NUM8, BLOCK_SIZE / NUM8, d0OfKey, BLOCK_SIZE),
                    return ge::GRAPH_FAILED);
      } else {
        OP_CHECK_IF((d0OfKey != BLOCK_SIZE),
                    OP_LOGE(ifaContext_->opName,
                    "When PA_NZ is enabled, the last dim (D0) of kvCache[%u] should be %u",
                    d0OfKey, BLOCK_SIZE),
                    return ge::GRAPH_FAILED);
      }
      uint32_t dimOfKey = d1OfKey * d0OfKey;
      if (inputKvType_ == ge::DT_INT4) {
        OP_CHECK_IF((dimOfKey != headDim_),
                  OP_LOGE(ifaContext_->opName,
                  "When Page Attention is enabled, if input kv dataType is INT32, headDim of kvCache[%u] should be %u; "
                  "if input kv dataType is INT4, headDim of kvCache[%u] should be %u",
                  dimOfKey / NUM8, headDim_ / NUM8, dimOfKey, headDim_),
                  return ge::GRAPH_FAILED);
      } else {
        OP_CHECK_IF((d1OfKey != (headDim_ / d0OfKey)),
                  OP_LOGE(ifaContext_->opName, "When Page Attention NZ is enabled, the third dim (D1) of kvCache[%u] should be %u", d1OfKey, headDim_ / d0OfKey),
                  return ge::GRAPH_FAILED);

        OP_CHECK_IF((dimOfKey != headDim_),
                  OP_LOGE(ifaContext_->opName, "When Page Attention is enabled, headDim of kvCache[%u] should be %u", dimOfKey, headDim_),
                  return ge::GRAPH_FAILED);
      }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATilingV2::KvShapePostProcess() {
  if (pageAttentionFlag_) {
    if (CheckEmptyTensor()) {
      emptyTensor_ = true;
      return ge::GRAPH_SUCCESS;
    }
    if (CheckKvCache() != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
  }

  for (size_t i = 0; i < ifaContext_->kCache.size(); i++) {
    auto keyShape = ifaContext_->kCache[i];
    auto valueShape = ifaContext_->vCache[i];

    OP_CHECK_IF((keyShape == nullptr || valueShape == nullptr),
                OP_LOGE(ifaContext_->opName, "Tensor shape of list[%zu] is null.", i),
                return ge::GRAPH_FAILED);

    if (!ShapeEqual(keyShape->GetStorageShape(), valueShape->GetStorageShape())) {
      OP_LOGE(ifaContext_->opName, "Key and value shape shoud be same.");
      return ge::GRAPH_FAILED;
    }

    if (CheckKVShapePre() != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
    }

    if (emptyTensor_) {
      return ge::GRAPH_SUCCESS;
    }

    if (CheckKeyShapeTensor(keyShape->GetStorageShape(), i) != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
    }

    uint32_t seqSize;
    if (inputLayout_ == IfaLayout::BSH_BSND) {
      seqSize = keyShape->GetStorageShape().GetDim(NUM1);
    } else if (inputLayout_ == IfaLayout::TND) {
      if (isMaxWorkspace_) {
        seqSize = ifaContext_->query.shape->GetStorageShape().GetDim(NUM0);
      } else {
        seqSize = GetMaxSeqLength(ifaContext_->actualSeqLengths.tensor);
      }
    } else {
      seqSize = keyShape->GetStorageShape().GetDim(NUM2);  // 2, dim of S
    }

    sMax_ = std::max(seqSize, sMax_);
    kvListSeqLens_.push_back(seqSize);
  }
  seqSize_ = sMax_;
  return ge::GRAPH_SUCCESS;
}

void IFATilingV2::IncreFlashAttentionInitSoftmaxLseOutputSplit() {
  totalSizeLse_ = ifaContext_->lseOut.shape->GetStorageShape().GetShapeSize();
  tilingData_->outputParams.set_totalLseOutputSize(totalSizeLse_);
}

void IFATilingV2::IncreFlashAttentionInitOutputSplit() {
  totalSize_ = ifaContext_->attenOut.shape->GetStorageShape().GetShapeSize();
  // Upward rounding, coreNum has been verified to be non-zero when obtained.
  singleCoreSize_ = (totalSize_ + coreNum_ - 1) / (coreNum_);

  if (enablePostQuant_) {
    // requiring that the number of points allocated to each kernel must be even.
    singleCoreSize_ = ((singleCoreSize_ + 1) / 2) * 2; // 2 : fill in 0
  }

  tilingData_->outputParams.set_singleCoreSize(singleCoreSize_);
  tilingData_->outputParams.set_totalOutputSize(totalSize_);
}

void IFATilingV2::SetEmptyTensor() {
  emptyTensor = true;
  quantMode = NoQuantMode;
  IncreFlashAttentionInitOutputSplit();
  tilingData_->outputParams.set_totalLseOutputSize(0);
  if (softmaxLseFlag_) {
    IncreFlashAttentionInitSoftmaxLseOutputSplit();
  }
  auto platformInfoPtr = context_->GetPlatformInfo();
  OP_CHECK_IF(platformInfoPtr == nullptr,
              OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "platformInfoPtr is null!"),
              return);
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
  ifaContext_->numBlocks = ascendcPlatform.CalcTschBlockDim(coreNum_, aicNum_, coreNum_);
  ifaContext_->workSpaces[0] = ascendcPlatform.GetLibApiWorkSpaceSize();
}

ge::graphStatus IFATilingV2::EmptyTensorProcess() {
  if (emptyTensor_) {
    SetEmptyTensor();
    return ge::GRAPH_SUCCESS;
  }
  return ge::GRAPH_FAILED;
}

ge::graphStatus IFATilingV2::CheckKeyShapeTensor(const gert::Shape& aShape, size_t idx) const {
  auto firstKeyShape = ifaContext_->kCache[0];
  std::string layOutStr = ifaContext_->layOut;
  if (layOutStr == "BNSD" || layOutStr == "BNSD_BSND") {
    OP_CHECK_IF(firstKeyShape->GetStorageShape().GetDim(NUM0) != aShape.GetDim(NUM0),
              OP_LOGE(ifaContext_->opName, "IFA check input param failed, the batch in keyShape must be same, batch in keyShape[0] is %ld, but batch in keyShape[%zu] is %ld.",
              firstKeyShape->GetStorageShape().GetDim(NUM0), idx, aShape.GetDim(NUM0)), return ge::GRAPH_FAILED);
    OP_CHECK_IF(firstKeyShape->GetStorageShape().GetDim(NUM1) != aShape.GetDim(NUM1),
              OP_LOGE(ifaContext_->opName, "IFA check input param failed, the numKvHeads in keyShape must be same, numKvHeads in keyShape[0] is %ld, but numKvHeads in keyShape[%zu] is %ld.",
              firstKeyShape->GetStorageShape().GetDim(NUM1), idx, aShape.GetDim(NUM1)), return ge::GRAPH_FAILED);
    OP_CHECK_IF(firstKeyShape->GetStorageShape().GetDim(NUM3) != aShape.GetDim(NUM3),
              OP_LOGE(ifaContext_->opName, "IFA check input param failed, the headDims in keyShape must be same, headDims in keyShape[0] is %ld, but headDims in keyShape[%zu] is %ld.",
              firstKeyShape->GetStorageShape().GetDim(NUM3), idx, aShape.GetDim(NUM3)), return ge::GRAPH_FAILED);
  } else if (layOutStr == "BSND") {
    OP_CHECK_IF(firstKeyShape->GetStorageShape().GetDim(NUM0) != aShape.GetDim(NUM0),
              OP_LOGE(ifaContext_->opName, "IFA check input param failed, the batch in keyShape must be same, batch in keyShape[0] is %ld, but batch in keyShape[%zu] is %ld.",
              firstKeyShape->GetStorageShape().GetDim(NUM0), idx, aShape.GetDim(NUM0)), return ge::GRAPH_FAILED);
    OP_CHECK_IF(firstKeyShape->GetStorageShape().GetDim(NUM2) != aShape.GetDim(NUM2),
              OP_LOGE(ifaContext_->opName, "IFA check input param failed, the numKvHeads in keyShape must be same, numKvHeads in keyShape[0] is %ld, but numKvHeads in keyShape[%zu] is %ld.",
              firstKeyShape->GetStorageShape().GetDim(NUM2), idx, aShape.GetDim(NUM2)), return ge::GRAPH_FAILED);
    OP_CHECK_IF(firstKeyShape->GetStorageShape().GetDim(NUM3) != aShape.GetDim(NUM3),
              OP_LOGE(ifaContext_->opName, "IFA check input param failed, the headDims in keyShape must be same, headDims in keyShape[0] is %ld, but headDims in keyShape[%zu] is %ld.",
              firstKeyShape->GetStorageShape().GetDim(NUM3), idx, aShape.GetDim(NUM3)), return ge::GRAPH_FAILED);
  } else if (layOutStr == "BSH") {
    OP_CHECK_IF(firstKeyShape->GetStorageShape().GetDim(NUM0) != aShape.GetDim(NUM0),
              OP_LOGE(ifaContext_->opName, "IFA check input param failed, the batch in keyShape must be same, batch in keyShape[0] is %ld, but batch in keyShape[%zu] is %ld.",
              firstKeyShape->GetStorageShape().GetDim(NUM0), idx, aShape.GetDim(NUM0)), return ge::GRAPH_FAILED);
    OP_CHECK_IF(firstKeyShape->GetStorageShape().GetDim(NUM2) != aShape.GetDim(NUM2),
              OP_LOGE(ifaContext_->opName, "IFA check input param failed, the H in keyShape must be same, H in keyShape[0] is %ld, but H in keyShape[%zu] is %ld.",
              firstKeyShape->GetStorageShape().GetDim(NUM2), idx, aShape.GetDim(NUM2)), return ge::GRAPH_FAILED);
  }
  return ge::GRAPH_SUCCESS;
}

bool IFATilingV2::ShapeEqual(const gert::Shape &aShape, const gert::Shape &bShape) const {
  if (aShape.GetDimNum() != bShape.GetDimNum()) {
    return false;
  }

  for (size_t idx = 0; idx < aShape.GetDimNum(); idx++) {
    if (aShape.GetDim(idx) != bShape.GetDim(idx)) {
      return false;
    }
  }

  return true;
}

bool IFATilingV2::CanChangeToNew() const {
  if (inOutMode_ == TilingInOutMode::BF16_BF16) {
    return true;
  }
  if (inOutMode_ == TilingInOutMode::BF16_INT8) {
    return true;
  }

  if (inOutMode_ == TilingInOutMode::FP16_FP16 || inOutMode_ == TilingInOutMode::FP16_INT8) {
    return true;
  }
  return false;
}

ge::graphStatus IFATilingV2::InitInOutMode() {
  if (inputQType_ == ge::DT_INT8 && outputType_ == ge::DT_FLOAT16) {
    inOutMode_ = TilingInOutMode::INT8_FP16;
  } else if (inputQType_ == ge::DT_FLOAT16 && outputType_ == ge::DT_FLOAT8_E4M3FN) {
    inOutMode_ = TilingInOutMode::FP16_FP8_E4M3FN;
  } else if (inputQType_ == ge::DT_FLOAT16 && outputType_ == ge::DT_HIFLOAT8) {
    inOutMode_ = TilingInOutMode::FP16_HIFLOAT8;
  } else if (inputQType_ == ge::DT_FLOAT16 && outputType_ == ge::DT_INT8) {
    inOutMode_ = TilingInOutMode::FP16_INT8;
  } else if (inputQType_ == ge::DT_FLOAT16 && outputType_ == ge::DT_FLOAT16) {
    inOutMode_ = TilingInOutMode::FP16_FP16;
  } else if (inputQType_ == ge::DT_BF16 && outputType_ == ge::DT_BF16) {
    inOutMode_ = TilingInOutMode::BF16_BF16;
  } else if (inputQType_ == ge::DT_BF16 && outputType_ == ge::DT_FLOAT8_E4M3FN) {
    inOutMode_ = TilingInOutMode::BF16_FP8_E4M3FN;
  } else if (inputQType_ == ge::DT_BF16 && outputType_ == ge::DT_HIFLOAT8) {
    inOutMode_ = TilingInOutMode::BF16_HIFLOAT8;
  } else if (inputQType_ == ge::DT_BF16 && outputType_ == ge::DT_INT8) {
    inOutMode_ = TilingInOutMode::BF16_INT8;
  } else if (inputQType_ == ge::DT_FLOAT && outputType_ == ge::DT_FLOAT) {
    inOutMode_ = TilingInOutMode::FP32_FP32;
  } else {
    OP_LOGE(ifaContext_->opName, "Input dataType[%d] with output dataType[%d] is not currently supported.",
                                inputQType_, outputType_);
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATilingV2::ProcessOptionalTensors() {
  if ((ProcessActualSeqLen() != ge::GRAPH_SUCCESS) || (ProcessPseShift() != ge::GRAPH_SUCCESS) ||
      (ProcessAttenMask() != ge::GRAPH_SUCCESS) || (ProcessAttenMaskSparsePFA() != ge::GRAPH_SUCCESS) ||
      (ProcessQuant2() != ge::GRAPH_SUCCESS) || (ProcessAntiQuant() != ge::GRAPH_SUCCESS) ||
      (ProcessBlockTable() != ge::GRAPH_SUCCESS) || (ProcessQPaddingSize() != ge::GRAPH_SUCCESS) ||
      (ProcessKVPaddingSize() != ge::GRAPH_SUCCESS) || (ProcessPrefix() != ge::GRAPH_SUCCESS)) {
    return ge::GRAPH_FAILED;
  }
  SetfaRunFlag();   // 判断是否走伪量化新模板
  if (!isPFAFlag_) {
    preToken_ = SPARSE_MODE_INT_MAX;
    nextToken_ = SPARSE_MODE_INT_MAX;
    if (inputLayout_ != IfaLayout::TND) {
      actualSeqLenQFlag_ = false;
      actualLenQDims_ = static_cast<uint32_t>(0);
    }
    qPaddingSizeFlag_ = false;
    needInit_ = needInitfaRun_;
  }
  return ge::GRAPH_SUCCESS;
}

bool IFATilingV2::CheckPFAMerge() { // PFA场景合轴条件检验
  const int64_t gS1 = nNumOfQInOneGroup_ * sOfQuery_;
  if (gS1 <= 0 || gS1 > pfaMergeGSLimit) { // 溢出或超出基本块大小
      return false;
  }

  // 隔离高阶特性
  std::string layout(ifaContext_->layOut);
  bool hasCrossoverAttr = actualSeqLenUnequal_ || qPaddingSizeFlag_ || attenMaskFlag_ || pseShiftFlag_ || pageAttentionFlag_ ||
    enableAlibiPse_ || enablePostQuant_ || softmaxLseFlag_ || layout == "BNSD_BSND";
  return !hasCrossoverAttr;
}

void IFATilingV2::SetfaRunFlag() {
  if (antiQuantFlag_) {
    if(sOfQuery_ == NUM1 && !enableAlibiPse_) {
      faRunGS_ = true;
      isGqa_ = 1;
    } else {
      bool enablePFAMerge = CheckPFAMerge();
      faRunGS_ = enablePFAMerge;
      isGqa_ = static_cast<uint8_t>(enablePFAMerge);
    }
  } else {
    faRunGS_ = false;
  }
  if (!isPFAFlag_) {
    faRunSparseType_ = static_cast<uint8_t>(IfaSparseEnum::IFA_ALL);
    if (attenMaskFlag_) {
      faRunAttenMaskShapeType_ = 1; // 1 is shape type
    } else {
      faRunAttenMaskShapeType_ = 0;
      attenMaskQSize_ = 0;
    }
  } else {
    SetPFASparseType(sOfQuery_);
    if (attenMaskFlag_) {
      faRunAttenMaskShapeType_ = attenMaskBatch_ > 1 ? 1 : 2; // 2 is shape type
    } else {
      faRunAttenMaskShapeType_ = 0;
    }
  }
}

void IFATilingV2::SetPFASparseType(uint32_t qS)
{
  if (sparseMode_ == SPARSE_MODE_NO_MASK) {
      if (preToken_ >= qS && nextToken_ == 0) {
          faRunSparseType_ = static_cast<uint8_t>(IfaSparseEnum::IFA_CAUSAL);
      } else if (preToken_ >= qS && !pageAttentionFlag_ && nextToken_ >= sMax_) {
          faRunSparseType_ = static_cast<uint8_t>(IfaSparseEnum::IFA_ALL);
      } else {
          faRunSparseType_ = static_cast<uint8_t>(IfaSparseEnum::IFA_BAND);
      }
  } else if (sparseMode_ == SPARSE_MODE_ALL_MASK) {
      faRunSparseType_ = static_cast<uint8_t>(IfaSparseEnum::IFA_ALL);
  } else if (sparseMode_ == SPARSE_MODE_LEFT_UP) {
      faRunSparseType_ = static_cast<uint8_t>(IfaSparseEnum::IFA_CAUSAL);
  } else if (sparseMode_ == SPARSE_MODE_RIGHT_DOWN) {
      if (!pageAttentionFlag_ && qS == sMax_) {
          faRunSparseType_ = static_cast<uint8_t>(IfaSparseEnum::IFA_CAUSAL);
      } else {
          faRunSparseType_ = static_cast<uint8_t>(IfaSparseEnum::IFA_BAND);
      }
  } else if (sparseMode_ == SPARSE_MODE_BAND) {
      faRunSparseType_ = static_cast<uint8_t>(IfaSparseEnum::IFA_BAND);
  }
}

bool IFATilingV2::SetQKVStartIdx()
{
    auto qStartIdxTensor = ifaContext_->qStartIdx.tensor;
    if (qStartIdxTensor != nullptr) {
        if (qStartIdxTensor->GetShapeSize() >= 1) {
            const int64_t *value = qStartIdxTensor->GetData<int64_t>();
            if (value != nullptr) {
                qStartIdx_ = value[0];
                OP_CHECK_IF(qStartIdx_ > INT32_MAX || qStartIdx_ < INT32_MIN,
                            OP_LOGE(ifaContext_->opName, "qStartIdx should >= %d and <= %d, but qStartIdx = %ld.",
                                    INT32_MIN, INT32_MAX, qStartIdx_),
                            return false);
            }
        }
    }

    auto kvStartIdxTensor = ifaContext_->kvStartIdx.tensor;
    if (kvStartIdxTensor != nullptr) {
        if (kvStartIdxTensor->GetShapeSize() >= 1) {
            const int64_t *kvValue = kvStartIdxTensor->GetData<int64_t>();
            if (kvValue != nullptr) {
                kvStartIdx_ = kvValue[0];
                OP_CHECK_IF(kvStartIdx_ > INT32_MAX || kvStartIdx_ < INT32_MIN,
                            OP_LOGE(ifaContext_->opName, "kvStartIdx should >= %d and <= %d, but kvStartIdx = %ld.",
                                    INT32_MIN, INT32_MAX, kvStartIdx_),
                            return false);
            }
        }
    }
    // 当kvStartIdx - qStartIdx超出范围后，kernel侧转为float会造成丢失精度。
    OP_CHECK_IF(kvStartIdx_ - qStartIdx_ > POS_SHIFT_MAX || kvStartIdx_ - qStartIdx_ < POS_SHIFT_MIN,
                OP_LOGE(ifaContext_->opName,
                        "kvStartIdx - qStartIdx should >= %d and <= %d, but qStartIdx = %ld, kvStartIdx = %ld.",
                        POS_SHIFT_MIN, POS_SHIFT_MAX, qStartIdx_, kvStartIdx_),
                return false);
    return true;
}

bool IFATilingV2::CheckAlibiPseShiftTypeAndShape()
{
    auto pseShape = ifaContext_->pseShift.tensor;
    OP_CHECK_IF(pseShape == nullptr, OP_LOGE(ifaContext_->opName, "When pseType = 2/3, pseShift shape is null."),
                return false);
    OP_CHECK_IF(ifaContext_->pseShift.desc == nullptr, OP_LOGE(ifaContext_->opName, "Desc of pseShift tensor is null."),
                return false);
    auto pseShiftDataType = ifaContext_->pseShift.desc->GetDataType();

    OP_CHECK_IF((pseShiftDataType != ge::DT_FLOAT),
                OP_LOGE(ifaContext_->opName, "When pseType = 2/3, pse shift type must be float, but pse shift type = %s.",
                        DataTypeToString(pseShiftDataType).c_str()),
                return false);

    if (pseShape->GetStorageShape().GetDimNum() != 0) {
        auto &pseShapeDims = pseShape->GetStorageShape();
        int64_t pseDimNum = pseShapeDims.GetDimNum();
        OP_CHECK_IF(pseDimNum != SLOPE_N_DIM_NUM,
                    OP_LOGE(ifaContext_->opName, "The dim of pseShift(%ld) must be 1, when pseType = 2/3.", pseDimNum),
                    return false);
        int64_t pseShiftN = pseShape->GetStorageShape().GetDim(0); // 0: The first dimension is N.
        OP_CHECK_IF(pseShiftN != numHeads_,
                    OP_LOGE(ifaContext_->opName,
                            "The length of pseShift(%ld) must be equal to query head number, when pseType = 2/3.",
                            pseShiftN),
                    return false);
        pseShapeType = IfaPseShapeType::PSE_1_N2_G_SLOPE;
        pseShiftBatch_ = 1;
    }
    return true;
}

bool IFATilingV2::AlibiCheckSeqLength()
{
    int64_t actSeqLenData;
    int64_t actSeqLenDataKV;
    for (uint32_t i = 0; i < batchSize_; i++) {
        GetActualSeqLength(actSeqLenData, actSeqLenDataKV, i);
        OP_CHECK_IF(actSeqLenData != actSeqLenDataKV,
                    OP_LOGE(ifaContext_->opName,
                            "When pseType = 2/3, actualSeqLengths[%u](seq size of query)=%ld must be equal to actualSeqLengthsKv[%u](seq size of key)=%ld.",
                            i, actSeqLenData, i, actSeqLenDataKV),
                    return false);
    }
    return true;
}

bool IFATilingV2::CheckAlibiPseShift()
{
    if (!CheckAlibiPseShiftTypeAndShape() || !SetQKVStartIdx() || !AlibiCheckSeqLength()) {
        return false;
    }
    return true;
}

ge::graphStatus IFATilingV2::ProcessPrefix() {
  if (!enableKVPrefix_) {
    actualSharedPrefixLenNullFlag_ = true;
    actualSharedPrefixLen_ = 0;
    return ge::GRAPH_SUCCESS;
  }
  std::string layOutStr = ifaContext_->layOut;
  // Not support prefix
  OP_CHECK_IF((layOutStr == "TND"), OP_LOGE(ifaContext_->opName, "when TND is used, system prefix is not supported!"),
              return false);
  OP_CHECK_IF(!batchContinuousFlag_, OP_LOGE(ifaContext_->opName, "when tensorlist is used, system prefix is not supported!"),
              return false);
  OP_CHECK_IF(qPaddingSizeFlag_ || kvPaddingSizeFlag_, OP_LOGE(ifaContext_->opName, "when leftpadding is used, system prefix is not supported!"),
              return false);
  OP_CHECK_IF(enableAlibiPse_, OP_LOGE(ifaContext_->opName, "when pseType = 2/3, system prefix is not supported!"),
              return false);
  OP_CHECK_IF(pageAttentionFlag_, OP_LOGE(ifaContext_->opName, "when page attention is enabled, system prefix is not supported!"),
              return false);
  OP_CHECK_IF((ifaContext_->query.desc->GetDataType() == ge::DT_INT8) &&
                  (ifaContext_->key.desc->GetDataType() == ge::DT_INT8),
              OP_LOGE(ifaContext_->opName, "when system prefix is used, query and key/value should not both be int8!"),
              return false);
  
  const gert::Shape keyPrefixShape = ifaContext_->keySharedPrefix.tensor->GetStorageShape();
  const gert::Shape valuePrefixShape = ifaContext_->valueSharedPrefix.tensor->GetStorageShape();
  const gert::Shape keyShape = ifaContext_->key.shape->GetStorageShape();
  const ge::DataType keyPrefixType = ifaContext_->keySharedPrefix.tensor->GetDataType();
  const ge::DataType valuePrefixType = ifaContext_->valueSharedPrefix.tensor->GetDataType();
  const ge::DataType keyType = ifaContext_->key.desc->GetDataType();

  // KV prefix dtype
  OP_CHECK_IF(keyPrefixType != valuePrefixType,
              OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName, "key shared prefix and value shared prefix dtype should be same."),
              return ge::GRAPH_FAILED);
  OP_CHECK_IF(keyPrefixType != keyType || valuePrefixType != keyType,
              OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName, "key shared prefix and value shared prefix dtype should have same dtype with key and value."),
              return ge::GRAPH_FAILED);
  // KV prefix shape
  OP_CHECK_IF(!GetAndCheckPrefixShape(layOutStr, keyPrefixShape, valuePrefixShape, keyShape),
              OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName, "Get and check KV shared prefix shape failed."),
              return ge::GRAPH_FAILED);
  // KV prefix consistency
  OP_CHECK_IF(!CheckKeyValuePrefixConsistency(keyPrefixShape, valuePrefixShape, keyShape),
              OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName, "KV shared prefix consistency check failed."),
              return ge::GRAPH_FAILED);
  // check actsharedPrefix
  auto actualSharedPrefixLenInput = ifaContext_->actualSharedPrefixLen.tensor;
  if (layOutStr == "BNSD" || layOutStr == "BNSD_BSND") {
    prefixSSize_ = keyPrefixShape.GetDim(2);
  } else if (layOutStr == "BSH" || layOutStr == "BSND") {
    prefixSSize_ = keyPrefixShape.GetDim(1);
  }
  if (actualSharedPrefixLenInput == nullptr) {
    actualSharedPrefixLenNullFlag_ = true;
    actualSharedPrefixLen_ = prefixSSize_;
  } else {
    if (!CheckActualSharedPrefixLen(actualSharedPrefixLenInput, keyPrefixShape, prefixSSize_)) {
            return ge::GRAPH_FAILED;
    }
    actualSharedPrefixLenNullFlag_ = 0;
  }
  return ge::GRAPH_SUCCESS;
}

bool IFATilingV2::GetAndCheckPrefixShape(std::string layOutStr, const gert::Shape keyPrefixShape, const gert::Shape valuePrefixShape, const gert::Shape keyShape) {
  int64_t keyPrefixBSize_ = 0L;
  int64_t keyPrefixSSize_ = 0L;
  int64_t keyPrefixNSize_ = 0L;
  int64_t keyPrefixDSize_ = 0L;
  int64_t keyPrefixHSize_ = 0L;
  int64_t valuePrefixBSize_ = 0L;
  int64_t valuePrefixSSize_ = 0L;
  int64_t valuePrefixNSize_ = 0L;
  int64_t valuePrefixDSize_ = 0L;
  int64_t valuePrefixHSize_ = 0L;
  int64_t kvNSize_ = 0L;
  int64_t kvDSize_ = 0L;
  int64_t kvHSize_ = 0L;

  keyPrefixBSize_ = keyPrefixShape.GetDim(0);
  valuePrefixBSize_ = valuePrefixShape.GetDim(0);
  if (layOutStr == "BNSD" || layOutStr == "BNSD_BSND") {
    keyPrefixNSize_ = keyPrefixShape.GetDim(1);
    keyPrefixSSize_ = keyPrefixShape.GetDim(2);
    keyPrefixDSize_ = keyPrefixShape.GetDim(3);
    valuePrefixNSize_ = valuePrefixShape.GetDim(1);
    valuePrefixSSize_ = valuePrefixShape.GetDim(2);
    valuePrefixDSize_ = valuePrefixShape.GetDim(3);
    kvNSize_ = keyShape.GetDim(1);
    kvDSize_ = keyShape.GetDim(3);
  } else if (layOutStr == "BSND") {
    keyPrefixSSize_ = keyPrefixShape.GetDim(1);
    keyPrefixNSize_ = keyPrefixShape.GetDim(2);
    keyPrefixDSize_ = keyPrefixShape.GetDim(3);
    valuePrefixSSize_ = valuePrefixShape.GetDim(1);
    valuePrefixNSize_ = valuePrefixShape.GetDim(2);
    valuePrefixDSize_ = valuePrefixShape.GetDim(3);
    kvNSize_ = keyShape.GetDim(2);
    kvDSize_ = keyShape.GetDim(3);
  } else if (layOutStr == "BSH") {
    keyPrefixSSize_ = keyPrefixShape.GetDim(1);
    keyPrefixHSize_ = keyPrefixShape.GetDim(2);
    valuePrefixSSize_ = valuePrefixShape.GetDim(1);
    valuePrefixHSize_ = valuePrefixShape.GetDim(2);
    kvHSize_ = keyShape.GetDim(2);
  }

  if (layOutStr == "BSH") {
    OP_CHECK_IF((keyPrefixBSize_ == 0 || keyPrefixSSize_ == 0 || keyPrefixHSize_ == 0) ||
                (valuePrefixBSize_ == 0 || valuePrefixSSize_ == 0 || valuePrefixHSize_ == 0),
            OP_LOGE(ifaContext_->opName, "key shared prefix and value shared prefix should not have 0 axis!"),
            return false);
  } else {
    OP_CHECK_IF((keyPrefixBSize_ == 0 || keyPrefixSSize_ == 0 || keyPrefixNSize_ == 0 || keyPrefixDSize_ == 0) ||
                (valuePrefixBSize_ == 0 || valuePrefixSSize_ == 0 || valuePrefixNSize_ == 0 || valuePrefixDSize_ == 0),
            OP_LOGE(ifaContext_->opName, "key shared prefix and value shared prefix should not have 0 axis!"),
            return false);
  }

  OP_CHECK_IF((keyPrefixSSize_ < 1 || keyPrefixSSize_ > SLIMIT || valuePrefixSSize_ < 1 || valuePrefixSSize_ > SLIMIT),
              OP_LOGE(ifaContext_->opName, "key shared prefix S(%ld) and value shared prefix S(%ld) invalid, they should not smaller than 1 or greater than %ld",
              keyPrefixSSize_, valuePrefixSSize_, SLIMIT),
              return false);
  OP_CHECK_IF((keyPrefixBSize_ != 1 || valuePrefixBSize_ != 1),
              OP_LOGE(ifaContext_->opName, "key shared prefix batch num(%ld) and value shared prefix batch num(%ld) only support 1!", keyPrefixBSize_, valuePrefixBSize_),
              return false);
  if (layOutStr == "BSH") {
    // prefix H 与 normal H
    OP_CHECK_IF((keyPrefixHSize_ != kvHSize_ || valuePrefixHSize_ != kvHSize_),
                OP_LOGE(ifaContext_->opName, "key shared prefix H(%ld) and value shared prefix H(%ld) should be same with KV H(%ld)!",
                keyPrefixHSize_, valuePrefixHSize_, kvHSize_),
                return false);
  } else {
    // prefix的N D 和 kv 的N D
    OP_CHECK_IF((keyPrefixNSize_ != kvNSize_) || (keyPrefixDSize_ != kvDSize_) || (valuePrefixNSize_ != kvNSize_) || (valuePrefixDSize_ != kvDSize_),
                OP_LOGE(ifaContext_->opName, "key shared prefix N(%ld) and D(%ld) / value shared prefix N(%ld) and D(%ld) should be same with KV N(%u) and D(%u)!",
                        keyPrefixNSize_, keyPrefixDSize_, valuePrefixNSize_, valuePrefixDSize_, kvNSize_, kvDSize_),
                return false);
  }
  return true;
}

bool IFATilingV2::CheckKeyValuePrefixConsistency(const gert::Shape keyPrefixShape, const gert::Shape valuePrefixShape,
                                                 const gert::Shape keyShape)
{
  int64_t prefixKeyDim = keyPrefixShape.GetDimNum();
  int64_t prefixValueDim = valuePrefixShape.GetDimNum();
  int64_t KVDim = keyShape.GetDimNum();
  OP_CHECK_IF(((prefixKeyDim != KVDim) || (prefixValueDim != KVDim) || (prefixKeyDim != prefixValueDim)),
              OP_LOGE(ifaContext_->opName,
                      "dim num of key_shared_prefix and value_shared_prefix should be same with KV, "
                      "but key_shared_prefix dim(%ld), value_shared_prefix dim(%ld), KV dim(%ld)!",
                      prefixKeyDim, prefixValueDim, KVDim),
              return false);
  for (uint32_t i = 0; i < prefixKeyDim; i++) {
    int64_t tmpPrefixKeyDim = keyPrefixShape.GetDim(i);
    int64_t tmpPrefixValueDim = valuePrefixShape.GetDim(i);
    OP_CHECK_IF(((tmpPrefixKeyDim == 0) || (tmpPrefixValueDim == 0)),
                OP_LOGE(ifaContext_->opName,
                        "key_shared_prefix and value_shared_prefix not support empty tensor,"
                        "but key_shared_prefix[%u]:%ld, value_shared_prefix[%u]:%ld!",
                        i, tmpPrefixKeyDim, i, tmpPrefixValueDim),
                return false);
    OP_CHECK_IF(((tmpPrefixKeyDim != tmpPrefixValueDim)),
                OP_LOGE(ifaContext_->opName,
                        "shape of key_shared_prefix should be same with value_shared_prefix,"
                        "but key_shared_prefix[%u]:%ld, value_shared_prefix[%u]:%ld!",
                        i, tmpPrefixKeyDim, i, tmpPrefixValueDim),
                return false);
  }
  return true;
}

bool IFATilingV2::CheckActualSharedPrefixLen(const gert::Tensor *actualSharedPrefixLenInput,
                                             const gert::Shape keyPrefixShape, uint32_t prefixSSizeLocal)
{
  uint32_t actualPrefixlenDim = actualSharedPrefixLenInput->GetStorageShape().GetDimNum();
  OP_CHECK_IF((actualPrefixlenDim != 1),
              OP_LOGE(ifaContext_->opName, "actualSharedPrefixLen dim num(%zu) should be 1!", actualPrefixlenDim),
              return false);
  uint32_t actualPrefixlenShapeSize = actualSharedPrefixLenInput->GetStorageShape().GetShapeSize();
  OP_CHECK_IF((actualPrefixlenShapeSize != 1),
              OP_LOGE(ifaContext_->opName, "actualSharedPrefixLen length(%zu) should be 1!", actualPrefixlenShapeSize),
              return false);
  OP_CHECK_IF((actualSharedPrefixLenInput->GetData<int64_t>() == nullptr),
              OP_LOGE(ifaContext_->opName, "actualSharedPrefixLen datas is null!"), return false);
  actualSharedPrefixLen_ = actualSharedPrefixLenInput->GetData<int64_t>()[0];
  OP_CHECK_IF((actualSharedPrefixLen_ > prefixSSizeLocal) || (actualSharedPrefixLen_ < 0),
              OP_LOGE(ifaContext_->opName, "actualSharedPrefixLen(%ld) must be in range[0, %ld]!",
                      actualSharedPrefixLen_, prefixSSizeLocal),
              return false);
  return true;
}

ge::graphStatus IFATilingV2::ProcessPseShift() {
  // get pse shift data
  auto pseShiftInput = ifaContext_->pseShift.tensor;
  auto pseType = ifaContext_->pseType;
  if (pseType != nullptr) {
      pseType_ = *pseType;
      OP_CHECK_IF((pseType_ != static_cast<int64_t>(IfaPseType::PSE_OUTER_MUL_ADD_TYPE)) &&
                      (pseType_ != static_cast<int64_t>(IfaPseType::PSE_INNER_MUL_ADD_TYPE)) &&
                      (pseType_ != static_cast<int64_t>(IfaPseType::PSE_INNER_MUL_ADD_SQRT_TYPE)),
                  OP_LOGE(ifaContext_->opName, "PseType(%ld) is not support, pseType must be 0/2/3.", pseType_),
                  return ge::GRAPH_FAILED);
  }

  if (pseShiftInput == nullptr && pseType_ == static_cast<int64_t>(IfaPseType::PSE_OUTER_MUL_ADD_TYPE)) {
      return ge::GRAPH_SUCCESS;
  }

  if (pseType_ == static_cast<int64_t>(IfaPseType::PSE_INNER_MUL_ADD_TYPE) ||
      pseType_ == static_cast<int64_t>(IfaPseType::PSE_INNER_MUL_ADD_SQRT_TYPE)) {
      enableAlibiPse_ = true;
      if (!CheckAlibiPseShift()) {
          return ge::GRAPH_FAILED;
      }
      return ge::GRAPH_SUCCESS;
  }

  OP_CHECK_IF(inputLayout_ == IfaLayout::TND, OP_LOGE(ifaContext_->opName,
             "TND not support pse."), return ge::GRAPH_FAILED);

  OP_CHECK_IF(ifaContext_->pseShift.desc == nullptr, OP_LOGE(ifaContext_->opName, "Desc of pseShift tensor is null."),
              return ge::GRAPH_FAILED);
  auto pseShiftDataType = ifaContext_->pseShift.desc->GetDataType();
  switch (pseShiftDataType) {
    case ge::DT_FLOAT16:
    case ge::DT_BF16:
      OP_CHECK_IF(inputQType_ != pseShiftDataType,
                  OP_LOGE(ifaContext_->opName,
                  "Datatype of pseShift is %s, which does not match datatype of query: %s.",
                  DataTypeToString(pseShiftDataType).c_str(), DataTypeToString(inputQType_).c_str()),
                  return ge::GRAPH_FAILED);
      break;
    default:
      OP_LOGE(ifaContext_->opName,"Datatype %s is not currently supported.",
                                  DataTypeToString(pseShiftDataType).c_str());
      return ge::GRAPH_FAILED;
  }

  // check pse shift shape (B/1, N, S0, Si)
  if (!CheckPseShiftShape(pseShiftInput)) {
    return ge::GRAPH_FAILED;
  }
  pseShiftFlag_ = true;
  return ge::GRAPH_SUCCESS;
}

bool IFATilingV2::CheckPseShiftShape(const gert::Tensor* pseShiftInput)
{
  const gert::Shape pseShiftShape = pseShiftInput->GetStorageShape();
  uint32_t pseShiftDimNum = pseShiftShape.GetDimNum();
  OP_CHECK_IF(pseShiftDimNum != 4, OP_LOGE(ifaContext_->opName,
             "The dimension of pseShift must be 4, current dimension num is %u.", pseShiftDimNum),
             return false);
  pseShiftBatch_ = pseShiftShape.GetDim(PSE_SHIFT_B);
  uint32_t pseShiftN = pseShiftShape.GetDim(PSE_SHIFT_N);
  pseShiftS0_ = pseShiftShape.GetDim(PSE_SHIFT_S0);
  pseShiftS1_ = pseShiftShape.GetDim(PSE_SHIFT_S1);
  OP_CHECK_IF(sOfQuery_ == NUM1 && ((pseShiftBatch_ != NUM1 && pseShiftBatch_ != batchSize_) || (pseShiftN != numHeads_) || (pseShiftS0_ != NUM1)),
              OP_LOGE(ifaContext_->opName,
              "The shape of pseShift is (%u, %u, %u, %u), which does not match (B, N, 1, S) or (1, N, 1, S).",
              pseShiftBatch_, pseShiftN, pseShiftS0_, pseShiftS1_),
              return false);
  OP_CHECK_IF(isPFAFlag_ && ((pseShiftBatch_ != NUM1 && pseShiftBatch_ != batchSize_) || (pseShiftN != numHeads_) || (pseShiftS0_ < sOfQuery_) ||
             (pseShiftS1_ < seqSize_)), OP_LOGE(ifaContext_->opName,
             "pseShift shape must be (1 or %u, %u, >=%u, >=%u), but now it is (%u, %u, %u, %u)",
             pseShiftBatch_, pseShiftN, sOfQuery_, seqSize_, pseShiftBatch_, pseShiftN, pseShiftS0_, pseShiftS1_),
             return false);
  OP_CHECK_IF(pseShiftS1_ < seqSize_,
    OP_LOGE(ifaContext_->opName, "The shape of pseShift is (%u, %u, %u, %u), pseShiftS[%u] shouldn't be less than sMax[%u]. "
              "When page attention is enabled, sMax is the second dimension of blockTable * blockSize.",
              pseShiftBatch_, pseShiftN, pseShiftS0_, pseShiftS1_, pseShiftS1_, seqSize_),
              return false);
  return true;
}

ge::graphStatus IFATilingV2::ProcessAttenMaskSparsePFA() {
  if (!isPFAFlag_) {
    return ge::GRAPH_SUCCESS;
  }
  
  const gert::Tensor* maskShape = ifaContext_->attenMask.tensor;
  ge::DataType attenMaskType = ifaContext_->attenMask.desc ? ifaContext_->attenMask.desc->GetDataType() : ge::DT_UNDEFINED;

  bool enableMask = (maskShape != nullptr);
  bool isDefaultSparseMode = (ifaContext_->sparseMode == nullptr) || ((ifaContext_->sparseMode != nullptr) && 
    (sparseMode_ == SPARSE_MODE_NO_MASK));

  if (enableMask) {
    if (!CheckMaskTypeAndShape(maskShape, attenMaskType)) {
      return ge::GRAPH_FAILED;
    }
    attenMaskFlag_ = true;
  } else {
    attenMaskFlag_ =false;
  }

  if (isMaxWorkspace_) {
    return ge::GRAPH_SUCCESS;
  }

  if (!CheckSparseMode(isDefaultSparseMode, enableMask)) {
    return ge::GRAPH_FAILED;
  }

  if (!CheckMaskCrossover(maskShape, attenMaskType, enableMask, isDefaultSparseMode)) {
    return ge::GRAPH_FAILED;
  }

  return ge::GRAPH_SUCCESS;
}

bool IFATilingV2::CheckMaskTypeAndShape(const gert::Tensor* maskShape, ge::DataType attenMaskType) const {
  OP_CHECK_IF((maskShape->GetStorageShape().GetShapeSize() == gert::Shape::kInvalidDimValue),
    OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName, "Get the shape size of attenMask failed."),
    return false);

  std::vector<ge::DataType> allowedMaskDtypes = {ge::DT_FLOAT16, ge::DT_BOOL, ge::DT_INT8, ge::DT_UINT8};
  auto maskTypeCheck = std::find(allowedMaskDtypes.begin(), allowedMaskDtypes.end(), attenMaskType);

  OP_CHECK_IF((maskTypeCheck == allowedMaskDtypes.end()), OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName,
    "maskType(%s) should be DT_FLOAT16, DT_BOOL, DT_INT8 and DT_UINT8",
    optiling::v2::GetPfaDataTypeStr(attenMaskType).c_str()), return false);

  // When in fp16 high-precision mode, the mask type only supports bool or int8.
  OP_CHECK_IF(((inputQType_ == ge::DT_FLOAT16) && (innerPrecise_ == IFA_HIGH_PRECISION)) &&
    (attenMaskType != ge::DT_BOOL) && (attenMaskType != ge::DT_INT8) && (attenMaskType != ge::DT_UINT8),
    OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName,
      "maskType[%s] should be bool, int8 or uint8 when innerprecise_ = %u and input type is fp16.",
      optiling::v2::GetPfaDataTypeStr(attenMaskType).c_str(), innerPrecise_),
    return false);
  // When bf16, the mask type only supports bool or int8.
  OP_CHECK_IF((inputQType_ == ge::DT_BF16) && (attenMaskType != ge::DT_BOOL) &&
    (attenMaskType != ge::DT_INT8) && (attenMaskType != ge::DT_UINT8),
    OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName,
      "maskType[%s] should be bool, int8 or uint8 when input type is bfloat16", optiling::v2::GetPfaDataTypeStr(attenMaskType).c_str()),
    return false);

  return true;
}

bool IFATilingV2::CheckSparseMode(bool isDefaultSparseMode, bool enableMask) {
  bool sparseCheck = false;
  if (ifaContext_->sparseMode != nullptr) {
    sparseCheck = ((sparseMode_ != SPARSE_MODE_NO_MASK) && (sparseMode_ != SPARSE_MODE_LEFT_UP) &&
      (sparseMode_ != SPARSE_MODE_RIGHT_DOWN) && (sparseMode_ != SPARSE_MODE_ALL_MASK) &&
      (sparseMode_ != SPARSE_MODE_BAND));
    OP_CHECK_IF(
      sparseCheck, OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName, "sparse_mode = %u is out of range. Currently only 0,1,2,3,4 are supported.", sparseMode_),
      return false);
  }
  bool isBandMode = false;
  SetSparseModeData(isBandMode, enableMask, isDefaultSparseMode);

  OP_CHECK_IF((nextToken_ * (-1)) > preToken_, OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName,
    "nexttoken line should be higher than pretoken line, preTokens = %ld, nextTokens = %ld.",
    preToken_, nextToken_),
  return false);

  OP_CHECK_IF((outputType_ == ge::DT_INT8 && isBandMode && ((preToken_ < 0) || nextToken_ < 0)),
    OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName,
      "When output type is int8, sparse mode = 4, preTokens (%ld) or nextTokens (%ld) cannot be negative.",
      preToken_, nextToken_),
    return false);

  return true;
}

void IFATilingV2::SetSparseModeData(bool& isBandMode, bool enableMask, bool isDefaultSparseMode) {
  if (preToken_ > SPARSE_MODE_INT_MAX) {
    preToken_ = SPARSE_MODE_INT_MAX;
  } else if (preToken_ < -(SPARSE_MODE_INT_MAX)) {
    preToken_ = -(SPARSE_MODE_INT_MAX);
  }

  if (nextToken_ > SPARSE_MODE_INT_MAX) {
    nextToken_ = SPARSE_MODE_INT_MAX;
  } else if (nextToken_ < -(SPARSE_MODE_INT_MAX)) {
    nextToken_ = -(SPARSE_MODE_INT_MAX);
  }

  if (enableMask && (ifaContext_->sparseMode != nullptr)) {
    if (sparseMode_ == SPARSE_MODE_LEFT_UP) {
      preToken_ = SPARSE_MODE_INT_MAX;
      nextToken_ = 0;
      // Right down tokens are calculated on the kernel side.
    } else if (sparseMode_ == SPARSE_MODE_RIGHT_DOWN) {
      preToken_ = SPARSE_MODE_INT_MAX;
    } else if (sparseMode_ == SPARSE_MODE_ALL_MASK) {
      preToken_ = SPARSE_MODE_INT_MAX;
      nextToken_ = SPARSE_MODE_INT_MAX;
    } else if (sparseMode_ == SPARSE_MODE_BAND) {
      isBandMode = true;
    } 
    OP_LOGI(ifaContext_->opName, "sparseMode is %u.", sparseMode_);
  }
  if ((ifaContext_->sparseMode != nullptr) && (sparseMode_ == SPARSE_MODE_NO_MASK)) {
    // sparse mode need process attention mask when empty tensor scenes as same.
    if (!enableMask) {
      preToken_ = SPARSE_MODE_INT_MAX;
      nextToken_ = SPARSE_MODE_INT_MAX;
    }
  }
  if (isDefaultSparseMode && qPaddingSizeFlag_) {
    // For scenes with sparse mode=0 and left padding, the attention mask part is fully calculated
    preToken_ = SPARSE_MODE_INT_MAX;
    nextToken_ = SPARSE_MODE_INT_MAX;
  }
}

bool IFATilingV2::CheckMaskCrossover(const gert::Tensor* maskShape, ge::DataType attenMaskType,
  bool enableMask, bool isDefaultSparseMode) {
  if ((ifaContext_->sparseMode != nullptr) && (sparseMode_ == SPARSE_MODE_LEFT_UP || sparseMode_ == SPARSE_MODE_RIGHT_DOWN ||
    sparseMode_ == SPARSE_MODE_ALL_MASK || sparseMode_ == SPARSE_MODE_BAND)) {
    OP_CHECK_IF(
      !enableMask, OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName, "attenMask should not be null when sparse_mode is %u.", sparseMode_),
      return false);

    // When sparse=2, 3, 4, the mask type only supports bool, int8, uint8
    OP_CHECK_IF((sparseMode_ != SPARSE_MODE_ALL_MASK) && (attenMaskType != ge::DT_BOOL) &&
      (attenMaskType != ge::DT_INT8) && (attenMaskType != ge::DT_UINT8),
      OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName,
        "maskType[%s] should be bool, int8 or uint8 when sparse mode is %u.",
        optiling::v2::GetPfaDataTypeStr(attenMaskType).c_str(), sparseMode_),
      return false);
  } else if (!enableMask) {
    return true;
  }

  // FP16 mask type does not support invalid line correction.
  OP_CHECK_IF((attenMaskType == ge::DT_FLOAT16 && isRowInvalid_ != 0),
    OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName,
      "maskType[%s] should not be float16 when innerPrecise = 2 or 3",
      optiling::v2::GetPfaDataTypeStr(attenMaskType).c_str()),
    return false);
  if (!CheckMaskShapeCrossSparse(maskShape, isDefaultSparseMode)) {
    return false;
  }
  return true;
}

bool IFATilingV2::CheckMaskShape(bool isDefaultSparseMode, const gert::Tensor* maskShape, std::string& strMaskShape, bool& checkMask) {
  size_t attenMaskDim = maskShape->GetStorageShape().GetDimNum();
  int64_t attenMaskN = 1U;
  if (attenMaskDim == MASKDIM_SS) {
    if (isDefaultSparseMode || (ifaContext_->sparseMode != nullptr && sparseMode_ == SPARSE_MODE_ALL_MASK)) { // sparse 0、1时不支持二维mask
      OP_LOGE(ifaContext_->opName, "The current dimension of the mask is 2. "
        "When sparseMode is 0 or 1, the mask dimension only supports 3 and 4. "
        "Please use 3D mask \[B,QS,KVS\]\/\[1,QS,KVS\] or 4D mask \[B,1,QS,KVS\]\/\[1,1,QS,KVS\].");
      return false;
    } else {
      attenMaskQSize_ = maskShape->GetStorageShape().GetDim(NUM0);
      attenMaskKvSize_ = maskShape->GetStorageShape().GetDim(NUM1);
      strMaskShape = std::to_string(attenMaskQSize_) + ", " + std::to_string(attenMaskKvSize_);
    }
  } else if (attenMaskDim == MASKDIM_1SS_BSS) {
    attenMaskBatch_ = maskShape->GetStorageShape().GetDim(NUM0);
    attenMaskQSize_ = maskShape->GetStorageShape().GetDim(NUM1);
    attenMaskKvSize_ = maskShape->GetStorageShape().GetDim(NUM2); // 2: When the dim is 3, the second dimension is S2.
    strMaskShape = std::to_string(attenMaskBatch_) + ", " + std::to_string(attenMaskQSize_) + ", " + 
      std::to_string(attenMaskKvSize_);
  } else if (attenMaskDim == MASKDIM_11SS_B1SS) {
    attenMaskBatch_ = maskShape->GetStorageShape().GetDim(NUM0);
    attenMaskN = maskShape->GetStorageShape().GetDim(NUM1);
    attenMaskQSize_ = maskShape->GetStorageShape().GetDim(NUM2); // 2: When the dim is 4, the second dimension is S1.
    attenMaskKvSize_ = maskShape->GetStorageShape().GetDim(NUM3); // 3: When the dim is 4, the third dimension is S2.
    strMaskShape = std::to_string(attenMaskBatch_) + ", " + std::to_string(attenMaskN) + ", " + 
      std::to_string(attenMaskQSize_) + ", " + std::to_string(attenMaskKvSize_);
  } else {
    OP_LOGE(ifaContext_->opName, "attenMask dim(%zu) must be 2 or 3 or 4!", attenMaskDim);
    return false;
  }

  if (isDefaultSparseMode || (ifaContext_->sparseMode != nullptr && sparseMode_ == SPARSE_MODE_ALL_MASK)) {
    checkMask = (attenMaskQSize_ >= sOfQuery_) && (attenMaskKvSize_ >= seqSize_) && 
      (attenMaskBatch_ == NUM1 || attenMaskBatch_ == batchSize_) && (static_cast<uint32_t>(attenMaskN) == NUM1);
    if (static_cast<uint32_t>(attenMaskN) != NUM1) {
      OP_LOGE(ifaContext_->opName, "The second dimension of the 4D mask must be 1, but now it is %lld!", attenMaskN);
    }
  } else if ((ifaContext_->sparseMode != nullptr) && ((sparseMode_ == SPARSE_MODE_LEFT_UP) ||
    (sparseMode_ == SPARSE_MODE_RIGHT_DOWN) || (sparseMode_ == SPARSE_MODE_BAND))) {
    checkMask = (attenMaskBatch_ == NUM1) && (static_cast<uint32_t>(attenMaskN) == NUM1) &&
      (attenMaskQSize_ == SPARSE_OPTIMIZE_ATTENTION_SIZE) && (attenMaskKvSize_ == SPARSE_OPTIMIZE_ATTENTION_SIZE);
  }
  return true;
}

bool IFATilingV2::CheckMaskShapeCrossSparse(const gert::Tensor* maskShape, bool isDefaultSparseMode) {
  attenMaskQSize_ = 0;
  attenMaskKvSize_ = 0;
  bool checkMask = false;
  std::string strMaskShape;

  if (!CheckMaskShape(isDefaultSparseMode, maskShape, strMaskShape, checkMask)) {
    return false;
  }

  if (isDefaultSparseMode || (ifaContext_->sparseMode != nullptr && sparseMode_ == SPARSE_MODE_ALL_MASK)) {
    OP_CHECK_IF(!checkMask, OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName,
        "attenMask batch(%u) must be 1 or %u, attenMask Q_S(%u) must be larger than or equal to sQ(%u),"
        "attenMask KV_S(%u) must be larger than or equal to sK(%u), please check",
        attenMaskBatch_, batchSize_, attenMaskQSize_, sOfQuery_, attenMaskKvSize_, seqSize_),
      return false);
  }
  if ((ifaContext_->sparseMode != nullptr) && ((sparseMode_ == SPARSE_MODE_LEFT_UP) ||
    (sparseMode_ == SPARSE_MODE_RIGHT_DOWN) || (sparseMode_ == SPARSE_MODE_BAND)) && !checkMask) {
    OP_LOGE(ifaContext_->opName,
      "attenMask shape must be (2048, 2048) or (1, 2048, 2048) or (1, 1, 2048, 2048) when sparse mode = %u.", sparseMode_);
    OP_LOGE(ifaContext_->opName, "attenMask shape is (%s).", strMaskShape.c_str());
    return false;
  }
  return true;
}

ge::graphStatus IFATilingV2::ProcessAttenMask() {
  if (isPFAFlag_) {
    return ge::GRAPH_SUCCESS;
  }
  OP_CHECK_IF((sparseMode_ != SPARSE_MODE_NO_MASK),
    OP_LOGE(ifaContext_->opName, "When S of query equal to 1, sparseMode only support 0(defaultMask),"
            "but got %u.", sparseMode_),
            return ge::GRAPH_FAILED);
  auto maskShape = ifaContext_->attenMask.tensor;  // input shape = 4
  if (maskShape == nullptr) {
    attenMaskFlag_ = false;
    OP_LOGD(ifaContext_->opName, "AttenMask tensor exists, but attenMask is null.");
    return ge::GRAPH_SUCCESS;
  }

  size_t attenMaskDim = maskShape->GetStorageShape().GetDimNum();
  attenMaskBatch_ = maskShape->GetStorageShape().GetDim(NUM0);
  ge::DataType attenMaskType = ifaContext_->attenMask.desc->GetDataType();
  if (attenMaskDim == MASKDIM_SS) { // 伪量化qs=1在未生效sparse时仅支持二维mask为BS，所以此处全部拦截
    OP_LOGE(ifaContext_->opName, "The current dimension of the mask is 2. "
      "When sparseMode is 0 or 1, the mask dimension only supports 3 and 4. "
      "Please use 3D mask \[B,QS,KVS\]\/\[1,QS,KVS\] or 4D mask \[B,1,QS,KVS\]\/\[1,1,QS,KVS\].");
    return ge::GRAPH_FAILED;
  } else {
    OP_CHECK_IF(!(attenMaskBatch_ == batchSize_ || attenMaskBatch_ == 1),
      OP_LOGE(ifaContext_->opName, "BatchSize[%u] of attenMask must be equal to batchSize[%u] of query or 1, "
                "when mask is three-dimensional or four-dimensional.", attenMaskBatch_, batchSize_),
                return ge::GRAPH_FAILED);
  }

  OP_CHECK_IF((attenMaskType != ge::DT_BOOL && attenMaskType != ge::DT_INT8 && attenMaskType != ge::DT_UINT8),
    OP_LOGE(ifaContext_->opName, "Not support attenMask type %s, only support bool, int8 and uint8.", DataTypeToString(attenMaskType).c_str()),
              return ge::GRAPH_FAILED);

  uint32_t minAttenMaskSize = pageAttentionFlag_ ? sMax_ : maxActualseq_;
  attenMaskKvSize_ = maskShape->GetStorageShape().GetDim(maskShape->GetStorageShape().GetDimNum() - 1);
  attenMaskQSize_ = maskShape->GetStorageShape().GetDim(maskShape->GetStorageShape().GetDimNum() - 2);
  OP_CHECK_IF(attenMaskKvSize_ < minAttenMaskSize,
    OP_LOGE(ifaContext_->opName, "S Size[%u] of attenMask must be greater than or equal to the second dimension of blockTable * blockSize([%u]).",
              attenMaskKvSize_, minAttenMaskSize),
              return ge::GRAPH_FAILED);

  attenMaskFlag_ = true;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATilingV2::CheckActualSeqLens()
{
    // Q的actual_seq要求非空
    if (ifaContext_->actualSeqLengthsQ.tensor == nullptr) {
        OP_LOGE(ifaContext_->opName, "TND the query's actual sequence lengths should not be null!");
        return ge::GRAPH_FAILED;
    }
    actualLenQDims_ = static_cast<uint32_t>(ifaContext_->actualSeqLengthsQ.tensor->GetShapeSize());
    if (actualLenQDims_ == 0U) {
        OP_LOGE(ifaContext_->opName, "TND actualLenQDims_ is 0!");
        return ge::GRAPH_FAILED;
    }

    // KV的actual_seq要求非空
    if (ifaContext_->actualSeqLengths.tensor == nullptr) {
        OP_LOGE(ifaContext_->opName, "TND the key/value's actual sequence lengths should not be null!");
        return ge::GRAPH_FAILED;
    }
    actualLenDims_ = ifaContext_->actualSeqLengths.tensor->GetShapeSize();
    if (actualLenDims_ == 0U) {
        OP_LOGE(ifaContext_->opName, "TND actualLenDims_ is 0!");
        return ge::GRAPH_FAILED;
    }
    if (!pageAttentionFlag_ && (actualLenQDims_ != actualLenDims_)) {
        OP_LOGE(ifaContext_->opName, "When layout is TND and page attention is not enable, "
                "the length of actualSequenceLengthQ (%u) and actualSequenceLengthKV (%u) must be equal",
                actualLenQDims_, actualLenDims_);
        return ge::GRAPH_FAILED;
    }
    // layout为TND, kv PA管理场景 actualSequenceLengthKV size可以为1或者>=B
    if (pageAttentionFlag_ && (actualLenDims_ != 1 && actualLenDims_ < actualLenQDims_)) {
        OP_LOGE(ifaContext_->opName, "When layout is TND and page attention is enable, "
                "the length of actualSequenceLengthKV (%u) should be greater than or equal to "
                "the length of actualSequenceLengthQ(%u) or equal to 1 ",
                actualLenDims_, actualLenQDims_);
        return ge::GRAPH_FAILED;
    }
    int64_t lastActSeq = 0;
    int64_t lastActSeqKV = 0;
    int64_t curActSeqKV = 0;
    for (uint32_t i = 0; i < actualLenQDims_; i++) {
        int64_t curActSeq = ifaContext_->actualSeqLengthsQ.tensor->GetData<int64_t>()[i];
        if (pageAttentionFlag_ && actualLenDims_ == 1) {
            curActSeqKV = ifaContext_->actualSeqLengths.tensor->GetData<int64_t>()[0];
        } else {
            curActSeqKV = ifaContext_->actualSeqLengths.tensor->GetData<int64_t>()[i];
        }
        if (curActSeq < 0) {
            OP_LOGE(ifaContext_->opName, "actualSeqLengths[%u] = %ld, should >= 0", i, curActSeq);
            return ge::GRAPH_FAILED;
        }
        if (i >= 1U && curActSeq < lastActSeq) {
            OP_LOGE(ifaContext_->opName, "When layout is TND, actualSeqLengths must not be decreasing, but actSeqLen[%u]=%ld is smaller than actSeqLen[%u]=%ld",
                i, curActSeq, i - 1U, lastActSeq);
            return ge::GRAPH_FAILED;
        }
        if (curActSeqKV < 0) {
            OP_LOGE(ifaContext_->opName, "actualSeqLengthsKv[%u] = %ld, should >= 0", i, curActSeqKV);
            return ge::GRAPH_FAILED;
        }
        if (!pageAttentionFlag_ && i >= 1U && curActSeqKV < lastActSeqKV) {
            OP_LOGE(ifaContext_->opName, "When layout is TND, actualSeqLengthsKv must not be decreasing, but actSeqLenKV[%u]=%ld is smaller than actSeqLenKV[%u]=%ld",
                i, curActSeqKV, i - 1U, lastActSeqKV);
            return ge::GRAPH_FAILED;
        }    
        lastActSeq = curActSeq;
        lastActSeqKV = curActSeqKV;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATilingV2::ProcessActualSeqLen() {
  if (isMaxWorkspace_) {
    actualSeqLenFlag_ = true;
    maxActualseq_ = sMax_;
    actualLenDims_ = 1;
    return ge::GRAPH_SUCCESS;
  }
  if (inputLayout_ == IfaLayout::TND) {
    actualSeqLenFlag_ = true;
    actualSeqLenQFlag_ = true;
    maxActualseq_ = sMax_;
  } else {
    if (ifaContext_->actualSeqLengthsQ.tensor != nullptr) {
      const gert::Tensor* actSeqLen = ifaContext_->actualSeqLengthsQ.tensor;
      actualLenQDims_ = actSeqLen->GetShapeSize();
      actualSeqLenQFlag_ = (actualLenQDims_ != 0 && actSeqLen->GetData<int64_t>() != nullptr);
      if (actualSeqLenQFlag_) {
      OP_CHECK_IF(actualLenQDims_ != NUM1 && actualLenQDims_ < batchSize_,
        OP_LOGE(ifaContext_->opName, "ActualSeqLengths size[%d] of query should be greater than or equal to query batch[%u] or equal to 1.",
        actualLenQDims_, batchSize_),
        return ge::GRAPH_FAILED);
      uint32_t actSeqLengthSize = std::min(actualLenQDims_, batchSize_);
      int64_t castSOfQuery_ = static_cast<int64_t>(sOfQuery_);
      for (uint32_t i = 0; i < actSeqLengthSize; ++i) {
        int64_t actSeqTmp = actSeqLen->GetData<int64_t>()[i];
        OP_CHECK_IF(actSeqTmp < 0 || actSeqTmp > sOfQuery_, OP_LOGE(ifaContext_->opName,
          "ActualSeqLengthsQ[%u](%ld) must be in range[0, %u]!", i, actSeqTmp, sOfQuery_),
          return ge::GRAPH_FAILED);
      if (actSeqTmp != castSOfQuery_) {
          needInit_ = true;
          }
        }
      }
    }
  }
  if (ifaContext_->actualSeqLengths.tensor == nullptr) {
    maxActualseq_ = sMax_;

    // pa场景必须带actual_seq_lens；第1次tiling调用时(isMaxWorkspace为true) actualSeqLengthsKv会被强制置None，需要跳过校验
    OP_CHECK_IF(pageAttentionFlag_ && (!isMaxWorkspace_),
               OP_LOGE(ifaContext_->opName, "When page attention scene, actualSeqLengthsKv must exist."),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
  }

  actualSeqLenFlag_ = true;
  actualLenDims_ = ifaContext_->actualSeqLengths.tensor->GetShapeSize();
  if (actualLenDims_ == 0) {
      // pa场景必须带actual_seq_lens
      if (pageAttentionFlag_) {
        OP_LOGE(ifaContext_->opName, "When page attention scene, size of actualSeqLengths can not be zero.");
        return ge::GRAPH_FAILED;
      }
      maxActualseq_ = sMax_;
      return ge::GRAPH_SUCCESS;
  }

  OP_CHECK_IF(actualLenDims_ != NUM1 && actualLenDims_ < batchSize_,
             OP_LOGE(ifaContext_->opName, "ActualSeqLengthsKv size[%u] should be greater than or equal to query batch[%u] or equal to 1.",
             actualLenDims_, batchSize_),
             return ge::GRAPH_FAILED);

  actualLenDims_ = std::min(actualLenDims_, batchSize_);

  const int64_t* actualLenData = ifaContext_->actualSeqLengths.tensor->GetData<int64_t>();
  if (actualLenData != nullptr) {
    uint32_t loop = ((actualLenDims_ == NUM1) && (kvListSeqLens_.size() == NUM1)) ? 1 : batchSize_;
    for (uint32_t i = 0; i < loop; i++) {
      int64_t actLen = (actualLenDims_ == NUM1) ? actualLenData[0] : actualLenData[i];
      if (inputLayout_ == IfaLayout::TND && i > 0 && !pageAttentionFlag_) {
        actLen -= actualLenData[i - 1];
      }
      OP_CHECK_IF(actLen < 0,  //actulSeqLengths必须大于0
               OP_LOGE(ifaContext_->opName, "ActualSeqLengthsKv must be greater than or equal to 0."),
               return ge::GRAPH_FAILED);
      if (!pageAttentionFlag_) {
        uint32_t seqSize = (kvListSeqLens_.size() == NUM1) ? kvListSeqLens_[0] : kvListSeqLens_[i];
        OP_CHECK_IF(
          static_cast<uint32_t>(actLen) > seqSize, OP_LOGE(ifaContext_->opName, "ActualSeqLengthsKv[%u](%ld) cannot be greater than seqLength(%u) in input key.", i, actLen, seqSize),
                   return ge::GRAPH_FAILED);
      }
      if (pageAttentionFlag_) {
        needBlockNum_ += (static_cast<uint64_t>(actLen) + blockSize_ - 1U) / blockSize_;
      }
      maxActualseq_ = maxActualseq_ < static_cast<uint32_t>(actLen) ? static_cast<uint32_t>(actLen) : maxActualseq_;
      if (actLen == 0 && needInitfaRun_ == false) {
        needInitfaRun_ = true;
      }

      if (actLen < sOfQuery_) { // 合轴校验actseqlen
          actualSeqLenUnequal_ = true;
      }
    }
  } else {
    // pa场景必须带actual_seq_lens
    if (pageAttentionFlag_ && (!isMaxWorkspace_)) {
      OP_LOGE(ifaContext_->opName, "When page attention scene, data of actualSeqLengthsKv can not be null.");
      return ge::GRAPH_FAILED;
    }
    maxActualseq_ = sMax_;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATilingV2::CheckQuant2Shape(const gert::Shape &inputParaShape) const {
  auto headsize = headDim_; // D
  auto headnum = numHeads_; // Q's N
  gert::Shape expectParamShapeBNSD = gert::Shape({1, headnum, 1, headsize});
  gert::Shape expectParamShapeBNSD_2 = gert::Shape({headnum, 1, headsize});
  gert::Shape expectParamShapeBNSD_3 = gert::Shape({headnum, headsize});
  gert::Shape expectParamShapeBSND = gert::Shape({1, 1, headnum, headsize});
  gert::Shape expectParamShapeBSND_2 = gert::Shape({1, headnum, headsize});
  gert::Shape expectParamShapeBSND_3 = gert::Shape({headnum, headsize});
  gert::Shape expectParamShapeBH = gert::Shape({1, headnum * headsize});
  gert::Shape expectParamShapeBH_2 = gert::Shape({1, 1, headnum * headsize});
  gert::Shape expectParamShapeBH_3 = gert::Shape({headnum * headsize});

  bool validShape = (inputParaShape == expectParamShapeBNSD) || (inputParaShape == expectParamShapeBSND) ||
      (inputParaShape == expectParamShapeBH) || (inputParaShape == expectParamShapeBNSD_2) ||
      (inputParaShape == expectParamShapeBSND_2) || (inputParaShape == expectParamShapeBH_2) ||
      (inputParaShape == expectParamShapeBNSD_3) || (inputParaShape == expectParamShapeBSND_3) ||
      (inputParaShape == expectParamShapeBH_3);

  if (!validShape && inputParaShape.GetDimNum() == DIM_BNSD) {
    OP_LOGE(ifaContext_->opName,
              "The shape of postquant parameter[%ld, %ld, %ld, %ld] is not expected shape. "
              "Expect [1, %u, 1, %u] or [1, 1, %u, %u].",
              inputParaShape.GetDim(BNSD_B_IDX), inputParaShape.GetDim(BNSD_N_IDX), inputParaShape.GetDim(BNSD_S_IDX),
              inputParaShape.GetDim(BNSD_D_IDX), headnum, headsize, headnum, headsize);
    return ge::GRAPH_FAILED;
  }

  if (!validShape && inputParaShape.GetDimNum() == NUM3) { // dim is 3
    OP_LOGE(ifaContext_->opName,
              "The shape of postquant parameter[%ld, %ld, %ld] is not expected shape. "
              "Expect [%u, 1, %u], [1, %u, %u] or [1, 1, %u].",
              inputParaShape.GetDim(BNSD_B_IDX), inputParaShape.GetDim(BNSD_N_IDX), inputParaShape.GetDim(BNSD_S_IDX),
              headnum, headsize, headnum, headsize, headnum * headsize);
    return ge::GRAPH_FAILED;
  }

  if (!validShape && inputParaShape.GetDimNum() == DIM_BH) {
    OP_LOGE(ifaContext_->opName, "The shape of postquant parameter[%ld, %ld] is not expected[1, %u] or [%u, %u].",
              inputParaShape.GetDim(BH_B_IDX), inputParaShape.GetDim(BH_H_IDX), headnum * headsize, headnum, headsize);
    return ge::GRAPH_FAILED;
  }

  if (!validShape && inputParaShape.GetDimNum() == NUM1) {
    OP_LOGE(ifaContext_->opName, "The shape of postquant parameter[%ld] is not expected[%u].",
              inputParaShape.GetDim(BH_B_IDX), headnum * headsize);
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATilingV2::ProcessQuant2Dtype() const {
  if (outputType_ == ge::DT_INT8) {
    OP_CHECK_IF(ifaContext_->quantScale2.tensor == nullptr,
               OP_LOGE(ifaContext_->opName,
               "Output datatype is int8, but input tensor quantScale2 is null."),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->quantScale2.desc == nullptr,
               OP_LOGE(ifaContext_->opName, "Desc of quantScale2 input tensor is null."),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->quantScale2.desc->GetDataType() != ge::DT_BF16 &&
               ifaContext_->quantScale2.desc->GetDataType() != ge::DT_FLOAT,
               OP_LOGE(ifaContext_->opName, "QuantScale2 type should be bf16 or fp32."),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->quantOffset2.desc != nullptr &&
               ifaContext_->quantScale2.desc->GetDataType() != ifaContext_->quantOffset2.desc->GetDataType(),
               OP_LOGE(ifaContext_->opName, "QuantScale2 datatype[%d] and quantOffset2 datatype[%d] are not same.",
                         ifaContext_->quantScale2.desc->GetDataType(), ifaContext_->quantOffset2.desc->GetDataType()),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputQType_ != ge::DT_BF16 && ifaContext_->quantScale2.desc->GetDataType() == ge::DT_BF16,
               OP_LOGE(ifaContext_->opName, "QuantScale2 and quantOffset2 support bf16 only when inputQ type is bf16."),
               return ge::GRAPH_FAILED);
    if (ifaContext_->quantScale2.desc->GetDataType() == ge::DT_BF16) {
      tilingData_->outputParams.set_isOutQuantTypeBf16(1);
    }
  } else {
    OP_CHECK_IF(ifaContext_->quantScale2.tensor != nullptr,
              OP_LOGE(ifaContext_->opName, "AttentionOut datatype is not int8, but quantScale2 exists."),
              return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->quantOffset2.tensor != nullptr,
              OP_LOGE(ifaContext_->opName, "AttentionOut datatype is not int8, but quantOffset2 exists."),
              return ge::GRAPH_FAILED);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATilingV2::ProcessQuant2Attribute(const gert::Tensor *qtScale2) {
  const ge::DataType quantScale2Type = qtScale2->GetDataType();
  int64_t quantScale2ShapeSize = qtScale2->GetShapeSize();
  size_t quantScale2Dim = qtScale2->GetStorageShape().GetDimNum();
  OP_CHECK_IF((quantScale2Type != ge::DT_BF16) && (quantScale2Type != ge::DT_FLOAT),
      OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName, "post quant scale dtype(%s) only support bf16 and fp32.", optiling::v2::GetPfaDataTypeStr(quantScale2Type).c_str()),
      return ge::GRAPH_FAILED);

  // input dtype verification
  if (inputQType_ == ge::DT_BF16) {
    OP_CHECK_IF(quantScale2Type != ge::DT_FLOAT && quantScale2Type != ge::DT_BF16,
        OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName,
        "invalid post quant scale dtype(%s), when q is %s, only support float32 and bf16",
        optiling::v2::GetPfaDataTypeStr(quantScale2Type).c_str(), optiling::v2::GetPfaDataTypeStr(inputQType_).c_str()),
    return ge::GRAPH_FAILED);
  } else {
    OP_CHECK_IF(quantScale2Type != ge::DT_FLOAT,
        OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName,
        "invalid post quant scale dtype(%s), when q is %s, only support float32",
        optiling::v2::GetPfaDataTypeStr(quantScale2Type).c_str(), optiling::v2::GetPfaDataTypeStr(inputQType_).c_str()),
        return ge::GRAPH_FAILED);
  }
  if (quantScale2Type == ge::DT_BF16) {
    isPostQuantBF16_ = true;
  }

  // per-tensor or per-channel verification
  uint64_t quantScale2ShapeSizePerChannel = static_cast<uint64_t>(numHeads_) * static_cast<uint64_t>(headDim_);
  std::string layoutString = ifaContext_->layOut;
  bool isSupportedLayout = layoutString == "BSH" || layoutString == "BSND" || layoutString == "BNSD" || layoutString == "BNSD_BSND";

  if (quantScale2Dim == 1) {
      if (static_cast<uint64_t>(quantScale2ShapeSize) == quantScale2ShapeSizePerChannel && isSupportedLayout) {
        // per-channel quant scale/offset shape is [H].
        isPostQuantPerChnl_ = true;
      } else {
        OP_CHECK_IF((static_cast<uint64_t>(quantScale2ShapeSize) != 1U),
            OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName,
                "For post quant per-tensor, quantScale2/quantOffset2 only support [1], now is [%d]", quantScale2ShapeSize),
            return ge::GRAPH_FAILED);
      }
  } else {
      if (isSupportedLayout) {
          OP_CHECK_IF((static_cast<uint64_t>(quantScale2ShapeSize) != quantScale2ShapeSizePerChannel),
                      OP_LOGE(ifaContext_->opName,
                              "For post quant per-channel,  when layout is %s, quantScale2/quantOffset2 dim multiply "
                              "result only support qN * vD(%u * %u = %lu), now is (%ld).",
                              layoutString.c_str(), numHeads_, headDim_, quantScale2ShapeSizePerChannel,
                              quantScale2ShapeSize),
                      return ge::GRAPH_FAILED);
      } else {
          OP_CHECK_IF(qtScale2->GetStorageShape() != gert::Shape({numHeads_, headDim_}),
                      OP_LOGE(ifaContext_->opName,
                              "For post quant per-channel, when layout is %s, "
                              "quantScale2/quantOffset2 expect shape is [%u, %u].",
                              layoutString.c_str(), numHeads_, headDim_),
                      return ge::GRAPH_FAILED);
      }
      isPostQuantPerChnl_ = true;
  }

  return ge::GRAPH_SUCCESS;
} 

ge::graphStatus IFATilingV2::ProcessQuant2() {
  if (outputType_ == ge::DT_BF16 || outputType_ == ge::DT_FLOAT16) {
    return ge::GRAPH_SUCCESS;
  }
  enablePostQuant_ = true;
  OP_CHECK_IF(outputType_ != ge::DT_INT8 &&
              outputType_ != ge::DT_FLOAT8_E4M3FN && outputType_ != ge::DT_HIFLOAT8,
              OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName,
              "invalid output type [%s], only support int8, fp8_e4m3fn_t, hifloat8_t",
              optiling::v2::GetPfaDataTypeStr(outputType_).c_str()),
              return ge::GRAPH_FAILED);
              
  // Basic verification: quantScale2 must be inputted and not an empty tensor
  auto qtScale2 = ifaContext_->quantScale2.tensor;
  OP_CHECK_IF(qtScale2 == nullptr, 
      OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName,
      "quant_scale2_shape is nullptr in post quant scenario."),
      return ge::GRAPH_FAILED);

  const ge::DataType quantScale2Type = qtScale2->GetDataType();
  size_t quantScale2Dim = qtScale2->GetStorageShape().GetDimNum();
  int64_t quantScale2ShapeSize = qtScale2->GetShapeSize();
  OP_CHECK_IF(quantScale2ShapeSize <= 0, OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName,
      "quant_scale2 is empty tensor in post quant scenario."),
      return ge::GRAPH_FAILED);
  
  // offset verification
  auto qtOffset2 = ifaContext_->quantOffset2.tensor;
  if (qtOffset2 != nullptr) {
    const ge::DataType quantOffset2Type = qtOffset2->GetDataType();
    OP_CHECK_IF(quantScale2Type != quantOffset2Type,
      OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName,
          "post quant scale dtype(%s) and offset dtype(%s) must be consistent.",
          optiling::v2::GetPfaDataTypeStr(quantScale2Type).c_str(), optiling::v2::GetPfaDataTypeStr(quantOffset2Type).c_str()),
          return ge::GRAPH_FAILED);

    OP_CHECK_IF(qtScale2->GetStorageShape() != qtOffset2->GetStorageShape(),
                OP_LOGE(ifaContext_->opName, "quantScale2 and quantOffset2 should have same shape."),
                return ge::GRAPH_FAILED);
  }
  OP_CHECK_IF(ProcessQuant2Attribute(qtScale2) != ge::GRAPH_SUCCESS,
      OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName, "post quant attribute process failed!"),
      return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATilingV2::CheckKVAntiQuantPerChannel(const gert::Shape& inputParaShape) const {
  std::string layOutStr = ifaContext_->layOut;
  gert::Shape expectParamShapeBNSD = gert::Shape({antiquantNum_, numKvHeads_, 1, headDim_});
  gert::Shape expectParamShapeBSNDType1 = gert::Shape({antiquantNum_, 1, numKvHeads_, headDim_});
  gert::Shape expectParamShapeBSNDType2 = gert::Shape({antiquantNum_, numKvHeads_, headDim_});
  gert::Shape expectParamShapeBH = gert::Shape({antiquantNum_, numKvHeads_ * headDim_});
  if (!kvAntiParamSplitFlag_ || sOfQuery_ == NUM1) {
    bool validOffsetShape = true;
    if (isPFAFlag_) {
      validOffsetShape = (inputParaShape == expectParamShapeBNSD) ||
                         (inputParaShape == expectParamShapeBSNDType2) || (inputParaShape == expectParamShapeBH);
    } else {
      validOffsetShape = (inputParaShape == expectParamShapeBNSD) || (inputParaShape == expectParamShapeBSNDType1) ||
                         (inputParaShape == expectParamShapeBSNDType2) || (inputParaShape == expectParamShapeBH);
    }

    OP_CHECK_IF((!validOffsetShape && inputParaShape.GetDimNum() == DIM_PER_CHANNEL_BNSD),
      OP_LOGE(ifaContext_->opName, "The shape of antiquant parameter[%ld, %ld, %ld, %ld] is not expected. "
        "Expect [%u, %u, 1, %u] or [%u, %u, %u] or [%u, %u] when per_channel mode.", inputParaShape.GetDim(BNSD_B_IDX),
        inputParaShape.GetDim(BNSD_N_IDX), inputParaShape.GetDim(BNSD_S_IDX), inputParaShape.GetDim(BNSD_D_IDX),
        antiquantNum_, numKvHeads_, headDim_, antiquantNum_, numKvHeads_, headDim_, antiquantNum_, numKvHeads_ * headDim_),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((!validOffsetShape && inputParaShape.GetDimNum() == DIM_PER_CHANNEL_BSND),
      OP_LOGE(ifaContext_->opName, "The shape of antiquant parameter[%ld, %ld, %ld] is not expected. "
        "Expect [%u, %u, 1, %u] or [%u, %u, %u] or [%u, %u] when per_channel mode.", inputParaShape.GetDim(BND_B_IDX),
        inputParaShape.GetDim(BND_N_IDX), inputParaShape.GetDim(BND_D_IDX), antiquantNum_, numKvHeads_, headDim_,
        antiquantNum_, numKvHeads_, headDim_, antiquantNum_, numKvHeads_ * headDim_),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((!validOffsetShape && inputParaShape.GetDimNum() == DIM_BH),
      OP_LOGE(ifaContext_->opName, "The shape of antiquant parameter[%ld, %ld] is not expected. "
        "Expect [%u, %u, 1, %u] or [%u, %u, %u] or [%u, %u] when per_channel mode.", inputParaShape.GetDim(BH_B_IDX),
        inputParaShape.GetDim(BH_H_IDX), antiquantNum_, numKvHeads_, headDim_, antiquantNum_, numKvHeads_, headDim_,
        antiquantNum_, numKvHeads_ * headDim_),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(!validOffsetShape,
      OP_LOGE(ifaContext_->opName, "The layout[%lu] does not match the dimension of antiquant, " 
                "when per_channel mode, it should be 2, 3 or 4.",
                inputParaShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    } else {
      gert::Shape expectParamShapeN1D = gert::Shape({numKvHeads_, 1, headDim_});
      gert::Shape expectParamShapeND = gert::Shape({numKvHeads_, headDim_});
      gert::Shape expectParamShapeH = gert::Shape({numKvHeads_ * headDim_});
      bool validOffsetShape = (inputParaShape == expectParamShapeN1D) || (inputParaShape == expectParamShapeND) ||
                              (inputParaShape == expectParamShapeH) || (inputParaShape == expectParamShapeBNSD) ||
                              (inputParaShape == expectParamShapeBSNDType2) || (inputParaShape == expectParamShapeBH);
      OP_CHECK_IF((!validOffsetShape && inputParaShape.GetDimNum() == DIM_PER_CHANNEL_BNSD), // 1N1D
        OP_LOGE(ifaContext_->opName, "The shape of antiquant parameter[%ld, %ld, %ld, %ld] is not expected. "
          "Expect [1, %u, 1, %u] or [1, %u, %u] or [1, %u] or [%u, 1, %u] or [%u, %u] or [%u] when per_channel mode.",
          inputParaShape.GetDim(BNSD_B_IDX), inputParaShape.GetDim(BNSD_N_IDX), inputParaShape.GetDim(BNSD_S_IDX),
          inputParaShape.GetDim(BNSD_D_IDX), numKvHeads_, headDim_, numKvHeads_, headDim_, numKvHeads_ * headDim_,
          numKvHeads_, headDim_, numKvHeads_, headDim_, numKvHeads_ * headDim_),
          return ge::GRAPH_FAILED);
      OP_CHECK_IF((!validOffsetShape && inputParaShape.GetDimNum() == DIM_PER_CHANNEL_BSND), // N1D
        OP_LOGE(ifaContext_->opName, "The shape of antiquant parameter[%ld, %ld, %ld] is not expected. "
          "Expect [1, %u, 1, %u] or [1, %u, %u] or [1, %u] or [%u, 1, %u] or [%u, %u] or [%u] when per_channel mode.",
          inputParaShape.GetDim(BND_B_IDX), inputParaShape.GetDim(BND_N_IDX), inputParaShape.GetDim(BND_D_IDX),
          numKvHeads_, headDim_, numKvHeads_, headDim_, numKvHeads_ * headDim_, numKvHeads_, headDim_, numKvHeads_,
          headDim_, numKvHeads_ * headDim_),
          return ge::GRAPH_FAILED);
      OP_CHECK_IF((!validOffsetShape && inputParaShape.GetDimNum() == DIM_BH), // ND or 1H
        OP_LOGE(ifaContext_->opName, "The shape of antiquant parameter[%ld, %ld] is not expected. "
          "Expect [1, %u, 1, %u] or [1, %u, %u] or [1, %u] or [%u, 1, %u] or [%u, %u] or [%u] when per_channel mode.",
          inputParaShape.GetDim(BH_B_IDX), inputParaShape.GetDim(BH_H_IDX),
          numKvHeads_, headDim_, numKvHeads_, headDim_, numKvHeads_ * headDim_, numKvHeads_, headDim_, numKvHeads_,
          headDim_, numKvHeads_ * headDim_),
          return ge::GRAPH_FAILED);
      OP_CHECK_IF(!validOffsetShape,
        OP_LOGE(ifaContext_->opName, "The layout[%lu] does not match the dimension of antiquant when per_channel mode, it should be 1, 2, 3 or 4.",
                  inputParaShape.GetDimNum()),
                  return ge::GRAPH_FAILED);
    }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATilingV2::CheckKVAntiQuantShapePA(const gert::Shape &inputParaShape) const {
  if (antiquantPerHeadFlag_ != NUM0) { // per-token-head, [block_num, kv_head_num, block_size]
    OP_CHECK_IF((inputParaShape.GetDim(NUM0) != totalBlockNum_),
        OP_LOGE(ifaContext_->opName,
            "The dimension 0 of antiquant parameter should be %u instead of the current %ld.",
            totalBlockNum_, inputParaShape.GetDim(NUM0)),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (inputParaShape.GetDim(NUM1) != numKvHeads_),
        OP_LOGE(ifaContext_->opName,
            "The dimension 1 of antiquant parameter should be %u instead of the current %ld.",
            numKvHeads_, inputParaShape.GetDim(NUM1)),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (inputParaShape.GetDim(NUM2) != blockSize_),
        OP_LOGE(ifaContext_->opName,
            "The dimension 2 of antiquant parameter should be %u instead of the current %ld.",
            blockSize_, inputParaShape.GetDim(NUM2)),
        return ge::GRAPH_FAILED);
  } else { // per-token, [block_num, block_size]
      OP_CHECK_IF((inputParaShape.GetDim(NUM0) != totalBlockNum_),
          OP_LOGE(ifaContext_->opName,
              "The dimension 0 of antiquant parameter should be %u instead of the current %ld.",
              totalBlockNum_, inputParaShape.GetDim(NUM0)),
          return ge::GRAPH_FAILED);
      OP_CHECK_IF(
          (inputParaShape.GetDim(NUM1) != blockSize_),
          OP_LOGE(ifaContext_->opName,
              "The dimension 1 of antiquant parameter should be %u instead of the current %ld.",
              blockSize_, inputParaShape.GetDim(NUM1)),
          return ge::GRAPH_FAILED);
  }
  return ge::GRAPH_SUCCESS;
}


ge::graphStatus IFATilingV2::CheckKVAntiQuantParaShapeLegal(const int64_t antiquantMode, const gert::Shape &inputParaShape) {
  if (kvAntiParamSplitFlag_) {
    antiquantNum_ = NUM1;
  }
  if (antiquantMode == PER_CHANNEL_MODE) { // per-channel or per-tensor
    if (inputParaShape.GetDimNum() == DIM_PER_TENSOR) {
      gert::Shape expectPerTensorShape = gert::Shape({antiquantNum_});
      if (inputParaShape == expectPerTensorShape) {
        antiquantPerTensorFlag_ = NUM1;
      } else {
        gert::Shape expectPerChannelShape = gert::Shape({numKvHeads_ * headDim_});
        if (isPFAFlag_ && inputParaShape == expectPerChannelShape) { // PFA PerChannel可能输入shape为H
          return ge::GRAPH_SUCCESS;
        } else if (isPFAFlag_ && inputParaShape != expectPerChannelShape) {
          OP_LOGE(ifaContext_->opName, "The shape of antiquant parameter[%ld] is not expected."
            "Expect[%u] when per-tensor mode or expect[%u] when per-channel mode.",
            inputParaShape.GetDim(BH_B_IDX), antiquantNum_, numKvHeads_ * headDim_);
            return ge::GRAPH_FAILED;
        } else {
          OP_LOGE(ifaContext_->opName, "The shape of antiquant parameter[%ld] is not expected. Expect[%u] when per-tensor mode.",
            inputParaShape.GetDim(BH_B_IDX), antiquantNum_);
            return ge::GRAPH_FAILED;
        }
      }
    } else {
      return CheckKVAntiQuantPerChannel(inputParaShape);
    }
  } else if (antiquantMode == PER_TOKEN_GROUP_MODE) {
    OP_CHECK_IF(((inputParaShape.GetDimNum() != NUM5)),
      OP_LOGE(ifaContext_->opName, "The dimension[%lu] of antiquant is illegal, it should be 5 when per_token_group mode.",
                inputParaShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    if (enableKVPrefix_) {
      antiquantParaSeqSize_ = inputParaShape.GetDim(NUM3);
      gert::Shape expectParamShape = gert::Shape({antiquantNum_, batchSize_, numKvHeads_, antiquantParaSeqSize_, headDim_ / NUM32});
      OP_CHECK_IF(inputParaShape != expectParamShape,
                  OP_LOGE(ifaContext_->opName,
                          "The shape of antiquant parameter is [%ld, %ld, %ld, %ld, %ld], "
                          "but [%u, %u, %u, %u, %u] is expected when per_token_group mode.",
                          inputParaShape.GetDim(NUM0), inputParaShape.GetDim(NUM1), inputParaShape.GetDim(NUM2),
                          inputParaShape.GetDim(NUM3), inputParaShape.GetDim(NUM4), antiquantNum_, batchSize_,
                          numKvHeads_, antiquantParaSeqSize_, headDim_ / NUM32),
                  return ge::GRAPH_FAILED);
    } else {
      gert::Shape expectParamShape = gert::Shape({antiquantNum_, batchSize_, numKvHeads_, seqSize_, headDim_ / NUM32});
      OP_CHECK_IF(inputParaShape != expectParamShape,
                  OP_LOGE(ifaContext_->opName,
                          "The shape of antiquant parameter is [%ld, %ld, %ld, %ld, %ld], "
                          "but [%u, %u, %u, %u, %u] is expected when per_token_group mode.",
                          inputParaShape.GetDim(NUM0), inputParaShape.GetDim(NUM1), inputParaShape.GetDim(NUM2),
                          inputParaShape.GetDim(NUM3), inputParaShape.GetDim(NUM4), antiquantNum_, batchSize_,
                          numKvHeads_, seqSize_, headDim_ / NUM32),
                  return ge::GRAPH_FAILED);
    }
  } else if (antiquantMode == PER_TOKEN_MODE) {
    //pertoken kv分离允许1BS/BS
    OP_CHECK_IF(((inputParaShape.GetDimNum() != NUM3) && !(kvAntiParamSplitFlag_ && inputParaShape.GetDimNum() == NUM2)),
      OP_LOGE(ifaContext_->opName, "The dimension[%lu] of antiquant is illegal, it should be 2 or 3 "
                "when split mode of per_token mode, or should be 3 when not split mode of per_token mode.",
                inputParaShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    // [2/1, B, S]
    OP_CHECK_IF(inputParaShape.GetDimNum() == NUM3 && (inputParaShape.GetDim(NUM0) != antiquantNum_ ||
               inputParaShape.GetDim(NUM1) != batchSize_ || inputParaShape.GetDim(NUM2) < seqSize_),
               OP_LOGE(ifaContext_->opName, "The shape of antiquant parameter[%ld, %ld, %ld] is not expected. Expect[%u, %u, %u] when per_token mode. "
                 "And antiquant seq size %ld should be equal to or greater than %u.",
                 inputParaShape.GetDim(NUM0), inputParaShape.GetDim(NUM1), inputParaShape.GetDim(NUM2),
                 antiquantNum_, batchSize_, seqSize_, inputParaShape.GetDim(NUM2), seqSize_),
               return ge::GRAPH_FAILED);
    // [B, S]
    OP_CHECK_IF(inputParaShape.GetDimNum() == NUM2 && (inputParaShape.GetDim(NUM0) != batchSize_ || inputParaShape.GetDim(NUM1) < seqSize_),
               OP_LOGE(ifaContext_->opName, "The shape of antiquant parameter[%ld, %ld] is not expected. Expect[%u, %u] when per_token mode. "
                 "And antiquant seq size %ld should be equal to or greater than %u.",
                 inputParaShape.GetDim(NUM0), inputParaShape.GetDim(NUM1),
                 batchSize_, seqSize_, inputParaShape.GetDim(NUM1), seqSize_),
               return ge::GRAPH_FAILED);
    antiquantParaSeqSize_ = inputParaShape.GetDimNum() == NUM3 ? inputParaShape.GetDim(NUM2) : inputParaShape.GetDim(NUM1);
  } else if (antiquantMode == PER_TOKEN_HEAD_MODE || antiquantMode == PER_TENSOR_HEAD_MODE) {
    antiquantPerHeadFlag_ = NUM1;
    return CheckKVAntiQuantPerHead(inputParaShape);
  } else if (antiquantMode == PER_TOKEN_PA_MODE) {
    antiquantParamsInPageAttentionFlag_ = true;
    OP_CHECK_IF((inputParaShape.GetDimNum() != DIM_BB),
      OP_LOGE(ifaContext_->opName, "The dimension of antiquant should be 2 instead of the current %lu, when antiquant mode is 4.",
                inputParaShape.GetDimNum()),
                return ge::GRAPH_FAILED);
  } else if (antiquantMode == PER_TOKEN_HEAD_PA_MODE) {
    antiquantParamsInPageAttentionFlag_ = true;
    antiquantPerHeadFlag_ = NUM1;
    OP_CHECK_IF((inputParaShape.GetDimNum() != DIM_BNB),
      OP_LOGE(ifaContext_->opName, "The dimension of antiquant should be 3 instead of the current %lu, when antiquant mode is 5.",
                inputParaShape.GetDimNum()),
                return ge::GRAPH_FAILED);
  }
  OP_CHECK_IF((antiquantParamsInPageAttentionFlag_ && !pageAttentionFlag_),
      OP_LOGE(ifaContext_->opName,
               "KeyAntiquantMode/valueAntiquantMode 4 or 5 use page attention to manage scale/offset, must to enble page attention."),
                return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATilingV2::CheckKVAntiQuantPerHead(const gert::Shape &inputParaShape)
{
  if (antiquantMode_ == PER_TOKEN_HEAD_MODE) { // per-token head
    OP_CHECK_IF((inputParaShape.GetDimNum() != NUM3), // 3: Dim of BGS is 3
                OP_LOGE(ifaContext_->opName, "The dimension of antiquant should be 3 instead of the current %lu, when per_token_head mode.",
                          inputParaShape.GetDimNum()),
                          return ge::GRAPH_FAILED);
    OP_CHECK_IF((inputParaShape.GetDim(NUM0) != batchSize_),
                OP_LOGE(ifaContext_->opName, "The 1st dimension of antiquant should be %u instead of the current %ld.",
                          batchSize_, inputParaShape.GetDim(NUM0)),
                          return ge::GRAPH_FAILED);
    OP_CHECK_IF((inputParaShape.GetDim(NUM1) != numKvHeads_),
                OP_LOGE(ifaContext_->opName, "The 2nd dimension of antiquant should be %u instead of the current %ld.",
                          numKvHeads_, inputParaShape.GetDim(NUM1)),
                          return ge::GRAPH_FAILED);
    OP_CHECK_IF((inputParaShape.GetDim(NUM2) < seqSize_),
                OP_LOGE(ifaContext_->opName, "The 3rd dimension of antiquant should be greater than or equal to %u instead of the current %ld.",
                          seqSize_, inputParaShape.GetDim(NUM2)), // 2 : BGS S index is 2
                return ge::GRAPH_FAILED);
    antiquantParaSeqSize_ = inputParaShape.GetDim(NUM2);
    return ge::GRAPH_SUCCESS;
  } else { // per-tensor head
    gert::Shape expectParamShape = gert::Shape({numKvHeads_});
    OP_CHECK_IF((inputParaShape.GetDimNum() != NUM1), // 1: Dim of N is 1
                OP_LOGE(ifaContext_->opName, "The dimension of antiquant should be 1 instead of the current %lu, when per_tensor_head mode.",
                          inputParaShape.GetDimNum()),
                          return ge::GRAPH_FAILED);
    OP_CHECK_IF((inputParaShape != expectParamShape),
                OP_LOGE(ifaContext_->opName,
                          "The shape of antiquant parameter[%ld] is not expected. Expect[%u] When per_tensor_head mode.",
                          inputParaShape.GetDim(NUM0), numKvHeads_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
  }
}

ge::graphStatus IFATilingV2::CheckAntiQuantParam(const int64_t antiquantMode, const gert::Tensor* antiquantScaleTensor, const gert::Tensor* antiquantOffsetTensor,
                                               const gert::CompileTimeTensorDesc * antiquantScaleDesc, const gert::CompileTimeTensorDesc * antiquantOffsetDesc) {
  // 3:per-token+pergroup
  OP_CHECK_IF(((antiquantMode != PER_CHANNEL_MODE) && (antiquantMode != PER_TOKEN_MODE) &&
             (socVersion_ == IfaSocVersion::SOC_ASCEND_950 && antiquantMode != PER_TENSOR_HEAD_MODE &&
              antiquantMode != PER_TOKEN_HEAD_MODE && antiquantMode != PER_TOKEN_PA_MODE &&
              antiquantMode != PER_TOKEN_HEAD_PA_MODE && antiquantMode != PER_TOKEN_GROUP_MODE)),
             OP_LOGE(ifaContext_->opName, "AntiquantMode value:%ld is invalid, it should be 0, 1, 2, 3, 4, 5 or 6.", antiquantMode),
             return ge::GRAPH_FAILED);

  OP_CHECK_IF(antiquantScaleTensor == nullptr,
            OP_LOGE(ifaContext_->opName, "Antiquant is enabled, but the input antiquantScale is null."),
            return ge::GRAPH_FAILED);
  OP_CHECK_IF((antiquantOffsetTensor != nullptr &&
              antiquantScaleTensor->GetStorageShape().GetDimNum() != antiquantOffsetTensor->GetStorageShape().GetDimNum()),
              OP_LOGE(ifaContext_->opName, "Antiquant is enabled, "
                                          "but antiquant params have different layouts[scale: %lu, offset: %lu].",
              antiquantScaleTensor->GetStorageShape().GetDimNum(), antiquantOffsetTensor->GetStorageShape().GetDimNum()),
              return ge::GRAPH_FAILED);

  OP_CHECK_IF((antiquantOffsetTensor != nullptr &&
             (inputKvType_ == ge::DT_HIFLOAT8 || inputKvType_ == ge::DT_FLOAT8_E4M3FN ||
              inputKvType_ == ge::DT_FLOAT4_E2M1)),
              OP_LOGE(ifaContext_->opName, "When input key/value dataType is fp8/hifp8/fp4_e2m1, antiquantOffset is not supported."),
              return ge::GRAPH_FAILED);
  OP_CHECK_IF(((antiquantMode == PER_TENSOR_HEAD_MODE || antiquantMode == PER_TOKEN_HEAD_PA_MODE) && inputKvType_ != ge::DT_INT8),
 	            OP_LOGE(ifaContext_->opName, "When antiquantMode of key/value is 2 or 5, input key/value type should be int8, "
 	                       "but now is %s.", DataTypeToString(inputKvType_).c_str()),
 	             return ge::GRAPH_FAILED);
 	OP_CHECK_IF(((antiquantMode == PER_TOKEN_PA_MODE) && (inputKvType_ != ge::DT_INT8 && inputKvType_ != ge::DT_FLOAT8_E4M3FN)),
 	            OP_LOGE(ifaContext_->opName, "When antiquantMode of key/value is 4, input key/value type should be int8 or fp8_e4m3, "
 	                       "but now is %s.", DataTypeToString(inputKvType_).c_str()),
 	             return ge::GRAPH_FAILED);
  OP_CHECK_IF((antiquantMode == PER_TOKEN_GROUP_MODE && !(inputKvType_ == ge::DT_FLOAT4_E2M1)),
            OP_LOGE(ifaContext_->opName, "When antiquantMode of key/value is PER_TOKEN_GROUP(6), input key/value type should be fp4_e2m1, "
                      "but now is %s.", DataTypeToString(inputKvType_).c_str()),
            return ge::GRAPH_FAILED);
  OP_CHECK_IF((antiquantOffsetTensor != nullptr && (!ShapeEqual(antiquantScaleTensor->GetStorageShape(), antiquantOffsetTensor->GetStorageShape()))),
            OP_LOGE(ifaContext_->opName, "antiquantScaleTensor and antiquantOffsetTensor should have the same shape"),
            return ge::GRAPH_FAILED);
  auto tmpAntiquantScale = antiquantScaleTensor->GetStorageShape();
  OP_CHECK_IF(CheckKVAntiQuantParaShapeLegal(antiquantMode, tmpAntiquantScale) == ge::GRAPH_FAILED,
            OP_LOGE(ifaContext_->opName, "Illegal shape of antiquant scale."),
            return ge::GRAPH_FAILED);

  if (antiquantOffsetTensor != nullptr) {
    auto tmpAntiquantOffset = antiquantOffsetTensor->GetStorageShape();
    if (CheckKVAntiQuantParaShapeLegal(antiquantMode, tmpAntiquantOffset) == ge::GRAPH_FAILED) {
      return ge::GRAPH_FAILED;
    }
  }

  ge::DataType antiquantScaleType = antiquantScaleDesc->GetDataType();

  OP_CHECK_IF(((antiquantMode == 0U || antiquantMode == 2U) && antiquantScaleType != inputQType_),  // per-tensor per-channel and per-tensor-head
             OP_LOGE(ifaContext_->opName,
                       "Datatype(%s) of antiquant scale is illegal, it should be same with input qtype(%s) when per-tensor mode or "
                       "per-channel mode or per-tensor-head mode.", DataTypeToString(antiquantScaleType).c_str(),
                       DataTypeToString(inputQType_).c_str()),
             return ge::GRAPH_FAILED);

  OP_CHECK_IF((antiquantMode == PER_TOKEN_MODE|| antiquantMode == PER_TOKEN_HEAD_MODE ||
              antiquantMode == PER_TOKEN_PA_MODE|| antiquantMode == PER_TOKEN_HEAD_PA_MODE) && (antiquantScaleType != ge::DT_FLOAT),
          OP_LOGE(ifaContext_->opName,
                    "Per-token/per-token-head/per-token-pa/per-token-head-pa mode is enabled, datatype of antiquant scale(%s) should be float32.",
                    DataTypeToString(antiquantScaleType).c_str()),
          return ge::GRAPH_FAILED);

  OP_CHECK_IF((antiquantMode == 6U && antiquantScaleType != ge::DT_FLOAT8_E8M0),
             OP_LOGE(ifaContext_->opName,
                       "Per-token-group mode is enabled, datatype of antiquant scale(%s) should be float8_e8m0.",
                       DataTypeToString(antiquantScaleType).c_str()),
             return ge::GRAPH_FAILED);

  if (antiquantOffsetTensor != nullptr && antiquantOffsetDesc != nullptr) {
    ge::DataType antiquantOffsetType = antiquantOffsetDesc->GetDataType();
    if (antiquantScaleType != antiquantOffsetType) {
      OP_LOGE(ifaContext_->opName, "Datatype of antiquant scale and antiquant offset should be the same.");
      return ge::GRAPH_FAILED;
    }
  }

  gert::Shape keyAntiquantScaleTensorShape = antiquantScaleTensor->GetStorageShape();
  gert::Shape expectedShape1 = gert::Shape({1});
  if (antiquantMode == PER_CHANNEL_MODE) {
    // per-tensor
    if (inputKvType_ == ge::DT_INT4 || inputKvType_ == ge::DT_INT32) {
      OP_CHECK_IF((ShapeEqual(expectedShape1, keyAntiquantScaleTensorShape) && inputKvType_ != ge::DT_INT8),
                  OP_LOGE(ifaContext_->opName,
                          "In per-tensor mode, the data type of key/value should be int8, but now is INT4/INT32."),
                  return ge::GRAPH_FAILED);
    } else {
      OP_CHECK_IF((ShapeEqual(expectedShape1, keyAntiquantScaleTensorShape) && inputKvType_ != ge::DT_INT8),
                  OP_LOGE(ifaContext_->opName,
                          "In per-tensor mode, the data type of key/value should be int8, but now is %s.",
                          DataTypeToString(inputKvType_).c_str()),
                  return ge::GRAPH_FAILED);
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATilingV2::KeyAndValueAntiQuantParamConsistencyCheck(const gert::Tensor* keyAntiquantTensor,
                                                                       const gert::Tensor* valueAntiquantTensor,
                                                                       const gert::CompileTimeTensorDesc* keyAntiquantDesc,
                                                                       const gert::CompileTimeTensorDesc* valueAntiquantDesc,
                                                                       int64_t keyAntiquantMode, int64_t valueAntiquantMode, const std::string sName) const {
  OP_CHECK_IF((keyAntiquantDesc == nullptr),
             OP_LOGE(ifaContext_->opName, "key %s exist, but it's dataType doesn't exist.", sName.c_str()),
             return ge::GRAPH_FAILED);
  OP_CHECK_IF((valueAntiquantDesc == nullptr),
            OP_LOGE(ifaContext_->opName, "value %s exist, but it's dataType doesn't exist.", sName.c_str()),
            return ge::GRAPH_FAILED);
  OP_CHECK_IF(((!kPerChnVPerTokFlag_) && (keyAntiquantDesc->GetDataType() != valueAntiquantDesc->GetDataType())),
            OP_LOGE(ifaContext_->opName, "key%s and value%s should have the same datatype "
                    "when keyAntiquantMode(%ld) isn't 0 and valueAntiquantMode(%ld) isn't 1.", sName.c_str(), sName.c_str(),
                    keyAntiquantMode, valueAntiquantMode),
            return ge::GRAPH_FAILED);
  if ((!kPerChnVPerTokFlag_) && (!ShapeEqual(keyAntiquantTensor->GetStorageShape(), valueAntiquantTensor->GetStorageShape()))) {
    OP_LOGE(ifaContext_->opName, "Key%s and value%s should have the same shape when keyAntiquantMode(%ld) "
              "isn't 0 and valueAntiquantMode(%ld) isn't 1.",sName.c_str(), sName.c_str(), keyAntiquantMode, valueAntiquantMode);
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATilingV2::ProcessAntiQuant() {
  auto antiquantScaleTensor = ifaContext_->antiquantScale.tensor;
  auto antiquantScaleDesc = ifaContext_->antiquantScale.desc;
  auto antiquantOffsetTensor = ifaContext_->antiquantOffset.tensor;
  auto antiquantOffsetDesc = ifaContext_->antiquantOffset.desc;
  auto keyAntiquantScaleTensor = ifaContext_->keyAntiquantScale.tensor;
  auto keyAntiquantScaleDesc = ifaContext_->keyAntiquantScale.desc;
  auto keyAntiquantOffsetTensor = ifaContext_->keyAntiquantOffset.tensor;
  auto keyAntiquantOffsetDesc = ifaContext_->keyAntiquantOffset.desc;
  auto valueAntiquantScaleTensor = ifaContext_->valueAntiquantScale.tensor;
  auto valueAntiquantScaleDesc = ifaContext_->valueAntiquantScale.desc;
  auto valueAntiquantOffsetTensor = ifaContext_->valueAntiquantOffset.tensor;
  auto valueAntiquantOffsetDesc = ifaContext_->valueAntiquantOffset.desc;
  auto queryRopeInputShape = ifaContext_->queryRopeInputShape;
  auto keyRopeInputShape = ifaContext_->keyRopeInputShape;

  OP_CHECK_IF((keyAntiquantScaleTensor == nullptr || valueAntiquantScaleTensor == nullptr),
    OP_LOGE(ifaContext_->opName, "In antiquant scenario, keyAntiquantScale and valueAntiquantScale must exist."),
      return ge::GRAPH_FAILED);
  OP_CHECK_IF((*ifaContext_->antiquantMode != 0 || antiquantScaleTensor != nullptr || antiquantOffsetTensor != nullptr),
    OP_LOGE(ifaContext_->opName, "Antiquant scenario only supports key/value split mode, antiquantMode,"
            "antiquantScale and antiquantOffset are not supported."),
      return ge::GRAPH_FAILED);
  if (!antiQuantFlag_ && (antiquantScaleTensor != nullptr || antiquantOffsetTensor != nullptr
    || keyAntiquantScaleTensor != nullptr || keyAntiquantOffsetTensor != nullptr
    || valueAntiquantScaleTensor != nullptr || valueAntiquantOffsetTensor != nullptr)) {
    OP_LOGE(ifaContext_->opName, "KeyAntiquant/valueAntiquant is unenabled, but antiquantScale/antiquantOffset exists.");
    return ge::GRAPH_FAILED;
  }
 
  if (!antiQuantFlag_ || isMaxWorkspace_) {
    return ge::GRAPH_SUCCESS;
  }
  kvAntiParamSplitFlag_ = false;
  OP_CHECK_IF(antiQuantFlag_ && (queryRopeInputShape != nullptr || keyRopeInputShape != nullptr),
    OP_LOGE(ifaContext_->opName, "Rope is not supported in antiquant scenario."),
      return ge::GRAPH_FAILED);
  OP_CHECK_IF((keyAntiquantScaleTensor != nullptr && valueAntiquantScaleTensor == nullptr),
    OP_LOGE(ifaContext_->opName, "ValueAntiquantScaleTensor is null, but keyAntiquantScaleTensor exists."),
      return ge::GRAPH_FAILED);
  OP_CHECK_IF((valueAntiquantScaleTensor != nullptr && keyAntiquantScaleTensor == nullptr),
    OP_LOGE(ifaContext_->opName, "KeyAntiquantScaleTensor is null, but valueAntiquantScaleTensor exists."),
      return ge::GRAPH_FAILED);
  OP_CHECK_IF((keyAntiquantOffsetTensor != nullptr && valueAntiquantOffsetTensor == nullptr),
    OP_LOGE(ifaContext_->opName, "ValueAntiquantOffsetTensor is null, but keyAntiquantOffsetTensor exists."),
      return ge::GRAPH_FAILED);
  OP_CHECK_IF((valueAntiquantOffsetTensor != nullptr && keyAntiquantOffsetTensor == nullptr),
    OP_LOGE(ifaContext_->opName, "KeyAntiquantOffsetTensor is null, but valueAntiquantOffsetTensor exists."),
      return ge::GRAPH_FAILED);
  OP_CHECK_IF((keyAntiquantScaleTensor == nullptr && keyAntiquantOffsetTensor != nullptr),
    OP_LOGE(ifaContext_->opName, "KeyAntiquantScaleTensor is null, but keyAntiquantOffsetTensor exists."),
      return ge::GRAPH_FAILED);
  int64_t keyAntiquantMode = (ifaContext_->keyAntiquantMode != nullptr) ? *ifaContext_->keyAntiquantMode : 0;
  int64_t valueAntiquantMode = (ifaContext_->valueAntiquantMode != nullptr) ? *ifaContext_->valueAntiquantMode : 0;
  kPerChnVPerTokFlag_  = (inputKvType_ == ge::DT_INT4 || inputKvType_ == ge::DT_INT8) &&
                         (keyAntiquantMode == PER_CHANNEL_MODE && valueAntiquantMode == PER_TOKEN_MODE);
  if (keyAntiquantOffsetTensor != nullptr && valueAntiquantOffsetTensor != nullptr) {
    if (KeyAndValueAntiQuantParamConsistencyCheck(keyAntiquantOffsetTensor, valueAntiquantOffsetTensor, keyAntiquantOffsetDesc, valueAntiquantOffsetDesc,
                              keyAntiquantMode, valueAntiquantMode, "AntiquantOffset") != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;                                          
    }
  }
  if (keyAntiquantScaleTensor != nullptr && valueAntiquantScaleTensor != nullptr) {
    if (KeyAndValueAntiQuantParamConsistencyCheck(keyAntiquantScaleTensor, valueAntiquantScaleTensor, keyAntiquantScaleDesc,
                                              valueAntiquantScaleDesc, keyAntiquantMode, valueAntiquantMode, "AntiquantScale") != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
    }
    kvAntiParamSplitFlag_ = true;
  }
  if (kvAntiParamSplitFlag_) {
    if ((!kPerChnVPerTokFlag_) && (keyAntiquantMode != valueAntiquantMode)) {
      OP_LOGE(ifaContext_->opName, "KeyAntiquantMode and valueAntiquantMode should be the same "
               "when keyAntiquantMode(%ld) isn't 0 and ValueAntiquantMode(%ld) isn't 1.", keyAntiquantMode, valueAntiquantMode);
      return ge::GRAPH_FAILED;
    }
    antiquantMode_ = keyAntiquantMode;
    if (CheckAntiQuantParam(keyAntiquantMode, keyAntiquantScaleTensor, keyAntiquantOffsetTensor,
                            keyAntiquantScaleDesc, keyAntiquantOffsetDesc) == ge::GRAPH_FAILED) {
      return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF((antiquantMode_ == PER_TOKEN_MODE || antiquantMode_ == PER_TOKEN_PA_MODE)
 	                   && (inputKvType_ == ge::DT_FLOAT8_E4M3FN && (outputType_ != ge::DT_BF16 && outputType_ != ge::DT_FLOAT16)),
 	        OP_LOGE(ifaContext_->opName, "When antiquantMode of key/value is 1 or 4, if data type of key/value is float8_e4m3, post quant is not supported."),
 	        return ge::GRAPH_FAILED);
    if (kPerChnVPerTokFlag_) {
      OP_CHECK_IF((inputKvType_ == ge::DT_INT8 && (inputQType_ == ge::DT_BF16 || outputType_ == ge::DT_BF16)),
        OP_LOGE(ifaContext_->opName, "When key in per-channel scenario and value in pre-token scenario,"
        "if inputKvType is Int8, inputQType and outputType only must be FP16, now inputQType is %s, outputType is %s.",
                optiling::v2::GetPfaDataTypeStr(inputQType_).c_str(), optiling::v2::GetPfaDataTypeStr(outputType_).c_str()),
          return ge::GRAPH_FAILED);
      if (CheckAntiQuantParam(valueAntiquantMode, valueAntiquantScaleTensor, valueAntiquantOffsetTensor,
                              valueAntiquantScaleDesc, valueAntiquantOffsetDesc) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
      }
    } else if (pageAttentionFlag_) {
        uint32_t dimNum = keyAntiquantScaleTensor->GetStorageShape().GetDimNum();
        if (keyAntiquantMode == PER_TOKEN_MODE) {
            OP_CHECK_IF(
              (keyAntiquantScaleTensor->GetStorageShape().GetDim(dimNum - NUM1) < maxBlockNumPerSeq_ * blockSize_),
              OP_LOGE(ifaContext_->opName,
                      "The last dimension(%u) of keyAntiquantScale is less than maxBlockNumPerSeq(%u) * blockSize(%u). "
                      "The last dimension of keyAntiquantScale should be larger than or equal to maxBlockNumPerSeq * blockSize when "
                      "keyAntiquantMode, valueAntiquantMode are per-token mode and keyAntiquant/valueAntiquant is splited.",
                      keyAntiquantScaleTensor->GetStorageShape().GetDim(dimNum - NUM1), maxBlockNumPerSeq_, blockSize_),
              return ge::GRAPH_FAILED);
        }
        if (keyAntiquantMode == PER_TOKEN_HEAD_MODE) {
            OP_CHECK_IF(
              (keyAntiquantScaleTensor->GetStorageShape().GetDim(dimNum - NUM1) < maxBlockNumPerSeq_ * blockSize_),
              OP_LOGE(ifaContext_->opName,
                      "The last dimension(%u) of keyAntiquantScale is less than maxBlockNumPerSeq(%u) * blockSize(%u). "
                      "The last dimension of keyAntiquantScale should be larger than or equal to maxBlockNumPerSeq * blockSize when "
                      "keyAntiquantMode, valueAntiquantMode are per-token-head mode and keyAntiquant/valueAntiquant is splited.",
                      keyAntiquantScaleTensor->GetStorageShape().GetDim(dimNum - NUM1), maxBlockNumPerSeq_, blockSize_),
              return ge::GRAPH_FAILED);
        }
        if (keyAntiquantMode == PER_TOKEN_GROUP_MODE) {
            OP_CHECK_IF(
              (keyAntiquantScaleTensor->GetStorageShape().GetDim(dimNum - NUM2) < maxBlockNumPerSeq_ * blockSize_),
              OP_LOGE(ifaContext_->opName,
                      "The second-to-last dimension(%u) of keyAntiquantScale is less than maxBlockNumPerSeq(%u) * blockSize(%u). "
                      "The second-to-last dimension of keyAntiquantScale should be larger than or equal to maxBlockNumPerSeq * blockSize when "
                      "keyAntiquantMode, valueAntiquantMode are per-token-group mode and keyAntiquant/valueAntiquant is splited.",
                      keyAntiquantScaleTensor->GetStorageShape().GetDim(dimNum - NUM2), maxBlockNumPerSeq_, blockSize_),
              return ge::GRAPH_FAILED);
        }
    }
    OP_CHECK_IF((inputKvType_ == ge::DT_INT8 && inputLayout_ == IfaLayout::TND),
                OP_LOGE(ifaContext_->opName, "In keyAntiquant/valueAntiquant split mode and data type of key/value is int8 scenario,"
                        "the layout of input does not support TND."),
                return ge::GRAPH_FAILED);
    if (isPFAFlag_) {
      OP_CHECK_IF((antiquantMode_ == PER_TENSOR_HEAD_MODE || antiquantMode_ == PER_TOKEN_HEAD_MODE || 
                   antiquantMode_ == PER_TOKEN_PA_MODE || antiquantMode_ == PER_TOKEN_HEAD_PA_MODE) && inputKvType_ == ge::DT_INT8,
          OP_LOGE(ifaContext_->opName, "In keyAntiquant/valueAntiquant split mode and data type of key/value is int8 scenario,"
                  "if S of query > 1, keyAntiquantMode/valueAntiquantMode 2, 3, 4, 5 are not supported."),
          return ge::GRAPH_FAILED);
      OP_CHECK_IF((antiquantMode_ == PER_CHANNEL_MODE || antiquantMode_ == PER_TOKEN_MODE)
                  && (inputKvType_ == ge::DT_INT8 && (inputQType_ != ge::DT_BF16 || outputType_ != ge::DT_BF16)),
                OP_LOGE(ifaContext_->opName, "In keyAntiquant/valueAntiquant split mode and data type of key/value is int8 scenario,"
                        "if keyAntiquantMode/valueAntiquantMode is 0 or 1, the data type of query and output only support BF16."),
                return ge::GRAPH_FAILED);
      OP_CHECK_IF((antiquantMode_ == PER_CHANNEL_MODE || antiquantMode_ == PER_TOKEN_MODE) && (inputKvType_ == ge::DT_INT8 && sOfQuery_ > 16),
                OP_LOGE(ifaContext_->opName, "In keyAntiquant/valueAntiquant split mode and data type of key/value is int8 scenario,"
                        "if keyAntiquantMode/valueAntiquantMode is 0 or 1, S of query should not be greater than 16."),
                return ge::GRAPH_FAILED);
      OP_CHECK_IF((antiquantMode_ == PER_CHANNEL_MODE || antiquantMode_ == PER_TOKEN_MODE) && (inputKvType_ == ge::DT_INT8 && !batchContinuousFlag_),
                OP_LOGE(ifaContext_->opName, "In keyAntiquant/valueAntiquant split mode and data type of key/value is int8 scenario,"
                        "if keyAntiquantMode/valueAntiquantMode is 0 or 1, tensorlist is not supported."),
                return ge::GRAPH_FAILED);
      OP_CHECK_IF((antiquantMode_ == PER_CHANNEL_MODE || antiquantMode_ == PER_TOKEN_MODE)
                  && (inputKvType_ == ge::DT_INT8 && (ifaContext_->queryPaddingSize.tensor || ifaContext_->kvPaddingSize.tensor)),
                OP_LOGE(ifaContext_->opName, "In keyAntiquant/valueAntiquant split mode and data type of key/value is int8 scenario,"
                        "if keyAntiquantMode/valueAntiquantMode is 0 or 1, leftpadding is not supported."),
                return ge::GRAPH_FAILED);
      OP_CHECK_IF((antiquantMode_ == PER_CHANNEL_MODE || antiquantMode_ == PER_TOKEN_MODE) && (inputKvType_ == ge::DT_INT8 && pageAttentionFlag_),
                OP_LOGE(ifaContext_->opName, "In keyAntiquant/valueAntiquant split mode and data type of key/value is int8 scenario,"
                        "if keyAntiquantMode/valueAntiquantMode is 0 or 1, page attention is not supported."),
                return ge::GRAPH_FAILED);
      OP_CHECK_IF((inputKvType_ == ge::DT_INT4 || inputKvType_ == ge::DT_INT32),
                OP_LOGE(ifaContext_->opName, "In keyAntiquant/valueAntiquant split mode scenario, int4 and int32 data types are not supported for the key and value."),
                return ge::GRAPH_FAILED);
    }
  } else {
    OP_CHECK_IF((antiquantScaleTensor !=nullptr && antiquantScaleDesc == nullptr),
             OP_LOGE(ifaContext_->opName, "antiquantScaleTensor exist, but it's datatype doesn't exist."),
             return ge::GRAPH_FAILED);
    OP_CHECK_IF((antiquantOffsetTensor !=nullptr && antiquantOffsetDesc == nullptr),
             OP_LOGE(ifaContext_->opName, "antiquantOffsetTensor exist, but it's datatype doesn't exist."),
             return ge::GRAPH_FAILED);
    OP_LOGD(ifaContext_->opName, "KeyAntiquant/valueAntiquant is not split mode.");
    OP_CHECK_IF((inputKvType_ == ge::DT_HIFLOAT8 || inputKvType_ == ge::DT_FLOAT8_E4M3FN ||
                inputKvType_ == ge::DT_FLOAT4_E2M1),
                OP_LOGE(ifaContext_->opName, "When input key/value dataType is fp8/hifp8/fp4_e2m1, keyAntiquant/valueAntiquant must be split mode."),
                return ge::GRAPH_FAILED);
    if (ifaContext_->antiquantMode != nullptr) {
      antiquantMode_ = *ifaContext_->antiquantMode;
    }
    
    if (CheckAntiQuantParam(antiquantMode_, antiquantScaleTensor, antiquantOffsetTensor, antiquantScaleDesc, antiquantOffsetDesc) == ge::GRAPH_FAILED) {
      return ge::GRAPH_FAILED;
    }
  }
  OP_LOGD(ifaContext_->opName, "Antiquant info antiquantMode:%ld.", antiquantMode_);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATilingV2::ProcessBlockTable() {
  if (!pageAttentionFlag_) {
    return ge::GRAPH_SUCCESS;
  }
  if (isMaxWorkspace_) {
    totalBlockNum_ = ifaContext_->kCache[0]->GetStorageShape().GetDim(NUM0);
    return ge::GRAPH_SUCCESS;
  }

  auto blockTable = ifaContext_->blockTable.tensor;
  OP_CHECK_IF(blockTable->GetStorageShape().GetDim(NUM0) != batchSize_,
             OP_LOGE(ifaContext_->opName, "The first dimension of blockTable is %ld, and it should be equal to batch[%u].",
             blockTable->GetStorageShape().GetDim(NUM0), batchSize_), return ge::GRAPH_FAILED);
  uint32_t maxBlockNumPerSeq = (maxActualseq_ + blockSize_ - 1U) / blockSize_;
  OP_CHECK_IF(blockTable->GetStorageShape().GetDim(NUM1) < maxBlockNumPerSeq,
             OP_LOGE(ifaContext_->opName, "The second dimension of blockTable is %ld, and it should be greater than or equal to maxBlockNumPerSeq[%u].",
             blockTable->GetStorageShape().GetDim(NUM1), maxBlockNumPerSeq), return ge::GRAPH_FAILED);
  // gm到l1，copynd2nz的srcDValue最大支持65535
  if ((pageAttentionKvLayoutType_ == KvCacheLayout::KV_CACHE_BSH) &&
    (numKvHeads_ * headDim_ > COPYND2NZ_SRC_STRIDE_LIMITATION)) { // 0: BSH
    OP_LOGE(ifaContext_->opName, "When input kvcache layout is BSH, the N * D of kvcache is %u, "
      "exceeds the maximum limit (%u) of the datacopy instruction.",
      numKvHeads_ * headDim_, COPYND2NZ_SRC_STRIDE_LIMITATION);
    return ge::GRAPH_FAILED;
  }

  uint32_t blockNum = ifaContext_->key.shape->GetStorageShape().GetDim(NUM0);
  OP_CHECK_IF(blockNum < needBlockNum_,
              OP_LOGE(ifaContext_->opName, "blockNum[%u] cannot be less than the sum of blocks in each batch calculated based on actualSeqLenKv and blockSize[%lu]", 
              blockNum, needBlockNum_), return ge::GRAPH_FAILED);

  totalBlockNum_ = ifaContext_->kCache[0]->GetStorageShape().GetDim(NUM0);
  if (antiquantParamsInPageAttentionFlag_) {
    auto keyAntiquantScaleTensor = ifaContext_->keyAntiquantScale.tensor;
    auto keyAntiquantScaleShape = keyAntiquantScaleTensor->GetStorageShape();
    if (CheckKVAntiQuantShapePA(keyAntiquantScaleShape) != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATilingV2::ProcessQPaddingSize() {
  if (sOfQuery_ == NUM1) {
    return ge::GRAPH_SUCCESS;
  }
  auto qPaddingSize = ifaContext_->queryPaddingSize.tensor;
  if (qPaddingSize == nullptr) {
    OP_LOGD(ifaContext_->opName, "QLeftPadding illegal condition: QPaddingSize.tensor is nullptr: %d.",
      ifaContext_->queryPaddingSize.tensor == nullptr);
    return ge::GRAPH_SUCCESS;
  }

  OP_CHECK_IF(enableAlibiPse_,
    OP_LOGE(ifaContext_->opName, "When pseType = 2/3, left padding is not supported!"),
    return ge::GRAPH_FAILED);

  if (pageAttentionFlag_) {
    OP_LOGE(ifaContext_->opName, "When page attention is used, left padding is not supported!");
    return ge::GRAPH_FAILED;
  }

  if (qPaddingSize->GetStorageShape().GetShapeSize() == 0) {
    OP_LOGD(ifaContext_->opName, "QueryLeftPadding illegal condition: QueryPaddingSize.tensor shape is empty: %d.",
      qPaddingSize->GetStorageShape().GetShapeSize() == 0);
    return ge::GRAPH_SUCCESS;
  }

  ge::graphStatus ret = CheckSupportQLeftPadding();

  return ret;
}

ge::graphStatus IFATilingV2::CheckSupportQLeftPadding() {
  if (!batchContinuousFlag_ || !actualSeqLenQFlag_ || pageAttentionFlag_ || inputLayout_ == IfaLayout::TND) {
    OP_LOGD(ifaContext_->opName,
      "QueryLeftPadding illegal condition:  \
      paged attention scene: %d, not isBatchContinues: %d, actualSeqLen not exist: %d, input layout TND.",
      pageAttentionFlag_, !batchContinuousFlag_, !actualSeqLenQFlag_);
    return ge::GRAPH_SUCCESS;
  }
  qPaddingSizeFlag_ = true;
  needInit_ = true;
  OP_LOGD(ifaContext_->opName, "QLeftPadding starts to be used.");
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATilingV2::ProcessKVPaddingSize() {
  auto kvPaddingSize = ifaContext_->kvPaddingSize.tensor;
  if (kvPaddingSize == nullptr) {
    OP_LOGD(ifaContext_->opName, "KVLeftPadding illegal condition: kvPaddingSize.tensor is nullptr: %d.",
              ifaContext_->kvPaddingSize.tensor == nullptr);
    return ge::GRAPH_SUCCESS;
  }

  if (pageAttentionFlag_) {
    OP_LOGE(ifaContext_->opName, "When page attention is used, left padding is not supported!");
    return ge::GRAPH_FAILED;
  }

  OP_CHECK_IF(enableAlibiPse_,
    OP_LOGE(ifaContext_->opName, "When pseType = 2/3, left padding is not supported!"),
    return ge::GRAPH_FAILED);

  if (kvPaddingSize->GetStorageShape().GetShapeSize() == 0) {
    OP_LOGD(ifaContext_->opName, "KVLeftPadding illegal condition: kvPaddingSize.tensor shape is empty: %d.",
              kvPaddingSize->GetStorageShape().GetShapeSize() == 0);
    return ge::GRAPH_SUCCESS;
  }

  ge::graphStatus ret = CheckSupportKVLeftPadding();

  return ret;
}

ge::graphStatus IFATilingV2::CheckSupportKVLeftPadding() {
  if (!batchContinuousFlag_ || !actualSeqLenFlag_ || pageAttentionFlag_ || inputLayout_ == IfaLayout::TND) {
    OP_LOGD(ifaContext_->opName,
      "KVLeftPadding illegal condition:  \
      paged attention scene: %d, not isBatchContinues: %d, actualSeqLen not exist: %d, input layout TND.",
      pageAttentionFlag_, !batchContinuousFlag_, !actualSeqLenFlag_);
    return ge::GRAPH_SUCCESS;
  }
  kvPaddingSizeFlag_ = true;
  if (isPFAFlag_) {
    needInit_ = true;
  }
  OP_LOGD(ifaContext_->opName, "KVLeftPadding starts to be used.");
  return ge::GRAPH_SUCCESS;
}

bool IFATilingV2::IsFlashDecode() const {
  // 0.4: 经验值
  if ((batchSize_ * numKvHeads_ * NUM5 < NUM2 * coreNum_) && (nNumOfQInOneGroup_ == NUM1)) {
    OP_LOGD(ifaContext_->opName, "Flash decode dplit key/value.");
    return true;
  }

  if ((batchSize_ * numKvHeads_ * NUM5 < NUM2 * coreNum_) &&
    (maxActualseq_ >= 2048)) { // 2048, 在flash decode + gqa时的经验值
    OP_LOGD(ifaContext_->opName, "Flash decode And GQA split key/value.");
    return true;
  }
  return false;
}

bool IFATilingV2::IsFlashDecodefaRun() const {
    float flashDecodeBNRatio = 0.4F; // 0.4, 经验值
    uint32_t sInnerDouble = sInnerSize_ * 2;
    if (enableKVPrefix_) {
      return false;
    }
    // 如果s2方向上最长还不超过两个sinnersize，不生效FD
    if (sMax_ < sInnerDouble) {
        return false;
    }
    uint64_t bng = batchSize_ * numKvHeads_ * ((nNumOfQInOneGroup_ + sOuterSize_ - 1) / sOuterSize_);  // 是否添加括号？
    if ((bng < flashDecodeBNRatio * aicNum_) && (nNumOfQInOneGroup_ == 1)) {
        OP_LOGD(ifaContext_->opName, "Flash decode dplit key/value.");
        return true;
    }
    if ((bng < flashDecodeBNRatio * aicNum_) && (maxActualseq_ >= 2048)) { // 2048, 在flash decode + gqa时的经验值
        OP_LOGD(ifaContext_->opName, "Flash decode And GQA split key/value.");
        return true;
    }
    return false;
}

ge::graphStatus IFATilingV2::Split() {
  CalcInnerSize(seqSize_);
  FlashAttentionCubeSplitBNSeq();
  if (!isPFAFlag_) {
    if (IsFlashDecodefaRun()) {
      splitKVFlag_ = true;
      return SplitBNSfaRun();
    }
  }
  return ge::GRAPH_SUCCESS;
}

std::vector<int64_t> IFATilingV2::InitSparseValidArray(const int64_t* actualLens) const {
  std::vector<int64_t> res((batchSize_ * numKvHeads_));
  for (uint32_t b = 0; b < batchSize_; b++) {
    for (uint32_t n = 0; n < numKvHeads_; n++) {
      int64_t estimatedLoad = seqSize_;
      if (actualLens != nullptr) {
        estimatedLoad = actualLens[b];
        if (antiQuantFlag_ && estimatedLoad < MSD_VEC_LOAD) {
          estimatedLoad = MSD_VEC_LOAD;
        } else if (actualLens[b] == 0) {
          // 空tensor输出也计入负载，否则当最后一个batch为空tensor时，分核算法会将该batch优化掉
          estimatedLoad = static_cast<int64_t>(NUM1);
        }
      }
      res[b * numKvHeads_ + n] = estimatedLoad;
    }
  }
  return res;
}
// code copy from flash_attention_score_tiling
bool IFATilingV2::BalanceLoad(const std::vector<int64_t>& sparseValidArray, int64_t totalSize, int64_t validAivNum,
                            std::vector<int64_t>& localValue, std::vector<int64_t>& sparseStartIdx) const {
  // to avoid buffer overflow, or maybe sometimes we want to only verify single
  // core
  int64_t maxVal = *std::max_element(localValue.begin(), localValue.end());
  int64_t tmpMaxVal = maxVal;

  // 从前往后遍历
  for (int64_t idx = NUM1; idx < validAivNum; ++idx) {
    int64_t start = sparseStartIdx[idx];
    if (start < totalSize && start > 0 && ((localValue[idx - 1] + sparseValidArray[start]) < maxVal)) {
      localValue[idx - 1] += sparseValidArray[start];
      localValue[idx] -= sparseValidArray[start];
      sparseStartIdx[idx] += NUM1;
    } else if (start == totalSize) {
      break;
    }
  }
  tmpMaxVal = *std::max_element(localValue.begin(), localValue.end());

  // 从后往前遍历
  for (int64_t idx = validAivNum - 1; idx > 0; --idx) {
    int64_t start = sparseStartIdx[idx];
    if (start == totalSize) {
      if (sparseStartIdx[idx - 1] == totalSize) {
        continue;
      }
      localValue[idx - 1] -= sparseValidArray[start - 1];
      localValue[idx] = sparseValidArray[start - 1];
      sparseStartIdx[idx] -= NUM1;
    } else if (start > 0) {
      if ((localValue[idx] + sparseValidArray[start - 1]) >= tmpMaxVal) {
        continue;
      }
      localValue[idx - 1] -= sparseValidArray[start - 1];
      localValue[idx] += sparseValidArray[start - 1];
      sparseStartIdx[idx] -= NUM1;
    } else {
      break;
    }
  }
  tmpMaxVal = *std::max_element(localValue.begin(), localValue.end());

  return (tmpMaxVal >= maxVal) ? false : true;
}

void IFATilingV2::InitLoadValue(const std::vector<int64_t>& sparseValidArray, int64_t totalSize, int64_t validAivNum,
                              const std::vector<int64_t>& sparseStartIdx, std::vector<int64_t>& localValue) const {
  for (int64_t idx = 0; idx < validAivNum; ++idx) {
    int64_t start = sparseStartIdx[idx];
    int64_t end = ((idx + 1) < validAivNum) ? sparseStartIdx[idx + 1] : totalSize;
    if (start < totalSize) {
      localValue[idx] = std::accumulate(sparseValidArray.begin() + start, sparseValidArray.begin() + end, 0);
    } else {
      break;
    }
  }
}

void IFATilingV2::SetSparseStartIdx(const std::vector<int64_t>& sparseValidArray, int64_t totalSize, int64_t validAivNum,
                                  uint32_t* sparseStartIdx, int64_t splitFactorSize) const {
  // initLoad: 使用均分策略, 保证后续不会比均分差
  std::vector<int64_t> localSparseStartIdx(MAX_CORE_NUM_REGBASE, totalSize);
  for (int64_t idx = 0; idx < MAX_CORE_NUM_REGBASE; ++idx) {
    localSparseStartIdx[idx] = std::min((idx * splitFactorSize), totalSize);
  }
  std::vector<int64_t> localValue(validAivNum, 0);
  InitLoadValue(sparseValidArray, totalSize, validAivNum, localSparseStartIdx, localValue);

  // 负载均衡粗调
  std::vector<int64_t> tmpLocalValue(validAivNum, 0);
  std::vector<int64_t> tmpsparseStartIdx(MAX_CORE_NUM_REGBASE, totalSize);
  int64_t sparseArraySum = std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0);
  int64_t avgVal = CeilDivision(sparseArraySum, validAivNum);

  tmpsparseStartIdx[0] = 0;
  for (int64_t idx = static_cast<int64_t>(static_cast<uint64_t>(NUM1)); idx < MAX_CORE_NUM_REGBASE; ++idx) {
    int64_t start = tmpsparseStartIdx[idx - 1];
    int64_t singleLoadValue = 0;
    tmpsparseStartIdx[idx] = start;
    while (singleLoadValue < avgVal && tmpsparseStartIdx[idx] < totalSize) {
      singleLoadValue += sparseValidArray[tmpsparseStartIdx[idx]];
      tmpsparseStartIdx[idx] += NUM1;
    }

    if ((start + 1) < tmpsparseStartIdx[idx]) {
      int64_t redoSingleLoadValue = singleLoadValue - sparseValidArray[tmpsparseStartIdx[idx] - 1];
      if ((singleLoadValue - avgVal) > (avgVal - redoSingleLoadValue)) {
        tmpsparseStartIdx[idx] = tmpsparseStartIdx[idx] - 1;
        singleLoadValue = redoSingleLoadValue;
      }
      sparseArraySum -= singleLoadValue;
      avgVal = CeilDivision(sparseArraySum, (validAivNum - idx));
    }
  }

  InitLoadValue(sparseValidArray, totalSize, validAivNum, tmpsparseStartIdx, tmpLocalValue);

  // 负载均衡精调
  while (BalanceLoad(sparseValidArray, totalSize, validAivNum, tmpLocalValue, tmpsparseStartIdx)) {
    // 根据负载均衡是否能得到更好预测结果决定是否结束循环
  }

  // exchange initLoad and 负载均衡
  if ((*std::max_element(localValue.begin(), localValue.end())) >
      (*std::max_element(tmpLocalValue.begin(), tmpLocalValue.end()))) {
    localSparseStartIdx.swap(tmpsparseStartIdx);
    localValue.swap(tmpLocalValue);
  }
  for (int64_t idx = 0; idx < MAX_CORE_NUM_REGBASE; ++idx) {
    sparseStartIdx[idx] = localSparseStartIdx[idx];
  }
}

void IFATilingV2::PromptFlashAttentionInitOutputSplit() {
  int64_t totalSize = ifaContext_->attenOut.shape->GetStorageShape().GetShapeSize();
  uint32_t singleCoreSize = (totalSize + coreNum_ - 1) / (coreNum_);
  if (enablePostQuant_) {
    // 2：In post quant scenario, when initializing, fill in 0 according to the half type,
    // requiring that the number of points allocated to each kernel must be even.
    singleCoreSize = ((singleCoreSize + 1) / 2) * 2; // 2 : fill in 0
  }
  singleCoreSize_ = singleCoreSize;
  tilingData_->outputParams.set_singleCoreSize(singleCoreSize);
  totalSize_ = totalSize;
  tilingData_->outputParams.set_totalOutputSize(totalSize);
  if (softmaxLseFlag_) {
    int64_t totalLseSize = ifaContext_->lseOut.shape->GetStorageShape().GetShapeSize();
    totalSizeLse_ = totalLseSize;
    uint32_t singleCoreLseSize = (totalLseSize + coreNum_ - 1) / (coreNum_);
    tilingData_->outputParams.set_singleCoreLseSize(singleCoreLseSize);
    tilingData_->outputParams.set_totalLseOutputSize(totalLseSize);
  }
}

void IFATilingV2::GetActualSeqLength(int64_t &actualSeqLengths, int64_t &actualSeqLengthsKV, uint32_t bIdx) {
  if (isMaxWorkspace_) {
    actualSeqLengths = sOfQuery_;
    actualSeqLengthsKV = seqSize_;
    return;
  }
  if (inputLayout_ == IfaLayout::TND) {
    actualSeqLengths = bIdx == 0 ? ifaContext_->actualSeqLengthsQ.tensor->GetData<int64_t>()[0] :
      ifaContext_->actualSeqLengthsQ.tensor->GetData<int64_t>()[bIdx] - ifaContext_->actualSeqLengthsQ.tensor->GetData<int64_t>()[bIdx - 1];
    if (faRunGS_) {
      actualSeqLengths *= nNumOfQInOneGroup_;
    }
    actualSeqLengthsKV = ifaContext_->actualSeqLengths.tensor->GetData<int64_t>()[bIdx];
    if (!pageAttentionFlag_ && bIdx > 0) {
      actualSeqLengthsKV -= ifaContext_->actualSeqLengths.tensor->GetData<int64_t>()[bIdx - 1];
    }
  } else {
    if (actualSeqLenFlag_  && actualLenDims_ > 0 && ifaContext_->actualSeqLengths.tensor->GetData<int64_t>() != nullptr) { // kvLengths
      actualSeqLengthsKV = actualLenDims_ == NUM1 ? ifaContext_->actualSeqLengths.tensor->GetData<int64_t>()[0] :
        ifaContext_->actualSeqLengths.tensor->GetData<int64_t>()[bIdx];
    } else {
      actualSeqLengthsKV = kvListSeqLens_.size() == NUM1 ? kvListSeqLens_[0] : kvListSeqLens_[bIdx];
    }
    if (actualSeqLengthsKV < seqSize_) {
      needInit_ = true;
    }
    if (actualSeqLenQFlag_) { // qLengths
      actualSeqLengths = actualLenQDims_ == NUM1 ? ifaContext_->actualSeqLengthsQ.tensor->GetData<int64_t>()[0] :
        ifaContext_->actualSeqLengthsQ.tensor->GetData<int64_t>()[bIdx];
    } else {
      actualSeqLengths = sOfQuery_;
      if (faRunGS_) {
        actualSeqLengths = sOfQuery_ * nNumOfQInOneGroup_;
      }
    }
  }
}

void IFATilingV2::GetPreNextTokensLeftUp(int64_t actualSeqLength, int64_t actualSeqLengthKV,
  int64_t& preTokensLeftUp, int64_t& nextTokensLeftUp) const {
  preTokensLeftUp = preToken_;
  nextTokensLeftUp = nextToken_;
  if (sparseMode_ == SPARSE_MODE_RIGHT_DOWN) {
    nextTokensLeftUp = actualSeqLengthKV - actualSeqLength;
  } else if (sparseMode_ == SPARSE_MODE_BAND) {
    preTokensLeftUp = preToken_ - actualSeqLengthKV + actualSeqLength;
    nextTokensLeftUp = nextToken_ + actualSeqLengthKV - actualSeqLength;
  }
}

void IFATilingV2::FixParamWithRowInvalid(int64_t& actualSeqLength, int64_t actualSeqLengthKV,
  int64_t& preTokensLeftUp, int64_t& nextTokensLeftUp) {
  // 若出现行无效，需要重新计算nexttokens，pretokens，actualseqlen，以便正确计算分核核数
  int64_t nextTokensError = (nextTokensLeftUp < 0) ? -nextTokensLeftUp : 0;
  nextTokensError = nextTokensError > actualSeqLength ? actualSeqLength : nextTokensError;
  int64_t preTokensError = (actualSeqLength > actualSeqLengthKV + preTokensLeftUp) ?
    (actualSeqLength - actualSeqLengthKV - preTokensLeftUp) : 0;
  preTokensError = preTokensError > actualSeqLength ? actualSeqLength : preTokensError;

  // 若出现上方行无效，需要重新计算nexttokens，pretokens，actualseqlen
  nextTokensLeftUp += nextTokensError;
  preTokensLeftUp -= nextTokensError;
  actualSeqLength -= nextTokensError;

  // 若出现下方行无效，需要重新计算actualseqlen
  actualSeqLength -= preTokensError;
  if (nextTokensError != 0 || preTokensError != 0) {
    needInit_ = true;
  }
}
 
int64_t IFATilingV2::SumOfArithmeticSeries(int64_t an, int64_t d) const {
  // 等差数列求和，an：等差数列第n项，d：等差数列公差
  if (d == 0) {
    return 0;
  }
  return (an > 0) ? (an % d + an) * (an / d + 1) / 2 : 0; // 2: 等差数列求和公式分母
}

int64_t IFATilingV2::GetCutBlockNums(int64_t blockSeqLengthKV, int64_t blockSeqLength,
  int64_t sInner, int64_t sOuter, int64_t token) const {
  int64_t blockNums = 0;
  int64_t blockToken = token > 0 ? ((token + sInner - 1) / sInner * sInner) : (token / sInner * sInner);
  int64_t outDivIn = sOuter > sInner ? sOuter / sInner : 1;
  int64_t InDivOut = sInner > sOuter ? sInner / sOuter : 1;
  int64_t tolerance = 0;
  int64_t smallSize = 0;
  if (outDivIn >= static_cast<int64_t>(NUM1)) {
    tolerance = outDivIn;
    smallSize = sInner;
  } else {
    tolerance = InDivOut;
    smallSize = sOuter;
  }

  // nextToken与上边右边构成的大三角形
  int64_t innerCutBlockNums = (blockSeqLengthKV - blockToken) / smallSize - tolerance;
  blockNums += SumOfArithmeticSeries(innerCutBlockNums, tolerance);

  // nextToken与上边左边构成的左侧三角形，需要减去
  int64_t innerCutBlockLeftNums = -blockToken / smallSize - tolerance;
  blockNums -= SumOfArithmeticSeries(innerCutBlockLeftNums, tolerance);

  // nextToken与下边右边构成的下侧三角形，需要减去
  int64_t innerCutBlockDownNums = (blockSeqLengthKV - blockSeqLength - blockToken) / smallSize - tolerance;
  blockNums -= SumOfArithmeticSeries(innerCutBlockDownNums, tolerance);

  // nextToken与下边左边构成的小三角形，是前两个三角形的重叠部分，需要加上
  int64_t innerCutBlockLeftDownNums = (-blockToken - blockSeqLength) / smallSize - tolerance;
  blockNums += SumOfArithmeticSeries(innerCutBlockLeftDownNums, tolerance);
  return blockNums;
}

int64_t IFATilingV2::GetCalcBlockNumsOneHead(int64_t outerBlockNums, int64_t innerBlockNums, int64_t sInnerLoopTimesPrefix,
  int64_t preTokensLeftUp, int64_t nextTokensLeftUp) const {
  if (!attenMaskFlag_) {
    return (innerBlockNums + sInnerLoopTimesPrefix) * outerBlockNums;
  } else {
    int64_t blockSeqLength = outerBlockNums * static_cast<int64_t>(sOuterSize_);
    int64_t blockSeqLengthKV = innerBlockNums * static_cast<int64_t>(sInnerSize_);
    int64_t toCalcBlockNums = innerBlockNums * outerBlockNums;
    // 必须满足pretoken + nexttoken > 0，否则会减出小于0的块数，这里需要去除prefix影响
    toCalcBlockNums -= GetCutBlockNums(blockSeqLengthKV, blockSeqLength, static_cast<int64_t>(sInnerSize_),
      static_cast<int64_t>(sOuterSize_), nextTokensLeftUp - actualSharedPrefixLen_);
    toCalcBlockNums -= GetCutBlockNums(blockSeqLengthKV, blockSeqLength, static_cast<int64_t>(sInnerSize_),
      static_cast<int64_t>(sOuterSize_), blockSeqLengthKV - blockSeqLength + preTokensLeftUp + actualSharedPrefixLen_);

    // prefix部分单独计算
    int64_t blockSharedPrefix = sInnerLoopTimesPrefix * static_cast<int64_t>(sInnerSize_);
    toCalcBlockNums += sInnerLoopTimesPrefix * outerBlockNums;
    toCalcBlockNums -= GetCutBlockNums(blockSharedPrefix, blockSeqLength, static_cast<int64_t>(sInnerSize_),
      static_cast<int64_t>(sOuterSize_), nextTokensLeftUp);
    toCalcBlockNums -= GetCutBlockNums(blockSharedPrefix, blockSeqLength, static_cast<int64_t>(sInnerSize_),
      static_cast<int64_t>(sOuterSize_), blockSharedPrefix - blockSeqLength + preTokensLeftUp);
    return toCalcBlockNums;
  }
}

int64_t IFATilingV2::GetActualInnerBlockNums(int64_t sInnerIndexStart, int64_t sInnerIndexEnd,
  int64_t innerBlockNums) const {
  int64_t sInnerBlockNums = 0;

  if (sInnerIndexEnd < 0) {
    sInnerBlockNums = 0;
  } else if (sInnerIndexEnd < innerBlockNums) {
    sInnerBlockNums = (sInnerIndexStart < 0) ? (sInnerIndexEnd + 1) : (sInnerIndexEnd - sInnerIndexStart + 1);
  } else {
    sInnerBlockNums = (sInnerIndexStart < 0) ? innerBlockNums :
      (sInnerIndexStart < innerBlockNums ? innerBlockNums - sInnerIndexStart : 0);
  }

  return sInnerBlockNums;
}

void IFATilingV2::ComputeSplitNBSeqfaRun(std::vector<int64_t> sOuterLoopTimes, std::vector<int64_t> sInnerLoopTimes,
  int64_t sInnerLoopTimesPrefix, double coreWightTarget, uint32_t& curCore, const size_t tilingElementArrayLen) {
  int64_t SplitNumHeads = numHeads_;
  if (faRunGS_) {
    SplitNumHeads = numKvHeads_;
  }
  std::vector<int64_t> sparseStartIdx(tilingElementArrayLen, 0L);
  std::vector<uint32_t> bnStartIdx(tilingElementArrayLen, 0U);
  std::vector<int64_t> gS1StartIdx(tilingElementArrayLen, 0L);
  // Temporary algorithm to be optimized
  int64_t curWight = 0;
  uint32_t tmpCoreNidEnd = 0; // actual seq为0时不分配核
  uint32_t tmpCoreSidEnd = 0;
  uint32_t tmpCoreSposEnd = 0;
  int64_t actualSeqLengths = 0;
  int64_t actualSeqLengthsKV = 0;
  for (uint32_t sIdx = 0; sIdx < batchSize_; sIdx++) {   
    GetActualSeqLength(actualSeqLengths, actualSeqLengthsKV, sIdx); 
    for (uint32_t headNum = 0; headNum < SplitNumHeads; headNum++) {
      int64_t preTokensLeftUp = 0;
      int64_t nextTokensLeftUp = 0;
      GetPreNextTokensLeftUp(actualSeqLengths, actualSeqLengthsKV + actualSharedPrefixLen_, preTokensLeftUp, nextTokensLeftUp);
      FixParamWithRowInvalid(actualSeqLengths, actualSeqLengthsKV + actualSharedPrefixLen_, preTokensLeftUp, nextTokensLeftUp);
      int64_t outerBlockNums = sOuterLoopTimes[sIdx];
      int64_t innerBlockNums = sInnerLoopTimes[sIdx];
      for (uint32_t sOuterIndex = 0; sOuterIndex < outerBlockNums; sOuterIndex++) {
        int64_t dif = static_cast<int64_t>(coreWightTarget * double(curCore + 1)) - curWight;
        // 非prefix部分计算，去除prefix影响
        int64_t preTokensNoPrefix = preTokensLeftUp + actualSharedPrefixLen_;
        int64_t nextTokensNoPrefix = nextTokensLeftUp - actualSharedPrefixLen_;
        int64_t sInnerIndexStart = -(preTokensNoPrefix > 0 ? (preTokensNoPrefix + static_cast<int64_t>(sInnerSize_) - 1) /
          static_cast<int64_t>(sInnerSize_) : preTokensNoPrefix / static_cast<int64_t>(sInnerSize_));
        int64_t sInnerIndexEnd = nextTokensNoPrefix > 0 ? (nextTokensNoPrefix + static_cast<int64_t>(sInnerSize_) - 1) /
          static_cast<int64_t>(sInnerSize_) : nextTokensNoPrefix / static_cast<int64_t>(sInnerSize_);
        
        // prefix部分单独计算
        int64_t sInnerIndexStartPrefix = -(preTokensLeftUp > 0 ? (preTokensLeftUp + static_cast<int64_t>(sInnerSize_) - 1) /
          static_cast<int64_t>(sInnerSize_) : preTokensLeftUp / static_cast<int64_t>(sInnerSize_));
        int64_t sInnerIndexEndPrefix = nextTokensLeftUp > 0 ? (nextTokensLeftUp + static_cast<int64_t>(sInnerSize_) - 1) /
          static_cast<int64_t>(sInnerSize_) : nextTokensLeftUp / static_cast<int64_t>(sInnerSize_);
 
        // 当前这一行有多少基本块需要计算
        int64_t actualInnerBlockNums = GetActualInnerBlockNums(sInnerIndexStart, sInnerIndexEnd, innerBlockNums) +
          GetActualInnerBlockNums(sInnerIndexStartPrefix, sInnerIndexEndPrefix, sInnerLoopTimesPrefix);
        if (actualInnerBlockNums - dif > dif && !(tmpCoreNidEnd == 0 && tmpCoreSidEnd == 0 && tmpCoreSposEnd == 0)) {
          curCore += 1;
          bnStartIdx[curCore] = sIdx * SplitNumHeads + headNum;
          gS1StartIdx[curCore] = sOuterIndex;
        }
        tmpCoreNidEnd = headNum + 1;
        tmpCoreSidEnd = sIdx + 1;
        tmpCoreSposEnd = sOuterIndex + 1;

        curWight += actualInnerBlockNums;
        preTokensLeftUp -= sOuterSize_;
        nextTokensLeftUp += sOuterSize_;
      }
    }
  }
  bnStartIdx[curCore + 1] = batchSize_ * SplitNumHeads;
  gS1StartIdx[curCore + 1] = tmpCoreSposEnd;

  faRunTilingAdapter.multiCoreParamsRegbase.set_bnStartIdx(bnStartIdx.data());
  faRunTilingAdapter.multiCoreParamsRegbase.set_sparseStartIdx(gS1StartIdx.data());
}

void IFATilingV2::FlashAttentionCubeSplitBNSeq()   //这里我们只用Cube视角分核
{
  //注意：实际上我们生效新模板时，是不带mask的，这里预留了mask的计算功能
  int64_t SplitNumHeads = numHeads_;
  if (faRunGS_) {
    SplitNumHeads = numKvHeads_;
  }
  int64_t totalBlockNumsOneHead = 0;
  std::vector<int64_t> sOuterLoopTimes(batchSize_, 0U);
  std::vector<int64_t> sInnerLoopTimes(batchSize_, 0U);
  int64_t multiSmaxsInnerLoopTimes = 0U;
  int64_t sInnerLoopTimesPrefix = (actualSharedPrefixLen_ + static_cast<int64_t>(sInnerSize_) - 1) / static_cast<int64_t>(sInnerSize_);
  for (uint32_t bIdx = 0; bIdx < batchSize_; bIdx++) {
    int64_t actualSeqLengths = 0;
    int64_t actualSeqLengthsKV = 0;
    GetActualSeqLength(actualSeqLengths, actualSeqLengthsKV, bIdx);
    int64_t preTokensLeftUp = 0;
    int64_t nextTokensLeftUp = 0;
    GetPreNextTokensLeftUp(actualSeqLengths, actualSeqLengthsKV + actualSharedPrefixLen_, preTokensLeftUp, nextTokensLeftUp);
    FixParamWithRowInvalid(actualSeqLengths, actualSeqLengthsKV + actualSharedPrefixLen_, preTokensLeftUp, nextTokensLeftUp);

    sOuterLoopTimes[bIdx] = (actualSeqLengths + static_cast<int64_t>(sOuterSize_) - 1) / static_cast<int64_t>(sOuterSize_);
    sInnerLoopTimes[bIdx] = (actualSeqLengthsKV + static_cast<int64_t>(sInnerSize_) - 1) / static_cast<int64_t>(sInnerSize_);
    multiSmaxsInnerLoopTimes = std::max(multiSmaxsInnerLoopTimes, sInnerLoopTimes[bIdx] + sInnerLoopTimesPrefix);

    totalBlockNumsOneHead += GetCalcBlockNumsOneHead(sOuterLoopTimes[bIdx], sInnerLoopTimes[bIdx], sInnerLoopTimesPrefix,
      preTokensLeftUp, nextTokensLeftUp);
  }
  double coreWightTarget = (double(totalBlockNumsOneHead * SplitNumHeads) / double(aicNum_));

  int64_t s1OuterSize = (sOfQuery_ + sOuterSize_ - 1) / sOuterSize_;
  faRunTilingAdapter.multiCoreParamsRegbase.set_s1OuterSize(s1OuterSize);
  const size_t tilingElementArrayLen = (static_cast<size_t>(aicNum_) > 64UL) ? static_cast<size_t>(aicNum_) : 64UL;
  uint32_t curIndx = 0;
  ComputeSplitNBSeqfaRun(sOuterLoopTimes, sInnerLoopTimes, sInnerLoopTimesPrefix, coreWightTarget, curIndx, tilingElementArrayLen);
  int64_t sinnerBlocknum = (sMax_ + sInnerSize_ - 1) / sInnerSize_;
  SetMultiCoreParamsRegbase((totalBlockNumsOneHead / sinnerBlocknum) * SplitNumHeads, static_cast<int64_t>((curIndx + 1)));
  if (needInit_) {
    PromptFlashAttentionInitOutputSplit();
  }
}

void IFATilingV2::SetMultiCoreParamsRegbase(int64_t totalSize, int64_t actualUsedCoreNum)
{
    faRunTilingAdapter.multiCoreParamsRegbase.set_coreNum(static_cast<int32_t>(actualUsedCoreNum));
    faRunTilingAdapter.multiCoreParamsRegbase.set_totalSize(totalSize);
    faRunTilingAdapter.multiCoreParamsRegbase.set_splitFactorSize(CeilDivision(totalSize, actualUsedCoreNum));
    faRunTilingAdapter.multiCoreParamsRegbase.set_splitFactorTailSize(CalcTailSizefaRun(totalSize, faRunTilingAdapter.multiCoreParamsRegbase.get_splitFactorSize()));
}

ge::graphStatus IFATilingV2::SplitBN_V0() {
  uint32_t bn = batchSize_ * numKvHeads_;
  formerCoreNum_ = bn % coreNum_;
  if (formerCoreNum_ == 0) {
    blockSplitBn2Range_ = bn / coreNum_;
    tailSplitedBatchRange_ = blockSplitBn2Range_;
  } else {
    blockSplitBn2Range_ = bn / coreNum_ + 1;
    tailSplitedBatchRange_ = blockSplitBn2Range_ - 1;
  }

  usedCoreNum_ = bn > coreNum_ ? coreNum_ : bn;

  for (uint32_t i = 0; i < formerCoreNum_; i++) {
    startIdxEachCore_[i] = blockSplitBn2Range_ * i;
  }

  uint32_t formerBase = formerCoreNum_ * blockSplitBn2Range_;
  for (uint32_t i = formerCoreNum_; i < usedCoreNum_; i++) {
    startIdxEachCore_[i] = formerBase + tailSplitedBatchRange_ * (i - formerCoreNum_);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATilingV2::SplitBNSfaRun() {
  uint64_t bng = batchSize_ * numKvHeads_ * (nNumOfQInOneGroup_ + sOuterSize_ - 1) / sOuterSize_;
  uint64_t headDimAlign = AlignUp(headDim_, BYTE_BLOCK);
  uint32_t kvSplitLimit = sInnerSize_ <= 256U ? 256U : sInnerSize_; // 256: 经验值  这里的门限值也需要调整，否则会分的很碎，每个核分到的还不到一个基本块
  kvSplitPart_ = aicNum_ / bng;

  while(((maxActualseq_ / kvSplitPart_) < kvSplitLimit) && (kvSplitPart_ > 1)) { // 512, 经验值
    kvSplitPart_--;
  }

  faRunTilingAdapter.inputParamsRegbase.set_kvSplitPart(kvSplitPart_);
  faRunTilingAdapter.inputParamsRegbase.set_accumOutSize(batchSize_ * numHeads_ * kvSplitPart_ * headDimAlign);
  faRunTilingAdapter.inputParamsRegbase.set_logSumExpSize(batchSize_ * numHeads_ * kvSplitPart_ * (BYTE_BLOCK / sizeof(float)));
  return ge::GRAPH_SUCCESS;
}

void IFATilingV2::SetfaRunBaseSize()
{
  sOuterSize_ = NUM16;
  if (sOfQuery_ > NUM1 && isGqa_) { // pfa gs1合轴时 s1base=32
      sOuterSize_ = NUM32;
  }
  if (headDim_ <= NUM64) {
    sInnerSize_ = NUM1024;
    if (pseShiftFlag_ || (sOfQuery_ > NUM1 && isGqa_)) { // pfa gs1合轴 s1base=32 s2base=512 dbase=64
      sInnerSize_ = NUM512;
    }
  } else if (headDim_ <= NUM128) {
    sInnerSize_ = NUM512;
  } else if (headDim_ <= NUM256) {
    sInnerSize_ = NUM256;
  } else {
    sInnerSize_ = NUM128;
  }
}

ge::graphStatus IFATilingV2::CalcInnerSize(uint32_t seqSize) {
  /**
   * sInnerSize：s2的切分大小，直接决定了MM的singleN/K和vector的切块大小，但当前切分也并非适用所有case。
   * 1、非GQA场景：按照vector的最大基本块8*1024进行切分，sInnerSize=8192
   * 2、GQA场景：(1) 非伪量化：vector计算比较少，cube MTE2bound，
   *                          因此，cube发射大块，减少通信次数。sInnerSize=8192
   *            (2) 伪量化：vector比较重，尽量较少vector的循环次数,
   *                          因此，cube发小块，期望vector尽量被cube的mte2掩盖。sInnerSize=1024
   */
  if (socVersion_ == IfaSocVersion::SOC_ASCEND_950) {
    SetfaRunBaseSize();
    sInnerSize2_ = sInnerSize_;
  } else {
    sInnerSize_ = MAX_SPLIT_SIZE;  // 8192
    if (antiQuantFlag_ && nNumOfQInOneGroup_ > 1) {
      sInnerSize_ = NUM1024;
    }
    // GS 不能超过32K，按照32K反推S2切块大小。 需要向下对齐。
    // BMM2需要local workspac :Align(nNumOfQInOneGroup_, 16) * Align(S, 16) =32K
    sInnerSize_ = (NUM32 * NUM1024 / sizeof(float) / ((nNumOfQInOneGroup_ + 15) / NUM16 * NUM16)); // 32*1024 Byte, 15/16 align
    sInnerSize_ = sInnerSize_ / BYTE_BLOCK * BYTE_BLOCK;
  }
  sInnerLoopTimes_ = (seqSize + sInnerSize_ - 1) / sInnerSize_;
  sInnerSizeTail_ = seqSize - (sInnerLoopTimes_ - 1) * sInnerSize_;
  if (sInnerSize_ > seqSize) {
      sInnerSize_ = seqSize;
  }
  sInnerSizeAlign_ = AlignUp(sInnerSize_, BYTE_BLOCK); // 元素个数按照基本块大小对齐

  CheckUbSpace();
  return ge::GRAPH_SUCCESS;
}

bool IFATilingV2::CheckWorkSpace() const {
  return true;
}

ge::graphStatus IFATilingV2::CheckUbSpace() {
  if (!CalcUbBmm() || !CalcUbSoftMax() || !CalcUbAttenMask() || !CalcUbPageAttention()) {
    return false;
  }
  return true;
}

bool IFATilingV2::CalcUbBmm() {
  mmResUbSize_ = msdIterNum_ * nNumOfQInOneGroup_ * sInnerSizeAlign_;
  bmm2ResUbSize_ = msdIterNum_ * nNumOfQInOneGroup_ * headDimAlign_;
  return true;
}

bool IFATilingV2::CalcUbSoftMax() {
  auto tmpShape = Shape({nNumOfQInOneGroup_, AlignUp(sInnerSize_, BYTE_BLOCK / blockTypeSize_)});
  softmaxFlashTmpSize_ = GetSoftMaxFlashV2MinTmpSize(tmpShape, blockTypeSize_, blockTypeSize_, true, false);

  return true;
}

bool IFATilingV2::CalcUbAttenMask() {
  if (!attenMaskFlag_) {
    selectWithByteMaskTmpMinSize_ = 0;
    return true;
  }
  // bool/int8/uint8类型的mask，每个占1字节
  attenMaskTypeSize_ = NUM1; // 1:sizeof(bool)
  auto selectWithByteMaskTmpShape = Shape({msdIterNum_ * nNumOfQInOneGroup_,
                                          AlignUp(sInnerSize_, BYTE_BLOCK / attenMaskTypeSize_)});
  selectWithByteMaskTmpMinSize_ = GetSelectWithBytesMaskMinTmpSize(selectWithByteMaskTmpShape, Shape({1, 1}),
      FP32_BYTES, selectWithByteMaskTmpShape, FP32_BYTES, false);

  return true;
}

bool IFATilingV2::CalcUbQuant() {
  if (!quantFlag_) {
    return true;
  }

  auto srcShape = Shape({1, AlignUp(sInnerSize_, BYTE_BLOCK / blockTypeSize_)});
  uint32_t minSize = 0;
  uint32_t maxSize = 0;
  GetAscendQuantMaxMinTmpSize(srcShape, BYTE_BLOCK / blockTypeSize_, maxSize, minSize);
  quantUbSize_ = AlignUp(minSize, BYTE_BLOCK);
  return true;
}

bool IFATilingV2::CalcUbPageAttention() {
  if (!pageAttentionFlag_) {
    kvPageResUbSize_ = 0;
    return true;
  }

  return false;
}

bool IFATilingV2::CalcUbDeQuant() const {
  return true;
}
bool IFATilingV2::CalcUbAntiQuant() const{
  return true;
}

bool IFATilingV2::CalcUbKvSplit() const {
  return true;
}

ge::graphStatus IFATilingV2::CalcWorkSpace() {
  uint32_t cubeL1UbSize = (NUM512 / 2) * NUM1024; // david L1 512K,提供给两个Vec使用,单个vector占用256K
  uint32_t cubeL0CUbSize = (NUM256 / 2) * NUM1024; // david L0C 256K,提供给两个Vec使用,单个vector占用128K

  uint32_t mmResElemSize = 4; // 4:fp32
  uint32_t vec1ResElemSize = 2; // 2:fp16/bf16
  uint32_t bmm2ResElemSize = 4; // 4:fp32
  uint32_t vec2ResElemSize = 4; // 4:fp32
  uint32_t qPreProcResElemSize = NUM1; // 当前tiling仅涉及伪量化场景
  uint32_t mmPACallBackDataSize = NUM64; // 64: matmul回调信息需要7个uint32值，dcci cacheline需要64B对齐
  workspaceSize_ = libapiSize_;
  if (perfMode_ != IfaPerfMode::BMM_ALL_BY_VEC) {
    workspaceSize_ += mmResUbSize_ * coreNum_ * mmResElemSize;
    workspaceSize_ += mmResUbSize_ * coreNum_ * vec1ResElemSize;
    workspaceSize_ += bmm2ResUbSize_ * coreNum_ * bmm2ResElemSize;
    workspaceSize_ += bmm2ResUbSize_ * coreNum_ * vec2ResElemSize;
    workspaceSize_ += bmm2ResUbSize_ * coreNum_ * qPreProcResElemSize;
  }
  // L1
  workspaceSize_ += cubeL1UbSize * coreNum_;
  // L0C
  workspaceSize_ += cubeL0CUbSize * coreNum_;
  if (isMaxWorkspace_) { // 计算maxWorkSpaceSize时默认开启FD且使用最大核数进行归约
    uint32_t maxAccumOutSize = aicNum_ * headDimAlign_;
    uint32_t maxLogSumExpSize = aicNum_ * (BYTE_BLOCK / sizeof(float));
    workspaceSize_ += (maxAccumOutSize + maxLogSumExpSize * 2) * blockTypeSize_;  // 2 : sMax 和 sSum
  } else if (splitKVFlag_) {
    workspaceSize_ += (tilingData_->splitKVParams.get_accumOutSize() +
                       tilingData_->splitKVParams.get_logSumExpSize() * 2) * blockTypeSize_;  // 2 : sMax 和 sSum
  }
  if (socVersion_ == IfaSocVersion::SOC_ASCEND_950) {
    workspaceSize_ += NUM100 * NUM1024 * NUM1024; // 100*1024*1024: extra workspace for dump in david
  }
  if (pageAttentionFlag_) {
    workspaceSize_ += coreNum_ * mmPACallBackDataSize * 2; // bmm1 bmm2 2份
  }
  if (ifaContext_->workSpaces) {
    ifaContext_->workSpaces[0] = workspaceSize_;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATilingV2::FillTiling() {
  FillTilingBaseParams();
  FillTilingSplitKV();
  FillTilingCoreParams();
  FillTilingSingleCoreParams();
  FillTilingSingleCoreTensorSize();
  FillTilingSoftmax();
  FillTilingTranspose();
  FillTilingOutputParams();
  return FillTilingBmm() ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

void IFATilingV2::FillTilingBaseParams() const {
  tilingData_->baseParams.set_batchSize(batchSize_);
  tilingData_->baseParams.set_seqSize(sMax_);
  tilingData_->baseParams.set_qSeqSize(sOfQuery_);
  tilingData_->baseParams.set_headSize(headDim_);
  tilingData_->baseParams.set_blockSize(blockSize_);
  tilingData_->baseParams.set_maxBlockNumPerSeq(maxBlockNumPerSeq_);
  tilingData_->baseParams.set_scaleValue(scaleValue_);
  tilingData_->baseParams.set_kvHeadNum(numKvHeads_);
  tilingData_->baseParams.set_qHeadNum(numHeads_);
  if (sOfQuery_ == NUM1) {
    tilingData_->baseParams.set_headNumRatio(1);
    tilingData_->baseParams.set_nNumOfQInOneGroup(numHeads_ / numKvHeads_);
  } else {
    tilingData_->baseParams.set_headNumRatio(numHeads_ / numKvHeads_);
    tilingData_->baseParams.set_nNumOfQInOneGroup(1);
  }
  tilingData_->baseParams.set_batchContinuousFlag(batchContinuousFlag_);
  tilingData_->baseParams.set_pseShiftB(pseShiftBatch_);
  tilingData_->baseParams.set_pseShiftS0(pseShiftS0_);
  tilingData_->baseParams.set_pseShiftS(pseShiftS1_);
  tilingData_->baseParams.set_selectWithByteMaskTmpMinSize(selectWithByteMaskTmpMinSize_);  // mask

  tilingData_->baseParams.set_actualLenDims(actualLenDims_);
  tilingData_->baseParams.set_actualLenQDims(actualLenQDims_);
  tilingData_->baseParams.set_msdIterNum(msdIterNum_);
  tilingData_->baseParams.set_qPaddingFlag(qPaddingSizeFlag_ ? 1 : 0);
  tilingData_->baseParams.set_kvPaddingFlag(kvPaddingSizeFlag_ ? 1 : 0);
  tilingData_->baseParams.set_antiquantPerTensorFlag(antiquantPerTensorFlag_);
  tilingData_->baseParams.set_antiquantPerHeadFlag(antiquantPerHeadFlag_);
  tilingData_->baseParams.set_attenMaskBatch(attenMaskBatch_);
  tilingData_->baseParams.set_attenMaskQSize(attenMaskQSize_);
  tilingData_->baseParams.set_attenMaskSize(attenMaskKvSize_);
  tilingData_->baseParams.set_sparseMode(sparseMode_);
  tilingData_->baseParams.set_preToken(preToken_);
  tilingData_->baseParams.set_nextToken(nextToken_);
  tilingData_->baseParams.set_isRowInvalid(isRowInvalid_);
  tilingData_->baseParams.set_l2CacheOffFlag(l2CacheOffFlag_);
  tilingData_->baseParams.set_softmaxLseFlag(softmaxLseFlag_); // whether return lse
  tilingData_->baseParams.set_totalBlockNum(totalBlockNum_);
  tilingData_->baseParams.set_paKvShapeType(static_cast<uint32_t>(pageAttentionKvLayoutType_));
  tilingData_->baseParams.set_antiqSeqSize(antiquantParaSeqSize_);
}

// for flash decode
void IFATilingV2::FillTilingSplitKV() const {
  tilingData_->splitKVParams.set_s2(kvSplitPart_);
  tilingData_->splitKVParams.set_sInnerLoopSize((maxActualseq_ + (kvSplitPart_ - 1)) / kvSplitPart_);
  tilingData_->splitKVParams.set_accumOutSize(batchSize_ * numHeads_ * kvSplitPart_ * headDimAlign_);
  tilingData_->splitKVParams.set_logSumExpSize(batchSize_ * numHeads_ * kvSplitPart_ * (BYTE_BLOCK / blockTypeSize_));
}

void IFATilingV2::FillTilingCoreParams() const {
  uint32_t* coreStartIdx = tilingData_->increFlashAttentionCoreParams.get_coreSidxEndRegbasePtr();
  memcpy_s(coreStartIdx, MAX_CORE_NUM_REGBASE * sizeof(uint32_t), startIdxEachCore_, MAX_CORE_NUM_REGBASE * sizeof(uint32_t));
  uint32_t* coreSposStartIdx = tilingData_->increFlashAttentionCoreParams.get_coreSposStartRegbasePtr();
  memcpy_s(coreSposStartIdx, MAX_CORE_NUM_REGBASE * sizeof(uint32_t), coreSposStart_, MAX_CORE_NUM_REGBASE * sizeof(uint32_t));
}

void IFATilingV2::FillTilingSingleCoreParams() {
  tilingData_->increFlashAttentionSingleCoreParams.set_sInnerLoopTimes(sInnerLoopTimes_);
  tilingData_->increFlashAttentionSingleCoreParams.set_singleProcessSInnerSize(sInnerSize_);
  tilingData_->increFlashAttentionSingleCoreParams.set_singleProcessSInnerSizeTail(sInnerSizeTail_);
  tilingData_->increFlashAttentionSingleCoreParams.set_usedCoreNum(usedCoreNum_);
  tilingData_->increFlashAttentionSingleCoreParams.set_formerCoreNum(formerCoreNum_);
  tilingData_->increFlashAttentionSingleCoreParams.set_blockSplitBn2Range(blockSplitBn2Range_);
  tilingData_->increFlashAttentionSingleCoreParams.set_tailSplitedBatchRange(tailSplitedBatchRange_);
}

void IFATilingV2::FillTilingSingleCoreTensorSize() const {
  tilingData_->increFlashAttentionSingleCoreTensorSize.set_mmResUbSize(mmResUbSize_);
  tilingData_->increFlashAttentionSingleCoreTensorSize.set_bmm2ResUbSize(bmm2ResUbSize_);
}

void IFATilingV2::FillTilingSoftmax() const {
  auto softmaxShape = Shape({1, AlignUp(sInnerSize_, BYTE_BLOCK / blockTypeSize_)});
  SoftMaxFlashV2TilingFunc(softmaxShape, blockTypeSize_, blockTypeSize_, softmaxFlashTmpSize_,
                           tilingData_->softmaxFlashTilingData, true, false);
}

void IFATilingV2::FillTilingTranspose() const {
}

// for zero output
void IFATilingV2::FillTilingOutputParams() const {
  std::string layout(ifaContext_->layOut);
  tilingData_->outputParams.set_needInit(needInit_);
  tilingData_->outputParams.set_isBSNDOut(layout == "BNSD_BSND");
}

void IFATilingV2::AdjustPABmm1Tiling(uint32_t& bmm1BaseN) const {
  if (bmm1BaseN < blockSize_) {
    while (blockSize_ % bmm1BaseN != 0) {
      bmm1BaseN /= 2; // 2:不断减半，确保1个base块不会跨block拷贝。已校验过blockSize 16/32对齐，因此bmm1BaseN最小值为16/32
    }
  } else if (bmm1BaseN > blockSize_) {
    // nd2nz拷贝时ndnum>1场景性能较差，通过设置baseN <= blocksize避免
    uint32_t tmpBaseN = increGcd(bmm1BaseN, blockSize_);
    bmm1BaseN = tmpBaseN;
  }
  OP_LOGD(ifaContext_->opName, "Page attention is enabled, blockSize is %u, bmm1 baseN is adjusted to %u.", blockSize_, bmm1BaseN);
}

void IFATilingV2::AdjustPABmm2Tiling() const {
  uint32_t targetBaseK = NUM128;
  if (targetBaseK < blockSize_) {
    while ((blockSize_ % targetBaseK != 0) || (targetBaseK * tilingData_->bmm2TilingData.get_baseN() * sizeof(float) > L0B_SIZE)) {
      targetBaseK /= 2; // 2:不断减半，确保1个base块不会跨block拷贝，已校验过blockSize_16/32对齐，因此targetBaseK最小值为16/32
    }
  } else {
    uint32_t tmpBaseK = increGcd(targetBaseK, blockSize_);
    while(tmpBaseK * tilingData_->bmm2TilingData.get_baseN() * sizeof(float) > L0B_SIZE) {
      tmpBaseK /= 2; // 2: 不断减半，确保base块大小在LOB有效范围内
    }
    targetBaseK = tmpBaseK;
  }
  // mm api不支持通过 SetFixSplit 设置baseK，需要直接配置tiling结构体
  tilingData_->bmm2TilingData.set_baseK(targetBaseK);
  OP_LOGD(ifaContext_->opName, "Page attention is enabled, blockSize is %u, bmm2 baseK is adjusted to %u.", blockSize_, targetBaseK);
}

bool IFATilingV2::FillTilingBmm() const {
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(ifaContext_->platformInfo);
  matmul_tiling::MatmulApiTiling bmm1(ascendcPlatform);
  matmul_tiling::MatmulApiTiling bmm2(ascendcPlatform);

  matmul_tiling::DataType qType;
  matmul_tiling::DataType kvType;

  OP_CHECK_IF((!GetMatmulType(inputQType_, &qType) || !GetMatmulType(inputKvType_, &kvType)),
             OP_LOGE(ifaContext_->opName, "Get matmul type error."),
             return false);
  uint32_t baseN;
  uint32_t bmm1OrgKa;
  uint32_t singleM;
  if (isPFAFlag_) {
    singleM = sOuterSize_; // 16 : sOuterSize of PFA
  } else {
    singleM = msdIterNum_ * nNumOfQInOneGroup_;
  }
  bmm1.SetShape(singleM, sInnerSize_, headDim_);
  bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, kvType, false);
  bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, kvType, true);
  if (antiQuantFlag_) {
    bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, qType, false);
    bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, qType, true);
    bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND_ALIGN, matmul_tiling::DataType::DT_FLOAT);
    baseN = MAX_MATMUL_BASE;  // antiquant to split K
    bmm1.SetOrgShape(singleM, sInnerSize_, headDim_, headDimAlign_);
  } else {
    bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    bmm1OrgKa = headDim_;
    baseN = MATMUL_BASE_N;
    if (inputLayout_ == IfaLayout::BSH_BSND ||
      (pageAttentionFlag_ && pageAttentionKvLayoutType_ == KvCacheLayout::KV_CACHE_BSH)) {
      bmm1.SetOrgShape(singleM, seqSize_, bmm1OrgKa, headDim_ * numKvHeads_);
    } else {
      bmm1.SetOrgShape(singleM, seqSize_, bmm1OrgKa, headDim_);
    }
  }
  // 存在输入query是BNSD格式，但使能PA，需要按BSH SetOrgShape
  bmm1.SetBias(false);

  uint32_t bmm1BaseN = std::min(AlignUp(sInnerSize_, NUM16), baseN);
  if (pageAttentionFlag_) {
    AdjustPABmm1Tiling(bmm1BaseN);
  }
  // 向下对齐保证M*N不超过L0C，且由于bmm1BaseN有最大限制，L0C_SIZE / sizeof(float) / bmm1BaseN不会小于16
  uint32_t bmm1MaxBaseM = AlignUp(static_cast<uint32_t>(L0C_SIZE / sizeof(float) / bmm1BaseN) - NUM16, NUM16);

  OP_CHECK_IF((bmm1.SetFixSplit(std::min(AlignUp(singleM, NUM16), bmm1MaxBaseM), AlignUp(bmm1BaseN, NUM16)) == -1),
             OP_LOGE(ifaContext_->opName, "Bmm1 SetFixSplit fail."),
             return false);
  OP_CHECK_IF((bmm1.SetTraverse(matmul_tiling::MatrixTraverse::FIRSTN) == -1),
             OP_LOGE(ifaContext_->opName, "Bmm1 SetTraverse fail."),
             return false);
  OP_CHECK_IF((bmm1.GetTiling(tilingData_->bmm1TilingData) == -1),
             OP_LOGE(ifaContext_->opName, "Bmm1 get tiling fail."),
             return false);

  // (m, n, k) (so, d, si)
  bmm2.SetShape(singleM, headDim_, sInnerSize_);
  if (antiQuantFlag_) {
    bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, qType, false);
    bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, qType, false);
    bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND_ALIGN, matmul_tiling::DataType::DT_FLOAT);
    bmm2.SetOrgShape(singleM, headDimAlign_, sInnerSizeAlign_, sInnerSize_);
  } else {
    bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, kvType, false);
    bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, kvType, false);
    bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND_ALIGN, matmul_tiling::DataType::DT_FLOAT);
    if (inputLayout_ == IfaLayout::BSH_BSND || (pageAttentionFlag_ && pageAttentionKvLayoutType_ == KvCacheLayout::KV_CACHE_BSH)) {
      bmm2.SetOrgShape(singleM, headDim_ * numKvHeads_, sInnerSizeAlign_, seqSize_);
    } else {
      bmm2.SetOrgShape(singleM, headDim_, sInnerSizeAlign_, seqSize_);
    } 
  }
  // 存在输入query是BNSD格式，但使能PA，需要按BSH SetOrgShape
  bmm2.SetBias(false);
  OP_CHECK_IF((bmm2.SetFixSplit(std::min(AlignUp(singleM, NUM16), MAX_MATMUL_BASE_M)) == -1),
             OP_LOGE(ifaContext_->opName, "Bmm2 SetFixSplit fail."),
             return false);
  OP_CHECK_IF((bmm2.GetTiling(tilingData_->bmm2TilingData) == -1),
             OP_LOGE(ifaContext_->opName, "Bmm2 get tiling fail."),
             return false);
  if (pageAttentionFlag_) {
    AdjustPABmm2Tiling();
  }

  return true;
}

bool IFATilingV2::GetMatmulType(ge::DataType getype, matmul_tiling::DataType* mmType) const {
  static struct {
    ge::DataType a;
    matmul_tiling::DataType b;
  } typeTrans[] = {{ge::DT_FLOAT16, matmul_tiling::DataType::DT_FLOAT16},
                   {ge::DT_BF16, matmul_tiling::DataType::DT_BF16},
                   {ge::DT_INT8, matmul_tiling::DataType::DT_INT8},
                   {ge::DT_FLOAT, matmul_tiling::DataType::DT_FLOAT},
                   {ge::DT_INT4, matmul_tiling::DataType::DT_INT8},
                   {ge::DT_HIFLOAT8, matmul_tiling::DataType::DT_INT8},
                   {ge::DT_FLOAT8_E4M3FN, matmul_tiling::DataType::DT_INT8},
                   {ge::DT_FLOAT4_E2M1, matmul_tiling::DataType::DT_INT8}
                   };

  for (uint32_t i = 0; i < sizeof(typeTrans) / sizeof(typeTrans[0]); i++) {
    if (typeTrans[i].a == getype) {
      *mmType = typeTrans[i].b;
      return true;
    }
  }
  return false;
}

uint8_t IFATilingV2::GenAntiquantModeVal() const {
  uint8_t ret = NUM0;
  switch (antiquantMode_) {
    case NUM0:
      ret = kPerChnVPerTokFlag_ ? NUM2 : NUM0;  // 2 : k_perchannel + v_pertoken
      break;
    case NUM1:
      ret = NUM1;
      break;
    case NUM2: // perTensorHead，使用0，通过perHeadFlag来区分
      ret = NUM0;
      break;
    case NUM3: // perTokenHead，使用1，通过perHeadFlag来区分
      ret = NUM1;
      break;
    case NUM4: // 4 : perTokenPa
      ret = NUM4; // 4 : tilingkey for perTokenPa
      break;
    case NUM5: // 5 : perTokenHeadPa
      ret = NUM5; // 5 : tilingkey for perTokenHeadPa
      break;
    default:
      break;
  }
  return ret;
}

ge::graphStatus IFATilingV2::GenTilingKey() {
    uint8_t inputQVal = 0;
    uint8_t inputKvVal = 0;
    uint8_t outputVal = 0;
    switch (inputQType_) {
    case ge::DT_FLOAT16:
    if ((inputKvType_ != ge::DT_FLOAT16) && (inputKvType_ != ge::DT_INT8) &&
         (inputKvType_ != ge::DT_INT4) && (inputKvType_ != ge::DT_HIFLOAT8) &&
         (inputKvType_ != ge::DT_FLOAT8_E4M3FN) &&
         (inputKvType_ != ge::DT_FLOAT4_E2M1)) {
        OP_LOGE(ifaContext_->opName, "When input query type is fp16, key/value type should be fp16/int8/fp8/hif8/int4/fp4_e2m1.");
        return ge::GRAPH_FAILED;
    }
    inputQVal = NUM0;
    break;
    case ge::DT_BF16:
    if ((inputKvType_ != ge::DT_BF16) && (inputKvType_ != ge::DT_INT8) &&
         (inputKvType_ != ge::DT_INT4) && (inputKvType_ != ge::DT_HIFLOAT8) &&
         (inputKvType_ != ge::DT_FLOAT8_E4M3FN) &&
         (inputKvType_ != ge::DT_FLOAT4_E2M1)) {
        OP_LOGE(ifaContext_->opName, "When input query type is bf16, key/value type should be bf16/int8/fp8/hif8/int4/fp4_e2m1.");
        return ge::GRAPH_FAILED;
    }
    inputQVal = NUM2;
    break;
    case ge::DT_INT8:
        if (inputKvType_ != ge::DT_INT8) {
            OP_LOGE(ifaContext_->opName, "When input query type is int8, key/value type should be int8.");
            return ge::GRAPH_FAILED;
        }
        inputQVal = NUM3;
        break;
    default :
        OP_LOGE(ifaContext_->opName, "Not support inputQType[%s].", DataTypeToString(inputQType_).c_str());
        return ge::GRAPH_FAILED;
    }
    switch (inputKvType_) {
    case ge::DT_FLOAT16:
        inputKvVal = NUM0;
        break;
    case ge::DT_BF16:
        inputKvVal = NUM2;
        break;
    case ge::DT_INT8:
        inputKvVal = NUM3;
        break;
    case ge::DT_INT4:
        inputKvVal = NUM4;
        break;
    case ge::DT_HIFLOAT8:
        inputKvVal = NUM5;
        break;
    case ge::DT_FLOAT8_E4M3FN:
        inputKvVal = NUM7;
        break;
    case ge::DT_FLOAT4_E2M1:
        inputKvVal = NUM8;
        break;
    default :
        OP_LOGE(ifaContext_->opName, "Not support inputKvType[%s].", DataTypeToString(inputKvType_).c_str());
        return ge::GRAPH_FAILED;
    }
    switch (outputType_) {
    case ge::DT_FLOAT16:
        outputVal = NUM0;
        break;
    case ge::DT_BF16:
        outputVal = NUM2;
        break;
    case ge::DT_INT8:
        outputVal = NUM3;
        break;
    case ge::DT_FLOAT8_E4M3FN:
        outputVal = NUM4;
        break;
    case ge::DT_HIFLOAT8:
        outputVal = NUM6;
        break;
    default :
        OP_LOGE(ifaContext_->opName, "Not support outputType[%s].", DataTypeToString(outputType_).c_str());
        return ge::GRAPH_FAILED;
    }
    UpdatePerfMode();
    uint64_t baseOffset = IFA_TILINGKEYOFFSET;
    if (socVersion_ != IfaSocVersion::SOC_ASCEND_950) {
        baseOffset += (static_cast<uint64_t>(perfMode_)) * IFA_PERF_MODE_TILINGKEYOFFSET;
    }
    UpdateTilingKeyLayoutType();
    UpdateTilingKeyConfig();
    UpdateTilingKeyPseMode();
    UpdateTilingKeyQuantMode();
    UpdateTilingKeyAttenMask();
    UpdateTilingKeyHasRope();
    UpdateTilingKeyIsPa();
    UpdateTilingKeyIsFd();
    UpdateTilingKeyEmptyTensor();
    UpdateTilingKeyEnableKVPrefix();
    UpdateTilingKeySplitCoreMode();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATilingV2::CalcNumBlocks() const {
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(ifaContext_->platformInfo);
  auto aicNum = aicNum_;
  auto aivNum = aivNum_;
  ifaContext_->numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);  // 暂时与当前代码一致
  OP_LOGD(ifaContext_->opName, "IFA block dim:%u aivNum:%u aicNum:%u.", ifaContext_->numBlocks, aivNum, aicNum);
  return ge::GRAPH_SUCCESS;
}

void IFATilingV2::UpdateTilingKeyLayoutType() {
  if (inputLayout_ == IfaLayout::BNSD) {
    inOutLayoutType = InOutLayoutType_BNSD_BNSD;
  } else if (inputLayout_ == IfaLayout::TND) {
    inOutLayoutType = InOutLayoutType_TND_TND;
  } else if (inputLayout_ == IfaLayout::BSH_BSND) {
    inOutLayoutType = InOutLayoutType_BSH_BSH;
  }
}

void IFATilingV2::UpdateTilingKeyConfig() {
  auto sInner = sInnerSize2_;
  auto sOuter = sOuterSize_;
  auto dSize = headDim_;
  auto dVsize = headDim_;
  if (sInner == 1024 && sOuter == 16 && dSize <= 64 && dVsize <= 64) {
    config = Config_S1Aligned16_S2Aligned1024_DAligned64_DVAligned64;
  } else if (sInner == 512 && sOuter == 16 && dSize <= 64 && dVsize <= 64) {
    config = Config_S1Aligned16_S2Aligned512_DAligned64_DVAligned64;
  } else if (sInner == 512 && sOuter == 16 && dSize <= 128 && dVsize <= 128) {
    config = Config_S1Aligned16_S2Aligned512_DAligned128_DVAligned128;
  } else if (sInner == 256 && sOuter == 16 && dSize <= 256 && dVsize <= 256) {
    config = Config_S1Aligned16_S2Aligned256_DAligned256_DVAligned256;
  } else if (sInner == 128 && sOuter == 16 && dSize <= 512 && dVsize <= 512) {
    config = Config_S1Aligned16_S2Aligned128_DAligned512_DVAligned512;
  } else if (sInner == 512 && sOuter == 32 && dSize <= 64 && dVsize <= 64) { // 以下为PFA伪量化合轴场景 32:s1base 512:s2base 64:dbase
    config = Config_S1Aligned32_S2Aligned512_DAligned64_DVAligned64;
  } else if (sInner == 512 && sOuter == 32 && dSize <= 128 && dVsize <= 128) { // 32:s1base 512:s2base 128:dbase
    config = Config_S1Aligned32_S2Aligned512_DAligned128_DVAligned128;
  } else if (sInner == 256 && sOuter == 32 && dSize <= 256 && dVsize <= 256) { // 32:s1base 256:s2base 256:dbase
    config = Config_S1Aligned32_S2Aligned256_DAligned256_DVAligned256;
  } else if (sInner == 128 && sOuter == 32 && dSize <= 512 && dVsize <= 512) { // 32:s1base 128:s2base 512:dbase
    config = Config_S1Aligned32_S2Aligned128_DAligned512_DVAligned512;
  } else {
        OP_LOGE(ifaContext_->opName, "The combination of parameters S1, S2, D, DV is not supported!");
    }
  OP_LOGI(ifaContext_->opName, "sInner is %llu. sOuter is %llu. dSize is %llu. dVsize is %llu.", sInner, sOuter, dSize, dVsize);
}

void IFATilingV2::UpdateTilingKeyPseMode() {
  if (!pseShiftFlag_ && !enableAlibiPse_) {
    pseMode = PSE_MODE_PSE_NONE_TYPE;
  }
  else {
    pseMode = pseType_;
  }
}

void IFATilingV2::UpdateTilingKeyQuantMode() {
  quantMode = GenAntiquantModeVal();
}

void IFATilingV2::UpdateTilingKeyAttenMask() {
  hasAttenMask = attenMaskFlag_;
}

void IFATilingV2::UpdateTilingKeyHasRope() {
  hasRope = 0;
}

void IFATilingV2::UpdateTilingKeyIsPa() {
  isPa = pageAttentionFlag_;
}

void IFATilingV2::UpdateTilingKeyIsFd() {
  isFd = static_cast<uint64_t>(splitKVFlag_);
}

void IFATilingV2::UpdateTilingKeyEmptyTensor() {
  emptyTensor = 0;
}

void IFATilingV2::UpdateTilingKeyEnableKVPrefix() {
  enableKVPrefix = enableKVPrefix_;
}

void IFATilingV2::UpdateTilingKeySplitCoreMode() 
{
  enableS1OutSplit = false;
}

ge::graphStatus IFATilingV2::DoTiling(gert::TilingContext& context) {
  IncreFlashAttentionTilingDataV2 tilingData;
  IncreFlashAttentionContext ifaContext {};

  if (ConvertContext(context, ifaContext) != ge::GRAPH_SUCCESS) {
    OP_LOGE(context.GetNodeName(), "Error occurred while convert tilingContext to ifa context.");
    return ge::GRAPH_FAILED;
  }

  if (RunBigKernelTiling(ifaContext, tilingData) == ge::SUCCESS) {
    context.SetTilingKey(ifaContext.tilingKey);
    context.SetBlockDim(ifaContext.numBlocks);
  } else {
    return ge::GRAPH_FAILED;
  }

  return IncreFlashAttentionSetTilingData(context, tilingData);
}

ge::graphStatus IFATilingV2::ConvertContext(gert::TilingContext& context,
                                          IncreFlashAttentionContext& ifaContext) {
  OP_CHECK_IF(context.GetNodeName() == nullptr,
             OP_LOGE("IncreFlashAttention", "OpName got from TilingContext is null."),
             return ge::GRAPH_FAILED);
  ifaContext.opName = context.GetNodeName();
  ifaContext.platformInfo = context.GetPlatformInfo();
  ifaContext.query.desc = context.GetInputDesc(QUERY_INPUT_INDEX);
  ifaContext.query.shape = context.GetInputShape(QUERY_INPUT_INDEX);
  ifaContext.key.desc = context.GetInputDesc(KEY_INPUT_INDEX);
  ifaContext.key.shape = context.GetInputShape(KEY_INPUT_INDEX);
  OP_CHECK_IF((ifaContext.query.shape == nullptr) || (ifaContext.key.shape == nullptr),
             OP_LOGE(context.GetNodeName(), "Shape of query or shape of key is null."),
             return ge::GRAPH_FAILED);
  auto batchOfQuery = ifaContext.query.shape->GetStorageShape().GetDim(NUM0);
  auto batchOfKey = ifaContext.key.shape->GetStorageShape().GetDim(NUM0);
  if (batchOfQuery != batchOfKey) {
    ifaContext.kCache.resize(batchOfQuery);
    ifaContext.vCache.resize(batchOfQuery);
    for (int64_t size = 0; size < batchOfQuery; ++size) {
      ifaContext.kCache[size] = const_cast<gert::StorageShape *>(context.GetDynamicInputShape(KEY_INPUT_INDEX, size));
      ifaContext.vCache[size] = const_cast<gert::StorageShape *>(context.GetDynamicInputShape(VALUE_INPUT_INDEX, size));
    }
  } else {
    ifaContext.kCache.resize(NUM1);
    ifaContext.vCache.resize(NUM1);
    ifaContext.kCache[0] = const_cast<gert::StorageShape *>(context.GetDynamicInputShape(KEY_INPUT_INDEX, 0));
    ifaContext.vCache[0] = const_cast<gert::StorageShape *>(context.GetDynamicInputShape(VALUE_INPUT_INDEX, 0));
  }

  ifaContext.value.desc = context.GetInputDesc(VALUE_INPUT_INDEX);
  ifaContext.value.shape = context.GetInputShape(VALUE_INPUT_INDEX);
  ifaContext.pseShift.desc = context.GetOptionalInputDesc(PSE_SHIFT_INPUT_INDEX);
  ifaContext.pseShift.tensor = context.GetOptionalInputTensor(PSE_SHIFT_INPUT_INDEX);
  ifaContext.attenMask.desc = context.GetOptionalInputDesc(ATTEN_MASK_INPUT_INDEX);
  ifaContext.attenMask.tensor = context.GetOptionalInputTensor(ATTEN_MASK_INPUT_INDEX);
  ifaContext.attenOut.desc = context.GetOutputDesc(OUTPUT_INDEX);
  ifaContext.attenOut.shape = context.GetOutputShape(OUTPUT_INDEX);
  ifaContext.lseOut.desc = context.GetOutputDesc(LSE_OUTPUT_INDEX);
  ifaContext.lseOut.shape = context.GetOutputShape(LSE_OUTPUT_INDEX);

  ifaContext.actualSeqLengths.tensor = context.GetOptionalInputTensor(ACT_SEQ_LEN_INPUT_INDEX);
  ifaContext.deqScale1.tensor = context.GetOptionalInputTensor(DEQUANT_SCALE_1_INPUT_INDEX);
  ifaContext.quantScale1.tensor = context.GetOptionalInputTensor(QUANT_SCALE_1_INPUT_INDEX);
  ifaContext.deqScale2.tensor = context.GetOptionalInputTensor(DEQUANT_SCALE_2_INPUT_INDEX);
  ifaContext.quantScale2.tensor = context.GetOptionalInputTensor(QUANT_SCALE_2_INPUT_INDEX);
  ifaContext.quantOffset2.tensor = context.GetOptionalInputTensor(QUANT_OFFSET_2_INPUT_INDEX);
  ifaContext.deqScale1.desc = context.GetOptionalInputDesc(DEQUANT_SCALE_1_INPUT_INDEX);
  ifaContext.quantScale1.desc = context.GetOptionalInputDesc(QUANT_SCALE_1_INPUT_INDEX);
  ifaContext.deqScale2.desc = context.GetOptionalInputDesc(DEQUANT_SCALE_2_INPUT_INDEX);
  ifaContext.quantScale2.desc = context.GetOptionalInputDesc(QUANT_SCALE_2_INPUT_INDEX);
  ifaContext.quantOffset2.desc = context.GetOptionalInputDesc(QUANT_OFFSET_2_INPUT_INDEX);
  ifaContext.antiquantScale.tensor = context.GetOptionalInputTensor(ANTIQUANT_SCALE_INPUT_INDEX);
  ifaContext.antiquantOffset.tensor = context.GetOptionalInputTensor(ANTIQUANT_OFFSET_INPUT_INDEX);
  ifaContext.antiquantScale.desc = context.GetOptionalInputDesc(ANTIQUANT_SCALE_INPUT_INDEX);
  ifaContext.antiquantOffset.desc = context.GetOptionalInputDesc(ANTIQUANT_OFFSET_INPUT_INDEX);
  ifaContext.blockTable.tensor = context.GetOptionalInputTensor(BLOCK_TABLE_INPUT_INDEX);
  ifaContext.kvPaddingSize.tensor = context.GetOptionalInputTensor(KV_PADDING_SIZE_INPUT_INDEX);

  auto attrs = context.GetAttrs();
  OP_CHECK_IF(attrs == nullptr, OP_LOGE(context.GetNodeName(), "Attrs got from ge is null."),
             return ge::GRAPH_FAILED);

  ifaContext.numHeads = attrs->GetAttrPointer<uint32_t>(NUM_HEADS_ATTR_INDEX);
  ifaContext.scaleValue = attrs->GetAttrPointer<float>(SCALE_VALUE_ATTR_INDEX);
  ifaContext.layOut = attrs->GetStr(LAYOUT_ATTR_INDEX);
  ifaContext.kvHeadNums = attrs->GetAttrPointer<uint32_t>(KV_NUM_HEADS_ATTR_INDEX);
  ifaContext.blockSize = attrs->GetAttrPointer<uint32_t>(BLOCK_SIZE_ATTR_INDEX);
  ifaContext.innerPrecise = attrs->GetAttrPointer<uint32_t>(INNER_PRECISE_ATTR_INDEX);

  OP_CHECK_IF(context.GetWorkspaceSizes(1) == nullptr,
             OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "WorkSpaceSize got from ge is null."),
             return ge::GRAPH_FAILED);
  ifaContext.workSpaces = context.GetWorkspaceSizes(1);
  return ge::GRAPH_SUCCESS;
}

void IFATilingV2::GetMaxWorkspaceFlag() {
  if ((ifaContext_->actualSeqLengthsQ.tensor != nullptr && ifaContext_->actualSeqLengthsQ.tensor->GetData<int64_t>() == nullptr) ||
    (ifaContext_->actualSeqLengths.tensor != nullptr && ifaContext_->actualSeqLengths.tensor->GetData<int64_t>() == nullptr)) {
    isMaxWorkspace_ = true;
  } else {
    isMaxWorkspace_ = false;
  }
}

ge::graphStatus IFATilingV2::RunBigKernelTiling(IncreFlashAttentionContext& context,
                                              IncreFlashAttentionTilingDataV2& tilingData) {
  this->ifaContext_ = &context;
  this->tilingData_ = &tilingData.tilingBase;

  GetMaxWorkspaceFlag();

  if ((GetNpuInfo() != ge::GRAPH_SUCCESS) || (PreProcess() != ge::GRAPH_SUCCESS)) {
    return ge::GRAPH_FAILED;
  }
  if (EmptyTensorProcess() != ge::GRAPH_SUCCESS) {
    if ((Split() != ge::GRAPH_SUCCESS) || (FillTiling() != ge::GRAPH_SUCCESS) ||
        (GenTilingKey() != ge::GRAPH_SUCCESS) || (CalcWorkSpace() != ge::GRAPH_SUCCESS) ||
        (CalcNumBlocks() != ge::GRAPH_SUCCESS)) {
      return ge::GRAPH_FAILED;
    }
  }
  if (tilingData_ != nullptr) {
    IFATilingDataconvert();
  }
  return ge::GRAPH_SUCCESS;
}

void IFATilingV2::SetLayoutTypefaRun()
{
    static std::map<IfaLayout, LayoutTypefaRun> layoutStrToLayoutTypeMap = {
        {IfaLayout::BSH_BSND, LayoutTypefaRun::LAYOUT_BSH},
        {IfaLayout::TND, LayoutTypefaRun::LAYOUT_TND},
        {IfaLayout::BSH_BSND, LayoutTypefaRun::LAYOUT_BSND},
        {IfaLayout::BNSD, LayoutTypefaRun::LAYOUT_BNSD},
    };
    auto itr = layoutStrToLayoutTypeMap.find(inputLayout_);
    if (itr == layoutStrToLayoutTypeMap.end()) {
        faRunTilingAdapter.inputParamsRegbase.set_layoutType(static_cast<uint8_t>(0));
    } else {
        faRunTilingAdapter.inputParamsRegbase.set_layoutType(static_cast<uint8_t>(itr->second));
    }
}

void IFATilingV2::SetAttenMaskCompressMode()
{
    static std::map<uint32_t, uint8_t> sparseToCompressModeMap = {
        {SPARSE_MODE_NO_MASK, PfaAttenMaskCompressModefaRun::PFA_NO_COMPRESS_MODE},
        {SPARSE_MODE_ALL_MASK, PfaAttenMaskCompressModefaRun::PFA_NO_COMPRESS_MODE},
        {SPARSE_MODE_LEFT_UP, PfaAttenMaskCompressModefaRun::PFA_LEFT_UP_CAUSAL_MODE},
        {SPARSE_MODE_RIGHT_DOWN, PfaAttenMaskCompressModefaRun::PFA_RIGHT_DOWN_CAUSAL_MODE},
        {SPARSE_MODE_BAND, PfaAttenMaskCompressModefaRun::PFA_BAND_MODE}
    };
    auto itr = sparseToCompressModeMap.find(sparseMode_);
    if (itr == sparseToCompressModeMap.end()) {
        faRunTilingAdapter.inputParamsRegbase.set_attenMaskCompressMode(0);
    } else {
        faRunTilingAdapter.inputParamsRegbase.set_attenMaskCompressMode(itr->second);
    }
}

void IFATilingV2::IFATilingDataconvert() {
  if (emptyTensor_) {
    auto &initOutputParams = faRunTilingAdapter.initOutputParams;
    initOutputParams.set_singleCoreSize(singleCoreSize_);
    initOutputParams.set_totalOutputSize(totalSize_);
    initOutputParams.set_totalSoftMaxLseOutputSize(totalSizeLse_);
    return;
  }
  SetLayoutTypefaRun();
  auto &inputParams = faRunTilingAdapter.inputParamsRegbase;
  inputParams.set_bSize(batchSize_);
  // 将GS1合轴与不合轴场景下，有不同含义的n2Size、gSize与s1Size参数，转化为各自实际的值
  // n2:KVN
  // gSize:原始的QN/KVN
  // s1:QS
  // headNumratio:  合轴1   不合轴QN/KVN
  inputParams.set_n2Size(numKvHeads_);
  inputParams.set_gSize(nNumOfQInOneGroup_);
  inputParams.set_s1Size(sOfQuery_);
  if (faRunGS_) {
    inputParams.set_headNumRatio(1);
  } else {
    inputParams.set_headNumRatio(numHeads_ / numKvHeads_);
  }
  inputParams.set_s2Size(sMax_);
  inputParams.set_alignedS2(0); // 默认值
  inputParams.set_dSize(headDim_);
  inputParams.set_dSizeV(headDim_);
  inputParams.set_dSizeRope(64); // 64 is norml value
  inputParams.set_scaleValue(scaleValue_);
  inputParams.set_preTokens(preToken_);
  inputParams.set_nextTokens(nextToken_);
  inputParams.set_pseS1Size(pseShiftS0_);
  inputParams.set_pseS2Size(pseShiftS1_);
  inputParams.set_pseBSize(pseShiftBatch_);
  inputParams.set_bandIndex(0); // 训练代码中在TND场景生效，用于计算s2方向循环的起始位置
  inputParams.set_pseEncodeType(0); // 默认值
  inputParams.set_pseAlibiBaseS1(0); // 默认值
  inputParams.set_pseAlibiBaseS2(0); // 默认值
  inputParams.set_attenMaskShapeType(faRunAttenMaskShapeType_);
  inputParams.set_attenMaskDataType(1); // 默认值
  SetAttenMaskCompressMode();
  inputParams.set_implMode(static_cast<uint8_t>(IFA_HIGH_PRECISION));
  inputParams.set_sparseType(faRunSparseType_);
  inputParams.set_needDropMaskOp(0); // 默认值
  inputParams.set_keepProb(0); // 默认值
  inputParams.set_keepProbUint8(0); // 默认值
  inputParams.set_dropMaskOuter(0); // 默认值
  inputParams.set_remain(0); // 默认值
  inputParams.set_attenMaskS2Size(attenMaskKvSize_);
  inputParams.set_rsv1(0); // 默认值
  inputParams.set_s1SparseValidSize(0); // 临时默认值
  inputParams.set_s2SparseValidSize(0); // 临时默认值
  inputParams.set_seed(0); // 默认值
  inputParams.set_offset(0); // 默认值
  inputParams.set_pseShapeType(static_cast<uint8_t>(pseShapeType));
  inputParams.set_pseType(pseType_);
  inputParams.set_qStartIdx(qStartIdx_);
  inputParams.set_kvStartIdx(kvStartIdx_);

  // PFA
  // 伪量化用到的PA相关的有
  inputParams.set_blockSize(blockSize_);
  inputParams.set_blockTableDim2(maxBlockNumPerSeq_);
  inputParams.set_paLayoutType(pageAttentionKvLayoutTypefaRun_);
  inputParams.set_paBlockNumSum(paBlockNumSumfaRun_);  // 用不到
  inputParams.set_prefixSeqInnerSize(prefixSSize_);
  inputParams.set_attenMaskS1Size(attenMaskQSize_);
  inputParams.set_isActualSeqLengthsNull(!actualSeqLenQFlag_ ? 1 : 0);
  inputParams.set_isActualSeqLengthsKVNull(!actualSeqLenFlag_ ? 1 : 0);
  inputParams.set_actualSeqLengthsSize(actualLenQDims_);
  inputParams.set_actualSeqLengthsKVSize(actualLenDims_);
  inputParams.set_deqScaleFlag(0);      // 伪量化模板没用到，默认值
  inputParams.set_deqScale2Flag(0);     // 伪量化模板没用到，默认值

  inputParams.set_isKvContinuous(batchContinuousFlag_);
  inputParams.set_fromFused(1);   //伪量化模板没有用到，设置为默认值
  std::string layout(ifaContext_->layOut);
  inputParams.set_isBSNDOut(layout == "BNSD_BSND");
  inputParams.set_transposeLayout(ifaContext_->transposeLayout);
  //关于合轴
  inputParams.set_isGqa(isGqa_);
  inputParams.set_isSoftMaxLseEnable(softmaxLseFlag_ );
  inputParams.set_isActualSharedPrefixLenNull(actualSharedPrefixLenNullFlag_);
  inputParams.set_isQHasLeftPadding(qPaddingSizeFlag_ ? 1 : 0);
  inputParams.set_isKVHasLeftPadding(kvPaddingSizeFlag_ ? 1 : 0);
  inputParams.set_ropeHeadSize(0);
  inputParams.set_isRowInvalid(isRowInvalid_);
  inputParams.set_antiquantPerTensorFlag(antiquantPerTensorFlag_);
  inputParams.set_antiquantPerHeadFlag(antiquantPerHeadFlag_);
  inputParams.set_antiquantParaSeqSize(antiquantParaSeqSize_);

  auto &initOutputParams = faRunTilingAdapter.initOutputParams;
  initOutputParams.set_singleCoreSize(singleCoreSize_);
  initOutputParams.set_totalOutputSize(totalSize_);
  initOutputParams.set_totalSoftMaxLseOutputSize(totalSizeLse_);
  initOutputParams.set_needInit(needInit_);   // 暂时先固定为1
  initOutputParams.set_isOneN(0);  // 伪量化没有用到

  inputParams.set_isPostQuantPerChnl(isPostQuantPerChnl_);  // 伪量化暂不支持后量化，默认值
  inputParams.set_isPostQuantBF16(isPostQuantBF16_);  //伪量化暂不支持后量化，默认值
}

ge::graphStatus IFATilingV2::IncreFlashAttentionSetTilingData(gert::TilingContext& ifaContextLocal,
                                                            IncreFlashAttentionTilingDataV2& tilingData) {
  return ge::GRAPH_SUCCESS;
}

#ifdef ASCEND_OPTILING_UT
// ut 测试框架通过算子名查找对应的tiling函数，
// 注册"IncreFlashAttentionNew" 作为算子名，模板归一后删除
static ge::graphStatus TilingIncreFlashAttentionNew(gert::TilingContext* context) {
  IFATilingV2 flash_tiling;
  OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("IncreFlashAttentionNew", "Context is nullptr."),
             return ge::GRAPH_FAILED);
  return flash_tiling.DoTiling(*context);
}

static ge::graphStatus TilingPrepareForIncreFlashAttentionNew(gert::TilingParseContext* context) {
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(IncreFlashAttentionNew)
    .Tiling(TilingIncreFlashAttentionNew)
    .TilingParse<IncreFlashAttentionCompileInfo>(TilingPrepareForIncreFlashAttentionNew)
    .TilingInputsDataDependency({5});

#endif // ASCEND_OPTILING_UT

void TilingGetTempCompileInfo(platform_ascendc::PlatformAscendC& ascendcPlatform, PromptFlashAttentionCompileInfo& compileInfo)
{
  compileInfo.aivNum = ascendcPlatform.GetCoreNumAiv();
  compileInfo.aicNum = ascendcPlatform.GetCoreNumAic();
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo.ubSize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfo.l1Size);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfo.l0CSize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfo.l0ASize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfo.l0BSize);
  compileInfo.socShortName = ascendcPlatform.GetSocVersion();
  if (compileInfo.socShortName == platform_ascendc::SocVersion::ASCEND310P) {
    compileInfo.defaultSysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
  } else {
    compileInfo.defaultSysWorkspaceSize = 0U;
  }
}

ge::graphStatus IFATilingV2::DoOpTiling()
{  
    IncreFlashAttentionContext ifaContext;
    auto ret = ConvertContext(*context_, ifaContext);
    OP_CHECK_IF(ret == ge::GRAPH_FAILED,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "fail to convert to IFAParams"),
                return ge::GRAPH_FAILED);
    ret = DoSubOpTiling(ifaContext);
    uint64_t tiling_key = GET_TPL_TILING_KEY(static_cast<uint64_t>(inOutLayoutType), static_cast<uint64_t>(config),
                                            static_cast<uint64_t>(pseMode), static_cast<uint64_t>(quantMode), hasAttenMask, hasRope, isPa, isFd, emptyTensor, 
                                            enableKVPrefix, enableS1OutSplit);
    context_->SetTilingKey(tiling_key);
    OP_LOGI(ifaContext.opName, "The new template tilingkey is %llu.", tiling_key);
    OP_LOGI(ifaContext.opName, "The new template tilingkey param is inOutLayoutType: %llu, config: %llu, pseMode: %llu,quantMode: %llu, "
            "hasAttenMask: %llu, hasRope: %llu, isPa: %llu, isFd: %llu, emptyTensor: %llu, enableKVPrefix: %llu, enableS1OutSplit: %llu.", 
            static_cast<uint64_t>(inOutLayoutType), static_cast<uint64_t>(config),
            static_cast<uint64_t>(pseMode), static_cast<uint64_t>(quantMode), hasAttenMask, hasRope, isPa, isFd, emptyTensor, 
            enableKVPrefix, enableS1OutSplit);
    return ret;
}

ge::graphStatus IFATilingV2::DoSubOpTiling(IncreFlashAttentionContext& ifaContext)
{
    // 使用SyncAll，需要设置为batchmode模式，所有核同时启动，否则多流方式下执行可能会卡死
    context_->SetScheduleMode(BATCH_MODE_SCHEDULE);
    if (ifaContext.key.desc->GetDataType() == ge::DT_FLOAT16 || ifaContext.key.desc->GetDataType() == ge::DT_BF16) {
        auto platformInfoPtr = context_->GetPlatformInfo();
        OP_CHECK_IF(platformInfoPtr == nullptr,
                    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "platformInfoPtr is null!"),
                    return ge::GRAPH_FAILED);
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        PromptFlashAttentionCompileInfo compileInfo = {0, 0, 0, 0, 0, 0, 0, 0, platform_ascendc::SocVersion::ASCEND310P};
        TilingGetTempCompileInfo(ascendcPlatform, compileInfo);
        PromptFlashAttentionTilingV2 flashTilingV2(context_);
        ContextParamsForPFATiling contextParamsForPFATiling;
        auto ret = PFAConvertContext(contextParamsForPFATiling, context_);
        contextParamsForPFATiling.compileInfoPtr = &compileInfo;
        OP_CHECK_IF(ret == ge::GRAPH_FAILED,
                    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "fail to convert to PFAParams"),
                    return ge::GRAPH_FAILED);
        PromptFlashAttentionTilingDataV2 tilingData;
        ret = flashTilingV2.DoSubOpTiling(tilingData, contextParamsForPFATiling);
        inOutLayoutType = flashTilingV2.inOutLayoutType;
        config = flashTilingV2.config;
        pseMode = flashTilingV2.pseMode;
        quantMode = flashTilingV2.quantMode;
        hasAttenMask = flashTilingV2.hasAttenMask;
        hasRope = flashTilingV2.hasRope;
        isPa = flashTilingV2.isPa;
        isFd = flashTilingV2.isFd;
        emptyTensor = flashTilingV2.emptyTensor;
        OP_LOGI(contextParamsForPFATiling.opName, "All the PFATiling work is done.");
        return ret;
    } else {
        IncreFlashAttentionTilingDataV2 tilingData;
        auto ret = RunBigKernelTiling(ifaContext, tilingData);
        OP_CHECK_IF(ret == ge::GRAPH_FAILED,
                    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "failed in IFA RunBigKernelTiling"),
                    return ge::GRAPH_FAILED);
        context_->SetBlockDim(ifaContext.numBlocks);
        FlashAttentionScoreSimplifiedTilingData* tiling = context_->GetTilingData<FlashAttentionScoreSimplifiedTilingData>();
        OP_CHECK_IF(tiling == nullptr,
                    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "tilingdata ptr should not be null"),
                    return ge::GRAPH_FAILED);
        *tiling = faRunTilingAdapter;
        return ret;
    }
  return ge::GRAPH_FAILED;
}

REGISTER_TILING_TEMPLATE_FIA(IncreFlashAttention, IFATilingV2, std::vector<int32_t>({(int32_t)NpuArch::DAV_3510}), 90);

} // namespace optiling