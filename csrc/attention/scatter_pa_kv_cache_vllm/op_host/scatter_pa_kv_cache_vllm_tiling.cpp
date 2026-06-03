/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file scatter_pa_kv_cache_vllm_tiling.cpp
 * \brief
 */

#include <algorithm>
#include <string>
#include "tiling/tiling_api.h"
#include "scatter_pa_kv_cache_vllm_tiling.h"

#include "log/ops_log.h"
#include "../../../moe/causal_conv1d/op_host/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_info_def.h"
// #include "platform_util.h"


namespace optiling {
constexpr uint64_t DIM_0 = 0;
constexpr uint64_t DIM_1 = 1;
constexpr uint64_t DIM_2 = 2;
constexpr uint64_t DIM_3 = 3;
constexpr uint64_t DIM_4 = 4;
constexpr uint64_t DIM_5 = 5;
constexpr uint64_t DIM_6 = 6;
constexpr uint64_t INPUT_KEY_INDEX = 0;
constexpr uint64_t INPUT_KEY_CACHE_INDEX = 1;
constexpr uint64_t INPUT_SLOT_MAPPING_INDEX = 2;
constexpr uint64_t INPUT_VALUE_INDEX = 3;
constexpr uint64_t INPUT_VALUE_CACHE_INDEX = 4;
constexpr uint64_t INPUT_COMPRESS_LENS = 5;
constexpr uint64_t INPUT_COMPRESS_SEQ_OFFSET = 6;
constexpr uint64_t INPUT_SEQ_LENS = 7;
constexpr uint64_t INPUT_CACHE_MODE_INDEX = 0;
constexpr uint64_t INPUT_SCATTER_MODE_INDEX = 1;
constexpr uint64_t INPUT_STRIDES_INDEX = 2;
constexpr uint64_t INPUT_OFFSET_INDEX = 3;
constexpr uint64_t INPUT_KV_CACHE_STRIDES_INDEX = 4;
constexpr uint64_t ALIGN = 32;
constexpr uint64_t RESERVED_BUFFER = 1024;
constexpr uint64_t NUM_HEAD_MAX = 4095;  // datacopypad基础接口中跳搬的块数，最大维4095

constexpr uint64_t TILING_ID_TEMPLATE = 1000;
constexpr uint64_t TILING_ID_FULL = 1;

constexpr int64_t TEMPLATE_NORMAL = 1;
constexpr int64_t TEMPLATE_NZ = 2;
constexpr int64_t TEMPLATE_ALIBI = 3;
constexpr int64_t TEMPLATE_ROPE = 4;
constexpr int64_t TEMPLATE_SISO = 5;
constexpr int64_t TEMPLATE_OMNI = 6;
constexpr int64_t TEMPLATE_NORM_NCT = 7;
constexpr int64_t TEMPLATE_SISO_NCT = 8;
constexpr int64_t TEMPLATE_NHSD = 9;

constexpr int64_t SINGLE_IN_OUT = 1;

constexpr uint64_t TASK_MULTIPLE = 2; // Compress_rope模式下KV分核，分核任务量翻倍
constexpr uint64_t SMALL_TOKEN = 1000; // token数



template <typename T>
inline T CeilDiv(T a, T b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}


bool ScatterPaKvCacheVllmMembaseTiling::IsCapable()
{
    return true;
}

ge::graphStatus ScatterPaKvCacheVllmMembaseTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheVllmMembaseTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context_, "platformInfo is nullptr."), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    params_.socVersion = ascendcPlatform.GetSocVersion();
    params_.usedCoreNum = static_cast<uint64_t>(ascendcPlatform.GetCoreNumAiv());
    if (params_.usedCoreNum == 0) {
        OP_LOGE(context_, "coreNum is 0");
        return ge::GRAPH_FAILED;
    }
    params_.sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == static_cast<uint64_t>(0)) {
        OP_LOGE(context_, "ubSize is 0");
        return ge::GRAPH_FAILED;
    }
    params_.ubSize = static_cast<int64_t>(ubSize) - RESERVED_BUFFER;
    return ge::GRAPH_SUCCESS;
}

/****************************************获取并校验输入类型******************************************* */
ge::graphStatus ScatterPaKvCacheVllmMembaseTiling::GetInputDtypeSiso()
{
    // key
    auto inputKeyDesc = context_->GetInputDesc(INPUT_KEY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputKeyDesc);
    inputKeyDtype_ = inputKeyDesc->GetDataType();
    // keyCache
    auto inputKeyCacheDesc = context_->GetInputDesc(INPUT_KEY_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputKeyCacheDesc);
    ge::DataType inputKeyCacheDtype = inputKeyCacheDesc->GetDataType();
    if (inputKeyDtype_ != inputKeyCacheDtype) {
        OP_LOGE(context_, "key, key_cache dtype must be same.");
        return ge::GRAPH_FAILED;
    }
    params_.typeByteK = GetSizeByDataType(inputKeyDtype_);
    if (params_.typeByteK <= 0) {
        OP_LOGE(context_, "get key dtype bytes failed.");
        return ge::GRAPH_FAILED;
    }
    if (!(params_.templateType == TEMPLATE_ROPE || params_.templateType == TEMPLATE_OMNI)) {
        if (!(inputKeyDtype_ == ge::DT_FLOAT16 || inputKeyDtype_ == ge::DT_BF16 || inputKeyDtype_ == ge::DT_INT8)) {
            OP_LOGE(context_, "input key dtype must be float16 or bfloat16 or int8.");
            return ge::GRAPH_FAILED;
        }
    } else {
        if (!(inputKeyDtype_ == ge::DT_FLOAT16 || inputKeyDtype_ == ge::DT_BF16)) {
            OP_LOGE(context_, "input key and value dtype must be float16 or bfloat16.");
            return ge::GRAPH_FAILED;
        }
    }
    // slotmapping
    auto inputSlotMappingDesc = context_->GetInputDesc(INPUT_SLOT_MAPPING_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputSlotMappingDesc);
    ge::DataType inputSlotMappingDtype = inputSlotMappingDesc->GetDataType();
    params_.typeByteSlot = GetSizeByDataType(inputSlotMappingDtype);
    if (params_.typeByteSlot <= 0) {
        OP_LOGE(context_, "get slot_mapping dtype bytes failed.");
        return ge::GRAPH_FAILED;
    }
    if (!(inputSlotMappingDtype == ge::DT_INT32 || inputSlotMappingDtype == ge::DT_INT64)) {
        OP_LOGE(context_, "slot_mapping dtype must be int32 or int64.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheVllmMembaseTiling::GetInputDtypeKv()
{
    if (GetInputDtypeSiso() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // value
    auto inputValueDesc = context_->GetInputDesc(INPUT_VALUE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputValueDesc);
    inputValueDtype_ = inputValueDesc->GetDataType();
    // valueCache
    auto inputValueCacheDesc = context_->GetInputDesc(INPUT_VALUE_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputValueCacheDesc);
    ge::DataType inputValueCacheDtype = inputValueCacheDesc->GetDataType();
    if (inputValueDtype_ != inputValueCacheDtype) {
        OP_LOGE(context_, "value, value_cache dtype must be same.");
        return ge::GRAPH_FAILED;
    }
    params_.typeByteV = GetSizeByDataType(inputValueDtype_);
    if (params_.typeByteV <= 0) {
        OP_LOGE(context_, "get value dtype bytes failed.");
        return ge::GRAPH_FAILED;
    }
    if (!(params_.templateType == TEMPLATE_ROPE || params_.templateType == TEMPLATE_OMNI)) {
        if (!(inputValueDtype_ == ge::DT_FLOAT16 || inputValueDtype_ == ge::DT_BF16 || inputValueDtype_ == ge::DT_INT8)) {
            OP_LOGE(context_, "input value dtype must be float16 or bfloat16 or int8.");
            return ge::GRAPH_FAILED;
        }
        if (inputKeyDtype_ != ge::DT_INT8 && inputValueDtype_ == ge::DT_INT8) {
            OP_LOGE(context_, "input key dtype must be int8 when input value dtype is int8.");
            return ge::GRAPH_FAILED;
        }
    } else {
        if (!(inputValueDtype_ == ge::DT_FLOAT16 || inputValueDtype_ == ge::DT_BF16)) {
            OP_LOGE(context_, "input value and value dtype must be float16 or bfloat16.");
            return ge::GRAPH_FAILED;
        }
    }
    if (params_.templateType == TEMPLATE_NHSD) {
        if (inputKeyDtype_ != inputValueDtype_) {
            OP_LOGE(context_, "key and value must be same dtype.");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheVllmMembaseTiling::GetInputDtype()
{
    if (params_.templateType == TEMPLATE_SISO || params_.templateType == TEMPLATE_SISO_NCT) {
        if (GetInputDtypeSiso() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else {
        if (GetInputDtypeKv() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

/****************************************获取并校验输入维度******************************************* */
ge::graphStatus ScatterPaKvCacheVllmMembaseTiling::CheckInputDimNumSiso()
{
    size_t kDimNum = inputKeyShape_.GetDimNum();
    size_t kCacheDimNum = inputKeyCacheInShape_.GetDimNum();
    size_t slotDimNum = slotMappingShape_.GetDimNum();

    OP_CHECK_IF((kDimNum != static_cast<size_t>(DIM_3)), OP_LOGE(context_, "key should be is 3 dim."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((kCacheDimNum != static_cast<size_t>(DIM_4)), OP_LOGE(context_, "key_cache should be is 4 dim."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((slotDimNum != static_cast<size_t>(DIM_1)), OP_LOGE(context_, "slot_mapping should be is 1 dim."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheVllmMembaseTiling::CheckInputDimNumNorm()
{
    if (CheckInputDimNumSiso() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    size_t vDimNum = inputValueShape_.GetDimNum();
    size_t vCacheDimNum = inputValueCacheInShape_.GetDimNum();

    OP_CHECK_IF((vDimNum != static_cast<size_t>(DIM_3)), OP_LOGE(context_, "value should be is 3 dim."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((vCacheDimNum != static_cast<size_t>(DIM_4)), OP_LOGE(context_, "value_cache should be is 4 dim."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheVllmMembaseTiling::CheckInputDimNumNz()
{
    size_t kDimNum = inputKeyShape_.GetDimNum();
    size_t kCacheDimNum = inputKeyCacheInShape_.GetDimNum();
    size_t slotDimNum = slotMappingShape_.GetDimNum();
    size_t vDimNum = inputValueShape_.GetDimNum();
    size_t vCacheDimNum = inputValueCacheInShape_.GetDimNum();

    OP_CHECK_IF((kDimNum != static_cast<size_t>(DIM_3)), OP_LOGE(context_, "key should be is 3 dim."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((kCacheDimNum != static_cast<size_t>(DIM_4) && kCacheDimNum != static_cast<size_t>(DIM_5)),
                OP_LOGE(context_, "key_cache should be is 4 dim or 5 dim."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((slotDimNum != static_cast<size_t>(DIM_1)), OP_LOGE(context_, "slot_mapping should be is 1 dim."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((vDimNum != static_cast<size_t>(DIM_3)), OP_LOGE(context_, "value should be is 3 dim."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((vCacheDimNum != static_cast<size_t>(DIM_4) && vCacheDimNum != static_cast<size_t>(DIM_5)),
                OP_LOGE(context_, "value_cache should be is 4 dim or 5 dim."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheVllmMembaseTiling::CheckInputDimNumCompress()
{
    if (CheckInputDimNumNorm() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    size_t compressLensDimNum = compressLensShape_.GetDimNum();
    size_t seqLensDimNum = seqLensShape_.GetDimNum();

    OP_CHECK_IF((compressLensDimNum != static_cast<size_t>(DIM_1)),
                OP_LOGE(context_, "compressLens should be is 1 dim."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((seqLensDimNum != static_cast<size_t>(DIM_1)), OP_LOGE(context_, "seqLensDimNum should be is 1 dim."),
                return ge::GRAPH_FAILED);

    if (params_.templateType == TEMPLATE_ROPE || params_.templateType == TEMPLATE_OMNI) {
        compressSeqOffsetShape_ = context_->GetInputShape(INPUT_SEQ_LENS)->GetOriginShape();
        size_t compressSeqOffsetDimNum = compressSeqOffsetShape_.GetDimNum();
        OP_CHECK_IF((compressSeqOffsetDimNum != static_cast<size_t>(DIM_1)),
                    OP_LOGE(context_, "compressSeqOffsetDimNum should be is 1 dim."), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheVllmMembaseTiling::CheckInputDimNumNHSD()
{
    size_t kDimNum = inputKeyShape_.GetDimNum();
    size_t kCacheDimNum = inputKeyCacheInShape_.GetDimNum();
    size_t slotDimNum = slotMappingShape_.GetDimNum();
    size_t vDimNum = inputValueShape_.GetDimNum();
    size_t vCacheDimNum = inputValueCacheInShape_.GetDimNum();
    OP_CHECK_IF((slotDimNum != static_cast<size_t>(DIM_1)), OP_LOGE(context_, "slot_mapping should be 1 dim."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((kCacheDimNum != static_cast<size_t>(DIM_4)), OP_LOGE(context_, "key_cache should be 4 dim."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((kDimNum != static_cast<size_t>(DIM_3)), OP_LOGE(context_, "key should be 3 dim."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((vCacheDimNum != static_cast<size_t>(DIM_4)), OP_LOGE(context_, "value_cache should be 4 dim."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((vDimNum != static_cast<size_t>(DIM_3)), OP_LOGE(context_, "value should be 3 dim."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheVllmMembaseTiling::CheckInputDimNum()
{
    inputKeyShape_ = context_->GetInputShape(INPUT_KEY_INDEX)->GetOriginShape();
    inputKeyCacheInShape_ = context_->GetInputShape(INPUT_KEY_CACHE_INDEX)->GetOriginShape();
    slotMappingShape_ = context_->GetInputShape(INPUT_SLOT_MAPPING_INDEX)->GetOriginShape();

    if (params_.templateType == TEMPLATE_SISO || params_.templateType == TEMPLATE_SISO_NCT) {
        return CheckInputDimNumSiso();
    } else {
        inputValueShape_ = context_->GetInputShape(INPUT_VALUE_INDEX)->GetOriginShape();
        inputValueCacheInShape_ = context_->GetInputShape(INPUT_VALUE_CACHE_INDEX)->GetOriginShape();
        if (params_.templateType == TEMPLATE_NORMAL || params_.templateType == TEMPLATE_NORM_NCT) {
            return CheckInputDimNumNorm();
        } else if (params_.templateType == TEMPLATE_NZ) {
            return CheckInputDimNumNz();
        } else if (params_.templateType == TEMPLATE_ALIBI || params_.templateType == TEMPLATE_ROPE ||
            params_.templateType == TEMPLATE_OMNI) {
            compressLensShape_ = context_->GetInputShape(INPUT_COMPRESS_LENS)->GetOriginShape();
            seqLensShape_ = context_->GetInputShape(INPUT_SEQ_LENS)->GetOriginShape();
            return CheckInputDimNumCompress();
        } else if (params_.templateType == TEMPLATE_NHSD) {
            return CheckInputDimNumNHSD();
        }
    }
    return ge::GRAPH_FAILED;
}

/****************************************获取并校验输入shape******************************************* */
ge::graphStatus ScatterPaKvCacheVllmMembaseTiling::CheckInputShapeSiso()
{
    params_.numTokens = inputKeyShape_.GetDim(DIM_0);
    params_.numHead = inputKeyShape_.GetDim(DIM_1);
    params_.kHeadSize = inputKeyShape_.GetDim(DIM_2);
    int64_t numBlocks = inputKeyCacheInShape_.GetDim(DIM_0);
    params_.blockSize = inputKeyCacheInShape_.GetDim(DIM_1);

    OP_CHECK_IF((static_cast<uint64_t>(numBlocks) * params_.blockSize < params_.numTokens),
                OP_LOGE(context_, "numBlocks * blockSize should larger than numTokens."), return ge::GRAPH_FAILED);
    if (!(params_.templateType == TEMPLATE_ALIBI || params_.templateType == TEMPLATE_ROPE ||
        params_.templateType == TEMPLATE_OMNI)) {
        OP_CHECK_IF((params_.numHead != inputKeyCacheInShape_.GetDim(DIM_2)),
                    OP_LOGE(context_, "dim2 of keyCache should be same as numHead."), return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF((params_.kHeadSize != inputKeyCacheInShape_.GetDim(DIM_3)),
                OP_LOGE(context_, "dim3 of keyCache should be same as kHeadSize."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheVllmMembaseTiling::CheckInputShapeNorm()
{
    if (CheckInputShapeSiso() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    params_.vHeadSize = inputValueShape_.GetDim(DIM_2);
    OP_CHECK_IF((params_.vHeadSize != inputValueCacheInShape_.GetDim(DIM_3)),
                OP_LOGE(context_, "dim3 of ValueCache should be same as vHeadSize."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((inputKeyShape_.GetDim(DIM_0) != inputValueShape_.GetDim(DIM_0)),
                OP_LOGE(context_, "dim0 of key should be same as Value."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((inputKeyShape_.GetDim(DIM_1) != inputValueShape_.GetDim(DIM_1)),
                OP_LOGE(context_, "dim1 of key should be same as Value."), return ge::GRAPH_FAILED);

    OP_CHECK_IF((inputKeyCacheInShape_.GetDim(DIM_0) != inputValueCacheInShape_.GetDim(DIM_0)),
                OP_LOGE(context_, "dim0 of keyCache should be same as ValueCache."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((inputKeyCacheInShape_.GetDim(DIM_1) != inputValueCacheInShape_.GetDim(DIM_1)),
                OP_LOGE(context_, "dim1 of keyCache should be same as ValueCache."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((inputKeyCacheInShape_.GetDim(DIM_2) != inputValueCacheInShape_.GetDim(DIM_2)),
                OP_LOGE(context_, "dim2 of keyCache should be same as ValueCache."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheVllmMembaseTiling::CheckInputShapeNHSD()
{
    params_.numTokens = inputKeyShape_.GetDim(DIM_0);
    params_.numHead = inputKeyShape_.GetDim(DIM_1);
    params_.kHeadSize = inputKeyShape_.GetDim(DIM_2);
    params_.vHeadSize = inputValueShape_.GetDim(DIM_2);
    int64_t numBlocks = inputKeyCacheInShape_.GetDim(DIM_0);
    params_.blockSize = inputKeyCacheInShape_.GetDim(DIM_2);
    bool isAlign = ((params_.kHeadSize * params_.typeByteK) % ALIGN == 0 &&
                        (params_.vHeadSize * params_.typeByteV) % ALIGN == 0);
    OP_CHECK_IF((!isAlign), OP_LOGE(context_, "kHeadSize and vHeadSize should be align to 32."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((params_.numHead != inputKeyCacheInShape_.GetDim(DIM_1)),
                OP_LOGE(context_, "dim1 of keyCache should be same as numHead."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((params_.numHead > NUM_HEAD_MAX),
                OP_LOGE(context_, "num head must less than or equal 4095."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((static_cast<uint64_t>(numBlocks) * params_.blockSize < params_.numTokens),
                OP_LOGE(context_, "numBlocks * blockSize should larger than numTokens."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((inputKeyCacheInShape_.GetDim(DIM_2) != inputValueCacheInShape_.GetDim(DIM_2)),
                OP_LOGE(context_, "dim2 of keyCache should be same as ValueCache."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((inputKeyCacheInShape_.GetDim(DIM_1) != inputValueCacheInShape_.GetDim(DIM_1)),
                OP_LOGE(context_, "dim1 of keyCache should be same as ValueCache."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((inputKeyShape_.GetDim(DIM_0) != inputValueShape_.GetDim(DIM_0)),
                OP_LOGE(context_, "dim0 of key should be same as Value."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((inputKeyShape_.GetDim(DIM_1) != inputValueShape_.GetDim(DIM_1)),
                OP_LOGE(context_, "dim1 of key should be same as Value."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((params_.kHeadSize != inputKeyCacheInShape_.GetDim(DIM_3)),
                OP_LOGE(context_, "dim3 of keyCache should be same as kHeadSize."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((params_.vHeadSize != inputValueCacheInShape_.GetDim(DIM_3)),
                OP_LOGE(context_, "dim3 of ValueCache should be same as vHeadSize."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((inputKeyCacheInShape_.GetDim(DIM_0) != inputValueCacheInShape_.GetDim(DIM_0)),
                OP_LOGE(context_, "dim0 of keyCache should be same as ValueCache."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheVllmMembaseTiling::CheckInputShapeNz()
{
    params_.numTokens = inputKeyShape_.GetDim(DIM_0);
    params_.numHead = inputKeyShape_.GetDim(DIM_1);
    params_.kHeadSize = inputKeyShape_.GetDim(DIM_2);
    params_.vHeadSize = inputValueShape_.GetDim(DIM_2);
    params_.blockSize = inputKeyCacheInShape_.GetDim(inputKeyCacheInShape_.GetDimNum() - DIM_2);
    int64_t numBlocks = inputKeyCacheInShape_.GetDim(DIM_0);
    int64_t lastDimK = ALIGN / params_.typeByteK;
    int64_t lastDimV = ALIGN / params_.typeByteV;

    OP_CHECK_IF((params_.blockSize > UINT16_MAX),
                OP_LOGE(context_, "blockSize should less than UINT16_MAX."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((static_cast<uint64_t>(numBlocks) * params_.blockSize < params_.numTokens),
                OP_LOGE(context_, "numBlocks * blockSize should larger than numTokens."), return ge::GRAPH_FAILED);

    OP_CHECK_IF((inputKeyShape_.GetDim(DIM_0) != inputValueShape_.GetDim(DIM_0)),
                OP_LOGE(context_, "dim0 of key should be same as Value."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((inputKeyShape_.GetDim(DIM_1) != inputValueShape_.GetDim(DIM_1)),
                OP_LOGE(context_, "dim1 of key should be same as Value."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((inputKeyCacheInShape_.GetDim(DIM_0) != inputValueCacheInShape_.GetDim(DIM_0)),
                OP_LOGE(context_, "dim0 of keyCache should be same as ValueCache."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((params_.blockSize != inputValueCacheInShape_.GetDim(inputValueCacheInShape_.GetDimNum() - DIM_2)),
                OP_LOGE(context_, "blockSize of keyCache should be same as ValueCache."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((lastDimK != inputKeyCacheInShape_.GetDim(inputKeyCacheInShape_.GetDimNum() - DIM_1)),
                OP_LOGE(context_, "lastDim of keyCache should be same as lastDimK."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((lastDimV != inputValueCacheInShape_.GetDim(inputValueCacheInShape_.GetDimNum() - DIM_1)),
                OP_LOGE(context_, "lastDim of valueCache should be same as lastDimV."), return ge::GRAPH_FAILED);

    if (inputKeyCacheInShape_.GetDimNum() == 4) {
        OP_CHECK_IF((params_.numHead * params_.kHeadSize / lastDimK != inputKeyCacheInShape_.GetDim(DIM_1)),
                    OP_LOGE(context_, "dim1 of keyCache should be same as numHead * kHeadSize / lastDimK."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((params_.numHead * params_.vHeadSize / lastDimV != inputValueCacheInShape_.GetDim(DIM_1)),
                    OP_LOGE(context_, "dim1 of valueCache should be same as numHead * vHeadSize / lastDimV."),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF((params_.numHead != inputKeyCacheInShape_.GetDim(DIM_1)),
                    OP_LOGE(context_, "dim1 of keyCache should be same as numHead."), return ge::GRAPH_FAILED);
        OP_CHECK_IF((params_.kHeadSize / lastDimK != inputKeyCacheInShape_.GetDim(DIM_2)),
                    OP_LOGE(context_, "dim2 of keyCache should be same as kHeadSize / lastDimK."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((params_.numHead != inputValueCacheInShape_.GetDim(DIM_1)),
                    OP_LOGE(context_, "dim1 of valueCache should be same as numHead."), return ge::GRAPH_FAILED);
        OP_CHECK_IF((params_.vHeadSize / lastDimV != inputValueCacheInShape_.GetDim(DIM_2)),
                    OP_LOGE(context_, "dim2 of keyCache should be same as vHeadSize / lastDimV."),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheVllmMembaseTiling::CheckInputShapeCompress()
{
    if (CheckInputShapeNorm() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF((inputKeyCacheInShape_.GetDim(DIM_2) != 1),
                OP_LOGE(context_, "dim2 of keyCache should be 1."), return ge::GRAPH_FAILED);

    auto seqLens = context_->GetOptionalInputTensor(INPUT_SEQ_LENS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, seqLens);

    params_.numBatchs = static_cast<uint32_t>(seqLens->GetStorageShape().GetDim(DIM_0));
    params_.numTokens = params_.numHead * params_.numBatchs;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheVllmMembaseTiling::CheckInputShape()
{
    if (params_.templateType == TEMPLATE_SISO || params_.templateType == TEMPLATE_SISO_NCT) {
        return CheckInputShapeSiso();
    } else if (params_.templateType == TEMPLATE_NORMAL || params_.templateType == TEMPLATE_NORM_NCT) {
        return CheckInputShapeNorm();
    } else if (params_.templateType == TEMPLATE_NZ) {
        return CheckInputShapeNz();
    } else if (params_.templateType == TEMPLATE_ALIBI || params_.templateType == TEMPLATE_ROPE ||
        params_.templateType == TEMPLATE_OMNI) {
        return CheckInputShapeCompress();
    } else if (params_.templateType == TEMPLATE_NHSD) {
        return CheckInputShapeNHSD();
    }
    return ge::GRAPH_FAILED;
}

/****************************************获取模式******************************************* */
ge::graphStatus ScatterPaKvCacheVllmMembaseTiling::GetTemplateType()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto cacheMode = attrs->GetStr(INPUT_CACHE_MODE_INDEX);
    auto scatterMode = attrs->GetStr(INPUT_SCATTER_MODE_INDEX);
    if (strcmp(cacheMode, "PA_NZ") == 0) {
        // entering template nz
        params_.templateType = TEMPLATE_NZ;
        return ge::GRAPH_SUCCESS;
    } else if (strcmp(cacheMode, "") == 0 || strcmp(cacheMode, "Norm") == 0) {
        if (strcmp(scatterMode, "Rope") == 0) {
            params_.templateType = TEMPLATE_ROPE;
        } else if (strcmp(scatterMode, "Alibi") == 0) {
            params_.templateType = TEMPLATE_ALIBI;
        } else if (strcmp(scatterMode, "Omni") == 0) {
            params_.templateType = TEMPLATE_OMNI;
        } else if (strcmp(scatterMode, "Nct") == 0) {
            params_.templateType = TEMPLATE_NORM_NCT;
        } else if (strcmp(scatterMode, "NHSD") == 0) {
            params_.templateType = TEMPLATE_NHSD;
        } else if (strcmp(scatterMode, "") == 0 || strcmp(scatterMode, "None") == 0) {
            params_.templateType = TEMPLATE_NORMAL;
        } else {
            OP_LOGE(context_, "scatterMode only support None, Rope, Alibi, Omni, Nct.");
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_LOGE(context_, "cacheMode only support None, Norm or PA_NZ.");
        return ge::GRAPH_FAILED;
    }

    if (context_->GetInputTensor(INPUT_VALUE_CACHE_INDEX)->GetOriginShape().GetDimNum() == 0) {
        if (params_.templateType == TEMPLATE_NORM_NCT) {
            params_.templateType = TEMPLATE_SISO_NCT;
        } else  if (params_.templateType == TEMPLATE_NORMAL) {
            params_.templateType = TEMPLATE_SISO;
        } else {
            OP_LOGE(context_, "Siso only support templateType is TEMPLATE_NORM_NCT or TEMPLATE_NORMAL.");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

/****************************************获取并校验输入与属性******************************************* */
ge::graphStatus ScatterPaKvCacheVllmMembaseTiling::GetShapeAttrsInfo()
{
    if (GetTemplateType() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetInputDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckInputDimNum() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckInputShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto attrs = context_->GetAttrs();
    auto strides = attrs->GetListInt(INPUT_STRIDES_INDEX);
    auto offsets = attrs->GetListInt(INPUT_OFFSET_INDEX);
    if (params_.templateType == TEMPLATE_NORM_NCT || params_.templateType == TEMPLATE_SISO_NCT) {
        params_.kStride = strides->GetData()[0];
        params_.vStride = strides->GetData()[1];
        params_.kOffset = offsets->GetData()[0];
        params_.vOffset = offsets->GetData()[1];
    }
    auto kvCacheStrides = attrs->GetListInt(INPUT_KV_CACHE_STRIDES_INDEX);
    if (kvCacheStrides != nullptr && kvCacheStrides->GetSize() >= 2) {
        params_.kCacheBlockStride = kvCacheStrides->GetData()[0];
        params_.vCacheBlockStride = kvCacheStrides->GetData()[1];
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheVllmMembaseTiling::DoNHSDOpTiling()
{
    OP_CHECK_IF(((params_.numHead * params_.kHeadSize + params_.numHead * params_.vHeadSize) * params_.typeByteK 
                    > params_.ubSize),
                    OP_LOGE(context_, "one token key size plus one token value must less than 196608."),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF((params_.blockSize * params_.kHeadSize > UINT32_MAX),
                    OP_LOGE(context_, "blockSize * kHeadSize must less than UINT32_MAX."),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF((params_.blockSize * params_.vHeadSize > UINT32_MAX),
                    OP_LOGE(context_, "blockSize * vHeadSize must less than UINT32_MAX."),
                    return ge::GRAPH_FAILED);
    params_.blockFactor = CeilDiv<int64_t>(params_.numTokens, params_.usedCoreNum);
    params_.usedCoreNum =
        std::min(CeilDiv<int64_t>(params_.numTokens, params_.blockFactor), params_.usedCoreNum);
    params_.tailBlockFactor = params_.numTokens - params_.blockFactor * (params_.usedCoreNum - 1);
    params_.tilingKey = TILING_ID_TEMPLATE * params_.templateType;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheVllmMembaseTiling::DoNoCompressOpTiling()
{
    if (params_.templateType == TEMPLATE_NZ) {
        bool isAlign = ((params_.kHeadSize * params_.typeByteK) % ALIGN == 0 &&
                        (params_.vHeadSize * params_.typeByteV) % ALIGN == 0);
        OP_CHECK_IF((!isAlign), OP_LOGE(context_, "kHeadSize and vHeadSize should be align to 32."),
                    return ge::GRAPH_FAILED);
    }
    bool canLoad = true;
    if (params_.templateType == TEMPLATE_SISO || params_.templateType == TEMPLATE_SISO_NCT) {
        canLoad = (static_cast<uint64_t>(params_.numHead) * params_.kHeadSize * params_.typeByteK <= params_.ubSize);
    } else {
        canLoad = ((static_cast<uint64_t>(params_.numHead) * params_.kHeadSize * params_.typeByteK +
                   static_cast<uint64_t>(params_.numHead) * params_.vHeadSize * params_.typeByteV) <= params_.ubSize);
    }

    params_.blockFactor = CeilDiv<int64_t>(params_.numTokens, params_.usedCoreNum);
    params_.usedCoreNum =
        std::min(CeilDiv<int64_t>(params_.numTokens, params_.blockFactor), params_.usedCoreNum);
    params_.tailBlockFactor = params_.numTokens - params_.blockFactor * (params_.usedCoreNum - 1);

    if (canLoad) {
        if (params_.templateType == TEMPLATE_NZ || params_.numTokens < SMALL_TOKEN) {
            params_.tilingKey = TILING_ID_TEMPLATE * params_.templateType + TILING_ID_FULL;
        } else {
            params_.tilingKey = TILING_ID_TEMPLATE * params_.templateType;
        }
    } else {
        params_.tilingKey = TILING_ID_TEMPLATE * params_.templateType;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheVllmMembaseTiling::DoCompressAlibiOpTiling()
{
    params_.usedCoreNum = params_.numTokens < params_.usedCoreNum ? params_.numTokens : params_.usedCoreNum;

    params_.tilingKey = TILING_ID_TEMPLATE * params_.templateType;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheVllmMembaseTiling::DoCompressRopeAndOmniOpTiling()
{
    params_.usedCoreNum = params_.numTokens * TASK_MULTIPLE <= params_.usedCoreNum ? params_.numTokens * TASK_MULTIPLE :
                                                                                     params_.usedCoreNum;
    params_.tilingKey = TILING_ID_TEMPLATE * params_.templateType;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheVllmMembaseTiling::DoOpTiling()
{
    if (params_.templateType == TEMPLATE_NORMAL || params_.templateType == TEMPLATE_NORM_NCT ||
        params_.templateType == TEMPLATE_SISO || params_.templateType == TEMPLATE_SISO_NCT ||
        params_.templateType == TEMPLATE_NZ) {
        return DoNoCompressOpTiling();
    } else if (params_.templateType == TEMPLATE_ROPE || params_.templateType == TEMPLATE_OMNI) {
        return DoCompressRopeAndOmniOpTiling();
    } else if (params_.templateType == TEMPLATE_ALIBI) {
        return DoCompressAlibiOpTiling();
    } else if (params_.templateType == TEMPLATE_NHSD) {
        return DoNHSDOpTiling();
    } else {
        OP_LOGE(context_, "templateType is invalid.");
    }
    return ge::GRAPH_FAILED;
}

uint64_t ScatterPaKvCacheVllmMembaseTiling::GetTilingKey() const
{
    return params_.tilingKey;
}

ge::graphStatus ScatterPaKvCacheVllmMembaseTiling::GetWorkspaceSize()
{
    workspaceSize_ = params_.sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheVllmMembaseTiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;

    context_->SetBlockDim(params_.usedCoreNum);
    tilingData_.set_usedCoreNum(params_.usedCoreNum);
    tilingData_.set_blockFactor(params_.blockFactor);
    tilingData_.set_tailBlockFactor(params_.tailBlockFactor);
    tilingData_.set_numTokens(params_.numTokens);
    tilingData_.set_kHeadSize(params_.kHeadSize);
    tilingData_.set_vHeadSize(params_.vHeadSize);
    tilingData_.set_numHead(params_.numHead);
    tilingData_.set_blockSize(params_.blockSize);
    tilingData_.set_ubSize(params_.ubSize);
    tilingData_.set_batch(params_.numBatchs);
    tilingData_.set_kStride(params_.kStride);
    tilingData_.set_vStride(params_.vStride);
    tilingData_.set_kOffset(params_.kOffset);
    tilingData_.set_vOffset(params_.vOffset);
    tilingData_.set_kCacheBlockStride(params_.kCacheBlockStride);
    tilingData_.set_vCacheBlockStride(params_.vCacheBlockStride);

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
void ScatterPaKvCacheVllmMembaseTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "totalCoreNum: " << params_.usedCoreNum << std::endl;
    info << "usedCoreNum: " << params_.usedCoreNum << std::endl;
    info << "blockFactor: " << params_.blockFactor << std::endl;
    info << "tailBlockFactor: " << params_.tailBlockFactor << std::endl;
    info << "numTokens: " << params_.numTokens << std::endl;
    info << "numHead: " << params_.numHead  << std::endl;
    info << "kHeadSize: " << params_.kHeadSize << std::endl;
    info << "vHeadSize: " << params_.vHeadSize << std::endl;
    info << "blockSize: " << params_.blockSize << std::endl;
    info << "typeByte: " << params_.typeByteSlot << std::endl;
    info << "kStride: " << params_.kStride << std::endl;
    info << "vStride: " << params_.vStride << std::endl;
    info << "kOffset: " << params_.kOffset << std::endl;
    info << "vOffset: " << params_.vOffset << std::endl;
    info << "kCacheBlockStride: " << params_.kCacheBlockStride << std::endl;
    info << "vCacheBlockStride: " << params_.vCacheBlockStride << std::endl;
    info << "tilingKey: " << params_.tilingKey << std::endl;
    OP_LOGD(context_, "%s", info.str().c_str());
}

ge::graphStatus Tiling4ScatterPaKvCacheVllm(gert::TilingContext *context_)
{
        ScatterPaKvCacheVllmMembaseTiling tiling(context_);
        return tiling.DoTiling();
}

ge::graphStatus TilingPrepare4ScatterPaKvCacheVllm(gert::TilingParseContext *context_)
{
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the ScatterPaKvCache op.
IMPL_OP_OPTILING(ScatterPaKvCacheVllm)
    .Tiling(Tiling4ScatterPaKvCacheVllm)
    .TilingParse<ScatterPaKvCacheVllmCompileInfo>(TilingPrepare4ScatterPaKvCacheVllm);

} // namespace optiling

