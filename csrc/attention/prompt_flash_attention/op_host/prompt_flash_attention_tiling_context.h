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
 * \file prompt_flash_attention_tiling_context.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_TILING_CONTEXT_H
#define PROMPT_FLASH_ATTENTION_TILING_CONTEXT_H
#include "op_host/data_copy_transpose_tiling_def.h"
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "prompt_flash_attention_tiling_compile_info.h"

namespace optiling {

/*
contextParams is a new structured defined for the use of FusedInferAttentionScore op.
It is meant to catch and organize all the necessary variables passed by FIAS tilling function.
It will be used as the input to the new 'runBigKernelWithParams' function in PFA tilling.
The old PFA tillingContext will also be transformed to this structure in the future.
*/
struct ContextParamsForPFATiling {
    const gert::Tensor *pseShift = nullptr;
    const gert::Tensor *attentionMask = nullptr;
    const gert::Tensor *actualSequenceLengthQ = nullptr;
    const gert::Tensor *actualSequenceLengthKV = nullptr;
    const gert::Tensor *antiquantScale = nullptr;
    const gert::Tensor *antiquantOffset = nullptr;
    const gert::Tensor *queryPaddingSize = nullptr;
    const gert::Tensor *kvPaddingSize = nullptr;
    const gert::Tensor *blockTable = nullptr;
    const gert::Tensor *keySharedPrefix = nullptr;
    const gert::Tensor *valueSharedPrefix = nullptr;
    const gert::Tensor *actualSharedPrefixLen = nullptr;
    const gert::Tensor *learnableSink = nullptr;

    const gert::Tensor *keyAntiquantScale = nullptr;
    const gert::Tensor *valueAntiquantScale = nullptr;
    const gert::Tensor *KeyAntiquantOffset = nullptr;
    const gert::Tensor *valueAntiquantOffset = nullptr;
    const gert::Tensor *dequantScaleQuery =nullptr;

    const gert::Tensor *qStartIdx = nullptr;
    const gert::Tensor *kvStartIdx = nullptr;

    ge::DataType inputDataType = ge::DataType::DT_FLOAT16;
    ge::DataType kDataType = ge::DataType::DT_FLOAT16;
    ge::DataType vDataType = ge::DataType::DT_FLOAT16;
    ge::DataType qRopeDataType = ge::DataType::DT_FLOAT16;
    ge::DataType kRopeDataType = ge::DataType::DT_FLOAT16;
    ge::DataType pseShiftDataType = ge::DataType::DT_FLOAT16;
    ge::DataType maskDataType = ge::DataType::DT_FLOAT16;
    ge::DataType blockTableType = ge::DataType::DT_FLOAT16;
    ge::DataType outputDataType = ge::DataType::DT_FLOAT16;
    ge::DataType keySharedPrefixDataType = ge::DataType::DT_FLOAT16;
    ge::DataType valueSharedPrefixDataType = ge::DataType::DT_FLOAT16;
    ge::DataType learnableSinkDataType = ge::DataType::DT_FLOAT16;
    const char *opName = nullptr;
    const gert::StorageShape *queryInputShape = nullptr;
    const gert::StorageShape *keyInputShape = nullptr;
    const gert::StorageShape *queryRopeInputShape = nullptr;
    const gert::StorageShape *keyRopeInputShape = nullptr;
    const gert::StorageShape *valueInputShape = nullptr;
    const gert::StorageShape *pseShiftShape = nullptr;
    const gert::StorageShape *attentionMaskShape = nullptr;
    const gert::StorageShape *deqScale1Shape = nullptr;
    const gert::StorageShape *scale1Shape = nullptr;
    const gert::StorageShape *deqScale2Shape = nullptr;
    const gert::StorageShape *scale2Shape = nullptr;
    const gert::StorageShape *offset2Shape = nullptr;
    const gert::StorageShape *antiquantScaleShape = nullptr;
    const gert::StorageShape *antiquantOffsetShape = nullptr;
    const gert::StorageShape *blockTableShape = nullptr;
    const gert::StorageShape *outputShape = nullptr;
    const gert::StorageShape *lseoutputShape = nullptr;

    const gert::StorageShape *dequantScaleQueryShape = nullptr;
    const gert::StorageShape *KeyAntiquantScaleShape = nullptr;
    const gert::StorageShape *valueAntiquantScaleShape = nullptr;
    const gert::StorageShape *KeyAntiquantOffsetShape = nullptr;
    const gert::StorageShape *valueAntiquantOffsetShape = nullptr;
    const gert::StorageShape *queryRope = nullptr;
    const gert::StorageShape *keyRope = nullptr;
    const gert::StorageShape *learnableSinkShape = nullptr;
    ge::DataType dequantScaleQueryType = ge::DataType::DT_FLOAT16;   
    ge::DataType KeyAntiquantScaleType = ge::DataType::DT_FLOAT16;
    ge::DataType valueAntiquantScaleType = ge::DataType::DT_FLOAT16;
    ge::DataType KeyAntiquantOffsetType = ge::DataType::DT_FLOAT16;
    ge::DataType valueAntiquantOffsetType = ge::DataType::DT_FLOAT16;

    const int64_t *innerPrecisePtr = nullptr;
    const int64_t *headsNumber = nullptr;
    const int32_t *sparseMode = nullptr;
    const int64_t *preToken = nullptr;
    const int64_t *nextToken = nullptr;
    const float *scaleValue = nullptr;
    const int32_t *blockSize = nullptr;
    const char *layout = nullptr;
    const int64_t *numKeyValueHeads = nullptr;
    size_t *workspaceSize = nullptr;
    const int64_t *pseType = nullptr;
    const PromptFlashAttentionCompileInfo *compileInfoPtr = nullptr;
    ge::DataType deqScaleType = ge::DataType::DT_FLOAT16;
    ge::DataType deqScale2Type = ge::DataType::DT_FLOAT16;
    ge::DataType quantScale2Type = ge::DataType::DT_FLOAT16;
    ge::DataType quantOffset2Type = ge::DataType::DT_FLOAT16;
    uint32_t isKvContinuous = 1;
    std::vector<const gert::StorageShape *> kTensorList = {nullptr};
    std::vector<const gert::StorageShape *> vTensorList = {nullptr};
    uint32_t maxKVs = 0;
    uint32_t fromFused = 0;
    uint32_t emptyTensor = 0;
    uint32_t isBSNDOut = 0;
    uint32_t transposeLayout = 0;
    const bool *softmaxLseFlag = nullptr;
    bool isSoftMaxLseEnable = false;
    uint32_t fromTilingSink = 0; // Flag indicating whether it is the step to enter the workspace calculation from tiling sinking
    bool hasKeyAntiquantScale = 0;
    bool hasValueAntiquantScale = 0;
    uint32_t isMsd = 0;
    const int64_t *keyAntiquantMode = nullptr;
    const int64_t *valueAntiquantMode = nullptr;
    const int64_t *queryQuantMode = nullptr;
    bool hasKeyAntiquantOffset = 0;
    bool hasLearnableSink = 0;
};

} // namespace optiling

#endif // PROMPT_FLASH_ATTENTION_TILING_CONTEXT_H