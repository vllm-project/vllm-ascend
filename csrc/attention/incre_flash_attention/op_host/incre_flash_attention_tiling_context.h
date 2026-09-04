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
 * \file incre_flash_attention_tiling_context.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_CONTEXT_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_CONTEXT_H_


#ifdef ASCENDC_OP_TEST
#define IFA_EXTERN_C extern "C"
#else
#define IFA_EXTERN_C
#endif
namespace optiling {
struct RequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct OptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

struct IncreFlashAttentionContext {
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    RequiredParaInfo query = {nullptr, nullptr};
    RequiredParaInfo key = {nullptr, nullptr};
    RequiredParaInfo value = {nullptr, nullptr};
    OptionalParaInfo pseShift = {nullptr, nullptr};
    OptionalParaInfo attenMask = {nullptr, nullptr};
    OptionalParaInfo actualSeqLengthsQ = {nullptr, nullptr};
    OptionalParaInfo actualSeqLengths = {nullptr, nullptr};
    OptionalParaInfo deqScale1 = {nullptr, nullptr};
    OptionalParaInfo quantScale1 = {nullptr, nullptr};
    OptionalParaInfo deqScale2 = {nullptr, nullptr};
    OptionalParaInfo quantScale2 = {nullptr, nullptr};
    OptionalParaInfo quantOffset2 = {nullptr, nullptr};
    OptionalParaInfo antiquantScale = {nullptr, nullptr};
    OptionalParaInfo antiquantOffset = {nullptr, nullptr};
    OptionalParaInfo blockTable = {nullptr, nullptr};
    OptionalParaInfo queryPaddingSize = {nullptr, nullptr};
    OptionalParaInfo kvPaddingSize = {nullptr, nullptr};
    OptionalParaInfo keyAntiquantScale = {nullptr, nullptr};
    OptionalParaInfo keyAntiquantOffset = {nullptr, nullptr};
    OptionalParaInfo valueAntiquantScale = {nullptr, nullptr};
    OptionalParaInfo valueAntiquantOffset = {nullptr, nullptr};
    OptionalParaInfo keySharedPrefix = {nullptr, nullptr};
    OptionalParaInfo valueSharedPrefix = {nullptr, nullptr};
    OptionalParaInfo actualSharedPrefixLen = {nullptr, nullptr};
    OptionalParaInfo queryRope = {nullptr, nullptr};
    OptionalParaInfo keyRope = {nullptr, nullptr};
    OptionalParaInfo keyRopeAntiquantScale = {nullptr, nullptr};
    OptionalParaInfo dequantScaleQuery = {nullptr, nullptr};
    OptionalParaInfo qStartIdx = {nullptr, nullptr};
    OptionalParaInfo kvStartIdx = {nullptr, nullptr};

    RequiredParaInfo attenOut = {nullptr, nullptr};
    RequiredParaInfo lseOut = {nullptr, nullptr};

    const uint32_t *numHeads = nullptr;
    const int64_t *preToken = nullptr;
    const int64_t *nextToken = nullptr;
    const float *scaleValue = nullptr;
    const uint32_t *kvHeadNums = nullptr;
    const char *layOut = nullptr;
    const uint32_t *blockSize = nullptr;
    const uint32_t *innerPrecise = nullptr;
    const int64_t *antiquantMode = nullptr;
    const bool *softmaxLseFlag = nullptr;
    const int64_t *keyAntiquantMode = nullptr;
    const int64_t *valueAntiquantMode = nullptr;
    const uint32_t *sparseMode = nullptr;
    const int64_t *queryQuantMode = nullptr;
    const int64_t *pseType = nullptr;
    const int64_t *windowSize = nullptr;
    const gert::StorageShape *queryRopeInputShape = nullptr;
    const gert::StorageShape *keyRopeInputShape = nullptr;

    size_t *workSpaces = nullptr;
    std::vector<gert::StorageShape *> kCache = {nullptr};
    std::vector<gert::StorageShape *> vCache = {nullptr};
    uint64_t tilingKey = 0;
    uint32_t numBlocks = 0;
    uint32_t transposeLayout = 0;
};
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_CONTEXT_H_