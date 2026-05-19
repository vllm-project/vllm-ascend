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
 * \file kv_quant_sparse_attn_sharedkv_check_single_para.cpp
 * \brief
 */

#include "kv_quant_sparse_attn_sharedkv_check.h"
#include "../op_kernel/kv_quant_sparse_attn_sharedkv_metadata.h"

using namespace ge;
using namespace AscendC;
using std::map;
using std::string;
using std::pair;
namespace optiling {

static constexpr uint32_t DIM_0 = 0;
static constexpr uint32_t DIM_1 = 1;
static constexpr uint32_t DIM_2 = 2;
static constexpr uint32_t DIM_3 = 3;

const std::map<std::string, std::vector<ge::DataType>> DTYPE_SUPPORT_MAP = {
    {QUERY_NAME,                     {ge::DT_BF16}},
    {ORI_KV_NAME,                    {ge::DT_INT8, ge::DT_FLOAT8_E4M3FN}},
    {CMP_KV_NAME,                    {ge::DT_INT8, ge::DT_FLOAT8_E4M3FN}},
    {ATTEN_OUT_NAME,                 {ge::DT_FLOAT16, ge::DT_BF16}},
    {CMP_SPARSE_INDICES_NAME,        {ge::DT_INT32}},
    {ORI_BLOCK_TABLE_NAME,           {ge::DT_INT32}},
    {CMP_BLOCK_TABLE_NAME,           {ge::DT_INT32}},
    {CU_SEQLENS_Q_NAME,              {ge::DT_INT32}},
    {SEQUSED_KV_NAME,                {ge::DT_INT32}},
    {SINKS_NAME,                     {ge::DT_FLOAT}},
};

const std::map<std::string, std::vector<SASLayout>> LAYOUT_SUPPORT_MAP = {
    {QUERY_NAME,             {SASLayout::BSND, SASLayout::TND}},
    {ORI_KV_NAME,            {SASLayout::PA_ND}},
    {CMP_KV_NAME,            {SASLayout::PA_ND}},
    {ATTEN_OUT_NAME,         {SASLayout::BSND, SASLayout::TND}},
};

template <typename T>
void KvQuantSASTilingCheck::LogErrorDimNumSupport(const std::vector<T> &expectNumberList,
    const T &actualValue, const std::string &name) const
{
    LogErrorNumberSupport(expectNumberList, actualValue, name, "dimension");
}

ge::graphStatus KvQuantSASTilingCheck::CheckDimNumInLayoutSupport(const SASLayout &layout,
    const gert::StorageShape *shape, const std::string &name) const
{
    const auto& dimIt = SAS_LAYOUT_DIM_MAP.find(layout);
    OP_CHECK_IF(shape->GetStorageShape().GetDimNum() != dimIt->second,
        OP_LOGE(opName_, "When layout is %s, %s dimension should be %zu, but it's %zu",
            KvQuantSASLayoutToSerialString(layout).c_str(), name.c_str(), dimIt->second,
            shape->GetStorageShape().GetDimNum()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckDimNumSupport(const gert::StorageShape *shape,
    const std::vector<size_t> &expectDimNumList, const std::string &name) const
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (std::find(expectDimNumList.begin(), expectDimNumList.end(),
        shape->GetStorageShape().GetDimNum()) == expectDimNumList.end()) {
        LogErrorDimNumSupport(expectDimNumList, shape->GetStorageShape().GetDimNum(), name);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckShapeNumSupport(const gert::StorageShape *shape,
    const std::vector<int64_t> &expectShapeNumList, const std::string &name) const
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (std::find(expectShapeNumList.begin(), expectShapeNumList.end(),
        shape->GetStorageShape().GetShapeSize()) == expectShapeNumList.end()) {
        LogErrorDimNumSupport(expectShapeNumList, shape->GetStorageShape().GetShapeSize(), name);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

void KvQuantSASTilingCheck::LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList,
    const ge::DataType &actualDtype, const std::string &name) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectDtypeList.size(); ++i) {
        oss << SASDataTypeToSerialString(expectDtypeList[i]);
        if (i < expectDtypeList.size() - 1) {
            oss << ", ";
        }
    }
    OP_LOGE(opName_, "Tensor %s only support dtype %s, but got %s",
        name.c_str(), oss.str().c_str(), SASDataTypeToSerialString(actualDtype).c_str());
}

ge::graphStatus KvQuantSASTilingCheck::CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc,
    const std::string &name) const
{
    if (desc != nullptr) {
        const auto& it = DTYPE_SUPPORT_MAP.find(name);
        OP_CHECK_IF(it == DTYPE_SUPPORT_MAP.end(),
            OP_LOGE(opName_, "%s datatype support list should be specify in DTYPE_SUPPORT_MAP", name.c_str()),
            return ge::GRAPH_FAILED);
        auto &expectDtypeList = it->second;
        OP_CHECK_IF(std::find(
            expectDtypeList.begin(), expectDtypeList.end(), desc->GetDataType()) == expectDtypeList.end(),
            LogErrorDtypeSupport(expectDtypeList, desc->GetDataType(), name),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
void KvQuantSASTilingCheck::LogErrorNumberSupport(const std::vector<T> &expectNumberList,
    const T &actualValue, const std::string &name, const std::string subName) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectNumberList.size(); ++i) {
        oss << std::to_string(expectNumberList[i]);
        if (i < expectNumberList.size() - 1) {
            oss << ", ";
        }
    }
    OP_LOGE(opName_, "%s %s only support %s, but got %s",
              name.c_str(), subName.c_str(), oss.str().c_str(), std::to_string(actualValue).c_str());
}


void KvQuantSASTilingCheck::LogErrorLayoutSupport(const std::vector<SASLayout> &expectLayoutList,
    const SASLayout &actualLayout, const std::string &name) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectLayoutList.size(); ++i) {
        oss << KvQuantSASLayoutToSerialString(expectLayoutList[i]);
        if (i < expectLayoutList.size() - 1) {
            oss << ", ";
        }
    }
    OP_LOGE(opName_, "Tensor %s only support layout %s, but got %s",
        name.c_str(), oss.str().c_str(), KvQuantSASLayoutToSerialString(actualLayout).c_str());
}

ge::graphStatus KvQuantSASTilingCheck::CheckSingleParaQuery() const
{
    OP_CHECK_IF(opParamInfo_.q.desc == nullptr,
        OP_LOGE(opName_, "Input q is required, but got nullptr."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.q.shape->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE(opName_, "Any dim of input q cannot be 0 "),
        return ge::GRAPH_FAILED);

    const std::vector<size_t> queryDimNumList = {DIM_NUM_THREE, DIM_NUM_FOUR};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.q.desc, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(qLayout_, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(opParamInfo_.q.shape, queryDimNumList, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(qLayout_, opParamInfo_.q.shape, QUERY_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckSingleParaKey() const
{
    const std::vector<size_t> keyDimNumList = {DIM_NUM_THREE, DIM_NUM_FOUR};
    OP_CHECK_IF(opParamInfo_.oriKv.tensor == nullptr,
        OP_LOGE(opName_, "input oriKv can not be nullptr, but it's empty"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.oriKv.tensor->GetShapeSize() == 0,
        OP_LOGE(opName_, "Any dim of input oriKv cannot be 0 "),
        return ge::GRAPH_FAILED);

    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.oriKv.desc, ORI_KV_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(kvLayout_, ORI_KV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.oriKv.tensor->GetShape(), keyDimNumList, ORI_KV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(kvLayout_, &opParamInfo_.oriKv.tensor->GetShape(), ORI_KV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(oriBlockSize_ <= 0 || oriBlockSize_ > 1024,
        OP_LOGE(opName_, "when page attention is enabled, ori_block_size(%u) should be in range (0, %u].",
        oriBlockSize_, MAX_BLOCK_SIZE), return ge::GRAPH_FAILED);

    OP_CHECK_IF(oriBlockSize_ % 16 > 0,
        OP_LOGE(opName_, "when page attention is enabled, ori_block_size(%u) should be 16-aligned.",
        oriBlockSize_), return ge::GRAPH_FAILED);

    OP_CHECK_IF(dSizeOriKvInput_ != 640,
        OP_LOGE(opName_, "Dimension of OriKv only support 640, but got %u", dSizeOriKvInput_),
        return ge::GRAPH_FAILED);

    if (opParamInfo_.cmpKv.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.cmpKv.tensor->GetShapeSize() == 0,
            OP_LOGE(opName_, "Any dim of input cmpKv cannot be 0 "),
            return ge::GRAPH_FAILED);

        if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.cmpKv.desc, CMP_KV_NAME) ||
            ge::GRAPH_SUCCESS != CheckLayoutSupport(kvLayout_, CMP_KV_NAME) ||
            ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.cmpKv.tensor->GetShape(), keyDimNumList, CMP_KV_NAME) ||
            ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(kvLayout_, &opParamInfo_.cmpKv.tensor->GetShape(), CMP_KV_NAME)) {
            return ge::GRAPH_FAILED;
        }

        OP_CHECK_IF(dSizeCmpKvInput_ != 640,
            OP_LOGE(opName_, "Dimension of CmpKv only support 640, but got %u", dSizeCmpKvInput_),
            return ge::GRAPH_FAILED);

        uint32_t cmpKvN2Size_ = GetAxisNum(opParamInfo_.cmpKv.tensor->GetStorageShape(), SASAxis::N, kvLayout_);
        OP_CHECK_IF(cmpKvN2Size_ != n2Size_,
            OP_LOGE(opName_, "N2 size check failed! Expected cmpKvN2 == oriKvN2."),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(cmpBlockSize_ <= 0 || cmpBlockSize_ > 1024,
            OP_LOGE(opName_, "when page attention is enabled, cmp_block_size(%ld) should be in range (0, %u].",
            cmpBlockSize_, MAX_BLOCK_SIZE), return ge::GRAPH_FAILED);

        OP_CHECK_IF(cmpBlockSize_ % 16 > 0,
            OP_LOGE(opName_, "when page attention is enabled, cmp_block_size(%ld) should be 16-aligned.",
            cmpBlockSize_), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckLayoutSupport(const SASLayout &actualLayout, const std::string &name) const
{
    const auto& it = LAYOUT_SUPPORT_MAP.find(name);
    OP_CHECK_IF(it == LAYOUT_SUPPORT_MAP.end(),
        OP_LOGE(opName_, "%s layout support list should be specify in LAYOUT_SUPPORT_MAP", name.c_str()),
        return ge::GRAPH_FAILED);
    auto &expectLayoutList = it->second;
    OP_CHECK_IF(std::find(
        expectLayoutList.begin(), expectLayoutList.end(), actualLayout) == expectLayoutList.end(),
        LogErrorLayoutSupport(expectLayoutList, actualLayout, name),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckSingleParaNumHeads() const
{
    OP_CHECK_IF(n1Size_ != 64 && n1Size_ != 128,
        OP_LOGE(opName_, "n1Size_ only support 64 and 128 now, but got %u.", n1Size_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckSingleParaKvHeadNums() const
{
    OP_CHECK_IF(n2Size_ != 1,
        OP_LOGE(opName_, "n2Size_ only support 1 now, but got %u.", n2Size_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckSingleParaSparseMode() const
{
    OP_CHECK_IF((*opParamInfo_.oriMaskMode != 4 || *opParamInfo_.cmpMaskMode != 3),
        OP_LOGE(opName_, "oriMaskMode only support 4 and cmpMaskMode only support 3, but got %u and %u.", *opParamInfo_.oriMaskMode, *opParamInfo_.cmpMaskMode),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckSingleParaSparseBlockSize() const
{
    OP_CHECK_IF(sparseBlockSize_ != 1,
        OP_LOGE(opName_, "sparseBlockSize_ only support 1, but got %u",
            sparseBlockSize_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckSingleParaCmpSparseIndices() const
{
    const std::vector<size_t> cmpSparseIndicesDimNumList = {DIM_NUM_FOUR, DIM_NUM_THREE};

    if (opParamInfo_.cmpSparseIndices.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.cmpSparseIndices.tensor->GetShapeSize() == 0,
            OP_LOGE(opName_, "Any dim of input cmpSparseIndices cannot be 0 "),
            return ge::GRAPH_FAILED);

        if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.cmpSparseIndices.desc, CMP_SPARSE_INDICES_NAME) ||
            ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.cmpSparseIndices.tensor->GetShape(), cmpSparseIndicesDimNumList, CMP_SPARSE_INDICES_NAME)) {
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckSingleParaBlockTable() const
{
    const std::vector<size_t> BlockTableDimNumList = {DIM_NUM_TWO};
    if (opParamInfo_.oriBlockTable.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.oriBlockTable.tensor->GetShapeSize() == 0,
            OP_LOGE(opName_, "Any dim of input oriBlockTable cannot be 0 "),
            return ge::GRAPH_FAILED);

        if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.oriBlockTable.desc, ORI_BLOCK_TABLE_NAME) ||
            ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.oriBlockTable.tensor->GetShape(), BlockTableDimNumList, ORI_BLOCK_TABLE_NAME)) {
            return ge::GRAPH_FAILED;
        }
    }

    if (opParamInfo_.cmpBlockTable.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.cmpBlockTable.tensor->GetShapeSize() == 0,
            OP_LOGE(opName_, "Any dim of input cmpBlockTable cannot be 0 "),
            return ge::GRAPH_FAILED);

        if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.cmpBlockTable.desc, CMP_BLOCK_TABLE_NAME) ||
            ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.cmpBlockTable.tensor->GetShape(), BlockTableDimNumList, CMP_BLOCK_TABLE_NAME)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckSingleParaCuSeqLensQ() const
{
    if (qLayout_ == SASLayout::BSND) {
        return ge::GRAPH_SUCCESS;
    }
    const std::vector<int64_t> cuSeqLensQDimNumList = {bSize_ + 1};
    OP_CHECK_IF((qLayout_ == SASLayout::TND && opParamInfo_.cuSeqLensQ.tensor == nullptr),
        OP_LOGE(opName_, "cuSeqLensQ can't be nullptr when layoutQ is TND"),
        return ge::GRAPH_FAILED);

    if (opParamInfo_.cuSeqLensQ.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.cuSeqLensQ.tensor->GetShapeSize() == 0,
            OP_LOGE(opName_, "Any dim of input cuSeqLensQ cannot be 0 "),
            return ge::GRAPH_FAILED);

        if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.cuSeqLensQ.desc, CU_SEQLENS_Q_NAME)) {
            return ge::GRAPH_FAILED;
        }

        OP_CHECK_IF((opParamInfo_.cuSeqLensQ.tensor->GetShapeSize() != bSize_ + 1),
            OP_LOGE(opName_, "cuSeqLensQ's shapeSize should be equal to bSize_+1:%u, but got %ld",
                bSize_ + 1, opParamInfo_.cuSeqLensQ.tensor->GetShapeSize()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckSingleParaSequsedKv() const
{
    OP_CHECK_IF(opParamInfo_.sequsedKv.tensor == nullptr,
        OP_LOGE(opName_, "input sequsedKv can not be nullptr, but it's empty"),
        return ge::GRAPH_FAILED);

    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.sequsedKv.desc, SEQUSED_KV_NAME)) {
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(opParamInfo_.sequsedKv.tensor->GetShapeSize() != bSize_,
        OP_LOGE(opName_, "input sequsedKv's shapeSize is not equal to B: %u, it is %ld", bSize_, opParamInfo_.sequsedKv.tensor->GetShapeSize()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckSingleParaSinks() const
{
    OP_CHECK_IF(opParamInfo_.sinks.tensor == nullptr,
        OP_LOGE(opName_, "Input sinks is nullptr, which is not supported"),
        return ge::GRAPH_FAILED);

    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.sinks.desc, SINKS_NAME)) {
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(opParamInfo_.sinks.tensor->GetShapeSize() != n1Size_,
        OP_LOGE(opName_, "Input sinks's shapeSize is not equal to n1: %u, it is %ld.", n1Size_, opParamInfo_.sinks.tensor->GetShapeSize()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckSingleParaMetadata() const
{
    OP_CHECK_IF(opParamInfo_.metadata.tensor == nullptr,
        OP_LOGE(opName_, "Input metadata is required, but got nullptr."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckSinglePara() const
{
    if (ge::GRAPH_SUCCESS != CheckSingleParaQuery() ||
        ge::GRAPH_SUCCESS != CheckSingleParaKey() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCmpSparseIndices() ||
        ge::GRAPH_SUCCESS != CheckSingleParaNumHeads() ||
        ge::GRAPH_SUCCESS != CheckSingleParaKvHeadNums() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSparseMode() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSparseBlockSize() ||
        ge::GRAPH_SUCCESS != CheckSingleParaBlockTable() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCuSeqLensQ() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSequsedKv() ||
        ge::GRAPH_SUCCESS != CheckSingleParaMetadata() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSinks() ) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

}