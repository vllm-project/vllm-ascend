/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recompute_wu_fwd_tiling_processor.h
 * \brief Tiling processor shared by aclnn tiling and fast kernel launch.
 */

#ifndef RECOMPUTE_WU_FWD_TILING_PROCESSOR_H
#define RECOMPUTE_WU_FWD_TILING_PROCESSOR_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include "exe_graph/runtime/storage_shape.h"
#include <register/op_impl_registry.h>
#include "tiling_base/data_copy_transpose_tiling.h"
#include "tiling_base/tiling_templates_registry.h"
#include "../../op_kernel/recompute_wu_fwd_struct.h"

#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

using GDN::RecomputeWUFwdTilingData;

namespace optiling {

static constexpr int64_t RECOMPUTE_WU_FWD_V_DIM_128 = 128;
static constexpr int64_t RECOMPUTE_WU_FWD_V_DIM_256 = 256;

static constexpr size_t RECOMPUTE_WU_FWD_INPUT_K_IDX = 0;
static constexpr size_t RECOMPUTE_WU_FWD_INPUT_V_IDX = 1;
static constexpr size_t RECOMPUTE_WU_FWD_INPUT_BETA_IDX = 2;
static constexpr size_t RECOMPUTE_WU_FWD_INPUT_A_IDX = 3;
static constexpr size_t RECOMPUTE_WU_FWD_INPUT_G_IDX = 4;
static constexpr size_t RECOMPUTE_WU_FWD_INPUT_SEQLENS_IDX = 5;
static constexpr size_t RECOMPUTE_WU_FWD_INPUT_CHUNK_INDICES_IDX = 6;

static constexpr size_t RECOMPUTE_WU_FWD_DIM_NUM_3 = 3;
static constexpr size_t RECOMPUTE_WU_FWD_DIM_NUM_4 = 4;

static constexpr size_t RECOMPUTE_WU_FWD_DIM_0 = 0;
static constexpr size_t RECOMPUTE_WU_FWD_DIM_1 = 1;
static constexpr size_t RECOMPUTE_WU_FWD_DIM_2 = 2;
static constexpr size_t RECOMPUTE_WU_FWD_DIM_3 = 3;

static constexpr int64_t RECOMPUTE_WU_FWD_CHUNK_SIZE_64 = 64;
static constexpr int64_t RECOMPUTE_WU_FWD_CHUNK_SIZE_128 = 128;
static constexpr int64_t RECOMPUTE_WU_FWD_VAR_LEN_B_DIM_1 = 1;

static constexpr const char *const RECOMPUTE_WU_FWD_INPUT_K_NAME = "k";
static constexpr const char *const RECOMPUTE_WU_FWD_INPUT_V_NAME = "v";
static constexpr const char *const RECOMPUTE_WU_FWD_INPUT_BETA_NAME = "beta";
static constexpr const char *const RECOMPUTE_WU_FWD_INPUT_A_NAME = "A";
static constexpr const char *const RECOMPUTE_WU_FWD_INPUT_G_NAME = "g";
static constexpr const char *const RECOMPUTE_WU_FWD_INPUT_CHUNK_INDICES_NAME = "chunk_indices";
static constexpr const char *const RECOMPUTE_WU_FWD_INPUT_SEQLENS_NAME = "cu_seqlens";

static constexpr uint64_t RECOMPUTE_WU_FWD_SIZE_HALF = 2;
static constexpr uint64_t RECOMPUTE_WU_FWD_SIZE_FP32 = 4;
static constexpr uint64_t RECOMPUTE_WU_FWD_ONE_BLOCK_32 = 32;

struct RecomputeWUFwdTilingContext {
    const char *nodeName;
    const gert::StorageShape *kShape;
    const gert::StorageShape *vShape;
    const gert::StorageShape *betaShape;
    const gert::StorageShape *aShape;
    const gert::StorageShape *gShape;
    const gert::StorageShape *cuSeqlensShape;
    const gert::StorageShape *chunkIndicesShape;
    const int64_t *cuSeqlensData;
    const int64_t *chunkIndicesData;
    int32_t chunkSize;
    ge::DataType kDtype;
    ge::DataType betaDtype;
    uint64_t ubSize;
    size_t sysWorkspaceSize;
};

class RecomputeWUFwdTilingProcessor {
    RecomputeWUFwdTilingContext &ctx_;
    RecomputeWUFwdTilingData &tiling_;
    size_t workspaceSize_ = 0;
    int64_t B = 0;
    int64_t Hk = 0;
    int64_t Hv = 0;
    int64_t hvPerHk = 1;
    int64_t K = 0;
    int64_t V = 0;
    int64_t T = 0;
    int64_t chunkSize = 0;

public:
    explicit RecomputeWUFwdTilingProcessor(RecomputeWUFwdTilingContext &ctx, RecomputeWUFwdTilingData &tiling)
        : ctx_(ctx), tiling_(tiling)
    {
    }

    size_t GetWorkspaceSize() const
    {
        return workspaceSize_;
    }

    bool IsVariableLength() const
    {
        return ctx_.cuSeqlensShape != nullptr;
    }

    ge::graphStatus RequiredInputDimNumCheck(const gert::StorageShape *curShape, size_t validDimNum,
                                             const char *inputName)
    {
        OP_CHECK_IF(curShape == nullptr,
                    OP_LOGE(ctx_.nodeName, "Input %s is required, but got nullptr.", inputName),
                    return ge::GRAPH_FAILED);
        const gert::Shape storageShape = curShape->GetStorageShape();
        size_t dimNum = storageShape.GetDimNum();
        OP_CHECK_IF(dimNum != validDimNum,
                    OP_LOGE(ctx_.nodeName,
                            "Check input %s shape failed, the dim num should be %zu, but get %zu.", inputName,
                            validDimNum, dimNum),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PreCheck()
    {
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.kShape, RECOMPUTE_WU_FWD_DIM_NUM_4, RECOMPUTE_WU_FWD_INPUT_K_NAME) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.vShape, RECOMPUTE_WU_FWD_DIM_NUM_4, RECOMPUTE_WU_FWD_INPUT_V_NAME) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.betaShape, RECOMPUTE_WU_FWD_DIM_NUM_3,
                                             RECOMPUTE_WU_FWD_INPUT_BETA_NAME) != ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.aShape, RECOMPUTE_WU_FWD_DIM_NUM_4, RECOMPUTE_WU_FWD_INPUT_A_NAME) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.gShape, RECOMPUTE_WU_FWD_DIM_NUM_3, RECOMPUTE_WU_FWD_INPUT_G_NAME) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CompareShape(const gert::Shape &shape1, const gert::Shape &shape2, const char *inputName1,
                                 const char *inputName2, size_t compareDimNum)
    {
        size_t shapeDim1 = 0;
        size_t shapeDim2 = 0;
        for (size_t dimIndex = 0; dimIndex < compareDimNum; dimIndex++) {
            shapeDim1 = shape1.GetDim(dimIndex);
            shapeDim2 = shape2.GetDim(dimIndex);
            OP_CHECK_IF(shapeDim1 != shapeDim2,
                        OP_LOGE(ctx_.nodeName,
                                "Compare input shape of %s and %s failed, the length of dim %zu should be same,but got "
                                "%zu and %zu.",
                                inputName1, inputName2, dimIndex, shapeDim1, shapeDim2),
                        return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus SetKbgExpVecRow(uint64_t ubSize, ge::DataType kType, ge::DataType betaType)
    {
        uint64_t rowNum = static_cast<uint64_t>(chunkSize);
        uint64_t sizeofKType = RECOMPUTE_WU_FWD_SIZE_FP32;
        uint64_t sizeofBetaType = RECOMPUTE_WU_FWD_SIZE_FP32;
        if (kType != ge::DataType::DT_FLOAT) {
            sizeofKType = RECOMPUTE_WU_FWD_SIZE_HALF;
        }
        if (betaType != ge::DataType::DT_FLOAT) {
            sizeofBetaType = RECOMPUTE_WU_FWD_SIZE_HALF;
        }
        while (rowNum >= 8) {
            uint64_t useUbSize = 0;
            useUbSize += 2 * rowNum * static_cast<uint64_t>(K) * sizeofKType;
            useUbSize += 2 * rowNum * sizeofBetaType;
            useUbSize += 2 * rowNum * sizeofBetaType;
            useUbSize += 2 * rowNum * static_cast<uint64_t>(K) * sizeofKType;
            useUbSize += rowNum * static_cast<uint64_t>(K) * sizeof(float);
            useUbSize += rowNum * sizeof(float);
            useUbSize += rowNum * sizeof(float);
            useUbSize += rowNum * RECOMPUTE_WU_FWD_ONE_BLOCK_32;

            if (useUbSize <= ubSize) {
                break;
            }
            rowNum = rowNum / 2;
        }
        tiling_.kbgExpVecRow = static_cast<int64_t>(rowNum);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus SetVbVecRow(uint64_t ubSize, ge::DataType kType, ge::DataType betaType)
    {
        uint64_t rowNum = static_cast<uint64_t>(chunkSize);
        uint64_t sizeofKType = RECOMPUTE_WU_FWD_SIZE_FP32;
        uint64_t sizeofBetaType = RECOMPUTE_WU_FWD_SIZE_FP32;
        if (kType != ge::DataType::DT_FLOAT) {
            sizeofKType = RECOMPUTE_WU_FWD_SIZE_HALF;
        }
        if (betaType != ge::DataType::DT_FLOAT) {
            sizeofBetaType = RECOMPUTE_WU_FWD_SIZE_HALF;
        }
        while (rowNum >= 8) {
            uint64_t useUbSize = 0;
            useUbSize += 2 * rowNum * static_cast<uint64_t>(V) * sizeofKType;
            useUbSize += 2 * rowNum * sizeofBetaType;
            useUbSize += 2 * rowNum * static_cast<uint64_t>(V) * sizeofKType;
            useUbSize += rowNum * static_cast<uint64_t>(V) * sizeof(float);
            useUbSize += rowNum * sizeof(float);
            useUbSize += rowNum * RECOMPUTE_WU_FWD_ONE_BLOCK_32;

            if (useUbSize <= ubSize) {
                break;
            }
            rowNum = rowNum / 2;
        }
        tiling_.vbVecRow = static_cast<int64_t>(rowNum);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CommonTiling()
    {
        const gert::Shape kStorageShape = ctx_.kShape->GetStorageShape();
        const gert::Shape vStorageShape = ctx_.vShape->GetStorageShape();
        const gert::Shape betaStorageShape = ctx_.betaShape->GetStorageShape();
        const gert::Shape AStorageShape = ctx_.aShape->GetStorageShape();
        const gert::Shape gStorageShape = ctx_.gShape->GetStorageShape();
        B = static_cast<int64_t>(vStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_0));
        Hk = static_cast<int64_t>(kStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_1));
        Hv = static_cast<int64_t>(vStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_1));
        OP_CHECK_IF(Hk <= 0 || Hv <= 0,
                    OP_LOGE(ctx_.nodeName,
                            "Invalid head dim: Hk and Hv must be positive, but got Hk=%ld, Hv=%ld.", Hk, Hv),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(Hv % Hk != 0,
                    OP_LOGE(ctx_.nodeName,
                            "GVA check: Hv must be divisible by Hk, but got Hk=%ld, Hv=%ld.", Hk, Hv),
                    return ge::GRAPH_FAILED);
        hvPerHk = Hv / Hk;
        OP_CHECK_IF(vStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_0) != kStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_0),
                    OP_LOGE(ctx_.nodeName, "Compare B: v B=%ld vs k B=%ld mismatch.",
                            vStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_0), kStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_0)),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(vStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_2) != kStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_2),
                    OP_LOGE(ctx_.nodeName, "Compare T: v T=%ld vs k T=%ld mismatch.",
                            vStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_2), kStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_2)),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(CompareShape(betaStorageShape, gStorageShape, RECOMPUTE_WU_FWD_INPUT_BETA_NAME,
                                 RECOMPUTE_WU_FWD_INPUT_G_NAME, RECOMPUTE_WU_FWD_DIM_NUM_3) != ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(kStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_0) != gStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_0),
                    OP_LOGE(ctx_.nodeName, "Compare B: k B=%ld vs g B=%ld mismatch.", kStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_0),
                            gStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_0)),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(kStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_2) != gStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_2),
                    OP_LOGE(ctx_.nodeName, "Compare T: k T=%ld vs g T=%ld mismatch.", kStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_2),
                            gStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_2)),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(kStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_0) != AStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_0),
                    OP_LOGE(ctx_.nodeName, "Compare B: k B=%ld vs A B=%ld mismatch.", kStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_0),
                            AStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_0)),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(kStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_2) != AStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_2),
                    OP_LOGE(ctx_.nodeName, "Compare T: k T=%ld vs A T=%ld mismatch.", kStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_2),
                            AStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_2)),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(AStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_1) != Hv,
                    OP_LOGE(ctx_.nodeName, "Compare head: A H=%ld must equal Hv=%ld.", AStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_1), Hv),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(betaStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_1) != Hv,
                    OP_LOGE(ctx_.nodeName, "Compare head: beta H=%ld must equal Hv=%ld.",
                            betaStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_1), Hv),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(gStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_1) != Hv,
                    OP_LOGE(ctx_.nodeName, "Compare head: g H=%ld must equal Hv=%ld.", gStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_1), Hv),
                    return ge::GRAPH_FAILED);
        T = static_cast<int64_t>(vStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_2));
        K = static_cast<int64_t>(kStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_3));
        V = static_cast<int64_t>(vStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_3));
        tiling_.B = B;
        tiling_.Hk = Hk;
        tiling_.Hv = Hv;
        tiling_.hvPerHk = hvPerHk;
        tiling_.T = T;
        tiling_.K = K;
        tiling_.V = V;
        OP_CHECK_IF(V != RECOMPUTE_WU_FWD_V_DIM_128 && V != RECOMPUTE_WU_FWD_V_DIM_256,
                    OP_LOGE(ctx_.nodeName,
                            "Check value dim V failed: only %ld or %ld is supported, but get %ld.",
                            RECOMPUTE_WU_FWD_V_DIM_128, RECOMPUTE_WU_FWD_V_DIM_256, V),
                    return ge::GRAPH_FAILED);
        chunkSize = static_cast<int64_t>(ctx_.chunkSize);
        OP_CHECK_IF(chunkSize != RECOMPUTE_WU_FWD_CHUNK_SIZE_64 && chunkSize != RECOMPUTE_WU_FWD_CHUNK_SIZE_128,
                    OP_LOGE(ctx_.nodeName,
                            "Check attr chunkSize failed, the chunkSize should be 64 or 128, but get %ld.", chunkSize),
                    return ge::GRAPH_FAILED);
        tiling_.chunkSize = chunkSize;

        OP_CHECK_IF(SetVbVecRow(ctx_.ubSize, ctx_.kDtype, ctx_.betaDtype) != ge::GRAPH_SUCCESS,
                    OP_LOGE(ctx_.nodeName, "SetVbVecRow Failed."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(SetKbgExpVecRow(ctx_.ubSize, ctx_.kDtype, ctx_.betaDtype) != ge::GRAPH_SUCCESS,
                    OP_LOGE(ctx_.nodeName, "SetKbgExpVecRow Failed."), return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    static int64_t CeilDiv(int64_t a, int64_t b)
    {
        if (unlikely(b == 0)) {
            return 0;
        }
        return (a + b - 1) / b;
    }

    ge::graphStatus FixLenTiling()
    {
        tiling_.chunkNum = tiling_.B * CeilDiv(tiling_.T, tiling_.chunkSize);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus VariableLenTiling()
    {
        OP_CHECK_IF(ctx_.cuSeqlensShape == nullptr,
                    OP_LOGE(ctx_.nodeName, "Input %s is required, but got nullptr.", RECOMPUTE_WU_FWD_INPUT_SEQLENS_NAME),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(ctx_.chunkIndicesShape == nullptr,
                    OP_LOGE(ctx_.nodeName, "Input %s is required, but got nullptr.",
                            RECOMPUTE_WU_FWD_INPUT_CHUNK_INDICES_NAME),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.chunkIndicesShape, 1, RECOMPUTE_WU_FWD_INPUT_CHUNK_INDICES_NAME) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.cuSeqlensShape, 1, RECOMPUTE_WU_FWD_INPUT_SEQLENS_NAME) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);

        const gert::Shape seqlensStorageShape = ctx_.cuSeqlensShape->GetStorageShape();
        int64_t seqlensDim0 = seqlensStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_0);
        OP_CHECK_IF(seqlensDim0 < 2,
                    OP_LOGE(ctx_.nodeName,
                            "Check seqlens shape failed, the dim 0 of seqlens should be larger than 1, but get %ld.",
                            seqlensDim0),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(ctx_.cuSeqlensData == nullptr,
                    OP_LOGE(ctx_.nodeName, "Input %s data is required, but got nullptr.", RECOMPUTE_WU_FWD_INPUT_SEQLENS_NAME),
                    return ge::GRAPH_FAILED);
        const int64_t *cuSeqlens = ctx_.cuSeqlensData;
        if (cuSeqlens[0] != 0) {
            OP_LOGE(ctx_.nodeName, "Check seqlens data failed, the seqlens[0] should be 0, but get %ld.", cuSeqlens[0]);
            return ge::GRAPH_FAILED;
        }
        std::vector<int64_t> expectChunkIndices;
        for (int64_t i = 1; i < seqlensDim0; i++) {
            int64_t curSeqLen = cuSeqlens[i] - cuSeqlens[i - 1];
            OP_CHECK_IF(curSeqLen <= 0,
                        OP_LOGE(ctx_.nodeName,
                                "Check seqlens data failed, the seqlens[%ld]:[%ld] should be larger than seqlens[%ld]:[%ld]",
                                i, cuSeqlens[i], i - 1, cuSeqlens[i - 1]),
                        return ge::GRAPH_FAILED);
            for (int64_t j = 0; j < curSeqLen; j += chunkSize) {
                expectChunkIndices.push_back(i - 1);
                expectChunkIndices.push_back(j / chunkSize);
            }
        }

        const gert::Shape chunkIndicesStorageShape = ctx_.chunkIndicesShape->GetStorageShape();
        int64_t chunkIndicesDim0 = chunkIndicesStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_0);
        OP_CHECK_IF(chunkIndicesDim0 != static_cast<int64_t>(expectChunkIndices.size()),
                    OP_LOGE(ctx_.nodeName,
                            "Check chunk_indices shape failed, the len of chunk_indices should be %zu, but get %ld.",
                            expectChunkIndices.size(), chunkIndicesDim0),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(ctx_.chunkIndicesData == nullptr,
                    OP_LOGE(ctx_.nodeName, "Input %s data is required, but got nullptr.",
                            RECOMPUTE_WU_FWD_INPUT_CHUNK_INDICES_NAME),
                    return ge::GRAPH_FAILED);
        const int64_t *chunkIndices = ctx_.chunkIndicesData;
        for (size_t i = 0; i < expectChunkIndices.size(); i++) {
            OP_CHECK_IF(expectChunkIndices[i] != chunkIndices[i],
                        OP_LOGE(ctx_.nodeName,
                                "Check chunk_indices data failed, the chunk_indices[%zu] should be %ld, but get %ld.",
                                i, expectChunkIndices[i], chunkIndices[i]),
                        return ge::GRAPH_FAILED);
        }
        tiling_.chunkNum = chunkIndicesStorageShape.GetDim(RECOMPUTE_WU_FWD_DIM_0) / 2;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus WorkspaceTiling()
    {
        uint64_t userWorkspaceSize =
            static_cast<uint64_t>(2) * static_cast<uint64_t>(tiling_.B) * static_cast<uint64_t>(tiling_.Hv) *
            static_cast<uint64_t>(tiling_.T) * static_cast<uint64_t>(tiling_.V);
        workspaceSize_ = ctx_.sysWorkspaceSize + static_cast<size_t>(userWorkspaceSize);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus Process()
    {
        OP_CHECK_IF(PreCheck() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        OP_CHECK_IF(CommonTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        if (IsVariableLength()) {
            OP_CHECK_IF(tiling_.B != RECOMPUTE_WU_FWD_VAR_LEN_B_DIM_1,
                        OP_LOGE(ctx_.nodeName,
                                "If cu_seqlens is not nullptr, the dim 0 of q needs to be 1, but now is %ld.",
                                tiling_.B),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(VariableLenTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
            tiling_.isVariable = 1;
        } else {
            OP_CHECK_IF(FixLenTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
            tiling_.isVariable = 0;
        }
        OP_CHECK_IF(WorkspaceTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
};

} // namespace optiling

#endif // RECOMPUTE_WU_FWD_TILING_PROCESSOR_H
