/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_fwd_o_tiling_processor.h
 * \brief Tiling processor shared by aclnn tiling and fast kernel launch.
 */

#ifndef CHUNK_FWD_O_TILING_PROCESSOR_H
#define CHUNK_FWD_O_TILING_PROCESSOR_H

#include <cstddef>
#include <cstdint>
#include "exe_graph/runtime/storage_shape.h"
#include <register/op_impl_registry.h>
#include "tiling_base/data_copy_transpose_tiling.h"
#include "tiling_base/tiling_templates_registry.h"
#include "../op_kernel/chunk_fwd_o_struct.h"

using GDN::ChunkFwdOTilingData;

namespace optiling {

static constexpr size_t CHUNK_FWD_O_INPUT_Q_IDX = 0;
static constexpr size_t CHUNK_FWD_O_INPUT_K_IDX = 1;
static constexpr size_t CHUNK_FWD_O_INPUT_V_IDX = 2;
static constexpr size_t CHUNK_FWD_O_INPUT_H_IDX = 3;
static constexpr size_t CHUNK_FWD_O_INPUT_G_IDX = 4;
static constexpr size_t CHUNK_FWD_O_INPUT_SEQLENS_IDX = 5;
static constexpr size_t CHUNK_FWD_O_INPUT_CHUNK_OFFSETS_IDX = 6;

static constexpr size_t CHUNK_FWD_O_ATTR_SCALE_IDX = 0;
static constexpr size_t CHUNK_FWD_O_ATTR_CHUNK_SIZE_IDX = 1;

static constexpr size_t CHUNK_FWD_O_QKV_DIM_NUM = 4;
static constexpr size_t CHUNK_FWD_O_H_DIM_NUM = 5;
static constexpr size_t CHUNK_FWD_O_G_DIM_NUM = 3;
static constexpr size_t CHUNK_FWD_O_SEQLENS_DIM_NUM = 1;
static constexpr size_t CHUNK_FWD_O_CHUNK_OFFSETS_DIM_NUM = 1;

static constexpr size_t CHUNK_FWD_O_DIM_BATCH = 0;
static constexpr size_t CHUNK_FWD_O_DIM_HEAD_NUM = 1;
static constexpr size_t CHUNK_FWD_O_DIM_SEQLEN = 2;
static constexpr size_t CHUNK_FWD_O_DIM_HEAD_DIM = 3;
static constexpr size_t CHUNK_FWD_O_H_DIM_CHUNKS = 2;
static constexpr size_t CHUNK_FWD_O_H_DIM_K = 3;
static constexpr size_t CHUNK_FWD_O_H_DIM_V = 4;

static constexpr int64_t CHUNK_FWD_O_DTYPE_FP16 = 0;
static constexpr int64_t CHUNK_FWD_O_DTYPE_BF16 = 1;
static constexpr int64_t CHUNK_FWD_O_DTYPE_FP32 = 2;

static constexpr size_t CHUNK_FWD_O_WORKSPACE_RSV_BYTE = 16 * 1024 * 1024;
static constexpr size_t CHUNK_FWD_O_GM_ALIGN = 512;
static constexpr int64_t CHUNK_FWD_O_PINGPONG_STAGES = 2;
static constexpr int64_t CHUNK_FWD_O_CHUNK_OFFSETS_PAIR_SIZE = 2;

static constexpr const char *const CHUNK_FWD_O_INPUT_Q_NAME = "q";
static constexpr const char *const CHUNK_FWD_O_INPUT_K_NAME = "k";
static constexpr const char *const CHUNK_FWD_O_INPUT_V_NAME = "v";
static constexpr const char *const CHUNK_FWD_O_INPUT_H_NAME = "h";
static constexpr const char *const CHUNK_FWD_O_INPUT_G_NAME = "g";
static constexpr const char *const CHUNK_FWD_O_INPUT_SEQLENS_NAME = "cu_seqlens";
static constexpr const char *const CHUNK_FWD_O_INPUT_CHUNK_OFFSETS_NAME = "chunk_offsets";

struct ChunkFwdOTilingContext {
    const char *nodeName;
    const gert::StorageShape *qShape;
    const gert::StorageShape *kShape;
    const gert::StorageShape *vShape;
    const gert::StorageShape *hShape;
    const gert::StorageShape *gShape;
    const gert::StorageShape *cuSeqlensShape;
    const gert::StorageShape *chunkOffsetsShape;
    double scale;
    int64_t chunkSize;
    int64_t dataType;
    int64_t gDataType;
    uint32_t aicCoreNum;
    size_t sysWorkspaceSize;
};

class ChunkFwdOTilingProcessor {
    ChunkFwdOTilingContext &ctx_;
    ChunkFwdOTilingData &tiling_;
    size_t workspaceSize_ = 0;

public:
    explicit ChunkFwdOTilingProcessor(ChunkFwdOTilingContext &ctx, ChunkFwdOTilingData &tiling)
        : ctx_(ctx), tiling_(tiling)
    {
    }

    size_t GetWorkspaceSize() const
    {
        return workspaceSize_;
    }

    bool IsVariableLength() const
    {
        return ctx_.cuSeqlensShape != nullptr || ctx_.chunkOffsetsShape != nullptr;
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
                            "Check input %s shape failed, the dim num should be %zu, but get %zu.",
                            inputName, validDimNum, dimNum),
                    return ge::GRAPH_FAILED);
        for (size_t dimIndex = 0; dimIndex < dimNum; ++dimIndex) {
            OP_CHECK_IF(storageShape.GetDim(dimIndex) == 0,
                        OP_LOGE(ctx_.nodeName,
                                "Check input %s shape failed, the dim %zu should be non-zero, but get 0.",
                                inputName, dimIndex),
                        return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PreCheck()
    {
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.qShape, CHUNK_FWD_O_QKV_DIM_NUM,
                                             CHUNK_FWD_O_INPUT_Q_NAME) != ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.kShape, CHUNK_FWD_O_QKV_DIM_NUM,
                                             CHUNK_FWD_O_INPUT_K_NAME) != ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.vShape, CHUNK_FWD_O_QKV_DIM_NUM,
                                             CHUNK_FWD_O_INPUT_V_NAME) != ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.hShape, CHUNK_FWD_O_H_DIM_NUM,
                                             CHUNK_FWD_O_INPUT_H_NAME) != ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.gShape, CHUNK_FWD_O_G_DIM_NUM,
                                             CHUNK_FWD_O_INPUT_G_NAME) != ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus ShapeCheck()
    {
        const gert::Shape qShape = ctx_.qShape->GetStorageShape();
        const gert::Shape kShape = ctx_.kShape->GetStorageShape();
        const gert::Shape vShape = ctx_.vShape->GetStorageShape();
        const gert::Shape hShape = ctx_.hShape->GetStorageShape();
        const gert::Shape gShape = ctx_.gShape->GetStorageShape();

        OP_CHECK_IF(qShape.GetDim(CHUNK_FWD_O_DIM_BATCH) != kShape.GetDim(CHUNK_FWD_O_DIM_BATCH) ||
                        qShape.GetDim(CHUNK_FWD_O_DIM_HEAD_NUM) != kShape.GetDim(CHUNK_FWD_O_DIM_HEAD_NUM) ||
                        qShape.GetDim(CHUNK_FWD_O_DIM_SEQLEN) != kShape.GetDim(CHUNK_FWD_O_DIM_SEQLEN) ||
                        qShape.GetDim(CHUNK_FWD_O_DIM_HEAD_DIM) != kShape.GetDim(CHUNK_FWD_O_DIM_HEAD_DIM),
                    OP_LOGE(ctx_.nodeName, "Check q/k shape failed, q and k should have the same shape."),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(qShape.GetDim(CHUNK_FWD_O_DIM_BATCH) != vShape.GetDim(CHUNK_FWD_O_DIM_BATCH) ||
                        qShape.GetDim(CHUNK_FWD_O_DIM_SEQLEN) != vShape.GetDim(CHUNK_FWD_O_DIM_SEQLEN),
                    OP_LOGE(ctx_.nodeName, "Check q/v shape failed, batch and seqlen should be the same."),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(vShape.GetDim(CHUNK_FWD_O_DIM_BATCH) != gShape.GetDim(CHUNK_FWD_O_DIM_BATCH) ||
                        vShape.GetDim(CHUNK_FWD_O_DIM_HEAD_NUM) != gShape.GetDim(CHUNK_FWD_O_DIM_HEAD_NUM) ||
                        vShape.GetDim(CHUNK_FWD_O_DIM_SEQLEN) != gShape.GetDim(CHUNK_FWD_O_DIM_SEQLEN),
                    OP_LOGE(ctx_.nodeName, "Check v/g shape failed, g should be [B, HV, T]."),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(hShape.GetDim(CHUNK_FWD_O_DIM_BATCH) != vShape.GetDim(CHUNK_FWD_O_DIM_BATCH) ||
                        hShape.GetDim(CHUNK_FWD_O_DIM_HEAD_NUM) != vShape.GetDim(CHUNK_FWD_O_DIM_HEAD_NUM) ||
                        hShape.GetDim(CHUNK_FWD_O_H_DIM_K) != qShape.GetDim(CHUNK_FWD_O_DIM_HEAD_DIM) ||
                        hShape.GetDim(CHUNK_FWD_O_H_DIM_V) != vShape.GetDim(CHUNK_FWD_O_DIM_HEAD_DIM),
                    OP_LOGE(ctx_.nodeName, "Check h shape failed, h should be [B, HV, chunks, K, V]."),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(vShape.GetDim(CHUNK_FWD_O_DIM_HEAD_NUM) % qShape.GetDim(CHUNK_FWD_O_DIM_HEAD_NUM) != 0,
                    OP_LOGE(ctx_.nodeName, "Check head num failed, vNumHead should be divisible by kNumHead."),
                    return ge::GRAPH_FAILED);

        return ge::GRAPH_SUCCESS;
    }

    size_t AlignWorkspaceSize(size_t value)
    {
        return (value + CHUNK_FWD_O_GM_ALIGN) / CHUNK_FWD_O_GM_ALIGN * CHUNK_FWD_O_GM_ALIGN;
    }

    ge::graphStatus CommonTiling()
    {
        const gert::Shape qShape = ctx_.qShape->GetStorageShape();
        const gert::Shape vShape = ctx_.vShape->GetStorageShape();

        tiling_.seqlen = static_cast<int64_t>(qShape.GetDim(CHUNK_FWD_O_DIM_SEQLEN));
        tiling_.kNumHead = static_cast<int64_t>(qShape.GetDim(CHUNK_FWD_O_DIM_HEAD_NUM));
        tiling_.vNumHead = static_cast<int64_t>(vShape.GetDim(CHUNK_FWD_O_DIM_HEAD_NUM));
        tiling_.kHeadDim = static_cast<int64_t>(qShape.GetDim(CHUNK_FWD_O_DIM_HEAD_DIM));
        tiling_.vHeadDim = static_cast<int64_t>(vShape.GetDim(CHUNK_FWD_O_DIM_HEAD_DIM));
        tiling_.scale = static_cast<float>(ctx_.scale);
        tiling_.chunkSize = ctx_.chunkSize;
        tiling_.dataType = ctx_.dataType;
        tiling_.gDataType = ctx_.gDataType;

        OP_CHECK_IF(tiling_.chunkSize <= 0,
                    OP_LOGE(ctx_.nodeName, "Check attr chunk_size failed, chunk_size should be positive."),
                    return ge::GRAPH_FAILED);

        if (IsVariableLength()) {
            OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.cuSeqlensShape, CHUNK_FWD_O_SEQLENS_DIM_NUM,
                                                 CHUNK_FWD_O_INPUT_SEQLENS_NAME) != ge::GRAPH_SUCCESS,
                        , return ge::GRAPH_FAILED);
            OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.chunkOffsetsShape, CHUNK_FWD_O_CHUNK_OFFSETS_DIM_NUM,
                                                 CHUNK_FWD_O_INPUT_CHUNK_OFFSETS_NAME) != ge::GRAPH_SUCCESS,
                        , return ge::GRAPH_FAILED);
            const gert::Shape cuSeqlensShape = ctx_.cuSeqlensShape->GetStorageShape();
            const gert::Shape chunkOffsetsShape = ctx_.chunkOffsetsShape->GetStorageShape();
            OP_CHECK_IF(chunkOffsetsShape.GetDim(CHUNK_FWD_O_DIM_BATCH) % CHUNK_FWD_O_CHUNK_OFFSETS_PAIR_SIZE != 0,
                        OP_LOGE(ctx_.nodeName,
                                "Check chunk_offsets shape failed, the dim 0 of chunk_offsets needs to be divisible by 2, but get %ld.",
                                chunkOffsetsShape.GetDim(CHUNK_FWD_O_DIM_BATCH)),
                        return ge::GRAPH_FAILED);
            tiling_.isVariedLen = 1;
            tiling_.shapeBatch = 1;
            tiling_.tokenBatch = static_cast<int64_t>(cuSeqlensShape.GetDim(CHUNK_FWD_O_DIM_BATCH)) - 1;
            OP_CHECK_IF(tiling_.tokenBatch <= 0,
                        OP_LOGE(ctx_.nodeName, "Check cu_seqlens shape failed, tokenBatch should be positive."),
                        return ge::GRAPH_FAILED);
        } else {
            tiling_.isVariedLen = 0;
            tiling_.shapeBatch = static_cast<int64_t>(qShape.GetDim(CHUNK_FWD_O_DIM_BATCH));
            tiling_.tokenBatch = 1;
        }

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus WorkspaceTiling()
    {
        size_t workspaceOffset = ctx_.sysWorkspaceSize;
        workspaceOffset += CHUNK_FWD_O_WORKSPACE_RSV_BYTE;

        tiling_.vWorkspaceOffset = static_cast<int64_t>(workspaceOffset);
        workspaceOffset += AlignWorkspaceSize(static_cast<size_t>(ctx_.aicCoreNum) * tiling_.chunkSize *
                                              tiling_.vHeadDim * sizeof(float) * CHUNK_FWD_O_PINGPONG_STAGES);

        tiling_.hWorkspaceOffset = static_cast<int64_t>(workspaceOffset);
        workspaceOffset += AlignWorkspaceSize(static_cast<size_t>(ctx_.aicCoreNum) * tiling_.chunkSize *
                                              tiling_.vHeadDim * sizeof(float) * CHUNK_FWD_O_PINGPONG_STAGES);

        tiling_.attnWorkspaceOffset = static_cast<int64_t>(workspaceOffset);
        workspaceOffset += AlignWorkspaceSize(static_cast<size_t>(ctx_.aicCoreNum) * tiling_.chunkSize *
                                              tiling_.chunkSize * sizeof(float) * CHUNK_FWD_O_PINGPONG_STAGES);

        tiling_.aftermaskWorkspaceOffset = static_cast<int64_t>(workspaceOffset);
        workspaceOffset += AlignWorkspaceSize(static_cast<size_t>(ctx_.aicCoreNum) * tiling_.chunkSize *
                                              tiling_.chunkSize * sizeof(float) * CHUNK_FWD_O_PINGPONG_STAGES);

        tiling_.maskWorkspaceOffset = static_cast<int64_t>(workspaceOffset);
        workspaceOffset += AlignWorkspaceSize(static_cast<size_t>(tiling_.chunkSize) * tiling_.chunkSize);

        workspaceOffset += CHUNK_FWD_O_WORKSPACE_RSV_BYTE;
        workspaceSize_ = workspaceOffset;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus Process()
    {
        OP_CHECK_IF(PreCheck() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        OP_CHECK_IF(ShapeCheck() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        OP_CHECK_IF(CommonTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        OP_CHECK_IF(WorkspaceTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
};

} // namespace optiling

#endif // CHUNK_FWD_O_TILING_PROCESSOR_H
