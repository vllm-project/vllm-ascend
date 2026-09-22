/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "chunk_kda_fwd_prepare_tiling.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {

enum class PrepareLayout {
    BNSD,
    BSND,
    NTD,
    TND,
};

bool ParseLayout(const char *layout, PrepareLayout &parsed)
{
    if (layout == nullptr || std::strcmp(layout, "BNSD") == 0) {
        parsed = PrepareLayout::BNSD;
        return true;
    }
    if (std::strcmp(layout, "BSND") == 0) {
        parsed = PrepareLayout::BSND;
        return true;
    }
    if (std::strcmp(layout, "NTD") == 0) {
        parsed = PrepareLayout::NTD;
        return true;
    }
    if (std::strcmp(layout, "TND") == 0) {
        parsed = PrepareLayout::TND;
        return true;
    }
    return false;
}

void SetShape3(gert::Shape *shape, int64_t d0, int64_t d1, int64_t d2)
{
    if (shape == nullptr) {
        return;
    }
    shape->SetDimNum(3);
    shape->SetDim(0, d0);
    shape->SetDim(1, d1);
    shape->SetDim(2, d2);
}

void SetShape4(gert::Shape *shape, int64_t d0, int64_t d1, int64_t d2,
               int64_t d3)
{
    if (shape == nullptr) {
        return;
    }
    shape->SetDimNum(4);
    shape->SetDim(0, d0);
    shape->SetDim(1, d1);
    shape->SetDim(2, d2);
    shape->SetDim(3, d3);
}

struct LogicalShape {
    bool packed = false;
    int64_t batch = 0;
    int64_t seqLen = 0;
    int64_t qkHeadNum = 0;
    int64_t valueHeadNum = 0;
    int64_t kDim = 0;
    int64_t vDim = 0;
};

bool ResolveLogicalShape(const gert::Shape &q, const gert::Shape &v,
                         PrepareLayout layout, LogicalShape &shape)
{
    shape.packed = layout == PrepareLayout::NTD || layout == PrepareLayout::TND;
    if (shape.packed) {
        if (q.GetDimNum() != 3 || v.GetDimNum() != 3) {
            return false;
        }
        shape.batch = 1;
        if (layout == PrepareLayout::TND) {
            shape.seqLen = q.GetDim(0);
            shape.qkHeadNum = q.GetDim(1);
            shape.valueHeadNum = v.GetDim(1);
        } else {
            shape.qkHeadNum = q.GetDim(0);
            shape.valueHeadNum = v.GetDim(0);
            shape.seqLen = q.GetDim(1);
        }
        shape.kDim = q.GetDim(2);
        shape.vDim = v.GetDim(2);
        return true;
    }

    if (q.GetDimNum() != 4 || v.GetDimNum() != 4) {
        return false;
    }
    shape.batch = q.GetDim(0);
    if (layout == PrepareLayout::BSND) {
        shape.seqLen = q.GetDim(1);
        shape.qkHeadNum = q.GetDim(2);
        shape.valueHeadNum = v.GetDim(2);
    } else {
        shape.qkHeadNum = q.GetDim(1);
        shape.valueHeadNum = v.GetDim(1);
        shape.seqLen = q.GetDim(2);
    }
    shape.kDim = q.GetDim(3);
    shape.vDim = v.GetDim(3);
    return true;
}

void SetValueMatrixShape(gert::Shape *output, const LogicalShape &shape,
                         int64_t dimension)
{
    if (shape.packed) {
        SetShape3(output, shape.valueHeadNum, shape.seqLen, dimension);
    } else {
        SetShape4(output, shape.batch, shape.valueHeadNum, shape.seqLen,
                  dimension);
    }
}

void SetValueScalarShape(gert::Shape *output, const LogicalShape &shape)
{
    if (shape.packed) {
        if (output != nullptr) {
            output->SetDimNum(2);
            output->SetDim(0, shape.valueHeadNum);
            output->SetDim(1, shape.seqLen);
        }
    } else {
        SetShape3(output, shape.batch, shape.valueHeadNum, shape.seqLen);
    }
}

void SetQkMatrixShape(gert::Shape *output, const LogicalShape &shape)
{
    if (shape.packed) {
        SetShape3(output, shape.qkHeadNum, shape.seqLen, shape.kDim);
    } else {
        SetShape4(output, shape.batch, shape.qkHeadNum, shape.seqLen,
                  shape.kDim);
    }
}

void SetQkScalarShape(gert::Shape *output, const LogicalShape &shape)
{
    if (shape.packed) {
        if (output != nullptr) {
            output->SetDimNum(2);
            output->SetDim(0, shape.qkHeadNum);
            output->SetDim(1, shape.seqLen);
        }
    } else {
        SetShape3(output, shape.batch, shape.qkHeadNum, shape.seqLen);
    }
}

} // namespace

static ge::graphStatus InferShapeChunkKdaFwdPrepare(
    gert::InferShapeContext *context)
{
    const auto *qShape = context->GetInputShape(optiling::PREPARE_INPUT_Q);
    const auto *vShape = context->GetInputShape(optiling::PREPARE_INPUT_V);
    if (qShape == nullptr || vShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const char *layout = "BNSD";
    int64_t chunkSize = 64;
    const auto *attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const char *attrLayout = attrs->GetStr(optiling::PREPARE_ATTR_LAYOUT);
        if (attrLayout != nullptr) {
            layout = attrLayout;
        }
        const auto *chunkSizePtr =
            attrs->GetAttrPointer<int64_t>(optiling::PREPARE_ATTR_CHUNK_SIZE);
        if (chunkSizePtr != nullptr) {
            chunkSize = *chunkSizePtr;
        }
    }
    if (chunkSize != 64) {
        return ge::GRAPH_FAILED;
    }

    PrepareLayout parsedLayout;
    LogicalShape shape;
    if (!ParseLayout(layout, parsedLayout) ||
        !ResolveLogicalShape(*qShape, *vShape, parsedLayout, shape)) {
        return ge::GRAPH_FAILED;
    }

    SetValueMatrixShape(context->GetOutputShape(optiling::PREPARE_OUTPUT_GK),
                        shape, shape.kDim);
    SetValueMatrixShape(context->GetOutputShape(optiling::PREPARE_OUTPUT_AQK),
                        shape, chunkSize);
    SetValueMatrixShape(context->GetOutputShape(optiling::PREPARE_OUTPUT_AKK),
                        shape, chunkSize);
    SetValueMatrixShape(context->GetOutputShape(optiling::PREPARE_OUTPUT_W),
                        shape, shape.kDim);
    SetValueMatrixShape(context->GetOutputShape(optiling::PREPARE_OUTPUT_U),
                        shape, shape.vDim);
    SetValueMatrixShape(context->GetOutputShape(optiling::PREPARE_OUTPUT_QG),
                        shape, shape.kDim);
    SetValueMatrixShape(context->GetOutputShape(optiling::PREPARE_OUTPUT_KG),
                        shape, shape.kDim);
    SetValueMatrixShape(
        context->GetOutputShape(optiling::PREPARE_OUTPUT_QG_SCALED), shape,
        shape.kDim);
    SetQkMatrixShape(context->GetOutputShape(optiling::PREPARE_OUTPUT_Q_HAT),
                     shape);
    SetQkMatrixShape(context->GetOutputShape(optiling::PREPARE_OUTPUT_K_HAT),
                     shape);
    SetQkScalarShape(context->GetOutputShape(optiling::PREPARE_OUTPUT_Q_RSTD),
                     shape);
    SetQkScalarShape(context->GetOutputShape(optiling::PREPARE_OUTPUT_K_RSTD),
                     shape);
    SetValueScalarShape(
        context->GetOutputShape(optiling::PREPARE_OUTPUT_BETA_EFF), shape);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeChunkKdaFwdPrepare(
    gert::InferDataTypeContext *context)
{
    constexpr size_t bf16Outputs[] = {
        optiling::PREPARE_OUTPUT_AQK,
        optiling::PREPARE_OUTPUT_AKK,
        optiling::PREPARE_OUTPUT_W,
        optiling::PREPARE_OUTPUT_U,
        optiling::PREPARE_OUTPUT_QG,
        optiling::PREPARE_OUTPUT_KG,
        optiling::PREPARE_OUTPUT_QG_SCALED,
        optiling::PREPARE_OUTPUT_Q_HAT,
        optiling::PREPARE_OUTPUT_K_HAT,
    };
    constexpr size_t fp32Outputs[] = {
        optiling::PREPARE_OUTPUT_GK,
        optiling::PREPARE_OUTPUT_Q_RSTD,
        optiling::PREPARE_OUTPUT_K_RSTD,
        optiling::PREPARE_OUTPUT_BETA_EFF,
    };
    for (size_t output : bf16Outputs) {
        context->SetOutputDataType(output, ge::DT_BF16);
    }
    for (size_t output : fp32Outputs) {
        context->SetOutputDataType(output, ge::DT_FLOAT);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ChunkKdaFwdPrepare)
    .InferShape(InferShapeChunkKdaFwdPrepare)
    .InferDataType(InferDataTypeChunkKdaFwdPrepare);

} // namespace ops
