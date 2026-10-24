// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <limits>
#include "register/op_impl_registry.h"
#include "../op_kernel/engram_hash_ids_constants.h"

namespace ops {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    using namespace EngramHashIds;
    const auto *attrs = context->GetAttrs();
    const auto *parameters = attrs == nullptr ? nullptr : attrs->GetListInt(PARAMETERS);
    auto *output = context->GetOutputShape(0);
    if (parameters == nullptr || parameters->GetSize() != PARAMETER_COUNT || output == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto *p = static_cast<const int64_t *>(parameters->GetData());
    if (p[NUM_TOKENS] < 0 || p[NUM_LAYERS] <= 0 || p[MAX_NGRAM] < 1 || p[NUM_HEADS] <= 0 ||
        p[MAX_NGRAM] - 1 > std::numeric_limits<int64_t>::max() / p[NUM_HEADS]) {
        return ge::GRAPH_FAILED;
    }
    output->SetDimNum(PARAM_RANK);
    output->SetDim(TOKEN_DIM, p[NUM_TOKENS]);
    output->SetDim(NGRAM_DIM, p[NUM_LAYERS]);
    output->SetDim(HEAD_DIM, (p[MAX_NGRAM] - 1) * p[NUM_HEADS]);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(EngramHashIds).InferShape(InferShape).InferDataType(InferDataType);
} // namespace ops
