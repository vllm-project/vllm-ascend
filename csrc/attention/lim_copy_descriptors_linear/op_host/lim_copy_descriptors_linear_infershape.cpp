/**
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file lim_copy_descriptors_linear_infershape.cpp
 * \brief InferShape implementation for LimCopyDescriptorsLinear (no outputs).
 */
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;
namespace ops {

static ge::graphStatus InferShape4LimCopyDescriptorsLinear(gert::InferShapeContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4LimCopyDescriptorsLinear(gert::InferDataTypeContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LimCopyDescriptorsLinear)
    .InferShape(InferShape4LimCopyDescriptorsLinear)
    .InferDataType(InferDtype4LimCopyDescriptorsLinear);
} // namespace ops
