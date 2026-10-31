/**
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file lim_copy_descriptors_infershape.cpp
 * \brief InferShape implementation for LimCopyDescriptors (no outputs).
 */
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;
namespace ops {

static ge::graphStatus InferShape4LimCopyDescriptors(gert::InferShapeContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4LimCopyDescriptors(gert::InferDataTypeContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LimCopyDescriptors)
    .InferShape(InferShape4LimCopyDescriptors)
    .InferDataType(InferDtype4LimCopyDescriptors);
} // namespace ops
