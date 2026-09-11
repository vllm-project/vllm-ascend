#ifndef CUSTOM_MULS_PROTO_H
#define CUSTOM_MULS_PROTO_H

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
REG_OP(customMuls)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(scalar, Float)
    .OP_END_FACTORY_REG(customMuls)
} // namespace ge

#endif // CUSTOM_MULS_PROTO_H
