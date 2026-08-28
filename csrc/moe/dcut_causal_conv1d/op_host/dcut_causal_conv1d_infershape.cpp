#define CausalConv1d DcutCausalConv1d

// Load the CANN registration definitions before overriding the infer-shape
// macro. The original macro stringifies and token-pastes its argument, which
// prevents the CausalConv1d alias above from expanding to the D-Cut op name.
#include "register/op_impl_registry.h"

#undef IMPL_OP_INFERSHAPE
#define DCUT_STRINGIFY(x) #x
#define DCUT_STRINGIFY_EXPAND(x) DCUT_STRINGIFY(x)
#define DCUT_CAT(a, b) DCUT_CAT_INNER(a, b)
#define DCUT_CAT_INNER(a, b) a##b
#define IMPL_OP_INFERSHAPE(op_type) \
  gert::OpImplRegisterV2 VAR_UNUSED DCUT_CAT(op_impl_register_infershape_, op_type) = gert::OpImplRegisterV2(DCUT_STRINGIFY_EXPAND(op_type))

#include "../../../../csrc/moe/causal_conv1d/op_host/causal_conv1d_infershape.cpp"
