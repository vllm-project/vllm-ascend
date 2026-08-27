#define RecurrentGatedDeltaRule DcutRecurrentGatedDeltaRule

// Include the CANN header FIRST so its IMPL_OP_INFERSHAPE definition is processed
// and the include guard INC_EXTERNAL_REGISTER_OP_IMPL_REGISTRY_H_ is set.
// Otherwise the header (pulled in transitively by the infershape file) would
// re-define IMPL_OP_INFERSHAPE and clobber our override below.
#include "register/op_impl_registry.h"

// Override IMPL_OP_INFERSHAPE to force macro expansion in #op_type (stringification)
// and ##op_type (token pasting). Per C standard, # and ## do NOT expand macro
// arguments, so without this override IMPL_OP_INFERSHAPE(RecurrentGatedDeltaRule)
// would register the infershape for "RecurrentGatedDeltaRule" instead of
// "DcutRecurrentGatedDeltaRule", and the runtime would fail with
// "Op has no infershape func, opType: DcutRecurrentGatedDeltaRule".
#undef IMPL_OP_INFERSHAPE
#define DCUT_STRINGIFY(x) #x
#define DCUT_STRINGIFY_EXPAND(x) DCUT_STRINGIFY(x)
#define DCUT_CAT(a, b) DCUT_CAT_INNER(a, b)
#define DCUT_CAT_INNER(a, b) a##b
#define IMPL_OP_INFERSHAPE(op_type) \
  gert::OpImplRegisterV2 VAR_UNUSED DCUT_CAT(op_impl_register_infershape_, op_type) = gert::OpImplRegisterV2(DCUT_STRINGIFY_EXPAND(op_type))

#include "../../../../csrc/attention/recurrent_gated_delta_rule/op_host/recurrent_gated_delta_rule_infershape.cpp"
