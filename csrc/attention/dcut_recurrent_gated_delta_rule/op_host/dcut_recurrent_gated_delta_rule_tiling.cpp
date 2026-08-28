#define DCUT_RECURRENT_FIXED_STATE_ROWS 1
#define RecurrentGatedDeltaRule DcutRecurrentGatedDeltaRule
#define RecurrentGatedDeltaRuleCompileInfo DcutRecurrentGatedDeltaRuleCompileInfo
#define RecurrentGatedDeltaRuleInfo DcutRecurrentGatedDeltaRuleInfo
#define RecurrentGatedDeltaRuleTiling DcutRecurrentGatedDeltaRuleTiling
#define RecurrentGatedDeltaRuleTilingData DcutRecurrentGatedDeltaRuleTilingData

// Include the CANN header FIRST so its IMPL_OP_OPTILING definition is processed
// and the include guard INC_EXTERNAL_REGISTER_OP_IMPL_REGISTRY_H_ is set.
// Otherwise the header (pulled in transitively by the tiling file) would
// re-define IMPL_OP_OPTILING and clobber our override below.
#include "register/op_impl_registry.h"

// Also include tiling_templates_registry.h first so REGISTER_OPS_TILING_TEMPLATE
// is defined before we override it.
#include "tiling_base/tiling_templates_registry.h"

// Override IMPL_OP_OPTILING to force macro expansion in #op_type (stringification)
// and ##op_type (token pasting). Per C standard, # and ## do NOT expand macro
// arguments, so without this override IMPL_OP_OPTILING(RecurrentGatedDeltaRule)
// would register the op type as "RecurrentGatedDeltaRule" instead of
// "DcutRecurrentGatedDeltaRule", and the runtime would fail with
// "Do not find tiling func of DcutRecurrentGatedDeltaRule!".
#undef IMPL_OP_OPTILING
#define DCUT_STRINGIFY(x) #x
#define DCUT_STRINGIFY_EXPAND(x) DCUT_STRINGIFY(x)
#define DCUT_CAT(a, b) DCUT_CAT_INNER(a, b)
#define DCUT_CAT_INNER(a, b) a##b
#define IMPL_OP_OPTILING(op_type) \
  gert::OpImplRegisterV2 VAR_UNUSED DCUT_CAT(op_impl_register_optiling_, op_type) = gert::OpImplRegisterV2(DCUT_STRINGIFY_EXPAND(op_type))

// Override REGISTER_OPS_TILING_TEMPLATE for the same reason: #op_type and
// ##op_type do NOT expand macros, so the tiling class would be registered
// under "RecurrentGatedDeltaRule" instead of "DcutRecurrentGatedDeltaRule",
// and DoTilingImpl would fail to find the class.
// Variable names are fixed (only one call per TU) to avoid complex token pasting.
#undef REGISTER_OPS_TILING_TEMPLATE
#define REGISTER_OPS_TILING_TEMPLATE(op_type, class_name, priority)                       \
    [[maybe_unused]] uint32_t dcut_op_impl_register_template;                             \
    static Ops::Transformer::OpTiling::Register                                           \
        __attribute__((unused)) dcut_tiling_template_register =                            \
            Ops::Transformer::OpTiling::Register(DCUT_STRINGIFY_EXPAND(op_type)).tiling<class_name>(priority)

#include "../vendor/op_host/recurrent_gated_delta_rule_tiling.cpp"
