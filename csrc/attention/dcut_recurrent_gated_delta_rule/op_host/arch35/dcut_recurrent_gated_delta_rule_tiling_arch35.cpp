#define DCUT_RECURRENT_FIXED_STATE_ROWS 1
#define RecurrentGatedDeltaRule DcutRecurrentGatedDeltaRule
#define RecurrentGatedDeltaRuleCompileInfo DcutRecurrentGatedDeltaRuleCompileInfo
#define RecurrentGatedDeltaRuleInfo DcutRecurrentGatedDeltaRuleInfo
#define RecurrentGatedDeltaRuleTiling DcutRecurrentGatedDeltaRuleTiling
#define RecurrentGatedDeltaRuleTilingArch35 DcutRecurrentGatedDeltaRuleTilingArch35
#define RecurrentGatedDeltaRuleTilingData DcutRecurrentGatedDeltaRuleTilingData

// Include tiling_templates_registry.h first so REGISTER_OPS_TILING_TEMPLATE
// is defined before we override it.
#include "tiling_base/tiling_templates_registry.h"

// Override REGISTER_OPS_TILING_TEMPLATE: #op_type and ##op_type do NOT expand
// macros (C standard), so the tiling class would be registered under
// "RecurrentGatedDeltaRule" instead of "DcutRecurrentGatedDeltaRule".
// Variable names are fixed (only one call per TU).
#undef REGISTER_OPS_TILING_TEMPLATE
#define DCUT_STRINGIFY(x) #x
#define DCUT_STRINGIFY_EXPAND(x) DCUT_STRINGIFY(x)
#define REGISTER_OPS_TILING_TEMPLATE(op_type, class_name, priority)                       \
    [[maybe_unused]] uint32_t dcut_arch35_op_impl_register_template;                      \
    static Ops::Transformer::OpTiling::Register                                           \
        __attribute__((unused)) dcut_arch35_tiling_template_register =                     \
            Ops::Transformer::OpTiling::Register(DCUT_STRINGIFY_EXPAND(op_type)).tiling<class_name>(priority)

#include "../../vendor/op_host/arch35/recurrent_gated_delta_rule_tiling_arch35.cpp"
