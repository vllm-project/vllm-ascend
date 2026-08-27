#define CFG_BUILD_DEBUG
#define RecurrentGatedDeltaRule DcutRecurrentGatedDeltaRule
#define aclnnRecurrentGatedDeltaRule aclnnDcutRecurrentGatedDeltaRule
#define aclnnRecurrentGatedDeltaRuleGetWorkspaceSize aclnnDcutRecurrentGatedDeltaRuleGetWorkspaceSize
#include "../../../../../csrc/attention/recurrent_gated_delta_rule/op_host/op_api/aclnn_recurrent_gated_delta_rule.cpp"
