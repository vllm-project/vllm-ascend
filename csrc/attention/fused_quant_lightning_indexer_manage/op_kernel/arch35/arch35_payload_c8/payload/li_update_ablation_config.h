#ifndef VLLM_FAST_LI_UPDATE_ABLATION_CONFIG_H
#define VLLM_FAST_LI_UPDATE_ABLATION_CONFIG_H

// P5 is the production path from ops_li_update_a5@0362e7e: construct the
// survivor payload, classify hit/miss entries, select victims, and commit the
// request-pool cache state.  Diagnostic P0-P4 builds are intentionally not
// exposed by this standalone nano-vLLM operator project.
#define LI_UPDATE_ABLATION_MODE 5

#endif // VLLM_FAST_LI_UPDATE_ABLATION_CONFIG_H
