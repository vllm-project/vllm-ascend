#ifndef HEADER_ACLRTLAUNCH_ADVANCESTEPFLASHATTNKERNEL_H
#define HEADER_ACLRTLAUNCH_ADVANCESTEPFLASHATTNKERNEL_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_AdvanceStepFlashAttnKernel(uint32_t blockDim, aclrtStream stream, int64_t num_seqs, int64_t num_queries, int64_t block_size, void* input_tokens_ptr, void* sampled_token_ids_ptr, void* input_positions_ptr, void* seq_lens_ptr, void* slot_mapping_ptr, void* block_tables_ptr, int64_t block_tables_stride, int32_t tasks_per_core);
#endif
