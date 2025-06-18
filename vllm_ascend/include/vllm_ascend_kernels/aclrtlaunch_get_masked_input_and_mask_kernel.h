#ifndef HEADER_ACLRTLAUNCH_GET_MASKED_INPUT_AND_MASK_KERNEL_H
#define HEADER_ACLRTLAUNCH_GET_MASKED_INPUT_AND_MASK_KERNEL_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_get_masked_input_and_mask_kernel(uint32_t blockDim, aclrtStream stream, void* input, void* masked_input, void* mask_out, const int64_t org_vocab_start_index, const int64_t org_vocab_end_index, const int64_t num_org_vocab_padding, const int64_t added_vocab_start_index, const int64_t added_vocab_end_index, const int64_t size, const uint32_t loop_cnt, const uint32_t aiv_num);
#endif
