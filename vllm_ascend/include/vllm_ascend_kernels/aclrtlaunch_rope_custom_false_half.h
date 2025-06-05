#ifndef HEADER_ACLRTLAUNCH_ROPE_CUSTOM_FALSE_HALF_H
#define HEADER_ACLRTLAUNCH_ROPE_CUSTOM_FALSE_HALF_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_rope_custom_false_half(uint32_t blockDim, aclrtStream stream, void* positions, void* queryDst, void* keyDst, void* query, void* key, void* cosSinCache, const int rotDim, const int64_t queryStride, const int64_t keyStride, const int64_t dstQueryStride, const int64_t dstKeyStride, const int numHeads, const int numKvHeads, const int headSize, const int64_t numTokens, const int loopNum, const int coreNum);
#endif
