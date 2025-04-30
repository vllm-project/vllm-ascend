
#ifndef HEADER_ACLRTLAUNCH_ADVANCESTEPFLASHATTNKERNEL_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_ADVANCESTEPFLASHATTNKERNEL_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_AdvanceStepFlashAttnKernel(uint32_t blockDim, void* stream, int64_t num_seqs, int64_t num_queries, int64_t block_size, void* input_tokens_ptr, void* sampled_token_ids_ptr, void* input_positions_ptr, void* seq_lens_ptr, void* slot_mapping_ptr, void* block_tables_ptr, int64_t block_tables_stride, int32_t tasks_per_core);

inline uint32_t AdvanceStepFlashAttnKernel(uint32_t blockDim, void* hold, void* stream, int64_t num_seqs, int64_t num_queries, int64_t block_size, void* input_tokens_ptr, void* sampled_token_ids_ptr, void* input_positions_ptr, void* seq_lens_ptr, void* slot_mapping_ptr, void* block_tables_ptr, int64_t block_tables_stride, int32_t tasks_per_core)
{
    (void)hold;
    return aclrtlaunch_AdvanceStepFlashAttnKernel(blockDim, stream, num_seqs, num_queries, block_size, input_tokens_ptr, sampled_token_ids_ptr, input_positions_ptr, seq_lens_ptr, slot_mapping_ptr, block_tables_ptr, block_tables_stride, tasks_per_core);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_ROPE_CUSTOM_FALSE_BFLOAT16_T_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_ROPE_CUSTOM_FALSE_BFLOAT16_T_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_rope_custom_false_bfloat16_t(uint32_t blockDim, void* stream, void* positions, void* queryDst, void* keyDst, void* query, void* key, void* cosSinCache, const int rotDim, const int64_t queryStride, const int64_t keyStride, const int64_t dstQueryStride, const int64_t dstKeyStride, const int numHeads, const int numKvHeads, const int headSize, const int64_t numTokens, const int loopNum, const int coreNum);

inline uint32_t rope_custom_false_bfloat16_t(uint32_t blockDim, void* hold, void* stream, void* positions, void* queryDst, void* keyDst, void* query, void* key, void* cosSinCache, const int rotDim, const int64_t queryStride, const int64_t keyStride, const int64_t dstQueryStride, const int64_t dstKeyStride, const int numHeads, const int numKvHeads, const int headSize, const int64_t numTokens, const int loopNum, const int coreNum)
{
    (void)hold;
    return aclrtlaunch_rope_custom_false_bfloat16_t(blockDim, stream, positions, queryDst, keyDst, query, key, cosSinCache, rotDim, queryStride, keyStride, dstQueryStride, dstKeyStride, numHeads, numKvHeads, headSize, numTokens, loopNum, coreNum);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_ROPE_CUSTOM_FALSE_HALF_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_ROPE_CUSTOM_FALSE_HALF_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_rope_custom_false_half(uint32_t blockDim, void* stream, void* positions, void* queryDst, void* keyDst, void* query, void* key, void* cosSinCache, const int rotDim, const int64_t queryStride, const int64_t keyStride, const int64_t dstQueryStride, const int64_t dstKeyStride, const int numHeads, const int numKvHeads, const int headSize, const int64_t numTokens, const int loopNum, const int coreNum);

inline uint32_t rope_custom_false_half(uint32_t blockDim, void* hold, void* stream, void* positions, void* queryDst, void* keyDst, void* query, void* key, void* cosSinCache, const int rotDim, const int64_t queryStride, const int64_t keyStride, const int64_t dstQueryStride, const int64_t dstKeyStride, const int numHeads, const int numKvHeads, const int headSize, const int64_t numTokens, const int loopNum, const int coreNum)
{
    (void)hold;
    return aclrtlaunch_rope_custom_false_half(blockDim, stream, positions, queryDst, keyDst, query, key, cosSinCache, rotDim, queryStride, keyStride, dstQueryStride, dstKeyStride, numHeads, numKvHeads, headSize, numTokens, loopNum, coreNum);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_ROPE_CUSTOM_TRUE_BFLOAT16_T_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_ROPE_CUSTOM_TRUE_BFLOAT16_T_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_rope_custom_true_bfloat16_t(uint32_t blockDim, void* stream, void* positions, void* queryDst, void* keyDst, void* query, void* key, void* cosSinCache, const int rotDim, const int64_t queryStride, const int64_t keyStride, const int64_t dstQueryStride, const int64_t dstKeyStride, const int numHeads, const int numKvHeads, const int headSize, const int64_t numTokens, const int loopNum, const int coreNum);

inline uint32_t rope_custom_true_bfloat16_t(uint32_t blockDim, void* hold, void* stream, void* positions, void* queryDst, void* keyDst, void* query, void* key, void* cosSinCache, const int rotDim, const int64_t queryStride, const int64_t keyStride, const int64_t dstQueryStride, const int64_t dstKeyStride, const int numHeads, const int numKvHeads, const int headSize, const int64_t numTokens, const int loopNum, const int coreNum)
{
    (void)hold;
    return aclrtlaunch_rope_custom_true_bfloat16_t(blockDim, stream, positions, queryDst, keyDst, query, key, cosSinCache, rotDim, queryStride, keyStride, dstQueryStride, dstKeyStride, numHeads, numKvHeads, headSize, numTokens, loopNum, coreNum);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_ROPE_CUSTOM_TRUE_HALF_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_ROPE_CUSTOM_TRUE_HALF_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_rope_custom_true_half(uint32_t blockDim, void* stream, void* positions, void* queryDst, void* keyDst, void* query, void* key, void* cosSinCache, const int rotDim, const int64_t queryStride, const int64_t keyStride, const int64_t dstQueryStride, const int64_t dstKeyStride, const int numHeads, const int numKvHeads, const int headSize, const int64_t numTokens, const int loopNum, const int coreNum);

inline uint32_t rope_custom_true_half(uint32_t blockDim, void* hold, void* stream, void* positions, void* queryDst, void* keyDst, void* query, void* key, void* cosSinCache, const int rotDim, const int64_t queryStride, const int64_t keyStride, const int64_t dstQueryStride, const int64_t dstKeyStride, const int numHeads, const int numKvHeads, const int headSize, const int64_t numTokens, const int loopNum, const int coreNum)
{
    (void)hold;
    return aclrtlaunch_rope_custom_true_half(blockDim, stream, positions, queryDst, keyDst, query, key, cosSinCache, rotDim, queryStride, keyStride, dstQueryStride, dstKeyStride, numHeads, numKvHeads, headSize, numTokens, loopNum, coreNum);
}

#endif
