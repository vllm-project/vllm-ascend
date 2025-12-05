#ifndef MOE_TOKEN_UNPERMUTE_TILING
#define MOE_TOKEN_UNPERMUTE_TILING

struct MoeTokenUnpermuteTilingData {
    int64_t hidden_size;
    int64_t top_k;
    int64_t num_out_tokens;
    int64_t hidden_splited_length;
    int64_t hidden_splited_num;
    int64_t hidden_splited_remain;
    int64_t tokens_core_length;
    int64_t tokens_core_remain;
    int64_t tokens_splited_length;
    int64_t tokens_splited_num;
    int64_t tokens_splited_remain;
    int64_t buffer_num;
};

__forceinline__ [host, aicore] void
MoeTokenUnpermuteTiling(int32_t m, int32_t n, int32_t topK, MoeTokenUnpermuteTilingData &tilingData, uint32_t coreNum)
{
    #define I64(x) static_cast<int64_t>(x)
    tilingData.hidden_size = I64(n);
    tilingData.top_k = I64(topK);
    tilingData.num_out_tokens = I64(m);
    tilingData.hidden_splited_length = tilingData.hidden_size;
    tilingData.hidden_splited_num = 1;
    tilingData.hidden_splited_remain = 0;
    uint32_t outTokens = m / topK;
    tilingData.tokens_core_length = I64(outTokens / coreNum);
    tilingData.tokens_core_remain = I64(outTokens % coreNum);
    tilingData.tokens_splited_length = I64(min(tilingData.tokens_core_length, 600));
    tilingData.tokens_splited_num = I64(tilingData.tokens_core_length / tilingData.tokens_splited_length);
    tilingData.tokens_splited_remain = I64(tilingData.tokens_core_length % tilingData.tokens_splited_length);
    tilingData.buffer_num = 4;
}

#endif