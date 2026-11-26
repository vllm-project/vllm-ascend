#ifndef KVTC_SHAPE_H
#define KVTC_SHAPE_H

#include <c10/util/Exception.h>

#include <cstdint>
#include <limits>
#include <utility>

namespace vllm_ascend {

template <typename Dim>
inline void check_kvtc_cache_shape(const Dim& num_gpu_blocks, const Dim& block_size, const Dim& num_heads,
                                  const Dim& head_size, const Dim& selected_blocks, bool with_rope,
                                  const char* op)
{
    TORCH_CHECK(num_gpu_blocks > 0 && block_size > 0 && num_heads > 0 && head_size > 0,
                op, ": cache dimensions must be positive");
    TORCH_CHECK(selected_blocks > 0, op, ": block_ids must not be empty");

    constexpr int64_t BF16_ELEMENTS_PER_COPY_BLOCK = 16;
    constexpr int64_t MAX_ROPE_HEAD_SIZE = 128;
    if (with_rope) {
        TORCH_CHECK(head_size <= MAX_ROPE_HEAD_SIZE && head_size % BF16_ELEMENTS_PER_COPY_BLOCK == 0,
                    op, ": head size must be <=128 and divisible by 16");
    } else {
        // Plain copies move a whole BF16 row, so only H*D needs 32-byte alignment.
        const auto heads_remainder = num_heads % BF16_ELEMENTS_PER_COPY_BLOCK;
        const auto size_remainder = head_size % BF16_ELEMENTS_PER_COPY_BLOCK;
        TORCH_CHECK(heads_remainder * size_remainder % BF16_ELEMENTS_PER_COPY_BLOCK == 0,
                    op, ": BF16 cache rows (H*D elements) must be 32-byte aligned");
    }
}

// Call after validating the cache and selected block dimensions.
inline std::pair<int64_t, int64_t> kvtc_matrix_shape(int64_t num_layers, int64_t block_size, int64_t num_heads,
                                                     int64_t head_size, int64_t selected_blocks, const char* op)
{
    constexpr int64_t MAX_DIM = std::numeric_limits<int64_t>::max();
    TORCH_CHECK(selected_blocks <= MAX_DIM / block_size && num_layers <= MAX_DIM / num_heads / head_size,
                op, ": matrix dimensions overflow int64");
    return {selected_blocks * block_size, num_layers * num_heads * head_size};
}

}  // namespace vllm_ascend

#endif  // KVTC_SHAPE_H
