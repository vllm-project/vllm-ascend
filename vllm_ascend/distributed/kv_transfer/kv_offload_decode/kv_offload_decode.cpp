#include <torch/extension.h>
#include <limits>
#include <optional>
#include <iostream>
#include <chrono>
#include <string>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <numeric>
#include <ATen/Parallel.h>
#include <torch/script.h>

#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <omp.h>
#include <assert.h>
#include <torch/torch.h>
#include <pthread.h>

#if defined(__GNUC__) || defined(__clang__)
#define HOT_FUNCTION __attribute__((hot))
#define FORCE_INLINE inline __attribute__((always_inline))
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#define RESTRICT __restrict__
#else
#define HOT_FUNCTION
#define FORCE_INLINE inline
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#define RESTRICT
#endif

constexpr int32_t EPOCH_RESET_THRESHOLD = 1 << 30;

FORCE_INLINE int choose_lru_resident_threads(
    const int num_reqs,
    const int workspace_threads,
    const int requested_threads
) {
    if (num_reqs <= 1) {
        return 1;
    }
    int active_threads = num_reqs;
    if (workspace_threads > 0) {
        active_threads = std::min(active_threads, workspace_threads);
    }
    if (requested_threads > 0) {
        active_threads = std::min(active_threads, requested_threads);
    }
    return std::max(active_threads, 1);
}

FORCE_INLINE bool is_valid_lru_resident_token(const int32_t token, const int32_t max_token) {
    return token >= 0 && token < max_token;
}

FORCE_INLINE void reset_lru_resident_row(
    int32_t* RESTRICT slot_to_token_row,
    int32_t* RESTRICT lru_slots_row,
    const int32_t capacity
) {
    std::fill(slot_to_token_row, slot_to_token_row + capacity, -1);
    for (int32_t slot = 0; slot < capacity; ++slot) {
        lru_slots_row[slot] = slot;
    }
}

FORCE_INLINE int32_t next_lru_resident_epoch(
    int32_t* RESTRICT token_mark,
    int32_t* RESTRICT token_pos,
    int32_t* RESTRICT epoch,
    const int32_t max_token
) {
    int32_t base = epoch[0] + 1;
    if (UNLIKELY(base >= EPOCH_RESET_THRESHOLD)) {
        std::fill(token_mark, token_mark + max_token, 0);
        std::fill(token_pos, token_pos + max_token, -1);
        base = 1;
    }
    epoch[0] = base;
    return base;
}

FORCE_INLINE void process_one_lru_resident_row(
    const int row,
    const int32_t topk,
    const int32_t capacity,
    const int32_t max_token,
    const int64_t* RESTRICT req_ids,
    int64_t* RESTRICT last_req_ids,
    const int32_t* RESTRICT topk_indices,
    const int32_t* RESTRICT stable_prefix_lens,
    int32_t* RESTRICT slot_to_token,
    int32_t* RESTRICT lru_slots,
    int32_t* RESTRICT current_slots,
    int32_t* RESTRICT miss_count,
    int32_t* RESTRICT miss_tokens_out,
    int32_t* RESTRICT miss_slots_out,
    int32_t* RESTRICT token_mark,
    int32_t* RESTRICT token_pos,
    int32_t* RESTRICT slot_workspace,
    int32_t* RESTRICT miss_positions,
    int32_t* RESTRICT epoch
) {
    int32_t* RESTRICT slot_to_token_row = slot_to_token + static_cast<int64_t>(row) * capacity;
    int32_t* RESTRICT lru_slots_row = lru_slots + static_cast<int64_t>(row) * capacity;
    int32_t* RESTRICT current_slots_row = current_slots + static_cast<int64_t>(row) * topk;
    int32_t* RESTRICT miss_tokens_row = miss_tokens_out + static_cast<int64_t>(row) * topk;
    int32_t* RESTRICT miss_slots_row = miss_slots_out + static_cast<int64_t>(row) * topk;
    const int32_t* RESTRICT topk_row = topk_indices + static_cast<int64_t>(row) * topk;
    int32_t* RESTRICT hit_slots = slot_workspace;
    int32_t* RESTRICT evictable_slots = slot_workspace + capacity;
    int32_t* RESTRICT assigned_miss_slots = slot_workspace + static_cast<int64_t>(capacity) * 2;

    std::fill(current_slots_row, current_slots_row + topk, -1);
    std::fill(miss_tokens_row, miss_tokens_row + topk, -1);
    std::fill(miss_slots_row, miss_slots_row + topk, -1);
    miss_count[row] = 0;

    const int64_t req_id = req_ids[row];
    if (UNLIKELY(last_req_ids[row] != req_id)) {
        reset_lru_resident_row(slot_to_token_row, lru_slots_row, capacity);
        last_req_ids[row] = req_id;
    }
    const int32_t stable_prefix_len = std::clamp(
        stable_prefix_lens[row],
        0,
        max_token
    );

    const int32_t base = next_lru_resident_epoch(token_mark, token_pos, epoch, max_token);
    for (int32_t pos = 0; pos < topk; ++pos) {
        const int32_t token = topk_row[pos];
        if (LIKELY(is_valid_lru_resident_token(token, max_token)) && token_mark[token] != base) {
            token_mark[token] = base;
            token_pos[token] = pos;
        }
    }

    int32_t hit_count = 0;
    int32_t evictable_count = 0;
    for (int32_t order = 0; order < capacity; ++order) {
        const int32_t slot = lru_slots_row[order];
        if (UNLIKELY(slot < 0 || slot >= capacity)) {
            continue;
        }
        int32_t token = slot_to_token_row[slot];
        // The speculative suffix may have been overwritten in the CPU KV pool.
        if (LIKELY(is_valid_lru_resident_token(token, max_token)) &&
            UNLIKELY(token >= stable_prefix_len)) {
            slot_to_token_row[slot] = -1;
            token = -1;
        }
        if (LIKELY(is_valid_lru_resident_token(token, max_token)) && token_mark[token] == base) {
            const int32_t pos = token_pos[token];
            current_slots_row[pos] = slot;
            hit_slots[hit_count] = slot;
            ++hit_count;
        } else {
            evictable_slots[evictable_count] = slot;
            ++evictable_count;
        }
    }

    int32_t local_miss_count = 0;
    for (int32_t pos = 0; pos < topk; ++pos) {
        const int32_t token = topk_row[pos];
        if (LIKELY(is_valid_lru_resident_token(token, max_token)) && current_slots_row[pos] < 0) {
            miss_tokens_row[local_miss_count] = token;
            miss_positions[local_miss_count] = pos;
            ++local_miss_count;
        }
    }

    const int32_t assign_count = std::min(local_miss_count, evictable_count);
    for (int32_t miss_idx = 0; miss_idx < assign_count; ++miss_idx) {
        const int32_t slot = evictable_slots[miss_idx];
        const int32_t token = miss_tokens_row[miss_idx];
        const int32_t pos = miss_positions[miss_idx];
        slot_to_token_row[slot] = token;
        current_slots_row[pos] = slot;
        miss_slots_row[miss_idx] = slot;
        assigned_miss_slots[miss_idx] = slot;
        miss_count[row] = miss_idx + 1;
    }

    int32_t write_pos = 0;
    for (int32_t idx = assign_count; idx < evictable_count; ++idx) {
        lru_slots_row[write_pos] = evictable_slots[idx];
        ++write_pos;
    }
    for (int32_t idx = 0; idx < assign_count; ++idx) {
        lru_slots_row[write_pos] = assigned_miss_slots[idx];
        ++write_pos;
    }
    for (int32_t idx = 0; idx < hit_count; ++idx) {
        lru_slots_row[write_pos] = hit_slots[idx];
        ++write_pos;
    }
}

HOT_FUNCTION void lru_resident_compact(
    uintptr_t req_ids_ptr,
    uintptr_t last_req_ids_ptr,
    uintptr_t topk_indices_ptr,
    uintptr_t stable_prefix_lens_ptr,
    uintptr_t slot_to_token_ptr,
    uintptr_t lru_slots_ptr,
    uintptr_t current_slots_ptr,
    uintptr_t miss_count_ptr,
    uintptr_t miss_tokens_ptr,
    uintptr_t miss_slots_ptr,
    uintptr_t token_mark_workspace_ptr,
    uintptr_t token_pos_workspace_ptr,
    uintptr_t slot_workspace_ptr,
    uintptr_t miss_position_workspace_ptr,
    uintptr_t epochs_ptr,
    int64_t num_reqs,
    int64_t topk,
    int64_t capacity,
    int64_t max_token,
    int64_t workspace_threads,
    int64_t requested_threads
) {
    if (num_reqs <= 0 || topk <= 0 || capacity <= 0 || max_token <= 0) {
        return;
    }

    auto* RESTRICT req_ids = reinterpret_cast<int64_t*>(req_ids_ptr);
    auto* RESTRICT last_req_ids = reinterpret_cast<int64_t*>(last_req_ids_ptr);
    auto* RESTRICT topk_indices = reinterpret_cast<int32_t*>(topk_indices_ptr);
    auto* RESTRICT stable_prefix_lens = reinterpret_cast<int32_t*>(stable_prefix_lens_ptr);
    auto* RESTRICT slot_to_token = reinterpret_cast<int32_t*>(slot_to_token_ptr);
    auto* RESTRICT lru_slots = reinterpret_cast<int32_t*>(lru_slots_ptr);
    auto* RESTRICT current_slots = reinterpret_cast<int32_t*>(current_slots_ptr);
    auto* RESTRICT miss_count = reinterpret_cast<int32_t*>(miss_count_ptr);
    auto* RESTRICT miss_tokens = reinterpret_cast<int32_t*>(miss_tokens_ptr);
    auto* RESTRICT miss_slots = reinterpret_cast<int32_t*>(miss_slots_ptr);
    auto* RESTRICT token_mark_workspace = reinterpret_cast<int32_t*>(token_mark_workspace_ptr);
    auto* RESTRICT token_pos_workspace = reinterpret_cast<int32_t*>(token_pos_workspace_ptr);
    auto* RESTRICT slot_workspace = reinterpret_cast<int32_t*>(slot_workspace_ptr);
    auto* RESTRICT miss_position_workspace = reinterpret_cast<int32_t*>(miss_position_workspace_ptr);
    auto* RESTRICT epochs = reinterpret_cast<int32_t*>(epochs_ptr);

    const int num_reqs_int = static_cast<int>(num_reqs);
    const int topk_int = static_cast<int>(topk);
    const int capacity_int = static_cast<int>(capacity);
    const int max_token_int = static_cast<int>(max_token);
    const int active_threads = choose_lru_resident_threads(
        num_reqs_int,
        static_cast<int>(workspace_threads),
        static_cast<int>(requested_threads)
    );

    if (active_threads == 1) {
        for (int row = 0; row < num_reqs_int; ++row) {
            process_one_lru_resident_row(
                row,
                topk_int,
                capacity_int,
                max_token_int,
                req_ids,
                last_req_ids,
                topk_indices,
                stable_prefix_lens,
                slot_to_token,
                lru_slots,
                current_slots,
                miss_count,
                miss_tokens,
                miss_slots,
                token_mark_workspace,
                token_pos_workspace,
                slot_workspace,
                miss_position_workspace,
                epochs
            );
        }
        return;
    }

    #pragma omp parallel num_threads(active_threads)
    {
        const int thread_id = omp_get_thread_num();
        int32_t* RESTRICT token_mark = token_mark_workspace + static_cast<int64_t>(thread_id) * max_token_int;
        int32_t* RESTRICT token_pos = token_pos_workspace + static_cast<int64_t>(thread_id) * max_token_int;
        int32_t* RESTRICT slots = slot_workspace + static_cast<int64_t>(thread_id) * capacity_int * 3;
        int32_t* RESTRICT miss_positions = miss_position_workspace + static_cast<int64_t>(thread_id) * topk_int;
        int32_t* RESTRICT epoch = epochs + thread_id;
        for (int row = thread_id; row < num_reqs_int; row += active_threads) {
            process_one_lru_resident_row(
                row,
                topk_int,
                capacity_int,
                max_token_int,
                req_ids,
                last_req_ids,
                topk_indices,
                stable_prefix_lens,
                slot_to_token,
                lru_slots,
                current_slots,
                miss_count,
                miss_tokens,
                miss_slots,
                token_mark,
                token_pos,
                slots,
                miss_positions,
                epoch
            );
        }
    }
}

int32_t
compute_lru_resident_addrs(
    const at::Tensor& miss_count,
    const at::Tensor& miss_tokens,
    const at::Tensor& miss_slots,
    const at::Tensor& block_table,
    const int32_t block_size,
    const int32_t token_size_bytes_k,
    const int32_t token_size_bytes_v,
    const int64_t gvas_k_base,
    const int64_t gvas_v_base,
    const int64_t addr_k_base,
    const int64_t addr_v_base,
    const int32_t resident_capacity,
    const int32_t max_num_threads,
    at::Tensor& gvas_buffer,
    at::Tensor& addr_buffer,
    at::Tensor& size_buffer,
    at::Tensor& num_tokens_buffer
) {
    const int32_t num_reqs = miss_tokens.size(0);
    const int32_t topk = miss_tokens.size(1);
    const int32_t max_num_tokens_to_load = num_reqs * topk;
    const int32_t max_num_blocks = block_table.size(1);
    const int32_t block_size_bytes_k = block_size * token_size_bytes_k;
    const int32_t block_size_bytes_v = block_size * token_size_bytes_v;
    TORCH_CHECK(miss_count.size(0) >= num_reqs,
        "miss_count size not enough for loading.");
    TORCH_CHECK(miss_slots.size(0) == num_reqs && miss_slots.size(1) == topk,
        "miss_slots shape should match miss_tokens.");
    TORCH_CHECK(gvas_buffer.size(0) >= max_num_tokens_to_load * 2,
        "gvas_buffer size not enough for loading.");
    TORCH_CHECK(addr_buffer.size(0) >= max_num_tokens_to_load * 2,
        "addr_buffer size not enough for loading.");
    TORCH_CHECK(size_buffer.size(0) >= max_num_tokens_to_load * 2,
        "size_buffer size not enough for loading.");
    TORCH_CHECK(miss_count.scalar_type() == at::kInt,
        "miss_count wrong dtype, should be int32.");
    TORCH_CHECK(miss_tokens.scalar_type() == at::kInt,
        "miss_tokens wrong dtype, should be int32.");
    TORCH_CHECK(miss_slots.scalar_type() == at::kInt,
        "miss_slots wrong dtype, should be int32.");
    TORCH_CHECK(block_table.scalar_type() == at::kInt,
        "block_table wrong dtype, should be int32.");
    TORCH_CHECK(gvas_buffer.scalar_type() == at::kLong,
        "gvas_buffer wrong dtype, should be int64.");
    TORCH_CHECK(addr_buffer.scalar_type() == at::kLong,
        "addr_buffer wrong dtype, should be int64.");
    TORCH_CHECK(size_buffer.scalar_type() == at::kInt,
        "size_buffer wrong dtype, should be int32.");

    const int32_t* miss_count_ptr = static_cast<int32_t*>(miss_count.data_ptr());
    const int32_t* miss_tokens_ptr = static_cast<int32_t*>(miss_tokens.data_ptr());
    const int32_t* miss_slots_ptr = static_cast<int32_t*>(miss_slots.data_ptr());
    const int32_t* block_table_ptr = static_cast<int32_t*>(block_table.data_ptr());
    int64_t* gvas_buffer_ptr = static_cast<int64_t*>(gvas_buffer.data_ptr());
    int64_t* addr_buffer_ptr = static_cast<int64_t*>(addr_buffer.data_ptr());
    int32_t* size_buffer_ptr = static_cast<int32_t*>(size_buffer.data_ptr());
    int32_t* num_tokens_buffer_ptr = static_cast<int32_t*>(num_tokens_buffer.data_ptr());

    int n_threads = std::min(num_reqs, max_num_threads);
    n_threads = std::max(n_threads, 1);

    std::vector<int32_t> num_tokens_to_load_req(num_reqs);
    #pragma omp parallel for num_threads(n_threads)
    for (size_t req_idx = 0; req_idx < num_reqs; ++req_idx) {
        int32_t count = miss_count_ptr[req_idx];
        count = std::max(count, 0);
        count = std::min(count, topk);
        num_tokens_to_load_req[req_idx] = count;
    }
    int32_t num_tokens_to_load_sum = std::accumulate(num_tokens_to_load_req.begin(), num_tokens_to_load_req.end(), 0);
    std::vector<int32_t> req_start_locs(num_reqs);
    int32_t cumsum = 0;
    for (size_t req_idx = 0; req_idx < num_reqs; ++req_idx) {
        req_start_locs[req_idx] = cumsum;
        cumsum += num_tokens_to_load_req[req_idx];
    }

    #pragma omp parallel for num_threads(n_threads)
    for (size_t req_idx = 0; req_idx < num_reqs; ++req_idx) {
        int32_t req_start_loc_k = req_start_locs[req_idx];
        int32_t req_start_loc_v = num_tokens_to_load_sum + req_start_loc_k;
        for (int32_t idx = 0; idx < num_tokens_to_load_req[req_idx]; ++idx) {
            const int32_t token = miss_tokens_ptr[req_idx * topk + idx];
            const int32_t slot = miss_slots_ptr[req_idx * topk + idx];
            if (token < 0 || slot < 0 || slot >= resident_capacity) {
                continue;
            }
            const int32_t block_id = token / block_size;
            if (block_id < 0 || block_id >= max_num_blocks) {
                continue;
            }
            const int32_t offset_in_block = token % block_size;
            const int32_t block_indice = block_table_ptr[req_idx * max_num_blocks + block_id];
            const int64_t gvas_k =
                gvas_k_base + block_indice * block_size_bytes_k + offset_in_block * token_size_bytes_k;
            const int64_t gvas_v =
                gvas_v_base + block_indice * block_size_bytes_v + offset_in_block * token_size_bytes_v;
            const int64_t addr_k = addr_k_base + (req_idx * resident_capacity + slot) * token_size_bytes_k;
            const int64_t addr_v = addr_v_base + (req_idx * resident_capacity + slot) * token_size_bytes_v;
            gvas_buffer_ptr[req_start_loc_k + idx] = gvas_k;
            gvas_buffer_ptr[req_start_loc_v + idx] = gvas_v;
            addr_buffer_ptr[req_start_loc_k + idx] = addr_k;
            addr_buffer_ptr[req_start_loc_v + idx] = addr_v;
        }
    }

    std::fill_n(size_buffer_ptr, num_tokens_to_load_sum, token_size_bytes_k);
    std::fill_n(&(size_buffer_ptr[num_tokens_to_load_sum]), num_tokens_to_load_sum, token_size_bytes_v);

    num_tokens_buffer_ptr[0] = num_tokens_to_load_sum * 2;
    return num_tokens_to_load_sum;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    namespace py = pybind11;
    m.def("lru_resident_compact", &lru_resident_compact,
          "CPU LRU resident compact miss prepare with OpenMP row-level parallelism");
    m.def("compute_lru_resident_addrs", &compute_lru_resident_addrs,
          "Compute sparse H2D metadata for compact LRU resident miss loads");
}
