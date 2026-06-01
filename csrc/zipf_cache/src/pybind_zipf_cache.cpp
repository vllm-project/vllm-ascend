/**
 * pybind11 bindings for ZipfCache
 *
 * High-performance Python bindings with minimal overhead.
 * Uses opaque pointer to avoid C/C++ atomic compatibility issues.
 *
 * 硬编码参数（不再暴露给 Python）：
 * - global_capacity = 33554432 (已移除 global_hash)
 * - shared_capacity = 8388608
 * - local_capacity = 40000
 * - max_local_hashes = 10000
 * - unigram_capacity = 131072
 * - unigram_threshold = 0.75
 * - enable_unigram = true
 * - init_score = 28800, decay_same = 0/8, decay_diff = 0/3
 *
 * 可配置参数：
 * - min_window, max_window
 * - skip_shared
 * - enable_generalized, gen_freq_threshold, gen_capacity, gen_bitmap_vocab_max, gen_min_score
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <optional>
#include <stdexcept>
#include <cstdint>
#include <memory>

// Forward declare C types as opaque
struct ZipfCache;
struct ZipfConfig;
struct ZipfKey;
struct QueryResult;
struct ZipfCacheStats;
struct LocalHashStats;
struct GeneralizerStats;

// C API declarations
extern "C" {

struct ZipfConfig {
    uint8_t min_window;
    uint8_t max_window;
    uint8_t skip_shared;
    uint8_t enable_generalized;
    uint32_t gen_freq_threshold;
    size_t gen_capacity;
    uint32_t gen_bitmap_vocab_max;
    uint8_t generalized_before_shared;
};

struct ZipfKey {
    uint32_t tokens[8];  // MAX_NGRAM_ORDER = 8
    uint8_t length;
};

struct QueryResult {
    uint32_t next_token;
    uint8_t hit;
    uint8_t from_local;
};

struct LocalHashStats {
    size_t total_hashes;
    size_t total_entries;
    size_t total_unigram;
    size_t total_gen_entries;
    size_t memory_bytes;
};

struct GeneralizerStats {
    size_t capacity;
    size_t count;
    uint32_t freq_threshold;
    uint32_t total_updates;
    size_t high_freq_count;
    size_t num_wildcard_buckets;
    size_t memory_bytes;
};

struct ZipfCacheStats {
    uint64_t query_count;
    uint64_t shared_hit_count;
    uint64_t local_hit_count;
    uint64_t unigram_hit_count;
    uint64_t generalized_hit_count;
    double shared_hit_rate;
    double local_hit_rate;
    double unigram_hit_rate;
    double generalized_hit_rate;
    double total_hit_rate;
    uint64_t update_count;
    LocalHashStats local_stats;
    size_t shared_entries;
    size_t shared_unigram;
    GeneralizerStats gen_stats;
};

ZipfCache* zipf_cache_create(const ZipfConfig* config);
void zipf_cache_free(ZipfCache* cache);
void zipf_cache_query_batch_with_req(ZipfCache* cache, const uint64_t* req_ids, const ZipfKey* keys, QueryResult* results, size_t batch_size);
void zipf_cache_update_with_req(ZipfCache* cache, uint64_t req_id, const uint32_t* context_tokens, uint8_t context_len, uint32_t next_token);
void zipf_cache_update_with_history(ZipfCache* cache, uint64_t req_id, const uint32_t* new_tokens, size_t new_token_count);
void zipf_cache_update_with_history_batch(ZipfCache* cache, const uint64_t* req_ids, const uint32_t* flat_tokens, const size_t* offsets, const size_t* token_counts, size_t batch_size);
void zipf_cache_set_history(ZipfCache* cache, uint64_t req_id, const uint32_t* tokens, size_t token_count);
void zipf_cache_warmup_from_prompt(ZipfCache* cache, uint64_t req_id, const uint32_t* prompt_tokens, size_t prompt_len);
void zipf_cache_reset_context(ZipfCache* cache, uint64_t req_id);
void zipf_cache_update_batch_with_req(ZipfCache* cache, const uint64_t* req_ids, const ZipfKey* contexts, const uint32_t* next_tokens, size_t batch_size);
ZipfCacheStats zipf_cache_stats(const ZipfCache* cache);
int zipf_cache_speculate_chain_with_req(ZipfCache* cache, uint64_t req_id, const uint32_t* context, uint8_t context_len, uint32_t* out_tokens, int max_tokens);
void zipf_cache_speculate_chain_batch_with_req(ZipfCache* cache, const uint64_t* req_ids, const ZipfKey* contexts, uint32_t* out_tokens, int* out_counts, size_t batch_size, int max_tokens);
uint64_t zipf_hash(const uint32_t* tokens, uint8_t length);

// Propose API
struct AdaptiveKConfig {
    float ema_alpha;
    int num_speculative_tokens;
    float fail_threshold;
    float success_threshold;
    int max_penalty;
    int max_bonus;
    int min_k;
    int max_k;
};

struct ProposeRequest {
    uint64_t req_hash;
    const uint32_t* token_ids;
    int num_tokens;
    int num_prompt_tokens;
    const uint32_t* sampled_ids;
    int num_sampled;
    int max_model_len;
    uint8_t is_valid;
};

struct ProposeResult {
    uint32_t* draft_tokens;
    int num_draft_tokens;
};

void zipf_cache_propose_batch(
    ZipfCache* cache,
    const ProposeRequest* requests,
    ProposeResult* results,
    size_t batch_size,
    const AdaptiveKConfig* ak_config
);

}

namespace py = pybind11;

class PyZipfCache {
public:
    PyZipfCache(
        // 可配置参数
        uint8_t min_window = 2,
        uint8_t max_window = 4,
        bool skip_shared = false,
        // Generalized parameters
        bool enable_generalized = true,
        uint32_t gen_freq_threshold = 8,
        size_t gen_capacity = 64 * 1024,
        uint32_t gen_bitmap_vocab_max = 256 * 1024,
        // Query priority
        bool generalized_before_shared = true
    ) {
        ZipfConfig config{};
        config.min_window = min_window;
        config.max_window = max_window;
        config.skip_shared = skip_shared ? 1 : 0;
        config.enable_generalized = enable_generalized ? 1 : 0;
        config.gen_freq_threshold = gen_freq_threshold;
        config.gen_capacity = gen_capacity;
        config.gen_bitmap_vocab_max = gen_bitmap_vocab_max;
        config.generalized_before_shared = generalized_before_shared ? 1 : 0;

        cache_ = zipf_cache_create(&config);
        if (!cache_) {
            throw std::runtime_error("Failed to create ZipfCache");
        }
        max_window_ = max_window;
    }

    ~PyZipfCache() {
        if (cache_) {
            zipf_cache_free(cache_);
        }
    }

    // Disable copy
    PyZipfCache(const PyZipfCache&) = delete;
    PyZipfCache& operator=(const PyZipfCache&) = delete;

    // Move constructor
    PyZipfCache(PyZipfCache&& other) noexcept : cache_(other.cache_), max_window_(other.max_window_) {
        other.cache_ = nullptr;
    }
    PyZipfCache& operator=(PyZipfCache&& other) noexcept {
        if (this != &other) {
            if (cache_) zipf_cache_free(cache_);
            cache_ = other.cache_;
            max_window_ = other.max_window_;
            other.cache_ = nullptr;
        }
        return *this;
    }

private:
    PyZipfCache(ZipfCache* cache, uint8_t max_window = 4) : cache_(cache), max_window_(max_window) {}

public:
    // Query with req_id (local + shared hash)
    std::optional<uint32_t> query_with_req(uint64_t req_id, const std::vector<uint32_t>& context) {
        if (context.empty() || context.size() > max_window_) return std::nullopt;
        ZipfKey key;
        key.length = context.size();
        for (size_t i = 0; i < context.size(); i++) key.tokens[i] = context[i];
        QueryResult result;
        zipf_cache_query_batch_with_req(cache_, &req_id, &key, &result, 1);
        if (result.hit) return result.next_token;
        return std::nullopt;
    }

    // Batch query with req_ids
    std::vector<std::optional<uint32_t>> query_batch_with_req(
        const std::vector<uint64_t>& req_ids,
        const std::vector<std::vector<uint32_t>>& contexts
    ) {
        size_t n = contexts.size();
        std::vector<std::optional<uint32_t>> results(n);
        if (n == 0) return results;
        if (n != req_ids.size()) throw std::invalid_argument("req_ids and contexts must have same length");
        std::vector<ZipfKey> keys(n);
        std::vector<QueryResult> query_results(n);
        for (size_t i = 0; i < n; i++) {
            const auto& ctx = contexts[i];
            size_t len = std::min(ctx.size(), size_t(max_window_));
            keys[i].length = len;
            for (size_t j = 0; j < len; j++) keys[i].tokens[j] = ctx[ctx.size() - len + j];
        }
        zipf_cache_query_batch_with_req(cache_, req_ids.data(), keys.data(), query_results.data(), n);
        for (size_t i = 0; i < n; i++) {
            if (query_results[i].hit) results[i] = query_results[i].next_token;
        }
        return results;
    }

    // Update with req_id (legacy API)
    void update_with_req(uint64_t req_id, const std::vector<uint32_t>& context, uint32_t next_token) {
        if (context.empty()) return;
        size_t len = std::min(context.size(), size_t(max_window_));
        const uint32_t* ptr = context.data() + context.size() - len;
        zipf_cache_update_with_req(cache_, req_id, ptr, len, next_token);
    }

    void update_with_history(uint64_t req_id, const std::vector<uint32_t>& new_tokens) {
        if (new_tokens.empty()) return;
        zipf_cache_update_with_history(cache_, req_id, new_tokens.data(), new_tokens.size());
    }

    void update_with_history_batch(
        const std::vector<uint64_t>& req_ids,
        const std::vector<std::vector<uint32_t>>& token_arrays
    ) {
        size_t n = req_ids.size();
        if (n == 0 || n != token_arrays.size()) return;
        size_t total_tokens = 0;
        std::vector<size_t> offsets(n);
        std::vector<size_t> counts(n);
        for (size_t i = 0; i < n; i++) {
            offsets[i] = total_tokens;
            counts[i] = token_arrays[i].size();
            total_tokens += counts[i];
        }
        std::vector<uint32_t> flat(total_tokens);
        for (size_t i = 0; i < n; i++) {
            if (counts[i] > 0) {
                std::memcpy(&flat[offsets[i]], token_arrays[i].data(), counts[i] * sizeof(uint32_t));
            }
        }
        zipf_cache_update_with_history_batch(cache_, req_ids.data(), flat.data(), offsets.data(), counts.data(), n);
    }

    void set_history(uint64_t req_id, const std::vector<uint32_t>& tokens) {
        if (tokens.empty()) return;
        zipf_cache_set_history(cache_, req_id, tokens.data(), tokens.size());
    }

    void warmup_from_prompt(uint64_t req_id, const std::vector<uint32_t>& prompt_tokens) {
        if (prompt_tokens.empty()) return;
        zipf_cache_warmup_from_prompt(cache_, req_id, prompt_tokens.data(), prompt_tokens.size());
    }

    void reset_context(uint64_t req_id) {
        zipf_cache_reset_context(cache_, req_id);
    }

    void update_batch_with_req(
        const std::vector<uint64_t>& req_ids,
        const std::vector<std::vector<uint32_t>>& contexts,
        const std::vector<uint32_t>& next_tokens
    ) {
        size_t n = contexts.size();
        if (n == 0) return;
        if (n != next_tokens.size() || n != req_ids.size()) {
            throw std::invalid_argument("req_ids, contexts and next_tokens must have same length");
        }
        std::vector<ZipfKey> keys(n);
        for (size_t i = 0; i < n; i++) {
            const auto& ctx = contexts[i];
            size_t len = std::min(ctx.size(), size_t(max_window_));
            keys[i].length = len;
            for (size_t j = 0; j < len; j++) keys[i].tokens[j] = ctx[ctx.size() - len + j];
        }
        zipf_cache_update_batch_with_req(cache_, req_ids.data(), keys.data(), next_tokens.data(), n);
    }

    // Chain speculation with req_id
    std::vector<uint32_t> speculate_chain_with_req(
        uint64_t req_id,
        const std::vector<uint32_t>& context,
        int max_tokens
    ) {
        if (context.empty() || max_tokens <= 0) return {};
        size_t len = std::min(context.size(), size_t(max_window_));
        const uint32_t* ptr = context.data() + context.size() - len;
        std::vector<uint32_t> out(max_tokens);
        int count = zipf_cache_speculate_chain_with_req(cache_, req_id, ptr, len, out.data(), max_tokens);
        out.resize(count);
        return out;
    }

    // Batch chain speculation with req_ids
    std::vector<std::vector<uint32_t>> speculate_chain_batch_with_req(
        const std::vector<uint64_t>& req_ids,
        const std::vector<std::vector<uint32_t>>& contexts,
        int max_tokens
    ) {
        size_t n = contexts.size();
        if (n == 0 || max_tokens <= 0) return std::vector<std::vector<uint32_t>>(n);
        if (n != req_ids.size()) throw std::invalid_argument("req_ids and contexts must have same length");
        std::vector<ZipfKey> keys(n);
        for (size_t i = 0; i < n; i++) {
            const auto& ctx = contexts[i];
            size_t len = std::min(ctx.size(), size_t(max_window_));
            keys[i].length = len;
            for (size_t j = 0; j < len; j++) keys[i].tokens[j] = ctx[ctx.size() - len + j];
        }
        std::vector<uint32_t> out_tokens(n * max_tokens);
        std::vector<int> out_counts(n);
        zipf_cache_speculate_chain_batch_with_req(cache_, req_ids.data(), keys.data(), out_tokens.data(), out_counts.data(), n, max_tokens);
        std::vector<std::vector<uint32_t>> results(n);
        for (size_t i = 0; i < n; i++) {
            int count = out_counts[i];
            results[i].resize(count);
            for (int j = 0; j < count; j++) results[i][j] = out_tokens[i * max_tokens + j];
        }
        return results;
    }

    // Get statistics
    py::dict stats() {
        ZipfCacheStats s = zipf_cache_stats(cache_);
        return py::dict(
            py::arg("query_count") = s.query_count,
            py::arg("shared_hit_count") = s.shared_hit_count,
            py::arg("local_hit_count") = s.local_hit_count,
            py::arg("unigram_hit_count") = s.unigram_hit_count,
            py::arg("generalized_hit_count") = s.generalized_hit_count,
            py::arg("total_hit_rate") = s.total_hit_rate,
            py::arg("shared_hit_rate") = s.shared_hit_rate,
            py::arg("local_hit_rate") = s.local_hit_rate,
            py::arg("unigram_hit_rate") = s.unigram_hit_rate,
            py::arg("generalized_hit_rate") = s.generalized_hit_rate,
            py::arg("update_count") = s.update_count,
            py::arg("shared_entries") = s.shared_entries,
            py::arg("shared_unigram") = s.shared_unigram,
            py::arg("local_hashes") = s.local_stats.total_hashes,
            py::arg("local_entries") = s.local_stats.total_entries,
            py::arg("local_unigram") = s.local_stats.total_unigram,
            py::arg("local_gen_entries") = s.local_stats.total_gen_entries,
            py::arg("local_memory_mb") = s.local_stats.memory_bytes / (1024.0 * 1024.0),
            py::arg("gen_unique_tokens") = s.gen_stats.count,
            py::arg("gen_high_freq_count") = s.gen_stats.high_freq_count,
            py::arg("gen_freq_threshold") = s.gen_stats.freq_threshold,
            py::arg("gen_memory_mb") = s.gen_stats.memory_bytes / (1024.0 * 1024.0),
            py::arg("gen_wildcard_buckets") = s.gen_stats.num_wildcard_buckets
        );
    }

    /**
     * propose_batch - 完整的 propose 逻辑下沉到 C 层
     *
     * 参数：
     *   req_hashes:        list[int]          请求 hash 列表
     *   token_ids_rows:    list[numpy.ndarray] 每个请求的 token_ids（int32 numpy array）
     *   num_tokens_list:   list[int]          每个请求的当前 token 数
     *   num_prompt_list:   list[int]          每个请求的 prompt token 数
     *   sampled_token_ids: list[list[int]]    每个请求上一轮验证通过的 token
     *   valid_mask:        list[bool]         每个请求是否参与推测
     *   max_model_len:     int                模型最大长度
     *   num_spec_tokens:   int                初始推测 token 数
     *   ema_alpha:         float              EMA 平滑系数
     *   max_spec_tokens:   int                最大推测 token 数
     *
     * 返回：list[list[int]]  每个请求的 draft token 列表
     */
    std::vector<std::vector<uint32_t>> propose_batch(
        const std::vector<uint64_t>& req_hashes,
        // token_ids_rows: 每个请求的完整 token 序列（numpy int32 array）
        const std::vector<py::array_t<int32_t>>& token_ids_rows,
        const std::vector<int>& num_tokens_list,
        const std::vector<int>& num_prompt_list,
        const std::vector<std::vector<uint32_t>>& sampled_token_ids,
        const std::vector<bool>& valid_mask,
        int max_model_len,
        int num_spec_tokens,
        float ema_alpha,
        int max_spec_tokens = 128
    ) {
        size_t n = req_hashes.size();
        std::vector<std::vector<uint32_t>> results(n);
        if (n == 0) return results;
        if (token_ids_rows.size() != n || num_tokens_list.size() != n ||
            num_prompt_list.size() != n || sampled_token_ids.size() != n ||
            valid_mask.size() != n) {
            throw std::invalid_argument(
                "All input lists must have the same length as req_hashes");
        }
        if (max_spec_tokens < num_spec_tokens) {
            throw std::invalid_argument(
                "max_spec_tokens must be >= num_spec_tokens");
        }

        // 配置 adaptive-k
        AdaptiveKConfig ak_config;
        ak_config.ema_alpha = ema_alpha;
        ak_config.num_speculative_tokens = num_spec_tokens;
        ak_config.fail_threshold = 0.3f;
        ak_config.success_threshold = 0.5f;
        ak_config.max_penalty = 3;
        ak_config.max_bonus = 3;
        ak_config.min_k = 1;
        ak_config.max_k = max_spec_tokens;

        // 预分配 draft token 缓冲区
        int max_draft = max_spec_tokens;
        std::vector<uint32_t> draft_buf(n * max_draft, 0);

        // 在 GIL 持有时提取所有 numpy 指针
        std::vector<const uint32_t*> token_ptrs(n);
        for (size_t i = 0; i < n; i++) {
            token_ptrs[i] = reinterpret_cast<const uint32_t*>(
                token_ids_rows[i].data());
        }

        // 构建请求和结果数组
        std::vector<ProposeRequest> requests(n);
        std::vector<ProposeResult> c_results(n);

        for (size_t i = 0; i < n; i++) {
            requests[i].req_hash = req_hashes[i];
            requests[i].token_ids = token_ptrs[i];
            requests[i].num_tokens = num_tokens_list[i];
            requests[i].num_prompt_tokens = num_prompt_list[i];

            if (!sampled_token_ids[i].empty()) {
                requests[i].sampled_ids = sampled_token_ids[i].data();
                requests[i].num_sampled = sampled_token_ids[i].size();
            } else {
                requests[i].sampled_ids = nullptr;
                requests[i].num_sampled = 0;
            }

            requests[i].max_model_len = max_model_len;
            requests[i].is_valid = valid_mask[i] ? 1 : 0;

            c_results[i].draft_tokens = &draft_buf[i * max_draft];
            c_results[i].num_draft_tokens = 0;
        }

        // 释放 GIL 调用 C 层
        {
            py::gil_scoped_release release;
            zipf_cache_propose_batch(
                cache_, requests.data(), c_results.data(), n, &ak_config);
        }

        // 转换结果
        for (size_t i = 0; i < n; i++) {
            int count = c_results[i].num_draft_tokens;
            if (count > 0) {
                results[i].assign(
                    c_results[i].draft_tokens,
                    c_results[i].draft_tokens + count);
            }
        }
        return results;
    }

private:
    ZipfCache* cache_ = nullptr;
    uint8_t max_window_ = 4;
};


PYBIND11_MODULE(_zipf_cache_cpp, m) {
    m.doc() = "High-performance ZipfCache with pybind11 bindings (simplified, no global hash)";

    m.def("zipf_hash", [](const std::vector<uint32_t>& tokens) -> uint64_t {
        if (tokens.empty() || tokens.size() > 8) {
            throw std::invalid_argument("tokens must have 1-8 elements");
        }
        return zipf_hash(tokens.data(), tokens.size());
    }, py::arg("tokens"), "Compute hash for n-gram tokens");

    py::class_<PyZipfCache>(m, "ZipfCache")
        .def(py::init<uint8_t, uint8_t, bool, bool, uint32_t, size_t, uint32_t, bool>(),
             py::arg("min_window") = 2,
             py::arg("max_window") = 4,
             py::arg("skip_shared") = false,
             py::arg("enable_generalized") = true,
             py::arg("gen_freq_threshold") = 8,
             py::arg("gen_capacity") = 64 * 1024,
             py::arg("gen_bitmap_vocab_max") = 256 * 1024,
             py::arg("generalized_before_shared") = true,
             "Create a new ZipfCache.\n\n"
             "Hardcoded params: shared_capacity=8M, local_capacity=40K, max_local_hashes=10K,\n"
             "unigram_capacity=128K, unigram_threshold=0.75.\n"
             "Chain length control is delegated to Python-layer adaptive-k.\n"
             "C layer uses pure hit-driven speculation: hit → continue, miss → stop.\n\n"
             "Configurable:\n"
             "- min_window/max_window: n-gram window range\n"
             "- skip_shared: skip shared hash queries/updates\n"
             "- Generalized layer parameters\n"
             "- generalized_before_shared: query priority (true=local→gen→shared, false=local→shared→gen)")
        .def("query_with_req", &PyZipfCache::query_with_req,
             py::arg("req_id"), py::arg("context"),
             "Query for next token prediction with req_id (local + shared hash)")
        .def("query_batch_with_req", &PyZipfCache::query_batch_with_req,
             py::arg("req_ids"), py::arg("contexts"),
             "Batch query with req_ids (local + shared hash)")
        .def("update_with_req", &PyZipfCache::update_with_req,
             py::arg("req_id"), py::arg("context"), py::arg("next_token"),
             "Update cache with observed token (legacy API)")
        .def("update_with_history", &PyZipfCache::update_with_history,
             py::arg("req_id"), py::arg("new_tokens"),
             py::call_guard<py::gil_scoped_release>(),
             "Update cache with new tokens using history window (GIL released)")
        .def("update_with_history_batch", &PyZipfCache::update_with_history_batch,
             py::arg("req_ids"), py::arg("token_arrays"),
             py::call_guard<py::gil_scoped_release>(),
             "Batch update with history window (GIL released)")
        .def("set_history", &PyZipfCache::set_history,
             py::arg("req_id"), py::arg("tokens"),
             "Set history window for a request")
        .def("warmup_from_prompt", &PyZipfCache::warmup_from_prompt,
             py::arg("req_id"), py::arg("prompt_tokens"),
             py::call_guard<py::gil_scoped_release>(),
             "Scan prompt and pre-fill local hash (GIL released)")
        .def("reset_context", &PyZipfCache::reset_context,
             py::arg("req_id"),
             "Reset history window")
        .def("update_batch_with_req", &PyZipfCache::update_batch_with_req,
             py::arg("req_ids"), py::arg("contexts"), py::arg("next_tokens"),
             "Batch update with req_ids")
        .def("speculate_chain_with_req", &PyZipfCache::speculate_chain_with_req,
             py::arg("req_id"), py::arg("context"), py::arg("max_tokens"),
             py::call_guard<py::gil_scoped_release>(),
             "Chain speculation with req_id (GIL released)")
        .def("speculate_chain_batch_with_req", &PyZipfCache::speculate_chain_batch_with_req,
             py::arg("req_ids"), py::arg("contexts"), py::arg("max_tokens"),
             "Batch chain speculation with req_ids")
        .def("stats", &PyZipfCache::stats,
             "Get cache statistics")
        .def("propose_batch", &PyZipfCache::propose_batch,
             py::arg("req_hashes"),
             py::arg("token_ids_rows"),
             py::arg("num_tokens_list"),
             py::arg("num_prompt_list"),
             py::arg("sampled_token_ids"),
             py::arg("valid_mask"),
             py::arg("max_model_len"),
             py::arg("num_spec_tokens"),
             py::arg("ema_alpha") = 0.3f,
             py::arg("max_spec_tokens") = 128,
             "Batch propose with full logic in C layer");
}
