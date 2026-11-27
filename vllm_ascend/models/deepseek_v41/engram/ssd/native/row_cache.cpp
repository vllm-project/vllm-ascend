// row_cache.cpp — Engram SSD 行缓存目录(生产版, 移植自 phase-2 V4 目录).
//
// 相对 V4 原型(phase-2/layerwise方案/cache-native/scripts/engram_bulk_v4.cpp)的生产化:
//   1. 目录按 capacity 一次性预分配最终大小, 消除运行期 grow/rehash 峰值(04 §7);
//   2. tombstone/探测长度/维护耗时全部入账, 提供 maintain() 批次边界重建与
//      有界清理, 触发阈值可配(桶数 1/4 只是默认待测值, 不是已定阈值);
//   3. 状态机: IDLE→RESERVED(plan/acquire)→PUBLISHED(publish)→IDLE(release);
//      单在途租约; abort 删除全部 pending; 命中先于 miss 加 pin 保护;
//   4. CLOCK 淘汰保留为基线, victim 扫描上限 cap*3, 全部被保护时报错而非死循环;
//   5. 独立实现, 无实验运行时编译依赖(v3/v4 的 hash/Clock/QueryPool 内联).
//
// key 为 packed_global_id 低 32 位(真实表 768M 行 < 2^32), limit 校验全局 id.
// 返回码: 0=ok, -1=参数/范围, -2=容量不足, -3=状态机冲突, -10=内部异常(已自愈).

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double elapsed(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

struct PhaseStats {
    double validate_ms = 0, lookup_ms = 0, sort_ms = 0, assign_ms = 0;
    double wall_ms = 0, maintain_ms = 0;
    uint64_t hit_rows = 0, miss_rows = 0, unique_misses = 0, evictions = 0;
    uint64_t probe_total = 0, probe_max = 0;
    uint64_t tombstones_after = 0, directory_buckets = 0;
    int rebuilt = 0;
};

// 单线程 plan(生产 decode 批次小; 需要并行段时按 V4 模式扩展)

struct RowCache {
    static constexpr uint64_t EMPTY = UINT64_MAX, DELETED = UINT64_MAX - 1;
    struct Slot {
        uint32_t key = 0, pin = 0;
        uint8_t state = 0, recent = 0;
        uint16_t pad = 0;
    };
    static_assert(sizeof(Slot) == 12, "adjacent slot metadata");

    uint32_t cap, next = 0, hand = 0, epoch = 0;
    uint64_t limit;
    unsigned threads;
    uint64_t rebuild_threshold;   // tombstone 数超过该值时 plan 前重建
    bool active = false, published = false, poisoned = false;
    size_t mask, used = 0, pending_count = 0, tombstones = 0;
    uint64_t lease_seq = 0;
    std::vector<uint64_t> directory;
    std::vector<Slot> slots;
    std::vector<uint32_t> pending;

    RowCache(uint32_t c, uint64_t l, unsigned t, uint64_t rebuild_at)
        : cap(c), limit(l), threads(t), rebuild_threshold(rebuild_at), slots(c) {
        if (!c || c >= uint32_t(1) << 31 || !l || l > (uint64_t(1) << 32) || !t || t > 32)
            throw std::invalid_argument("capacity/range/threads");
        // 预分配最终目录: capacity*3/2 向上取 2 的幂(69.4M 槽 → 2^27 桶, 04 §7.1)
        size_t size = 8;
        while (size < size_t(c) * 3 / 2 + 1) size *= 2;
        mask = size - 1;
        directory.assign(size, EMPTY);
    }

    static uint64_t hash(uint64_t x) {
        x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27; x *= 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }
    static uint64_t entry(uint32_t key, uint32_t slot) { return (uint64_t(key) << 32) | slot; }

    uint32_t find(uint32_t key, uint64_t& probe) const {
        size_t p = hash(key) & mask;
        for (size_t j = 0; j <= mask; j++, p = (p + 1) & mask) {
            auto e = directory[p];
            probe++;
            if (e == EMPTY) return UINT32_MAX;
            if (e != DELETED && uint32_t(e >> 32) == key) return uint32_t(e);
        }
        return UINT32_MAX;
    }
    void insert_from(uint32_t key, uint32_t slot, size_t p) {
        for (size_t j = 0; j <= mask; j++, p = (p + 1) & mask) {
            auto e = directory[p];
            if (e == EMPTY) { directory[p] = entry(key, slot); used++; return; }
            if (e == DELETED) { directory[p] = entry(key, slot); tombstones--; used++; return; }
            if (uint32_t(e >> 32) == key) throw std::runtime_error("duplicate insertion");
        }
        throw std::runtime_error("directory full");
    }
    void erase(uint32_t key) {
        size_t p = hash(key) & mask;
        for (size_t j = 0; j <= mask; j++, p = (p + 1) & mask) {
            auto e = directory[p];
            if (e == EMPTY) return;
            if (e != DELETED && uint32_t(e >> 32) == key) {
                directory[p] = DELETED; used--; tombstones++;
                return;
            }
        }
    }
    // 批次边界重建: 需要一份新目录的峰值内存(04 §7 预算计入)
    void rebuild() {
        size_t size = mask + 1;
        std::vector<uint64_t> replacement(size, EMPTY);
        std::swap(replacement, directory);
        used = 0; tombstones = 0;
        for (auto e : replacement) {
            if (e == EMPTY || e == DELETED) continue;
            insert_from(uint32_t(e >> 32), uint32_t(e), hash(uint32_t(e >> 32)) & mask);
        }
    }
    uint32_t victim(PhaseStats& st) {
        for (uint64_t i = 0; i < uint64_t(cap) * 3; i++) {
            uint32_t s = hand; hand = (hand + 1) % cap;
            auto& m = slots[s];
            if (m.pin == epoch && m.state) continue;   // 本批命中保护中
            if (m.state && m.recent) { m.recent = 0; continue; }
            if (m.state) { erase(m.key); st.evictions++; }
            return s;
        }
        throw std::runtime_error("all slots protected");
    }

    int plan(const uint64_t* ids, size_t n, int32_t* out, uint64_t* miss_rows,
             uint32_t* miss_slots, PhaseStats& st) {
        if (poisoned) return -5;
        if (active) return -3;
        if (n > UINT32_MAX) return -1;
        st = PhaseStats{};
        auto start = Clock::now();
        for (size_t i = 0; i < n; i++)
            if (ids[i] >= limit) return -1;
        if (++epoch == 0) { for (auto& m : slots) m.pin = 0; epoch = 1; }

        // tombstone 治理: 超阈值时在批次边界重建(无在飞)
        auto m0 = Clock::now();
        if (tombstones > rebuild_threshold) { rebuild(); st.rebuilt = 1; }
        st.maintain_ms = elapsed(m0, Clock::now());

        // 单线程查找(生产 decode 批次小; 大批次的并行段按需再加)
        std::vector<uint32_t> misses;       // key
        std::vector<uint32_t> miss_idx;     // 请求内下标
        misses.reserve(n / 2 + 1); miss_idx.reserve(n / 2 + 1);
        uint64_t protected_hits = 0;        // 本批命中的唯一槽(增量计数)
        for (size_t i = 0; i < n; i++) {
            uint64_t probe = 0;
            uint32_t s = find(uint32_t(ids[i]), probe);
            st.probe_total += probe;
            if (probe > st.probe_max) st.probe_max = probe;
            if (s != UINT32_MAX) {
                auto& m = slots[s];
                if (m.pin != epoch) { m.pin = epoch; protected_hits++; }
                m.recent = 1;
                out[i] = int32_t(s);
                st.hit_rows++;
            } else {
                out[i] = -1;
                misses.push_back(uint32_t(ids[i]));
                miss_idx.push_back(uint32_t(i));
            }
        }
        auto b = Clock::now();
        st.lookup_ms = elapsed(start, b);
        st.miss_rows = misses.size();

        // miss 排序去重(基数排序, 11bit×3 趟)
        size_t count = misses.size();
        if (count >= 2) {
            std::vector<uint32_t> kk(count), ii(count);
            for (unsigned shift : {0u, 11u, 22u}) {
                alignas(64) std::array<size_t, 2048> histogram{};
                for (size_t i = 0; i < count; i++) histogram[(misses[i] >> shift) & 2047]++;
                size_t offset = 0;
                for (auto& v : histogram) { size_t old = v; v = offset; offset += old; }
                for (size_t i = 0; i < count; i++) {
                    size_t h = histogram[(misses[i] >> shift) & 2047]++;
                    kk[h] = misses[i]; ii[h] = miss_idx[i];
                }
                misses.swap(kk); miss_idx.swap(ii);
            }
        }
        for (size_t i = 0; i < count; i++)
            if (!i || misses[i] != misses[i - 1]) st.unique_misses++;
        auto c = Clock::now();
        st.sort_ms = elapsed(b, c);

        // 容量检查: 本批命中保护 + 唯一 miss 不得超过 capacity
        if (protected_hits + st.unique_misses > cap) return -2;

        active = true; published = false; pending_count = 0;
        lease_seq++;
        uint32_t fresh = uint32_t(std::min(uint64_t(cap - next), st.unique_misses)), first = next;
        next += fresh;
        size_t j = 0;
        for (size_t i = 0; i < count; i++) {
            if (i && misses[i] == misses[i - 1]) continue;
            uint32_t s = j < fresh ? first + uint32_t(j) : victim(st);
            slots[s] = {misses[i], epoch, 2, 1, 0};   // state=2 RESERVED
            pending.push_back(s);
            pending_count++;
            miss_rows[j] = misses[i];
            miss_slots[j] = s;
            j++;
        }
        // 恢复请求顺序: out[req_idx] = slot(重复 miss 复用同槽)
        {
            size_t u = 0; uint32_t s = 0;
            for (size_t i = 0; i < count; i++) {
                if (!i || misses[i] != misses[i - 1]) s = miss_slots[u++];
                out[miss_idx[i]] = int32_t(s);
            }
        }
        auto end = Clock::now();
        st.assign_ms = elapsed(c, end);
        st.wall_ms = elapsed(start, end);
        st.tombstones_after = tombstones;
        st.directory_buckets = mask + 1;
        return 0;
    }

    int publish() {
        if (!active || published) return -3;
        // 目录可见性切换: 完整批次成功才把新行插入目录(04 §7)
        for (size_t i = 0; i < pending_count; i++) {
            auto& m = slots[pending[i]];
            insert_from(m.key, pending[i], hash(m.key) & mask);
            m.state = 1;  // PUBLISHED
        }
        published = true;
        pending.clear();
        return 0;
    }
    int release() {
        if (!active || !published) return -3;
        active = false; published = false; pending_count = 0;
        return 0;
    }
    int abort() {
        if (!active) return -3;
        for (size_t i = 0; i < pending_count; i++) {
            auto& m = slots[pending[i]];
            erase(m.key);
            m.state = 0; m.recent = 0;
        }
        pending.clear();
        pending_count = 0; active = false; published = false;
        return 0;
    }
    void poison() { poisoned = true; }
    uint64_t lease_id() const { return lease_seq; }
    bool is_active() const { return active; }
};

}  // namespace

extern "C" {
void* rc_create(uint32_t cap, uint64_t limit, unsigned threads, uint64_t rebuild_threshold) {
    try { return new RowCache(cap, limit, threads, rebuild_threshold); }
    catch (...) { return nullptr; }
}
void rc_destroy(void* p) { delete static_cast<RowCache*>(p); }
int rc_plan(void* p, const uint64_t* ids, size_t n, int32_t* out,
            uint64_t* miss_rows, uint32_t* miss_slots, PhaseStats* st) {
    auto* c = static_cast<RowCache*>(p);
    try { return c->plan(ids, n, out, miss_rows, miss_slots, *st); }
    catch (...) { if (c->is_active()) c->abort(); return -10; }
}
int rc_publish(void* p) { return static_cast<RowCache*>(p)->publish(); }
int rc_release(void* p) { return static_cast<RowCache*>(p)->release(); }
int rc_abort(void* p) { return static_cast<RowCache*>(p)->abort(); }
void rc_poison(void* p) { static_cast<RowCache*>(p)->poison(); }
uint64_t rc_lease_id(void* p) { return static_cast<RowCache*>(p)->lease_id(); }
}
