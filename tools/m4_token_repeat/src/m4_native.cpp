// CPU implementation of CONFIG_M4 at upstream c455af6c4fa15dd027b8388b70691df7c81af57e.
// Unsigned hashing deliberately wraps modulo 2^64. No fast-math: thresholds
// compare double(float(feature)) to the original JSON double constants.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "position_queue.h"

namespace py = pybind11;
constexpr int BLOCK = 64, H = 4096, RAW_KEEP = 6064;
constexpr uint64_t MASK = ~((uint64_t(1) << 24) - 1);
static uint64_t mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

// Absolute-index circular storage; capacity includes the cumulative value at
// the left edge and the largest block that can arrive before a query.
struct Ring {
  std::vector<int64_t> a;
  int64_t end = 0;
  explicit Ring(size_t cap) : a(cap, 0) {}
  void push(int64_t x) { a[end++ % a.size()] = x; }
  int64_t at(int64_t i) const {
    if (i < std::max<int64_t>(0, end - static_cast<int64_t>(a.size())) || i >= end)
      throw std::out_of_range("ring history");
    return a[i % a.size()];
  }
  void add(int64_t x) { push(at(end - 1) + x); }
  double share(int64_t n, int w) const {
    return n < w ? std::numeric_limits<double>::quiet_NaN() : double(at(n) - at(n - w)) / w;
  }
};

struct Ngrams {
  int ng;
  int64_t m = 0;
  std::array<uint64_t, 8> context{};
  int cursor = 0;
  uint64_t rolling = 0, outgoing_weight = 1;
  static constexpr uint64_t HASH_BASE = 0x100000001B3ULL;
  std::unordered_map<uint64_t, PositionQueue> positions;
  std::deque<std::pair<int64_t, uint64_t>> history;
  // A3, A5, P, N, all kept for independent feature comparison.
  std::array<Ring, 4> counts{Ring(1089), Ring(1089), Ring(1089), Ring(1089)};
  explicit Ngrams(int size) : ng(size) {
    for (int i = 1; i < ng; ++i) outgoing_weight *= HASH_BASE;
    positions.reserve(H + BLOCK);
    for (auto& r : counts) r.push(0);
  }
  void push(int32_t id, int64_t answer) {
    if (m >= (int64_t(1) << 24)) throw std::length_error("m4 position limit 2^24");
    const uint64_t incoming = mix(uint64_t(id));
    // Keep all 64 bits between updates. MASK applies to the lookup key
    // only; masking rolling itself would change future n-gram hashes.
    rolling = (rolling - context[cursor] * outgoing_weight) * HASH_BASE + incoming;
    context[cursor] = incoming;
    if (++cursor == ng) cursor = 0;
    int64_t count = 0, d1 = 0, d2 = 0;
    if (m >= ng - 1) {
      while (!history.empty() && history.front().first < m - H) {
        auto [pos, key] = history.front();
        history.pop_front();
        auto found = positions.find(key);
        if (found == positions.end() || found->second.front() != pos) throw std::logic_error("ngram eviction mismatch");
        found->second.pop_front();
        if (found->second.empty()) positions.erase(found);
      }
      const uint64_t key = rolling & MASK;
      auto& q = positions[key];
      int64_t floor = std::max<int64_t>(m - H, 0);
      count = static_cast<int64_t>(q.size());
      if (answer > floor && m >= answer) count = q.end() - std::lower_bound(q.begin(), q.end(), answer);
      if (count >= 1) d1 = m - q.back();
      if (count >= 2) d2 = m - q[q.size() - 2];
      q.push_back(m);
      history.emplace_back(m, key);
    }
    counts[0].add(count >= 3);
    counts[1].add(count >= 5);
    counts[2].add(count >= 2 && std::abs(d2 - 2 * d1) <= (d1 >> 3));
    counts[3].add(count == 0);
    ++m;
  }
};

static bool exact(const Ring& raw, int nnew, int pmin, int pmax, int copies) {
  const int64_t end = raw.end, base = std::max<int64_t>(0, end - static_cast<int64_t>(raw.a.size()));
  const int64_t b0 = end - nnew;
  int hi = std::min<int64_t>(pmax, (end - base) / copies);
  if (nnew <= 0 || hi < pmin) return false;
  for (int p = pmin; p <= hi; ++p) {
    int need = (copies - 1) * p;
    if ((copies - 1) * pmin >= nnew) {
      bool candidate = true;
      for (int k = 0; k < 6; ++k) {
        int64_t x = b0 - int64_t(need - nnew) * k / 5;
        if (x - p < base || raw.at(x) != raw.at(x - p)) {
          candidate = false;
          break;
        }
      }
      if (!candidate) continue;
    }
    int64_t lo = std::max(b0 - need + 1, base + p), run = 0;
    for (int64_t q = lo; q < end; ++q) {
      run = raw.at(q) == raw.at(q - p) ? run + 1 : 0;
      if (q >= b0 && run >= need) return true;
    }
  }
  return false;
}

struct Tables {
  std::vector<int32_t> norm, chars, ws, run_char, marker, send;
  int64_t think_end;
  // A valid int32 norm ID can be INT32_MAX. Widen before adding one;
  // the bitmap's size is a count, not another signed token ID.
  size_t nbits;
};

struct State {
  std::shared_ptr<const Tables> tables;
  std::array<int32_t, BLOCK> pending{};
  int n_pending = 0;
  int64_t t = 0, nx = 0, answer = -1, fired_at = -1;
  int reason = -1;
  bool done = false;
  bool failed = false;  // A partially computed block must never be reused.
  Ngrams n4{4}, n8{8};
  Ring raw{RAW_KEEP}, sep{321}, chars{321}, ws{321};
  Ring repeat{2113}, marker{1089}, duplicate{81};
  std::vector<uint8_t> seen;
  int32_t run_char = -1;
  int64_t run_len = 0, run_max = 0, sentences = 0;
  int sentence_len = 0;
  std::array<uint64_t, 4> signature;
  std::array<std::array<uint64_t, 4>, 512> sentence_history{};
  std::array<int, 14> streak{};
  explicit State(std::shared_ptr<const Tables> tab) : tables(std::move(tab)), seen(tables->nbits / 8 + 1, 0) {
    for (auto r : {&sep, &chars, &ws, &repeat, &marker, &duplicate}) r->push(0);
    signature.fill(UINT64_MAX);
  }
  void close_sentence() {
    if (sentence_len >= 4) {
      bool dup = false;
      for (int i = 0; i < std::min<int64_t>(sentences, 512); ++i) {
        int agree = 0;
        for (int k = 0; k < 4; ++k) agree += sentence_history[i][k] == signature[k];
        if (agree >= 3) {
          dup = true;
          break;
        }
      }
      sentence_history[sentences % 512] = signature;
      ++sentences;
      duplicate.add(dup);
    }
    sentence_len = 0;
    signature.fill(UINT64_MAX);
  }
  std::array<double, 9> features() const {
    if (failed) throw std::runtime_error("failed detector state; discard it");
    return {
        n4.counts[0].share(nx, 1024),
        n4.counts[1].share(nx, 512),
        n4.counts[3].share(nx, 512),
        n8.counts[2].share(nx, 1024),
        repeat.share(nx, 2048),
        repeat.share(nx, 1024),
        marker.share(t, 1024),
        duplicate.share(sentences, 16),
        double(sep.at(t) - sep.at(std::max<int64_t>(t - 256, 0))) / std::max<int64_t>(std::min<int64_t>(t, 256), 1)};
  }
  void block(int n) {
    // Find the answer boundary before feeding any normalized positions in
    // this block, matching the vectorized Python implementation.
    if (answer < 0 && tables->think_end >= 0) {
      int before = 0;
      for (int i = 0; i < n; ++i) {
        int32_t id = pending[i];
        if (id == tables->think_end) {
          answer = nx + before;
          break;
        }
        before += tables->norm[id] != -1;
      }
    }
    constexpr std::array<uint64_t, 4> seeds{0x51ED27, 0x9E3779, 0x7F4A7C, 0x2545F4};
    for (int i = 0; i < n; ++i) {
      const int32_t id = pending[i], norm = tables->norm[id], ch = tables->chars[id], rc = tables->run_char[id];
      if (norm != -1) {
        n4.push(norm, answer);
        n8.push(norm, answer);
        uint8_t mask = uint8_t(1) << (norm & 7);
        repeat.add((seen[norm >> 3] & mask) != 0);
        seen[norm >> 3] |= mask;
        ++nx;
        ++sentence_len;
        uint64_t h = mix(uint64_t(norm));
        for (int k = 0; k < 4; ++k) signature[k] = std::min(signature[k], h * (seeds[k] | 1) + seeds[k]);
        if (sentence_len == 256) close_sentence();
      }
      if (tables->send[id]) close_sentence();
      raw.push(id);
      sep.add(norm == -1);
      chars.add(ch);
      ws.add(tables->ws[id]);
      marker.add(tables->marker[id]);
      if (rc < 0) {
        run_char = -1;
        run_len = 0;
      } else if (rc == run_char)
        run_len += ch;
      else {
        run_char = rc;
        run_len = ch;
      }
      run_max = std::max(run_max, run_len);
    }
    t += n;
    auto f = features();
    // Preserve NumPy float32 rounding followed by Python float comparison.
    for (auto& v : f) v = double(float(v));
    std::array<bool, 14> ok{};
    // Evaluation order determines the first alert reason.
    ok[0] = exact(raw, n, 32, 500, 5);
    if (!ok[0]) ok[1] = exact(raw, n, 32, 2000, 3);
    ok[2] = run_max >= 500;
    ok[3] = f[0] >= .6 && f[2] <= .05;
    ok[4] = f[1] >= .7 && f[2] <= .1;
    ok[5] = f[1] >= .5 && f[8] <= .5 && f[2] <= .05;
    ok[6] = f[3] >= .6 && f[2] <= .02;
    ok[7] = f[4] >= .95 && f[6] >= .05;
    ok[8] = f[4] >= .99;
    ok[9] = f[5] >= .99 && f[6] >= .04;
    ok[10] = f[5] >= .98 && f[6] >= .05;
    ok[11] = f[7] >= .7 && f[4] >= .9 && f[6] >= .03;
    ok[12] = f[7] >= .6 && f[6] >= .03;
    ok[13] = f[4] >= .95 && f[6] >= .03;
    constexpr std::array<int, 14> persist{1, 1, 1, 8, 4, 4, 24, 16, 24, 8, 8, 24, 40, 48};
    for (int i = 0; i < 14; ++i) {
      streak[i] = ok[i] ? streak[i] + 1 : 0;
      if (streak[i] >= persist[i]) {
        reason = i;
        fired_at = t;
        done = true;
        break;
      }
    }
  }
  void compute_pending() {
    // block() mutates several owning containers. Rollback is not safe after
    // allocation/capacity failure. Fail closed, including during finish().
    try {
      block(n_pending);
      n_pending = 0;
    } catch (...) {
      failed = true;
      throw;
    }
  }
  void push(int64_t token) {
    // Iterators can re-enter Python/native code between public preflight
    // and push(). Never index pending after a nested block failure.
    if (failed) throw std::runtime_error("failed detector state; discard it");
    if (done) return;
    if (token < 0 || token > std::numeric_limits<int32_t>::max() || uint64_t(token) >= tables->norm.size())
      throw std::out_of_range("token id");
    pending[n_pending++] = int32_t(token);
    if (n_pending == BLOCK) compute_pending();
  }
};

struct Detector {
  std::shared_ptr<Tables> tables = std::make_shared<Tables>();
  // The Python wrapper performs range-checked conversion. Direct native
  // callers must not silently truncate float/int64/unsigned lookup tables.
  using Array = py::array;
  Detector(Array norm, Array chars, Array ws, Array rc, Array marker, Array send, int64_t think) {
    auto copy = [](const Array& x) {
      if (x.ndim() != 1) throw std::invalid_argument("lookup tables must be one-dimensional");
      if (!x.dtype().is(py::dtype::of<int32_t>()) || !(x.flags() & py::array::c_style))
        throw std::invalid_argument("lookup tables must be contiguous native int32 arrays");
      std::vector<int32_t> values(static_cast<size_t>(x.size()));
      // NumPy allows contiguous arrays with unaligned data pointers.
      // Copy bytes, not typed elements, to avoid undefined alignment reads.
      if (!values.empty()) std::memcpy(values.data(), x.data(), values.size() * sizeof(int32_t));
      return values;
    };
    tables->norm = copy(norm);
    tables->chars = copy(chars);
    tables->ws = copy(ws);
    tables->run_char = copy(rc);
    tables->marker = copy(marker);
    tables->send = copy(send);
    auto size = tables->norm.size();
    if (!size || tables->chars.size() != size || tables->ws.size() != size || tables->run_char.size() != size ||
        tables->marker.size() != size || tables->send.size() != size)
      throw std::invalid_argument("lookup table sizes");
    for (auto id : tables->norm)
      if (id < -1) throw std::invalid_argument("normalized id");
    tables->nbits = static_cast<size_t>(int64_t(*std::max_element(tables->norm.begin(), tables->norm.end())) + 1);
    tables->think_end = think;
  }
  std::shared_ptr<State> start() const { return std::make_shared<State>(tables); }
  void check(const State& s) const {
    if (s.tables.get() != tables.get()) throw std::invalid_argument("state belongs to another detector");
    if (s.failed) throw std::runtime_error("failed detector state; discard it");
  }
  void finish(State& s) const {
    check(s);
    if (!s.done && s.n_pending) s.compute_pending();
    s.n_pending = 0;
    s.done = true;
  }
  // P1: cross-request submission without cross-step buffering or reblocking.
  // The existing State::push is the only token-consumption implementation.
  py::list feed_events(const std::vector<std::shared_ptr<State>>& states, py::sequence tokens) const {
    if (states.size() != size_t(py::len(tokens))) throw std::invalid_argument("batch sizes");
    // Reject an invalid request set before mutating any valid state. A state
    // cannot have two distinct new-token slices in one decode step.
    std::unordered_set<const State*> unique;
    for (const auto& state : states) {
      if (!state) throw std::invalid_argument("state must not be None");
      check(*state);
      if (!unique.insert(state.get()).second) throw std::invalid_argument("duplicate state in feed_events");
    }
    py::list events;
    for (size_t i = 0; i < states.size(); ++i) {
      State& s = *states[i];
      const bool was_done = s.done;
      // Do not obtain an iterator for an already-done state.
      if (!s.done) {
        try {
          for (auto id : py::cast<py::iterable>(tokens[i])) {
            // Do not prevalidate or convert an unread suffix: detection
            // stops consuming immediately when this request alerts.
            s.push(py::cast<int64_t>(id));
            if (s.done) break;
          }
        } catch (...) {
          // Only this slot is unusable, including conversion/iteration failures
          // before its first token. Other slots retain trustworthy receipts.
          s.failed = true;
          throw;
        }
      }
      // An input iterator can finish a state without an alert. Terminal
      // state alone must not fabricate an event with reason == -1.
      if (!was_done && s.fired_at >= 0) events.append(py::make_tuple(i, s.fired_at, s.reason));
    }
    return events;
  }
};

PYBIND11_MODULE(_m4_native, m) {
  py::class_<State, std::shared_ptr<State>>(m, "State")
      .def_readonly("t", &State::t)
      .def_readonly("fired_at", &State::fired_at)
      .def_readonly("reason", &State::reason)
      .def_readonly("failed", &State::failed)
      .def_property_readonly("received", [](const State& s) {
        if (s.failed) throw std::runtime_error("failed detector state; accepted prefix unknown");
        return s.t + s.n_pending;
      });
  py::class_<Detector>(m, "Detector")
      .def(py::init<Detector::Array, Detector::Array, Detector::Array, Detector::Array, Detector::Array,
                    Detector::Array, int64_t>(),
           py::arg("norm").noconvert(), py::arg("chars").noconvert(), py::arg("ws").noconvert(),
           py::arg("rc").noconvert(), py::arg("marker").noconvert(), py::arg("send").noconvert(), py::arg("think"))
      .def("start", &Detector::start)
      .def("finish", &Detector::finish)
      .def("feed_events", &Detector::feed_events);
}
