#pragma once
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

// Preserve R5's sorted live sequence. One/two-occurrence keys avoid a separate
// allocation. Frequent keys retain the amortized compact vector. No count,
// eviction, or lookup semantics change.
class PositionQueue {
  std::vector<int64_t> values;
  int64_t small[2]{};
  size_t head = 0, small_end = 0;
  bool heap() const { return !values.empty(); }
  size_t limit() const { return heap() ? values.size() : small_end; }
  const int64_t* data() const { return heap() ? values.data() : small; }

 public:
  using const_iterator = const int64_t*;
  size_t size() const { return limit() - head; }
  bool empty() const { return size() == 0; }
  int64_t front() const { return data()[head]; }
  int64_t back() const { return data()[limit() - 1]; }
  int64_t operator[](size_t index) const { return data()[head + index]; }
  const_iterator begin() const { return data() + head; }
  const_iterator end() const { return data() + limit(); }
  void push_back(int64_t value) {
    if (heap()) {
      values.push_back(value);
      return;
    }
    if (head) {
      const size_t live = small_end - head;
      for (size_t i = 0; i < live; ++i) small[i] = small[head + i];
      small_end = live;
      head = 0;
    }
    if (small_end < 2) {
      small[small_end++] = value;
      return;
    }
    // Allocate before changing the inline state, preserving it on failure.
    std::vector<int64_t> next;
    next.reserve(4);
    next.push_back(small[0]);
    next.push_back(small[1]);
    next.push_back(value);
    values = std::move(next);
  }
  void pop_front() {
    ++head;
    if (head == limit()) {
      values.clear();
      head = small_end = 0;
    } else if (heap() && head >= 64 && head >= values.size() - head) {
      values.erase(values.begin(), values.begin() + head);
      head = 0;
    }
  }
};
