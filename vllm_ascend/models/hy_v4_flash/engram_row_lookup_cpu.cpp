// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cstdint>
#include <cstring>

// Tables remain owned by PyTorch. Copy native BF16 bits without conversion.
extern "C" __attribute__((visibility("default"))) int hy4_engram_rows(
    const int32_t* hashes, const uint16_t* const* weights, const int64_t* metadata, uint16_t* output, int64_t layers,
    int64_t tokens, int64_t total_heads, int64_t local_heads, int64_t dim) {
  if (!hashes || !weights || !metadata || !output || layers != 4 || tokens < 0 || total_heads <= 0 ||
      local_heads <= 0 || dim <= 0)
    return -1;
  for (int64_t l = 0; l < layers; ++l) {
    const auto* m = metadata + l * 4;
    // head_start, head_end, vocab_start, vocab_end are shard-specific.
    if (!weights[l] || m[0] < 0 || m[1] > total_heads || m[1] - m[0] != local_heads || m[3] <= m[2]) return -2;
  }
  for (int64_t l = 0; l < layers; ++l) {
    const auto* m = metadata + l * 4;
    for (int64_t t = 0; t < tokens; ++t) {
      const auto* h = hashes + (l * tokens + t) * total_heads + m[0];
      auto* dst = output + (l * tokens + t) * local_heads * dim;
      for (int64_t j = 0; j < local_heads; ++j) {
        const int64_t index = std::clamp(int64_t(h[j]) - m[2], int64_t(0), m[3] - m[2] - 1);
        std::memcpy(dst + j * dim, weights[l] + index * dim, size_t(dim) * sizeof(uint16_t));
      }
    }
  }
  return 0;
}
