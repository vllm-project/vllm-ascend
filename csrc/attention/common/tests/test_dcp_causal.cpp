// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "../dcp_causal.h"

#include <array>
#include <cstdint>
#include <iostream>

int main() {
  constexpr int64_t maxLength = 262144;
  uint64_t checks = 0;
  for (uint32_t ratio : {4U, 128U}) {
    for (uint32_t size : {1U, 2U, 4U, 8U}) {
      for (uint32_t interleave : {128U, 256U}) {
        std::array<uint32_t, 8> expected{};
        // Enumerate completed global groups, independently of the
        // arithmetic prefix-count implementation used by the kernels.
        for (int64_t rawEnd = -32; rawEnd <= maxLength; ++rawEnd) {
          if (rawEnd > 0 && rawEnd % ratio == 0) {
            const uint32_t owner = ((rawEnd - 1) / interleave) % size;
            ++expected[owner];
          }
          uint32_t total = 0;
          for (uint32_t rank = 0; rank < size; ++rank) {
            const auto actual = vllm_ascend::dcp::LocalCompressedLength(rawEnd, ratio, size, rank, interleave);
            if (actual != expected[rank]) {
              std::cerr << "Mismatch: end=" << rawEnd << " c=" << ratio << " D=" << size << " rank=" << rank
                        << " I=" << interleave << " actual=" << actual << " expected=" << expected[rank] << '\n';
              return 1;
            }
            total += actual;
            ++checks;
          }
          if (rawEnd >= 0 && total != rawEnd / ratio) {
            std::cerr << "A complete group was lost or assigned twice\n";
            return 1;
          }
        }
      }
    }
  }

  // The reported MTP case: the first of 8 queries ends at raw position
  // 8185 (exclusive). C4 must retain older ranks' entire 256-entry history;
  // C128 rank 7 must exclude global group 63, which completes at 8192.
  for (uint32_t rank = 0; rank < 8; ++rank) {
    const auto c4 = vllm_ascend::dcp::LocalCompressedLength(8185, 4, 8, rank, 128);
    const auto c128 = vllm_ascend::dcp::LocalCompressedLength(8185, 128, 8, rank, 128);
    if (c4 != (rank == 7 ? 254U : 256U) || c128 != (rank == 7 ? 7U : 8U)) {
      return 1;
    }
  }

  // Chunked prefill, MTP, single-query decode, and page/stripe transitions:
  // compare every local prefix against explicit global group identities.
  for (uint32_t ratio : {4U, 128U}) {
    for (int64_t length : {1024, 8191, 8192, 8193, 262144}) {
      for (int64_t queryLength : {1, 4, 8, 4096}) {
        if (queryLength > length) {
          continue;
        }
        for (int64_t q = 0; q < queryLength; ++q) {
          const int64_t rawEnd = length - queryLength + q + 1;
          for (uint32_t rank = 0; rank < 8; ++rank) {
            const uint32_t visible = vllm_ascend::dcp::LocalCompressedLength(rawEnd, ratio, 8, rank, 128);
            // First/last visible and first invisible local entries,
            // mapped back to their raw group completion positions.
            for (uint32_t local : {0U, visible == 0 ? 0U : visible - 1, visible}) {
              const uint32_t stripe = 128 / ratio;
              const int64_t global = (local / stripe * 8 + rank) * stripe + local % stripe;
              const bool legal = (global + 1) * ratio <= rawEnd;
              if (legal != (local < visible)) {
                std::cerr << "Causal prefix disagrees with global group completion\n";
                return 1;
              }
              ++checks;
            }
          }
        }
      }
    }
  }
  std::cout << checks << " DCP causal-boundary checks passed\n";
}
