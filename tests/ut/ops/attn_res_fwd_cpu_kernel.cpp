// SPDX-License-Identifier: Apache-2.0
// Runs the production kernel with a synchronous CPU API shim. This checks
// actual task/address/control-flow code, not NPU API or pipeline correctness.
#include "attn_res_fwd_cpu_stubs.h"
#include "csrc/attention/attn_res_fwd/op_host/attn_res_fwd_parallel_plan.h"
#include "csrc/attention/attn_res_fwd/op_kernel/arch22/attn_res_fwd_cooperative.h"
#include "csrc/attention/attn_res_fwd/op_kernel/arch22/attn_res_fwd_token_group.h"
#include "csrc/attention/attn_res_fwd/op_kernel/arch22/attn_res_fwd_resident.h"
#include "csrc/attention/attn_res_fwd/op_host/attn_res_fwd_legacy_plan.h"
#ifdef ATTN_RES_TEST_REGBASE
  #include "attn_res_fwd_regbase_cpu_stubs.h"
  #include "csrc/attention/attn_res_fwd/op_kernel/arch35/attn_res_fwd_cooperative.h"
  #include "csrc/attention/attn_res_fwd/op_kernel/arch35/attn_res_fwd_token_group.h"
#endif

#include <exception>
#include <iostream>
#include <random>
#include <thread>

namespace {
struct Case {
  uint32_t tokens;
  uint32_t valid;
  uint32_t hidden;
  uint32_t cores;
  bool add;
  bool norm;
  bool save;
  bool write;
};
enum class Path { COOPERATIVE, RESIDENT, RELOAD };

void CheckPrecision(const std::vector<bfloat16_t>& actual, const std::vector<bfloat16_t>& expected) {
  size_t passed = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    const float error = std::abs(static_cast<float>(actual[i]) - static_cast<float>(expected[i]));
    assert(std::isfinite(static_cast<float>(actual[i])) && error <= 1.0f);
    passed += error <= (1.0f + std::abs(static_cast<float>(expected[i]))) / 64.0f;
  }
  assert(passed >= (actual.size() * 99 + 99) / 100);
}

void Run(Case c, Path path = Path::COOPERATIVE, uint64_t ubBytes = 192 * 1024) {
#ifdef ATTN_RES_TEST_REGBASE
  constexpr bool regBase = true;
#else
  constexpr bool regBase = false;
#endif
  auto plan = AttnResFwd::MakeParallelPlan(c.tokens, c.valid + 1, c.hidden, c.cores, ubBytes, c.norm, true, regBase);
  // Cooperative cases must use the real Host plan without a test-only override.
  if (path != Path::COOPERATIVE) {
    plan = {};
    plan.usedCores = std::min(c.tokens, c.cores);
  }
  assert(plan.usedCores != 0);
  const size_t elements = static_cast<size_t>(c.tokens) * c.hidden;
  const size_t stride = static_cast<size_t>(c.valid + 3) * c.hidden;  // Padded token stride.
  std::vector<bfloat16_t> prefix(elements), addend(elements), projection(c.hidden), gamma(c.hidden),
      outputGamma(c.hidden), bank(c.tokens * stride, bfloat16_t(NAN)), raw(elements, bfloat16_t(NAN)),
      output(elements, bfloat16_t(NAN)), materialized(elements, bfloat16_t(NAN));
  std::mt19937 generator(42 + c.valid + c.tokens);
  std::normal_distribution<float> normal;
  for (auto* tensor : {&prefix, &addend, &projection, &gamma, &outputGamma}) {
    for (auto& x : *tensor) {
      x = normal(generator);
    }
  }
  for (auto& x : projection) {
    x = static_cast<float>(x) / std::sqrt(static_cast<float>(c.hidden));
  }
  for (uint32_t t = 0; t < c.tokens; ++t) {
    for (uint32_t n = 0; n < c.valid; ++n) {
      for (uint32_t h = 0; h < c.hidden; ++h) {
        bank[t * stride + n * c.hidden + h] = normal(generator);
      }
    }
  }
  const auto prefixBefore = prefix;
  const auto addendBefore = addend;
  auto expectedBank = bank;
  std::vector<bfloat16_t> expectedRaw(elements), expectedMixture(elements), expectedOutput(elements);
  constexpr float normEps = 1e-5f, outputEps = 1e-4f;

  // Independent row-wise reference with both required BF16 boundaries.
  for (uint32_t t = 0; t < c.tokens; ++t) {
    for (uint32_t h = 0; h < c.hidden; ++h) {
      const size_t index = static_cast<size_t>(t) * c.hidden + h;
      expectedRaw[index] =
          c.add ? bfloat16_t(static_cast<float>(prefix[index]) + static_cast<float>(addend[index])) : prefix[index];
      if (c.write) {
        expectedBank[t * stride + c.valid * c.hidden + h] = expectedRaw[index];
      }
    }
    auto value = [&](uint32_t n, uint32_t h) {
      return static_cast<float>(n < c.valid ? bank[t * stride + n * c.hidden + h]
                                            : expectedRaw[static_cast<size_t>(t) * c.hidden + h]);
    };
    std::vector<double> scores(c.valid + 1);
    for (uint32_t n = 0; n <= c.valid; ++n) {
      double square = 0;
      for (uint32_t h = 0; h < c.hidden; ++h) {
        square += static_cast<double>(value(n, h)) * value(n, h);
      }
      const double inv = 1.0 / std::sqrt(square / c.hidden + normEps);
      for (uint32_t h = 0; h < c.hidden; ++h) {
        scores[n] += value(n, h) * inv * static_cast<float>(gamma[h]) * static_cast<float>(projection[h]);
      }
    }
    const double maximum = *std::max_element(scores.begin(), scores.end());
    double denominator = 0;
    for (auto& score : scores) {
      score = std::exp(score - maximum);
      denominator += score;
    }
    for (auto& score : scores) {
      score /= denominator;
    }
    double square = 0;
    for (uint32_t h = 0; h < c.hidden; ++h) {
      double mixture = 0;
      for (uint32_t n = 0; n <= c.valid; ++n) {
        mixture += scores[n] * value(n, h);
      }
      const size_t index = static_cast<size_t>(t) * c.hidden + h;
      expectedMixture[index] = static_cast<float>(mixture);
      const float rounded = expectedMixture[index];
      square += static_cast<double>(rounded) * rounded;
    }
    const double inv = 1.0 / std::sqrt(square / c.hidden + outputEps);
    for (uint32_t h = 0; h < c.hidden; ++h) {
      const size_t index = static_cast<size_t>(t) * c.hidden + h;
      expectedOutput[index] =
          c.norm ? bfloat16_t(static_cast<float>(expectedMixture[index]) * inv * static_cast<float>(outputGamma[h]))
                 : expectedMixture[index];
    }
  }

  AttnResFwd::AttnResFwdTilingData tiling{};
  tiling.numTokens = c.tokens;
  tiling.numBlocks = c.valid;
  tiling.hiddenSize = c.hidden;
  tiling.usedCoreNum = plan.usedCores;
  tiling.tokensPerCore = (c.tokens + plan.usedCores - 1) / plan.usedCores;
  tiling.blockCount = c.valid + 1;
  tiling.normEps = normEps;
  tiling.invHiddenSize = 1.0f / c.hidden;
  tiling.blockTokenStride = stride;
  tiling.blockWriteIdx = c.write ? static_cast<int32_t>(c.valid) : -1;
  tiling.fusedChain = 1;
  tiling.fusedAdd = c.add;
  tiling.mix = 1;
  tiling.saveMaterialized = c.save;
  tiling.outputNormEps = c.norm ? outputEps : 0.0f;
  tiling.outputSplits = plan.outputSplits;
  tiling.scoreSplits = plan.scoreSplits;
  tiling.cooperativeTileElems = plan.tileElems;
  tiling.residentGroup = plan.residentGroup;
  tiling.tokenGroupSize = plan.tokenGroupSize;
  tiling.cachedRows = plan.cachedRows;
  // Guard both ends of the user workspace and poison all unwritten slots.
  constexpr size_t guardFloats = 8;
  std::vector<float> workspace(plan.workspaceBytes / sizeof(float) + 2 * guardFloats, NAN);
  std::vector<CpuAttnTest::Region> regions;
  for (auto* tensor : {&prefix, &addend, &projection, &gamma, &outputGamma, &bank, &raw, &output, &materialized}) {
    regions.push_back({tensor->data(), tensor->size() * sizeof(bfloat16_t)});
  }
  regions.push_back({workspace.data() + guardFloats, static_cast<size_t>(plan.workspaceBytes)});
  auto address = [](auto& tensor) { return reinterpret_cast<GM_ADDR>(tensor.data()); };
  AttnResFwd::AttnResFwdInitParams params{address(prefix),
                                          address(bank),
                                          address(projection),
                                          address(gamma),
                                          address(output),
                                          nullptr,
                                          nullptr,
                                          c.add ? address(addend) : nullptr,
                                          c.add ? address(raw) : address(prefix),
                                          c.norm ? address(outputGamma) : nullptr,
                                          c.save ? address(materialized) : address(output)};
  CpuAttnTest::Barrier barrier(plan.usedCores);
  std::mutex errorMutex;
  std::exception_ptr error;
  std::vector<std::thread> workers;
  std::vector<uint32_t> outputWrites(plan.usedCores, 0);
  const size_t scoreSlots = static_cast<size_t>(c.tokens) * (c.valid + 1);
  std::vector<uint32_t> scoreWriteCounts(plan.usedCores * scoreSlots, 0);
  for (uint32_t core = 0; core < plan.usedCores; ++core) {
    workers.emplace_back([&, core] {
      CpuAttnTest::Context context{core, barrier, regions};
      context.bankBegin = reinterpret_cast<uintptr_t>(bank.data());
      context.bankEnd = context.bankBegin + bank.size() * sizeof(bfloat16_t);
      context.outputBegin = reinterpret_cast<uintptr_t>(output.data());
      context.outputEnd = context.outputBegin + output.size() * sizeof(bfloat16_t);
      if (path == Path::COOPERATIVE && plan.rowParallel && plan.tokenGroupSize != 1) {
        context.scoreBegin = reinterpret_cast<uintptr_t>(workspace.data() + guardFloats);
        context.scoreEnd = context.scoreBegin + scoreSlots * 32;
        context.scoreWriteCounts = scoreWriteCounts.data() + core * scoreSlots;
      }
      CpuAttnTest::context = &context;
      try {
        AscendC::TPipe pipe;
        if (path == Path::COOPERATIVE) {
          const auto execute = [&](auto& kernel) {
            kernel.Init(params, reinterpret_cast<GM_ADDR>(workspace.data() + guardFloats));
            assert(pipe.ubBytes + AttnResFwd::COOPERATIVE_UB_ALLOWANCE <= ubBytes);
            kernel.Process();
          };
          if (plan.tokenGroupSize != 0) {
#ifdef ATTN_RES_TEST_REGBASE
            AttnResFwd::AttnResFwdTokenGroupRegBase<bfloat16_t> kernel(&pipe, &tiling);
#else
            AttnResFwd::AttnResFwdTokenGroup<bfloat16_t> kernel(&pipe, &tiling);
#endif
            execute(kernel);
          } else {
#ifdef ATTN_RES_TEST_REGBASE
            AttnResFwd::AttnResFwdCooperativeRegBase<bfloat16_t> kernel(&pipe, &tiling);
#else
            AttnResFwd::AttnResFwdCooperative<bfloat16_t> kernel(&pipe, &tiling);
#endif
            execute(kernel);
          }
          if (plan.rowParallel) {
            assert(context.syncCount == (plan.tokenGroupSize == 1 ? 0U : 1U));
            // Every complete output row must have a distinct owning core.
            assert(context.outputWrites <= 1);
            if (plan.tokenGroupSize != 0) {
              const uint32_t lane = core % plan.tokenGroupSize;
              const uint32_t owned = (c.valid + plan.tokenGroupSize - lane) / plan.tokenGroupSize;
              const uint32_t expectedReads = lane == 0 ? owned - 1 + c.valid - plan.cachedRows : owned;
              assert(context.bankReads == expectedReads);
            }
          }
        } else if (path == Path::RESIDENT) {
          AttnResFwd::AttnResFwdResident<bfloat16_t, true> kernel(&pipe, &tiling);
          kernel.Init(params);
          assert(pipe.ubBytes <= ubBytes);
          const AttnResFwd::LegacyUbConfig budget{path == Path::RESIDENT, true, false, false, false};
          assert(AttnResFwd::LegacyUbBytes(c.hidden, c.valid + 1, budget) ==
                 pipe.ubBytes + AttnResFwd::LEGACY_UB_RESERVE_BYTES);
          kernel.Process();
        } else {
          AttnResFwd::AttnResFwdReload<bfloat16_t, true> kernel(&pipe, &tiling);
          kernel.Init(params);
          assert(pipe.ubBytes <= ubBytes);
          const AttnResFwd::LegacyUbConfig budget{path == Path::RESIDENT, true, false, false, false};
          assert(AttnResFwd::LegacyUbBytes(c.hidden, c.valid + 1, budget) ==
                 pipe.ubBytes + AttnResFwd::LEGACY_UB_RESERVE_BYTES);
          kernel.Process();
        }
      } catch (...) {
        {
          std::lock_guard<std::mutex> lock(errorMutex);
          if (!error) error = std::current_exception();
        }
        barrier.Cancel();
      }
      outputWrites[core] = context.outputWrites;
      CpuAttnTest::context = nullptr;
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  if (error) {
    std::rethrow_exception(error);
  }
  if (path == Path::COOPERATIVE && plan.rowParallel) {
    for (size_t slot = 0; slot < scoreSlots; ++slot) {
      uint32_t writes = 0;
      for (uint32_t core = 0; core < plan.usedCores; ++core) {
        writes += scoreWriteCounts[core * scoreSlots + slot];
      }
      const uint32_t residual = slot % (c.valid + 1);
      const bool local = plan.tokenGroupSize != 0 && (residual == c.valid || (residual + 1) % plan.tokenGroupSize == 0);
      assert(writes == (local ? 0U : 1U));
    }
  }
  if (plan.rowParallel) {
    uint32_t totalWrites = 0;
    for (auto count : outputWrites) {
      totalWrites += count;
    }
    assert(totalWrites == c.tokens);
  }
  CheckPrecision(output, expectedOutput);
  if (c.save) {
    CheckPrecision(materialized, expectedMixture);
  }
  for (size_t i = 0; i < elements; ++i) {
    assert(prefix[i].bits == prefixBefore[i].bits && addend[i].bits == addendBefore[i].bits);
    if (c.add) {
      assert(raw[i].bits == expectedRaw[i].bits);
    }
  }
  for (size_t i = 0; i < bank.size(); ++i) {
    assert(bank[i].bits == expectedBank[i].bits);
  }
  for (size_t i = 0; i < guardFloats; ++i) {
    assert(std::isnan(workspace[i]) && std::isnan(workspace[workspace.size() - i - 1]));
  }
  for (size_t i = guardFloats; i < workspace.size() - guardFloats; ++i) {
    const uint32_t residual = ((i - guardFloats) / 8) % (c.valid + 1);
    const bool local = plan.rowParallel && plan.tokenGroupSize != 0 &&
                       (residual == c.valid || (residual + 1) % plan.tokenGroupSize == 0);
    if (local) {
      assert(std::isnan(workspace[i]));
    } else {
      assert(std::isfinite(workspace[i]));
    }
  }
}
}  // namespace

int main() {
  const Case cases[] = {
      {12, 4, 4096, 40, true, true, true, true},
      {16, 5, 8193, 56, false, true, true, false},
      {20, 3, 1537, 40, true, true, false, true},
      {8, 5, 4096, 56, false, true, true, true},
      {16, 3, 7168, 40, true, true, true, true},
      {16, 3, 7168, 40, false, true, false, false},
      {16, 6, 1535, 56, true, true, true, true},
      {16, 6, 1536, 56, false, true, false, false},
      {16, 6, 1537, 56, true, true, true, true},
      {16, 6, 1600, 56, true, true, false, true},
      {16, 6, 1601, 56, false, true, true, false},
      {1, 8, 7168, 40, true, true, false, true},
      {4, 8, 7168, 40, true, true, true, true},
      {16, 1, 7168, 40, true, true, true, true},
      {16, 2, 7168, 40, true, true, false, false},
      {16, 2, 1537, 40, true, true, true, true},
      {8, 5, 1537, 40, true, true, true, true},
      {16, 2, 7168, 56, false, true, true, true},
      {16, 3, 1537, 56, true, false, true, false},
      {16, 7, 7168, 40, false, true, true, true},
      {16, 4, 7168, 40, true, true, false, true},
      {16, 8, 7168, 40, true, true, true, true},    // Four row tasks on some A2/A3 cores.
      {16, 8, 7168, 56, true, true, false, false},  // Three row tasks on some A5 cores.
      {16, 3, 7168, 56, true, true, false, false},  // R=4 previously reused output owners.
      {16, 6, 7168, 56, true, true, true, true},
      {16, 7, 7168, 56, false, true, true, false},
      {8, 7, 7168, 56, true, true, false, true},
      {16, 4, 7168, 40, true, true, true, true},
      {20, 8, 513, 40, false, true, true, true},  // Inclusive token-parallelism cutoff.
      {28, 8, 513, 56, true, false, false, true},
      {4, 4, 7168, 40, false, true, false, true},
      {1, 1, 7168, 40, true, false, false, false},
      {1, 16, 7168, 40, false, true, true, false},
      {4, 8, 1537, 40, true, true, false, true},
      {2, 2, 7169, 40, true, false, true, true},
      {4, 8, 1025, 40, false, true, true, false},
      {4, 8, 7168, 40, true, false, false, false},
      {10, 4, 7168, 40, false, true, false, true},
      {10, 7, 7168, 40, true, true, true, true},  // Wrapped score tasks still need distinct output owners.
      {4, 1, 16384, 40, true, true, true, true},  // H slices fit the UB budget.
      {1, 64, 1024, 40, true, true, true, true},
      {1, 8, 512, 40, false, false, true, false},
      {1, 2, 7169, 40, true, true, true, true},
      {1, 3, 8193, 40, false, true, true, true},
      {1, 8, 32768, 40, true, true, true, true},
      {4, 8, 32768, 40, true, true, true, true},  // More score slices than cores.
      {4, 8, 32769, 40, false, false, false, true},
      {1, 2, 7168, 8, true, true, true, true},  // Multiple rows per scoring core.
      {1, 4, 7168, 8, false, false, false, true},
      {1, 1, 7168, 48, true, true, true, false},
      {1, 8, 1538, 64, true, true, true, true},  // Even H without an aligned half-row split.
      {1, 63, 1024, 64, true, true, true, true},
      {1, 127, 1024, 64, true, true, true, true},
  };
  for (const auto& c : cases) {
    Run(c);
  }
  Run({16, 4, 1537, 40, true, true, true, true}, Path::RESIDENT);
  Run({16, 64, 1537, 40, false, true, true, true}, Path::RELOAD);
  Run({16, 4, 7168, 40, true, true, false, false}, Path::RELOAD);
  Run({1, 2048, 1, 40, true, false, true, true}, Path::RESIDENT);
  Run({1, 2048, 32, 40, true, false, true, true}, Path::RELOAD);
  Run({16, 0, 7168, 40, true, true, true, true}, Path::RESIDENT);
  Run({16, 2, 7168, 40, true, true, true, true}, Path::RESIDENT);
  Run({16, 5, 7168, 56, false, true, true, true}, Path::RESIDENT, 248 * 1024);
  Run({16, 6, 7169, 56, true, true, true, true}, Path::RESIDENT, 248 * 1024);
  std::cout << "CPU shim: " << sizeof(cases) / sizeof(cases[0]) << " cooperative kernel cases passed\n";
  std::cout << "CPU shim: 9 resident/reload regression cases passed\n";
}
