// SPDX-License-Identifier: Apache-2.0
// Runs the production kernel with a synchronous CPU API shim. This checks
// actual task/address/control-flow code, not NPU API or pipeline correctness.
#include "attn_res_fwd_cpu_stubs.h"
#include "csrc/attention/attn_res_fwd/op_host/attn_res_fwd_parallel_plan.h"
#include "csrc/attention/attn_res_fwd/op_kernel/arch22/attn_res_fwd_cooperative.h"
#include "csrc/attention/attn_res_fwd/op_kernel/arch22/attn_res_fwd_resident.h"
#ifdef ATTN_RES_TEST_REGBASE
  #include "attn_res_fwd_regbase_cpu_stubs.h"
  #include "csrc/attention/attn_res_fwd/op_kernel/arch35/attn_res_fwd_cooperative.h"
#endif

#include <exception>
#include <iostream>
#include <random>
#include <thread>

namespace {
struct Case {
  uint32_t tokens, valid, hidden, cores;
  bool add, norm, save, write;
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

void Run(Case c, Path path = Path::COOPERATIVE) {
  constexpr uint64_t ubBytes = 192 * 1024;
#ifdef ATTN_RES_TEST_REGBASE
  constexpr bool regBase = true;
#else
  constexpr bool regBase = false;
#endif
  auto plan = AttnResFwd::MakeParallelPlan(c.tokens, c.valid + 1, c.hidden, c.cores, ubBytes, c.norm, true, regBase);
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
  for (auto* tensor : {&prefix, &addend, &projection, &gamma, &outputGamma})
    for (auto& x : *tensor) x = normal(generator);
  for (auto& x : projection) x = static_cast<float>(x) / std::sqrt(static_cast<float>(c.hidden));
  for (uint32_t t = 0; t < c.tokens; ++t)
    for (uint32_t n = 0; n < c.valid; ++n)
      for (uint32_t h = 0; h < c.hidden; ++h) bank[t * stride + n * c.hidden + h] = normal(generator);
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
      if (c.write) expectedBank[t * stride + c.valid * c.hidden + h] = expectedRaw[index];
    }
    auto value = [&](uint32_t n, uint32_t h) {
      return static_cast<float>(n < c.valid ? bank[t * stride + n * c.hidden + h]
                                            : expectedRaw[static_cast<size_t>(t) * c.hidden + h]);
    };
    std::vector<double> scores(c.valid + 1);
    for (uint32_t n = 0; n <= c.valid; ++n) {
      double square = 0;
      for (uint32_t h = 0; h < c.hidden; ++h) square += static_cast<double>(value(n, h)) * value(n, h);
      const double inv = 1.0 / std::sqrt(square / c.hidden + normEps);
      for (uint32_t h = 0; h < c.hidden; ++h)
        scores[n] += value(n, h) * inv * static_cast<float>(gamma[h]) * static_cast<float>(projection[h]);
    }
    const double maximum = *std::max_element(scores.begin(), scores.end());
    double denominator = 0;
    for (auto& score : scores) {
      score = std::exp(score - maximum);
      denominator += score;
    }
    for (auto& score : scores) score /= denominator;
    double square = 0;
    for (uint32_t h = 0; h < c.hidden; ++h) {
      double mixture = 0;
      for (uint32_t n = 0; n <= c.valid; ++n) mixture += scores[n] * value(n, h);
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
  // Guard both ends of the user workspace and poison all unwritten slots.
  constexpr size_t guardFloats = 8;
  std::vector<float> workspace(plan.workspaceBytes / sizeof(float) + 2 * guardFloats, NAN);
  std::vector<CpuAttnTest::Region> regions;
  for (auto* tensor : {&prefix, &addend, &projection, &gamma, &outputGamma, &bank, &raw, &output, &materialized})
    regions.push_back({tensor->data(), tensor->size() * sizeof(bfloat16_t)});
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
  for (uint32_t core = 0; core < plan.usedCores; ++core)
    workers.emplace_back([&, core] {
      CpuAttnTest::Context context{core, barrier, regions};
      CpuAttnTest::context = &context;
      try {
        AscendC::TPipe pipe;
        if (path == Path::COOPERATIVE) {
#ifdef ATTN_RES_TEST_REGBASE
          AttnResFwd::AttnResFwdCooperativeRegBase<bfloat16_t> kernel(&pipe, &tiling);
#else
          AttnResFwd::AttnResFwdCooperative<bfloat16_t> kernel(&pipe, &tiling);
#endif
          kernel.Init(params, reinterpret_cast<GM_ADDR>(workspace.data() + guardFloats));
          assert(pipe.ubBytes + AttnResFwd::COOPERATIVE_UB_ALLOWANCE <= ubBytes);
          kernel.Process();
          if (plan.rowParallel) assert(context.syncCount == 1);
        } else if (path == Path::RESIDENT) {
          AttnResFwd::AttnResFwdResident<bfloat16_t, true> kernel(&pipe, &tiling);
          kernel.Init(params);
          assert(pipe.ubBytes <= ubBytes);
          kernel.Process();
        } else {
          AttnResFwd::AttnResFwdReload<bfloat16_t, true> kernel(&pipe, &tiling);
          kernel.Init(params);
          assert(pipe.ubBytes <= ubBytes);
          kernel.Process();
        }
      } catch (...) {
        {
          std::lock_guard<std::mutex> lock(errorMutex);
          if (!error) error = std::current_exception();
        }
        barrier.Cancel();
      }
      CpuAttnTest::context = nullptr;
    });
  for (auto& worker : workers) worker.join();
  if (error) std::rethrow_exception(error);
  CheckPrecision(output, expectedOutput);
  if (c.save) CheckPrecision(materialized, expectedMixture);
  for (size_t i = 0; i < elements; ++i) {
    assert(prefix[i].bits == prefixBefore[i].bits && addend[i].bits == addendBefore[i].bits);
    if (c.add) assert(raw[i].bits == expectedRaw[i].bits);
  }
  for (size_t i = 0; i < bank.size(); ++i) assert(bank[i].bits == expectedBank[i].bits);
  for (size_t i = 0; i < guardFloats; ++i)
    assert(std::isnan(workspace[i]) && std::isnan(workspace[workspace.size() - i - 1]));
  for (size_t i = guardFloats; i < workspace.size() - guardFloats; ++i) assert(std::isfinite(workspace[i]));
}
}  // namespace

int main() {
  const Case cases[] = {
      {1, 8, 7168, 40, true, true, false, true},
      {4, 8, 7168, 40, true, true, true, true},
      {16, 8, 7168, 40, true, true, true, true},    // Four row tasks on some A2/A3 cores.
      {16, 8, 7168, 56, true, true, false, false},  // Three row tasks on some A5 cores.
      {20, 8, 513, 40, false, true, true, true},    // Inclusive token-parallelism cutoff.
      {28, 8, 513, 56, true, false, false, true},
      {4, 4, 7168, 40, false, true, false, true},
      {1, 1, 7168, 40, true, false, false, false},
      {1, 16, 7168, 40, false, true, true, false},
      {4, 8, 1537, 40, true, true, false, true},
      {2, 2, 7169, 40, true, false, true, true},
      {4, 8, 1025, 40, false, true, true, false},
      {4, 8, 7168, 40, true, false, false, false},
      {10, 4, 7168, 40, false, true, false, true},
      {10, 7, 7168, 40, true, true, true, true},  // Several tokens share one output leader.
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
  for (const auto& c : cases) Run(c);
  Run({16, 4, 1537, 40, true, true, true, true}, Path::RESIDENT);
  Run({16, 64, 1537, 40, false, true, true, true}, Path::RELOAD);
  Run({16, 4, 7168, 40, true, true, false, false}, Path::RELOAD);
  Run({1, 2048, 1, 40, true, false, true, true}, Path::RESIDENT);
  Run({1, 2048, 32, 40, true, false, true, true}, Path::RELOAD);
  std::cout << "CPU shim: " << sizeof(cases) / sizeof(cases[0]) << " cooperative kernel cases passed\n";
  std::cout << "CPU shim: 5 resident/reload regression cases passed\n";
}
