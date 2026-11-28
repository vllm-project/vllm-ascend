/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include <cstdint>

#include "hc_post_pre_norm_config.h"
#include "kernel_operator.h"

#if __has_include("../../hc_pre/op_kernel/hc_pre_base.h")
  #include "../../hc_pre/op_kernel/hc_pre_base.h"
#else
  #include "../hc_pre/hc_pre_base.h"
#endif

namespace HcPostPreNorm {

using namespace AscendC;

class KernelPostMix {
 public:
  __aicore__ inline void Init(GM_ADDR attnOut, GM_ADDR residual, GM_ADDR attnPost, GM_ADDR attnComb, GM_ADDR attnPre,
                              GM_ADDR newResidual, GM_ADDR mixedInput, uint32_t numRows, uint32_t hidden,
                              uint32_t vectorCores, TPipe* pipe) {
    hidden_ = hidden;
    vectorCores_ = vectorCores;
    tilesPerRow_ = (hidden + kTileElements - 1) / kTileElements;
    totalTiles_ = static_cast<uint64_t>(numRows) * tilesPerRow_;

    const uint64_t activationElements = static_cast<uint64_t>(numRows) * hidden;
    const uint64_t residualElements = activationElements * kRoutes;
    const uint64_t mixElements = static_cast<uint64_t>(numRows) * kRoutes;

    attnOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(attnOut), activationElements);
    residualGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(residual), residualElements);
    attnPostGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(attnPost), mixElements);
    attnCombGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(attnComb), mixElements * kRoutes);
    attnPreGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(attnPre), mixElements);
    newResidualGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(newResidual), residualElements);
    mixedInputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(mixedInput), activationElements);

    pipe->InitBuffer(bf16InputBuf_, (kRoutes + 1) * kTileElements * sizeof(bfloat16_t));
    pipe->InitBuffer(bf16OutputBuf_, (kRoutes + 1) * kTileElements * sizeof(bfloat16_t));
    pipe->InitBuffer(fp32InputBuf_, (kRoutes + 1) * kTileElements * sizeof(float));
    pipe->InitBuffer(accumulatorBuf_, kRoutes * kTileElements * sizeof(float));
    pipe->InitBuffer(productsBuf_, kRoutes * kTileElements * sizeof(float));
    pipe->InitBuffer(weightsBuf_, (2 * kFp32BlockElements + kRoutes * kRoutes) * sizeof(float));
    pipe->InitBuffer(broadcastBuf_, HcPre::ONE_REPEAT_BLOCK_NUMS * HcPre::BLOCK_SIZE);
  }

  __aicore__ inline void Process() {
    for (uint64_t tile = GetBlockIdx(); tile < totalTiles_; tile += vectorCores_) {
      const uint64_t token = tile / tilesPerRow_;
      const uint32_t col = static_cast<uint32_t>(tile % tilesPerRow_) * kTileElements;
      const uint32_t remaining = hidden_ - col;
      const uint32_t count = remaining < kTileElements ? remaining : kTileElements;
      ProcessTile(token, col, count);
    }
  }

 private:
  static constexpr uint32_t kRoutes = kHcMult;
  static constexpr uint32_t kTileElements = 1024;
  static constexpr uint32_t kFp32BlockElements = HcPre::BLOCK_SIZE / sizeof(float);

  template <HardEvent event>
  __aicore__ inline void WaitForPipe() {
    const auto eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
  }

  __aicore__ inline void ProcessTile(uint64_t token, uint32_t col, uint32_t count) {
    const uint64_t activationOffset = token * hidden_ + col;
    const uint64_t residualOffset = token * kRoutes * hidden_ + col;
    const uint64_t mixOffset = token * kRoutes;
    const uint32_t rowGap = hidden_ - count;

    auto bf16Input = bf16InputBuf_.Get<bfloat16_t>();
    auto updated = bf16OutputBuf_.Get<bfloat16_t>();
    auto mixed = updated[kRoutes * count];
    auto fp32Input = fp32InputBuf_.Get<float>();
    auto accumulator = accumulatorBuf_.Get<float>();
    auto products = productsBuf_.Get<float>();
    auto pre = weightsBuf_.Get<float>();
    auto post = pre[kFp32BlockElements];
    auto comb = pre[2 * kFp32BlockElements];
    auto broadcast = broadcastBuf_.Get<float>();

    HcPre::CopyIn(attnOutGm_[activationOffset], bf16Input, 1, count);
    HcPre::CopyIn(residualGm_[residualOffset], bf16Input[count], kRoutes, count, rowGap);
    HcPre::CopyIn(attnPreGm_[mixOffset], pre, 1, kRoutes);
    HcPre::CopyIn(attnPostGm_[mixOffset], post, 1, kRoutes);
    HcPre::CopyIn(attnCombGm_[mixOffset * kRoutes], comb, 1, kRoutes * kRoutes);

    WaitForPipe<HardEvent::MTE2_V>();
    WaitForPipe<HardEvent::MTE2_S>();
    Cast(fp32Input, bf16Input, RoundMode::CAST_NONE, (kRoutes + 1) * count);
    PipeBarrier<PIPE_V>();

    for (uint32_t dst = 0; dst < kRoutes; ++dst) {
      Muls(accumulator[dst * count], fp32Input, post.GetValue(dst), count);
    }
    PipeBarrier<PIPE_V>();

    // comb is indexed by [input route, output route].
    for (uint32_t src = 0; src < kRoutes; ++src) {
      const auto source = fp32Input[(src + 1) * count];
      for (uint32_t dst = 0; dst < kRoutes; ++dst) {
        Muls(products[dst * count], source, comb.GetValue(src * kRoutes + dst), count);
      }
      PipeBarrier<PIPE_V>();
      for (uint32_t dst = 0; dst < kRoutes; ++dst) {
        const uint32_t offset = dst * count;
        Add(accumulator[offset], accumulator[offset], products[offset], count);
      }
      PipeBarrier<PIPE_V>();
    }

    // Pre consumes the same BF16 residual that is written to GM.
    Cast(updated, accumulator, RoundMode::CAST_RINT, kRoutes * count);
    PipeBarrier<PIPE_V>();
    HcPre::ProcessYHc4(mixed, updated, pre, broadcast, fp32Input, accumulator, 1, count);

    WaitForPipe<HardEvent::V_MTE3>();
    HcPre::CopyOut(updated, newResidualGm_[residualOffset], kRoutes, count, rowGap);
    HcPre::CopyOut(mixed, mixedInputGm_[activationOffset], 1, count);
    WaitForPipe<HardEvent::MTE3_MTE2>();
  }

  uint32_t hidden_ = 0;
  uint32_t vectorCores_ = 0;
  uint32_t tilesPerRow_ = 0;
  uint64_t totalTiles_ = 0;

  GlobalTensor<bfloat16_t> attnOutGm_;
  GlobalTensor<bfloat16_t> residualGm_;
  GlobalTensor<float> attnPostGm_;
  GlobalTensor<float> attnCombGm_;
  GlobalTensor<float> attnPreGm_;
  GlobalTensor<bfloat16_t> newResidualGm_;
  GlobalTensor<bfloat16_t> mixedInputGm_;

  TBuf<TPosition::VECCALC> bf16InputBuf_;
  TBuf<TPosition::VECCALC> bf16OutputBuf_;
  TBuf<TPosition::VECCALC> fp32InputBuf_;
  TBuf<TPosition::VECCALC> accumulatorBuf_;
  TBuf<TPosition::VECCALC> productsBuf_;
  TBuf<TPosition::VECCALC> weightsBuf_;
  TBuf<TPosition::VECCALC> broadcastBuf_;
};

}  // namespace HcPostPreNorm
