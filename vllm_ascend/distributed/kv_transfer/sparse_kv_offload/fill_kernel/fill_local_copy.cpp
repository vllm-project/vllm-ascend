/*
 * FillLocalCopy: fused HBM->HBM fill for sparse KV offload.
 *
 * Copies up to maxN (source row -> resident slot) byte rows for both K and V
 * in a single kernel launch. Source/dest bases are passed as kernel arguments
 * (fixed at graph capture time, one set per layer); the per-forward dynamic
 * part is only the small index arrays below, refreshed once per forward:
 *
 *   rows  [int32 x maxN] : source row index into the activation tensor
 *   slots [int32 x maxN] : destination slot index into the resident buffer
 *   valid [int32 x maxN] : 0/1 mask; entries with 0 are strict no-ops
 *   params[u32 x 3]      : [0]=maxN [1]=kRowBytes [2]=vRowBytes (static)
 *
 * Modeled on memfabric_hybrid OffloadSparseCopyOps (Mulan PSL v2), with two
 * changes: descriptor arrays hold indices instead of absolute addresses so
 * all 79 layers can share one H2D upload per forward, and a strided loop so
 * small entry counts still spread across AIVs.
 */

#include "kernel_operator.h"

namespace {
constexpr int64_t UB_ONCE_SIZE = 176 * 1024;

class FillLocalCopyKernel {
 public:
  __aicore__ inline void Init(GM_ADDR rows, GM_ADDR slots, GM_ADDR valid, GM_ADDR params, GM_ADDR kSrc, GM_ADDR kDst,
                              GM_ADDR vSrc, GM_ADDR vDst) {
    aivNum_ = AscendC::GetBlockNum();
    aivIdx_ = AscendC::GetBlockIdx();

    __gm__ uint32_t* p = reinterpret_cast<__gm__ uint32_t*>(params);
    maxN_ = p[0];
    kRowBytes_ = p[1];
    vRowBytes_ = p[2];

    rows_ = reinterpret_cast<__gm__ int32_t*>(rows);
    slots_ = reinterpret_cast<__gm__ int32_t*>(slots);
    valid_ = reinterpret_cast<__gm__ int32_t*>(valid);
    kSrc_ = reinterpret_cast<__gm__ uint8_t*>(kSrc);
    kDst_ = reinterpret_cast<__gm__ uint8_t*>(kDst);
    vSrc_ = reinterpret_cast<__gm__ uint8_t*>(vSrc);
    vDst_ = reinterpret_cast<__gm__ uint8_t*>(vDst);

    pipe_.InitBuffer(bindQueue_, 1, UB_ONCE_SIZE);
  }

  __aicore__ inline void Process() {
    for (uint32_t i = aivIdx_; i < maxN_; i += aivNum_) {
      if (valid_[i] == 0) {
        continue;
      }
      uint64_t row = static_cast<uint64_t>(static_cast<uint32_t>(rows_[i]));
      uint64_t slot = static_cast<uint64_t>(static_cast<uint32_t>(slots_[i]));
      CopyRow(kDst_ + slot * kRowBytes_, kSrc_ + row * kRowBytes_, kRowBytes_);
      CopyRow(vDst_ + slot * vRowBytes_, vSrc_ + row * vRowBytes_, vRowBytes_);
    }
  }

 private:
  __aicore__ inline void CopyRow(__gm__ uint8_t* dst, __gm__ uint8_t* src, uint32_t len) {
    inputGm_.SetGlobalBuffer(src, len);
    outputGm_.SetGlobalBuffer(dst, len);

    uint32_t leftLen = len;
    uint32_t times = 0;
    uint32_t preCopyNum = UB_ONCE_SIZE;  // uint8: elements == bytes
    AscendC::DataCopyPadExtParams<uint8_t> padParams;

    while (leftLen > 0) {
      uint32_t curCopySize = (leftLen > UB_ONCE_SIZE) ? UB_ONCE_SIZE : leftLen;
      AscendC::LocalTensor<uint8_t> local = bindQueue_.AllocTensor<uint8_t>();
      AscendC::DataCopyExtParams dataCopyParams(1, curCopySize, 0, 0, 0);
      AscendC::DataCopyPad(local, inputGm_[times * preCopyNum], dataCopyParams, padParams);
      bindQueue_.EnQue(local);
      local = bindQueue_.DeQue<uint8_t>();
      AscendC::DataCopyPad(outputGm_[times * preCopyNum], local, dataCopyParams);
      bindQueue_.FreeTensor(local);
      leftLen = (leftLen > UB_ONCE_SIZE) ? leftLen - UB_ONCE_SIZE : 0;
      times++;
    };
  }

 private:
  AscendC::TPipe pipe_;
  AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, 1> bindQueue_;
  AscendC::GlobalTensor<uint8_t> inputGm_;
  AscendC::GlobalTensor<uint8_t> outputGm_;
  uint32_t aivNum_;
  uint32_t aivIdx_;
  uint32_t maxN_;
  uint32_t kRowBytes_;
  uint32_t vRowBytes_;
  __gm__ int32_t* rows_;
  __gm__ int32_t* slots_;
  __gm__ int32_t* valid_;
  __gm__ uint8_t* kSrc_;
  __gm__ uint8_t* kDst_;
  __gm__ uint8_t* vSrc_;
  __gm__ uint8_t* vDst_;
};
}  // namespace

extern "C" __global__ __aicore__ void FillLocalCopyOps(GM_ADDR rows, GM_ADDR slots, GM_ADDR valid, GM_ADDR params,
                                                       GM_ADDR kSrc, GM_ADDR kDst, GM_ADDR vSrc, GM_ADDR vDst) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  FillLocalCopyKernel op;
  op.Init(rows, slots, valid, params, kSrc, kDst, vSrc, vDst);
  op.Process();
}

extern "C" void FillLocalCopy(uint64_t rowsDev, uint64_t slotsDev, uint64_t validDev, uint64_t paramsDev,
                              uint64_t kSrcBase, uint64_t kDstBase, uint64_t vSrcBase, uint64_t vDstBase,
                              void* stream) {
  constexpr uint32_t blockDim = 32;
  FillLocalCopyOps<<<blockDim, nullptr, stream>>>(
      reinterpret_cast<GM_ADDR>(rowsDev), reinterpret_cast<GM_ADDR>(slotsDev), reinterpret_cast<GM_ADDR>(validDev),
      reinterpret_cast<GM_ADDR>(paramsDev), reinterpret_cast<GM_ADDR>(kSrcBase), reinterpret_cast<GM_ADDR>(kDstBase),
      reinterpret_cast<GM_ADDR>(vSrcBase), reinterpret_cast<GM_ADDR>(vDstBase));
}
