/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * BSD 3-Clause License.
 *
 * SolveTri Vector 部分 - AIV 核执行
 * 负责在 UB 中生成辅助矩阵 (I, -I, ZERO)，然后写到 GM（ND 格式）
 */
 #ifndef SOLVE_TRI_VECTOR_H
 #define SOLVE_TRI_VECTOR_H

 #include "kernel_operator.h"
 #include "catlass/arch/cross_core_sync.hpp"
 #include "solve_tri_common.h"

 namespace NsSolveTri {

 using namespace AscendC;

template <int MATRIX_SIZE, typename T = half>
class SolveTriVector {
     static constexpr int32_t TILE_LEN = MATRIX_SIZE * MATRIX_SIZE;
     static constexpr int32_t NUM_FRACS = MATRIX_SIZE / 16;
     static constexpr int32_t STRIP_LEN = ROWS_PER_AIV_CORE * MATRIX_SIZE;
     static constexpr int32_t NUM_AUX_CORES = NUM_FRACS * 2;
     static constexpr int32_t UB_DIAG_I_OFF = STRIP_LEN;
     static constexpr int32_t UB_DIAG_INEG_OFF = STRIP_LEN + DIAG_BLOCK_ELEMS;
     static constexpr int32_t UB_AIV_ELEMS = STRIP_LEN + 2 * DIAG_BLOCK_ELEMS;

 public:
     __aicore__ inline SolveTriVector() {}

     __aicore__ inline void Init(GM_ADDR workspace,
                                  int64_t totalTiles,
                                  int64_t matrixSize);
     __aicore__ inline void Process();

 private:
     __aicore__ inline void GenerateAuxMatrices();

     TPipe pipe_;
    GlobalTensor<T> workspaceGM_;
    TBuf<TPosition::VECCALC> ubBuf_;
    LocalTensor<T> ub_;

     int64_t totalTiles_;
     int64_t matrixSize_;

     Catlass::Arch::CrossCoreFlagWithReverse<> flagAivFinish_{SYNC_AIV_AIC_FLAG_SOLVE, SYNC_AIC_AIV_FLAG_SOLVE};
 };

template <int MATRIX_SIZE, typename T>
__aicore__ inline void SolveTriVector<MATRIX_SIZE, T>::Init(
    GM_ADDR workspace, int64_t totalTiles, int64_t matrixSize)
{
    totalTiles_ = totalTiles;
    matrixSize_ = matrixSize;

    workspaceGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace));

    pipe_.InitBuffer(ubBuf_, UB_AIV_ELEMS * sizeof(T));
    ub_ = ubBuf_.Get<T>();
}

template <int MATRIX_SIZE, typename T>
__aicore__ inline void SolveTriVector<MATRIX_SIZE, T>::Process()
 {
     int32_t subIdx = static_cast<int32_t>(GetSubBlockIdx());
     int32_t blockIdx = static_cast<int32_t>(GetBlockIdx());
     // 只让核组 0 的 AIV 子核 0 生成辅助矩阵，其他核直接参与全核同步
     if (subIdx != 0 || blockIdx != 0) {
         SyncAll<false>();
         return;
     }

     GenerateAuxMatrices();

     SyncAll<false>();
 }

template <int MATRIX_SIZE, typename T>
__aicore__ inline void SolveTriVector<MATRIX_SIZE, T>::GenerateAuxMatrices()
 {
     // 单核循环生成所有条带（block dim=1 时只有 1 个 AIV 核）
     for (int32_t stripIdx = 0; stripIdx < NUM_AUX_CORES; stripIdx++) {
         // Step 1: UB 全部清零
         Duplicate(ub_, T(0), UB_AIV_ELEMS);
         SetFlag<HardEvent::V_MTE3>(0);
         WaitFlag<HardEvent::V_MTE3>(0);
         PipeBarrier<PIPE_V>();

         // Step 2: 零条带写入 ZERO / I / -I 三个 GM slot
         DataCopyExtParams stripParams;
         stripParams.blockCount = 1;
         stripParams.blockLen = static_cast<uint32_t>(STRIP_LEN * sizeof(T));
         stripParams.srcStride = 0;
         stripParams.dstStride = 0;

         int32_t stripOff = stripIdx * STRIP_LEN;
         DataCopyPad(workspaceGM_[GM_WS_ZERO * TILE_LEN + stripOff], ub_, stripParams);
         DataCopyPad(workspaceGM_[GM_WS_I * TILE_LEN + stripOff], ub_, stripParams);
         DataCopyPad(workspaceGM_[GM_WS_INEG * TILE_LEN + stripOff], ub_, stripParams);
         SetFlag<HardEvent::MTE3_V>(0);
         WaitFlag<HardEvent::MTE3_V>(0);

         // Step 3: mask 写 8x16 对角块
         uint64_t diagMask[2] = {
             DIAG_MASK_8X16_EVEN[0],
             DIAG_MASK_8X16_EVEN[1]
         };
        Duplicate(ub_[UB_DIAG_I_OFF], T(1.0f), diagMask, 1, 1, 8);
        Duplicate(ub_[UB_DIAG_INEG_OFF], T(-1.0f), diagMask, 1, 1, 8);
         SetFlag<HardEvent::V_MTE3>(1);
         WaitFlag<HardEvent::V_MTE3>(1);

         // Step 4: 搬对角块到 GM
         int32_t rowStart = stripIdx * ROWS_PER_AIV_CORE;
         int32_t colStart = stripIdx * ROWS_PER_AIV_CORE;
         int32_t gmDiagOff = rowStart * MATRIX_SIZE + colStart;

         DataCopyExtParams diagParams;
         diagParams.blockCount = ROWS_PER_AIV_CORE;
         diagParams.blockLen = static_cast<uint32_t>(ROWS_PER_AIV_CORE * sizeof(T));
         diagParams.srcStride = 0;
         diagParams.dstStride = static_cast<uint32_t>((MATRIX_SIZE - ROWS_PER_AIV_CORE) * sizeof(T));

         DataCopyPad(workspaceGM_[GM_WS_I * TILE_LEN + gmDiagOff], ub_[UB_DIAG_I_OFF], diagParams);
         DataCopyPad(workspaceGM_[GM_WS_INEG * TILE_LEN + gmDiagOff], ub_[UB_DIAG_INEG_OFF], diagParams);
         SetFlag<HardEvent::MTE3_V>(0);
         WaitFlag<HardEvent::MTE3_V>(0);
     }
 }

 }  // namespace NsSolveTri

 #endif  // SOLVE_TRI_VECTOR_H
