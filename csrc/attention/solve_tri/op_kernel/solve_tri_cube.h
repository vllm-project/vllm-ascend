/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * BSD 3-Clause License.
 *
 * SolveTri Cube 部分 - AIC 核执行
 * 负责从 GM 读取辅助矩阵，执行 CUBE 矩阵乘法
 */
 #ifndef SOLVE_TRI_CUBE_H
 #define SOLVE_TRI_CUBE_H

 #include "kernel_operator.h"
 #include "lib/matmul_intf.h"
 #include "catlass/arch/cross_core_sync.hpp"
 #include <type_traits>

// 内联 l0c_to_gm：L0C -> GM (NZ→ND, FP32→FP16/BF16)
namespace NsSolveTri {

template <typename T>
__aicore__ inline void L0CToGM(AscendC::GlobalTensor<T> gmTensor,
                               AscendC::LocalTensor<float> l0cTensor,
                               uint32_t mTileActual,
                               uint32_t nTileActual,
                               uint32_t srcStride,
                               uint32_t dstStride)
{
    auto intriParams = AscendC::FixpipeParamsV220(nTileActual, // nSize
                                                  mTileActual, // mSize
                                                  srcStride,   // srcStride
                                                  dstStride,   // dstStride
                                                  false);      // enRelu
    if constexpr (std::is_same_v<T, half>) {
        intriParams.quantPre = QuantMode_t::F322F16;
    } else {
        intriParams.quantPre = QuantMode_t::F322BF16;
    }
    AscendC::Fixpipe<T, float, AscendC::CFG_ROW_MAJOR>(gmTensor, l0cTensor, intriParams);
}

}  // namespace NsSolveTri


 #include "solve_tri_common.h"

 namespace NsSolveTri {

 using namespace AscendC;

 constexpr int32_t FRAC = 16;
 constexpr int32_t FRAC_LEN = FRAC * FRAC;

template <int MATRIX_SIZE, typename T = half>
class SolveTriCube {
     static constexpr int32_t TILE_LEN = MATRIX_SIZE * MATRIX_SIZE;
     static constexpr int32_t NUM_FRACS = MATRIX_SIZE / FRAC;
     static constexpr int32_t L1_SLOT_ELEMS = TILE_LEN;

    static constexpr int32_t SLOT_INEG = 0;
    static constexpr int32_t SLOT_I = 1;
    static constexpr int32_t SLOT_MNEG = 2;
    static constexpr int32_t SLOT_X = 3;
    static constexpr int32_t SLOT_Y = 4;
    static constexpr int32_t SLOT_INPUT = 5;
    static constexpr int32_t SLOT_ZERO = 6;  // 全零矩阵，用于 MBH 优化
    static constexpr int32_t L1_SLOT_COUNT = 7;
    static constexpr int32_t L1_TOTAL_ELEMS = L1_SLOT_COUNT * L1_SLOT_ELEMS;

    // Event IDs for pipeline sync
    static constexpr int32_t EVT_MTE2_MTE1 = 0;
    static constexpr int32_t EVT_MTE1_M = 0;
    static constexpr int32_t EVT_M_MTE1 = 0;
    static constexpr int32_t EVT_M_FIX = 0;
    static constexpr int32_t EVT_FIX_MTE2 = 0;
    static constexpr int32_t EVT_FIX_M = 0;   // FIX -> M 同步
    static constexpr int32_t EVT_MBH_LOAD = 1;  // MBH 专用：GM→L1 搬运同步（避免与 EVT_MTE2_MTE1 冲突）
    static constexpr int32_t EVT_MTE2_FIX = 1;  // MTE2 -> FIX 同步（避免与 MTE2_MTE1 event 0 冲突）
    static constexpr int32_t EVT_MTE3_MTE2 = 0;
    static constexpr int32_t EVT_MTE2_MTE3 = 0;

    // Double-buffering event IDs for MCH (X flow = 0, Y flow = 1)
    static constexpr int32_t EVT_X = 0;
    static constexpr int32_t EVT_Y = 1;

 public:
     __aicore__ inline SolveTriCube() {}
     __aicore__ inline void Init(GM_ADDR x, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
                                 GM_ADDR x_out, GM_ADDR workspace,
                                 const SolveTriTilingData* tilingData);
     __aicore__ inline void Process();

private:
   __aicore__ inline void ProcessOneTile(int64_t tileIdx);
   __aicore__ inline int64_t GetTileGMOffset(int64_t tileIdx);
   __aicore__ inline int64_t GetTileValidSize(int64_t tileIdx);
   __aicore__ inline void PrepareConstants();
   __aicore__ inline void LoadInputTile(int64_t gmOffset, int64_t validSize = MATRIX_SIZE);
   __aicore__ inline void LoadFullInputForMBH(int64_t gmOffset, int64_t validSize = MATRIX_SIZE);
   __aicore__ inline void StoreFinalResult(int64_t gmOffset, int64_t validSize = MATRIX_SIZE);
   __aicore__ inline void MCHInvertDiagonal(int64_t gmOffset, int64_t validSize = MATRIX_SIZE);
   __aicore__ inline void RecursiveMerge();

   // 非对齐尾块处理
   __aicore__ inline void ProcessPartialTile(int64_t gmOffset, int64_t validSize);

     // Matmul: load slotA->L0A, slotB->L0B, Mmad, fixpipe->slotDst
     __aicore__ inline void MatmulToSlot(int32_t slotA, int32_t slotB, int32_t slotDst, bool initC);
     // Matmul: load and compute, leave result in L0C
     __aicore__ inline void MatmulToL0C(int32_t slotA, int32_t slotB, bool initC);
     // L0C -> workspace GM (NZ FP16) -> L1 slot
     __aicore__ inline void L0CToSlot(int32_t slotDst);

    // Clear a slot by copying ZERO from GM
    __aicore__ inline void ClearSlot(int32_t slot);

    // === MCH 双缓冲辅助函数 ===
    __aicore__ inline void LoadToL0A_Offset(int32_t slot, int32_t offset);
    __aicore__ inline void LoadToL0B_Offset(int32_t slot, int32_t offset);
    __aicore__ inline void Mmad_Offset(int32_t bufOffset, bool initC);
    __aicore__ inline void Mmad_ACC_Offset(int32_t xBufOffset, int32_t yBufOffset);
    __aicore__ inline void StoreL0CToSlot_Offset(int32_t slot, int32_t l0cOffset);
    __aicore__ inline void StoreL0CToSlot_Y_Offset(int32_t slot, int32_t l0cOffset);  // 使用 scratchGM_Y_
    __aicore__ inline void MCH_InitXY();  // MCH 初始化：计算 Y=A² 和 X=I-A

    // === MBH 优化：直接从 L1 提取对角块到 L0 ===
    __aicore__ inline void LoadDiagonalBlocksToL0A(int32_t slot, int32_t l0Offset,
                                                    int32_t blockSize, int32_t startBlock);
    __aicore__ inline void LoadDiagonalBlocksToL0B(int32_t slot, int32_t l0Offset,
                                                    int32_t blockSize, int32_t startBlock);

 private:
     TPipe pipe_;
    GlobalTensor<T> inputGM_;
    GlobalTensor<T> outputGM_;
   GlobalTensor<T> workspaceGM_;
   GlobalTensor<T> scratchGM_;    // 每核独立的 GM 中转缓冲区 (X 流)
   GlobalTensor<T> scratchGM_Y_;  // 每核独立的 GM 中转缓冲区 (Y 流，双缓冲优化)

     TBuf<TPosition::A1> l1Buf_;
    LocalTensor<T> l1_;
    TBuf<TPosition::A2> l0aBuf_;
    LocalTensor<T> l0a_;
    TBuf<TPosition::B2> l0bBuf_;
    LocalTensor<T> l0b_;
     TBuf<TPosition::CO1> l0cBuf_;
     LocalTensor<float> l0c_;

    int64_t totalTiles_;
    int64_t matrixSize_;
    int64_t numHeads_;
    int64_t seqLen_;
    int64_t batchSize_;
    int64_t isLower_;
    int64_t hasCuSeqlens_;
    int64_t tilesPerCore_;
    int64_t aicIdx_;
    int64_t numChunks_;
    int64_t lastChunkValidSize_;
    int64_t isVarlen_;
    int64_t totalChunks_;
    int64_t rowStride_;

    // 预计算的 Nd2NzParams 模板（减少 scalar 开销）
    Nd2NzParams scratchToL1Params_;    // scratch GM → L1 slot（MatmulToSlot 等使用）
    Nd2NzParams diagLoadParams_;       // 对角块 GM → L1（LoadInputTile 使用）
    Nd2NzParams fullLoadParams_;       // 整矩阵 GM → L1（MCH 最后一轮使用）
    int64_t layoutMode_;
    GlobalTensor<int64_t> cuSeqlensGM_;
    GlobalTensor<int64_t> chunkIndicesGM_;

    // AIC/AIV 同步标志 - 等待 AIV 完成
    Catlass::Arch::CrossCoreFlagWithReverse<> flagAivFinish_{SYNC_AIC_AIV_FLAG_SOLVE, SYNC_AIV_AIC_FLAG_SOLVE};
 };


 // ============ Implementation ============

template <int MATRIX_SIZE, typename T>
__aicore__ inline void SolveTriCube<MATRIX_SIZE, T>::Init(
     GM_ADDR x, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
     GM_ADDR x_out, GM_ADDR workspace,
     const SolveTriTilingData* tilingData)
 {
    totalTiles_ = tilingData->totalTiles;
    matrixSize_ = tilingData->matrixSize;
    numHeads_ = tilingData->numHeads;
    seqLen_ = tilingData->seqLen;
    batchSize_ = tilingData->batchSize;
    isLower_ = tilingData->isLower;
    hasCuSeqlens_ = tilingData->hasCuSeqlens;
    tilesPerCore_ = tilingData->tilesPerCore;
    numChunks_ = tilingData->numChunks;
    lastChunkValidSize_ = tilingData->lastChunkValidSize;
    isVarlen_ = tilingData->isVarlen;
    totalChunks_ = tilingData->totalChunks;
    layoutMode_ = tilingData->layoutMode;

    // 行间步长: BHTD=BT, BSND/THD=H*BT
    if (layoutMode_ == 0) {
        rowStride_ = matrixSize_;  // BHTD
    } else {
        rowStride_ = numHeads_ * matrixSize_;  // BSND or THD
    }

    // AIC 的核索引
    aicIdx_ = GetBlockIdx();

     inputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
     outputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x_out));
     workspaceGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace));

     if (isVarlen_) {
         cuSeqlensGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(cu_seqlens));
         chunkIndicesGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(chunk_indices));
     }
     // 每核独立中转区：位于共享矩阵之后，每核两个 TILE_LEN（X 流和 Y 流双缓冲）
     int64_t scratchOffset = GM_NUM_SHARED_SLOTS * TILE_LEN + aicIdx_ * 2 * TILE_LEN;
     scratchGM_ = workspaceGM_[scratchOffset];
     scratchGM_Y_ = workspaceGM_[scratchOffset + TILE_LEN];

     // AIC 初始化 L1/L0 buffers
     pipe_.InitBuffer(l1Buf_, L1_TOTAL_ELEMS * sizeof(T));
     l1_ = l1Buf_.Get<T>();
     pipe_.InitBuffer(l0aBuf_, 2 * TILE_LEN * sizeof(T));
     l0a_ = l0aBuf_.Get<T>();
     pipe_.InitBuffer(l0bBuf_, 2 * TILE_LEN * sizeof(T));
     l0b_ = l0bBuf_.Get<T>();
    pipe_.InitBuffer(l0cBuf_, 2 * TILE_LEN * sizeof(float));  // MCH 双缓冲需要两份
    l0c_ = l0cBuf_.Get<float>();

    // 预计算 Nd2NzParams 模板（减少热路径上的 scalar 开销）
    // 1. scratch GM → L1 slot（MatmulToSlot 等使用）
    scratchToL1Params_.ndNum = 1;
    scratchToL1Params_.nValue = MATRIX_SIZE;
    scratchToL1Params_.dValue = MATRIX_SIZE;
    scratchToL1Params_.srcDValue = MATRIX_SIZE;  // scratch 中是紧凑存储
    scratchToL1Params_.srcNdMatrixStride = 0;
    scratchToL1Params_.dstNzNStride = 1;
    scratchToL1Params_.dstNzC0Stride = MATRIX_SIZE;
    scratchToL1Params_.dstNzMatrixStride = 0;

    // 2. 对角块 GM → L1（LoadInputTile 使用）
    diagLoadParams_.nValue = FRAC;
    diagLoadParams_.dValue = FRAC;
    diagLoadParams_.srcDValue = static_cast<uint32_t>(rowStride_);
    diagLoadParams_.srcNdMatrixStride = FRAC * static_cast<int32_t>(rowStride_) + FRAC;
    diagLoadParams_.dstNzNStride = 1;
    diagLoadParams_.dstNzC0Stride = FRAC;
    diagLoadParams_.dstNzMatrixStride = (NUM_FRACS + 1) * FRAC_LEN;
    // diagLoadParams_.ndNum 在运行时设置

    // 3. 整矩阵 GM → L1（MCH 最后一轮使用）
    fullLoadParams_.ndNum = 1;
    fullLoadParams_.srcDValue = static_cast<uint32_t>(rowStride_);
    fullLoadParams_.srcNdMatrixStride = 0;
    fullLoadParams_.dstNzNStride = 1;
    fullLoadParams_.dstNzC0Stride = MATRIX_SIZE;
    fullLoadParams_.dstNzMatrixStride = 0;
    // fullLoadParams_.nValue/dValue 在运行时设置（validSize）
}

template <int MATRIX_SIZE, typename T>
__aicore__ inline void SolveTriCube<MATRIX_SIZE, T>::Process()
 {
     int64_t startTile = aicIdx_ * tilesPerCore_;
     int64_t endTile = startTile + tilesPerCore_;

     if (endTile > totalTiles_) endTile = totalTiles_;
     if (startTile >= totalTiles_) {
         return;
     }

     // 等待 AIV 完成辅助矩阵的生成
     SyncAll<false>();

     PrepareConstants();

     // 处理每个 tile
     for (int64_t t = startTile; t < endTile; t++) {
         ProcessOneTile(t);
     }
 }

template <int MATRIX_SIZE, typename T>
__aicore__ inline int64_t SolveTriCube<MATRIX_SIZE, T>::GetTileGMOffset(int64_t tileIdx)
{
    int64_t H = numHeads_;
    int64_t BT = matrixSize_;

    if (layoutMode_ == 2) {
        // THD 变长格式: [total_T, H, BT]
        // 遍历顺序: chunk_global → H (H 变化最快)
        int64_t chunk_global_idx = tileIdx / H;
        int64_t h = tileIdx % H;

        // 从 GM 读取 chunk_indices[chunk_global_idx] = (seq_idx, chunk_in_seq)
        int64_t seq_idx = chunkIndicesGM_.GetValue(chunk_global_idx * 2);
        int64_t chunk_in_seq = chunkIndicesGM_.GetValue(chunk_global_idx * 2 + 1);

        // 从 cu_seqlens 读取该序列的起始位置
        int64_t bos = cuSeqlensGM_.GetValue(seq_idx);

        // 偏移: (bos + chunk_in_seq * BT) * H * BT + h * BT
        return (bos + chunk_in_seq * BT) * H * BT + h * BT;

    } else if (layoutMode_ == 1) {
        // BSND 格式: [B, T, H, BT]
        // 遍历顺序: B → chunk → H (H 变化最快)
        int64_t seqT = seqLen_;

        int64_t h = tileIdx % H;
        int64_t chunk = (tileIdx / H) % numChunks_;
        int64_t b = tileIdx / (H * numChunks_);

        // offset = b×T×H×BT + chunk×BT×H×BT + h×BT
        return b * seqT * H * BT + chunk * BT * H * BT + h * BT;

    } else {
        // BHTD 格式: [B, H, T, BT]
        // 遍历顺序: B → H → chunk (chunk 变化最快)
        int64_t seqT = seqLen_;
        int64_t chunk = tileIdx % numChunks_;
        int64_t h = (tileIdx / numChunks_) % H;
        int64_t b = tileIdx / (numChunks_ * H);
        return b * H * seqT * BT + h * seqT * BT + chunk * BT * BT;
    }
}

template <int MATRIX_SIZE, typename T>
__aicore__ inline int64_t SolveTriCube<MATRIX_SIZE, T>::GetTileValidSize(int64_t tileIdx)
{
    if (layoutMode_ == 2) {
        // THD: 动态计算每个序列的尾块
        int64_t H = numHeads_;
        int64_t BT = matrixSize_;
        int64_t chunk_global_idx = tileIdx / H;

        // 从 chunk_indices 读取 (seq_idx, chunk_in_seq)
        int64_t seq_idx = chunkIndicesGM_.GetValue(chunk_global_idx * 2);
        int64_t chunk_in_seq = chunkIndicesGM_.GetValue(chunk_global_idx * 2 + 1);

        // 从 cu_seqlens 计算该序列的长度
        int64_t bos = cuSeqlensGM_.GetValue(seq_idx);
        int64_t eos = cuSeqlensGM_.GetValue(seq_idx + 1);
        int64_t seq_len = eos - bos;

        // remaining < BT 即为尾块
        int64_t chunk_start = chunk_in_seq * BT;
        int64_t remaining = seq_len - chunk_start;
        return (remaining >= BT) ? BT : remaining;
    } else {
        // BHTD/BSND: 使用预计算的 lastChunkValidSize
        int64_t chunk;
        if (layoutMode_ == 1) {
            // BSND: tileIdx = b×(H×numChunks) + chunk×H + h
            chunk = (tileIdx / numHeads_) % numChunks_;
        } else {
            // BHTD: tileIdx = b×(H×numChunks) + h×numChunks + chunk
            chunk = tileIdx % numChunks_;
        }
        if (chunk == numChunks_ - 1) {
            return lastChunkValidSize_;
        }
        return matrixSize_;
    }
}

template <int MATRIX_SIZE, typename T>
__aicore__ inline void SolveTriCube<MATRIX_SIZE, T>::ProcessOneTile(int64_t tileIdx)
{
    int64_t gmOffset = GetTileGMOffset(tileIdx);
    int64_t validSize = GetTileValidSize(tileIdx);

    if (validSize < matrixSize_) {
        // 非对齐尾块：特殊路径
        ProcessPartialTile(gmOffset, validSize);
    } else {
        // 正常 chunk：原有逻辑
        LoadInputTile(gmOffset);

        MCHInvertDiagonal(gmOffset);

        if constexpr (MATRIX_SIZE > FRAC) {
            LoadFullInputForMBH(gmOffset);
            RecursiveMerge();
        } else {
            // 单 fractal：MCH 结果在 SLOT_X，搬到 L0C 供 StoreFinalResult 使用
            MatmulToL0C(SLOT_X, SLOT_I, true);
            SetFlag<HardEvent::M_FIX>(EVT_M_FIX);
            WaitFlag<HardEvent::M_FIX>(EVT_M_FIX);
        }
        StoreFinalResult(gmOffset);
    }
}

 // Load slotA->L0A(A2), slotB->L0B(B2), Mmad, result in L0C(CO1)
 // L1(NZ): 分形内行主序(小Z)，分形间列主序(大N) -> (0,0),(1,0),(0,1),(1,1)
 // L0A(ZZ): 分形内行主序(小Z)，分形间行主序(大Z) -> (0,0),(0,1),(1,0),(1,1)
 // L0B(ZN): 分形内列主序(小N)，分形间行主序(大Z) -> (0,0),(0,1),(1,0),(1,1)
 // 需要重排分形间顺序：NZ(大N) -> ZZ/ZN(大Z)
 // 参照 mmad_custom.h 的 SplitA/SplitB 做法：循环 + srcStride 跳读
 template <int MATRIX_SIZE, typename T>
 __aicore__ inline void SolveTriCube<MATRIX_SIZE, T>::MatmulToL0C(int32_t slotA, int32_t slotB, bool initC)
 {
     // L1(NZ) -> L0A(ZZ): 重排分形间顺序，分形内不转置
     LoadData2DParams loadParamsA;
     loadParamsA.startIndex = 0;
     loadParamsA.repeatTimes = NUM_FRACS;    // K方向分形数（每次搬一行的所有列）
     loadParamsA.srcStride = NUM_FRACS;      // 跳过M方向，读同一行的下一列
     loadParamsA.dstGap = 0;
     loadParamsA.ifTranspose = false;        // 分形内不转置
    //  PipeBarrier<PIPE_ALL>();
     for (int32_t i = 0; i < NUM_FRACS; ++i) {
         int32_t srcOffsetA = slotA * L1_SLOT_ELEMS + i * FRAC_LEN;  // 从第i个M行块开始
         int32_t dstOffsetA = i * NUM_FRACS * FRAC_LEN;              // 写到L0A的第i个M行块
         LoadData(l0a_[dstOffsetA], l1_[srcOffsetA], loadParamsA);
     }
    //  PipeBarrier<PIPE_ALL>();
     // L1(NZ) -> L0B(ZN): 重排分形间顺序，分形内转置(Z->N)
     LoadData2DParams loadParamsB;
     loadParamsB.startIndex = 0;
     loadParamsB.repeatTimes = NUM_FRACS;    // N方向分形数（每次搬一行的所有列）
     loadParamsB.srcStride = NUM_FRACS;      // 跳过K方向，读同一行的下一列
     loadParamsB.dstGap = 0;
     loadParamsB.ifTranspose = true;         // 分形内 Z->N 转置

     for (int32_t i = 0; i < NUM_FRACS; ++i) {
         int32_t srcOffsetB = slotB * L1_SLOT_ELEMS + i * FRAC_LEN;  // 从第i个K行块开始
         int32_t dstOffsetB = i * NUM_FRACS * FRAC_LEN;              // 写到L0B的第i个K行块
         LoadData(l0b_[dstOffsetB], l1_[srcOffsetB], loadParamsB);
     }

     SetFlag<HardEvent::MTE1_M>(EVT_MTE1_M);
     WaitFlag<HardEvent::MTE1_M>(EVT_MTE1_M);
     MmadParams mmadParams;
     mmadParams.m = MATRIX_SIZE;
     mmadParams.n = MATRIX_SIZE;
     mmadParams.k = MATRIX_SIZE;
     mmadParams.cmatrixInitVal = initC;
     mmadParams.cmatrixSource = false;
     mmadParams.unitFlag = 0;
    Mmad(l0c_, l0a_, l0b_, mmadParams);
}

 // L0C(FP32) -> workspace GM(ND, FP16) -> L1(NZ, FP16)
 // Fixpipe CFG_ROW_MAJOR 输出 ND 格式
 // 对于 16x16：ND = NZ，直接 DataCopy 读回
 // 对于 32x32+：ND ≠ NZ，需要 Nd2Nz 转换读回
 template <int MATRIX_SIZE, typename T>
__aicore__ inline void SolveTriCube<MATRIX_SIZE, T>::L0CToSlot(int32_t slotDst)
{
    // Step 1: L0C -> scratch GM (NZ→ND, FP32→FP16)
    int32_t rowStride = MATRIX_SIZE;
    NsSolveTri::L0CToGM(
        scratchGM_,
        l0c_,
        MATRIX_SIZE,
        MATRIX_SIZE,
        MATRIX_SIZE,
        rowStride
    );
    SetFlag<HardEvent::FIX_MTE2>(EVT_FIX_MTE2);
    WaitFlag<HardEvent::FIX_MTE2>(EVT_FIX_MTE2);
    // Step 2: scratch GM(ND) -> L1(NZ)，使用预计算的参数模板
    DataCopy(l1_[slotDst * L1_SLOT_ELEMS], scratchGM_, scratchToL1Params_);
    SetFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
    WaitFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
}

 // Matmul + fixpipe to slot (convenience wrapper)
 template <int MATRIX_SIZE, typename T>
 __aicore__ inline void SolveTriCube<MATRIX_SIZE, T>::MatmulToSlot(
     int32_t slotA, int32_t slotB, int32_t slotDst, bool initC)
 {
    MatmulToL0C(slotA, slotB, initC);
    SetFlag<HardEvent::M_FIX>(EVT_M_FIX);
    WaitFlag<HardEvent::M_FIX>(EVT_M_FIX);
    L0CToSlot(slotDst);
 }

// Clear a slot by copying ZERO matrix from GM
template <int MATRIX_SIZE, typename T>
__aicore__ inline void SolveTriCube<MATRIX_SIZE, T>::ClearSlot(int32_t slot)
{
    DataCopyParams params;
    params.blockCount = 1;
    params.blockLen = TILE_LEN * sizeof(T) / 32;
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopy(l1_[slot * L1_SLOT_ELEMS], workspaceGM_[GM_WS_ZERO * TILE_LEN], params);
}

// === MCH 双缓冲辅助函数实现 ===

// 加载 L1 slot 到 L0A 指定 offset
template <int MATRIX_SIZE, typename T>
__aicore__ inline void SolveTriCube<MATRIX_SIZE, T>::LoadToL0A_Offset(int32_t slot, int32_t offset)
{
    LoadData2DParams loadParams;
    loadParams.startIndex = 0;
    loadParams.repeatTimes = NUM_FRACS;
    loadParams.srcStride = NUM_FRACS;
    loadParams.dstGap = 0;
    loadParams.ifTranspose = false;
    for (int32_t i = 0; i < NUM_FRACS; ++i) {
        int32_t src = slot * L1_SLOT_ELEMS + i * FRAC_LEN;
        int32_t dst = offset + i * NUM_FRACS * FRAC_LEN;
        LoadData(l0a_[dst], l1_[src], loadParams);
    }
}

// 加载 L1 slot 到 L0B 指定 offset（带转置）
template <int MATRIX_SIZE, typename T>
__aicore__ inline void SolveTriCube<MATRIX_SIZE, T>::LoadToL0B_Offset(int32_t slot, int32_t offset)
{
    LoadData2DParams loadParams;
    loadParams.startIndex = 0;
    loadParams.repeatTimes = NUM_FRACS;
    loadParams.srcStride = NUM_FRACS;
    loadParams.dstGap = 0;
    loadParams.ifTranspose = true;
    for (int32_t i = 0; i < NUM_FRACS; ++i) {
        int32_t src = slot * L1_SLOT_ELEMS + i * FRAC_LEN;
        int32_t dst = offset + i * NUM_FRACS * FRAC_LEN;
        LoadData(l0b_[dst], l1_[src], loadParams);
    }
}

// Mmad：从 L0A/L0B 指定 offset 计算，结果写入 L0C 指定 offset
template <int MATRIX_SIZE, typename T>
__aicore__ inline void SolveTriCube<MATRIX_SIZE, T>::Mmad_Offset(int32_t bufOffset, bool initC)
{
    MmadParams mmadParams;
    mmadParams.m = MATRIX_SIZE;
    mmadParams.n = MATRIX_SIZE;
    mmadParams.k = MATRIX_SIZE;
    mmadParams.cmatrixInitVal = initC;
    mmadParams.cmatrixSource = false;
    mmadParams.unitFlag = 0;
    Mmad(l0c_[bufOffset], l0a_[bufOffset], l0b_[bufOffset], mmadParams);
}

// Mmad 累加：c_l0[xBufOffset] += a_l0[xBufOffset] @ b_l0[yBufOffset]
template <int MATRIX_SIZE, typename T>
__aicore__ inline void SolveTriCube<MATRIX_SIZE, T>::Mmad_ACC_Offset(int32_t xBufOffset, int32_t yBufOffset)
{
    MmadParams mmadParams;
    mmadParams.m = MATRIX_SIZE;
    mmadParams.n = MATRIX_SIZE;
    mmadParams.k = MATRIX_SIZE;
    mmadParams.cmatrixInitVal = false;  // 累加模式
    mmadParams.cmatrixSource = false;
    mmadParams.unitFlag = 0;
    Mmad(l0c_[xBufOffset], l0a_[xBufOffset], l0b_[yBufOffset], mmadParams);
}

// 从 L0C 指定 offset 写回到 L1 slot
template <int MATRIX_SIZE, typename T>
__aicore__ inline void SolveTriCube<MATRIX_SIZE, T>::StoreL0CToSlot_Offset(int32_t slot, int32_t l0cOffset)
{
    int32_t rowStride = MATRIX_SIZE;
    NsSolveTri::L0CToGM(
        scratchGM_,
        l0c_[l0cOffset],
        MATRIX_SIZE,
        MATRIX_SIZE,
        MATRIX_SIZE,
        rowStride
    );
    SetFlag<HardEvent::FIX_MTE2>(EVT_FIX_MTE2);
    WaitFlag<HardEvent::FIX_MTE2>(EVT_FIX_MTE2);

    // 使用预计算的参数模板
    DataCopy(l1_[slot * L1_SLOT_ELEMS], scratchGM_, scratchToL1Params_);
    SetFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
    WaitFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
}

// 从 L0C 指定 offset 写回到 L1 slot（使用 scratchGM_Y_，实现双缓冲）
template <int MATRIX_SIZE, typename T>
__aicore__ inline void SolveTriCube<MATRIX_SIZE, T>::StoreL0CToSlot_Y_Offset(int32_t slot, int32_t l0cOffset)
{
    int32_t rowStride = MATRIX_SIZE;
    NsSolveTri::L0CToGM(
        scratchGM_Y_,
        l0c_[l0cOffset],
        MATRIX_SIZE,
        MATRIX_SIZE,
        MATRIX_SIZE,
        rowStride
    );
    SetFlag<HardEvent::FIX_MTE2>(EVT_FIX_MTE2);
    WaitFlag<HardEvent::FIX_MTE2>(EVT_FIX_MTE2);

    // 使用预计算的参数模板
    DataCopy(l1_[slot * L1_SLOT_ELEMS], scratchGM_Y_, scratchToL1Params_);
    SetFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
    WaitFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
}

// MCH 初始化：计算 Y=A² 和 X=I-A，结果存入 SLOT_Y 和 SLOT_X
template <int MATRIX_SIZE, typename T>
__aicore__ inline void SolveTriCube<MATRIX_SIZE, T>::MCH_InitXY()
{
    constexpr int32_t X_BUF = 0;
    constexpr int32_t Y_BUF = TILE_LEN;

    // ========== 全并行双缓冲初始化：Y=A² 和 X=I-A ==========
    // Y 使用 L0[1]，X 使用 L0[0]，搬运和计算完全并行

    // --- 阶段 1：并行搬运 ---
    // Y: 搬运 A 到 L0A[1], L0B[1]
    LoadToL0A_Offset(SLOT_INPUT, Y_BUF);
    LoadToL0B_Offset(SLOT_INPUT, Y_BUF);
    SetFlag<HardEvent::MTE1_M>(EVT_Y);
    // X: 搬运 I 到 L0A[0], L0B[0]（与 Y 搬运并行）
    LoadToL0A_Offset(SLOT_I, X_BUF);
    LoadToL0B_Offset(SLOT_I, X_BUF);
    SetFlag<HardEvent::MTE1_M>(EVT_X);

    // --- 阶段 2：并行计算 ---
    // Y: L0C[1] = A * A
    WaitFlag<HardEvent::MTE1_M>(EVT_Y);
    Mmad_Offset(Y_BUF, true);
    SetFlag<HardEvent::M_FIX>(EVT_Y);
    // X: L0C[0] = I * I = I
    WaitFlag<HardEvent::MTE1_M>(EVT_X);
    Mmad_Offset(X_BUF, true);

    // --- 阶段 3：X 第二次搬运 + Y 写回（并行）---
    SetFlag<HardEvent::M_MTE1>(EVT_X);
    WaitFlag<HardEvent::M_MTE1>(EVT_X);
    SetFlag<HardEvent::M_MTE1>(EVT_X);
    WaitFlag<HardEvent::M_MTE1>(EVT_X);
    LoadToL0A_Offset(SLOT_INEG, X_BUF);
    LoadToL0B_Offset(SLOT_INPUT, X_BUF);
    SetFlag<HardEvent::MTE1_M>(EVT_X);
    // Y: 写回
    WaitFlag<HardEvent::M_FIX>(EVT_Y);
    StoreL0CToSlot_Y_Offset(SLOT_Y, Y_BUF);

    // --- 阶段 4：X 第二次计算 ---
    WaitFlag<HardEvent::MTE1_M>(EVT_X);
    Mmad_Offset(X_BUF, false);
    SetFlag<HardEvent::M_FIX>(EVT_X);

    // --- 阶段 5：X 写回 ---
    WaitFlag<HardEvent::M_FIX>(EVT_X);
    StoreL0CToSlot_Offset(SLOT_X, X_BUF);
}

// === MBH 优化：直接从 L1 slot 的 X 矩阵提取对角块到 L0A ===
// 避免 L1→GM→L1 的中转，直接 L1→L0
template <int MATRIX_SIZE, typename T>
__aicore__ inline void SolveTriCube<MATRIX_SIZE, T>::LoadDiagonalBlocksToL0A(
    int32_t slot, int32_t l0Offset, int32_t blockSize, int32_t startBlock)
{
    int32_t numBlocks = MATRIX_SIZE / blockSize;
    int32_t fracsPerBlock = blockSize / FRAC;

    // 先加载 SLOT_ZERO 作为全零基底
    LoadData2DParams loadParamsZero;
    loadParamsZero.startIndex = 0;
    loadParamsZero.repeatTimes = NUM_FRACS;
    loadParamsZero.srcStride = NUM_FRACS;
    loadParamsZero.dstGap = 0;
    loadParamsZero.ifTranspose = false;

    for (int32_t i = 0; i < NUM_FRACS; ++i) {
        int32_t src = SLOT_ZERO * L1_SLOT_ELEMS + i * FRAC_LEN;
        int32_t dst = l0Offset + i * NUM_FRACS * FRAC_LEN;
        LoadData(l0a_[dst], l1_[src], loadParamsZero);
    }

    // 等待 ZERO 加载完成，再加载对角块（避免 WAW）
    PipeBarrier<PIPE_MTE1>();

    LoadData2DParams loadParams;
    loadParams.startIndex = 0;
    loadParams.repeatTimes = fracsPerBlock;
    loadParams.srcStride = NUM_FRACS;
    loadParams.dstGap = 0;
    loadParams.ifTranspose = false;

    // 只搬运对角块的 fractal，覆盖 ZERO 的对应位置
    for (int32_t blk = startBlock; blk < numBlocks; blk += 2) {
        for (int32_t fi = 0; fi < fracsPerBlock; fi++) {
            int32_t row = blk * fracsPerBlock + fi;
            int32_t col = blk * fracsPerBlock;  // 对角块的起始列

            // L1 NZ 中 fractal 的 offset: (col * NUM_FRACS + row) * FRAC_LEN
            int32_t srcOffset = slot * L1_SLOT_ELEMS + (col * NUM_FRACS + row) * FRAC_LEN;

            // L0A ZZ 格式中对角块位置: (row * NUM_FRACS + col) * FRAC_LEN
            int32_t dstOffset = l0Offset + (row * NUM_FRACS + col) * FRAC_LEN;

            LoadData(l0a_[dstOffset], l1_[srcOffset], loadParams);
        }
    }
}

// === MBH 优化：直接从 L1 slot 的 X 矩阵提取对角块到 L0B ===
template <int MATRIX_SIZE, typename T>
__aicore__ inline void SolveTriCube<MATRIX_SIZE, T>::LoadDiagonalBlocksToL0B(
    int32_t slot, int32_t l0Offset, int32_t blockSize, int32_t startBlock)
{
    int32_t numBlocks = MATRIX_SIZE / blockSize;
    int32_t fracsPerBlock = blockSize / FRAC;

    // 先加载 SLOT_ZERO 作为全零基底
    LoadData2DParams loadParamsZero;
    loadParamsZero.startIndex = 0;
    loadParamsZero.repeatTimes = NUM_FRACS;
    loadParamsZero.srcStride = NUM_FRACS;
    loadParamsZero.dstGap = 0;
    loadParamsZero.ifTranspose = true;  // L0B 需要转置

    for (int32_t i = 0; i < NUM_FRACS; ++i) {
        int32_t src = SLOT_ZERO * L1_SLOT_ELEMS + i * FRAC_LEN;
        int32_t dst = l0Offset + i * NUM_FRACS * FRAC_LEN;
        LoadData(l0b_[dst], l1_[src], loadParamsZero);
    }

    // 等待 ZERO 加载完成，再加载对角块（避免 WAW）
    PipeBarrier<PIPE_MTE1>();

    LoadData2DParams loadParams;
    loadParams.startIndex = 0;
    loadParams.repeatTimes = fracsPerBlock;
    loadParams.srcStride = NUM_FRACS;
    loadParams.dstGap = 0;
    loadParams.ifTranspose = true;

    // 只搬运对角块的 fractal
    for (int32_t blk = startBlock; blk < numBlocks; blk += 2) {
        for (int32_t fi = 0; fi < fracsPerBlock; fi++) {
            int32_t row = blk * fracsPerBlock + fi;
            int32_t col = blk * fracsPerBlock;

            int32_t srcOffset = slot * L1_SLOT_ELEMS + (col * NUM_FRACS + row) * FRAC_LEN;
            int32_t dstOffset = l0Offset + (row * NUM_FRACS + col) * FRAC_LEN;

            LoadData(l0b_[dstOffset], l1_[srcOffset], loadParams);
        }
    }
}


 template <int MATRIX_SIZE, typename T>
 __aicore__ inline void SolveTriCube<MATRIX_SIZE, T>::PrepareConstants()
 {
     // GM 中辅助矩阵是 ND 格式，L1 中需要 NZ fractal 格式供 LoadData/Mmad 使用
     // 使用 DataCopy + Nd2NzParams 在 MTE2 搬运时随路转换
     Nd2NzParams nd2nzParams;
     nd2nzParams.ndNum = 1;
     nd2nzParams.nValue = MATRIX_SIZE;
     nd2nzParams.dValue = MATRIX_SIZE;
     nd2nzParams.srcDValue = MATRIX_SIZE;
     nd2nzParams.srcNdMatrixStride = 0;
     nd2nzParams.dstNzNStride = 1;
     nd2nzParams.dstNzC0Stride = MATRIX_SIZE;
     nd2nzParams.dstNzMatrixStride = 0;

    // GM layout: [I, -I, ZERO] each TILE_LEN elements
    DataCopy(l1_[SLOT_I * L1_SLOT_ELEMS], workspaceGM_[GM_WS_I * TILE_LEN], nd2nzParams);
    DataCopy(l1_[SLOT_INEG * L1_SLOT_ELEMS], workspaceGM_[GM_WS_INEG * TILE_LEN], nd2nzParams);
    DataCopy(l1_[SLOT_ZERO * L1_SLOT_ELEMS], workspaceGM_[GM_WS_ZERO * TILE_LEN], nd2nzParams);
    SetFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
    WaitFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
}

template <int MATRIX_SIZE, typename T>
__aicore__ inline void SolveTriCube<MATRIX_SIZE, T>::LoadInputTile(int64_t gmOffset, int64_t validSize)
{
    // MCH 路径：批量搬运所有对角 fractals
    // 使用 Nd2Nz 的 ndNum + srcNdMatrixStride 一次性搬运所有对角块
    // validSize < MATRIX_SIZE 时为尾块，最后一个对角块如果不足 16x16，硬件自动补零
    ClearSlot(SLOT_INPUT);
    PipeBarrier<PIPE_MTE2>();

    int32_t numDiagFracs = (static_cast<int32_t>(validSize) + FRAC - 1) / FRAC;

    // 使用预计算的参数模板，只修改变化的字段
    diagLoadParams_.ndNum = numDiagFracs;
    DataCopy(l1_[SLOT_INPUT * L1_SLOT_ELEMS], inputGM_[gmOffset], diagLoadParams_);
    SetFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
    WaitFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
}

template <int MATRIX_SIZE, typename T>
__aicore__ inline void SolveTriCube<MATRIX_SIZE, T>::StoreFinalResult(int64_t gmOffset, int64_t validSize)
{
    // L0C -> output GM directly with NZ->ND conversion + FP32->FP16
    // validSize 控制实际写回的行/列数
    NsSolveTri::L0CToGM(
        outputGM_[gmOffset],  // 目标 GM
        l0c_,                 // 源 L0C
        static_cast<uint32_t>(validSize),   // mTileActual
        static_cast<uint32_t>(validSize),   // nTileActual
        MATRIX_SIZE,          // srcStride (L0C 中 Z 排布间距)
        static_cast<uint32_t>(rowStride_)   // dstStride (输出行步长)
    );
    SetFlag<HardEvent::FIX_MTE2>(EVT_FIX_MTE2);
    WaitFlag<HardEvent::FIX_MTE2>(EVT_FIX_MTE2);
}


template <int MATRIX_SIZE, typename T>
__aicore__ inline void SolveTriCube<MATRIX_SIZE, T>::MCHInvertDiagonal(int64_t gmOffset, int64_t validSize)
{
    // SLOT_INPUT 中已经是对角 fractals（LoadInputTile 分块搬入的）
    // 双缓冲：buffer[0] (offset=0) 给 X 计算流，buffer[1] (offset=TILE_LEN) 给 Y 计算流
    // L0A、L0B、L0C 各两份

    constexpr int32_t X_BUF = 0;
    constexpr int32_t Y_BUF = TILE_LEN;

    // 初始化：计算 Y=A² 和 X=I-A
    MCH_InitXY();

    // MCH 迭代：X_new = X + X@Y, Y_new = Y@Y
    constexpr int32_t NUM_ITERS = 3;

    // 循环前初始化：让首轮的 WaitFlag 能通过
    // 初始化阶段的 FIX 已完成，L0C 可被 M 写（复用上面的状态）
    SetFlag<HardEvent::M_MTE1>(EVT_X);   // L0A[0]/L0B[0] 可写
    SetFlag<HardEvent::M_MTE1>(EVT_Y);   // L0A[1]/L0B[1] 可写
    SetFlag<HardEvent::FIX_M>(EVT_X);    // L0C[0] 可被 M 写
    SetFlag<HardEvent::FIX_M>(EVT_Y);    // L0C[1] 可被 M 写

    for (int32_t iter = 0; iter < NUM_ITERS; iter++) {
        // === Step 1: 加载 X, I 到 buffer[0] ===
        WaitFlag<HardEvent::M_MTE1>(EVT_X);
        LoadToL0A_Offset(SLOT_X, X_BUF);
        LoadToL0B_Offset(SLOT_I, X_BUF);
        SetFlag<HardEvent::MTE1_M>(EVT_X);

        // === Step 2: 加载 Y 到 buffer[1] ===
        WaitFlag<HardEvent::M_MTE1>(EVT_Y);
        LoadToL0A_Offset(SLOT_Y, Y_BUF);
        LoadToL0B_Offset(SLOT_Y, Y_BUF);
        SetFlag<HardEvent::MTE1_M>(EVT_Y);

        // 【流水线优化】最后一轮时，SLOT_INPUT 不再需要，提前发起 GM→L1 搬运
        if constexpr (MATRIX_SIZE > FRAC) {
            if (iter == NUM_ITERS - 1) {
                // 异步发起整个矩阵的搬运（与后续 Mmad 并行）
                ClearSlot(SLOT_INPUT);
                PipeBarrier<PIPE_MTE2>();

                // 使用预计算的参数模板，只修改变化的字段
                fullLoadParams_.nValue = static_cast<uint32_t>(validSize);
                fullLoadParams_.dValue = static_cast<uint32_t>(validSize);
                DataCopy(l1_[SLOT_INPUT * L1_SLOT_ELEMS], inputGM_[gmOffset], fullLoadParams_);
                SetFlag<HardEvent::MTE2_MTE1>(EVT_MBH_LOAD);  // 使用独立 event，避免与 StoreL0CToSlot 冲突
                // 不等待，让搬运与后续 Mmad 并行执行
            }
        }

        // // === Step 3: c_l0[0] = X @ I = X ===
        WaitFlag<HardEvent::FIX_M>(EVT_X);
        WaitFlag<HardEvent::MTE1_M>(EVT_X);
        Mmad_Offset(X_BUF, true);

        // === Step 4: 计算 Y² (如果不是最后一轮) ===
        if (iter < NUM_ITERS - 1) {
            WaitFlag<HardEvent::FIX_M>(EVT_Y);
            WaitFlag<HardEvent::MTE1_M>(EVT_Y);
            Mmad_Offset(Y_BUF, true);
            SetFlag<HardEvent::M_FIX>(EVT_Y);

            WaitFlag<HardEvent::M_FIX>(EVT_Y);
            StoreL0CToSlot_Y_Offset(SLOT_Y, Y_BUF);  // 使用独立的 scratchGM_Y_
            SetFlag<HardEvent::FIX_M>(EVT_Y);
            // 双缓冲优化：Y 使用 scratchGM_Y_，X 使用 scratchGM_，互不干扰，无需等待
        }

        // // === Step 5: c_l0[0] += X @ Y ===
        PipeBarrier<PIPE_M>();  // 等 Step 3/4 的 Mmad 完成，解决 L0C RAW/WAW
        if (iter >= NUM_ITERS - 1) {
            WaitFlag<HardEvent::MTE1_M>(EVT_Y);
        }
        Mmad_ACC_Offset(X_BUF, Y_BUF);
        SetFlag<HardEvent::M_FIX>(EVT_X);

        // 【流水线优化】最后一轮时，先发起 -M 的 MTE1 搬运，再写回 X
        if constexpr (MATRIX_SIZE > FRAC) {
            if (iter == NUM_ITERS - 1) {
                // 等待 GM→L1 搬运完成（在 Step 2 后发起的）
                WaitFlag<HardEvent::MTE2_MTE1>(EVT_MBH_LOAD);

                // 通知 L0A[1]/L0B[1] 空闲，供 -M 计算使用
                SetFlag<HardEvent::M_MTE1>(EVT_Y);
                WaitFlag<HardEvent::M_MTE1>(EVT_Y);
                SetFlag<HardEvent::M_MTE1>(EVT_Y);
                WaitFlag<HardEvent::M_MTE1>(EVT_Y);

                LoadToL0A_Offset(SLOT_INEG, Y_BUF);   // L0A[1] ← -I
                LoadToL0B_Offset(SLOT_INPUT, Y_BUF);  // L0B[1] ← M
                SetFlag<HardEvent::MTE1_M>(EVT_Y);
            }
        }

        // // === Step 6: 写回 X ===
        WaitFlag<HardEvent::M_FIX>(EVT_X);
        // 双缓冲优化：X 使用 scratchGM_，Y 使用 scratchGM_Y_，无需等待 Y 的 MTE2
        StoreL0CToSlot_Offset(SLOT_X, X_BUF);

        // 【流水线优化】最后一轮时，完成 -M 的计算和写回
        if constexpr (MATRIX_SIZE > FRAC) {
            if (iter == NUM_ITERS - 1) {
                // 计算 -M = (-I) * M（MTE1 已在 Step 6 之前发起）
                WaitFlag<HardEvent::MTE1_M>(EVT_Y);
                Mmad_Offset(Y_BUF, true);             // L0C[1] = L0A[1] @ L0B[1]
                SetFlag<HardEvent::M_FIX>(EVT_Y);

                // 写回 SLOT_MNEG（使用 scratchGM_Y_）
                WaitFlag<HardEvent::M_FIX>(EVT_Y);
                StoreL0CToSlot_Y_Offset(SLOT_MNEG, Y_BUF);
            }
        }

        SetFlag<HardEvent::M_MTE1>(EVT_X);
        SetFlag<HardEvent::M_MTE1>(EVT_Y);
        SetFlag<HardEvent::FIX_M>(EVT_X);
    }

    // 消费最后一轮的 SetFlag，避免 unpaired set_flag
    WaitFlag<HardEvent::M_MTE1>(EVT_X);
    WaitFlag<HardEvent::M_MTE1>(EVT_Y);
    WaitFlag<HardEvent::FIX_M>(EVT_X);
    WaitFlag<HardEvent::FIX_M>(EVT_Y);
}


// MBH: -M 的搬运和计算已在 MCH 最后一轮完成，这里只是占位
template <int MATRIX_SIZE, typename T>
__aicore__ inline void SolveTriCube<MATRIX_SIZE, T>::LoadFullInputForMBH(int64_t gmOffset, int64_t validSize)
{
    // 【流水线优化】GM→L1 搬运和 -M 计算已在 MCH 最后一轮完成
    // SLOT_INPUT 已有整个矩阵 M
    // SLOT_MNEG 已有 -M
    // 无需额外操作
}

template <int MATRIX_SIZE, typename T>
__aicore__ inline void SolveTriCube<MATRIX_SIZE, T>::RecursiveMerge()
{
    for (int32_t blockSize = FRAC; blockSize < MATRIX_SIZE; blockSize *= 2) {
        int32_t drvStart = isLower_ ? 1 : 0;
        int32_t othStart = isLower_ ? 0 : 1;

        // Y = driving * M_neg + I
        // Step 1: L0C = I
        MatmulToL0C(SLOT_I, SLOT_I, true);
        SetFlag<HardEvent::M_MTE1>(EVT_M_MTE1);
        WaitFlag<HardEvent::M_MTE1>(EVT_M_MTE1);
        SetFlag<HardEvent::M_MTE1>(EVT_M_MTE1);
        WaitFlag<HardEvent::M_MTE1>(EVT_M_MTE1);
        // Step 2: 加载 driving 到 L0A，M_neg 到 L0B
        LoadDiagonalBlocksToL0A(SLOT_X, 0, blockSize, drvStart);
        LoadToL0B_Offset(SLOT_MNEG, 0);
        SetFlag<HardEvent::MTE1_M>(EVT_MTE1_M);
        WaitFlag<HardEvent::MTE1_M>(EVT_MTE1_M);

        // Step 3: L0C += driving * M_neg
        MmadParams mmadParams;
        mmadParams.m = MATRIX_SIZE;
        mmadParams.n = MATRIX_SIZE;
        mmadParams.k = MATRIX_SIZE;
        mmadParams.cmatrixInitVal = false;
        mmadParams.cmatrixSource = false;
        mmadParams.unitFlag = 0;
        Mmad(l0c_, l0a_, l0b_, mmadParams);
        SetFlag<HardEvent::M_FIX>(EVT_M_FIX);
        WaitFlag<HardEvent::M_FIX>(EVT_M_FIX);

        // Step 4: Y -> SLOT_Y
        L0CToSlot(SLOT_Y);

        // Step 5: 加载 Y 到 L0A，other 到 L0B
        LoadToL0A_Offset(SLOT_Y, 0);
        LoadDiagonalBlocksToL0B(SLOT_X, 0, blockSize, othStart);
        SetFlag<HardEvent::MTE1_M>(EVT_MTE1_M);
        WaitFlag<HardEvent::MTE1_M>(EVT_MTE1_M);

        // Step 6: L0C = Y * other
        mmadParams.cmatrixInitVal = true;
        Mmad(l0c_, l0a_, l0b_, mmadParams);
        SetFlag<HardEvent::M_MTE1>(EVT_M_MTE1);
        WaitFlag<HardEvent::M_MTE1>(EVT_M_MTE1);
        SetFlag<HardEvent::M_MTE1>(EVT_M_MTE1);
        WaitFlag<HardEvent::M_MTE1>(EVT_M_MTE1);
        // Step 7: 加载 I 到 L0A，driving 到 L0B
        LoadToL0A_Offset(SLOT_I, 0);
        LoadDiagonalBlocksToL0B(SLOT_X, 0, blockSize, drvStart);
        SetFlag<HardEvent::MTE1_M>(EVT_MTE1_M);
        WaitFlag<HardEvent::MTE1_M>(EVT_MTE1_M);

        // Step 8: L0C += I * driving
        mmadParams.cmatrixInitVal = false;
        Mmad(l0c_, l0a_, l0b_, mmadParams);
        SetFlag<HardEvent::M_FIX>(EVT_M_FIX);
        WaitFlag<HardEvent::M_FIX>(EVT_M_FIX);

        if (blockSize < MATRIX_SIZE / 2) {
            L0CToSlot(SLOT_X);
            SetFlag<HardEvent::MTE2_MTE3>(EVT_MTE2_MTE3);
            WaitFlag<HardEvent::MTE2_MTE3>(EVT_MTE2_MTE3);
        }
    }
}

// ============ 非对齐尾块处理函数 ============

template <int MATRIX_SIZE, typename T>
__aicore__ inline void SolveTriCube<MATRIX_SIZE, T>::ProcessPartialTile(int64_t gmOffset, int64_t validSize)
{
    // Step 1-2: 搬入 validSize x validSize 对角块数据
    LoadInputTile(gmOffset, validSize);

    // Step 3: MCH（按完整 MATRIX_SIZE 计算，多余位置是零）
    MCHInvertDiagonal(gmOffset, validSize);

    // Step 4: MBH（如果 MATRIX_SIZE > FRAC）
    if constexpr (MATRIX_SIZE > FRAC) {
        LoadFullInputForMBH(gmOffset, validSize);
        RecursiveMerge();
    } else {
        MatmulToL0C(SLOT_X, SLOT_I, true);
        SetFlag<HardEvent::M_FIX>(EVT_M_FIX);
        WaitFlag<HardEvent::M_FIX>(EVT_M_FIX);
    }

    // Step 5: 只写回 validSize x validSize 的结果
    StoreFinalResult(gmOffset, validSize);
}

}  // namespace NsSolveTri

#endif  // SOLVE_TRI_CUBE_H
