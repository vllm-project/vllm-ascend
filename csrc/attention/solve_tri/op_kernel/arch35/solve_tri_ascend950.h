/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */

#ifndef SOLVE_TRI_ASCEND950_H
#define SOLVE_TRI_ASCEND950_H

#include "kernel_operator.h"
#include "solve_tri_ascend950_common.h"
#include "mem.h"

using namespace AscendC;

// ============================================================================
// SolveTri —— 完整下三角逆 (I+A)^{-1} 的 ascend950 实现（MCH + MBH 合一）
//
// 两段算法合并在同一 SolveTri 类：
//   - MCH（块对角逆）：AIV 生成 NZ 辅助矩阵 + gather 对角块，AIC 牛顿迭代求每个
//     16×16 对角块逆；cur==16 直接写 gm_out，cur>16 结果(NZ)暂存 ub_Res(UB) 供 MBH。
//   - MBH（递归合并）：X 常驻 ub_Res(UB,NZ)；逐层 blockSize=16->cur：AIV 从 ub_Res
//     提取 drv/oth 对角块到 L1，AIC 做 B/C/E/G 四步矩乘合并，层间结果 Fixpipe 回
//     ub_Res，末层写 gm_out。
//
// 【分核 / 同步】（本版：CrossCoreFlag + round-robin，统一 MCH 与 MBH）
//   - 参照仓1（recurrent_gdn/solve_tril）已验证的 MCH 写法：跨核用 CrossCoreSetFlag/
//     WaitFlag 每 tile / 每 MBH 层握手，AIC 与其配对 AIV 处理同一 tile（cur 一致 -> 层数
//     一致 -> 握手计数天然对齐，无需全局屏障 / 固定调度，无死锁）。
//   - 多核 round-robin，以 num_core 为间隔分核：
//       AIV(sub0): loop_idx = core_idx/2; loop_idx += num_core
//       AIC     : loop_idx = core_idx;   loop_idx += num_core
//   - flag：mode 0x4；0x2 = AIV->AIC(数据/提取就绪)，0x0 = AIC->AIV(计算/回写完成)。
//     两 flag 严格交替复用于 MCH 与每个 MBH 层。
//   - 注：原 SyncAll 固定调度版已整体替换为 CrossCoreFlag（MBH 一并切换）。
//
// 【布局 / 变长 TND】
//   - GM 偏移按 [B,H,T,BT](bhtd) / [B,T,H,BT](bsnd) / [total_T,H,BT](tnd 变长) 完整公式；
//     tnd 由 cu_seqlens / chunk_indices（INT64，GetValue）确定每 tile 偏移与序列长度。
//   - 行跨度 row_stride = (bhtd) chunk_size : num_head*chunk_size。
//
// 【尾块（partial chunk）正确性】
//   - actual_size = 该 chunk 的有效行数（尾块 < cur）。cur = ChunkAlign(actual_size)。
//   - 输入 gather 只读 actual_size 行（逐分形 min(16, actual_size-i*16)）；越界分形不读，
//     由 aux 置 I，使部分对角块逆为 [[valid^-1,0],[0,I]]。
//   - MBH 的 -A 只读 actual_size 行（其余清 0），保证 padding 参与 MBH 得 I。
//   - 输出只写 actual_size 行（FixpipeL0cToGM 的 validRows），避免跨序列覆写 / OOB。
// ============================================================================

constexpr AscendC::FixpipeConfig CFG_NZ_L1 = {AscendC::CO2Layout::NZ, false};
constexpr AscendC::FixpipeConfig CFG_NZ_UB = {AscendC::CO2Layout::NZ, true};

template <typename InDtype, typename OutDtype>
class SolveTri {
public:
    __aicore__ inline void Init(GM_ADDR aGm, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR outGm,
                                GM_ADDR workspace, const SolveTriTilingData *tilingData)
    {
        // Tiling
        batch_size = tilingData->batchSize;
        seq_length = tilingData->seqLen;
        num_head = tilingData->numHeads;
        chunk_size = tilingData->chunkSize;
        chunk_num_in_seq = tilingData->numChunks;
        chunk_num_total = tilingData->totalTiles; // 主循环上界 = 全部 tile 数
        mode = tilingData->layoutMode;            // 0=bhtd, 1=bsnd, 2=tnd
        is_lower = tilingData->isLower;
        tiles_per_core = tilingData->tilesPerCore;

        // GM（INT64 索引）
        gm_a.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(aGm));
        gm_cu_seqlens.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cu_seqlens));
        gm_chunk_indices.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(chunk_indices));
        gm_out.SetGlobalBuffer(reinterpret_cast<__gm__ OutDtype *>(outGm));

        OnChipBuffer buf;

        // UB（每块 chunk_size*chunk_size 个 InDtype）
        ub_I = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(0);
        ub_Zero = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(chunk_size * chunk_size * sizeof(InDtype));
        ub_A = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(chunk_size * chunk_size * sizeof(InDtype) * 2);
        ub_I_A = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(chunk_size * chunk_size * sizeof(InDtype) * 3);
        ub_Res = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(chunk_size * chunk_size * sizeof(InDtype) * 4);
        ub_FullA = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(chunk_size * chunk_size * sizeof(InDtype) * 5);
        // L1（NZ 槽）
        l1_I = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(0);
        l1_X = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(chunk_size * chunk_size * sizeof(InDtype));
        l1_Y = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(chunk_size * chunk_size * sizeof(InDtype) * 2);
        l1_MNEG = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(chunk_size * chunk_size * sizeof(InDtype) * 3);
        l1_INPUT = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(chunk_size * chunk_size * sizeof(InDtype) * 4);

        // L0
        l0a_X = buf.template GetBuffer<BufferType::ASCEND_L0A, InDtype>(0);
        l0a_Y = buf.template GetBuffer<BufferType::ASCEND_L0A, InDtype>(chunk_size * chunk_size * sizeof(InDtype));
        l0b_X = buf.template GetBuffer<BufferType::ASCEND_L0B, InDtype>(0);
        l0b_Y = buf.template GetBuffer<BufferType::ASCEND_L0B, InDtype>(chunk_size * chunk_size * sizeof(InDtype));
        l0c_X = buf.template GetBuffer<BufferType::ASCEND_L0C, float>(0);
        l0c_Y = buf.template GetBuffer<BufferType::ASCEND_L0C, float>(chunk_size * chunk_size * sizeof(float));

        // Core
        num_core = AscendC::GetBlockNum();
        core_idx = AscendC::GetBlockIdx();
        sub_block_idx = AscendC::GetSubBlockIdx();

        // 辅助矩阵缓存标记：0 为无效初值（cur 恒 >=16），首个 tile 必触发生成
        last_chunk_size = 0;
    }

    __aicore__ inline int64_t CeilDiv(int64_t a, int64_t b)
    {
        return (a + b - 1) / b;
    }

    __aicore__ inline void ub_to_l1(AscendC::LocalTensor<InDtype> l1Tensor,
                                    AscendC::LocalTensor<InDtype> ubTensor, uint32_t chunkSize)
    {
        AscendC::DataCopy(l1Tensor, ubTensor,
                          AscendC::DataCopyParams(1, chunkSize * chunkSize / 16, 0, 0));
    }

    __aicore__ inline void FixpipeL0cToL1(AscendC::LocalTensor<InDtype> l1Tensor,
                                          AscendC::LocalTensor<float> l0CTensor, uint32_t chunkSize)
    {
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::NZ> fixPipeParams;
        fixPipeParams.nSize = chunkSize;
        fixPipeParams.mSize = chunkSize;
        fixPipeParams.srcStride = chunkSize;
        fixPipeParams.dstStride = chunkSize * 16;
        if constexpr (std::is_same_v<InDtype, half>) {
            fixPipeParams.quantPre = QuantMode_t::F322F16;
        } else {
            fixPipeParams.quantPre = QuantMode_t::F322BF16;
        }
        AscendC::Fixpipe<InDtype, float, CFG_NZ_L1>(l1Tensor, l0CTensor, fixPipeParams);
    }

    __aicore__ inline void FixpipeL0cToUB(AscendC::LocalTensor<InDtype> ubTensor,
                                          AscendC::LocalTensor<float> l0CTensor, uint32_t chunkSize)
    {
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::NZ> fixPipeParams;
        fixPipeParams.nSize = chunkSize;
        fixPipeParams.mSize = chunkSize;
        fixPipeParams.srcStride = chunkSize;
        fixPipeParams.dstStride = chunkSize * 16;
        fixPipeParams.dualDstCtl = 0;
        fixPipeParams.subBlockId = 0;
        if constexpr (std::is_same_v<InDtype, half>) {
            fixPipeParams.quantPre = QuantMode_t::F322F16;
        } else {
            fixPipeParams.quantPre = QuantMode_t::F322BF16;
        }
        AscendC::Fixpipe<InDtype, float, CFG_NZ_UB>(ubTensor, l0CTensor, fixPipeParams);
    }

    __aicore__ inline int64_t ChunkAlign(int64_t cur_chunk)
    {
        if (cur_chunk <= 16)
            return 16;
        if (cur_chunk <= 32)
            return 32;
        if (cur_chunk <= 64)
            return 64;
        return 128;
    }

    // 结果写回：L0C(FP32, NZ) -> gm_out(ND, 行优先)。
    //   validRows = 有效行数（尾块 < curSize），只写 validRows 行避免跨序列覆写。
    //   curSize   = 对齐后 chunk 尺寸（= L0C/输出的列数与源行跨度）。
    //   dstStride = GM 物理行跨度。
    __aicore__ inline void FixpipeL0cToGM(AscendC::GlobalTensor<OutDtype> gmTensor,
                                          AscendC::LocalTensor<float> l0CTensor,
                                          uint32_t validRows, uint32_t curSize, uint32_t dstStride)
    {
        auto intriParams = AscendC::FixpipeParamsV220(curSize, validRows, curSize, dstStride, false);
        if constexpr (std::is_same_v<OutDtype, half>) {
            intriParams.quantPre = QuantMode_t::F322F16;
        } else {
            intriParams.quantPre = QuantMode_t::F322BF16;
        }
        AscendC::Fixpipe<OutDtype, float, AscendC::CFG_ROW_MAJOR>(gmTensor, l0CTensor, intriParams);
    }

    // MCH V2 MatmulToL0C（块对角操作数专用；非对角分形恒 0，V1/V2 等价）
    __aicore__ inline void MatmulToL0C(AscendC::LocalTensor<InDtype> l1A, AscendC::LocalTensor<InDtype> l1B,
                                       AscendC::LocalTensor<InDtype> l0A, AscendC::LocalTensor<InDtype> l0B,
                                       AscendC::LocalTensor<float> l0C, int64_t chunkSize, bool initC)
    {
        int64_t numFracs = chunkSize / 16;

        AscendC::LoadData2DParamsV2 loadDataParamsA;
        loadDataParamsA.mStartPosition = 0;
        loadDataParamsA.kStartPosition = 0;
        loadDataParamsA.mStep = numFracs;
        loadDataParamsA.kStep = numFracs;
        loadDataParamsA.srcStride = numFracs;
        loadDataParamsA.dstStride = numFracs;
        loadDataParamsA.ifTranspose = false;
        AscendC::LoadData(l0A, l1A, loadDataParamsA);

        AscendC::LoadData2DParamsV2 loadDataParamsB;
        loadDataParamsB.mStartPosition = 0;
        loadDataParamsB.kStartPosition = 0;
        loadDataParamsB.mStep = numFracs;
        loadDataParamsB.kStep = numFracs;
        loadDataParamsB.srcStride = numFracs;
        loadDataParamsB.dstStride = numFracs;
        loadDataParamsB.ifTranspose = true;
        AscendC::LoadData(l0B, l1B, loadDataParamsB);

        SetFlag<AscendC::HardEvent::MTE1_M>(0);   // LoadData(MTE1, 写 l0A/l0B) -> Mmad(M, 读 l0A/l0B)
        WaitFlag<AscendC::HardEvent::MTE1_M>(0);

        AscendC::MmadParams mmadParams;
        mmadParams.m = chunkSize;
        mmadParams.n = chunkSize;
        mmadParams.k = chunkSize;
        mmadParams.cmatrixInitVal = initC;
        mmadParams.cmatrixSource = false;
        mmadParams.unitFlag = 0;
        AscendC::Mmad(l0C, l0A, l0B, mmadParams);
    }

    // 按给定 cur_size 生成 NZ 块对角辅助矩阵（尾块安全）：单位阵 I、全零阵 Zero，并清零 ub_A。
    __aicore__ inline void AuxMatrixGen(int64_t cur_size)
    {
        uint64_t NUM_FRACS = cur_size / 16;
        uint64_t NUM_ITER = NUM_FRACS * 2;
        int32_t chunkElems = static_cast<int32_t>(cur_size * cur_size);

        Duplicate(ub_A, (InDtype)0, chunkElems);
        Duplicate(ub_I, (InDtype)0, chunkElems);
        for (uint64_t stripIdx = 0; stripIdx < NUM_ITER; stripIdx++) {
            uint64_t fracsIdx = stripIdx / 2;
            uint64_t oldEvenIdx = stripIdx % 2;
            uint64_t diagMask[2] = {
                DIAG_MASK_8X16[oldEvenIdx ? 0 : 1][0],
                DIAG_MASK_8X16[oldEvenIdx ? 0 : 1][1]
            };
            uint64_t UB_DIAG_I_OFF = fracsIdx * (cur_size + 16) * 16 + oldEvenIdx * 8 * 16;
            Duplicate(ub_I[UB_DIAG_I_OFF], (InDtype)1.0f, diagMask, 1, 1, 1);
        }
        Duplicate(ub_Zero, (InDtype)0, chunkElems);
    }

    // 由 loop_idx 计算该 tile 的 GM 偏移、对齐后 chunk 尺寸 cur_size 与有效行数 actual_size。
    __aicore__ inline void ComputeTile(int64_t loop_idx, int64_t &x_gm_offset,
                                       int64_t &cur_size, int64_t &actual_size)
    {
        int64_t seq_idx = 0;
        int64_t chunk_in_seq_idx = 0;
        int64_t head_idx = 0;
        int64_t chunk_idx = 0;
        int64_t local_seq_length = seq_length;
        int64_t local_chunk_num_in_seq = chunk_num_in_seq;

        if (mode == 0) { // BHTD: [B, H, T, BT]
            seq_idx = loop_idx / (chunk_num_in_seq * num_head);
            head_idx = (loop_idx / chunk_num_in_seq) % num_head;
            chunk_in_seq_idx = loop_idx % chunk_num_in_seq;
            x_gm_offset = seq_idx * num_head * seq_length * chunk_size +
                          head_idx * seq_length * chunk_size +
                          chunk_in_seq_idx * chunk_size * chunk_size;
        } else if (mode == 1) { // BSND: [B, T, H, BT]
            seq_idx = loop_idx / (chunk_num_in_seq * num_head);
            chunk_in_seq_idx = loop_idx % (chunk_num_in_seq * num_head) / num_head;
            head_idx = loop_idx % (chunk_num_in_seq * num_head) % num_head;
            x_gm_offset = seq_idx * seq_length * num_head * chunk_size +
                          chunk_in_seq_idx * chunk_size * num_head * chunk_size +
                          head_idx * chunk_size;
        } else { // TND varlen: [total_T, H, BT]; B = 1
            chunk_idx = loop_idx / num_head;
            head_idx = loop_idx % num_head;
            seq_idx = gm_chunk_indices.GetValue(chunk_idx * 2);
            chunk_in_seq_idx = gm_chunk_indices.GetValue(chunk_idx * 2 + 1);
            local_seq_length = gm_cu_seqlens.GetValue(seq_idx + 1) - gm_cu_seqlens.GetValue(seq_idx);
            local_chunk_num_in_seq = CeilDiv(local_seq_length, chunk_size);
            int64_t bos = gm_cu_seqlens.GetValue(seq_idx);
            x_gm_offset = (bos + chunk_in_seq_idx * chunk_size) * num_head * chunk_size +
                          head_idx * chunk_size;
        }

        bool is_last = (chunk_in_seq_idx == (local_chunk_num_in_seq - 1));
        actual_size = is_last ? (local_seq_length - chunk_in_seq_idx * chunk_size) : chunk_size;
        cur_size = is_last ? ChunkAlign(actual_size) : chunk_size;
    }

    // ---- AIV：MCH 单 tile 备数（含 MBH 所需的完整 -A 暂存）----
    __aicore__ inline void AivMchPrep(int64_t cur, int64_t actual_size, int64_t x_gm_offset, int64_t row_stride)
    {
        // [尾块辅助矩阵] 仅 chunk 尺寸变化时重建 NZ 单位/全零阵并重拷 l1_I
        if (cur != last_chunk_size) {
            AuxMatrixGen(cur);
            SetFlag<AscendC::HardEvent::V_MTE3>(0);   // AuxMatrixGen(V, 写 ub_I) -> ub_to_l1(MTE3, 读 ub_I)
            WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            ub_to_l1(l1_I, ub_I, static_cast<uint32_t>(cur));
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);   // ub_to_l1(MTE3) -> 下面对角块 gather(MTE2, 写 ub_A)
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
            last_chunk_size = cur;
        }

        // 对角 16x16 块 GM(ND) -> ub_A(NZ 块对角)；尾块只读 actual_size 行（逐分形裁剪）。
        uint16_t src_blk_stride = static_cast<uint16_t>(row_stride / 16 - 1);
        uint64_t num_valid_fracs = static_cast<uint64_t>(CeilDiv(actual_size, 16));
        for (uint64_t i = 0; i < num_valid_fracs; i++) {
            int64_t rows64 = actual_size - static_cast<int64_t>(i) * 16;
            uint16_t rows = static_cast<uint16_t>(rows64 >= 16 ? 16 : rows64);
            uint64_t srcOffset = i * (16 * (uint64_t)row_stride + 16);
            uint64_t dstOffset = i * ((uint64_t)cur * 16 + 16 * 16);
            AscendC::DataCopy(ub_A[dstOffset], gm_a[x_gm_offset + srcOffset],
                              AscendC::DataCopyParams(rows, 1, src_blk_stride, 0));
        }
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);   // gather(MTE2, 写 ub_A) -> ub_to_l1(MTE3, 读 ub_A)
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);

        ub_to_l1(l1_Y, ub_A, static_cast<uint32_t>(cur));        // l1_Y = A(块对角)
        SetFlag<AscendC::HardEvent::MTE3_V>(0);   // ub_to_l1(MTE3, 读 ub_A) -> Sub(V, 读 ub_A) [RAR/顺序，保证 gather 已消费]
        WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::Sub(ub_I_A, ub_I, ub_A, (int32_t)(cur * cur));  // I - A
        SetFlag<AscendC::HardEvent::V_MTE3>(0);   // Sub(V, 写 ub_I_A) -> ub_to_l1(MTE3, 读 ub_I_A)
        WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        ub_to_l1(l1_X, ub_I_A, static_cast<uint32_t>(cur));      // l1_X = I - A
        SetFlag<AscendC::HardEvent::MTE3_V>(0);   // ub_to_l1(MTE3) -> 下面 Duplicate ub_FullA(V) [顺序]
        WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(0x2);   // 数据就绪 -> AIC

        // MBH 预备（cur>16）：完整 A GM(ND)->ub_FullA(NZ)，尾块只读 actual_size 行（其余清 0），取负 -> l1_MNEG。
        if (cur > 16) {
            Duplicate(ub_FullA, (InDtype)0, (int32_t)(cur * cur));  // 清 padding 行
            SetFlag<AscendC::HardEvent::V_MTE2>(0);   // Duplicate(V, 写 ub_FullA) -> nd2nz DataCopy(MTE2, 写 ub_FullA) [WAW]
            WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::Nd2NzParams p;
            p.ndNum = 1;
            p.nValue = static_cast<uint32_t>(actual_size);
            p.dValue = static_cast<uint32_t>(cur);
            p.srcDValue = static_cast<uint32_t>(row_stride);
            p.srcNdMatrixStride = 0;
            p.dstNzNStride = 1;
            p.dstNzC0Stride = static_cast<uint16_t>(cur);
            p.dstNzMatrixStride = 0;
            AscendC::DataCopy(ub_FullA, gm_a[x_gm_offset], p);
            SetFlag<AscendC::HardEvent::MTE2_V>(0);   // nd2nz(MTE2, 写 ub_FullA) -> Muls(V, 读写 ub_FullA)
            WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::Muls(ub_FullA, ub_FullA, (InDtype)(-1.0f), (int32_t)(cur * cur));
            SetFlag<AscendC::HardEvent::V_MTE3>(0);   // Muls(V, 写 ub_FullA) -> ub_to_l1(MTE3, 读 ub_FullA)
            WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            ub_to_l1(l1_MNEG, ub_FullA, static_cast<uint32_t>(cur));  // l1_MNEG = -A
        }
    }

    // ---- AIC：MCH 牛顿迭代，求 16x16 对角块逆 ----
    // cur>16：结果 Fixpipe 暂存 ub_Res(UB,NZ) 供 MBH；cur==16：纯 MCH，直接写 gm_out（只写 actual_size 行）。
    __aicore__ inline void AicMchNewton(int64_t cur, int64_t actual_size, int64_t x_gm_offset, int64_t row_stride)
    {
        MatmulToL0C(l1_Y, l1_Y, l0a_Y, l0b_Y, l0c_Y, cur, true);
        SetFlag<AscendC::HardEvent::M_FIX>(0);
        WaitFlag<AscendC::HardEvent::M_FIX>(0);
        FixpipeL0cToL1(l1_Y, l0c_Y, cur);
        SetFlag<AscendC::HardEvent::FIX_MTE1>(0);

        MatmulToL0C(l1_I, l1_X, l0a_X, l0b_X, l0c_X, cur, true);
        SetFlag<AscendC::HardEvent::M_FIX>(1);
        WaitFlag<AscendC::HardEvent::M_FIX>(1);
        FixpipeL0cToL1(l1_X, l0c_X, cur);
        SetFlag<AscendC::HardEvent::FIX_MTE1>(1);

        WaitFlag<AscendC::HardEvent::FIX_MTE1>(0);
        WaitFlag<AscendC::HardEvent::FIX_MTE1>(1);
        MatmulToL0C(l1_X, l1_Y, l0a_X, l0b_X, l0c_X, cur, false);
        SetFlag<AscendC::HardEvent::M_FIX>(1);
        WaitFlag<AscendC::HardEvent::M_FIX>(1);
        FixpipeL0cToL1(l1_X, l0c_X, cur);
        SetFlag<AscendC::HardEvent::FIX_MTE1>(1);

        for (uint64_t iter = 0; iter < 2; iter++) {
            MatmulToL0C(l1_Y, l1_Y, l0a_Y, l0b_Y, l0c_Y, cur, true);
            SetFlag<AscendC::HardEvent::M_FIX>(0);
            WaitFlag<AscendC::HardEvent::M_FIX>(0);

            FixpipeL0cToL1(l1_Y, l0c_Y, cur);
            SetFlag<AscendC::HardEvent::FIX_MTE1>(0);

            WaitFlag<AscendC::HardEvent::FIX_MTE1>(0);
            WaitFlag<AscendC::HardEvent::FIX_MTE1>(1);

            MatmulToL0C(l1_X, l1_Y, l0a_X, l0b_X, l0c_X, cur, false);
            SetFlag<AscendC::HardEvent::M_FIX>(1);
            WaitFlag<AscendC::HardEvent::M_FIX>(1);

            if (iter == 1) {
                if (cur > 16) {
                    FixpipeL0cToUB(ub_Res, l0c_X, cur); // -> MBH 消费
                } else {
                    FixpipeL0cToGM(gm_out[x_gm_offset], l0c_X, static_cast<uint32_t>(actual_size),
                                   static_cast<uint32_t>(cur), static_cast<uint32_t>(row_stride));
                }
            } else {
                FixpipeL0cToL1(l1_X, l0c_X, cur);
                SetFlag<AscendC::HardEvent::FIX_MTE1>(1);
            }
        }
    }

    // ---- AIV：清 L1 槽（zeroUB->L1，整槽置零）----
    __aicore__ inline void ClearSlotUB(AscendC::LocalTensor<InDtype> l1Slot, int64_t cur)
    {
        AscendC::DataCopy(l1Slot, ub_Zero,
                          AscendC::DataCopyParams(1, (uint16_t)(cur * cur / 16), 0, 0));
    }

    // ---- AIV：从 ub_Res(NZ) 按块 raw UB->L1 提取选中对角块到 l1Slot ----
    __aicore__ inline void ExtractFromUB(AscendC::LocalTensor<InDtype> l1Slot,
                                         int64_t cur, int32_t blockSize, int32_t startBlock)
    {
        int32_t numFracsTotal = static_cast<int32_t>(cur / 16);
        int32_t numBlocks = static_cast<int32_t>(cur) / blockSize;
        int32_t fracsPerBlock = blockSize / 16;
        constexpr int32_t FRAC_LEN = 16 * 16;

        for (int32_t blk = startBlock; blk < numBlocks; blk += 2) {
            for (int32_t fi = 0; fi < fracsPerBlock; fi++) {
                for (int32_t fj = 0; fj < fracsPerBlock; fj++) {
                    int32_t fr = blk * fracsPerBlock + fi;
                    int32_t fc = blk * fracsPerBlock + fj;
                    int32_t off = (fc * numFracsTotal + fr) * FRAC_LEN;
                    AscendC::DataCopy(l1Slot[off], ub_Res[off],
                                      AscendC::DataCopyParams(1, (uint16_t)(FRAC_LEN / 16), 0, 0));
                }
            }
        }
    }

    // ---- AIC：MBH 矩乘（V1 约定，对 A 做块转置预交换，运行期 cur 参数化）----
    __aicore__ inline void MbhMatmulToL0C(AscendC::LocalTensor<InDtype> l1A, AscendC::LocalTensor<InDtype> l1B,
                                          AscendC::LocalTensor<InDtype> l0A, AscendC::LocalTensor<InDtype> l0B,
                                          AscendC::LocalTensor<float> l0C, int64_t cur, bool initC)
    {
        constexpr int32_t FRAC_LEN = 16 * 16;
        int32_t numFracs = static_cast<int32_t>(cur / 16);

        AscendC::LoadData2DParams loadParamsA;
        loadParamsA.startIndex = 0;
        loadParamsA.repeatTimes = numFracs;
        loadParamsA.srcStride = 1;
        loadParamsA.dstGap = 0;
        loadParamsA.ifTranspose = false;
        for (int32_t i = 0; i < numFracs; ++i) {
            AscendC::LoadData(l0A[i * numFracs * FRAC_LEN], l1A[i * numFracs * FRAC_LEN], loadParamsA);
        }
        AscendC::LoadData2DParams loadParamsB;
        loadParamsB.startIndex = 0;
        loadParamsB.repeatTimes = numFracs;
        loadParamsB.srcStride = numFracs;
        loadParamsB.dstGap = 0;
        loadParamsB.ifTranspose = true;
        for (int32_t i = 0; i < numFracs; ++i) {
            AscendC::LoadData(l0B[i * numFracs * FRAC_LEN], l1B[i * FRAC_LEN], loadParamsB);
        }
        SetFlag<AscendC::HardEvent::MTE1_M>(0); // LoadData(MTE1, 写 l0a_X/l0b_X) -> Mmad(M, 读 l0a_X/l0b_X)
        WaitFlag<AscendC::HardEvent::MTE1_M>(0);

        AscendC::MmadParams mmadParams;
        mmadParams.m = cur;
        mmadParams.n = cur;
        mmadParams.k = cur;
        mmadParams.cmatrixInitVal = initC;
        mmadParams.cmatrixSource = false;
        mmadParams.unitFlag = 0;
        AscendC::Mmad(l0C, l0A, l0B, mmadParams);
    }

    // ---- AIC：MBH 单层 B/C/E/G 四步矩乘 + 层间/末层写出 ----
    __aicore__ inline void MbhLevelAic(int64_t cur, int64_t actual_size, int64_t x_gm_offset,
                                       int64_t row_stride, bool lastLevel)
    {
        // step B: L0C = I × I（完整单位阵）
        MbhMatmulToL0C(l1_I, l1_I, l0a_X, l0b_X, l0c_X, cur, true);

        // step C: Y = drv(l1_X) × (-A) + I -> l1_Y
        MbhMatmulToL0C(l1_X, l1_MNEG, l0a_Y, l0b_Y, l0c_X, cur, false);
        SetFlag<AscendC::HardEvent::M_FIX>(0);   // Mmad(M, 写 l0c_X) -> Fixpipe(FIX, 读 l0c_X)
        WaitFlag<AscendC::HardEvent::M_FIX>(0);
        FixpipeL0cToL1(l1_Y, l0c_X, cur);

        AscendC::PipeBarrier<PIPE_ALL>();

        // step G: L0C += I × drv(l1_X)
        MbhMatmulToL0C(l1_I, l1_X, l0a_X, l0b_X, l0c_Y, cur, true);

        // step E: L0C = Y × oth(l1_INPUT)
        MbhMatmulToL0C(l1_Y, l1_INPUT, l0a_Y, l0b_Y, l0c_Y, cur, false);
        SetFlag<AscendC::HardEvent::M_FIX>(1);   // Mmad(M, 写 l0c_X) -> Fixpipe(FIX, 读 l0c_X)
        WaitFlag<AscendC::HardEvent::M_FIX>(1);

        if (!lastLevel) {
            FixpipeL0cToUB(ub_Res, l0c_Y, cur);   // 层间结果 -> ub_Res，下层提取
        } else {
            FixpipeL0cToGM(gm_out[x_gm_offset], l0c_Y,
                           static_cast<uint32_t>(actual_size), static_cast<uint32_t>(cur),
                           static_cast<uint32_t>(row_stride));
        }
    }

    // ---- Process：CrossCoreFlag + round-robin，MCH 与 MBH 每 tile 逐层握手 ----
    __aicore__ inline void Process()
    {
        int32_t drvStart = is_lower ? 1 : 0;
        int32_t othStart = is_lower ? 0 : 1;
        int64_t row_stride = (mode == 0) ? chunk_size : (num_head * chunk_size);

        if ASCEND_IS_AIV {
            if (sub_block_idx == 0) {
                for (int64_t loop_idx = core_idx / 2; loop_idx < chunk_num_total; loop_idx += num_core) {
                    int64_t x_gm_offset = 0;
                    int64_t cur = 0;
                    int64_t actual_size = 0;
                    ComputeTile(loop_idx, x_gm_offset, cur, actual_size);

                    // ---- MCH 备数 ----
                    AivMchPrep(cur, actual_size, x_gm_offset, row_stride);
                    AscendC::CrossCoreWaitFlag<0x4>(0x0);             // 等 AIC：MCH 完成（ub_Res 就绪 / cur==16 已写 GM）

                    // ---- MBH 各层：提取 drv/oth -> 握手 ----
                    for (int32_t blockSize = 16; blockSize < cur; blockSize *= 2) {
                        ClearSlotUB(l1_X, cur);
                        ClearSlotUB(l1_INPUT, cur);
                        ExtractFromUB(l1_X, cur, blockSize, drvStart);     // drv -> l1_X
                        ExtractFromUB(l1_INPUT, cur, blockSize, othStart); // oth -> l1_INPUT
                        AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(0x2);   // 提取就绪 -> AIC
                        AscendC::CrossCoreWaitFlag<0x4>(0x0);             // 等 AIC：本层矩乘/回写完成
                    }
                }
            }
        }

        if ASCEND_IS_AIC {
            for (int64_t loop_idx = core_idx; loop_idx < chunk_num_total; loop_idx += num_core) {
                int64_t x_gm_offset = 0;
                int64_t cur = 0;
                int64_t actual_size = 0;
                ComputeTile(loop_idx, x_gm_offset, cur, actual_size);

                // ---- MCH 牛顿 ----
                AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE1>(0x2);   // 等 AIV：数据就绪
                AicMchNewton(cur, actual_size, x_gm_offset, row_stride);
                AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(0x0);     // MCH 完成 -> AIV

                // ---- MBH 各层：等提取 -> 四步矩乘 -> 写出 -> 握手 ----
                for (int32_t blockSize = 16; blockSize < cur; blockSize *= 2) {
                    bool lastLevel = !(blockSize < cur / 2);
                    AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE1>(0x2);   // 等 AIV：提取就绪
                    MbhLevelAic(cur, actual_size, x_gm_offset, row_stride, lastLevel);
                    AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(0x0);     // 本层完成 -> AIV
                }
            }
        }
    }

private:
    // Gm
    AscendC::GlobalTensor<InDtype> gm_a;
    AscendC::GlobalTensor<int64_t> gm_cu_seqlens;
    AscendC::GlobalTensor<int64_t> gm_chunk_indices;
    AscendC::GlobalTensor<OutDtype> gm_out;

    // UB
    AscendC::LocalTensor<InDtype> ub_A;
    AscendC::LocalTensor<InDtype> ub_I_A;
    AscendC::LocalTensor<InDtype> ub_I;
    AscendC::LocalTensor<InDtype> ub_Zero;
    AscendC::LocalTensor<InDtype> ub_Res;
    AscendC::LocalTensor<InDtype> ub_FullA;   // MBH: 完整 -A 的 UB 暂存

    // L1
    AscendC::LocalTensor<InDtype> l1_X;
    AscendC::LocalTensor<InDtype> l1_Y;
    AscendC::LocalTensor<InDtype> l1_I;
    AscendC::LocalTensor<InDtype> l1_MNEG;    // MBH: -A（NZ）
    AscendC::LocalTensor<InDtype> l1_INPUT;   // MBH: 提取的 oth 对角块（NZ）

    // L0
    AscendC::LocalTensor<InDtype> l0a_X;
    AscendC::LocalTensor<InDtype> l0a_Y;
    AscendC::LocalTensor<InDtype> l0b_X;
    AscendC::LocalTensor<InDtype> l0b_Y;
    AscendC::LocalTensor<float> l0c_X;
    AscendC::LocalTensor<float> l0c_Y;

    // Tiling
    int64_t batch_size;
    int64_t seq_length;
    int64_t num_head;
    int64_t chunk_size;
    int64_t chunk_num_in_seq;
    int64_t chunk_num_total;
    int64_t mode;
    int64_t is_lower;
    int64_t tiles_per_core;

    // Core
    int64_t num_core;
    int64_t core_idx;
    int64_t sub_block_idx;

    // 辅助矩阵当前缓存对应的 chunk 尺寸（仅尺寸变化时重建 I/Zero/l1_I）
    int64_t last_chunk_size;
};


#endif  // SOLVE_TRI_ASCEND950_H
