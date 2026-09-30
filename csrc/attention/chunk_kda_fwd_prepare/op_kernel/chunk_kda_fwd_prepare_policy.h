/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef CHUNK_KDA_FWD_PREPARE_POLICY_H
#define CHUNK_KDA_FWD_PREPARE_POLICY_H

#include <cstdint>

#define CHUNK_KDA_FWD_PREPARE_TPL_BF16 10
#define CHUNK_KDA_FWD_PREPARE_TPL_FP32 30

#define CHUNK_KDA_FWD_PREPARE_NORM_IDENTITY 0
#define CHUNK_KDA_FWD_PREPARE_NORM_L2 1

#define CHUNK_KDA_FWD_PREPARE_BETA_RAW 0
#define CHUNK_KDA_FWD_PREPARE_BETA_SIGMOID 1
#define CHUNK_KDA_FWD_PREPARE_BETA_TWO_SIGMOID 2

#define CHUNK_KDA_FWD_PREPARE_GATE_PRECOMPUTED_STEP 0
#define CHUNK_KDA_FWD_PREPARE_GATE_SOFTPLUS 1
#define CHUNK_KDA_FWD_PREPARE_GATE_SAFE_SIGMOID 2

// 13 个输出形参始终保留；该模式只在编译期裁剪公开 GM 搬出，
// 不改变 Prepare 内部为后续 Stage 生成 workspace/L1 数据的计算。
#define CHUNK_KDA_FWD_PREPARE_OUTPUT_NONE 0
#define CHUNK_KDA_FWD_PREPARE_OUTPUT_RECOMPUTE 1
#define CHUNK_KDA_FWD_PREPARE_OUTPUT_SAVE 2
// 只额外搬出 Akk：Aqk/Akk 是公开必选输出，而 qHat/kHat/qRstd/kRstd/betaEff
// 只有反向重计算路径才需要。
#define CHUNK_KDA_FWD_PREPARE_OUTPUT_FORWARD 3

namespace KdaPrepare {

enum class QkNormMode : uint8_t {
    Identity = CHUNK_KDA_FWD_PREPARE_NORM_IDENTITY,
    L2 = CHUNK_KDA_FWD_PREPARE_NORM_L2,
};

enum class BetaMode : uint8_t {
    Raw = CHUNK_KDA_FWD_PREPARE_BETA_RAW,
    Sigmoid = CHUNK_KDA_FWD_PREPARE_BETA_SIGMOID,
    TwoSigmoid = CHUNK_KDA_FWD_PREPARE_BETA_TWO_SIGMOID,
};

enum class GateMode : uint8_t {
    PrecomputedStep = CHUNK_KDA_FWD_PREPARE_GATE_PRECOMPUTED_STEP,
    Softplus = CHUNK_KDA_FWD_PREPARE_GATE_SOFTPLUS,
    SafeSigmoid = CHUNK_KDA_FWD_PREPARE_GATE_SAFE_SIGMOID,
};

enum class OutputMode : uint8_t {
    // 只搬出 fwd_h/finalize 必需输出。
    None = CHUNK_KDA_FWD_PREPARE_OUTPUT_NONE,
    // 额外搬出 Akk（公开必选），但不搬出反向重计算中间量。
    Forward = CHUNK_KDA_FWD_PREPARE_OUTPUT_FORWARD,
    // 额外搬出反向重计算需要的 Akk、归一化结果和 beta_eff。
    Recompute = CHUNK_KDA_FWD_PREPARE_OUTPUT_RECOMPUTE,
    // 再额外搬出 qg，保留全部反向中间量。
    Save = CHUNK_KDA_FWD_PREPARE_OUTPUT_SAVE,
};

namespace ExpDomain {
constexpr float kLn2 = 0.69314718055994530942F;
constexpr float kRcpLn2 = 1.44269504088896340736F;
constexpr float kV1Bf16LowerBase2 = -126.0F;
constexpr float kV1Bf16UpperBase2 = 120.0F;
constexpr float kV6LowerBase2 = -80.0F;
constexpr float kV6UpperBase2 = 80.0F;
} // namespace ExpDomain

// 两套架构共同消费这一份域合同，host 测试只验证这里的纯数值选择。
template <bool USE_EXP2>
struct ExpDomainTraits {
    static constexpr bool useExp2 = USE_EXP2;
    static constexpr float stepScale = USE_EXP2 ? ExpDomain::kRcpLn2 : 1.0F;
    static constexpr float expInputScale = USE_EXP2 ? ExpDomain::kLn2 : 1.0F;

    static constexpr float StoredBound(float base2Bound)
    {
        return USE_EXP2 ? base2Bound : base2Bound * ExpDomain::kLn2;
    }

    static constexpr float ClampStored(float value, float base2Lower,
                                       float base2Upper)
    {
        const float lower = StoredBound(base2Lower);
        const float upper = StoredBound(base2Upper);
        return value < lower ? lower : (value > upper ? upper : value);
    }

    static constexpr float ToExpInput(float storedExponent)
    {
        return storedExponent * expInputScale;
    }
};

template <QkNormMode NORM_MODE, BetaMode BETA_MODE, GateMode GATE_MODE,
          bool USE_EXP2, bool SAFE_GATE, OutputMode OUTPUT_MODE>
struct PrepareCompilePolicy {
    static constexpr QkNormMode normMode = NORM_MODE;
    static constexpr BetaMode betaMode = BETA_MODE;
    static constexpr GateMode gateMode = GATE_MODE;
    static constexpr bool useExp2 = USE_EXP2;
    static constexpr bool safeGate = SAFE_GATE;
    static constexpr OutputMode outputMode = OUTPUT_MODE;
    // None/Forward/Recompute/Save 四档的搬出集合：
    //   outputAkk          : Akk（None 之外都写）
    //   outputRecomputeAux : qHat/kHat/qRstd/kRstd/betaEff
    //   outputQg           : qg
    static constexpr bool outputAkk = OUTPUT_MODE != OutputMode::None;
    static constexpr bool outputRecomputeAux =
        OUTPUT_MODE == OutputMode::Recompute || OUTPUT_MODE == OutputMode::Save;
    static constexpr bool outputQg = OUTPUT_MODE == OutputMode::Save;
};

namespace Shape {
constexpr uint32_t kChunkRows = 64;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kValueDim = 128;
constexpr uint32_t kSubChunkRows = 16;
constexpr uint32_t kSubChunkCount = 4;
constexpr uint32_t kHeadsPerGroup = 4;
constexpr uint32_t kAivPerAic = 2;
constexpr uint32_t kBf16MatrixBytes = 0x4000;    // [64,128] * BF16
constexpr uint32_t kGateMatrixBytes = 0x8000;    // [64,128] * FP32
constexpr uint32_t kScoreMatrixBytes = 0x4000;   // [64,64] * FP32
constexpr uint32_t kQuadrantFp32Bytes = 0x1000;  // [32,32] * FP32
constexpr uint32_t kQuadrantBf16Bytes = 0x0800;  // [32,32] * BF16
constexpr uint32_t kRstdBytes = 0x0100;          // [64] * FP32
constexpr uint32_t kPrefixRows[kSubChunkCount] = {16, 32, 48, 64};
constexpr uint32_t kKMinusBytes[kSubChunkCount] = {
    0x1000, 0x2000, 0x3000, 0x4000};
constexpr uint32_t kScorePayloadBytes = 0x12000; // 16K Q+ + 16K K+ + 40K K-
} // namespace Shape

namespace Workspace {
// 两套架构共用前 105 KiB：33 KiB context + 72 KiB stage payload。
constexpr uint32_t kQHat = 0x00000;
constexpr uint32_t kKHat = 0x04000;
constexpr uint32_t kBetaEff = 0x08000;
constexpr uint32_t kPayload = 0x08400;
constexpr uint32_t kSlotStride = 0x1A400;
constexpr uint32_t kArch35SlotStride = kSlotStride;

// Arch22 的 Cube 结果先经 GM 交给 Vector。该区只由 AIC 写：
// C2 放四段连续 raw score，V3 消费后 C4 复用前 4 KiB 放 T。
constexpr uint32_t kArch22CubeRelay = 0x1A400;
constexpr uint32_t kArch22RawScoreBytes = 0x5000;
constexpr uint32_t kArch22TRelay = kArch22CubeRelay;
constexpr uint32_t kArch22SlotStride =
    kArch22CubeRelay + kArch22RawScoreBytes;

// payload 在不同 Stage 原址换义，不在 UB/L1 内搬位。
constexpr uint32_t kX0 = 0x0000;
constexpr uint32_t kNegX1 = 0x2000;
constexpr uint32_t kB = 0x3000;
constexpr uint32_t kAkk = 0x5800;
constexpr uint32_t kKBetaG = 0x7800;
constexpr uint32_t kVBeta = 0xB800;

constexpr uint32_t kArch22SlotCount = 4;
constexpr uint32_t kArch22WorkgroupStride =
    kArch22SlotCount * kArch22SlotStride;
constexpr uint32_t kArch35SlotCount = 4;
constexpr uint32_t kArch35WorkgroupStride =
    kArch35SlotCount * kSlotStride;
} // namespace Workspace

namespace ScorePayload {
constexpr uint32_t kQPlus = 0x0000;
constexpr uint32_t kKPlus = 0x4000;
constexpr uint32_t kKMinus[Shape::kSubChunkCount] = {
    0x8000, 0x9000, 0xB000, 0xE000};
} // namespace ScorePayload

namespace L1 {
constexpr uint32_t kCapacity = 0x80000;
constexpr uint32_t kHeadLaneBytes = 0x12000;
constexpr uint32_t kHeadLane[Shape::kHeadsPerGroup] = {
    0x00000, 0x12000, 0x24000, 0x36000};
constexpr uint32_t kX0 = 0x48000;
constexpr uint32_t kNegX1 = 0x4C000;
constexpr uint32_t kT = 0x50000;
constexpr uint32_t kAkk = 0x54000;
constexpr uint32_t kQuadrantStride = 0x1000;
constexpr uint32_t kAkkStride = 0x2000;
// BF16 NZ 为 [N1,M1,M0,N0]；q10=(row 32,col 0) 的元素偏移。
constexpr uint32_t kAkkQ10Elements = 32 * 16;
constexpr uint32_t kPeak = 0x5C000;
} // namespace L1

namespace Arch35Ub {
constexpr uint32_t kCapacity = 0x3E000; // 248 KiB
constexpr uint32_t kComputeSlotBytes = 0x1C000;
constexpr uint32_t kStateBytes = 0x03000;
constexpr uint32_t kComputeSlotBase[2] = {0x00000, 0x1C000};
constexpr uint32_t kStateBase[2] = {0x38000, 0x3B000};

// V0/V1 的主计算区。
constexpr uint32_t kQ = 0x0000;
constexpr uint32_t kK = 0x4000;
constexpr uint32_t kG = 0x8000;
constexpr uint32_t kGateInput = 0x10000;
constexpr uint32_t kV0Work = 0x18000;
// V1 先生成四段 Kminus，再原位把 Q/K 改写为 Qplus/Kplus。
// 最终布局为 Qplus 16 KiB + Kplus 16 KiB + G 32 KiB + Kminus 40 KiB。
constexpr uint32_t kKMinus = 0x10000;

// V3 主计算区。
constexpr uint32_t kRawScore = 0x0000;
constexpr uint32_t kAqk = 0x5000;
constexpr uint32_t kLkk = 0x9000;
constexpr uint32_t kB = 0xD000;
constexpr uint32_t kX0 = 0xE000;
constexpr uint32_t kX1 = 0xF000;
constexpr uint32_t kNegX1 = 0x10000;
constexpr uint32_t kAkkPack = 0x11000;

// V6 主计算区。
constexpr uint32_t kQg = 0x0000;
constexpr uint32_t kKg = 0x4000;
constexpr uint32_t kVBeta = 0x8000;
constexpr uint32_t kGForPost = 0xC000;
constexpr uint32_t kKBetaG = 0x14000;
constexpr uint32_t kQgScaled = 0x18000;

// 每个 local head 的向量状态与临时区。
// sequence-major beta 按每行一个 32 Byte data block 搬入；head-major
// beta 仍从同一起点连续存放。后续状态避开完整的 2 KiB 暂存区。
constexpr uint32_t kBetaRaw = 0x0000;
constexpr uint32_t kBetaEff = 0x0800;
constexpr uint32_t kGRef[4] = {0x0A00, 0x0C00, 0x0E00, 0x1000};
constexpr uint32_t kQRstd = 0x1200;
constexpr uint32_t kKRstd = 0x1300;
constexpr uint32_t kGLast = 0x1400;
constexpr uint32_t kVfScratch = 0x1600;
} // namespace Arch35Ub

namespace Arch22Ub {
constexpr uint32_t kHardwareBytes = 0x30000; // 192 KiB
constexpr uint32_t kUsableBytes = 0x2E000;   // 末尾 8 KiB 保留
constexpr uint32_t kPrivateBytes = 0x12000;
constexpr uint32_t kPrivateBase[2] = {0x00000, 0x1C000};
constexpr uint32_t kSharedBase = 0x12000;
constexpr uint32_t kSharedG = kSharedBase;
constexpr uint32_t kSharedScratch = kSharedBase + 0x8000;

constexpr uint32_t kQ = 0x0000;
constexpr uint32_t kK = 0x4000;
constexpr uint32_t kGateOrKMinus = 0x8000;
constexpr uint32_t kKMinus[4] = {0x8000, 0x9000, 0xB000, 0xE000};
// A2/A3 先把 sequence-major beta 按 32 Byte 行距搬入，再 Gather 成
// 连续标量；head-major beta 直接写入 kBetaRaw。
constexpr uint32_t kBetaRawStrided = 0x10000;
constexpr uint32_t kBetaRaw = 0x10800;
constexpr uint32_t kBetaEff = 0x10A00;
constexpr uint32_t kDtBias = 0x10C00;
constexpr uint32_t kALog = 0x10E00;
constexpr uint32_t kQRstd = 0x10F00;
constexpr uint32_t kKRstd = 0x11000;
constexpr uint32_t kGLast = 0x11200;
constexpr uint32_t kBetaGatherOffsets = kSharedScratch + 0x1F00;

constexpr uint32_t kV3Aqk = 0x0000;
constexpr uint32_t kV3Lkk = 0x2000;
constexpr uint32_t kV3Leaf0 = 0x6000;
constexpr uint32_t kV3Leaf1 = 0x7000;
constexpr uint32_t kV3B = 0x8000;
constexpr uint32_t kV3X0 = 0x9000;
constexpr uint32_t kV3X1 = 0xA000;
constexpr uint32_t kV3NegX1 = 0xB000;
constexpr uint32_t kV3CompactRaw = 0xD000;
constexpr uint32_t kV3AkkPack = 0xE000;
// V3 读取 compact raw 时，betaEff 放在已经结束 G 生命周期的共享区。
// 不能复用 kBetaEff=0x10200，它位于 compact raw [0xD000,0x12000) 内。
constexpr uint32_t kV3BetaEff = kSharedBase;

constexpr uint32_t kV6Qg = 0x0000;
constexpr uint32_t kV6Kg = 0x4000;
constexpr uint32_t kV6VBeta = 0x8000;
constexpr uint32_t kV6KBetaG = 0xC000;
// V6 正序消费 FP32 G 后，将 BF16 qgScaled 压缩写入同一共享区低 16 KiB。
constexpr uint32_t kV6QgScaled = kSharedG;
} // namespace Arch22Ub

} // namespace KdaPrepare

#endif // CHUNK_KDA_FWD_PREPARE_POLICY_H
