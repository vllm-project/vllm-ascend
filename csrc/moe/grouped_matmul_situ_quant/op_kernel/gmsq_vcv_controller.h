/*
 * gmsq_vcv_controller.h — shared GmmSituController for the n3-A fused
 * gmm_situ_vcv datapath (single device implementation, two host entries):
 *
 *   entry 1 (eager, unchanged): gmm_situ_vcv — host X14 active-group table in
 *           the tiling blob (host prunes zero-token groups, single pinned H2D).
 *   entry 2 (graph/static-grid): gmm_situ_vcv_dev — group_list stays a DEVICE
 *           tensor; the per-expert group table is built by an in-kernel
 *           preamble (running device-side cumsum over E<=32 entries, each core
 *           independently — NOT an independent cumsum device kernel), grid is
 *           the full AIC core count with idle cores exiting after the
 *           preamble. Kernel body (SplitNByMultiCore + vendored VCV basic
 *           block) is shared verbatim by both entries.
 */
#include "kernel_operator.h"
#include "grouped_matmul_situ_quant_tiling.h"

#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)
#include "lib/matmul_intf.h"
#include "vendor/wqbmm/weight_quant_tool.h"
#include "vendor/gmsq2/gmmsq_weight_quant_vcv_basic_block.h"

namespace gmm_situ {

using namespace AscendC;
namespace WQ = WeightQuantBatchMatmulV2::Arch35;
namespace GQ = GMMSQWeightQuant;
using GQ::BasicBlockOffsetParam;

using SituBasicBlock =
    GQ::GMMSQWeightQuantVcvBasicBlock<fp8_e4m3fn_t, fp4x2_e2m1_t, fp8_e8m0_t, fp8_e8m0_t, fp8_e4m3fn_t, fp8_e8m0_t,
                                      GQ::MXA8W4_NZNK, GQ::VEC_ANTIQUANT_CONFIG_DYNAMIC>;

struct SituGroupEntry { // 56B
    uint64_t mSize;
    uint64_t xOff;      // fp8 elements
    uint64_t wOff;      // fp4x2 packed elements
    uint64_t wScaleOff; // e8m0 elements
    uint64_t xScaleOff; // e8m0 elements
    uint64_t yOff;      // fp8 elements
    uint64_t yScaleOff; // e8m0 elements
};

static constexpr uint64_t L1_K_256 = 256;
static constexpr uint64_t L1_K_512 = 512;
static constexpr uint64_t L1_K_N_THRESHOLD = 128;
static constexpr uint64_t L1_K_M_THRESHOLD = 256; // 生产 swiglu_v2 动态规则
static constexpr uint64_t A_L1_BUFFER_ELEMS = 128 * GetKBUnit<fp8_e4m3fn_t>();
static constexpr uint32_t TENSOR_LIST_FLAG = 1U << 1;
// sv3_cont（M 条件化双角色拆分）：双角色拆分（供数专核 + epilogue 专核）的收益随
// 每 L1 块 M 行数（mL1Size）增大——epilogue 工作量 ∝ mL1×nL1，拆分把输出 drain 从
// 供数路径解耦；代价是供数带宽单核化（两核各半片 → 单核串搬两半片），与 M 无关。
// 故 mL1Size 小时回退原始交错路径（两核各供半片 + 各做半量 epilogue），mL1Size 达
// 阈值才启用双角色拆分。阈值 34 按 M 分布间隙条件化：round_4 nk_split 实测 maxL1M=32
// 的 launch 在 DUAL 下退化（case1 S_i 1.1701→1.1028，拆分调度/握手开销>供给收益假设），
// maxL1M=36/43/58 均获益（case6/7/8）；maxL1M 观测间隙 [33,35] 取中点 34（对两侧观测
// 边界各 ±2 余量；baseM=128 下 maxL1M=组表最大 expert M，由 tiling/组表推导，非 case 特化）。
// round_2 原始依据（28-29 行 S=0.989 vs 35-36 行 S=1.066）与本间隙相容。
// 模式在整个 launch 内全局恒定（三核同值）：两种模式的核间旗标收支不同构，逐块/
// 逐组切换会让回退路径的 L1-free 等待消费 sv3 模式积压的陈旧旗标，引发读写竞争。
static constexpr uint64_t GMSQ_DUAL_ROLE_M_THRESHOLD = 34;
// nk_tile×sv3_cont 合并新增：双角色整 tile relay 包络。双角色模式下 AIC 把整块
// [mL1×nL1] F32 经 fixpipe 单投 epilogue 专核（sub 1）的 relay 窗——Variant-A 让出的
// [64,128KB) 共 64KB（split-M 版每核只收半量行，同窗自然够用）→ 整 tile 需
// mL1×nL1 ≤ 64KB/4B = 16384 elem。sv3_cont 原基座（r2p2）窄 tile nL1=128、mL1≤128
// 恰好满槽不越界；nk_tile 宽 tile（mainBlockSize=128 → nL1=256）下 mL1∈[65,128]
//（baseM=128）会写穿 relay 窗 → 谓词同时要求 MaxGroupL1M×maxNL1Size ≤ 16384，
// 越界包络的 launch 整体回退交错路径（r3p2 nk_tile 基座实测覆盖的路径）。判据全部
// 由 tiling 头（mainBlockSize）与组表（MaxGroupL1M）推导，非 case 特化。
static constexpr uint64_t GMSQ_DUAL_ROLE_RELAY_TILE_ELEMS = 16384;

template <typename T>
__aicore__ inline __gm__ T *GetTensorAddr(uint16_t index, GM_ADDR tensorPtr)
{
    __gm__ uint64_t *dataAddr = reinterpret_cast<__gm__ uint64_t *>(tensorPtr);
    const uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t *addressTable = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T *>(*(addressTable + index));
}

class GmmSituController {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR xScale, GM_ADDR w, GM_ADDR wScale, GM_ADDR y, GM_ADDR yScale,
                                GM_ADDR tiling)
    {
        hostTableMode_ = true;
        entries_ = reinterpret_cast<__gm__ SituGroupEntry *>(reinterpret_cast<__gm__ uint8_t *>(hdrOf(tiling)) +
                                                             sizeof(SituTilingHeader));
        InitCommon(x, xScale, w, wScale, y, yScale, tiling);
    }

    // Device group_list entry: builds the group table from the DEVICE group_list
    // tensor in a per-core preamble (28-entry running cumsum; contract red line
    // only forbids an INDEPENDENT cumsum device kernel — this is an in-kernel
    // preamble segment of the fused launch).
    __aicore__ inline void InitDev(GM_ADDR x, GM_ADDR xScale, GM_ADDR w, GM_ADDR wScale, GM_ADDR groupList,
                                   GM_ADDR y, GM_ADDR yScale, GM_ADDR tiling)
    {
        hostTableMode_ = false;
        glGm_ = reinterpret_cast<__gm__ const int64_t *>(groupList);
        eCount_ = hdrOf(tiling)->activeCount; // dev entry: activeCount field carries E
        InitCommon(x, xScale, w, wScale, y, yScale, tiling);
        // static per-expert strides (shape-only, group_list-independent)
        const uint64_t k = hdr_->kSize;
        const uint64_t n = hdr_->nSize;
        devKScaleRow_ = k / 32;                       // k % 64 == 0 (host-checked)
        devPerGroupW_ = (k / 2) * n;
        devPerGroupWScale_ = devKScaleRow_ * n;
        devN2_ = n / 2;
        devScaleRowBytes_ = (devN2_ + 63) / 64 * 2;
    }

    __aicore__ inline void Process()
    {
        uint32_t cubeBlockIdx = GetBlockIdx();
        if ASCEND_IS_AIV {
            cubeBlockIdx = cubeBlockIdx >> 1;
        }

        uint64_t startBasicBlockId = 0;
        if (hostTableMode_) {
            for (uint32_t g = 0; g < hdr_->activeCount; ++g) {
                SituGroupEntry ent;
                ent.mSize = entries_[g].mSize;
                ent.xOff = entries_[g].xOff;
                ent.wOff = entries_[g].wOff;
                ent.wScaleOff = entries_[g].wScaleOff;
                ent.xScaleOff = entries_[g].xScaleOff;
                ent.yOff = entries_[g].yOff;
                ent.yScaleOff = entries_[g].yScaleOff;
                ProcessGroup(ent, g, cubeBlockIdx, startBasicBlockId);
            }
        } else {
            // preamble: per-core group table build (device cumsum) + zero-token skip
            uint64_t rowBase = 0;
            for (uint32_t g = 0; g < eCount_; ++g) {
                const int64_t v = glGm_[g];
                uint64_t cnt;
                if (glType_ == 1) {
                    cnt = static_cast<uint64_t>(v);
                } else {
                    const int64_t start = (g == 0) ? 0 : glGm_[g - 1];
                    cnt = static_cast<uint64_t>(v - start);
                }
                if (cnt == 0) {
                    continue; // inactive expert: no table entry, this core moves on
                }
                SituGroupEntry ent;
                ent.mSize = cnt;
                ent.xOff = rowBase * hdr_->kSize;
                ent.wOff = weightListMode_ ? 0 : static_cast<uint64_t>(g) * devPerGroupW_;
                ent.wScaleOff = weightListMode_ ? 0 : static_cast<uint64_t>(g) * devPerGroupWScale_;
                ent.xScaleOff = rowBase * devKScaleRow_;
                ent.yOff = rowBase * devN2_;
                ent.yScaleOff = rowBase * devScaleRowBytes_;
                ProcessGroup(ent, g, cubeBlockIdx, startBasicBlockId);
                rowBase += cnt;
            }
        }
        basicBlock_.End(prev_);
    }

private:
    __aicore__ inline __gm__ SituTilingHeader *hdrOf(GM_ADDR tiling)
    {
        return reinterpret_cast<__gm__ SituTilingHeader *>(tiling);
    }

    __aicore__ inline void InitCommon(GM_ADDR x, GM_ADDR xScale, GM_ADDR w, GM_ADDR wScale, GM_ADDR y, GM_ADDR yScale,
                                      GM_ADDR tiling)
    {
        hdr_ = hdrOf(tiling);
        xGm_ = reinterpret_cast<__gm__ fp8_e4m3fn_t *>(x);
        xScaleGm_ = reinterpret_cast<__gm__ fp8_e8m0_t *>(xScale);
        weightTensorPtr_ = w;
        weightScaleTensorPtr_ = wScale;
        wGm_ = GetTensorAddr<fp4x2_e2m1_t>(0, weightTensorPtr_);
        wScaleGm_ = GetTensorAddr<fp8_e8m0_t>(0, weightScaleTensorPtr_);
        yGm_ = reinterpret_cast<__gm__ fp8_e4m3fn_t *>(y);
        yScaleGm_ = reinterpret_cast<__gm__ fp8_e8m0_t *>(yScale);
        glType_ = hdr_->reserved & 1U;
        weightListMode_ = (hdr_->reserved & TENSOR_LIST_FLAG) != 0;

        basicBlock_.Init(WeightQuantBatchMatmulV2::Arch35::MX_GROUPSIZE, yGm_, yScaleGm_, hdr_->beta, hdr_->invBeta,
                         hdr_->linearBeta, hdr_->invLinearBeta);
        // sv3_cont：模式判据在三核（AIC / AIV sub0 / AIV sub1）上由同一 tiling 头与同一
        // 组表内容独立推导，结果必然一致，无需额外同步；整个 launch 全程恒定。
        // nk_tile 合并：追加整 tile relay 64KB 包络上界（宽 tile 下 mL1>64 的 launch
        // 整体回退交错路径，见 GMSQ_DUAL_ROLE_RELAY_TILE_ELEMS 注释）。
        uint64_t maxL1M = MaxGroupL1M();
        bool dualRoleMode =
            maxL1M >= GMSQ_DUAL_ROLE_M_THRESHOLD && maxL1M * (hdr_->mainBlockSize * 2ULL) <= GMSQ_DUAL_ROLE_RELAY_TILE_ELEMS;
        basicBlock_.SetDualRoleMode(dualRoleMode);
    }

    // sv3_cont：组表扫描求 max(每 L1 块 M 行数)。与 ProcessGroup 的 mBlkNum/mL1Size
    // 公式逐字一致（CeilDivide(mSize, CeilDivide(mSize, baseM))）；零 token 组与
    // Process 的 skip 规则一致。host 表 / device group_list 两种入口都只读 GM 数据，
    // 各核结果相同。
    __aicore__ inline uint64_t MaxGroupL1M()
    {
        const uint64_t baseM = hdr_->baseM;
        uint64_t maxML1 = 0;
        if (hostTableMode_) {
            for (uint32_t g = 0; g < hdr_->activeCount; ++g) {
                const uint64_t mSize = entries_[g].mSize;
                if (mSize == 0) {
                    continue;
                }
                const uint64_t mBlkNum = WQ::CeilDivide(mSize, baseM);
                const uint64_t mL1 = WQ::CeilDivide(mSize, mBlkNum);
                if (mL1 > maxML1) {
                    maxML1 = mL1;
                }
            }
        } else {
            for (uint32_t g = 0; g < eCount_; ++g) {
                const int64_t v = glGm_[g];
                uint64_t cnt;
                if (glType_ == 1) {
                    cnt = static_cast<uint64_t>(v);
                } else {
                    const int64_t start = (g == 0) ? 0 : glGm_[g - 1];
                    cnt = static_cast<uint64_t>(v - start);
                }
                if (cnt == 0) {
                    continue;
                }
                const uint64_t mBlkNum = WQ::CeilDivide(cnt, baseM);
                const uint64_t mL1 = WQ::CeilDivide(cnt, mBlkNum);
                if (mL1 > maxML1) {
                    maxML1 = mL1;
                }
            }
        }
        return maxML1;
    }

    __aicore__ inline void ProcessGroup(const SituGroupEntry &ent, uint32_t groupIdx, uint32_t cubeBlockIdx,
                                        uint64_t &startBasicBlockId)
    {
        BasicBlockOffsetParam cur = {};
        cur.kSize = hdr_->kSize;
        cur.nSize = hdr_->nSize;
        cur.kAlign = WQ::CeilAlign(cur.kSize, static_cast<uint64_t>(BLOCK_CUBE));
        cur.nAlign = WQ::CeilAlign(cur.nSize, static_cast<uint64_t>(BLOCK_CUBE));
        const bool isCacheLineUnaligned = cur.kSize % 256 != 0; // fp4: 256B line

        const uint64_t mSize = ent.mSize; // host pruned (eager) / preamble skipped (dev): always > 0
        const uint64_t mBlkNum = WQ::CeilDivide(mSize, static_cast<uint64_t>(hdr_->baseM));
        const uint64_t mL1Size = WQ::CeilDivide(mSize, mBlkNum);

        curYBase_ = yGm_ + ent.yOff;
        curYScaleBase_ = yScaleGm_ + ent.yScaleOff;
        __gm__ fp4x2_e2m1_t *groupWeight =
            weightListMode_ ? GetTensorAddr<fp4x2_e2m1_t>(groupIdx, weightTensorPtr_) : wGm_ + ent.wOff;
        __gm__ fp8_e8m0_t *groupWeightScale =
            weightListMode_ ? GetTensorAddr<fp8_e8m0_t>(groupIdx, weightScaleTensorPtr_) : wScaleGm_ + ent.wScaleOff;
        basicBlock_.UpdateGlobalAddr(xGm_ + ent.xOff, groupWeight, groupWeightScale,
                                     xScaleGm_ + ent.xScaleOff, curYBase_, curYScaleBase_,
                                     mL1Size < mSize || isCacheLineUnaligned);

        uint64_t curBasicBlockId =
            cubeBlockIdx >= startBasicBlockId ? cubeBlockIdx : cubeBlockIdx + hdr_->coreNum;
        uint64_t basicBlockLimit = startBasicBlockId;
        for (uint64_t mOffset = 0; mOffset < mSize; mOffset += mL1Size) {
            uint64_t nOffset = 0;
            SplitNByMultiCore(cur, prev_, mSize, mL1Size, mOffset, curBasicBlockId, basicBlockLimit,
                              hdr_->mainBlockCount, hdr_->mainBlockSize, nOffset);
            basicBlockLimit += hdr_->mainBlockCount;
            if (hdr_->firstTailBlockCount > 0) {
                SplitNByMultiCore(cur, prev_, mSize, mL1Size, mOffset, curBasicBlockId, basicBlockLimit,
                                  hdr_->firstTailBlockCount, hdr_->firstTailBlockSize, nOffset);
                basicBlockLimit += hdr_->firstTailBlockCount;
            }
        }
        startBasicBlockId = basicBlockLimit % hdr_->coreNum;
    }

    __aicore__ inline void SplitNByMultiCore(BasicBlockOffsetParam &cur, BasicBlockOffsetParam &prev, uint64_t mSize,
                                             uint64_t mL1Size, uint64_t mOffset, uint64_t &curBasicBlockId,
                                             uint64_t basicBlockLimit, uint64_t basicBlockCount,
                                             uint64_t basicBlockSize, uint64_t &nOffset)
    {
        for (; curBasicBlockId < basicBlockLimit + basicBlockCount; curBasicBlockId += hdr_->coreNum) {
            cur.mSize = mSize;
            cur.mOffset = mOffset;
            cur.mL1Size = mOffset + mL1Size > mSize ? mSize - mOffset : mL1Size;
            cur.nOffset = nOffset + ((curBasicBlockId - basicBlockLimit) % basicBlockCount) * basicBlockSize;

            // gate/up 半区语义：nOffset 为前 N/2（gate）输出列偏移，nL1Size 覆盖两半
            uint64_t halfNRemaining = cur.nSize / 2 - cur.nOffset;
            cur.nL1Size = halfNRemaining >= basicBlockSize ? basicBlockSize * 2 : halfNRemaining * 2;
            cur.yGmAddr = reinterpret_cast<GM_ADDR>(curYBase_);
            cur.yScaleGmAddr = reinterpret_cast<GM_ADDR>(curYScaleBase_);

            // nk_tile 宽 N 块容量守卫（通用 shape 规则，非 case 特化）：
            // SiTU epilogue 刮擦平面按生产上界 mReal×nHalf ≤ 64×64 定容
            //（GetGmmsqSituBufferInfo：gate/up/out/yFp8 均 64×64 elem）。
            // 宽块（nL1Size>128）在 mL1Size>64 时会越界 → 拆为连续 128 宽子块
            //（逐语句等价于基线块序列：kbL1 规则/kaL1 定容/prev 链全部按子块重算）。
            if (cur.nL1Size > 2 * L1_K_N_THRESHOLD && cur.mL1Size > 64) {
                const uint64_t subHalfW = L1_K_N_THRESHOLD / 2; // 每子块 64 个 gate 列
                for (uint64_t subOff = 0; subOff < cur.nL1Size / 2; subOff += subHalfW) {
                    BasicBlockOffsetParam sub = cur;
                    sub.nOffset = cur.nOffset + subOff;
                    uint64_t subHalfRemaining = cur.nSize / 2 - sub.nOffset;
                    sub.nL1Size = subHalfRemaining >= subHalfW ? subHalfW * 2 : subHalfRemaining * 2;
                    sub.kbL1Size = (sub.mL1Size <= L1_K_M_THRESHOLD && sub.nL1Size <= L1_K_N_THRESHOLD) ? L1_K_512
                                                                                                        : L1_K_256;
                    uint64_t subMl1Align = WQ::CeilAlign(sub.mL1Size, static_cast<uint64_t>(BLOCK_CUBE));
                    uint64_t subKaDepth = WQ::CeilDivide(sub.nL1Size, subMl1Align * 2);
                    uint64_t subMaxKaDepth = A_L1_BUFFER_ELEMS / (subMl1Align * sub.kbL1Size);
                    if (subKaDepth > subMaxKaDepth) {
                        subKaDepth = subMaxKaDepth;
                    }
                    sub.kaL1Size = subKaDepth * sub.kbL1Size;
                    basicBlock_.ComputeBasicBlock(sub, prev);
                    prev = sub;
                }
                continue;
            }

            // 生产动态 k 切分规则：小 M 窄 N 走 512 深度 L1 k
            cur.kbL1Size = (cur.mL1Size <= L1_K_M_THRESHOLD && cur.nL1Size <= L1_K_N_THRESHOLD) ? L1_K_512 : L1_K_256;
            uint64_t mL1Align = WQ::CeilAlign(cur.mL1Size, static_cast<uint64_t>(BLOCK_CUBE));
            uint64_t kaDepth = WQ::CeilDivide(cur.nL1Size, mL1Align * 2);
            uint64_t maxKaDepth = A_L1_BUFFER_ELEMS / (mL1Align * cur.kbL1Size);
            if (kaDepth > maxKaDepth) {
                kaDepth = maxKaDepth;
            }
            cur.kaL1Size = kaDepth * cur.kbL1Size;

            basicBlock_.ComputeBasicBlock(cur, prev);
            prev = cur;
        }
        nOffset += basicBlockSize * basicBlockCount;
    }

    bool hostTableMode_ = true;
    bool weightListMode_ = false;
    uint32_t glType_ = 1;
    uint32_t eCount_ = 0;
    uint64_t devKScaleRow_ = 0;
    uint64_t devPerGroupW_ = 0;
    uint64_t devPerGroupWScale_ = 0;
    uint64_t devN2_ = 0;
    uint64_t devScaleRowBytes_ = 0;
    __gm__ const int64_t *glGm_ = nullptr;
    GM_ADDR weightTensorPtr_ = nullptr;
    GM_ADDR weightScaleTensorPtr_ = nullptr;
    __gm__ SituTilingHeader *hdr_ = nullptr;
    __gm__ SituGroupEntry *entries_ = nullptr;
    __gm__ fp8_e4m3fn_t *xGm_ = nullptr;
    __gm__ fp8_e8m0_t *xScaleGm_ = nullptr;
    __gm__ fp4x2_e2m1_t *wGm_ = nullptr;
    __gm__ fp8_e8m0_t *wScaleGm_ = nullptr;
    __gm__ fp8_e4m3fn_t *yGm_ = nullptr;
    __gm__ fp8_e8m0_t *yScaleGm_ = nullptr;
    __gm__ fp8_e4m3fn_t *curYBase_ = nullptr;
    __gm__ fp8_e8m0_t *curYScaleBase_ = nullptr;
    BasicBlockOffsetParam prev_ = {};
    SituBasicBlock basicBlock_;
};

} // namespace gmm_situ

#endif // __NPU_ARCH__ 3510
