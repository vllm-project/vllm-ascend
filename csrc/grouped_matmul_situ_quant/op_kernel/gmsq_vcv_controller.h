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

// ---- tiling blob layout (must match op_host/{gmm_situ_vcv,gmm_situ_vcv_dev}.cpp) ----
struct SituTilingHeader { // 64B
    uint32_t coreNum;
    uint32_t activeCount; // eager: pruned active-group count; dev entry: E (full expert count)
    uint32_t kSize;
    uint32_t nSize;
    uint32_t baseM;
    uint32_t mainBlockSize;
    uint32_t firstTailBlockSize;
    uint32_t reserved;
    uint64_t mainBlockCount;
    uint64_t firstTailBlockCount;
    float beta;
    float invBeta;
    float linearBeta;
    float invLinearBeta;
};

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
                ProcessGroup(ent, cubeBlockIdx, startBasicBlockId);
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
                ent.wOff = static_cast<uint64_t>(g) * devPerGroupW_;
                ent.wScaleOff = static_cast<uint64_t>(g) * devPerGroupWScale_;
                ent.xScaleOff = rowBase * devKScaleRow_;
                ent.yOff = rowBase * devN2_;
                ent.yScaleOff = rowBase * devScaleRowBytes_;
                ProcessGroup(ent, cubeBlockIdx, startBasicBlockId);
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
        wGm_ = reinterpret_cast<__gm__ fp4x2_e2m1_t *>(w);
        wScaleGm_ = reinterpret_cast<__gm__ fp8_e8m0_t *>(wScale);
        yGm_ = reinterpret_cast<__gm__ fp8_e4m3fn_t *>(y);
        yScaleGm_ = reinterpret_cast<__gm__ fp8_e8m0_t *>(yScale);
        glType_ = hdr_->reserved; // dev entry reuses the reserved u32 slot for groupListType

        basicBlock_.Init(WeightQuantBatchMatmulV2::Arch35::MX_GROUPSIZE, yGm_, yScaleGm_, hdr_->beta, hdr_->invBeta,
                         hdr_->linearBeta, hdr_->invLinearBeta);
    }

    __aicore__ inline void ProcessGroup(const SituGroupEntry &ent, uint32_t cubeBlockIdx,
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
        basicBlock_.UpdateGlobalAddr(xGm_ + ent.xOff, wGm_ + ent.wOff, wScaleGm_ + ent.wScaleOff,
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
    uint32_t glType_ = 1;
    uint32_t eCount_ = 0;
    uint64_t devKScaleRow_ = 0;
    uint64_t devPerGroupW_ = 0;
    uint64_t devPerGroupWScale_ = 0;
    uint64_t devN2_ = 0;
    uint64_t devScaleRowBytes_ = 0;
    __gm__ const int64_t *glGm_ = nullptr;
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
