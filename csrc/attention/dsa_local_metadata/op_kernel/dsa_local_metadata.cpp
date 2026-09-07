#include "kernel_operator.h"

using namespace AscendC;

// Fused single-core kernel for DSA context-parallel local token metadata.
// Mirrors build_local_metadata_triton semantics:
//   lqs = clamp(qsl[i],   local_start, local_end)
//   lqe = clamp(qsl[i+1], local_start, local_end)
//   lql = lqe - lqs
//   local_query_start_loc[i+1] = inclusive_cumsum(lql)[i]   (index 0 = 0)
//   local_seq_lens[i] = (lql > 0 && seq_lens[i] > 0)
//                       ? max(seq_lens[i] - (qsl[i+1] - lqe), 0) : 0
//   start_pos_out[i]   = seq_lens[i] - (qsl[i+1] - qsl[i])   [optional]
class DsaLocalMetadataKernel {
public:
    __aicore__ inline DsaLocalMetadataKernel() {}

    __aicore__ inline void Init(GM_ADDR queryStartLoc, GM_ADDR seqLens,
                                GM_ADDR outLocalQueryStartLoc, GM_ADDR outLocalSeqLens,
                                GM_ADDR outStartPos, const DsaLocalMetadataTilingData *tilingData)
    {
        numReqs = tilingData->numReqs;
        localStart = tilingData->localStart;
        localEnd = tilingData->localEnd;
        computeStartPos = tilingData->computeStartPos;

        gmQsl.SetGlobalBuffer((__gm__ int32_t *)queryStartLoc, numReqs + 1);
        gmSeqLens.SetGlobalBuffer((__gm__ int32_t *)seqLens, numReqs);
        gmOutQsl.SetGlobalBuffer((__gm__ int32_t *)outLocalQueryStartLoc, numReqs + 1);
        gmOutSeqLens.SetGlobalBuffer((__gm__ int32_t *)outLocalSeqLens, numReqs);
        gmOutStartPos.SetGlobalBuffer((__gm__ int32_t *)outStartPos, numReqs);

        uint32_t qslBytes = AlignUp((numReqs + 1) * sizeof(int32_t), ONE_BLK_SIZE);
        uint32_t slBytes = AlignUp(numReqs * sizeof(int32_t), ONE_BLK_SIZE);
        pipe.InitBuffer(qslBuf, qslBytes);
        pipe.InitBuffer(seqLensBuf, slBytes);
        pipe.InitBuffer(outQslBuf, qslBytes);
        pipe.InitBuffer(outSeqLensBuf, slBytes);
        pipe.InitBuffer(outStartPosBuf, slBytes);

        if (numReqs > 0) {
            LocalTensor<int32_t> lQsl = qslBuf.Get<int32_t>();
            DataCopyPadIn(lQsl, gmQsl, 0, numReqs + 1);
            LocalTensor<int32_t> lSeqLens = seqLensBuf.Get<int32_t>();
            DataCopyPadIn(lSeqLens, gmSeqLens, 0, numReqs);
        }
    }

    __aicore__ inline void Process()
    {
        LocalTensor<int32_t> lOutQsl = outQslBuf.Get<int32_t>();
        if (numReqs == 0) {
            lOutQsl.SetValue(0, 0);
            DataCopyPadOut(gmOutQsl, lOutQsl, 0, 1);
            return;
        }

        LocalTensor<int32_t> lQsl = qslBuf.Get<int32_t>();
        LocalTensor<int32_t> lSeqLens = seqLensBuf.Get<int32_t>();
        LocalTensor<int32_t> lOutSeqLens = outSeqLensBuf.Get<int32_t>();
        LocalTensor<int32_t> lOutStartPos = outStartPosBuf.Get<int32_t>();

        int32_t cum = 0;
        lOutQsl.SetValue(0, 0);
        for (uint32_t i = 0; i < numReqs; i++) {
            int32_t qs = lQsl.GetValue(i);
            int32_t qe = lQsl.GetValue(i + 1);
            int32_t sl = lSeqLens.GetValue(i);

            int32_t lqs = Clamp(qs);
            int32_t lqe = Clamp(qe);
            int32_t lql = lqe - lqs;

            cum += lql;
            lOutQsl.SetValue(i + 1, cum);

            int32_t offset = qe - lqe;
            int32_t lsl = 0;
            if (lql > 0 && sl > 0) {
                lsl = sl - offset;
                if (lsl < 0) {
                    lsl = 0;
                }
            }
            lOutSeqLens.SetValue(i, lsl);

            if (computeStartPos != 0) {
                lOutStartPos.SetValue(i, sl - (qe - qs));
            }
        }

        DataCopyPadOut(gmOutQsl, lOutQsl, 0, numReqs + 1);
        DataCopyPadOut(gmOutSeqLens, lOutSeqLens, 0, numReqs);
        if (computeStartPos != 0) {
            DataCopyPadOut(gmOutStartPos, lOutStartPos, 0, numReqs);
        }
    }

private:
    static __aicore__ inline uint32_t AlignUp(uint32_t x, uint32_t a)
    {
        return (x + a - 1) / a * a;
    }

    __aicore__ inline int32_t Clamp(int32_t v) const
    {
        if (v < localStart) {
            return localStart;
        }
        if (v > localEnd) {
            return localEnd;
        }
        return v;
    }

    // GM -> UB with arbitrary element count (no over-read past the tensor end).
    __aicore__ inline void DataCopyPadIn(LocalTensor<int32_t> &dst,
                                         GlobalTensor<int32_t> &src,
                                         int32_t gmOffset, int32_t count)
    {
        if (count <= 0) {
            return;
        }
        // sizeof(int32_t) is size_t (64-bit); cast it back to uint32_t so the
        // braced-init-list does not trigger -Wc++11-narrowing.
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count) * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(dst, src[gmOffset], copyParams, padParams);
        pipe_barrier(PIPE_ALL);
    }

    // UB -> GM with arbitrary element count (block-granular stores stay inside
    // the requested byte range thanks to DataCopyPad).
    __aicore__ inline void DataCopyPadOut(GlobalTensor<int32_t> &dst,
                                          LocalTensor<int32_t> &src,
                                          int32_t gmOffset, int32_t count)
    {
        if (count <= 0) {
            return;
        }
        pipe_barrier(PIPE_ALL);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count) * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(dst[gmOffset], src, copyParams);
        pipe_barrier(PIPE_ALL);
    }

    GlobalTensor<int32_t> gmQsl, gmSeqLens;
    GlobalTensor<int32_t> gmOutQsl, gmOutSeqLens, gmOutStartPos;

    uint32_t numReqs = 0;
    int32_t localStart = 0;
    int32_t localEnd = 0;
    uint32_t computeStartPos = 0;

    TPipe pipe;
    TBuf<QuePosition::VECCALC> qslBuf, seqLensBuf;
    TBuf<QuePosition::VECCALC> outQslBuf, outSeqLensBuf, outStartPosBuf;
};

extern "C" __global__ __aicore__ void dsa_local_metadata(
    GM_ADDR queryStartLoc, GM_ADDR seqLens,
    GM_ADDR outLocalQueryStartLoc, GM_ADDR outLocalSeqLens, GM_ADDR outStartPos,
    GM_ADDR workspace, GM_ADDR tiling)
{
    // Pure vector operator: declare AIV-only task type explicitly (required on
    // 910C/A3 where AIC and AIV are separate core groups).
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (g_coreType == AIC) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(1)) {
        DsaLocalMetadataKernel op;
        op.Init(queryStartLoc, seqLens, outLocalQueryStartLoc, outLocalSeqLens,
                outStartPos, &tilingData);
        op.Process();
    }
}
