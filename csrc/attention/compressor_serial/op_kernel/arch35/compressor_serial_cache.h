// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kernel_operator.h"
#include "compressor_serial_params.h"

namespace CompressorSerial {
using namespace AscendC;
class CacheUpdate {
public:
    __aicore__ inline void Run(GM_ADDR projection, GM_ADDR state, GM_ADDR ape,
        GM_ADDR table, GM_ADDR cu, GM_ADDR used, GM_ADDR start,
        __gm__ const CompressorSerialParams* p)
    {
        TPipe pipe;
        TBuf<TPosition::VECCALC> buffer;
        pipe.InitBuffer(buffer, 3 * 1024 * sizeof(float));
        auto value = buffer.Get<float>();
        auto bias = value[2048];
        GlobalTensor<float> src, dst, gmApe;
        src.SetGlobalBuffer((__gm__ float*)projection);
        dst.SetGlobalBuffer((__gm__ float*)state);
        gmApe.SetGlobalBuffer((__gm__ float*)ape);
        auto gmCu = (__gm__ const int32_t*)cu;
        auto gmUsed = (__gm__ const int32_t*)used;
        auto gmStart = (__gm__ const int32_t*)start;
        auto gmTable = (__gm__ const int32_t*)table;
        uint32_t width = p->coff * p->dim;
        for (int32_t row = GetBlockIdx(); row < p->tokens; row += GetBlockNum()) {
            int32_t b = 0, begin = 0;
            for (; b < p->batch; ++b) {
                begin = (p->flags & HAS_CU) ? gmCu[b] : b * p->sequence;
                int32_t n = (p->flags & HAS_USED) ? gmUsed[b] :
                    ((p->flags & HAS_CU) ? gmCu[b + 1] - begin : p->sequence);
                if (row >= begin && row < begin + n) break;
            }
            if (b >= p->batch) continue;
            int32_t pos = ((p->flags & HAS_START) ? gmStart[b] : 0) + row - begin;
            int32_t page, slot = pos % p->blockSize;
            if (p->cacheMode == 2) {
                page = (p->flags & HAS_TABLE) ? gmTable[b] : b;
            } else {
                int32_t logical = pos / p->blockSize;
                if (logical >= p->tableWidth) continue;
                page = (p->flags & HAS_TABLE) ? gmTable[b * p->tableWidth + logical]
                                            : b * p->tableWidth + logical;
                if ((p->flags & HAS_TABLE) && page == 0) continue;
            }
            if (page < 0 || page >= p->cacheBlocks) continue;
            DataCopy(value, src[row * p->projectionStride], 2 * width);
            DataCopy(bias, gmApe[(pos % p->ratio) * width], width);
            PipeBarrier<PIPE_ALL>();
            Add(value[width], value[width], bias, width);
            PipeBarrier<PIPE_ALL>();
            DataCopyExtParams copy{1, static_cast<uint32_t>(2 * width * sizeof(float)), 0, 0, 0};
            DataCopyPad(dst[page * p->cacheStride + slot * (2 * width)], value, copy);
            PipeBarrier<PIPE_ALL>();
        }
    }
};
} // namespace CompressorSerial
