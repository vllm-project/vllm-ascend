// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
#ifndef VLLM_ASCEND_GATING_EPLB_MAP_H
#define VLLM_ASCEND_GATING_EPLB_MAP_H

#include "kernel_operator.h"

namespace MoeGatingTopKHash {
using namespace AscendC;
// Map only the final output IDs: the logical IDs used to gather scores remain
// untouched. The table is the same periodic table as the standalone EPLB op.
class EplbMap {
public:
    __aicore__ inline void Init(GM_ADDR table, int64_t rows, int64_t experts,
                               int64_t k, AscendC::TPipe *pipe)
    {
        rows_ = rows;
        experts_ = experts;
        k_ = k;
        if (rows_ == 0) {
            return;
        }
        table_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(table));
        pipe->InitBuffer(rowQueue_, 1, AlignBytes(experts_));
        pipe->InitBuffer(offsetsBuf_, AlignBytes(k_));
    }

    __aicore__ inline void Apply(AscendC::LocalTensor<int32_t> ids, int64_t globalRow)
    {
        if (rows_ == 0) {
            return;
        }
        auto row = rowQueue_.AllocTensor<int32_t>();
        AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(experts_ * sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> padding{false, 0, 0, 0};
        AscendC::DataCopyPad(row, table_[(globalRow % rows_) * experts_], params, padding);
        rowQueue_.EnQue(row);
        row = rowQueue_.DeQue<int32_t>();
        auto offsets = offsetsBuf_.Get<int32_t>();
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(offsets, ids, static_cast<int32_t>(sizeof(int32_t)), k_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(ids, row, offsets.ReinterpretCast<uint32_t>(), 0, k_);
        // CopyOut consumes the modified UB IDs through MTE3.
        auto event = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event);
        rowQueue_.FreeTensor(row);
    }

private:
    __aicore__ inline uint32_t AlignBytes(int64_t elements)
    {
        constexpr int64_t blockBytes = 32;
        return (elements * sizeof(int32_t) + blockBytes - 1) / blockBytes * blockBytes;
    }
    int64_t rows_ = 0;
    int64_t experts_ = 0;
    int64_t k_ = 0;
    AscendC::GlobalTensor<int32_t> table_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> rowQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> offsetsBuf_;
};
} // namespace MoeGatingTopKHash
#endif
