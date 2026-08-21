/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernel_operator.h"

class DcpRemapCompactKernel {
public:
    __aicore__ inline DcpRemapCompactKernel() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, int64_t rows,
                                int64_t width, int64_t alignedWidth,
                                int64_t dcpRank, int64_t dcpSize,
                                int64_t interleaveSize)
    {
        inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(input), rows * width);
        outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(output), rows * width);
        rows_ = rows;
        width_ = width;
        alignedWidth_ = alignedWidth;
        dcpRank_ = static_cast<int32_t>(dcpRank);
        dcpSize_ = static_cast<int32_t>(dcpSize);
        interleaveSize_ = static_cast<int32_t>(interleaveSize);
        maskBytes_ = ((alignedWidth_ + 255) / 256) * 32;
        pipe.InitBuffer(inputBuf, alignedWidth_ * sizeof(int32_t));
        pipe.InitBuffer(outputBuf, alignedWidth_ * sizeof(int32_t));
        pipe.InitBuffer(inputFloatBuf, alignedWidth_ * sizeof(float));
        pipe.InitBuffer(blockFloatBuf, alignedWidth_ * sizeof(float));
        pipe.InitBuffer(groupFloatBuf, alignedWidth_ * sizeof(float));
        pipe.InitBuffer(tempFloatBuf, alignedWidth_ * sizeof(float));
        pipe.InitBuffer(gatheredFloatBuf, alignedWidth_ * sizeof(float));
        pipe.InitBuffer(ownerMaskBuf, maskBytes_);
        pipe.InitBuffer(validMaskBuf, maskBytes_);
        pipe.InitBuffer(combinedMaskBuf, maskBytes_);
    }

    __aicore__ inline void Process()
    {
        const int64_t block = AscendC::GetBlockIdx();
        const int64_t blocks = AscendC::GetBlockNum();
        for (int64_t row = block; row < rows_; row += blocks) {
            ProcessRow(row);
        }
    }

private:
    __aicore__ inline void ProcessRow(int64_t row)
    {
        AscendC::LocalTensor<int32_t> inputLocal = inputBuf.Get<int32_t>();
        AscendC::LocalTensor<int32_t> outputLocal = outputBuf.Get<int32_t>();
        AscendC::LocalTensor<float> inputFloat = inputFloatBuf.Get<float>();
        AscendC::LocalTensor<float> blockFloat = blockFloatBuf.Get<float>();
        AscendC::LocalTensor<float> groupFloat = groupFloatBuf.Get<float>();
        AscendC::LocalTensor<float> tempFloat = tempFloatBuf.Get<float>();
        AscendC::LocalTensor<float> gatheredFloat = gatheredFloatBuf.Get<float>();
        AscendC::LocalTensor<uint8_t> ownerMask = ownerMaskBuf.Get<uint8_t>();
        AscendC::LocalTensor<uint8_t> validMask = validMaskBuf.Get<uint8_t>();
        AscendC::LocalTensor<uint8_t> combinedMask = combinedMaskBuf.Get<uint8_t>();

        AscendC::DataCopyExtParams copyIn;
        copyIn.blockCount = 1;
        copyIn.blockLen = static_cast<uint32_t>(width_ * sizeof(int32_t));
        copyIn.srcStride = 0;
        copyIn.dstStride = 0;
        AscendC::DataCopyPadExtParams<int32_t> pad;
        pad.isPad = false;
        pad.leftPadding = 0;
        pad.rightPadding = 0;
        pad.paddingValue = -1;
        AscendC::Duplicate(inputLocal, static_cast<int32_t>(-1), alignedWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopyPad(inputLocal, inputGm[row * width_], copyIn, pad);

        AscendC::Duplicate(outputLocal, static_cast<int32_t>(-1), alignedWidth_);
        AscendC::PipeBarrier<PIPE_ALL>();

        AscendC::Cast(inputFloat, inputLocal, AscendC::RoundMode::CAST_ROUND,
                      alignedWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(blockFloat, inputFloat,
                      1.0f / static_cast<float>(interleaveSize_), alignedWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Floor(blockFloat, blockFloat, alignedWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(groupFloat, blockFloat,
                      1.0f / static_cast<float>(dcpSize_), alignedWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Floor(groupFloat, groupFloat, alignedWidth_);
        AscendC::PipeBarrier<PIPE_V>();

        // owner = block - floor(block / dcp_size) * dcp_size
        AscendC::Muls(tempFloat, groupFloat, static_cast<float>(dcpSize_),
                      alignedWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(tempFloat, blockFloat, tempFloat, alignedWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::CompareScalar(ownerMask, tempFloat,
                               static_cast<float>(dcpRank_),
                               AscendC::CMPMODE::EQ, alignedWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::CompareScalar(validMask, inputFloat, 0.0f,
                               AscendC::CMPMODE::GE, alignedWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::And(combinedMask.ReinterpretCast<uint16_t>(),
                     ownerMask.ReinterpretCast<uint16_t>(),
                     validMask.ReinterpretCast<uint16_t>(),
                     static_cast<int32_t>(alignedWidth_ / 16));
        AscendC::PipeBarrier<PIPE_V>();

        // local = group * interleave + (index - block * interleave)
        AscendC::Muls(tempFloat, blockFloat,
                      static_cast<float>(interleaveSize_), alignedWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(tempFloat, inputFloat, tempFloat, alignedWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(groupFloat, groupFloat,
                      static_cast<float>(interleaveSize_), alignedWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(tempFloat, groupFloat, tempFloat, alignedWidth_);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::GatherMaskParams gatherParams;
        gatherParams.repeatTimes = 1;
        gatherParams.src0BlockStride = 1;
        gatherParams.src0RepeatStride = 8;
        gatherParams.src1RepeatStride = 8;
        uint64_t validCount = 0;
        AscendC::GatherMask(gatheredFloat, tempFloat,
                            combinedMask.ReinterpretCast<uint32_t>(), true,
                            static_cast<uint32_t>(width_), gatherParams,
                            validCount);
        AscendC::PipeBarrier<PIPE_V>();
        if (validCount > 0) {
            AscendC::Cast(outputLocal, gatheredFloat,
                          AscendC::RoundMode::CAST_ROUND, validCount);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::DataCopyExtParams copyOut;
        copyOut.blockCount = 1;
        copyOut.blockLen = static_cast<uint32_t>(width_ * sizeof(int32_t));
        copyOut.srcStride = 0;
        copyOut.dstStride = 0;
        AscendC::DataCopyPad(outputGm[row * width_], outputLocal, copyOut);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputFloatBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> blockFloatBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> groupFloatBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tempFloatBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gatheredFloatBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ownerMaskBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> validMaskBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> combinedMaskBuf;
    AscendC::GlobalTensor<int32_t> inputGm;
    AscendC::GlobalTensor<int32_t> outputGm;
    int64_t rows_ = 0;
    int64_t width_ = 0;
    int64_t alignedWidth_ = 0;
    int32_t dcpRank_ = 0;
    int32_t dcpSize_ = 1;
    int32_t interleaveSize_ = 1;
    int64_t maskBytes_ = 0;
};

extern "C" __global__ __aicore__ void dcp_remap_compact_kernel(
    GM_ADDR input, GM_ADDR output, int64_t rows, int64_t width,
    int64_t alignedWidth, int64_t dcpRank, int64_t dcpSize,
    int64_t interleaveSize)
{
    DcpRemapCompactKernel op;
    op.Init(input, output, rows, width, alignedWidth, dcpRank, dcpSize,
            interleaveSize);
    op.Process();
}

namespace vllm_ascend {
extern void dcp_remap_compact_impl(void *stream, void *input, void *output,
                                   int64_t rows, int64_t width,
                                   int64_t alignedWidth, int64_t dcpRank,
                                   int64_t dcpSize, int64_t interleaveSize,
                                   uint32_t blockDim)
{
    dcp_remap_compact_kernel<<<blockDim, nullptr, stream>>>(
        input, output, rows, width, alignedWidth, dcpRank, dcpSize,
        interleaveSize);
}
} // namespace vllm_ascend
