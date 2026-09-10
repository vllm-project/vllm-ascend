#ifndef COMPRESSOR_METADATA_H
#define COMPRESSOR_METADATA_H

#include "kernel_operator.h"

namespace CompressorMetadata {
using namespace AscendC;

constexpr uint32_t SLOT_MAPPING_FLAT = 1;
constexpr uint32_t ALIGN_BYTES = 32;
constexpr uint32_t BUFFER_NUM = 2;
constexpr int64_t MAX_INT32_VALUE = 0x7FFFFFFFLL;

__aicore__ inline uint32_t MinU32(uint32_t lhs, uint32_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline uint32_t MaxU32(uint32_t lhs, uint32_t rhs)
{
    return lhs > rhs ? lhs : rhs;
}

__aicore__ inline uint32_t AlignUpU32(uint32_t value, uint32_t align)
{
    return (value + align - 1) / align * align;
}

__aicore__ inline uint32_t Int32BytesU32(uint32_t elems)
{
    return elems * static_cast<uint32_t>(sizeof(int32_t));
}

__aicore__ inline void PipeMte2ToS()
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventID);
    WaitFlag<HardEvent::MTE2_S>(eventID);
}

__aicore__ inline void PipeMte3ToS()
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventID);
    WaitFlag<HardEvent::MTE3_S>(eventID);
}

__aicore__ inline void PipeSToMte3()
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventID);
    WaitFlag<HardEvent::S_MTE3>(eventID);
}

__aicore__ inline void PipeVToMte3()
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventID);
    WaitFlag<HardEvent::V_MTE3>(eventID);
}

struct CompressorMetadataTilingData {
    uint32_t numRows;
    uint32_t numReqs;
    uint32_t actualNumReqs;
    uint32_t ropeRows;
    uint32_t ropeDim;
    uint32_t kvBlockTableStride;
    uint32_t kvBlockSize;
    uint32_t slotMappingFormat;
    uint32_t cmpRatio;
    uint32_t usedCoreNum;
    uint32_t tileRows;
    uint32_t ropeRowBytes;
    uint32_t ropeRowBytesAligned;
    uint32_t slotCols;
};

template <typename T>
class CompressorMetadataKernel {
public:
    __aicore__ inline CompressorMetadataKernel() {}

    __aicore__ inline void Init(CompressorMetadataTilingData* tilingData, TPipe* pipe)
    {
        numRows_ = tilingData->numRows;
        actualNumReqs_ = tilingData->actualNumReqs;
        ropeRows_ = tilingData->ropeRows;
        ropeDim_ = tilingData->ropeDim;
        kvBlockTableStride_ = tilingData->kvBlockTableStride;
        kvBlockSize_ = tilingData->kvBlockSize;
        slotMappingFormat_ = tilingData->slotMappingFormat;
        cmpRatio_ = tilingData->cmpRatio;
        tileRows_ = tilingData->tileRows;
        ropeRowBytes_ = tilingData->ropeRowBytes;
        ropeRowBytesAligned_ = tilingData->ropeRowBytesAligned;
        slotCols_ = tilingData->slotCols;
        reqTableBytes_ = AlignUpU32(Int32BytesU32(actualNumReqs_ + 1), ALIGN_BYTES);
        ropeDimAligned_ = ropeRowBytesAligned_ / sizeof(T);
        ropePadElems_ = ropeDimAligned_ - ropeDim_;
        slotTileBytes_ = AlignUpU32(Int32BytesU32(tileRows_ * slotCols_), ALIGN_BYTES);
        blockTableTileBytes_ = AlignUpU32(Int32BytesU32(tileRows_), ALIGN_BYTES);

        pipe->InitBuffer(prefixBuf_, reqTableBytes_);
        pipe->InitBuffer(startPosBuf_, reqTableBytes_);
        pipe->InitBuffer(cuSeqlensBuf_, reqTableBytes_);
        pipe->InitBuffer(blockTableBuf_, blockTableTileBytes_);
        pipe->InitBuffer(slotBuf_, slotTileBytes_);
        pipe->InitBuffer(cosQueue_, BUFFER_NUM, tileRows_ * ropeRowBytesAligned_);
        pipe->InitBuffer(sinQueue_, BUFFER_NUM, tileRows_ * ropeRowBytesAligned_);
    }

    __aicore__ inline void Process(
        GM_ADDR ropeCos,
        GM_ADDR ropeSin,
        GM_ADDR cuSeqlens,
        GM_ADDR startPos,
        GM_ADDR kvBlockTable,
        GM_ADDR compressCos,
        GM_ADDR compressSin,
        GM_ADDR slotMapping,
        GM_ADDR)
    {
        ropeCosGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(ropeCos));
        ropeSinGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(ropeSin));
        cuSeqlensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cuSeqlens));
        startPosGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(startPos));
        kvBlockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(kvBlockTable));
        compressCosGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(compressCos));
        compressSinGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(compressSin));
        slotMappingGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(slotMapping));

        LocalTensor<int32_t> prefixLocal = prefixBuf_.Get<int32_t>();
        LocalTensor<int32_t> startPosLocal = startPosBuf_.Get<int32_t>();
        LocalTensor<int32_t> cuSeqlensLocal = cuSeqlensBuf_.Get<int32_t>();
        BuildCompressedPrefix(prefixLocal, startPosLocal, cuSeqlensLocal);

        uint32_t validRows = static_cast<uint32_t>(prefixLocal.GetValue(actualNumReqs_));
        validRows = MinU32(validRows, numRows_);
        ProcessValidRows(prefixLocal, startPosLocal, validRows);
        ProcessPaddingRows(validRows);
    }

private:
    __aicore__ inline void BuildCompressedPrefix(
        LocalTensor<int32_t>& prefixLocal,
        LocalTensor<int32_t>& startPosLocal,
        LocalTensor<int32_t>& cuSeqlensLocal)
    {
        DataCopyExtParams startCopyParams{1, Int32BytesU32(actualNumReqs_), 0, 0, 0};
        DataCopyExtParams cuCopyParams{1, Int32BytesU32(actualNumReqs_ + 1), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
        DataCopyPad(startPosLocal, startPosGm_, startCopyParams, padParams);
        DataCopyPad(cuSeqlensLocal, cuSeqlensGm_, cuCopyParams, padParams);
        PipeMte2ToS();

        uint32_t prefix = 0;
        prefixLocal.SetValue(0, 0);
        for (uint32_t reqIdx = 0; reqIdx < actualNumReqs_; ++reqIdx) {
            int64_t startPos = static_cast<int64_t>(startPosLocal.GetValue(reqIdx));
            int64_t seqLen = static_cast<int64_t>(cuSeqlensLocal.GetValue(reqIdx + 1)) -
                             static_cast<int64_t>(cuSeqlensLocal.GetValue(reqIdx));
            uint32_t compressedRows = 0;
            if (startPos >= 0 && seqLen > 0) {
                compressedRows = static_cast<uint32_t>(((startPos + seqLen) / cmpRatio_) - (startPos / cmpRatio_));
            }
            prefix += compressedRows;
            prefixLocal.SetValue(reqIdx + 1, static_cast<int32_t>(prefix));
        }
    }

    __aicore__ inline void SplitRange(uint32_t totalRows, uint32_t& begin, uint32_t& end)
    {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t blockNum = MaxU32(GetBlockNum(), 1);
        uint32_t rowsPerBlock = (totalRows + blockNum - 1) / blockNum;
        begin = MinU32(blockIdx * rowsPerBlock, totalRows);
        end = MinU32(begin + rowsPerBlock, totalRows);
    }

    __aicore__ inline uint32_t FindRequest(LocalTensor<int32_t>& prefixLocal, uint32_t row)
    {
        uint32_t reqIdx = 0;
        while (reqIdx < actualNumReqs_ && static_cast<uint32_t>(prefixLocal.GetValue(reqIdx + 1)) <= row) {
            ++reqIdx;
        }
        return reqIdx;
    }

    __aicore__ inline void ProcessValidRows(
        LocalTensor<int32_t>& prefixLocal,
        LocalTensor<int32_t>& startPosLocal,
        uint32_t validRows)
    {
        uint32_t begin = 0;
        uint32_t end = 0;
        SplitRange(validRows, begin, end);
        if (begin >= end) {
            return;
        }

        uint32_t reqIdx = FindRequest(prefixLocal, begin);
        uint32_t row = begin;
        while (row < end && reqIdx < actualNumReqs_) {
            uint32_t reqBegin = static_cast<uint32_t>(prefixLocal.GetValue(reqIdx));
            uint32_t reqEnd = static_cast<uint32_t>(prefixLocal.GetValue(reqIdx + 1));
            if (row >= reqEnd) {
                ++reqIdx;
                continue;
            }
            uint32_t rowsInReq = MinU32(end - row, reqEnd - row);
            int64_t startPos = static_cast<int64_t>(startPosLocal.GetValue(reqIdx));
            uint32_t localCompressedIdx = row - reqBegin;
            // KV slot uses compressed position; RoPE uses the original group-start position.
            uint32_t compressedPos = static_cast<uint32_t>(startPos / cmpRatio_) + localCompressedIdx;
            ProcessRequestRows(reqIdx, row, compressedPos, rowsInReq);
            row += rowsInReq;
        }
    }

    __aicore__ inline void ProcessRequestRows(
        uint32_t reqIdx,
        uint32_t outputRow,
        uint32_t compressedPos,
        uint32_t rows)
    {
        while (rows > 0) {
            uint32_t blockOffset = compressedPos % kvBlockSize_;
            uint32_t rowsToBlockEnd = kvBlockSize_ - blockOffset;
            uint32_t curRows = MinU32(rows, tileRows_);
            curRows = MinU32(curRows, rowsToBlockEnd);
            ProcessTile(reqIdx, outputRow, compressedPos, curRows);
            outputRow += curRows;
            compressedPos += curRows;
            rows -= curRows;
        }
    }

    __aicore__ inline void ProcessTile(
        uint32_t reqIdx,
        uint32_t outputRow,
        uint32_t compressedPos,
        uint32_t rows)
    {
        uint32_t blockIdOffset = compressedPos / kvBlockSize_;
        if (blockIdOffset >= kvBlockTableStride_) {
            WriteInvalidTile(outputRow, rows);
            return;
        }

        LocalTensor<int32_t> blockTableLocal = blockTableBuf_.Get<int32_t>();
        DataCopyExtParams blockCopyParams{1, Int32BytesU32(1), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
        uint64_t blockTableGmOffset = static_cast<uint64_t>(reqIdx) * kvBlockTableStride_ + blockIdOffset;
        DataCopyPad(blockTableLocal, kvBlockTableGm_[blockTableGmOffset], blockCopyParams, padParams);
        PipeMte2ToS();

        int32_t blockId = blockTableLocal.GetValue(0);
        if (blockId < 0) {
            WriteInvalidTile(outputRow, rows);
            return;
        }

        uint32_t blockOffset = compressedPos % kvBlockSize_;
        if (slotMappingFormat_ == SLOT_MAPPING_FLAT) {
            int64_t maxSlot = static_cast<int64_t>(blockId) * kvBlockSize_ + blockOffset + rows - 1;
            if (maxSlot > MAX_INT32_VALUE) {
                WriteInvalidTile(outputRow, rows);
                return;
            }
        }

        uint64_t lastRopePos = (static_cast<uint64_t>(compressedPos) + rows - 1) * cmpRatio_;
        if (lastRopePos >= ropeRows_) {
            WriteInvalidTile(outputRow, rows);
            return;
        }

        CopyRopeTile(outputRow, compressedPos, rows);
        WriteSlotTile(outputRow, compressedPos, rows, blockId);
    }

    __aicore__ inline void CopyRopeTile(uint32_t outputRow, uint32_t compressedPos, uint32_t rows)
    {
        LocalTensor<T> cosLocal = cosQueue_.AllocTensor<T>();
        LocalTensor<T> sinLocal = sinQueue_.AllocTensor<T>();
        uint64_t ropePos = static_cast<uint64_t>(compressedPos) * cmpRatio_;
        uint32_t srcStride = (cmpRatio_ - 1) * ropeRowBytes_;

        DataCopyExtParams copyInParams{
            static_cast<uint16_t>(rows), ropeRowBytes_, srcStride, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(ropePadElems_), 0};
        DataCopyPad(cosLocal, ropeCosGm_[ropePos * ropeDim_], copyInParams, padParams);
        DataCopyPad(sinLocal, ropeSinGm_[ropePos * ropeDim_], copyInParams, padParams);
        PipeMte2ToS();

        DataCopyExtParams copyOutParams{
            static_cast<uint16_t>(rows), ropeRowBytes_, 0, 0, 0};
        uint64_t outputBase = static_cast<uint64_t>(outputRow) * ropeDim_;
        DataCopyPad(compressCosGm_[outputBase], cosLocal, copyOutParams);
        DataCopyPad(compressSinGm_[outputBase], sinLocal, copyOutParams);
        PipeMte3ToS();

        cosQueue_.FreeTensor<T>(cosLocal);
        sinQueue_.FreeTensor<T>(sinLocal);
    }

    __aicore__ inline void WriteSlotTile(
        uint32_t outputRow,
        uint32_t compressedPos,
        uint32_t rows,
        int32_t blockId)
    {
        LocalTensor<int32_t> slotLocal = slotBuf_.Get<int32_t>();
        int32_t blockOffset = static_cast<int32_t>(compressedPos % kvBlockSize_);
        if (slotMappingFormat_ == SLOT_MAPPING_FLAT) {
            int32_t slotBase = blockId * static_cast<int32_t>(kvBlockSize_) + blockOffset;
            for (uint32_t row = 0; row < rows; ++row) {
                slotLocal.SetValue(row, slotBase + static_cast<int32_t>(row));
            }
        } else {
            for (uint32_t row = 0; row < rows; ++row) {
                uint32_t slotOffset = row * slotCols_;
                slotLocal.SetValue(slotOffset, blockId);
                slotLocal.SetValue(slotOffset + 1, blockOffset + static_cast<int32_t>(row));
            }
        }

        DataCopyExtParams slotCopyParams{1, Int32BytesU32(rows * slotCols_), 0, 0, 0};
        PipeSToMte3();
        DataCopyPad(slotMappingGm_[static_cast<uint64_t>(outputRow) * slotCols_], slotLocal, slotCopyParams);
        PipeMte3ToS();
    }

    __aicore__ inline void WriteInvalidTile(uint32_t outputRow, uint32_t rows)
    {
        LocalTensor<T> cosLocal = cosQueue_.AllocTensor<T>();
        LocalTensor<T> sinLocal = sinQueue_.AllocTensor<T>();

        Duplicate<T>(cosLocal, static_cast<T>(1.0f), rows * ropeDimAligned_);
        Duplicate<T>(sinLocal, static_cast<T>(0.0f), rows * ropeDimAligned_);
        PipeVToMte3();

        DataCopyExtParams ropeCopyParams{
            static_cast<uint16_t>(rows), ropeRowBytes_, 0, 0, 0};
        uint64_t outputBase = static_cast<uint64_t>(outputRow) * ropeDim_;
        DataCopyPad(compressCosGm_[outputBase], cosLocal, ropeCopyParams);
        DataCopyPad(compressSinGm_[outputBase], sinLocal, ropeCopyParams);
        PipeMte3ToS();

        cosQueue_.FreeTensor<T>(cosLocal);
        sinQueue_.FreeTensor<T>(sinLocal);

        LocalTensor<int32_t> slotLocal = slotBuf_.Get<int32_t>();
        if (slotMappingFormat_ == SLOT_MAPPING_FLAT) {
            for (uint32_t row = 0; row < rows; ++row) {
                slotLocal.SetValue(row, -1);
            }
        } else {
            int32_t padOffset = static_cast<int32_t>(kvBlockSize_ - 1);
            for (uint32_t row = 0; row < rows; ++row) {
                uint32_t slotOffset = row * slotCols_;
                slotLocal.SetValue(slotOffset, -1);
                slotLocal.SetValue(slotOffset + 1, padOffset);
            }
        }
        PipeSToMte3();
        DataCopyExtParams slotCopyParams{1, Int32BytesU32(rows * slotCols_), 0, 0, 0};
        DataCopyPad(slotMappingGm_[static_cast<uint64_t>(outputRow) * slotCols_], slotLocal, slotCopyParams);
        PipeMte3ToS();
    }

    __aicore__ inline void ProcessPaddingRows(uint32_t validRows)
    {
        if (validRows >= numRows_) {
            return;
        }
        uint32_t padRows = numRows_ - validRows;
        uint32_t begin = 0;
        uint32_t end = 0;
        SplitRange(padRows, begin, end);
        uint32_t row = validRows + begin;
        uint32_t padEnd = validRows + end;
        while (row < padEnd) {
            uint32_t curRows = MinU32(tileRows_, padEnd - row);
            WriteInvalidTile(row, curRows);
            row += curRows;
        }
    }

    uint32_t numRows_{0};
    uint32_t actualNumReqs_{0};
    uint32_t ropeRows_{0};
    uint32_t ropeDim_{0};
    uint32_t kvBlockTableStride_{0};
    uint32_t kvBlockSize_{0};
    uint32_t slotMappingFormat_{0};
    uint32_t cmpRatio_{1};
    uint32_t tileRows_{1};
    uint32_t ropeRowBytes_{0};
    uint32_t ropeRowBytesAligned_{0};
    uint32_t slotCols_{1};
    uint32_t reqTableBytes_{0};
    uint32_t ropeDimAligned_{0};
    uint32_t ropePadElems_{0};
    uint32_t slotTileBytes_{0};
    uint32_t blockTableTileBytes_{0};

    TBuf<TPosition::VECCALC> prefixBuf_;
    TBuf<TPosition::VECCALC> startPosBuf_;
    TBuf<TPosition::VECCALC> cuSeqlensBuf_;
    TBuf<TPosition::VECCALC> blockTableBuf_;
    TBuf<TPosition::VECCALC> slotBuf_;
    TQue<TPosition::VECOUT, BUFFER_NUM> cosQueue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> sinQueue_;

    GlobalTensor<T> ropeCosGm_;
    GlobalTensor<T> ropeSinGm_;
    GlobalTensor<T> compressCosGm_;
    GlobalTensor<T> compressSinGm_;
    GlobalTensor<int32_t> cuSeqlensGm_;
    GlobalTensor<int32_t> startPosGm_;
    GlobalTensor<int32_t> kvBlockTableGm_;
    GlobalTensor<int32_t> slotMappingGm_;
};
}  // namespace CompressorMetadata

#endif
