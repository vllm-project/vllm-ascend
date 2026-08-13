/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#ifndef CATEGORICAL_SAMPLE_H
#define CATEGORICAL_SAMPLE_H

#include "kernel_operator.h"

namespace CategoricalSample {
using namespace AscendC;

constexpr float POS_INFINITY = __builtin_inff();
constexpr float NEG_INFINITY = -__builtin_inff();
// Standard Philox4x32-10 multipliers and Weyl key increments.
constexpr uint32_t PHILOX_M0 = 0xD2511F53U;
constexpr uint32_t PHILOX_M1 = 0xCD9E8D57U;
constexpr uint32_t PHILOX_W0 = 0x9E3779B9U;
constexpr uint32_t PHILOX_W1 = 0xBB67AE85U;
constexpr float UINT24_TO_UNIT = 1.0f / 16777216.0f;
constexpr uint32_t VECTOR_ALIGNMENT_ELEMENTS = 256;
constexpr int32_t FP64_WEIGHT_FRACTION_BITS = 42;
constexpr int32_t FP32_ABS_MASK = 0x7FFFFFFF;
constexpr int32_t FP32_EXP_MASK = 0x7F800000;

#define CATEGORICAL_SAMPLE_CHECK(condition, message) \
    do {                                             \
        if (!(condition)) {                         \
            AscendC::AssertPrint(message);           \
            AscendC::Trap();                         \
        }                                            \
    } while (0)

struct CategoricalSampleTilingData {
    uint32_t numRows;
    uint32_t vocabSize;
    uint32_t numRequests;
    uint32_t rowStride;
    uint32_t outputProcessedLogitsStride;
    uint32_t outputProcessedLogitsNumCols;
    uint32_t tileElements;
    uint32_t tileCount;
    uint32_t hasOutputProcessedLogits;
    uint32_t hasOutputProcessedLogitsCol;
    uint32_t outputProcessedLogitsColPerToken;
    uint32_t applyTemperature;
    uint32_t returnLse;
    uint32_t useFp64;
};

__aicore__ inline uint32_t MinU32(uint32_t lhs, uint32_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline uint32_t AlignUpU32(uint32_t value, uint32_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

__aicore__ inline void PipeVToS()
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventId);
    WaitFlag<HardEvent::V_S>(eventId);
}

__aicore__ inline void PipeSToV()
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventId);
    WaitFlag<HardEvent::S_V>(eventId);
}

__aicore__ inline uint32_t MulHi(uint32_t lhs, uint32_t rhs)
{
    return static_cast<uint32_t>((static_cast<uint64_t>(lhs) * static_cast<uint64_t>(rhs)) >> 32U);
}

__aicore__ inline uint32_t MulLo(uint32_t lhs, uint32_t rhs)
{
    return static_cast<uint32_t>(static_cast<uint64_t>(lhs) * static_cast<uint64_t>(rhs));
}

__aicore__ inline void PhiloxRandom(int64_t seed, int64_t position, uint32_t& random0, uint32_t& random1)
{
    uint32_t counter0 = static_cast<uint32_t>(position);
    uint32_t counter1 = static_cast<uint32_t>(static_cast<uint64_t>(position) >> 32U);
    uint32_t counter2 = 0;
    uint32_t counter3 = 0;
    uint32_t key0 = static_cast<uint32_t>(seed);
    uint32_t key1 = static_cast<uint32_t>(static_cast<uint64_t>(seed) >> 32U);

    for (uint32_t round = 0; round < 10; ++round) {
        const uint32_t hi0 = MulHi(PHILOX_M0, counter0);
        const uint32_t lo0 = MulLo(PHILOX_M0, counter0);
        const uint32_t hi1 = MulHi(PHILOX_M1, counter2);
        const uint32_t lo1 = MulLo(PHILOX_M1, counter2);
        const uint32_t next0 = hi1 ^ counter1 ^ key0;
        const uint32_t next1 = lo1;
        const uint32_t next2 = hi0 ^ counter3 ^ key1;
        const uint32_t next3 = lo0;
        counter0 = next0;
        counter1 = next1;
        counter2 = next2;
        counter3 = next3;
        key0 += PHILOX_W0;
        key1 += PHILOX_W1;
    }

    random0 = counter0;
    random1 = counter1;
}

__aicore__ inline int32_t PhiloxRandom24(int64_t seed, int64_t position)
{
    uint32_t random0 = 0;
    uint32_t random1 = 0;
    PhiloxRandom(seed, position, random0, random1);
    return static_cast<int32_t>(random0 >> 8U);
}

__aicore__ inline uint64_t PhiloxRandom64(int64_t seed, int64_t position)
{
    uint32_t random0 = 0;
    uint32_t random1 = 0;
    PhiloxRandom(seed, position, random0, random1);
    return (static_cast<uint64_t>(random1) << 32U) | static_cast<uint64_t>(random0);
}

__aicore__ inline uint64_t MulHigh64(uint64_t lhs, uint64_t rhs)
{
    const uint64_t lhsLow = static_cast<uint32_t>(lhs);
    const uint64_t lhsHigh = lhs >> 32U;
    const uint64_t rhsLow = static_cast<uint32_t>(rhs);
    const uint64_t rhsHigh = rhs >> 32U;
    const uint64_t lowProduct = lhsLow * rhsLow;
    const uint64_t middle = lhsHigh * rhsLow + (lowProduct >> 32U);
    const uint64_t middleLow = middle & 0xFFFFFFFFULL;
    const uint64_t middleHigh = middle >> 32U;
    const uint64_t upperMiddle = middleLow + lhsLow * rhsHigh;
    return lhsHigh * rhsHigh + middleHigh + (upperMiddle >> 32U);
}

template <typename T>
class CategoricalSampleKernel {
public:
    __aicore__ inline CategoricalSampleKernel() {}

    __aicore__ inline void Init(
        GM_ADDR processedLogits,
        GM_ADDR expandedIdxMapping,
        GM_ADDR temperature,
        GM_ADDR seed,
        GM_ADDR pos,
        GM_ADDR outputProcessedLogits,
        GM_ADDR outputProcessedLogitsCol,
        GM_ADDR sampledTokenIds,
        GM_ADDR lse,
        CategoricalSampleTilingData* tilingData,
        TPipe* pipe)
    {
        numRows_ = tilingData->numRows;
        vocabSize_ = tilingData->vocabSize;
        numRequests_ = tilingData->numRequests;
        rowStride_ = tilingData->rowStride;
        outputProcessedLogitsStride_ = tilingData->outputProcessedLogitsStride;
        outputProcessedLogitsNumCols_ = tilingData->outputProcessedLogitsNumCols;
        tileElements_ = tilingData->tileElements;
        tileCount_ = tilingData->tileCount;
        hasOutputProcessedLogits_ = tilingData->hasOutputProcessedLogits != 0;
        hasOutputProcessedLogitsCol_ = tilingData->hasOutputProcessedLogitsCol != 0;
        outputProcessedLogitsColPerToken_ = tilingData->outputProcessedLogitsColPerToken != 0;
        applyTemperature_ = tilingData->applyTemperature != 0;
        returnLse_ = tilingData->returnLse != 0;
        useFp64_ = tilingData->useFp64 != 0;
        pipe_ = pipe;

        processedLogitsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(processedLogits), (numRows_ - 1) * rowStride_ + vocabSize_);
        expandedIdxMappingGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(expandedIdxMapping), numRows_);
        temperatureGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(temperature), numRequests_);
        seedGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(seed), numRequests_);
        posGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(pos), numRows_);
        if (hasOutputProcessedLogits_) {
            outputProcessedLogitsGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ float*>(outputProcessedLogits),
                (numRequests_ - 1) * outputProcessedLogitsStride_ +
                    outputProcessedLogitsNumCols_ * vocabSize_);
        }
        if (hasOutputProcessedLogitsCol_) {
            outputProcessedLogitsColGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ int32_t*>(outputProcessedLogitsCol),
                outputProcessedLogitsColPerToken_ ? numRows_ : 1);
        }
        sampledTokenIdsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(sampledTokenIds), numRows_);
        if (returnLse_) {
            lseGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(lse), numRows_);
        }

        pipe_->InitBuffer(inputQueue_, 1, tileElements_ * sizeof(T));
        pipe_->InitBuffer(logitsFloatBuf_, tileElements_ * sizeof(float));
        pipe_->InitBuffer(workBuf_, tileElements_ * sizeof(float));
        pipe_->InitBuffer(nanValuesBuf_, tileElements_ * sizeof(float));
        pipe_->InitBuffer(nanMaskBuf_, tileElements_ * sizeof(uint8_t));
        pipe_->InitBuffer(scalarBuf_, 64);
        pipe_->InitBuffer(scalarIntBuf_, 32);
        pipe_->InitBuffer(tileSumsBuf_, AlignUpU32(tileCount_ * sizeof(float), 32));
        pipe_->InitBuffer(tileMassesBuf_, AlignUpU32(tileCount_ * sizeof(uint64_t), 32));
        pipe_->InitBuffer(sampledOutputBuf_, 32);
        pipe_->InitBuffer(lseOutputBuf_, 32);
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreIndex = GetBlockIdx();
        const uint32_t coreCount = GetBlockNum();
        for (uint32_t row = coreIndex; row < numRows_; row += coreCount) {
            ProcessRow(row);
        }
    }

private:
    __aicore__ inline uint32_t TileLength(uint32_t tileIndex) const
    {
        const uint32_t tileOffset = tileIndex * tileElements_;
        return MinU32(tileElements_, vocabSize_ - tileOffset);
    }

    __aicore__ inline uint32_t TileVectorLength(uint32_t tileIndex) const
    {
        return AlignUpU32(TileLength(tileIndex), VECTOR_ALIGNMENT_ELEMENTS);
    }

    __aicore__ inline void WriteProcessedLogitsTile(
        LocalTensor<float> logitsFloat, uint32_t row, uint32_t tileIndex, uint32_t validElements)
    {
        if (!hasOutputProcessedLogits_) {
            return;
        }
        const uint64_t cacheOffset = static_cast<uint64_t>(activeRequestIndex_) * outputProcessedLogitsStride_ +
            static_cast<uint64_t>(activeOutputProcessedLogitsCol_) * vocabSize_ +
            static_cast<uint64_t>(tileIndex) * tileElements_;
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(
            outputProcessedLogitsGm_[cacheOffset],
            logitsFloat,
            DataCopyExtParams(1, validElements * static_cast<uint32_t>(sizeof(float)), 0, 0, 0));
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline LocalTensor<float> LoadTile(uint32_t row, uint32_t tileIndex, bool writeCache = false)
    {
        const uint32_t validElements = TileLength(tileIndex);
        const uint32_t vectorElements = TileVectorLength(tileIndex);
        const uint64_t gmOffset = static_cast<uint64_t>(row) * rowStride_ +
            static_cast<uint64_t>(tileIndex) * tileElements_;
        LocalTensor<T> inputLocal = inputQueue_.AllocTensor<T>();
        DataCopyExtParams copyParams{1, validElements * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(inputLocal, processedLogitsGm_[gmOffset], copyParams, padParams);
        inputQueue_.EnQue(inputLocal);
        inputLocal = inputQueue_.DeQue<T>();

        LocalTensor<float> logitsFloat = logitsFloatBuf_.Get<float>();
        if constexpr (IsSameType<T, float>::value) {
            Adds(logitsFloat, inputLocal, 0.0f, vectorElements);
        } else {
            Cast(logitsFloat, inputLocal, RoundMode::CAST_NONE, vectorElements);
        }
        inputQueue_.FreeTensor(inputLocal);
        PipeBarrier<PIPE_V>();
        if (applyTemperature_ && activeTemperature_ != 0.0f) {
            LocalTensor<float> divisor = workBuf_.Get<float>();
            Duplicate(divisor, activeTemperature_, vectorElements);
            PipeBarrier<PIPE_V>();
            Div(logitsFloat, logitsFloat, divisor, vectorElements);
            PipeBarrier<PIPE_V>();
        }
        if (writeCache) {
            WriteProcessedLogitsTile(logitsFloat, row, tileIndex, validElements);
        }
        return logitsFloat;
    }

    __aicore__ inline LocalTensor<float> BuildNanIndicator(LocalTensor<float> logitsFloat, uint32_t vectorElements)
    {
        LocalTensor<float> nanValues = nanValuesBuf_.Get<float>();
        LocalTensor<uint8_t> nanMask = nanMaskBuf_.Get<uint8_t>();
        LocalTensor<int32_t> logitsBits = logitsFloat.ReinterpretCast<int32_t>();
        LocalTensor<int32_t> workBits = workBuf_.Get<int32_t>();

        Duplicate(workBits, FP32_EXP_MASK, vectorElements);
        PipeBarrier<PIPE_V>();
        // C220 implements Level-2 And with 16-bit lanes for compatibility.
        And(workBits, logitsBits, workBits, static_cast<int32_t>(2 * vectorElements));
        PipeBarrier<PIPE_V>();
        Compares(nanMask, workBits, FP32_EXP_MASK, CMPMODE::EQ, vectorElements);
        PipeBarrier<PIPE_V>();
        Duplicate(nanValues, 1.0f, vectorElements);
        PipeBarrier<PIPE_V>();
        Select(nanValues, nanMask, nanValues, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, vectorElements);

        Duplicate(workBits, FP32_ABS_MASK, vectorElements);
        PipeBarrier<PIPE_V>();
        And(workBits, logitsBits, workBits, static_cast<int32_t>(2 * vectorElements));
        PipeBarrier<PIPE_V>();
        Compares(nanMask, workBits, FP32_EXP_MASK, CMPMODE::EQ, vectorElements);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> infinityValues = workBuf_.Get<float>();
        Duplicate(infinityValues, 1.0f, vectorElements);
        PipeBarrier<PIPE_V>();
        Select(infinityValues, nanMask, infinityValues, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, vectorElements);
        PipeBarrier<PIPE_V>();
        Sub(nanValues, nanValues, infinityValues, vectorElements);
        PipeBarrier<PIPE_V>();
        return nanValues;
    }

    __aicore__ inline float TileMax(
        uint32_t row, uint32_t tileIndex, bool needFirstMaxIndex, uint32_t& firstMaxIndex)
    {
        const uint32_t validElements = TileLength(tileIndex);
        const uint32_t vectorElements = TileVectorLength(tileIndex);
        LocalTensor<float> logitsFloat = LoadTile(row, tileIndex);
        LocalTensor<float> work = workBuf_.Get<float>();
        LocalTensor<float> nanValues = BuildNanIndicator(logitsFloat, vectorElements);
        LocalTensor<float> scalar = scalarBuf_.Get<float>();

        if (validElements != vectorElements) {
            PipeVToS();
            bool hasNan = false;
            for (uint32_t index = 0; index < validElements; ++index) {
                hasNan = hasNan || nanValues.GetValue(index) != 0.0f;
            }
            CATEGORICAL_SAMPLE_CHECK(!hasNan, "CategoricalSample processed logits must not contain NaN\n");
            float tileMax = NEG_INFINITY;
            for (uint32_t index = 0; index < validElements; ++index) {
                const float value = logitsFloat.GetValue(index);
                if (value > tileMax) {
                    tileMax = value;
                    firstMaxIndex = index;
                }
            }
            return tileMax;
        }

        ReduceMax(scalar[8], nanValues, work, vectorElements);
        PipeVToS();
        CATEGORICAL_SAMPLE_CHECK(
            scalar.GetValue(8) == 0.0f, "CategoricalSample processed logits must not contain NaN\n");
        ReduceMax(scalar, logitsFloat, work, vectorElements);
        PipeVToS();
        const float tileMax = scalar.GetValue(0);
        if (needFirstMaxIndex) {
            for (uint32_t index = 0; index < validElements; ++index) {
                if (logitsFloat.GetValue(index) == tileMax) {
                    firstMaxIndex = index;
                    break;
                }
            }
        }
        return tileMax;
    }

    __aicore__ inline float ComputeTileExpSum(
        uint32_t row, uint32_t tileIndex, float rowMax, bool writeCache)
    {
        const uint32_t validElements = TileLength(tileIndex);
        const uint32_t vectorElements = TileVectorLength(tileIndex);
        LocalTensor<float> logitsFloat = LoadTile(row, tileIndex, writeCache);
        LocalTensor<float> work = workBuf_.Get<float>();
        LocalTensor<float> scalar = scalarBuf_.Get<float>();
        Adds(logitsFloat, logitsFloat, -rowMax, vectorElements);
        PipeBarrier<PIPE_V>();
        Exp(logitsFloat, logitsFloat, vectorElements);
        PipeBarrier<PIPE_V>();
        if (validElements != vectorElements) {
            PipeVToS();
            float tileSum = 0.0f;
            for (uint32_t index = 0; index < validElements; ++index) {
                tileSum += logitsFloat.GetValue(index);
            }
            return tileSum;
        }

        ReduceSum(scalar, logitsFloat, work, vectorElements);
        PipeVToS();
        return scalar.GetValue(0);
    }

    __aicore__ inline float ComputeAllExpSums(uint32_t row, float rowMax, bool writeCache)
    {
        LocalTensor<float> tileSums = tileSumsBuf_.Get<float>();
        float total = 0.0f;
        for (uint32_t tile = 0; tile < tileCount_; ++tile) {
            const float tileSum = ComputeTileExpSum(row, tile, rowMax, writeCache);
            tileSums.SetValue(tile, tileSum);
            total += tileSum;
        }
        return total;
    }

    __aicore__ inline uint64_t FloatToFixedMass(float value)
    {
        if (value <= 0.0f) {
            return 0;
        }
        LocalTensor<float> scalar = scalarBuf_.Get<float>();
        scalar.SetValue(0, value);
        const uint32_t bits = scalar.ReinterpretCast<uint32_t>().GetValue(0);
        const uint32_t exponent = (bits >> 23U) & 0xFFU;
        const uint64_t mantissa = exponent == 0 ? (bits & 0x7FFFFFU) : ((bits & 0x7FFFFFU) | 0x800000U);
        const int32_t binaryExponent = exponent == 0 ? -149 : static_cast<int32_t>(exponent) - 150;
        const int32_t shift = binaryExponent + FP64_WEIGHT_FRACTION_BITS;
        if (shift >= 0) {
            return mantissa << static_cast<uint32_t>(shift);
        }
        const uint32_t rightShift = static_cast<uint32_t>(-shift);
        if (rightShift >= 64U) {
            return 0;
        }
        return (mantissa + (1ULL << (rightShift - 1U))) >> rightShift;
    }

    __aicore__ inline uint64_t ComputeTileFixedMass(
        uint32_t row, uint32_t tileIndex, float rowMax, bool writeCache)
    {
        const uint32_t validElements = TileLength(tileIndex);
        const uint32_t vectorElements = TileVectorLength(tileIndex);
        LocalTensor<float> logitsFloat = LoadTile(row, tileIndex, writeCache);
        Adds(logitsFloat, logitsFloat, -rowMax, vectorElements);
        PipeBarrier<PIPE_V>();
        Exp(logitsFloat, logitsFloat, vectorElements);
        PipeVToS();

        uint64_t tileMass = 0;
        for (uint32_t index = 0; index < validElements; ++index) {
            tileMass += FloatToFixedMass(logitsFloat.GetValue(index));
        }
        return tileMass;
    }

    __aicore__ inline uint64_t ComputeAllFixedMasses(uint32_t row, float rowMax, bool writeCache)
    {
        LocalTensor<uint64_t> tileMasses = tileMassesBuf_.Get<uint64_t>();
        uint64_t totalMass = 0;
        for (uint32_t tile = 0; tile < tileCount_; ++tile) {
            const uint64_t tileMass = ComputeTileFixedMass(row, tile, rowMax, writeCache);
            tileMasses.SetValue(tile, tileMass);
            totalMass += tileMass;
        }
        return totalMass;
    }

    __aicore__ inline float ComputeLse(float rowMax, float expSum)
    {
        LocalTensor<float> scalar = scalarBuf_.Get<float>();
        scalar.SetValue(0, expSum);
        PipeSToV();
        Ln(scalar, scalar, 1);
        PipeVToS();
        return rowMax + scalar.GetValue(0);
    }

    __aicore__ inline float IntToFloat(int32_t value)
    {
        LocalTensor<float> scalar = scalarBuf_.Get<float>();
        LocalTensor<int32_t> scalarInt = scalarIntBuf_.Get<int32_t>();
        scalarInt.SetValue(0, value);
        PipeSToV();
        Cast(scalar, scalarInt, RoundMode::CAST_NONE, 1);
        PipeVToS();
        return scalar.GetValue(0);
    }

    __aicore__ inline float Uniform(int64_t seed, int64_t position)
    {
        LocalTensor<float> scalar = scalarBuf_.Get<float>();
        LocalTensor<int32_t> scalarInt = scalarIntBuf_.Get<int32_t>();
        scalarInt.SetValue(0, PhiloxRandom24(seed, position));
        PipeSToV();
        Cast(scalar, scalarInt, RoundMode::CAST_NONE, 1);
        PipeBarrier<PIPE_V>();
        Adds(scalar, scalar, 0.5f, 1);
        PipeBarrier<PIPE_V>();
        Muls(scalar, scalar, UINT24_TO_UNIT, 1);
        PipeVToS();
        return scalar.GetValue(0);
    }

    __aicore__ inline int64_t SelectCategorical(uint32_t row, float rowMax, float uniform, float total)
    {
        LocalTensor<float> tileSums = tileSumsBuf_.Get<float>();
        const float target = uniform * total;
        float prefix = 0.0f;
        uint32_t selectedTile = tileCount_ - 1;
        for (uint32_t tile = 0; tile < tileCount_; ++tile) {
            const float nextPrefix = prefix + tileSums.GetValue(tile);
            if (target <= nextPrefix || tile + 1 == tileCount_) {
                selectedTile = tile;
                break;
            }
            prefix = nextPrefix;
        }

        const uint32_t validElements = TileLength(selectedTile);
        const uint32_t vectorElements = TileVectorLength(selectedTile);
        LocalTensor<float> logitsFloat = LoadTile(row, selectedTile);
        Adds(logitsFloat, logitsFloat, -rowMax, vectorElements);
        PipeBarrier<PIPE_V>();
        Exp(logitsFloat, logitsFloat, vectorElements);
        PipeVToS();

        float cumulative = prefix;
        int64_t fallback = static_cast<int64_t>(selectedTile) * tileElements_;
        for (uint32_t index = 0; index < validElements; ++index) {
            const float weight = logitsFloat.GetValue(index);
            if (weight > 0.0f) {
                fallback = static_cast<int64_t>(selectedTile) * tileElements_ + index;
            }
            cumulative += weight;
            if (cumulative >= target) {
                return static_cast<int64_t>(selectedTile) * tileElements_ + index;
            }
        }
        return fallback;
    }

    __aicore__ inline int64_t SelectCategoricalFp64(
        uint32_t row, float rowMax, uint64_t random, uint64_t totalMass)
    {
        LocalTensor<uint64_t> tileMasses = tileMassesBuf_.Get<uint64_t>();
        const uint64_t target = MulHigh64(random, totalMass);
        uint64_t prefix = 0;
        uint32_t selectedTile = tileCount_ - 1;
        for (uint32_t tile = 0; tile < tileCount_; ++tile) {
            const uint64_t nextPrefix = prefix + tileMasses.GetValue(tile);
            if (target < nextPrefix || tile + 1 == tileCount_) {
                selectedTile = tile;
                break;
            }
            prefix = nextPrefix;
        }

        const uint32_t validElements = TileLength(selectedTile);
        const uint32_t vectorElements = TileVectorLength(selectedTile);
        LocalTensor<float> logitsFloat = LoadTile(row, selectedTile);
        Adds(logitsFloat, logitsFloat, -rowMax, vectorElements);
        PipeBarrier<PIPE_V>();
        Exp(logitsFloat, logitsFloat, vectorElements);
        PipeVToS();

        uint64_t cumulative = prefix;
        int64_t fallback = static_cast<int64_t>(selectedTile) * tileElements_;
        for (uint32_t index = 0; index < validElements; ++index) {
            const uint64_t mass = FloatToFixedMass(logitsFloat.GetValue(index));
            if (mass > 0) {
                fallback = static_cast<int64_t>(selectedTile) * tileElements_ + index;
            }
            cumulative += mass;
            if (cumulative > target) {
                return static_cast<int64_t>(selectedTile) * tileElements_ + index;
            }
        }
        return fallback;
    }

    __aicore__ inline int32_t CountPositiveInfinity(uint32_t row, int64_t& firstIndex, bool writeCache)
    {
        int32_t count = 0;
        firstIndex = 0;
        bool foundFirst = false;
        for (uint32_t tile = 0; tile < tileCount_; ++tile) {
            const uint32_t validElements = TileLength(tile);
            const uint32_t vectorElements = TileVectorLength(tile);
            LocalTensor<float> logitsFloat = LoadTile(row, tile, writeCache);
            PipeVToS();
            for (uint32_t index = 0; index < validElements; ++index) {
                if (logitsFloat.GetValue(index) == POS_INFINITY) {
                    if (!foundFirst) {
                        firstIndex = static_cast<int64_t>(tile) * tileElements_ + index;
                        foundFirst = true;
                    }
                    ++count;
                }
            }
        }
        return count;
    }

    __aicore__ inline int64_t SelectPositiveInfinity(uint32_t row, float rank)
    {
        for (uint32_t tile = 0; tile < tileCount_; ++tile) {
            const uint32_t validElements = TileLength(tile);
            const uint32_t vectorElements = TileVectorLength(tile);
            LocalTensor<float> logitsFloat = LoadTile(row, tile);
            PipeVToS();
            for (uint32_t index = 0; index < validElements; ++index) {
                if (logitsFloat.GetValue(index) == POS_INFINITY) {
                    if (rank < 1.0f) {
                        return static_cast<int64_t>(tile) * tileElements_ + index;
                    }
                    rank -= 1.0f;
                }
            }
        }
        return 0;
    }

    __aicore__ inline int64_t SelectPositiveInfinityFp64(uint32_t row, uint64_t rank)
    {
        for (uint32_t tile = 0; tile < tileCount_; ++tile) {
            const uint32_t validElements = TileLength(tile);
            const uint32_t vectorElements = TileVectorLength(tile);
            LocalTensor<float> logitsFloat = LoadTile(row, tile);
            PipeVToS();
            for (uint32_t index = 0; index < validElements; ++index) {
                if (logitsFloat.GetValue(index) == POS_INFINITY) {
                    if (rank == 0) {
                        return static_cast<int64_t>(tile) * tileElements_ + index;
                    }
                    --rank;
                }
            }
        }
        return 0;
    }

    __aicore__ inline void WriteProcessedLogitsRow(uint32_t row)
    {
        if (!hasOutputProcessedLogits_) {
            return;
        }
        for (uint32_t tile = 0; tile < tileCount_; ++tile) {
            LoadTile(row, tile, true);
        }
    }

    __aicore__ inline void WriteRow(uint32_t row, int64_t sampledIndex, float lseValue)
    {
        LocalTensor<int64_t> sampledOutput = sampledOutputBuf_.Get<int64_t>();
        LocalTensor<float> lseOutput = lseOutputBuf_.Get<float>();
        sampledOutput.SetValue(0, sampledIndex);
        if (returnLse_) {
            lseOutput.SetValue(0, lseValue);
        }

        PipeBarrier<PIPE_ALL>();
        DataCopyPad(sampledTokenIdsGm_[row], sampledOutput, DataCopyExtParams(1, sizeof(int64_t), 0, 0, 0));
        if (returnLse_) {
            DataCopyPad(lseGm_[row], lseOutput, DataCopyExtParams(1, sizeof(float), 0, 0, 0));
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void WritePaddingRow(uint32_t row)
    {
        WriteRow(row, 0, 0.0f);
    }

    __aicore__ inline void ProcessRow(uint32_t row)
    {
        const int32_t requestIndex = expandedIdxMappingGm_.GetValue(row);
        if (requestIndex == -1) {
            WritePaddingRow(row);
            return;
        }
        CATEGORICAL_SAMPLE_CHECK(
            requestIndex >= 0 && static_cast<uint32_t>(requestIndex) < numRequests_,
            "CategoricalSample expanded index mapping is outside request state\n");

        activeRequestIndex_ = static_cast<uint32_t>(requestIndex);
        activeOutputProcessedLogitsCol_ = 0;
        if (hasOutputProcessedLogitsCol_) {
            activeOutputProcessedLogitsCol_ =
                outputProcessedLogitsColGm_.GetValue(outputProcessedLogitsColPerToken_ ? row : 0);
        }
        CATEGORICAL_SAMPLE_CHECK(
            !hasOutputProcessedLogits_ ||
                (activeOutputProcessedLogitsCol_ >= 0 &&
                 static_cast<uint32_t>(activeOutputProcessedLogitsCol_) < outputProcessedLogitsNumCols_),
            "CategoricalSample output processed logits column is outside cache bounds\n");
        activeTemperature_ = temperatureGm_.GetValue(requestIndex);
        const bool isGreedy = activeTemperature_ == 0.0f;
        float rowMax = NEG_INFINITY;
        int64_t firstMaxIndex = 0;
        for (uint32_t tile = 0; tile < tileCount_; ++tile) {
            uint32_t tileFirstMaxIndex = 0;
            const float tileMax = TileMax(row, tile, isGreedy, tileFirstMaxIndex);
            if (tileMax > rowMax) {
                rowMax = tileMax;
                firstMaxIndex = static_cast<int64_t>(tile) * tileElements_ + tileFirstMaxIndex;
            }
        }
        CATEGORICAL_SAMPLE_CHECK(
            rowMax != NEG_INFINITY, "CategoricalSample processed logits row must not be all -inf\n");

        if (rowMax == POS_INFINITY) {
            int64_t firstIndex = 0;
            const int32_t infCount = CountPositiveInfinity(row, firstIndex, hasOutputProcessedLogits_);
            int64_t sampledIndex = firstIndex;
            if (!isGreedy) {
                if (useFp64_) {
                    const uint64_t random = PhiloxRandom64(seedGm_.GetValue(requestIndex), posGm_.GetValue(row));
                    const uint64_t rank = MulHigh64(random, static_cast<uint64_t>(infCount));
                    sampledIndex = SelectPositiveInfinityFp64(row, rank);
                } else {
                    const float uniform = Uniform(seedGm_.GetValue(requestIndex), posGm_.GetValue(row));
                    const float rank = uniform * IntToFloat(infCount);
                    sampledIndex = SelectPositiveInfinity(row, rank);
                }
            }
            WriteRow(row, sampledIndex, POS_INFINITY);
            return;
        }

        int64_t sampledIndex = 0;
        float expSum = 0.0f;
        if (isGreedy) {
            sampledIndex = firstMaxIndex;
            if (returnLse_) {
                expSum = ComputeAllExpSums(row, rowMax, hasOutputProcessedLogits_);
            } else {
                WriteProcessedLogitsRow(row);
            }
        } else if (useFp64_) {
            if (returnLse_) {
                expSum = ComputeAllExpSums(row, rowMax, hasOutputProcessedLogits_);
            }
            const uint64_t totalMass =
                ComputeAllFixedMasses(row, rowMax, hasOutputProcessedLogits_ && !returnLse_);
            const uint64_t random = PhiloxRandom64(seedGm_.GetValue(requestIndex), posGm_.GetValue(row));
            sampledIndex = SelectCategoricalFp64(row, rowMax, random, totalMass);
        } else {
            expSum = ComputeAllExpSums(row, rowMax, hasOutputProcessedLogits_);
            const float uniform = Uniform(seedGm_.GetValue(requestIndex), posGm_.GetValue(row));
            sampledIndex = SelectCategorical(row, rowMax, uniform, expSum);
        }

        const float lseValue = returnLse_ ? ComputeLse(rowMax, expSum) : 0.0f;
        WriteRow(row, sampledIndex, lseValue);
    }

private:
    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TBuf<QuePosition::VECCALC> logitsFloatBuf_;
    TBuf<QuePosition::VECCALC> workBuf_;
    TBuf<QuePosition::VECCALC> nanValuesBuf_;
    TBuf<QuePosition::VECCALC> nanMaskBuf_;
    TBuf<QuePosition::VECCALC> scalarBuf_;
    TBuf<QuePosition::VECCALC> scalarIntBuf_;
    TBuf<QuePosition::VECCALC> tileSumsBuf_;
    TBuf<QuePosition::VECCALC> tileMassesBuf_;
    TBuf<QuePosition::VECCALC> sampledOutputBuf_;
    TBuf<QuePosition::VECCALC> lseOutputBuf_;

    GlobalTensor<T> processedLogitsGm_;
    GlobalTensor<int32_t> expandedIdxMappingGm_;
    GlobalTensor<float> temperatureGm_;
    GlobalTensor<int64_t> seedGm_;
    GlobalTensor<int64_t> posGm_;
    GlobalTensor<float> outputProcessedLogitsGm_;
    GlobalTensor<int32_t> outputProcessedLogitsColGm_;
    GlobalTensor<int64_t> sampledTokenIdsGm_;
    GlobalTensor<float> lseGm_;

    uint32_t numRows_ = 0;
    uint32_t vocabSize_ = 0;
    uint32_t numRequests_ = 0;
    uint32_t rowStride_ = 0;
    uint32_t outputProcessedLogitsStride_ = 0;
    uint32_t outputProcessedLogitsNumCols_ = 0;
    uint32_t tileElements_ = 0;
    uint32_t tileCount_ = 0;
    uint32_t activeRequestIndex_ = 0;
    int32_t activeOutputProcessedLogitsCol_ = 0;
    float activeTemperature_ = 1.0f;
    bool hasOutputProcessedLogits_ = false;
    bool hasOutputProcessedLogitsCol_ = false;
    bool outputProcessedLogitsColPerToken_ = false;
    bool applyTemperature_ = false;
    bool returnLse_ = false;
    bool useFp64_ = false;
};
}  // namespace CategoricalSample

#endif
