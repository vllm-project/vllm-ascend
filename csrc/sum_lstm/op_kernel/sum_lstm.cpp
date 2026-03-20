#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;

// Constant definitions
constexpr float SQRT_2_PI = 0.7978845608f;      // sqrt(2/pi)
constexpr float GELU_COEF = 0.044715f;          // GELU tanh approximation coefficient
constexpr float SQRT_2_INV = 0.7071067812f;     // 1/sqrt(2)

template <typename T>
class SumLstmKernel {
public:
    __aicore__ inline SumLstmKernel() {}

    __aicore__ inline void Init(GM_ADDR states4d, GM_ADDR z4_4d, GM_ADDR prevCell,
                                GM_ADDR wCell, GM_ADDR bCell, GM_ADDR wState, GM_ADDR bState,
                                GM_ADDR outState, GM_ADDR outCell, GM_ADDR workspace,
                                const SumLstmTilingData* tilingData) {
        // Get Tiling parameters
        totalSamples = tilingData->totalSamples;
        hiddenDim = tilingData->hiddenDim;
        gatedDim = tilingData->gatedDim;
        coreNum = tilingData->coreNum;
        samplesPerCore = tilingData->samplesPerCore;
        remainSamples = tilingData->remainSamples;
        tileNumPerCore = tilingData->tileNumPerCore;
        tileSamples = tilingData->tileSamples;
        lastTileSamples = tilingData->lastTileSamples;
        hiddenDimAligned = tilingData->hiddenDimAligned;
        gatedDimAligned = tilingData->gatedDimAligned;
        alpha = tilingData->alpha;
        epsCell = tilingData->epsCell;
        epsState = tilingData->epsState;
        useFastGelu = tilingData->useFastGelu;
        hasWCell = tilingData->hasWCell;
        hasBCell = tilingData->hasBCell;
        hasWState = tilingData->hasWState;
        hasBState = tilingData->hasBState;
        floatHiddenDimAligned = tilingData->floatHiddenDimAligned;
        bufferCount = tilingData->bufferCount;
        preloadWeights = tilingData->preloadWeights;

        // Compute current core processing range
        uint32_t blockIdx = GetBlockIdx();
        uint32_t startSample = blockIdx * samplesPerCore;
        if (blockIdx < remainSamples) {
            startSample += blockIdx;
            coreSamples = samplesPerCore + 1;
        } else {
            startSample += remainSamples;
            coreSamples = samplesPerCore;
        }

        // Set Global Memory addresses
        gmStates4d.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(states4d) + startSample * gatedDim, coreSamples * gatedDim);
        gmZ4_4d.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(z4_4d) + startSample * gatedDim, coreSamples * gatedDim);
        gmPrevCell.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(prevCell) + startSample * hiddenDim, coreSamples * hiddenDim);
        gmOutState.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outState) + startSample * hiddenDim, coreSamples * hiddenDim);
        gmOutCell.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outCell) + startSample * hiddenDim, coreSamples * hiddenDim);

        // Optional inputs
        if (hasWCell) {
            gmWCell.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(wCell), hiddenDim);
        }
        if (hasBCell) {
            gmBCell.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bCell), hiddenDim);
        }
        if (hasWState) {
            gmWState.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(wState), hiddenDim);
        }
        if (hasBState) {
            gmBState.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bState), hiddenDim);
        }

        // Initialize Pipe Buffer
        uint32_t bufferSize = tileSamples * hiddenDimAligned * sizeof(T);
        uint32_t bufferSizeGated = tileSamples * gatedDimAligned * sizeof(T);

        // Input queues (half, dynamic buffer depth)
        pipe.InitBuffer(inQueueStates, bufferCount, bufferSizeGated);
        pipe.InitBuffer(inQueueZ4, bufferCount, bufferSizeGated);
        pipe.InitBuffer(inQueuePrevCell, bufferCount, bufferSize);

        // Output queues (half, dynamic buffer depth)
        pipe.InitBuffer(outQueueState, bufferCount, bufferSize);
        pipe.InitBuffer(outQueueCell, bufferCount, bufferSize);

        // 5 float buffers, each floatHiddenDimAligned * sizeof(float), fixed size not scaling with tileSamples
        // Layout:
        //   floatBuf1: preF -> reused as preO (delayed computation)
        //   floatBuf2: preI -> freed then idle
        //   floatBuf3: cpre/cact/sact + castTemp (for weight Cast)
        //   floatBuf4: outCellF
        //   floatBuf5: castTemp -> outStateF -> final out_state
        uint32_t floatBufSize = floatHiddenDimAligned * sizeof(float);
        pipe.InitBuffer(floatBuffer1, floatBufSize);
        pipe.InitBuffer(floatBuffer2, floatBufSize);
        pipe.InitBuffer(floatBuffer3, floatBufSize);
        pipe.InitBuffer(floatBuffer4, floatBufSize);
        pipe.InitBuffer(floatBuffer5, floatBufSize);

        // Optional input buffer (half, for loading optional weights)
        if (hasWCell || hasBCell || hasWState || hasBState) {
            uint32_t optSlots = preloadWeights ? 4 : 1;
            pipe.InitBuffer(optBuffer, optSlots * hiddenDimAligned * sizeof(T));
        }

        // Calculate tile count
        actualTileNum = (coreSamples + tileSamples - 1) / tileSamples;
    }

    __aicore__ inline void PreloadWeights() {
        LocalTensor<T> optLocal = optBuffer.Get<T>();
        uint32_t wBytes = hiddenDim * sizeof(T);
        uint8_t wRpad = (hiddenDimAligned - hiddenDim);
        AscendC::DataCopyExtParams cpParams = {1, wBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams = {false, 0, wRpad, (T)0};

        if (hasWCell) {
            AscendC::DataCopyPad<T>(optLocal, gmWCell[0], cpParams, padParams);
        }
        if (hasBCell) {
            AscendC::DataCopyPad<T>(optLocal[hiddenDimAligned], gmBCell[0], cpParams, padParams);
        }
        if (hasWState) {
            AscendC::DataCopyPad<T>(optLocal[2 * hiddenDimAligned], gmWState[0], cpParams, padParams);
        }
        if (hasBState) {
            AscendC::DataCopyPad<T>(optLocal[3 * hiddenDimAligned], gmBState[0], cpParams, padParams);
        }
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void Process() {
        if (preloadWeights) {
            PreloadWeights();
        }

        for (uint32_t tileIdx = 0; tileIdx < actualTileNum; ++tileIdx) {
            uint32_t currentTileSamples = tileSamples;
            if (tileIdx == actualTileNum - 1) {
                currentTileSamples = coreSamples - tileIdx * tileSamples;
            }
            currentTileSize = currentTileSamples;
            tileOffset = tileIdx * tileSamples;

            CopyIn(tileIdx, currentTileSamples);
            Compute(currentTileSamples);
            CopyOut(tileIdx, currentTileSamples);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t tileIdx, uint32_t samples) {
        // Copy states_4d (GM -> Local)
        LocalTensor<T> statesLocal = inQueueStates.AllocTensor<T>();
        uint32_t gatedBytes = gatedDim * sizeof(T);
        uint8_t gatedRpad = (gatedDimAligned - gatedDim);
        AscendC::DataCopyExtParams copyParamsGated = {(uint16_t)samples, gatedBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParamsGated = {false, 0, gatedRpad, (T)0};
        AscendC::DataCopyPad<T>(statesLocal, gmStates4d[tileOffset * gatedDim], copyParamsGated, padParamsGated);
        inQueueStates.EnQue(statesLocal);

        // Copy z4_4d
        LocalTensor<T> z4Local = inQueueZ4.AllocTensor<T>();
        AscendC::DataCopyPad<T>(z4Local, gmZ4_4d[tileOffset * gatedDim], copyParamsGated, padParamsGated);
        inQueueZ4.EnQue(z4Local);

        // Copy prev_cell
        LocalTensor<T> prevCellLocal = inQueuePrevCell.AllocTensor<T>();
        uint32_t hiddenBytes = hiddenDim * sizeof(T);
        uint8_t hiddenRpad = (hiddenDimAligned - hiddenDim);
        AscendC::DataCopyExtParams copyParamsHidden = {(uint16_t)samples, hiddenBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParamsHidden = {false, 0, hiddenRpad, (T)0};
        AscendC::DataCopyPad<T>(prevCellLocal, gmPrevCell[tileOffset * hiddenDim], copyParamsHidden, padParamsHidden);
        inQueuePrevCell.EnQue(prevCellLocal);
    }

    __aicore__ inline void Compute(uint32_t samples) {
        // Get input data (half)
        LocalTensor<T> statesLocal = inQueueStates.DeQue<T>();
        LocalTensor<T> z4Local = inQueueZ4.DeQue<T>();
        LocalTensor<T> prevCellLocal = inQueuePrevCell.DeQue<T>();

        // Get output buffer (half)
        LocalTensor<T> outStateLocal = outQueueState.AllocTensor<T>();
        LocalTensor<T> outCellLocal = outQueueCell.AllocTensor<T>();

        // Get 5 float buffers
        LocalTensor<float> fBuf1 = floatBuffer1.Get<float>();  // preF -> reused as preO
        LocalTensor<float> fBuf2 = floatBuffer2.Get<float>();  // preI -> freed
        LocalTensor<float> fBuf3 = floatBuffer3.Get<float>();  // cpre/cact/sact/castTemp(weight)
        LocalTensor<float> fBuf4 = floatBuffer4.Get<float>();  // outCellF
        LocalTensor<float> fBuf5 = floatBuffer5.Get<float>();  // castTemp -> outStateF

        float alphaF = alpha;

        // Clear float buffers, ensure padding region is 0
        Duplicate(fBuf1, 0.0f, floatHiddenDimAligned);
        Duplicate(fBuf2, 0.0f, floatHiddenDimAligned);
        Duplicate(fBuf3, 0.0f, floatHiddenDimAligned);
        Duplicate(fBuf4, 0.0f, floatHiddenDimAligned);
        Duplicate(fBuf5, 0.0f, floatHiddenDimAligned);
        pipe_barrier(PIPE_ALL);

        // Process each sample
        for (uint32_t s = 0; s < samples; ++s) {
            uint32_t statesOffsetAligned = s * gatedDimAligned;
            uint32_t hiddenOffsetAligned = s * hiddenDimAligned;

            // ===== Step 1: Compute preF, preI, cpre (3 gates + cpre, delayed preO) =====
            // Use fBuf5 as castTemp

            // Forget gate: preF = s0 + alpha * z0  -> fBuf1
            // Note: Gate data in half tensor is contiguous by hiddenDim, Cast uses hiddenDim
            Cast(fBuf5, z4Local[statesOffsetAligned], RoundMode::CAST_NONE, hiddenDim);
            pipe_barrier(PIPE_ALL);
            Muls(fBuf1, fBuf5, alphaF, floatHiddenDimAligned);
            Cast(fBuf5, statesLocal[statesOffsetAligned], RoundMode::CAST_NONE, hiddenDim);
            pipe_barrier(PIPE_ALL);
            Add(fBuf1, fBuf5, fBuf1, floatHiddenDimAligned);

            // Input gate: preI = s1 + alpha * z1  → fBuf2
            Cast(fBuf5, z4Local[statesOffsetAligned + hiddenDim], RoundMode::CAST_NONE, hiddenDim);
            pipe_barrier(PIPE_ALL);
            Muls(fBuf2, fBuf5, alphaF, floatHiddenDimAligned);
            Cast(fBuf5, statesLocal[statesOffsetAligned + hiddenDim], RoundMode::CAST_NONE, hiddenDim);
            pipe_barrier(PIPE_ALL);
            Add(fBuf2, fBuf5, fBuf2, floatHiddenDimAligned);

            // Cell pre-activation: cpre = s3 + alpha * z3  → fBuf3
            Cast(fBuf5, z4Local[statesOffsetAligned + 3 * hiddenDim], RoundMode::CAST_NONE, hiddenDim);
            pipe_barrier(PIPE_ALL);
            Muls(fBuf3, fBuf5, alphaF, floatHiddenDimAligned);
            Cast(fBuf5, statesLocal[statesOffsetAligned + 3 * hiddenDim], RoundMode::CAST_NONE, hiddenDim);
            pipe_barrier(PIPE_ALL);
            Add(fBuf3, fBuf5, fBuf3, floatHiddenDimAligned);

            // ===== Step 2: Sigmoid on preF, preI =====
            pipe_barrier(PIPE_ALL);
            Sigmoid(fBuf1, fBuf1, floatHiddenDimAligned);  // f in fBuf1
            Sigmoid(fBuf2, fBuf2, floatHiddenDimAligned);  // i in fBuf2

            // ===== Step 3: RMSNorm cpre → outCellF =====
            pipe_barrier(PIPE_ALL);
            ComputeRMSNormFloat(fBuf4, fBuf3, hiddenDim, epsCell);

            // ===== Step 4: Optional weights w_cell, b_cell (Cast half->float using fBuf5) =====
            if (hasWCell) {
                LocalTensor<T> optLocal = optBuffer.Get<T>();
                if (preloadWeights) {
                    Cast(fBuf5, optLocal, RoundMode::CAST_NONE, hiddenDim);
                } else {
                    uint32_t wBytes = hiddenDim * sizeof(T);
                    uint8_t wRpad = (hiddenDimAligned - hiddenDim);
                    AscendC::DataCopyExtParams cpParams = {1, wBytes, 0, 0, 0};
                    AscendC::DataCopyPadExtParams<T> padParams = {false, 0, wRpad, (T)0};
                    AscendC::DataCopyPad<T>(optLocal, gmWCell[0], cpParams, padParams);
                    pipe_barrier(PIPE_ALL);
                    Cast(fBuf5, optLocal, RoundMode::CAST_NONE, hiddenDim);
                }
                pipe_barrier(PIPE_ALL);
                Mul(fBuf4, fBuf4, fBuf5, floatHiddenDimAligned);
            }
            if (hasBCell) {
                LocalTensor<T> optLocal = optBuffer.Get<T>();
                if (preloadWeights) {
                    Cast(fBuf5, optLocal[hiddenDimAligned], RoundMode::CAST_NONE, hiddenDim);
                } else {
                    uint32_t bBytes = hiddenDim * sizeof(T);
                    uint8_t bRpad = (hiddenDimAligned - hiddenDim);
                    AscendC::DataCopyExtParams cpParams = {1, bBytes, 0, 0, 0};
                    AscendC::DataCopyPadExtParams<T> padParams = {false, 0, bRpad, (T)0};
                    AscendC::DataCopyPad<T>(optLocal, gmBCell[0], cpParams, padParams);
                    pipe_barrier(PIPE_ALL);
                    Cast(fBuf5, optLocal, RoundMode::CAST_NONE, hiddenDim);
                }
                pipe_barrier(PIPE_ALL);
                Add(fBuf4, fBuf4, fBuf5, floatHiddenDimAligned);
            }

            // ===== Step 5: GELU → cact in fBuf3 =====
            pipe_barrier(PIPE_ALL);
            ComputeGeluFloat(fBuf3, fBuf4, floatHiddenDimAligned);

            // ===== Step 6: out_cell = prevCell * f + cact * i =====
            // Use fBuf5 as castTemp
            Cast(fBuf5, prevCellLocal[hiddenOffsetAligned], RoundMode::CAST_NONE, hiddenDim);
            pipe_barrier(PIPE_ALL);
            Mul(fBuf4, fBuf5, fBuf1, floatHiddenDimAligned);    // outCellF = prevCell * f
            Mul(fBuf3, fBuf3, fBuf2, floatHiddenDimAligned);    // cact * i
            Add(fBuf4, fBuf4, fBuf3, floatHiddenDimAligned);    // outCellF final

            // >>> fBuf1, fBuf2, fBuf3 all freed, fBuf4=outCellF, fBuf5 available

            // ===== Step 7: Delayed compute preO -> fBuf1 =====
            // Use fBuf5 as castTemp
            Cast(fBuf5, z4Local[statesOffsetAligned + 2 * hiddenDim], RoundMode::CAST_NONE, hiddenDim);
            pipe_barrier(PIPE_ALL);
            Muls(fBuf1, fBuf5, alphaF, floatHiddenDimAligned);
            Cast(fBuf5, statesLocal[statesOffsetAligned + 2 * hiddenDim], RoundMode::CAST_NONE, hiddenDim);
            pipe_barrier(PIPE_ALL);
            Add(fBuf1, fBuf5, fBuf1, floatHiddenDimAligned);
            Sigmoid(fBuf1, fBuf1, floatHiddenDimAligned);  // o in fBuf1

            // ===== Step 8: RMSNorm outCellF → outStateF in fBuf5 =====
            pipe_barrier(PIPE_ALL);
            ComputeRMSNormFloat(fBuf5, fBuf4, hiddenDim, epsState);

            // ===== Step 9: Optional weights w_state, b_state (Cast half->float using fBuf3) =====
            if (hasWState) {
                LocalTensor<T> optLocal = optBuffer.Get<T>();
                if (preloadWeights) {
                    Cast(fBuf3, optLocal[2 * hiddenDimAligned], RoundMode::CAST_NONE, hiddenDim);
                } else {
                    uint32_t wBytes = hiddenDim * sizeof(T);
                    uint8_t wRpad = (hiddenDimAligned - hiddenDim);
                    AscendC::DataCopyExtParams cpParams = {1, wBytes, 0, 0, 0};
                    AscendC::DataCopyPadExtParams<T> padParams = {false, 0, wRpad, (T)0};
                    AscendC::DataCopyPad<T>(optLocal, gmWState[0], cpParams, padParams);
                    pipe_barrier(PIPE_ALL);
                    Cast(fBuf3, optLocal, RoundMode::CAST_NONE, hiddenDim);
                }
                pipe_barrier(PIPE_ALL);
                Mul(fBuf5, fBuf5, fBuf3, floatHiddenDimAligned);
            }
            if (hasBState) {
                LocalTensor<T> optLocal = optBuffer.Get<T>();
                if (preloadWeights) {
                    Cast(fBuf3, optLocal[3 * hiddenDimAligned], RoundMode::CAST_NONE, hiddenDim);
                } else {
                    uint32_t bBytes = hiddenDim * sizeof(T);
                    uint8_t bRpad = (hiddenDimAligned - hiddenDim);
                    AscendC::DataCopyExtParams cpParams = {1, bBytes, 0, 0, 0};
                    AscendC::DataCopyPadExtParams<T> padParams = {false, 0, bRpad, (T)0};
                    AscendC::DataCopyPad<T>(optLocal, gmBState[0], cpParams, padParams);
                    pipe_barrier(PIPE_ALL);
                    Cast(fBuf3, optLocal, RoundMode::CAST_NONE, hiddenDim);
                }
                pipe_barrier(PIPE_ALL);
                Add(fBuf5, fBuf5, fBuf3, floatHiddenDimAligned);
            }

            // ===== Step 10: GELU → sact in fBuf3 =====
            pipe_barrier(PIPE_ALL);
            ComputeGeluFloat(fBuf3, fBuf5, floatHiddenDimAligned);

            // ===== Step 11: out_state = sact * o =====
            Mul(fBuf5, fBuf3, fBuf1, floatHiddenDimAligned);

            // ===== Step 12: Cast output float->half =====
            pipe_barrier(PIPE_ALL);
            Cast(outCellLocal[hiddenOffsetAligned], fBuf4, RoundMode::CAST_RINT, hiddenDim);
            Cast(outStateLocal[hiddenOffsetAligned], fBuf5, RoundMode::CAST_RINT, hiddenDim);
        }

        // Free input buffer
        inQueueStates.FreeTensor(statesLocal);
        inQueueZ4.FreeTensor(z4Local);
        inQueuePrevCell.FreeTensor(prevCellLocal);

        // Enqueue output
        outQueueState.EnQue(outStateLocal);
        outQueueCell.EnQue(outCellLocal);
    }

    __aicore__ inline void CopyOut(uint32_t tileIdx, uint32_t samples) {
        // Copy out_state (Local -> GM)
        LocalTensor<T> outStateLocal = outQueueState.DeQue<T>();
        uint32_t hiddenBytes = hiddenDim * sizeof(T);
        uint32_t srcStrideBytes = (hiddenDimAligned - hiddenDim) * sizeof(T);
        AscendC::DataCopyExtParams copyParams = {(uint16_t)samples, hiddenBytes, srcStrideBytes, 0, 0};
        AscendC::DataCopyPad<T>(gmOutState[tileOffset * hiddenDim], outStateLocal, copyParams);
        outQueueState.FreeTensor(outStateLocal);

        // Copy out_cell
        LocalTensor<T> outCellLocal = outQueueCell.DeQue<T>();
        AscendC::DataCopyPad<T>(gmOutCell[tileOffset * hiddenDim], outCellLocal, copyParams);
        outQueueCell.FreeTensor(outCellLocal);
    }

    // RMSNorm full float implementation: x / sqrt(mean(x^2) + eps)
    // Use vector tree reduction instead of scalar loop summation
    __aicore__ inline void ComputeRMSNormFloat(LocalTensor<float>& dst, LocalTensor<float>& src, uint32_t len, float eps) {
        // Step 1: Compute x^2 (float)
        Mul(dst, src, src, floatHiddenDimAligned);
        pipe_barrier(PIPE_ALL);

        // Step 2: Vector tree reduction sum
        // Add upper half to lower half each time
        // halfLen must be multiple of 8 to ensure dst[halfLen] UB address 32-byte aligned
        uint32_t reduceLen = floatHiddenDimAligned;
        while (reduceLen > 8) {
            uint32_t halfLen = reduceLen / 2;
            if (halfLen < 8 || halfLen % 8 != 0) break;
            Add(dst, dst, dst[halfLen], halfLen);
            pipe_barrier(PIPE_ALL);
            reduceLen = halfLen;
        }

        // Scalar sum remaining elements
        float sum = 0.0f;
        for (uint32_t idx = 0; idx < reduceLen; ++idx) {
            sum += dst.GetValue(idx);
        }

        // Step 3: Compute rsqrt (float)
        // Note: Divide by actual hiddenDim not aligned length (padding elements are 0, does not affect sum)
        float mean = sum / static_cast<float>(static_cast<int32_t>(len));
        float rsqrtVal = 1.0f / sqrt(mean + eps);

        // Step 4: dst = src * rsqrt
        Muls(dst, src, rsqrtVal, floatHiddenDimAligned);
    }

    // GELU full float implementation
    __aicore__ inline void ComputeGeluFloat(LocalTensor<float>& dst, LocalTensor<float>& src, uint32_t len) {
        if (useFastGelu) {
            // Fast GELU (tanh approximation): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            Mul(dst, src, src, len);
            Mul(dst, dst, src, len);
            Muls(dst, dst, GELU_COEF, len);
            Add(dst, src, dst, len);
            Muls(dst, dst, SQRT_2_PI, len);
            Tanh(dst, dst, len);
            Adds(dst, dst, 1.0f, len);
            Mul(dst, src, dst, len);
            Muls(dst, dst, 0.5f, len);
        } else {
            // Precise GELU (erf): 0.5 * x * (1 + erf(x / sqrt(2)))
            Muls(dst, src, SQRT_2_INV, len);
            Erf(dst, dst, len);
            Adds(dst, dst, 1.0f, len);
            Mul(dst, src, dst, len);
            Muls(dst, dst, 0.5f, len);
        }
    }

private:
    // Tiling parameters
    uint32_t totalSamples;
    uint32_t hiddenDim;
    uint32_t gatedDim;
    uint32_t coreNum;
    uint32_t samplesPerCore;
    uint32_t remainSamples;
    uint32_t tileNumPerCore;
    uint32_t tileSamples;
    uint32_t lastTileSamples;
    uint32_t hiddenDimAligned;
    uint32_t gatedDimAligned;
    uint32_t floatHiddenDimAligned;
    uint32_t bufferCount;
    uint32_t preloadWeights;
    float alpha;
    float epsCell;
    float epsState;
    uint32_t useFastGelu;
    uint32_t hasWCell;
    uint32_t hasBCell;
    uint32_t hasWState;
    uint32_t hasBState;

    // Runtime parameters
    uint32_t coreSamples;
    uint32_t actualTileNum;
    uint32_t currentTileSize;
    uint32_t tileOffset;

    // Pipe
    TPipe pipe;

    // Input queues (half)
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueStates;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueZ4;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueuePrevCell;

    // Output queues (half)
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueState;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueCell;

    // float compute buffer (5)
    TBuf<QuePosition::VECCALC> floatBuffer1;  // preF -> preO
    TBuf<QuePosition::VECCALC> floatBuffer2;  // preI
    TBuf<QuePosition::VECCALC> floatBuffer3;  // cpre / cact / sact / castTemp(weight)
    TBuf<QuePosition::VECCALC> floatBuffer4;  // outCellF
    TBuf<QuePosition::VECCALC> floatBuffer5;  // castTemp -> outStateF

    // Optional input buffer (half)
    TBuf<QuePosition::VECCALC> optBuffer;

    // Global Memory
    GlobalTensor<T> gmStates4d;
    GlobalTensor<T> gmZ4_4d;
    GlobalTensor<T> gmPrevCell;
    GlobalTensor<T> gmWCell;
    GlobalTensor<T> gmBCell;
    GlobalTensor<T> gmWState;
    GlobalTensor<T> gmBState;
    GlobalTensor<T> gmOutState;
    GlobalTensor<T> gmOutCell;
};

extern "C" __global__ __aicore__ void sum_lstm(GM_ADDR states4d, GM_ADDR z4_4d, GM_ADDR prevCell,
                                                GM_ADDR wCell, GM_ADDR bCell, GM_ADDR wState, GM_ADDR bState,
                                                GM_ADDR outState, GM_ADDR outCell,
                                                GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    // Get data type size to determine which template to use
    uint32_t dataTypeSize = tilingData.dataTypeSize;

    if (TILING_KEY_IS(0)) {
        // Standard processing path
        if (dataTypeSize == 2) {
            // bfloat16
            SumLstmKernel<bfloat16_t> op;
            op.Init(states4d, z4_4d, prevCell, wCell, bCell, wState, bState,
                    outState, outCell, workspace, &tilingData);
            op.Process();
        }
    }
}
