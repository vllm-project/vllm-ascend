/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_NO_MASK_HPP
 #define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_NO_MASK_HPP
 
 #include "catlass/catlass.hpp"
 #include "catlass/arch/cross_core_sync.hpp"
 #include "catlass/arch/resource.hpp"
 #include "catlass/epilogue/dispatch_policy.hpp"
 #include "catlass/epilogue/tile/tile_copy.hpp"
 #include "catlass/gemm_coord.hpp"
 #include "catlass/matrix_coord.hpp"
 
 namespace Catlass::Epilogue::Block {
 
 template <
     class OutputType_,
     class InputType_,
     class MaskType_>
 class BlockEpilogue<
     EpilogueAtlasA2OnlineSoftmax,
     OutputType_,
     InputType_,
     MaskType_>
 {
 public:
     using DispatchPolicy = EpilogueAtlasA2OnlineSoftmax;
     using ArchTag = typename DispatchPolicy::ArchTag;
     using ElementOutput = typename OutputType_::Element;
     using ElementInput = typename InputType_::Element;
     using ElementMask = typename MaskType_::Element;
 
     using LayoutOutput = typename OutputType_::Layout;
     using LayoutInput = typename InputType_::Layout;
     using LayoutMask = typename MaskType_::Layout;
 
     static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
     static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
     static constexpr uint32_t HALF_VECTOR_SIZE = 128;
     static constexpr uint32_t BLOCK_SIZE = 16;
     static constexpr uint32_t UB_UINT8_VECTOR_SIZE = 1024;
     static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 16384;
     static constexpr uint32_t VECTOR_SIZE = 128;
     static constexpr uint32_t MAX_UB_S_ELEM_NUM = 8192;
 
     static constexpr uint32_t REDUCE_UB_SIZE = 1024;
     static constexpr uint32_t ROW_OPS_SPEC_MASK_32 = 32;
     static constexpr uint32_t ROW_OPS_SPEC_MASK_4 = 4;
     static constexpr uint32_t MAX_ROW_NUM_SUB_CORE = 128;
     static constexpr int64_t UB_FLOAT_LINE_SIZE = 64;
     enum class MaskCategory{
         NO_MASK = 0,
         CAUSAL_MASK = 1
     };
     CATLASS_DEVICE
     BlockEpilogue(Arch::Resource<ArchTag> &resource, float scaleValue_)
     {
         // Allocate UB space
         constexpr uint32_t LS_UB_TENSOR_OFFSET = 0;
         constexpr uint32_t LP_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;
         constexpr uint32_t MASK32_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;
         
         constexpr uint32_t TV_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE;
         constexpr uint32_t LM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 8 * UB_UINT8_VECTOR_SIZE;
         
         constexpr uint32_t HM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_VECTOR_SIZE;
         constexpr uint32_t GM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 10 * UB_UINT8_VECTOR_SIZE;
         constexpr uint32_t LL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 11 * UB_UINT8_VECTOR_SIZE;
         constexpr uint32_t GL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 12 * UB_UINT8_VECTOR_SIZE;
         constexpr uint32_t DM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 13 * UB_UINT8_VECTOR_SIZE;
 
         constexpr uint32_t MASK_UB_TENSOR_OFFSET = 11 * UB_UINT8_BLOCK_SIZE;
 
 
         scaleValue = scaleValue_;
         lsUbTensor = resource.ubBuf.template GetBufferByByte<float>(LS_UB_TENSOR_OFFSET);
         lpUbTensor = resource.ubBuf.template GetBufferByByte<ElementOutput>(LP_UB_TENSOR_OFFSET);
         maskUbTensor = resource.ubBuf.template GetBufferByByte<ElementMask>(MASK_UB_TENSOR_OFFSET);
         maskUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(MASK32_UB_TENSOR_OFFSET);
         lmUbTensor = resource.ubBuf.template GetBufferByByte<float>(LM_UB_TENSOR_OFFSET);
         hmUbTensor = resource.ubBuf.template GetBufferByByte<float>(HM_UB_TENSOR_OFFSET);
         gmUbTensor = resource.ubBuf.template GetBufferByByte<float>(GM_UB_TENSOR_OFFSET);
         dmUbTensor = resource.ubBuf.template GetBufferByByte<float>(DM_UB_TENSOR_OFFSET);
         llUbTensor = resource.ubBuf.template GetBufferByByte<float>(LL_UB_TENSOR_OFFSET);
         tvUbTensor = resource.ubBuf.template GetBufferByByte<float>(TV_UB_TENSOR_OFFSET);
         glUbTensor = resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET);
     }
 
     CATLASS_DEVICE
     ~BlockEpilogue() {}
 
     template <typename T>
     CATLASS_DEVICE T Min(T a, T b)
     {
         return (a > b) ? b : a;
     }
 
     CATLASS_DEVICE
     void SetVecMask(int32_t len)
     {
         uint64_t mask = 0;
         uint64_t one = 1;
         uint64_t temp = len % FLOAT_VECTOR_SIZE;
         for (int64_t i = 0; i < temp; i++) {
             mask |= one << i;
         }
 
         if (len == VECTOR_SIZE || len == 0) {
             AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
         } else if (len >= FLOAT_VECTOR_SIZE) {
             AscendC::SetVectorMask<int8_t>(mask, (uint64_t)-1);
         } else {
             AscendC::SetVectorMask<int8_t>(0x0, mask);
         }
     }
 
     CATLASS_DEVICE
     void SetBlockReduceMask(int32_t len)
     {
         if (len > 8 || len < 1) {
             AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
             return;
         }
         uint64_t subMask = ((uint64_t)1 << len) - 1;
         uint64_t maskValue = (subMask << 48) + (subMask << 32) + (subMask << 16) + subMask + (subMask << 56) +
                              (subMask << 40) + (subMask << 24) + (subMask << 8);
         AscendC::SetVectorMask<int8_t>(maskValue, maskValue);
     }
 
     CATLASS_DEVICE
     void RowsumSPECTILE512(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowsumUb,
                            const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
                            uint32_t numElemsAligned)
     {
         AscendC::BlockReduceSum<float, false>(tvUbTensor, srcUb, numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE, 0,
                                               1, 1, 8);
         AscendC::PipeBarrier<PIPE_V>();
 
         AscendC::BlockReduceSum<float, false>(tvUbTensor[REDUCE_UB_SIZE], tvUbTensor,
                                               numRowsRound * numElemsAligned / FLOAT_BLOCK_SIZE / FLOAT_VECTOR_SIZE, 0,
                                               1, 1, 8);
         AscendC::PipeBarrier<PIPE_V>();
         AscendC::BlockReduceSum<float, false>(rowsumUb, tvUbTensor[REDUCE_UB_SIZE],
                                               numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE / FLOAT_VECTOR_SIZE, 0,
                                               1, 1, 8);
         AscendC::PipeBarrier<PIPE_V>();
     }
 
     CATLASS_DEVICE
     void RowsumSPECTILE256(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowsumUb,
                            const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
                            uint32_t numElemsAligned)
     {
         AscendC::BlockReduceSum<float, false>(tvUbTensor, srcUb, numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE, 0,
                                               1, 1, 8);
         AscendC::PipeBarrier<PIPE_V>();
         SetVecMask(ROW_OPS_SPEC_MASK_32);
         AscendC::BlockReduceSum<float, false>(tvUbTensor[REDUCE_UB_SIZE], tvUbTensor, numRowsRound, 0, 1, 1, 4);
         AscendC::PipeBarrier<PIPE_V>();
         SetBlockReduceMask(ROW_OPS_SPEC_MASK_4);
         AscendC::BlockReduceSum<float, false>(
             rowsumUb, tvUbTensor[REDUCE_UB_SIZE],
             (numRowsRound * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE, 0, 1, 1, 8);
         AscendC::PipeBarrier<PIPE_V>();
         AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
     }
 
     CATLASS_DEVICE
     void RowsumTAILTILE(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowsumUb,
                         const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
                         uint32_t numElemsAligned)
     {
         if (numElems >= FLOAT_VECTOR_SIZE) {
             AscendC::BlockReduceSum<float, false>(tvUbTensor, srcUb, numRowsRound, 0, 1, 1,
                                                   numElemsAligned / FLOAT_BLOCK_SIZE);
             AscendC::PipeBarrier<PIPE_V>();
             AscendC::BlockReduceSum<float, false>(
                 rowsumUb, tvUbTensor, (numRowsRound * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE, 0,
                 1, 1, 8);
             AscendC::PipeBarrier<PIPE_V>();
             for (uint64_t rowSumIdx = 1; rowSumIdx < (uint64_t)numElems / FLOAT_VECTOR_SIZE; ++rowSumIdx) {
                 AscendC::BlockReduceSum<float, false>(tvUbTensor, srcUb[rowSumIdx * FLOAT_VECTOR_SIZE], numRowsRound, 0,
                                                       1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
                 AscendC::PipeBarrier<PIPE_V>();
                 AscendC::BlockReduceSum<float, false>(
                     tvUbTensor[REDUCE_UB_SIZE], tvUbTensor,
                     (numRowsRound * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE, 0, 1, 1, 8);
                 AscendC::PipeBarrier<PIPE_V>();
                 SetVecMask(numRowsRound);
                 AscendC::Add<float, false>(rowsumUb, rowsumUb, tvUbTensor[REDUCE_UB_SIZE], (uint64_t)0, 1,
                                            AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                 AscendC::PipeBarrier<PIPE_V>();
                 AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
             }
         }
         if (numElems % FLOAT_VECTOR_SIZE > 0) {
             SetVecMask(numElems % FLOAT_VECTOR_SIZE);
             AscendC::BlockReduceSum<float, false>(tvUbTensor, srcUb[numElems / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                                   numRowsRound, 0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
             AscendC::PipeBarrier<PIPE_V>();
             SetBlockReduceMask((numElems % FLOAT_VECTOR_SIZE + FLOAT_BLOCK_SIZE - 1) / FLOAT_BLOCK_SIZE);
             if (numElems < FLOAT_VECTOR_SIZE) {
                 AscendC::BlockReduceSum<float, false>(
                     rowsumUb, tvUbTensor, (numRowsRound * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                     0, 1, 1, 8);
                 AscendC::PipeBarrier<PIPE_V>();
             } else {
                 AscendC::BlockReduceSum<float, false>(
                     tvUbTensor[REDUCE_UB_SIZE], tvUbTensor,
                     (numRowsRound * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE, 0, 1, 1, 8);
                 AscendC::PipeBarrier<PIPE_V>();
                 SetVecMask(numRowsRound);
                 AscendC::Add<float, false>(rowsumUb, rowsumUb, tvUbTensor[REDUCE_UB_SIZE], (uint64_t)0, 1,
                                            AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                 AscendC::PipeBarrier<PIPE_V>();
             }
             AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
         }
     }
 
     CATLASS_DEVICE
     void RowmaxSPECTILE512(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowmaxUb,
                            const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
                            uint32_t numElemsAligned)
     {
         AscendC::BlockReduceMax<float, false>(tvUbTensor, srcUb, numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE, 0,
                                               1, 1, 8);
         AscendC::PipeBarrier<PIPE_V>();
         AscendC::BlockReduceMax<float, false>(tvUbTensor[REDUCE_UB_SIZE], tvUbTensor,
                                               numRowsRound * numElemsAligned / FLOAT_BLOCK_SIZE / FLOAT_VECTOR_SIZE, 0,
                                               1, 1, 8);
         AscendC::PipeBarrier<PIPE_V>();
         AscendC::BlockReduceMax<float, false>(rowmaxUb, tvUbTensor[REDUCE_UB_SIZE],
                                               numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE / FLOAT_VECTOR_SIZE, 0,
                                               1, 1, 8);
         AscendC::PipeBarrier<PIPE_V>();
     }
 
     CATLASS_DEVICE
     void RowmaxSPECTILE256(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowmaxUb,
                            const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
                            uint32_t numElemsAligned)
     {
         AscendC::BlockReduceMax<float, false>(tvUbTensor, srcUb, numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE, 0,
                                               1, 1, 8);
         AscendC::PipeBarrier<PIPE_V>();
         SetVecMask(ROW_OPS_SPEC_MASK_32);
         AscendC::BlockReduceMax<float, false>(tvUbTensor[REDUCE_UB_SIZE], tvUbTensor, numRowsRound, 0, 1, 1, 4);
         AscendC::PipeBarrier<PIPE_V>();
         SetBlockReduceMask(ROW_OPS_SPEC_MASK_4);
         AscendC::BlockReduceMax<float, false>(
             rowmaxUb, tvUbTensor[REDUCE_UB_SIZE],
             (numRowsRound * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE, 0, 1, 1, 8);
         AscendC::PipeBarrier<PIPE_V>();
         AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
     }
 
     CATLASS_DEVICE
     void RowmaxTAILTILE(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowmaxUb,
                         const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
                         uint32_t numElemsAligned)
     {
         if (numElems >= FLOAT_VECTOR_SIZE) {
             AscendC::BlockReduceMax<float, false>(tvUbTensor, srcUb, numRowsRound, 0, 1, 1,
                                                   numElemsAligned / FLOAT_BLOCK_SIZE);
             AscendC::PipeBarrier<PIPE_V>();
             AscendC::BlockReduceMax<float, false>(
                 rowmaxUb, tvUbTensor, (numRowsRound * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE, 0,
                 1, 1, 8);
             AscendC::PipeBarrier<PIPE_V>();
             for (uint64_t rowmax_idx = 1; rowmax_idx < (uint64_t)numElems / FLOAT_VECTOR_SIZE; ++rowmax_idx) {
                 AscendC::BlockReduceMax<float, false>(tvUbTensor, srcUb[rowmax_idx * FLOAT_VECTOR_SIZE], numRowsRound,
                                                       0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
                 AscendC::PipeBarrier<PIPE_V>();
                 AscendC::BlockReduceMax<float, false>(
                     tvUbTensor[REDUCE_UB_SIZE], tvUbTensor,
                     (numRowsRound * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE, 0, 1, 1, 8);
                 AscendC::PipeBarrier<PIPE_V>();
                 SetVecMask(numRowsRound);
                 AscendC::Max<float, false>(rowmaxUb, rowmaxUb, tvUbTensor[REDUCE_UB_SIZE], (uint64_t)0, 1,
                                            AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                 AscendC::PipeBarrier<PIPE_V>();
                 AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
             }
         }
         if (numElems % FLOAT_VECTOR_SIZE > 0) {
             SetVecMask(numElems % FLOAT_VECTOR_SIZE);
             AscendC::BlockReduceMax<float, false>(tvUbTensor, srcUb[numElems / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                                   numRowsRound, 0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
             AscendC::PipeBarrier<PIPE_V>();
             SetBlockReduceMask((numElems % FLOAT_VECTOR_SIZE + FLOAT_BLOCK_SIZE - 1) / FLOAT_BLOCK_SIZE);
             if (numElems < FLOAT_VECTOR_SIZE) {
                 AscendC::BlockReduceMax<float, false>(
                     rowmaxUb, tvUbTensor, (numRowsRound * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                     0, 1, 1, 8);
                 AscendC::PipeBarrier<PIPE_V>();
             } else {
                 AscendC::BlockReduceMax<float, false>(
                     tvUbTensor[REDUCE_UB_SIZE], tvUbTensor,
                     (numRowsRound * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE, 0, 1, 1, 8);
                 AscendC::PipeBarrier<PIPE_V>();
                 SetVecMask(numRowsRound);
                 AscendC::Max<float, false>(rowmaxUb, rowmaxUb, tvUbTensor[REDUCE_UB_SIZE], (uint64_t)0, 1,
                                            AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                 AscendC::PipeBarrier<PIPE_V>();
             }
             AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
         }
     }
 
     CATLASS_DEVICE
     void CopySGmToUb(
         AscendC::GlobalTensor<ElementInput> gInput,
         uint32_t sUbOffset,
         uint32_t rowNumCurLoop,
         uint32_t columnNumRound,
         uint32_t columnNumPad)
     {
         // input S
         AscendC::DataCopy(
             lsUbTensor[sUbOffset], gInput,
             AscendC::DataCopyParams(
                 rowNumCurLoop, columnNumRound / FLOAT_BLOCK_SIZE,
                 (columnNumPad - columnNumRound) / FLOAT_BLOCK_SIZE, 0));
     }
 
     CATLASS_DEVICE
     void CopyMaskGmToUb(
         AscendC::GlobalTensor<ElementMask> gMask,   
         uint32_t columnNum,
         uint32_t columnNumRound,
         uint32_t maskStride,
         uint32_t qSBlockSize,
         uint32_t proTokenIdx,
         uint32_t proTokenNum,
         uint32_t integralHeadNum,
         uint32_t epiTokenNum)
     {
         uint32_t innerUbRowOffset = 0;
         if (proTokenNum != 0) {
             AscendC::DataCopyPad(
                 maskUbTensor[innerUbRowOffset], gMask[proTokenIdx * maskStride],
                 AscendC::DataCopyExtParams(
                     proTokenNum, columnNum * 2, (maskStride - columnNum) * 2, 0, 0),
                 AscendC::DataCopyPadExtParams<ElementMask>(false, 0, 0, 0));
             innerUbRowOffset += proTokenNum * columnNumRound;
         }
         for (uint32_t headIdx = 0; headIdx < integralHeadNum; headIdx++) {
             AscendC::DataCopyPad(
                 maskUbTensor[innerUbRowOffset], gMask,
                 AscendC::DataCopyExtParams(
                     qSBlockSize, columnNum * 2, (maskStride - columnNum) * 2, 0, 0),
                 AscendC::DataCopyPadExtParams<ElementMask>(false, 0, 0, 0));
             innerUbRowOffset += qSBlockSize * columnNumRound;
         }
         if (epiTokenNum != 0) {
             AscendC::DataCopyPad(
                 maskUbTensor[innerUbRowOffset], gMask,
                 AscendC::DataCopyExtParams(
                     epiTokenNum, columnNum * 2, (maskStride - columnNum) * 2, 0, 0),
                 AscendC::DataCopyPadExtParams<ElementMask>(false, 0, 0, 0));
         }
     }
 
     CATLASS_DEVICE
     void ScaleS(
         uint32_t sUbOffset,
         uint32_t rowNumCurLoop,
         uint32_t columnNumRound)
     {
         // *** ls = scaleValue * ls
         AscendC::Muls<float, false>(
             lsUbTensor[sUbOffset], lsUbTensor[sUbOffset], scaleValue, (uint64_t)0,
             (rowNumCurLoop * columnNumRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
             AscendC::UnaryRepeatParams(1, 1, 8, 8));
 
         AscendC::PipeBarrier<PIPE_V>();
     }
 
     CATLASS_DEVICE
     void UpCastMask(
         uint32_t rowNumCurLoop,
         uint32_t columnNumRound)
     {
         AscendC::Cast<float, ElementMask, false>(
             maskUbTensor32, maskUbTensor, AscendC::RoundMode::CAST_NONE, (uint64_t)0,
             (rowNumCurLoop * columnNumRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
             AscendC::UnaryRepeatParams(1, 1, 8, 4));
         AscendC::PipeBarrier<PIPE_V>();
     }
     
     CATLASS_DEVICE
     void ApplyMask(
         uint32_t sUbOffset,
         uint32_t rowNumCurLoop,
         uint32_t columnNumRound)
     {
         AscendC::Muls<float, false>(
             maskUbTensor32, maskUbTensor32, (float)-3e38, (uint64_t)0,
             (rowNumCurLoop * columnNumRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
             AscendC::UnaryRepeatParams(1, 1, 8, 8));
         AscendC::PipeBarrier<PIPE_V>();
         AscendC::Add<float, false>(
             lsUbTensor[sUbOffset], lsUbTensor[sUbOffset], maskUbTensor32,
             (uint64_t)0, (rowNumCurLoop * columnNumRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
             AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
         AscendC::PipeBarrier<PIPE_V>();
     }
 
     CATLASS_DEVICE
     void CalcLocalRowMax(
         uint32_t sUbOffset,
         uint32_t rowNumCurLoopRound,
         uint32_t columnNum,
         uint32_t columnNumRound,
         uint32_t rowOffset)
     {
         if (columnNum == 512) {
             RowmaxSPECTILE512(
                 lsUbTensor[sUbOffset], lmUbTensor[rowOffset],
                 tvUbTensor, rowNumCurLoopRound, columnNum, columnNumRound);
         } else if (columnNum == 256) {
             RowmaxSPECTILE256(
                 lsUbTensor[sUbOffset], lmUbTensor[rowOffset],
                 tvUbTensor, rowNumCurLoopRound, columnNum, columnNumRound);
         } else {
             RowmaxTAILTILE(
                 lsUbTensor[sUbOffset], lmUbTensor[rowOffset],
                 tvUbTensor, rowNumCurLoopRound, columnNum, columnNumRound);
         }
         
     }
 
     CATLASS_DEVICE
     void UpdateGlobalRowMax(
         uint32_t rowNumCurLoop,
         uint32_t rowNumCurLoopRound,
         uint32_t columnNum,
         uint32_t columnNumRound,
         uint32_t dmUbOffsetCurCycle,
         uint32_t rowOffset,
         uint32_t isFirstStackTile)
     {
         if (isFirstStackTile) {
             AscendC::DataCopy(
                 hmUbTensor[rowOffset], lmUbTensor[rowOffset],
                 AscendC::DataCopyParams(1, rowNumCurLoopRound / FLOAT_BLOCK_SIZE, 0, 0));
             AscendC::PipeBarrier<PIPE_V>();
         } else {
             SetVecMask(rowNumCurLoop);
             // *** hm = vmax(lm, gm)
             AscendC::Max<float, false>(
                 hmUbTensor[rowOffset], lmUbTensor[rowOffset], gmUbTensor[rowOffset], (uint64_t)0,
                 1, AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
 
             AscendC::PipeBarrier<PIPE_V>();
             // *** dm = gm - hm
             AscendC::Sub<float, false>(
                 dmUbTensor[dmUbOffsetCurCycle],
                 gmUbTensor[rowOffset], hmUbTensor[rowOffset], (uint64_t)0, 1,
                 AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
 
             AscendC::PipeBarrier<PIPE_V>();
             // *** dm = exp(dm)
             AscendC::Exp<float, false>(
                 dmUbTensor[dmUbOffsetCurCycle],
                 dmUbTensor[dmUbOffsetCurCycle],
                 (uint64_t)0, 1, AscendC::UnaryRepeatParams(1, 1, 8, 8));
         }
         AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
         AscendC::PipeBarrier<PIPE_V>();
         // *** gm = hm
         AscendC::DataCopy(gmUbTensor[rowOffset], hmUbTensor[rowOffset],
                           AscendC::DataCopyParams(1, rowNumCurLoopRound / FLOAT_BLOCK_SIZE, 0, 0));
         AscendC::PipeBarrier<PIPE_V>();
     }
 
     CATLASS_DEVICE
     void CalcExp(
         uint32_t sUbOffset,
         uint32_t rowNumCurLoop,
         uint32_t rowNumCurLoopRound,
         uint32_t columnNum,
         uint32_t columnNumRound,
         uint32_t rowOffset)
     {
         // *** hm_block = expand_to_block(hm), 存放于 tv
         AscendC::Brcb(
             tvUbTensor.template ReinterpretCast<uint32_t>(),
             hmUbTensor[rowOffset].template ReinterpretCast<uint32_t>(), rowNumCurLoopRound / FLOAT_BLOCK_SIZE,
             AscendC::BrcbRepeatParams(1, 8));
         AscendC::PipeBarrier<PIPE_V>();
         // *** ls = ls - hm_block
         for (uint32_t subIdx = 0; subIdx < columnNum / FLOAT_VECTOR_SIZE; ++subIdx) {
             AscendC::Sub<float, false>(
                 lsUbTensor[sUbOffset][subIdx * FLOAT_VECTOR_SIZE], lsUbTensor[sUbOffset][subIdx * FLOAT_VECTOR_SIZE],
                 tvUbTensor, (uint64_t)0, rowNumCurLoop,
                 AscendC::BinaryRepeatParams(1, 1, 0, columnNumRound / FLOAT_BLOCK_SIZE, columnNumRound / FLOAT_BLOCK_SIZE, 1));
         }
         if (columnNum % FLOAT_VECTOR_SIZE > 0) {
             SetVecMask(columnNum % FLOAT_VECTOR_SIZE);
             AscendC::Sub<float, false>(
                 lsUbTensor[sUbOffset][columnNum / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                 lsUbTensor[sUbOffset][columnNum / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE], tvUbTensor, (uint64_t)0, rowNumCurLoop,
                 AscendC::BinaryRepeatParams(1, 1, 0, columnNumRound / FLOAT_BLOCK_SIZE, columnNumRound / FLOAT_BLOCK_SIZE, 1));
             AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
         }
         AscendC::PipeBarrier<PIPE_V>();
         // *** ls = exp(ls)
         AscendC::Exp<float, false>(lsUbTensor[sUbOffset], lsUbTensor[sUbOffset], (uint64_t)0,
                                    (rowNumCurLoop * columnNumRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                    AscendC::UnaryRepeatParams(1, 1, 8, 8));
         AscendC::PipeBarrier<PIPE_V>();
     }
 
     CATLASS_DEVICE
     void CalcLocalRowSum(
         uint32_t sUbOffset,
         uint32_t rowNumCurLoopRound,
         uint32_t columnNum,
         uint32_t columnNumRound,
         uint32_t rowOffset)
     {
         // *** ll = rowsum(ls32)
         if (columnNum == 512) {
             RowsumSPECTILE512(
                 lsUbTensor[sUbOffset], llUbTensor[rowOffset], tvUbTensor,
                 rowNumCurLoopRound, columnNum, columnNumRound);
         } else if (columnNum == 256) {
             RowsumSPECTILE256(
                 lsUbTensor[sUbOffset], llUbTensor[rowOffset], tvUbTensor,
                 rowNumCurLoopRound, columnNum, columnNumRound);
         } else {
             RowsumTAILTILE(
                 lsUbTensor[sUbOffset], llUbTensor[rowOffset], tvUbTensor,
                 rowNumCurLoopRound, columnNum, columnNumRound);
         }
     }
 
     CATLASS_DEVICE
     void UpdateGlobalRowSum(
         uint32_t sUbOffset,
         uint32_t rowNumCurLoop,
         uint32_t rowNumCurLoopRound,
         uint32_t dmUbOffsetCurCycle,
         uint32_t rowOffset,
         uint32_t isFirstStackTile)
     {
         if (isFirstStackTile) {
             // *** gl = ll
             AscendC::DataCopy(
                 glUbTensor[rowOffset], llUbTensor[rowOffset],
                 AscendC::DataCopyParams(1, rowNumCurLoopRound / FLOAT_BLOCK_SIZE, 0, 0));
             AscendC::PipeBarrier<PIPE_V>();
         } else {
             SetVecMask(rowNumCurLoop);
             // *** gl = dm * gl
             AscendC::Mul<float, false>(
                 glUbTensor[rowOffset], dmUbTensor[dmUbOffsetCurCycle],
                 glUbTensor[rowOffset], (uint64_t)0, 1, AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
             AscendC::PipeBarrier<PIPE_V>();
             // *** gl = ll + gl
             AscendC::Add<float, false>(glUbTensor[rowOffset], glUbTensor[rowOffset], llUbTensor[rowOffset], (uint64_t)0,
                                        1, AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
             AscendC::PipeBarrier<PIPE_V>();
             AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
         }
     }
 
     CATLASS_DEVICE
     void DownCastP(
         uint32_t sUbOffset,
         uint32_t rowNumCurLoop,
         uint32_t columnNumRound)
     {
         // *** lp = castfp32to16(ls)
         if (std::is_same<ElementOutput, bfloat16_t>::value) {
             AscendC::Cast<ElementOutput, float, false>(
                 lpUbTensor[sUbOffset], lsUbTensor[sUbOffset], AscendC::RoundMode::CAST_RINT, (uint64_t)0,
                 (rowNumCurLoop * columnNumRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE, AscendC::UnaryRepeatParams(1, 1, 4, 8));
         } else {
             AscendC::Cast<ElementOutput, float, false>(
                 lpUbTensor[sUbOffset], lsUbTensor[sUbOffset], AscendC::RoundMode::CAST_NONE, (uint64_t)0,
                 (rowNumCurLoop * columnNumRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE, AscendC::UnaryRepeatParams(1, 1, 4, 8));
         }
     }
 
     CATLASS_DEVICE
     void CopyPUbToGm(
         AscendC::GlobalTensor<ElementOutput> gOutput,
         uint32_t sUbOffset,
         uint32_t rowNumCurLoop,
         uint32_t columnNumRound,
         uint32_t columnNumPad)
     {
         if(columnNumRound == columnNumPad){
             AscendC::DataCopy(
                 gOutput, lpUbTensor[sUbOffset],
                 AscendC::DataCopyParams(
                     1, rowNumCurLoop * columnNumRound / BLOCK_SIZE,
                     0, 0));
         }
         else{
             AscendC::DataCopy(
                 gOutput, lpUbTensor[sUbOffset],
                 AscendC::DataCopyParams(
                     rowNumCurLoop, columnNumRound / BLOCK_SIZE,
                     0, (columnNumPad - columnNumRound) / BLOCK_SIZE));
         }
         
     }
 
     template<MaskCategory maskCat>
     CATLASS_DEVICE
     void SubCoreCompute(AscendC::GlobalTensor<ElementOutput> gOutput, 
                         const LayoutOutput &layoutOutput,
                         uint32_t rowOffset, uint32_t isFirstStackTile,
                         uint32_t columnNumRound, uint32_t pingpongFlag,
                         uint32_t curStackTileMod)
     {
         uint32_t rowNumCurLoop = layoutOutput.shape(0);
         uint32_t rowNumCurLoopRound = RoundUp(rowNumCurLoop, FLOAT_BLOCK_SIZE);
         uint32_t columnNum = layoutOutput.shape(1);
         // Align colNum to 16, for both float&half compute&copy
         uint32_t columnNumPad = layoutOutput.stride(0);
         uint32_t sUbOffset = pingpongFlag * MAX_UB_S_ELEM_NUM;
         uint32_t dmUbOffsetCurCycle = curStackTileMod * MAX_ROW_NUM_SUB_CORE + rowOffset;
 
         CalcLocalRowMax(sUbOffset, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);
         UpdateGlobalRowMax(rowNumCurLoop, rowNumCurLoopRound, columnNum, columnNumRound, dmUbOffsetCurCycle, rowOffset, isFirstStackTile);
 
         CalcExp(sUbOffset, rowNumCurLoop, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);
         if constexpr (maskCat == MaskCategory::NO_MASK){
             AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pingpongFlag);
         }   
         
         DownCastP(sUbOffset, rowNumCurLoop, columnNumRound);
         AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pingpongFlag);
 
         CalcLocalRowSum(sUbOffset, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);
         AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(pingpongFlag);
 
         AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pingpongFlag);
         CopyPUbToGm(gOutput, sUbOffset, rowNumCurLoop, columnNumRound, columnNumPad);
         if constexpr (maskCat == MaskCategory::NO_MASK){
             AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pingpongFlag);
         }
         else if constexpr (maskCat == MaskCategory::CAUSAL_MASK){
             AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
             AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
         }
         UpdateGlobalRowSum(sUbOffset, rowNumCurLoop, rowNumCurLoopRound, dmUbOffsetCurCycle, rowOffset, isFirstStackTile);
     }
 
     CATLASS_DEVICE
     void operator()(AscendC::GlobalTensor<ElementOutput> gOutput, AscendC::GlobalTensor<ElementInput> gInput,
                     const LayoutOutput &layoutOutput, const LayoutInput &layoutInput, GemmCoord actualBlockShape,
                     uint32_t isFirstStackTile, uint32_t qSBlockSize, uint32_t qNBlockSize, uint32_t curStackTileMod)
     {
         uint32_t rowNum = actualBlockShape.m();
         uint32_t columnNum = actualBlockShape.n();
         uint32_t columnNumRound = RoundUp(columnNum, BLOCK_SIZE);
         uint32_t columnNumPad = layoutInput.stride(0);
 
         uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
         uint32_t subBlockNum = AscendC::GetSubBlockNum();
 
         uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;
         uint32_t qNThisSubBlock = (qNBlockSize == 1) ?
             0 : (subBlockIdx == 1) ? (qNBlockSize - qNSplitSubBlock) : qNSplitSubBlock;
         uint32_t rowSplitSubBlock = (qNBlockSize == 1) ?
             (qSBlockSize / 2) : (qSBlockSize * qNSplitSubBlock);
         uint32_t rowActualThisSubBlock = (subBlockIdx == 1) ?
             (rowNum - rowSplitSubBlock) : rowSplitSubBlock;
         uint32_t rowOffsetThisSubBlock = subBlockIdx * rowSplitSubBlock;
         uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / columnNumRound;
         uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, FLOAT_BLOCK_SIZE);
         uint32_t rowLoopNum = CeilDiv(rowActualThisSubBlock, rowNumTile);
         uint32_t preLoad = 1;
 
         
         for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum + preLoad; rowLoopIdx++) {
             if(rowLoopIdx < rowLoopNum){
                 
                 uint32_t pingpongFlag = rowLoopIdx % 2;
                 uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                 uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                 uint32_t rowNumCurLoop = (rowLoopIdx == rowLoopNum - 1) ?
                     (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;
             // loop 0 mask load before cross core sync
             
             
                int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gInputCurLoop = gInput[offsetInput];
                
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(pingpongFlag);
                CopySGmToUb(gInputCurLoop, (pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound, columnNumPad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
 
            }
            if(rowLoopIdx >= preLoad){
                uint32_t delayedRowLoopIdx = rowLoopIdx - preLoad;
                uint32_t pingpongFlag = delayedRowLoopIdx % 2;
                uint32_t rowOffsetCurLoop = delayedRowLoopIdx * rowNumTile;
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                uint32_t rowNumCurLoop = (delayedRowLoopIdx == rowLoopNum - 1) ?
                    (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

                int64_t offsetOutput = layoutOutput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gOutputCurLoop = gOutput[offsetOutput];
                auto layoutOutputCurLoop = layoutOutput.GetTileLayout(MatrixCoord(rowNumCurLoop, columnNum));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
                ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                SubCoreCompute<MaskCategory::NO_MASK>(
                    gOutputCurLoop,
                    layoutOutputCurLoop,
                    rowOffsetCurLoop,
                    isFirstStackTile,
                    columnNumRound,
                    pingpongFlag,
                    curStackTileMod
                );
            }
        }
    }


    CATLASS_DEVICE
     void operator()(AscendC::GlobalTensor<ElementOutput> gOutput, AscendC::GlobalTensor<ElementInput> gInput,
                     AscendC::GlobalTensor<ElementMask> gMask,
                     const LayoutOutput &layoutOutput, const LayoutInput &layoutInput, const LayoutInput &layoutMask,
                     GemmCoord actualBlockShape,
                     uint32_t isFirstStackTile, uint32_t qSBlockSize, uint32_t qNBlockSize, uint32_t curStackTileMod, 
                    Arch::CrossCoreFlag qkReady)
     {
         uint32_t rowNum = actualBlockShape.m();
         uint32_t columnNum = actualBlockShape.n();
         uint32_t columnNumRound = RoundUp(columnNum, BLOCK_SIZE);
         uint32_t columnNumPad = layoutInput.stride(0);
         uint32_t maskStride = layoutMask.stride(0);
         uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
         uint32_t subBlockNum = AscendC::GetSubBlockNum();
 
         uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;
         uint32_t qNThisSubBlock = (qNBlockSize == 1) ?
             0 : (subBlockIdx == 1) ? (qNBlockSize - qNSplitSubBlock) : qNSplitSubBlock;
         uint32_t rowSplitSubBlock = (qNBlockSize == 1) ?
             (qSBlockSize / 2) : (qSBlockSize * qNSplitSubBlock);
         uint32_t rowActualThisSubBlock = (subBlockIdx == 1) ?
             (rowNum - rowSplitSubBlock) : rowSplitSubBlock;
         uint32_t rowOffsetThisSubBlock = subBlockIdx * rowSplitSubBlock;
 
         uint32_t tokenNumPerHeadThisSubBlock = Min(qSBlockSize, rowActualThisSubBlock);
         
         uint32_t maskOffsetThisSubBlock = (qNBlockSize == 1) ?
             rowOffsetThisSubBlock : 0;
         int64_t offsetMask = layoutMask.GetOffset(MatrixCoord(maskOffsetThisSubBlock, 0));
         auto gMaskThisSubBlock = gMask[offsetMask];
         auto layoutMaskThisSubBlock = layoutMask;

         uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / columnNumRound;
         uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, FLOAT_BLOCK_SIZE);
         uint32_t rowLoopNum = CeilDiv(rowActualThisSubBlock, rowNumTile);
         uint32_t preLoad = 1;
         
         if (rowActualThisSubBlock == 0) {
             Arch::CrossCoreWaitFlag(qkReady);
             return;
         }

        for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum + preLoad; rowLoopIdx++) {
            if(rowLoopIdx < rowLoopNum){
                uint32_t pingpongFlag = rowLoopIdx % 2;
                uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                uint32_t rowNumCurLoop = (rowLoopIdx == rowLoopNum - 1) ?
                    (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;
                // loop 0 mask load before cross core sync
                if(rowLoopIdx == 0){
                    // the token idx of the start token of the prologue part
                    uint32_t proTokenIdx = rowOffsetCurLoop % tokenNumPerHeadThisSubBlock;
                    // the token num of the prologue part
                    uint32_t proTokenNum = Min(rowNumCurLoop, (tokenNumPerHeadThisSubBlock - proTokenIdx)) %
                        tokenNumPerHeadThisSubBlock;
                    // the token num of the epilogue part
                    uint32_t integralHeadNum = (rowNumCurLoop - proTokenNum) / tokenNumPerHeadThisSubBlock;
                    // the number of integral heads within a cycle
                    uint32_t epiTokenNum = rowNumCurLoop - proTokenNum - integralHeadNum * tokenNumPerHeadThisSubBlock;
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
                    CopyMaskGmToUb(gMaskThisSubBlock, columnNum, columnNumRound, maskStride,
                        tokenNumPerHeadThisSubBlock, proTokenIdx, proTokenNum, integralHeadNum, epiTokenNum);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                    Arch::CrossCoreWaitFlag(qkReady);
                }
                int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gInputCurLoop = gInput[offsetInput];
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(pingpongFlag);
                CopySGmToUb(gInputCurLoop, (pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound, columnNumPad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
            }
            if(rowLoopIdx >= preLoad){
                uint32_t delayedRowLoopIdx = rowLoopIdx - preLoad;
                uint32_t pingpongFlag = delayedRowLoopIdx % 2;
                uint32_t rowOffsetCurLoop = delayedRowLoopIdx * rowNumTile;
                uint32_t rowNumCurLoop = (delayedRowLoopIdx == rowLoopNum - 1) ?
                    (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                UpCastMask(rowNumCurLoop, columnNumRound);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
                ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                ApplyMask((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
                // next loop mask load
                if (rowLoopIdx < rowLoopNum){
                    uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                    uint32_t rowNumCurLoop = (rowLoopIdx == rowLoopNum - 1) ?
                        (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;
                    // the token idx of the start token of the prologue part
                    uint32_t proTokenIdx = rowOffsetCurLoop % tokenNumPerHeadThisSubBlock;
                    // the token num of the prologue part
                    uint32_t proTokenNum = Min(rowNumCurLoop, (tokenNumPerHeadThisSubBlock - proTokenIdx)) %
                        tokenNumPerHeadThisSubBlock;
                    // the number of integral heads within a cycle
                    uint32_t integralHeadNum = (rowNumCurLoop - proTokenNum) / tokenNumPerHeadThisSubBlock;
                    // the token num of the epilogue part
                    uint32_t epiTokenNum = rowNumCurLoop - proTokenNum - integralHeadNum * tokenNumPerHeadThisSubBlock;
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
                    CopyMaskGmToUb(gMaskThisSubBlock, columnNum, columnNumRound, maskStride,
                        tokenNumPerHeadThisSubBlock, proTokenIdx, proTokenNum, integralHeadNum, epiTokenNum);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                }
                // online softmax vectorized compute
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                int64_t offsetOutput = layoutOutput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gOutputCurLoop = gOutput[offsetOutput];
                auto layoutOutputCurLoop = layoutOutput.GetTileLayout(MatrixCoord(rowNumCurLoop, columnNum));
                SubCoreCompute<MaskCategory::CAUSAL_MASK>(
                    gOutputCurLoop,
                    layoutOutputCurLoop,
                    rowOffsetCurLoop,
                    isFirstStackTile,
                    columnNumRound,
                    pingpongFlag,
                    curStackTileMod);
            }
        }
         
     }
 
 private:
     float scaleValue;
     AscendC::LocalTensor<float> lsUbTensor;
     AscendC::LocalTensor<ElementOutput> lpUbTensor;
     AscendC::LocalTensor<ElementMask> maskUbTensor;
     AscendC::LocalTensor<float> maskUbTensor32;
     AscendC::LocalTensor<float> lmUbTensor;
     AscendC::LocalTensor<float> hmUbTensor;
     AscendC::LocalTensor<float> gmUbTensor;
     AscendC::LocalTensor<float> dmUbTensor;
     AscendC::LocalTensor<float> llUbTensor;
     AscendC::LocalTensor<float> tvUbTensor;
     AscendC::LocalTensor<float> glUbTensor;
 };
 } // namespace Catlass::Epilogue::Block
 
 #endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_NO_MASK_HPP