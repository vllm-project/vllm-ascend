/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_NO_SPLIT_ROW_HPP
 #define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_NO_SPLIT_ROW_HPP
 
 #include "catlass/catlass.hpp"
 #include "catlass/arch/resource.hpp"
 #include "catlass/epilogue/dispatch_policy.hpp"
 #include "catlass/epilogue/tile/tile_copy.hpp"
 #include "catlass/gemm_coord.hpp"
 #include "catlass/matrix_coord.hpp"
 
 namespace Catlass::Epilogue::Block {
 
 template <
     class OutputType_,
     class InputType_,
     class UpdateType_>
 class BlockEpilogue<
     EpilogueAtlasA2RescaleO,
     OutputType_,
     InputType_,
     UpdateType_>
 {
 public:
     // Type aliases
     using DispatchPolicy = EpilogueAtlasA2RescaleO;
     using ArchTag = typename DispatchPolicy::ArchTag;
 
     using ElementOutput = typename OutputType_::Element;
     using ElementInput = typename InputType_::Element;
     using ElementUpdate = typename UpdateType_::Element;
 
     using LayoutOutput = typename OutputType_::Layout;
     using LayoutInput = typename InputType_::Layout;
     using LayoutUpdate = typename UpdateType_::Layout;
 
     static constexpr uint32_t HALF_ELENUM_PER_BLK = 16;
     static constexpr uint32_t BLOCK_SIZE = 16;
     static constexpr uint32_t HALF_ELENUM_PER_VECCALC = 128;
     static constexpr uint32_t FLOAT_ELENUM_PER_VECCALC = 64;
     static constexpr uint32_t HALF_ELENUM_PER_LINE = 256;
     static constexpr uint32_t FLOAT_ELENUM_PER_LINE = 128;
     static constexpr uint32_t MULTIPLIER = 2;
     static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
     static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
     static constexpr uint32_t UB_UINT8_VECTOR_SIZE = 1024;
     static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 16384;
     static constexpr uint32_t HALF_DM_UB_SIZE = 64;
     static constexpr uint32_t HALF_LL_UB_SIZE = 256;
     static constexpr uint32_t VECTOR_SIZE = 128;
     static constexpr uint32_t NUM4 = 4;
     static constexpr uint32_t MAX_UB_O_ELEM_NUM = 4096;
     static constexpr uint32_t MAX_ROW_NUM_SUB_CORE = 128;
 
     CATLASS_DEVICE
     BlockEpilogue(Arch::Resource<ArchTag> &resource)
     {
         // Allocate UB space
         constexpr uint32_t LO_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE;
         constexpr uint32_t GO_UB_TENSOR_OFFSET = 8 * UB_UINT8_BLOCK_SIZE;
         constexpr uint32_t TV_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE;
         
         constexpr uint32_t HM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_VECTOR_SIZE;
         constexpr uint32_t GL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 12 * UB_UINT8_VECTOR_SIZE;
         constexpr uint32_t DM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 13 * UB_UINT8_VECTOR_SIZE;
 
         loUbTensor = resource.ubBuf.template GetBufferByByte<float>(LO_UB_TENSOR_OFFSET);
         dmUbTensor = resource.ubBuf.template GetBufferByByte<float>(DM_UB_TENSOR_OFFSET);
         glUbTensor = resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET);
         tvUbTensor = resource.ubBuf.template GetBufferByByte<float>(TV_UB_TENSOR_OFFSET);
         goUbTensor16 = resource.ubBuf.template GetBufferByByte<ElementOutput>(GO_UB_TENSOR_OFFSET);
         goUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(GO_UB_TENSOR_OFFSET);
         hmUbTensor = resource.ubBuf.template GetBufferByByte<float>(HM_UB_TENSOR_OFFSET);
     }
 
     CATLASS_DEVICE
     ~BlockEpilogue()
     {
     }
 
     CATLASS_DEVICE
     void SetMask(int32_t len)
     {
         uint64_t mask = 0;
         uint64_t one = 1;
         uint64_t temp = len % FLOAT_VECTOR_SIZE;
         for (int64_t i = 0; i < temp; i++) {
             mask |= one << i;
         }
 
         if (len == VECTOR_SIZE) {
             AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
         } else if (len >= FLOAT_VECTOR_SIZE) {
             AscendC::SetVectorMask<int8_t>(mask, (uint64_t)-1);
         } else {
             AscendC::SetVectorMask<int8_t>(0x0, mask);
         }
     }
 
     CATLASS_DEVICE
     void CopyOToGm(
         AscendC::GlobalTensor<ElementOutput> gOutput,
         uint32_t curRowNum, uint32_t qSBlockSize, uint32_t embed,
         uint32_t embedRound, uint32_t qNThisSubBlock, uint32_t oHiddenSize)
     {
         if (qNThisSubBlock == 0) {
             AscendC::DataCopyPad(
                 gOutput, 
                 goUbTensor16,
                 AscendC::DataCopyExtParams(curRowNum, embed * 2, 0, (oHiddenSize - embed) * 2, 0));
         } else {
             for (uint32_t qNIdx = 0; qNIdx < qNThisSubBlock; qNIdx++) {
                 AscendC::DataCopyPad(
                     gOutput[qNIdx * embed],
                     goUbTensor16[qNIdx * embedRound * qSBlockSize],
                     AscendC::DataCopyExtParams(qSBlockSize, embed * 2, 0, (oHiddenSize - embed) * 2, 0));
             }
         }
     }
 
     CATLASS_DEVICE
     void SubCoreCompute(
         AscendC::GlobalTensor<ElementOutput> gOutput,
         AscendC::GlobalTensor<ElementInput> gInput,
         const LayoutOutput &layoutOutput,
         const LayoutInput &layoutInput,
         uint32_t qNThisSubBlock,
         uint32_t isFirstStackTile, uint32_t isLastStackTile, uint32_t curStackTileMod)
     {
         uint32_t curRowNum = layoutInput.shape(0);
         uint32_t embed = layoutInput.shape(1);
         uint32_t embedRound = layoutInput.stride(0);
         uint32_t curRowNumRound = RoundUp(curRowNum, FLOAT_BLOCK_SIZE);
         uint32_t qSBlockSize = layoutOutput.shape(0);
         uint32_t oHiddenSize = layoutOutput.shape(1);
         uint32_t dmUbOffsetCurStackTile =
             curStackTileMod * MAX_ROW_NUM_SUB_CORE;
         
         AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
         if (!isFirstStackTile) {
             AscendC::DataCopy(loUbTensor, gInput,
                               AscendC::DataCopyParams(1, curRowNum * embedRound / FLOAT_BLOCK_SIZE, 0, 0));
             AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
             AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
 
             AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
             AscendC::Brcb(tvUbTensor.ReinterpretCast<uint32_t>(),
                           dmUbTensor[dmUbOffsetCurStackTile].ReinterpretCast<uint32_t>(), curRowNumRound / FLOAT_BLOCK_SIZE,
                           AscendC::BrcbRepeatParams(1, 8));
             AscendC::PipeBarrier<PIPE_V>();
             // *** go = go * dm_block
             AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
             for (uint32_t vmul_idx = 0; vmul_idx < embed / FLOAT_VECTOR_SIZE; ++vmul_idx) {
                 AscendC::Mul<float, false>(goUbTensor32[vmul_idx * FLOAT_VECTOR_SIZE],
                                            goUbTensor32[vmul_idx * FLOAT_VECTOR_SIZE],
                                            tvUbTensor, (uint64_t)0,
                                            curRowNum,
                                            AscendC::BinaryRepeatParams(1, 1, 0, embedRound / FLOAT_BLOCK_SIZE,
                                                                        embedRound / FLOAT_BLOCK_SIZE, 1));
             }
             if (embed % FLOAT_VECTOR_SIZE > 0) {
                 SetMask(embed % FLOAT_VECTOR_SIZE);
                 AscendC::Mul<float, false>(goUbTensor32[embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                            goUbTensor32[embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                            tvUbTensor,
                                            (uint64_t)0, curRowNum,
                                            AscendC::BinaryRepeatParams(1, 1, 0, embedRound / FLOAT_BLOCK_SIZE,
                                                                        embedRound / FLOAT_BLOCK_SIZE, 1));
                 AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
             }
             AscendC::PipeBarrier<PIPE_V>();
             // *** go = lo + go
             AscendC::Add<float, false>(goUbTensor32, goUbTensor32,
                                        loUbTensor, (uint64_t)0,
                                        (curRowNum * embedRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                        AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
             AscendC::PipeBarrier<PIPE_V>();
         } else {
             // *** go = lo
             AscendC::DataCopy(goUbTensor32, gInput,
                               AscendC::DataCopyParams(1, curRowNum * embedRound / FLOAT_BLOCK_SIZE, 0, 0));
             AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
             AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
         }
         AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
 
         if (isLastStackTile) {
             // *** gl_block = expand_to_block(gl), 存放于 tv
             AscendC::Brcb(tvUbTensor.ReinterpretCast<uint32_t>(),
                           glUbTensor.ReinterpretCast<uint32_t>(),
                           curRowNumRound / FLOAT_BLOCK_SIZE,
                           AscendC::BrcbRepeatParams(1, 8));
             AscendC::PipeBarrier<PIPE_V>();
             // *** go = go / gl_block
             AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
             for (uint32_t vdiv_idx = 0; vdiv_idx < embed / FLOAT_VECTOR_SIZE; ++vdiv_idx) {
                 AscendC::Div<float, false>(goUbTensor32[vdiv_idx * FLOAT_VECTOR_SIZE],
                                            goUbTensor32[vdiv_idx * FLOAT_VECTOR_SIZE],
                                            tvUbTensor, (uint64_t)0,
                                            curRowNum,
                                            AscendC::BinaryRepeatParams(1, 1, 0, embedRound / FLOAT_BLOCK_SIZE,
                                                                        embedRound / FLOAT_BLOCK_SIZE, 1));
             }
             if (embed % FLOAT_VECTOR_SIZE > 0) {
                 SetMask(embed % FLOAT_VECTOR_SIZE);
                 AscendC::Div<float, false>(goUbTensor32[embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                            goUbTensor32[embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                            tvUbTensor,
                                            (uint64_t)0, curRowNum,
                                            AscendC::BinaryRepeatParams(1, 1, 0, embedRound / FLOAT_BLOCK_SIZE,
                                                                        embedRound / FLOAT_BLOCK_SIZE, 1));
                 AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
             }
             AscendC::PipeBarrier<PIPE_V>();
 
             // *** go = castfp32to16(go)
             if (std::is_same<ElementOutput, bfloat16_t>::value) {
                 AscendC::Cast<ElementOutput, float, false>(
                     goUbTensor16, goUbTensor32,
                     AscendC::RoundMode::CAST_RINT, (uint64_t)0,
                     (curRowNum * embedRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                     AscendC::UnaryRepeatParams(1, 1, 4, 8));
             } else {
                 AscendC::Cast<ElementOutput, float, false>(
                     goUbTensor16, goUbTensor32,
                     AscendC::RoundMode::CAST_NONE, (uint64_t)0,
                     (curRowNum * embedRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                     AscendC::UnaryRepeatParams(1, 1, 4, 8));
             }
             AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
             AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
 
             // ***move O to GM
             CopyOToGm(
                 gOutput, curRowNum, qSBlockSize, embed,
                 embedRound, qNThisSubBlock, oHiddenSize);
         }
     }
 
     CATLASS_DEVICE
     void operator()(
         AscendC::GlobalTensor<ElementOutput> gOutput,
         AscendC::GlobalTensor<ElementInput> gInput,
         const LayoutOutput &layoutOutput,
         const LayoutInput &layoutInput,
         GemmCoord actualBlockShape,
         uint32_t qSBlockSize, uint32_t qNBlockSize,
         uint32_t isFirstStackTile, uint32_t isLastStackTile, uint32_t curStackTileMod)
     {
         uint32_t rowNum = actualBlockShape.m();
         uint32_t embed = actualBlockShape.n();
 
         uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
         uint32_t subBlockNum = AscendC::GetSubBlockNum();
 
         uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;
         uint32_t qNThisSubBlock = (qNBlockSize == 1) ?
             0: (subBlockIdx == 1) ?
             (qNBlockSize - qNSplitSubBlock) : qNSplitSubBlock;
         uint32_t inRowSplitSubBlock = (qNBlockSize == 1) ?
             (qSBlockSize / subBlockNum) : (qSBlockSize * qNSplitSubBlock);
         uint32_t inRowActualThisSubBlock = (subBlockIdx == 1) ?
             (rowNum - inRowSplitSubBlock) : inRowSplitSubBlock;
         uint32_t inRowOffsetThisSubBlock = subBlockIdx * inRowSplitSubBlock;
         uint32_t outRowOffsetThisSubBlock = (qNBlockSize == 1) ?
             inRowOffsetThisSubBlock : 0;
         uint32_t outColOffsetThisSubBlock = (qNBlockSize == 1) ?
             0 : subBlockIdx * qNSplitSubBlock * embed;
 
         if (inRowActualThisSubBlock > 0) {
             int64_t offsetOutput =
                 layoutOutput.GetOffset(MatrixCoord(outRowOffsetThisSubBlock, outColOffsetThisSubBlock));
             auto gOutputThisSubBlock = gOutput[offsetOutput];
             auto layoutOutputThisSubBlock = layoutOutput;
 
             int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(inRowOffsetThisSubBlock, 0));
             auto gInputThisSubBlock = gInput[offsetInput];
             auto layoutInputThisSubBlock = layoutInput.GetTileLayout(MatrixCoord(inRowActualThisSubBlock, embed));
             SubCoreCompute(
                 gOutputThisSubBlock,
                 gInputThisSubBlock,
                 layoutOutputThisSubBlock,
                 layoutInputThisSubBlock,
                 qNThisSubBlock, isFirstStackTile, isLastStackTile, curStackTileMod);
         }
     }
 
 private:
     AscendC::LocalTensor<float> loUbTensor;
     AscendC::LocalTensor<float> dmUbTensor;
     AscendC::LocalTensor<float> hmUbTensor;
     AscendC::LocalTensor<float> glUbTensor;
     AscendC::LocalTensor<float> tvUbTensor;
     AscendC::LocalTensor<ElementOutput> goUbTensor16;
     AscendC::LocalTensor<float> goUbTensor32;
 };
 }
 
 #endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_NO_SPLIT_ROW_HPP