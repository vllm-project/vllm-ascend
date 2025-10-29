/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #ifndef CATLASS_EPILOGUE_TILE_TILE_ELEMWISE_MULS_HPP
 #define CATLASS_EPILOGUE_TILE_TILE_ELEMWISE_MULS_HPP
 
 #include "catlass/gemm/helper.hpp"
 
 namespace Catlass::Epilogue::Tile{
 template<
     class ArchTag_,
     class ComputeType_,
     uint32_t COMPUTE_LENGTH_
 >
 struct TileElemWiseMuls{
     using ArchTag = ArchTag_;
     using ElementCompute = typename ComputeType_::Element;
 
     static constexpr uint32_t COMPUTE_LENGTH = COMPUTE_LENGTH_;
 
     CATLASS_DEVICE
     TileElemWiseMuls(){}
 
     CATLASS_DEVICE
     void operator()(
         AscendC::LocalTensor<ElementCompute> dstLocal,
         AscendC::LocalTensor<ElementCompute> srcTensor,
         ElementCompute scalar
     ){
         AscendC::Muls(dstLocal, srcTensor, scalar, COMPUTE_LENGTH);
     }
 };
 }
 
 #endif // CATLASS_EPILOGUE_TILE_TILE_ELEMWISE_MULS_HPP