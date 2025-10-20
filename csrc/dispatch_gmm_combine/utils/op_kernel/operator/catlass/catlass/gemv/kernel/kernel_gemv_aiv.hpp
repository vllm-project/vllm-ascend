/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMV_KERNLE_GEMV_AIV_HPP
#define CATLASS_GEMV_KERNLE_GEMV_AIV_HPP
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemv_coord.hpp"

namespace Catlass::Gemv::Kernel {

// tmeplate for gemv kernle, Compute z = αAx + βy
template <
    class BlockGemv_,
    class BlockEpilogue_
>
class KernelGemvAiv {
public:
    using BlockGemv = BlockGemv_;
    using ArchTag = typename BlockGemv::ArchTag;
    using UBTileShape = typename BlockGemv::UBTileShape;
    using ElementA = typename BlockGemv::ElementA;
    using LayoutA = typename BlockGemv::LayoutA;
    using ElementX = typename BlockGemv::ElementX;
    using LayoutX = typename BlockGemv::LayoutX;
    using ElementY = typename BlockGemv::ElementY;
    using LayoutY = typename BlockGemv::LayoutY;
    using ElementAccumulator = typename BlockGemv::ElementAccumulator;


    /// Parameters structure
    struct Params {
        // Data members
        GemvCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrX;
        LayoutX layoutX;
        GM_ADDR ptrY;
        LayoutY layoutY;
        GM_ADDR ptrZ;
        float alpha;
        float beta;
        uint32_t split;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemvCoord const &problemShape_,  GM_ADDR ptrA_, LayoutA layoutA_,  GM_ADDR ptrX_,LayoutX layoutX_,
        GM_ADDR ptrY_,LayoutY layoutY_,GM_ADDR ptrZ_,float alpha_,float beta_,uint32_t split_)
            : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), ptrX(ptrX_),layoutX(layoutX_),
            ptrY(ptrY_),layoutY(layoutY_),ptrZ(ptrZ_),alpha(alpha_),beta(beta_),split(split_) {}
    };

    //TODO: add arguments
    struct Arguments {
        GemvCoord problemShape;
        GM_ADDR ptrA;
        GM_ADDR ptrX;
        GM_ADDR ptrY;
        GM_ADDR ptrZ;
        float alpha;
        float beta;
        uint32_t split;
    };

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return sizeof(ElementY) * args.problemShape.m();
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        GemvCoord problemShape = args.problemShape;
        uint32_t m = problemShape.m();
        uint32_t n = problemShape.n();
        LayoutA layoutA{m, n};
        LayoutX layoutX{n};
        LayoutY layoutY{m};
        Params params{problemShape, args.ptrA, layoutA, args.ptrX, layoutX, args.ptrY, layoutY, args.ptrZ, args.alpha, args.beta, args.split};
        return params;
    }

    // Methods
    CATLASS_DEVICE
    KernelGemvAiv(){}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params){
    };

    /// Executes one Matmul
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params) {
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params) {
        AscendC::SetAtomicNone();
        Arch::Resource<ArchTag> resource;
        BlockGemv blockGemv(resource);
        uint32_t align = BYTE_PER_C0 / sizeof(ElementA);
        uint32_t maxmPerBlock_round = RoundUp(UBTileShape::M,align);
        uint32_t maxnPerBlock_round = RoundUp(UBTileShape::N,align);

        //add split k
        uint32_t N_Split = RoundDown(params.problemShape.n(),params.split)/params.split;
        uint32_t Mloopnum = CeilDiv(params.problemShape.m(),maxmPerBlock_round);
        int32_t loopnum;
        float Realbeta= params.beta;
        if constexpr (std::is_same_v<LayoutA, Catlass::layout::ColumnMajor>){
            loopnum = Mloopnum * params.split;
            Realbeta = params.beta - 1.0f;
        }else{
            loopnum = Mloopnum;
        }
            
        uint32_t offset_matrix;
        uint32_t offset_vector_out;
        uint32_t offset_vector_in = 0;

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementX> gmX;
        gmX.SetGlobalBuffer((__gm__ ElementX *)params.ptrX);
        AscendC::GlobalTensor<ElementY> gmY;
        gmY.SetGlobalBuffer((__gm__ ElementY *)params.ptrY);
        AscendC::GlobalTensor<ElementY> gmZ;
        gmZ.SetGlobalBuffer((__gm__ ElementY *)params.ptrZ);
        uint32_t aiv_num = AscendC::GetBlockNum()*AscendC::GetTaskRation();
        for(uint32_t loop_id = 0;loop_id < loopnum;loop_id++){
            uint32_t aiv_id = AscendC::GetBlockIdx();   
            if(loop_id % aiv_num != aiv_id)continue;
            uint32_t m_catlassual = ((int32_t)loop_id > (int32_t)(loopnum - params.split - 1) ) ? params.problemShape.m() - ((loop_id/params.split) * maxmPerBlock_round) : maxmPerBlock_round;
            uint32_t n_catlassual = params.problemShape.n();

            if constexpr (std::is_same_v<LayoutA, Catlass::layout::ColumnMajor>) {
                offset_matrix = (loop_id % params.split) * N_Split*params.problemShape.m()+(loop_id/params.split) * maxmPerBlock_round;
                offset_vector_out = (loop_id/params.split) * maxmPerBlock_round;
                offset_vector_in = (loop_id % params.split) * N_Split; 
                
                if((loop_id%params.split) == params.split - 1){
                    n_catlassual = params.problemShape.n() - N_Split * (params.split - 1);
                }
                else{
                    n_catlassual = N_Split;
                }
            } else {
                offset_matrix = loop_id * maxmPerBlock_round * params.problemShape.n();
                offset_vector_out = loop_id * maxmPerBlock_round;
            }
            GemvCoord actualBlockShape = GemvCoord{m_catlassual,n_catlassual};
            
            float realbeta = (loop_id % params.split == 0) ? Realbeta:0.0f;

            blockGemv(gmA[offset_matrix], params.layoutA,
                gmX[offset_vector_in], params.layoutX,
                gmY[offset_vector_out], params.layoutY,
                gmZ[offset_vector_out],
                actualBlockShape,
                params.alpha,
                realbeta
            );

        }
    }
};

} 

#endif // CATLASS_GEMV_KERNLE_GEMV_AIV_HPP