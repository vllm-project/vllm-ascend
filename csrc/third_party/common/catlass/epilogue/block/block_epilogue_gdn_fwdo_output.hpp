
#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDO_OUTPUT_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDO_OUTPUT_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

namespace Catlass::Epilogue::Block {

template <
    class HOutputType_,
    class GInputType_,
    class AInputType_,
    class HInputType_
>
class BlockEpilogue <
    EpilogueAtlasA2GDNFwdOOutput,
    HOutputType_,
    GInputType_,
    AInputType_,
    HInputType_
> {
public:
    // Type aliases
    using DispatchPolicy = EpilogueAtlasA2GDNFwdOOutput;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using HElementOutput = typename HOutputType_::Element;
    using GElementInput = typename GInputType_::Element;
    using AElementInput = typename AInputType_::Element;
    using HElementInput = typename HInputType_::Element;
 
    // using CopyGmToUbInput = Tile::CopyGm2Ub<ArchTag, InputType_>;
    // using CopyUbToGmOutput = Tile::CopyUb2Gm<ArchTag, OutputType_>;

    static constexpr uint32_t HALF_ELENUM_PER_BLK = 16;
    static constexpr uint32_t FLOAT_ELENUM_PER_BLK = 8;
    static constexpr uint32_t HALF_ELENUM_PER_VECCALC = 128;
    static constexpr uint32_t FLOAT_ELENUM_PER_VECCALC = 64;
    static constexpr uint32_t UB_TILE_SIZE = 16384;  // 64 * 128 * 2B
    static constexpr uint32_t UB_LINE_SIZE = 512;   // 128 * 2 * 2B
    static constexpr uint32_t HALF_ELENUM_PER_LINE = 256;    // 128 * 2
    static constexpr uint32_t FLOAT_ELENUM_PER_LINE = 128;   // 128
    static constexpr uint32_t MULTIPLIER = 2;

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource)
    {
        constexpr uint32_t BASE = 0;

        constexpr uint32_t FLOAT_UB_TENSOR_SIZE = 64 * UB_LINE_SIZE;
        constexpr uint32_t G_HALF_UB_TENSOR_SIZE = 2 * UB_LINE_SIZE;
        constexpr uint32_t G_UB_TENSOR_SIZE = 2 * UB_LINE_SIZE;
        constexpr uint32_t GBRCLEFTCAST_UB_TENSOR_SIZE = 80 * UB_LINE_SIZE;
        constexpr uint32_t GBRCUP_UB_TENSOR_SIZE = 64 * UB_LINE_SIZE;
        constexpr uint32_t HALF_UB_TENSOR_SIZE = 32 * UB_LINE_SIZE;
        constexpr uint32_t MASK_UB_TENSOR_SIZE = 64 * UB_LINE_SIZE;

        constexpr uint32_t MASK_UB_TENSOR_OFFSET = BASE;
        constexpr uint32_t GBRCLEFTCAST_UB_TENSOR_OFFSET = MASK_UB_TENSOR_OFFSET + MASK_UB_TENSOR_SIZE;
        constexpr uint32_t GBRCUP_UB_TENSOR_OFFSET = GBRCLEFTCAST_UB_TENSOR_OFFSET + GBRCLEFTCAST_UB_TENSOR_SIZE;
        constexpr uint32_t G_HALF_UB_TENSOR_OFFSET = GBRCUP_UB_TENSOR_OFFSET + GBRCUP_UB_TENSOR_SIZE;
        constexpr uint32_t SHARE_TENSOR_OFFSET = G_HALF_UB_TENSOR_OFFSET + G_HALF_UB_TENSOR_SIZE;

        // boolUbTensor = resource.ubBuf.template GetBufferByByte<MaskElementInput>(BOOL_UB_TENSOR_OFFSET);
        gbrcleftcastUbTensor = resource.ubBuf.template GetBufferByByte<float>(GBRCLEFTCAST_UB_TENSOR_OFFSET);
        gbrcuphalfUbTensor = resource.ubBuf.template GetBufferByByte<half>(GBRCUP_UB_TENSOR_OFFSET);
        // gbrcupfloatUbTensor = resource.ubBuf.template GetBufferByByte<float>(GBRCUP_UB_TENSOR_OFFSET);
        ghalfUbTensor = resource.ubBuf.template GetBufferByByte<half>(G_HALF_UB_TENSOR_OFFSET);
        shareBuffer_ = resource.ubBuf.template GetBufferByByte<uint8_t>(SHARE_TENSOR_OFFSET);

        constexpr uint32_t ATTN_UB_TENSOR_OFFSET = SHARE_TENSOR_OFFSET + HALF_UB_TENSOR_SIZE;
        constexpr uint32_t H_UB_TENSOR_OFFSET = ATTN_UB_TENSOR_OFFSET + HALF_UB_TENSOR_SIZE;
        constexpr uint32_t G_UB_TENSOR_OFFSET = H_UB_TENSOR_OFFSET + HALF_UB_TENSOR_SIZE;
        constexpr uint32_t OUT_TENSOR_OFFSET = G_UB_TENSOR_OFFSET + G_UB_TENSOR_SIZE;
        
        aUbTensor = resource.ubBuf.template GetBufferByByte<AElementInput>(ATTN_UB_TENSOR_OFFSET);
        hUbTensor = resource.ubBuf.template GetBufferByByte<HElementInput>(H_UB_TENSOR_OFFSET);
        gUbTensor = resource.ubBuf.template GetBufferByByte<GElementInput>(G_UB_TENSOR_OFFSET);
        outputFPUbTensor = resource.ubBuf.template GetBufferByByte<half>(OUT_TENSOR_OFFSET);
        outputBFUbTensor = resource.ubBuf.template GetBufferByByte<HElementOutput>(OUT_TENSOR_OFFSET);
        
    }
    CATLASS_DEVICE
    ~BlockEpilogue()
    {}

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<HElementOutput> hOutput,
        AscendC::GlobalTensor<GElementInput> gInput,
        AscendC::GlobalTensor<AElementInput> attnInput,
        AscendC::GlobalTensor<HElementInput> hInput,
        float scale,
        uint32_t chunkSize,
        uint32_t kHeadDim,
        uint32_t vHeadDim
        // Arch::CrossCoreFlag cube2Done,
        // Arch::CrossCoreFlag cube3Done
        )
    {
        uint32_t mCVActual = chunkSize;
        uint32_t nCVActual = vHeadDim;
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t mCVActualPerSubBlock = CeilDiv(mCVActual, subBlockNum);
        uint32_t mCVActualThisSubBlock = (subBlockIdx == 0) ? mCVActualPerSubBlock : (mCVActual - mCVActualPerSubBlock);
        uint32_t mCVOffset = subBlockIdx * mCVActualPerSubBlock;
        uint32_t nOffset = 0;
        int64_t offsetCV = mCVOffset * nCVActual + nOffset;

        uint32_t gbrcStart, gbrcRealStart, gbrcReptime, gbrcEffStart, gbrcEffEnd;
        if(subBlockIdx==0)
        {
            gbrcStart = 0;
            gbrcRealStart = 0;
            gbrcReptime = (mCVActualThisSubBlock + 8 - 1) / 8;
        }
        else 
        {
            gbrcStart = mCVActualPerSubBlock;
            gbrcRealStart = gbrcStart & ~15;
            gbrcReptime = (mCVActual - gbrcRealStart + 8 - 1) / 8;   
        }
        gbrcEffStart = gbrcStart-gbrcRealStart;
        gbrcEffEnd = gbrcEffStart + mCVActualThisSubBlock;
        uint32_t dstShape_[2] = {gbrcReptime*8, nCVActual};
        uint32_t srcShape_[2] = {gbrcReptime*8, 1};

        AscendC::ResetMask();
        AscendC::GlobalTensor<HElementOutput> hOutputThisSubBlock = hOutput[offsetCV];
        AscendC::GlobalTensor<AElementInput> attnInputThisSubBlock = attnInput[offsetCV];
        AscendC::GlobalTensor<HElementInput> hInputThisSubBlock = hInput[offsetCV];
        AscendC::GlobalTensor<GElementInput> gInputThisSubBlock = gInput;
        AscendC::DataCopyExtParams copyParams{1, (uint32_t)(mCVActualThisSubBlock * nCVActual * sizeof(half)), 0, 0, 0};
        AscendC::DataCopyParams gUbParams{1, (uint16_t)(mCVActual*sizeof(float)), 0, 0};
        AscendC::DataCopyPadParams gUbPadParams{false, 0, 0, 0};
        // AscendC::PipeBarrier<PIPE_ALL>();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0); 
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1); 
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2); 

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0); 
        // AscendC::DataCopy(gUbTensor, gInputThisSubBlock, (mCVActual + 8 - 1)/8 * 8);
        AscendC::DataCopyPad(gUbTensor, gInputThisSubBlock, gUbParams, gUbPadParams);



        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0); 
        // AscendC::PipeBarrier<PIPE_ALL>();

        // AscendC::PipeBarrier<PIPE_ALL>();
 
        // AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0); 
        AscendC::Cast(ghalfUbTensor, gUbTensor, AscendC::RoundMode::CAST_NONE, mCVActual);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(ghalfUbTensor, ghalfUbTensor, mCVActual);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Broadcast<half, 2, 1>(gbrcuphalfUbTensor, ghalfUbTensor[gbrcRealStart], dstShape_, srcShape_, shareBuffer_);
        AscendC::PipeBarrier<PIPE_V>();


        // Arch::CrossCoreWaitFlag(cube2Done);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1); 
        AscendC::DataCopy(hUbTensor, hInputThisSubBlock, mCVActualThisSubBlock * nCVActual);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1); 


        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1); 
        AscendC::Mul(gbrcuphalfUbTensor, hUbTensor, gbrcuphalfUbTensor[gbrcEffStart*nCVActual], mCVActualThisSubBlock * nCVActual);
        AscendC::PipeBarrier<PIPE_V>();

        // Arch::CrossCoreWaitFlag(cube3Done);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2); 
        AscendC::DataCopy(aUbTensor, attnInputThisSubBlock, mCVActualThisSubBlock * nCVActual);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2); 


        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2); 
        AscendC::Add(gbrcuphalfUbTensor, aUbTensor, gbrcuphalfUbTensor, mCVActualThisSubBlock * nCVActual);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(outputFPUbTensor, gbrcuphalfUbTensor, (half)scale, mCVActualThisSubBlock * nCVActual);

        if constexpr(!std::is_same<HElementOutput, half>::value) {
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(gbrcleftcastUbTensor, outputFPUbTensor, AscendC::RoundMode::CAST_NONE, mCVActualThisSubBlock * nCVActual);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(outputBFUbTensor, gbrcleftcastUbTensor, AscendC::RoundMode::CAST_RINT, mCVActualThisSubBlock * nCVActual);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopyPad(hOutputThisSubBlock, outputBFUbTensor, copyParams);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopyPad(hOutputThisSubBlock, outputFPUbTensor, copyParams);
        }

    }

private:
    // uint32_t goFlag = 1;
    AscendC::LocalTensor<float> gbrcleftcastUbTensor;
    AscendC::LocalTensor<half> gbrcuphalfUbTensor;
    AscendC::LocalTensor<half> ghalfUbTensor;
    AscendC::LocalTensor<uint8_t> shareBuffer_;

    AscendC::LocalTensor<AElementInput> aUbTensor;
    AscendC::LocalTensor<HElementInput> hUbTensor;
    AscendC::LocalTensor<GElementInput> gUbTensor;
    AscendC::LocalTensor<half> outputFPUbTensor;
    AscendC::LocalTensor<HElementOutput> outputBFUbTensor;

};
}

#endif


