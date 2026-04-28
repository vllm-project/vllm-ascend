#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDO_QKMASK_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDO_QKMASK_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

namespace Catlass::Epilogue::Block {

template <
    class AOutputType_,
    class GInputType_,
    class AInputType_,
    class MaskInputType_
>
class BlockEpilogue <
    EpilogueAtlasA2GDNFwdOQkmask,
    AOutputType_,
    GInputType_,
    AInputType_,
    MaskInputType_
> {
public:
    // Type aliases
    using DispatchPolicy = EpilogueAtlasA2GDNFwdOQkmask;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using AElementOutput = typename AOutputType_::Element;
    using GElementInput = typename GInputType_::Element;
    using AElementInput = typename AInputType_::Element;
    using MaskElementInput = typename MaskInputType_::Element;

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

        maskUbTensor = resource.ubBuf.template GetBufferByByte<half>(MASK_UB_TENSOR_OFFSET);
        gbrcleftcastUbTensor = resource.ubBuf.template GetBufferByByte<float>(GBRCLEFTCAST_UB_TENSOR_OFFSET);
        gbrcuphalfUbTensor = resource.ubBuf.template GetBufferByByte<half>(GBRCUP_UB_TENSOR_OFFSET);
        gbrcupfloatUbTensor = resource.ubBuf.template GetBufferByByte<float>(GBRCUP_UB_TENSOR_OFFSET);
        ghalfUbTensor = resource.ubBuf.template GetBufferByByte<half>(G_HALF_UB_TENSOR_OFFSET);
        shareBuffer_ = resource.ubBuf.template GetBufferByByte<uint8_t>(SHARE_TENSOR_OFFSET);
        
        constexpr uint32_t G_UB_TENSOR_OFFSET_PING = SHARE_TENSOR_OFFSET + HALF_UB_TENSOR_SIZE;
        constexpr uint32_t OUT_TENSOR_OFFSET_PING = G_UB_TENSOR_OFFSET_PING + G_UB_TENSOR_SIZE;

        gUbTensor_ping = resource.ubBuf.template GetBufferByByte<GElementInput>(G_UB_TENSOR_OFFSET_PING);
        outputFPUbTensor_ping = resource.ubBuf.template GetBufferByByte<half>(OUT_TENSOR_OFFSET_PING);
        outputBFUbTensor_ping = resource.ubBuf.template GetBufferByByte<AElementOutput>(OUT_TENSOR_OFFSET_PING);

        // pub 121KB
        // vec1: 17KB  VS  vec2: 49KB
        // share 预留16KB
        constexpr uint32_t G_UB_TENSOR_OFFSET_PONG = OUT_TENSOR_OFFSET_PING + HALF_UB_TENSOR_SIZE + 64 * UB_LINE_SIZE;
        constexpr uint32_t OUT_TENSOR_OFFSET_PONG = G_UB_TENSOR_OFFSET_PONG + G_UB_TENSOR_SIZE;

        gUbTensor_pong = resource.ubBuf.template GetBufferByByte<GElementInput>(G_UB_TENSOR_OFFSET_PONG);
        outputFPUbTensor_pong = resource.ubBuf.template GetBufferByByte<half>(OUT_TENSOR_OFFSET_PONG);
        outputBFUbTensor_pong = resource.ubBuf.template GetBufferByByte<AElementOutput>(OUT_TENSOR_OFFSET_PONG);

        
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {}

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<AElementOutput> maskOutput,
        AscendC::GlobalTensor<GElementInput> gInput,
        AscendC::GlobalTensor<AElementInput> attnInput,
        AscendC::GlobalTensor<MaskElementInput> boolInput,
        uint32_t fullChunkSize,
        uint32_t chunkSize,
        uint32_t kHeadDim,
        uint32_t vHeadDim
        )
    {
        uint32_t mActual = chunkSize;
        uint32_t nActual = chunkSize;
        uint32_t alignedNActual = (nActual+15)/16*16;

        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();

        uint32_t blockIdx = AscendC::GetBlockIdx();


        uint32_t mActualPerSubBlock = CeilDiv(mActual, subBlockNum);
        uint32_t mActualThisSubBlock = (subBlockIdx == 0) ? mActualPerSubBlock : (mActual - mActualPerSubBlock);
        uint32_t mOffset = subBlockIdx * mActualPerSubBlock;
        uint32_t nOffset = 0;
        int64_t offsetA = mOffset * nActual + nOffset;

        uint32_t gbrcStart, gbrcRealStart, gbrcReptime, gbrcEffStart, gbrcEffEnd;
        if(subBlockIdx==0)
        {
            gbrcStart = 0;
            gbrcRealStart = 0;
            gbrcReptime = (mActualThisSubBlock + 8 - 1) / 8;
        }
        else 
        {
            gbrcStart = mActualPerSubBlock;
            gbrcRealStart = gbrcStart & ~15;
            gbrcReptime = (mActual - gbrcRealStart + 8 - 1) / 8;   
        }
        gbrcEffStart = gbrcStart-gbrcRealStart;
        gbrcEffEnd = gbrcEffStart + mActualThisSubBlock;

        uint32_t dstUpShape_[2] = {mActualThisSubBlock, alignedNActual};
        uint32_t srcUpShape_[2] = {1, alignedNActual};
        uint32_t dstLeftShape_[2] = {gbrcReptime*8, alignedNActual};
        uint32_t srcLeftShape_[2] = {gbrcReptime*8, 1};

        AscendC::ResetMask();
        AscendC::GlobalTensor<AElementOutput> maskOutputThisSubBlock = maskOutput[offsetA];
        AscendC::GlobalTensor<AElementInput> attnInputThisSubBlock = attnInput[offsetA];
        AscendC::GlobalTensor<GElementInput> gInputThisSubBlock = gInput;

        AscendC::DataCopyParams aInputUbParams{(uint16_t)mActualThisSubBlock, (uint16_t)(nActual*sizeof(half)), 0, 0};
        AscendC::DataCopyPadParams aInputUbPadParams{false, 0, 0, 0};
        AscendC::DataCopyExtParams aOutputUbParams{(uint16_t)mActualThisSubBlock, (uint32_t)(nActual*sizeof(half)), 0, 0, 0};

        AscendC::DataCopyParams gUbParams{1, (uint16_t)(mActual*sizeof(float)), 0, 0};
        AscendC::DataCopyPadParams gUbPadParams{false, 0, 0, 0};

        pingpongFlag = isFirst ? 0 : 4;
        AscendC::LocalTensor<GElementInput> gUbTensor = pingpongFlag == 0 ? gUbTensor_ping : gUbTensor_pong;
        AscendC::LocalTensor<half> outputFPUbTensor = pingpongFlag == 0 ? outputFPUbTensor_ping : outputFPUbTensor_pong;
        AscendC::LocalTensor<AElementOutput> outputBFUbTensor = pingpongFlag == 0 ? outputBFUbTensor_ping : outputBFUbTensor_pong;

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pingpongFlag);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
        AscendC::DataCopyPad(gUbTensor, gInputThisSubBlock, gUbParams, gUbPadParams);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
        AscendC::Broadcast<float, 2, 0>(gbrcupfloatUbTensor, gUbTensor, dstUpShape_, srcUpShape_, shareBuffer_);
        AscendC::Broadcast<float, 2, 1>(gbrcleftcastUbTensor, gUbTensor[gbrcRealStart], dstLeftShape_, srcLeftShape_, shareBuffer_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(gbrcupfloatUbTensor, gbrcleftcastUbTensor[gbrcEffStart*alignedNActual], gbrcupfloatUbTensor, mActualThisSubBlock * alignedNActual);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(gbrcuphalfUbTensor, gbrcupfloatUbTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock * alignedNActual); 
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mins(gbrcuphalfUbTensor, gbrcuphalfUbTensor, (half)0.0, mActualThisSubBlock * alignedNActual);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(gbrcuphalfUbTensor, gbrcuphalfUbTensor, mActualThisSubBlock * alignedNActual);
        AscendC::PipeBarrier<PIPE_V>();
        // chunk64 128B 4DB      {na/16, na/16, 4} 
        // chunk128 256B 8DB     {na/16, na/16, 8}
        AscendC::Mul(gbrcuphalfUbTensor, gbrcuphalfUbTensor, maskUbTensor[gbrcStart*fullChunkSize], alignedNActual, mActualThisSubBlock, 
        {1,1,1, static_cast<uint8_t>(alignedNActual/16), static_cast<uint8_t>(alignedNActual/16), static_cast<uint8_t>(fullChunkSize/16)});
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pingpongFlag);
        if(chunkSize==fullChunkSize) AscendC::DataCopy(outputFPUbTensor, attnInputThisSubBlock, mActualThisSubBlock*nActual);
        else AscendC::DataCopyPad(outputFPUbTensor, attnInputThisSubBlock, aInputUbParams, aInputUbPadParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + pingpongFlag);
        AscendC::Mul(outputFPUbTensor, outputFPUbTensor, gbrcuphalfUbTensor, mActualThisSubBlock * alignedNActual);

        if constexpr(!std::is_same<AElementOutput, half>::value) {
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(gbrcleftcastUbTensor, outputFPUbTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock * alignedNActual);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(outputBFUbTensor, gbrcleftcastUbTensor, AscendC::RoundMode::CAST_RINT, mActualThisSubBlock * alignedNActual);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
            if(chunkSize==fullChunkSize) AscendC::DataCopy(maskOutputThisSubBlock, outputBFUbTensor, mActualThisSubBlock*nActual);
            else AscendC::DataCopyPad(maskOutputThisSubBlock, outputBFUbTensor, aOutputUbParams);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
            if(chunkSize==fullChunkSize) AscendC::DataCopy(maskOutputThisSubBlock, outputFPUbTensor, mActualThisSubBlock*nActual);
            else AscendC::DataCopyPad(maskOutputThisSubBlock, outputFPUbTensor, aOutputUbParams);
        }
        isFirst = false;
    }

private:
    uint32_t pingpongFlag = 0;
    bool isFirst = true;
    AscendC::LocalTensor<half> maskUbTensor;
    AscendC::LocalTensor<float> gbrcleftcastUbTensor;
    AscendC::LocalTensor<half> gbrcuphalfUbTensor;
    AscendC::LocalTensor<float> gbrcupfloatUbTensor;
    AscendC::LocalTensor<half> ghalfUbTensor;
    AscendC::LocalTensor<uint8_t> shareBuffer_;
    
    AscendC::LocalTensor<GElementInput> gUbTensor_ping;
    AscendC::LocalTensor<half> outputFPUbTensor_ping;
    AscendC::LocalTensor<AElementOutput> outputBFUbTensor_ping;

    AscendC::LocalTensor<GElementInput> gUbTensor_pong;
    AscendC::LocalTensor<half> outputFPUbTensor_pong;
    AscendC::LocalTensor<AElementOutput> outputBFUbTensor_pong;


};
}

#endif

