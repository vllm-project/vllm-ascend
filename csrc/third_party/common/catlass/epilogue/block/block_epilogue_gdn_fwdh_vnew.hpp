#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDH_VNEW_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDH_VNEW_HPP
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"



namespace Catlass::Epilogue::Block {

template <
    class VOutputType_,
    class GInputType_,
    class UInputType_,
    class WSInputType_
>
class BlockEpilogue <
    EpilogueAtlasA2GDNFwdHVnew,
    VOutputType_,
    GInputType_,
    UInputType_,
    WSInputType_
> {
public:
    // Type aliases
    using DispatchPolicy = EpilogueAtlasA2GDNFwdHVnew;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using VElementOutput = typename VOutputType_::Element;
    using GElementInput = typename GInputType_::Element;
    using UElementInput = typename UInputType_::Element;
    using WSElementInput = typename WSInputType_::Element;
 

    // using ElementOutput = bfloat16_t;
    // using ElementInput = bfloat16_t;

    // using LayoutOutput = typename OutputType_::Layout;
    // using LayoutInput = typename InputType_::Layout;

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
        // chunk = 128 dim = 128
        // chunk * dim * 2B / 2 : 32 line : 16KB
        // chunk * dim * 2B / 2 : 32 line : 16KB
        // chunk * 4B : 1 line
        // chunk * 2B : 1 line
        // chunk * 2B : 1 line
        // constexpr uint32_t BASE = 6*UB_TILE_SIZE;
        constexpr uint32_t BASE = 0;

//        constexpr uint32_t U_UB_TENSOR_OFFSET = BASE; // 16*128*2B
//        constexpr uint32_t WS_UB_TENSOR_OFFSET = U_UB_TENSOR_OFFSET + 32 * UB_LINE_SIZE;
//        constexpr uint32_t SRC_UB_TENSOR_OFFSET = WS_UB_TENSOR_OFFSET + 32 * UB_LINE_SIZE;
//        constexpr uint32_t FLOAT_UB_TENSOR_OFFSET = SRC_UB_TENSOR_OFFSET + 32 * UB_LINE_SIZE;
//        constexpr uint32_t G_UB_TENSOR_OFFSET = FLOAT_UB_TENSOR_OFFSET + 64 * UB_LINE_SIZE;
//        constexpr uint32_t G_UB_HALF_TENSOR_OFFSET = G_UB_TENSOR_OFFSET + 1 * UB_LINE_SIZE;
//        constexpr uint32_t GSUB_UB_TENSOR_OFFSET = G_UB_HALF_TENSOR_OFFSET + 1 * UB_LINE_SIZE;
//        constexpr uint32_t SHARE_TENSOR_OFFSET = GSUB_UB_TENSOR_OFFSET + 1 * UB_LINE_SIZE;
//
//        uUbTensor = resource.ubBuf.template GetBufferByByte<half>(U_UB_TENSOR_OFFSET);
//        srcUbTensor = resource.ubBuf.template GetBufferByByte<UElementInput>(SRC_UB_TENSOR_OFFSET);
//        floatUbTensor = resource.ubBuf.template GetBufferByByte<float>(FLOAT_UB_TENSOR_OFFSET);
//        halfUbTensor = resource.ubBuf.template GetBufferByByte<half>(FLOAT_UB_TENSOR_OFFSET); //复用
//        gUbTensor = resource.ubBuf.template GetBufferByByte<GElementInput>(G_UB_TENSOR_OFFSET);
//        gUbhalfTensor = resource.ubBuf.template GetBufferByByte<half>(G_UB_HALF_TENSOR_OFFSET);
//        wsUbTensor = resource.ubBuf.template GetBufferByByte<half>(WS_UB_TENSOR_OFFSET);
//        gsubUbTensor = resource.ubBuf.template GetBufferByByte<half>(GSUB_UB_TENSOR_OFFSET);
//        shareBuffer_ = resource.ubBuf.template GetBufferByByte<uint8_t>(SHARE_TENSOR_OFFSET);


        constexpr uint32_t FLOAT_UB_TENSOR_SIZE = 64 * UB_LINE_SIZE;
        constexpr uint32_t HALF_UB_TENSOR_SIZE = 32 * UB_LINE_SIZE;
        constexpr uint32_t GSUB_UB_TENSOR_SIZE = 1 * UB_LINE_SIZE;
        constexpr uint32_t SHARE_TENSOR_SIZE = 1 * UB_LINE_SIZE;

        constexpr uint32_t FLOAT_UB_TENSOR_OFFSET = BASE;
        constexpr uint32_t HALF_UB_TENSOR_OFFSET = FLOAT_UB_TENSOR_OFFSET + FLOAT_UB_TENSOR_SIZE;
        constexpr uint32_t U_UB_TENSOR_OFFSET = HALF_UB_TENSOR_OFFSET + HALF_UB_TENSOR_SIZE;
        constexpr uint32_t WS_UB_TENSOR_OFFSET = U_UB_TENSOR_OFFSET + HALF_UB_TENSOR_SIZE;
        constexpr uint32_t G_UB_TENSOR_OFFSET = WS_UB_TENSOR_OFFSET + HALF_UB_TENSOR_SIZE;
        constexpr uint32_t VNEW_OUTPUT_UB_TENSOR_OFFSET = G_UB_TENSOR_OFFSET + GSUB_UB_TENSOR_SIZE;
        constexpr uint32_t VNEW_DECAY_UB_TENSOR_OFFSET = VNEW_OUTPUT_UB_TENSOR_OFFSET + HALF_UB_TENSOR_SIZE;


        floatUbTensor = resource.ubBuf.template GetBufferByByte<float>(FLOAT_UB_TENSOR_OFFSET);
        halfUbTensor = resource.ubBuf.template GetBufferByByte<half>(HALF_UB_TENSOR_OFFSET);

        uUbTensor_ping = resource.ubBuf.template GetBufferByByte<UElementInput>(U_UB_TENSOR_OFFSET);
        uUbHalfTensor_ping = resource.ubBuf.template GetBufferByByte<half>(U_UB_TENSOR_OFFSET);
        wsUbTensor_ping = resource.ubBuf.template GetBufferByByte<half>(WS_UB_TENSOR_OFFSET);
        gUbTensor_ping = resource.ubBuf.template GetBufferByByte<GElementInput>(G_UB_TENSOR_OFFSET);
        gUbHalfTensor_ping = resource.ubBuf.template GetBufferByByte<half>(G_UB_TENSOR_OFFSET);
        vNewOutputUbTensor_ping = resource.ubBuf.template GetBufferByByte<VElementOutput>(VNEW_OUTPUT_UB_TENSOR_OFFSET);
        vNewOutputUbHalfTensor_ping = resource.ubBuf.template GetBufferByByte<half>(VNEW_OUTPUT_UB_TENSOR_OFFSET);
        vNewDecayUbTensor_ping = resource.ubBuf.template GetBufferByByte<VElementOutput>(VNEW_DECAY_UB_TENSOR_OFFSET);
        vNewDecayUbHalfTensor_ping = resource.ubBuf.template GetBufferByByte<half>(VNEW_DECAY_UB_TENSOR_OFFSET);


        constexpr uint32_t U_UB_TENSOR_OFFSET_pong = VNEW_DECAY_UB_TENSOR_OFFSET + HALF_UB_TENSOR_SIZE;
        constexpr uint32_t WS_UB_TENSOR_OFFSET_pong = U_UB_TENSOR_OFFSET_pong + HALF_UB_TENSOR_SIZE;
        constexpr uint32_t G_UB_TENSOR_OFFSET_pong = WS_UB_TENSOR_OFFSET_pong + HALF_UB_TENSOR_SIZE;
        constexpr uint32_t VNEW_OUTPUT_UB_TENSOR_OFFSET_pong = G_UB_TENSOR_OFFSET_pong + GSUB_UB_TENSOR_SIZE;
        constexpr uint32_t VNEW_DECAY_UB_TENSOR_OFFSET_pong = VNEW_OUTPUT_UB_TENSOR_OFFSET_pong + HALF_UB_TENSOR_SIZE;

        constexpr uint32_t SHARE_TENSOR_OFFSET = VNEW_DECAY_UB_TENSOR_OFFSET_pong + HALF_UB_TENSOR_SIZE;

        uUbTensor_pong = resource.ubBuf.template GetBufferByByte<UElementInput>(U_UB_TENSOR_OFFSET_pong);
        uUbHalfTensor_pong = resource.ubBuf.template GetBufferByByte<half>(U_UB_TENSOR_OFFSET_pong);
        wsUbTensor_pong = resource.ubBuf.template GetBufferByByte<half>(WS_UB_TENSOR_OFFSET_pong);
        gUbTensor_pong = resource.ubBuf.template GetBufferByByte<GElementInput>(G_UB_TENSOR_OFFSET_pong);
        gUbHalfTensor_pong = resource.ubBuf.template GetBufferByByte<half>(G_UB_TENSOR_OFFSET_pong);
        vNewOutputUbTensor_pong = resource.ubBuf.template GetBufferByByte<VElementOutput>(VNEW_OUTPUT_UB_TENSOR_OFFSET_pong);
        vNewOutputUbHalfTensor_pong = resource.ubBuf.template GetBufferByByte<half>(VNEW_OUTPUT_UB_TENSOR_OFFSET_pong);
        vNewDecayUbTensor_pong = resource.ubBuf.template GetBufferByByte<VElementOutput>(VNEW_DECAY_UB_TENSOR_OFFSET_pong);
        vNewDecayUbHalfTensor_pong = resource.ubBuf.template GetBufferByByte<half>(VNEW_DECAY_UB_TENSOR_OFFSET_pong);


        shareBuffer_ = resource.ubBuf.template GetBufferByByte<uint8_t>(SHARE_TENSOR_OFFSET);

        // AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        // AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        // AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        // AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }


    float taylor_exp(float x, int n) {
        float sum = 1.0; // 第一项 (n=0) 是 1
        float term = 1.0; // 当前项的值
        for (int i = 1; i <= n; i++) {
            term = term * x / i; // 利用前一项推导当前项
            sum += term;
        }
        return sum;
    }


    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<VElementOutput> vnewOutput,
        AscendC::GlobalTensor<VElementOutput> vnewdecayOutput,
        AscendC::GlobalTensor<float> gInput,
        AscendC::GlobalTensor<UElementInput> uInput,
        AscendC::GlobalTensor<WSElementInput> wsInput,
        uint32_t chunkSize,
        uint32_t kHeadDim,
        uint32_t vHeadDim,
        Arch::CrossCoreFlag cube1Done
        // const LayoutOutput &layoutOutput,
        // const LayoutInput &LayoutInput    
        )
    {
        uint32_t mActual = chunkSize;
        uint32_t nkActual = kHeadDim;
        uint32_t nvActual = vHeadDim;

        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t mActualPerSubBlock = CeilDiv(mActual, subBlockNum);
        uint32_t mActualThisSubBlock = (subBlockIdx == 0) ? mActualPerSubBlock : (mActual - mActualPerSubBlock);
        uint32_t mOffset = subBlockIdx * mActualPerSubBlock;
        uint32_t nOffset = 0;
        // 当前场景内部一定连续
        // k [B, H, T, D]
        // g [B, H, T]
        // 在外部offset的基础上进一步offset
        // 当前asset kdim == vHeadDim
        int64_t offsetK = mOffset * nvActual + nOffset;
        int64_t offsetD = 0; // 因为要用最后一个数减去之前所有，所以全部读入

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

        AscendC::ResetMask();

        AscendC::GlobalTensor<VElementOutput> vnewOutputThisSubBlock = vnewOutput[offsetK];
        AscendC::GlobalTensor<VElementOutput> vnewdecayOutputThisSubBlock = vnewdecayOutput[offsetK];
        AscendC::GlobalTensor<float> gInputThisSubBlock = gInput;
        AscendC::GlobalTensor<UElementInput> uInputThisSubBlock = uInput[offsetK];
        AscendC::GlobalTensor<WSElementInput> wsInputThisSubBlock = wsInput[offsetK];
        AscendC::LocalTensor<half> vNewUbHalfTensor;

        pingpongFlag = isFirst ? 0 : 4;
        AscendC::LocalTensor<UElementInput> uUbTensor = pingpongFlag == 0 ? uUbTensor_ping : uUbTensor_pong;
        AscendC::LocalTensor<half> uUbHalfTensor = pingpongFlag == 0 ? uUbHalfTensor_ping : uUbHalfTensor_pong;
        AscendC::LocalTensor<half> wsUbTensor = pingpongFlag == 0 ? wsUbTensor_ping : wsUbTensor_pong;
        AscendC::LocalTensor<float> gUbTensor = pingpongFlag == 0 ? gUbTensor_ping : gUbTensor_pong;
        AscendC::LocalTensor<half> gUbHalfTensor = pingpongFlag == 0 ? gUbHalfTensor_ping : gUbHalfTensor_pong;
        AscendC::LocalTensor<VElementOutput> vNewOutputUbTensor = pingpongFlag == 0 ? vNewOutputUbTensor_ping : vNewOutputUbTensor_pong;
        AscendC::LocalTensor<half> vNewOutputUbHalfTensor = pingpongFlag == 0 ? vNewOutputUbHalfTensor_ping : vNewOutputUbHalfTensor_pong;
        AscendC::LocalTensor<VElementOutput> vNewDecayUbTensor = pingpongFlag == 0 ? vNewDecayUbTensor_ping : vNewDecayUbTensor_pong;
        AscendC::LocalTensor<half> vNewDecayUbHalfTensor = pingpongFlag == 0 ? vNewDecayUbHalfTensor_ping : vNewDecayUbHalfTensor_pong;

        AscendC::DataCopyParams gUbParams{1, (uint16_t)(mActual * sizeof(float)), 0, 0};
        AscendC::DataCopyPadParams gUbPadParams{false, 0, 0, 0};

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + pingpongFlag);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pingpongFlag);

//        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
//        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pingpongFlag);
        AscendC::DataCopyPad(gUbTensor, gInputThisSubBlock, gUbParams, gUbPadParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + pingpongFlag);

        // AscendC::Cast(gUbHalfTensor, gUbTensor, AscendC::RoundMode::CAST_NONE, mActual);

        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2 + pingpongFlag);
        float inputVal = gUbTensor.GetValue(mActual-1);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID2 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID2 + pingpongFlag);

        AscendC::Duplicate<float>(floatUbTensor, inputVal, mActual);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Sub<float>(gUbTensor, floatUbTensor, gUbTensor, mActual);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Exp(gUbTensor, gUbTensor, mActual);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(gUbHalfTensor, gUbTensor, AscendC::RoundMode::CAST_NONE, mActual);
        AscendC::PipeBarrier<PIPE_V>();

        uint32_t dstShape_[2] = {gbrcReptime*8, nvActual};
        uint32_t srcShape_[2] = {gbrcReptime*8, 1};
        AscendC::Broadcast<half, 2, 1>(halfUbTensor, gUbHalfTensor[gbrcRealStart], dstShape_, srcShape_, shareBuffer_);
        AscendC::PipeBarrier<PIPE_V>();

        Arch::CrossCoreWaitFlag(cube1Done);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
        if constexpr(!std::is_same<UElementInput, half>::value) {
            AscendC::DataCopy(uUbTensor, uInputThisSubBlock, mActualThisSubBlock * nvActual);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
            AscendC::Cast(floatUbTensor, uUbTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock * nvActual);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(uUbHalfTensor, floatUbTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock * nvActual);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            AscendC::DataCopy(uUbHalfTensor, uInputThisSubBlock, mActualThisSubBlock * nvActual);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
        }

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + pingpongFlag);
        AscendC::DataCopy(wsUbTensor, wsInputThisSubBlock, mActualThisSubBlock * nvActual);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + pingpongFlag);

        if constexpr(!std::is_same<VElementOutput, half>::value) {
            AscendC::Sub<half>(uUbHalfTensor, uUbHalfTensor, wsUbTensor, mActualThisSubBlock * nvActual);
            vNewUbHalfTensor = uUbHalfTensor;
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(floatUbTensor, uUbHalfTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock * nvActual);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(vNewOutputUbTensor, floatUbTensor, AscendC::RoundMode::CAST_RINT, mActualThisSubBlock * nvActual);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
            AscendC::DataCopy(vnewOutputThisSubBlock, vNewOutputUbTensor, mActualThisSubBlock * nvActual);
        } else {
            AscendC::Sub<half>(vNewOutputUbHalfTensor, uUbHalfTensor, wsUbTensor, mActualThisSubBlock * nvActual);
            vNewUbHalfTensor = vNewOutputUbHalfTensor;
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
            AscendC::DataCopy(vnewOutputThisSubBlock, vNewOutputUbHalfTensor, mActualThisSubBlock * nvActual);
        }

        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(vNewDecayUbHalfTensor, vNewUbHalfTensor, halfUbTensor[gbrcEffStart*nvActual], mActualThisSubBlock * nvActual);

        if constexpr(!std::is_same<VElementOutput, half>::value) {
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(floatUbTensor, vNewDecayUbHalfTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock * nvActual);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(vNewDecayUbTensor, floatUbTensor, AscendC::RoundMode::CAST_RINT, mActualThisSubBlock * nvActual);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
            AscendC::DataCopy(vnewdecayOutputThisSubBlock, vNewDecayUbTensor, mActualThisSubBlock * nvActual);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
            AscendC::DataCopy(vnewdecayOutputThisSubBlock, vNewDecayUbHalfTensor, mActualThisSubBlock * nvActual);
        }

        isFirst = false;
    }

private:
    uint32_t pingpongFlag = 0;
    bool isFirst = true;

    AscendC::LocalTensor<float> floatUbTensor;
    AscendC::LocalTensor<half> halfUbTensor;

    AscendC::LocalTensor<UElementInput> uUbTensor_ping;
    AscendC::LocalTensor<half> uUbHalfTensor_ping;
    AscendC::LocalTensor<half> wsUbTensor_ping;
    AscendC::LocalTensor<float> gUbTensor_ping;
    AscendC::LocalTensor<half> gUbHalfTensor_ping;
    AscendC::LocalTensor<VElementOutput> vNewOutputUbTensor_ping;
    AscendC::LocalTensor<half> vNewOutputUbHalfTensor_ping;
    AscendC::LocalTensor<VElementOutput> vNewDecayUbTensor_ping;
    AscendC::LocalTensor<half> vNewDecayUbHalfTensor_ping;

    AscendC::LocalTensor<UElementInput> uUbTensor_pong;
    AscendC::LocalTensor<half> uUbHalfTensor_pong;
    AscendC::LocalTensor<half> wsUbTensor_pong;
    AscendC::LocalTensor<float> gUbTensor_pong;
    AscendC::LocalTensor<half> gUbHalfTensor_pong;
    AscendC::LocalTensor<VElementOutput> vNewOutputUbTensor_pong;
    AscendC::LocalTensor<half> vNewOutputUbHalfTensor_pong;
    AscendC::LocalTensor<VElementOutput> vNewDecayUbTensor_pong;
    AscendC::LocalTensor<half> vNewDecayUbHalfTensor_pong;

    AscendC::LocalTensor<uint8_t> shareBuffer_;

    };
}

#endif