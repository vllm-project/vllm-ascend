#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDH_UPDATE_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDH_UPDATE_HPP
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
    class HInputType_,
    class HUpdateInputType_,
    class FinalStateType_
>
class BlockEpilogue <
    EpilogueAtlasA2GDNFwdHUpdate,
    HOutputType_,
    GInputType_,
    HInputType_,
    HUpdateInputType_,
    FinalStateType_
> {
public:
    // Type aliases
    using DispatchPolicy = EpilogueAtlasA2GDNFwdHUpdate;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using HElementOutput = typename HOutputType_::Element;
    using GElementInput = typename GInputType_::Element;
    using HElementInput = typename HInputType_::Element;
    using HUpdateElementInput = typename HUpdateInputType_::Element;
    using FinalStateElement = typename FinalStateType_::Element;


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

//        constexpr uint32_t BASE = 0;
//        constexpr uint32_t H_TENSOR_OFFSET = BASE ; // 16*128*2B
//        constexpr uint32_t HUPDATE_UB_TENSOR_OFFSET = H_TENSOR_OFFSET + 32 * UB_LINE_SIZE;
//        constexpr uint32_t SRC_UB_TENSOR_OFFSET = HUPDATE_UB_TENSOR_OFFSET + 32 * UB_LINE_SIZE; // 16*128*2B
//        constexpr uint32_t FLOAT_UB_TENSOR_OFFSET = SRC_UB_TENSOR_OFFSET + 32 * UB_LINE_SIZE;
//        constexpr uint32_t G_TENSOR_OFFSET = FLOAT_UB_TENSOR_OFFSET + 64 * UB_LINE_SIZE;
//
//        hUbTensor = resource.ubBuf.template GetBufferByByte<half>(H_TENSOR_OFFSET);
//        hUpdateUbTensor = resource.ubBuf.template GetBufferByByte<half>(HUPDATE_UB_TENSOR_OFFSET);
//        srcUbTensor = resource.ubBuf.template GetBufferByByte<HElementOutput>(SRC_UB_TENSOR_OFFSET);
//        floatUbTensor = resource.ubBuf.template GetBufferByByte<float>(FLOAT_UB_TENSOR_OFFSET);
//      glastUbTensor = resource.ubBuf.template GetBufferByByte<half>(G_TENSOR_OFFSET);

        constexpr uint32_t FLOAT_UB_TENSOR_SIZE = 64 * UB_LINE_SIZE;
        constexpr uint32_t HALF_UB_TENSOR_SIZE = 32 * UB_LINE_SIZE;

        constexpr uint32_t BASE = 0;
        constexpr uint32_t FLOAT_UB_TENSOR_OFFSET = BASE;
        constexpr uint32_t HALF_UB_TENSOR_OFFSET = FLOAT_UB_TENSOR_OFFSET + FLOAT_UB_TENSOR_SIZE;

        constexpr uint32_t H_TENSOR_OFFSET = HALF_UB_TENSOR_OFFSET + HALF_UB_TENSOR_SIZE;
        constexpr uint32_t HUPDATE_UB_TENSOR_OFFSET = H_TENSOR_OFFSET + HALF_UB_TENSOR_SIZE;
        constexpr uint32_t HOUTPUT_UB_TENSOR_OFFSET = HUPDATE_UB_TENSOR_OFFSET + HALF_UB_TENSOR_SIZE;
        constexpr uint32_t G_TENSOR_OFFSET = HOUTPUT_UB_TENSOR_OFFSET + HALF_UB_TENSOR_SIZE;

        floatUbTensor = resource.ubBuf.template GetBufferByByte<float>(FLOAT_UB_TENSOR_OFFSET);
        halfUbTensor = resource.ubBuf.template GetBufferByByte<half>(HALF_UB_TENSOR_OFFSET);

        hUbTensor = resource.ubBuf.template GetBufferByByte<HElementInput>(H_TENSOR_OFFSET);
        hUbHalfTensor = resource.ubBuf.template GetBufferByByte<half>(H_TENSOR_OFFSET);
        hUpdateUbHalfTensor = resource.ubBuf.template GetBufferByByte<half>(HUPDATE_UB_TENSOR_OFFSET);

        hOutputUbFloatTensor = resource.ubBuf.template GetBufferByByte<float>(HUPDATE_UB_TENSOR_OFFSET);
        hOutputUbTensor = resource.ubBuf.template GetBufferByByte<HElementOutput>(HOUTPUT_UB_TENSOR_OFFSET);
        hOutputUbHalfTensor = resource.ubBuf.template GetBufferByByte<half>(HOUTPUT_UB_TENSOR_OFFSET);

        glastUbTensor = resource.ubBuf.template GetBufferByByte<half>(G_TENSOR_OFFSET);

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
        AscendC::GlobalTensor<HElementOutput> hOutput,
        AscendC::GlobalTensor<FinalStateElement> finalState,
        AscendC::GlobalTensor<float> gInput,
        AscendC::GlobalTensor<HElementInput> hInput,
        AscendC::GlobalTensor<HUpdateElementInput> hUpdateInput,
        uint32_t chunkSize,
        uint32_t kHeadDim,
        uint32_t vHeadDim,
        Arch::CrossCoreFlag cube2Done,
        bool isFinalState
    )
    {
        uint32_t mActual = kHeadDim;
        uint32_t nActual = vHeadDim;
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t mActualPerSubBlock = CeilDiv(mActual, subBlockNum);
        uint32_t mActualThisSubBlock = (subBlockIdx == 0) ? mActualPerSubBlock : (mActual - mActualPerSubBlock);
        uint32_t mOffset = subBlockIdx * mActualPerSubBlock;
        uint32_t nOffset = 0;
        int64_t offsetH = mOffset * nActual + nOffset;

        AscendC::ResetMask();

        AscendC::GlobalTensor<HElementOutput> hOutputThisSubBlock = hOutput[offsetH];
        AscendC::GlobalTensor<FinalStateElement> finalStateThisSubBlock = finalState[offsetH];
        AscendC::GlobalTensor<float> gInputThisSubBlock = gInput;
        AscendC::GlobalTensor<HElementInput> hInputThisSubBlock = hInput[offsetH];
        AscendC::GlobalTensor<HUpdateElementInput> hUpdateInputThisSubBlock = hUpdateInput[offsetH];

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        if constexpr(!std::is_same<HElementInput, half>::value) {
            AscendC::DataCopy(hUbTensor, hInputThisSubBlock, mActualThisSubBlock * nActual);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Cast(floatUbTensor, hUbTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock * nActual);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(hUbHalfTensor, floatUbTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock * nActual);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            AscendC::DataCopy(hUbHalfTensor, hInputThisSubBlock, mActualThisSubBlock * nActual);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        }
        glastUbTensor.SetValue(0, (half)gInputThisSubBlock.GetValue(chunkSize-1));
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::Exp(glastUbTensor, glastUbTensor, 1);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        half muls = glastUbTensor.GetValue(0);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::Muls(hUbHalfTensor, hUbHalfTensor, muls, mActualThisSubBlock * nActual);

        Arch::CrossCoreWaitFlag(cube2Done);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::DataCopy(hUpdateUbHalfTensor, hUpdateInputThisSubBlock, mActualThisSubBlock * nActual);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::Add<half>(hUbHalfTensor, hUbHalfTensor, hUpdateUbHalfTensor, mActualThisSubBlock * nActual);

        if (isFinalState) {
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(hOutputUbFloatTensor, hUbHalfTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock * nActual);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopy(finalStateThisSubBlock, hOutputUbFloatTensor, mActualThisSubBlock * nActual);
        } else {
            if constexpr(!std::is_same<HElementOutput, half>::value) {
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(floatUbTensor, hUbHalfTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock * nActual);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(hOutputUbTensor, floatUbTensor, AscendC::RoundMode::CAST_RINT, mActualThisSubBlock * nActual);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::DataCopy(hOutputThisSubBlock, hOutputUbTensor, mActualThisSubBlock * nActual);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::DataCopy(hOutputThisSubBlock, hUbHalfTensor, mActualThisSubBlock * nActual);
            }
        }
    }

private:
    AscendC::LocalTensor<float> floatUbTensor;
    AscendC::LocalTensor<half> halfUbTensor;

    AscendC::LocalTensor<HElementInput> hUbTensor;
    AscendC::LocalTensor<half> hUbHalfTensor;
    AscendC::LocalTensor<half> hUpdateUbHalfTensor;

    AscendC::LocalTensor<float> hOutputUbFloatTensor;
    AscendC::LocalTensor<HElementOutput> hOutputUbTensor;
    AscendC::LocalTensor<half> hOutputUbHalfTensor;

    AscendC::LocalTensor<half> glastUbTensor;

    };
}

#endif