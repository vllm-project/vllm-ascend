#ifndef COPY_L0C_TO_GM_CUSTOM_HPP
#define COPY_L0C_TO_GM_CUSTOM_HPP

namespace Catlass::Gemm::Tile {
    template <
        class ElementAccumulator_,
        class ElementDst_,
        bool ReluEnable_
    >
    struct CopyL0CToGm<Catlass::Arch::AtlasA2,
                    ElementAccumulator_,
                    Gemm::GemmType<ElementDst_, layout::RowMajor>,
                    ScaleGranularity::PER_CHANNEL,
                    ReluEnable_>
    {
        using ArchTag = Catlass::Arch::AtlasA2;
        using ElementDst = ElementDst_;
        using ElementSrc = ElementAccumulator_;
        using LayoutSrc = Catlass::layout::zN;
        using LayoutDst = Catlass::layout::RowMajor;
        static constexpr auto quantPre = CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst,
            ScaleGranularity::PER_CHANNEL>::VALUE;
        static constexpr auto reluEn = ReluEnable_;

        CATLASS_DEVICE
        void operator()(AscendC::GlobalTensor<ElementDst> const &dst, AscendC::LocalTensor<ElementSrc> const &src, AscendC::LocalTensor<uint64_t> cbufWorkspace,
            LayoutDst const &dstLayout, LayoutSrc const &srcLayout, uint8_t unitFlag = 0)
        {
            AscendC::FixpipeParamsV220 intriParams;

            // Fixpipe layout information
            intriParams.nSize = dstLayout.shape(1);
            intriParams.mSize = dstLayout.shape(0);
            intriParams.srcStride = srcLayout.stride(3) / srcLayout.stride(0);
            intriParams.dstStride = dstLayout.stride(0);

            // Fixpipe auxiliary arguments
            intriParams.quantPre = quantPre;
            intriParams.reluEn = reluEn;
            intriParams.unitFlag = unitFlag;

            // Call AscendC Fixpipe
            AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_ROW_MAJOR>(dst, src, cbufWorkspace, intriParams);
        }
    };
}
#endif // COPY_L0C_TO_GM_CUSTOM_HPP