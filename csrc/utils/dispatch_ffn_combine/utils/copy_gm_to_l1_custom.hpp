#ifndef COPY_GM_TO_L1_CUSTOM_HPP
#define COPY_GM_TO_L1_CUSTOM_HPP

namespace Catlass::Gemm::Tile {
    /// Partial specialization for nZ in and nZ out.
    template <
        class ArchTag,
        class Element
    >
    struct CopyGmToL1<ArchTag, Gemm::GemmType<Element, layout::VectorLayout>> {
        using LayoutDst = layout::VectorLayout;
        using LayoutSrc = layout::VectorLayout;

        static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);   // int64, 32/8=4

        // Mehtods  

        CATLASS_DEVICE
        CopyGmToL1() {};

        CATLASS_DEVICE
        void operator()(
            AscendC::LocalTensor<Element> const &dstTensor,
            AscendC::GlobalTensor<Element> const &srcTensor,
            LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
        {
            uint32_t blockCount = 1;
            uint32_t blockLen = CeilDiv<ELE_NUM_PER_C0>(layoutSrc.shape(0));

            AscendC::DataCopyParams repeatParams;

            repeatParams.blockCount = blockCount;
            repeatParams.blockLen = blockLen;
            repeatParams.srcStride = 0;
            repeatParams.dstStride = 0;
            AscendC::DataCopy(dstTensor, srcTensor, repeatParams);
        }
    };
}
#endif // COPY_GM_TO_L1_CUSTOM_HPP