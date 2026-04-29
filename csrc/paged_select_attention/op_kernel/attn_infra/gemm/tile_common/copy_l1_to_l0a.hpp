/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_TILE_COPY_L1_TO_L0A_HPP
#define GEMM_TILE_COPY_L1_TO_L0A_HPP

#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/arch/arch.hpp"
#include "../../../attn_infra/layout/layout.hpp"
#include "../../../attn_infra/gemm/gemm_type.hpp"
#include "../../../attn_infra/gemm/tile_common/tile_copy_tla.hpp"


namespace NpuArch::Gemm::Tile {

template <
    class ArchTag,
    class L1Type,
    class L0Type = void
>
struct CopyL1ToL0A {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to l0, can not find the specialization.");
};

////////////////////////////////
/// new add gemm
template<class ArchTag, class Element>
struct CopyL1ToL0A<ArchTag, NpuArch::Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>, NpuArch::Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::A2>>{
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 =  BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    __aicore__ inline
    CopyL1ToL0A(){}

    __aicore__ inline
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(3));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRACTAL - 1;
        loadDataParams.ifTranspose = false;
        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < layoutDst.shape(1); i++) {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

template<class ArchTag, class Element>
struct CopyL1ToL0A<ArchTag, NpuArch::Gemm::GemmType<Element, layout::nN, AscendC::TPosition::A1>, NpuArch::Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::A2>>{
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::nN;

    static constexpr uint32_t ELE_NUM_PER_C0 =  BYTE_PER_C0 / sizeof(Element);

    __aicore__ inline
    CopyL1ToL0A(){}

    __aicore__ inline
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes =
            static_cast<uint16_t>(NpuArch::Detail::Alignment::CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(1)));
        loadDataParams.srcStride =
            static_cast<uint16_t>(NpuArch::Detail::Alignment::CeilDiv<ELE_NUM_PER_C0>(layoutSrc.orgShape(0)));;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = true;
        loadDataParams.addrMode = 0;
        for(uint32_t i = 0; i < NpuArch::Detail::Alignment::CeilDiv<ELE_NUM_PER_C0>(layoutSrc.orgShape(0)); i++){
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

template<class ArchTag>
struct CopyL1ToL0A<ArchTag, NpuArch::Gemm::GemmType<float, layout::nN, AscendC::TPosition::A1>, NpuArch::Gemm::GemmType<float, layout::zZ, AscendC::TPosition::A2>>{
    using Element = float;
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::nN;

    static constexpr uint32_t ELE_NUM_PER_C0 =  BYTE_PER_C0 / sizeof(Element);

    __aicore__ inline
    CopyL1ToL0A(){}

    __aicore__ inline
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        AscendC::LoadData2dTransposeParams loadDataParams;
        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes =
            static_cast<uint16_t>(NpuArch::Detail::Alignment::CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(1)));
        loadDataParams.srcStride =
            static_cast<uint16_t>(NpuArch::Detail::Alignment::CeilDiv<C0_NUM_PER_FRACTAL>(layoutSrc.orgShape(0)));
        loadDataParams.dstGap = 1;
        loadDataParams.dstFracGap = 0;
        for(uint32_t i = 0; i < NpuArch::Detail::Alignment::CeilDiv<C0_NUM_PER_FRACTAL>(layoutSrc.orgShape(0)); i++){
            AscendC::LoadDataWithTranspose(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1) * 2], loadDataParams);
        }
    }
};

template<class ArchTag>
struct CopyL1ToL0A<ArchTag, NpuArch::Gemm::GemmType<int8_t, layout::nZ, AscendC::TPosition::A1>, NpuArch::Gemm::GemmType<int8_t, layout::zZ, AscendC::TPosition::A2>>{
    using Element = int8_t;
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 =  BYTE_PER_C0 / sizeof(Element);

    __aicore__ inline
    CopyL1ToL0A(){}

    __aicore__ inline
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        AscendC::LoadData2dTransposeParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes =
            static_cast<uint16_t>(NpuArch::Detail::Alignment::CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.dstFracGap = NpuArch::Detail::Alignment::CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)) - 1;

        for (uint32_t i = 0; i < NpuArch::Detail::Alignment::CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(0)); i++) {
            AscendC::LoadDataWithTranspose(dstTensor[i * layoutDst.stride(1) * 2],
                                           srcTensor[i * layoutSrc.stride(1)],
                                           loadDataParams);
        }
    }
};
//////////////////////////////////////////

/// Partial specialization for zN in and zZ out.
template <class ArchTag, class Element>
struct CopyL1ToL0A<ArchTag, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>> {
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    // Methods

    __aicore__ inline
    CopyL1ToL0A() {};

    __aicore__ inline
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(3));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRACTAL - 1;
        loadDataParams.ifTranspose = false;
        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < layoutDst.shape(1); i++) {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

/// Partial specialization for float, zN in and zZ out.
template <class ArchTag>
struct CopyL1ToL0A<ArchTag, Gemm::GemmType<float, layout::zN, AscendC::TPosition::A1>> {
    using Element = float;
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    // Methods

    __aicore__ inline
    CopyL1ToL0A() {};

    __aicore__ inline
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        constexpr uint8_t PAD_LIST[4] = {0, 0, 0, 0};
        uint16_t l1M = layoutSrc.shape(0) * layoutSrc.shape(1);
        uint16_t l1K = layoutSrc.shape(2) * layoutSrc.shape(3);
        uint16_t l0M = layoutDst.shape(0) * layoutDst.shape(1);
        uint16_t l0K = layoutDst.shape(2) * layoutDst.shape(3);
        AscendC::SetFmatrix(1, l1M, PAD_LIST, AscendC::FmatrixMode::FMATRIX_LEFT);
        static constexpr AscendC::IsResetLoad3dConfig config = {false, false};
        AscendC::LoadData3DParamsV2<Element> loadDataParams;
        loadDataParams.kExtension = l0K;
        loadDataParams.mExtension = l0M;
        loadDataParams.channelSize = l1K;

        AscendC::LoadData<Element, config>(dstTensor, srcTensor, loadDataParams);
    }
};

template <class ArchTag, class Element>
struct CopyL1ToL0A<ArchTag, Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>> {
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    __aicore__ inline
    CopyL1ToL0A() {};

    __aicore__ inline
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes =
            static_cast<uint16_t>(NpuArch::Detail::Alignment::CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRACTAL - 1;
        loadDataParams.ifTranspose = true;
        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < NpuArch::Detail::Alignment::CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(0)); i++) {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

/// Partial specialization for int8_t, nZ in and zZ out. (Transpose A)
template <class ArchTag>
struct CopyL1ToL0A<ArchTag, Gemm::GemmType<int8_t, layout::nZ, AscendC::TPosition::A1>> {
    using Element = int8_t;
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    // Methods

    __aicore__ inline
    CopyL1ToL0A() {};

    __aicore__ inline
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2dTransposeParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes =
            static_cast<uint16_t>(NpuArch::Detail::Alignment::CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.dstFracGap = NpuArch::Detail::Alignment::CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)) - 1;

        for (uint32_t i = 0; i < NpuArch::Detail::Alignment::CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(0)); i++) {
            AscendC::LoadDataWithTranspose(dstTensor[i * layoutDst.stride(1) * 2],
                                           srcTensor[i * layoutSrc.stride(1)],
                                           loadDataParams);
        }
    }
};

/// Partial specialization for float, nZ in and zZ out. (Transpose A)
template <class ArchTag>
struct CopyL1ToL0A<ArchTag, Gemm::GemmType<float, layout::nZ, AscendC::TPosition::A1>> {
    using Element = float;
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    // Methods

    __aicore__ inline
    CopyL1ToL0A() {};

    __aicore__ inline
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        constexpr uint8_t PAD_LIST[4] = {0, 0, 0, 0};
        uint16_t l1M = layoutSrc.shape(0) * layoutSrc.shape(1);
        uint16_t l1K = layoutSrc.shape(2) * layoutSrc.shape(3);
        uint16_t l0M = layoutDst.shape(0) * layoutDst.shape(1);
        uint16_t l0K = layoutDst.shape(2) * layoutDst.shape(3);
        // K, M need to be 16 aligned for f32
        uint16_t l1MAlign = NpuArch::Detail::Alignment::RoundUp<C0_NUM_PER_FRACTAL>(l1M);
        uint16_t l1KAlign = NpuArch::Detail::Alignment::RoundUp<C0_NUM_PER_FRACTAL>(l1K);
        uint16_t l0MAlign = NpuArch::Detail::Alignment::RoundUp<C0_NUM_PER_FRACTAL>(l0M);
        uint16_t l0KAlign = NpuArch::Detail::Alignment::RoundUp<C0_NUM_PER_FRACTAL>(l0K);
        AscendC::SetFmatrix(1, l1KAlign, PAD_LIST, AscendC::FmatrixMode::FMATRIX_LEFT);
        static constexpr AscendC::IsResetLoad3dConfig config = {false, false};
        AscendC::LoadData3DParamsV2<Element> loadDataParams;
        loadDataParams.kExtension = l0MAlign;
        loadDataParams.mExtension = l0KAlign;
        loadDataParams.enTranspose = true;
        loadDataParams.channelSize = l1MAlign;

        AscendC::LoadData<Element, config>(dstTensor, srcTensor, loadDataParams);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace NpuArch::Gemm::Tile

#endif // GEMM_TILE_COPY_L1_TO_L0A_HPP