/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_HELPER_HPP
#define CATLASS_GEMM_HELPER_HPP

#include "catlass/catlass.hpp"
#include "catlass/numeric_size.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "tla/layout.hpp"

namespace Catlass::Gemm::helper {

template<class Element, class Layout>
struct L1AlignHelper {
    static_assert(DEPENDENT_FALSE<Element>, "Unsupported align helper, can not find the specialization.");
};

template<class Element>
struct L1AlignHelper<Element, layout::RowMajor> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t M_ALIGNED = C0_NUM_PER_FRACTAL;
    static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t N_ALIGNED = ELE_NUM_PER_C0;
};

template<class Element>
struct L1AlignHelper<Element, layout::ColumnMajor> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t M_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t N_ALIGNED = C0_NUM_PER_FRACTAL;
};

template<class Element>
struct L1AlignHelper<Element, layout::PaddingRowMajor> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t M_ALIGNED = C0_NUM_PER_FRACTAL;
    static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t N_ALIGNED = ELE_NUM_PER_C0;
};

template<class Element>
struct L1AlignHelper<Element, layout::PaddingColumnMajor> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t M_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t N_ALIGNED = C0_NUM_PER_FRACTAL;
};

template<class Element>
struct L1AlignHelper<Element, layout::zN> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t M_ALIGNED = C0_NUM_PER_FRACTAL;
    static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t N_ALIGNED = ELE_NUM_PER_C0;
};

template<class Element>
struct L1AlignHelper<Element, layout::nZ> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t M_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t N_ALIGNED = C0_NUM_PER_FRACTAL;
};

template<class Element>
struct L1AlignHelper<Element, layout::NC1HWC0> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t HOWO_ALIGNED = C0_NUM_PER_FRACTAL;
};

template<class Element>
struct L1AlignHelper<Element, layout::CI1KHKWCOCI0> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t COUT_ALIGNED = C0_NUM_PER_FRACTAL;
};


template<class ElementA, class ElementB>
struct ElementAccumulatorSelector {
    static_assert(DEPENDENT_FALSE<ElementA>,
        "Unsupported element accumulator selector, can not find the specialization.");
};

template<>
struct ElementAccumulatorSelector<half, half> {
    using ElementAccumulator = float;
};

template<>
struct ElementAccumulatorSelector<float, float> {
    using ElementAccumulator = float;
};

template<>
struct ElementAccumulatorSelector<int8_t, int8_t> {
    using ElementAccumulator = int32_t;
};

template<>
struct ElementAccumulatorSelector<bfloat16_t, bfloat16_t> {
    using ElementAccumulator = float;
};

template<>
struct ElementAccumulatorSelector<AscendC::int4b_t, AscendC::int4b_t> {
    using ElementAccumulator = int32_t;
};

template<class GmAType>
struct L1ATypeSelector {
    static_assert(DEPENDENT_FALSE<GmAType>,
        "Unsupported layout selector, can not find the specialization.");
};

template<class Element>
struct L1ATypeSelector<Gemm::GemmType<Element, layout::RowMajor>> {
    using L1AType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
};

template<class Element>
struct L1ATypeSelector<Gemm::GemmType<Element, layout::zN>> {
    using L1AType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
};

template<class Element>
struct L1ATypeSelector<Gemm::GemmType<Element, layout::PaddingRowMajor>> {
    using L1AType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
};

template<class Element>
struct L1ATypeSelector<Gemm::GemmType<Element, layout::ColumnMajor>> {
    using L1AType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>;
};

template<class Element>
struct L1ATypeSelector<Gemm::GemmType<Element, layout::nZ>> {
    using L1AType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>;
};

template<class Element>
struct L1ATypeSelector<Gemm::GemmType<Element, layout::PaddingColumnMajor>> {
    using L1AType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>;
};

template<class Element>
struct L1ATypeSelector<Gemm::GemmType<Element, layout::NDC1HWC0>> {
    using L1AType = Gemm::GemmType<Element, layout::NDC1HWC0, AscendC::TPosition::A1>;
};

template<class Element>
struct L1ATypeSelector<Gemm::GemmType<Element, layout::NC1HWC0>> {
    using L1AType = Gemm::GemmType<Element, layout::NC1HWC0, AscendC::TPosition::A1>;
};

template<class GmBType>
struct L1BTypeSelector {
    static_assert(DEPENDENT_FALSE<GmBType>,
        "Unsupported layout selector, can not find the specialization.");
};

template<class Element>
struct L1BTypeSelector<Gemm::GemmType<Element, layout::RowMajor>> {
    using L1BType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
};

template<class Element>
struct L1BTypeSelector<Gemm::GemmType<Element, layout::zN>> {
    using L1BType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
};

template<class Element>
struct L1BTypeSelector<Gemm::GemmType<Element, layout::PaddingRowMajor>> {
    using L1BType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
};

template<class Element>
struct L1BTypeSelector<Gemm::GemmType<Element, layout::ColumnMajor>> {
    using L1BType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>;
};

template<class Element>
struct L1BTypeSelector<Gemm::GemmType<Element, layout::nZ>> {
    using L1BType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>;
};

template<class Element>
struct L1BTypeSelector<Gemm::GemmType<Element, layout::PaddingColumnMajor>> {
    using L1BType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>;
};

template<class Element>
struct L1BTypeSelector<Gemm::GemmType<Element, layout::KDC1KHKWN1N0C0>> {
    using L1BType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>;
};

template<class Element>
struct L1BTypeSelector<Gemm::GemmType<Element, layout::CI1KHKWCOCI0>> {
    using L1BType = Gemm::GemmType<Element, layout::CI1KHKWCOCI0, AscendC::TPosition::A1>;
};

template<class GmBiasType, class ElementAccumulator>
struct L1BiasTypeSelector {
    static_assert(DEPENDENT_FALSE<GmBiasType>,
        "Unsupported layout selector, can not find the specialization.");
};

template<class ElementAccumulator>
struct L1BiasTypeSelector<void, ElementAccumulator> {
    using GMBiasType = void;
    using L1BiasType = void;
    using L0BiasType = void;
};

template<class Element, class ElementAccumulator>
struct L1BiasTypeSelector<Gemm::GemmType<Element, layout::VectorLayout>, ElementAccumulator> {
    using GMBiasType = Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::GM>;
    using L1BiasType = Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::A1>;
    using L0BiasType = Gemm::GemmType<ElementAccumulator, layout::VectorLayout, AscendC::TPosition::C2>;
};

template<class Element, class Layout, class Enable = void>
struct L1AlignHelperTla {
    static_assert(DEPENDENT_FALSE<Element>, "Unsupported align helper tla, can not find the specialization.");
};

template<class Element, class Layout>
struct L1AlignHelperTla<Element, Layout, std::enable_if_t<tla::detail::isRowMajor<Layout>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t M_ALIGNED = C0_NUM_PER_FRACTAL;
    static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t N_ALIGNED = ELE_NUM_PER_C0;
};

template<class Element, class Layout>
struct L1AlignHelperTla<Element, Layout, std::enable_if_t<tla::detail::isColumnMajor<Layout>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t M_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t N_ALIGNED = C0_NUM_PER_FRACTAL;
};

///////////////////////////////////////
// new add
template<>
struct ElementAccumulatorSelector<int32_t, int32_t> {
    using ElementAccumulator = int32_t;
};

template<class GmAType, class GmBType>
struct L1AndL0TypeSelectorGemm{
    static_assert(DEPENDENT_FALSE<GmAType>,
        "Unsupported layout selector, can not find the specialization.");
    static_assert(DEPENDENT_FALSE<GmBType>,
        "Unsupported layout selector, can not find the specialization.");
};

template<class Element>
struct L1AndL0TypeSelectorGemm<Gemm::GemmType<Element, layout::RowMajor>, Gemm::GemmType<Element, layout::RowMajor>>{
    using L1AType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
    using L1BType = Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::B1>;
    using L0AType = Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::A2>;
    using L0BType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>;
};

template<>
struct L1AndL0TypeSelectorGemm<Gemm::GemmType<int8_t, layout::RowMajor>, Gemm::GemmType<int8_t, layout::RowMajor>>{
    using L1AType = Gemm::GemmType<int8_t, layout::zN, AscendC::TPosition::A1>;
    using L1BType = Gemm::GemmType<int8_t, layout::zN, AscendC::TPosition::B1>;
    using L0AType = Gemm::GemmType<int8_t, layout::zZ, AscendC::TPosition::A2>;
    using L0BType = Gemm::GemmType<int8_t, layout::nZ, AscendC::TPosition::B2>;
};

template<class Element>
struct L1AndL0TypeSelectorGemm<Gemm::GemmType<Element, layout::ColumnMajor>, Gemm::GemmType<Element, layout::ColumnMajor>>{
    using L1AType = Gemm::GemmType<Element, layout::nN, AscendC::TPosition::A1>;
    using L1BType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B1>;
    using L0AType = Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::A2>;
    using L0BType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>;
};

template<>
struct L1AndL0TypeSelectorGemm<Gemm::GemmType<int8_t, layout::ColumnMajor>, Gemm::GemmType<int8_t, layout::ColumnMajor>>{
    using L1AType = Gemm::GemmType<int8_t, layout::nZ, AscendC::TPosition::A1>;
    using L1BType = Gemm::GemmType<int8_t, layout::nZ, AscendC::TPosition::B1>;
    using L0AType = Gemm::GemmType<int8_t, layout::zZ, AscendC::TPosition::A2>;
    using L0BType = Gemm::GemmType<int8_t, layout::nZ, AscendC::TPosition::B2>;
};

template<class Element>
struct L1AndL0TypeSelectorGemm<Gemm::GemmType<Element, layout::RowMajor>, Gemm::GemmType<Element, layout::ColumnMajor>>{
    using L1AType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
    using L1BType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B1>;
    using L0AType = Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::A2>;
    using L0BType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>;
};

template<class Element>
struct L1AndL0TypeSelectorGemm<Gemm::GemmType<Element, layout::ColumnMajor>, Gemm::GemmType<Element, layout::RowMajor>>{
    using L1AType = Gemm::GemmType<Element, layout::nN, AscendC::TPosition::A1>;
    using L1BType = Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::B1>;
    using L0AType = Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::A2>;
    using L0BType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>;
};

template<>
struct L1AndL0TypeSelectorGemm<Gemm::GemmType<int8_t, layout::ColumnMajor>, Gemm::GemmType<int8_t, layout::RowMajor>>{
    using L1AType = Gemm::GemmType<int8_t, layout::nZ, AscendC::TPosition::A1>;
    using L1BType = Gemm::GemmType<int8_t, layout::zN, AscendC::TPosition::B1>;
    using L0AType = Gemm::GemmType<int8_t, layout::zZ, AscendC::TPosition::A2>;
    using L0BType = Gemm::GemmType<int8_t, layout::nZ, AscendC::TPosition::B2>;
};

///////////////////////////////////////
// Check for whether the tile shape is aligned (Default: 32Bytes).
template<class L1TileShape, class L0TileShape, class ElementA, class ElementB, uint32_t _aligned=32>
struct TileShapeAlignChecker {
    static constexpr uint32_t _ALIGN = _aligned*8;

    static_assert(L1TileShape::M * SizeOfBits<ElementA>::value % _ALIGN == 0, "L1TileShape::M is not aligned.");
    static_assert(L0TileShape::M * SizeOfBits<ElementA>::value % _ALIGN == 0, "L0TileShape::M is not aligned.");
    static_assert(L1TileShape::K * SizeOfBits<ElementA>::value % _ALIGN == 0, "L1TileShape::K is not aligned.");
    static_assert(L1TileShape::K * SizeOfBits<ElementB>::value % _ALIGN == 0, "L1TileShape::K is not aligned.");
    static_assert(L0TileShape::K * SizeOfBits<ElementA>::value % _ALIGN == 0, "L0TileShape::K is not aligned.");
    static_assert(L0TileShape::K * SizeOfBits<ElementB>::value % _ALIGN == 0, "L0TileShape::K is not aligned.");
    static_assert(L1TileShape::N * SizeOfBits<ElementB>::value % _ALIGN == 0, "L1TileShape::N is not aligned.");
    static_assert(L0TileShape::N * SizeOfBits<ElementB>::value % _ALIGN == 0, "L0TileShape::N is not aligned.");
};

///////////////////////////////////////
} // namespace Catlass::Gemm::helper

#endif // CATLASS_GEMM_HELPER_HPP
