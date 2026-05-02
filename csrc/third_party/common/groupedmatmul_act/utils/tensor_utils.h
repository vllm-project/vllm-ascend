/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tensor_utils.h
 * \brief
 */

#ifndef UTILS_TENSOR_UTILS_H
#define UTILS_TENSOR_UTILS_H
#include "integral_constant.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "tuple_utils.h"
#include "matmul_layout_type.h"

namespace AscendC {
template <typename Tp>
struct is_global_tensor : public Std::false_type {};

template <typename Tp>
struct is_global_tensor<GlobalTensor<Tp>> : public Std::true_type {};

template <typename Tp>
constexpr bool is_global_tensor_v = is_global_tensor<Tp>::value;

template <typename Tp>
struct is_local_tensor : public Std::false_type {};

template <typename Tp>
struct is_local_tensor<LocalTensor<Tp>> : public Std::true_type {};

template <typename Tp>
constexpr bool is_local_tensor_v = is_local_tensor<Tp>::value;

template <typename Tp>
struct tensor_trait {
    static_assert(Std::always_false_v<Tp>, "Unsupported tensor type");
};

template <typename Tp>
struct tensor_trait<GlobalTensor<Tp>> {
    using trait_type = Tp;
};

template <typename Tp>
struct tensor_trait<LocalTensor<Tp>> {
    using trait_type = Tp;
};
} // namespace AscendC

namespace Act {
namespace Gemm {

template <class Layout_, class AGlobalTensor_, class ATensorTrait_, class AType_>
__aicore__ inline void InitGlobalTensorA(AGlobalTensor_& aGlobal, GM_ADDR aGmAddr, bool transA, int64_t m, int64_t k)
{
    Layout_ aLayout;
    if (!transA) {
        aLayout = AscendC::MakeLayout(AscendC::MakeShape(m, k), AscendC::MakeStride(k, static_cast<int64_t>(1)));
    } else {
        aLayout = AscendC::MakeLayout(AscendC::MakeShape(k, m), AscendC::MakeStride(m, static_cast<int64_t>(1)));
    }
    aGlobal.SetTensorTrait(ATensorTrait_(aLayout));
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
    AscendC::GlobalTensor<AType_> aTmp;
    aTmp.SetGlobalBuffer(reinterpret_cast<__gm__ AType_*>(aGmAddr), m * k);
    aGlobal.address_ = aTmp.address_;
#else
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ AType_*>(aGmAddr), m * k);
#endif
}

template <class Layout_, class BGlobalTensor_, class BTensorTrait_, class BType_>
__aicore__ inline void InitGlobalTensorB(BGlobalTensor_& bGlobal, GM_ADDR bGmAddr, bool transB, int64_t n, int64_t k)
{
    Layout_ bLayout;
    if (!transB) {
        bLayout = AscendC::MakeLayout(AscendC::MakeShape(k, n), AscendC::MakeStride(n, static_cast<int64_t>(1)));
    } else {
        bLayout = AscendC::MakeLayout(AscendC::MakeShape(n, k), AscendC::MakeStride(k, static_cast<int64_t>(1)));
    }
    bGlobal.SetTensorTrait(BTensorTrait_(bLayout));
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
    AscendC::GlobalTensor<BType_> bTmp;
    bTmp.SetGlobalBuffer(reinterpret_cast<__gm__ BType_*>(bGmAddr), k * n);
    bGlobal.address_ = bTmp.address_;
#else
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ BType_*>(bGmAddr), k * n);
#endif
}

template <class Layout_, class CGlobalTensor_, class CTensorTrait_, class CType_>
__aicore__ inline void InitGlobalTensorC(CGlobalTensor_& cGlobal, GM_ADDR cGmAddr, int64_t m, int64_t n)
{
    Layout_ cLayout = AscendC::MakeLayout(AscendC::MakeShape(m, n), AscendC::MakeStride(n, static_cast<int64_t>(1)));
    cGlobal.SetTensorTrait(CTensorTrait_(cLayout));
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
    AscendC::GlobalTensor<CType_> cTmp;
    cTmp.SetGlobalBuffer(reinterpret_cast<__gm__ CType_*>(cGmAddr), m * n);
    cGlobal.address_ = cTmp.address_;
#else
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ CType_*>(cGmAddr), m * n);
#endif
}

template <class Layout_, class CTensorTrait_, class CType_, class BlockShape_>
__aicore__ inline AscendC::GlobalTensor<CTensorTrait_> GetWorkSpaceGlobal(BlockShape_ blockShape, GM_ADDR cGmAddr)
{
    int64_t blockShapeM = Get<0>(blockShape);
    int64_t blockShapeN = Get<1>(blockShape);
    Layout_ cLayout = AscendC::MakeLayout(AscendC::MakeShape(blockShapeM, blockShapeN),
                        AscendC::MakeStride(blockShapeN, static_cast<int64_t>(1)));
    CTensorTrait_ cTensorTrait = AscendC::MakeTensorTrait<CType_, AscendC::TPosition::GM>(cLayout);
    AscendC::GlobalTensor<CTensorTrait_> workspaceGlobal;
    workspaceGlobal.SetTensorTrait(cTensorTrait);
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
    AscendC::GlobalTensor<CType_> cTmp;
    cTmp.SetGlobalBuffer(reinterpret_cast<__gm__ CType_*>(cGmAddr));
    workspaceGlobal.address_ = cTmp.address_;
#else
    workspaceGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ CType_*>(cGmAddr));
#endif
    return workspaceGlobal;
}

template <typename T>
__aicore__ inline __gm__ T* GetTensorAddr(uint64_t index, GM_ADDR tensorPtr)
{
    AscendC::ListTensorDesc listTensorDesc(reinterpret_cast<__gm__ void*>(tensorPtr));
    return listTensorDesc.GetDataPtr<T>(index);
}

template <class T, AscendC::TPosition Pos, class Layout, class Coord, class Shape>
__aicore__ inline constexpr auto GetTile(AscendC::GlobalTensor<AscendC::TensorTrait<T, Pos, Layout>> const &tensor,
                                         Coord const &coord, Shape const &shape)
{
    auto layout = tensor.GetTensorTrait().GetLayout();
    auto offset = layout(coord);
    typename AscendC::Std::remove_cvref_t<decltype(tensor)> newTensor;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 3102)
    if constexpr (AscendC::IsSameTypeV<T, fp4x2_e2m1_t> || AscendC::IsSameTypeV<T, fp4x2_e1m2_t>) {
        newTensor.address_ = (__gm__ T *)((__gm__ uint8_t *)tensor.address_ + (offset >> 1));
    } else {
        newTensor.address_ = (__gm__ T *)((__gm__ uint8_t *)tensor.address_ + offset * sizeof(T));
    }
#else
    newTensor.address_ = (__gm__ T *)((__gm__ uint8_t *)tensor.address_ + offset * sizeof(T));
#endif
    newTensor.SetTensorTrait(
        AscendC::MakeTensorTrait<T, Pos>(AscendC::MakeLayout(shape, tensor.GetTensorTrait().GetLayout().GetStride())));
    return newTensor;
}

template <class T, class Layout, AscendC::TPosition Pos = AscendC::TPosition::GM>
__aicore__ inline constexpr auto MakeTensor(__gm__ T *addr, Layout const &layout)
{
    using TensorTraitType = AscendC::TensorTrait<T, Pos, Layout>;
    using TensorType =
        AscendC::Std::conditional_t<Pos == AscendC::TPosition::GM, AscendC::GlobalTensor<TensorTraitType>,
                                    AscendC::LocalTensor<TensorTraitType>>;
    TensorType tensor;
    tensor.address_ = addr;
    tensor.SetTensorTrait(AscendC::MakeTensorTrait<T, Pos>(layout));
    return tensor;
}

/**
 * @brief Convert the input tensor to a standard tensor without layout information
 * @param [in] DataType: data type of input tensor
 * @param [in] Tensor: type of input tensor
 * @param [in] tensor: input tensor to be converted
 * @return Return the converted standard tensor
 */
template <class DataType, class Tensor>
__aicore__ inline auto ConvertToTensorWithoutLayout(const Tensor& tensor)
{
    typename AscendC::Std::conditional_t<AscendC::is_global_tensor_v<Tensor>, AscendC::GlobalTensor<DataType>,
                                         AscendC::LocalTensor<DataType>>
        normalTensor;
    if constexpr (AscendC::is_global_tensor_v<Tensor>) {
        normalTensor.address_ = tensor.address_;
    } else {
        normalTensor.SetAddr(tensor.address_);
    }
    return normalTensor;
}

/**
 * @brief Check if the given matrix type is ND format
 * @param [in] MatmulType: matrix type
 * @return Return true if the matrix type is ND format, otherwise false
 */

template<size_t N, typename T>
struct ShapeTrait {
    using value = typename AscendC::Std::tuple_element<N, typename AscendC::GetTypeFromTrait<T>::ShapeType>::type;
};

template <class MatmulType>
__aicore__ inline constexpr bool IsNDOrAlign()
{
    if constexpr(!AscendC::is_tensorTrait_v<MatmulType>) {
        if constexpr(Act::Gemm::IsMatmulLayoutTypeV<MatmulType>) {
            return (Act::Gemm::ToMatmulTypeT<MatmulType>::format == CubeFormat::ND ||
            Act::Gemm::ToMatmulTypeT<MatmulType>::format == CubeFormat::ND_ALIGN);
        } else {
            return (MatmulType::format == CubeFormat::ND || MatmulType::format == CubeFormat::ND_ALIGN);
        }
    } else {
        if constexpr (!AscendC::Std::is_tuple_v<typename ShapeTrait<0, MatmulType>::value>) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Check if the given matrix type is NZ format
 * @param [in] MatmulType: matrix type
 * @return Return true if the matrix type is NZ format, otherwise false
 */

template<typename T>
struct typeTrait {
    using type = T;
};

template<size_t N0, size_t N1, typename T>
struct GetTuple {
    using ShapeTraitType = typename ShapeTrait<N0, T>::value;
    using type = typename AscendC::Std::tuple_element<N1, ShapeTraitType>::type;
};

template <class MatmulType>
__aicore__ inline constexpr bool IsNz()
{
    if constexpr(!AscendC::is_tensorTrait_v<MatmulType>) {
        if constexpr(Act::Gemm::IsMatmulLayoutTypeV<MatmulType>) {
            return (Act::Gemm::ToMatmulTypeT<MatmulType>::format == CubeFormat::NZ);
        } else {
            return (MatmulType::format == CubeFormat::NZ);
        }
    } else {
        if constexpr (AscendC::Std::is_tuple_v<typename ShapeTrait<0, MatmulType>::value>) {
            if constexpr(AscendC::Std::is_same_v<typename GetTuple<0, 0, MatmulType>::type, typeTrait<_16>::type>) {
                return true;
            }
        }
    }
    return false;
}

/**
 * @brief Check if physical position is L1
 * @param [in] pos: TPosition
 * @return Return true if pos is A1 or B1
 */
template <AscendC::TPosition pos>
__aicore__ inline constexpr bool PosIsL1()
{
    return pos == AscendC::TPosition::A1 || pos == AscendC::TPosition::B1;
}

/**
 * @brief Check if physical position is CO1
 * @param [in] pos: TPosition
 * @return Return true if pos is CO1
 */
template <AscendC::TPosition pos>
__aicore__ inline constexpr bool PosIsCO1()
{
    return pos == AscendC::TPosition::CO1;
}

/**
 * @brief Check if physical position is GM
 * @param [in] pos: TPosition
 * @return Return true if pos is GM
 */
template <AscendC::TPosition pos>
__aicore__ inline constexpr bool PosIsGM()
{
    return pos == AscendC::TPosition::GM;
}

/**
 * @brief Check if physical position is UB
 * @param [in] pos: TPosition
 * @return Return true if pos is UB
 */
template <AscendC::TPosition pos>
__aicore__ inline constexpr bool PosIsUB()
{
    return AscendC::PhyPosIsUB(pos);
}

/**
 * @brief Check if physical position is L0C
 * @param [in] pos: TPosition
 * @return Return true if pos is L0C
 */
template <AscendC::TPosition pos>
__aicore__ inline constexpr bool PosIsL0C()
{
    return AscendC::PhyPosIsL0C(pos);
}

} // namespace Gemm
} // namespace Act
#endif