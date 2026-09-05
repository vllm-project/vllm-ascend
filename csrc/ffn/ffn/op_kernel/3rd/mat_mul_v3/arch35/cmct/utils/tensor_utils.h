/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

#pragma once
#include "common_utils.h"
#include "integral_constant.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "tuple_utils.h"

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

namespace Cmct {
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
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
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
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
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
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
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
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
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
    // depend on kernel_operator_list_tensor_intf.h
    AscendC::ListTensorDesc listTensorDesc(reinterpret_cast<__gm__ void*>(tensorPtr));
    return listTensorDesc.GetDataPtr<T>(index);
}

template <class T, AscendC::TPosition Pos, class Layout, class Coord, class Shape>
__aicore__ inline constexpr auto GetTile(AscendC::GlobalTensor<AscendC::TensorTrait<T, Pos, Layout>> const& tensor,
                                         Coord const& coord, Shape const& shape)
{
    auto layout = tensor.GetTensorTrait().GetLayout();
    auto offset = layout(coord);
    typename AscendC::Std::remove_cvref_t<decltype(tensor)> newTensor;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 3102)
    if constexpr (AscendC::IsSameType<T, fp4x2_e2m1_t>::value || AscendC::IsSameType<T, fp4x2_e1m2_t>::value ||
                  AscendC::IsSameType<T, AscendC::int4b_t>::value) {
        newTensor.address_ = (__gm__ T*)((__gm__ uint8_t*)tensor.address_ + (offset >> 1));
    } else {
        newTensor.address_ = (__gm__ T*)((__gm__ uint8_t*)tensor.address_ + offset * sizeof(T));
    }
#else
    newTensor.address_ = (__gm__ T*)((__gm__ uint8_t*)tensor.address_ + offset * sizeof(T));
#endif
    newTensor.SetTensorTrait(
        AscendC::MakeTensorTrait<T, Pos>(AscendC::MakeLayout(shape, tensor.GetTensorTrait().GetLayout().GetStride())));
    return newTensor;
}

// 不设置layout，需要手动更新
template <typename CAST_T, typename T>
__aicore__ inline auto ReinerpretCast(const AscendC::LocalTensor<T>& tensorIn)
{
    static_assert(AscendC::is_tensorTrait_v<CAST_T> && AscendC::is_tensorTrait_v<T>,
                  "only support TensorTrait type Tensor!");
    if constexpr (AscendC::Std::is_same_v<CAST_T, T>) {
        return tensorIn;
    }
    static_assert(all_static_v<decltype(CAST_T{}.GetStride())> && all_static_v<decltype(T{}.GetStride())>);
    AscendC::LocalTensor<CAST_T> tensorOut;
    tensorOut.address_.logicPos = static_cast<uint8_t>(tensorIn.GetPosition());
    tensorOut.address_.bufferHandle = tensorIn.GetBufferHandle();
    tensorOut.address_.bufferAddr = tensorIn.address_.bufferAddr;
    return tensorOut;
}

template <class T, class Layout, AscendC::TPosition Pos = AscendC::TPosition::GM>
__aicore__ inline constexpr auto MakeTensor(__gm__ T* addr, Layout const& layout)
{
    using TensorTraitType = AscendC::TensorTrait<T, Pos, Layout>;
    using TensorType = AscendC::Std::conditional_t<
        Pos == AscendC::TPosition::GM, AscendC::GlobalTensor<TensorTraitType>, AscendC::LocalTensor<TensorTraitType>>;
    TensorType tensor;
    tensor.address_ = addr;
    tensor.SetTensorTrait(AscendC::MakeTensorTrait<T, Pos>(layout));
    return tensor;
}

template <typename T, AscendC::TPosition Pos, typename Layout, typename Stride>
__aicore__ inline auto SetStride(AscendC::LocalTensor<AscendC::TensorTrait<T, Pos, Layout>>& tensor,
                                 Stride const& stride)
{
    tensor.SetTensorTrait(AscendC::MakeTensorTrait<T, Pos>(AscendC::MakeLayout(tensor.GetShape(), stride)));
    return tensor;
}

template <typename TensorTrait>
__aicore__ inline auto CreateTensor(uint32_t addr)
{
    static_assert(AscendC::is_tensorTrait_v<TensorTrait>, "only support TensorTrait type Tensor!");
    AscendC::LocalTensor<TensorTrait> tensor;
    tensor.address_.dataLen = 32;
    tensor.address_.bufferAddr = addr;
    tensor.address_.logicPos = static_cast<uint8_t>(TensorTrait::tPos);
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
template <class MatmulType>
__aicore__ inline constexpr bool IsNDOrAlign()
{
    return (MatmulType::format == CubeFormat::ND || MatmulType::format == CubeFormat::ND_ALIGN);
}

/**
 * @brief Check if the given matrix type is NZ format
 * @param [in] MatmulType: matrix type
 * @return Return true if the matrix type is NZ format, otherwise false
 */
template <class MatmulType>
__aicore__ inline constexpr bool IsNz()
{
    return MatmulType::format == CubeFormat::NZ;
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
} // namespace Cmct
