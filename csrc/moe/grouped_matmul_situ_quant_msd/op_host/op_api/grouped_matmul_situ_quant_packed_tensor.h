#pragma once

#include "opdev/op_executor.h"

namespace situ_quant {

constexpr int64_t INT4_PER_INT32 = 8;

// Descriptor offsets and strides are measured in dtype elements. Preserve the
// address of an expert view when changing each INT32 carrier into eight INT4s.
inline void UnpackInt32ToInt4(const aclTensor *tensor)
{
    if (tensor == nullptr || tensor->GetDataType() != op::DataType::DT_INT32) {
        return;
    }
    auto *tensorS4 = const_cast<aclTensor *>(tensor);
    op::Shape viewShape = tensorS4->GetViewShape();
    const int64_t viewDimNum = static_cast<int64_t>(viewShape.GetDimNum());
    if (viewDimNum == 0) {
        return;
    }
    viewShape[viewDimNum - 1] = viewShape.GetDim(viewDimNum - 1) * INT4_PER_INT32;
    auto strides = tensorS4->GetViewStrides();
    for (int64_t dim = 0; dim + 1 < viewDimNum; ++dim) {
        strides[dim] *= INT4_PER_INT32;
    }
    tensorS4->SetViewStrides(strides);
    tensorS4->SetViewOffset(tensorS4->GetViewOffset() * INT4_PER_INT32);
    tensorS4->SetStorageOffset(tensorS4->GetStorageOffset() * INT4_PER_INT32);
    tensorS4->SetViewShape(viewShape);
    tensorS4->SetDataType(op::DataType::DT_INT4);
    op::Shape storageShape = tensorS4->GetStorageShape();
    const int64_t storageDimNum = static_cast<int64_t>(storageShape.GetDimNum());
    if (storageDimNum > 0) {
        storageShape[storageDimNum - 1] = storageShape.GetDim(storageDimNum - 1) * INT4_PER_INT32;
        tensorS4->SetStorageShape(storageShape);
    }
}

// A8W4 MSD 的 weightScale 是打包的 UINT64，torch 侧没有无符号 64 位整型，按 int64 传入。
inline void CastInt64ScaleToUint64(const aclTensor *tensor)
{
    if (tensor == nullptr || tensor->GetDataType() != op::DataType::DT_INT64) {
        return;
    }
    const_cast<aclTensor *>(tensor)->SetDataType(op::DataType::DT_UINT64);
}

} // namespace situ_quant
