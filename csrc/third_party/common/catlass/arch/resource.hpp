/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INCLUDE_CATLASS_ARCH_RESOURCE_HPP
#define INCLUDE_CATLASS_ARCH_RESOURCE_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/local_tensor_buffer.hpp"

namespace Catlass::Arch {

template<class ArchTag>
struct Resource {
public:
    AscendC::TPipe pipe;

    LocalTensorBuffer<ArchTag, AscendC::TPosition::A1> l1Buf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::A2> l0ABuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::B2> l0BBuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::C2> btBuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::CO1> l0CBuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::VECCALC> ubBuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::C2PIPE2GM> fpBuf;

    CATLASS_DEVICE
    Resource()
    {
        // The initialization of AscendC::Tpipe will insert some synchronization interfaces,
        // which may conflict with the usage by users. Therefore, the "destroy" interface is used for releasing.
        pipe.Destroy();
    }
};

} // namespace Catlass::Arch

#endif // INCLUDE_CATLASS_ARCH_RESOURCE_HPP