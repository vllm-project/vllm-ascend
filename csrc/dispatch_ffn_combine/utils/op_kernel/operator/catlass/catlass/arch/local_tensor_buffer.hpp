/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INCLUDE_CATLASS_ARCH_MEMORY_H
#define INCLUDE_CATLASS_ARCH_MEMORY_H

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"

namespace Catlass::Arch {

struct LocalTensorBufferBase {
public:
    template <class Element = half>
    CATLASS_DEVICE
    AscendC::LocalTensor<Element> GetBufferByByte(const uint32_t offset) const
    {
        return tensor[offset].template ReinterpretCast<Element>();
    }

protected:
    CATLASS_DEVICE
    LocalTensorBufferBase() = default;

    AscendC::LocalTensor<uint8_t> tensor;
};

template <
    class ArchTag,
    AscendC::TPosition Position
>
struct LocalTensorBuffer {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported local tensor buffer, can not find the specialization.");
};

/// Partial specialization for TPosition::A1
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::A1> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::A1;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::A1> tbufA1;
        GetTPipePtr()->InitBuffer(tbufA1, ArchTag::L1_SIZE);
        tensor = tbufA1.Get<uint8_t>();
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::A2
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::A2> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::A2;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::A2> tbufA2;
        GetTPipePtr()->InitBuffer(tbufA2, ArchTag::L0A_SIZE);
        tensor = tbufA2.Get<uint8_t>();
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::B1
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::B1> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::B1;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::B1> tbufB1;
        GetTPipePtr()->InitBuffer(tbufB1, ArchTag::L1_SIZE);
        tensor = tbufB1.Get<uint8_t>();
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for AtlasA2, TPosition::B2
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::B2> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::B2;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::B2> tbufB2;
        GetTPipePtr()->InitBuffer(tbufB2, ArchTag::L0B_SIZE);
        tensor = tbufB2.Get<uint8_t>();
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for AtlasA2, TPosition::C1
template <>
struct LocalTensorBuffer<Arch::AtlasA2, AscendC::TPosition::C1> : LocalTensorBufferBase {
public:
    using ArchTag = Arch::AtlasA2;
    static constexpr AscendC::TPosition Position = AscendC::TPosition::C1;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::C1> tbufC1;
        GetTPipePtr()->InitBuffer(tbufC1, ArchTag::L1_SIZE);
        tensor = tbufC1.Get<uint8_t>();
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for AtlasA2, TPosition::C2
template <>
struct LocalTensorBuffer<Arch::AtlasA2, AscendC::TPosition::C2> : LocalTensorBufferBase {
public:
    using ArchTag = Arch::AtlasA2;
    static constexpr AscendC::TPosition Position = AscendC::TPosition::C2;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::C2> tbufC2;
        GetTPipePtr()->InitBuffer(tbufC2, ArchTag::BIAS_SIZE);
        tensor = tbufC2.Get<uint8_t>();
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::CO1
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::CO1> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::CO1;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::CO1> tbufCO1;
        GetTPipePtr()->InitBuffer(tbufCO1, ArchTag::L0C_SIZE);
        tensor = tbufCO1.Get<uint8_t>();
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for AtlasA2, TPosition::C2PIPE2GM
template <>
struct LocalTensorBuffer<Arch::AtlasA2, AscendC::TPosition::C2PIPE2GM> : LocalTensorBufferBase {
public:
    using ArchTag = Arch::AtlasA2;
    static constexpr AscendC::TPosition Position = AscendC::TPosition::C2PIPE2GM;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::C2PIPE2GM> tbufC2PIPE2GM;
        GetTPipePtr()->InitBuffer(tbufC2PIPE2GM, ArchTag::FIXBUF_SIZE);
        tensor = tbufC2PIPE2GM.Get<uint8_t>();
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::VECIN
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::VECIN> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::VECIN;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::VECIN> tbufVECIN;
        GetTPipePtr()->InitBuffer(tbufVECIN, ArchTag::UB_SIZE);
        tensor = tbufVECIN.Get<uint8_t>();
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::VECOUT
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::VECOUT> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::VECOUT;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::VECOUT> tbufVECOUT;
        GetTPipePtr()->InitBuffer(tbufVECOUT, ArchTag::UB_SIZE);
        tensor = tbufVECOUT.Get<uint8_t>();
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::VECCALC
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::VECCALC> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::VECCALC;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::VECCALC> tbufVECCALC;
        GetTPipePtr()->InitBuffer(tbufVECCALC, ArchTag::UB_SIZE);
        tensor = tbufVECCALC.Get<uint8_t>();
    }
};

}  // namespace Catlass::Arch

#endif  // INCLUDE_CATLASS_ARCH_MEMORY_H