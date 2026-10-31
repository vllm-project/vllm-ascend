/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LOCAL_TENSOR_BUFFER_HPP
#define LOCAL_TENSOR_BUFFER_HPP

#include "../../attn_infra/base_defs.hpp"
#include "../../attn_infra/arch/arch.hpp"

namespace NpuArch::Arch
{

struct LocalTensorBufferBase {
public:
    template <class Element = half>
    __aicore__ inline
    AscendC::LocalTensor<Element> GetBufferByByte(const uint32_t offset) const
    {
        return tensor[offset].template ReinterpretCast<Element>();
    }

protected:
    __aicore__ inline
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
template <>
struct LocalTensorBuffer<Arch::AtlasA5, AscendC::TPosition::A1> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::A1;

    __aicore__ inline
    LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(AscendC::TPosition::A1, 0, Arch::AtlasA5::L1_SIZE);
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::A2
template <>
struct LocalTensorBuffer<Arch::AtlasA5, AscendC::TPosition::A2> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::A2;

    __aicore__ inline
    LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(AscendC::TPosition::A2, 0, Arch::AtlasA5::L0A_SIZE);
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::B1
template <>
struct LocalTensorBuffer<Arch::AtlasA5, AscendC::TPosition::B1> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::B1;

    __aicore__ inline
    LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(AscendC::TPosition::B1, 0, Arch::AtlasA5::L1_SIZE);
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::B2
template <>
struct LocalTensorBuffer<Arch::AtlasA5, AscendC::TPosition::B2> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::B2;

    __aicore__ inline
    LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(AscendC::TPosition::B2, 0, Arch::AtlasA5::L0B_SIZE);
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::C1
template <>
struct LocalTensorBuffer<Arch::AtlasA5, AscendC::TPosition::C1> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::C1;

    __aicore__ inline
    LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(AscendC::TPosition::C1, 0, Arch::AtlasA5::L1_SIZE);
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::C2
template <>
struct LocalTensorBuffer<Arch::AtlasA5, AscendC::TPosition::C2> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::C2;

    __aicore__ inline
    LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(AscendC::TPosition::C2, 0, Arch::AtlasA5::BIAS_SIZE);
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::CO1
template <>
struct LocalTensorBuffer<Arch::AtlasA5, AscendC::TPosition::CO1> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::CO1;

    __aicore__ inline
    LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(AscendC::TPosition::CO1, 0, Arch::AtlasA5::L0C_SIZE);
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::C2PIPE2GM
template <>
struct LocalTensorBuffer<Arch::AtlasA5, AscendC::TPosition::C2PIPE2GM> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::C2PIPE2GM;

    __aicore__ inline
    LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(AscendC::TPosition::C2PIPE2GM, 0, Arch::AtlasA5::FIXBUF_SIZE);
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::VECIN
template <>
struct LocalTensorBuffer<Arch::AtlasA5, AscendC::TPosition::VECIN> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::VECIN;

    __aicore__ inline
    LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(AscendC::TPosition::VECIN, 0, Arch::AtlasA5::UB_SIZE);
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::VECOUT
template <>
struct LocalTensorBuffer<Arch::AtlasA5, AscendC::TPosition::VECOUT> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::VECOUT;

    __aicore__ inline
    LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(AscendC::TPosition::VECOUT, 0, Arch::AtlasA5::UB_SIZE);
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::VECCALC
template <>
struct LocalTensorBuffer<Arch::AtlasA5, AscendC::TPosition::VECCALC> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::VECCALC;

    __aicore__ inline
    LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(AscendC::TPosition::VECCALC, 0, Arch::AtlasA5::UB_SIZE);
    }
};


/// Partial specialization for TPosition::A1
template <>
struct LocalTensorBuffer<Arch::AtlasA2, AscendC::TPosition::A1> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::A1;

    __aicore__ inline
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::A1> tbufA1;
        GetTPipePtr()->InitBuffer(tbufA1, Arch::AtlasA2::L1_SIZE);
        tensor = tbufA1.Get<uint8_t>();
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::A2
template <>
struct LocalTensorBuffer<Arch::AtlasA2, AscendC::TPosition::A2> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::A2;

    __aicore__ inline
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::A2> tbufA2;
        GetTPipePtr()->InitBuffer(tbufA2, Arch::AtlasA2::L0A_SIZE);
        tensor = tbufA2.Get<uint8_t>();
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::B1
template <>
struct LocalTensorBuffer<Arch::AtlasA2, AscendC::TPosition::B1> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::B1;

    __aicore__ inline
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::B1> tbufB1;
        GetTPipePtr()->InitBuffer(tbufB1, Arch::AtlasA2::L1_SIZE);
        tensor = tbufB1.Get<uint8_t>();
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for AtlasA2, TPosition::B2
template <>
struct LocalTensorBuffer<Arch::AtlasA2, AscendC::TPosition::B2> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::B2;

    __aicore__ inline
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::B2> tbufB2;
        GetTPipePtr()->InitBuffer(tbufB2, Arch::AtlasA2::L0B_SIZE);
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

    __aicore__ inline
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::C1> tbufC1;
        GetTPipePtr()->InitBuffer(tbufC1, Arch::AtlasA2::L1_SIZE);
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

    __aicore__ inline
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::C2> tbufC2;
        GetTPipePtr()->InitBuffer(tbufC2, Arch::AtlasA2::BIAS_SIZE);
        tensor = tbufC2.Get<uint8_t>();
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::CO1
template <>
struct LocalTensorBuffer<Arch::AtlasA2, AscendC::TPosition::CO1> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::CO1;

    __aicore__ inline
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::CO1> tbufCO1;
        GetTPipePtr()->InitBuffer(tbufCO1, Arch::AtlasA2::L0C_SIZE);
        tensor = tbufCO1.Get<uint8_t>();
    }
};

/// Partial specialization for TPosition::VECCALC
template <>
struct LocalTensorBuffer<Arch::AtlasA2, AscendC::TPosition::VECCALC> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::VECCALC;

    __aicore__ inline
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::VECCALC> tbufVECCALC;
        GetTPipePtr()->InitBuffer(tbufVECCALC, Arch::AtlasA2::UB_SIZE);
        tensor = tbufVECCALC.Get<uint8_t>();
    }
};
}  // namespace NpuArch::Arch

#endif  // INCLUDE_ARCH_MEMORY_H
