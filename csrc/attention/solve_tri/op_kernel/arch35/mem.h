/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */

#ifndef MEM_H
#define MEM_H

struct HardwareInfo {
    static uint32_t const l1Size = 512 * 1024;
    static uint32_t const l0ASize = 64 * 1024;
    static uint32_t const l0BSize = 64 * 1024;
    static uint32_t const l0CSize = 256 * 1024;
    static uint32_t const l2Size = 128 * 1024 * 1024;
    static uint32_t const biasSize = 4 * 1024;
    static uint32_t const fixBufSize = 6 * 1024;
    static uint32_t const ubSize = 248 * 1024;
};

enum class BufferType : uint32_t {
    ASCEND_UB,
    ASCEND_CB,
    ASCEND_L0A,
    ASCEND_L0B,
    ASCEND_L0C,
    ASCEND_MAX
};

struct OnChipBuffer {
    template <typename T>
    using Tensor = AscendC::LocalTensor<T>;

public:
    __aicore__ inline OnChipBuffer()
    {
        constexpr uint32_t bufferSize[(uint32_t)BufferType::ASCEND_MAX] = {
            HardwareInfo::ubSize, HardwareInfo::l1Size, HardwareInfo::l0ASize,
            HardwareInfo::l0BSize, HardwareInfo::l0CSize};

        buffer_[(uint32_t)BufferType::ASCEND_UB] =
            Tensor<uint8_t>(AscendC::TPosition::VECIN, 0, bufferSize[(uint32_t)BufferType::ASCEND_UB]);
        buffer_[(uint32_t)BufferType::ASCEND_CB] =
            Tensor<uint8_t>(AscendC::TPosition::A1, 0, bufferSize[(uint32_t)BufferType::ASCEND_CB]);
        buffer_[(uint32_t)BufferType::ASCEND_L0A] =
            Tensor<uint8_t>(AscendC::TPosition::A2, 0, bufferSize[(uint32_t)BufferType::ASCEND_L0A]);
        buffer_[(uint32_t)BufferType::ASCEND_L0B] =
            Tensor<uint8_t>(AscendC::TPosition::B2, 0, bufferSize[(uint32_t)BufferType::ASCEND_L0B]);
        buffer_[(uint32_t)BufferType::ASCEND_L0C] =
            Tensor<uint8_t>(AscendC::TPosition::CO1, 0, bufferSize[(uint32_t)BufferType::ASCEND_L0C]);
    }

    template <BufferType bufferType, typename Dtype = half>
    __aicore__ inline Tensor<Dtype> GetBuffer(const uint32_t offset) const
    {
        return buffer_[(uint32_t)bufferType][offset].template ReinterpretCast<Dtype>();
    }

private:
    AscendC::LocalTensor<uint8_t> buffer_[(uint32_t)BufferType::ASCEND_MAX];
};

#endif  // MEM_H
