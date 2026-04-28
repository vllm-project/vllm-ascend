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
 * \file tbuf_pool_l0_default.h
 * \brief
 */
#ifndef MATMUL_TILE_LOAD_DATA_TBUF_POOL_L0_DEFAULT_H
#define MATMUL_TILE_LOAD_DATA_TBUF_POOL_L0_DEFAULT_H

namespace Act {
namespace Gemm {
namespace Tile {
/**
 * @class TBufPoolL0
 * @brief Class for managing L0 buffer pool operations
 *
 * This class provides methods for initializing, allocating, and managing L0 buffers
 */
class TBufPoolL0 {
public:
    __aicore__ inline TBufPoolL0(){};
    __aicore__ inline ~TBufPoolL0(){};

    /**
     * @brief Initialize the L0 buffer pool
     * @param [in] isL0Db: boolean flag to indicate if L0 double buffering is used
     */
    __aicore__ inline void Init(bool isL0Db = true)
    {
        useL0PingPong_ = static_cast<uint16_t>(isL0Db);
        GetTPipePtr()->InitBuffer(l0aBuf_, L0A_SIZE);
        GetTPipePtr()->InitBuffer(l0bBuf_, L0B_SIZE);
    }

    /**
     * @brief Set the double buffering flag for L0
     * @param [in] isL0Db: boolean flag to indicate if L0 double buffering is used
     */
    __aicore__ inline void SetDBFlag(bool isL0Db = true)
    {
        useL0PingPong_ = static_cast<uint16_t>(isL0Db);
    }

    /**
     * @brief Allocate an L0 buffer
     * @return Reference to the current TBufPoolL0 instance
     */
    __aicore__ inline TBufPoolL0& Allocate()
    {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0PingPongFlag_);
        return *this;
    }

    /**
     * @brief Get a L0 buffer
     * @param [in] Pos: the position of the buffer
     * @param [in] T: the type of the buffer
     * @param [in] subIdx: the sub-index of the buffer
     * @return LocalTensor of type T
     */
    template <AscendC::TPosition Pos, typename T>
    __aicore__ inline AscendC::LocalTensor<T> GetBuffer(uint8_t subIdx = 0)
    {
        AscendC::LocalTensor<typename T::LiteType> tempTensor;
        if constexpr (Pos == AscendC::TPosition::A2) {
            tempTensor = l0aBuf_.Get<typename T::LiteType>();
            if (l0PingPongFlag_ != 0) {
                tempTensor = tempTensor[L0A_SIZE / DOUBLE_BUFFER_COUNT / sizeof(typename T::LiteType)];
            }
        } else {
            tempTensor = l0bBuf_.Get<typename T::LiteType>();
            if (l0PingPongFlag_ != 0) {
                tempTensor = tempTensor[L0B_SIZE / DOUBLE_BUFFER_COUNT / sizeof(typename T::LiteType)];
            }
        }
        AscendC::LocalTensor<T> retTensor;
        retTensor.SetAddr(tempTensor.address_);
        return retTensor;
    }

    /**
     * @brief Check if a buffer is hit
     * @param [in] Pos: the position of the buffer
     * @param [in] pos: the position to check
     * @return Boolean indicating if the buffer is hit
     */
    template <AscendC::TPosition Pos>
    __aicore__ inline bool Hit(uint32_t pos = 0)
    {
        return false;
    }

    /**
     * @brief Reset the cache
     */
    __aicore__ inline void ResetCache() {}

    /**
     * @brief Enqueue a buffer
     */
    __aicore__ inline void EnQue()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPongFlag_);
    }

    /**
     * @brief Dequeue a buffer
     */
    __aicore__ inline void DeQue()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPongFlag_);
    }

    /**
     * @brief Free a buffer
     */
    __aicore__ inline void Free()
    {
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0PingPongFlag_);
        l0PingPongFlag_ = useL0PingPong_ - l0PingPongFlag_;
    }

private:
    AscendC::TBuf<AscendC::TPosition::A2> l0aBuf_;
    AscendC::TBuf<AscendC::TPosition::B2> l0bBuf_;
    uint16_t l0PingPongFlag_{0};
    uint16_t useL0PingPong_{1};
};

} // namespace Tile
} // namespace Gemm
} // namespace Act
#endif // MATMUL_TILE_LOAD_DATA_TBUF_POOL_L0_DEFAULT_H
