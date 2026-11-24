// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

struct QuantMlaShape {
    uint32_t requests;
    uint32_t tokens;
    uint32_t heads;
    uint32_t tableWidth;
    uint32_t cores;
    uint64_t pageStride;
    float scale;
};

extern "C" void launch_quant_flash_mla_nz(void* stream, uint8_t* q, uint8_t* kv, uint8_t* qScale, uint8_t* kvScale,
    uint8_t* table, uint8_t* lengths, uint8_t* cu, uint8_t* used, uint8_t* mask, uint8_t* schedule, uint8_t* out,
    uint8_t* workspace, QuantMlaShape shape);
extern "C" void launch_quant_flash_mla_cache_write(void* stream, uint32_t blocks, uint8_t* latent, uint8_t* rope,
    uint8_t* latentPages, uint8_t* ropePages, void* slots, bool wideSlots, uint32_t count, uint64_t pages,
    uint64_t latentStride, uint64_t ropeStride, uint64_t pageStride);
