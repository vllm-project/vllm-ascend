// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
#ifndef FLASH_MLA_C8_MEMORY_H
#define FLASH_MLA_C8_MEMORY_H
#include "memory_copy_arch35.h"
// 格式转换, 暂时放在这里
template <FlashMlaC8Layout LAYOUT>
__aicore__ inline constexpr ActualSeqLensMode GetQActSeqMode()
{
    if constexpr (LAYOUT == FlashMlaC8Layout::LAYOUT_TND || LAYOUT == FlashMlaC8Layout::LAYOUT_NTD) {
        return ActualSeqLensMode::ACCUM;
    } else {
        return ActualSeqLensMode::BY_BATCH;
    }
}

template <FlashMlaC8Layout LAYOUT, const bool PAGE_ATTENTION>
__aicore__ inline constexpr ActualSeqLensMode GetKvActSeqMode()
{
    if constexpr (PAGE_ATTENTION) {
        return ActualSeqLensMode::BY_BATCH;
    }
    if constexpr (LAYOUT == FlashMlaC8Layout::LAYOUT_TND || LAYOUT == FlashMlaC8Layout::LAYOUT_NTD) {
        return ActualSeqLensMode::ACCUM;
    } else {
        return ActualSeqLensMode::BY_BATCH;
    }
}

template <FlashMlaC8Layout LAYOUT>
__aicore__ inline constexpr GmFormat GetQueryGmFormat()
{
    if constexpr (LAYOUT == FlashMlaC8Layout::LAYOUT_BSH) {
        return GmFormat::BSNGD;
    } else if constexpr (LAYOUT == FlashMlaC8Layout::LAYOUT_SBH) {
        return GmFormat::SBNGD;
    } else if constexpr (LAYOUT == FlashMlaC8Layout::LAYOUT_BNSD) {
        return GmFormat::BNGSD;
    } else if constexpr (LAYOUT == FlashMlaC8Layout::LAYOUT_TND) {
        return GmFormat::TNGD;
    } else {
        return GmFormat::NGTD;
    }
}

template <FlashMlaC8Layout LAYOUT, bool useDn = false, bool isPerTokenHead = false, bool isMlaFullQuant = false>
__aicore__ inline constexpr GmFormat GetQueryScaleGmFormat()
{
    if constexpr (LAYOUT == FlashMlaC8Layout::LAYOUT_BSH || LAYOUT == FlashMlaC8Layout::LAYOUT_BNSD) {
        return GmFormat::BNGSD;
    } else if constexpr (LAYOUT == FlashMlaC8Layout::LAYOUT_TND || LAYOUT == FlashMlaC8Layout::LAYOUT_NTD) {
        if constexpr (isMlaFullQuant) {
            return GmFormat::TNG;
        }
        if constexpr (isPerTokenHead) {
            return GmFormat::NGT;
        }
        if constexpr (!useDn) {
            return GmFormat::NTGD;
        } else {
            return GmFormat::TNGD;
        }
    } else {
        return GmFormat::TNGD;
    }
}

template <FlashMlaC8Layout LAYOUT, uint8_t kvLayoutType = 0, bool isPa = false>
__aicore__ inline constexpr GmFormat GetKeyScaleGmFormat()
{
    if constexpr (kvLayoutType == 0) { // KvLayoutType_NO_PA
        if constexpr (LAYOUT == FlashMlaC8Layout::LAYOUT_BSH) {
            return GmFormat::BSND;
        } else if constexpr (LAYOUT == FlashMlaC8Layout::LAYOUT_SBH) {
            return GmFormat::SBND;
        } else if constexpr (LAYOUT == FlashMlaC8Layout::LAYOUT_BNSD) {
            return GmFormat::BNSD;
        } else if constexpr (LAYOUT == FlashMlaC8Layout::LAYOUT_TND) {
            return GmFormat::TND;
        } else {
            return GmFormat::NTD;
        }
    } else if constexpr (kvLayoutType == 1) { // KvLayoutType_PA_BBH
        return GmFormat::PA_BnBsND;
    } else if constexpr (kvLayoutType == 2) { // KvLayoutType_PA_BNBD
        return GmFormat::PA_BnNBsD;
    } else {
        return GmFormat::PA_NZ_K_SCALE;
    }
}

template <FlashMlaC8Layout LAYOUT, uint8_t kvLayoutType = 0, bool isPa = false>
__aicore__ inline constexpr GmFormat GetValueScaleGmFormat()
{
    if constexpr (kvLayoutType == 0) { // KvLayoutType_NO_PA
        if constexpr (LAYOUT == FlashMlaC8Layout::LAYOUT_BSH) {
            return GmFormat::BSND;
        } else if constexpr (LAYOUT == FlashMlaC8Layout::LAYOUT_SBH) {
            return GmFormat::SBND;
        } else if constexpr (LAYOUT == FlashMlaC8Layout::LAYOUT_BNSD) {
            return GmFormat::BNSD;
        } else if constexpr (LAYOUT == FlashMlaC8Layout::LAYOUT_TND) {
            return GmFormat::TND2;
        } else {
            return GmFormat::NTD;
        }
    } else if constexpr (kvLayoutType == 1) { // KvLayoutType_PA_BBH
        return GmFormat::PA_BnBsND;
    } else if constexpr (kvLayoutType == 2) { // KvLayoutType_PA_BNBD
        return GmFormat::PA_BnNBsD;
    } else {
        return GmFormat::PA_NZ;
    }
}

template <FlashMlaC8Layout LAYOUT>
__aicore__ inline constexpr UbFormat GetOutUbFormat()
{
    static_assert((LAYOUT == FlashMlaC8Layout::LAYOUT_BSH) || (LAYOUT == FlashMlaC8Layout::LAYOUT_BNSD) ||
                      (LAYOUT == FlashMlaC8Layout::LAYOUT_TND) || (LAYOUT == FlashMlaC8Layout::LAYOUT_NTD),
                  "Get OutAttention UB GmFormat fail, LAYOUT is incorrect");
    if constexpr (LAYOUT == FlashMlaC8Layout::LAYOUT_BSH || LAYOUT == FlashMlaC8Layout::LAYOUT_TND) {
        return UbFormat::S1G;
    } else if constexpr (LAYOUT == FlashMlaC8Layout::LAYOUT_BNSD || LAYOUT == FlashMlaC8Layout::LAYOUT_NTD) {
        return UbFormat::GS1;
    }
}

template <FlashMlaC8Layout LAYOUT, uint8_t KvLayoutType = 0, bool isPa = false>
__aicore__ inline constexpr GmFormat GetKVGmFormat()
{
    if constexpr (KvLayoutType == 0) { // KvLayoutType_NO_PA
        if constexpr (LAYOUT == FlashMlaC8Layout::LAYOUT_BSH) {
            return GmFormat::BSND;
        } else if constexpr (LAYOUT == FlashMlaC8Layout::LAYOUT_SBH) {
            return GmFormat::SBND;
        } else if constexpr (LAYOUT == FlashMlaC8Layout::LAYOUT_BNSD) {
            return GmFormat::BNSD;
        } else if constexpr (LAYOUT == FlashMlaC8Layout::LAYOUT_TND) {
            return GmFormat::TND;
        } else {
            return GmFormat::NTD;
        }
    } else if constexpr (KvLayoutType == 1) { // KvLayoutType_PA_BBH
        return GmFormat::PA_BnBsND;
    } else if constexpr (KvLayoutType == 2) { // KvLayoutType_PA_BNBD
        return GmFormat::PA_BnNBsD;
    } else { // KvLayoutType_PA_NZ
        return GmFormat::PA_NZ;
    }
}


#endif
