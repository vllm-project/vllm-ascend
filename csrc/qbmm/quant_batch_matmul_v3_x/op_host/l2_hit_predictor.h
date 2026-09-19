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
 * \file l2_hit_predictor.h
 * \brief L2 hit-rate predictor used by the tiling cost model. Auto-generated; do not edit by hand.
 * \author Feodor Pisnitchenko
 */
#ifndef QBMV3_L2_HIT_PREDICTOR_H
#define QBMV3_L2_HIT_PREDICTOR_H

#include <cmath>
#include <cstdint>

namespace optiling {

// Returns L2 hit % in [0, 100].
static inline double PredictL2HitPct(
    uint32_t baseM, uint32_t baseN, uint32_t baseK,
    uint32_t B, uint32_t M, uint32_t K, uint32_t N,
    uint32_t mtilesPerCore, uint32_t cbsPerCore,
    uint64_t actPerCore, uint64_t wPerCore, uint64_t perCoreWS) {
    if ((double)M <= 136.0000) {
        if ((double)baseN <= 24.0000) {
            if ((double)B <= 2.5000) {
                return 9.5000;
            } else {
                if ((double)B <= 10.0000) {
                    return 35.9957;
                } else {
                    return 14.5039;
                }
            }
        } else {
            if (std::log2((double)actPerCore) <= 20.4383) {
                if ((double)baseK <= 192.0000) {
                    if ((double)mtilesPerCore <= 1.5000) {
                        return 2.6455;
                    } else {
                        return 0.7147;
                    }
                } else {
                    if ((double)mtilesPerCore <= 1.5000) {
                        return 5.9211;
                    } else {
                        return 9.4043;
                    }
                }
            } else {
                return 24.6970;
            }
        }
    } else {
        if ((double)baseM <= 72.0000) {
            if ((double)K <= 2624.0000) {
                if ((double)baseM <= 48.0000) {
                    if (std::log2((double)perCoreWS) <= 22.2189) {
                        return 82.6461;
                    } else {
                        return 84.9615;
                    }
                } else {
                    if (std::log2((double)actPerCore) <= 18.1699) {
                        return 66.9246;
                    } else {
                        return 33.2412;
                    }
                }
            } else {
                if (std::log2((double)actPerCore) <= 24.4771) {
                    if ((double)baseM <= 48.0000) {
                        return 86.1109;
                    } else {
                        return 83.8506;
                    }
                } else {
                    if ((double)cbsPerCore <= 12.0000) {
                        return 77.1189;
                    } else {
                        return 75.4291;
                    }
                }
            }
        } else {
            if ((double)B <= 10.0000) {
                if ((double)M <= 1280.0000) {
                    if ((double)B <= 2.5000) {
                        return 69.3583;
                    } else {
                        return 51.9279;
                    }
                } else {
                    if ((double)mtilesPerCore <= 9.5000) {
                        return 81.4611;
                    } else {
                        return 74.3779;
                    }
                }
            } else {
                if (std::log2((double)actPerCore) <= 23.1610) {
                    if (std::log2((double)wPerCore) <= 21.2083) {
                        return 54.3114;
                    } else {
                        return 7.3394;
                    }
                } else {
                    if (std::log2((double)actPerCore) <= 24.0112) {
                        return 81.2807;
                    } else {
                        return 69.3785;
                    }
                }
            }
        }
    }
    return 50.0;  // unreachable fallback
}

}  // namespace optiling

#endif  // QBMV3_L2_HIT_PREDICTOR_H
