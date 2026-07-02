#ifndef ASCENDC_ZB_MOE_DISTRIBUTE_DISPATCH_TILING_H
#define ASCENDC_ZB_MOE_DISTRIBUTE_DISPATCH_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct ZbMoeDistributeDispatchInfo {
    uint32_t epWorldSize;          // epWorldSize
    uint32_t tpWorldSize;          // tpWorldSize
    uint32_t epRankId;             // epRankId
    uint32_t tpRankId;             // tpRankId
    uint32_t expertShardType;      // expert type
    uint32_t sharedExpertNum;      // shared expert number
    uint32_t sharedExpertRankNum;  // shared expert rank number
    uint32_t moeExpertNum;         // moe expert number
    uint32_t quantMode;            // quant mode
    uint32_t globalBs;             // globalBs = BS * worldSize
    uint32_t bs;                   // bs
    uint32_t k;                    // k
    uint32_t h;                    // h
    uint32_t aivNum;               // aivNum
    bool isTokenMask;              // input active mask 1dims or not
    bool isExpertMask;             // input active mask 2dims or not
    bool hasElasticInfo;           // has elasticinfo or not
    bool reserved3;                // reserved
    uint64_t totalUbSize;          // epWorldSize
    uint64_t totalWinSize;
    uint32_t expertTokenNumsType;  // expert token nums type, support 0: cumsum mode, 1: count mode
    int32_t zeroComputeExpertNum;  // sum of zero、copy and const expert nums
    uint32_t cumSumUBMinValue;     // Minimum value for CumSum remainder（in UB）
    uint64_t shmemptr;             // shmem ptr for MetaInfo
};

struct ZbMoeDistributeDispatchTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling1;
    Mc2CcTiling mc2CcTiling2;
    ZbMoeDistributeDispatchInfo moeDistributeDispatchInfo;
};

#endif
