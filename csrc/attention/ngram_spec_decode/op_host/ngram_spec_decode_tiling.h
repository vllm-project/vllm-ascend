#ifndef NGRAM_SPEC_DECODE_TILING_H
#define NGRAM_SPEC_DECODE_TILING_H

#include "kernel_tiling/kernel_tiling.h"

struct NgramSpecDecodeInfo {
    uint32_t batchSize;
    uint32_t maxSeqLen;
    uint32_t maxNewTokens;
    uint32_t vocabSize;
    uint32_t minN;
    uint32_t maxN;
    uint32_t k;
    uint32_t formerNum;
    uint32_t rowsPerCore;
    uint32_t tailRows;
    uint32_t blockRows;
};

struct NgramSpecDecodeTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling1;
    NgramSpecDecodeInfo ngramInfo;
};

#endif  // NGRAM_SPEC_DECODE_TILING_H
