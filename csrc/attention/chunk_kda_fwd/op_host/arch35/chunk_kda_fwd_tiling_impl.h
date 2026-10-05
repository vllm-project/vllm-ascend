#pragma once

namespace optiling::arch35 {

struct ChunkKdaFwdArch35Options {
    bool computeGateInPrepare = false;
    bool fusePostWu = false;
    bool fusePostWuIntoFwdH = false;
    bool useDenseFwdH = false;
};

inline ChunkKdaFwdArch35Options ConfigureChunkKdaFwdArch35(
    bool isAscend950, bool qIsBf16, bool rawGIsFp32, bool hasALog,
    bool useGateInKernel, bool safeGate, bool isVarLen, int64_t seqlen,
    int64_t vHeads, int64_t chunkSize, int64_t kDim, int64_t vDim,
    bool storeQG, bool storeVNew, bool storeH)
{
    ChunkKdaFwdArch35Options options;
    const bool shapeSupported =
        isAscend950 && chunkSize == 64 && kDim == 128 && vDim == 128;
    if (!shapeSupported) {
        return options;
    }

    // Tiling keys describe shape families independently of the SoC. These
    // options only enable arch35 sub-pipelines within the selected family.
    options.computeGateInPrepare =
        qIsBf16 && rawGIsFp32 && hasALog &&
        useGateInKernel && safeGate;
    const bool denseAligned = !isVarLen && seqlen % chunkSize == 0;
    options.useDenseFwdH = denseAligned && qIsBf16;
    const bool canFusePreparePostWu =
        denseAligned && qIsBf16 && safeGate && vHeads % 2 == 0;
    options.fusePostWuIntoFwdH =
        options.useDenseFwdH && canFusePreparePostWu &&
        options.computeGateInPrepare &&
        !storeQG && !storeVNew && !storeH;
    options.fusePostWu =
        canFusePreparePostWu && !options.fusePostWuIntoFwdH;
    return options;
}

} // namespace optiling::arch35
