#include <map>
#include "tiling_data.h"
#include "common.h"
#include "common_tiling.h"

namespace pp_matmul {

constexpr uint32_t L1_DESCALE_BUFFER_LEN_MAX = 6144;
constexpr uint32_t CONST_3 = 3;
constexpr uint32_t CONST_4 = 4;
constexpr uint32_t CONST_16 = 16;
constexpr uint32_t CONST_32 = 32;
constexpr uint32_t CONST_256 = 256;
constexpr uint32_t CONST_512 = 512;

const std::map<TensorDType, uint32_t> G_DTYPE_MAP = {{TensorDType::TENSOR_DTYPE_FLOAT16, 1u},
                                                     {TensorDType::TENSOR_DTYPE_BF16, 2u}};
const std::map<TensorFormat, uint32_t> G_FORMAT_MAP = {{TensorFormat::TENSOR_FORMAT_ND, 0u},
                                                       {TensorFormat::TENSOR_FORMAT_NZ, 1u}};
using MmType = MatMul::MatMulType;
using QmType = MatMul::QuantMode;
using namespace host_utils;

bool IsI8Bf16Kernel(const MatMulInfo &mmInfo)
{
    bool isI8Bf16 = mmInfo.isInt8 && mmInfo.dtypeC == TensorDType::TENSOR_DTYPE_BF16;
    bool isI8Fp16 = mmInfo.isInt8 && mmInfo.dtypeC == TensorDType::TENSOR_DTYPE_FLOAT16 &&
                    mmInfo.quantMode == QmType::PER_TOKEN_SYMM;
    return isI8Bf16 || isI8Fp16;
}

HardwareInfo::HardwareInfo()
{
    auto &platform = PlatformInfo::Instance();
    coreNum = platform.coreNumAic;
    l2Size = platform.l2Size;
    l1Size = platform.l1Size;
    l0aSize = platform.l0aSize;
    l0bSize = platform.l0bSize;
    l0cSize = platform.l0cSize;
    hbmBandWidth = 1;
    l2BandWidth = 5;  // 5x faster than hbm.
}

void PpMatmulTilingData::SetBaseShape(uint32_t batchSize, uint32_t m, uint32_t k, uint32_t n)
{
    opShape.batchSize = batchSize;
    opShape.m = m;
    opShape.k = k;
    opShape.n = n;
}

void PpMatmulTilingData::SetBaseOp(uint32_t coreNum, uint32_t mBase, uint32_t nBase, const MatMulInfo &mmInfo)
{
    opShape.m0 = mBase;
    opShape.n0 = nBase;
    mLoop = CeilDiv(opShape.m, opShape.m0);
    nLoop = CeilDiv(opShape.n, opShape.n0);
    coreLoop = opShape.batchSize * mLoop * nLoop;

    if (mLoop == 1 && mmInfo.transB && coreLoop % coreNum < coreNum / CONST_4 * CONST_3) {
        mBase = RoundUp<uint32_t>(opShape.m, CONST_16);
        opShape.m0 = mBase;
        uint32_t maxN0 = PlatformInfo::Instance().l0cSize / (mBase * sizeof(float));
        if (mmInfo.isInt8 || mmInfo.mmType == MmType::MATMUL_WITH_BIAS) {
            maxN0 = maxN0 < CONST_256 ? maxN0 : CONST_256;
        }
        uint32_t x = CeilDiv(opShape.n, coreNum);
        uint32_t y = CeilDiv(x, maxN0);
        nBase = RoundUp<uint32_t>(CeilDiv(x, y), CONST_16);
        uint32_t rqdL0CSize = mBase * nBase * sizeof(float);
        if (rqdL0CSize < PlatformInfo::Instance().l0cSize &&
            (mBase + nBase) * CONST_256 * sizeof(uint16_t) < L1AB_PINGPONG_BUFFER_LEN) {
            opShape.n0 = nBase;
            nLoop = CeilDiv(opShape.n, opShape.n0);
            coreLoop = opShape.batchSize * nLoop;
        }
    }
    blockDim = std::min(coreLoop, coreNum);
}

// transA transB quantMode [dtype] format
void PpMatmulTilingData::SetTilingKey(const MatMulInfo &mmInfo, uint32_t swizzleDirect, uint32_t enSplitK)
{
    if (mmInfo.mmType == MmType::MATMUL_ACCUM_ATOMIC || mmInfo.mmType == MmType::MATMUL_WITH_BIAS ||
        mmInfo.mmType == MmType::MATMUL_EIN_SUM || mmInfo.mmType == MmType::MATMUL_DEQUANT || IsI8Bf16Kernel(mmInfo)) {
        // SwizzleDir[1] TransA[1] TransB[1] DtypeA[3] DtypeB[3] DtypeC[3] FormatA[1] FormatB[1] FormatC[1] WithBias[1]
        tilingKey = swizzleDirect;
        tilingKey = (tilingKey << 1) + static_cast<uint32_t>(mmInfo.transA);
        tilingKey = (tilingKey << 1) + static_cast<uint32_t>(mmInfo.transB);
        tilingKey = (tilingKey << 3) + G_DTYPE_MAP.at(mmInfo.dtypeA);  // 3bit for dtypeA.
        tilingKey = (tilingKey << 3) + G_DTYPE_MAP.at(mmInfo.dtypeB);  // 3bit for dtypeB.
        tilingKey = (tilingKey << 3) + G_DTYPE_MAP.at(mmInfo.dtypeC);  // 3bit for dtypeC.
        tilingKey = (tilingKey << 1) + G_FORMAT_MAP.at(mmInfo.formatA);
        tilingKey = (tilingKey << 1) + G_FORMAT_MAP.at(mmInfo.formatB);
        tilingKey = (tilingKey << 1) + G_FORMAT_MAP.at(mmInfo.formatC);
        tilingKey = (tilingKey << 1) + static_cast<uint32_t>(mmInfo.biasFlag);
    } else {
        tilingKey = swizzleDirect;
        tilingKey = (tilingKey << 1) + static_cast<uint32_t>(mmInfo.transA);
        tilingKey = (tilingKey << 1) + static_cast<uint32_t>(mmInfo.transB);
        tilingKey = (tilingKey << 1) + static_cast<uint32_t>(mmInfo.isInt8);
        tilingKey = (tilingKey << 1) + static_cast<uint32_t>(mmInfo.biasFlag);
        tilingKey = (tilingKey << 1) + enSplitK;
    }
}

uint32_t PpMatmulTilingData::End(const MatMulInfo &mmInfo)
{
    uint32_t cubeBlockSize = mmInfo.isInt8 ? CUBE_BLOCK_SIZE_INT8 : CUBE_BLOCK_SIZE;
    uint32_t kBlockSize = mmInfo.isInt8 ? BLOCK_SIZE_INT8_K : BLOCK_SIZE;
    uint32_t scaleBlockSize = mmInfo.isInt8 ? L1_DESCALE_BUFFER_LEN_MAX : 0;
    uint32_t shapeSum = opShape.m0 + opShape.n0;
    if (mmInfo.isInt8 && (mmInfo.transA || !mmInfo.transB)) {
        shapeSum = RoundUp<uint32_t>(opShape.m0, CONST_32) + RoundUp<uint32_t>(opShape.n0, CONST_32);
    }
    uint32_t k0Max = shapeSum == 0
                         ? L1AB_PINGPONG_BUFFER_LEN
                         : static_cast<uint32_t>(static_cast<float>(L1AB_PINGPONG_BUFFER_LEN - scaleBlockSize) /
                                                 (shapeSum * mmInfo.inDtype));
    if (mmInfo.mmType == MatMul::MatMulType::MATMUL_WITH_BIAS) {
        uint32_t l1AbSize = L1AB_PINGPONG_BUFFER_LEN - opShape.n0 * sizeof(float);
        k0Max = l1AbSize / (shapeSum * mmInfo.inDtype);
    }

    opShape.k0 =
        k0Max < cubeBlockSize ? RoundDown<uint32_t>(k0Max, kBlockSize) : RoundDown<uint32_t>(k0Max, cubeBlockSize);
    if (opShape.k0 > CONST_512) {
        opShape.k0 = RoundDown<uint32_t>(opShape.k0, CONST_512);
    }
    kLoop = CeilDiv(opShape.k, opShape.k0);
    return blockDim;
}

void GetPpMatmulTiling(const MatMulInfo &mmInfo, const HardwareInfo &hwInfo, uint32_t &blockDim,
                       PpMatmulTilingData &tilingData)
{
    OpShape opShape;
    opShape.batchSize = mmInfo.batchSize;
    opShape.m = mmInfo.m;
    opShape.n = mmInfo.n;
    opShape.k = mmInfo.k;
    tilingData.opShape = opShape;
    tilingData.quantMode = static_cast<uint32_t>(mmInfo.quantMode);
    tilingData.SetTilingKey(mmInfo, 0, 0);  // init tilingkey with transA transB.
    if (opShape.m < opShape.n) {
        TilingFunc<false, OpShape, PpMatmulTilingData, HardwareInfo, MatMulInfo>(opShape, tilingData, hwInfo, mmInfo);
    } else {
        TilingFunc<true, OpShape, PpMatmulTilingData, HardwareInfo, MatMulInfo>(opShape, tilingData, hwInfo, mmInfo);
    }
    uint32_t direct = Swizzl<PpMatmulTilingData>(tilingData);
    blockDim = tilingData.End(mmInfo);
    tilingData.SetTilingKey(mmInfo, direct, 0);
}
}  // namespace pp_matmul
