#ifndef PP_MATMUL_TILING_DATA
#define PP_MATMUL_TILING_DATA
#include <cstdint>

namespace pp_matmul {
struct MatMul {
    enum class MatMulType : uint32_t {
        MATMUL_DEFAULT = 0,   // C = op(A) * op(B)
        MATMUL_DEQUANT,       //
        MATMUL_ACCUM_ATOMIC,  // C += op(A) * op(B)
        MATMUL_WITH_BIAS,     // C = op(A) * op(B) + Bias, where Bias is a vector.
        MATMUL_EIN_SUM
    };
    enum class QuantMode : uint32_t { PER_CHANNEL_SYMM = 0, PER_CHANNEL_ASYMM, PER_TOKEN_SYMM };
};

enum class TensorDType : uint32_t { TENSOR_DTYPE_FLOAT16 = 0, TENSOR_DTYPE_BF16 };

enum class TensorFormat : uint32_t { TENSOR_FORMAT_ND = 0, TENSOR_FORMAT_NZ };

struct MatMulInfo {
    uint32_t batchSize{0};
    uint32_t m{0};  // actual input m
    uint32_t k{0};  // actual input k
    uint32_t n{0};  // actual input n
    TensorDType dtypeA{TensorDType::TENSOR_DTYPE_FLOAT16};
    TensorDType dtypeB{TensorDType::TENSOR_DTYPE_FLOAT16};
    TensorDType dtypeC{TensorDType::TENSOR_DTYPE_FLOAT16};
    TensorFormat formatA{TensorFormat::TENSOR_FORMAT_ND};
    TensorFormat formatB{TensorFormat::TENSOR_FORMAT_ND};
    TensorFormat formatC{TensorFormat::TENSOR_FORMAT_ND};
    MatMul::MatMulType mmType{MatMul::MatMulType::MATMUL_DEFAULT};
    bool transA{0};    // false: 0, true: 1
    bool transB{0};    // false: 0, true: 1
    bool biasFlag{0};  // false: 0, true: 1
    bool isInt8{0};    // false: 0, true: 1
    float inDtype{0};
    float outDtype{0};
    MatMul::QuantMode quantMode{MatMul::QuantMode::PER_CHANNEL_SYMM};
};

struct OpShape {
    uint32_t batchSize{0};
    uint32_t m{0};
    uint32_t k{0};
    uint32_t n{0};
    uint32_t m0{0};
    uint32_t k0{0};
    uint32_t n0{0};
};

struct HardwareInfo {
    uint32_t coreNum{0};
    uint32_t l2Size{0};
    uint32_t l1Size{0};
    uint32_t l0aSize{0};
    uint32_t l0bSize{0};
    uint32_t l0cSize{0};
    uint32_t hbmBandWidth{0};
    uint32_t l2BandWidth{0};

    HardwareInfo();
};

#pragma pack(push, 1)
struct PpMatmulTilingData {
    OpShape opShape{};
    uint32_t mLoop{1};
    uint32_t kLoop{1};
    uint32_t nLoop{1};
    uint32_t coreLoop{1};
    uint32_t swizzlCount{1};
    uint32_t tilingKey{0};
    uint32_t blockDim{1};
    uint32_t swizzlDirect{0};
    uint32_t splitk{0};
    uint32_t enShuffleK{0};
    uint32_t quantMode{0};

    void SetBaseShape(uint32_t batchSize, uint32_t m, uint32_t k, uint32_t n);
    void SetBaseOp(uint32_t coreNum, uint32_t mBase, uint32_t nBase, const MatMulInfo &mmInfo);
    void SetTilingKey(const MatMulInfo &mmInfo, uint32_t swizzleDirect, uint32_t enSplitK);
    uint32_t End(const MatMulInfo &mmInfo);
};
#pragma pack(pop)

void GetPpMatmulTiling(const MatMulInfo &mmInfo, const HardwareInfo &hwInfo, uint32_t &blockDim,
                       PpMatmulTilingData &tilingData);
}  // namespace pp_matmul
#endif
