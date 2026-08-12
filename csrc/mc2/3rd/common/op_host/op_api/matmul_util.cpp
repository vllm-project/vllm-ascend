/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "common/op_host/op_api/mc2_3rd_matmul_util.h"

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "level0/fill.h"
#include "level0/padv3.h"
#include "level0/mul.h"
#include "level0/matmul_v2tov3.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/platform.h"
#include "opdev/op_log.h"
#include "mc2_log_compat.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/shape_utils.h"
#include "op_api/op_api_def.h"
#include "common/op_host/op_api/cube_util.h"
#include "common/op_host/math_util.h"

using namespace std;
using namespace op;
using namespace Ops::Transformer;

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;

namespace {
static const int64_t SPLIT_K_MULTI = 8;
static const int64_t MKN_MAX = 8000000000;
static const int64_t MN_MULTI = 50;
static const int64_t N_KEQAL1_LIMIT = 4000;
static const int64_t MM_KEQAL1_LIMIT = 10000;
static const size_t MM_DIM = 2;
static const int32_t INNER_AXIS = 1;
static const int32_t OUTER_AXIS = 2;
static const int64_t DIM_EQUAL_ONE = 1;
static const uint64_t SMALL_SHAPE_LIMIT = 524288UL;
static const uint32_t kDeqScaleMul = 0xFFFFE000;
static const uint64_t BASIC_BLOCK_SIZE_256 = 256;
static const uint64_t NUM_HALF = 2;
static const uint64_t FP32_HF32_DTYPE_SIZE = 4;
static const uint64_t BASIC_BLOCK_K_256_BYTE = 256;
static const uint64_t BASIC_BLOCK_SIZE_32 = 32;
static const uint64_t BASIC_BLOCK_SIZE_128 = 128;
static const double THRESHOLD = 0.7;
static const uint64_t CACHELINE = 512;
static const uint64_t NUM_TWO = 2;
static const int64_t DIMS_TWO = 2;
static const int64_t HALF_ALIGN_UNIT = 256;
static const int64_t ALIGN_UNIT = 512;
static const int64_t M_DIM_SELF_IDX = 0;
static const int64_t K_DIM_SELF_IDX = 1;
static const int64_t N_DIM_SELF_IDX = 1;
static const int64_t ALIGN_UNIT_16 = 16;
static const int64_t ALIGN_UNIT_128 = 128;
static const int64_t MIN_V3_SHAPE_310 = 2048;
static const int64_t MAX_V3_SHAPE_310 = 5504;
static const int64_t SINGLE_CORE_SPLIT_K = 27392;
static const int64_t BLOCK_CUBE = 16;
static const int64_t BLOCK_BYTE_SIZE = 32;
static const uint64_t MB = 1024UL * 1024UL;

static const std::initializer_list<op::DataType> V100_DTYPE_SUPPORT = {DataType::DT_FLOAT16, DataType::DT_BF16};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16,
                                                                       DataType::DT_BF16};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_WITHOUT_BF16 = {DataType::DT_FLOAT,
                                                                                    DataType::DT_FLOAT16};

static inline bool CheckMathType(const aclTensor *self, const aclTensor *mat2, int8_t cubeMathType)
{
    bool mat2Float = mat2->GetDataType() == DataType::DT_FLOAT;
    bool selfFloat = self->GetDataType() == DataType::DT_FLOAT;
    auto promoteType = (mat2Float || selfFloat) ? DataType::DT_FLOAT : self->GetDataType();
    return CheckCubeMathTypeForMm(promoteType, cubeMathType);
}

static inline bool CheckKEqual1Support(void)
{
    return GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B;
}

static inline bool CheckSocVersionIsSupportBf16(void)
{
    return GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E &&
           GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *mat2, const aclTensor *bias, const aclTensor *out,
                            int8_t cubeMathType)
{
    bool bf16flag = CheckSocVersionIsSupportBf16();
    if (bf16flag) {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(mat2, DTYPE_SUPPORT_LIST, return false);
        if (bias != nullptr) {
            OP_CHECK_DTYPE_NOT_SUPPORT(bias, DTYPE_SUPPORT_LIST, return false);
        }
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST_WITHOUT_BF16, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(mat2, DTYPE_SUPPORT_LIST_WITHOUT_BF16, return false);
        if (bias != nullptr) {
            OP_CHECK_DTYPE_NOT_SUPPORT(bias, DTYPE_SUPPORT_LIST_WITHOUT_BF16, return false);
        }
    }

    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (!bf16flag && (self->GetDataType() == op::DataType::DT_BF16 || mat2->GetDataType() == op::DataType::DT_BF16)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Bfloat16 is unsupported by the current SOC version [%s], now self is %s, mat2 is %s",
                op::ToString(socVersion).GetString(), op::ToString(self->GetDataType()).GetString(),
                op::ToString(mat2->GetDataType()).GetString());
        return false;
    }
    if (out != nullptr && cubeMathType == KEEP_DTYPE && out->GetDataType() == op::DataType::DT_FLOAT16 &&
        self->GetDataType() == op::DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Input tensor's dtype[DT_FLOAT] should be same with output's dtype[DT_FLOAT16].");
        return false;
    }
    // self和mat2的dtype不相等时，会做promote处理。
    bool dtype_match = self->GetDataType() == mat2->GetDataType();
    if (!dtype_match) {
        OP_LOGW("Self's dtype [%s] and mat2's dtype [%s] are not equal. Promotion of Data Type will be applied",
                op::ToString(self->GetDataType()).GetString(), op::ToString(mat2->GetDataType()).GetString());
    }
    return true;
}

static bool CheckShapeValid(const aclTensor *self, const aclTensor *mat2, bool transposeX2 = false)
{
    if (mat2->GetViewShape().GetDimNum() != DIMS_TWO) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("Matmul", "mat2",
                                                 (std::to_string(mat2->GetViewShape().GetDimNum()) + "D").c_str(),
                                                 "The shape of mat2 must be 2D.");
        return false;
    }
    if (self->GetViewShape().GetDimNum() != DIMS_TWO) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("Matmul", "self",
                                                 (std::to_string(self->GetViewShape().GetDimNum()) + "D").c_str(),
                                                 "The shape of self must be 2D.");
        return false;
    }
    op::Shape mat2Shape = mat2->GetViewShape();
    op::Shape selfShape = self->GetViewShape();
    int64_t mat2KDim = transposeX2 ? mat2Shape.GetDim(K_DIM_SELF_IDX) : mat2Shape.GetDim(M_DIM_SELF_IDX);
    int64_t selfKDim = selfShape.GetDim(K_DIM_SELF_IDX); // self固定不转置
    if (mat2KDim != selfKDim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The k-axis of the two inputs are different, self Kdim[%ld], mat2 Kdim[%ld].",
                selfKDim, mat2KDim);
        return false;
    }
    return true;
}


static bool CheckSupportSingleSplitKFp16Bf16(const aclTensor *self, const aclTensor *mat2, const DataType selfDtype,
                                             const DataType mat2Dtype)
{
    // 判决门限
    // 1. 输入数据类型为fp16/bf16
    // 2. 在K轴非256字节对齐场景下，输入数据大小不超过INT32最大值
    // 3. K轴大于27392
    // 4. M、N中最大不超过K轴的一半
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) {
        return false;
    }
    op::Shape selfShape = self->GetViewShape();
    op::Shape mat2Shape = mat2->GetViewShape();

    int64_t splitKMultiThres = 2;
    int64_t kDim = selfShape.GetDim(1);
    bool dtypeCorrect = ((selfDtype == DataType::DT_BF16) && (mat2Dtype == DataType::DT_BF16)) ||
                        ((selfDtype == DataType::DT_FLOAT16) && (mat2Dtype == DataType::DT_FLOAT16));
    int64_t checkSize = 0;

    int dtypeASize = ge::GetSizeByDataType(selfDtype);
    int dtypeBSize = ge::GetSizeByDataType(mat2Dtype);
    if ((kDim * dtypeASize) % 256 != 0) { // check if inner-dim is aligned to 256 byte
        checkSize += selfShape[selfShape.GetDimNum() - 1] * selfShape[selfShape.GetDimNum() - DIMS_TWO] * dtypeASize;
        checkSize += mat2Shape[mat2Shape.GetDimNum() - 1] * mat2Shape[mat2Shape.GetDimNum() - DIMS_TWO] * dtypeBSize;
    }

    bool checkMemSize = checkSize <= INT32_MAX;
    return dtypeCorrect && checkMemSize && kDim >= SINGLE_CORE_SPLIT_K &&
           kDim >= splitKMultiThres * std::max(selfShape.GetDim(0), mat2Shape.GetDim(1));
}

static aclnnStatus SetMatmulOpSupportInfo(const aclTensor *self, const aclTensor *mat2, MmOpInfo &mmOpInfo,
                                          int8_t cubeMathType)
{
    // 判断传入L0接口，用于计算的Dtype
    SetMmSupportDType(mmOpInfo, cubeMathType);

    // 判断当前Shape是否支持使用ND输入输出
    SetMmSupportFormat(self, mat2, mmOpInfo);

    TensorInfo SpTensor_sefl = {self, mmOpInfo.support_info.self_dtype, mmOpInfo.support_info.self_format};
    TensorInfo SpTensor_mat2 = {mat2, mmOpInfo.support_info.mat2_dtype, mmOpInfo.support_info.output_format};

    if (IsSplitk(&SpTensor_sefl, &SpTensor_mat2)) {
        mmOpInfo.supporSplitK = true;
        if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P) {
            mmOpInfo.support_info.output_dtype = DataType::DT_FLOAT;
        } else if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910) {
            mmOpInfo.support_info.output_format = Format::FORMAT_FRACTAL_NZ;
            mmOpInfo.support_info.output_dtype = DataType::DT_FLOAT;
        }
    }

    if (CheckSupportSingleSplitKFp16Bf16(self, mat2, mmOpInfo.support_info.self_dtype,
                                         mmOpInfo.support_info.mat2_dtype)) {
        OP_LOGI("Hit mat_mul_v3 ND fp16/bf16 single core splitK case channel.");
        mmOpInfo.support_info.output_format = Format::FORMAT_ND;
        mmOpInfo.support_info.self_format = Format::FORMAT_ND;
        mmOpInfo.support_info.mat2_format = Format::FORMAT_ND;
        mmOpInfo.supporSplitK = true;
    }

    // self=nd, mat2=nz不支持切K
    bool isNdNzIn =
        self->GetStorageFormat() == Format::FORMAT_ND && mat2->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ;
    mmOpInfo.support_info.mat2_format = isNdNzIn ? Format::FORMAT_FRACTAL_NZ : mmOpInfo.support_info.mat2_format;
    mmOpInfo.support_info.output_dtype =
        isNdNzIn ? mmOpInfo.support_info.mat2_dtype : mmOpInfo.support_info.output_dtype;
    return ACLNN_SUCCESS;
}


static inline bool IsSplitKThenForbiddenNd2Nz(const uint64_t mDim, const uint64_t kDim, const uint64_t nDim,
                                              const bool transposeX1, const bool transposeX2)
{
    constexpr uint64_t align128 = 128;
    constexpr uint64_t numTwo = 2;
    constexpr uint64_t smallMNsplitKThres = 15000;
    bool kIsEnoughMultiCore = kDim >= smallMNsplitKThres;
    bool mnIsNotEnoughCore = (std::ceil(mDim / align128) * std::ceil(nDim / align128) <
                              static_cast<int64_t>(GetCurrentPlatformInfo().GetCubeCoreNum() / numTwo));
    // M/N轴在内轴的场景切m/n不影响MTE2搬运效率，M/N可以切小保证多核能开启，属于cube_bound场景
    return kIsEnoughMultiCore && mnIsNotEnoughCore && !(!transposeX1 && transposeX2);
}

static inline int64_t ComputePadNum(int64_t kDim, int64_t dataSize)
{
    return CeilAlign(kDim, CeilDiv(ALIGN_UNIT, dataSize)) - kDim;
}

static inline const aclTensor *GetPadTensor(int64_t padNum, int64_t padDim, aclOpExecutor *executor)
{
    // pad: top bottom left right
    size_t dims = 4;
    std::vector<int64_t> padVec(dims, 0);

    padVec[padDim] = padNum;

    auto padArray = executor->AllocIntArray(padVec.data(), dims);
    if (padArray == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try alloc padVec failed");
        return nullptr;
    }

    auto padTensor = executor->ConvertToTensor(padArray, DataType::DT_INT64);
    return padTensor;
}

static bool CheckStreamKSKTiling(MmOpInfo &mmOpInfo)
{
    // 判断k轴是否大于32*512 / DtypeSize_, 小于就不走stream-k-sk
    uint64_t kAlign =
        static_cast<uint64_t>(CeilAlign(mmOpInfo.shapeInfo.kDim, static_cast<int64_t>(BASIC_BLOCK_SIZE_256)));
    uint64_t aiCoreCnt = std::max(uint64_t{1}, static_cast<uint64_t>(mmOpInfo.aiCoreCnt));
    uint64_t dtypeASize = std::max(uint64_t{1}, static_cast<uint64_t>(mmOpInfo.shapeInfo.dtypeASize));
    uint64_t kThreshold = aiCoreCnt * NUM_HALF * BASIC_BLOCK_K_256_BYTE / dtypeASize;
    if (kAlign < kThreshold) {
        return false;
    }

    uint64_t alignValue = BASIC_BLOCK_SIZE_256;
    //
    if (mmOpInfo.shapeInfo.dtypeASize == FP32_HF32_DTYPE_SIZE && !mmOpInfo.enableHf32) {
        alignValue = BASIC_BLOCK_SIZE_32; // 如果是Fp32 基本块判断要用32
    }
    // 判断mn是否需要已经能切32份及以上
    uint64_t mCnt = static_cast<uint64_t>(CeilDiv(mmOpInfo.shapeInfo.mDim, static_cast<int64_t>(alignValue)));
    uint64_t nCnt = static_cast<uint64_t>(CeilDiv(mmOpInfo.shapeInfo.nDim, static_cast<int64_t>(alignValue)));
    return !(mCnt * nCnt > aiCoreCnt / NUM_HALF);
}

static bool CheckStreamKDPSKTiling(MmOpInfo &mmOpInfo)
{
    uint64_t aiCoreCnt = std::max(uint64_t{1}, static_cast<uint64_t>(mmOpInfo.aiCoreCnt));
    uint64_t dtypeASize = std::max(uint64_t{1}, static_cast<uint64_t>(mmOpInfo.shapeInfo.dtypeASize));
    uint64_t kThreshold = aiCoreCnt * BASIC_BLOCK_K_256_BYTE / dtypeASize;
    uint64_t kDim = static_cast<uint64_t>(mmOpInfo.shapeInfo.kDim);
    // 如果k轴小于32*256/DtypeSize_ 或 mn轴不是256对齐 或 输入是fp32类型，不走stream-k-dpsk
    if (mmOpInfo.shapeInfo.mDim % BASIC_BLOCK_SIZE_256 != 0UL ||
        mmOpInfo.shapeInfo.nDim % BASIC_BLOCK_SIZE_256 != 0UL || kDim < kThreshold ||
        (dtypeASize == FP32_HF32_DTYPE_SIZE && !mmOpInfo.enableHf32)) {
        return false;
    }
    // 如果mn用256切分的份数小于核数 或者 取余核数为0或大于一半的核数，则不使用stream-k-dpsk
    uint64_t mCnt = static_cast<uint64_t>(CeilDiv(mmOpInfo.shapeInfo.mDim, static_cast<int64_t>(BASIC_BLOCK_SIZE_256)));
    uint64_t nCnt = static_cast<uint64_t>(CeilDiv(mmOpInfo.shapeInfo.nDim, static_cast<int64_t>(BASIC_BLOCK_SIZE_256)));
    uint64_t tatalMNCnt = mCnt * nCnt;
    if ((tatalMNCnt < aiCoreCnt) || (tatalMNCnt % aiCoreCnt == 0UL) || (tatalMNCnt % aiCoreCnt > aiCoreCnt / NUM_TWO)) {
        return false;
    }
    return true;
}

// 判断shape是否支持GemmV3
static bool CheckShapeSupport(MmOpInfo &mmOpInfo)
{
    // 判断是否不走stream-k
    bool notSkTiling = !CheckStreamKSKTiling(mmOpInfo) && !CheckStreamKDPSKTiling(mmOpInfo);
    // 判断m和n是否大于等于512
    bool mnValid = mmOpInfo.shapeInfo.mDim >= static_cast<int64_t>(CACHELINE) &&
                   mmOpInfo.shapeInfo.nDim >= static_cast<int64_t>(CACHELINE);
    // 判断k是否大于256
    bool kValid = mmOpInfo.shapeInfo.kDim > static_cast<int64_t>(BASIC_BLOCK_SIZE_256);
    // 满足条件shape范围
    bool shapeAswt = notSkTiling && mnValid && kValid;
    if (!shapeAswt) {
        OP_LOGI("Not support this shape in gemmV3.");
        return false;
    }
    OP_LOGD("Check shape success in gemmV3.");
    return true;
}
} // namespace

namespace Ops {
namespace Transformer {
op::Shape SwapLastTwoDimValue(const op::Shape tensorShape)
{
    op::Shape swapedShape = tensorShape;
    int64_t dimNum = tensorShape.GetDimNum();
    if (dimNum >= static_cast<int64_t>(MM_DIM)) {
        int64_t lastDim = tensorShape.GetDim(dimNum - 1);
        // dimNum - 1, 这里1指的是取最后一维的dim值。dimNum - 2, 这里2指的是取倒数第二维的dim值
        swapedShape.SetDim(dimNum - 1, tensorShape.GetDim(dimNum - 2));
        // dimNum - 2, 这里2指的是取倒数第二维的dim值
        swapedShape.SetDim(dimNum - 2, lastDim);
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimNum is not supported , which is %ld.", dimNum);
    }
    return swapedShape;
}

MmOpInfo GetMatmulOpInfo(const aclTensor *self, const aclTensor *mat2, int8_t cubeMathType)
{
    // 获取m、k、n轴的大小
    op::Shape selfShape = self->GetViewShape();
    op::Shape mat2Shape = mat2->GetViewShape();
    int64_t mDim = selfShape.GetDim(M_DIM_SELF_IDX);
    int64_t kDim = selfShape.GetDim(K_DIM_SELF_IDX);
    int64_t nDim = mat2Shape.GetDim(N_DIM_SELF_IDX);

    // Dtype和Format初始化
    MmOpInfo mmOpInfo;
    mmOpInfo.ori_info.self_dtype = self->GetDataType();
    mmOpInfo.ori_info.self_format = op::Format::FORMAT_ND;
    mmOpInfo.ori_info.mat2_dtype = mat2->GetDataType();
    mmOpInfo.ori_info.mat2_format = op::Format::FORMAT_ND;
    mmOpInfo.ori_info.output_dtype = self->GetDataType();
    if (FP16FP32_KEEP_DTYPE == cubeMathType) {
        mmOpInfo.ori_info.output_dtype = DataType::DT_FLOAT;
    }
    mmOpInfo.ori_info.output_format = op::Format::FORMAT_ND;

    mmOpInfo.shapeInfo.kDim = kDim;
    mmOpInfo.shapeInfo.nDim = nDim;
    mmOpInfo.shapeInfo.mDim = mDim;
    mmOpInfo.shapeInfo.transposeX1 = false;
    mmOpInfo.shapeInfo.transposeX2 = false;
    mmOpInfo.shapeInfo.dtypeASize = ge::GetSizeByDataType(self->GetDataType());
    mmOpInfo.shapeInfo.dtypeBSize = ge::GetSizeByDataType(mat2->GetDataType());
    OP_LOGD("mDim=%ld, kDim=%ld, nDim=%ld, dtypeASize=%ld, dtypeBSize=%ld", mDim, kDim, nDim,
            mmOpInfo.shapeInfo.dtypeASize, mmOpInfo.shapeInfo.dtypeBSize);
    mmOpInfo.support_info = mmOpInfo.ori_info;

    // 不同芯片能力不同
    // fp16 fp32 选择，1980 vector支持fp32
    SetMatmulOpSupportInfo(self, mat2, mmOpInfo, cubeMathType);

    // 获取aicore数目
    mmOpInfo.aiCoreCnt = GetCurrentPlatformInfo().GetCubeCoreNum();

    bool inputFp32Flag = mmOpInfo.support_info.self_dtype == DataType::DT_FLOAT &&
                         mmOpInfo.support_info.mat2_dtype == DataType::DT_FLOAT;
    // 如果允许降精度处理， 则开启HF32模式（0x40），否则采用默认模式; 后续此字段配置需要按照字段表进行配置
    mmOpInfo.opImplModeEnum =
        (inputFp32Flag && ((cubeMathType == ALLOW_FP32_DOWN_PRECISION) || (cubeMathType == USE_HF32))) ? 0x40 : 0x1;
    mmOpInfo.enableHf32 = inputFp32Flag && ((cubeMathType == ALLOW_FP32_DOWN_PRECISION) || (cubeMathType == USE_HF32));
    OP_LOGD("opImplModeEnum=%ld, enableHf32=%d, cubeMathType=%d, inputFp32Flag= %d", mmOpInfo.opImplModeEnum,
            mmOpInfo.enableHf32, cubeMathType, inputFp32Flag);
    // Log mm info
    GetMmInfo(mmOpInfo);
    return mmOpInfo;
}

bool ContiguousAndCast(const aclTensor *&contiguousInput, const aclTensor *&castOut, bool &transposeFlag,
                       op::DataType dtype, aclOpExecutor *executor)
{
    auto contiguousOut = contiguousInput;
    if (IsTransposeLastTwoDims(contiguousInput)) {
        contiguousOut = executor->CreateView(contiguousInput, SwapLastTwoDimValue(contiguousInput->GetViewShape()),
                                             contiguousInput->GetViewOffset());
        transposeFlag = true;
    } else {
        contiguousOut = l0op::Contiguous(contiguousInput, executor);
    }
    CHECK_RET(contiguousOut != nullptr, false);

    // cast
    castOut = l0op::Cast(contiguousOut, dtype, executor);
    CHECK_RET(castOut != nullptr, false);
    return true;
}

bool CheckGemmV3Support(const aclTensor *mat1, const aclTensor *mat2, MmOpInfo &mmOpInfo, int8_t cubeMathType)
{
    CHECK_RET(mat1 != nullptr, false);
    CHECK_RET(mat2 != nullptr, false);
    CHECK_RET(CheckDtypeValid(mat1, mat2, nullptr, nullptr, cubeMathType), false);
    CHECK_RET(CheckShapeValid(mat1, mat2), false);
    CHECK_RET(CheckMathType(mat1, mat2, cubeMathType), false);
    // 空Tensor不由gemmV3处理
    if (mat1->IsEmpty() || mat2->IsEmpty()) {
        OP_LOGI("mat1 or mat2 is empty, does not support GemmV3.");
        return false;
    }

    if (mat2->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ ||
        mat1->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ) {
        OP_LOGI("mat1 or mat2 StorageFormat is FORMAT_FRACTAL_NZ, does not support GemmV3.");
        return false;
    }
    // 当前支持平台
    if (GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        OP_LOGI("Current SOC version does not support GemmV3.");
        return false;
    }

    // 解析当前规格matmulop支持的dtype format能力
    mmOpInfo = GetMatmulOpInfo(mat1, mat2, cubeMathType);

    // 当前支持shape范围
    return CheckShapeSupport(mmOpInfo);
}

bool IsInputSupportFp32()
{
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_2201 &&
        op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        return false;
    }
    return true;
}

bool CheckBatchDimBroadcast(size_t batch1DimNum, size_t batch2DimNum, const op::Shape &batch1, const op::Shape &batch2)
{
    size_t batchIndex = MM_DIM;
    while (batch1DimNum > batchIndex && batch2DimNum > batchIndex) {
        if (batch1[batch1DimNum - batchIndex - 1] != 1 && batch2[batch1DimNum - batchIndex - 1] != 1 &&
            batch1[batch1DimNum - batchIndex - 1] != batch2[batch1DimNum - batchIndex - 1]) {
            return false;
        }
        batchIndex++;
    }
    return true;
}

// bmm 相对于 mm 取坐标需偏移
int64_t GetOffSet(int64_t DimNum)
{
    int64_t rightMove = 0;
    // bmm DimNum 为 3, mm DimNum 为 2 ，bmm需要相对于mm向后偏移一位取行列值，默认rightMove为 0
    rightMove = DimNum == 3 ? 1 : 0;
    return rightMove;
}

// 检查单Tensor是否为支持带bias的mm的dtype
static inline bool CheckDtypeSupport(const aclTensor *tensor)
{
    if (!IsInputSupportFp32()) {
        auto iter = std::find(V100_DTYPE_SUPPORT.begin(), V100_DTYPE_SUPPORT.end(), tensor->GetDataType());
        return iter != V100_DTYPE_SUPPORT.end();
    }
    auto iter = std::find(DTYPE_SUPPORT_LIST.begin(), DTYPE_SUPPORT_LIST.end(), tensor->GetDataType());
    return iter != DTYPE_SUPPORT_LIST.end();
}

// 检查是否为支持带bias的mm的dtype
static inline bool CheckDtypeSupportBias(const aclTensor *self, const aclTensor *mat1, const aclTensor *mat2)
{
    bool matMulDtypeCorrect = CheckDtypeSupport(mat1) && CheckDtypeSupport(mat2);
    if (mat1->GetDataType() == DataType::DT_BF16) {
        return matMulDtypeCorrect &&
               (self->GetDataType() == DataType::DT_BF16 || self->GetDataType() == DataType::DT_FLOAT);
    }
    return CheckDtypeSupport(self) && matMulDtypeCorrect;
}

// 如果beta==1 && alpha == 1 && self.shape[0] == mat2.shape[1] && 不属于切k，直接走matmul的bias模式
bool NeedToConvertBias(const aclTensor *self, const aclTensor *mat1, const aclTensor *mat2, const aclScalar *beta,
                       const aclScalar *alpha)
{
    int64_t mat1DimNum = static_cast<int64_t>(mat1->GetViewShape().GetDimNum());
    // rightMove to distinguish different shape of mm and bmm
    int64_t rightMove = 0;
    rightMove = GetOffSet(mat1DimNum);

    TensorInfo Tensor_matl = {mat1, mat1->GetDataType(), Format::FORMAT_ND};
    TensorInfo Tensor_mat2 = {mat2, mat2->GetDataType(), Format::FORMAT_ND};

    bool isSplitK = false;
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_2201 &&
        op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        isSplitK = IsSplitk(&Tensor_matl, &Tensor_mat2);
        ;
    }
    op::Shape selfShape = self->GetViewShape();
    op::Shape mat2Shape = mat2->GetViewShape();
    int64_t selfDimNum = static_cast<int64_t>(selfShape.GetDimNum());
    bool canBeBiasFlag = false;
    // bmm (DimNum==3) only apply the case of batch == 1
    bool batchIsOne = !(mat1DimNum == 3 && mat1->GetViewShape().GetDim(0) != 1);

    if (selfDimNum == 1) {
        canBeBiasFlag = (mat2->GetViewShape().GetDim(1 + rightMove) == self->GetViewShape().GetDim(0)) &&
                        CheckDtypeSupportBias(self, mat1, mat2) && batchIsOne;
        // When input tensor is a 2 dimentional tensor
    } else if (selfDimNum == 2) {
        canBeBiasFlag = (selfShape.GetDim(0) == 1) && (selfShape.GetDim(1) == mat2Shape.GetDim(1 + rightMove)) &&
                        CheckDtypeSupportBias(self, mat1, mat2) && batchIsOne;
    }
    OP_LOGI("Current Shape's canBeBiasFlag = %ld", static_cast<int64_t>(canBeBiasFlag));
    return (std::abs(alpha->ToFloat() - 1.0f) <= std::numeric_limits<float>::epsilon()) &&
           (std::abs(beta->ToFloat() - 1.0f) <= std::numeric_limits<float>::epsilon()) && !isSplitK && canBeBiasFlag;
}

// Nz fp16 in fp32 out experimental rules
bool GetNzSplitKFlag(const aclTensor *self, const aclTensor *mat2, const Format selfSuppFormat,
                     const Format outSuppFormat)
{
    if ((selfSuppFormat == Format::FORMAT_ND) && (outSuppFormat == Format::FORMAT_ND)) {
        return true;
    }
    op::Shape selfShape = self->GetViewShape();
    op::Shape mat2Shape = mat2->GetViewShape();
    int64_t selfDimNum = static_cast<int64_t>(selfShape.GetDimNum());
    // rightMove to distinguish different shape of mm and bmm
    int64_t rightMove = 0;
    rightMove = GetOffSet(selfDimNum);

    int64_t m = selfShape.GetDim(rightMove);
    int64_t k = selfShape.GetDim(rightMove + 1);
    int64_t n = mat2Shape.GetDim(rightMove + 1);
    bool mn_multi = m > n ? m < (MN_MULTI * n) : n < (MN_MULTI * m);
    return (m * n * k < MKN_MAX) && mn_multi;
}

bool IsSplitk(const TensorInfo *self, const TensorInfo *mat2)
{
    op::Shape selfShape = self->tensor->GetViewShape();
    op::Shape mat2Shape = mat2->tensor->GetViewShape();
    int64_t selfDimNum = static_cast<int64_t>(selfShape.GetDimNum());
    // rightMove to distinguish different shape of mm and bmm
    int64_t rightMove = 0;
    rightMove = GetOffSet(selfDimNum);
    bool NzSplitKFlag = true;
    // only apply on mm now
    if (!rightMove) {
        NzSplitKFlag = GetNzSplitKFlag(self->tensor, mat2->tensor, self->format, mat2->format);
    }

    int64_t k_dim = selfShape.GetDim(1 + rightMove);
    bool dtype_correct = (self->dataType == DataType::DT_FLOAT16) && (mat2->dataType == DataType::DT_FLOAT16);
    return dtype_correct &&
           k_dim >= SPLIT_K_MULTI * std::max(selfShape.GetDim(rightMove), mat2Shape.GetDim(1 + rightMove)) &&
           NzSplitKFlag;
}

bool IsFormatSupportNd(const aclTensor *self, const aclTensor *mat2)
{
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        return true;
    }
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_93) {
        op::Shape selfShape = self->GetViewShape();
        op::Shape mat2Shape = mat2->GetViewShape();
        int64_t dimNum = selfShape.GetDimNum();
        auto isAligin = [selfShape, mat2Shape, dimNum]() {
            return (!(static_cast<uint64_t>(selfShape.GetDim(dimNum - 2)) & 0x0000000F)) &&
                   (!(static_cast<uint64_t>(selfShape.GetDim(dimNum - 1)) & 0x0000000F)) &&
                   (!(static_cast<uint64_t>(mat2Shape.GetDim(dimNum - 2)) & 0x0000000F)) &&
                   (!(static_cast<uint64_t>(mat2Shape.GetDim(dimNum - 1)) & 0x0000000F));
        };
        if (isAligin() && self->GetDataType() == op::DataType::DT_FLOAT16) {
            return true;
        }
        return false;
    }
    if ((self->GetDataType() == DataType::DT_FLOAT16 && mat2->GetDataType() == DataType::DT_FLOAT16) ||
        (self->GetDataType() == DataType::DT_BF16 && mat2->GetDataType() == DataType::DT_BF16)) {
        return IsNdToNzOnTheFly(self, mat2);
    }
    return true;
}

bool IsSupportNzNzNd(const aclTensor *self, const aclTensor *mat2)
{
    op::Shape selfShape = self->GetViewShape();
    op::Shape mat2Shape = mat2->GetViewShape();
    int64_t dimNum = selfShape.GetDimNum();
    auto isNAligin = [mat2Shape, dimNum]() {
        return ((static_cast<uint64_t>(mat2Shape.GetDim(dimNum - 1)) & 0x0000000F) == 0);
    };
    if (isNAligin() && self->GetDataType() == op::DataType::DT_FLOAT16) {
        return true;
    }
    return false;
}

bool IsNdToNzOnTheFly(const aclTensor *self, const aclTensor *mat2)
{
    uint64_t kInnerAxisMinLimit = 128U;
    uint64_t kInnerAxisMaxLimit = 65535U;
    uint64_t kAxisLengthOne = 1U;
    // 如果self或mat2的维度数量小于2，则不符合判断是否16对齐的条件，返回失败
    if (self->GetViewShape().GetDimNum() < 2 || mat2->GetViewShape().GetDimNum() < 2) {
        return false;
    }
    bool isTransposeSelf = IsTransposeLastTwoDims(self);
    bool isTransposeMat2 = IsTransposeLastTwoDims(mat2);
    uint64_t selfInnerAxis =
        isTransposeSelf ? static_cast<uint64_t>(self->GetViewShape().GetDim(self->GetViewShape().GetDimNum() - 2)) :
                          static_cast<uint64_t>(self->GetViewShape().GetDim(self->GetViewShape().GetDimNum() - 1));
    uint64_t mat2InnerAxis =
        isTransposeMat2 ? static_cast<uint64_t>(mat2->GetViewShape().GetDim(mat2->GetViewShape().GetDimNum() - 2)) :
                          static_cast<uint64_t>(mat2->GetViewShape().GetDim(mat2->GetViewShape().GetDimNum() - 1));

    uint64_t selfOuterAxis =
        isTransposeSelf ? static_cast<uint64_t>(self->GetViewShape().GetDim(self->GetViewShape().GetDimNum() - 1)) :
                          static_cast<uint64_t>(self->GetViewShape().GetDim(self->GetViewShape().GetDimNum() - 2));
    uint64_t mat2OuterAxis =
        isTransposeMat2 ? static_cast<uint64_t>(mat2->GetViewShape().GetDim(mat2->GetViewShape().GetDimNum() - 1)) :
                          static_cast<uint64_t>(mat2->GetViewShape().GetDim(mat2->GetViewShape().GetDimNum() - 2));
    uint64_t mAxis =
        static_cast<uint64_t>(self->GetViewShape().GetDim(self->GetViewShape().GetDimNum() - 2)); // 倒数第2维
    uint64_t kAxis = static_cast<uint64_t>(self->GetViewShape().GetDim(self->GetViewShape().GetDimNum() - 1));
    uint64_t nAxis = static_cast<uint64_t>(mat2->GetViewShape().GetDim(mat2->GetViewShape().GetDimNum() - 1));
    if (selfInnerAxis * selfOuterAxis <= kInnerAxisMaxLimit && mat2InnerAxis * mat2OuterAxis <= kInnerAxisMaxLimit) {
        // too small tensor size
        return true;
    }
    OP_LOGD("Check IsNdToNzOnTheFly, if k=1 scenerio then remains ND.");
    if (kAxis == kAxisLengthOne) {
        return true;
    }

    if (IsSplitKThenForbiddenNd2Nz(mAxis, kAxis, nAxis, isTransposeSelf, isTransposeMat2)) {
        OP_LOGD("Hit small mn multi split k.");
        return true;
    }

    return ((selfInnerAxis >= kInnerAxisMinLimit && selfInnerAxis <= kInnerAxisMaxLimit) ||
            (selfInnerAxis < kInnerAxisMinLimit && ((selfInnerAxis & 0xF) == 0))) &&
           ((mat2InnerAxis >= kInnerAxisMinLimit && mat2InnerAxis <= kInnerAxisMaxLimit) ||
            (mat2InnerAxis < kInnerAxisMinLimit && ((mat2InnerAxis & 0xF) == 0)));
}

bool IsTransposeLastTwoDims(const aclTensor *tensor)
{
    // 当输入tensor的shape小于2或者大于6的时候，返回错误
    if (tensor->GetViewShape().GetDimNum() < 2 || tensor->GetViewShape().GetDimNum() > 6) {
        return false;
    }
    int64_t dim1 = tensor->GetViewShape().GetDimNum() - 1;
    int64_t dim2 = tensor->GetViewShape().GetDimNum() - 2;
    // BMM 场景下，Batch维度的stride需要等于 N, D 的乘积
    if (tensor->GetViewStrides()[dim2] == 1 && tensor->GetViewStrides()[dim1] == tensor->GetViewShape().GetDim(dim2)) {
        int64_t tmpNxD = tensor->GetViewShape().GetDim(dim1) * tensor->GetViewShape().GetDim(dim2);
        // 多batch连续，3是batch索引
        for (int64_t batchDim = tensor->GetViewShape().GetDimNum() - 3; batchDim >= 0; batchDim--) {
            if (tensor->GetViewStrides()[batchDim] != tmpNxD) {
                return false;
            }
            tmpNxD *= tensor->GetViewShape().GetDim(batchDim);
        }
        if (tensor->GetViewShape().GetDim(dim1) == 1 && tensor->GetViewShape().GetDim(dim2) == 1) {
            return false;
        }
        return true;
    }
    return false;
}

aclnnStatus SetMmSupportDType(MmOpInfo &mmOpInfo, int8_t cubeMathType)
{
    bool dtypeMismatch = mmOpInfo.ori_info.self_dtype != mmOpInfo.ori_info.mat2_dtype;
    bool tensorFloat =
        mmOpInfo.ori_info.self_dtype == DataType::DT_FLOAT || mmOpInfo.ori_info.mat2_dtype == DataType::DT_FLOAT;
    bool tensorBfloat16 =
        mmOpInfo.ori_info.self_dtype == DataType::DT_BF16 || mmOpInfo.ori_info.mat2_dtype == DataType::DT_BF16;

    if (!IsInputSupportFp32()) {
        mmOpInfo.support_info.self_dtype = DataType::DT_FLOAT16;
        mmOpInfo.support_info.mat2_dtype = DataType::DT_FLOAT16;
    } else if (IsInputSupportFp32() && cubeMathType == USE_FP16 && (!tensorBfloat16)) {
        // FP16
        mmOpInfo.support_info.self_dtype = DataType::DT_FLOAT16;
        mmOpInfo.support_info.mat2_dtype = DataType::DT_FLOAT16;
        mmOpInfo.support_info.output_dtype = DataType::DT_FLOAT16;
    } else if (IsInputSupportFp32() && dtypeMismatch && (tensorFloat || tensorBfloat16)) {
        // BF16或者存在FP32输入则全部dtype统一到FP32
        mmOpInfo.support_info.self_dtype = DataType::DT_FLOAT;
        mmOpInfo.support_info.mat2_dtype = DataType::DT_FLOAT;
        mmOpInfo.support_info.output_dtype = DataType::DT_FLOAT;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus SetMmSupportFormat(const aclTensor *self, const aclTensor *mat2, MmOpInfo &mmOpInfo)
{
    if (IsFormatSupportNd(self, mat2)) {
        OP_LOGD("Matmul support NDNDND");
        mmOpInfo.support_info.output_format = Format::FORMAT_ND;
        mmOpInfo.support_info.self_format = Format::FORMAT_ND;
        mmOpInfo.support_info.mat2_format = Format::FORMAT_ND;
    } else {
        OP_LOGD("Matmul do not support NDNDND");
        // if 310p and n%16==0
        bool is310p = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P;
        if (IsSupportNzNzNd(self, mat2) && is310p) {
            mmOpInfo.support_info.output_format = Format::FORMAT_ND;
            mmOpInfo.support_info.self_format = Format::FORMAT_FRACTAL_NZ;
            mmOpInfo.support_info.mat2_format = Format::FORMAT_FRACTAL_NZ;
            return ACLNN_SUCCESS;
        }
        mmOpInfo.support_info.output_format = Format::FORMAT_FRACTAL_NZ;
        mmOpInfo.support_info.self_format = Format::FORMAT_FRACTAL_NZ;
        mmOpInfo.support_info.mat2_format = Format::FORMAT_FRACTAL_NZ;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus GetMmInfo(MmOpInfo mmOpInfo)
{
    OP_LOGI(
        "Self tensor input's ori dtype = %s and format = %s; Mat2 tensor input's ori dtype = %s and format = %s;"
        "Output tensor's ori dtype = %s and ori format = %s;"
        "Self tensor input's Npu dtype = %s and Npu format = %s; Mat2 tensor input's Npu dtype = %s and Npuformat = %s;"
        "Output tensor's Npu dtype = %s and Npu format = %s.",
        op::ToString(mmOpInfo.ori_info.self_dtype).GetString(), op::ToString(mmOpInfo.ori_info.self_format).GetString(),
        op::ToString(mmOpInfo.ori_info.mat2_dtype).GetString(), op::ToString(mmOpInfo.ori_info.mat2_format).GetString(),
        op::ToString(mmOpInfo.ori_info.output_dtype).GetString(),
        op::ToString(mmOpInfo.ori_info.output_format).GetString(),
        op::ToString(mmOpInfo.support_info.self_dtype).GetString(),
        op::ToString(mmOpInfo.support_info.self_format).GetString(),
        op::ToString(mmOpInfo.support_info.mat2_dtype).GetString(),
        op::ToString(mmOpInfo.support_info.mat2_format).GetString(),
        op::ToString(mmOpInfo.support_info.output_dtype).GetString(),
        op::ToString(mmOpInfo.support_info.output_format).GetString());
    return ACLNN_SUCCESS;
}

aclIntArray *NeedTransPerm(const aclTensor *x, aclOpExecutor *executor)
{
    op::Shape shape = x->GetViewShape();
    int64_t dimSize = x->GetViewShape().GetDimNum();
    std::vector<int64_t> valuePerm(dimSize, 0);
    for (int64_t i = 0; i < dimSize; i++) {
        valuePerm[i] = shape[i];
    }
    std::swap(valuePerm[dimSize - INNER_AXIS], valuePerm[dimSize - OUTER_AXIS]);
    return executor->AllocIntArray(valuePerm.data(), dimSize);
}

bool checkBF16SizeValid(const aclTensor *&mat2, const bool &transX2Flag)
{
    // 校验N轴是否在优化范围内
    int64_t nDimNumWhenNoTrans = mat2->GetViewShape().GetDimNum() - INNER_AXIS;
    int64_t nDimNumWhenTrans = mat2->GetViewShape().GetDimNum() - OUTER_AXIS;
    int64_t nDim =
        transX2Flag ? mat2->GetViewShape().GetDim(nDimNumWhenTrans) : mat2->GetViewShape().GetDim(nDimNumWhenNoTrans);
    if (nDim > N_KEQAL1_LIMIT) {
        return false;
    }
    return true;
}

bool checkBF16MMValid(const aclTensor *&self, const aclTensor *&mat2, const bool &transX2Flag)
{
    // 校验MN轴是否在优化范围内
    int64_t mDimNumWhenNoTrans = self->GetViewShape().GetDimNum() - OUTER_AXIS;
    int64_t mDimNumWhenTrans = self->GetViewShape().GetDimNum() - INNER_AXIS;
    int64_t mDim =
        transX2Flag ? self->GetViewShape().GetDim(mDimNumWhenTrans) : self->GetViewShape().GetDim(mDimNumWhenNoTrans);
    int64_t nDimNumWhenNoTrans = mat2->GetViewShape().GetDimNum() - INNER_AXIS;
    int64_t nDimNumWhenTrans = mat2->GetViewShape().GetDimNum() - OUTER_AXIS;
    int64_t nDim =
        transX2Flag ? mat2->GetViewShape().GetDim(nDimNumWhenTrans) : mat2->GetViewShape().GetDim(nDimNumWhenNoTrans);
    if (mDim * nDim < MM_KEQAL1_LIMIT) {
        return false;
    }
    return true;
}

bool IfKEqual1(const aclTensor *&selfInput, const MmOpInfo &mmOpInfo, const bool &transX1Flag, const aclTensor *&bias)
{
    // 不支持nz场景
    if (mmOpInfo.support_info.self_format == Format::FORMAT_FRACTAL_NZ ||
        mmOpInfo.support_info.mat2_format == Format::FORMAT_FRACTAL_NZ) {
        return false;
    }
    OP_LOGD("Check MatMul or BatchMatmul k=1 scenario, and support_info is not NZ");
    if (mmOpInfo.support_info.output_dtype != mmOpInfo.support_info.self_dtype ||
        mmOpInfo.support_info.output_dtype != mmOpInfo.support_info.mat2_dtype) {
        return false;
    }
    // 判断是否带bias
    if (bias != nullptr) {
        return false;
    }
    // 判断k轴是否满足切mul需求(等于1)
    int64_t kDimNumWhenNoTrans = selfInput->GetViewShape().GetDimNum() - INNER_AXIS;
    int64_t kDimNumWhenTrans = selfInput->GetViewShape().GetDimNum() - OUTER_AXIS;
    int64_t kDim = transX1Flag ? selfInput->GetViewShape().GetDim(kDimNumWhenTrans) :
                                 selfInput->GetViewShape().GetDim(kDimNumWhenNoTrans);
    if (kDim != DIM_EQUAL_ONE) {
        return false;
    }
    return true;
}

aclnnStatus IfKEqual1SelfToMK(const aclTensor *&selfInput, const aclTensor *&selfReshapeOutput, bool &transX1Flag,
                              aclOpExecutor *executor)
{
    auto x1Perm = NeedTransPerm(selfInput, executor);
    selfReshapeOutput = transX1Flag ? l0op::Reshape(selfInput, x1Perm, executor) : selfInput;
    CHECK_RET(selfReshapeOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
    transX1Flag = false;
    return ACLNN_SUCCESS;
}

aclnnStatus IfKEqual1Mat2ToKN(const aclTensor *&mat2Input, const aclTensor *&mat2ReshapeOutput, bool &transX2Flag,
                              aclOpExecutor *executor)
{
    auto x2Perm = NeedTransPerm(mat2Input, executor);
    mat2ReshapeOutput = transX2Flag ? l0op::Reshape(mat2Input, x2Perm, executor) : mat2Input;
    CHECK_RET(mat2ReshapeOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
    transX2Flag = false;
    return ACLNN_SUCCESS;
}

aclnnStatus IfMEqual1SelfToMK(const aclTensor *&selfInput, const aclTensor *&selfReshapeOutput,
                              const Format selfInputFormat, bool &transX1Flag, aclOpExecutor *executor)
{
    // 不支持nz场景
    if (selfInputFormat == Format::FORMAT_FRACTAL_NZ) {
        return ACLNN_SUCCESS;
    }
    OP_LOGD("Check MatMul or BatchMatmul m=1 scenario, and support_info is not NZ");
    // 首先判断m轴是否已经为外轴，是外轴则return
    if (!transX1Flag) {
        return ACLNN_SUCCESS;
    }
    // 判断m/n轴是否满足等于1，满足则reshape为外轴再进行mm/bmm计算
    int64_t mDimNumWhenInner = selfInput->GetViewShape().GetDimNum() - INNER_AXIS;
    int64_t mDimSize = selfInput->GetViewShape().GetDim(mDimNumWhenInner);
    if (mDimSize != DIM_EQUAL_ONE) {
        return ACLNN_SUCCESS;
    }
    auto x1Perm = NeedTransPerm(selfInput, executor);
    selfReshapeOutput = l0op::Reshape(selfInput, x1Perm, executor);
    transX1Flag = false;
    CHECK_RET(selfReshapeOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
    OP_LOGI("Hit MatMul or BatchMatmul m=1 and m is inner scenario, trans m axis to outer");
    return ACLNN_SUCCESS;
}

aclnnStatus IfNEqual1Mat2ToNK(const aclTensor *&mat2Input, const aclTensor *&mat2ReshapeOutput,
                              const Format mat2InputFormat, bool &transX2Flag, aclOpExecutor *executor)
{
    // 不支持nz场景。
    if (mat2InputFormat == Format::FORMAT_FRACTAL_NZ) {
        return ACLNN_SUCCESS;
    }
    OP_LOGD("Check MatMul or BatchMatmul n=1 scenario, and support_info is not NZ");
    // 首先判断n轴是否已经为外轴，是外轴则return
    if (transX2Flag) {
        return ACLNN_SUCCESS;
    }
    // 判断m/n轴是否满足等于1，满足则reshape为外轴再进行mm/bmm计算
    int64_t nDimNumWhenInner = mat2Input->GetViewShape().GetDimNum() - INNER_AXIS;
    int64_t nDimSize = mat2Input->GetViewShape().GetDim(nDimNumWhenInner);
    if (nDimSize != DIM_EQUAL_ONE) {
        return ACLNN_SUCCESS;
    }
    auto x2Perm = NeedTransPerm(mat2Input, executor);
    mat2ReshapeOutput = l0op::Reshape(mat2Input, x2Perm, executor);
    transX2Flag = true;
    CHECK_RET(mat2ReshapeOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
    OP_LOGI("Hit MatMul or BatchMatmul n=1 and n is inner scenario, trans n axis to outer");
    return ACLNN_SUCCESS;
}

uint64_t TransDequantScaleToM1(const float deqScale)
{
    union {
        float scaleFloat;
        uint32_t scaleInt;
    } dequantScale;
    dequantScale.scaleFloat = deqScale;
    uint64_t fixpipeDeqScale = static_cast<uint64_t>(dequantScale.scaleInt) & kDeqScaleMul;
    return fixpipeDeqScale;
}

op::FVector<int64_t> GetShape(const aclTensor *tensor)
{
    op::FVector<int64_t> shape;
    if (tensor == nullptr) {
        shape.push_back(1);
        OP_LOGW("The input tensor of Func GetShape is nullptr");
        return shape;
    }
    if (tensor->GetViewShape().GetDimNum() == 0U) {
        shape.push_back(1);
    } else {
        size_t dimNum = tensor->GetViewShape().GetDimNum();
        for (size_t idx = 0U; idx < dimNum; idx++) {
            int64_t tmpVal = tensor->GetViewShape().GetDim(idx);
            shape.push_back(tmpVal);
        }
    }
    return shape;
}

const aclTensor *ContiguousBias(const aclTensor *self, const aclTensor *bias, aclOpExecutor *executor)
{
    auto contiguousBias = l0op::Contiguous(bias, executor);
    CHECK_RET(contiguousBias != nullptr, nullptr);
    // bias为bf16时cast为fp32保证精度
    if ((contiguousBias->GetDataType() == DataType::DT_BF16 &&
         GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) ||
        self->GetDataType() == DataType::DT_FLOAT) {
        contiguousBias = l0op::Cast(contiguousBias, op::DataType::DT_FLOAT, executor);
        CHECK_RET(contiguousBias != nullptr, nullptr);
    }
    return contiguousBias;
}

} // namespace Transformer
} // namespace Ops
