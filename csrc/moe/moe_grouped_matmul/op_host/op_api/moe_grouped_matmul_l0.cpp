#include "moe_grouped_matmul_l0.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MoeGroupedMatmul);

const aclTensorList *MoeGroupedMatmul(const aclTensorList *x,
                                   const aclTensorList *weight,
                                   const aclTensor *groupList,
                                   bool transposeWeight,
                                   op::Shape yShape,
                                   size_t outLength,
                                   op::DataType yDtype,
                                   aclOpExecutor *executor) {
    L0_DFX(MoeGroupedMatmul, x, weight, groupList, transposeWeight, outLength);

    std::vector<const aclTensor*> tensorsVec;
    const aclTensor *x0 = x->Size() > 0 ? (*x)[0] : nullptr;
    if (x0 == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "(*x)[0] is nullptr.");
        return nullptr;
    }

    for (size_t i(0); i < outLength; ++i) {
        tensorsVec.emplace_back(executor->AllocTensor(yShape, yDtype));
    }
    auto out = executor->AllocTensorList(tensorsVec.data(), outLength);

    auto x0_dim_num = x0->GetStorageShape().GetDimNum();
    auto x0_dim0 = x0->GetStorageShape().GetDim(0);
    printf("x0_dim_num %d x0_dim0 %d\n", x0_dim_num, x0_dim0);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(MoeGroupedMatmul,
                                      OP_INPUT(x, weight, groupList),
                                      OP_OUTPUT(out),
                                      OP_ATTR(transposeWeight));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }

    return out;
}

}  // namespace l0op