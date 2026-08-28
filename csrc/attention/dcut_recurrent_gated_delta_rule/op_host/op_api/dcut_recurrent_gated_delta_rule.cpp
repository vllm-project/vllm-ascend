#define CFG_BUILD_DEBUG
#define RecurrentGatedDeltaRule DcutRecurrentGatedDeltaRule

// Include CANN headers FIRST so their include guards are set and our
// overrides below are NOT clobbered when the l0op file re-includes them.
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

// Override OP_TYPE_REGISTER and ADD_TO_LAUNCHER_LIST_AICORE: # (stringify)
// and ## (token paste) do NOT expand macro arguments (C standard), so without
// this override OP_TYPE_REGISTER(RecurrentGatedDeltaRule) registers the op type
// as "RecurrentGatedDeltaRule" instead of "DcutRecurrentGatedDeltaRule", and
// ADD_TO_LAUNCHER_LIST_AICORE calls RecurrentGatedDeltaRuleOpTypeId() (wrong ID)
// instead of DcutRecurrentGatedDeltaRuleOpTypeId(). The runtime then fails with
// "RecurrentGatedDeltaRule ADD_TO_LAUNCHER_LIST_AICORE failed."
#undef OP_TYPE_REGISTER
#undef ADD_TO_LAUNCHER_LIST_AICORE
#undef INFER_SHAPE
#undef ADD_TO_LAUNCHER_LIST_DSA

#define DCUT_STRINGIFY(x) #x
#define DCUT_STRINGIFY_EXPAND(x) DCUT_STRINGIFY(x)
#define DCUT_CAT(a, b) DCUT_CAT_INNER(a, b)
#define DCUT_CAT_INNER(a, b) a##b

#define INFER_SHAPE(KERNEL_NAME, op_args...)                                                           \
    ({  aclnnStatus inferShapeRet;                                                                     \
        do {                                                                                           \
            op::OpArgContext *opArgCtx = GetOpArgContext(op_args);                                     \
            if (opArgCtx == nullptr){                                                                  \
                inferShapeRet = ACLNN_ERR_PARAM_NULLPTR;                                               \
            } else {                                                                                   \
                inferShapeRet = InferShape(DCUT_CAT(KERNEL_NAME, OpTypeId)(),                          \
                                            *opArgCtx->GetOpArg(op::OP_INPUT_ARG),                     \
                                            *opArgCtx->GetOpArg(op::OP_OUTPUT_ARG),                    \
                                            *opArgCtx->GetOpArg(op::OP_ATTR_ARG));                     \
                op::DestroyOpArgContext(opArgCtx);                                                     \
            }                                                                                          \
        } while (0); inferShapeRet;                                                                    \
    })

#define ADD_TO_LAUNCHER_LIST_AICORE(KERNEL_NAME, op_args...)                                 \
    ({  aclnnStatus addToLaunchRet;                                                          \
        do {                                                                                 \
            op::OpArgContext *opArgCtx = GetOpArgContext(op_args);                           \
            addToLaunchRet = CreatAiCoreKernelLauncher(DCUT_STRINGIFY_EXPAND(KERNEL_NAME),  \
                                                       DCUT_CAT(KERNEL_NAME, OpTypeId)(),     \
                                                       executor, opArgCtx);                   \
        } while (0); addToLaunchRet;                                                         \
    })

#define ADD_TO_LAUNCHER_LIST_DSA(KERNEL_NAME, op_args...)                                    \
    do {                                                                                     \
        op::OpArgContext *opArgCtx = GetOpArgContext(op_args);                               \
        CreatDSAKernelLauncher(DCUT_STRINGIFY_EXPAND(KERNEL_NAME),                          \
                               DCUT_CAT(KERNEL_NAME, OpTypeId)(),                           \
                               DCUT_CAT(KERNEL_NAME, TaskType),                             \
                               executor, opArgCtx);                                         \
    } while (0)

#ifdef ACLNN_WITH_BINARY
#undef OP_RESOURCE
#undef DECLARE_OP_RESOURCE
#undef OP_RESOURCES_VALUE

#define DECLARE_OP_RESOURCE(kernelName)                            \
    extern void* DCUT_CAT(kernelName, TilingRegisterResource)();            \
    extern void* DCUT_CAT(kernelName, InferShapeRegisterResource)();       \
    extern void* DCUT_CAT(kernelName, TuningRegisterResource)();            \
    extern const op::OP_BINARY_RES& DCUT_CAT(kernelName, KernelResource)(); \
    extern const op::OP_RUNTIME_KB_RES& DCUT_CAT(kernelName, TuningResource)()

#define OP_RESOURCES_VALUE(kernelName)                                                                  \
    {                                                                                                   \
        {                                                                                               \
            ge::AscendString(DCUT_STRINGIFY_EXPAND(kernelName)),                                        \
            {                                                                                           \
                {l0op::DCUT_CAT(kernelName, TilingRegisterResource)(),                                  \
                 l0op::DCUT_CAT(kernelName, InferShapeRegisterResource)(),                             \
                 l0op::DCUT_CAT(kernelName, TuningRegisterResource)()},                                  \
                    l0op::DCUT_CAT(kernelName, KernelResource)(),                                       \
                    l0op::DCUT_CAT(kernelName, TuningResource)()                                        \
            }                                                                                           \
        }                                                                                               \
    }

#define OP_RESOURCE(kernelName)      \
    DECLARE_OP_RESOURCE(kernelName); \
    const op::OP_RESOURCES DCUT_CAT(kernelName, _RESOURCES) OP_RESOURCES_VALUE(kernelName)

#define OP_TYPE_REGISTER(kernelName)                                                                    \
    OP_RESOURCE(kernelName);                                                                            \
    [[maybe_unused]] uint32_t DCUT_CAT(kernelName, _kernelName_Be_Defined_Multi_Times__);               \
    [[maybe_unused]] inline uint32_t DCUT_CAT(kernelName, OpTypeId)()                                   \
    {                                                                                                   \
        DCUT_CAT(kernelName, _kernelName_Be_Defined_Multi_Times__) = 0;                                 \
        op::GenInternalOpTypeId();                                                                      \
        static uint32_t DCUT_CAT(kernelName, OpTypeId) =                                                \
            op::GenOpTypeId(DCUT_STRINGIFY_EXPAND(kernelName), DCUT_CAT(kernelName, _RESOURCES));       \
        return DCUT_CAT(kernelName, OpTypeId);                                                           \
    }
#else
#define OP_TYPE_REGISTER(kernelName)                                                    \
    [[maybe_unused]] uint32_t DCUT_CAT(kernelName, _kernelName_Be_Defined_Multi_Times__);\
    [[maybe_unused]] inline uint32_t DCUT_CAT(kernelName, OpTypeId)()                   \
    {                                                                                    \
        DCUT_CAT(kernelName, _kernelName_Be_Defined_Multi_Times__) = 0;                  \
        op::GenInternalOpTypeId();                                                       \
        static uint32_t DCUT_CAT(kernelName, OpTypeId) =                                 \
            op::GenOpTypeId(DCUT_STRINGIFY_EXPAND(kernelName));                          \
        return DCUT_CAT(kernelName, OpTypeId);                                           \
    }
#endif

#include "../../../../../csrc/attention/recurrent_gated_delta_rule/op_host/op_api/recurrent_gated_delta_rule.cpp"
