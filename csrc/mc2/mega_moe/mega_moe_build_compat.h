/*
 * Build-only compatibility for the CANN 9.0.1 custom-op headers shipped in
 * the target image. Keep this outside the imported MegaMoe implementation.
 */
#ifndef VLLM_ASCEND_MEGA_MOE_BUILD_COMPAT_H
#define VLLM_ASCEND_MEGA_MOE_BUILD_COMPAT_H

#include "log/log.h"

/* CANN 9.0.1's macro body refers to the misspelled `param_name`. */
#ifdef OP_LOGE_WITH_INVALID_INPUT
#undef OP_LOGE_WITH_INVALID_INPUT
#define OP_LOGE_WITH_INVALID_INPUT(opName, paramName)                                 \
    do {                                                                              \
        const std::string compatOpName = (opName);                                    \
        const std::string compatParamName = (paramName);                              \
        OP_LOGE_WITHOUT_REPORT(compatOpName.c_str(), "OP[%s] get [%s] failed.",      \
                               compatOpName.c_str(), compatParamName.c_str());         \
        const std::vector<const char *> msgKey = {"op_name", "param_name"};          \
        const std::vector<const char *> msgvalue = {compatOpName.c_str(),              \
                                                     compatParamName.c_str()};          \
        REPORT_PREDEFINED_ERR_MSG("EZ0004", msgKey, msgvalue);                        \
    } while (0)
#endif

#endif
