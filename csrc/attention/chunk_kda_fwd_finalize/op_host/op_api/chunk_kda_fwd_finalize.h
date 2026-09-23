#ifndef OP_API_INC_LEVEL0_CHUNK_KDA_FWD_FINALIZE_H
#define OP_API_INC_LEVEL0_CHUNK_KDA_FWD_FINALIZE_H

#include "opdev/op_executor.h"

namespace l0op {

const aclTensor *ChunkKdaFwdFinalize(
    const aclTensor *qgScaled,
    const aclTensor *aqk,
    const aclTensor *vNew,
    const aclTensor *h,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const char *outputLayout,
    bool stateVFirst,
    const aclTensor *attnOut,
    aclOpExecutor *executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_CHUNK_KDA_FWD_FINALIZE_H
