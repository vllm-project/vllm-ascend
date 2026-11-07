#include <torch/extension.h>
#include <torch/library.h>
#include "aclnn_torch_adapter/op_api_common.h"
thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset = 0;
#include "attention/quant_lightning_indexer_a3_dcp/torch_adpt.h"
TORCH_LIBRARY_FRAGMENT(_C_ascend, ops) {
#include "attention/quant_lightning_indexer_a3_dcp/register.inc"
}
