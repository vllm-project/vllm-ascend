#include "kernel_operator.h"
#include "notify_dispatch.h"
#include "notify_dispatch_tiling.h"

#define TILING_KEY_FLOAT16 20
#define TILING_KEY_BFLOAT16 21
#define TILING_KEY_FLOAT 22
#define TILING_KEY_INT 23

#define KERNEL_USE_WORKSPACE (1 * 1024 * 1024)

extern "C" __global__ __aicore__ void notify_dispatch(
    GM_ADDR sendData, GM_ADDR tokenPerExpertData, GM_ADDR sendDataOffset, GM_ADDR recvData, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(NotifyDispatchTilingData);
    GET_TILING_DATA_WITH_STRUCT(NotifyDispatchTilingData, tilingData, tiling);

    // hcomm will set magic later in init
    uint32_t magic = 1;
    GM_ADDR commArgs = nullptr;

    int localRank = tilingData.notifyDispatchInfo.localRankId;
    int localRankSize = tilingData.notifyDispatchInfo.localRankSize;
    int rank = tilingData.notifyDispatchInfo.rankId;
    int rankSize = tilingData.notifyDispatchInfo.rankSize;
    int64_t len = tilingData.notifyDispatchInfo.sendCount;
    int64_t numTokens = tilingData.notifyDispatchInfo.numTokens;

    GM_ADDR sendDataInput = sendData;
    GM_ADDR tokenPerExpertDataInput = tokenPerExpertData;
    GM_ADDR sendDataOffsetOutput = sendDataOffset;
    GM_ADDR recvDataOutput = recvData;

    // fill in unused args
    uint32_t extraFlag = 0;
    GM_ADDR scale = nullptr;
    int root = 0;
    int op = 0;
    int cycleCount = 0;
    int64_t scaleCount = 0;
    GM_ADDR offset = nullptr;
    int blockNum = GetBlockNum();

    if (TILING_KEY_IS(TILING_KEY_FLOAT16)) {
        NotifyDispatch<float16_t> opKernel(rank, rankSize, extraFlag);
        opKernel.Init(KERNELS_ARGS_CALL_ALL2ALL());
        opKernel.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FLOAT)) {
        NotifyDispatch<float> opKernel(rank, rankSize, extraFlag);
        opKernel.Init(KERNELS_ARGS_CALL_ALL2ALL());
        opKernel.Process();
    } else if (TILING_KEY_IS(TILING_KEY_INT)) {
        NotifyDispatch<int> opKernel(rank, rankSize, extraFlag);
        opKernel.Init(KERNELS_ARGS_CALL_ALL2ALL());
        opKernel.Process();
    }
}