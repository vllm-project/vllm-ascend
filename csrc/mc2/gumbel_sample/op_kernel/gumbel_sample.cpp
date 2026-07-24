#include "gumbel_sample.h"
#include "gumbel_sample_tiling_key.h"

// ===========================================================================
// GumbelSample kernel 入口
//   参数顺序：5 必选输入 + 1 可选输入 + 1 必选输出 + 1 可选输出 + workspace + tiling
//   TilingKey 单维：applyTemp ∈ {0, 1}（0=不缩放, 1=z/τ 缩放）
// ===========================================================================
template <uint32_t applyTemp>
__global__ __aicore__ void gumbel_sample(
    GM_ADDR logits, GM_ADDR idxMapping, GM_ADDR temperature, GM_ADDR seeds,
    GM_ADDR pos, GM_ADDR processedLogitsCol,
    GM_ADDR sampled, GM_ADDR processedLogits, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(GumbelSampleTilingData, tilingData, tiling);

    // [opt-1] TPipe 在核函数入口创建，传指针给 Op 类（减少头尾开销）
    TPipe pipe;
    if constexpr (applyTemp == 1) {
        NsGumbelSample::GumbelSampleOp<true> op;
        op.Init(logits, idxMapping, temperature, seeds, pos, processedLogitsCol,
                sampled, processedLogits, workspace, tilingData, &pipe);
        op.Process();
    } else {
        NsGumbelSample::GumbelSampleOp<false> op;
        op.Init(logits, idxMapping, temperature, seeds, pos, processedLogitsCol,
                sampled, processedLogits, workspace, tilingData, &pipe);
        op.Process();
    }
}
