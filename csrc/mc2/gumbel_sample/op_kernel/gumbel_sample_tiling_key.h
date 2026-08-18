#ifndef GUMBEL_SAMPLE_TILING_KEY_H
#define GUMBEL_SAMPLE_TILING_KEY_H
#include "tiling/template_argument.h"

// 单维 TilingKey：applyTemp ∈ {0, 1}
//   0 = 跳过 z/temp 缩放（apply_temperature=false）
//   1 = 应用 z/temp 缩放（apply_temperature=true，默认）
ASCENDC_TPL_ARGS_DECL(GumbelSample,
    ASCENDC_TPL_UINT_DECL(applyTemp, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(applyTemp, ASCENDC_TPL_UI_LIST, 0, 1)));

#endif  // GUMBEL_SAMPLE_TILING_KEY_H
