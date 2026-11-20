#pragma once

#include "../../grouped_matmul_situ_quant/op_host/grouped_matmul_situ_quant_a8w4_tiling.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(GroupedMatmulSituQuantMlpTilingData)
TILING_DATA_FIELD_DEF_STRUCT(GMMSituQuantBaseParams, upParams)
TILING_DATA_FIELD_DEF_STRUCT(GMMSituQuantBaseParams, downParams)
TILING_DATA_FIELD_DEF_STRUCT(GMMSituQuant, upVector)
TILING_DATA_FIELD_DEF_STRUCT(GMMSituQuant, downVector)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, upDefault)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, upDecode)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, upSingleToken)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, downCube)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(GroupedMatmulSituQuantMlp, GroupedMatmulSituQuantMlpTilingData)

} // namespace optiling
