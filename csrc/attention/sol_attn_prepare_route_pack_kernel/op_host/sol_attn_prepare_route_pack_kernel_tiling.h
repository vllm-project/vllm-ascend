#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SolAttnPrepareRoutePackKernelTilingData)
  TILING_DATA_FIELD_DEF(int32_t, HC);
  TILING_DATA_FIELD_DEF(int32_t, NT);
  TILING_DATA_FIELD_DEF(int32_t, NT_Q);
  TILING_DATA_FIELD_DEF(int32_t, NG);
  TILING_DATA_FIELD_DEF(int32_t, NPAD);
  TILING_DATA_FIELD_DEF(int32_t, NT_Q_PAD8);
  TILING_DATA_FIELD_DEF(int32_t, TILE_M);
  TILING_DATA_FIELD_DEF(int32_t, sink_s0);
  TILING_DATA_FIELD_DEF(int32_t, sink_s1);
  TILING_DATA_FIELD_DEF(float, scale_log2);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SolAttnPrepareRoutePackKernel, SolAttnPrepareRoutePackKernelTilingData)
}
