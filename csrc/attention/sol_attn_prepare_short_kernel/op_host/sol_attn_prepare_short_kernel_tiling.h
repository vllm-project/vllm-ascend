#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SolAttnPrepareShortKernelTilingData)
  TILING_DATA_FIELD_DEF(int32_t, B);
  TILING_DATA_FIELD_DEF(int32_t, T);
  TILING_DATA_FIELD_DEF(int32_t, H);
  TILING_DATA_FIELD_DEF(int32_t, NT);
  TILING_DATA_FIELD_DEF(int32_t, T_PAD_KV);
  TILING_DATA_FIELD_DEF(int32_t, NT_Q);
  TILING_DATA_FIELD_DEF(int32_t, NT_Q_PAD8);
  TILING_DATA_FIELD_DEF(int32_t, query_blocks_per_tile);
  TILING_DATA_FIELD_DEF(int32_t, sink_start);
  TILING_DATA_FIELD_DEF(int32_t, sink_end);
  TILING_DATA_FIELD_DEF(float, scale_log2);
  TILING_DATA_FIELD_DEF(float, scale_sq);
  TILING_DATA_FIELD_DEF(float, tau);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SolAttnPrepareShortKernel, SolAttnPrepareShortKernelTilingData)
}
