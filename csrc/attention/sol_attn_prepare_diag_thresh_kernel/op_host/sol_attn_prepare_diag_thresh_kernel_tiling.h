#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SolAttnPrepareDiagThreshKernelTilingData)
  TILING_DATA_FIELD_DEF(int32_t, B);
  TILING_DATA_FIELD_DEF(int32_t, H);
  TILING_DATA_FIELD_DEF(int32_t, NT);
  TILING_DATA_FIELD_DEF(int32_t, NG);
  TILING_DATA_FIELD_DEF(int32_t, NPAD);
  TILING_DATA_FIELD_DEF(float, scale_log2);
  TILING_DATA_FIELD_DEF(float, tau);
  TILING_DATA_FIELD_DEF(float, inv_nt);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SolAttnPrepareDiagThreshKernel, SolAttnPrepareDiagThreshKernelTilingData)
}
