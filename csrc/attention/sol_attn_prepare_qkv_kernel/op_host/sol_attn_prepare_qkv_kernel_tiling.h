#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SolAttnPrepareQkvKernelTilingData)
  TILING_DATA_FIELD_DEF(int32_t, B);
  TILING_DATA_FIELD_DEF(int32_t, T);
  TILING_DATA_FIELD_DEF(int32_t, H);
  TILING_DATA_FIELD_DEF(int32_t, NT);
  TILING_DATA_FIELD_DEF(int32_t, NG);
  TILING_DATA_FIELD_DEF(int32_t, NPAD);
  TILING_DATA_FIELD_DEF(int32_t, T_PAD_KV);
  TILING_DATA_FIELD_DEF(float, inv_tail_len);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SolAttnPrepareQkvKernel, SolAttnPrepareQkvKernelTilingData)
}
