#ifndef QSA_EXPAND_E3_TILING_H
#define QSA_EXPAND_E3_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(QsaExpandE3TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, rows);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(QsaExpandE3, QsaExpandE3TilingData)
}  // namespace optiling

using optiling::QsaExpandE3TilingData;

#endif
