#ifndef REBATCH_AND_UPDATE_TILING_H
#define REBATCH_AND_UPDATE_TILING_H
#if __has_include("../attn_metadata_update/attn_metadata_update_tiling.h")
// Device compilation: dependencies are copied beside rebatch_and_update.
#include "../attn_metadata_update/attn_metadata_update_tiling.h"
#include "../combine_and_dispatch_rebatch/combine_and_dispatch_rebatch_tiling.h"
#else
// Host tiling compilation: use the dependencies from the source tree.
#include "../../attn_metadata_update/op_kernel/attn_metadata_update_tiling.h"
#include "../../combine_and_dispatch_rebatch/op_kernel/combine_and_dispatch_rebatch_tiling.h"
#endif

struct RebatchAndUpdateTilingData {
    CombineAndDispatchRebatchTilingData rebatch;
    AttnMetadataUpdateTilingData update;
};

#endif  // REBATCH_AND_UPDATE_TILING_H
