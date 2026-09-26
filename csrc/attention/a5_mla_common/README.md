# A5 attention shared headers

Shared tiling, memory-copy, vector and cube helpers used by the A5
FlashAttention operators. Sources retain the CANN license in `LICENSE`.
The original MLA source snapshot derives from ops-transformer commit
`8d5d69c35e6517c064158b6e0c40267f66856533`.

`op_kernel/arch35/c8_pipeline` contains the FP8/BF16 pipeline header
closure reused by expanded C8 FlashAttention. Its original `flash_mla_*`
identifiers are retained to avoid changing template logic. These headers
include decode/output helpers referenced by the pipeline; they do not
register or build the standalone FlashMLA operator.

The operators that consume these headers select them through their CMake
source dependencies. This directory has no independent runtime switch.
