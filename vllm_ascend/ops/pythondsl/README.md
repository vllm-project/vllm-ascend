# DeepSeek V4.1 A5 PythonDSL sources

These six files are copied without functional edits from the 2026-09-23
operator delivery used for the A5 QLI, QSLI, and MQSMLA integration:

- `mixed_quant_sparse_flash_mla.py`, `quant_lightning_indexer_dsl.py`, and
  `quant_sparse_lightning_indexer_dsl.py` come from
  `cannbot_arena_net_ops-0.1.0-cp311-cp311-linux_aarch64.whl`.
- `mixed_quant_sparse_flash_mla_metadata.py`,
  `quant_lightning_indexer_metadata_dsl.py` and
  `quant_sparse_lightning_indexer_metadata_dsl.py` come from the matching
  installed 0923 CANN transformer payload.

The transformer wrappers still provide the Torch operator schemas.  Their
`ops.*` imports are redirected to this directory by the A5 package loader.
Only the `cannbotdsl` compiler/runtime wheel is required for these kernels;
the arena net-ops wheel's native variants and AICPU binary are not loaded.
The AICPU metadata kernels compile from the included Python source when no
precompiled binary is present.

Before changing these copies, compare against the operator team's delivery
and preserve its licensing terms.
