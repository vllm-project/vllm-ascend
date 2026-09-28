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

The transformer wrappers still provide the Torch operator schemas.  For the
near-term A5 test, the package loader defaults to the installed arena net-ops
wheel and `CANNBOTDSL_NATIVE_BINARY_MODE=prefer`.  These copies remain in the
repository for a later source-only experiment: set
`DSV41_A5_DSL_SOURCE=local` and `CANNBOTDSL_NATIVE_BINARY_MODE=off` to select
them explicitly.  The source-only path was validated separately but is not
the default in the `vllm_0300` image.

Before changing these copies, compare against the operator team's delivery
and preserve its licensing terms.
