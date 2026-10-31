# Design Documents

This section provides an overview of the features implemented in vLLM Ascend. Developers can refer to this guide to understand how vLLM Ascend works.

- [KVPP: KV Cache Layer Parallelism](kvpp.md) — Physical cache placement, full-layer broadcast, and test design.
- [Persistent Incremental Csrc Build Cache](persistent_csrc_build_cache.md) — Fine-grained native build reuse, correctness identity, concurrency, and CI integration.
- [YOCO Prefill Truncation (Mask / Compact)](yoco_prefill_truncation.md) — DeepSeek-V4.1 consumer keep-tail mask vs compact, vs decoder bounded replay.
- [Speculative Decoding KV Sliding Window](speculative_kv_sliding_window.md) — Draft-only `draft_window_size` for EAGLE3 / DFlash / DSpark (orthogonal to YOCO).
