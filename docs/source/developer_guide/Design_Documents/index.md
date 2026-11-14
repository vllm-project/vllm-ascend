# Design Documents

This section provides an overview of the features implemented in vLLM Ascend. Developers can refer to this guide to understand how vLLM Ascend works.

- [KVPP: KV Cache Layer Parallelism](kvpp.md) — Physical cache placement, full-layer broadcast, and test design.
- [Kimi K3.1 block-outermost KV cache](kimi_k31_packed_kv_cache.md) — Spec-driven MLA/indexer/draft packing and recurrent overlays.
- [Persistent Incremental Csrc Build Cache](persistent_csrc_build_cache.md) — Fine-grained native build reuse, correctness identity, concurrency, and CI integration.
