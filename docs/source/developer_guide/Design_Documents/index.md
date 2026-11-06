# Design Documents

This section provides an overview of the features implemented in vLLM Ascend. Developers can refer to this guide to understand how vLLM Ascend works.

- [Context Parallel Piecewise Graph](context_parallel_piecewise_graph.md) — Eager attention islands for `sfa_cp`/`mla_cp`/DSA-CP under piecewise ACL graphs.
- [KVPP: KV Cache Layer Parallelism](kvpp.md) — Physical cache placement, full-layer broadcast, and test design.
- [Persistent Incremental Csrc Build Cache](persistent_csrc_build_cache.md) — Fine-grained native build reuse, correctness identity, concurrency, and CI integration.
