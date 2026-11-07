# Worker inheritance contracts

RFC: <https://github.com/vllm-project/vllm-ascend/issues/18920>.

## Dependency order

The migration targets the verified vLLM main lane, based on `24420b7a` plus
<https://github.com/vllm-project/vllm/pull/65565>. The verification pin records
that exact companion commit so the parent contracts remain reproducible.
Merge the vLLM companion first, update the pin to the merged commit, then merge
Ascend phases 0.5a, 0.5b, 0.5c, 1a, 1b, 2, 3, 4 and 5 in order.
The older v0.32.0 lane needs the same Worker-hook backport before this migration
can be released on that lane. Missing hooks fail before the parent constructor;
there is no fallback that bypasses the parent constructor.

## Parent-owned constructor state (phase 1a)

| State | Ascend contract |
| --- | --- |
| Shared config and ranks | Initialized by the real GPU Worker constructor |
| Float32 matmul precision | Adopt upstream `VLLM_FLOAT32_MATMUL_PRECISION` |
| Elastic EP executor | Explicit unsupported executor, no CUDA imports or thread pool |
| Fault-tolerance sentinel | Reject enabled fault tolerance before parent construction |
| Sleep buffer, parameter and draft dictionaries | Created by parent; NPU sleep retains its own policy |
| Weight transfer engine and active/draft flags | Created by parent; transitions migrate in phase 3 |
| Profiler config and wrapper | Validate NPU torch profiler through a constructor hook; create lazily |
| PP send handles | Created by parent; device handles only |
| Sleep backend cache | Created by parent; CaMem integration follows in phase 3 |

V1/V2, 310P and Xlite retain NPU device initialization and their existing runner
construction paths in this phase. Parent `init_device` is never called.
Runtime startup and serving qualification remain required before merging this
stack; portable contract tests are not a substitute for those checks.
