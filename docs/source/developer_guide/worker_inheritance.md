# Worker inheritance contracts

RFC: <https://github.com/vllm-project/vllm-ascend/issues/18920>.

## Direct inheritance entry point

The migration targets the verified vLLM main lane, based on `24420b7a` plus
<https://github.com/vllm-project/vllm/pull/65565>. The verification pin records
that exact companion commit so the parent contracts remain reproducible.
PR #20865 targets Ascend main directly. There is no prerequisite Ascend PR,
Phase 0.5 alignment step, or stacked successor migration in this change.
Phase 0 contract tooling remains separate work, not a prerequisite branch.
Merge the vLLM companion and update the pin to its merged commit before rollout.
The current v0.33.0 lane needs the same hooks to be qualified or backported.
Missing hooks fail before the parent constructor; there is no fallback that
bypasses the parent constructor.

Keep the existing Ascend method overrides in this PR, including config updates,
reload, PP sends, decorators, loading and dummy batches. Their alignment or
removal is separate follow-up work that must be reviewed against main when it
is proposed. No future PR stack is a dependency of this inheritance change.

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
| PP send handles | Parent initializes the list; the existing Ascend send override is retained |
| Sleep backend cache | Created by parent; CaMem integration follows in phase 3 |

Current main uses V2, encoder-only, 310P and Xlite construction paths; their NPU
device initialization is retained. Parent `init_device` is never called.
Runtime startup and serving qualification remain required before merging this
PR; portable contract tests are not a substitute for those checks.
