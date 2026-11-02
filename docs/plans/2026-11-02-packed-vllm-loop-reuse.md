# Reuse installed vLLM packed loops with a shared NPU stream patch

## Scope and pairing

PR #19127 stays independent. This branch directly executes vLLM's installed
packed producer and consumer; no copied loop and no FunctionType/private globals
clone. The latest checked Ascend main `426a26bc1` still declares vLLM
`544f182d9` as its verified main pairing. Also validate release v0.32.0 and
latest vLLM main `5c8ae7c7`, without silently changing the repository's official
pairing or either installed vLLM checkout.

## One minimal API patch

`patch/torch_streams.py::patch_cuda_streams` installs exactly three NPU aliases:

- `torch.cuda.Stream = torch.npu.Stream`
- `torch.cuda.stream = torch.npu.stream`
- `torch.cuda.current_stream = torch.npu.current_stream`

Platform startup (`patch_torch_accelerator.py`), standalone packed trainers and
V2's existing CUDA wrapper share this function. It is reapplied before use so a
standalone trainer does not depend on platform/worker initialization order.
Remove the former packed-module torch proxy. Do not patch allocations, CUDA IPC,
graph APIs or generic `torch.Event` here. Generic events were checked against
actual NPU streams on the 188 environment and can record/synchronize directly.

This is deliberately process-wide compatibility for NPU-only processes, like the
existing V2 wrapper. Mixed CUDA/NPU engines are unsupported. Patching earlier
cannot update CUDA aliases another module has already saved; call the patch at
platform startup before consumers create streams. Other legacy wrapper APIs are
outside this change.

## Boundaries that remain separate

`packed_adapter.py` now contains HCCL/correctness adapters only: capture vLLM's
cached slots and join caller work; preserve owning byte clones and reader stream
tracking; order unpack after previous callback completion; keep native HCCL
packing/receive and collective-order boundaries; drain slots on all exits.

A ContextVar holds each transfer's state and per-device locks prevent concurrent
reuse of vLLM's cached slots. Nested transfers are rejected. The remaining
_get_streams and unpack helper hooks are temporary until vLLM exposes suitable
extension points. API replacement alone cannot remove these correctness duties.

## Latest vLLM differences and removal plan

vLLM #63457 moved caller-stream lookup to torch.accelerator, but latest checked
main still creates CUDA Streams and enters CUDA stream contexts. The current
stream alias remains for older supported pairings. Latest main also uses
slot_done events for slot reuse; validate that behavior rather than assuming
all future versions drain a whole slot stream before reuse.

As supported vLLM versions become device-neutral, remove unused aliases after
NPU validation. When owning-unpack and callback-completion hooks are available,
remove private helper wrapping too. Future loop fixes run directly, but changes
to helper signatures/semantics still require review; this is not zero maintenance.

## Validation

Run weight-transfer UT, minimal-patch invariants, the V2 wrapper tests and actual
NPU two-process HCCL, delayed-clone and allocator-pressure controls. Independently
remove adapter readiness and clone tracking to confirm expected corruption.
Model/graph E2E remains a separate acceptance step, not implied by CPU or transport
passes. No performance improvement is claimed.

## Shared API patch results on 188

Runtime implementation tested at `54dd8952b` in an isolated worktree inside
`lch-packed-fix`: Ascend 910C, torch 2.10.0+cpu, torch_npu 2.10.0.post4, physical
8/9. Later commit `51d729e47` adds the fake Event.synchronize required by latest
vLLM's slot_done tests and updates comments; it does not change production
runtime behavior. Source revisions, imported paths, tracked cleanliness and
transfer-file SHA256 are checked by the native harness.

| Test | v0.32.0 | Verified main 544f182d9 | Latest main 5c8ae7c7 |
| --- | --- | --- | --- |
| Weight-transfer + stream patch + V2 wrapper UT | 169 passed | 169 passed | Full collection blocked by unrelated model patches |
| Isolated packed + stream patch UT | Included above | Included above | 110 passed |
| Two-process HCCL, 16 chunks/update, 2/3 slots | 6/6 correct | 6/6 correct | 6/6 correct |
| Delayed-clone NPU control | 10/10 correct | 10/10 correct | 10/10 correct |
| Allocator-pressure NPU control | 10/10 correct | 10/10 correct | 10/10 correct |
| Disable only adapter completion wait | 0/10 correct, expected | 0/10 correct, expected | 0/10 correct, expected |
| Disable only clone tracking | 0/10 correct, expected | 0/10 correct, expected | 0/10 correct, expected |

All allocator positive controls report zero unsafe reuse. Mechanism controls use
actual NPU work and a simulated collective through the worker engine; they are
distinct from the two-process native HCCL tests. Runtime profiling confirms entry
into the original vLLM consumer on every update, including latest slot_done logic.

Latest-main general UT collection fails because Ascend's existing Kimi patch
imports removed get_rope_shape_decorate; an isolated V2 utility import also finds
an absent Qwen4 qsa_cache module. Do not bypass or stub these and claim whole-stack
compatibility. The 110-test run isolates only packed and stream API behavior;
its copied test files include the updated Event fake and are saved with evidence.

Ruff/format, Markdown lint, spelling, forbidden-import, package-init and applicable
Linux function/context checks pass. Full bash format.sh ci was attempted with
Git Bash and pre-commit, but actionlint setup fails downloading Go dependencies.
This is not a complete pre-commit pass. Full model/graph E2E, serving performance
and long-run acceptance have not been rerun on the shared-patch implementation.
No installed vLLM checkout was modified; source selection is explicit per run.

Harness/logs: `/tmp/lch-packed-unified-20261102` on 188. Local evidence copy:
`C:/lichenhan/packed-pr/packed-unified-research-20261102`. The environment's editable
vLLM remains v0.32.0; main tests import their isolated fixed checkouts explicitly.
