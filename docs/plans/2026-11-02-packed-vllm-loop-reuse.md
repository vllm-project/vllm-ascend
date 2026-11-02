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

## Historical module-proxy prototype results

The following evidence predates replacement of the module proxy and must not be
used as acceptance evidence for the shared API patch.

### Results on the existing 188 container

Transfer code at `07b0bc01a` was tested in an isolated worktree inside
`lch-packed-fix`, using Ascend 910C, torch 2.10.0+cpu and torch_npu 2.10.0.post4.
Source revisions, imported module paths, tracked cleanliness and the transfer
files' SHA256 were checked before native runs. Installed vLLM stays v0.32.0;
main tests select the fixed main source explicitly, without editing either
vLLM checkout. Native tests use physical devices 8/9.

| Test | v0.32.0 | Paired main |
| --- | --- | --- |
| Weight-transfer unit tests | 167 passed | 167 passed |
| Two-process native HCCL, 16 chunks/update, 2/3 slots | 6/6 correct | 6/6 correct |
| Delayed-clone NPU control | 10/10 correct | 10/10 correct |
| Allocator-pressure NPU control | 10/10 correct; zero unsafe reuse | 10/10 correct; zero unsafe reuse |
| Disable only adapter completion wait | 0/10 correct, as expected | 0/10 correct, as expected |
| Disable only clone storage tracking | 0/10 correct, as expected | 0/10 correct, as expected |

Mechanism controls use actual NPU operations and a simulated collective through
the worker engine; they are distinct from the real two-process HCCL test.
Runtime profiling counts entry into the original vLLM consumer on every update.
Negative readiness controls retain main's original receive-event chain.

Ruff check/format, Markdown lint, spelling, package-init, forbidden-import,
long-function and boolean-context checks pass. Full `bash format.sh ci` could
not complete: the container lacks pre-commit; the Windows pre-commit setup also
fails downloading actionlint's Go dependencies. Do not interpret these as a
complete pre-commit pass. The full model/graph E2E is retained for follow-up but
has not been rerun on this branch; serving performance and long-run acceptance
also remain unverified.

Harness and logs: `/tmp/lch-packed-loop-reuse-20261102` on 188. A local copy of
the evidence archive and provenance manifest is kept in
`C:/lichenhan/packed-pr/packed-loop-reuse-research-20261102`.
