# Reuse installed vLLM packed loops on Ascend

## Scope

Keep PR #19127 independent. This branch starts at Ascend main `c567dde89`
and calls the producer/consumer functions from the installed vLLM directly.
The paired main remains `544f182d9`; release compatibility targets v0.32.0.
IPC is unchanged. No vLLM source checkout is modified by this adapter.

## Existing precedents

- `patch/platform/patch_torch_accelerator.py` redirects accelerator memory APIs
  to NPU; it does not make CUDA stream contexts usable on NPU.
- `worker/v2/utils.py::torch_cuda_wrapper` redirects CUDA stream, event and graph
  APIs globally. It demonstrates the necessary device routing but is wider than
  needed by a standalone weight-transfer trainer.

## Alternatives and decision

1. Retain local loops: least change, but every loop bug fix requires review and
   potentially a manual port. This is the existing correctness branch.
2. Redirect all of `torch.cuda`: allows direct calls, but changes unrelated
   modules and relies on initialization order. Do not add another global patch.
3. Redirect the packed module's torch reference and narrowly adapt helper and
   communicator boundaries. This experimental branch chooses this option.

`upstream.packed_nccl_broadcast_producer/consumer` are never replaced, copied,
compiled or cloned with `FunctionType`. Their installed code runs unchanged.
Future changes inside those loops are therefore executed directly, although
changes to helper interfaces/semantics still need compatibility validation.
This does not promise zero future maintenance.

## Adapter boundaries

- A module-local proxy routes `torch.cuda` and generic `torch.Event` to NPU,
  forwarding all other attributes to real torch. Real `torch.cuda` is unchanged.
- Wrap `_get_streams` rather than duplicating its implementation; retain vLLM's
  cache, join caller work, and capture the actual slots for cleanup.
- Override `unpack_tensor` only during active HCCL transfers to clone bytes before
  interpreting dtype/shape. This preserves independent ownership and unaligned
  dtype handling. Outside a transfer, defer to the original helper.
- A communicator adapter establishes native HCCL packing/receive boundaries and
  producer collective order, forwarding the actual NPU stream explicitly.
- A callback adapter records completion after model loading. Before unpack,
  wait for the preceding callback completion and track clones on supported
  readers. This supplements both the release loop without events and the main
  loop whose receive event precedes clone/callback work.
- Always drain captured slots in `finally`, including failures in upstream's
  own normal drain. Retain the existing failed-buffer protection and termination
  exception priority.

## State and concurrency

An import/install lock protects one-time module adaptation. Transfer state lives
in a ContextVar, reset on every exit. Since vLLM caches streams, a cached lock per
device serializes transfers using those slots. Nested transfers are rejected.
This adapter is for NPU processes; mixed CUDA/NPU weight-transfer engines in the
same process are not supported. Caller writers on other streams must join before
entry; callbacks must join relevant additional-stream work onto their slot.

## Removal plan

Contribute device-neutral stream/context/event access and an owning-unpack or
callback-completion hook to vLLM. Once both supported pairings expose suitable
hooks, remove the torch proxy and private-helper wrapping. Do not port the whole
consumer merely to adjust a boundary.

## Validation

Run existing packed and engine tests against both fixed pairings. Add tests for
unchanged upstream loop identities, no global CUDA mutation, callback-after-load
completion, clone tracking, event reuse and context cleanup. Reuse actual-NPU
delayed-clone/allocator-pressure controls and the two-process HCCL harness.
CPU tests alone are insufficient for native HCCL readiness and allocator safety.
Record hardware results separately; no performance improvement is claimed.

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
