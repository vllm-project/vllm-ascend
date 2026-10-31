# 310P GDN operator optimization scope

Branch: `GruntovDima/opt-gdn-updated`.

Upstream base after rebase: `02615df12c0ad44bb7401cfc5c654fe16bebb7d0`.
Historical integration base: `2ee50f3847e4873e61d280911b2a24d85f5278d6`.
Earlier integration validation used vLLM
`ee0da84ab9e04ac7610e28580af62c365e898389` (0.24 API).

## Included changes

- Native compute-WY operator, tiling, registration, bindings and Python
  dispatch, with the existing torch-WY fallback for unsupported inputs.
- 310P FwdH and FwdO kernels, their schedulers and epilogues, and the shared
  hand-written matmul helper.
- Kernel correctness fixes, including causal masking, stable gate differences,
  cumulative gate scan ordering and guarded compute-WY forward substitution.
- Necessary operator build integration and regression tests.
- Initial-state packing required by the paired FwdH/FwdO state-layout contract.

The rebase preserves the optimized 310P calculations, adapts integration to
upstream interfaces and uses the upstream `chunk_fwd_o_vllm` operator name.
The new compute-WY operator also follows upstream's kernel-build policy
without `-Werror`. FwdH uses the current contiguous-state contract instead of
the removed initial-state-stride attribute.

## State-layout contract

External Q/K/W/U/V and gate inputs remain ND. The torch binding accepts an ND
initial state and packs the initial per-chunk state into the 310P kernel's zN
layout. FwdH writes intermediate per-chunk `h` in that same zN layout, and FwdO
consumes it directly. Final recurrent state, updated values and attention output
remain ND.

The intermediate `h` allocation retains its ordinary tensor shape; its physical
data is packed. An ND `h` from an older FwdH must not be passed to this 310P FwdO.
Producer, consumer and initial-state binding must be deployed together.
This is an internal paired-kernel contract, not a request to change model weights.

## Excluded integration optimizations

This PR does not introduce host-boundary metadata reuse, a host-slot state
commit, single-sequence packing/copy elimination, or shared input quantization.
Their helpers, flags, model-runner/metadata-builder/weight-loader changes and
feature-specific tests have been removed from the PR's net diff.

An independent boolean-mask IndexPut state-clear fix has also been separated
from this operator PR. Removing it here does not establish that the historical
IndexPut path works on every 310P runtime; that repair needs its own integration
change and validation.

DFlash, Tree-MTP, QBM, LM-head pruning/routing, last-layer prefill selection and
unrelated normalization/fusion experiments are also outside this scope.
The separately preserved HQ_TEST integration is not modified by this cleanup.

## Validation boundaries

The FwdH launcher matches the headers present in the rebased tree: arch20 uses
four template parameters, arch22 uses six, and arch35 has eight with defaults
for its additional options. The non-310P dispatch and architecture-specific
headers are preserved from upstream. All entries retain the thirteen-argument
OpDef ABI, including the optional `gk` slot. The internal arch20 implementation
still takes twelve arguments and supports scalar gating only; tiling rejects
non-null `gk` on 310P. Host-stub tests derive the external entry arguments from
the OpDef and compile a generated-wrapper-like call for each architecture.
They are not a CANN build or NPU correctness test.

Prior integrated standalone and e2e results remain evidence for their exact
recorded source, binaries, environment and configuration. They are not a new
build, correctness or performance result for this cleaned standalone branch.

The cleanup requires scope/source-preservation checks and Python syntax checks.
A fresh CANN build, NPU operator tests and full-plugin validation are still
required before declaring this branch validated on a new upstream stack.
Do not infer current-main compatibility from the earlier vLLM 0.24 runs.
