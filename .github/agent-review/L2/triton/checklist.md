# Triton Operator Review Checklist

Use the [Triton operator contribution guide](../../../../docs/source/developer_guide/contribution/triton_operators.md)
when reviewing kernels, their public wrappers, callers and tests. Trace each
finding to a changed line and a concrete input or execution path.

- `L2-triton#dispatch-contract`: Follow each changed dispatch branch into its
  wrapper. For a valid input outside the fast path's supported dtype, shape or
  layout, verify the documented fallback can consume the same input. Check
  optional arguments, output allocation and mutations against the caller.
- `L2-triton#indexing`: Trace program IDs, tile indices and masks to each
  load and store. Check tails, padding, sentinels and empty inputs where
  supported. Verify stride units and the integer width of offset products
  before pointer addition. For example, a request-length view accepted by
  preparation must still be indexed with its logical stride during packing.
- `L2-triton#write-ownership`: For in-place or cache updates, identify which
  program owns each destination and which readers need its previous value.
  Check for overlapping stores, clobbered history or writes to padded slots
  under the declared input and stream contract.
- `L2-triton#numerics`: Compare reductions, casts and rounding points with
  the reference for the changed path. For a data-movement output, check exact
  values. For fused arithmetic, distinguish tests against an independent
  reference from tests comparing two modes that share the same computation.
- `L2-triton#specialization`: Trace changing token counts, strides, optional
  pointers and launch options into the JIT signature. Identify which concrete
  inputs can create distinct variants. Where warmup is used, compare its
  registered variants and wrapper calls with those used at runtime.
- `L2-triton#performance-evidence`: Match each speedup claim to its baseline,
  shapes, device and timing scope. Inspect the grid for the claimed workload
  (for example, a token-only grid at single-token decode). Record missing
  measurements in the review summary. A small grid or a possible compiler
  optimization alone does not establish a regression.
