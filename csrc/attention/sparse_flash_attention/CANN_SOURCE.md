# Ascend950 kernel source

The Ascend950 (`arch35`) SFA kernel is imported from the official
[CANN ops-transformer repository](https://gitcode.com/cann/ops-transformer),
tag `v9.2.0.pre`, commit `1bba359a9ae31401c63f8d63e2ce8a72cf7c986b`.
Original copyright and CANN Open Software License Agreement Version 2.0
notices are retained. `cann_source_manifest.json` records the upstream paths
and SHA256 hashes of the original files, with LF line endings.

The attention and base utility dependencies are copied into
`op_kernel/arch35/cann_common` to keep this version independent of shared
headers used by other operators. Quoted include paths and the three kernel
service filenames are adapted to the repository layout. Two comment spelling
corrections satisfy the repository's codespell check. The external ACLNN
and PyTorch interfaces are retained: the existing kernel entry passes
`nullptr` for CANN's optional `sinks` argument. This import does not expose a
new sinks API. The Ascend910 (`arch22`) implementation is unchanged.

The host tiling adds CANN's `keyStride0` field. The existing operator defines
key inputs as `AutoContiguous`, so the page stride passed to the kernel is
the page block size in KV rows. This preserves the current input contract
without exposing CANN's optional non-contiguous-kernel interface.

Two local numerical adaptations retain the LSE behavior required by A5 DCP:

- With `return_softmax_lse=True` and sparse mode 0, the valid sorted-index
  prefix bounds the KV loops, excluding trailing `-1` padding from the
  softmax denominator.
- Queries with no selected local keys retain zero output and zero softmax
  max/sum, representing LSE `-inf`, including when the KV cache is nonempty.

The non-LSE path retains upstream behavior. These adaptations assume the
valid indices form a prefix followed by `-1`; interleaved invalid indices
are outside that contract.
