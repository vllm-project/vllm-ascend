# Bundled GBSA provenance

Source: [L_Euler/ops-transformer_gbsa](https://gitcode.com/L_Euler/ops-transformer_gbsa),
branch `br_a2a3_gbsa_1016`, commit
`50eb8f5b66f20b110106240dc6ee31260572169b` (the branch head when imported).
Both `attention/generic_block_sparse_attention` and
`attention/generic_block_sparse_attention_metadata` use this same revision.
The CANN Open Software License Agreement Version 2.0 is preserved in each
operator directory. Operator-private `attn_infra` and `tla` headers are included.
The metadata operator also bundles its matching `aicpu_common.h` as
`op_kernel_aicpu/gbsa_aicpu_common.h`.

## Integration changes

- Use the repository's `tiling_base/tiling_base.h` include path.
- Adapt metadata AICPU targets to this repository's three-argument CMake helper
  and enable packaging its JSON registration and `libtransformer_aicpu_kernels.so`.
- Apply the upstream explicit-sync kernel compile options to custom-package builds.
- Correct spelling in inherited comments for repository lint checks.
- Add `_C_ascend.npu_generic_block_sparse_attention` and
  `_C_ascend.npu_generic_block_sparse_attention_metadata` dispatcher operators
  with symbolic Meta implementations and process-lifetime layout attributes.
- Build both operators through the normal A2/A3/A5 custom-op lists. An A5
  compile/runtime result is required before claiming A5 support for this port.
- Route MiniMax through these bundled operators. Preserve the metadata cache
  and query-prefix reuse from upstream PR 18426.

Upstream documentation is retained for the operator contract. Upstream's separate
`torch_extension` package, examples, and tests are not installed by this port;
repository tests cover the new dispatcher and model wiring.

## Startup and standalone use

Rebuild and reinstall vllm-ascend after checking out this change; changing Python
source alone does not replace the compiled operators. The normal custom-op build
lists include both operators, including metadata's AICPU registration and library.
The separate `cann_ops_transformer` Python package is not needed for this path;
the base CANN runtime/toolkit is still required.

Register the bundled OPP path before initializing the NPU runtime. Existing vLLM
platform setup calls `bootstrap_custom_op_env()` for this purpose. Standalone
scripts should use this order:

```python
import torch
import torch_npu
from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env(include_vendor_lib=True)
torch.npu.init()
import vllm_ascend.vllm_ascend_C
```

Restart an already initialized process after rebuilding. Updating an environment
variable after runtime initialization can leave the previous tiling provider active,
even if ACLNN symbol lookup finds the new custom API library.

## Static kernel blacklist

Keep the underlying GE/ACLNN name `GenericBlockSparseAttention` unchanged.
PR 19317's existing `GenericBlockSparseAttention` blacklist entry must remain:
npugraph_ex matches substrings of generated JSON filenames, not PyTorch's
`_C_ascend` namespace. Do not change it to `GenericBlockSparseAttn` or
`npu_generic_block_sparse_attention`.

The blacklist intentionally prevents this operator from static-kernel
recompilation; it does not disable ACL Graph capture/replay of the ACLNN operator.
A full model static/super-kernel startup still needs separate NPU validation.

## Focused validation

Run the numerical, Meta, graph-replay, and model metadata-reuse tests with:

```bash
pytest -q tests/e2e/pull_request/one_card/test_generic_block_sparse_attention.py
```

The standalone smoke below runs GBSA plus a vector add through npugraph_ex,
checks output after input changes, and requires GBSA runtime JSONs to appear in
the blacklist directory. Repeat with query length 1/4 and with/without
`--super-kernel`. Use a fresh process and output directory for each case.
The smoke explicitly disables `force_eager` because it owns graph capture;
it does not validate vLLM's outer graph lifecycle or full model startup.

```bash
python tests/e2e/pull_request/one_card/gbsa_static_kernel_smoke.py \
  --query-len 4 --super-kernel --output-dir "$PWD/gbsa-static-q4"
```
