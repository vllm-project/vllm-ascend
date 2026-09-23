# Official CANN QLI bindings

Custom OPP libraries can export ACLNN names also present in CANN. Resolving a
name successfully does not establish ABI compatibility. In particular, the
custom candidate/stride QLI implementation has extra arguments that the
official QLI V2 interface does not accept.

Two explicit entry points are available under `torch.ops._C_ascend`:

- `npu_quant_lightning_indexer_v2_cann`
- `npu_quant_lightning_indexer_v2_metadata_cann`

Use these as a pair with a matching CANN installation. Both workspace-size
functions and execution functions are resolved from one official component:
`libopapi_transformer.so`, or `libopapi.so` if it supplies the complete group.
An incomplete installation raises an error; there is no custom-library
fallback. Existing custom QLI/candidate entry points and the custom OPP search
path are unchanged. Other custom operators remain available.

These bindings use native PyTorch tensor dtypes. For packed FP4 inputs, view
the packed bytes as `torch.float4_e2m1fn_x2`; E8M0 scales must use
`torch.float8_e8m0fnu`. Passing raw uint8 tensors does not request FP4
reinterpretation. Shape, layout, mask, and quantization constraints are those
of the installed official operator. Custom candidate and explicit stride
extensions are not part of this interface.

This does not alter the separate `cann_ops_transformer` package or migrate
model callers automatically. Callers requesting the official implementation
must select both explicit framework entry points. Do not mix opaque metadata
from a custom provider with an official QLI invocation.

## Validation

CPU-only dynamic-linker regression tests compile tiny ELF libraries. They do
not require importing PyTorch, CANN, or initializing an NPU:

```bash
python -m pytest --noconftest -q tests/ut/ops/test_cann_op_api.py
```

On an available Ascend 950 device, after rebuilding the framework extension
against the intended CANN installation:

```bash
python -m pytest -v \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_cann_qli_binding.py \
  --junitxml=cann_qli_results.xml
```

The device tests compare selected top-k scores against a CPU reference and
check index bounds, uniqueness, and causal masking for FP8 and FP4, including
non-identity page mappings. They also record five device-event timing samples
for metadata, QLI, and their pair after warmup. Timings require an otherwise
idle device; they are not an end-to-end performance claim. A clean official
direct-call baseline must be measured separately before claiming an overhead
or speedup. Device accuracy and performance must be validated before deploying
these bindings; CPU loader tests alone do not establish either.
