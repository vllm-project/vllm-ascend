# Adding a custom aclnn operation

This document describes how to add a custom aclnn operation to vllm-ascend.

## How custom aclnn operation works in vllm-ascend?

Custom aclnn operations are built and installed into `vllm_ascend/cann_ops_custom` directory during the build process of vllm-ascend. Then the aclnn operators are bound to `torch.ops._C_ascend` module, enabling users to invoke them in vllm-ascend Python code.

To enable custom operations, use the following code:

```python
from vllm_ascend.utils import enable_custom_op

enable_custom_op()
```

## How to add a custom aclnn operation?

  1. Create a new operation folder under `csrc` directory.
  2. Create `op_host` and `op_kernel` directories for host and kernel source code.
  3. Add build options in `csrc/build_aclnn.sh` for supported SOC. Note that multiple ops should be separated with `;`, i.e. `CUSTOM_OPS="op1;op2;op3"`.
  4. Bind aclnn operators to torch.ops._C_ascend module in `csrc/torch_binding.cpp`.
  5. Write a meta implementation in `csrc/torch_binding_meta.cpp` for the op to be captured into the aclgraph.

If the new operation participates in the cached generated-kernel build in
`csrc/cmake/func.cmake`, check that its compiler-visible source, generated and
shared inputs, build recipe, and toolchain environment are represented in the
[cache identity contract](persistent_csrc_build_cache.md#cmake-integration-contract).
When changing one of those inputs or commands, update the matching cache
identity arguments in the same change; otherwise an old compiled action could
be reused incorrectly.

After a successful build of vllm-ascend, the custom aclnn operation can be invoked in Python code.
