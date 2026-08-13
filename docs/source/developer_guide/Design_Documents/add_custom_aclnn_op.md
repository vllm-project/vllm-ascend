# Adding a custom aclnn operation

This document describes how to add a custom aclnn operation to vllm-ascend.

## How custom aclnn operation works in vllm-ascend?

Custom aclnn operations are built and installed into `vllm_ascend/cann_ops_custom` directory during the build process of vllm-ascend. Then the aclnn operators are bound to `torch.ops._C_ascend` module, enabling users to invoke them in vllm-ascend Python code.

To enable custom operations, use the following code:

```python
from vllm_ascend.utils import enable_custom_op

enable_custom_op()
```

`enable_custom_op()` is the default registration path, but it is not a promise that every supported device generation enables every packaged operator through one global gate. When an operator has a device-specific loading requirement, keep that bootstrap in an operator-scoped helper and register only the intended Torch schema. Do not enable unrelated call sites as a side effect. The [MRV2 categorical sampler](categorical_sampling.md) is an example: A2 and A3 use the general initializer, while A5 loads the packaged ACLNN operator through a sampling-specific path.

## How to add a custom aclnn operation?

  1. Create a new operation folder under `csrc` directory.
  2. Create `op_host` and `op_kernel` directories for host and kernel source code.
  3. Add build options in `csrc/build_aclnn.sh` for supported SOC. Note that multiple ops should be separated with `;`, i.e. `CUSTOM_OPS="op1;op2;op3"`.
  4. Bind aclnn operators to torch.ops._C_ascend module in `csrc/torch_binding.cpp`.
  5. Write a meta implementation in `csrc/torch_binding_meta.cpp` for the op to be captured into the aclgraph.
  6. If registration differs by device generation, expose one operator-scoped capability check and test every supported loading path. Missing registration must fail clearly instead of silently selecting a different implementation.

After a successful build of vllm-ascend, the custom aclnn operation can be invoked in Python code.
